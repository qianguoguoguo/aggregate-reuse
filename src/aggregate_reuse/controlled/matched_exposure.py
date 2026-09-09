from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .campaign import requested_k_on, coalition_action_sample
from ..evidence import (
    HistogramSpec,
    NullBaselineCache,
    controlled_reference_mass,
    histogram_from_actions,
    exact_null_baseline_w1_cached,
    signed_w1_evidence,
    wasserstein1_ordered_hist,
)
from ..metrics import coalition_placebo_auc, fit_cumulative_slope
from ..seeding import derive_stage_seed


def _balanced_coalition_matrix(
    I_t: np.ndarray,
    N_coalition: int,
    k_on: int,
) -> np.ndarray:
    """
    Boolean exposure matrix of shape (N_coalition, T) using the exact same
    cyclic balanced schedule as BalancedCyclicScheduler, but vectorized.
    """
    T = len(I_t)
    A = np.zeros((N_coalition, T), dtype=np.uint8)
    pointer = 0
    for t in range(T):
        if I_t[t] == 0:
            continue
        idx = (pointer + np.arange(k_on)) % N_coalition
        A[idx, t] = 1
        pointer = (pointer + k_on) % N_coalition
    return A


def controlled_matched_exposure_divergence(
    config: Dict[str, Any],
    seed: int,
    run_dir: Path,
    stage_seed: int,
) -> Tuple[Dict, Dict]:

    ccfg = config["controlled_model"]
    acfg = ccfg["action_space"]
    normal = ccfg["normal_distribution"]
    attack = ccfg["coalition_distributions"]["mean_shift"]

    T = int(ccfg["T"])
    N_normal = int(ccfg["N_normal"])
    N_coalition = int(ccfg["N_coalition"])
    p_normal = float(ccfg["p_normal"])
    p_on = float(ccfg["campaign"]["principal_p_on"])
    R_exp = float(ccfg["campaign"]["principal_exposure_ratio"])

    spec = HistogramSpec(
        support_min=float(acfg["support"][0]),
        support_max=float(acfg["support"][1]),
        n_bins=int(acfg["histogram_bins"]),
    )
    reference = controlled_reference_mass(
        spec,
        mean=float(normal["mean"]),
        std=float(normal["std"]),
    )

    rng = np.random.default_rng(stage_seed)

    # Campaign state.
    I_t = rng.binomial(1, p_on, size=T).astype(np.uint8)

    # Normal exposure matrix: exact Bernoulli per account per interval.
    A_n = (rng.random((N_normal, T)) < p_normal).astype(np.uint8)
    n_normal_t = A_n.sum(axis=0).astype(int)

    # Balanced coalition exposure matrix.
    k_on = requested_k_on(
        R_exp=R_exp,
        p_normal=p_normal,
        N_coalition=N_coalition,
        p_on=p_on,
    )
    A_c = _balanced_coalition_matrix(I_t, N_coalition, k_on)
    n_coalition_t = A_c.sum(axis=0).astype(int)

    normal_exposure_counts = A_n.sum(axis=1).astype(int)
    coalition_exposure_counts = A_c.sum(axis=1).astype(int)

    realized_normal_rate = float(A_n.mean())
    realized_coalition_rate = float(A_c.mean())
    realized_R_exp = realized_coalition_rate / realized_normal_rate
    realized_p_on = float(I_t.mean())

    # Aggregate evidence stream.
    cache = NullBaselineCache(Path(run_dir).parent / "_baseline_cache")
    d = np.empty(T, dtype=float)
    raw_w1 = np.empty(T, dtype=float)
    baseline_used = np.empty(T, dtype=float)

    for t in range(T):
        n_n = int(n_normal_t[t])
        n_c = int(n_coalition_t[t])

        x_n = rng.normal(
            loc=float(normal["mean"]),
            scale=float(normal["std"]),
            size=n_n,
        )
        if n_c:
            x_c = coalition_action_sample(rng, n_c, attack)
            x = np.concatenate([x_n, x_c])
        else:
            x = x_n

        hist, n_total = histogram_from_actions(x, spec, clip=True)
        b, _ = exact_null_baseline_w1_cached(
            reference,
            n_total,
            bin_positions=spec.centers,
            cache=cache,
        )
        baseline_used[t] = b
        raw_w1[t] = wasserstein1_ordered_hist(
            hist, reference, bin_positions=spec.centers
        )
        d[t] = signed_w1_evidence(
            hist,
            reference,
            baseline=b,
            bin_positions=spec.centers,
        )

    # Exposure attribution entirely by matrix multiplication.
    # increments: accounts x T
    inc_n = A_n * d[None, :]
    inc_c = A_c * d[None, :]
    traj_n = np.cumsum(inc_n, axis=1)
    traj_c = np.cumsum(inc_c, axis=1)
    final_n = traj_n[:, -1]
    final_c = traj_c[:, -1]

    empirical_gap_traj = traj_c.mean(axis=0) - traj_n.mean(axis=0)
    empirical_final_gap = float(empirical_gap_traj[-1])

    # Direct realized q and nu estimates.
    active = I_t == 1
    inactive = ~active
    n_active = int(active.sum())
    n_inactive = int(inactive.sum())

    if n_active == 0 or n_inactive == 0:
        raise RuntimeError("Need both active and inactive intervals for theorem validation.")

    q_c1_hat = float(A_c[:, active].mean())
    q_n1_hat = float(A_n[:, active].mean())
    q_n0_hat = float(A_n[:, inactive].mean())

    # Conditional evidence upon participation is exposure-weighted mean d_t.
    c_active_exposure = A_c[:, active].sum()
    n_active_exposure = A_n[:, active].sum()
    n_inactive_exposure = A_n[:, inactive].sum()

    nu_c1_hat = float((A_c[:, active] * d[active][None, :]).sum() / c_active_exposure)
    nu_n1_hat = float((A_n[:, active] * d[active][None, :]).sum() / n_active_exposure)
    nu_n0_hat = float((A_n[:, inactive] * d[inactive][None, :]).sum() / n_inactive_exposure)

    p_hat = realized_p_on
    q_n_hat = realized_normal_rate
    Delta_hat = q_n_hat * (
        nu_c1_hat
        - p_hat * nu_n1_hat
        - (1.0 - p_hat) * nu_n0_hat
    )
    predicted_gap_traj = np.arange(1, T + 1, dtype=float) * Delta_hat
    predicted_final_gap = float(predicted_gap_traj[-1])

    # Pair first N_coalition normal accounts with coalition accounts.
    n_pairs = min(N_coalition, N_normal)
    pair_gap_traj = traj_c[:n_pairs, :] - traj_n[:n_pairs, :]
    pair_error_traj = np.mean(pair_gap_traj <= 0.0, axis=0)
    final_pair_error = float(pair_error_traj[-1])

    frequency_auc = coalition_placebo_auc(
        coalition_exposure_counts.astype(float),
        normal_exposure_counts.astype(float),
    )
    score_auc = coalition_placebo_auc(final_c, final_n)

    empirical_slope = fit_cumulative_slope(empirical_gap_traj)
    slope_relative_error = (
        abs(empirical_slope - Delta_hat) / abs(Delta_hat)
        if Delta_hat != 0 else float("nan")
    )

    data_dir = Path(run_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_dir / "controlled_matched_exposure_divergence.npz",
        I_t=I_t,
        d_t=d,
        raw_w1=raw_w1,
        baseline=baseline_used,
        empirical_gap=empirical_gap_traj,
        predicted_gap=predicted_gap_traj,
        pair_error=pair_error_traj,
        normal_exposure_counts=normal_exposure_counts,
        coalition_exposure_counts=coalition_exposure_counts,
    )

    metrics = {
        "experiment": "controlled_matched_exposure_divergence",
        "status": "OK",
        "master_seed": int(seed),
        "stage_seed": int(stage_seed),
        "requested_R_exp": R_exp,
        "realized_R_exp": realized_R_exp,
        "requested_p_on": p_on,
        "realized_p_on": realized_p_on,
        "k_on": int(k_on),
        "frequency_auc": float(frequency_auc),
        "score_auc": float(score_auc),
        "empirical_final_gap": empirical_final_gap,
        "predicted_final_gap": predicted_final_gap,
        "Delta_hat": float(Delta_hat),
        "empirical_gap_slope": float(empirical_slope),
        "slope_relative_error": float(slope_relative_error),
        "final_pair_misordering": final_pair_error,
        "q_c1_hat": q_c1_hat,
        "q_n1_hat": q_n1_hat,
        "q_n0_hat": q_n0_hat,
        "nu_c1_hat": nu_c1_hat,
        "nu_n1_hat": nu_n1_hat,
        "nu_n0_hat": nu_n0_hat,
    }

    manifest = {
        "experiment": "controlled_matched_exposure_divergence",
        "master_seed": int(seed),
        "stage_seed": int(stage_seed),
        "implementation": "vectorized_exact_w1_v2",
        "inputs": {
            "baseline_method": "exact_binomial_marginal_expectation",
            "T": T,
            "N_normal": N_normal,
            "N_coalition": N_coalition,
            "p_normal": p_normal,
            "p_on": p_on,
            "R_exp": R_exp,
            "attack": attack,
        },
        "outputs": {
            "timeseries": "data/controlled_matched_exposure_divergence.npz",
        },
        "invariants": {
            "coalition_only_active": bool(np.all(n_coalition_t[inactive] == 0)),
            "k_on_constant_when_active": bool(np.all(n_coalition_t[active] == k_on)),
            "expected_exposure_ratio_matches_request": bool(
                abs((p_on * k_on / N_coalition) / p_normal - R_exp) < 1e-12
            ),
            "all_evidence_finite": bool(np.all(np.isfinite(d))),
            "all_baselines_nonnegative": bool(np.all(baseline_used >= 0)),
            "balanced_coalition_exposure": bool(
                coalition_exposure_counts.max() - coalition_exposure_counts.min() <= 1
            ),
        },
        "realized_schedule": {
            "realized_R_exp": realized_R_exp,
            "realized_p_on": realized_p_on,
            "normal_rate": realized_normal_rate,
            "coalition_rate": realized_coalition_rate,
        },
    }

    if not all(manifest["invariants"].values()):
        raise RuntimeError(f"Invariant failure: {manifest['invariants']}")

    return metrics, manifest
