"""Controlled normal-only null calibration.

Phase 2D migration of the validated `controlled_null_centering` implementation.
Scientific behavior is intentionally preserved.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ..evidence import (
    HistogramSpec,
    NullBaselineCache,
    controlled_reference_mass,
    histogram_from_actions,
    exact_null_baseline_w1_cached,
    signed_w1_evidence,
    wasserstein1_ordered_hist,
)

def controlled_null_centering(
    config: Dict[str, Any],
    seed: int,
    run_dir: Path,
    stage_seed: int,
) -> Tuple[Dict, Dict]:
    """
    Normal-only null-centering experiment under the frozen participation model.

    Each of N_normal accounts participates independently with probability
    p_normal in every interval, so n_t is realized rather than fixed.
    The matched null baseline is therefore b(n_t,h), exactly as specified
    by the framework.
    """
    ccfg = config["controlled_model"]
    acfg = ccfg["action_space"]
    normal = ccfg["normal_distribution"]

    T = int(ccfg["T"])
    p_normal = float(ccfg["p_normal"])
    n_normal_accounts = int(ccfg["N_normal"])

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

    # Realized interval sizes from independent account participation.
    n_t = rng.binomial(n_normal_accounts, p_normal, size=T)
    if np.any(n_t <= 0):
        raise RuntimeError(
            "Encountered an empty interval. Under the frozen parameters "
            "this should be effectively impossible; investigate before proceeding."
        )

    # Build each unique matched baseline once, then reuse it.
    cache = NullBaselineCache(Path(run_dir).parent / "_baseline_cache")
    unique_n = np.unique(n_t)
    baselines = {}
    baseline_meta_by_n = {}
    for n in unique_n:
        b, meta = exact_null_baseline_w1_cached(
            reference,
            int(n),
            bin_positions=spec.centers,
            cache=cache,
        )
        baselines[int(n)] = b
        baseline_meta_by_n[str(int(n))] = meta

    raw = np.empty(T, dtype=float)
    signed = np.empty(T, dtype=float)
    used_baseline = np.empty(T, dtype=float)

    for t in range(T):
        n = int(n_t[t])
        x = rng.normal(
            loc=float(normal["mean"]),
            scale=float(normal["std"]),
            size=n,
        )
        hist, observed_n = histogram_from_actions(x, spec, clip=True)
        if observed_n != n:
            raise RuntimeError("Unexpected action-count mismatch.")

        raw[t] = wasserstein1_ordered_hist(
            hist,
            reference,
            bin_positions=spec.centers,
        )
        used_baseline[t] = baselines[n]
        signed[t] = signed_w1_evidence(
            hist,
            reference,
            baseline=used_baseline[t],
            bin_positions=spec.centers,
        )

    cumulative_raw = np.cumsum(raw)
    cumulative_signed = np.cumsum(signed)
    t_axis = np.arange(1, T + 1, dtype=float)
    raw_slope = float(np.polyfit(t_axis, cumulative_raw, deg=1)[0])
    signed_slope = float(np.polyfit(t_axis, cumulative_signed, deg=1)[0])

    data_dir = Path(run_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_dir / "controlled_null_centering.npz",
        n_t=n_t,
        raw_w1=raw,
        matched_baseline=used_baseline,
        signed_d=signed,
        cumulative_raw=cumulative_raw,
        cumulative_signed=cumulative_signed,
        reference=reference,
        bin_centers=spec.centers,
    )

    metrics = {
        "experiment": "controlled_null_centering",
        "status": "OK",
        "master_seed": int(seed),
        "stage_seed": int(stage_seed),
        "T": T,
        "mean_n_t": float(np.mean(n_t)),
        "std_n_t": float(np.std(n_t, ddof=1)),
        "min_n_t": int(np.min(n_t)),
        "max_n_t": int(np.max(n_t)),
        "n_unique_n_t": int(len(unique_n)),
        "mean_matched_baseline": float(np.mean(used_baseline)),
        "mean_raw_w1": float(np.mean(raw)),
        "mean_signed_increment": float(np.mean(signed)),
        "raw_cumulative_slope": raw_slope,
        "signed_cumulative_slope": signed_slope,
        "final_raw_cumulative": float(cumulative_raw[-1]),
        "final_signed_cumulative": float(cumulative_signed[-1]),
    }

    manifest = {
        "experiment": "controlled_null_centering",
        "master_seed": int(seed),
        "stage_seed": int(stage_seed),
        "inputs": {
            "T": T,
            "N_normal": n_normal_accounts,
            "p_normal": p_normal,
            "participation_model": "independent Bernoulli per normal account and interval",
            "histogram_support": [spec.support_min, spec.support_max],
            "histogram_bins": spec.n_bins,
            "normal_mean": float(normal["mean"]),
            "normal_std": float(normal["std"]),
            "baseline_method": "exact_binomial_marginal_expectation",
        },
        "outputs": {
            "timeseries": "data/controlled_null_centering.npz",
        },
        "invariants": {
            "reference_sums_to_one": bool(
                abs(float(reference.sum()) - 1.0) < 1e-12
            ),
            "all_interval_counts_positive": bool(np.all(n_t > 0)),
            "all_raw_w1_nonnegative": bool(np.all(raw >= -1e-15)),
            "signed_equals_raw_minus_matched_baseline": bool(
                np.allclose(
                    signed,
                    raw - used_baseline,
                    atol=1e-15,
                    rtol=0.0,
                )
            ),
            "baseline_matched_to_realized_n": bool(
                all(
                    used_baseline[t] == baselines[int(n_t[t])]
                    for t in range(T)
                )
            ),
        },
        "baseline_cache_entries": baseline_meta_by_n,
    }

    if not all(manifest["invariants"].values()):
        raise RuntimeError(f"Invariant failure: {manifest['invariants']}")

    return metrics, manifest

