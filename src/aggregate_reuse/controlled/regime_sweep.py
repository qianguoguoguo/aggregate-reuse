from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import sparse

from .campaign import requested_k_on
from ..evidence import (
    HistogramSpec,
    NullBaselineCache,
    controlled_reference_mass,
    exact_null_baseline_w1,
    wasserstein1_ordered_hist,
)
from ..metrics import coalition_placebo_auc, matched_pair_misordering
from ..seeding import derive_stage_seed


def _balanced_scores_and_frequency(
    I_t: np.ndarray,
    k_on: int,
    N_coalition: int,
    d_t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balanced cyclic coalition allocation without materializing a dense matrix.
    """
    scores = np.zeros(N_coalition, dtype=float)
    freq = np.zeros(N_coalition, dtype=np.int32)
    pointer = 0

    for t in np.flatnonzero(I_t):
        k = int(k_on)
        end = pointer + k
        if end <= N_coalition:
            scores[pointer:end] += d_t[t]
            freq[pointer:end] += 1
        else:
            first = N_coalition - pointer
            scores[pointer:] += d_t[t]
            freq[pointer:] += 1
            remain = k - first
            if remain:
                scores[:remain] += d_t[t]
                freq[:remain] += 1
        pointer = end % N_coalition

    return scores, freq


def _reference_from_attack(spec: HistogramSpec, cfg: Dict[str, Any]) -> np.ndarray:
    family = cfg["family"]
    if family == "normal":
        return controlled_reference_mass(
            spec, mean=float(cfg["mean"]), std=float(cfg["std"])
        )
    if family == "symmetric_gaussian_mixture":
        from ..evidence import gaussian_mixture_reference_mass
        return gaussian_mixture_reference_mass(
            spec,
            weights=cfg["weights"],
            means=cfg["component_means"],
            stds=cfg["component_stds"],
        )
    raise ValueError(f"Unsupported attack family: {family}")


def controlled_exposure_intermittency_regime(
    config: Dict[str, Any],
    seed: int,
    run_dir: Path,
    stage_seed: int,
) -> Tuple[Dict, Dict]:
    """
    Full frozen R_exp x p_on sweep for the primary mean-shift attack.

    Efficiency strategy (scientifically neutral):
    - one normal participation/background realization per seed is reused
      across grid cells (common random numbers);
    - aggregate normal histograms are drawn exactly from the binned reference
      conditional on the realized normal action count;
    - normal account scores for all cells are computed in one sparse-dense
      matrix multiplication.
    """
    ccfg = config["controlled_model"]
    acfg = ccfg["action_space"]
    normal = ccfg["normal_distribution"]
    attack = ccfg["coalition_distributions"]["mean_shift"]
    campaign = ccfg["campaign"]

    T = int(ccfg["T"])
    Nn = int(ccfg["N_normal"])
    Nc = int(ccfg["N_coalition"])
    pn = float(ccfg["p_normal"])
    R_grid = [float(x) for x in campaign["exposure_ratio_grid"]]
    p_grid = [float(x) for x in campaign["p_on_grid"]]
    draws = int(ccfg["baseline_monte_carlo"]["draws_per_configuration"])

    spec = HistogramSpec(
        support_min=float(acfg["support"][0]),
        support_max=float(acfg["support"][1]),
        n_bins=int(acfg["histogram_bins"]),
    )
    ref = controlled_reference_mass(
        spec, mean=float(normal["mean"]), std=float(normal["std"])
    )
    attack_mass = _reference_from_attack(spec, attack)

    # --------------------------------------------------------------
    # Shared normal background for this seed.
    # --------------------------------------------------------------
    rng_normal = np.random.default_rng(
        derive_stage_seed(seed, "controlled_sweep::normal_background")
    )
    # Sparse Bernoulli exposure matrix A_n of shape Nn x T.
    # scipy.sparse.random would not give exact Bernoulli entries cleanly;
    # generate interval-wise participant indices and assemble CSR.
    rows = []
    cols = []
    n_normal_t = np.empty(T, dtype=np.int32)
    normal_hist_counts = np.empty((T, spec.n_bins), dtype=np.int32)

    for t in range(T):
        mask = rng_normal.random(Nn) < pn
        idx = np.flatnonzero(mask)
        n = len(idx)
        if n <= 0:
            raise RuntimeError("Empty normal interval under frozen parameters.")
        rows.extend(idx.tolist())
        cols.extend([t] * n)
        n_normal_t[t] = n
        normal_hist_counts[t] = rng_normal.multinomial(n, ref)

    data = np.ones(len(rows), dtype=np.float64)
    A_normal = sparse.csr_matrix(
        (data, (np.asarray(rows), np.asarray(cols))),
        shape=(Nn, T),
    )
    normal_freq = np.asarray(A_normal.sum(axis=1)).ravel()

    # Shared uniforms make ON sets nested across p_on values.
    rng_state = np.random.default_rng(
        derive_stage_seed(seed, "controlled_sweep::campaign_uniforms")
    )
    campaign_uniform = rng_state.random(T)

    cells = [(R, p) for R in R_grid for p in p_grid]
    D = np.empty((T, len(cells)), dtype=np.float64)
    per_cell_aux: List[Dict[str, Any]] = []

    cache = NullBaselineCache(Path(run_dir).parent / "_baseline_cache")

    all_needed_n = set(int(n) for n in n_normal_t.tolist())
    cell_k = {}
    for R_exp, p_on in cells:
        k = requested_k_on(
            R_exp=R_exp, p_normal=pn, N_coalition=Nc, p_on=p_on
        )
        cell_k[(R_exp, p_on)] = k
        all_needed_n.update(int(n + k) for n in n_normal_t.tolist())

    baseline_by_n = {
        n_total: exact_null_baseline_w1(
            ref, n_total, bin_positions=spec.centers
        )
        for n_total in sorted(all_needed_n)
    }

    gaps = np.diff(spec.centers)

    for j, (R_exp, p_on) in enumerate(cells):
        I_t = (campaign_uniform < p_on).astype(np.int8)
        k_on = cell_k[(R_exp, p_on)]

        rng_cell = np.random.default_rng(
            derive_stage_seed(
                seed,
                f"controlled_sweep::R={R_exp:.8g}::p={p_on:.8g}"
            )
        )

        coalition_hist_counts = np.zeros(
            (T, spec.n_bins), dtype=np.int32
        )
        active_idx = np.flatnonzero(I_t)
        if len(active_idx):
            coalition_hist_counts[active_idx] = rng_cell.multinomial(
                k_on, attack_mass, size=len(active_idx)
            )

        counts = normal_hist_counts + coalition_hist_counts
        n_total = counts.sum(axis=1).astype(np.int32)
        hist = counts.astype(float) / n_total[:, None]
        cdf_diff = np.cumsum(hist - ref[None, :], axis=1)[:, :-1]
        raw = np.sum(np.abs(cdf_diff) * gaps[None, :], axis=1)
        bvec = np.fromiter(
            (baseline_by_n[int(n)] for n in n_total),
            dtype=float, count=T
        )
        d = raw - bvec
        D[:, j] = d

        p_hat = float(I_t.mean())
        coalition_rate = p_hat * k_on / Nc
        normal_rate = float(normal_freq.mean()) / T
        realized_R = coalition_rate / normal_rate

        per_cell_aux.append({
            "R_exp": R_exp,
            "p_on": p_on,
            "k_on": int(k_on),
            "I_t": I_t,
            "realized_p_on": p_hat,
            "realized_R_exp": realized_R,
        })

    # All normal account scores for all cells at once.
    normal_scores_matrix = A_normal @ D

    rows_out: List[Dict[str, Any]] = []
    for j, aux in enumerate(per_cell_aux):
        d = D[:, j]
        c_scores, c_freq = _balanced_scores_and_frequency(
            aux["I_t"], aux["k_on"], Nc, d
        )
        n_scores = np.asarray(normal_scores_matrix[:, j]).ravel()

        # Pair first Nc normal accounts with Nc coalition accounts for the
        # controlled matched-pair diagnostic.
        n_pair = n_scores[:Nc]

        frequency_auc = coalition_placebo_auc(c_freq, normal_freq)
        score_auc = coalition_placebo_auc(c_scores, n_scores)
        pair_error = matched_pair_misordering(c_scores, n_pair)

        row = {
            "master_seed": int(seed),
            "requested_R_exp": float(aux["R_exp"]),
            "requested_p_on": float(aux["p_on"]),
            "realized_R_exp": float(aux["realized_R_exp"]),
            "realized_p_on": float(aux["realized_p_on"]),
            "k_on": int(aux["k_on"]),
            "mean_signed_increment": float(np.mean(d)),
            "mean_active_increment": float(
                np.mean(d[aux["I_t"] == 1])
            ) if np.any(aux["I_t"] == 1) else float("nan"),
            "mean_inactive_increment": float(
                np.mean(d[aux["I_t"] == 0])
            ) if np.any(aux["I_t"] == 0) else float("nan"),
            "mean_coalition_score": float(np.mean(c_scores)),
            "mean_normal_score": float(np.mean(n_scores)),
            "mean_score_gap": float(np.mean(c_scores) - np.mean(n_scores)),
            "frequency_auc": float(frequency_auc),
            "score_auc": float(score_auc),
            "matched_pair_misordering": float(pair_error),
        }
        rows_out.append(row)

    data_dir = Path(run_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # JSON rows are easy to aggregate across seeds.
    (data_dir / "controlled_exposure_intermittency_rows.json").write_text(
        json.dumps(rows_out, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez_compressed(
        data_dir / "controlled_exposure_intermittency_core.npz",
        D=D,
        normal_n_t=n_normal_t,
        normal_frequency=normal_freq,
        campaign_uniform=campaign_uniform,
    )

    principal = [
        r for r in rows_out
        if abs(r["requested_R_exp"] - campaign["principal_exposure_ratio"]) < 1e-12
        and abs(r["requested_p_on"] - campaign["principal_p_on"]) < 1e-12
    ][0]

    metrics = {
        "experiment": "controlled_exposure_intermittency_regime",
        "status": "OK",
        "master_seed": int(seed),
        "stage_seed": int(stage_seed),
        "n_grid_cells": len(rows_out),
        "R_grid": R_grid,
        "p_on_grid": p_grid,
        "principal_frequency_auc": principal["frequency_auc"],
        "principal_score_auc": principal["score_auc"],
        "principal_pair_misordering": principal["matched_pair_misordering"],
        "principal_realized_R_exp": principal["realized_R_exp"],
        "principal_realized_p_on": principal["realized_p_on"],
    }

    manifest = {
        "experiment": "controlled_exposure_intermittency_regime",
        "master_seed": int(seed),
        "stage_seed": int(stage_seed),
        "inputs": {
            "T": T,
            "N_normal": Nn,
            "N_coalition": Nc,
            "p_normal": pn,
            "R_grid": R_grid,
            "p_on_grid": p_grid,
            "attack": attack,
        },
        "outputs": {
            "grid_rows": "data/controlled_exposure_intermittency_rows.json",
            "core_arrays": "data/controlled_exposure_intermittency_core.npz",
        },
        "implementation": {
            "common_random_normal_background_within_seed": True,
            "nested_campaign_states_across_p_on": True,
            "normal_account_matrix": "scipy CSR",
            "normal_histogram_sampling": "multinomial from exact binned reference",
            "primary_w1_null_baseline": "exact binomial-marginal expectation",
        },
        "invariants": {
            "grid_has_70_cells": len(rows_out) == len(R_grid) * len(p_grid),
            "normal_frequency_finite": bool(np.all(np.isfinite(normal_freq))),
            "all_D_finite": bool(np.all(np.isfinite(D))),
            "all_requested_cells_present": len({
                (r["requested_R_exp"], r["requested_p_on"]) for r in rows_out
            }) == len(rows_out),
        },
    }
    if not all(manifest["invariants"].values()):
        raise RuntimeError(f"Invariant failure: {manifest['invariants']}")

    return metrics, manifest
