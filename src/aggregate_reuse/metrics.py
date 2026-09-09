from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:
    roc_auc_score = None
    average_precision_score = None


def _finite_1d(x: Sequence[float], name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if len(a) == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values.")
    return a


def mean_score_gap(
    coalition_scores: Sequence[float],
    control_scores: Sequence[float],
) -> float:
    c = _finite_1d(coalition_scores, "coalition_scores")
    n = _finite_1d(control_scores, "control_scores")
    return float(np.mean(c) - np.mean(n))


def paired_score_gaps(
    coalition_scores: Sequence[float],
    placebo_scores: Sequence[float],
) -> np.ndarray:
    c = _finite_1d(coalition_scores, "coalition_scores")
    p = _finite_1d(placebo_scores, "placebo_scores")
    if len(c) != len(p):
        raise ValueError("Matched score vectors must have equal length.")
    return c - p


def matched_pair_misordering(
    coalition_scores: Sequence[float],
    placebo_scores: Sequence[float],
    *,
    ties_count_as_error: bool = True,
) -> float:
    gaps = paired_score_gaps(coalition_scores, placebo_scores)
    if ties_count_as_error:
        return float(np.mean(gaps <= 0.0))
    return float(np.mean(gaps < 0.0))


def coalition_placebo_auc(
    coalition_scores: Sequence[float],
    placebo_scores: Sequence[float],
) -> float:
    c = _finite_1d(coalition_scores, "coalition_scores")
    p = _finite_1d(placebo_scores, "placebo_scores")
    if roc_auc_score is None:
        raise RuntimeError("scikit-learn is required for ROC-AUC.")
    y = np.concatenate([np.ones(len(c), dtype=int), np.zeros(len(p), dtype=int)])
    s = np.concatenate([c, p])
    return float(roc_auc_score(y, s))


def coalition_placebo_average_precision(
    coalition_scores: Sequence[float],
    placebo_scores: Sequence[float],
) -> float:
    c = _finite_1d(coalition_scores, "coalition_scores")
    p = _finite_1d(placebo_scores, "placebo_scores")
    if average_precision_score is None:
        raise RuntimeError("scikit-learn is required for average precision.")
    y = np.concatenate([np.ones(len(c), dtype=int), np.zeros(len(p), dtype=int)])
    s = np.concatenate([c, p])
    return float(average_precision_score(y, s))


def cumulative_group_mean_gap(
    coalition_trajectories: Sequence[Sequence[float]],
    control_trajectories: Sequence[Sequence[float]],
) -> np.ndarray:
    c = np.asarray(coalition_trajectories, dtype=float)
    n = np.asarray(control_trajectories, dtype=float)
    if c.ndim != 2 or n.ndim != 2:
        raise ValueError("Trajectory inputs must be two-dimensional.")
    if c.shape[1] != n.shape[1]:
        raise ValueError("Groups must share the same horizon.")
    if c.shape[0] == 0 or n.shape[0] == 0:
        raise ValueError("Both groups must contain at least one account.")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(n)):
        raise ValueError("Trajectories contain non-finite values.")
    return np.mean(c, axis=0) - np.mean(n, axis=0)


def cumulative_matched_pair_misordering(
    coalition_trajectories: Sequence[Sequence[float]],
    placebo_trajectories: Sequence[Sequence[float]],
    *,
    ties_count_as_error: bool = True,
) -> np.ndarray:
    c = np.asarray(coalition_trajectories, dtype=float)
    p = np.asarray(placebo_trajectories, dtype=float)
    if c.ndim != 2 or p.ndim != 2 or c.shape != p.shape:
        raise ValueError("Matched trajectories must have identical 2-D shape.")
    gaps = c - p
    if ties_count_as_error:
        return np.mean(gaps <= 0.0, axis=0)
    return np.mean(gaps < 0.0, axis=0)


def fit_cumulative_slope(values: Sequence[float], start_t: int = 1) -> float:
    y = _finite_1d(values, "values")
    if start_t < 1:
        raise ValueError("start_t must be >= 1.")
    x = np.arange(start_t, start_t + len(y), dtype=float)
    return float(np.polyfit(x, y, deg=1)[0])


def percentile_bootstrap_ci(
    values: Sequence[float],
    *,
    level: float = 0.95,
    replicates: int = 10000,
    seed: int = 0,
    statistic: str = "mean",
) -> Dict[str, float]:
    """
    Percentile bootstrap over independent experimental units (seeds).

    This function intentionally accepts one scalar per independent unit.
    It should not be called on individual accounts when seeds are the
    declared statistical unit.
    """
    x = _finite_1d(values, "values")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1.")
    if replicates <= 0:
        raise ValueError("replicates must be positive.")
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'.")

    stat_fn = np.mean if statistic == "mean" else np.median
    estimate = float(stat_fn(x))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(replicates, len(x)))
    samples = x[idx]
    boot = stat_fn(samples, axis=1)

    alpha = 1.0 - level
    lower = float(np.quantile(boot, alpha / 2.0))
    upper = float(np.quantile(boot, 1.0 - alpha / 2.0))

    return {
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "level": float(level),
        "replicates": int(replicates),
        "n_units": int(len(x)),
        "statistic": statistic,
        "bootstrap_seed": int(seed),
    }


def summarize_seed_metrics(
    seed_rows: Sequence[Mapping[str, float]],
    *,
    metric_names: Sequence[str],
    level: float,
    replicates: int,
    bootstrap_seed: int,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate one scalar per seed for each declared metric.

    Each row must represent exactly one independent randomized seed.
    """
    rows = list(seed_rows)
    if not rows:
        raise ValueError("seed_rows cannot be empty.")

    out = {}
    for j, metric in enumerate(metric_names):
        vals = []
        for row in rows:
            if metric not in row:
                raise KeyError(f"Metric '{metric}' missing from a seed row.")
            vals.append(float(row[metric]))
        out[metric] = percentile_bootstrap_ci(
            vals,
            level=level,
            replicates=replicates,
            seed=int(bootstrap_seed) + j,
            statistic="mean",
        )
    return out
