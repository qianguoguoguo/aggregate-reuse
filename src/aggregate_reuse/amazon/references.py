from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.special import betaln, gammaln


STAR_RATINGS = np.asarray([1., 2., 3., 4., 5.], dtype=float)
LAMBDA_GRID = [0., 5., 10., 20., 40., 80., 160.]


def w1_hist_1to5(counts: np.ndarray, q: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    q = np.asarray(q, dtype=float)
    p = counts / counts.sum()
    return float(np.abs(np.cumsum(p - q)[:-1]).sum())


def exact_plugin_w1_baseline(q: np.ndarray, n: int) -> float:
    q = np.asarray(q, dtype=float)
    F = np.cumsum(q)[:-1]
    out = 0.0
    ks = np.arange(n + 1)
    log_choose = gammaln(n + 1) - gammaln(ks + 1) - gammaln(n - ks + 1)

    for p in F:
        if p <= 0.0 or p >= 1.0:
            continue
        logpmf = log_choose + ks * np.log(p) + (n - ks) * np.log1p(-p)
        pmf = np.exp(logpmf)
        out += float(np.sum(np.abs(ks / n - p) * pmf))
    return out


def exact_predictive_w1_baseline(alpha: np.ndarray, n: int) -> float:
    alpha = np.asarray(alpha, dtype=float)
    if np.any(alpha < 0):
        raise ValueError("alpha must be nonnegative")

    a0 = float(alpha.sum())
    if a0 <= 0:
        raise ValueError("alpha must have positive total mass")

    qbar = alpha / a0
    A = np.cumsum(alpha)[:-1]
    F = np.cumsum(qbar)[:-1]

    x = np.arange(n + 1, dtype=float)
    log_choose = gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1)

    out = 0.0
    for a, f in zip(A, F):
        b = a0 - a
        if a <= 0.0 or b <= 0.0:
            continue
        logpmf = (
            log_choose
            + betaln(x + a, n - x + b)
            - betaln(a, b)
        )
        pmf = np.exp(logpmf)
        pmf /= pmf.sum()
        out += float(np.sum(np.abs(x / n - f) * pmf))
    return out


def shrunk_reference(
    item_ref_counts: np.ndarray,
    loo_probs: np.ndarray,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    c = np.asarray(item_ref_counts, dtype=float)
    loo = np.asarray(loo_probs, dtype=float)
    alpha = c + float(lam) * loo
    q = alpha / alpha.sum()
    return alpha, q


def load_reference_table(path: str | Path):
    rows = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            asin = row["asin"]
            ref = np.asarray(
                [float(row[f"reference_count_{r}"]) for r in range(1, 6)],
                dtype=float,
            )
            loo = np.asarray(
                [float(row[f"loo_domain_prob_{r}"]) for r in range(1, 6)],
                dtype=float,
            )
            if abs(ref.sum() - 120.0) > 1e-9:
                raise RuntimeError(f"{asin}: reference count sum != 120")
            if abs(loo.sum() - 1.0) > 1e-9:
                raise RuntimeError(f"{asin}: LOO probabilities do not sum to 1")
            rows[asin] = (ref, loo)
    return rows


def block_counts(block: List[Dict]) -> np.ndarray:
    out = np.zeros(5, dtype=int)
    for r in block:
        rating = int(r["rating"])
        if rating < 1 or rating > 5:
            raise RuntimeError(f"Unsupported rating in Stage 2: {rating}")
        out[rating - 1] += 1
    return out


def load_stage3_blocks(path: str | Path):
    wanted = [
        "calibration_1",
        "calibration_2",
        "holdout_1",
        "holdout_2",
    ]
    rows = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            asin = str(obj["asin"])
            blocks = obj["blocks"]
            d = {}
            for role in wanted:
                c = block_counts(blocks[role])
                if c.sum() != 30:
                    raise RuntimeError(f"{asin}/{role}: expected 30 reviews.")
                d[role] = c
            rows[asin] = d
    return rows


def item_cluster_bootstrap_mean(
    values: np.ndarray,
    *,
    replicates: int = 10000,
    seed: int = 314159,
    level: float = 0.95,
):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must have shape (n_items, n_blocks)")

    n = values.shape[0]
    rng = np.random.default_rng(seed)
    item_means = values.mean(axis=1)
    estimate = float(item_means.mean())

    boots = np.empty(replicates, dtype=float)
    chunk = 200
    pos = 0
    while pos < replicates:
        m = min(chunk, replicates - pos)
        idx = rng.integers(0, n, size=(m, n))
        boots[pos:pos+m] = item_means[idx].mean(axis=1)
        pos += m

    alpha = (1.0 - level) / 2.0
    return {
        "mean": estimate,
        "ci_lower": float(np.quantile(boots, alpha)),
        "ci_upper": float(np.quantile(boots, 1.0 - alpha)),
        "n_items": int(n),
        "replicates": int(replicates),
        "level": float(level),
    }
