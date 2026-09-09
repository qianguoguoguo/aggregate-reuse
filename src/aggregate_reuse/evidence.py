from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class HistogramSpec:
    support_min: float
    support_max: float
    n_bins: int

    def __post_init__(self):
        if not np.isfinite(self.support_min) or not np.isfinite(self.support_max):
            raise ValueError("Histogram support bounds must be finite.")
        if self.support_max <= self.support_min:
            raise ValueError("support_max must exceed support_min.")
        if int(self.n_bins) != self.n_bins or self.n_bins <= 0:
            raise ValueError("n_bins must be a positive integer.")

    @property
    def edges(self) -> np.ndarray:
        return np.linspace(self.support_min, self.support_max, self.n_bins + 1)

    @property
    def centers(self) -> np.ndarray:
        e = self.edges
        return 0.5 * (e[:-1] + e[1:])

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)


def _as_probability_vector(x: Sequence[float], *, atol: float = 1e-12) -> np.ndarray:
    p = np.asarray(x, dtype=float)
    if p.ndim != 1:
        raise ValueError("Probability vector must be one-dimensional.")
    if len(p) == 0:
        raise ValueError("Probability vector cannot be empty.")
    if not np.all(np.isfinite(p)):
        raise ValueError("Probability vector contains non-finite values.")
    if np.any(p < -atol):
        raise ValueError("Probability vector contains negative entries.")
    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    if total <= 0:
        raise ValueError("Probability vector has zero total mass.")
    p = p / total
    return p


def histogram_from_actions(
    actions: Iterable[float],
    spec: HistogramSpec,
    *,
    clip: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Convert scalar actions to a normalized histogram.

    The frozen controlled design clips to [support_min, support_max] before
    histogramming. With clip=False, actions outside the support raise.
    """
    x = np.asarray(list(actions), dtype=float)
    if x.ndim != 1:
        raise ValueError("Actions must form a one-dimensional array.")
    if len(x) == 0:
        raise ValueError("Cannot form a histogram from zero actions.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Actions contain non-finite values.")

    if clip:
        x = np.clip(x, spec.support_min, spec.support_max)
    else:
        if np.any(x < spec.support_min) or np.any(x > spec.support_max):
            raise ValueError("Action falls outside histogram support.")

    counts, _ = np.histogram(x, bins=spec.edges)
    n = int(counts.sum())
    if n != len(x):
        raise RuntimeError("Histogram count mismatch.")
    return counts.astype(float) / n, n


def project_reference_to_bins(
    reference_mass: Sequence[float],
    target_bins: int,
) -> np.ndarray:
    """
    Project a discrete reference defined on ordered equal-width bins to a
    coarser equal-width resolution.

    Exact aggregation is required: the source bin count must be divisible by
    target_bins. This prevents silent interpolation from changing mass.
    """
    p = _as_probability_vector(reference_mass)
    source_bins = len(p)
    if target_bins <= 0:
        raise ValueError("target_bins must be positive.")
    if source_bins == target_bins:
        return p.copy()
    if source_bins % target_bins != 0:
        raise ValueError(
            f"Cannot exactly aggregate {source_bins} bins to {target_bins}; "
            "source bin count must be divisible by target bin count."
        )
    factor = source_bins // target_bins
    return p.reshape(target_bins, factor).sum(axis=1)


def wasserstein1_ordered_hist(
    p: Sequence[float],
    q: Sequence[float],
    *,
    bin_positions: Optional[Sequence[float]] = None,
    bin_widths: Optional[Sequence[float]] = None,
) -> float:
    """
    Exact 1-D W1 for probability masses on common ordered support.

    For equally spaced bins, provide bin_positions or constant bin_widths.
    With neither argument, unit-spaced support {0,1,...,h-1} is used.

    W1 = sum_{k=1}^{h-1} |CDF_p(k)-CDF_q(k)| * gap_k
    """
    p = _as_probability_vector(p)
    q = _as_probability_vector(q)
    if len(p) != len(q):
        raise ValueError("p and q must have the same number of bins.")
    h = len(p)
    if h == 1:
        return 0.0

    if bin_positions is not None and bin_widths is not None:
        raise ValueError("Provide bin_positions or bin_widths, not both.")

    if bin_positions is not None:
        pos = np.asarray(bin_positions, dtype=float)
        if pos.ndim != 1 or len(pos) != h:
            raise ValueError("bin_positions must have one position per bin.")
        if not np.all(np.isfinite(pos)) or np.any(np.diff(pos) <= 0):
            raise ValueError("bin_positions must be finite and strictly increasing.")
        gaps = np.diff(pos)
    elif bin_widths is not None:
        widths = np.asarray(bin_widths, dtype=float)
        if widths.ndim != 1 or len(widths) not in (h, h - 1):
            raise ValueError("bin_widths must have length h or h-1.")
        if not np.all(np.isfinite(widths)) or np.any(widths <= 0):
            raise ValueError("bin_widths must be finite and positive.")
        if len(widths) == h:
            # Adjacent bin-center gaps for contiguous bins.
            gaps = 0.5 * (widths[:-1] + widths[1:])
        else:
            gaps = widths
    else:
        gaps = np.ones(h - 1, dtype=float)

    cdf_diff = np.cumsum(p - q)[:-1]
    return float(np.sum(np.abs(cdf_diff) * gaps))


def _baseline_cache_key(
    reference: np.ndarray,
    n: int,
    bin_positions: np.ndarray,
    draws: int,
    seed: int,
) -> str:
    payload = {
        "reference": np.asarray(reference, dtype=float).round(17).tolist(),
        "n": int(n),
        "bin_positions": np.asarray(bin_positions, dtype=float).round(17).tolist(),
        "draws": int(draws),
        "seed": int(seed),
        "version": 1,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class NullBaselineCache:
    """
    Deterministic on-disk cache for matched finite-sample null expectations.
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for_key(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> Optional[dict]:
        path = self.path_for_key(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, payload: dict) -> None:
        path = self.path_for_key(key)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)


def matched_null_baseline_w1(
    reference_mass: Sequence[float],
    n: int,
    *,
    bin_positions: Sequence[float],
    draws: int,
    seed: int,
    cache: Optional[NullBaselineCache] = None,
) -> Tuple[float, dict]:
    """
    Monte Carlo estimate of

      E[ W1(N/n, reference) ], N ~ Multinomial(n, reference).

    Returns (baseline, metadata). The estimator is deterministic given all
    inputs, including seed.
    """
    p = _as_probability_vector(reference_mass)
    pos = np.asarray(bin_positions, dtype=float)
    if len(pos) != len(p):
        raise ValueError("bin_positions must match reference bins.")
    if n <= 0:
        raise ValueError("n must be positive.")
    if draws <= 0:
        raise ValueError("draws must be positive.")

    key = _baseline_cache_key(p, n, pos, draws, seed)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return float(hit["baseline"]), {**hit, "cache_hit": True}

    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, p, size=draws)
    empirical = counts / float(n)

    gaps = np.diff(pos)
    cdf_diff = np.cumsum(empirical - p[None, :], axis=1)[:, :-1]
    w1 = np.sum(np.abs(cdf_diff) * gaps[None, :], axis=1)

    baseline = float(np.mean(w1))
    se = float(np.std(w1, ddof=1) / np.sqrt(draws)) if draws > 1 else 0.0

    payload = {
        "key": key,
        "baseline": baseline,
        "mc_standard_error": se,
        "draws": int(draws),
        "n": int(n),
        "seed": int(seed),
        "n_bins": int(len(p)),
        "cache_hit": False,
    }
    if cache is not None:
        cache.put(key, payload)

    return baseline, payload


def signed_w1_evidence(
    observed_hist: Sequence[float],
    reference_hist: Sequence[float],
    *,
    baseline: float,
    bin_positions: Sequence[float],
) -> float:
    if not np.isfinite(baseline) or baseline < 0:
        raise ValueError("baseline must be finite and nonnegative.")
    raw = wasserstein1_ordered_hist(
        observed_hist, reference_hist, bin_positions=bin_positions
    )
    return float(raw - baseline)


def controlled_reference_mass(
    spec: HistogramSpec,
    *,
    mean: float = 0.0,
    std: float = 1.0,
) -> np.ndarray:
    """
    Probability mass induced by a Normal(mean,std) after clipping values to
    the histogram support. Tail mass below/above support is assigned to the
    first/last bins, matching histogram_from_actions(..., clip=True).

    Requires scipy.
    """
    if std <= 0:
        raise ValueError("std must be positive.")
    try:
        from scipy.stats import norm
    except Exception as exc:
        raise RuntimeError("scipy is required for controlled_reference_mass") from exc

    edges = spec.edges
    z = (edges - mean) / std
    cdf = norm.cdf(z)
    mass = np.diff(cdf)
    mass[0] += cdf[0]
    mass[-1] += 1.0 - cdf[-1]
    return _as_probability_vector(mass)
def gaussian_mixture_reference_mass(
    spec: HistogramSpec,
    *,
    weights: Sequence[float],
    means: Sequence[float],
    stds: Sequence[float],
) -> np.ndarray:
    """
    Exact binned probability mass for a clipped Gaussian mixture.
    Tail mass is assigned to the first/last bins, matching clipping.
    """
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(means, dtype=float)
    sd = np.asarray(stds, dtype=float)
    if not (w.ndim == mu.ndim == sd.ndim == 1):
        raise ValueError("weights, means, and stds must be one-dimensional.")
    if not (len(w) == len(mu) == len(sd)) or len(w) == 0:
        raise ValueError("weights, means, and stds must have equal nonzero length.")
    if np.any(w < 0) or float(w.sum()) <= 0:
        raise ValueError("Mixture weights must be nonnegative with positive sum.")
    if np.any(sd <= 0):
        raise ValueError("Mixture standard deviations must be positive.")
    w = w / w.sum()
    out = np.zeros(spec.n_bins, dtype=float)
    for wi, mui, sdi in zip(w, mu, sd):
        out += wi * controlled_reference_mass(spec, mean=float(mui), std=float(sdi))
    return _as_probability_vector(out)

def exact_binomial_mean_absolute_deviation(n: int, p: float) -> float:
    """
    Exact E[|X/n - p|] for X ~ Binomial(n,p), up to floating-point
    evaluation of scipy's binomial CDF.

    If m=floor(np), then

        E|X/n-p|
        = 2 p [P(Bin(n,p) <= m) - P(Bin(n-1,p) <= m-1)].

    This follows from E[X-np]=0 and the truncated-binomial identity
    E[X 1{X<=m}] = np P(Bin(n-1,p)<=m-1).
    """
    if int(n) != n or n <= 0:
        raise ValueError("n must be a positive integer.")
    p = float(p)
    if not np.isfinite(p) or p < 0.0 or p > 1.0:
        raise ValueError("p must lie in [0,1].")
    if p == 0.0 or p == 1.0:
        return 0.0

    from scipy.stats import binom

    m = int(np.floor(n * p))
    F_n = float(binom.cdf(m, n, p))
    F_nm1 = 0.0 if m == 0 else float(binom.cdf(m - 1, n - 1, p))
    value = 2.0 * p * (F_n - F_nm1)

    # Protect against tiny negative roundoff at extreme p.
    if value < 0.0 and value > -1e-14:
        value = 0.0
    if value < 0.0 or not np.isfinite(value):
        raise RuntimeError(
            f"Numerically invalid binomial MAD for n={n}, p={p}: {value}"
        )
    return float(value)


def exact_null_baseline_w1(
    reference_mass: Sequence[float],
    n: int,
    *,
    bin_positions: Sequence[float],
) -> float:
    """
    Exact matched finite-sample null expectation of ordered 1-D W1.

    On ordered common support x_1 < ... < x_h,

        W1(H_n,p)
        = sum_{k=1}^{h-1} |F_n(k)-F(k)| (x_{k+1}-x_k).

    Under N ~ Multinomial(n,p), the cumulative count through boundary k is
    Binomial(n,F(k)). By linearity of expectation,

        E[W1(H_n,p)]
        = sum_k gap_k E|Binomial(n,F_k)/n - F_k|.

    No independence assumption between cumulative boundaries is required,
    because only linearity of expectation is used.
    """
    p = _as_probability_vector(reference_mass)
    pos = np.asarray(bin_positions, dtype=float)

    if int(n) != n or n <= 0:
        raise ValueError("n must be a positive integer.")
    if pos.ndim != 1 or len(pos) != len(p):
        raise ValueError("bin_positions must have one entry per reference bin.")
    if not np.all(np.isfinite(pos)) or np.any(np.diff(pos) <= 0):
        raise ValueError("bin_positions must be finite and strictly increasing.")
    if len(p) == 1:
        return 0.0

    cumulative_p = np.cumsum(p)[:-1]
    gaps = np.diff(pos)

    # Vectorized evaluation of the same closed form used by
    # exact_binomial_mean_absolute_deviation.
    from scipy.stats import binom
    interior = (cumulative_p > 0.0) & (cumulative_p < 1.0)
    terms = np.zeros_like(cumulative_p, dtype=float)
    F = cumulative_p[interior]
    m = np.floor(n * F).astype(int)
    cdf_n = binom.cdf(m, n, F)
    cdf_nm1 = binom.cdf(m - 1, n - 1, F)
    terms[interior] = 2.0 * F * (cdf_n - cdf_nm1)

    # Numerical guard.
    terms[(terms < 0.0) & (terms > -1e-14)] = 0.0
    if np.any(terms < 0.0) or not np.all(np.isfinite(terms)):
        raise RuntimeError("Invalid exact W1 null-baseline boundary term.")

    return float(np.dot(gaps, terms))


def exact_null_baseline_w1_cached(
    reference_mass: Sequence[float],
    n: int,
    *,
    bin_positions: Sequence[float],
    cache: Optional[NullBaselineCache] = None,
) -> Tuple[float, dict]:
    """Cached exact finite-sample W1 null expectation."""
    p = _as_probability_vector(reference_mass)
    pos = np.asarray(bin_positions, dtype=float)
    payload = {
        "reference": p.round(17).tolist(),
        "n": int(n),
        "bin_positions": pos.round(17).tolist(),
        "method": "exact_binomial_cdf",
        "version": 2,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()

    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return float(hit["baseline"]), {**hit, "cache_hit": True}

    baseline = exact_null_baseline_w1(
        p, int(n), bin_positions=pos
    )
    saved = {
        "key": key,
        "method": "exact_binomial_cdf",
        "baseline": float(baseline),
        "n": int(n),
        "n_bins": int(len(p)),
        "cache_hit": False,
    }
    if cache is not None:
        cache.put(key, saved)
    return float(baseline), saved
