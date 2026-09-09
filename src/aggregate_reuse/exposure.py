from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ExposureRecord:
    """
    Sparse exposure record for one account in one interval.

    account_id is deliberately opaque: the exposure layer consumes IDs only
    after interval evidence has already been fixed.
    """
    account_id: str
    count: int = 1

    def __post_init__(self):
        if not self.account_id:
            raise ValueError("account_id cannot be empty.")
        if int(self.count) != self.count or self.count <= 0:
            raise ValueError("Exposure count must be a positive integer.")


def validate_interval_evidence(d_t: Sequence[float]) -> np.ndarray:
    d = np.asarray(d_t, dtype=float)
    if d.ndim != 1:
        raise ValueError("d_t must be one-dimensional.")
    if len(d) == 0:
        raise ValueError("d_t cannot be empty.")
    if not np.all(np.isfinite(d)):
        raise ValueError("d_t contains non-finite values.")
    return d


def validate_sparse_exposures(
    exposures_by_interval: Sequence[Sequence[ExposureRecord]],
    T: int,
) -> None:
    if len(exposures_by_interval) != T:
        raise ValueError(
            f"Expected exposure records for {T} intervals, "
            f"received {len(exposures_by_interval)}."
        )
    for t, records in enumerate(exposures_by_interval):
        seen = set()
        for rec in records:
            if not isinstance(rec, ExposureRecord):
                raise TypeError(
                    f"Interval {t} contains a non-ExposureRecord object."
                )
            if rec.account_id in seen:
                raise ValueError(
                    f"Duplicate account '{rec.account_id}' in interval {t}. "
                    "Aggregate multiple actions into one count."
                )
            seen.add(rec.account_id)


def accumulate_scores_sparse(
    d_t: Sequence[float],
    exposures_by_interval: Sequence[Sequence[ExposureRecord]],
    *,
    return_trajectories: bool = False,
    account_order: Sequence[str] | None = None,
):
    """
    Compute S_u(T) = sum_t a_{u,t} d_t from sparse interval exposures.

    Parameters
    ----------
    d_t:
        Already-fixed interval evidence.
    exposures_by_interval:
        One sparse list per interval. The function never computes or changes
        d_t; it only transfers fixed evidence to exposed accounts.
    return_trajectories:
        If True, also return cumulative score trajectories shaped
        (n_accounts, T).
    account_order:
        Optional deterministic account order. If omitted, accounts are
        sorted lexicographically after discovery.

    Returns
    -------
    scores : dict[str, float]
    trajectories : optional tuple(list[str], np.ndarray)
    """
    d = validate_interval_evidence(d_t)
    T = len(d)
    validate_sparse_exposures(exposures_by_interval, T)

    discovered = set()
    for records in exposures_by_interval:
        for rec in records:
            discovered.add(rec.account_id)

    if account_order is None:
        accounts = sorted(discovered)
    else:
        accounts = list(account_order)
        if len(accounts) != len(set(accounts)):
            raise ValueError("account_order contains duplicates.")
        missing = discovered.difference(accounts)
        if missing:
            raise ValueError(
                f"account_order omits discovered accounts: {sorted(missing)[:5]}"
            )

    scores: Dict[str, float] = {u: 0.0 for u in accounts}

    if not return_trajectories:
        for t, records in enumerate(exposures_by_interval):
            dt = float(d[t])
            for rec in records:
                scores[rec.account_id] += rec.count * dt
        return scores

    index = {u: i for i, u in enumerate(accounts)}
    increments = np.zeros((len(accounts), T), dtype=float)

    for t, records in enumerate(exposures_by_interval):
        dt = float(d[t])
        for rec in records:
            increments[index[rec.account_id], t] += rec.count * dt

    trajectories = np.cumsum(increments, axis=1)
    if len(accounts):
        final = trajectories[:, -1]
        for i, u in enumerate(accounts):
            scores[u] = float(final[i])

    return scores, (accounts, trajectories)


def accumulate_scores_dense(
    d_t: Sequence[float],
    exposure_matrix: Sequence[Sequence[float]],
) -> np.ndarray:
    """
    Dense reference implementation.

    exposure_matrix has shape (n_accounts, T), with nonnegative
    count-valued exposures. Returns final account scores.
    """
    d = validate_interval_evidence(d_t)
    A = np.asarray(exposure_matrix, dtype=float)
    if A.ndim != 2:
        raise ValueError("exposure_matrix must be two-dimensional.")
    if A.shape[1] != len(d):
        raise ValueError("Exposure matrix interval dimension must match d_t.")
    if not np.all(np.isfinite(A)) or np.any(A < 0):
        raise ValueError("Exposure matrix must be finite and nonnegative.")
    return A @ d


def group_scores(
    scores: Mapping[str, float],
    account_ids: Sequence[str],
) -> np.ndarray:
    """
    Extract a deterministic score vector for a declared account set.
    """
    values = []
    for u in account_ids:
        if u not in scores:
            raise KeyError(f"Account '{u}' does not have a score.")
        values.append(float(scores[u]))
    return np.asarray(values, dtype=float)
