from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..exposure import ExposureRecord


@dataclass
class BalancedCyclicScheduler:
    account_ids: Sequence[str]
    pointer: int = 0

    def __post_init__(self):
        self.account_ids = list(self.account_ids)
        if len(self.account_ids) == 0:
            raise ValueError("Scheduler requires at least one account.")
        if len(set(self.account_ids)) != len(self.account_ids):
            raise ValueError("Scheduler account IDs must be unique.")
        self.pointer %= len(self.account_ids)

    def select(self, k: int) -> List[str]:
        if k < 0:
            raise ValueError("k must be nonnegative.")
        n = len(self.account_ids)
        if k > n:
            raise ValueError("k cannot exceed scheduler pool size.")
        selected = [self.account_ids[(self.pointer + j) % n] for j in range(k)]
        self.pointer = (self.pointer + k) % n
        return selected


def requested_k_on(
    *,
    R_exp: float,
    p_normal: float,
    N_coalition: int,
    p_on: float,
) -> int:
    if R_exp <= 0 or p_normal <= 0 or p_on <= 0 or N_coalition <= 0:
        raise ValueError("Exposure-ratio parameters must be positive.")
    k = int(round(R_exp * p_normal * N_coalition / p_on))
    if k < 1 or k > N_coalition:
        raise ValueError(
            f"Infeasible k_on={k} for N_coalition={N_coalition}. "
            "Adjust R_exp or p_on."
        )
    return k


def generate_controlled_campaign_schedule(
    *,
    T: int,
    N_normal: int,
    N_coalition: int,
    p_normal: float,
    p_on: float,
    R_exp: float,
    rng: np.random.Generator,
) -> Dict:
    """
    Generate campaign state and sparse exposure schedules.

    Normal accounts participate independently in each interval.
    Coalition accounts participate only in active intervals and are chosen
    by balanced cyclic rotation.

    Returns realized exposure statistics needed for theory validation.
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    if N_normal <= 0 or N_coalition <= 0:
        raise ValueError("Account-pool sizes must be positive.")
    if not 0 < p_normal <= 1:
        raise ValueError("p_normal must lie in (0,1].")
    if not 0 < p_on <= 1:
        raise ValueError("p_on must lie in (0,1].")

    normal_ids = [f"n{j:06d}" for j in range(N_normal)]
    coalition_ids = [f"c{j:06d}" for j in range(N_coalition)]

    I_t = rng.binomial(1, p_on, size=T).astype(int)
    k_on = requested_k_on(
        R_exp=R_exp,
        p_normal=p_normal,
        N_coalition=N_coalition,
        p_on=p_on,
    )
    scheduler = BalancedCyclicScheduler(coalition_ids)

    exposures_by_interval: List[List[ExposureRecord]] = []
    n_normal_t = np.empty(T, dtype=int)
    n_coalition_t = np.zeros(T, dtype=int)

    normal_exposure_counts = np.zeros(N_normal, dtype=int)
    coalition_exposure_counts = np.zeros(N_coalition, dtype=int)
    coalition_index = {u: i for i, u in enumerate(coalition_ids)}

    for t in range(T):
        # Independent Bernoulli participation for each normal account.
        mask = rng.random(N_normal) < p_normal
        normal_idx = np.flatnonzero(mask)
        n_normal_t[t] = len(normal_idx)
        normal_exposure_counts[normal_idx] += 1

        records = [ExposureRecord(normal_ids[j], 1) for j in normal_idx]

        if I_t[t] == 1:
            selected = scheduler.select(k_on)
            n_coalition_t[t] = k_on
            for u in selected:
                coalition_exposure_counts[coalition_index[u]] += 1
                records.append(ExposureRecord(u, 1))

        exposures_by_interval.append(records)

    realized_normal_rate = float(normal_exposure_counts.sum()) / (N_normal * T)
    realized_coalition_rate = float(coalition_exposure_counts.sum()) / (N_coalition * T)
    realized_R_exp = realized_coalition_rate / realized_normal_rate

    return {
        "I_t": I_t,
        "k_on": int(k_on),
        "normal_ids": normal_ids,
        "coalition_ids": coalition_ids,
        "exposures_by_interval": exposures_by_interval,
        "normal_exposure_counts": normal_exposure_counts,
        "coalition_exposure_counts": coalition_exposure_counts,
        "n_normal_t": n_normal_t,
        "n_coalition_t": n_coalition_t,
        "realized_normal_rate": realized_normal_rate,
        "realized_coalition_rate": realized_coalition_rate,
        "realized_R_exp": realized_R_exp,
        "realized_p_on": float(I_t.mean()),
    }


def coalition_action_sample(
    rng: np.random.Generator,
    n: int,
    distribution_cfg: Dict,
) -> np.ndarray:
    family = distribution_cfg["family"]
    if family == "normal":
        x = rng.normal(
            loc=float(distribution_cfg["mean"]),
            scale=float(distribution_cfg["std"]),
            size=n,
        )
    elif family == "symmetric_gaussian_mixture":
        weights = np.asarray(distribution_cfg["weights"], dtype=float)
        means = np.asarray(distribution_cfg["component_means"], dtype=float)
        stds = np.asarray(distribution_cfg["component_stds"], dtype=float)
        if not (len(weights) == len(means) == len(stds)):
            raise ValueError("Mixture weights/means/stds must have equal length.")
        weights = weights / weights.sum()
        z = rng.choice(len(weights), size=n, p=weights)
        x = rng.normal(loc=means[z], scale=stds[z])
    else:
        raise ValueError(f"Unsupported coalition distribution family: {family}")

    clip = distribution_cfg.get("clipping")
    if clip is not None:
        x = np.clip(x, float(clip[0]), float(clip[1]))
    return x
