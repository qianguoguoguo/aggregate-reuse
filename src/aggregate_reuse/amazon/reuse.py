"""Fixed-attack reuse incidence and account attribution.

Phase 2J migration from the validated Stage-4 implementation.
The attack world is supplied by Phase 2I and is NEVER regenerated here.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List

import numpy as np
from scipy.stats import rankdata

def regular_incidence(
    asins: List[str],
    m: int,
    r: int,
    rng: np.random.Generator,
) -> Dict[str, List[str]]:
    """
    Exact bipartite incidence:
      every item has degree m,
      every synthetic coalition account has degree r,
      no account appears twice on an item.

    Requires len(asins)*m divisible by r.
    """
    L = len(asins)
    E = L * m
    if E % r != 0:
        raise ValueError(f"L*m={E} is not divisible by r={r}")
    M = E // r
    if M < m:
        raise ValueError("Need at least m coalition accounts.")

    item_order = list(asins)
    rng.shuffle(item_order)

    # Circulant exact-regular construction over the shuffled item order.
    numeric = {}
    for j, asin in enumerate(item_order):
        numeric[asin] = [int((j * m + s) % M) for s in range(m)]

    # Exact row-degree and no duplicate account-item edges.
    degrees = np.zeros(M, dtype=int)
    for asin, ids in numeric.items():
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Duplicate coalition account within item {asin}")
        for a in ids:
            degrees[a] += 1
    if not np.all(degrees == r):
        raise RuntimeError(
            f"Regular-incidence invariant failed for r={r}: "
            f"degrees range {degrees.min()}..{degrees.max()}"
        )

    # Randomize visible account labels, preserving incidence exactly.
    perm = rng.permutation(M)
    out = {}
    for asin in asins:
        out[asin] = [
            f"syn_coal_r{r}_{int(perm[a]):06d}" for a in numeric[asin]
        ]
    return out


def auc_rank(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    scores_pos = np.asarray(scores_pos, dtype=float)
    scores_neg = np.asarray(scores_neg, dtype=float)
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return float("nan")
    values = np.concatenate([scores_pos, scores_neg])
    ranks = rankdata(values, method="average")
    n1 = len(scores_pos)
    n0 = len(scores_neg)
    rank_sum_pos = ranks[:n1].sum()
    u = rank_sum_pos - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0))


def strict_no_larger_probability(
    scores_pos: np.ndarray,
    scores_neg: np.ndarray,
) -> float:
    """
    P(S_positive <= S_negative), counting ties as no-larger.
    """
    pos = np.asarray(scores_pos, dtype=float)
    neg = np.sort(np.asarray(scores_neg, dtype=float))
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    total = 0
    for s in pos:
        left = np.searchsorted(neg, s, side="left")
        total += len(neg) - left
    return float(total / (len(pos) * len(neg)))


def summarize_accounts(
    account_score: Dict[str, float],
    account_freq: Dict[str, int],
    coalition_prefix: str = "syn_coal_",
):
    coalition = sorted(
        [a for a in account_score if a.startswith(coalition_prefix)]
    )
    normal = sorted(
        [a for a in account_score if not a.startswith(coalition_prefix)]
    )

    c_score = np.asarray([account_score[a] for a in coalition], dtype=float)
    n_score = np.asarray([account_score[a] for a in normal], dtype=float)
    c_freq = np.asarray([account_freq[a] for a in coalition], dtype=float)
    n_freq = np.asarray([account_freq[a] for a in normal], dtype=float)

    return {
        "coalition_accounts": coalition,
        "normal_accounts": normal,
        "coalition_scores": c_score,
        "normal_scores": n_score,
        "coalition_freq": c_freq,
        "normal_freq": n_freq,
        "score_auc": auc_rank(c_score, n_score),
        "frequency_auc": auc_rank(c_freq, n_freq),
        "score_no_larger_probability": strict_no_larger_probability(
            c_score, n_score
        ),
    }
