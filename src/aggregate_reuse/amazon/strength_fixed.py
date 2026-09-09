from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

STAR = np.asarray([1., 2., 3., 4., 5.])


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_reference_table(path):
    out = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            asin = row["asin"]
            c = np.asarray(
                [float(row[f"reference_count_{r}"]) for r in range(1, 6)]
            )
            loo = np.asarray(
                [float(row[f"loo_domain_prob_{r}"]) for r in range(1, 6)]
            )
            out[asin] = (c, loo)
    return out


def load_experimental_blocks(path):
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            out[str(obj["asin"])] = {
                "experimental_A": obj["blocks"]["experimental_A"],
                "experimental_B": obj["blocks"]["experimental_B"],
            }
    return out


def shrunk_reference(c, loo, lam):
    alpha = np.asarray(c, float) + float(lam) * np.asarray(loo, float)
    return alpha / alpha.sum()


def hist(block):
    c = np.zeros(5, dtype=int)
    for r in block:
        v = int(r["rating"])
        if v < 1 or v > 5:
            raise RuntimeError(f"Unsupported rating {v}")
        c[v - 1] += 1
    return c


def w1_counts(c, q):
    p = np.asarray(c, float)
    p /= p.sum()
    q = np.asarray(q, float)
    return float(np.abs(np.cumsum(p - q)[:-1]).sum())


def auc_rank(pos, neg):
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    vals = np.concatenate([pos, neg])
    ranks = rankdata(vals, method="average")
    n1, n0 = len(pos), len(neg)
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0))


def fixed_nested_incidence(asins, item_degree, reuse_r, rng):
    """
    Construct ONE fixed exact-regular account-item incidence and a nested
    donor-rank -> account assignment.

    Requirements for the current experiment:
      L = 2000 items
      item_degree = 9
      reuse_r = 8
      M = L * item_degree / reuse_r = 2250 accounts

    Each item has exactly 9 synthetic accounts; each account appears on exactly
    8 items.  Each item's nine incident edges are assigned ranks 0,...,8.

    Rank construction:
      numeric account on shuffled-item position j and local edge s:
          a = (j * item_degree + s) mod M
      donor rank:
          rank = (s + j) mod item_degree

    For L=2000, item_degree=9, reuse_r=8, each account receives eight distinct
    ranks among 0,...,8.  Therefore:
      k=3 -> each account has 2 or 3 modified exposures
      k=6 -> each account has 5 or 6 modified exposures
      k=9 -> each account has exactly 8 modified exposures

    This gives a balanced nested intervention while preserving the SAME
    identity population and SAME account-item exposures for every k.
    """
    L = len(asins)
    m = int(item_degree)
    r = int(reuse_r)
    E = L * m
    if E % r:
        raise ValueError(f"L*item_degree={E} not divisible by reuse_r={r}")
    M = E // r

    order = list(asins)
    rng.shuffle(order)

    # numeric_by_rank[asin][rank] = numeric account id
    numeric_by_rank = {}
    account_degree = np.zeros(M, dtype=int)

    for j, asin in enumerate(order):
        rank_to_account = [None] * m
        seen_accounts = set()
        for s in range(m):
            a = int((j * m + s) % M)
            rank = int((s + j) % m)
            if a in seen_accounts:
                raise RuntimeError("Duplicate account within item.")
            if rank_to_account[rank] is not None:
                raise RuntimeError("Duplicate donor rank within item.")
            rank_to_account[rank] = a
            seen_accounts.add(a)
            account_degree[a] += 1
        if any(x is None for x in rank_to_account):
            raise RuntimeError("Missing donor rank within item.")
        numeric_by_rank[asin] = rank_to_account

    if not np.all(account_degree == r):
        raise RuntimeError("Fixed incidence account-degree failure.")

    # Randomize visible identity labels without changing the graph/rank structure.
    perm = rng.permutation(M)
    account_by_rank = {
        asin: [f"syn_strength_fixed_r{r}_{int(perm[a]):06d}" for a in numeric_by_rank[asin]]
        for asin in asins
    }

    # Verify exact item degree and exact account degree after relabeling.
    deg = defaultdict(int)
    for asin in asins:
        ids = account_by_rank[asin]
        if len(ids) != m or len(set(ids)) != m:
            raise RuntimeError("Fixed incidence item-degree/simplicity failure.")
        for a in ids:
            deg[a] += 1
    if len(deg) != M or any(v != r for v in deg.values()):
        raise RuntimeError("Fixed incidence relabeled account-degree failure.")

    return account_by_rank


def modification_counts(account_by_rank, k):
    """Count how many of each account's fixed exposures are modified at strength k."""
    counts = defaultdict(int)
    for ids in account_by_rank.values():
        for rank, a in enumerate(ids):
            if rank < k:
                counts[a] += 1
    return counts
