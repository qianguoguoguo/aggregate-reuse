"""Dedicated same-k=6 feasibility-population audit.

Purpose
-------
The paper's primary k=6 experiment and the fixed-identity strength experiment
operate on different feasible item populations. This audit holds the
intervention itself at k=6 and changes only the population eligibility rule:

  P6: both experimental blocks have >=6 non-five-star reviews
  P9: both experimental blocks have >=9 non-five-star reviews

Two outputs are produced:

1. Exact full-population expected mean d_cf, integrating exactly over the
   randomized A/B treatment choice and uniform k=6 donor selection.
2. A 30-seed finite-sample matched-twin audit with item degree 6 and r=8.

The exact expectation is deterministic and does not depend on Monte Carlo.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from itertools import product
from math import comb
from typing import Dict, List

import numpy as np
from scipy.stats import rankdata

STAR_RATINGS = (1,2,3,4,5)


def hist(block):
    c = np.zeros(5, dtype=int)
    for rev in block:
        c[int(rev["rating"])-1] += 1
    return c


def shrunk_reference(ref_counts, loo_probs, lam):
    alpha = np.asarray(ref_counts, float) + float(lam)*np.asarray(loo_probs,float)
    return alpha/alpha.sum()


def w1_counts(counts, q):
    p = np.asarray(counts,float)
    p /= p.sum()
    q = np.asarray(q,float)
    return float(np.abs(np.cumsum(p-q)[:-1]).sum())


def auc_rank(pos, neg):
    pos=np.asarray(pos,float); neg=np.asarray(neg,float)
    vals=np.concatenate([pos,neg])
    ranks=rankdata(vals,method="average")
    n1,n0=len(pos),len(neg)
    u=ranks[:n1].sum()-n1*(n1+1)/2
    return float(u/(n1*n0))


def feasible(block, threshold):
    return sum(int(x["rating"]) != 5 for x in block) >= int(threshold)


def regular_incidence(asins, m, r, rng):
    L=len(asins); E=L*m
    if E % r:
        raise ValueError("L*m must be divisible by r")
    M=E//r
    order=list(asins)
    rng.shuffle(order)
    numeric={}
    for j,asin in enumerate(order):
        numeric[asin]=[int((j*m+s)%M) for s in range(m)]

    deg=np.zeros(M,dtype=int)
    for asin,ids in numeric.items():
        if len(ids)!=len(set(ids)):
            raise RuntimeError("duplicate account-item edge")
        for a in ids:
            deg[a]+=1
    if not np.all(deg==r):
        raise RuntimeError("account-degree invariant failed")

    perm=rng.permutation(M)
    return {
        asin:[f"syn_k6audit_r{r}_{int(perm[a]):06d}" for a in numeric[asin]]
        for asin in asins
    }


def donor_count_vectors(nonfive_counts, k):
    """Enumerate all rating-count donor compositions for ratings 1..4."""
    n1,n2,n3,n4 = [int(x) for x in nonfive_counts]
    for x1 in range(min(n1,k)+1):
        for x2 in range(min(n2,k-x1)+1):
            for x3 in range(min(n3,k-x1-x2)+1):
                x4=k-x1-x2-x3
                if 0 <= x4 <= n4:
                    yield (x1,x2,x3,x4)


def exact_expected_block_dcf(clean_counts, q, k=6):
    """Exact E[W1(attack,q)-W1(clean,q)] under uniform donor slots."""
    clean=np.asarray(clean_counts,dtype=int)
    nonfive=clean[:4]
    N=int(nonfive.sum())
    if N < k:
        raise ValueError("block not k-feasible")

    denom=comb(N,k)
    clean_w1=w1_counts(clean,q)
    expected=0.0

    for x in donor_count_vectors(nonfive,k):
        ways=1
        for n,xx in zip(nonfive,x):
            ways *= comb(int(n), int(xx))
        if ways == 0:
            continue
        attack=clean.copy()
        for j,xx in enumerate(x):
            attack[j]-=xx
        attack[4]+=k
        expected += (ways/denom)*(w1_counts(attack,q)-clean_w1)

    return float(expected)


def exact_population_expected_mean(blocks, refs, lam, threshold, k=6):
    """Average exact k=6 d_cf expectation over full eligible population."""
    values=[]
    for asin in sorted(blocks):
        if asin not in refs:
            continue
        A=blocks[asin]["experimental_A"]
        B=blocks[asin]["experimental_B"]
        if not (feasible(A,threshold) and feasible(B,threshold)):
            continue

        c,loo=refs[asin]
        q=shrunk_reference(c,loo,lam)
        ea=exact_expected_block_dcf(hist(A),q,k)
        eb=exact_expected_block_dcf(hist(B),q,k)
        values.append(0.5*(ea+eb))

    return {
        "population_size": len(values),
        "exact_expected_mean_d_cf": float(np.mean(values)),
    }


def build_sample(seed, blocks, refs, lam, threshold, *, L=2000, k=6,
                 attack_seed_offset=900000):
    """Sample one k=6 attack world from a specified feasibility population."""
    rng=np.random.default_rng(int(seed)+int(attack_seed_offset))
    universe=sorted([
        asin for asin in blocks
        if asin in refs
        and feasible(blocks[asin]["experimental_A"],threshold)
        and feasible(blocks[asin]["experimental_B"],threshold)
    ])
    if len(universe) < L:
        raise RuntimeError("insufficient eligible items")

    chosen=sorted(rng.choice(universe,size=L,replace=False).tolist())
    rows={}

    for asin in chosen:
        c,loo=refs[asin]
        q=shrunk_reference(c,loo,lam)
        block_name="experimental_A" if rng.random()<0.5 else "experimental_B"
        block=blocks[asin][block_name]
        cand=[j for j,rev in enumerate(block) if int(rev["rating"]) != 5]
        idx=rng.choice(cand,size=k,replace=False).tolist()

        clean=hist(block)
        attack=clean.copy()
        for j in idx:
            old=int(block[j]["rating"])
            attack[old-1]-=1
            attack[4]+=1

        rows[asin]={
            "d_cf": w1_counts(attack,q)-w1_counts(clean,q),
            "block": block_name,
        }

    return universe, rows


def matched_twin_auc(seed, rows, *, k=6, r=8):
    asins=sorted(rows)
    # Keep the incidence independent of the feasibility population's attack RNG.
    rng=np.random.default_rng(int(seed)*100003 + int(r)*997 + 41)
    inc=regular_incidence(asins,k,r,rng)

    score=defaultdict(float)
    for asin in asins:
        d=float(rows[asin]["d_cf"])
        for a in inc[asin]:
            score[a]+=d

    vals=np.asarray([score[a] for a in sorted(score)],float)
    zeros=np.zeros_like(vals)
    return {
        "n_accounts": len(vals),
        "mean_d_cf": float(np.mean([rows[a]["d_cf"] for a in asins])),
        "counterfactual_score_auc": auc_rank(vals,zeros),
        "paired_misordering_probability": float(np.mean(vals <= 0)),
        "mean_paired_score_gap": float(vals.mean()),
    }
