from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.special import betaln, gammaln
from scipy.stats import rankdata

STAR = np.asarray([1.,2.,3.,4.,5.])

def load_roles(path):
    rows = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            asin = str(obj["asin"])
            rows[asin] = obj["blocks"]
    return rows

def counts_from_reviews(reviews):
    c = np.zeros(5, dtype=int)
    for r in reviews:
        v = int(r["rating"])
        if v < 1 or v > 5:
            raise RuntimeError(f"Unsupported rating {v}")
        c[v-1] += 1
    return c

def load_attack_world(path):
    out = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            out[row["asin"]] = {
                "clean_counts": np.asarray(json.loads(row["clean_counts"]), dtype=int),
                "attack_counts": np.asarray(json.loads(row["attack_counts"]), dtype=int),
                "d_cf_stage4": float(row["d_cf"]),
            }
    return out

def load_r8_assignment(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            rows.append({
                "asin": row["asin"],
                "account": row["synthetic_account_id"],
            })
    return rows

def w1_counts(counts, q):
    p = np.asarray(counts, float)
    p /= p.sum()
    q = np.asarray(q, float)
    return float(np.abs(np.cumsum(p-q)[:-1]).sum())

def exact_predictive_w1_baseline(alpha, n=30):
    alpha = np.asarray(alpha, float)
    a0 = float(alpha.sum())
    q = alpha / a0
    A = np.cumsum(alpha)[:-1]
    F = np.cumsum(q)[:-1]
    x = np.arange(n+1, dtype=float)
    log_choose = gammaln(n+1)-gammaln(x+1)-gammaln(n-x+1)
    out = 0.0
    for a,f in zip(A,F):
        b = a0-a
        if a <= 0 or b <= 0:
            continue
        logpmf = log_choose + betaln(x+a, n-x+b) - betaln(a,b)
        pmf = np.exp(logpmf)
        pmf /= pmf.sum()
        out += float(np.sum(np.abs(x/n-f)*pmf))
    return out

def auc_rank(pos, neg):
    pos=np.asarray(pos,float); neg=np.asarray(neg,float)
    vals=np.concatenate([pos,neg])
    ranks=rankdata(vals,method="average")
    n1,n0=len(pos),len(neg)
    u=ranks[:n1].sum()-n1*(n1+1)/2
    return float(u/(n1*n0))
