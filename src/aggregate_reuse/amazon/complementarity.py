from __future__ import annotations
import csv, gzip, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

def read_csv_gz(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def load_attack_world(path):
    out = {}
    for row in read_csv_gz(path):
        asin = row["asin"]
        out[asin] = {
            "d_cf": float(row["d_cf"]),
            "clean_w1": float(row["clean_w1"]),
            "attack_w1": float(row["attack_w1"]),
            "treatment_block": row["treatment_block"],
        }
    return out

def validate_regular_incidence(adj, *, asins, account_degree, item_degree):
    for u, items in adj.items():
        if len(items) != account_degree:
            raise RuntimeError(f"Account {u} degree {len(items)} != {account_degree}")
    counts = Counter()
    for items in adj.values():
        for asin in items:
            counts[asin] += 1
    if set(counts) != set(asins):
        raise RuntimeError("Item universe mismatch.")
    if any(counts[a] != item_degree for a in asins):
        raise RuntimeError("Item-degree invariant failed.")

def team_rich_incidence(asins, *, team_size, repetitions, rng):
    L = len(asins)
    if L % repetitions:
        raise ValueError("L must be divisible by repetitions.")
    n_teams = L // repetitions
    n_accounts = n_teams * team_size
    order = list(asins)
    rng.shuffle(order)
    adj = {u: set() for u in range(n_accounts)}
    pos = 0
    for team in range(n_teams):
        members = range(team*team_size, (team+1)*team_size)
        items = order[pos:pos+repetitions]
        pos += repetitions
        for u in members:
            adj[u].update(items)
    validate_regular_incidence(
        adj, asins=asins, account_degree=repetitions, item_degree=team_size
    )
    return adj

def degree_preserving_randomize(adj, *, rng, successful_swaps):
    out = {u: set(items) for u, items in adj.items()}
    edges = [(u, a) for u in sorted(out) for a in sorted(out[u])]
    E = len(edges)
    success = attempts = 0
    max_attempts = max(successful_swaps * 50, 10000)
    while success < successful_swaps and attempts < max_attempts:
        attempts += 1
        ia, ib = rng.integers(0, E, size=2)
        if ia == ib:
            continue
        u, i = edges[int(ia)]
        v, j = edges[int(ib)]
        if u == v or i == j or j in out[u] or i in out[v]:
            continue
        out[u].remove(i); out[v].remove(j)
        out[u].add(j); out[v].add(i)
        edges[int(ia)] = (u, j); edges[int(ib)] = (v, i)
        success += 1
    if success != successful_swaps:
        raise RuntimeError(f"Only {success}/{successful_swaps} swaps succeeded.")
    return out, {"successful_swaps": success, "attempts": attempts, "edges": E}

def coactivity_scores(adj):
    item_accounts = defaultdict(list)
    for u, items in adj.items():
        for asin in items:
            item_accounts[asin].append(u)
    pc = Counter()
    for users in item_accounts.values():
        users = sorted(users)
        for i in range(len(users)):
            for j in range(i+1, len(users)):
                pc[(users[i], users[j])] += 1
    s = np.zeros(len(adj), dtype=float)
    for (u,v), c in pc.items():
        x = max(c-1, 0)
        s[u] += x; s[v] += x
    return s

def aggregate_counterfactual_scores(adj, attack):
    out = np.zeros(len(adj), dtype=float)
    for u in sorted(adj):
        out[u] = sum(float(attack[a]["d_cf"]) for a in adj[u])
    return out

def auc_rank(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    vals = np.concatenate([pos, neg])
    ranks = rankdata(vals, method="average")
    n1, n0 = len(pos), len(neg)
    u = ranks[:n1].sum() - n1*(n1+1)/2
    return float(u/(n1*n0))

def pooled_z(pos, neg):
    pos = np.asarray(pos,float); neg = np.asarray(neg,float)
    pooled = np.concatenate([pos,neg])
    mu = float(pooled.mean()); sd = float(pooled.std(ddof=0))
    if sd == 0:
        return np.zeros_like(pos), np.zeros_like(neg), {"mean":mu,"std":sd}
    return (pos-mu)/sd, (neg-mu)/sd, {"mean":mu,"std":sd}

def incidence_edges(adj):
    return [(u,a) for u in sorted(adj) for a in sorted(adj[u])]

def repeated_pair_summary(adj):
    item_accounts = defaultdict(list)
    for u, items in adj.items():
        for a in items:
            item_accounts[a].append(u)
    pc = Counter()
    for users in item_accounts.values():
        users = sorted(users)
        for i in range(len(users)):
            for j in range(i+1,len(users)):
                pc[(users[i],users[j])] += 1
    return {
        "unique_account_pairs": len(pc),
        "repeated_account_pairs": sum(c>=2 for c in pc.values()),
        "max_pair_cooccurrence": max(pc.values()) if pc else 0,
        "mean_pair_cooccurrence": float(np.mean(list(pc.values()))) if pc else 0.0,
    }
