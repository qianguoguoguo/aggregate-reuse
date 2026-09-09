from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import rankdata


STAR_VALUES = np.asarray([1., 2., 3., 4., 5.], dtype=float)

# Unordered pair -> more polarized unordered pair with identical sum.
PAIR_MAP = {
    (2, 2): (1, 3),
    (2, 3): (1, 4),
    (2, 4): (1, 5),
    (3, 3): (1, 5),
    (3, 4): (2, 5),
    (4, 4): (3, 5),
}


def load_json(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
            rows[asin] = (ref, loo)
    return rows


def load_experimental_blocks(path: str | Path):
    rows = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            asin = str(obj["asin"])
            rows[asin] = {
                "experimental_A": obj["blocks"]["experimental_A"],
                "experimental_B": obj["blocks"]["experimental_B"],
            }
    return rows


def shrunk_reference(ref_counts, loo_probs, lam):
    alpha = np.asarray(ref_counts, dtype=float) + float(lam) * np.asarray(
        loo_probs, dtype=float
    )
    return alpha / alpha.sum()


def hist_from_reviews(reviews: List[Dict]) -> np.ndarray:
    c = np.zeros(5, dtype=int)
    for r in reviews:
        v = int(r["rating"])
        if v < 1 or v > 5:
            raise RuntimeError(f"Unsupported rating {v}")
        c[v - 1] += 1
    return c


def mean_from_counts(counts: np.ndarray) -> float:
    c = np.asarray(counts, dtype=float)
    return float(np.dot(c, STAR_VALUES) / c.sum())


def w1_counts_to_reference(counts: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(counts, dtype=float)
    p /= p.sum()
    q = np.asarray(q, dtype=float)
    return float(np.abs(np.cumsum(p - q)[:-1]).sum())


def js_divergence_counts_to_reference(counts: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(counts, dtype=float)
    p /= p.sum()
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def abs_mean_discrepancy(counts: np.ndarray, q: np.ndarray) -> float:
    return abs(mean_from_counts(counts) - float(np.dot(q, STAR_VALUES)))


def candidate_pairs(block: List[Dict]):
    out = []
    n = len(block)
    for i in range(n):
        a = int(block[i]["rating"])
        for j in range(i + 1, n):
            b = int(block[j]["rating"])
            key = tuple(sorted((a, b)))
            if key in PAIR_MAP:
                out.append((i, j, key))
    return out


def find_three_disjoint_pairs(
    block: List[Dict],
    rng: Optional[np.random.Generator] = None,
):
    """
    Return three disjoint transformable pairs, or None.

    If rng is supplied, the candidate order is randomized. If rng is None,
    deterministic input order is used for feasibility testing.
    """
    cand = candidate_pairs(block)
    if rng is not None and len(cand) > 1:
        order = rng.permutation(len(cand))
        cand = [cand[int(k)] for k in order]

    def rec(start: int, chosen, used):
        if len(chosen) == 3:
            return list(chosen)
        remaining_needed = 3 - len(chosen)
        if len(cand) - start < remaining_needed:
            return None
        for k in range(start, len(cand)):
            i, j, key = cand[k]
            if i in used or j in used:
                continue
            chosen.append((i, j, key))
            used.add(i)
            used.add(j)
            ans = rec(k + 1, chosen, used)
            if ans is not None:
                return ans
            used.remove(i)
            used.remove(j)
            chosen.pop()
        return None

    return rec(0, [], set())


def block_is_feasible(block: List[Dict]) -> bool:
    return find_three_disjoint_pairs(block, rng=None) is not None


def apply_shape_intervention(
    block: List[Dict],
    rng: np.random.Generator,
):
    pairs = find_three_disjoint_pairs(block, rng=rng)
    if pairs is None:
        raise RuntimeError("Selected block is not shape-intervention feasible.")

    clean_counts = hist_from_reviews(block)
    attack_counts = clean_counts.copy()

    slots = []
    used = set()

    for i, j, key in pairs:
        if i in used or j in used:
            raise RuntimeError("Pair overlap invariant failed.")
        used.add(i)
        used.add(j)

        old = [int(block[i]["rating"]), int(block[j]["rating"])]
        new = list(PAIR_MAP[key])

        # Randomize orientation only; histogram and pair sum are unchanged.
        if rng.random() < 0.5:
            new = [new[1], new[0]]

        if sum(old) != sum(new):
            raise RuntimeError("Pair-sum preservation failed.")
        if old[0] == new[0] or old[1] == new[1]:
            # With the frozen mappings, every selected slot should change.
            raise RuntimeError("A selected rating did not change.")

        for idx, replacement in zip((i, j), new):
            original = int(block[idx]["rating"])
            attack_counts[original - 1] -= 1
            attack_counts[replacement - 1] += 1
            slots.append({
                "local_index": int(idx),
                "position": int(block[idx]["position"]),
                "source_line": int(block[idx]["source_line"]),
                "original_rating": original,
                "replacement_rating": int(replacement),
            })

    slots = sorted(slots, key=lambda x: x["source_line"])

    if len(slots) != 6:
        raise RuntimeError("Expected exactly six manipulated slots.")
    if len({s["source_line"] for s in slots}) != 6:
        raise RuntimeError("Duplicate manipulated source line.")
    if np.any(attack_counts < 0) or attack_counts.sum() != 30:
        raise RuntimeError("Attack histogram invariant failed.")

    clean_sum = int(np.dot(clean_counts, STAR_VALUES))
    attack_sum = int(np.dot(attack_counts, STAR_VALUES))
    if clean_sum != attack_sum:
        raise RuntimeError("Block mean is not exactly preserved.")

    return clean_counts, attack_counts, slots


def canonical_attack_hash(rows: List[Dict]) -> str:
    h = hashlib.sha256()
    for row in sorted(rows, key=lambda x: x["asin"]):
        payload = {
            "asin": row["asin"],
            "treatment_block": row["treatment_block"],
            "slots": row["slots"],
            "clean_counts": row["clean_counts"],
            "attack_counts": row["attack_counts"],
            "clean_mean": row["clean_mean"],
            "attack_mean": row["attack_mean"],
            "clean_w1": row["clean_w1"],
            "attack_w1": row["attack_w1"],
            "d_w1": row["d_w1"],
            "clean_js": row["clean_js"],
            "attack_js": row["attack_js"],
            "d_js": row["d_js"],
            "clean_abs_mean": row["clean_abs_mean"],
            "attack_abs_mean": row["attack_abs_mean"],
            "d_abs_mean": row["d_abs_mean"],
        }
        h.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        h.update(b"\n")
    return h.hexdigest()


def regular_incidence(asins: List[str], m: int, r: int, rng):
    L = len(asins)
    E = L * m
    if E % r != 0:
        raise ValueError(f"L*m={E} not divisible by r={r}")
    M = E // r
    item_order = list(asins)
    rng.shuffle(item_order)

    numeric = {}
    for j, asin in enumerate(item_order):
        numeric[asin] = [int((j * m + s) % M) for s in range(m)]

    deg = np.zeros(M, dtype=int)
    for asin, ids in numeric.items():
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Duplicate account within {asin}")
        for a in ids:
            deg[a] += 1
    if not np.all(deg == r):
        raise RuntimeError("Regular incidence degree invariant failed.")

    perm = rng.permutation(M)
    return {
        asin: [f"syn_shape_r{r}_{int(perm[a]):06d}" for a in numeric[asin]]
        for asin in asins
    }


def auc_rank(pos, neg) -> float:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    vals = np.concatenate([pos, neg])
    ranks = rankdata(vals, method="average")
    n1 = len(pos)
    n0 = len(neg)
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def paired_no_larger(pos, neg) -> float:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if pos.shape != neg.shape:
        raise ValueError("Paired arrays differ in shape.")
    return float(np.mean(pos <= neg))
