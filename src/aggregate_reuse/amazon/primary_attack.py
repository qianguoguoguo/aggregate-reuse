from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.stats import rankdata


STAR_RATINGS = (1, 2, 3, 4, 5)


def load_json(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_reference_table(path: str | Path):
    rows = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            asin = row["asin"]
            c = np.asarray(
                [float(row[f"reference_count_{r}"]) for r in STAR_RATINGS],
                dtype=float,
            )
            loo = np.asarray(
                [float(row[f"loo_domain_prob_{r}"]) for r in STAR_RATINGS],
                dtype=float,
            )
            rows[asin] = (c, loo)
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


def shrunk_reference(
    ref_counts: np.ndarray,
    loo_probs: np.ndarray,
    lam: float,
) -> np.ndarray:
    alpha = np.asarray(ref_counts, dtype=float) + float(lam) * np.asarray(
        loo_probs, dtype=float
    )
    return alpha / alpha.sum()


def hist_from_reviews(reviews: List[Dict]) -> np.ndarray:
    c = np.zeros(5, dtype=int)
    for r in reviews:
        rating = int(r["rating"])
        if rating not in STAR_RATINGS:
            raise RuntimeError(f"Unsupported rating {rating}")
        c[rating - 1] += 1
    return c


def w1_counts_to_reference(counts: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(counts, dtype=float)
    p = p / p.sum()
    q = np.asarray(q, dtype=float)
    return float(np.abs(np.cumsum(p - q)[:-1]).sum())


def feasible_block(block: List[Dict], m: int) -> bool:
    return sum(int(x["rating"]) != 5 for x in block) >= m


def canonical_attack_hash(rows: List[Dict]) -> str:
    """
    Hash only aggregate attack construction fields. Identity assignments are
    intentionally excluded.
    """
    h = hashlib.sha256()
    for row in sorted(rows, key=lambda x: x["asin"]):
        payload = {
            "asin": row["asin"],
            "treatment_block": row["treatment_block"],
            "treated_positions": row["treated_positions"],
            "treated_source_lines": row["treated_source_lines"],
            "original_ratings": row["original_ratings"],
            "replacement_ratings": row["replacement_ratings"],
            "clean_counts": row["clean_counts"],
            "attack_counts": row["attack_counts"],
            "clean_w1": row["clean_w1"],
            "attack_w1": row["attack_w1"],
            "d_cf": row["d_cf"],
        }
        h.update(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        h.update(b"\n")
    return h.hexdigest()


def build_attack_world(
    *,
    seed: int,
    L: int,
    m: int,
    blocks,
    refs,
    lam: float,
):
    rng = np.random.default_rng(seed)

    feasible = sorted([
        asin for asin in blocks
        if asin in refs
        and feasible_block(blocks[asin]["experimental_A"], m)
        and feasible_block(blocks[asin]["experimental_B"], m)
    ])

    if len(feasible) < L:
        raise RuntimeError(
            f"Only {len(feasible):,} items satisfy the Stage-4 feasibility "
            f"rule, fewer than L={L:,}."
        )

    chosen = sorted(rng.choice(feasible, size=L, replace=False).tolist())
    rows = []
    participant_blocks = {}

    for asin in chosen:
        ref_counts, loo_probs = refs[asin]
        q = shrunk_reference(ref_counts, loo_probs, lam)

        treatment_block = (
            "experimental_A" if rng.random() < 0.5 else "experimental_B"
        )
        clean_reviews = blocks[asin][treatment_block]

        candidate_indices = [
            j for j, rev in enumerate(clean_reviews)
            if int(rev["rating"]) != 5
        ]
        chosen_idx = sorted(
            rng.choice(candidate_indices, size=m, replace=False).tolist()
        )

        clean_counts = hist_from_reviews(clean_reviews)
        attack_counts = clean_counts.copy()

        original_ratings = []
        treated_positions = []
        treated_source_lines = []
        donor_users = []

        for j in chosen_idx:
            rev = clean_reviews[j]
            old = int(rev["rating"])
            if old == 5:
                raise RuntimeError("Five-star donor selected unexpectedly.")
            attack_counts[old - 1] -= 1
            attack_counts[4] += 1
            original_ratings.append(old)
            treated_positions.append(int(rev["position"]))
            treated_source_lines.append(int(rev["source_line"]))
            donor_users.append(str(rev["user_id"]))

        if int(clean_counts.sum()) != 30 or int(attack_counts.sum()) != 30:
            raise RuntimeError(f"{asin}: histogram-size invariant failed.")
        if np.any(attack_counts < 0):
            raise RuntimeError(f"{asin}: negative attack histogram count.")

        clean_w1 = w1_counts_to_reference(clean_counts, q)
        attack_w1 = w1_counts_to_reference(attack_counts, q)
        d_cf = attack_w1 - clean_w1

        # All original participants are needed later. Donor identities will be
        # removed from the attacked world and replaced by synthetic identities.
        donor_source_lines = set(treated_source_lines)
        normal_users = [
            str(rev["user_id"]) for rev in clean_reviews
            if int(rev["source_line"]) not in donor_source_lines
        ]
        if len(normal_users) != 30 - m:
            raise RuntimeError(f"{asin}: normal-participant count invariant failed.")

        participant_blocks[asin] = {
            "normal_users": normal_users,
            "treated_positions": treated_positions,
            "treated_source_lines": treated_source_lines,
        }

        rows.append({
            "asin": asin,
            "treatment_block": treatment_block,
            "treated_positions": treated_positions,
            "treated_source_lines": treated_source_lines,
            "original_ratings": original_ratings,
            "replacement_ratings": [5] * m,
            "clean_counts": clean_counts.astype(int).tolist(),
            "attack_counts": attack_counts.astype(int).tolist(),
            "clean_w1": float(clean_w1),
            "attack_w1": float(attack_w1),
            "d_cf": float(d_cf),
        })

    return feasible, rows, participant_blocks
