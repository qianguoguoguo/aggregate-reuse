from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROLE_RANGES = {
    "reference": (1, 120),
    "calibration_1": (121, 150),
    "calibration_2": (151, 180),
    "experimental_A": (181, 210),
    "experimental_B": (211, 240),
    "holdout_1": (241, 270),
    "holdout_2": (271, 300),
}

STAR_RATINGS = (1, 2, 3, 4, 5)


def role_for_position(position: int) -> str:
    for role, (lo, hi) in ROLE_RANGES.items():
        if lo <= position <= hi:
            return role
    raise ValueError(f"Position outside frozen 1..300 range: {position}")


def rating_counts(reviews: Iterable[Dict]) -> Dict[int, int]:
    c = Counter(int(r["rating"]) for r in reviews)
    bad = set(c) - set(STAR_RATINGS)
    if bad:
        raise RuntimeError(f"Unsupported ratings survived Stage 1: {sorted(bad)}")
    return {r: int(c.get(r, 0)) for r in STAR_RATINGS}


def mean_from_counts(counts: Dict[int, int]) -> float:
    n = sum(counts.values())
    if n <= 0:
        raise ValueError("Cannot compute mean from empty counts.")
    return sum(r * counts[r] for r in STAR_RATINGS) / n


def probs_from_counts(counts: Dict[int, int]) -> Dict[int, float]:
    n = sum(counts.values())
    if n <= 0:
        raise ValueError("Cannot normalize empty counts.")
    return {r: counts[r] / n for r in STAR_RATINGS}


def load_stage1_items(path: str | Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            yield line_no, obj


def validate_stage1_item(obj: Dict) -> Tuple[str, List[Dict]]:
    asin = str(obj["asin"])
    reviews = list(obj["reviews"])

    if len(reviews) != 300:
        raise RuntimeError(
            f"{asin}: Stage-1 invariant failed, expected 300 reviews, got {len(reviews)}"
        )

    positions = [int(r["position"]) for r in reviews]
    if positions != list(range(1, 301)):
        raise RuntimeError(f"{asin}: positions are not exactly 1..300")

    users = [str(r["user_id"]) for r in reviews]
    if len(users) != len(set(users)):
        raise RuntimeError(f"{asin}: duplicate user IDs remain within item")

    # Verify chronological order as frozen in Stage 1.
    keys = [(int(r["timestamp"]), int(r["source_line"])) for r in reviews]
    if keys != sorted(keys):
        raise RuntimeError(f"{asin}: reviews are not ordered by (timestamp, source_line)")

    for r in reviews:
        rating = int(r["rating"])
        if rating not in STAR_RATINGS:
            raise RuntimeError(f"{asin}: unsupported rating {rating}")

    return asin, reviews


def split_reviews(reviews: List[Dict]) -> Dict[str, List[Dict]]:
    out = {}
    for role, (lo, hi) in ROLE_RANGES.items():
        block = reviews[lo - 1 : hi]
        expected = hi - lo + 1
        if len(block) != expected:
            raise RuntimeError(
                f"Role {role}: expected {expected} reviews, got {len(block)}"
            )
        out[role] = block
    return out
