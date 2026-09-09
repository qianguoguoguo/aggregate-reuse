from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import rankdata


def read_csv_gz(path: str | Path) -> List[Dict]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_json_field(value):
    if isinstance(value, (list, dict)):
        return value
    return json.loads(value)


def auc_rank(scores_pos, scores_neg) -> float:
    pos = np.asarray(scores_pos, dtype=float)
    neg = np.asarray(scores_neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    vals = np.concatenate([pos, neg])
    ranks = rankdata(vals, method="average")
    n1 = len(pos)
    n0 = len(neg)
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def paired_no_larger_probability(pos, neg) -> float:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if pos.shape != neg.shape:
        raise ValueError("Matched arrays must have identical shape.")
    return float(np.mean(pos <= neg))


def load_attack_world(path: str | Path):
    rows = read_csv_gz(path)
    out = {}
    for row in rows:
        asin = row["asin"]
        if asin in out:
            raise RuntimeError(f"Duplicate ASIN in attack world: {asin}")
        out[asin] = {
            "asin": asin,
            "treatment_block": row["treatment_block"],
            "treated_positions": [int(x) for x in parse_json_field(row["treated_positions"])],
            "treated_source_lines": [int(x) for x in parse_json_field(row["treated_source_lines"])],
            "original_ratings": [int(x) for x in parse_json_field(row["original_ratings"])],
            "replacement_ratings": [int(x) for x in parse_json_field(row["replacement_ratings"])],
            "clean_w1": float(row["clean_w1"]),
            "attack_w1": float(row["attack_w1"]),
            "d_cf": float(row["d_cf"]),
        }
    return out


def load_r8_assignment(path: str | Path):
    rows = read_csv_gz(path)
    for row in rows:
        row["treated_position"] = int(row["treated_position"])
        row["treated_source_line"] = int(row["treated_source_line"])
        row["d_cf"] = float(row["d_cf"])
    return rows
