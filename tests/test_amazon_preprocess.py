from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.preprocessing import (
    normalize_star_rating,
    normalize_timestamp,
)
from aggregate_reuse.amazon.roles import ROLE_RANGES, split_reviews


def test_dataset_location_is_parent_of_repo():
    cfg = yaml.safe_load(
        (ROOT/"configs"/"amazon_preprocess.yaml").read_text()
    )
    assert cfg["dataset"]["location"] == "parent_of_repository"
    assert cfg["dataset"]["filename"] == "Home_and_Kitchen.jsonl.gz"


def test_exact_star_policy():
    for x in [1,2,3,4,5,1.0,5.0]:
        rating,status = normalize_star_rating(x)
        assert status == "ok"
        assert rating in {1,2,3,4,5}
    assert normalize_star_rating(0)[1] == "zero"
    assert normalize_star_rating(4.5)[1] == "unsupported"


def test_role_ranges_are_frozen():
    assert ROLE_RANGES == {
        "reference": (1,120),
        "calibration_1": (121,150),
        "calibration_2": (151,180),
        "experimental_A": (181,210),
        "experimental_B": (211,240),
        "holdout_1": (241,270),
        "holdout_2": (271,300),
    }
    reviews = [{"position": i} for i in range(1,301)]
    blocks = split_reviews(reviews)
    assert len(blocks["reference"]) == 120
    assert all(len(blocks[k]) == 30 for k in blocks if k != "reference")


def test_integral_timestamp_normalization():
    assert normalize_timestamp(123)[0] == 123
    assert normalize_timestamp("123")[0] == 123
    assert normalize_timestamp(123.0)[0] == 123
