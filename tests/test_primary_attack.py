from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.primary_attack import (
    canonical_attack_hash,
    feasible_block,
    shrunk_reference,
)


def test_primary_attack_config_frozen():
    cfg = yaml.safe_load(
        (ROOT / "configs" / "amazon_primary.yaml").read_text()
    )
    assert cfg["attack"]["n_items_per_seed"] == 2000
    assert cfg["attack"]["manipulated_slots_per_item"] == 6
    assert cfg["attack"]["replacement_rating"] == 5
    assert cfg["eligibility"]["expected_feasible_item_universe"] == 21197
    assert cfg["output"]["identity_assignment_in_this_phase"] is False


def test_shared_paths_external():
    cfg = yaml.safe_load(
        (ROOT / "configs" / "amazon_primary.yaml").read_text()
    )
    assert cfg["shared_data"]["root"] == "../amazon_preprocess"
    assert cfg["shared_data"]["output_subdir"] == "stage4_attack"


def test_feasible_block_requires_six_nonfive():
    block = [{"rating": 5}] * 24 + [{"rating": 4}] * 6
    assert feasible_block(block, 6)
    assert not feasible_block(block, 7)


def test_shrunk_reference_lambda_20():
    c = np.array([10,20,30,40,20], dtype=float)
    loo = np.array([.1,.1,.2,.2,.4], dtype=float)
    q = shrunk_reference(c, loo, 20.0)
    expected = (c + 20.0*loo) / 140.0
    assert np.allclose(q, expected)
    assert np.isclose(q.sum(), 1.0)


def test_attack_hash_excludes_identity_by_construction():
    rows = [{
        "asin":"A",
        "treatment_block":"experimental_A",
        "treated_positions":[181,182,183,184,185,186],
        "treated_source_lines":[1,2,3,4,5,6],
        "original_ratings":[1,2,3,4,4,3],
        "replacement_ratings":[5,5,5,5,5,5],
        "clean_counts":[1,1,2,2,24],
        "attack_counts":[0,0,0,0,30],
        "clean_w1":0.1,
        "attack_w1":0.2,
        "d_cf":0.1,
    }]
    h1 = canonical_attack_hash(rows)
    rows2 = [dict(rows[0])]
    rows2[0]["some_identity_field_not_hashed"] = "x"
    h2 = canonical_attack_hash(rows2)
    assert h1 == h2
