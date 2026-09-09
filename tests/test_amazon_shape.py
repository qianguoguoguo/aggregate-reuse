from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.shape import (
    PAIR_MAP,
    apply_shape_intervention,
    block_is_feasible,
)


def test_shape_config_frozen():
    cfg = yaml.safe_load(
        (ROOT / "configs" / "amazon_shape.yaml").read_text()
    )
    assert cfg["n_items_per_seed"] == 2000
    assert cfg["m_manipulated_slots_per_block"] == 6
    assert cfg["pairs_per_block"] == 3
    assert cfg["reuse_r"] == 8
    assert cfg["shared_data"]["root"] == "../amazon_preprocess"
    assert cfg["shared_data"]["output_subdir"] == "stage6_shape"


def test_pair_map_preserves_sum_and_changes_both_slots():
    for old, new in PAIR_MAP.items():
        assert sum(old) == sum(new)
        # For both possible orientations, no selected rating is unchanged.
        assert old[0] not in (new[0],) or old[1] != new[1]
        assert tuple(sorted(old)) != tuple(sorted(new))


def test_simple_block_is_feasible():
    # Three disjoint transformable (2,2) pairs plus filler.
    ratings = [2,2,2,2,2,2] + [5]*24
    block = [
        {
            "rating": r,
            "position": 181+i,
            "source_line": 1000+i,
        }
        for i,r in enumerate(ratings)
    ]
    assert block_is_feasible(block)


def test_shape_intervention_preserves_exact_sum_and_changes_six():
    ratings = [2,2,2,2,2,2] + [5]*24
    block = [
        {
            "rating": r,
            "position": 181+i,
            "source_line": 1000+i,
        }
        for i,r in enumerate(ratings)
    ]
    rng = np.random.default_rng(123)
    clean, attack, slots = apply_shape_intervention(block, rng)

    stars = np.arange(1,6)
    assert len(slots) == 6
    assert int(np.dot(clean, stars)) == int(np.dot(attack, stars))
    assert all(s["original_rating"] != s["replacement_rating"] for s in slots)


def test_runner_resolves_shared_stage2_stage3_from_config_only():
    text = (
        ROOT / "experiments" / "amazon" / "run_shape.py"
    ).read_text(encoding="utf-8")
    assert "AmazonPathLayout.from_config" in text
    assert 'shared["roles_file"]' in text
    assert 'shared["reference_histograms"]' in text
    assert 'shared["reference_freeze"]' in text
    assert 'ROOT.parent / "amazon_preprocess"' not in text
    assert "stage4_attack" not in text
    assert "stage4_reuse" not in text
