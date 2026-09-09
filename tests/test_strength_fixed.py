from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.strength_fixed import (
    fixed_nested_incidence,
    modification_counts,
)


def test_strength_config_frozen():
    cfg = yaml.safe_load(
        (ROOT / "configs" / "amazon_strength_fixed.yaml").read_text()
    )
    assert cfg["strength_grid_k"] == [3,6,9]
    assert cfg["fixed_item_degree"] == 9
    assert cfg["reuse_r"] == 8
    assert cfg["n_items_per_seed"] == 2000
    assert cfg["attack_seed_offset"] == 900000
    assert cfg["incidence_seed_multiplier"] == 100003
    assert cfg["incidence_seed_offset"] == 9091
    assert cfg["shared_data"]["root"] == "../amazon_preprocess"
    assert cfg["shared_data"]["output_subdir"] == "stage9_strength_fixed"


def test_fixed_incidence_exact_degrees():
    asins = [f"A{i}" for i in range(16)]
    inc = fixed_nested_incidence(
        asins,
        item_degree=8,
        reuse_r=8,
        rng=np.random.default_rng(123),
    )

    degree = {}
    for asin, ids in inc.items():
        assert len(ids) == 8
        assert len(set(ids)) == 8
        for a in ids:
            degree[a] = degree.get(a,0) + 1
    assert set(degree.values()) == {8}


def test_modification_counts_nested():
    asins = [f"A{i}" for i in range(16)]
    inc = fixed_nested_incidence(
        asins,
        item_degree=8,
        reuse_r=8,
        rng=np.random.default_rng(123),
    )
    c3 = modification_counts(inc, 3)
    c6 = modification_counts(inc, 6)
    for a in set(c3) | set(c6):
        assert c3.get(a,0) <= c6.get(a,0)


def test_runner_is_self_contained_strength_branch():
    text = (
        ROOT / "experiments" / "amazon" / "run_strength_fixed.py"
    ).read_text(encoding="utf-8")
    assert "stage4_attack" not in text
    assert "stage4_reuse" not in text
    assert "build_common_seed" in text
    assert "fixed_nested_incidence" in text
