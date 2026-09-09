from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.twins import (
    auc_rank,
    paired_no_larger_probability,
)


def test_twins_config_frozen():
    cfg = yaml.safe_load(
        (ROOT / "configs" / "amazon_twins.yaml").read_text()
    )
    assert cfg["reuse_r"] == 8
    assert cfg["matching"]["frequency"] == "exact"
    assert cfg["matching"]["item_set"] == "exact"
    assert cfg["matching"]["block"] == "exact"
    assert cfg["matching"]["slot_position"] == "exact"
    assert cfg["matching"]["source_line"] == "exact"


def test_shared_paths_use_canonical_directory():
    cfg = yaml.safe_load(
        (ROOT / "configs" / "amazon_twins.yaml").read_text()
    )
    assert cfg["shared_data"]["root"] == "../amazon_preprocess"
    assert cfg["shared_data"]["attack_dir"] == "stage4_attack"
    assert cfg["shared_data"]["reuse_dir"] == "stage4_reuse"
    assert cfg["shared_data"]["output_subdir"] == "stage5_twins"


def test_equal_frequency_auc_is_half():
    x = np.full(1500, 8.0)
    assert auc_rank(x, x) == 0.5


def test_paired_no_larger_probability():
    pos = np.array([1., 3., 2., 5.])
    neg = np.array([2., 2., 2., 5.])
    # <= holds for pair 0, pair 2, pair 3.
    assert paired_no_larger_probability(pos, neg) == 0.75


def test_runner_does_not_construct_attack_or_reuse_incidence():
    text = (
        ROOT / "experiments" / "amazon" / "run_matched_twins.py"
    ).read_text(encoding="utf-8")
    assert "build_attack_world" not in text
    assert "regular_incidence" not in text
