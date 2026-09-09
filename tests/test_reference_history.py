from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.reference_history import (
    counts_from_reviews,
    exact_predictive_w1_baseline,
    w1_counts,
)


def test_reference_history_config_frozen():
    cfg = yaml.safe_load(
        (ROOT/"configs"/"amazon_reference_history.yaml").read_text()
    )
    assert cfg["reference_lengths"] == [60,90,120]
    assert cfg["lambda_grid"] == [0,5,10,20,40,80,160]
    assert cfg["reuse_r"] == 8
    assert cfg["calibration_roles"] == ["calibration_1","calibration_2"]
    assert cfg["shared_data"]["root"] == "../amazon_preprocess"
    assert cfg["shared_data"]["attack_dir"] == "stage4_attack"
    assert cfg["shared_data"]["reuse_dir"] == "stage4_reuse"
    assert cfg["shared_data"]["twins_dir"] == "stage5_twins"


def test_counts_from_reviews():
    reviews = [{"rating":1},{"rating":2},{"rating":2},{"rating":5}]
    assert counts_from_reviews(reviews).tolist() == [1,2,0,0,1]


def test_w1_zero_for_identical_histogram():
    c = np.array([1,2,3,4,5])
    q = c/c.sum()
    assert w1_counts(c,q) == 0.0


def test_predictive_baseline_finite_nonnegative():
    alpha = np.array([10.,20.,30.,40.,20.])
    b = exact_predictive_w1_baseline(alpha, 30)
    assert np.isfinite(b)
    assert b >= 0.0


def test_runner_reuses_frozen_attack_and_r8_assignment():
    text = (
        ROOT/"experiments"/"amazon"/"run_reference_history.py"
    ).read_text(encoding="utf-8")
    assert "build_attack_world" not in text
    assert "regular_incidence" not in text
    assert "stage4_attack" in text
    assert "stage4_reuse" in text
    assert "stage5_twins" in text
