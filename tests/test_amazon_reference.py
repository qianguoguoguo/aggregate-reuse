from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.references import (
    LAMBDA_GRID,
    exact_plugin_w1_baseline,
    exact_predictive_w1_baseline,
    shrunk_reference,
)


def test_lambda_grid_frozen():
    assert LAMBDA_GRID == [0.,5.,10.,20.,40.,80.,160.]
    cfg = yaml.safe_load((ROOT/"configs"/"amazon_reference.yaml").read_text())
    assert cfg["lambda_grid"] == [0,5,10,20,40,80,160]


def test_shared_paths_are_external():
    cfg = yaml.safe_load((ROOT/"configs"/"amazon_reference.yaml").read_text())
    assert cfg["shared_data"]["root"] == "../amazon_preprocess"
    assert cfg["shared_data"]["output_subdir"] == "stage3"


def test_shrunk_reference_formula():
    c = np.array([10,20,30,40,20], dtype=float)
    loo = np.array([0.1,0.1,0.2,0.2,0.4], dtype=float)
    alpha,q = shrunk_reference(c, loo, 20.0)
    assert np.allclose(alpha, c + 20.0*loo)
    assert np.isclose(q.sum(), 1.0)
    assert np.allclose(q, alpha / 140.0)


def test_exact_baselines_positive():
    q = np.array([0.1,0.1,0.2,0.2,0.4], dtype=float)
    b1 = exact_plugin_w1_baseline(q, 30)
    alpha = 120*q + 20*q
    b2 = exact_predictive_w1_baseline(alpha, 30)
    assert b1 > 0
    assert b2 > 0
