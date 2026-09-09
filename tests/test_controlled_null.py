from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.evidence import (
    HistogramSpec,
    controlled_reference_mass,
    exact_null_baseline_w1,
)
from aggregate_reuse.seeding import derive_stage_seed


def test_phase2d_config_is_frozen():
    cfg = yaml.safe_load((ROOT / "configs" / "controlled.yaml").read_text())
    assert cfg["controlled_model"]["T"] == 4000
    assert cfg["controlled_model"]["N_normal"] == 20000
    assert cfg["controlled_model"]["p_normal"] == 0.04
    assert cfg["controlled_model"]["action_space"]["histogram_bins"] == 40
    assert cfg["implementation"]["baseline_method"] == "exact_binomial_marginal_expectation"
    assert cfg["seeds"] == {"start": 27001, "stop": 27030}


def test_stage_seed_compatibility():
    assert derive_stage_seed(27001, "controlled_null_centering") == 4271325630


def test_reference_is_probability_mass():
    spec = HistogramSpec(-4.0, 4.0, 40)
    ref = controlled_reference_mass(spec, mean=0.0, std=1.0)
    assert np.isclose(ref.sum(), 1.0, rtol=0.0, atol=1e-15)
    assert np.all(ref >= 0)


def test_exact_baseline_is_positive():
    spec = HistogramSpec(-4.0, 4.0, 40)
    ref = controlled_reference_mass(spec, mean=0.0, std=1.0)
    b = exact_null_baseline_w1(ref, 800, bin_positions=spec.centers)
    assert b > 0
    assert 0.04 < b < 0.05
