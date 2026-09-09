from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.controlled.campaign import requested_k_on
from aggregate_reuse.controlled.matched_exposure import _balanced_coalition_matrix
from aggregate_reuse.seeding import derive_stage_seed


def test_matched_config_frozen():
    cfg = yaml.safe_load((ROOT / "configs" / "controlled.yaml").read_text())
    m = cfg["matched_exposure"]
    assert m["campaign"]["principal_p_on"] == 0.2
    assert m["campaign"]["principal_exposure_ratio"] == 1.0
    assert m["coalition_distribution"]["mean"] == 0.5
    assert m["coalition_distribution"]["std"] == 1.0


def test_k_on_principal_is_400():
    assert requested_k_on(
        R_exp=1.0,
        p_normal=0.04,
        N_coalition=2000,
        p_on=0.2,
    ) == 400


def test_balanced_matrix_only_active_and_degree_balanced():
    I = np.array([1, 0, 1, 1, 0, 1], dtype=np.uint8)
    A = _balanced_coalition_matrix(I, N_coalition=8, k_on=2)
    assert np.all(A[:, I == 0] == 0)
    assert np.all(A[:, I == 1].sum(axis=0) == 2)
    degrees = A.sum(axis=1)
    assert degrees.max() - degrees.min() <= 1


def test_stage_seed_matches_frozen_seed_27001():
    assert derive_stage_seed(
        27001, "controlled_matched_exposure_divergence"
    ) == 302562997
