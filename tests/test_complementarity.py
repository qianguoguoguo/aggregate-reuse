from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.complementarity import (
    degree_preserving_randomize,
    team_rich_incidence,
    validate_regular_incidence,
)


def test_complementarity_config_frozen():
    cfg = yaml.safe_load(
        (ROOT / "configs" / "amazon_complementarity.yaml").read_text()
    )
    assert cfg["reuse_r"] == 8
    assert cfg["item_degree_m"] == 6
    assert cfg["n_items"] == 2000
    assert cfg["n_accounts"] == 1500
    assert cfg["team_size"] == 6
    assert cfg["team_repetitions"] == 8
    assert cfg["swap_success_multiplier"] == 20
    assert cfg["shared_data"]["root"] == "../amazon_preprocess"
    assert cfg["shared_data"]["attack_dir"] == "stage4_attack"
    assert cfg["shared_data"]["output_subdir"] == "stage7_complementarity"


def test_team_rich_exact_degrees():
    asins = [f"A{i}" for i in range(16)]
    adj = team_rich_incidence(
        asins,
        team_size=4,
        repetitions=4,
        rng=np.random.default_rng(123),
    )
    validate_regular_incidence(
        adj,
        asins=asins,
        account_degree=4,
        item_degree=4,
    )


def test_degree_preserving_randomization_preserves_degrees():
    asins = [f"A{i}" for i in range(16)]
    adj = team_rich_incidence(
        asins,
        team_size=4,
        repetitions=4,
        rng=np.random.default_rng(123),
    )
    rand, diag = degree_preserving_randomize(
        adj,
        rng=np.random.default_rng(456),
        successful_swaps=100,
    )
    assert diag["successful_swaps"] == 100
    validate_regular_incidence(
        rand,
        asins=asins,
        account_degree=4,
        item_degree=4,
    )


def test_runner_reuses_configured_phase2I_attack_only():
    text = (
        ROOT / "experiments" / "amazon" / "run_complementarity.py"
    ).read_text(encoding="utf-8")
    assert "build_attack_world" not in text
    assert "AmazonPathLayout.from_config" in text
    assert 'shared["attack_dir"]' in text
    assert 'ROOT.parent/"amazon_preprocess"' not in text
    assert "stage4_reuse" not in text
