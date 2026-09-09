from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.reuse import regular_incidence


def test_reuse_config_frozen():
    cfg = yaml.safe_load(
        (ROOT/"configs"/"amazon_primary.yaml").read_text()
    )
    assert cfg["reuse"]["grid"] == [1,2,4,8,16]
    assert cfg["reuse"]["item_degree_m"] == 6
    assert cfg["attribution"]["no_attack_recomputation"] is True
    assert cfg["shared_data"]["reuse_output_subdir"] == "stage4_reuse"


def test_regular_incidence_exact_degrees():
    asins = [f"A{i}" for i in range(16)]
    m = 6
    for r in [1,2,4,8,16]:
        rng = np.random.default_rng(12345 + r)
        inc = regular_incidence(asins, m, r, rng)

        assert all(len(ids) == m for ids in inc.values())
        assert all(len(ids) == len(set(ids)) for ids in inc.values())

        degree = {}
        for ids in inc.values():
            for a in ids:
                degree[a] = degree.get(a,0) + 1
        assert set(degree.values()) == {r}
        assert len(degree) == len(asins)*m//r


def test_identity_rng_formula_frozen():
    cfg = yaml.safe_load(
        (ROOT/"configs"/"amazon_primary.yaml").read_text()
    )
    assert cfg["reuse"]["identity_rng_formula"] == \
        "seed * 100003 + r * 997 + 41"


def test_phase2j_does_not_contain_attack_builder():
    text = (
        ROOT/"experiments"/"amazon"/"run_primary_reuse.py"
    ).read_text(encoding="utf-8")
    assert "build_attack_world" not in text
    assert "canonical_attack_hash" in text
