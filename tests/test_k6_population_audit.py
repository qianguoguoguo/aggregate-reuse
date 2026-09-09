from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.population_audit import (
    donor_count_vectors,
    exact_expected_block_dcf,
)


def test_final_config_is_deterministic_same_k6_audit():
    cfg = yaml.safe_load(
        (ROOT / "configs" / "amazon_k6_population_audit.yaml").read_text()
    )
    assert cfg["status"] == "final"
    assert cfg["k"] == 6
    assert cfg["population_rules"] == {
        "k6_feasible_population": 6,
        "k9_feasible_population": 9,
    }


def test_donor_compositions_have_exact_k():
    xs = list(donor_count_vectors([3,4,5,6], 6))
    assert xs
    assert all(sum(x) == 6 for x in xs)
    assert len(xs) == len(set(xs))


def test_exact_expected_block_dcf_is_finite():
    clean = np.array([3,4,5,6,12])
    q = clean / clean.sum()
    x = exact_expected_block_dcf(clean, q, k=6)
    assert np.isfinite(x)


def test_runner_has_no_finite_sample_rng():
    text = (
        ROOT / "experiments" / "amazon" / "run_k6_population_audit.py"
    ).read_text(encoding="utf-8")
    assert "default_rng" not in text
    assert "matched_twin_auc" not in text
    assert "build_sample" not in text
    assert "exact_population_expected_mean" in text


def test_verifier_does_not_target_legacy_empirical_values():
    text = (
        ROOT / "tools" / "verify_k6_population_audit.py"
    ).read_text(encoding="utf-8")
    assert "reported_empirical_rounded" not in text
    assert "legacy_empirical_not_used_as_target" in text
