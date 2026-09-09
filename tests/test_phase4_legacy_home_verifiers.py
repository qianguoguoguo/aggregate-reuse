from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import _verify_common as common
from _verify_phase4_home import (
    CATEGORY,
    aggregate_provenance_matches,
    expected_raw_sha256,
    sha256_file,
)


def test_legacy_compare_accepts_only_whitelisted_phase4_top_level_metadata():
    expected = {
        "stage": "legacy_stage",
        "n_seeds": 30,
        "metrics": {"auc": 0.75},
    }
    actual = {
        **copy.deepcopy(expected),
        "category": CATEGORY,
        "raw_dataset_sha256": "a" * 64,
        "selected_lambda": 20.0,
        "provenance": {"category": CATEGORY},
    }
    report = common.compare_legacy_json_with_report(expected, actual)
    assert report.status == "PASS"

    actual["new_scientific_result"] = 123
    report = common.compare_legacy_json_with_report(expected, actual)
    assert report.status == "FAIL"
    assert "unexpected_keys" in report.differences[0][1]


def test_legacy_compare_still_rejects_missing_or_changed_scientific_content():
    expected = {"stage": "s", "metrics": {"auc": 0.75}, "n_seeds": 30}

    missing = {"stage": "s", "metrics": {"auc": 0.75}, "category": CATEGORY}
    assert common.compare_legacy_json_with_report(expected, missing).status == "FAIL"

    changed = {
        **copy.deepcopy(expected),
        "metrics": {"auc": 0.7501},
        "category": CATEGORY,
    }
    assert common.compare_legacy_json_with_report(expected, changed).status == "FAIL"


def test_phase4_home_provenance_is_checked_independently_from_legacy_gold():
    raw_sha = expected_raw_sha256(ROOT)
    config_relpath = "configs/amazon_primary.yaml"
    config_path = ROOT / config_relpath
    actual = {
        "category": CATEGORY,
        "raw_dataset_sha256": raw_sha,
        "selected_lambda": 20.0,
        "seed_ids": list(range(30)),
        "provenance": {
            "category": CATEGORY,
            "raw_dataset_sha256": raw_sha,
            "selected_lambda": 20.0,
            "seed_ids": list(range(30)),
            "config": {
                "path": config_relpath,
                "sha256": sha256_file(config_path),
            },
            "upstream_artifacts": {},
            "environment_versions": {"python": "3.12.10"},
        },
    }
    assert aggregate_provenance_matches(
        actual, root=ROOT, config_relpath=config_relpath
    )

    wrong = copy.deepcopy(actual)
    wrong["provenance"]["raw_dataset_sha256"] = "0" * 64
    assert not aggregate_provenance_matches(
        wrong, root=ROOT, config_relpath=config_relpath
    )


def test_remaining_home_verifiers_use_legacy_aware_comparison_where_needed():
    affected = [
        "verify_primary_reuse.py",
        "verify_matched_twins.py",
        "verify_amazon_shape.py",
        "verify_complementarity.py",
        "verify_reference_history.py",
        "verify_strength_fixed.py",
        "verify_full_background_ranking.py",
        "verify_self_influence.py",
    ]
    for name in affected:
        text = (TOOLS / name).read_text(encoding="utf-8")
        assert "compare_legacy_json_with_report" in text, name
        assert "aggregate_provenance_matches" in text, name

    # Selective verifiers do not compare entire legacy aggregate JSON, but they
    # must still validate the new category-aware provenance explicitly.
    for name in ["verify_primary_attack.py", "verify_k6_population_audit.py"]:
        text = (TOOLS / name).read_text(encoding="utf-8")
        assert "aggregate_provenance_matches" in text, name


def test_windows_orchestrator_can_resume_after_completed_home_reuse():
    runner = (TOOLS / "run_phase4_windows.py").read_text(encoding="utf-8")
    cmd = (ROOT / "RUN_PHASE4_WINDOWS.cmd").read_text(encoding="utf-8")
    assert '"resume-after-home-reuse": resume_after_home_reuse' in runner
    assert "resume-after-home-reuse" in cmd
