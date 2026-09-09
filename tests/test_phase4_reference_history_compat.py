from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from _verify_common import LEGACY_ADDITIVE_TOP_LEVEL_KEYS, compare_legacy_json_with_report


def _load_verifier_module():
    spec = importlib.util.spec_from_file_location(
        "phase4_verify_reference_history", TOOLS / "verify_reference_history.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def test_reference_history_legacy_seed_csv_accepts_only_required_category(tmp_path):
    verifier = _load_verifier_module()
    gold = tmp_path / "gold.csv"
    actual = tmp_path / "actual.csv"
    fields = ["seed", "reference_length", "counterfactual_score_auc"]
    rows = [
        {"seed": 0, "reference_length": 60, "counterfactual_score_auc": 0.608},
        {"seed": 0, "reference_length": 90, "counterfactual_score_auc": 0.680},
    ]
    _write_csv(gold, fields, rows)
    _write_csv(
        actual,
        ["category", *fields],
        [{"category": "home_and_kitchen", **row} for row in rows],
    )
    ok, diff = verifier.typed_equal_csv(
        gold,
        actual,
        required_actual_metadata={"category": "home_and_kitchen"},
    )
    assert ok
    assert diff == 0.0

    _write_csv(
        actual,
        ["category", *fields],
        [{"category": "electronics", **row} for row in rows],
    )
    ok, _ = verifier.typed_equal_csv(
        gold,
        actual,
        required_actual_metadata={"category": "home_and_kitchen"},
    )
    assert not ok


def test_reference_history_legacy_summary_allows_seed_ids_only_when_explicit():
    expected = json.loads(
        (ROOT / "paper_results" / "expected" / "amazon_reference_history"
         / "home_and_kitchen_stage8_refhistory_summary.json").read_text(encoding="utf-8")
    )
    actual = {
        **expected,
        "category": "home_and_kitchen",
        "raw_dataset_sha256": "a" * 64,
        "selected_lambda": 20.0,
        "selected_lambda_scope": "primary reference length n_ref=120",
        "seed_ids": list(range(30)),
        "provenance": {},
    }
    default = compare_legacy_json_with_report(expected, actual)
    assert default.status == "FAIL"
    report = compare_legacy_json_with_report(
        expected,
        actual,
        allowed_extra_keys=LEGACY_ADDITIVE_TOP_LEVEL_KEYS | frozenset({"seed_ids"}),
    )
    assert report.status == "PASS"

    actual["unexpected_scientific_field"] = 1
    report = compare_legacy_json_with_report(
        expected,
        actual,
        allowed_extra_keys=LEGACY_ADDITIVE_TOP_LEVEL_KEYS | frozenset({"seed_ids"}),
    )
    assert report.status == "FAIL"


def test_windows_orchestrator_can_resume_after_reference_history():
    runner = (TOOLS / "run_phase4_windows.py").read_text(encoding="utf-8")
    cmd = (ROOT / "RUN_PHASE4_WINDOWS.cmd").read_text(encoding="utf-8")
    assert '"resume-after-home-reference-history": resume_after_home_reference_history' in runner
    assert "resume-after-home-reference-history" in cmd
