from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import _verify_electronics as verifier


def test_verification_scope_matches_phase_12_contract():
    assert verifier.VERIFICATION_SCOPE == (
        "input_provenance",
        "schemas",
        "row_counts",
        "seed_sets",
        "exact_discrete_fields",
        "hashes",
        "conservation_laws",
        "category_consistency",
        "declared_floating_point_tolerances",
    )


def test_category_check_rejects_home_and_kitchen_content():
    verifier.ensure_category(
        {"category": "electronics", "stage": "test"},
        "valid Electronics summary",
    )

    with pytest.raises(verifier.VerificationError, match="category mismatch"):
        verifier.ensure_category(
            {"category": "home_and_kitchen"},
            "wrong category",
        )
    with pytest.raises(
        verifier.VerificationError,
        match="Home-and-Kitchen reference",
    ):
        verifier.ensure_category(
            {
                "category": "electronics",
                "upstream": "shared_data/home_and_kitchen/stage3.json",
            },
            "cross-category upstream",
        )


def test_float_policy_accepts_machine_drift_but_rejects_scientific_change():
    verifier.ensure_close(0.5 + 5e-13, 0.5, "machine drift")

    with pytest.raises(verifier.VerificationError, match="floating-point mismatch"):
        verifier.ensure_close(0.5001, 0.5, "scientific change")


def test_seed_directory_set_is_exact(tmp_path):
    for seed in verifier.EXPECTED_SEEDS:
        (tmp_path / f"seed_{seed:03d}").mkdir()
    verifier.ensure_seed_directories(tmp_path)

    (tmp_path / "seed_030").mkdir()
    with pytest.raises(verifier.VerificationError, match="seed directory set"):
        verifier.ensure_seed_directories(tmp_path)


def test_csv_schema_and_row_count_are_structural(tmp_path):
    source = tmp_path / "rows.csv"
    source.write_text("seed,value\n0,1.0\n", encoding="utf-8")
    assert verifier.read_csv(source, ("seed", "value")) == [
        {"seed": "0", "value": "1.0"}
    ]

    with pytest.raises(verifier.VerificationError, match="CSV header mismatch"):
        verifier.read_csv(source, ("value", "seed"))


def test_stage_report_explicitly_disclaims_gold_reproduction():
    audit = verifier.StageAudit("unit")
    audit.capture("invariant", lambda: {"rows": 30})

    report = audit.to_dict()
    assert report["status"] == "PASS"
    assert report["verification_mode"] == "invariant_only"
    assert report["gold_comparison_performed"] is False
    assert report["independent_gold_reproduction_claimed"] is False
    assert report["tolerances"] == {
        "rtol": verifier.DEFAULT_RTOL,
        "atol": verifier.DEFAULT_ATOL,
    }


def test_consolidated_checker_writes_reports_and_required_signal(
    tmp_path,
    monkeypatch,
    capsys,
):
    def passing(stage):
        def check(_context):
            audit = verifier.StageAudit(stage)
            audit.capture("contract", lambda: True)
            return audit

        return check

    for stage in verifier.STAGE_ORDER:
        monkeypatch.setitem(verifier.STAGE_VERIFIERS, stage, passing(stage))

    context = verifier.VerificationContext(
        repo_root=tmp_path,
        config_dir=tmp_path / "configs",
        shared_root=tmp_path / "shared",
        report_dir=tmp_path / "reports",
        raw_path=tmp_path / "Electronics.jsonl.gz",
        raw_sha256="a" * 64,
        n_items=1,
        selected_lambda=20.0,
    )
    status, report = verifier.run_verification(context)

    assert status == 0
    assert report is not None
    assert report["status"] == "PASS"
    assert report["gold_comparison_performed"] is False
    assert report["independent_gold_reproduction_claimed"] is False
    assert set(report["stage_reports"]) == set(verifier.STAGE_ORDER)
    assert capsys.readouterr().out.rstrip().endswith(
        "ELECTRONICS FINAL STATUS: PASS"
    )

    final_path = context.report_dir / "electronics_final_report.json"
    persisted = json.loads(final_path.read_text(encoding="utf-8"))
    assert persisted == report
    assert all(
        (context.report_dir / f"{stage}_report.json").is_file()
        for stage in verifier.STAGE_ORDER
    )
