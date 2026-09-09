#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from _verify_common import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    ComparisonReport,
    compare_with_report,
)

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "paper_results" / "expected" / "amazon_reference"
ACT = ROOT.parent / "amazon_preprocess" / "stage3"
CATEGORY = "home_and_kitchen"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


CSV_SCHEMAS = {
    "home_and_kitchen_lambda_calibration.csv": {
        "exact": {"lambda"},
        "integer": set(),
    },
    "home_and_kitchen_holdout_validation.csv": {
        "exact": {"method", "level"},
        "integer": {"n_items", "replicates"},
    },
    "home_and_kitchen_holdout_item_metrics.csv": {
        "exact": {"asin", "lambda"},
        "integer": set(),
    },
    "home_and_kitchen_selected_lambda_chronology.csv": {
        "exact": {"role"},
        "integer": set(),
    },
}


def _drop_additive_provenance(obj):
    """Drop Phase-3/4 portable provenance before scientific gold comparison.

    The immutable Home reference gold predates category-aware propagation of
    the raw-dataset digest and the portable provenance block.  Those additive
    fields are verified independently below against the actual Stage-2 inputs
    and frozen raw-data identity.
    """
    obj = json.loads(json.dumps(obj))
    obj.pop("raw_dataset_sha256", None)
    obj.pop("provenance", None)
    return obj


def strip_paths_summary(obj):
    obj = _drop_additive_provenance(obj)
    obj.pop("outputs", None)
    return obj


def strip_paths_freeze(obj):
    return _drop_additive_provenance(obj)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_raw_sha256() -> str:
    candidates = [
        ROOT / "artifacts" / "verification" / "phase3_smoke_home_and_kitchen_windows_seed0_full.json",
        ROOT / "artifacts" / "verification" / "phase3_smoke_home_and_kitchen_windows_seed0_core.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        value = obj.get("inputs", {}).get("raw_dataset_sha256")
        if isinstance(value, str) and _SHA256.fullmatch(value):
            return value
    raise RuntimeError(
        "Cannot locate the frozen Home_and_Kitchen raw SHA-256 in the Phase-3 verification manifests."
    )


def reference_provenance_matches(obj, expected_sha: str) -> bool:
    provenance = obj.get("provenance")
    if not isinstance(provenance, dict):
        return False
    config = provenance.get("config")
    upstream = provenance.get("upstream_artifacts")
    if not isinstance(config, dict) or not isinstance(upstream, dict):
        return False

    roles = ROOT.parent / "amazon_preprocess" / "stage2" / "home_and_kitchen_roles.items.jsonl.gz"
    refs = ROOT.parent / "amazon_preprocess" / "stage2" / "home_and_kitchen_reference_histograms.csv"
    if not roles.is_file() or not refs.is_file():
        return False

    expected_upstream = {
        "stage2_roles": {
            "path": "shared_data/stage2/home_and_kitchen_roles.items.jsonl.gz",
            "sha256": sha256(roles),
        },
        "stage2_reference_histograms": {
            "path": "shared_data/stage2/home_and_kitchen_reference_histograms.csv",
            "sha256": sha256(refs),
        },
    }
    config_path = ROOT / "configs" / "amazon_reference.yaml"
    return (
        obj.get("category") == CATEGORY
        and obj.get("raw_dataset_sha256") == expected_sha
        and provenance.get("category") == CATEGORY
        and provenance.get("raw_dataset_sha256") == expected_sha
        and provenance.get("selected_lambda") == float(obj.get("selected_lambda"))
        and provenance.get("seed_ids") == []
        and config.get("path") == "configs/amazon_reference.yaml"
        and config.get("sha256") == sha256(config_path)
        and upstream == expected_upstream
        and isinstance(provenance.get("environment_versions"), dict)
    )


def parse_csv_value(value, *, exact, integer):
    if exact:
        return value
    if integer:
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def load_typed_csv(path, schema):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = []
        for raw in reader:
            rows.append({
                field: parse_csv_value(
                    raw[field],
                    exact=field in schema["exact"],
                    integer=field in schema["integer"],
                )
                for field in header
            })
    return {"header": header, "rows": rows}


def compare_csv(name):
    schema = CSV_SCHEMAS[name]
    expected = load_typed_csv(GOLD / name, schema)
    actual = load_typed_csv(ACT / name, schema)
    return compare_with_report(expected, actual)


def exact_summary_configuration(expected, actual):
    try:
        return (
            expected["lambda_grid"] == actual["lambda_grid"]
            and expected["selected_lambda"] == actual["selected_lambda"]
            and expected["selected_calibration_row"]["lambda"]
                == actual["selected_calibration_row"]["lambda"]
        )
    except (KeyError, TypeError):
        return False


def combined_numeric_envelope(reports: dict[str, ComparisonReport]):
    max_abs_source, max_abs_report = max(
        reports.items(), key=lambda item: item[1].max_abs_diff
    )
    max_rel_source, max_rel_report = max(
        reports.items(), key=lambda item: item[1].max_rel_diff
    )
    return {
        "float_values_checked": sum(
            report.float_values_checked for report in reports.values()
        ),
        "nonzero_float_differences": sum(
            report.nonzero_float_differences for report in reports.values()
        ),
        "max_abs_diff": max_abs_report.max_abs_diff,
        "max_abs_source": max_abs_source,
        "max_abs_path": max_abs_report.max_abs_path,
        "max_abs_expected": max_abs_report.max_abs_expected,
        "max_abs_actual": max_abs_report.max_abs_actual,
        "max_rel_diff": max_rel_report.max_rel_diff,
        "max_rel_source": max_rel_source,
        "max_rel_path": max_rel_report.max_rel_path,
        "max_rel_expected": max_rel_report.max_rel_expected,
        "max_rel_actual": max_rel_report.max_rel_actual,
    }


def main():
    out = ROOT / "artifacts" / "verification"
    out.mkdir(parents=True, exist_ok=True)

    checks = {}
    comparison_reports = {}

    for name in CSV_SCHEMAS:
        comparison = compare_csv(name)
        comparison_reports[name] = comparison
        checks[name] = comparison.status == "PASS"

    gold_freeze = json.loads(
        (GOLD/"home_and_kitchen_reference_freeze.json").read_text()
    )
    act_freeze = json.loads(
        (ACT/"home_and_kitchen_reference_freeze.json").read_text()
    )
    checks["reference_freeze"] = (
        strip_paths_freeze(gold_freeze) == strip_paths_freeze(act_freeze)
    )

    gold_summary = json.loads(
        (GOLD/"home_and_kitchen_stage3_summary.json").read_text()
    )
    act_summary = json.loads(
        (ACT/"home_and_kitchen_stage3_summary.json").read_text()
    )
    summary_comparison = compare_with_report(
        strip_paths_summary(gold_summary),
        strip_paths_summary(act_summary),
    )
    summary_configuration_exact = exact_summary_configuration(
        gold_summary, act_summary
    )
    comparison_reports["summary"] = summary_comparison
    checks["summary"] = (
        summary_comparison.status == "PASS"
        and summary_configuration_exact
    )

    expected_sha = expected_raw_sha256()
    checks["reference_freeze_provenance"] = reference_provenance_matches(
        act_freeze, expected_sha
    )
    checks["summary_provenance"] = reference_provenance_matches(
        act_summary, expected_sha
    )

    headline = {
        "selected_lambda": act_summary["selected_lambda"],
        "plugin_holdout_mean":
            act_summary["holdout_validation"]["plugin_centered_w1"]["mean"],
        "predictive_holdout_mean":
            act_summary["holdout_validation"]["predictive_centered_w1"]["mean"],
        "predictive_holdout_ci_lower":
            act_summary["holdout_validation"]["predictive_centered_w1"]["ci_lower"],
        "predictive_holdout_ci_upper":
            act_summary["holdout_validation"]["predictive_centered_w1"]["ci_upper"],
        "predictive_holdout_1_mean":
            act_summary["absolute_null_diagnostic"]["holdout_1_mean"],
        "predictive_holdout_2_mean":
            act_summary["absolute_null_diagnostic"]["holdout_2_mean"],
        "stage4_evidence_mode":
            act_summary["semantic_decision"]["stage4_evidence_mode"],
    }

    expected_headline = {
        "selected_lambda": 20.0,
        "plugin_holdout_mean": 0.12268886315849689,
        "predictive_holdout_mean": 0.10399279718735407,
        "predictive_holdout_ci_lower": 0.1015633437187101,
        "predictive_holdout_ci_upper": 0.10640330714362015,
        "predictive_holdout_1_mean": 0.09583661995080951,
        "predictive_holdout_2_mean": 0.11214897442389861,
        "stage4_evidence_mode": "paired_counterfactual",
    }
    headline_comparison = compare_with_report(expected_headline, headline)
    comparison_reports["headline"] = headline_comparison
    checks["headline"] = (
        headline_comparison.status == "PASS"
        and headline["selected_lambda"]
            == expected_headline["selected_lambda"]
    )

    numeric_envelope = combined_numeric_envelope(comparison_reports)

    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "phase": "2H",
        "status": status,
        "checks": checks,
        "headline": headline,
        "tolerances": {
            "rtol": DEFAULT_RTOL,
            "atol": DEFAULT_ATOL,
        },
        "exact_configuration_checks": {
            "reference_freeze": checks["reference_freeze"],
            "summary_lambda_fields": summary_configuration_exact,
            "headline_selected_lambda": (
                headline["selected_lambda"]
                == expected_headline["selected_lambda"]
            ),
        },
        "numeric_envelope": numeric_envelope,
        "comparison_details": {
            name: comparison.to_dict()
            for name, comparison in comparison_reports.items()
        },
    }
    (out/"phase2H_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("="*72)
    print("PHASE 2H AMAZON REFERENCE-CALIBRATION VERIFICATION")
    print("="*72)
    for k,v in checks.items():
        print(f"{k:42s}: {'PASS' if v else 'FAIL'}")
    print(
        "max numeric abs diff                       : "
        f"{numeric_envelope['max_abs_diff']:.3e}"
    )
    print(
        "max numeric rel diff                       : "
        f"{numeric_envelope['max_rel_diff']:.3e}"
    )
    print(f"OVERALL                                    : {status}")
    print("="*72)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
