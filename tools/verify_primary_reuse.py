#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

from _verify_common import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    ComparisonReport,
    compare_with_report,
    compare_legacy_json_with_report,
)

ROOT = Path(__file__).resolve().parents[1]
from _verify_phase4_home import aggregate_provenance_matches, per_seed_category_matches
GOLD = ROOT / "paper_results" / "expected" / "amazon_primary_reuse"
ACT = ROOT.parent / "amazon_preprocess" / "stage4_reuse"

REUSE_VALUES = (1, 2, 4, 8, 16)
INTEGER_FIELDS = {
    "seed",
    "reuse_r",
    "n_items",
    "m",
    "coalition_accounts",
    "normal_accounts",
}
HASH_FIELDS = {"attack_world_sha256", "identity_assignment_sha256"}
INVARIANT_NAMES = {
    "coalition_row_degree_exactly_r",
    "item_column_degree_exactly_m",
    "no_duplicate_coalition_account_within_item",
    "phase2I_attack_hash_verified_before_identity_assignment",
    "same_attack_world_across_reuse",
    "score_conservation_verified",
}


def load_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_row(row):
    # Compare every scientific/hash field using its typed representation.
    out = {}
    for k, v in row.items():
        if k in INTEGER_FIELDS:
            out[k] = int(v)
        elif k in HASH_FIELDS:
            out[k] = v
        else:
            out[k] = float(v)
    return out


def metric_rows_by_key(rows):
    return {
        f"seed_{row['seed']:03d}.reuse_{row['reuse_r']}": {
            field: value
            for field, value in row.items()
            if field not in INTEGER_FIELDS | HASH_FIELDS
        }
        for row in rows
    }


def exact_row_fields_match(expected, actual, fields):
    return all(actual.get(field) == expected.get(field) for field in fields)


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


def exact_aggregate_structure(expected, actual):
    try:
        return (
            expected["stage"] == actual["stage"]
            and expected["interpretation_guard"]
                == actual["interpretation_guard"]
            and expected["n_seeds"] == actual["n_seeds"]
            and expected["seed_ids"] == actual["seed_ids"]
            and set(expected["reuse_summary"])
                == set(actual["reuse_summary"])
            and all(
                expected["reuse_summary"][reuse]["n_seeds"]
                    == actual["reuse_summary"][reuse]["n_seeds"]
                for reuse in expected["reuse_summary"]
            )
        )
    except (KeyError, TypeError):
        return False


def main():
    out = ROOT / "artifacts" / "verification"
    out.mkdir(parents=True, exist_ok=True)

    gold_rows = [
        normalize_row(r)
        for r in load_rows(GOLD/"home_and_kitchen_stage4_seed_metrics.csv")
    ]
    actual_rows = [
        normalize_row(r)
        for r in load_rows(ACT/"home_and_kitchen_stage4_seed_metrics.csv")
    ]

    checks = {}
    expected_keys = {
        (seed, reuse) for seed in range(30) for reuse in REUSE_VALUES
    }
    gold_keys = {(row["seed"], row["reuse_r"]) for row in gold_rows}
    actual_keys = {(row["seed"], row["reuse_r"]) for row in actual_rows}
    checks["all_150_seed_reuse_rows_present"] = (
        len(gold_rows) == 150
        and len(actual_rows) == 150
        and gold_keys == expected_keys
        and actual_keys == expected_keys
    )

    gold_fields = set(gold_rows[0]) if gold_rows else set()
    actual_fields = set(actual_rows[0]) if actual_rows else set()
    checks["row_schema"] = (
        bool(gold_fields)
        and gold_fields == actual_fields
        and all(set(row) == gold_fields for row in gold_rows)
        and all(set(row) == gold_fields for row in actual_rows)
    )

    by_key_g = {(row["seed"], row["reuse_r"]): row for row in gold_rows}
    by_key_a = {(row["seed"], row["reuse_r"]): row for row in actual_rows}

    hashes_ok = True
    integers_ok = True

    for key, g in by_key_g.items():
        a = by_key_a.get(key)
        if a is None:
            hashes_ok = integers_ok = False
            continue
        if not exact_row_fields_match(g, a, HASH_FIELDS):
            hashes_ok = False
        if not exact_row_fields_match(g, a, INTEGER_FIELDS):
            integers_ok = False

    checks["attack_and_identity_hashes"] = hashes_ok
    checks["integer_counts"] = integers_ok

    seed_metric_comparison = compare_with_report(
        metric_rows_by_key(gold_rows),
        metric_rows_by_key(actual_rows),
        path="$.seed_reuse_metrics",
    )
    checks["all_reuse_metrics"] = seed_metric_comparison.status == "PASS"

    # Verify per-seed invariants and frozen attack hash continuity.
    inv_ok = True
    fixed_attack_ok = True
    for seed in range(30):
        p = ACT/f"seed_{seed:03d}"/"reuse_summary.json"
        if not p.exists():
            inv_ok = fixed_attack_ok = False
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        if not per_seed_category_matches(s):
            inv_ok = False
        invariants = s.get("invariants")
        if (
            not isinstance(invariants, dict)
            or set(invariants) != INVARIANT_NAMES
            or not all(value is True for value in invariants.values())
        ):
            inv_ok = False
        attack_hashes = {
            r["attack_world_sha256"] for r in s["reuse_metrics"]
        }
        if len(attack_hashes) != 1:
            fixed_attack_ok = False

        expected_attack = by_key_g[(seed,1)]["attack_world_sha256"]
        if s["attack_world_sha256"] != expected_attack:
            fixed_attack_ok = False

    checks["all_incidence_invariants"] = inv_ok
    checks["fixed_attack_across_reuse"] = fixed_attack_ok

    gold_summary = json.loads(
        (GOLD/"home_and_kitchen_stage4_summary.json").read_text()
    )
    act_summary = json.loads(
        (ACT/"home_and_kitchen_stage4_summary.json").read_text()
    )
    aggregate_comparison = compare_legacy_json_with_report(
        gold_summary,
        act_summary,
        path="$.aggregate_summary",
    )
    aggregate_structure_exact = exact_aggregate_structure(
        gold_summary, act_summary
    )
    checks["aggregate_summary"] = (
        aggregate_comparison.status == "PASS"
        and aggregate_structure_exact
    )
    checks["aggregate_provenance"] = aggregate_provenance_matches(
        act_summary, root=ROOT, config_relpath="configs/amazon_primary.yaml"
    )

    comparison_reports = {
        "seed_reuse_metrics": seed_metric_comparison,
        "aggregate_summary": aggregate_comparison,
    }
    numeric_envelope = combined_numeric_envelope(comparison_reports)

    headline = {
        "reuse_auc": {
            r: act_summary["reuse_summary"][r]["score_auc"]["mean"]
            for r in ["1","2","4","8","16"]
        },
        "frequency_auc": {
            r: act_summary["reuse_summary"][r]["frequency_auc"]["mean"]
            for r in ["1","2","4","8","16"]
        },
        "mean_d_cf":
            act_summary["aggregate_visibility_mean_d_cf"]["mean"],
    }

    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "phase": "2J",
        "status": status,
        "checks": checks,
        "max_absolute_numeric_difference": numeric_envelope["max_abs_diff"],
        "max_relative_numeric_difference": numeric_envelope["max_rel_diff"],
        "headline": headline,
        "tolerances": {
            "rtol": DEFAULT_RTOL,
            "atol": DEFAULT_ATOL,
        },
        "exact_structural_checks": {
            "seed_reuse_keys": checks["all_150_seed_reuse_rows_present"],
            "row_schema": checks["row_schema"],
            "integer_counts": checks["integer_counts"],
            "attack_and_identity_hashes": (
                checks["attack_and_identity_hashes"]
            ),
            "incidence_invariants": checks["all_incidence_invariants"],
            "fixed_attack_across_reuse": (
                checks["fixed_attack_across_reuse"]
            ),
            "aggregate_structure": aggregate_structure_exact,
        },
        "numeric_envelope": numeric_envelope,
        "comparison_details": {
            name: comparison.to_dict()
            for name, comparison in comparison_reports.items()
        },
    }
    (out/"phase2J_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("="*72)
    print("PHASE 2J FIXED-ATTACK REUSE VERIFICATION")
    print("="*72)
    for k,v in checks.items():
        print(f"{k:34s}: {'PASS' if v else 'FAIL'}")
    print(
        "max numeric abs diff               : "
        f"{numeric_envelope['max_abs_diff']:.3e}"
    )
    print(
        "max numeric rel diff               : "
        f"{numeric_envelope['max_rel_diff']:.3e}"
    )
    print(f"OVERALL                            : {status}")
    print("="*72)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
