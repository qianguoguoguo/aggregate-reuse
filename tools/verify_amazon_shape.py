#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

from _verify_common import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    ComparisonReport,
    compare_legacy_json_with_report,
    compare_with_report,
    float_close,
)

ROOT = Path(__file__).resolve().parents[1]
from _verify_phase4_home import aggregate_provenance_matches, per_seed_category_matches
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.shape import canonical_attack_hash

GOLD_PER_SEED = (
    ROOT / "paper_results" / "expected" / "per_seed"
    / "stage6_per_seed_full.json"
)
GOLD_AGG = (
    ROOT / "paper_results" / "expected" / "amazon_shape"
    / "home_and_kitchen_stage6_summary.json"
)
GOLD_HEADLINE = (
    ROOT / "paper_results" / "expected" / "amazon_shape"
    / "shape_headline.json"
)
ACT = ROOT.parent / "amazon_preprocess" / "stage6_shape"

INVARIANT_NAMES = {
    "all_six_selected_ratings_change",
    "exact_block_mean_preservation",
    "frequency_exactly_matched_at_r8",
    "item_block_slot_exposure_pairwise_matched",
    "mean_based_paired_effect_exactly_zero",
    "three_disjoint_sum_preserving_pairs",
}
SEED_EXACT_FIELDS = {
    "stage",
    "seed",
    "selected_lambda",
    "reuse_r",
    "n_items",
    "m",
    "n_account_pairs",
    "feasible_item_universe",
    "interpretation_guard",
}
AGGREGATE_EXACT_FIELDS = {
    "stage",
    "n_seeds",
    "seed_ids",
    "n_items_per_seed",
    "reuse_r",
    "interpretation_guard",
    "invariants",
}
ATTACK_WORLD_FIELDS = (
    "asin",
    "treatment_block",
    "slots",
    "clean_counts",
    "attack_counts",
    "clean_mean",
    "attack_mean",
    "clean_w1",
    "attack_w1",
    "d_w1",
    "clean_js",
    "attack_js",
    "d_js",
    "clean_abs_mean",
    "attack_abs_mean",
    "d_abs_mean",
)
STRUCTURAL_FINGERPRINT_FIELDS = (
    "asin",
    "treatment_block",
    "slots",
    "clean_counts",
    "attack_counts",
)
SLOT_FIELDS = {
    "local_index",
    "position",
    "source_line",
    "original_rating",
    "replacement_rating",
}


def without_attack_hash(summary):
    return {
        key: value
        for key, value in summary.items()
        if key != "attack_world_sha256"
    }


def exact_fields_match(expected, actual, fields):
    try:
        return all(expected[field] == actual[field] for field in fields)
    except (KeyError, TypeError):
        return False


def exact_invariants_hold(invariants):
    return (
        isinstance(invariants, dict)
        and set(invariants) == INVARIANT_NAMES
        and all(value is True for value in invariants.values())
    )


def load_attack_world(path):
    json_fields = {"slots", "clean_counts", "attack_counts"}
    string_fields = {"asin", "treatment_block"}
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != ATTACK_WORLD_FIELDS:
            raise ValueError(
                "shape attack CSV schema/order does not match the frozen schema"
            )
        for row in reader:
            rows.append({
                key: (
                    value
                    if key in string_fields
                    else json.loads(value)
                    if key in json_fields
                    else float(value)
                )
                for key, value in row.items()
            })
    return rows


def structural_attack_hash(rows):
    h = hashlib.sha256()
    for row in sorted(rows, key=lambda value: value["asin"]):
        payload = {
            field: row[field] for field in STRUCTURAL_FINGERPRINT_FIELDS
        }
        h.update(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        h.update(b"\n")
    return h.hexdigest()


def validate_attack_world(rows, *, expected_n_items):
    failures = []

    def record(path, reason):
        if len(failures) < 50:
            failures.append({"path": path, "reason": reason})

    if len(rows) != expected_n_items:
        record("$.rows", f"expected {expected_n_items}, found {len(rows)}")

    asins = [row.get("asin") for row in rows]
    if len(set(asins)) != len(asins):
        record("$.rows.asin", "duplicate ASIN")
    if asins != sorted(asins):
        record("$.rows.asin", "rows are not in canonical ASIN order")

    for index, row in enumerate(rows):
        path = f"$.rows[{index}]"
        if set(row) != set(ATTACK_WORLD_FIELDS):
            record(path, "row schema")
            continue
        if row["treatment_block"] not in {
            "experimental_A", "experimental_B"
        }:
            record(f"{path}.treatment_block", "unexpected category")

        slots = row["slots"]
        clean_counts = row["clean_counts"]
        attack_counts = row["attack_counts"]
        counts_valid = True

        if not isinstance(slots, list) or len(slots) != 6:
            record(f"{path}.slots", "expected six manipulated slots")
            slots = []
        if (
            not isinstance(clean_counts, list)
            or len(clean_counts) != 5
            or any(type(value) is not int or value < 0 for value in clean_counts)
            or sum(clean_counts) != 30
        ):
            record(f"{path}.clean_counts", "invalid five-star histogram")
            counts_valid = False
        if (
            not isinstance(attack_counts, list)
            or len(attack_counts) != 5
            or any(type(value) is not int or value < 0 for value in attack_counts)
            or sum(attack_counts) != 30
        ):
            record(f"{path}.attack_counts", "invalid five-star histogram")
            counts_valid = False

        slot_values_valid = True
        source_lines = []
        for slot_index, slot in enumerate(slots):
            slot_path = f"{path}.slots[{slot_index}]"
            if not isinstance(slot, dict) or set(slot) != SLOT_FIELDS:
                record(slot_path, "slot schema")
                slot_values_valid = False
                continue
            if any(type(slot[field]) is not int for field in SLOT_FIELDS):
                record(slot_path, "slot fields must be integers")
                slot_values_valid = False
                continue
            source_lines.append(slot["source_line"])
            original = slot["original_rating"]
            replacement = slot["replacement_rating"]
            if not (1 <= original <= 5 and 1 <= replacement <= 5):
                record(slot_path, "rating outside 1..5")
                slot_values_valid = False
            elif original == replacement:
                record(slot_path, "selected rating did not change")
                slot_values_valid = False

        if len(source_lines) != len(set(source_lines)):
            record(f"{path}.slots", "duplicate manipulated source line")
            slot_values_valid = False
        if source_lines != sorted(source_lines):
            record(f"{path}.slots", "slots are not in canonical source-line order")

        if counts_valid and slot_values_valid:
            reconstructed = list(clean_counts)
            for slot in slots:
                reconstructed[slot["original_rating"] - 1] -= 1
                reconstructed[slot["replacement_rating"] - 1] += 1
            if reconstructed != attack_counts:
                record(
                    f"{path}.attack_counts",
                    "histogram does not match recorded slot replacements",
                )
            clean_sum = sum(
                (star + 1) * count
                for star, count in enumerate(clean_counts)
            )
            attack_sum = sum(
                (star + 1) * count
                for star, count in enumerate(attack_counts)
            )
            if clean_sum != attack_sum:
                record(path, "block rating sum is not exactly preserved")

            clean_mean = clean_sum / sum(clean_counts)
            attack_mean = attack_sum / sum(attack_counts)
            if not float_close(row["clean_mean"], clean_mean):
                record(f"{path}.clean_mean", "inconsistent with clean counts")
            if not float_close(row["attack_mean"], attack_mean):
                record(f"{path}.attack_mean", "inconsistent with attack counts")

        for metric in ("w1", "js", "abs_mean"):
            clean = row[f"clean_{metric}"]
            attack = row[f"attack_{metric}"]
            delta = row[f"d_{metric}"]
            if not float_close(delta, attack - clean):
                record(f"{path}.d_{metric}", "inconsistent derived difference")

        if (
            row["clean_mean"] != row["attack_mean"]
            or row["clean_abs_mean"] != row["attack_abs_mean"]
            or row["d_abs_mean"] != 0.0
        ):
            record(path, "exact mean-preservation fields failed")

    return failures


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
    differences = []

    gold_bundle = json.loads(
        GOLD_PER_SEED.read_text(encoding="utf-8")
    )

    per_seed_ok = True
    attack_hash_ok = True
    attack_world_structure_ok = True
    exact_seed_structure_ok = True
    invariant_ok = True
    legacy_attack_hash_matches = 0
    legacy_attack_hash_mismatches = []
    canonical_attack_hash_failures = []
    attack_world_structure_failures = []
    structural_fingerprints = []
    comparison_reports = {}

    for entry in gold_bundle:
        seed = int(entry["seed"])
        expected = entry["summary"]
        p = ACT / f"seed_{seed:03d}" / "summary.json"

        if not p.exists():
            per_seed_ok = attack_hash_ok = attack_world_structure_ok = False
            exact_seed_structure_ok = invariant_ok = False
            differences.append(f"seed_{seed}:missing")
            continue

        actual = json.loads(p.read_text(encoding="utf-8"))
        comparison = compare_legacy_json_with_report(
            without_attack_hash(expected),
            without_attack_hash(actual),
            path=f"$.seed_{seed:03d}",
        )
        comparison_reports[f"seed_{seed:03d}"] = comparison
        if comparison.differences or not per_seed_category_matches(actual):
            per_seed_ok = False
            differences.extend(
                [
                    f"seed_{seed}:{path}:{reason}"
                    for path, reason in comparison.differences[:8]
                ]
            )

        if not exact_fields_match(expected, actual, SEED_EXACT_FIELDS):
            exact_seed_structure_ok = False

        actual_hash = actual.get("attack_world_sha256")
        expected_hash = expected.get("attack_world_sha256")
        if actual_hash == expected_hash:
            legacy_attack_hash_matches += 1
        else:
            legacy_attack_hash_mismatches.append({
                "seed": seed,
                "expected": expected_hash,
                "actual": actual_hash,
            })

        attack_path = ACT / f"seed_{seed:03d}" / "shape_attack_world.csv.gz"
        try:
            attack_rows = load_attack_world(attack_path)
            computed_hash = canonical_attack_hash(attack_rows)
            structural_fingerprints.append({
                "seed": seed,
                "sha256": structural_attack_hash(attack_rows),
            })
            seed_structure_failures = validate_attack_world(
                attack_rows,
                expected_n_items=actual["n_items"],
            )
            if seed_structure_failures:
                attack_world_structure_ok = False
                attack_world_structure_failures.append({
                    "seed": seed,
                    "failures": seed_structure_failures,
                })
        except (
            OSError,
            csv.Error,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            computed_hash = None
            attack_world_structure_ok = False
            canonical_attack_hash_failures.append({
                "seed": seed,
                "recorded": actual_hash,
                "computed": None,
                "reason": f"{type(exc).__name__}: {exc}",
            })
            attack_world_structure_failures.append({
                "seed": seed,
                "failures": [{
                    "path": "$.attack_world",
                    "reason": f"{type(exc).__name__}: {exc}",
                }],
            })

        if computed_hash != actual_hash:
            attack_hash_ok = False
            if not any(
                failure["seed"] == seed
                for failure in canonical_attack_hash_failures
            ):
                canonical_attack_hash_failures.append({
                    "seed": seed,
                    "recorded": actual_hash,
                    "computed": computed_hash,
                    "reason": "canonical hash mismatch",
                })

        if not exact_invariants_hold(actual.get("invariants")):
            invariant_ok = False

    checks["all_30_per_seed_summaries"] = per_seed_ok
    checks["all_30_shape_attack_hashes"] = attack_hash_ok
    checks["all_30_attack_world_structures"] = attack_world_structure_ok
    checks["exact_seed_structure"] = exact_seed_structure_ok
    checks["all_shape_invariants"] = invariant_ok

    agg_path = ACT / "home_and_kitchen_stage6_summary.json"
    if agg_path.exists():
        gold_agg = json.loads(GOLD_AGG.read_text(encoding="utf-8"))
        actual_agg = json.loads(agg_path.read_text(encoding="utf-8"))
        aggregate_comparison = compare_legacy_json_with_report(
            gold_agg,
            actual_agg,
            path="$.aggregate_summary",
        )
        comparison_reports["aggregate_summary"] = aggregate_comparison
        aggregate_structure_exact = exact_fields_match(
            gold_agg, actual_agg, AGGREGATE_EXACT_FIELDS
        )
        checks["aggregate_summary"] = (
            aggregate_comparison.status == "PASS"
            and aggregate_structure_exact
        )
        checks["aggregate_provenance"] = aggregate_provenance_matches(
            actual_agg, root=ROOT, config_relpath="configs/amazon_shape.yaml"
        )
        differences.extend(
            [
                f"aggregate:{path}:{reason}"
                for path, reason in aggregate_comparison.differences[:8]
            ]
        )
    else:
        actual_agg = {}
        aggregate_structure_exact = False
        checks["aggregate_summary"] = False
        checks["aggregate_provenance"] = False

    if actual_agg:
        m = actual_agg["metrics"]
        headline = {
            "frequency_auc": m["frequency_auc"]["mean"],
            "w1_counterfactual_score_auc":
                m["w1_counterfactual_score_auc"]["mean"],
            "js_counterfactual_score_auc":
                m["js_counterfactual_score_auc"]["mean"],
            "mean_counterfactual_score_auc":
                m["mean_counterfactual_score_auc"]["mean"],
            "w1_paired_misordering_probability":
                m["w1_paired_misordering_probability"]["mean"],
            "js_paired_misordering_probability":
                m["js_paired_misordering_probability"]["mean"],
            "mean_paired_misordering_probability":
                m["mean_paired_misordering_probability"]["mean"],
            "mean_d_abs_mean": m["mean_d_abs_mean"]["mean"],
            "max_abs_d_abs_mean": m["max_abs_d_abs_mean"]["mean"],
        }
    else:
        headline = {}

    expected_headline = json.loads(
        GOLD_HEADLINE.read_text(encoding="utf-8")
    )
    headline_comparison = compare_with_report(
        expected_headline,
        headline,
        path="$.headline",
    )
    comparison_reports["headline"] = headline_comparison
    checks["headline"] = headline_comparison.status == "PASS"
    differences.extend(
        [
            f"headline:{path}:{reason}"
            for path, reason in headline_comparison.differences[:8]
        ]
    )

    # Explicit conceptual guards central to Figure 2(c).
    checks["exact_mean_preservation"] = (
        headline.get("mean_d_abs_mean") == 0.0
        and headline.get("max_abs_d_abs_mean") == 0.0
        and headline.get("mean_counterfactual_score_auc") == 0.5
    )
    checks["frequency_exactly_matched"] = (
        headline.get("frequency_auc") == 0.5
    )

    numeric_envelope = combined_numeric_envelope(comparison_reports)
    status = "PASS" if all(checks.values()) else "FAIL"

    fingerprint_report = {
        "phase": "2L",
        "status": (
            "PASS"
            if attack_world_structure_ok
            and len(structural_fingerprints) == len(gold_bundle)
            else "FAIL"
        ),
        "algorithm": "sha256-canonical-json-lines-v1",
        "included_fields": list(STRUCTURAL_FINGERPRINT_FIELDS),
        "excluded_fields": [
            field
            for field in ATTACK_WORLD_FIELDS
            if field not in STRUCTURAL_FINGERPRINT_FIELDS
        ],
        "fingerprints": structural_fingerprints,
        "cross_platform_comparison": "PENDING_WINDOWS_CONFIRMATION",
        "comparison_note": (
            "Frozen gold does not contain the Windows row-level Stage-6 "
            "attack CSVs or float-free fingerprints."
        ),
    }
    fingerprint_path = out / "phase2L_shape_structural_fingerprints.json"
    fingerprint_path.write_text(
        json.dumps(fingerprint_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    report = {
        "phase": "2L",
        "status": status,
        "checks": checks,
        "headline": headline,
        "first_differences": differences[:20],
        "tolerances": {
            "rtol": DEFAULT_RTOL,
            "atol": DEFAULT_ATOL,
        },
        "exact_structural_checks": {
            "per_seed_structure": exact_seed_structure_ok,
            "per_seed_invariants": invariant_ok,
            "attack_world_rows": attack_world_structure_ok,
            "aggregate_structure": aggregate_structure_exact,
            "exact_mean_preservation": checks["exact_mean_preservation"],
            "frequency_exactly_matched": checks["frequency_exactly_matched"],
        },
        "numeric_envelope": numeric_envelope,
        "comparison_details": {
            name: comparison.to_dict()
            for name, comparison in comparison_reports.items()
        },
        "attack_hash_verification": {
            "mode": "regenerated_artifact_self_consistency",
            "canonical_hash_failures": canonical_attack_hash_failures,
            "attack_world_structure_failures": (
                attack_world_structure_failures
            ),
            "legacy_gold_exact_matches": legacy_attack_hash_matches,
            "legacy_gold_total": len(gold_bundle),
            "legacy_gold_mismatches": legacy_attack_hash_mismatches,
            "legacy_hash_gating": False,
            "legacy_hash_note": (
                "Legacy Stage-6 hashes include derived floating-point W1/JS "
                "fields and are not portable across Windows and Linux."
            ),
            "structural_fingerprint_artifact": fingerprint_path.name,
            "cross_platform_structural_comparison": (
                "PENDING_WINDOWS_CONFIRMATION"
            ),
        },
    }
    (out / "phase2L_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("=" * 72)
    print("PHASE 2L AMAZON MEAN-PRESERVING SHAPE VERIFICATION")
    print("=" * 72)
    for k, v in checks.items():
        print(f"{k:36s}: {'PASS' if v else 'FAIL'}")
    print(
        "max numeric abs diff                 : "
        f"{numeric_envelope['max_abs_diff']:.3e}"
    )
    print(
        "max numeric rel diff                 : "
        f"{numeric_envelope['max_rel_diff']:.3e}"
    )
    print(f"OVERALL                              : {status}")
    if differences:
        print("First differences:")
        for d in differences[:10]:
            print("  " + d)
    print("=" * 72)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
