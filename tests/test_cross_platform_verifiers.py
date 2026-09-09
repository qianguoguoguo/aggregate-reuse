from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import _verify_common as common
import verify_amazon_reference as reference_verifier
import verify_amazon_shape as shape_verifier
import verify_primary_reuse as reuse_verifier


def primary_reuse_row():
    return {
        "seed": 0,
        "reuse_r": 1,
        "n_items": 2000,
        "m": 6,
        "coalition_accounts": 12000,
        "normal_accounts": 47805,
        "attack_world_sha256": "a" * 64,
        "identity_assignment_sha256": "b" * 64,
        "score_auc": 0.5,
        "mean_d_cf": 0.075,
    }


def shape_attack_row(asin="A1", source_line_offset=0):
    replacements = [1, 3, 1, 3, 1, 3]
    slots = [
        {
            "local_index": index,
            "position": 181 + index,
            "source_line": source_line_offset + 1000 + index,
            "original_rating": 2,
            "replacement_rating": replacement,
        }
        for index, replacement in enumerate(replacements)
    ]
    return {
        "asin": asin,
        "treatment_block": "experimental_A",
        "slots": slots,
        "clean_counts": [0, 6, 0, 0, 24],
        "attack_counts": [3, 0, 3, 0, 24],
        "clean_mean": 4.4,
        "attack_mean": 4.4,
        "clean_w1": 0.1,
        "attack_w1": 0.2,
        "d_w1": 0.1,
        "clean_js": 0.01,
        "attack_js": 0.03,
        "d_js": 0.02,
        "clean_abs_mean": 0.4,
        "attack_abs_mean": 0.4,
        "d_abs_mean": 0.0,
    }


def shape_seed_summary():
    return {
        "stage": "amazon_stage6_mean_preserving_shape_r8",
        "seed": 0,
        "selected_lambda": 20.0,
        "reuse_r": 8,
        "n_items": 2000,
        "m": 6,
        "n_account_pairs": 1500,
        "feasible_item_universe": 14690,
        "interpretation_guard": "guard",
        "attack_world_sha256": "a" * 64,
        "invariants": {
            name: True for name in shape_verifier.INVARIANT_NAMES
        },
        "matched_twins": {"score_auc": 0.9},
    }


def test_shared_default_tolerances_are_frozen():
    assert common.DEFAULT_RTOL == 1e-10
    assert common.DEFAULT_ATOL == 1e-12


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        (1_000_000.0, 1_000_000.0 + 5e-5),
        (0.0, 5e-13),
    ],
    ids=["relative-tolerance", "near-zero-absolute-tolerance"],
)
def test_shared_comparison_accepts_machine_precision_drift(expected, actual):
    report = common.compare_with_report(
        {"metric": expected}, {"metric": actual}
    )
    assert report.status == "PASS"
    assert report.float_values_checked == 1
    assert report.nonzero_float_differences == 1


def test_shared_comparison_rejects_meaningful_numeric_change_and_reports_path():
    report = common.compare_with_report(
        {"metric": 0.75}, {"metric": 0.7501}
    )
    assert report.status == "FAIL"
    assert report.differences == [("$.metric", "float")]
    assert report.max_abs_path == "$.metric"
    assert report.max_abs_diff == pytest.approx(1e-4)
    assert report.to_dict()["numeric_envelope"]["max_abs_actual"] == 0.7501


@pytest.mark.parametrize(
    ("expected", "actual", "reason"),
    [
        ({"value": 30}, {"value": 30.0}, "int"),
        ({"value": True}, {"value": 1}, "bool"),
        ({"value": "seed_001"}, {"value": "seed_002"}, "value"),
    ],
    ids=["integer-type", "boolean-type", "string-value"],
)
def test_shared_comparison_keeps_discrete_leaves_exact(
    expected, actual, reason
):
    report = common.compare_with_report(expected, actual)
    assert report.status == "FAIL"
    assert report.differences == [("$.value", reason)]


def test_shared_comparison_keeps_schema_row_count_and_order_exact():
    missing_key = common.compare_with_report(
        {"seed": 0, "metric": 0.5}, {"seed": 0}
    )
    short_rows = common.compare_with_report(
        [{"seed": 0}, {"seed": 1}], [{"seed": 0}]
    )
    reordered_rows = common.compare_with_report(
        [{"seed": 0}, {"seed": 1}],
        [{"seed": 1}, {"seed": 0}],
    )

    assert missing_key.differences == [("$", "keys")]
    assert short_rows.differences == [("$", "list")]
    assert reordered_rows.status == "FAIL"
    assert {path for path, _ in reordered_rows.differences} == {
        "$[0].seed", "$[1].seed"
    }


def test_shared_nan_policy_preserves_legacy_modes_and_locations():
    expected = {"values": [math.nan, 1.0]}
    same = {"values": [math.nan, 1.0]}
    moved = {"values": [1.0, math.nan]}

    assert common.compare_with_report(expected, same).status == "FAIL"
    assert not common.recursive_compare_nan_equal(expected, same)
    assert common.recursive_compare_nan_equal(expected, moved)


def test_shared_comparison_rejects_negative_tolerances():
    with pytest.raises(ValueError, match="non-negative"):
        common.compare_with_report(1.0, 1.0, rtol=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        common.compare_with_report(1.0, 1.0, atol=-1.0)


def test_shared_compatibility_wrapper_preserves_positional_tolerances():
    expected = {"metric": 0.0}
    actual = {"metric": 5e-13}

    assert not common.recursive_compare(
        expected, actual, "$", 1e-12, 0.0
    )
    assert common.recursive_compare(
        expected, actual, "$", 1e-14, 0.0
    ) == [("$.metric", "float")]


def test_reference_typed_csv_allows_numeric_drift_but_not_schema_or_labels(
    tmp_path,
):
    schema = {"exact": {"method"}, "integer": {"n_items"}}
    expected_path = tmp_path / "expected.csv"
    actual_path = tmp_path / "actual.csv"
    expected_path.write_text(
        "method,n_items,mean\nplugin,30,0.125\n", encoding="utf-8"
    )
    actual_path.write_text(
        "method,n_items,mean\nplugin,30,0.1250000000005\n",
        encoding="utf-8",
    )

    expected = reference_verifier.load_typed_csv(expected_path, schema)
    actual = reference_verifier.load_typed_csv(actual_path, schema)
    assert common.compare_with_report(expected, actual).status == "PASS"

    changed_label = copy.deepcopy(actual)
    changed_label["rows"][0]["method"] = "predictive"
    assert common.compare_with_report(expected, changed_label).status == "FAIL"

    reordered_header = copy.deepcopy(actual)
    reordered_header["header"] = ["n_items", "method", "mean"]
    assert common.compare_with_report(expected, reordered_header).status == "FAIL"


def test_reference_configuration_floats_remain_exact():
    expected = {
        "lambda_grid": [0.0, 20.0],
        "selected_lambda": 20.0,
        "selected_calibration_row": {"lambda": 20.0, "mean": 0.1},
    }
    actual = copy.deepcopy(expected)
    actual["selected_lambda"] += 5e-13

    assert common.compare_with_report(expected, actual).status == "PASS"
    assert not reference_verifier.exact_summary_configuration(
        expected, actual
    )


def test_reference_exact_csv_fields_are_not_coerced_to_numbers():
    assert reference_verifier.parse_csv_value(
        "20.0", exact=True, integer=False
    ) == "20.0"
    assert reference_verifier.parse_csv_value(
        "20.0", exact=False, integer=False
    ) == 20.0
    assert "lambda" in reference_verifier.CSV_SCHEMAS[
        "home_and_kitchen_lambda_calibration.csv"
    ]["exact"]


def test_primary_reuse_metrics_accept_drift_and_reject_scientific_change():
    expected_rows = [primary_reuse_row()]
    actual_rows = copy.deepcopy(expected_rows)
    actual_rows[0]["score_auc"] += 5e-13
    report = common.compare_with_report(
        reuse_verifier.metric_rows_by_key(expected_rows),
        reuse_verifier.metric_rows_by_key(actual_rows),
    )
    assert report.status == "PASS"

    actual_rows[0]["score_auc"] += 1e-4
    report = common.compare_with_report(
        reuse_verifier.metric_rows_by_key(expected_rows),
        reuse_verifier.metric_rows_by_key(actual_rows),
    )
    assert report.status == "FAIL"


def test_primary_reuse_keys_counts_and_hashes_remain_exact():
    expected = primary_reuse_row()

    changed_key = copy.deepcopy(expected)
    changed_key["reuse_r"] = 2
    key_report = common.compare_with_report(
        reuse_verifier.metric_rows_by_key([expected]),
        reuse_verifier.metric_rows_by_key([changed_key]),
    )
    assert key_report.differences == [("$", "keys")]

    for field, replacement in (
        ("coalition_accounts", 12001),
        ("attack_world_sha256", "c" * 64),
        ("identity_assignment_sha256", "d" * 64),
    ):
        changed = copy.deepcopy(expected)
        changed[field] = replacement
        metric_report = common.compare_with_report(
            reuse_verifier.metric_rows_by_key([expected]),
            reuse_verifier.metric_rows_by_key([changed]),
        )
        assert metric_report.status == "PASS"
        exact_fields = (
            reuse_verifier.INTEGER_FIELDS
            if field == "coalition_accounts"
            else reuse_verifier.HASH_FIELDS
        )
        assert not reuse_verifier.exact_row_fields_match(
            expected, changed, exact_fields
        )


def test_primary_reuse_aggregate_structure_is_exact():
    expected = {
        "stage": "amazon_stage4_fixed_attack_reuse",
        "interpretation_guard": "guard",
        "n_seeds": 30,
        "seed_ids": list(range(30)),
        "reuse_summary": {"1": {"n_seeds": 30, "score_auc": 0.5}},
    }
    numeric_drift = copy.deepcopy(expected)
    numeric_drift["reuse_summary"]["1"]["score_auc"] += 5e-13
    changed_structure = copy.deepcopy(expected)
    changed_structure["reuse_summary"]["1"]["n_seeds"] = 29

    assert common.compare_with_report(expected, numeric_drift).status == "PASS"
    assert reuse_verifier.exact_aggregate_structure(expected, numeric_drift)
    assert not reuse_verifier.exact_aggregate_structure(
        expected, changed_structure
    )


def test_shape_legacy_hash_is_diagnostic_but_scientific_change_fails():
    expected = shape_seed_summary()
    legacy_hash_only = copy.deepcopy(expected)
    legacy_hash_only["attack_world_sha256"] = "b" * 64

    assert common.compare_with_report(
        shape_verifier.without_attack_hash(expected),
        shape_verifier.without_attack_hash(legacy_hash_only),
    ).status == "PASS"

    legacy_hash_only["matched_twins"]["score_auc"] += 1e-4
    assert common.compare_with_report(
        shape_verifier.without_attack_hash(expected),
        shape_verifier.without_attack_hash(legacy_hash_only),
    ).status == "FAIL"


def test_shape_configuration_and_invariant_flags_remain_exact():
    expected = shape_seed_summary()
    changed_configuration = copy.deepcopy(expected)
    changed_configuration["selected_lambda"] += 5e-13
    changed_invariant = copy.deepcopy(expected)
    changed_invariant["invariants"][
        "exact_block_mean_preservation"
    ] = "true"

    assert common.compare_with_report(
        shape_verifier.without_attack_hash(expected),
        shape_verifier.without_attack_hash(changed_configuration),
    ).status == "PASS"
    assert not shape_verifier.exact_fields_match(
        expected,
        changed_configuration,
        shape_verifier.SEED_EXACT_FIELDS,
    )
    assert common.compare_with_report(
        shape_verifier.without_attack_hash(expected),
        shape_verifier.without_attack_hash(changed_invariant),
    ).status == "FAIL"
    assert shape_verifier.exact_invariants_hold(expected["invariants"])
    assert not shape_verifier.exact_invariants_hold(
        changed_invariant["invariants"]
    )


def test_shape_structural_fingerprint_excludes_floats_but_canonical_hash_does_not():
    expected = shape_attack_row()
    float_changed = copy.deepcopy(expected)
    float_changed["clean_w1"] += 5e-13

    assert (
        shape_verifier.structural_attack_hash([expected])
        == shape_verifier.structural_attack_hash([float_changed])
    )
    assert (
        shape_verifier.canonical_attack_hash([expected])
        != shape_verifier.canonical_attack_hash([float_changed])
    )


def test_shape_attack_world_validator_rejects_structural_and_derived_changes():
    expected = shape_attack_row()
    assert not shape_verifier.validate_attack_world(
        [expected], expected_n_items=1
    )

    bad_counts = copy.deepcopy(expected)
    bad_counts["clean_counts"][0] += 1
    unchanged_slot = copy.deepcopy(expected)
    unchanged_slot["slots"][0]["replacement_rating"] = 2
    bad_delta = copy.deepcopy(expected)
    bad_delta["d_js"] = 0.2

    for changed in (bad_counts, unchanged_slot, bad_delta):
        assert shape_verifier.validate_attack_world(
            [changed], expected_n_items=1
        )

    assert (
        shape_verifier.structural_attack_hash([expected])
        != shape_verifier.structural_attack_hash([bad_counts])
    )


def test_shape_row_count_order_and_schema_are_exact():
    first = shape_attack_row("A1", 0)
    second = shape_attack_row("A2", 100)
    reordered = [second, first]
    missing_field = copy.deepcopy(first)
    missing_field.pop("attack_counts")

    assert not shape_verifier.validate_attack_world(
        [first, second], expected_n_items=2
    )
    assert shape_verifier.validate_attack_world(
        reordered, expected_n_items=2
    )
    assert shape_verifier.validate_attack_world(
        [first], expected_n_items=2
    )
    assert shape_verifier.validate_attack_world(
        [missing_field], expected_n_items=1
    )

    # Fingerprints canonicalize row order, while the row validator rejects a
    # noncanonical artifact order as an exact structural mismatch.
    assert (
        shape_verifier.structural_attack_hash([first, second])
        == shape_verifier.structural_attack_hash(reordered)
    )


def test_reference_legacy_gold_normalization_drops_only_additive_provenance():
    expected = {
        "category": "home_and_kitchen",
        "selected_lambda": 20.0,
        "semantic": {"mode": "paired_counterfactual"},
    }
    actual = copy.deepcopy(expected)
    actual["raw_dataset_sha256"] = "a" * 64
    actual["provenance"] = {
        "category": "home_and_kitchen",
        "raw_dataset_sha256": "a" * 64,
    }
    assert reference_verifier.strip_paths_freeze(expected) == expected
    assert reference_verifier.strip_paths_freeze(actual) == expected

    changed = copy.deepcopy(actual)
    changed["selected_lambda"] = 40.0
    assert reference_verifier.strip_paths_freeze(changed) != expected


def test_reference_summary_normalization_also_drops_output_paths():
    expected = {"selected_lambda": 20.0, "metric": 0.125}
    actual = {
        **expected,
        "raw_dataset_sha256": "a" * 64,
        "provenance": {"category": "home_and_kitchen"},
        "outputs": {"summary": "machine/dependent/path"},
    }
    assert reference_verifier.strip_paths_summary(actual) == expected
