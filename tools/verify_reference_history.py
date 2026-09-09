#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

from _verify_common import (
    LEGACY_ADDITIVE_TOP_LEVEL_KEYS,
    compare_legacy_json_with_report,
)

ROOT = Path(__file__).resolve().parents[1]
from _verify_phase4_home import aggregate_provenance_matches
GOLD = ROOT / "paper_results" / "expected" / "amazon_reference_history"
ACT = ROOT.parent / "amazon_preprocess" / "stage8_reference_history"


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def typed_equal_csv(
    gold_path,
    actual_path,
    atol=1e-12,
    *,
    required_actual_metadata=None,
):
    """Compare legacy CSV scientific columns plus explicit Phase-4 metadata.

    Legacy Home gold predates the category column added by the category-aware
    runner.  The scientific columns remain exact-schema and value checked; any
    requested additive metadata column must be present with the required value.
    No other extra columns are accepted.
    """
    required_actual_metadata = dict(required_actual_metadata or {})
    g = read_csv(gold_path)
    a = read_csv(actual_path)
    if len(g) != len(a):
        return False, float("inf")
    if not g and not a:
        return True, 0.0
    if not g or not a:
        return False, float("inf")

    gold_columns = list(g[0].keys())
    actual_columns = list(a[0].keys())
    expected_actual_columns = set(gold_columns) | set(required_actual_metadata)
    if set(actual_columns) != expected_actual_columns:
        return False, float("inf")

    max_abs = 0.0
    for gr, ar in zip(g, a):
        if set(gr) != set(gold_columns) or set(ar) != expected_actual_columns:
            return False, float("inf")
        for key, required_value in required_actual_metadata.items():
            if ar.get(key) != required_value:
                return False, float("inf")
        for k in gold_columns:
            gv, av = gr[k], ar[k]
            try:
                gf = float(gv)
                af = float(av)
                d = abs(gf-af)
                max_abs = max(max_abs, d)
                if d > atol:
                    return False, max_abs
            except ValueError:
                if gv != av:
                    return False, max_abs
    return True, max_abs



def main():
    out = ROOT / "artifacts" / "verification"
    out.mkdir(parents=True, exist_ok=True)

    checks = {}
    first_diffs = []
    max_abs = 0.0

    cal_ok, cal_diff = typed_equal_csv(
        GOLD/"reference_history_calibration.csv",
        ACT/"reference_history_calibration.csv",
    )
    seed_ok, seed_diff = typed_equal_csv(
        GOLD/"reference_history_seed_metrics.csv",
        ACT/"reference_history_seed_metrics.csv",
        required_actual_metadata={"category": "home_and_kitchen"},
    )
    max_abs = max(cal_diff, seed_diff)
    checks["calibration_grid"] = cal_ok
    checks["all_90_seed_length_rows"] = seed_ok

    expected_summary = json.loads(
        (GOLD/"home_and_kitchen_stage8_refhistory_summary.json").read_text(
            encoding="utf-8"
        )
    )
    actual_summary_path = ACT/"home_and_kitchen_stage8_refhistory_summary.json"
    if actual_summary_path.exists():
        actual_summary = json.loads(
            actual_summary_path.read_text(encoding="utf-8")
        )
        comparison = compare_legacy_json_with_report(
            expected_summary,
            actual_summary,
            path="$.aggregate_summary",
            allowed_extra_keys=(
                LEGACY_ADDITIVE_TOP_LEVEL_KEYS | frozenset({"seed_ids"})
            ),
        )
        diffs = comparison.differences
        checks["aggregate_summary"] = not diffs
        checks["aggregate_provenance"] = aggregate_provenance_matches(
            actual_summary, root=ROOT, config_relpath="configs/amazon_reference_history.yaml"
        )
        first_diffs.extend(
            [f"summary:{d[0]}:{d[1]}" for d in diffs[:10]]
        )
    else:
        actual_summary = {}
        checks["aggregate_summary"] = False
        checks["aggregate_provenance"] = False

    # Headline.
    headline = {}
    if actual_summary:
        for nref in ["60","90","120"]:
            row = actual_summary["metrics_by_reference_length"][nref]
            headline[nref] = {
                "selected_lambda": row["selected_lambda"],
                "counterfactual_score_auc":
                    row["counterfactual_score_auc"]["mean"],
                "paired_misordering_probability":
                    row["paired_misordering_probability"]["mean"],
                "mean_paired_score_gap":
                    row["mean_paired_score_gap"]["mean"],
                "mean_d_cf":
                    row["mean_d_cf"]["mean"],
            }

    expected_headline = json.loads(
        (GOLD/"reference_history_headline.json").read_text(
            encoding="utf-8"
        )
    )
    checks["headline"] = headline == expected_headline

    # Explicit robustness-design guards.
    checks["lambda_20_all_lengths"] = (
        bool(headline)
        and all(headline[n]["selected_lambda"] == 20.0
                for n in ["60","90","120"])
    )
    checks["reference_lengths_exact"] = (
        actual_summary.get("reference_lengths") == [60,90,120]
    )
    checks["seed_ids_exact"] = (
        actual_summary.get("seed_ids") == list(range(30))
    )
    checks["reuse_r8_fixed"] = actual_summary.get("reuse_r") == 8
    checks["all_summary_invariants"] = (
        bool(actual_summary)
        and all(bool(v) for v in actual_summary["invariants"].values())
    )

    # Directly verify the n_ref=120 endpoint equals Phase-2K matched twins.
    twins_summary_path = (
        ROOT.parent / "amazon_preprocess" / "stage5_twins"
        / "home_and_kitchen_stage5_summary.json"
    )
    if twins_summary_path.exists() and actual_summary:
        twins = json.loads(twins_summary_path.read_text(encoding="utf-8"))
        a120 = actual_summary["metrics_by_reference_length"]["120"]
        checks["nref120_matches_phase2K_auc"] = (
            abs(
                a120["counterfactual_score_auc"]["mean"]
                - twins["metrics"]["counterfactual_score_auc"]["mean"]
            ) <= 1e-12
        )
        checks["nref120_matches_phase2K_gap"] = (
            abs(
                a120["mean_paired_score_gap"]["mean"]
                - twins["metrics"]["mean_paired_score_gap"]["mean"]
            ) <= 1e-12
        )
    else:
        checks["nref120_matches_phase2K_auc"] = False
        checks["nref120_matches_phase2K_gap"] = False

    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "phase": "2N",
        "status": status,
        "checks": checks,
        "headline": headline,
        "max_absolute_numeric_difference": max_abs,
        "first_differences": first_diffs[:20],
        "tolerance": {"absolute": 1e-12},
    }
    (out/"phase2N_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("="*72)
    print("PHASE 2N AMAZON REFERENCE-HISTORY VERIFICATION")
    print("="*72)
    for k,v in checks.items():
        print(f"{k:38s}: {'PASS' if v else 'FAIL'}")
    print(f"max numeric abs diff                   : {max_abs:.3e}")
    print(f"OVERALL                                : {status}")
    if first_diffs:
        print("First differences:")
        for d in first_diffs[:10]:
            print("  " + d)
    print("="*72)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
