#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from _verify_common import compare_legacy_json_with_report

ROOT = Path(__file__).resolve().parents[1]
from _verify_phase4_home import aggregate_provenance_matches, per_seed_category_matches
GOLD_PER_SEED = (
    ROOT / "paper_results" / "expected" / "per_seed"
    / "stage9_strength_fixed_identity_per_seed_full.json"
)
GOLD_AGG = (
    ROOT / "paper_results" / "expected" / "amazon_strength_fixed"
    / "home_and_kitchen_stage9_strength_fixed_identity_summary.json"
)
GOLD_HEADLINE = (
    ROOT / "paper_results" / "expected" / "amazon_strength_fixed"
    / "strength_fixed_headline.json"
)
ACT = ROOT.parent / "amazon_preprocess" / "stage9_strength_fixed"



def main():
    out = ROOT / "artifacts" / "verification"
    out.mkdir(parents=True, exist_ok=True)

    checks = {}
    differences = []

    bundle = json.loads(GOLD_PER_SEED.read_text(encoding="utf-8"))
    per_seed_ok = True
    invariants_ok = True
    nested_ok = True
    fixed_identity_ok = True
    balance_ok = True

    for entry in bundle:
        seed = int(entry["seed"])
        expected = entry["summary"]
        p = ACT / f"seed_{seed:03d}" / "summary.json"
        if not p.exists():
            per_seed_ok = invariants_ok = nested_ok = fixed_identity_ok = balance_ok = False
            differences.append(f"seed_{seed}:missing")
            continue

        actual = json.loads(p.read_text(encoding="utf-8"))
        comparison = compare_legacy_json_with_report(expected, actual, path=f"$.seed_{seed:03d}")
        diffs = comparison.differences
        if diffs or not per_seed_category_matches(actual):
            per_seed_ok = False
            differences.extend(
                [f"seed_{seed}:{d[0]}:{d[1]}" for d in diffs[:8]]
            )

        if not all(bool(v) for v in actual["invariants"].values()):
            invariants_ok = False

        inv = actual["invariants"]
        nested_ok &= (
            inv["nested_donor_slots_3_in_6_in_9"]
            and inv["modified_edges_nested_across_strengths"]
        )
        fixed_identity_ok &= (
            inv["same_identity_population_across_strengths"]
            and inv["same_account_item_incidence_across_strengths"]
            and inv["account_degree_exactly_r_across_strengths"]
            and inv["item_degree_fixed_at_9_across_strengths"]
        )
        balance_ok &= inv[
            "modified_exposure_count_balanced_within_one_across_accounts"
        ]

    checks["all_30_per_seed_summaries"] = per_seed_ok
    checks["all_strength_invariants"] = invariants_ok
    checks["nested_strength_slots"] = nested_ok
    checks["fixed_identity_and_incidence"] = fixed_identity_ok
    checks["balanced_modified_exposures"] = balance_ok

    agg_path = ACT / "home_and_kitchen_stage9_strength_fixed_identity_summary.json"
    if agg_path.exists():
        gold_agg = json.loads(GOLD_AGG.read_text(encoding="utf-8"))
        actual_agg = json.loads(agg_path.read_text(encoding="utf-8"))
        agg_comparison = compare_legacy_json_with_report(gold_agg, actual_agg, path="$.aggregate_summary")
        agg_diffs = agg_comparison.differences
        checks["aggregate_summary"] = not agg_diffs
        checks["aggregate_provenance"] = aggregate_provenance_matches(
            actual_agg, root=ROOT, config_relpath="configs/amazon_strength_fixed.yaml"
        )
        differences.extend(
            [f"aggregate:{d[0]}:{d[1]}" for d in agg_diffs[:8]]
        )
    else:
        actual_agg = {}
        checks["aggregate_summary"] = False
        checks["aggregate_provenance"] = False

    headline = {}
    if actual_agg:
        for k in ["3","6","9"]:
            row = actual_agg["metrics_by_k"][k]
            headline[k] = {
                "mean_d_cf": row["mean_d_cf"]["mean"],
                "counterfactual_score_auc":
                    row["counterfactual_score_auc"]["mean"],
                "paired_misordering_probability":
                    row["paired_misordering_probability"]["mean"],
                "mean_paired_score_gap":
                    row["mean_paired_score_gap"]["mean"],
                "positive_block_fraction":
                    row["positive_block_fraction"]["mean"],
                "n_accounts_per_seed": row["n_accounts_per_seed"],
                "min_modified_exposures_per_account":
                    row["min_modified_exposures_per_account"],
                "max_modified_exposures_per_account":
                    row["max_modified_exposures_per_account"],
            }

    expected_headline = json.loads(
        GOLD_HEADLINE.read_text(encoding="utf-8")
    )
    checks["headline"] = headline == expected_headline

    checks["strength_grid_exact"] = (
        actual_agg.get("strength_grid_k") == [3,6,9]
    )
    checks["reuse_r8_fixed"] = actual_agg.get("reuse_r") == 8
    checks["item_degree9_fixed"] = actual_agg.get("fixed_item_degree") == 9

    # Conceptual trend is not used as gold replacement; it is an extra guard.
    checks["reported_strength_response"] = (
        bool(headline)
        and headline["3"]["mean_d_cf"] < 0
        and abs(headline["6"]["mean_d_cf"]) < 0.01
        and headline["9"]["mean_d_cf"] > 0
        and headline["3"]["counterfactual_score_auc"] < 0.5
        and headline["9"]["counterfactual_score_auc"] > 0.5
    )

    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "phase": "2O",
        "status": status,
        "checks": checks,
        "headline": headline,
        "first_differences": differences[:20],
        "tolerance": {"absolute": 1e-12},
    }
    (out / "phase2O_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("="*72)
    print("PHASE 2O AMAZON FIXED-IDENTITY STRENGTH VERIFICATION")
    print("="*72)
    for k,v in checks.items():
        print(f"{k:38s}: {'PASS' if v else 'FAIL'}")
    print(f"OVERALL                                : {status}")
    if differences:
        print("First differences:")
        for d in differences[:10]:
            print("  " + d)
    print("="*72)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
