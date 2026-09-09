#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from _verify_common import compare_legacy_json_with_report

ROOT = Path(__file__).resolve().parents[1]
from _verify_phase4_home import aggregate_provenance_matches, per_seed_category_matches
GOLD_PER_SEED = (
    ROOT / "paper_results" / "expected" / "per_seed"
    / "stage5_per_seed_full.json"
)
GOLD_AGG = (
    ROOT / "paper_results" / "expected" / "amazon_matched_twins"
    / "home_and_kitchen_stage5_summary.json"
)
ACT = ROOT.parent / "amazon_preprocess" / "stage5_twins"



def main():
    out = ROOT / "artifacts" / "verification"
    out.mkdir(parents=True, exist_ok=True)

    gold_bundle = json.loads(GOLD_PER_SEED.read_text(encoding="utf-8"))
    checks = {}
    per_seed_ok = True
    max_diffs = []

    for entry in gold_bundle:
        seed = int(entry["seed"])
        expected = entry["summary"]
        path = ACT / f"seed_{seed:03d}" / "summary.json"
        if not path.exists():
            per_seed_ok = False
            max_diffs.append(f"seed_{seed}:missing")
            continue
        actual = json.loads(path.read_text(encoding="utf-8"))
        comparison = compare_legacy_json_with_report(expected, actual, path=f"$.seed_{seed:03d}")
        diffs = comparison.differences
        if diffs or not per_seed_category_matches(actual):
            per_seed_ok = False
            max_diffs.extend([f"seed_{seed}:{x[0]}:{x[1]}" for x in diffs[:5]])

    checks["all_30_per_seed_summaries"] = per_seed_ok

    gold_agg = json.loads(GOLD_AGG.read_text(encoding="utf-8"))
    actual_agg_path = ACT / "home_and_kitchen_stage5_summary.json"
    if actual_agg_path.exists():
        actual_agg = json.loads(actual_agg_path.read_text(encoding="utf-8"))
        agg_comparison = compare_legacy_json_with_report(gold_agg, actual_agg, path="$.aggregate_summary")
        agg_diffs = agg_comparison.differences
        checks["aggregate_summary"] = not agg_diffs
        checks["aggregate_provenance"] = aggregate_provenance_matches(
            actual_agg, root=ROOT, config_relpath="configs/amazon_twins.yaml"
        )
        max_diffs.extend([f"aggregate:{x[0]}:{x[1]}" for x in agg_diffs[:5]])
    else:
        actual_agg = {}
        checks["aggregate_summary"] = False
        checks["aggregate_provenance"] = False

    # Handoff and exact-matching guards.
    handoff_ok = True
    matching_ok = True
    for seed in range(30):
        p = ACT / f"seed_{seed:03d}" / "summary.json"
        if not p.exists():
            handoff_ok = matching_ok = False
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        if s["reuse_r"] != 8:
            handoff_ok = False
        if s["metrics"]["frequency_auc"] != 0.5:
            matching_ok = False
        if not all(
            bool(v) if isinstance(v, bool) else True
            for k,v in s["matching"].items()
            if k != "only_difference"
        ):
            matching_ok = False

        # Compare reported attack hash to Phase2I seed summary.
        attack_summary = (
            ROOT.parent / "amazon_preprocess" / "stage4_attack"
            / f"seed_{seed:03d}" / "attack_summary.json"
        )
        if not attack_summary.exists():
            handoff_ok = False
        else:
            a = json.loads(attack_summary.read_text(encoding="utf-8"))
            if s["stage4_attack_world_sha256"] != a["attack_world_sha256"]:
                handoff_ok = False

    checks["phase2I_phase2J_handoff"] = handoff_ok
    checks["exact_matching_guards"] = matching_ok

    # Headline exact values.
    if actual_agg:
        m = actual_agg["metrics"]
        headline = {
            "frequency_auc": m["frequency_auc"]["mean"],
            "counterfactual_score_auc":
                m["counterfactual_score_auc"]["mean"],
            "raw_world_score_auc":
                m["raw_world_score_auc"]["mean"],
            "predictive_centered_world_score_auc":
                m["predictive_centered_world_score_auc"]["mean"],
            "paired_gap_nonpositive_fraction":
                m["paired_gap_nonpositive_fraction"]["mean"],
            "mean_paired_score_gap":
                m["mean_paired_score_gap"]["mean"],
            "mean_d_cf":
                m["mean_d_cf"]["mean"],
        }
    else:
        headline = {}

    expected_headline = json.loads(
        (
            ROOT / "paper_results" / "expected" / "amazon_matched_twins"
            / "matched_twins_headline.json"
        ).read_text(encoding="utf-8")
    )
    expected_headline.pop("reuse_r", None)
    checks["headline"] = headline == expected_headline

    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "phase": "2K",
        "status": status,
        "checks": checks,
        "headline": headline,
        "first_differences": max_diffs[:20],
        "tolerance": {"absolute": 1e-12},
    }
    (out / "phase2K_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("="*72)
    print("PHASE 2K EXACT MATCHED-TWIN VERIFICATION")
    print("="*72)
    for k,v in checks.items():
        print(f"{k:34s}: {'PASS' if v else 'FAIL'}")
    print(f"OVERALL                            : {status}")
    if max_diffs:
        print("First differences:")
        for d in max_diffs[:10]:
            print("  " + d)
    print("="*72)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
