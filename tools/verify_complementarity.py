#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from _verify_common import compare_legacy_json_with_report

ROOT = Path(__file__).resolve().parents[1]
from _verify_phase4_home import aggregate_provenance_matches, per_seed_category_matches
GOLD_PER_SEED = (
    ROOT / "paper_results" / "expected" / "per_seed"
    / "stage7_per_seed_full.json"
)
GOLD_AGG = (
    ROOT / "paper_results" / "expected" / "amazon_complementarity"
    / "home_and_kitchen_stage7_summary.json"
)
GOLD_HEADLINE = (
    ROOT / "paper_results" / "expected" / "amazon_complementarity"
    / "complementarity_headline.json"
)
ACT = ROOT.parent / "amazon_preprocess" / "stage7_complementarity"



def main():
    out = ROOT / "artifacts" / "verification"
    out.mkdir(parents=True, exist_ok=True)

    checks = {}
    differences = []

    gold_bundle = json.loads(GOLD_PER_SEED.read_text(encoding="utf-8"))
    per_seed_ok = True
    attack_hash_ok = True
    invariant_ok = True

    for entry in gold_bundle:
        seed = int(entry["seed"])
        expected = entry["summary"]
        p = ACT / f"seed_{seed:03d}" / "summary.json"

        if not p.exists():
            per_seed_ok = attack_hash_ok = invariant_ok = False
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

        # Exact attack handoff to Phase 2I.
        attack_summary = (
            ROOT.parent / "amazon_preprocess" / "stage4_attack"
            / f"seed_{seed:03d}" / "attack_summary.json"
        )
        if not attack_summary.exists():
            attack_hash_ok = False
        else:
            a = json.loads(attack_summary.read_text(encoding="utf-8"))
            if actual["stage4_attack_world_sha256"] != a["attack_world_sha256"]:
                attack_hash_ok = False

        if not all(bool(v) for v in actual["invariants"].values()):
            invariant_ok = False

    checks["all_30_per_seed_summaries"] = per_seed_ok
    checks["phase2I_attack_hash_handoff"] = attack_hash_ok
    checks["all_complementarity_invariants"] = invariant_ok

    agg_path = ACT / "home_and_kitchen_stage7_summary.json"
    if agg_path.exists():
        gold_agg = json.loads(GOLD_AGG.read_text(encoding="utf-8"))
        actual_agg = json.loads(agg_path.read_text(encoding="utf-8"))
        agg_comparison = compare_legacy_json_with_report(gold_agg, actual_agg, path="$.aggregate_summary")
        agg_diffs = agg_comparison.differences
        checks["aggregate_summary"] = not agg_diffs
        checks["aggregate_provenance"] = aggregate_provenance_matches(
            actual_agg, root=ROOT, config_relpath="configs/amazon_complementarity.yaml"
        )
        differences.extend(
            [f"aggregate:{d[0]}:{d[1]}" for d in agg_diffs[:8]]
        )
    else:
        actual_agg = {}
        checks["aggregate_summary"] = False
        checks["aggregate_provenance"] = False

    if actual_agg:
        m = actual_agg["metrics"]
        headline = {
            "frequency_auc": m["frequency_auc"]["mean"],
            "aggregate_auc": m["aggregate_auc"]["mean"],
            "coactivity_auc": m["coactivity_auc"]["mean"],
            "combined_auc": m["combined_auc"]["mean"],
            "combined_minus_best_single":
                m["combined_minus_best_single"]["mean"],
            "evidence_only_aggregate_auc":
                m["evidence_only_aggregate_auc"]["mean"],
            "evidence_only_coactivity_auc":
                m["evidence_only_coactivity_auc"]["mean"],
            "topology_only_aggregate_auc":
                m["topology_only_aggregate_auc"]["mean"],
            "topology_only_coactivity_auc":
                m["topology_only_coactivity_auc"]["mean"],
        }
    else:
        headline = {}

    expected_headline = json.loads(
        GOLD_HEADLINE.read_text(encoding="utf-8")
    )
    checks["headline"] = headline == expected_headline

    # Explicit conceptual guards for Figure 3(a).
    checks["frequency_exactly_matched"] = (
        headline.get("frequency_auc") == 0.5
    )
    checks["evidence_only_isolates_aggregate"] = (
        headline.get("evidence_only_coactivity_auc") == 0.5
    )
    checks["topology_only_isolates_coactivity"] = (
        headline.get("topology_only_aggregate_auc") == 0.5
    )
    checks["combined_improves_over_best_single"] = (
        headline.get("combined_minus_best_single", 0.0) > 0.0
    )

    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "phase": "2M",
        "status": status,
        "checks": checks,
        "headline": headline,
        "first_differences": differences[:20],
        "tolerance": {"absolute": 1e-12},
    }
    (out / "phase2M_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("="*72)
    print("PHASE 2M AMAZON COMPLEMENTARITY VERIFICATION")
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
