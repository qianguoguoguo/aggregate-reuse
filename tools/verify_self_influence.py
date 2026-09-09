#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

from _verify_common import compare_legacy_json_with_report

ROOT=Path(__file__).resolve().parents[1]
from _verify_phase4_home import aggregate_provenance_matches, per_seed_category_matches
ACT=ROOT.parent/"amazon_preprocess"/"stage11_self_influence"
GOLD_AGG=(
    ROOT/"paper_results"/"expected"/"amazon_self_influence"
    /"home_and_kitchen_stage11_self_influence_loo_summary.json"
)
GOLD_HEADLINE=(
    ROOT/"paper_results"/"expected"/"amazon_self_influence"
    /"self_influence_headline.json"
)
GOLD_PER_SEED=(
    ROOT/"paper_results"/"expected"/"per_seed"
    /"stage11_self_influence_loo_per_seed_full.json"
)



def main():
    out=ROOT/"artifacts"/"verification"
    out.mkdir(parents=True,exist_ok=True)
    checks={}
    diffs=[]

    # Aggregate.
    agg_path=ACT/"home_and_kitchen_stage11_self_influence_loo_summary.json"
    if agg_path.exists():
        expected=json.loads(GOLD_AGG.read_text(encoding="utf-8"))
        actual=json.loads(agg_path.read_text(encoding="utf-8"))
        comparison=compare_legacy_json_with_report(expected,actual,path="$.aggregate_summary",equal_nan=True)
        dd=comparison.differences
        checks["aggregate_summary"]=not dd
        checks["aggregate_provenance"]=aggregate_provenance_matches(
            actual, root=ROOT, config_relpath="configs/amazon_self_influence.yaml"
        )
        diffs.extend([f"aggregate:{x[0]}:{x[1]}" for x in dd[:10]])
    else:
        actual={}
        checks["aggregate_summary"]=False
        checks["aggregate_provenance"]=False

    # Full per-seed summaries.
    bundle=json.loads(GOLD_PER_SEED.read_text(encoding="utf-8"))
    per_seed=True
    handoff=True
    inv=True
    for entry in bundle:
        seed=int(entry["seed"])
        p=ACT/f"seed_{seed:03d}"/"summary.json"
        if not p.exists():
            per_seed=handoff=inv=False
            diffs.append(f"seed_{seed}:missing")
            continue
        s=json.loads(p.read_text(encoding="utf-8"))
        comparison=compare_legacy_json_with_report(entry["summary"],s,path=f"$.seed_{seed:03d}",equal_nan=True)
        dd=comparison.differences
        if dd or not per_seed_category_matches(s):
            per_seed=False
            diffs.extend([f"seed_{seed}:{x[0]}:{x[1]}" for x in dd[:6]])
        ap=(
            ROOT.parent/"amazon_preprocess"/"stage4_attack"
            /f"seed_{seed:03d}"/"attack_summary.json"
        )
        if not ap.exists():
            handoff=False
        else:
            a=json.loads(ap.read_text(encoding="utf-8"))
            if s["source_stage4"]["attack_world_sha256"] != a["attack_world_sha256"]:
                handoff=False
        if not all(bool(v) for v in s["invariants"].values()):
            inv=False

    checks["all_30_per_seed_summaries"]=per_seed
    checks["phase2I_attack_hash_handoff"]=handoff
    checks["all_self_influence_invariants"]=inv

    # Headline.
    if actual:
        m=actual["metrics"]
        h={
            "original_auc":m["original_matched_twin_auc"]["mean"],
            "loo_auc":m["loo_matched_twin_auc"]["mean"],
            "original_mean_gap":m["original_mean_paired_gap"]["mean"],
            "loo_mean_gap":m["loo_mean_paired_gap"]["mean"],
            "original_misordering":
                m["original_misordering_probability"]["mean"],
            "loo_misordering":
                m["loo_misordering_probability"]["mean"],
            "retained_mean_gap_fraction":
                m["retained_mean_gap_fraction"]["mean"],
            "pearson":m["pearson_original_vs_loo_score"]["mean"],
            "spearman":m["spearman_original_vs_loo_score"]["mean"],
        }
    else:
        h={}
    gold_h=json.loads(GOLD_HEADLINE.read_text(encoding="utf-8"))
    checks["headline"]=h==gold_h

    # Explicit conceptual guards.
    checks["loo_improves_auc"]=(
        bool(h) and h["loo_auc"] > h["original_auc"]
    )
    checks["loo_reduces_misordering"]=(
        bool(h) and h["loo_misordering"] < h["original_misordering"]
    )
    checks["offline_diagnostic_guard"]=(
        bool(actual)
        and "offline diagnostic" in actual["interpretation_guard"]
        and "not the deployable online score" in actual["interpretation_guard"]
    )

    status="PASS" if all(checks.values()) else "FAIL"
    report={
        "phase":"2R",
        "status":status,
        "checks":checks,
        "headline":h,
        "first_differences":diffs[:20],
        "tolerance":{"absolute":1e-12},
    }
    (out/"phase2R_report.json").write_text(
        json.dumps(report,indent=2,sort_keys=True),encoding="utf-8"
    )

    print("="*72)
    print("PHASE 2R AMAZON LOO SELF-INFLUENCE VERIFICATION")
    print("="*72)
    for k,v in checks.items():
        print(f"{k:40s}: {'PASS' if v else 'FAIL'}")
    print(f"OVERALL                                 : {status}")
    if diffs:
        print("First differences:")
        for d in diffs[:10]:
            print("  "+d)
    print("="*72)
    return 0 if status=="PASS" else 1

if __name__=="__main__":
    raise SystemExit(main())
