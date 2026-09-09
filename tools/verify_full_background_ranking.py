#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

from _verify_common import compare_legacy_json_with_report

ROOT=Path(__file__).resolve().parents[1]
from _verify_phase4_home import aggregate_provenance_matches
ACT=ROOT.parent/"amazon_preprocess"/"stage10_full_background"
GOLD_AGG=(
    ROOT/"paper_results"/"expected"/"amazon_full_background_ranking"
    /"home_and_kitchen_stage10_ranking_full_background_summary.json"
)
GOLD_HEADLINE=(
    ROOT/"paper_results"/"expected"/"amazon_full_background_ranking"
    /"ranking_headline.json"
)
GOLD_PER_SEED=(
    ROOT/"paper_results"/"expected"/"per_seed"
    /"stage10_per_seed_metrics.json"
)



def get_path(obj,path):
    for p in path.split("."):
        obj=obj[p]
    return obj


def main():
    out=ROOT/"artifacts"/"verification"
    out.mkdir(parents=True,exist_ok=True)
    checks={}
    diffs=[]

    agg_path=ACT/"home_and_kitchen_stage10_ranking_full_background_summary.json"
    if agg_path.exists():
        expected=json.loads(GOLD_AGG.read_text(encoding="utf-8"))
        actual=json.loads(agg_path.read_text(encoding="utf-8"))
        comparison=compare_legacy_json_with_report(expected,actual,path="$.aggregate_summary",equal_nan=True)
        dd=comparison.differences
        checks["aggregate_summary"]=not dd
        checks["aggregate_provenance"]=aggregate_provenance_matches(
            actual, root=ROOT, config_relpath="configs/amazon_full_background.yaml"
        )
        diffs.extend([f"aggregate:{x[0]}:{x[1]}" for x in dd[:10]])
    else:
        actual={}
        checks["aggregate_summary"]=False
        checks["aggregate_provenance"]=False

    # Compact per-seed vector from Step 1C.
    gold_rows=json.loads(GOLD_PER_SEED.read_text(encoding="utf-8"))
    compact_ok=True
    max_abs=0.0

    for gold in gold_rows:
        seed=int(gold["seed"])
        p=ACT/f"seed_{seed:03d}"/"summary.json"
        if not p.exists():
            compact_ok=False
            diffs.append(f"seed_{seed}:missing")
            continue
        s=json.loads(p.read_text(encoding="utf-8"))

        # Every compact key except metadata maps directly into summary.
        for key,gv in gold.items():
            if key in {"seed","source_summary","source_summary_sha256"}:
                continue
            if key.startswith("population."):
                av=get_path(s,key)
            else:
                av=get_path(s,"metrics."+key)

            if isinstance(gv,(int,float)) and not isinstance(gv,bool):
                gf=float(gv); af=float(av)
                if math.isnan(gf) and math.isnan(af):
                    continue
                d=abs(gf-af)
                max_abs=max(max_abs,d)
                if d>1e-12:
                    compact_ok=False
                    diffs.append(f"seed_{seed}:{key}:{gf}!={af}")
            elif gv!=av:
                compact_ok=False
                diffs.append(f"seed_{seed}:{key}")

    checks["all_30_per_seed_compact_metrics"]=compact_ok

    # Handoff/invariants.
    handoff=True
    inv=True
    for seed in range(30):
        p=ACT/f"seed_{seed:03d}"/"summary.json"
        ap=ROOT.parent/"amazon_preprocess"/"stage4_attack"/f"seed_{seed:03d}"/"attack_summary.json"
        if not p.exists() or not ap.exists():
            handoff=inv=False
            continue
        s=json.loads(p.read_text(encoding="utf-8"))
        a=json.loads(ap.read_text(encoding="utf-8"))
        if s["source_stage4"]["attack_world_sha256"] != a["attack_world_sha256"]:
            handoff=False
        if not all(bool(v) for v in s["invariants"].values()):
            inv=False

    checks["phase2I_attack_hash_handoff"]=handoff
    checks["all_ranking_invariants"]=inv

    # Headline exact.
    if actual:
        h={
            "full_predictive":{
                "median_background_percentile":
                    actual["metrics"]["predictive_centered_w1"]["all"]
                    ["planted_background_percentile_median"]["mean"],
                "top_1pct_capture":
                    actual["metrics"]["predictive_centered_w1"]["all"]
                    ["top_0p01_planted_capture"]["mean"],
                "top_5pct_capture":
                    actual["metrics"]["predictive_centered_w1"]["all"]
                    ["top_0p05_planted_capture"]["mean"],
                "burden_50_accounts":
                    actual["metrics"]["predictive_centered_w1"]["all"]
                    ["burden_0p5_accounts"]["mean"],
                "burden_50_fraction":
                    actual["metrics"]["predictive_centered_w1"]["all"]
                    ["burden_0p5_fraction_of_universe"]["mean"],
            },
            "freq_eq_8_predictive":{
                "median_background_percentile":
                    actual["metrics"]["predictive_centered_w1"]["freq_eq_8"]
                    ["planted_background_percentile_median"]["mean"],
                "q25":
                    actual["metrics"]["predictive_centered_w1"]["freq_eq_8"]
                    ["planted_background_percentile_q25"]["mean"],
                "q75":
                    actual["metrics"]["predictive_centered_w1"]["freq_eq_8"]
                    ["planted_background_percentile_q75"]["mean"],
                "above_p95":
                    actual["metrics"]["predictive_centered_w1"]["freq_eq_8"]
                    ["fraction_planted_above_background_p95"]["mean"],
                "above_p99":
                    actual["metrics"]["predictive_centered_w1"]["freq_eq_8"]
                    ["fraction_planted_above_background_p99"]["mean"],
                "background_above_planted_median":
                    actual["metrics"]["predictive_centered_w1"]["freq_eq_8"]
                    ["background_accounts_strictly_above_planted_median_score"]["mean"],
            },
        }
    else:
        h={}
    gold_h=json.loads(GOLD_HEADLINE.read_text(encoding="utf-8"))
    checks["headline"]=h==gold_h

    # Interpretation guard: no fraud-negative metrics against historical users.
    checks["historical_background_remains_unlabeled"]=(
        bool(actual)
        and "unlabeled background" in actual["interpretation_guard"]
        and "not precision" in actual["interpretation_guard"]
    )

    status="PASS" if all(checks.values()) else "FAIL"
    report={
        "phase":"2Q",
        "status":status,
        "checks":checks,
        "headline":h,
        "max_absolute_per_seed_numeric_difference":max_abs,
        "first_differences":diffs[:20],
        "tolerance":{"absolute":1e-12},
    }
    (out/"phase2Q_report.json").write_text(
        json.dumps(report,indent=2,sort_keys=True),encoding="utf-8"
    )

    print("="*72)
    print("PHASE 2Q AMAZON FULL-BACKGROUND RANKING VERIFICATION")
    print("="*72)
    for k,v in checks.items():
        print(f"{k:42s}: {'PASS' if v else 'FAIL'}")
    print(f"max per-seed numeric abs diff             : {max_abs:.3e}")
    print(f"OVERALL                                   : {status}")
    if diffs:
        print("First differences:")
        for d in diffs[:10]:
            print("  "+d)
    print("="*72)
    return 0 if status=="PASS" else 1

if __name__=="__main__":
    raise SystemExit(main())
