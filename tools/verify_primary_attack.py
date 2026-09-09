#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from _verify_phase4_home import aggregate_provenance_matches, per_seed_category_matches
GOLD = ROOT / "paper_results" / "expected" / "amazon_primary_attack"
ACT = ROOT.parent / "amazon_preprocess" / "stage4_attack"


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    out = ROOT / "artifacts" / "verification"
    out.mkdir(parents=True, exist_ok=True)

    gold_rows = json.loads(
        (GOLD / "primary_attack_per_seed.json").read_text(encoding="utf-8")
    )
    gold_by_seed = {int(r["seed"]): r for r in gold_rows}

    actual_rows = read_csv(ACT / "primary_attack_seed_metrics.csv")
    actual_by_seed = {int(r["seed"]): r for r in actual_rows}

    checks = {}
    checks["all_30_seeds_present"] = (
        set(actual_by_seed) == set(range(30))
    )

    hash_ok = True
    metric_ok = True
    invariant_ok = True
    feasible_ok = True

    numeric_fields = [
        "mean_d_cf",
        "positive_d_cf_fraction",
        "zero_d_cf_fraction",
        "negative_d_cf_fraction",
        "clean_w1_mean",
        "attack_w1_mean",
    ]

    for seed in range(30):
        gold = gold_by_seed[seed]
        actual = actual_by_seed.get(seed)
        if actual is None:
            hash_ok = metric_ok = invariant_ok = feasible_ok = False
            continue

        if actual["attack_world_sha256"] != gold["attack_world_sha256"]:
            hash_ok = False

        if int(actual["n_items"]) != 2000 or int(actual["m"]) != 6:
            metric_ok = False

        for field in numeric_fields:
            if abs(float(actual[field]) - float(gold[field])) > 1e-12:
                metric_ok = False

        summary_path = ACT / f"seed_{seed:03d}" / "attack_summary.json"
        if not summary_path.exists():
            invariant_ok = feasible_ok = False
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not per_seed_category_matches(summary):
            invariant_ok = False
        if not all(bool(v) for v in summary["invariants"].values()):
            invariant_ok = False
        if summary["feasible_item_universe"] != 21197:
            feasible_ok = False

    checks["attack_world_hashes"] = hash_ok
    checks["per_seed_attack_metrics"] = metric_ok
    checks["all_attack_invariants"] = invariant_ok
    checks["feasible_item_universe"] = feasible_ok

    global_gold = json.loads(
        (GOLD / "primary_attack_global_gold.json").read_text(encoding="utf-8")
    )
    global_actual = json.loads(
        (ACT / "primary_attack_summary.json").read_text(encoding="utf-8")
    )

    g = global_gold["mean_d_cf"]
    a = global_actual["aggregate_visibility_mean_d_cf"]
    checks["aggregate_visibility"] = all(
        abs(float(a[k]) - float(g[k])) <= 1e-12
        for k in ["mean", "ci_lower", "ci_upper"]
    )
    checks["aggregate_provenance"] = aggregate_provenance_matches(
        global_actual, root=ROOT, config_relpath="configs/amazon_primary.yaml"
    )

    checks["identity_assignment_absent"] = (
        global_actual["identity_assignment_performed"] is False
        and not any(ACT.rglob("identity_assignment*"))
        and not any(ACT.rglob("account_metrics*"))
    )

    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "phase": "2I",
        "status": status,
        "checks": checks,
        "headline": {
            "mean_d_cf": a["mean"],
            "mean_d_cf_ci": [a["ci_lower"], a["ci_upper"]],
            "feasible_item_universe": 21197,
            "n_items_per_seed": 2000,
            "m": 6,
            "identity_assignment_performed": False,
        },
    }

    (out / "phase2I_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("="*72)
    print("PHASE 2I PRIMARY AMAZON ATTACK-WORLD VERIFICATION")
    print("="*72)
    for k,v in checks.items():
        print(f"{k:32s}: {'PASS' if v else 'FAIL'}")
    print(f"OVERALL                          : {status}")
    print("="*72)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
