#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from _verify_phase4_home import aggregate_provenance_matches
ACT = ROOT.parent / "amazon_preprocess" / "stage9_k6_population_audit"
GOLD = (
    ROOT / "paper_results" / "expected" / "amazon_k6_population_audit"
    / "legacy_population_audit_gold.json"
)


def main():
    out = ROOT / "artifacts" / "verification"
    out.mkdir(parents=True, exist_ok=True)

    summary_path = ACT / "home_and_kitchen_k6_population_audit_summary.json"
    csv_path = ACT / "k6_population_exact_expectations.csv"
    gold = json.loads(GOLD.read_text(encoding="utf-8"))

    checks = {}

    checks["named_artifacts_exist"] = (
        summary_path.exists() and csv_path.exists()
    )

    if checks["named_artifacts_exist"]:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        summary = {}
        rows = []

    res = summary.get("population_results", {})
    sizes = gold["known_feasible_population_sizes"]

    checks["population_sizes_exact"] = (
        bool(res)
        and int(res["k6_feasible_population"]["population_size"])
            == int(sizes["k6_feasible_population"])
        and int(res["k9_feasible_population"]["population_size"])
            == int(sizes["k9_feasible_population"])
    )

    exact_gold = gold["reported_exact_expected_mean_d_cf_rounded"]
    p6 = (
        res.get("k6_feasible_population", {})
        .get("exact_expected_mean_d_cf")
    )
    p9 = (
        res.get("k9_feasible_population", {})
        .get("exact_expected_mean_d_cf")
    )

    checks["exact_expected_mean_d_cf_matches_reported_rounding"] = (
        p6 is not None
        and p9 is not None
        and round(float(p6), 5)
            == round(float(exact_gold["k6_feasible_population"]), 5)
        and round(float(p9), 5)
            == round(float(exact_gold["k9_feasible_population"]), 5)
    )

    checks["csv_has_exact_two_population_rows"] = (
        len(rows) == 2
        and {r["population"] for r in rows}
        == {"k6_feasible_population", "k9_feasible_population"}
    )

    checks["all_invariants"] = (
        bool(summary)
        and all(bool(v) for v in summary["invariants"].values())
    )
    checks["same_k6_design"] = (
        summary.get("k") == 6
        and summary.get("selected_lambda") == 20.0
    )
    checks["population_shift_direction"] = (
        p6 is not None and p9 is not None and float(p6) > float(p9)
    )
    checks["legacy_empirical_not_used_as_target"] = (
        summary.get("legacy_empirical_values", {}).get("status")
        == "not_used_as_reproduction_targets"
    )
    checks["aggregate_provenance"] = (
        bool(summary)
        and aggregate_provenance_matches(
            summary, root=ROOT, config_relpath="configs/amazon_k6_population_audit.yaml"
        )
    )

    status = "PASS" if all(checks.values()) else "FAIL"

    report = {
        "phase": "2P",
        "status": status,
        "checks": checks,
        "headline": {
            "k6_feasible_population": {
                "population_size":
                    res.get("k6_feasible_population", {}).get("population_size"),
                "exact_expected_mean_d_cf": p6,
            },
            "k9_feasible_population": {
                "population_size":
                    res.get("k9_feasible_population", {}).get("population_size"),
                "exact_expected_mean_d_cf": p9,
            },
        },
        "provenance_resolution": (
            "Phase 2P is now deterministic. The legacy exploratory empirical "
            "values are documented but are not reproduction targets because "
            "their exact finite-sample protocol was not retained."
        ),
    }

    (out / "phase2P_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("="*72)
    print("PHASE 2P DETERMINISTIC SAME-k=6 POPULATION AUDIT VERIFICATION")
    print("="*72)
    for k, v in checks.items():
        print(f"{k:52s}: {'PASS' if v else 'FAIL'}")
    print(f"OVERALL                                              : {status}")
    print("="*72)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
