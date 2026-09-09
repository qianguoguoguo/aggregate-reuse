#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "tools" / "verify_results.py"


def compare(expected, actual, report):
    cmd = [
        sys.executable, str(VERIFY), "compare",
        "--expected", str(expected),
        "--actual", str(actual),
        "--rtol", "1e-10",
        "--atol", "1e-12",
        "--report", str(report),
    ]
    return subprocess.run(cmd).returncode == 0


def compare_trajectories():
    gold = np.load(
        ROOT / "paper_results" / "expected" / "per_seed"
        / "controlled_matched_trajectories.npz"
    )
    seeds = gold["seeds"].astype(int)
    e, p, q = [], [], []
    for seed in seeds:
        z = np.load(
            ROOT / "artifacts" / "results" / "controlled_matched_runs"
            / f"seed_{seed}" / "data"
            / "controlled_matched_exposure_divergence.npz"
        )
        e.append(z["empirical_gap"])
        p.append(z["predicted_gap"])
        q.append(z["pair_error"])

    e, p, q = np.stack(e), np.stack(p), np.stack(q)
    diffs = [
        float(np.max(np.abs(e - gold["empirical_gap"]))),
        float(np.max(np.abs(p - gold["predicted_gap"]))),
        float(np.max(np.abs(q - gold["pair_error"]))),
    ]
    ok = (
        np.allclose(e, gold["empirical_gap"], rtol=1e-10, atol=1e-12)
        and np.allclose(p, gold["predicted_gap"], rtol=1e-10, atol=1e-12)
        and np.allclose(q, gold["pair_error"], rtol=1e-10, atol=1e-12)
    )
    return ok, max(diffs)


def main():
    out = ROOT / "artifacts" / "verification"
    out.mkdir(parents=True, exist_ok=True)

    checks = {}
    checks["per_seed_json"] = compare(
        ROOT / "paper_results" / "expected" / "per_seed"
             / "controlled_matched_per_seed.json",
        ROOT / "artifacts" / "per_seed" / "controlled_matched_per_seed.json",
        out / "controlled_matched_per_seed_compare.json",
    )
    checks["aggregate_summary"] = compare(
        ROOT / "paper_results" / "expected" / "source_snapshots"
             / "controlled_matched_30seed_summary.json",
        ROOT / "artifacts" / "results" / "controlled_matched_summary.json",
        out / "controlled_matched_summary_compare.json",
    )
    checks["figure_trajectory"] = compare(
        ROOT / "paper_results" / "expected" / "figure_data"
             / "controlled_theory_trajectory.csv",
        ROOT / "artifacts" / "figure_data" / "controlled_theory_trajectory.csv",
        out / "controlled_matched_trajectory_compare.json",
    )

    trajectory_ok, max_abs = compare_trajectories()
    checks["per_seed_trajectories"] = trajectory_ok

    with (
        ROOT / "artifacts" / "per_seed"
        / "controlled_matched_invariants.csv"
    ).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fields = [k for k in rows[0] if k != "seed"]
    checks["all_invariants"] = (
        len(rows) == 30
        and all(
            row[k].strip().lower() == "true"
            for row in rows
            for k in fields
        )
    )

    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "phase": "2E",
        "status": status,
        "checks": checks,
        "max_absolute_per_seed_trajectory_difference": max_abs,
        "tolerances": {"rtol": 1e-10, "atol": 1e-12},
    }
    (out / "phase2E_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("=" * 68)
    print("PHASE 2E MATCHED-EXPOSURE VERIFICATION")
    print("=" * 68)
    for name, ok in checks.items():
        print(f"{name:28s}: {'PASS' if ok else 'FAIL'}")
    print(f"max trajectory abs diff      : {max_abs:.3e}")
    print(f"OVERALL                      : {status}")
    print("=" * 68)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
