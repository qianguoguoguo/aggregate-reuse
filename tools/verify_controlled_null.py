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


def run_compare(expected: Path, actual: Path, report: Path) -> bool:
    cmd = [
        sys.executable, str(VERIFY), "compare",
        "--expected", str(expected),
        "--actual", str(actual),
        "--rtol", "1e-10",
        "--atol", "1e-12",
        "--report", str(report),
    ]
    return subprocess.run(cmd).returncode == 0


def compare_npz() -> tuple[bool, float]:
    expected = np.load(
        ROOT / "paper_results" / "expected" / "per_seed"
        / "controlled_null_trajectories.npz"
    )
    seeds = expected["seeds"].astype(int).tolist()

    raw = []
    signed = []
    for seed in seeds:
        z = np.load(
            ROOT / "artifacts" / "results" / "controlled_null_runs"
            / f"seed_{seed}" / "data" / "controlled_null_centering.npz"
        )
        raw.append(z["cumulative_raw"])
        signed.append(z["cumulative_signed"])

    raw = np.stack(raw)
    signed = np.stack(signed)

    max_abs = max(
        float(np.max(np.abs(raw - expected["cumulative_raw"]))),
        float(np.max(np.abs(signed - expected["cumulative_signed"]))),
    )
    ok = (
        np.allclose(raw, expected["cumulative_raw"], rtol=1e-10, atol=1e-12)
        and np.allclose(
            signed, expected["cumulative_signed"], rtol=1e-10, atol=1e-12
        )
    )
    return ok, max_abs


def main() -> int:
    out = ROOT / "artifacts" / "verification"
    out.mkdir(parents=True, exist_ok=True)

    checks = {}

    checks["per_seed_json"] = run_compare(
        ROOT / "paper_results" / "expected" / "per_seed"
             / "controlled_null_per_seed.json",
        ROOT / "artifacts" / "per_seed" / "controlled_null_per_seed.json",
        out / "controlled_null_per_seed_compare.json",
    )

    checks["aggregate_summary"] = run_compare(
        ROOT / "paper_results" / "expected" / "source_snapshots"
             / "controlled_null_30seed_summary.json",
        ROOT / "artifacts" / "results" / "controlled_null_summary.json",
        out / "controlled_null_summary_compare.json",
    )

    checks["figure_trajectory"] = run_compare(
        ROOT / "paper_results" / "expected" / "figure_data"
             / "controlled_null_trajectory.csv",
        ROOT / "artifacts" / "figure_data" / "controlled_null_trajectory.csv",
        out / "controlled_null_trajectory_compare.json",
    )

    traj_ok, max_abs = compare_npz()
    checks["per_seed_trajectories"] = traj_ok

    inv_path = ROOT / "artifacts" / "per_seed" / "controlled_null_invariants.csv"
    with inv_path.open("r", newline="", encoding="utf-8") as f:
        inv_rows = list(csv.DictReader(f))

    bool_fields = [k for k in inv_rows[0] if k != "seed"]
    inv_ok = (
        len(inv_rows) == 30
        and all(
            row[k].strip().lower() == "true"
            for row in inv_rows
            for k in bool_fields
        )
    )
    checks["all_invariants"] = inv_ok

    status = "PASS" if all(checks.values()) else "FAIL"

    report = {
        "phase": "2D",
        "status": status,
        "checks": checks,
        "max_absolute_per_seed_trajectory_difference": max_abs,
        "tolerances": {"rtol": 1e-10, "atol": 1e-12},
    }
    (out / "phase2D_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("=" * 68)
    print("PHASE 2D CONTROLLED NULL VERIFICATION")
    print("=" * 68)
    for name, ok in checks.items():
        print(f"{name:28s}: {'PASS' if ok else 'FAIL'}")
    print(f"max trajectory abs diff      : {max_abs:.3e}")
    print(f"OVERALL                      : {status}")
    print("=" * 68)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
