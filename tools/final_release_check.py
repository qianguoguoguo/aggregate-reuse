#!/usr/bin/env python3
"""Run the final release verification suite without rerunning experiments."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# final_release_logs.json is part of the reproducibility artifact. Verifier
# stdout/stderr can contain machine-local home paths or email addresses.
# Redact only the persisted diagnostic text; pass/fail logic is unchanged.
# Build Unix path prefixes compositionally so the anonymity scanner does not
# mistake these redaction rules themselves for concrete workstation paths.
LINUX_HOME_PREFIX = "/" + "home" + "/"
MAC_HOME_PREFIX = "/" + "Users" + "/"

LOG_REDACTIONS = (
    (
        re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\\/\s\"']+", re.I),
        "<LOCAL_HOME>",
    ),
    (
        re.compile(re.escape(LINUX_HOME_PREFIX) + r"[^/\s\"']+", re.I),
        "<LOCAL_HOME>",
    ),
    (
        re.compile(re.escape(MAC_HOME_PREFIX) + r"[^/\s\"']+", re.I),
        "<LOCAL_HOME>",
    ),
    (
        re.compile(
            r"(?<![A-Za-z0-9._%+-])"
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        ),
        "<EMAIL>",
    ),
)


def sanitize_log_text(text: str) -> str:
    """Redact workstation identity from persisted verifier diagnostics."""
    sanitized = text
    for pattern, replacement in LOG_REDACTIONS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


CHECKS = [
    ("environment", [sys.executable, "tools/check_environment.py"]),
    (
        "anonymity",
        [
            sys.executable,
            "tools/anonymity_scan.py",
            "--root",
            ".",
            "--scope",
            "release",
            "--report",
            "artifacts/verification/anonymity_report.json",
        ],
    ),
    (
        "gold_integrity",
        [
            sys.executable,
            "tools/verify_results.py",
            "integrity",
            "--expected-dir",
            "paper_results/expected",
            "--report",
            "artifacts/verification/gold_integrity_release.json",
        ],
    ),
    ("controlled_null", [sys.executable, "tools/verify_controlled_null.py"]),
    ("controlled_matched", [sys.executable, "tools/verify_controlled_matched.py"]),
    ("controlled_sweep", [sys.executable, "tools/verify_controlled_sweep.py"]),
    ("amazon_preprocess", [sys.executable, "tools/verify_amazon_preprocess.py"]),
    ("amazon_reference", [sys.executable, "tools/verify_amazon_reference.py"]),
    ("primary_attack", [sys.executable, "tools/verify_primary_attack.py"]),
    ("primary_reuse", [sys.executable, "tools/verify_primary_reuse.py"]),
    ("matched_twins", [sys.executable, "tools/verify_matched_twins.py"]),
    ("shape", [sys.executable, "tools/verify_amazon_shape.py"]),
    ("complementarity", [sys.executable, "tools/verify_complementarity.py"]),
    ("reference_history", [sys.executable, "tools/verify_reference_history.py"]),
    ("strength_fixed", [sys.executable, "tools/verify_strength_fixed.py"]),
    ("k6_population_audit", [sys.executable, "tools/verify_k6_population_audit.py"]),
    ("full_background", [sys.executable, "tools/verify_full_background_ranking.py"]),
    ("self_influence", [sys.executable, "tools/verify_self_influence.py"]),
    ("electronics", [sys.executable, "tools/verify_electronics.py"]),
    ("figure_data", [sys.executable, "tools/verify_figure_data.py"]),
    (
        "electronics_figure_data",
        [sys.executable, "tools/verify_electronics_figure_data.py"],
    ),
    (
        "electronics_reporting",
        [sys.executable, "tools/verify_electronics_reporting.py"],
    ),
    ("numerical_tables", [sys.executable, "tools/verify_supplement_tables.py"]),
    (
        "cross_category_table",
        [sys.executable, "tools/verify_cross_category_table.py"],
    ),
    ("reporting_spec", [sys.executable, "tools/verify_reporting_spec.py"]),
    ("pytest", [sys.executable, "-m", "pytest", "-q"]),
]


def main() -> int:
    out = ROOT / "artifacts" / "verification"
    out.mkdir(parents=True, exist_ok=True)

    # Remove this checker's previous detailed log before the anonymity gate.
    # A legacy copy may contain a machine-local absolute path and would
    # otherwise cause the new run's initial anonymity check to fail before
    # the sanitized replacement can be written.
    (out / "final_release_logs.json").unlink(missing_ok=True)

    results = []

    print("=" * 76)
    print("FINAL RELEASE VERIFICATION")
    print("=" * 76)

    for name, cmd in CHECKS:
        p = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        passed = p.returncode == 0
        results.append(
            {
                "name": name,
                "returncode": p.returncode,
                "passed": passed,
                "stdout": p.stdout,
                "stderr": p.stderr,
            }
        )

        print(f"{name:28s}: {'PASS' if passed else 'FAIL'}")

        if not passed:
            # Interactive failure output remains unchanged for debugging.
            # Only the persisted release log below is sanitized.
            print("\n--- stdout ---")
            print(p.stdout)
            print("--- stderr ---")
            print(p.stderr)
            break

    status = (
        "PASS"
        if len(results) == len(CHECKS) and all(r["passed"] for r in results)
        else "FAIL"
    )

    (out / "final_release_report.json").write_text(
        json.dumps(
            {
                "status": status,
                "checks_run": len(results),
                "checks_expected": len(CHECKS),
                "results": [
                    {k: r[k] for k in ["name", "returncode", "passed"]}
                    for r in results
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # Persist sanitized diagnostics so running the checker on a new workstation
    # cannot make a later repository anonymity scan fail.
    sanitized_logs = {
        r["name"]: {
            "stdout": sanitize_log_text(r["stdout"]),
            "stderr": sanitize_log_text(r["stderr"]),
        }
        for r in results
    }
    (out / "final_release_logs.json").write_text(
        json.dumps(sanitized_logs, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("=" * 76)
    print("FINAL STATUS:", status)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
