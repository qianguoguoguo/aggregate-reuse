#!/usr/bin/env python3
"""
Generic numerical/provenance verifier for the Aggregate Reuse artifact.

Phase 2B scope:
  * verify integrity of immutable pre-refactor gold artifacts;
  * compare JSON and CSV result artifacts recursively;
  * use strict numerical tolerances for floating-point values;
  * require exact equality for strings, booleans, integers, keys, and shapes;
  * provide a self-test that MUST detect an intentionally perturbed result.

This module contains no scientific experiment implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RTOL = 1e-10
DEFAULT_ATOL = 1e-12


@dataclass
class Difference:
    path: str
    expected: Any
    actual: Any
    reason: str


@dataclass
class VerificationResult:
    status: str
    checks: int
    failures: int
    differences: list[Difference]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "checks": self.checks,
            "failures": self.failures,
            "differences": [asdict(d) for d in self.differences],
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_number(x: Any) -> bool:
    # bool is a subclass of int, so exclude it explicitly.
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _numbers_equal(expected: float, actual: float, rtol: float, atol: float) -> bool:
    e = float(expected)
    a = float(actual)

    if math.isnan(e) or math.isnan(a):
        return math.isnan(e) and math.isnan(a)
    if math.isinf(e) or math.isinf(a):
        return e == a

    return math.isclose(a, e, rel_tol=rtol, abs_tol=atol)


def compare_objects(
    expected: Any,
    actual: Any,
    *,
    path: str = "$",
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    differences: list[Difference] | None = None,
    counter: list[int] | None = None,
) -> tuple[list[Difference], int]:
    """
    Recursively compare arbitrary JSON-like objects.

    Numeric values use isclose; nonnumeric scalar values require exact equality.
    Dict key sets and list lengths must match exactly.
    """
    if differences is None:
        differences = []
    if counter is None:
        counter = [0]

    if isinstance(expected, dict):
        counter[0] += 1
        if not isinstance(actual, dict):
            differences.append(Difference(path, type(expected).__name__,
                                          type(actual).__name__, "type mismatch"))
            return differences, counter[0]

        exp_keys = set(expected.keys())
        act_keys = set(actual.keys())
        if exp_keys != act_keys:
            differences.append(
                Difference(
                    path,
                    sorted(exp_keys),
                    sorted(act_keys),
                    "dictionary keys differ",
                )
            )
        for key in sorted(exp_keys & act_keys):
            compare_objects(
                expected[key],
                actual[key],
                path=f"{path}.{key}",
                rtol=rtol,
                atol=atol,
                differences=differences,
                counter=counter,
            )
        return differences, counter[0]

    if isinstance(expected, list):
        counter[0] += 1
        if not isinstance(actual, list):
            differences.append(Difference(path, type(expected).__name__,
                                          type(actual).__name__, "type mismatch"))
            return differences, counter[0]
        if len(expected) != len(actual):
            differences.append(Difference(path, len(expected), len(actual),
                                          "list lengths differ"))
        for i, (e, a) in enumerate(zip(expected, actual)):
            compare_objects(
                e, a,
                path=f"{path}[{i}]",
                rtol=rtol,
                atol=atol,
                differences=differences,
                counter=counter,
            )
        return differences, counter[0]

    counter[0] += 1

    # Preserve integer exactness when both are integers.
    if isinstance(expected, int) and not isinstance(expected, bool):
        if isinstance(actual, int) and not isinstance(actual, bool):
            if expected != actual:
                differences.append(Difference(path, expected, actual,
                                              "integer values differ"))
            return differences, counter[0]

    if _is_number(expected) and _is_number(actual):
        if not _numbers_equal(expected, actual, rtol, atol):
            differences.append(
                Difference(
                    path,
                    expected,
                    actual,
                    f"numeric mismatch (rtol={rtol:g}, atol={atol:g})",
                )
            )
        return differences, counter[0]

    if type(expected) is not type(actual):
        differences.append(Difference(path, type(expected).__name__,
                                      type(actual).__name__, "type mismatch"))
    elif expected != actual:
        differences.append(Difference(path, expected, actual, "values differ"))

    return differences, counter[0]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_json_files(
    expected_path: Path,
    actual_path: Path,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> VerificationResult:
    expected = load_json(expected_path)
    actual = load_json(actual_path)
    differences, checks = compare_objects(expected, actual, rtol=rtol, atol=atol)
    return VerificationResult(
        status="PASS" if not differences else "FAIL",
        checks=checks,
        failures=len(differences),
        differences=differences,
    )


def _parse_csv_scalar(x: str) -> Any:
    text = x.strip()
    if text == "":
        return ""
    try:
        # Keep plain integer strings as exact integers.
        if text.lstrip("+-").isdigit():
            return int(text)
        return float(text)
    except ValueError:
        return text


def compare_csv_files(
    expected_path: Path,
    actual_path: Path,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> VerificationResult:
    with expected_path.open("r", newline="", encoding="utf-8") as f:
        exp_rows = list(csv.DictReader(f))
    with actual_path.open("r", newline="", encoding="utf-8") as f:
        act_rows = list(csv.DictReader(f))

    differences: list[Difference] = []
    checks = 1

    exp_fields = list(exp_rows[0].keys()) if exp_rows else []
    act_fields = list(act_rows[0].keys()) if act_rows else []
    if exp_fields != act_fields:
        differences.append(
            Difference("$.header", exp_fields, act_fields, "CSV headers differ")
        )

    checks += 1
    if len(exp_rows) != len(act_rows):
        differences.append(
            Difference("$.rows", len(exp_rows), len(act_rows), "CSV row counts differ")
        )

    common_fields = [x for x in exp_fields if x in act_fields]
    for i, (er, ar) in enumerate(zip(exp_rows, act_rows)):
        for field in common_fields:
            e = _parse_csv_scalar(er[field])
            a = _parse_csv_scalar(ar[field])
            d, n = compare_objects(
                e, a, path=f"$[{i}].{field}", rtol=rtol, atol=atol
            )
            differences.extend(d)
            checks += n

    return VerificationResult(
        status="PASS" if not differences else "FAIL",
        checks=checks,
        failures=len(differences),
        differences=differences,
    )


def verify_gold_integrity(expected_dir: Path) -> VerificationResult:
    """
    Verify immutable-gold internal provenance and per-seed structure.

    This intentionally verifies only information that exists inside the
    pre-refactor gold package. It does not attempt to rerun experiments.
    """
    expected_dir = expected_dir.resolve()
    gold_path = expected_dir / "gold_results.json"
    if not gold_path.exists():
        return VerificationResult(
            status="FAIL",
            checks=1,
            failures=1,
            differences=[Difference(
                "$.gold_results.json", "present", "missing",
                "gold master file missing"
            )],
        )

    gold = load_json(gold_path)
    differences: list[Difference] = []
    checks = 0

    # Master status/schema checks.
    checks += 1
    if gold.get("status") != "FROZEN_STEP_1B":
        differences.append(
            Difference("$.status", "FROZEN_STEP_1B", gold.get("status"),
                       "unexpected gold status")
        )

    checks += 1
    if gold.get("schema_version") != 1:
        differences.append(
            Difference("$.schema_version", 1, gold.get("schema_version"),
                       "unexpected schema version")
        )

    # Verify compact source snapshots against the frozen hashes.
    for alias, info in sorted(gold.get("source_files", {}).items()):
        snapshot = info.get("snapshot")
        if not snapshot:
            continue
        path = expected_dir / snapshot
        checks += 1
        if not path.exists():
            differences.append(
                Difference(f"$.source_files.{alias}.snapshot",
                           snapshot, "missing", "frozen snapshot missing")
            )
            continue
        actual_hash = sha256_file(path)
        expected_hash = info["sha256"]
        checks += 1
        if actual_hash != expected_hash:
            differences.append(
                Difference(
                    f"$.source_files.{alias}.sha256",
                    expected_hash,
                    actual_hash,
                    "frozen snapshot hash mismatch",
                )
            )

    # Verify known per-seed bundles are complete and ordered 0..29.
    per_seed = expected_dir / "per_seed"
    json_patterns = {
        "stage5": "stage5_per_seed_full.json",
        "stage6": "stage6_per_seed_full.json",
        "stage7": "stage7_per_seed_full.json",
        "stage9_fixed": "stage9_strength_fixed_identity_per_seed_full.json",
        "stage10_full_background": "stage10_per_seed_metrics.json",
        "stage11_loo": "stage11_self_influence_loo_per_seed_full.json",
    }

    for stage, filename in json_patterns.items():
        path = per_seed / filename
        checks += 1
        if not path.exists():
            differences.append(
                Difference(f"$.per_seed.{stage}", filename, "missing",
                           "per-seed gold file missing")
            )
            continue

        data = load_json(path)
        seeds = [int(x["seed"]) for x in data]
        checks += 1
        if seeds != list(range(30)):
            differences.append(
                Difference(
                    f"$.per_seed.{stage}.seeds",
                    list(range(30)),
                    seeds,
                    "expected exactly seeds 0..29",
                )
            )

    # Stage10 compact manifest must itself claim 30 seeds 0..29.
    st10_manifest_path = per_seed / "stage10_compact_manifest.json"
    checks += 1
    if not st10_manifest_path.exists():
        differences.append(
            Difference("$.stage10.manifest", "present", "missing",
                       "Stage10 compact manifest missing")
        )
    else:
        m = load_json(st10_manifest_path)
        checks += 1
        if m.get("num_seeds") != 30:
            differences.append(
                Difference("$.stage10.num_seeds", 30, m.get("num_seeds"),
                           "Stage10 seed count mismatch")
            )
        checks += 1
        if m.get("seeds") != list(range(30)):
            differences.append(
                Difference("$.stage10.seeds", list(range(30)), m.get("seeds"),
                           "Stage10 seeds mismatch")
            )

        # Cross-check hashes in Stage10 manifest against per-seed JSON records.
        st10_json_path = per_seed / "stage10_per_seed_metrics.json"
        if st10_json_path.exists():
            rows = load_json(st10_json_path)
            expected_hashes = m.get("input_summary_hashes", {})
            for row in rows:
                seed = int(row["seed"])
                key = f"seed_{seed:03d}"
                checks += 1
                if row.get("source_summary_sha256") != expected_hashes.get(key):
                    differences.append(
                        Difference(
                            f"$.stage10.hashes.{key}",
                            expected_hashes.get(key),
                            row.get("source_summary_sha256"),
                            "Stage10 source-summary hash disagreement",
                        )
                    )

    return VerificationResult(
        status="PASS" if not differences else "FAIL",
        checks=checks,
        failures=len(differences),
        differences=differences,
    )


def compare_paths(
    expected: Path,
    actual: Path,
    *,
    rtol: float,
    atol: float,
) -> VerificationResult:
    if expected.suffix.lower() == ".json" and actual.suffix.lower() == ".json":
        return compare_json_files(expected, actual, rtol=rtol, atol=atol)
    if expected.suffix.lower() == ".csv" and actual.suffix.lower() == ".csv":
        return compare_csv_files(expected, actual, rtol=rtol, atol=atol)
    raise ValueError("compare currently supports JSON-to-JSON or CSV-to-CSV")


def print_result(result: VerificationResult, *, max_diffs: int = 20) -> None:
    print("=" * 72)
    print(f"STATUS:   {result.status}")
    print(f"CHECKS:   {result.checks}")
    print(f"FAILURES: {result.failures}")
    if result.differences:
        print("-" * 72)
        for diff in result.differences[:max_diffs]:
            print(f"{diff.path}: {diff.reason}")
            print(f"  expected: {diff.expected!r}")
            print(f"  actual:   {diff.actual!r}")
        if len(result.differences) > max_diffs:
            print(f"... {len(result.differences) - max_diffs} additional differences omitted")
    print("=" * 72)


def write_report(result: VerificationResult, report_path: Path | None) -> None:
    if report_path is None:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_selftest(expected_dir: Path) -> bool:
    """
    Safety test:
      1. gold integrity must PASS;
      2. comparing an untouched copy of gold_results.json must PASS;
      3. an intentional scientific-number perturbation must FAIL;
      4. a CSV perturbation in a per-seed metric must FAIL.
    """
    print("[selftest] 1/4 gold integrity")
    integrity = verify_gold_integrity(expected_dir)
    print_result(integrity, max_diffs=5)
    if integrity.status != "PASS":
        return False

    with tempfile.TemporaryDirectory(prefix="aggregate_reuse_verifier_") as td:
        td = Path(td)
        expected_gold = expected_dir / "gold_results.json"

        print("[selftest] 2/4 untouched JSON copy")
        clean_json = td / "gold_clean.json"
        shutil.copy2(expected_gold, clean_json)
        r = compare_json_files(expected_gold, clean_json)
        print_result(r, max_diffs=5)
        if r.status != "PASS":
            return False

        print("[selftest] 3/4 intentional JSON perturbation")
        bad_json = td / "gold_bad.json"
        obj = load_json(expected_gold)
        obj["reported_prose_full_precision"]["controlled_null"]["raw_mean_w1"] += 0.01
        bad_json.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
        r = compare_json_files(expected_gold, bad_json)
        print_result(r, max_diffs=5)
        if r.status != "FAIL":
            print("ERROR: verifier failed to catch intentional JSON perturbation")
            return False

        print("[selftest] 4/4 intentional CSV perturbation")
        exp_csv = expected_dir / "per_seed" / "stage10_per_seed_metrics.csv"
        bad_csv = td / "stage10_bad.csv"
        with exp_csv.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            fieldnames = rows[0].keys()

        target = "predictive_centered_w1.all.planted_background_percentile_median"
        rows[0][target] = str(float(rows[0][target]) + 0.01)

        with bad_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        r = compare_csv_files(exp_csv, bad_csv)
        print_result(r, max_diffs=5)
        if r.status != "FAIL":
            print("ERROR: verifier failed to catch intentional CSV perturbation")
            return False

    print("VERIFIER SELFTEST: PASS")
    return True


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("integrity", help="verify immutable gold internal integrity")
    p.add_argument("--expected-dir", default="paper_results/expected")
    p.add_argument("--report")

    p = sub.add_parser("compare", help="compare one expected JSON/CSV with one actual JSON/CSV")
    p.add_argument("--expected", required=True)
    p.add_argument("--actual", required=True)
    p.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    p.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    p.add_argument("--report")

    p = sub.add_parser("selftest", help="prove the verifier passes clean data and rejects perturbations")
    p.add_argument("--expected-dir", default="paper_results/expected")

    return ap


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "integrity":
        result = verify_gold_integrity(Path(args.expected_dir))
        print_result(result)
        write_report(result, Path(args.report) if args.report else None)
        return 0 if result.status == "PASS" else 1

    if args.command == "compare":
        result = compare_paths(
            Path(args.expected),
            Path(args.actual),
            rtol=args.rtol,
            atol=args.atol,
        )
        print_result(result)
        write_report(result, Path(args.report) if args.report else None)
        return 0 if result.status == "PASS" else 1

    if args.command == "selftest":
        ok = run_selftest(Path(args.expected_dir))
        return 0 if ok else 1

    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
