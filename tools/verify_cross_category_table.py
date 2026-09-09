#!/usr/bin/env python3
"""Verify the Home-vs-Electronics supplement table from canonical summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.reporting.cross_category_table import (
    ELECTRONICS,
    HOME,
    build_cross_category_table,
    load_category_sources,
)
from aggregate_reuse.reporting.supplement_tables import render_simple_tabular


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home-shared-root", type=Path, default=ROOT.parent / "amazon_preprocess")
    parser.add_argument(
        "--electronics-shared-root",
        type=Path,
        default=ROOT.parent / "amazon_preprocess" / "electronics",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts" / "table_data")
    parser.add_argument("--table-dir", type=Path, default=ROOT / "artifacts" / "tables")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "artifacts" / "verification" / "cross_category_table_report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks: dict[str, bool] = {}
    failures: list[str] = []
    try:
        home = load_category_sources(repository_root=ROOT, category=HOME, shared_root=args.home_shared_root)
        electronics = load_category_sources(
            repository_root=ROOT,
            category=ELECTRONICS,
            shared_root=args.electronics_shared_root,
        )
        rows = build_cross_category_table(home, electronics)
        expected_data = {"tab:supp-electronics": rows}
        data_path = args.data_dir / "cross_category_table.json"
        actual_data = json.loads(data_path.read_text(encoding="utf-8"))
        checks["table_data_matches_canonical_sources"] = actual_data == expected_data
        expected_tex = render_simple_tabular(
            ["Quantity", "Home \\& Kitchen", "Electronics"], rows, alignment="lcc"
        )
        tex_path = args.table_dir / "supp_electronics.tex"
        checks["latex_fragment_matches_table_data"] = tex_path.read_text(encoding="utf-8") == expected_tex
        manifest = json.loads((args.data_dir / "cross_category_table_manifest.json").read_text(encoding="utf-8"))
        checks["manifest_has_both_categories"] = (
            manifest.get("table_label") == "tab:supp-electronics"
            and manifest.get("n_rows") == 10
            and set(manifest.get("home", {})) == {
                "stage2", "stage3", "stage5", "stage6", "stage7", "stage8", "stage9", "population_audit"
            }
            and set(manifest.get("electronics", {})) == {
                "stage2", "stage3", "stage5", "stage6", "stage7", "stage8", "stage9", "population_audit"
            }
        )
        generator = (ROOT / "experiments" / "reporting" / "generate_cross_category_table.py").read_text(encoding="utf-8")
        checks["generator_does_not_read_gold"] = (
            "paper_results/expected" not in generator
            and "supplement_table_gold" not in generator
        )
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")

    status = "PASS" if checks and all(checks.values()) and not failures else "FAIL"
    report = {
        "schema_version": 1,
        "status": status,
        "checks": checks,
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, value in checks.items():
        print(f"{name:45s}: {'PASS' if value else 'FAIL'}")
    for failure in failures:
        print(failure)
    print(f"CROSS-CATEGORY TABLE STATUS: {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
