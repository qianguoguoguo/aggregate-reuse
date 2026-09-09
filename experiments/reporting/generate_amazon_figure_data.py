#!/usr/bin/env python3
"""Generate Amazon figure-data CSVs directly from category stage summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.reporting.amazon_figure_data import (
    FigureDataError,
    generate_figure_data,
    load_layout,
    verify_figure_data,
    write_verification_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Amazon main and robustness figure data directly from "
            "the configured category's Stage 4--9 aggregate summaries."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "electronics" / "amazon_preprocess.yaml",
        help="Category-aware Amazon preprocessing configuration.",
    )
    parser.add_argument(
        "--shared-root",
        type=Path,
        help="Override the configuration's shared category root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory; defaults to artifacts/<category>/figure_data.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help=(
            "Verification report path; defaults to "
            "artifacts/<category>/verification/figure_data_report.json."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    layout = load_layout(
        args.config,
        repo_root=ROOT,
        shared_root=args.shared_root,
    )
    output_dir = (
        args.out_dir
        if args.out_dir is not None
        else ROOT / "artifacts" / layout.category / "figure_data"
    )
    report_path = (
        args.report
        if args.report is not None
        else ROOT
        / "artifacts"
        / layout.category
        / "verification"
        / "figure_data_report.json"
    )

    try:
        main_path, robustness_path = generate_figure_data(layout, output_dir)
        report = verify_figure_data(layout, output_dir)
    except (FigureDataError, KeyError, TypeError, ValueError) as exc:
        failure = {
            "schema_version": 1,
            "category": layout.category,
            "status": "FAIL",
            "verification_mode": "source_summary_cell_match",
            "gold_comparison_performed": False,
            "failure": {"error_type": type(exc).__name__, "message": str(exc)},
        }
        write_verification_report(report_path, failure)
        print(f"{layout.category.upper()} FIGURE DATA STATUS: FAIL")
        print(str(exc))
        return 1

    write_verification_report(report_path, report)
    print(f"Generated: {main_path}")
    print(f"Generated: {robustness_path}")
    print(f"Verification report: {report_path}")
    print(
        "Verified "
        f"{report['totals']['data_cells_checked']} data cells and "
        f"{report['totals']['header_fields_checked']} header fields against "
        f"{report['source_summary_count']} source summaries."
    )
    print(f"{layout.category.upper()} FIGURE DATA STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
