#!/usr/bin/env python3
"""Verify Electronics figure CSVs directly against Stage 4--9 summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.reporting.amazon_figure_data import (
    FigureDataError,
    load_layout,
    verify_figure_data,
    write_verification_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check every Electronics figure-data CSV value against its "
            "canonical source summary without consulting result gold."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "electronics" / "amazon_preprocess.yaml",
    )
    parser.add_argument("--shared-root", type=Path)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "artifacts" / "electronics" / "figure_data",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            ROOT
            / "artifacts"
            / "electronics"
            / "verification"
            / "figure_data_report.json"
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
    if layout.category != "electronics":
        raise ValueError(
            "The Electronics figure-data verifier requires category: electronics."
        )

    try:
        report = verify_figure_data(layout, args.data_dir)
    except (FigureDataError, KeyError, TypeError, ValueError) as exc:
        report = {
            "schema_version": 1,
            "category": "electronics",
            "status": "FAIL",
            "verification_mode": "source_summary_cell_match",
            "gold_comparison_performed": False,
            "failure": {"error_type": type(exc).__name__, "message": str(exc)},
        }
        write_verification_report(args.report, report)
        print("ELECTRONICS FIGURE DATA STATUS: FAIL")
        print(str(exc))
        return 1

    write_verification_report(args.report, report)
    print(
        "Verified "
        f"{report['totals']['data_cells_checked']} data cells and "
        f"{report['totals']['header_fields_checked']} header fields against "
        f"{report['source_summary_count']} source summaries."
    )
    print(f"Verification report: {args.report}")
    print("ELECTRONICS FIGURE DATA STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
