#!/usr/bin/env python3
"""Run invariant-only verification for the Electronics Amazon experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from _verify_electronics import (
    ROOT,
    STAGE_ORDER,
    build_context,
    run_selftest,
    run_verification,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Electronics artifacts internally without comparing them "
            "against Home-and-Kitchen gold."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("all", *STAGE_ORDER),
        default="all",
        help="Run one stage verifier or the complete consolidated checker.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=ROOT / "configs" / "electronics",
    )
    parser.add_argument(
        "--shared-root",
        type=Path,
        help="Override the configured ../amazon_preprocess/electronics root.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "artifacts" / "electronics" / "verification",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run verifier negative controls without reading experiment data.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.selftest:
        return run_selftest()
    context = build_context(
        config_dir=args.config_dir,
        shared_root=args.shared_root,
        report_dir=args.report_dir,
    )
    stages = STAGE_ORDER if args.stage == "all" else (args.stage,)
    status, _ = run_verification(context, stages=stages)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
