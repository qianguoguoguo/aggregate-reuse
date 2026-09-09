#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _phase3_smoke_common import DEFAULT_ATOL, DEFAULT_RTOL, compare_reports


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare Windows/Linux Phase-3 smoke manifests.")
    ap.add_argument("left", type=Path)
    ap.add_argument("right", type=Path)
    ap.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    ap.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    left = json.loads(args.left.read_text(encoding="utf-8"))
    right = json.loads(args.right.read_text(encoding="utf-8"))
    result = compare_reports(left, right, rtol=args.rtol, atol=args.atol)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
