#!/usr/bin/env python3
"""Publish verified staged Home preprocessing to ../amazon_preprocess safely."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "artifacts" / "amazon_preprocess"
DEFAULT_DESTINATION = ROOT.parent / "amazon_preprocess"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    expected_destination = DEFAULT_DESTINATION.resolve()
    if destination != expected_destination:
        raise RuntimeError(
            f"Refusing noncanonical publish destination: {destination}; expected {expected_destination}"
        )
    if not source.is_dir():
        raise FileNotFoundError(source)
    required = [
        source / "stage2" / "home_and_kitchen_stage2_summary.json",
        source / "home_and_kitchen_amazon_preprocess_audit.json",
    ]
    # Legacy Home audit name is allowed for backward-compatible staged output.
    if not required[1].exists():
        required[1] = source / "amazon_preprocess_audit.json"
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Staged Home preprocessing is incomplete:\n  " + "\n  ".join(missing))
    if destination.exists():
        if not args.clean:
            raise FileExistsError(
                f"Destination already exists: {destination}. Use --clean for a fresh publication."
            )
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    print(f"Published Home preprocessing: {source} -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
