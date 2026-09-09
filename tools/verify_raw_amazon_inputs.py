#!/usr/bin/env python3
"""Verify the two raw Amazon files against the frozen Phase-3 provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_hash(category: str) -> str:
    manifest = ROOT / "artifacts" / "verification" / f"phase3_smoke_{category}_windows_seed0_full.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    return value["inputs"]["raw_dataset_sha256"]


def main() -> int:
    pairs = [
        ("home_and_kitchen", ROOT.parent / "Home_and_Kitchen.jsonl.gz"),
        ("electronics", ROOT.parent / "Electronics.jsonl.gz"),
    ]
    failures = 0
    for category, path in pairs:
        if not path.is_file():
            print(f"{category:18s}: FAIL (missing {path})")
            failures += 1
            continue
        expected = expected_hash(category)
        actual = sha256_file(path)
        passed = actual == expected
        print(f"{category:18s}: {'PASS' if passed else 'FAIL'}")
        print(f"  expected: {expected}")
        print(f"  actual:   {actual}")
        failures += int(not passed)
    print("RAW AMAZON INPUT VERIFICATION:", "PASS" if failures == 0 else "FAIL")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
