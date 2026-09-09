#!/usr/bin/env python3
"""Remove Python/test caches that are intentionally excluded from the release."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    removed = 0
    for path in sorted(ROOT.rglob("__pycache__"), reverse=True):
        if path.is_dir():
            shutil.rmtree(path)
            removed += 1
    for pattern in ("*.pyc", "*.pyo"):
        for path in ROOT.rglob(pattern):
            if path.is_file():
                path.unlink()
                removed += 1
    pytest_cache = ROOT / ".pytest_cache"
    if pytest_cache.exists():
        shutil.rmtree(pytest_cache)
        removed += 1
    print(f"Removed {removed} cache objects.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
