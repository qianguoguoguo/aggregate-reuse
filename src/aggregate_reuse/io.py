"""Small shared I/O helpers used by Amazon experiment runners."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path


def write_csv_gz(path: Path, rows, fieldnames):
    """Write rows to a gzip-compressed CSV with the historical exact format."""
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
