#!/usr/bin/env python3
"""Sanitize machine-specific paths from generated release metadata.

This tool intentionally edits only generated JSON/CSV metadata under
``paper_results`` and ``artifacts``. Scientific numerical values, identifiers,
and hashes of experimental worlds are left unchanged. When frozen source
snapshots change only because path strings were sanitized, the corresponding
compact-gold snapshot hashes are refreshed so the internal integrity check
remains self-consistent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path, PureWindowsPath, PurePosixPath
from typing import Any


TEXT_ROOTS = ("paper_results", "artifacts")
_REPO_MARKERS = (
    "src", "experiments", "tools", "configs", "config", "artifacts",
    "paper_results", "figures", "tables", "tests", "docs",
)
_WINDOWS_ABS = re.compile(r"^[A-Za-z]:[\\/]")
_POSIX_HOME = re.compile(r"^/(?:home|Users)/[^/]+/")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _parts_for_absolute(text: str) -> list[str]:
    if _WINDOWS_ABS.match(text):
        return list(PureWindowsPath(text).parts)
    if text.startswith("/"):
        return list(PurePosixPath(text).parts)
    return []


def portableize_path_string(text: str) -> str:
    """Convert a machine-absolute path string into a portable logical label."""
    if not isinstance(text, str):
        return text
    parts = _parts_for_absolute(text)
    if not parts:
        return text

    lowered = [p.lower() for p in parts]
    # Preserve the scientifically meaningful repository-relative suffix when
    # one of the standard repository roots is visible in the historical path.
    for marker in _REPO_MARKERS:
        indices = [i for i, p in enumerate(lowered) if p == marker.lower()]
        if indices:
            i = indices[-1]
            return "/".join(parts[i:]).replace("\\", "/")

    # Raw datasets and interpreters live intentionally outside the repository;
    # retain only the basename. This conveys the resource identity without a
    # workstation username or checkout location.
    name = parts[-1].replace("\\", "/")
    return f"external/{name}"


def _sanitize_obj(value: Any, *, key: str | None = None) -> tuple[Any, int]:
    changes = 0
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            nv, n = _sanitize_obj(v, key=str(k))
            out[k] = nv
            changes += n
        return out, changes
    if isinstance(value, list):
        out = []
        for item in value:
            nv, n = _sanitize_obj(item, key=key)
            out.append(nv)
            changes += n
        return out, changes
    if isinstance(value, str):
        if key == "project_root" and (_WINDOWS_ABS.match(value) or value.startswith("/")):
            return ".", int(value != ".")
        new = portableize_path_string(value)
        return new, int(new != value)
    return value, 0


def sanitize_json(path: Path) -> int:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    new, changes = _sanitize_obj(obj)
    if changes:
        path.write_text(
            json.dumps(new, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return changes


def sanitize_csv(path: Path) -> int:
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
    except Exception:
        return 0
    changes = 0
    for row in rows:
        for i, cell in enumerate(row):
            new = portableize_path_string(cell)
            if new != cell:
                row[i] = new
                changes += 1
    if changes:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
    return changes


def refresh_gold_hashes(repo: Path) -> dict[str, int]:
    expected = repo / "paper_results" / "expected"
    gold_path = expected / "gold_results.json"
    source_hashes_path = expected / "source_hashes.csv"
    refreshed_gold = 0
    refreshed_csv = 0

    if gold_path.is_file():
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        for info in gold.get("source_files", {}).values():
            snapshot = info.get("snapshot")
            if not snapshot:
                continue
            path = expected / snapshot
            if not path.is_file():
                continue
            digest = sha256_file(path)
            size = path.stat().st_size
            if info.get("sha256") != digest or info.get("size_bytes") != size:
                info["sha256"] = digest
                info["size_bytes"] = size
                refreshed_gold += 1
        if refreshed_gold:
            gold_path.write_text(
                json.dumps(gold, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )

    if source_hashes_path.is_file():
        with source_hashes_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            fields = list(rows[0].keys()) if rows else []
        for row in rows:
            snapshot = (row.get("snapshot") or "").strip()
            if not snapshot:
                continue
            path = expected / snapshot
            if not path.is_file():
                continue
            digest = sha256_file(path)
            size = str(path.stat().st_size)
            if row.get("sha256") != digest or row.get("size_bytes") != size:
                row["sha256"] = digest
                row["size_bytes"] = size
                refreshed_csv += 1
        if refreshed_csv:
            with source_hashes_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    return {"gold_entries": refreshed_gold, "source_hash_rows": refreshed_csv}


def sanitize_repository(repo: Path) -> dict[str, Any]:
    modified_files: list[dict[str, Any]] = []
    for root_name in TEXT_ROOTS:
        root = repo / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() == ".json":
                changes = sanitize_json(path)
            elif path.suffix.lower() == ".csv":
                changes = sanitize_csv(path)
            else:
                changes = 0
            if changes:
                modified_files.append({
                    "path": path.relative_to(repo).as_posix(),
                    "replacements": changes,
                })

    refreshed = refresh_gold_hashes(repo)
    report = {
        "schema_version": 1,
        "scope": list(TEXT_ROOTS),
        "modified_file_count": len(modified_files),
        "replacement_count": sum(x["replacements"] for x in modified_files),
        "modified_files": modified_files,
        "refreshed_compact_gold": refreshed,
    }
    report_path = repo / "artifacts" / "verification" / "anonymity_sanitization.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    args = ap.parse_args()
    repo = args.root.resolve()
    report = sanitize_repository(repo)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
