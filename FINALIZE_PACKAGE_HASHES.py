#!/usr/bin/env python3
"""Create PACKAGE_MANIFEST_SHA256.txt from Git-tracked release files."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

MANIFEST = "PACKAGE_MANIFEST_SHA256.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_files(root: Path) -> list[Path]:
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )

    files: list[Path] = []
    for raw in proc.stdout.split(b"\0"):
        if not raw:
            continue
        rel = Path(raw.decode("utf-8"))
        if rel.as_posix() == MANIFEST:
            continue
        path = root / rel
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", nargs="?", type=Path, default=Path("."))
    args = parser.parse_args()

    root = args.repository.resolve()
    if not (root / "README.md").is_file() or not (root / ".git").exists():
        raise SystemExit(f"Not a Git release repository: {root}")

    out = root / MANIFEST
    rows = []
    for path in tracked_files(root):
        rel = path.relative_to(root).as_posix()
        rows.append(f"{sha256(path)}  {rel}")

    out.write_text("\n".join(rows) + "\n", encoding="ascii")
    print(f"Wrote {len(rows)} hashes to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
