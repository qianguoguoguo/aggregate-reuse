#!/usr/bin/env python3
"""Fail-fast anonymity scanner for the anonymous review artifact."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Any

TEXT_EXTENSIONS = {
    ".py", ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml", ".toml",
    ".md", ".txt", ".tex", ".bib", ".cff", ".ini", ".cfg", ".sh", ".bat",
    ".ps1", ".mk",
}
TEXT_NAMES = {"Makefile", "LICENSE", ".gitignore", "README"}
SKIP_DIRS = {".git", ".venv", "venv", "env", ".pytest_cache"}

# Match concrete workstation identities, not merely words such as "Windows".
HOME_PREFIX = "/" + "home" + "/"
MAC_PREFIX = "/" + "Users" + "/"
PATTERNS = (
    ("windows_user_home", re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\\/\s\"']+", re.I)),
    ("linux_user_home", re.compile(re.escape(HOME_PREFIX) + r"[^/\s\"']+", re.I)),
    ("mac_user_home", re.compile(re.escape(MAC_PREFIX) + r"[^/\s\"']+", re.I)),
    (
        "email_address",
        re.compile(
            r"(?<![A-Za-z0-9._%+-])"
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        ),
    ),
)


def _is_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS or path.name in TEXT_NAMES


def _scan_text(text: str, *, location: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for label, pattern in PATTERNS:
            for match in pattern.finditer(line):
                findings.append({
                    "kind": label,
                    "location": location,
                    "line": line_no,
                    "match": match.group(0),
                })
    return findings


def _scan_json_value(
    value: Any,
    *,
    location: str,
    path: str = "$",
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            findings.extend(
                _scan_json_value(
                    item,
                    location=location,
                    path=f"{path}.{key}",
                )
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(
                _scan_json_value(
                    item,
                    location=location,
                    path=f"{path}[{index}]",
                )
            )
    elif isinstance(value, str):
        for label, pattern in PATTERNS:
            for match in pattern.finditer(value):
                findings.append({
                    "kind": label,
                    "location": location,
                    "json_path": path,
                    "line": None,
                    "match": match.group(0),
                })
    return findings


def _scan_zip(path: Path, repo: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                name = info.filename
                findings.extend(
                    _scan_text(
                        name,
                        location=f"{path.relative_to(repo).as_posix()}::{name}",
                    )
                )
                if info.is_dir() or info.file_size > 5 * 1024 * 1024:
                    continue
                suffix = Path(name).suffix.lower()
                if suffix not in TEXT_EXTENSIONS:
                    continue
                try:
                    text = zf.read(info).decode("utf-8")
                except Exception:
                    continue
                findings.extend(
                    _scan_text(
                        text,
                        location=f"{path.relative_to(repo).as_posix()}::{name}",
                    )
                )
    except zipfile.BadZipFile:
        findings.append({
            "kind": "invalid_zip",
            "location": path.relative_to(repo).as_posix(),
            "line": None,
            "match": "invalid zip archive",
        })
    return findings



# Files created by a reproduction run but not distributed in the GitHub
# artifact. These mirror the current repository .gitignore and are used only
# when the source tree has no .git directory (for example, a GitHub source ZIP).
RELEASE_EXCLUDED_PREFIXES = (
    "artifacts/results/",
    "artifacts/per_seed/",
    "artifacts/figure_data/",
    "artifacts/figures/",
    "artifacts/table_data/",
    "artifacts/tables/",
    "artifacts/manifests/",
    "artifacts/verification/",
    "amazon_preprocess/",
    "phase3-inputs/",
    "_smoke/",
)
RELEASE_EXCLUDED_FILES = {
    ".env",
    "Home_and_Kitchen.jsonl.gz",
    "Electronics.jsonl.gz",
    "PACKAGE_MANIFEST_SHA256.txt.tmp",
}


def _git_tracked_files(repo: Path) -> list[Path] | None:
    """Return tracked files when *repo* is a Git worktree, else ``None``."""
    if not (repo / ".git").exists():
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "-z"],
            capture_output=True,
            check=True,
        )
    except Exception:
        return None

    paths: list[Path] = []
    for raw in proc.stdout.split(b"\0"):
        if not raw:
            continue
        rel = Path(raw.decode("utf-8", errors="surrogateescape"))
        path = repo / rel
        if path.is_file():
            paths.append(path)
    return sorted(paths)


def _fallback_release_files(repo: Path) -> list[Path]:
    """Approximate the GitHub source tree when .git metadata is absent."""
    files: list[Path] = []
    for path in repo.rglob("*"):
        rel = path.relative_to(repo)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if not path.is_file():
            continue
        posix = rel.as_posix()
        if posix in RELEASE_EXCLUDED_FILES:
            continue
        if any(posix.startswith(prefix) for prefix in RELEASE_EXCLUDED_PREFIXES):
            continue
        if path.suffix.lower() in {".pyc", ".pyo", ".bak"}:
            continue
        if any("phase4" in part.lower() and "backup" in part.lower() for part in rel.parts[:-1]):
            continue
        files.append(path)
    return sorted(files)


def _candidate_files(repo: Path, scope: str) -> tuple[list[Path], str]:
    if scope == "worktree":
        return sorted(path for path in repo.rglob("*") if path.is_file()), "worktree"

    tracked = _git_tracked_files(repo)
    if tracked is not None:
        return tracked, "git_tracked"
    return _fallback_release_files(repo), "source_archive_fallback"

def scan_repository(
    repo: Path,
    *,
    check_git: bool = False,
    skip_paths: set[Path] | None = None,
    scope: str = "release",
) -> dict[str, Any]:
    """Scan the anonymous release artifact.

    ``scope="release"`` scans only Git-tracked files when Git metadata is
    available. This is the canonical anonymity gate for the GitHub artifact.
    For a GitHub source ZIP, where ``.git`` is absent, it falls back to the
    same release exclusions used by this repository's ``.gitignore``.

    ``scope="worktree"`` preserves the legacy whole-working-tree hygiene
    scan and may intentionally flag local/generated files.
    """
    repo = repo.resolve()
    resolved_skip_paths = {
        path.resolve()
        for path in (skip_paths or set())
    }

    findings: list[dict[str, Any]] = []
    scanned_files = 0
    candidates, effective_scope = _candidate_files(repo, scope)

    for path in candidates:
        rel = path.relative_to(repo)

        if any(part in SKIP_DIRS for part in rel.parts):
            continue

        # Skip the scanner's requested output file exactly, if any. This makes
        # repeated scans stable even when the previous report recorded a
        # finding containing a local username/path.
        if path.resolve() in resolved_skip_paths:
            continue

        # Backward-compatible exclusion for the historical default report name.
        if path.name == "anonymity_report.json":
            continue

        scanned_files += 1

        if path.suffix.lower() in {".pyc", ".pyo"}:
            findings.append({
                "kind": "python_bytecode",
                "location": rel.as_posix(),
                "line": None,
                "match": path.name,
            })
            continue

        if path.suffix.lower() == ".zip":
            findings.extend(_scan_zip(path, repo))
            continue

        if _is_text(path):
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue

            if path.suffix.lower() == ".json":
                try:
                    value = json.loads(text)
                except Exception:
                    findings.extend(_scan_text(text, location=rel.as_posix()))
                else:
                    findings.extend(_scan_json_value(value, location=rel.as_posix()))
            else:
                findings.extend(_scan_text(text, location=rel.as_posix()))

        elif path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".webp"}:
            # Metadata and embedded generator paths are generally ASCII/UTF-8.
            try:
                text = path.read_bytes().decode("latin-1", errors="ignore")
            except Exception:
                continue
            findings.extend(_scan_text(text, location=rel.as_posix()))

    git_checked = False

    if check_git and (repo / ".git").exists():
        git_checked = True
        try:
            proc = subprocess.run(
                ["git", "-C", str(repo), "log", "--format=%an <%ae>"],
                text=True,
                capture_output=True,
                check=True,
            )
            for line_no, line in enumerate(proc.stdout.splitlines(), start=1):
                email_hits = _scan_text(line, location="git-history")
                findings.extend(email_hits)
                if line.strip() and not line.lower().startswith("anonymous"):
                    findings.append({
                        "kind": "nonanonymous_git_author",
                        "location": "git-history",
                        "line": line_no,
                        "match": line.strip(),
                    })
        except Exception as exc:
            findings.append({
                "kind": "git_history_unreadable",
                "location": ".git",
                "line": None,
                "match": str(exc),
            })

    return {
        "schema_version": 1,
        "status": "PASS" if not findings else "FAIL",
        "requested_scope": scope,
        "effective_scope": effective_scope,
        "scanned_files": scanned_files,
        "git_history_checked": git_checked,
        "finding_count": len(findings),
        "findings": findings,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--report", type=Path)
    ap.add_argument(
        "--check-git",
        action="store_true",
        help="Also require anonymous Git author metadata.",
    )
    ap.add_argument(
        "--scope",
        choices=("release", "worktree"),
        default="release",
        help=(
            "release: scan only the GitHub release file set (default); "
            "worktree: scan every local file under the repository"
        ),
    )
    args = ap.parse_args()

    repo = args.root.resolve()

    destination: Path | None = None
    skip_paths: set[Path] = set()

    if args.report:
        destination = args.report
        if not destination.is_absolute():
            destination = repo / destination
        destination = destination.resolve()
        skip_paths.add(destination)

    report = scan_repository(
        repo,
        check_git=args.check_git,
        skip_paths=skip_paths,
        scope=args.scope,
    )

    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
