from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_NAME = "amazon_" + "preprocessed"

TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".txt", ".toml"}


def test_no_stale_shared_directory_name():
    offenders = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if ".pytest_cache" in p.parts or "tests" in p.parts:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if OLD_NAME in text:
            offenders.append(str(p.relative_to(ROOT)))

    assert not offenders, offenders
