"""Shared checks for category-aware Home-and-Kitchen Phase-4 artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CATEGORY = "home_and_kitchen"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_raw_sha256(root: Path) -> str:
    candidates = [
        root / "artifacts" / "verification" / "phase3_smoke_home_and_kitchen_windows_seed0_full.json",
        root / "artifacts" / "verification" / "phase3_smoke_home_and_kitchen_windows_seed0_core.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        value = obj.get("inputs", {}).get("raw_dataset_sha256")
        if isinstance(value, str) and len(value) == 64:
            return value
    raise RuntimeError(
        "Cannot locate the frozen Home_and_Kitchen raw SHA-256 in the Phase-3 verification manifests."
    )


def per_seed_category_matches(actual: dict[str, Any]) -> bool:
    return actual.get("category") == CATEGORY


def aggregate_provenance_matches(
    actual: dict[str, Any],
    *,
    root: Path,
    config_relpath: str,
) -> bool:
    """Validate additive Phase-4 provenance independently of legacy gold."""

    try:
        raw_sha = expected_raw_sha256(root)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError):
        return False

    provenance = actual.get("provenance")
    if not isinstance(provenance, dict):
        return False
    config = provenance.get("config")
    if not isinstance(config, dict):
        return False
    config_path = root / config_relpath
    if not config_path.is_file():
        return False

    if actual.get("category") != CATEGORY:
        return False
    if actual.get("raw_dataset_sha256") != raw_sha:
        return False
    if provenance.get("category") != CATEGORY:
        return False
    if provenance.get("raw_dataset_sha256") != raw_sha:
        return False
    if config.get("path") != config_relpath.replace("\\", "/"):
        return False
    if config.get("sha256") != sha256_file(config_path):
        return False
    if not isinstance(provenance.get("upstream_artifacts"), dict):
        return False
    if not isinstance(provenance.get("environment_versions"), dict):
        return False

    seed_ids = actual.get("seed_ids")
    if seed_ids is not None and provenance.get("seed_ids") != seed_ids:
        return False

    if "selected_lambda" in actual:
        try:
            actual_lambda = float(actual["selected_lambda"])
            provenance_lambda = float(provenance.get("selected_lambda"))
        except (TypeError, ValueError):
            return False
        if actual_lambda != provenance_lambda:
            return False

    return True
