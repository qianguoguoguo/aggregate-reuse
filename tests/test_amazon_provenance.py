from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.provenance import (
    build_provenance,
    discover_raw_dataset_sha256,
    environment_versions,
    raw_dataset_sha256_from_artifact,
    resolve_raw_dataset_sha256,
    require_artifact_category,
    sha256_file,
)
from aggregate_reuse.amazon.paths import AmazonPathLayout


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(b"amazon-electronics\n")
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_provenance_records_portable_paths_and_hashes(tmp_path):
    repo = tmp_path / "repo"
    shared = tmp_path / "amazon_preprocess" / "electronics"
    config = repo / "configs" / "electronics" / "amazon_primary.yaml"
    upstream = shared / "stage3" / "electronics_reference_freeze.json"
    config.parent.mkdir(parents=True)
    upstream.parent.mkdir(parents=True)
    config.write_text("category: electronics\n", encoding="utf-8")
    upstream.write_text('{"category":"electronics"}\n', encoding="utf-8")
    raw_hash = hashlib.sha256(b"raw").hexdigest()

    result = build_provenance(
        category="electronics",
        config_path=config,
        raw_dataset_sha256=raw_hash,
        selected_lambda=40,
        seed_ids=[0, 1, 2],
        upstream_artifacts={"reference_freeze": upstream},
        repo_root=repo,
        shared_root=shared,
    )

    assert result["category"] == "electronics"
    assert result["raw_dataset_sha256"] == raw_hash
    assert result["config"]["path"] == (
        "configs/electronics/amazon_primary.yaml"
    )
    assert result["config"]["sha256"] == sha256_file(config)
    assert result["selected_lambda"] == 40.0
    assert result["seed_ids"] == [0, 1, 2]
    assert result["upstream_artifacts"]["reference_freeze"] == {
        "path": "shared_data/stage3/electronics_reference_freeze.json",
        "sha256": sha256_file(upstream),
    }
    assert set(result["environment_versions"]) == {
        "python",
        "numpy",
        "scipy",
        "sklearn",
        "pyyaml",
    }


def test_environment_versions_match_frozen_environment():
    assert environment_versions() == {
        "python": "3.12.10",
        "numpy": "2.5.1",
        "scipy": "1.18.0",
        "sklearn": "1.9.0",
        "pyyaml": "6.0.3",
    }


def test_raw_dataset_sha_is_read_from_provenance():
    raw_hash = hashlib.sha256(b"raw").hexdigest()
    artifact = {"provenance": {"raw_dataset_sha256": raw_hash}}
    assert raw_dataset_sha256_from_artifact(
        artifact,
        label="freeze",
    ) == raw_hash


def test_discover_raw_sha_supports_legacy_scan_input_path(tmp_path):
    repo = tmp_path / "repo"
    shared = repo / "artifacts" / "amazon_preprocess"
    raw = tmp_path / "Home_and_Kitchen.jsonl.gz"
    scan = shared / "scan" / "home_and_kitchen_scan.json"
    raw.write_bytes(b"legacy raw data")
    scan.parent.mkdir(parents=True)
    scan.write_text(
        json.dumps(
            {
                "category": "home_and_kitchen",
                "input_path": str(raw),
            }
        ),
        encoding="utf-8",
    )
    layout = AmazonPathLayout(
        repository_root=repo,
        category="home_and_kitchen",
        shared_category_root=shared,
    )

    assert discover_raw_dataset_sha256(layout) == sha256_file(raw)


def test_resolve_raw_sha_prefers_upstream_artifact(tmp_path):
    raw_hash = hashlib.sha256(b"raw").hexdigest()
    layout = AmazonPathLayout(
        repository_root=tmp_path,
        category="electronics",
        shared_category_root=tmp_path / "shared" / "electronics",
    )
    assert resolve_raw_dataset_sha256(
        layout,
        {"provenance": {"raw_dataset_sha256": raw_hash}},
    ) == raw_hash


def test_missing_or_invalid_raw_dataset_sha_is_rejected():
    with pytest.raises(RuntimeError, match="does not record"):
        raw_dataset_sha256_from_artifact({}, label="freeze")
    with pytest.raises(ValueError, match="64-character"):
        raw_dataset_sha256_from_artifact(
            {"raw_dataset_sha256": "not-a-hash"},
            label="freeze",
        )


def test_category_guard_rejects_cross_category_artifact():
    with pytest.raises(RuntimeError, match="category mismatch"):
        require_artifact_category(
            {"category": "home_and_kitchen"},
            expected="electronics",
            label="reference freeze",
        )


def test_category_guard_allows_legacy_home_missing_category_only():
    require_artifact_category(
        {},
        expected="home_and_kitchen",
        label="legacy Home summary",
    )
    with pytest.raises(RuntimeError, match="category mismatch"):
        require_artifact_category(
            {},
            expected="electronics",
            label="Electronics summary",
        )
