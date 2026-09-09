from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aggregate_reuse.amazon.paths import AmazonPathLayout
from aggregate_reuse.amazon.provenance import sha256_file
from experiments.amazon import preprocess


@pytest.fixture
def synthetic_electronics_workspace(tmp_path):
    """Create two qualifying Electronics items without touching shared data."""
    repo = tmp_path / "repo"
    repo.mkdir()
    raw = tmp_path / "Electronics.jsonl.gz"

    records = []
    for item_number, asin in enumerate(["ELEC_A", "ELEC_B"]):
        for index in range(300):
            records.append({
                "parent_asin": asin,
                "user_id": f"{asin}_user_{index:03d}",
                "rating": index % 5 + 1,
                "timestamp": 10_000 * item_number + index,
            })

        # A later source line with an earlier timestamp verifies the exact
        # earliest-review user/item deduplication rule.
        records.append({
            "parent_asin": asin,
            "user_id": f"{asin}_user_000",
            "rating": 5,
            "timestamp": -2 + item_number,
        })

    with gzip.open(raw, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    config = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "electronics"
            / "amazon_preprocess.yaml"
        ).read_text(encoding="utf-8")
    )
    config["implementation"] = {
        "scan_progress_every": 0,
        "ingest_progress_every": 0,
        "sqlite_batch_size": 100,
    }
    config_path = repo / "electronics_preprocess.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    shared = tmp_path / "amazon_preprocess"
    home_sentinel = shared / "stage2" / "home_summary.json"
    home_sentinel.parent.mkdir(parents=True)
    home_sentinel.write_text("home remains", encoding="utf-8")

    electronics_old = shared / "electronics" / "obsolete.txt"
    electronics_old.parent.mkdir(parents=True)
    electronics_old.write_text("remove me", encoding="utf-8")

    return {
        "repo": repo,
        "raw": raw,
        "config": config,
        "config_path": config_path,
        "home_sentinel": home_sentinel,
        "electronics_old": electronics_old,
    }


def test_synthetic_electronics_preprocessing_is_isolated_and_complete(
    synthetic_electronics_workspace,
    monkeypatch,
):
    fixture = synthetic_electronics_workspace
    monkeypatch.setattr(preprocess, "ROOT", fixture["repo"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocess.py",
            "--config",
            str(fixture["config_path"]),
            "--clean",
        ],
    )

    assert preprocess.main() == 0

    layout = AmazonPathLayout.from_config(
        fixture["config"],
        repo_root=fixture["repo"],
    )
    raw_hash = sha256_file(fixture["raw"])

    assert fixture["home_sentinel"].read_text(encoding="utf-8") == (
        "home remains"
    )
    assert not fixture["electronics_old"].exists()
    assert layout.shared_category_root == (
        fixture["repo"].parent / "amazon_preprocess" / "electronics"
    )

    expected_artifacts = [
        "scan_summary",
        "scan_item_counts",
        "stage1_corpus",
        "stage1_manifest",
        "stage1_excluded",
        "stage1_summary",
        "stage2_roles",
        "stage2_manifest",
        "stage2_reference_histograms",
        "stage2_summary",
        "preprocess_audit",
    ]
    assert all(layout.artifact(name).is_file() for name in expected_artifacts)

    scan = json.loads(
        layout.artifact("scan_summary").read_text(encoding="utf-8")
    )
    stage1 = json.loads(
        layout.artifact("stage1_summary").read_text(encoding="utf-8")
    )
    stage2 = json.loads(
        layout.artifact("stage2_summary").read_text(encoding="utf-8")
    )
    audit = json.loads(
        layout.artifact("preprocess_audit").read_text(encoding="utf-8")
    )

    assert scan["category"] == "electronics"
    assert scan["duplicate_user_item_records"] == 2
    assert scan["threshold_counts_after_dedup"]["300"] == 2
    assert stage1["category"] == "electronics"
    assert stage1["final"]["final_eligible_items"] == 2
    assert stage2["category"] == "electronics"
    assert stage2["n_items"] == 2
    assert stage2["leave_one_item_out_reference"]["n_per_item"] == 120
    assert audit["category"] == "electronics"
    assert audit["both_experimental_blocks_feasible_k6"] == 2
    assert audit["both_experimental_blocks_feasible_k9"] == 2

    for artifact in [scan, stage1, stage2, audit]:
        assert artifact["raw_dataset_sha256"] == raw_hash
        assert artifact["provenance"]["category"] == "electronics"
        assert artifact["provenance"]["raw_dataset_sha256"] == raw_hash

    with gzip.open(
        layout.artifact("stage1_corpus"),
        "rt",
        encoding="utf-8",
    ) as handle:
        items = [json.loads(line) for line in handle]

    assert [item["asin"] for item in items] == ["ELEC_A", "ELEC_B"]
    for item_number, item in enumerate(items):
        reviews = item["reviews"]
        assert len(reviews) == 300
        assert [review["position"] for review in reviews] == list(range(1, 301))
        assert reviews[0]["user_id"] == f"{item['asin']}_user_000"
        assert reviews[0]["timestamp"] == -2 + item_number
        assert reviews[0]["rating"] == 5

