from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import _verify_electronics as verifier


def test_inputs_context_does_not_require_stage3_reference_freeze(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    config_dir = repo / "configs" / "electronics"
    shared = tmp_path / "amazon_preprocess" / "electronics"
    stage2 = shared / "stage2"
    config_dir.mkdir(parents=True)
    stage2.mkdir(parents=True)

    (config_dir / "amazon_preprocess.yaml").write_text(
        "category: electronics\ndataset:\n  filename: Electronics.jsonl.gz\n  location: parent_of_repository\n",
        encoding="utf-8",
    )
    (shared / "electronics_amazon_preprocess_audit.json").write_text(
        json.dumps({"raw_dataset_sha256": "a" * 64}), encoding="utf-8"
    )
    (stage2 / "electronics_stage2_summary.json").write_text(
        json.dumps({"n_items": 21409}), encoding="utf-8"
    )

    fake_layout = SimpleNamespace(
        shared_category_root=shared.resolve(),
        raw_dataset=(repo.parent / "Electronics.jsonl.gz").resolve(),
    )
    monkeypatch.setattr(
        verifier.AmazonPathLayout,
        "from_config",
        lambda *args, **kwargs: fake_layout,
    )

    ctx = verifier.build_context(
        repo_root=repo,
        config_dir=config_dir,
        shared_root=shared,
        report_dir=repo / "artifacts" / "electronics" / "verification",
    )

    assert ctx.n_items == 21409
    assert ctx.selected_lambda is None
    assert not (shared / "stage3" / "electronics_reference_freeze.json").exists()
