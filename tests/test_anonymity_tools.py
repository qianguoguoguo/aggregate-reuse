from __future__ import annotations

import importlib.util
from pathlib import Path

from aggregate_reuse.amazon.provenance import external_path_label, logical_path


ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_logical_path_hides_external_parent(tmp_path):
    repo = tmp_path / "repo"
    shared = tmp_path / "amazon_preprocess" / "electronics"
    repo.mkdir()
    shared.mkdir(parents=True)
    inside = repo / "configs" / "x.yaml"
    inside.parent.mkdir()
    inside.write_text("x", encoding="utf-8")
    upstream = shared / "stage3" / "summary.json"
    upstream.parent.mkdir()
    upstream.write_text("{}", encoding="utf-8")
    outside = tmp_path / "raw" / "Electronics.jsonl.gz"
    outside.parent.mkdir()
    outside.write_text("x", encoding="utf-8")

    assert logical_path(inside, repo_root=repo) == "configs/x.yaml"
    assert logical_path(upstream, repo_root=repo, shared_root=shared) == "shared_data/stage3/summary.json"
    assert logical_path(outside, repo_root=repo, shared_root=shared) == "external/Electronics.jsonl.gz"
    assert external_path_label(outside) == "external/Electronics.jsonl.gz"


def test_sanitizer_removes_workstation_paths(tmp_path):
    tool = _load_tool("sanitize_release_metadata")
    repo = tmp_path / "repo"
    target = repo / "paper_results" / "expected" / "sample.json"
    target.parent.mkdir(parents=True)
    # Assemble the path dynamically so the anonymous-release scanner does not
    # mistake this test source itself for leaked workstation metadata.
    win_path = "C:" + "\\" + "Users" + "\\" + "example" + "\\" + "project" + "\\" + "artifacts" + "\\" + "x.json"
    target.write_text('{"path": ' + repr(win_path).replace("'", '"').replace("\\", "\\\\") + '}', encoding="utf-8")
    tool.sanitize_repository(repo)
    text = target.read_text(encoding="utf-8")
    assert "example" not in text
    assert "artifacts/x.json" in text


def test_anonymity_scanner_detects_and_then_clears_path(tmp_path):
    scanner = _load_tool("anonymity_scan")
    sanitizer = _load_tool("sanitize_release_metadata")
    repo = tmp_path / "repo"
    target = repo / "paper_results" / "expected" / "sample.json"
    target.parent.mkdir(parents=True)
    win_path = "C:" + "\\" + "Users" + "\\" + "example" + "\\" + "project" + "\\" + "artifacts" + "\\" + "x.json"
    import json
    target.write_text(json.dumps({"path": win_path}), encoding="utf-8")
    first = scanner.scan_repository(repo)
    assert first["status"] == "FAIL"
    sanitizer.sanitize_repository(repo)
    second = scanner.scan_repository(repo)
    assert second["status"] == "PASS"
