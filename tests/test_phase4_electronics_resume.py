from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_phase4_windows.py"


def _module():
    spec = importlib.util.spec_from_file_location("phase4_windows_runner_v7", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_cmd_exposes_electronics_only_resume_targets():
    cmd = (ROOT / "RUN_PHASE4_WINDOWS.cmd").read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    targets = [
        "resume-after-electronics-preprocess",
        "resume-after-electronics-reference",
        "resume-after-electronics-primary-attack",
        "resume-after-electronics-primary-reuse",
        "resume-after-electronics-matched-twins",
        "resume-after-electronics-shape",
        "resume-after-electronics-complementarity",
        "resume-after-electronics-reference-history",
        "resume-after-electronics-strength",
        "resume-after-electronics-k6-audit",
        "resume-after-electronics-full-background",
        "resume-after-electronics-self-influence",
    ]
    for target in targets:
        assert target in cmd
        assert f'"{target}"' in runner


def test_electronics_pipeline_verifies_each_stage_immediately(monkeypatch):
    mod = _module()
    events = []

    monkeypatch.setattr(mod, "run", lambda label, args: events.append(("run", label)))
    monkeypatch.setattr(mod, "electronics_verify", lambda stage: events.append(("verify", stage)))
    monkeypatch.setattr(mod, "write_resume_hint", lambda target: events.append(("hint", target)))

    # Avoid the final consolidated verifier runs being relevant to pair checks.
    mod.run_electronics_from(0)

    triples = []
    for stage, label, *_rest in mod.ELECTRONICS_PIPELINE:
        run_pos = events.index(("run", label))
        verify_pos = events.index(("verify", stage))
        assert verify_pos == run_pos + 1
        triples.append((stage, run_pos, verify_pos))
    assert len(triples) == len(mod.ELECTRONICS_PIPELINE)


def test_resume_after_electronics_shape_stays_in_electronics(monkeypatch):
    mod = _module()
    calls = []

    monkeypatch.setattr(mod, "check_environment", lambda: calls.append("env"))
    monkeypatch.setattr(mod, "verify_raw", lambda: calls.append("raw"))
    monkeypatch.setattr(mod, "verify_electronics_prefix", lambda stage: calls.append(("prefix", stage)))
    monkeypatch.setattr(mod, "run_electronics_from", lambda start: calls.append(("run_from", start)))
    monkeypatch.setattr(mod, "finalization_after_experiments", lambda: calls.append("finalize"))

    # If any Home routine is reached, fail loudly.
    for name in [
        "home_preprocess",
        "home_downstream",
        "home_downstream_after_reference",
        "home_downstream_after_reuse",
        "home_downstream_after_shape",
        "home_downstream_after_reference_history",
    ]:
        monkeypatch.setattr(mod, name, lambda n=name: (_ for _ in ()).throw(AssertionError(n)))

    mod.resume_after_electronics_shape()

    shape_idx = mod.electronics_stage_index("shape")
    assert calls == ["env", "raw", ("prefix", "shape"), ("run_from", shape_idx + 1), "finalize"]


def test_electronics_resume_prefix_uses_electronics_verifier_only(monkeypatch):
    mod = _module()
    stages = []
    monkeypatch.setattr(mod, "electronics_verify", lambda stage: stages.append(stage))
    mod.verify_electronics_prefix("shape")
    assert stages == [
        "inputs",
        "reference",
        "primary_attack",
        "primary_reuse",
        "matched_twins",
        "shape",
    ]
