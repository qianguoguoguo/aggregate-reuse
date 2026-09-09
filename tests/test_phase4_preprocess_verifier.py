from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "tools" / "verify_amazon_preprocess.py"
    spec = importlib.util.spec_from_file_location("verify_amazon_preprocess", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_preprocess_summary_normalization_ignores_only_additive_provenance():
    mod = _module()
    base = {"category": "home_and_kitchen", "x": 1}
    augmented = {
        **base,
        "raw_dataset_sha256": "a" * 64,
        "provenance": {"category": "home_and_kitchen"},
    }
    assert mod.normalized_scan(base) == mod.normalized_scan(augmented)
    assert mod.normalized_stage1(base) == mod.normalized_stage1(augmented)
    assert mod.normalized_stage2(base) == mod.normalized_stage2(augmented)


def test_phase4_verifies_staged_home_preprocess_explicitly():
    text = (ROOT / "tools" / "run_phase4_windows.py").read_text(encoding="utf-8")
    assert '"tools/verify_amazon_preprocess.py", "--actual-root", "artifacts/amazon_preprocess"' in text

def test_windows_orchestrator_can_resume_after_home_preprocess():
    text = (ROOT / "tools" / "run_phase4_windows.py").read_text(encoding="utf-8")
    cmd = (ROOT / "RUN_PHASE4_WINDOWS.cmd").read_text(encoding="utf-8")
    assert '"resume-after-home-preprocess": resume_after_home_preprocess' in text
    assert "resume-after-home-preprocess" in cmd
