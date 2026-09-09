from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "verify_results.py"

spec = importlib.util.spec_from_file_location("verify_results", MODULE_PATH)
vr = importlib.util.module_from_spec(spec)
assert spec.loader is not None
# dataclasses expects the module to be registered while class decorators run.
sys.modules[spec.name] = vr
spec.loader.exec_module(vr)


def test_recursive_numeric_tolerance():
    expected = {"x": [1.0, 2.0], "n": 3, "s": "abc"}
    actual = {"x": [1.0 + 1e-13, 2.0], "n": 3, "s": "abc"}
    diffs, _ = vr.compare_objects(expected, actual, rtol=1e-10, atol=1e-12)
    assert not diffs


def test_recursive_detects_perturbation():
    expected = {"x": {"auc": 0.7436}}
    actual = {"x": {"auc": 0.7536}}
    diffs, _ = vr.compare_objects(expected, actual, rtol=1e-10, atol=1e-12)
    assert diffs
    assert diffs[0].path == "$.x.auc"


def test_integer_exactness():
    diffs, _ = vr.compare_objects({"n": 30}, {"n": 31})
    assert diffs


def test_gold_integrity():
    result = vr.verify_gold_integrity(ROOT / "paper_results" / "expected")
    assert result.status == "PASS", result.to_dict()
