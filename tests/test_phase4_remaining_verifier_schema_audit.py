from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def test_only_reference_history_has_legacy_csv_category_compatibility_issue():
    # The remaining Home verifiers compare JSON or invariants.  Reference
    # history is the only downstream verifier that compares a legacy seed CSV
    # whose category-aware producer adds a category column.
    csv_gold_comparers = []
    for path in TOOLS.glob("verify_*.py"):
        text = path.read_text(encoding="utf-8")
        if "typed_equal_csv" in text or "load_typed_csv(GOLD" in text:
            csv_gold_comparers.append(path.name)
    assert set(csv_gold_comparers) == {
        "verify_amazon_reference.py",
        "verify_reference_history.py",
    }
    ref_text = (TOOLS / "verify_reference_history.py").read_text(encoding="utf-8")
    assert 'required_actual_metadata={"category": "home_and_kitchen"}' in ref_text


def test_remaining_legacy_aggregate_gold_already_contains_seed_ids():
    paths = [
        ROOT / "paper_results/expected/amazon_strength_fixed/home_and_kitchen_stage9_strength_fixed_identity_summary.json",
        ROOT / "paper_results/expected/amazon_full_background_ranking/home_and_kitchen_stage10_ranking_full_background_summary.json",
        ROOT / "paper_results/expected/amazon_self_influence/home_and_kitchen_stage11_self_influence_loo_summary.json",
    ]
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value["seed_ids"] == list(range(30)), path


def test_electronics_reference_history_verifier_is_category_aware():
    text = (TOOLS / "_verify_electronics.py").read_text(encoding="utf-8")
    assert '"category", "seed", "reference_length", "selected_lambda"' in text
    assert '{row["category"] for row in metrics} == {CATEGORY}' in text
    assert '"reference_history": {' in text
    assert '"seed_ids"' in text
