from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.paths import AmazonPathLayout
from aggregate_reuse.reporting.cross_category_table import (
    build_cross_category_table,
    load_category_sources,
)


def ci(mean: float):
    return {"mean": mean, "ci_lower": mean - 0.01, "ci_upper": mean + 0.01}


def write_category(tmp_path: Path, category: str, offset: float):
    layout = AmazonPathLayout(
        repository_root=tmp_path,
        category=category,
        shared_category_root=tmp_path / "shared" / category,
    )
    sources = {
        "stage2_summary": {
            "category": category,
            "stage": "amazon_stage2_frozen_roles_and_references",
            "n_items": 1000 + int(offset * 100),
        },
        "stage3_summary": {
            "category": category,
            "stage": "amazon_stage3_reference_calibration_and_drift_diagnostic",
            "selected_lambda": 20.0,
        },
        "stage5_twins_summary": {
            "category": category,
            "stage": "amazon_stage5_exact_matched_twins_r8",
            "n_seeds": 30,
            "seed_ids": list(range(30)),
            "metrics": {"counterfactual_score_auc": ci(0.70 + offset)},
        },
        "stage6_shape_summary": {
            "category": category,
            "stage": "amazon_stage6_mean_preserving_shape_r8",
            "n_seeds": 30,
            "seed_ids": list(range(30)),
            "metrics": {
                "w1_counterfactual_score_auc": ci(0.80 + offset),
                "js_counterfactual_score_auc": ci(0.90 + offset),
            },
        },
        "stage7_complementarity_summary": {
            "category": category,
            "stage": "amazon_stage7_complementarity",
            "n_seeds": 30,
            "seed_ids": list(range(30)),
            "metrics": {"combined_auc": ci(0.85 + offset)},
        },
        "stage8_reference_history_summary": {
            "category": category,
            "stage": "amazon_stage8_reference_history_robustness",
            "n_seeds": 30,
            "seed_ids": list(range(30)),
            "metrics_by_reference_length": {
                "60": {"counterfactual_score_auc": ci(0.60 + offset)},
                "90": {"counterfactual_score_auc": ci(0.65 + offset)},
                "120": {"counterfactual_score_auc": ci(0.70 + offset)},
            },
        },
        "stage9_strength_summary": {
            "category": category,
            "stage": "amazon_stage9_intervention_strength_fixed_identity",
            "n_seeds": 30,
            "seed_ids": list(range(30)),
            "metrics_by_k": {
                "3": {"mean_d_cf": ci(-0.05 + offset)},
                "6": {"mean_d_cf": ci(0.01 + offset)},
                "9": {"mean_d_cf": ci(0.15 + offset)},
            },
        },
        "stage9_k6_population_summary": {
            "category": category,
            "population_results": {
                "k6_feasible_population": {"population_size": 900},
                "k9_feasible_population": {"population_size": 700},
            },
        },
    }
    for artifact, value in sources.items():
        path = layout.artifact(artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return layout


def test_cross_category_table_is_built_only_from_category_sources(tmp_path):
    home_layout = write_category(tmp_path, "home_and_kitchen", 0.0)
    elec_layout = write_category(tmp_path, "electronics", 0.05)
    home = load_category_sources(
        repository_root=tmp_path,
        category="home_and_kitchen",
        shared_root=home_layout.shared_category_root,
    )
    elec = load_category_sources(
        repository_root=tmp_path,
        category="electronics",
        shared_root=elec_layout.shared_category_root,
    )
    rows = build_cross_category_table(home, elec)
    assert len(rows) == 10
    assert rows[0][0].startswith("Eligible items")
    assert rows[4] == ["Matched-twin ROC--AUC", "0.700", "0.750"]
    assert rows[5] == ["Shape $W_1$ ROC--AUC", "0.800", "0.850"]
    assert rows[8][1] == ".600/.650/.700"
    assert rows[9][2] == ".000/.060/.200"


def test_cross_category_generator_does_not_read_frozen_gold():
    paths = [
        ROOT / "src" / "aggregate_reuse" / "reporting" / "cross_category_table.py",
        ROOT / "experiments" / "reporting" / "generate_cross_category_table.py",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "paper_results/expected" not in text
        assert "supplement_table_gold" not in text
