from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def test_reporting_registry_has_17_unique_tables():
    x=json.loads((ROOT/"configs"/"reporting_spec.json").read_text())
    labels=[r["label"] for r in x["supplement_tables"]]
    assert len(labels)==17
    assert len(set(labels))==17


def test_static_generator_does_not_read_gold():
    text=(ROOT/"experiments"/"reporting"/"generate_static_tables.py").read_text()
    assert "paper_results/expected" not in text
    assert "supplement_table_gold" not in text


def test_static_generator_uses_canonical_configs():
    text=(ROOT/"experiments"/"reporting"/"generate_static_tables.py").read_text()
    for name in [
        "controlled.yaml",
        "amazon_preprocess.yaml",
        "amazon_shape.yaml",
        "amazon_primary.yaml",
        "amazon_complementarity.yaml",
        "amazon_reference_history.yaml",
        "amazon_strength_fixed.yaml",
        "amazon_full_background.yaml",
        "amazon_self_influence.yaml",
    ]:
        assert name in text


def test_static_outputs_are_tabular_fragments():
    text=(ROOT/"experiments"/"reporting"/"generate_static_tables.py").read_text()
    assert "render_simple_tabular" in text
