from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


numerical = load_script(
    "phase14_numerical_generator",
    "experiments/reporting/generate_supplement_tables.py",
)
static = load_script(
    "phase14_static_generator",
    "experiments/reporting/generate_static_tables.py",
)
fig2 = load_script("phase14_fig2", "figures/make_fig2_amazon_main.py")
fig3 = load_script("phase14_fig3", "figures/make_fig3_amazon_robustness.py")


def test_electronics_numerical_paths_are_category_scoped():
    shared = ROOT.parent / "amazon_preprocess" / "electronics"
    resolved = numerical.paths(
        category="electronics",
        shared_root=shared,
        include_controlled_grid=False,
    )

    assert set(resolved) == set(numerical.AMAZON_ARTIFACTS)
    assert all(path.is_relative_to(shared) for path in resolved.values())
    assert all("home_and_kitchen" not in str(path) for path in resolved.values())
    assert resolved["reuse"].name == "electronics_stage4_summary.json"
    assert (
        resolved["ranking"].name
        == "electronics_stage10_ranking_full_background_summary.json"
    )


def test_electronics_static_subset_uses_only_electronics_amazon_configs():
    tables, metadata = static.build_static_tables(
        config_dir=ROOT / "configs" / "electronics",
        amazon_only=True,
    )

    assert metadata == {
        "category": "electronics",
        "config_dir": "configs/electronics",
        "shared_root": "../amazon_preprocess/electronics",
        "amazon_only": True,
    }
    assert set(tables) == {
        "tab:supp-map",
        "tab:supp-notation",
        "tab:supp-amazon-splits",
        "tab:supp-shape-transforms",
        "tab:supp-controls",
    }
    assert len(tables["tab:supp-map"]) == 12
    assert all(
        row[1].startswith("configs/electronics/")
        and row[2].startswith("../amazon_preprocess/electronics/")
        for row in tables["tab:supp-map"]
    )
    symbols = {row[0] for row in tables["tab:supp-notation"]}
    assert "$R_{\\rm exp}$" not in symbols
    assert "$p_{\\rm on}$" not in symbols


def test_home_reporting_defaults_remain_legacy_compatible():
    resolved = numerical.paths()
    assert resolved["reuse"] == (
        ROOT.parent
        / "amazon_preprocess"
        / "stage4_reuse"
        / "home_and_kitchen_stage4_summary.json"
    )
    assert "controlled_grid" in resolved

    tables, metadata = static.build_static_tables(
        config_dir=ROOT / "configs",
        amazon_only=False,
    )
    assert metadata["category"] == "home_and_kitchen"
    assert len(tables) == 6
    assert len(tables["tab:supp-map"]) == 15
    assert "tab:supp-rotation-parameters" in tables


def test_only_legacy_home_may_omit_summary_category():
    source = (
        ROOT / "experiments" / "reporting" / "generate_supplement_tables.py"
    ).read_text(encoding="utf-8")
    assert "legacy_home_category" in source
    assert "args.category == HOME_AND_KITCHEN" in source


@pytest.mark.parametrize("module", [fig2, fig3])
def test_figure_stems_are_category_specific_and_path_safe(module):
    assert all("electronics" in stem for stem in module.output_stems("electronics"))
    assert all("amazon" in stem for stem in module.output_stems("amazon"))
    with pytest.raises(ValueError, match="Invalid figure stem prefix"):
        module.output_stems("../home")


def test_phase14_make_target_uses_only_electronics_output_directories():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "electronics-reporting:" in makefile
    assert "artifacts/electronics/table_data" in makefile
    assert "artifacts/electronics/tables" in makefile
    assert "artifacts/electronics/figures" in makefile
    assert "--stem-prefix electronics" in makefile


def test_phase14_generators_and_verifier_do_not_read_gold():
    paths = (
        ROOT / "experiments" / "reporting" / "generate_supplement_tables.py",
        ROOT / "experiments" / "reporting" / "generate_static_tables.py",
        ROOT / "tools" / "verify_electronics_reporting.py",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "paper_results/expected" not in text
        assert "supplement_table_gold" not in text
