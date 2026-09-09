from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.paths import AmazonPathLayout
from aggregate_reuse.reporting import amazon_figure_data as figure_data


RAW_SHA256 = "a" * 64


def ci(mean: float) -> dict[str, float]:
    return {
        "mean": mean,
        "ci_lower": mean - 0.001,
        "ci_upper": mean + 0.001,
    }


def summary_base(stage: str) -> dict:
    return {
        "category": "electronics",
        "stage": stage,
        "n_seeds": 30,
        "seed_ids": list(range(30)),
        "selected_lambda": 20.0,
        "raw_dataset_sha256": RAW_SHA256,
        "provenance": {
            "category": "electronics",
            "seed_ids": list(range(30)),
            "selected_lambda": 20.0,
            "raw_dataset_sha256": RAW_SHA256,
        },
    }


def make_layout(tmp_path: Path) -> AmazonPathLayout:
    return AmazonPathLayout(
        repository_root=tmp_path,
        category="electronics",
        shared_category_root=tmp_path / "shared" / "electronics",
        raw_dataset=tmp_path / "Electronics.jsonl.gz",
    )


def write_sources(layout: AmazonPathLayout) -> dict[str, dict]:
    by_spec = {spec.key: spec for spec in figure_data.SOURCE_SPECS}

    stage4 = summary_base(by_spec["stage4_reuse"].stage)
    stage4["reuse_summary"] = {
        str(reuse): {
            "score_auc": ci(0.50 + reuse / 1000),
            "frequency_auc": ci(0.60 + reuse / 1000),
            "coalition_mean_score": ci(0.70 + reuse / 1000),
        }
        for reuse in figure_data.REUSE_GRID
    }

    stage5 = summary_base(by_spec["stage5_twins"].stage)
    stage5["metrics"] = {
        "frequency_auc": ci(0.51),
        "counterfactual_score_auc": ci(0.52),
        "raw_world_score_auc": ci(0.53),
        "predictive_centered_world_score_auc": ci(0.54),
    }

    stage6 = summary_base(by_spec["stage6_shape"].stage)
    stage6["metrics"] = {
        "frequency_auc": ci(0.61),
        "mean_counterfactual_score_auc": ci(0.62),
        "w1_counterfactual_score_auc": ci(0.63),
        "js_counterfactual_score_auc": ci(0.64),
    }

    stage7 = summary_base(by_spec["stage7_complementarity"].stage)
    stage7["metrics"] = {
        "aggregate_auc": ci(0.71),
        "coactivity_auc": ci(0.72),
        "combined_auc": ci(0.73),
    }

    stage8 = summary_base(by_spec["stage8_reference_history"].stage)
    stage8["metrics_by_reference_length"] = {
        str(length): {
            "counterfactual_score_auc": ci(0.80 + length / 10000),
            "selected_lambda": 20.0,
        }
        for length in figure_data.REFERENCE_LENGTHS
    }

    stage9 = summary_base(by_spec["stage9_strength_fixed"].stage)
    stage9["metrics_by_k"] = {
        str(strength): {
            "counterfactual_score_auc": ci(0.90 + strength / 1000),
            "mean_d_cf": ci(0.10 + strength / 1000),
        }
        for strength in figure_data.STRENGTH_GRID
    }

    summaries = {
        "stage4_reuse": stage4,
        "stage5_twins": stage5,
        "stage6_shape": stage6,
        "stage7_complementarity": stage7,
        "stage8_reference_history": stage8,
        "stage9_strength_fixed": stage9,
    }
    for spec in figure_data.SOURCE_SPECS:
        path = layout.artifact(spec.artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(summaries[spec.key], sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summaries


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_generator_maps_all_six_stages_and_verifies_every_csv_cell(tmp_path):
    layout = make_layout(tmp_path)
    summaries = write_sources(layout)
    output = tmp_path / "artifacts" / "electronics" / "figure_data"

    main_path, robustness_path = figure_data.generate_figure_data(layout, output)
    report = figure_data.verify_figure_data(layout, output)

    assert main_path.name == "amazon_main_figure_data.csv"
    assert robustness_path.name == "amazon_robustness_figure_data.csv"
    assert report["status"] == "PASS"
    assert report["gold_comparison_performed"] is False
    assert report["source_summary_count"] == 6
    assert report["totals"] == {
        "csv_files_checked": 2,
        "header_fields_checked": 14,
        "data_rows_checked": 32,
        "data_cells_checked": 210,
        "all_values_match_source_summaries": True,
    }

    main = read_csv(main_path)
    robustness = read_csv(robustness_path)
    assert len(main) == 23
    assert len(robustness) == 9
    assert float(main[0]["mean"]) == summaries["stage4_reuse"][
        "reuse_summary"
    ]["1"]["score_auc"]["mean"]
    assert float(main[16]["mean"]) == summaries["stage5_twins"]["metrics"][
        "counterfactual_score_auc"
    ]["mean"]
    assert float(main[21]["mean"]) == summaries["stage6_shape"]["metrics"][
        "w1_counterfactual_score_auc"
    ]["mean"]
    assert float(robustness[2]["mean"]) == summaries[
        "stage7_complementarity"
    ]["metrics"]["combined_auc"]["mean"]
    assert float(robustness[4]["aux_value"]) == summaries[
        "stage8_reference_history"
    ]["metrics_by_reference_length"]["90"]["selected_lambda"]
    assert float(robustness[8]["aux_value"]) == summaries[
        "stage9_strength_fixed"
    ]["metrics_by_k"]["9"]["mean_d_cf"]["mean"]


@pytest.mark.parametrize(
    ("filename", "fields", "field", "replacement", "message"),
    [
        (
            figure_data.MAIN_FILENAME,
            figure_data.MAIN_FIELDS,
            "panel",
            "wrong_panel",
            "exact CSV mismatch",
        ),
        (
            figure_data.MAIN_FILENAME,
            figure_data.MAIN_FIELDS,
            "mean",
            "0.999",
            "source-summary mismatch",
        ),
        (
            figure_data.ROBUSTNESS_FILENAME,
            figure_data.ROBUSTNESS_FIELDS,
            "aux_value",
            "0.0",
            "exact CSV mismatch",
        ),
    ],
)
def test_verifier_rejects_changed_static_numeric_and_blank_values(
    tmp_path,
    filename,
    fields,
    field,
    replacement,
    message,
):
    layout = make_layout(tmp_path)
    write_sources(layout)
    output = tmp_path / "out"
    figure_data.generate_figure_data(layout, output)
    path = output / filename
    rows = read_csv(path)
    rows[0][field] = replacement
    write_csv(path, fields, rows)

    with pytest.raises(figure_data.FigureDataError, match=message):
        figure_data.verify_figure_data(layout, output)


def test_verifier_rejects_header_row_count_and_source_changes(tmp_path):
    layout = make_layout(tmp_path)
    sources = write_sources(layout)
    output = tmp_path / "out"
    figure_data.generate_figure_data(layout, output)
    main_path = output / figure_data.MAIN_FILENAME

    rows = read_csv(main_path)
    write_csv(main_path, tuple(reversed(figure_data.MAIN_FIELDS)), rows)
    with pytest.raises(figure_data.FigureDataError, match="CSV header mismatch"):
        figure_data.verify_figure_data(layout, output)

    figure_data.generate_figure_data(layout, output)
    rows = read_csv(main_path)
    write_csv(main_path, figure_data.MAIN_FIELDS, rows[:-1])
    with pytest.raises(figure_data.FigureDataError, match="row count mismatch"):
        figure_data.verify_figure_data(layout, output)

    figure_data.generate_figure_data(layout, output)
    spec = next(spec for spec in figure_data.SOURCE_SPECS if spec.key == "stage5_twins")
    sources["stage5_twins"]["metrics"]["counterfactual_score_auc"]["mean"] += 0.0001
    layout.artifact(spec.artifact).write_text(
        json.dumps(sources["stage5_twins"], sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(figure_data.FigureDataError, match="source-summary mismatch"):
        figure_data.verify_figure_data(layout, output)


def test_electronics_sources_reject_home_category_leakage(tmp_path):
    layout = make_layout(tmp_path)
    sources = write_sources(layout)
    spec = figure_data.SOURCE_SPECS[0]
    sources[spec.key]["provenance"]["upstream"] = (
        "shared_data/home_and_kitchen/stage3.json"
    )
    layout.artifact(spec.artifact).write_text(
        json.dumps(sources[spec.key], sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(figure_data.FigureDataError, match="Home-and-Kitchen"):
        figure_data.load_source_summaries(layout)


def test_generator_and_verifier_do_not_read_packaged_gold():
    paths = (
        Path(figure_data.__file__),
        ROOT / "experiments"
        / "reporting"
        / "generate_amazon_figure_data.py",
        ROOT / "tools"
        / "verify_electronics_figure_data.py",
    )
    for path in paths:
        assert "paper_results/expected" not in path.read_text(encoding="utf-8")


def test_home_figure_data_preserves_legacy_robustness_filename(tmp_path):
    layout = AmazonPathLayout(
        repository_root=tmp_path,
        category="home_and_kitchen",
        shared_category_root=tmp_path / "shared" / "home",
        raw_dataset=tmp_path / "Home_and_Kitchen.jsonl.gz",
    )
    assert figure_data.robustness_filename(layout) == "fig3_amazon_robustness_figure_data.csv"
