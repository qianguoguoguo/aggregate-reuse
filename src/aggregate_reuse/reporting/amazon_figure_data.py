"""Generate and verify Amazon figure data from aggregate stage summaries.

The mapping in this module is category-aware and summary-driven.  It never
reads packaged figure data or frozen result gold.  Generated CSV values are
verified cell by cell against the six canonical Stage 4--9 summaries before a
successful report is returned.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from aggregate_reuse.amazon.paths import AmazonPathLayout
from aggregate_reuse.amazon.provenance import sha256_file


MAIN_FILENAME = "amazon_main_figure_data.csv"
ROBUSTNESS_FILENAME = "amazon_robustness_figure_data.csv"
HOME_ROBUSTNESS_FILENAME = "fig3_amazon_robustness_figure_data.csv"

MAIN_FIELDS = (
    "panel",
    "x",
    "metric",
    "mean",
    "ci_lower",
    "ci_upper",
)
ROBUSTNESS_FIELDS = MAIN_FIELDS + ("aux_metric", "aux_value")

EXPECTED_SEEDS = tuple(range(30))
REUSE_GRID = (1, 2, 4, 8, 16)
REFERENCE_LENGTHS = (60, 90, 120)
STRENGTH_GRID = (3, 6, 9)




def robustness_filename(layout: AmazonPathLayout) -> str:
    """Preserve the historical Home filename while keeping category output generic."""
    return (
        HOME_ROBUSTNESS_FILENAME
        if layout.category == "home_and_kitchen"
        else ROBUSTNESS_FILENAME
    )

class FigureDataError(RuntimeError):
    """Raised when source summaries or generated figure data are invalid."""


@dataclass(frozen=True)
class SourceSpec:
    key: str
    artifact: str
    stage: str


SOURCE_SPECS = (
    SourceSpec(
        "stage4_reuse",
        "stage4_reuse_summary",
        "amazon_stage4_fixed_five_star_reuse",
    ),
    SourceSpec(
        "stage5_twins",
        "stage5_twins_summary",
        "amazon_stage5_exact_matched_twins_r8",
    ),
    SourceSpec(
        "stage6_shape",
        "stage6_shape_summary",
        "amazon_stage6_mean_preserving_shape_r8",
    ),
    SourceSpec(
        "stage7_complementarity",
        "stage7_complementarity_summary",
        "amazon_stage7_complementarity",
    ),
    SourceSpec(
        "stage8_reference_history",
        "stage8_reference_history_summary",
        "amazon_stage8_reference_history_robustness",
    ),
    SourceSpec(
        "stage9_strength_fixed",
        "stage9_strength_summary",
        "amazon_stage9_intervention_strength_fixed_identity",
    ),
)


@dataclass(frozen=True)
class LoadedSources:
    layout: AmazonPathLayout
    summaries: Mapping[str, Mapping[str, Any]]
    paths: Mapping[str, Path]
    hashes: Mapping[str, str]
    raw_dataset_sha256: str
    selected_lambda: float


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise FigureDataError(message)


def _finite_number(value: Any, label: str) -> float:
    _ensure(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    _ensure(math.isfinite(result), f"{label} is not finite")
    return result


def _ci_triplet(value: Any, label: str) -> dict[str, float]:
    _ensure(isinstance(value, Mapping), f"{label} is not an object")
    _ensure(
        set(value) == {"mean", "ci_lower", "ci_upper"},
        f"{label} does not have the exact confidence-interval schema",
    )
    result = {
        field: _finite_number(value[field], f"{label}.{field}")
        for field in ("mean", "ci_lower", "ci_upper")
    }
    _ensure(
        result["ci_lower"] <= result["mean"] <= result["ci_upper"],
        f"{label} confidence interval does not contain its mean",
    )
    return result


def load_layout(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    shared_root: str | Path | None = None,
) -> AmazonPathLayout:
    """Load one category layout from an Amazon preprocessing configuration."""

    config_source = Path(config_path).resolve()
    config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    layout = AmazonPathLayout.from_config(config, repo_root=repo_root)
    if shared_root is None:
        return layout
    return AmazonPathLayout(
        repository_root=layout.repository_root,
        category=layout.category,
        shared_category_root=Path(shared_root),
        raw_dataset=layout.raw_dataset,
    )


def load_source_summaries(layout: AmazonPathLayout) -> LoadedSources:
    """Load and cross-check the six summaries mapped into the figure CSVs."""

    summaries: dict[str, Mapping[str, Any]] = {}
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    raw_hashes: set[str] = set()
    selected_lambdas: set[float] = set()

    for spec in SOURCE_SPECS:
        path = layout.artifact(spec.artifact)
        _ensure(path.is_file(), f"missing figure-data source summary: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        _ensure(isinstance(value, Mapping), f"summary root is not an object: {path}")
        _ensure(
            value.get("category") == layout.category,
            f"category mismatch in {path}",
        )
        _ensure(value.get("stage") == spec.stage, f"stage mismatch in {path}")
        _ensure(value.get("n_seeds") == 30, f"seed count mismatch in {path}")
        _ensure(
            value.get("seed_ids") == list(EXPECTED_SEEDS),
            f"seed set mismatch in {path}",
        )
        if layout.category == "electronics":
            _ensure(
                "home_and_kitchen" not in json.dumps(value, sort_keys=True).lower(),
                f"Home-and-Kitchen reference found in Electronics summary: {path}",
            )

        raw_hash = value.get("raw_dataset_sha256")
        _ensure(
            isinstance(raw_hash, str)
            and len(raw_hash) == 64
            and all(char in "0123456789abcdef" for char in raw_hash),
            f"invalid raw dataset SHA-256 in {path}",
        )
        raw_hashes.add(raw_hash)
        selected_lambda = _finite_number(
            value.get("selected_lambda"), f"selected_lambda in {path}"
        )
        selected_lambdas.add(selected_lambda)

        provenance = value.get("provenance")
        _ensure(isinstance(provenance, Mapping), f"missing provenance in {path}")
        _ensure(
            provenance.get("category") == layout.category,
            f"provenance category mismatch in {path}",
        )
        _ensure(
            provenance.get("raw_dataset_sha256") == raw_hash,
            f"provenance raw hash mismatch in {path}",
        )
        _ensure(
            provenance.get("seed_ids") == list(EXPECTED_SEEDS),
            f"provenance seed set mismatch in {path}",
        )
        _ensure(
            _finite_number(
                provenance.get("selected_lambda"),
                f"provenance selected_lambda in {path}",
            )
            == selected_lambda,
            f"provenance selected lambda mismatch in {path}",
        )

        summaries[spec.key] = value
        paths[spec.key] = path
        hashes[spec.key] = sha256_file(path)

    _ensure(len(raw_hashes) == 1, "source summaries disagree on raw dataset SHA-256")
    _ensure(len(selected_lambdas) == 1, "source summaries disagree on selected lambda")
    return LoadedSources(
        layout=layout,
        summaries=summaries,
        paths=paths,
        hashes=hashes,
        raw_dataset_sha256=next(iter(raw_hashes)),
        selected_lambda=next(iter(selected_lambdas)),
    )


def _row(
    panel: str,
    x: str | int,
    metric: str,
    triplet: Any,
    *,
    label: str,
    aux_metric: str | None = None,
    aux_value: float | None = None,
) -> dict[str, str | float]:
    values = _ci_triplet(triplet, label)
    result: dict[str, str | float] = {
        "panel": panel,
        "x": str(x),
        "metric": metric,
        **values,
    }
    if aux_metric is not None:
        result["aux_metric"] = aux_metric
        result["aux_value"] = (
            "" if aux_value is None else _finite_number(aux_value, f"{label}.aux_value")
        )
    return result


def build_main_rows(sources: LoadedSources) -> list[dict[str, str | float]]:
    """Map Stage 4 reuse, Stage 5 twins, and Stage 6 shape summaries."""

    summaries = sources.summaries
    stage4 = summaries["stage4_reuse"]["reuse_summary"]
    rows: list[dict[str, str | float]] = []
    for reuse in REUSE_GRID:
        values = stage4[str(reuse)]
        for metric in ("score_auc", "frequency_auc", "coalition_mean_score"):
            rows.append(
                _row(
                    "a_reuse",
                    reuse,
                    metric,
                    values[metric],
                    label=f"stage4 reuse={reuse} {metric}",
                )
            )

    stage5 = summaries["stage5_twins"]["metrics"]
    for x, source_metric in (
        ("Frequency", "frequency_auc"),
        ("Counterfactual", "counterfactual_score_auc"),
        ("Raw world", "raw_world_score_auc"),
        ("Predictive", "predictive_centered_world_score_auc"),
    ):
        rows.append(
            _row(
                "b_matched",
                x,
                "auc",
                stage5[source_metric],
                label=f"stage5 {source_metric}",
            )
        )

    stage6 = summaries["stage6_shape"]["metrics"]
    for x, source_metric in (
        ("Frequency", "frequency_auc"),
        ("Mean", "mean_counterfactual_score_auc"),
        ("W1", "w1_counterfactual_score_auc"),
        ("JS", "js_counterfactual_score_auc"),
    ):
        rows.append(
            _row(
                "c_shape",
                x,
                "auc",
                stage6[source_metric],
                label=f"stage6 {source_metric}",
            )
        )

    _ensure(len(rows) == 23, "main figure mapping did not produce 23 rows")
    return rows


def build_robustness_rows(
    sources: LoadedSources,
) -> list[dict[str, str | float]]:
    """Map Stages 7--9 into the robustness figure-data schema."""

    summaries = sources.summaries
    stage7 = summaries["stage7_complementarity"]["metrics"]
    rows: list[dict[str, str | float]] = []
    for x, source_metric in (
        ("Aggregate", "aggregate_auc"),
        ("Co-activity", "coactivity_auc"),
        ("Combined", "combined_auc"),
    ):
        rows.append(
            _row(
                "complementarity",
                x,
                "roc_auc",
                stage7[source_metric],
                label=f"stage7 {source_metric}",
                aux_metric="",
            )
        )

    stage8 = summaries["stage8_reference_history"]["metrics_by_reference_length"]
    for reference_length in REFERENCE_LENGTHS:
        values = stage8[str(reference_length)]
        rows.append(
            _row(
                "reference_history",
                reference_length,
                "matched_twin_roc_auc",
                values["counterfactual_score_auc"],
                label=f"stage8 n_ref={reference_length} counterfactual_score_auc",
                aux_metric="selected_lambda",
                aux_value=values["selected_lambda"],
            )
        )

    stage9 = summaries["stage9_strength_fixed"]["metrics_by_k"]
    for strength in STRENGTH_GRID:
        values = stage9[str(strength)]
        rows.append(
            _row(
                "intervention_strength",
                strength,
                "matched_twin_roc_auc",
                values["counterfactual_score_auc"],
                label=f"stage9 k={strength} counterfactual_score_auc",
                aux_metric="mean_d_cf",
                aux_value=values["mean_d_cf"]["mean"],
            )
        )

    _ensure(len(rows) == 9, "robustness figure mapping did not produce 9 rows")
    return rows


def _write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def generate_figure_data(
    layout: AmazonPathLayout,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Generate both CSVs, then verify them against the current summaries."""

    destination = Path(output_dir).resolve()
    sources = load_source_summaries(layout)
    main_path = destination / MAIN_FILENAME
    robustness_path = destination / robustness_filename(layout)
    _write_csv(main_path, MAIN_FIELDS, build_main_rows(sources))
    _write_csv(
        robustness_path,
        ROBUSTNESS_FIELDS,
        build_robustness_rows(sources),
    )
    verify_figure_data(layout, destination)
    return main_path, robustness_path


def _read_csv(path: Path, expected_fields: Sequence[str]) -> list[dict[str, str]]:
    _ensure(path.is_file(), f"missing generated figure-data CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _ensure(
            tuple(reader.fieldnames or ()) == tuple(expected_fields),
            f"CSV header mismatch: {path}",
        )
        rows = list(reader)
    for index, row in enumerate(rows, start=2):
        _ensure(
            set(row) == set(expected_fields) and None not in row.values(),
            f"malformed CSV row {index}: {path}",
        )
    return rows


def _verify_rows(
    path: Path,
    fields: Sequence[str],
    expected_rows: Sequence[Mapping[str, str | float]],
) -> dict[str, int]:
    actual_rows = _read_csv(path, fields)
    _ensure(
        len(actual_rows) == len(expected_rows),
        f"CSV row count mismatch: {path}",
    )
    checked = 0
    numeric_checked = 0
    exact_checked = 0
    for row_number, (actual, expected) in enumerate(
        zip(actual_rows, expected_rows),
        start=2,
    ):
        for field in fields:
            actual_value = actual[field]
            expected_value = expected[field]
            if isinstance(expected_value, float):
                try:
                    parsed = float(actual_value)
                except (TypeError, ValueError) as exc:
                    raise FigureDataError(
                        f"non-numeric CSV value at {path}:{row_number}:{field}"
                    ) from exc
                _ensure(
                    math.isfinite(parsed) and parsed == expected_value,
                    f"source-summary mismatch at {path}:{row_number}:{field}",
                )
                numeric_checked += 1
            else:
                _ensure(
                    actual_value == expected_value,
                    f"exact CSV mismatch at {path}:{row_number}:{field}",
                )
                exact_checked += 1
            checked += 1
    return {
        "header_fields_checked": len(fields),
        "rows_checked": len(actual_rows),
        "data_cells_checked": checked,
        "numeric_cells_checked": numeric_checked,
        "exact_cells_checked": exact_checked,
    }


def _logical_path(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        try:
            relative = resolved.relative_to(repository_root.resolve().parent)
        except ValueError:
            return f"external/{resolved.name}"
        return f"../{relative.as_posix()}"


def verify_figure_data(
    layout: AmazonPathLayout,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Verify every generated header and data cell against source summaries."""

    destination = Path(output_dir).resolve()
    sources = load_source_summaries(layout)
    main_path = destination / MAIN_FILENAME
    robustness_path = destination / robustness_filename(layout)
    main_checks = _verify_rows(
        main_path,
        MAIN_FIELDS,
        build_main_rows(sources),
    )
    robustness_checks = _verify_rows(
        robustness_path,
        ROBUSTNESS_FIELDS,
        build_robustness_rows(sources),
    )

    generated = {
        MAIN_FILENAME: {
            "path": _logical_path(main_path, layout.repository_root),
            "sha256": sha256_file(main_path),
            **main_checks,
            "source_stages": ["stage4_reuse", "stage5_twins", "stage6_shape"],
        },
        robustness_filename(layout): {
            "path": _logical_path(robustness_path, layout.repository_root),
            "sha256": sha256_file(robustness_path),
            **robustness_checks,
            "source_stages": [
                "stage7_complementarity",
                "stage8_reference_history",
                "stage9_strength_fixed",
            ],
        },
    }
    total_cells = sum(item["data_cells_checked"] for item in generated.values())
    total_headers = sum(item["header_fields_checked"] for item in generated.values())
    return {
        "schema_version": 1,
        "category": layout.category,
        "status": "PASS",
        "verification_mode": "source_summary_cell_match",
        "gold_comparison_performed": False,
        "raw_dataset_sha256": sources.raw_dataset_sha256,
        "selected_lambda": sources.selected_lambda,
        "source_summary_count": len(SOURCE_SPECS),
        "source_summaries": {
            spec.key: {
                "path": _logical_path(sources.paths[spec.key], layout.repository_root),
                "sha256": sources.hashes[spec.key],
                "stage": spec.stage,
            }
            for spec in SOURCE_SPECS
        },
        "generated_csvs": generated,
        "totals": {
            "csv_files_checked": 2,
            "header_fields_checked": total_headers,
            "data_rows_checked": main_checks["rows_checked"]
            + robustness_checks["rows_checked"],
            "data_cells_checked": total_cells,
            "all_values_match_source_summaries": True,
        },
    }


def write_verification_report(path: str | Path, report: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
