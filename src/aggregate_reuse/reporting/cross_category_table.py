"""Build the Home-and-Kitchen vs Electronics cross-category supplement table.

This reporting module reads canonical category-scoped experiment summaries only.
It does not read frozen result gold and it does not perform scientific
computation.  Its purpose is to expose a compact, auditable comparison of the
primary Home-and-Kitchen evaluation and the Electronics replication.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from aggregate_reuse.amazon.paths import AmazonPathLayout
from aggregate_reuse.amazon.provenance import sha256_file


HOME = "home_and_kitchen"
ELECTRONICS = "electronics"
EXPECTED_SEEDS = list(range(30))


class CrossCategoryTableError(RuntimeError):
    """Raised when cross-category source artifacts are incomplete or invalid."""


@dataclass(frozen=True)
class CategorySources:
    category: str
    layout: AmazonPathLayout
    stage2: Mapping[str, Any]
    stage3: Mapping[str, Any]
    stage5: Mapping[str, Any]
    stage6: Mapping[str, Any]
    stage7: Mapping[str, Any]
    stage8: Mapping[str, Any]
    stage9: Mapping[str, Any]
    population_audit: Mapping[str, Any]
    paths: Mapping[str, Path]


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise CrossCategoryTableError(message)


def _load_json(path: Path) -> Mapping[str, Any]:
    _ensure(path.is_file(), f"missing cross-category source: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _ensure(isinstance(value, Mapping), f"source root is not an object: {path}")
    return value


def _number(value: Any, label: str) -> float:
    _ensure(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    _ensure(math.isfinite(result), f"{label} is not finite")
    return result


def _mean(summary: Mapping[str, Any], *keys: str) -> float:
    value: Any = summary
    label_parts: list[str] = []
    for key in keys:
        label_parts.append(key)
        _ensure(isinstance(value, Mapping) and key in value, f"missing {'.'.join(label_parts)}")
        value = value[key]
    if isinstance(value, Mapping) and "mean" in value:
        value = value["mean"]
        label_parts.append("mean")
    return _number(value, ".".join(label_parts))


def _validate_category(summary: Mapping[str, Any], category: str, label: str) -> None:
    _ensure(summary.get("category") == category, f"category mismatch in {label}")


def _validate_seeded(summary: Mapping[str, Any], category: str, label: str) -> None:
    _validate_category(summary, category, label)
    _ensure(summary.get("n_seeds") == 30, f"seed count mismatch in {label}")
    _ensure(summary.get("seed_ids") == EXPECTED_SEEDS, f"seed set mismatch in {label}")


def load_category_sources(
    *,
    repository_root: str | Path,
    category: str,
    shared_root: str | Path,
) -> CategorySources:
    root = Path(repository_root).resolve()
    layout = AmazonPathLayout(
        repository_root=root,
        category=category,
        shared_category_root=Path(shared_root),
    )
    paths = {
        "stage2": layout.artifact("stage2_summary"),
        "stage3": layout.artifact("stage3_summary"),
        "stage5": layout.artifact("stage5_twins_summary"),
        "stage6": layout.artifact("stage6_shape_summary"),
        "stage7": layout.artifact("stage7_complementarity_summary"),
        "stage8": layout.artifact("stage8_reference_history_summary"),
        "stage9": layout.artifact("stage9_strength_summary"),
        "population_audit": layout.artifact("stage9_k6_population_summary"),
    }
    loaded = {name: _load_json(path) for name, path in paths.items()}

    _validate_category(loaded["stage2"], category, "stage2")
    _validate_category(loaded["stage3"], category, "stage3")
    for label in ("stage5", "stage6", "stage7", "stage8", "stage9"):
        _validate_seeded(loaded[label], category, label)
    _validate_category(loaded["population_audit"], category, "population_audit")

    _ensure(
        loaded["stage2"].get("stage") == "amazon_stage2_frozen_roles_and_references",
        f"unexpected Stage-2 summary for {category}",
    )
    _ensure(
        loaded["stage3"].get("stage") == "amazon_stage3_reference_calibration_and_drift_diagnostic",
        f"unexpected Stage-3 summary for {category}",
    )
    _ensure(
        loaded["stage5"].get("stage") == "amazon_stage5_exact_matched_twins_r8",
        f"unexpected Stage-5 summary for {category}",
    )
    _ensure(
        loaded["stage6"].get("stage") == "amazon_stage6_mean_preserving_shape_r8",
        f"unexpected Stage-6 summary for {category}",
    )
    _ensure(
        loaded["stage7"].get("stage") == "amazon_stage7_complementarity",
        f"unexpected Stage-7 summary for {category}",
    )
    _ensure(
        loaded["stage8"].get("stage") == "amazon_stage8_reference_history_robustness",
        f"unexpected Stage-8 summary for {category}",
    )
    _ensure(
        loaded["stage9"].get("stage") == "amazon_stage9_intervention_strength_fixed_identity",
        f"unexpected Stage-9 summary for {category}",
    )

    selected_lambda = _number(loaded["stage3"].get("selected_lambda"), f"{category}.selected_lambda")
    _ensure(selected_lambda == 20.0, f"unexpected selected lambda for {category}")

    return CategorySources(
        category=category,
        layout=layout,
        stage2=loaded["stage2"],
        stage3=loaded["stage3"],
        stage5=loaded["stage5"],
        stage6=loaded["stage6"],
        stage7=loaded["stage7"],
        stage8=loaded["stage8"],
        stage9=loaded["stage9"],
        population_audit=loaded["population_audit"],
        paths=paths,
    )


def _f3(value: float) -> str:
    return f"{value:.3f}"


def _fint(value: Any) -> str:
    return f"{int(round(_number(value, 'integer value'))):,}"


def _compact3(value: float) -> str:
    text = _f3(value)
    if text.startswith("-0."):
        return "-." + text[3:]
    if text.startswith("0."):
        return "." + text[2:]
    return text


def _joined(values: list[float]) -> str:
    return "/".join(_compact3(value) for value in values)


def category_values(source: CategorySources) -> dict[str, str]:
    p = source.population_audit["population_results"]
    return {
        "eligible_items": _fint(source.stage2["n_items"]),
        "k6_feasible": _fint(p["k6_feasible_population"]["population_size"]),
        "k9_feasible": _fint(p["k9_feasible_population"]["population_size"]),
        "selected_lambda": f"{_number(source.stage3['selected_lambda'], 'selected_lambda'):.0f}",
        "matched_twin_auc": _f3(_mean(source.stage5, "metrics", "counterfactual_score_auc")),
        "shape_w1_auc": _f3(_mean(source.stage6, "metrics", "w1_counterfactual_score_auc")),
        "shape_js_auc": _f3(_mean(source.stage6, "metrics", "js_counterfactual_score_auc")),
        "combined_auc": _f3(_mean(source.stage7, "metrics", "combined_auc")),
        "reference_auc": _joined([
            _mean(source.stage8, "metrics_by_reference_length", str(n), "counterfactual_score_auc")
            for n in (60, 90, 120)
        ]),
        "strength_dcf": _joined([
            _mean(source.stage9, "metrics_by_k", str(k), "mean_d_cf")
            for k in (3, 6, 9)
        ]),
    }


def build_cross_category_table(
    home: CategorySources,
    electronics: CategorySources,
) -> list[list[str]]:
    _ensure(home.category == HOME, "Home source category mismatch")
    _ensure(electronics.category == ELECTRONICS, "Electronics source category mismatch")
    h = category_values(home)
    e = category_values(electronics)
    return [
        ["Eligible items ($\\ge 300$ reviews)", h["eligible_items"], e["eligible_items"]],
        ["Primary $k=6$-feasible items", h["k6_feasible"], e["k6_feasible"]],
        ["Common $k=9$-feasible items", h["k9_feasible"], e["k9_feasible"]],
        ["Selected $\\lambda$", h["selected_lambda"], e["selected_lambda"]],
        ["Matched-twin ROC--AUC", h["matched_twin_auc"], e["matched_twin_auc"]],
        ["Shape $W_1$ ROC--AUC", h["shape_w1_auc"], e["shape_w1_auc"]],
        ["Shape JS ROC--AUC", h["shape_js_auc"], e["shape_js_auc"]],
        ["Combined complementarity ROC--AUC", h["combined_auc"], e["combined_auc"]],
        ["Reference ROC--AUC ($60/90/120$)", h["reference_auc"], e["reference_auc"]],
        ["Strength $\\bar d_{\\rm cf}$ ($k=3/6/9$)", h["strength_dcf"], e["strength_dcf"]],
    ]


def source_manifest(source: CategorySources, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    result: dict[str, Any] = {}
    for label, path in source.paths.items():
        resolved = path.resolve()
        try:
            logical = resolved.relative_to(root).as_posix()
        except ValueError:
            try:
                logical = f"../{resolved.relative_to(root.parent).as_posix()}"
            except ValueError:
                logical = f"external/{resolved.name}"
        result[label] = {"path": logical, "sha256": sha256_file(resolved)}
    return result
