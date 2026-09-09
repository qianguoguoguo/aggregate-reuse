#!/usr/bin/env python3
"""Generate static/specification tables from category-aware configurations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.paths import HOME_AND_KITCHEN
from aggregate_reuse.reporting.supplement_tables import render_simple_tabular


AMAZON_CONFIG_NAMES = (
    "amazon_preprocess.yaml",
    "amazon_reference.yaml",
    "amazon_primary.yaml",
    "amazon_twins.yaml",
    "amazon_shape.yaml",
    "amazon_complementarity.yaml",
    "amazon_reference_history.yaml",
    "amazon_strength_fixed.yaml",
    "amazon_k6_population_audit.yaml",
    "amazon_full_background.yaml",
    "amazon_self_influence.yaml",
)

AMAZON_MAP_SPECS = (
    ("Amazon preprocessing", "amazon_preprocess.yaml", "stage2"),
    ("Reference/calibration", "amazon_reference.yaml", "stage3"),
    ("Primary attack", "amazon_primary.yaml", "stage4_attack"),
    ("Fixed-attack reuse", "amazon_primary.yaml", "stage4_reuse"),
    ("Exact matched twins", "amazon_twins.yaml", "stage5_twins"),
    ("Mean-preserving shape", "amazon_shape.yaml", "stage6_shape"),
    (
        "Complementarity",
        "amazon_complementarity.yaml",
        "stage7_complementarity",
    ),
    (
        "Reference history",
        "amazon_reference_history.yaml",
        "stage8_reference_history",
    ),
    (
        "Fixed-identity strength",
        "amazon_strength_fixed.yaml",
        "stage9_strength_fixed",
    ),
    (
        "Same-k=6 population audit",
        "amazon_k6_population_audit.yaml",
        "stage9_k6_population_audit",
    ),
    (
        "Full-background ranking",
        "amazon_full_background.yaml",
        "stage10_full_background",
    ),
    (
        "LOO self-influence",
        "amazon_self_influence.yaml",
        "stage11_self_influence",
    ),
)

CONTROLLED_ONLY_NOTATION = {"$R_{\\rm exp}$", "$p_{\\rm on}$"}


def load_yaml(config_dir: Path, name: str) -> dict[str, Any]:
    value = yaml.safe_load((config_dir / name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Configuration is not an object: {config_dir / name}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON configuration is not an object: {path}")
    return value


def fmt_list(values: list[Any]) -> str:
    return ", ".join(str(value) for value in values)


def _logical_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return f"external/{resolved.name}"


def _shared_root_label(preprocess: dict[str, Any]) -> str:
    shared = preprocess.get("shared_data")
    if isinstance(shared, dict) and isinstance(shared.get("root"), str):
        return shared["root"].rstrip("/")
    return "../amazon_preprocess"


def build_static_tables(
    *,
    config_dir: str | Path,
    amazon_only: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build category-specific static tables without reading result artifacts."""

    category_configs = Path(config_dir).resolve()
    loaded = {
        name: load_yaml(category_configs, name)
        for name in AMAZON_CONFIG_NAMES
    }
    preprocess = loaded["amazon_preprocess.yaml"]
    category = preprocess.get("category")
    if not isinstance(category, str):
        raise ValueError("amazon_preprocess.yaml has no category")
    mismatched = {
        name: value.get("category")
        for name, value in loaded.items()
        if value.get("category") != category
    }
    if mismatched:
        raise ValueError(f"Amazon config category mismatch: {mismatched}")

    spec = load_json(ROOT / "configs" / "reporting_static_spec.json")
    config_prefix = _logical_path(category_configs)
    shared_root = _shared_root_label(preprocess)
    amazon_map = [
        [
            experiment,
            f"{config_prefix}/{config_name}",
            f"{shared_root}/{stage}/",
        ]
        for experiment, config_name, stage in AMAZON_MAP_SPECS
    ]
    map_rows = amazon_map
    if not amazon_only:
        map_rows = [
            [
                "Controlled null",
                "configs/controlled.yaml",
                "artifacts/results/controlled_null_summary.json",
            ],
            [
                "Controlled matched exposure",
                "configs/controlled.yaml",
                "artifacts/results/controlled_matched_summary.json",
            ],
            [
                "Controlled regime sweep",
                "configs/controlled.yaml",
                "artifacts/results/controlled_sweep_summary.json",
            ],
            *amazon_map,
        ]

    notation_rows = spec["notation"]
    if amazon_only:
        notation_rows = [
            row for row in notation_rows if row[0] not in CONTROLLED_ONLY_NOTATION
        ]

    split_rows = [
        [role, f"{bounds[0]}--{bounds[1]}", str(bounds[1] - bounds[0] + 1)]
        for role, bounds in preprocess["roles"].items()
    ]

    pair_rows = []
    for old, new in loaded["amazon_shape.yaml"]["intervention"][
        "pair_maps"
    ].items():
        pair_rows.append(
            [
                f"({old})",
                f"({new[0]},{new[1]})",
                str(sum(int(value) for value in str(old).split(","))),
            ]
        )

    tables: dict[str, Any] = {
        "tab:supp-map": map_rows,
        "tab:supp-notation": notation_rows,
        "tab:supp-amazon-splits": split_rows,
        "tab:supp-shape-transforms": pair_rows,
        "tab:supp-controls": spec["controls"],
    }

    if not amazon_only:
        controlled = load_yaml(ROOT / "configs", "controlled.yaml")
        controlled_model = controlled["controlled_model"]
        matched = controlled["matched_exposure"]
        sweep = controlled["regime_sweep"]
        tables["tab:supp-rotation-parameters"] = [
            ["Intervals $T$", str(controlled_model["T"])],
            ["Normal accounts", str(controlled_model["N_normal"])],
            ["Coalition accounts", str(controlled_model["N_coalition"])],
            ["Normal participation $p$", str(controlled_model["p_normal"])],
            ["Scheduler", str(matched["campaign"]["scheduler"])],
            ["Principal $p_{\\rm on}$", str(matched["campaign"]["principal_p_on"])],
            [
                "Principal $R_{\\rm exp}$",
                str(matched["campaign"]["principal_exposure_ratio"]),
            ],
            [
                "Active coalition size rule",
                str(matched["campaign"]["k_on_rule"]),
            ],
            ["Sweep $p_{\\rm on}$ grid", fmt_list(sweep["p_on_grid"])],
            [
                "Sweep $R_{\\rm exp}$ grid",
                fmt_list(sweep["exposure_ratio_grid"]),
            ],
        ]

    metadata = {
        "category": category,
        "config_dir": config_prefix,
        "shared_root": shared_root,
        "amazon_only": amazon_only,
    }
    return tables, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=ROOT / "configs",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "tables",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "artifacts" / "table_data",
    )
    parser.add_argument(
        "--amazon-only",
        action="store_true",
        help="Generate only static tables applicable to Amazon experiments.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tables, metadata = build_static_tables(
        config_dir=args.config_dir,
        amazon_only=args.amazon_only,
    )
    out = args.out_dir.resolve()
    data_dir = args.data_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    static_data_path = data_dir / "supplement_static_tables.json"
    static_data_path.write_text(
        json.dumps(tables, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tex_specs = {
        "tab:supp-map": (
            "supp_map.tex",
            ["Experiment", "Canonical config", "Canonical output"],
            "lll",
        ),
        "tab:supp-notation": (
            "supp_notation.tex",
            ["Symbol", "Meaning"],
            "ll",
        ),
        "tab:supp-amazon-splits": (
            "supp_amazon_splits.tex",
            ["Role", "Positions", "Reviews"],
            "lcc",
        ),
        "tab:supp-shape-transforms": (
            "supp_shape_transforms.tex",
            ["Original pair", "Replacement pair", "Pair sum"],
            "ccc",
        ),
        "tab:supp-controls": (
            "supp_controls.tex",
            ["Experiment", "Control/isolation principle"],
            "ll",
        ),
        "tab:supp-rotation-parameters": (
            "supp_rotation_parameters.tex",
            ["Parameter", "Value"],
            "ll",
        ),
    }
    generated_tex = []
    for label, rows in tables.items():
        filename, header, alignment = tex_specs[label]
        path = out / filename
        path.write_text(
            render_simple_tabular(header, rows, alignment=alignment),
            encoding="utf-8",
        )
        generated_tex.append(path.name)

    registry = load_json(ROOT / "configs" / "reporting_spec.json")
    sources = {
        "map": f"{metadata['config_dir']}/amazon_*.yaml",
        "notation": "configs/reporting_static_spec.json",
        "amazon_splits": f"{metadata['config_dir']}/amazon_preprocess.yaml",
        "shape_transforms": f"{metadata['config_dir']}/amazon_shape.yaml",
        "controls": "configs/reporting_static_spec.json + experiment configs",
    }
    if not args.amazon_only:
        sources["rotation_parameters"] = "configs/controlled.yaml"
    manifest = {
        "phase": (
            "14" if metadata["category"] != HOME_AND_KITCHEN else "2T"
        ),
        **metadata,
        "static_tables": sorted(tables),
        "n_static_tables": len(tables),
        "n_applicable_table_groups": len(tables) + (9 if args.amazon_only else 11),
        "all_supplement_tables": registry["supplement_tables"],
        "n_all_supplement_tables": len(registry["supplement_tables"]),
        "generated_tex_files": sorted(generated_tex),
        "sources": sources,
        "generator_reads_numerical_gold": False,
    }
    manifest_path = data_dir / "reporting_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"Generated {len(tables)} static/specification table groups for "
        f"{metadata['category']}."
    )
    print(f"Table data: {static_data_path}")
    print(f"LaTeX fragments: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
