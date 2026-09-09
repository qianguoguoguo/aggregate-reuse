#!/usr/bin/env python3
"""Generate numerical supplement tables from category-scoped outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.paths import AmazonPathLayout, HOME_AND_KITCHEN
from aggregate_reuse.reporting.supplement_tables import (
    read_csv,
    read_json,
    render_key_value_tabular,
    render_simple_tabular,
    table_complementarity,
    table_controlled_grid,
    table_k6_population,
    table_one_world,
    table_reference_history,
    table_reuse,
    table_self_influence,
    table_shape,
    table_strength,
    table_twins,
)


AMAZON_ARTIFACTS = {
    "reuse": "stage4_reuse_summary",
    "twins": "stage5_twins_summary",
    "shape": "stage6_shape_summary",
    "complementarity": "stage7_complementarity_summary",
    "reference_history": "stage8_reference_history_summary",
    "strength": "stage9_strength_summary",
    "population_audit": "stage9_k6_population_summary",
    "ranking": "stage10_full_background_summary",
    "self_influence": "stage11_self_influence_summary",
}


def paths(
    *,
    category: str = HOME_AND_KITCHEN,
    shared_root: str | Path | None = None,
    include_controlled_grid: bool = True,
) -> dict[str, Path]:
    """Resolve canonical numerical inputs without hard-coding one category."""

    shared = (
        ROOT.parent / "amazon_preprocess"
        if shared_root is None
        else Path(shared_root)
    )
    layout = AmazonPathLayout(
        repository_root=ROOT,
        category=category,
        shared_category_root=shared,
    )
    result = {
        key: layout.artifact(artifact)
        for key, artifact in AMAZON_ARTIFACTS.items()
    }
    if include_controlled_grid:
        result["controlled_grid"] = (
            ROOT / "artifacts" / "figure_data" / "controlled_regime_auc.csv"
        )
    return result


def _logical_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return f"../{resolved.relative_to(ROOT.parent).as_posix()}"
        except ValueError:
            return f"external/{resolved.name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=HOME_AND_KITCHEN)
    parser.add_argument(
        "--shared-root",
        type=Path,
        default=ROOT.parent / "amazon_preprocess",
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
        help="Exclude the category-independent controlled-regime table.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_paths = paths(
        category=args.category,
        shared_root=args.shared_root,
        include_controlled_grid=not args.amazon_only,
    )
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Canonical inputs missing:\n  " + "\n  ".join(missing)
        )

    reuse = read_json(input_paths["reuse"])
    twins = read_json(input_paths["twins"])
    shape = read_json(input_paths["shape"])
    complementarity = read_json(input_paths["complementarity"])
    reference_history = read_json(input_paths["reference_history"])
    strength = read_json(input_paths["strength"])
    population_audit = read_json(input_paths["population_audit"])
    ranking = read_json(input_paths["ranking"])
    self_influence = read_json(input_paths["self_influence"])

    summaries = {
        "reuse": reuse,
        "twins": twins,
        "shape": shape,
        "complementarity": complementarity,
        "reference_history": reference_history,
        "strength": strength,
        "population_audit": population_audit,
        "ranking": ranking,
        "self_influence": self_influence,
    }
    for label, summary in summaries.items():
        actual_category = summary.get("category")
        legacy_home_category = (
            args.category == HOME_AND_KITCHEN and actual_category is None
        )
        if actual_category != args.category and not legacy_home_category:
            raise ValueError(
                f"Category mismatch in {label}: "
                f"{actual_category!r} != {args.category!r}"
            )

    display = {
        "tab:supp-reuse-results": table_reuse(reuse),
        "tab:supp-twin-results": table_twins(twins),
        "tab:supp-shape-results": table_shape(shape),
        "tab:supp-complementarity": table_complementarity(complementarity),
        "tab:supp-reference-history": table_reference_history(reference_history),
        "tab:supp-strength-results": table_strength(strength),
        "tab:supp-k6-population": table_k6_population(
            twins,
            strength,
            population_audit,
        ),
        "tab:supp-one-world-ranking": table_one_world(ranking),
        "tab:supp-self-influence": table_self_influence(self_influence),
    }
    if not args.amazon_only:
        grid = read_csv(input_paths["controlled_grid"])
        display = {
            "tab:supp-controlled-grid": table_controlled_grid(grid),
            **display,
        }

    out = args.out_dir.resolve()
    data_dir = args.data_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    table_data_path = data_dir / "supplement_numerical_tables.json"
    table_data_path.write_text(
        json.dumps(display, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    generated_tex: list[Path] = []

    def write_tex(name: str, content: str) -> None:
        path = out / name
        path.write_text(content, encoding="utf-8")
        generated_tex.append(path)

    write_tex(
        "supp_reuse_results.tex",
        render_simple_tabular(
            ["Reuse $r$", "Evidence AUC", "Frequency AUC"],
            display["tab:supp-reuse-results"],
        ),
    )
    write_tex(
        "supp_twin_results.tex",
        render_key_value_tabular(display["tab:supp-twin-results"]),
    )
    write_tex(
        "supp_shape_results.tex",
        render_simple_tabular(
            ["Evidence channel", "ROC--AUC", "Misordering"],
            display["tab:supp-shape-results"],
        ),
    )
    write_tex(
        "supp_complementarity.tex",
        render_simple_tabular(
            ["Condition", "Aggregate", "Co-activity", "Combined"],
            display["tab:supp-complementarity"],
        ),
    )
    write_tex(
        "supp_reference_history.tex",
        render_simple_tabular(
            ["$n_{\\rm ref}$", "$\\lambda$", "ROC--AUC", "Misordering"],
            display["tab:supp-reference-history"],
        ),
    )
    write_tex(
        "supp_strength_results.tex",
        render_simple_tabular(
            [
                "$k$",
                "Mean $d^{\\rm cf}$",
                "Positive blocks",
                "ROC--AUC",
                "Misordering",
            ],
            display["tab:supp-strength-results"],
        ),
    )
    write_tex(
        "supp_k6_population.tex",
        render_simple_tabular(
            ["Population", "Eligible items", "Mean $d^{\\rm cf}$", "ROC--AUC"],
            display["tab:supp-k6-population"],
        ),
    )
    write_tex(
        "supp_self_influence.tex",
        render_simple_tabular(
            ["Metric", "Original", "Leave-one-out"],
            display["tab:supp-self-influence"],
        ),
    )
    write_tex(
        "supp_one_world_ranking_full.tex",
        render_simple_tabular(
            ["Metric", "Frequency", "Raw W1", "Predictive W1"],
            display["tab:supp-one-world-ranking"]["full"],
        ),
    )
    write_tex(
        "supp_one_world_ranking_freq8.tex",
        render_simple_tabular(
            ["Metric", "Frequency", "Raw W1", "Predictive W1"],
            display["tab:supp-one-world-ranking"]["freq_eq_8"],
        ),
    )

    if not args.amazon_only:
        p_headers = [f"{value / 10:.1f}" for value in range(1, 11)]
        grid_rows = [
            [ratio] + values
            for ratio, values in display["tab:supp-controlled-grid"].items()
        ]
        write_tex(
            "supp_controlled_grid.tex",
            render_simple_tabular(
                ["$R_{\\rm exp}$"] + p_headers,
                grid_rows,
                alignment="l" + "c" * 10,
            ),
        )

    manifest = {
        "phase": "14" if args.category != HOME_AND_KITCHEN else "2S",
        "category": args.category,
        "scope": (
            "amazon_numerical_tables_only"
            if args.amazon_only
            else "numerical_supplement_tables_only"
        ),
        "n_tables": len(display),
        "source_paths": {
            key: _logical_path(path) for key, path in input_paths.items()
        },
        "generated_table_data": _logical_path(table_data_path),
        "generated_tex_files": sorted(path.name for path in generated_tex),
        "static_spec_tables_deferred": True,
        "controlled_grid_included": not args.amazon_only,
    }
    manifest_path = data_dir / "supplement_table_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"Generated {len(display)} numerical table groups for {args.category}."
    )
    print(f"Table data: {table_data_path}")
    print(f"LaTeX fragments: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
