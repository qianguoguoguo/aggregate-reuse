#!/usr/bin/env python3
"""Verify Phase 14 Electronics tables and figures from canonical sources."""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPORTING_SCRIPTS = ROOT / "experiments" / "reporting"
for path in (SRC, REPORTING_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aggregate_reuse.amazon.paths import AmazonPathLayout
from aggregate_reuse.amazon.provenance import sha256_file
from aggregate_reuse.reporting.amazon_figure_data import verify_figure_data
from aggregate_reuse.reporting.supplement_tables import (
    read_json,
    render_key_value_tabular,
    render_simple_tabular,
    table_complementarity,
    table_k6_population,
    table_one_world,
    table_reference_history,
    table_reuse,
    table_self_influence,
    table_shape,
    table_strength,
    table_twins,
)
from generate_static_tables import build_static_tables


NUMERICAL_LABELS = {
    "tab:supp-reuse-results",
    "tab:supp-twin-results",
    "tab:supp-shape-results",
    "tab:supp-complementarity",
    "tab:supp-reference-history",
    "tab:supp-strength-results",
    "tab:supp-k6-population",
    "tab:supp-one-world-ranking",
    "tab:supp-self-influence",
}
STATIC_LABELS = {
    "tab:supp-map",
    "tab:supp-notation",
    "tab:supp-amazon-splits",
    "tab:supp-shape-transforms",
    "tab:supp-controls",
}
NUMERICAL_TEX = {
    "supp_reuse_results.tex",
    "supp_twin_results.tex",
    "supp_shape_results.tex",
    "supp_complementarity.tex",
    "supp_reference_history.tex",
    "supp_strength_results.tex",
    "supp_k6_population.tex",
    "supp_self_influence.tex",
    "supp_one_world_ranking_full.tex",
    "supp_one_world_ranking_freq8.tex",
}
STATIC_TEX = {
    "supp_map.tex",
    "supp_notation.tex",
    "supp_amazon_splits.tex",
    "supp_shape_transforms.tex",
    "supp_controls.tex",
}
FIGURE_STEMS = {
    "fig2a_electronics_reuse",
    "fig2b_electronics_matched",
    "fig2c_electronics_shape",
    "fig3a_electronics_complementarity",
    "fig3b_electronics_reference_history",
    "fig3c_electronics_intervention_strength",
}


class ReportingError(RuntimeError):
    pass


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ReportingError(message)


def load_json_object(path: Path) -> dict[str, Any]:
    ensure(path.is_file(), f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    ensure(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def electronics_layout(shared_root: Path) -> AmazonPathLayout:
    return AmazonPathLayout(
        repository_root=ROOT,
        category="electronics",
        shared_category_root=shared_root,
    )


def expected_numerical(layout: AmazonPathLayout) -> dict[str, Any]:
    summaries = {
        "reuse": read_json(layout.artifact("stage4_reuse_summary")),
        "twins": read_json(layout.artifact("stage5_twins_summary")),
        "shape": read_json(layout.artifact("stage6_shape_summary")),
        "complementarity": read_json(
            layout.artifact("stage7_complementarity_summary")
        ),
        "reference_history": read_json(
            layout.artifact("stage8_reference_history_summary")
        ),
        "strength": read_json(layout.artifact("stage9_strength_summary")),
        "population_audit": read_json(
            layout.artifact("stage9_k6_population_summary")
        ),
        "ranking": read_json(
            layout.artifact("stage10_full_background_summary")
        ),
        "self_influence": read_json(
            layout.artifact("stage11_self_influence_summary")
        ),
    }
    ensure(
        all(summary.get("category") == "electronics" for summary in summaries.values()),
        "non-Electronics numerical source summary",
    )
    return {
        "tab:supp-reuse-results": table_reuse(summaries["reuse"]),
        "tab:supp-twin-results": table_twins(summaries["twins"]),
        "tab:supp-shape-results": table_shape(summaries["shape"]),
        "tab:supp-complementarity": table_complementarity(
            summaries["complementarity"]
        ),
        "tab:supp-reference-history": table_reference_history(
            summaries["reference_history"]
        ),
        "tab:supp-strength-results": table_strength(summaries["strength"]),
        "tab:supp-k6-population": table_k6_population(
            summaries["twins"],
            summaries["strength"],
            summaries["population_audit"],
        ),
        "tab:supp-one-world-ranking": table_one_world(summaries["ranking"]),
        "tab:supp-self-influence": table_self_influence(
            summaries["self_influence"]
        ),
    }


def expected_numerical_tex(display: dict[str, Any]) -> dict[str, str]:
    return {
        "supp_reuse_results.tex": render_simple_tabular(
            ["Reuse $r$", "Evidence AUC", "Frequency AUC"],
            display["tab:supp-reuse-results"],
        ),
        "supp_twin_results.tex": render_key_value_tabular(
            display["tab:supp-twin-results"]
        ),
        "supp_shape_results.tex": render_simple_tabular(
            ["Evidence channel", "ROC--AUC", "Misordering"],
            display["tab:supp-shape-results"],
        ),
        "supp_complementarity.tex": render_simple_tabular(
            ["Condition", "Aggregate", "Co-activity", "Combined"],
            display["tab:supp-complementarity"],
        ),
        "supp_reference_history.tex": render_simple_tabular(
            ["$n_{\\rm ref}$", "$\\lambda$", "ROC--AUC", "Misordering"],
            display["tab:supp-reference-history"],
        ),
        "supp_strength_results.tex": render_simple_tabular(
            [
                "$k$",
                "Mean $d^{\\rm cf}$",
                "Positive blocks",
                "ROC--AUC",
                "Misordering",
            ],
            display["tab:supp-strength-results"],
        ),
        "supp_k6_population.tex": render_simple_tabular(
            ["Population", "Eligible items", "Mean $d^{\\rm cf}$", "ROC--AUC"],
            display["tab:supp-k6-population"],
        ),
        "supp_self_influence.tex": render_simple_tabular(
            ["Metric", "Original", "Leave-one-out"],
            display["tab:supp-self-influence"],
        ),
        "supp_one_world_ranking_full.tex": render_simple_tabular(
            ["Metric", "Frequency", "Raw W1", "Predictive W1"],
            display["tab:supp-one-world-ranking"]["full"],
        ),
        "supp_one_world_ranking_freq8.tex": render_simple_tabular(
            ["Metric", "Frequency", "Raw W1", "Predictive W1"],
            display["tab:supp-one-world-ranking"]["freq_eq_8"],
        ),
    }


def expected_static_tex(display: dict[str, Any]) -> dict[str, str]:
    specs = {
        "supp_map.tex": (
            "tab:supp-map",
            ["Experiment", "Canonical config", "Canonical output"],
            "lll",
        ),
        "supp_notation.tex": (
            "tab:supp-notation",
            ["Symbol", "Meaning"],
            "ll",
        ),
        "supp_amazon_splits.tex": (
            "tab:supp-amazon-splits",
            ["Role", "Positions", "Reviews"],
            "lcc",
        ),
        "supp_shape_transforms.tex": (
            "tab:supp-shape-transforms",
            ["Original pair", "Replacement pair", "Pair sum"],
            "ccc",
        ),
        "supp_controls.tex": (
            "tab:supp-controls",
            ["Experiment", "Control/isolation principle"],
            "ll",
        ),
    }
    return {
        filename: render_simple_tabular(headers, display[label], alignment=align)
        for filename, (label, headers, align) in specs.items()
    }


def png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:24]
    ensure(data[:8] == b"\x89PNG\r\n\x1a\n", f"invalid PNG signature: {path}")
    return struct.unpack(">II", data[16:24])


def pdf_dimensions(path: Path) -> tuple[float, float]:
    data = path.read_bytes()
    ensure(data.startswith(b"%PDF-"), f"invalid PDF signature: {path}")
    match = re.search(
        rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]",
        data,
    )
    ensure(match is not None, f"missing PDF MediaBox: {path}")
    return float(match.group(1)), float(match.group(2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shared-root",
        type=Path,
        default=ROOT.parent / "amazon_preprocess" / "electronics",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=ROOT / "configs" / "electronics",
    )
    parser.add_argument(
        "--table-data-dir",
        type=Path,
        default=ROOT / "artifacts" / "electronics" / "table_data",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=ROOT / "artifacts" / "electronics" / "tables",
    )
    parser.add_argument(
        "--figure-data-dir",
        type=Path,
        default=ROOT / "artifacts" / "electronics" / "figure_data",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "artifacts" / "electronics" / "figures",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            ROOT
            / "artifacts"
            / "electronics"
            / "verification"
            / "reporting_report.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    failures: list[dict[str, str]] = []

    def capture(name: str, check: Callable[[], Any]) -> None:
        try:
            detail = check()
        except Exception as exc:
            checks[name] = False
            failures.append(
                {
                    "check": name,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        else:
            checks[name] = True
            if detail is not None:
                details[name] = detail

    layout = electronics_layout(args.shared_root.resolve())

    expected_num: dict[str, Any] = {}
    expected_static: dict[str, Any] = {}

    def numerical_data() -> dict[str, Any]:
        nonlocal expected_num
        expected_num = expected_numerical(layout)
        actual = load_json_object(
            args.table_data_dir / "supplement_numerical_tables.json"
        )
        ensure(set(actual) == NUMERICAL_LABELS, "numerical table set mismatch")
        ensure(actual == expected_num, "numerical tables differ from source summaries")
        return {"table_groups": len(actual)}

    def static_data() -> dict[str, Any]:
        nonlocal expected_static
        expected_static, metadata = build_static_tables(
            config_dir=args.config_dir,
            amazon_only=True,
        )
        ensure(metadata["category"] == "electronics", "static config category mismatch")
        actual = load_json_object(
            args.table_data_dir / "supplement_static_tables.json"
        )
        ensure(set(actual) == STATIC_LABELS, "static table set mismatch")
        ensure(actual == expected_static, "static tables differ from Electronics configs")
        ensure(
            "home_and_kitchen" not in json.dumps(actual, sort_keys=True).lower(),
            "Home-and-Kitchen reference in static table data",
        )
        return {"table_groups": len(actual)}

    def manifests() -> dict[str, Any]:
        numerical = load_json_object(
            args.table_data_dir / "supplement_table_manifest.json"
        )
        static = load_json_object(args.table_data_dir / "reporting_manifest.json")
        ensure(
            numerical.get("category") == "electronics"
            and numerical.get("n_tables") == 9
            and numerical.get("controlled_grid_included") is False,
            "numerical manifest mismatch",
        )
        ensure(
            len(numerical.get("source_paths", {})) == 9
            and all(
                path.startswith("../amazon_preprocess/electronics/")
                for path in numerical["source_paths"].values()
            ),
            "numerical source path mismatch",
        )
        ensure(
            static.get("category") == "electronics"
            and static.get("amazon_only") is True
            and static.get("n_static_tables") == 5
            and static.get("n_applicable_table_groups") == 14,
            "static manifest mismatch",
        )
        combined = json.dumps({"numerical": numerical, "static": static}).lower()
        ensure("home_and_kitchen" not in combined, "Home category leaked into manifests")
        return {"numerical_sources": 9, "applicable_table_groups": 14}

    def tex_fragments() -> dict[str, Any]:
        ensure(expected_num and expected_static, "table-data checks did not run first")
        expected = {
            **expected_numerical_tex(expected_num),
            **expected_static_tex(expected_static),
        }
        actual_files = {path.name for path in args.table_dir.glob("*.tex")}
        ensure(actual_files == NUMERICAL_TEX | STATIC_TEX, "LaTeX file set mismatch")
        for filename, content in expected.items():
            actual = (args.table_dir / filename).read_text(encoding="utf-8")
            ensure(actual == content, f"LaTeX source derivation mismatch: {filename}")
        return {"tex_fragments": len(expected)}

    def figure_data() -> dict[str, Any]:
        result = verify_figure_data(layout, args.figure_data_dir)
        ensure(result["status"] == "PASS", "figure-data verification failed")
        return result["totals"]

    def figures() -> dict[str, Any]:
        pdfs = {path.stem: path for path in args.figure_dir.glob("*.pdf")}
        pngs = {path.stem: path for path in args.figure_dir.glob("*.png")}
        ensure(set(pdfs) == FIGURE_STEMS, "Electronics PDF set mismatch")
        ensure(set(pngs) == FIGURE_STEMS, "Electronics PNG set mismatch")
        for stem in sorted(FIGURE_STEMS):
            ensure(
                pdf_dimensions(pdfs[stem]) == (619.2, 331.2),
                f"unexpected PDF canvas: {pdfs[stem]}",
            )
            ensure(
                png_dimensions(pngs[stem]) == (3440, 1840),
                f"unexpected PNG canvas: {pngs[stem]}",
            )
        return {
            "pdf_files": len(pdfs),
            "png_files": len(pngs),
            "files": {
                path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                for path in sorted((*pdfs.values(), *pngs.values()))
            },
        }

    capture("numerical_tables_match_source_summaries", numerical_data)
    capture("static_tables_match_electronics_configs", static_data)
    capture("manifests_are_category_scoped", manifests)
    capture("latex_fragments_match_table_data", tex_fragments)
    capture("figure_data_matches_source_summaries", figure_data)
    capture("figure_files_and_canvases", figures)

    status = "PASS" if checks and all(checks.values()) else "FAIL"
    report = {
        "schema_version": 1,
        "phase": 14,
        "category": "electronics",
        "status": status,
        "gold_comparison_performed": False,
        "checks": checks,
        "details": details,
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 76)
    print("ELECTRONICS PHASE 14 REPORTING VERIFICATION")
    print("=" * 76)
    for name, passed in checks.items():
        print(f"{name:45s}: {'PASS' if passed else 'FAIL'}")
    print("=" * 76)
    print(f"ELECTRONICS REPORTING STATUS: {status}")
    if failures:
        for failure in failures:
            print(f"{failure['check']}: {failure['message']}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
