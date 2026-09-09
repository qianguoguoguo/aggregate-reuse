#!/usr/bin/env python3
"""Generate the Home-vs-Electronics cross-category supplement table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.reporting.cross_category_table import (
    ELECTRONICS,
    HOME,
    build_cross_category_table,
    load_category_sources,
    source_manifest,
)
from aggregate_reuse.reporting.supplement_tables import render_simple_tabular


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--home-shared-root",
        type=Path,
        default=ROOT.parent / "amazon_preprocess",
    )
    parser.add_argument(
        "--electronics-shared-root",
        type=Path,
        default=ROOT.parent / "amazon_preprocess" / "electronics",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    home = load_category_sources(
        repository_root=ROOT,
        category=HOME,
        shared_root=args.home_shared_root,
    )
    electronics = load_category_sources(
        repository_root=ROOT,
        category=ELECTRONICS,
        shared_root=args.electronics_shared_root,
    )
    rows = build_cross_category_table(home, electronics)

    out = args.out_dir.resolve()
    data = args.data_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    json_path = data / "cross_category_table.json"
    json_path.write_text(
        json.dumps({"tab:supp-electronics": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tex_path = out / "supp_electronics.tex"
    tex_path.write_text(
        render_simple_tabular(
            ["Quantity", "Home \\& Kitchen", "Electronics"],
            rows,
            alignment="lcc",
        ),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "table_label": "tab:supp-electronics",
        "table_kind": "numerical_cross_category",
        "generator_reads_result_gold": False,
        "home": source_manifest(home, ROOT),
        "electronics": source_manifest(electronics, ROOT),
        "generated_data": str(json_path.relative_to(ROOT)).replace("\\", "/"),
        "generated_tex": str(tex_path.relative_to(ROOT)).replace("\\", "/"),
        "n_rows": len(rows),
    }
    manifest_path = data / "cross_category_table_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Generated: {json_path}")
    print(f"Generated: {tex_path}")
    print(f"Manifest:  {manifest_path}")
    print("CROSS-CATEGORY TABLE GENERATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
