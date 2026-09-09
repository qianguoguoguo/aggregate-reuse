#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "artifacts" / "table_data" / "supplement_numerical_tables.json"
GOLD = ROOT / "paper_results" / "expected" / "supplement_table_gold.json"
TABLE_DIR = ROOT / "artifacts" / "tables"

EXPECTED_TEX = {
    "supp_controlled_grid.tex",
    "supp_reuse_results.tex",
    "supp_twin_results.tex",
    "supp_shape_results.tex",
    "supp_complementarity.tex",
    "supp_reference_history.tex",
    "supp_strength_results.tex",
    "supp_k6_population.tex",
    "supp_one_world_ranking_full.tex",
    "supp_one_world_ranking_freq8.tex",
    "supp_self_influence.tex",
}


def main():
    out = ROOT / "artifacts" / "verification"
    out.mkdir(parents=True, exist_ok=True)

    checks = {}
    differences = []

    if not ACT.exists():
        actual = {}
        checks["table_data_exists"] = False
    else:
        actual = json.loads(ACT.read_text(encoding="utf-8"))
        checks["table_data_exists"] = True

    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    expected = gold["expected_display_strings"]

    checks["exact_table_set"] = set(actual) == set(expected)
    if set(actual) != set(expected):
        differences.append(
            f"table-set actual={sorted(actual)} expected={sorted(expected)}"
        )

    display_ok = actual == expected
    checks["all_display_strings_exact"] = display_ok

    if not display_ok:
        for key in sorted(set(actual) & set(expected)):
            if actual[key] != expected[key]:
                differences.append(key)

    tex_files = {p.name for p in TABLE_DIR.glob("supp_*.tex")}
    checks["all_tex_fragments_exist"] = EXPECTED_TEX.issubset(tex_files)

    # The generator must never import/read the frozen gold package.
    generator = (
        ROOT / "experiments" / "reporting" / "generate_supplement_tables.py"
    ).read_text(encoding="utf-8")
    checks["generator_does_not_read_gold"] = (
        "paper_results/expected" not in generator
        and "supplement_table_gold" not in generator
    )

    # Confirm each table has at least one canonical source declared.
    manifest_path = (
        ROOT / "artifacts" / "table_data" / "supplement_table_manifest.json"
    )
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks["source_manifest_complete"] = (
            manifest["n_tables"] == 10
            and len(manifest["source_paths"]) == 10
            and manifest["static_spec_tables_deferred"] is True
        )
    else:
        checks["source_manifest_complete"] = False

    checks["numerical_tables_only"] = set(actual) == {
        "tab:supp-controlled-grid",
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

    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "phase": "2S",
        "status": status,
        "checks": checks,
        "n_numerical_tables": len(actual),
        "first_differences": differences[:20],
    }
    (out / "phase2S_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("="*72)
    print("PHASE 2S AUTOMATIC SUPPLEMENT-TABLE VERIFICATION")
    print("="*72)
    for k,v in checks.items():
        print(f"{k:38s}: {'PASS' if v else 'FAIL'}")
    print(f"OVERALL                                : {status}")
    if differences:
        print("First differences:")
        for d in differences[:10]:
            print("  " + d)
    print("="*72)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
