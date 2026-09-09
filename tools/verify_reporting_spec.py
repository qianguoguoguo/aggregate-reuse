#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"artifacts"/"table_data"
TABLES=ROOT/"artifacts"/"tables"

STATIC_LABELS={
    "tab:supp-map",
    "tab:supp-notation",
    "tab:supp-rotation-parameters",
    "tab:supp-amazon-splits",
    "tab:supp-shape-transforms",
    "tab:supp-controls",
}
NUMERICAL_LABELS={
    "tab:supp-reference-history",
    "tab:supp-controlled-grid",
    "tab:supp-reuse-results",
    "tab:supp-twin-results",
    "tab:supp-shape-results",
    "tab:supp-strength-results",
    "tab:supp-one-world-ranking",
    "tab:supp-self-influence",
    "tab:supp-complementarity",
    "tab:supp-k6-population",
}
CROSS_CATEGORY_LABELS={"tab:supp-electronics"}


def main():
    out=ROOT/"artifacts"/"verification"
    out.mkdir(parents=True,exist_ok=True)
    checks={}
    differences=[]

    static_path=DATA/"supplement_static_tables.json"
    registry_path=ROOT/"configs"/"reporting_spec.json"
    manifest_path=DATA/"reporting_manifest.json"

    checks["static_table_data_exists"]=static_path.exists()
    checks["reporting_registry_exists"]=registry_path.exists()
    checks["reporting_manifest_exists"]=manifest_path.exists()

    static=json.loads(static_path.read_text()) if static_path.exists() else {}
    registry=json.loads(registry_path.read_text()) if registry_path.exists() else {}
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    checks["exact_static_table_set"]=set(static)==STATIC_LABELS

    entries=registry.get("supplement_tables",[])
    labels=[x["label"] for x in entries]
    checks["all_17_tables_registered"]=(
        len(entries)==17 and len(set(labels))==17
        and set(labels)==STATIC_LABELS|NUMERICAL_LABELS|CROSS_CATEGORY_LABELS
    )

    pre=yaml.safe_load((ROOT/"configs"/"amazon_preprocess.yaml").read_text())
    expected_splits=[
        [role,f"{b[0]}--{b[1]}",str(b[1]-b[0]+1)]
        for role,b in pre["roles"].items()
    ]
    checks["amazon_splits_derived_from_config"]=(
        static.get("tab:supp-amazon-splits")==expected_splits
    )

    shape=yaml.safe_load((ROOT/"configs"/"amazon_shape.yaml").read_text())
    expected_pairs=[]
    for old,new in shape["intervention"]["pair_maps"].items():
        expected_pairs.append([
            f"({old})",f"({new[0]},{new[1]})",
            str(sum(int(x) for x in str(old).split(",")))
        ])
    checks["shape_transforms_derived_from_config"]=(
        static.get("tab:supp-shape-transforms")==expected_pairs
    )

    ctrl=yaml.safe_load((ROOT/"configs"/"controlled.yaml").read_text())
    rot=static.get("tab:supp-rotation-parameters",[])
    rot_map={r[0]:r[1] for r in rot}
    checks["rotation_parameters_derived_from_config"]=(
        rot_map.get("Intervals $T$")==str(ctrl["controlled_model"]["T"])
        and rot_map.get("Normal accounts")==str(ctrl["controlled_model"]["N_normal"])
        and rot_map.get("Coalition accounts")==str(ctrl["controlled_model"]["N_coalition"])
        and rot_map.get("Scheduler")==ctrl["matched_exposure"]["campaign"]["scheduler"]
        and rot_map.get("Active coalition size rule")
            ==ctrl["matched_exposure"]["campaign"]["k_on_rule"]
    )

    expected_tex={
        "supp_map.tex","supp_notation.tex","supp_rotation_parameters.tex",
        "supp_amazon_splits.tex","supp_shape_transforms.tex","supp_controls.tex",
    }
    checks["all_6_static_tex_fragments_exist"]=(
        {p.name for p in TABLES.glob("supp_*.tex")} >= expected_tex
    )

    numerical=DATA/"supplement_numerical_tables.json"
    checks["phase2S_numerical_tables_retained"]=numerical.exists()
    if numerical.exists():
        num=json.loads(numerical.read_text())
        checks["legacy_numerical_table_set_still_10"]=set(num)==NUMERICAL_LABELS
    else:
        checks["legacy_numerical_table_set_still_10"]=False

    cross_entries=[entry for entry in entries if entry.get("label")=="tab:supp-electronics"]
    checks["cross_category_table_registered_as_separate_group"]=(
        len(cross_entries)==1
        and cross_entries[0].get("generator")=="experiments/reporting/generate_cross_category_table.py"
        and cross_entries[0].get("output")=="artifacts/tables/supp_electronics.tex"
    )

    gen=(ROOT/"experiments"/"reporting"/"generate_static_tables.py").read_text()
    checks["static_generator_does_not_read_gold"]=(
        "paper_results/expected" not in gen
        and "supplement_table_gold" not in gen
    )
    checks["manifest_reports_17_tables"]=(
        manifest.get("n_all_supplement_tables")==17
        and manifest.get("n_static_tables")==6
    )

    forbidden=[
        "0.7436000000000001","0.9090666666666667",
        "0.8737004111111111","0.9973519971116672",
        "0.7794222222222221",
    ]
    checks["no_scientific_result_literals_in_static_generator"]=all(
        x not in gen for x in forbidden
    )

    status="PASS" if all(checks.values()) else "FAIL"
    report={
        "phase":"4",
        "status":status,
        "checks":checks,
        "n_static_tables":len(static),
        "n_registered_tables":len(entries),
        "first_differences":differences[:20],
        "verification_note":(
            "The original 10 numerical groups remain frozen-gold verified; "
            "the new cross-category table is source-summary verified by "
            "tools/verify_cross_category_table.py."
        ),
    }
    (out/"phase4_reporting_spec_report.json").write_text(
        json.dumps(report,indent=2,sort_keys=True),encoding="utf-8"
    )

    print("="*72)
    print("PHASE 4 REPORTING SPEC / STATIC-TABLE VERIFICATION")
    print("="*72)
    for k,v in checks.items():
        print(f"{k:52s}: {'PASS' if v else 'FAIL'}")
    print(f"OVERALL                                              : {status}")
    print("="*72)
    return 0 if status=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())
