#!/usr/bin/env python3
"""Windows-friendly Phase-4 orchestration without requiring make.

Run from the repository root through RUN_PHASE4_WINDOWS.cmd, or directly:
    python tools/run_phase4_windows.py <target>

The runner has category-specific resume points.  In particular, every
``resume-after-electronics-*`` target stays entirely inside the Electronics
pipeline and never re-runs Home-and-Kitchen experiments.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ECFG = ROOT / "configs" / "electronics"
ESHARED = ROOT.parent / "amazon_preprocess" / "electronics"
HSHARED = ROOT.parent / "amazon_preprocess"
PROGRESS_HINT = ROOT / "artifacts" / "verification" / "NEXT_WINDOWS_RESUME.txt"


def run(label: str, args: list[str]) -> None:
    print("=" * 78, flush=True)
    print(label, flush=True)
    print(" ".join(str(x) for x in args), flush=True)
    print("=" * 78, flush=True)
    proc = subprocess.run(args, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"FAILED [{label}] with exit code {proc.returncode}")


def write_resume_hint(target: str) -> None:
    PROGRESS_HINT.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_HINT.write_text(
        "Last successfully verified Phase-4 checkpoint.\n"
        f"If a later step fails, resume with:\n"
        f"RUN_PHASE4_WINDOWS.cmd {target}\n",
        encoding="utf-8",
    )
    print(f"Checkpoint: {target}", flush=True)


def check_environment() -> None:
    run("check frozen environment", [PY, "tools/check_environment.py"])


def controlled() -> None:
    run("controlled null calibration", [PY, "experiments/controlled/run_null_calibration.py", "--clean"])
    run("verify controlled null", [PY, "tools/verify_controlled_null.py"])
    run("controlled matched exposure", [PY, "experiments/controlled/run_matched_exposure.py", "--clean"])
    run("verify controlled matched", [PY, "tools/verify_controlled_matched.py"])
    run("controlled regime sweep", [PY, "experiments/controlled/run_regime_sweep.py", "--clean"])
    run("verify controlled sweep", [PY, "tools/verify_controlled_sweep.py"])


def verify_raw() -> None:
    run("verify raw Amazon inputs", [PY, "tools/verify_raw_amazon_inputs.py"])


def home_preprocess() -> None:
    run("Home preprocessing", [PY, "experiments/amazon/preprocess.py", "--clean"])
    run("verify Home preprocessing", [PY, "tools/verify_amazon_preprocess.py", "--actual-root", "artifacts/amazon_preprocess"])
    run("publish verified Home preprocessing", [PY, "tools/publish_home_preprocess.py", "--clean"])
    write_resume_hint("resume-after-home-preprocess")


def home_downstream_after_reference_history() -> None:
    cmds = [
        ("Home strength", [PY, "experiments/amazon/run_strength_fixed.py", "--all-seeds", "--overwrite"]),
        ("verify Home strength", [PY, "tools/verify_strength_fixed.py"]),
        ("Home k6 population audit", [PY, "experiments/amazon/run_k6_population_audit.py"]),
        ("verify Home k6 population audit", [PY, "tools/verify_k6_population_audit.py"]),
        ("Home full-background ranking", [PY, "experiments/amazon/run_full_background_ranking.py", "--all-seeds", "--overwrite"]),
        ("verify Home full-background ranking", [PY, "tools/verify_full_background_ranking.py"]),
        ("Home self-influence", [PY, "experiments/amazon/run_self_influence.py", "--all-seeds", "--overwrite"]),
        ("verify Home self-influence", [PY, "tools/verify_self_influence.py"]),
    ]
    for label, cmd in cmds:
        run(label, cmd)


def home_downstream_after_shape() -> None:
    cmds = [
        ("Home complementarity", [PY, "experiments/amazon/run_complementarity.py", "--all-seeds", "--overwrite"]),
        ("verify Home complementarity", [PY, "tools/verify_complementarity.py"]),
        ("Home reference history", [PY, "experiments/amazon/run_reference_history.py"]),
        ("verify Home reference history", [PY, "tools/verify_reference_history.py"]),
    ]
    for label, cmd in cmds:
        run(label, cmd)
    home_downstream_after_reference_history()


def home_downstream_after_reuse() -> None:
    cmds = [
        ("Home matched twins", [PY, "experiments/amazon/run_matched_twins.py", "--all-seeds", "--overwrite"]),
        ("verify Home matched twins", [PY, "tools/verify_matched_twins.py"]),
        ("Home shape", [PY, "experiments/amazon/run_shape.py", "--all-seeds", "--overwrite"]),
        ("verify Home shape", [PY, "tools/verify_amazon_shape.py"]),
    ]
    for label, cmd in cmds:
        run(label, cmd)
    home_downstream_after_shape()


def home_downstream_after_reference() -> None:
    cmds = [
        ("Home primary attack", [PY, "experiments/amazon/run_primary_attack.py", "--all-seeds", "--overwrite"]),
        ("verify Home primary attack", [PY, "tools/verify_primary_attack.py"]),
        ("Home primary reuse", [PY, "experiments/amazon/run_primary_reuse.py", "--all-seeds", "--overwrite"]),
        ("verify Home primary reuse", [PY, "tools/verify_primary_reuse.py"]),
    ]
    for label, cmd in cmds:
        run(label, cmd)
    home_downstream_after_reuse()


def home_downstream() -> None:
    run("Home reference calibration", [PY, "experiments/amazon/calibrate_reference.py", "--overwrite"])
    run("verify Home reference", [PY, "tools/verify_amazon_reference.py"])
    write_resume_hint("resume-after-home-reference")
    home_downstream_after_reference()


def electronics_verify(stage: str) -> None:
    run(
        f"verify Electronics {stage}",
        [PY, "tools/verify_electronics.py", "--stage", stage],
    )


def electronics_preprocess() -> None:
    run(
        "Electronics preprocessing",
        [PY, "experiments/amazon/preprocess.py", "--config", str(ECFG / "amazon_preprocess.yaml"), "--clean"],
    )
    electronics_verify("inputs")
    write_resume_hint("resume-after-electronics-preprocess")


# stage_name, label, script, config, extra_args, resume_target
ELECTRONICS_PIPELINE = [
    ("reference", "Electronics reference", "calibrate_reference.py", "amazon_reference.yaml", ["--overwrite"], "resume-after-electronics-reference"),
    ("primary_attack", "Electronics primary attack", "run_primary_attack.py", "amazon_primary.yaml", ["--all-seeds", "--overwrite"], "resume-after-electronics-primary-attack"),
    ("primary_reuse", "Electronics primary reuse", "run_primary_reuse.py", "amazon_primary.yaml", ["--all-seeds", "--overwrite"], "resume-after-electronics-primary-reuse"),
    ("matched_twins", "Electronics matched twins", "run_matched_twins.py", "amazon_twins.yaml", ["--all-seeds", "--overwrite"], "resume-after-electronics-matched-twins"),
    ("shape", "Electronics shape", "run_shape.py", "amazon_shape.yaml", ["--all-seeds", "--overwrite"], "resume-after-electronics-shape"),
    ("complementarity", "Electronics complementarity", "run_complementarity.py", "amazon_complementarity.yaml", ["--all-seeds", "--overwrite"], "resume-after-electronics-complementarity"),
    ("reference_history", "Electronics reference history", "run_reference_history.py", "amazon_reference_history.yaml", [], "resume-after-electronics-reference-history"),
    ("strength_fixed", "Electronics strength", "run_strength_fixed.py", "amazon_strength_fixed.yaml", ["--all-seeds", "--overwrite"], "resume-after-electronics-strength"),
    ("k6_population_audit", "Electronics k6 population audit", "run_k6_population_audit.py", "amazon_k6_population_audit.yaml", [], "resume-after-electronics-k6-audit"),
    ("full_background", "Electronics full-background ranking", "run_full_background_ranking.py", "amazon_full_background.yaml", ["--all-seeds", "--overwrite"], "resume-after-electronics-full-background"),
    ("self_influence", "Electronics self-influence", "run_self_influence.py", "amazon_self_influence.yaml", ["--all-seeds", "--overwrite"], "resume-after-electronics-self-influence"),
]


def electronics_stage_index(stage: str) -> int:
    for i, row in enumerate(ELECTRONICS_PIPELINE):
        if row[0] == stage:
            return i
    raise ValueError(f"unknown Electronics stage: {stage}")


def verify_electronics_prefix(last_stage: str | None) -> None:
    """Verify completed Electronics work only; never invoke Home verifiers."""
    electronics_verify("inputs")
    if last_stage is None:
        return
    stop = electronics_stage_index(last_stage)
    for stage, *_ in ELECTRONICS_PIPELINE[: stop + 1]:
        electronics_verify(stage)


def run_electronics_from(start_index: int = 0) -> None:
    """Run Electronics stages from start_index, verifying each stage immediately."""
    for stage, label, script, config, extra, resume_target in ELECTRONICS_PIPELINE[start_index:]:
        run(
            label,
            [PY, f"experiments/amazon/{script}", "--config", str(ECFG / config), *extra],
        )
        electronics_verify(stage)
        write_resume_hint(resume_target)
    run("Electronics verifier self-test", [PY, "tools/verify_electronics.py", "--selftest"])
    run("Electronics consolidated verification", [PY, "tools/verify_electronics.py"])


def electronics_downstream() -> None:
    # A normal Electronics run assumes preprocessing has already completed.
    electronics_verify("inputs")
    run_electronics_from(0)


def amazon_both_from_raw() -> None:
    verify_raw()
    home_preprocess()
    home_downstream()
    electronics_preprocess()
    electronics_downstream()


def home_figure_data() -> None:
    run(
        "Home Amazon figure data",
        [
            PY,
            "experiments/reporting/generate_amazon_figure_data.py",
            "--config",
            "configs/amazon_preprocess.yaml",
            "--shared-root",
            str(HSHARED),
            "--out-dir",
            "artifacts/figure_data",
            "--report",
            "artifacts/verification/home_amazon_figure_data_source_report.json",
        ],
    )


def electronics_figures() -> None:
    run("Electronics figure data", [PY, "experiments/reporting/generate_amazon_figure_data.py", "--config", str(ECFG / "amazon_preprocess.yaml")])
    run("verify Electronics figure data", [PY, "tools/verify_electronics_figure_data.py", "--config", str(ECFG / "amazon_preprocess.yaml")])
    run(
        "render Electronics Figure 2 panels",
        [PY, "figures/make_fig2_amazon_main.py", "--data", "artifacts/electronics/figure_data/amazon_main_figure_data.csv", "--out-dir", "artifacts/electronics/figures", "--stem-prefix", "electronics"],
    )
    run(
        "render Electronics Figure 3 panels",
        [PY, "figures/make_fig3_amazon_robustness.py", "--data", "artifacts/electronics/figure_data/amazon_robustness_figure_data.csv", "--out-dir", "artifacts/electronics/figures", "--stem-prefix", "electronics"],
    )


def electronics_reporting() -> None:
    electronics_figures()
    run(
        "Electronics numerical supplement tables",
        [PY, "experiments/reporting/generate_supplement_tables.py", "--category", "electronics", "--shared-root", str(ESHARED), "--amazon-only", "--out-dir", "artifacts/electronics/tables", "--data-dir", "artifacts/electronics/table_data"],
    )
    run(
        "Electronics static supplement tables",
        [PY, "experiments/reporting/generate_static_tables.py", "--config-dir", str(ECFG), "--amazon-only", "--out-dir", "artifacts/electronics/tables", "--data-dir", "artifacts/electronics/table_data"],
    )
    run("verify Electronics reporting", [PY, "tools/verify_electronics_reporting.py"])


def cross_category_table() -> None:
    run("generate cross-category table", [PY, "experiments/reporting/generate_cross_category_table.py"])
    run("verify cross-category table", [PY, "tools/verify_cross_category_table.py"])


def reporting() -> None:
    home_figure_data()
    electronics_reporting()
    run("generate Home numerical supplement tables", [PY, "experiments/reporting/generate_supplement_tables.py"])
    run("verify Home numerical supplement tables", [PY, "tools/verify_supplement_tables.py"])
    run("generate static/specification tables", [PY, "experiments/reporting/generate_static_tables.py"])
    run("verify reporting registry", [PY, "tools/verify_reporting_spec.py"])
    cross_category_table()
    run("verify main-paper figure data", [PY, "tools/verify_figure_data.py"])
    run("render Figure 1", [PY, "figures/make_fig1_controlled.py"])
    run("render Figure 2", [PY, "figures/make_fig2_amazon_main.py"])
    run("render Figure 3", [PY, "figures/make_fig3_amazon_robustness.py"])


def clean_caches() -> None:
    run("clean release caches", [PY, "tools/clean_release_caches.py"])


def final_check() -> None:
    run("final release checker", [PY, "tools/final_release_check.py"])


def finalization_after_experiments() -> None:
    reporting()
    clean_caches()
    final_check()
    clean_caches()
    run(
        "final frozen-gold integrity",
        [PY, "tools/verify_results.py", "integrity", "--expected-dir", "paper_results/expected", "--report", "artifacts/verification/gold_integrity_final.json"],
    )
    clean_caches()
    run(
        "final anonymity scan",
        [PY, "tools/anonymity_scan.py", "--root", ".", "--report", "artifacts/verification/final_anonymity_after_reproduction.json"],
    )
    write_resume_hint("final-check")


def reproduce_final_from_raw() -> None:
    check_environment()
    controlled()
    amazon_both_from_raw()
    finalization_after_experiments()


def resume_after_home_preprocess() -> None:
    check_environment()
    verify_raw()
    run("verify staged Home preprocessing", [PY, "tools/verify_amazon_preprocess.py", "--actual-root", "artifacts/amazon_preprocess"])
    run("publish verified Home preprocessing", [PY, "tools/publish_home_preprocess.py", "--clean"])
    home_downstream()
    electronics_preprocess()
    electronics_downstream()
    finalization_after_experiments()


def resume_after_home_reference() -> None:
    check_environment()
    verify_raw()
    run("verify existing Home reference", [PY, "tools/verify_amazon_reference.py"])
    home_downstream_after_reference()
    electronics_preprocess()
    electronics_downstream()
    finalization_after_experiments()


def resume_after_home_reuse() -> None:
    check_environment()
    verify_raw()
    run("verify existing Home reference", [PY, "tools/verify_amazon_reference.py"])
    run("verify existing Home primary attack", [PY, "tools/verify_primary_attack.py"])
    run("verify existing Home primary reuse", [PY, "tools/verify_primary_reuse.py"])
    home_downstream_after_reuse()
    electronics_preprocess()
    electronics_downstream()
    finalization_after_experiments()


def resume_after_home_shape() -> None:
    check_environment()
    verify_raw()
    run("verify existing Home reference", [PY, "tools/verify_amazon_reference.py"])
    run("verify existing Home primary attack", [PY, "tools/verify_primary_attack.py"])
    run("verify existing Home primary reuse", [PY, "tools/verify_primary_reuse.py"])
    run("verify existing Home matched twins", [PY, "tools/verify_matched_twins.py"])
    run("verify existing Home shape", [PY, "tools/verify_amazon_shape.py"])
    home_downstream_after_shape()
    electronics_preprocess()
    electronics_downstream()
    finalization_after_experiments()


def resume_after_home_reference_history() -> None:
    check_environment()
    verify_raw()
    run("verify existing Home reference", [PY, "tools/verify_amazon_reference.py"])
    run("verify existing Home primary attack", [PY, "tools/verify_primary_attack.py"])
    run("verify existing Home primary reuse", [PY, "tools/verify_primary_reuse.py"])
    run("verify existing Home matched twins", [PY, "tools/verify_matched_twins.py"])
    run("verify existing Home shape", [PY, "tools/verify_amazon_shape.py"])
    run("verify existing Home complementarity", [PY, "tools/verify_complementarity.py"])
    run("verify existing Home reference history", [PY, "tools/verify_reference_history.py"])
    home_downstream_after_reference_history()
    electronics_preprocess()
    electronics_downstream()
    finalization_after_experiments()


def resume_after_electronics(last_stage: str | None) -> None:
    """Resume inside Electronics only; Home work is never re-run or re-verified."""
    check_environment()
    verify_raw()
    verify_electronics_prefix(last_stage)
    start = 0 if last_stage is None else electronics_stage_index(last_stage) + 1
    run_electronics_from(start)
    finalization_after_experiments()


def resume_after_electronics_preprocess() -> None:
    resume_after_electronics(None)


def resume_after_electronics_reference() -> None:
    resume_after_electronics("reference")


def resume_after_electronics_primary_attack() -> None:
    resume_after_electronics("primary_attack")


def resume_after_electronics_primary_reuse() -> None:
    resume_after_electronics("primary_reuse")


def resume_after_electronics_matched_twins() -> None:
    resume_after_electronics("matched_twins")


def resume_after_electronics_shape() -> None:
    resume_after_electronics("shape")


def resume_after_electronics_complementarity() -> None:
    resume_after_electronics("complementarity")


def resume_after_electronics_reference_history() -> None:
    resume_after_electronics("reference_history")


def resume_after_electronics_strength() -> None:
    resume_after_electronics("strength_fixed")


def resume_after_electronics_k6_audit() -> None:
    resume_after_electronics("k6_population_audit")


def resume_after_electronics_full_background() -> None:
    resume_after_electronics("full_background")


def resume_after_electronics_self_influence() -> None:
    resume_after_electronics("self_influence")


TARGETS = {
    "verify-raw": verify_raw,
    "controlled": controlled,
    "home-preprocess": home_preprocess,
    "home-downstream": home_downstream,
    "electronics-preprocess": electronics_preprocess,
    "electronics-downstream": electronics_downstream,
    "amazon-both-from-raw": amazon_both_from_raw,
    "home-figure-data": home_figure_data,
    "electronics-figures": electronics_figures,
    "electronics-reporting": electronics_reporting,
    "cross-category-table": cross_category_table,
    "reporting": reporting,
    "final-check": final_check,
    "reproduce-final-from-raw": reproduce_final_from_raw,
    "resume-after-home-preprocess": resume_after_home_preprocess,
    "resume-after-home-reference": resume_after_home_reference,
    "resume-after-home-reuse": resume_after_home_reuse,
    "resume-after-home-shape": resume_after_home_shape,
    "resume-after-home-reference-history": resume_after_home_reference_history,
    "resume-after-electronics-preprocess": resume_after_electronics_preprocess,
    "resume-after-electronics-reference": resume_after_electronics_reference,
    "resume-after-electronics-primary-attack": resume_after_electronics_primary_attack,
    "resume-after-electronics-primary-reuse": resume_after_electronics_primary_reuse,
    "resume-after-electronics-matched-twins": resume_after_electronics_matched_twins,
    "resume-after-electronics-shape": resume_after_electronics_shape,
    "resume-after-electronics-complementarity": resume_after_electronics_complementarity,
    "resume-after-electronics-reference-history": resume_after_electronics_reference_history,
    "resume-after-electronics-strength": resume_after_electronics_strength,
    "resume-after-electronics-k6-audit": resume_after_electronics_k6_audit,
    "resume-after-electronics-full-background": resume_after_electronics_full_background,
    "resume-after-electronics-self-influence": resume_after_electronics_self_influence,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", choices=sorted(TARGETS))
    args = ap.parse_args()
    TARGETS[args.target]()
    print("=" * 78)
    print(f"PHASE 4 WINDOWS TARGET COMPLETE: {args.target}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
