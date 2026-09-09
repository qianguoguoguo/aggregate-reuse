#!/usr/bin/env python3
"""Run an isolated one-seed Amazon smoke test using frozen Stage-2/3 inputs.

This deliberately reuses an already-prepared Stage-2/3 category root.  It is
intended to validate category routing and downstream Windows/Linux agreement
before the expensive full preprocessing + 30-seed reproduction matrix.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.paths import AmazonPathLayout
from aggregate_reuse.amazon.provenance import require_artifact_category
from tools._phase3_smoke_common import (
    canonical_csv_fingerprint,
    load_summary,
    logical_sha256,
    sha256_file,
    split_summary,
)

CONFIG_NAMES = {
    "primary": "amazon_primary.yaml",
    "twins": "amazon_twins.yaml",
    "shape": "amazon_shape.yaml",
    "complementarity": "amazon_complementarity.yaml",
    "reference_history": "amazon_reference_history.yaml",
    "strength": "amazon_strength_fixed.yaml",
    "population_audit": "amazon_k6_population_audit.yaml",
    "full_background": "amazon_full_background.yaml",
    "self_influence": "amazon_self_influence.yaml",
}

STAGE_DIRS = {
    "attack": "stage4_attack",
    "reuse": "stage4_reuse",
    "twins": "stage5_twins",
    "shape": "stage6_shape",
    "complementarity": "stage7_complementarity",
    "reference_history": "stage8_reference_history",
    "strength": "stage9_strength_fixed",
    "population_audit": "stage9_k6_population_audit",
    "full_background": "stage10_full_background",
    "self_influence": "stage11_self_influence",
}

CORE_STAGES = [
    "attack", "reuse", "twins", "shape", "complementarity", "strength", "self_influence"
]
FULL_EXTRA_STAGES = ["reference_history", "population_audit", "full_background"]


def config_dir(category: str) -> Path:
    return ROOT / "configs" if category == "home_and_kitchen" else ROOT / "configs" / "electronics"


def load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected YAML mapping")
    return value


def smoke_rel(run_id: str, stage: str) -> str:
    return f"_smoke/{run_id}/{stage}"


def prepare_configs(category: str, seed: int, run_id: str, output_dir: Path) -> dict[str, Path]:
    src_dir = config_dir(category)
    output_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for label, name in CONFIG_NAMES.items():
        src = src_dir / name
        cfg = load_yaml(src)
        if cfg.get("category") != category:
            raise RuntimeError(f"{src}: category mismatch")
        if "seeds" in cfg:
            cfg["seeds"] = [seed]
        if "seed_ids" in cfg:
            cfg["seed_ids"] = [seed]
        shared = cfg.get("shared_data")
        if isinstance(shared, dict):
            if "output_subdir" in shared:
                shared["output_subdir"] = smoke_rel(run_id, STAGE_DIRS[label if label in STAGE_DIRS else {
                    "primary":"attack", "twins":"twins", "shape":"shape", "complementarity":"complementarity",
                    "reference_history":"reference_history", "strength":"strength", "population_audit":"population_audit",
                    "full_background":"full_background", "self_influence":"self_influence"}[label]])
            # Primary config owns both stage-4 outputs.
            if label == "primary":
                shared["output_subdir"] = smoke_rel(run_id, STAGE_DIRS["attack"])
                shared["reuse_output_subdir"] = smoke_rel(run_id, STAGE_DIRS["reuse"])
            if "attack_dir" in shared:
                shared["attack_dir"] = smoke_rel(run_id, STAGE_DIRS["attack"])
            if "reuse_dir" in shared:
                shared["reuse_dir"] = smoke_rel(run_id, STAGE_DIRS["reuse"])
            if "twins_dir" in shared:
                shared["twins_dir"] = smoke_rel(run_id, STAGE_DIRS["twins"])
        cfg["phase3_smoke"] = {"seed": seed, "run_id": run_id}
        dst = output_dir / name
        dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        out[label] = dst
    return out


def safe_clean_smoke(shared_root: Path, run_id: str) -> None:
    base = (shared_root / "_smoke").resolve()
    target = (base / run_id).resolve()
    if target.parent != base or target.name != run_id:
        raise RuntimeError(f"Unsafe smoke cleanup target: {target}")
    if target.exists():
        shutil.rmtree(target)


def _redact(text: str, redactions: list[tuple[str, str]]) -> str:
    # Longest paths first so nested prefixes do not leave identifying tails.
    for actual, replacement in sorted(redactions, key=lambda x: len(x[0]), reverse=True):
        if actual:
            text = text.replace(actual, replacement)
            text = text.replace(actual.replace("\\", "/"), replacement)
    return text


def run_cmd(cmd: list[str], *, log_path: Path, redactions: list[tuple[str, str]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    display = _redact(" ".join(cmd), redactions)
    print("+", display, flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    stdout = _redact(proc.stdout, redactions)
    stderr = _redact(proc.stderr, redactions)
    log_path.write_text(
        "$ " + display + "\n\n--- stdout ---\n" + stdout + "\n--- stderr ---\n" + stderr,
        encoding="utf-8",
    )
    if proc.returncode:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"Command failed ({proc.returncode}); see {log_path}")


def stage_paths(shared_root: Path, run_id: str, seed: int) -> dict[str, Path]:
    base = shared_root / "_smoke" / run_id
    s = f"seed_{seed:03d}"
    return {
        "attack_summary": base / "stage4_attack" / s / "attack_summary.json",
        "attack_world": base / "stage4_attack" / s / "attack_world.csv.gz",
        "reuse_summary": base / "stage4_reuse" / s / "reuse_summary.json",
        "twins_summary": base / "stage5_twins" / s / "summary.json",
        "twins_edges": base / "stage5_twins" / s / "matched_twin_edges.csv.gz",
        "shape_summary": base / "stage6_shape" / s / "summary.json",
        "shape_world": base / "stage6_shape" / s / "shape_attack_world.csv.gz",
        "shape_identity": base / "stage6_shape" / s / "shape_identity_assignment_r8.csv.gz",
        "complementarity_summary": base / "stage7_complementarity" / s / "summary.json",
        "comp_team": base / "stage7_complementarity" / s / "team_incidence.csv.gz",
        "comp_r1": base / "stage7_complementarity" / s / "random_incidence_1.csv.gz",
        "comp_r2": base / "stage7_complementarity" / s / "random_incidence_2.csv.gz",
        "strength_summary": base / "stage9_strength_fixed" / s / "summary.json",
        "strength_k3": base / "stage9_strength_fixed" / s / "attack_world_k3.csv.gz",
        "strength_k6": base / "stage9_strength_fixed" / s / "attack_world_k6.csv.gz",
        "strength_k9": base / "stage9_strength_fixed" / s / "attack_world_k9.csv.gz",
        "full_background_summary": base / "stage10_full_background" / s / "summary.json",
        "self_influence_summary": base / "stage11_self_influence" / s / "summary.json",
    }


def add_summary(report: dict, stage: str, path: Path) -> None:
    summary = load_summary(path)
    exact, numeric = split_summary(summary)
    for key, value in exact.items():
        report["exact_summary"][f"{stage}.{key}"] = value
    for key, value in numeric.items():
        report["numeric_summary"][f"{stage}.{key}"] = value


def build_manifest(category: str, seed: int, scope: str, shared_root: Path, run_id: str, configs: dict[str, Path]) -> dict:
    primary_cfg = load_yaml(configs["primary"])
    layout = AmazonPathLayout.from_config(primary_cfg, repo_root=ROOT, shared_root=shared_root)
    roles = layout.resolve_shared_path(primary_cfg["shared_data"]["roles_file"])
    refs = layout.resolve_shared_path(primary_cfg["shared_data"]["reference_histograms"])
    freeze = layout.resolve_shared_path(primary_cfg["shared_data"]["reference_freeze"])
    freeze_obj = load_summary(freeze)
    require_artifact_category(freeze_obj, expected=category, label="Phase-3 reference freeze", allow_legacy_home_missing=False)

    # Phase-3 input bundles may carry raw-dataset provenance outside the
    # legacy Stage-3 freeze.  Prefer the freeze when available, then fall
    # back to the bundle manifest and portable scan stub.  This affects only
    # the compact verification manifest; no experiment inputs or metrics.
    input_manifest_path = shared_root / "phase3_input_manifest.json"
    input_manifest_obj = load_summary(input_manifest_path) if input_manifest_path.is_file() else {}
    scan_path = shared_root / "scan" / f"{category}_scan.json"
    scan_obj = load_summary(scan_path) if scan_path.is_file() else {}
    raw_dataset_sha256 = (
        ((freeze_obj.get("provenance") or {}).get("raw_dataset_sha256"))
        or freeze_obj.get("raw_dataset_sha256")
        or input_manifest_obj.get("raw_dataset_sha256")
        or ((scan_obj.get("provenance") or {}).get("raw_dataset_sha256"))
        or scan_obj.get("raw_dataset_sha256")
    )

    report = {
        "schema_version": 1,
        "status": "PASS",
        "category": category,
        "seed": seed,
        "scope": scope,
        "platform": {"system": platform.system().lower(), "python": platform.python_version()},
        "inputs": {
            "roles_logical_sha256": logical_sha256(roles),
            "reference_histograms_logical_sha256": logical_sha256(refs),
            "reference_freeze_selected_lambda": freeze_obj.get("selected_lambda"),
            "reference_freeze_category": freeze_obj.get("category") or (freeze_obj.get("provenance") or {}).get("category"),
            "raw_dataset_sha256": raw_dataset_sha256,
        },
        "structural_fingerprints": {},
        "exact_summary": {},
        "numeric_summary": {},
    }
    p = stage_paths(shared_root, run_id, seed)

    report["structural_fingerprints"]["attack_world"] = canonical_csv_fingerprint(
        p["attack_world"],
        fields=["asin","treatment_block","treated_positions","treated_source_lines","original_ratings","replacement_ratings","clean_counts","attack_counts"],
        json_fields=["treated_positions","treated_source_lines","original_ratings","replacement_ratings","clean_counts","attack_counts"],
    )
    for r in (1,2,4,8,16):
        report["structural_fingerprints"][f"reuse_r{r}"] = canonical_csv_fingerprint(
            shared_root / "_smoke" / run_id / "stage4_reuse" / f"seed_{seed:03d}" / f"identity_assignment_r{r}.csv.gz",
            fields=["asin","treatment_block","treated_position","treated_source_line","synthetic_account_id"],
        )
    report["structural_fingerprints"]["matched_twin_edges"] = canonical_csv_fingerprint(
        p["twins_edges"],
        fields=["pair_id","coalition_account_id","control_account_id","asin","block","position","source_line","clean_rating","attack_rating"],
    )
    report["structural_fingerprints"]["shape_world"] = canonical_csv_fingerprint(
        p["shape_world"], fields=["asin","treatment_block","slots","clean_counts","attack_counts"],
        json_fields=["slots","clean_counts","attack_counts"],
    )
    report["structural_fingerprints"]["shape_identity"] = canonical_csv_fingerprint(
        p["shape_identity"], fields=["account_id","asin","block","position","source_line","original_rating","replacement_rating"],
    )
    for key in ("comp_team","comp_r1","comp_r2"):
        report["structural_fingerprints"][key] = canonical_csv_fingerprint(p[key], fields=["account_id","asin"])
    for k in (3,6,9):
        report["structural_fingerprints"][f"strength_k{k}"] = canonical_csv_fingerprint(
            p[f"strength_k{k}"],
            fields=["asin","treatment_block","k","fixed_item_degree","reuse_r","modified_slots","fixed_assignments","clean_counts","attack_counts"],
            json_fields=["modified_slots","fixed_assignments","clean_counts","attack_counts"],
        )

    for stage, key in [
        ("attack","attack_summary"), ("reuse","reuse_summary"), ("twins","twins_summary"),
        ("shape","shape_summary"), ("complementarity","complementarity_summary"),
        ("strength","strength_summary"), ("self_influence","self_influence_summary"),
    ]:
        add_summary(report, stage, p[key])

    if scope == "full":
        ref_cfg = load_yaml(configs["reference_history"])
        ref_layout = AmazonPathLayout.from_config(ref_cfg, repo_root=ROOT, shared_root=shared_root)
        add_summary(report, "reference_history", ref_layout.resolve_shared_path(ref_cfg["shared_data"]["output_subdir"]) / ref_layout.artifact_name("stage8_reference_history_summary"))
        audit_cfg = load_yaml(configs["population_audit"])
        audit_layout = AmazonPathLayout.from_config(audit_cfg, repo_root=ROOT, shared_root=shared_root)
        add_summary(report, "population_audit", audit_layout.resolve_shared_path(audit_cfg["shared_data"]["output_subdir"]) / audit_layout.artifact_name("stage9_k6_population_summary"))
        add_summary(report, "full_background", p["full_background_summary"])

    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", choices=["home_and_kitchen", "electronics"], required=True)
    ap.add_argument("--shared-root", type=Path, required=True,
                    help="Existing category root containing canonical stage2/ and stage3/ inputs.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scope", choices=["core", "full"], default="core")
    ap.add_argument("--run-id", default="phase3_seed0")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    shared_root = args.shared_root.resolve()
    if not shared_root.is_dir():
        raise FileNotFoundError(shared_root)
    if args.clean:
        safe_clean_smoke(shared_root, args.run_id)

    config_out = ROOT / "artifacts" / "phase3_smoke" / "configs" / args.category / args.run_id
    if args.clean and config_out.exists():
        shutil.rmtree(config_out)
    configs = prepare_configs(args.category, args.seed, args.run_id, config_out)

    log_root = ROOT / "artifacts" / "phase3_smoke" / "logs" / args.category / args.run_id / platform.system().lower()
    py = sys.executable
    common = ["--shared-root", str(shared_root)]
    commands = [
        ("attack", [py, "experiments/amazon/run_primary_attack.py", "--config", str(configs["primary"]), *common, "--seed", str(args.seed), "--overwrite"]),
        ("reuse", [py, "experiments/amazon/run_primary_reuse.py", "--config", str(configs["primary"]), *common, "--seed", str(args.seed), "--overwrite"]),
        ("twins", [py, "experiments/amazon/run_matched_twins.py", "--config", str(configs["twins"]), *common, "--seed", str(args.seed), "--overwrite"]),
        ("shape", [py, "experiments/amazon/run_shape.py", "--config", str(configs["shape"]), *common, "--seed", str(args.seed), "--overwrite"]),
        ("complementarity", [py, "experiments/amazon/run_complementarity.py", "--config", str(configs["complementarity"]), *common, "--seed", str(args.seed), "--overwrite"]),
        ("strength", [py, "experiments/amazon/run_strength_fixed.py", "--config", str(configs["strength"]), *common, "--seed", str(args.seed), "--overwrite"]),
        ("self_influence", [py, "experiments/amazon/run_self_influence.py", "--config", str(configs["self_influence"]), *common, "--seed", str(args.seed), "--overwrite"]),
    ]
    if args.scope == "full":
        # Run reference history before the full-background stage; both consume only frozen upstream worlds.
        commands.insert(5, ("reference_history", [py, "experiments/amazon/run_reference_history.py", "--config", str(configs["reference_history"]), *common]))
        commands.insert(7, ("population_audit", [py, "experiments/amazon/run_k6_population_audit.py", "--config", str(configs["population_audit"]), *common]))
        commands.insert(8, ("full_background", [py, "experiments/amazon/run_full_background_ranking.py", "--config", str(configs["full_background"]), *common, "--seed", str(args.seed), "--overwrite"]))

    redactions = [
        (str(shared_root), "<SHARED_ROOT>"),
        (str(ROOT), "<REPO_ROOT>"),
        (str(sys.executable), "<PYTHON>"),
    ]
    for label, cmd in commands:
        run_cmd(cmd, log_path=log_root / f"{label}.log", redactions=redactions)

    manifest = build_manifest(args.category, args.seed, args.scope, shared_root, args.run_id, configs)
    report = args.report
    if report is None:
        report = ROOT / "artifacts" / "verification" / f"phase3_smoke_{args.category}_{platform.system().lower()}_seed{args.seed}_{args.scope}.json"
    elif not report.is_absolute():
        report = ROOT / report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Anonymity scan of newly generated smoke artifacts.
    smoke_root = shared_root / "_smoke" / args.run_id
    anon_report = ROOT / "artifacts" / "verification" / f"phase3_anonymity_{args.category}_{platform.system().lower()}_seed{args.seed}_{args.scope}.json"
    run_cmd([py, "tools/anonymity_scan.py", "--root", str(smoke_root), "--report", str(anon_report)], log_path=log_root / "anonymity.log", redactions=redactions)

    print(f"PHASE 3 {args.category} {args.scope} SMOKE: PASS")
    print(f"Manifest: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
