#!/usr/bin/env python3
"""Construct the primary Amazon k=6 attack worlds only.

Phase 2I stops BEFORE synthetic identity assignment.

Reads shared Phase-2G/2H data:
  ../amazon_preprocess/stage2/
  ../amazon_preprocess/stage3/

Writes reusable attack worlds:
  ../amazon_preprocess/stage4_attack/

Per seed:
  seed_000/attack_world.csv.gz
  seed_000/attack_summary.json
  ...

No reuse incidence, account score, AUC, or matched-twin calculation occurs here.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.primary_attack import (
    build_attack_world,
    canonical_attack_hash,
    load_experimental_blocks,
    load_json,
    load_reference_table,
)
from aggregate_reuse.amazon.paths import AmazonPathLayout
from aggregate_reuse.amazon.provenance import (
    build_provenance,
    require_artifact_category,
    resolve_raw_dataset_sha256,
)


def write_csv_gz(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def load_cfg(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def shared_paths(cfg, *, shared_root=None):
    layout = AmazonPathLayout.from_config(
        cfg,
        repo_root=ROOT,
        shared_root=shared_root,
    )
    root = layout.shared_category_root
    shared = cfg["shared_data"]
    return {
        "layout": layout,
        "root": root,
        "roles": layout.resolve_shared_path(shared["roles_file"]),
        "references": layout.resolve_shared_path(
            shared["reference_histograms"]
        ),
        "freeze": layout.resolve_shared_path(shared["reference_freeze"]),
        "output": layout.resolve_shared_path(shared["output_subdir"]),
    }


def run_seed(seed, cfg, blocks, refs, freeze, out_root, overwrite=False):
    L = int(cfg["attack"]["n_items_per_seed"])
    m = int(cfg["attack"]["manipulated_slots_per_item"])
    lam = float(freeze["selected_lambda"])

    seed_dir = out_root / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    summary_path = seed_dir / "attack_summary.json"
    attack_path = seed_dir / "attack_world.csv.gz"

    if summary_path.exists() and attack_path.exists() and not overwrite:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        require_artifact_category(
            existing,
            expected=cfg["category"],
            label=f"existing attack seed {seed}",
        )
        return existing

    feasible, attack_rows, _participant_blocks = build_attack_world(
        seed=seed,
        L=L,
        m=m,
        blocks=blocks,
        refs=refs,
        lam=lam,
    )

    attack_hash = canonical_attack_hash(attack_rows)
    d = np.asarray([x["d_cf"] for x in attack_rows], dtype=float)

    csv_rows = []
    for x in attack_rows:
        csv_rows.append({
            "asin": x["asin"],
            "treatment_block": x["treatment_block"],
            "treated_positions": json.dumps(x["treated_positions"]),
            "treated_source_lines": json.dumps(x["treated_source_lines"]),
            "original_ratings": json.dumps(x["original_ratings"]),
            "replacement_ratings": json.dumps(x["replacement_ratings"]),
            "clean_counts": json.dumps(x["clean_counts"]),
            "attack_counts": json.dumps(x["attack_counts"]),
            "clean_w1": x["clean_w1"],
            "attack_w1": x["attack_w1"],
            "d_cf": x["d_cf"],
        })

    write_csv_gz(
        attack_path,
        csv_rows,
        list(csv_rows[0].keys()),
    )

    # Explicit attack-only invariants.
    exactly_m = all(len(x["treated_positions"]) == m for x in attack_rows)
    donor_nonfive = all(
        all(int(r) != 5 for r in x["original_ratings"])
        for x in attack_rows
    )
    replacement_five = all(
        all(int(r) == 5 for r in x["replacement_ratings"])
        for x in attack_rows
    )
    counts_size = all(
        sum(x["clean_counts"]) == 30 and sum(x["attack_counts"]) == 30
        for x in attack_rows
    )
    unique_slots = all(
        len(x["treated_source_lines"]) == len(set(x["treated_source_lines"]))
        for x in attack_rows
    )

    summary = {
        "experiment": "amazon_primary_attack",
        "category": cfg["category"],
        "seed": int(seed),
        "selected_lambda": lam,
        "evidence_mode": freeze["stage4_evidence_mode"],
        "feasible_item_universe": len(feasible),
        "sampled_items": L,
        "m": m,
        "attack_world_sha256": attack_hash,
        "aggregate_visibility": {
            "mean_d_cf": float(d.mean()),
            "positive_fraction": float((d > 0).mean()),
            "zero_fraction": float((d == 0).mean()),
            "negative_fraction": float((d < 0).mean()),
            "clean_w1_mean": float(
                np.mean([x["clean_w1"] for x in attack_rows])
            ),
            "attack_w1_mean": float(
                np.mean([x["attack_w1"] for x in attack_rows])
            ),
        },
        "invariants": {
            "exactly_m_manipulated_slots_per_item": bool(exactly_m),
            "all_selected_donor_ratings_are_nonfive": bool(donor_nonfive),
            "all_replacements_are_five": bool(replacement_five),
            "clean_and_attack_histograms_have_30_reviews": bool(counts_size),
            "no_duplicate_treated_source_line_within_item": bool(unique_slots),
            "identity_assignment_not_constructed": True,
        },
    }

    if not all(summary["invariants"].values()):
        raise RuntimeError(f"Seed {seed}: attack-world invariant failed.")

    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def aggregate(
    summaries,
    out_root,
    *,
    layout,
    config_path,
    raw_dataset_sha256,
    selected_lambda,
    upstream_artifacts,
):
    means = np.asarray(
        [s["aggregate_visibility"]["mean_d_cf"] for s in summaries],
        dtype=float,
    )
    mean = float(means.mean())
    se = float(means.std(ddof=1) / math.sqrt(len(means)))
    crit = float(t.ppf(0.975, df=len(means)-1))

    result = {
        "experiment": "amazon_primary_attack",
        "category": layout.category,
        "raw_dataset_sha256": raw_dataset_sha256,
        "selected_lambda": float(selected_lambda),
        "n_seeds": len(summaries),
        "seed_ids": [int(s["seed"]) for s in summaries],
        "aggregate_visibility_mean_d_cf": {
            "mean": mean,
            "ci_lower": float(mean - crit*se),
            "ci_upper": float(mean + crit*se),
        },
        "all_attack_invariants_pass": all(
            all(s["invariants"].values()) for s in summaries
        ),
        "identity_assignment_performed": False,
    }
    result["provenance"] = build_provenance(
        category=layout.category,
        config_path=config_path,
        raw_dataset_sha256=raw_dataset_sha256,
        selected_lambda=selected_lambda,
        seed_ids=result["seed_ids"],
        upstream_artifacts=upstream_artifacts,
        repo_root=ROOT,
        shared_root=layout.shared_category_root,
    )

    (out_root / layout.artifact_name("stage4_attack_summary")).write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rows = []
    for s in summaries:
        v = s["aggregate_visibility"]
        rows.append({
            "seed": s["seed"],
            "attack_world_sha256": s["attack_world_sha256"],
            "n_items": s["sampled_items"],
            "m": s["m"],
            "mean_d_cf": v["mean_d_cf"],
            "positive_d_cf_fraction": v["positive_fraction"],
            "zero_d_cf_fraction": v["zero_fraction"],
            "negative_d_cf_fraction": v["negative_fraction"],
            "clean_w1_mean": v["clean_w1_mean"],
            "attack_w1_mean": v["attack_w1_mean"],
        })

    with (out_root / layout.artifact_name("stage4_attack_seed_metrics")).open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(ROOT / "configs" / "amazon_primary.yaml"),
    )
    ap.add_argument(
        "--shared-root",
        type=Path,
        help="Operational shared-root override; does not alter the config file.",
    )
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--seed", type=int)
    g.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_cfg(config_path)
    paths = shared_paths(cfg, shared_root=args.shared_root)

    for key in ["roles", "references", "freeze"]:
        if not paths[key].exists():
            raise FileNotFoundError(paths[key])

    freeze = load_json(paths["freeze"])
    require_artifact_category(
        freeze,
        expected=cfg["category"],
        label="reference freeze",
        allow_legacy_home_missing=False,
    )
    if freeze["stage4_evidence_mode"] != "paired_counterfactual":
        raise RuntimeError("Reference freeze does not authorize paired evidence.")
    if freeze["holdout_used_for_selection"]:
        raise RuntimeError("Invalid freeze: holdout was used for selection.")
    raw_dataset_sha256 = resolve_raw_dataset_sha256(paths["layout"], freeze)

    blocks = load_experimental_blocks(paths["roles"])
    refs = load_reference_table(paths["references"])
    paths["output"].mkdir(parents=True, exist_ok=True)

    seeds = [int(x) for x in cfg["seeds"]]
    if args.seed is not None:
        if args.seed not in seeds:
            raise ValueError("Seed outside frozen list.")
        summary = run_seed(
            args.seed, cfg, blocks, refs, freeze, paths["output"], args.overwrite
        )
        print(json.dumps(summary, indent=2))
        return 0

    # Default is all seeds; --all-seeds is accepted for readability.
    summaries = []
    for seed in seeds:
        print(f"=== Primary attack seed {seed} ===")
        summaries.append(
            run_seed(
                seed, cfg, blocks, refs, freeze,
                paths["output"], args.overwrite
            )
        )

    result = aggregate(
        summaries,
        paths["output"],
        layout=paths["layout"],
        config_path=config_path,
        raw_dataset_sha256=raw_dataset_sha256,
        selected_lambda=freeze["selected_lambda"],
        upstream_artifacts={
            "stage2_roles": paths["roles"],
            "stage2_reference_histograms": paths["references"],
            "stage3_reference_freeze": paths["freeze"],
        },
    )
    print("Primary Amazon attack construction complete.")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
