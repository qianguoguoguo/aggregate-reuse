#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.io import write_csv_gz
from aggregate_reuse.amazon.shape import (
    abs_mean_discrepancy,
    apply_shape_intervention,
    auc_rank,
    block_is_feasible,
    canonical_attack_hash,
    js_divergence_counts_to_reference,
    load_experimental_blocks,
    load_json,
    load_reference_table,
    mean_from_counts,
    paired_no_larger,
    regular_incidence,
    shrunk_reference,
    w1_counts_to_reference,
)
from aggregate_reuse.amazon.paths import AmazonPathLayout
from aggregate_reuse.amazon.provenance import (
    build_provenance,
    require_artifact_category,
    resolve_raw_dataset_sha256,
)



def build_attack_world(seed, L, blocks, refs, lam):
    rng = np.random.default_rng(seed + 600_000)

    feasible = sorted([
        asin for asin in blocks
        if asin in refs
        and block_is_feasible(blocks[asin]["experimental_A"])
        and block_is_feasible(blocks[asin]["experimental_B"])
    ])
    if len(feasible) < L:
        raise RuntimeError(
            f"Only {len(feasible):,} items satisfy Stage-6 feasibility, "
            f"fewer than L={L:,}."
        )

    chosen = sorted(rng.choice(feasible, size=L, replace=False).tolist())
    rows = []

    for asin in chosen:
        ref_counts, loo_probs = refs[asin]
        q = shrunk_reference(ref_counts, loo_probs, lam)

        treatment_block = (
            "experimental_A" if rng.random() < 0.5 else "experimental_B"
        )
        block = blocks[asin][treatment_block]

        clean_counts, attack_counts, slots = apply_shape_intervention(block, rng)

        clean_mean = mean_from_counts(clean_counts)
        attack_mean = mean_from_counts(attack_counts)
        if abs(clean_mean - attack_mean) > 1e-12:
            raise RuntimeError("Mean-preservation invariant failed.")

        clean_w1 = w1_counts_to_reference(clean_counts, q)
        attack_w1 = w1_counts_to_reference(attack_counts, q)
        clean_js = js_divergence_counts_to_reference(clean_counts, q)
        attack_js = js_divergence_counts_to_reference(attack_counts, q)
        clean_abs_mean = abs_mean_discrepancy(clean_counts, q)
        attack_abs_mean = abs_mean_discrepancy(attack_counts, q)

        d_abs_mean = attack_abs_mean - clean_abs_mean
        if abs(d_abs_mean) > 1e-12:
            raise RuntimeError(
                f"{asin}: mean-based paired discrepancy should be exactly zero."
            )

        rows.append({
            "asin": asin,
            "treatment_block": treatment_block,
            "slots": slots,
            "clean_counts": clean_counts.astype(int).tolist(),
            "attack_counts": attack_counts.astype(int).tolist(),
            "clean_mean": clean_mean,
            "attack_mean": attack_mean,
            "clean_w1": clean_w1,
            "attack_w1": attack_w1,
            "d_w1": attack_w1 - clean_w1,
            "clean_js": clean_js,
            "attack_js": attack_js,
            "d_js": attack_js - clean_js,
            "clean_abs_mean": clean_abs_mean,
            "attack_abs_mean": attack_abs_mean,
            "d_abs_mean": d_abs_mean,
        })

    return feasible, rows


def run_seed(seed, cfg, blocks, refs, freeze, out_root, overwrite):
    L = int(cfg["n_items_per_seed"])
    m = int(cfg["m_manipulated_slots_per_block"])
    r = int(cfg["reuse_r"])
    lam = float(freeze["selected_lambda"])

    seed_dir = out_root / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    summary_path = seed_dir / "summary.json"
    if summary_path.exists() and not overwrite:
        print(f"Seed {seed}: already complete, skipping.")
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        require_artifact_category(
            existing,
            expected=cfg["category"],
            label=f"existing Stage-6 seed {seed}",
        )
        return existing

    feasible, attack_rows = build_attack_world(
        seed=seed, L=L, blocks=blocks, refs=refs, lam=lam
    )
    attack_hash = canonical_attack_hash(attack_rows)
    by_asin = {x["asin"]: x for x in attack_rows}
    asins = sorted(by_asin)

    # Save aggregate attack world.
    attack_csv = []
    for x in attack_rows:
        attack_csv.append({
            "asin": x["asin"],
            "treatment_block": x["treatment_block"],
            "slots": json.dumps(x["slots"]),
            "clean_counts": json.dumps(x["clean_counts"]),
            "attack_counts": json.dumps(x["attack_counts"]),
            "clean_mean": x["clean_mean"],
            "attack_mean": x["attack_mean"],
            "clean_w1": x["clean_w1"],
            "attack_w1": x["attack_w1"],
            "d_w1": x["d_w1"],
            "clean_js": x["clean_js"],
            "attack_js": x["attack_js"],
            "d_js": x["d_js"],
            "clean_abs_mean": x["clean_abs_mean"],
            "attack_abs_mean": x["attack_abs_mean"],
            "d_abs_mean": x["d_abs_mean"],
        })
    write_csv_gz(
        seed_dir / "shape_attack_world.csv.gz",
        attack_csv,
        list(attack_csv[0].keys()),
    )

    # Exact r=8 incidence over the six manipulated slots per item.
    incidence_rng = np.random.default_rng(seed * 100_003 + 6_008)
    incidence = regular_incidence(asins, m, r, incidence_rng)

    account_edges = defaultdict(list)
    edge_rows = []
    for asin in asins:
        row = by_asin[asin]
        slots = sorted(row["slots"], key=lambda x: x["source_line"])
        ids = incidence[asin]
        if len(slots) != m or len(ids) != m:
            raise RuntimeError("Column degree mismatch.")
        for k in range(m):
            a = ids[k]
            slot = slots[k]
            account_edges[a].append((asin, row, slot))
            edge_rows.append({
                "account_id": a,
                "asin": asin,
                "block": row["treatment_block"],
                "position": slot["position"],
                "source_line": slot["source_line"],
                "original_rating": slot["original_rating"],
                "replacement_rating": slot["replacement_rating"],
                "d_w1": row["d_w1"],
                "d_js": row["d_js"],
                "d_abs_mean": row["d_abs_mean"],
            })

    coalition_ids = sorted(account_edges)
    for a in coalition_ids:
        if len(account_edges[a]) != r:
            raise RuntimeError(f"{a}: degree != {r}")
        seen_items = [x[0] for x in account_edges[a]]
        if len(seen_items) != len(set(seen_items)):
            raise RuntimeError(f"{a}: duplicate item exposure.")

    write_csv_gz(
        seed_dir / "shape_identity_assignment_r8.csv.gz",
        edge_rows,
        list(edge_rows[0].keys()),
    )

    # Exact clean twins: same r item/block/slot exposures.
    w1_attack, w1_clean = [], []
    js_attack, js_clean = [], []
    mean_attack, mean_clean = [], []
    cf_w1, cf_js, cf_mean = [], [], []
    pair_rows = []

    for a in coalition_ids:
        exposures = account_edges[a]
        aw1 = cw1 = ajs = cjs = am = cm = 0.0
        dw1 = djs = dm = 0.0

        for asin, row, slot in exposures:
            aw1 += float(row["attack_w1"])
            cw1 += float(row["clean_w1"])
            ajs += float(row["attack_js"])
            cjs += float(row["clean_js"])
            am += float(row["attack_abs_mean"])
            cm += float(row["clean_abs_mean"])
            dw1 += float(row["d_w1"])
            djs += float(row["d_js"])
            dm += float(row["d_abs_mean"])

        if abs((aw1 - cw1) - dw1) > 1e-10:
            raise RuntimeError("W1 paired-gap identity failed.")
        if abs((ajs - cjs) - djs) > 1e-10:
            raise RuntimeError("JS paired-gap identity failed.")
        if abs((am - cm) - dm) > 1e-12 or abs(dm) > 1e-12:
            raise RuntimeError("Mean-channel zero-effect invariant failed.")

        w1_attack.append(aw1)
        w1_clean.append(cw1)
        js_attack.append(ajs)
        js_clean.append(cjs)
        mean_attack.append(am)
        mean_clean.append(cm)
        cf_w1.append(dw1)
        cf_js.append(djs)
        cf_mean.append(dm)

        pair_rows.append({
            "pair_id": a,
            "coalition_account_id": a,
            "control_account_id": a.replace("syn_shape_r8_", "syn_shape_ctrl_r8_", 1),
            "frequency_each": r,
            "w1_attack_score": aw1,
            "w1_clean_score": cw1,
            "w1_paired_gap": dw1,
            "js_attack_score": ajs,
            "js_clean_score": cjs,
            "js_paired_gap": djs,
            "abs_mean_attack_score": am,
            "abs_mean_clean_score": cm,
            "abs_mean_paired_gap": dm,
        })

    write_csv_gz(
        seed_dir / "shape_matched_twin_accounts.csv.gz",
        pair_rows,
        list(pair_rows[0].keys()),
    )

    w1_attack = np.asarray(w1_attack)
    w1_clean = np.asarray(w1_clean)
    js_attack = np.asarray(js_attack)
    js_clean = np.asarray(js_clean)
    mean_attack = np.asarray(mean_attack)
    mean_clean = np.asarray(mean_clean)
    cf_w1 = np.asarray(cf_w1)
    cf_js = np.asarray(cf_js)
    cf_mean = np.asarray(cf_mean)

    mean_d_w1 = float(np.mean([x["d_w1"] for x in attack_rows]))
    mean_d_js = float(np.mean([x["d_js"] for x in attack_rows]))
    mean_d_mean = float(np.mean([x["d_abs_mean"] for x in attack_rows]))

    summary = {
        "stage": "amazon_stage6_mean_preserving_shape_r8",
        "category": cfg["category"],
        "seed": seed,
        "selected_lambda": lam,
        "reuse_r": r,
        "n_items": L,
        "m": m,
        "n_account_pairs": len(coalition_ids),
        "feasible_item_universe": len(feasible),
        "attack_world_sha256": attack_hash,
        "aggregate": {
            "mean_d_w1": mean_d_w1,
            "positive_d_w1_fraction": float(np.mean([x["d_w1"] > 0 for x in attack_rows])),
            "mean_d_js": mean_d_js,
            "positive_d_js_fraction": float(np.mean([x["d_js"] > 0 for x in attack_rows])),
            "mean_d_abs_mean": mean_d_mean,
            "max_abs_d_abs_mean": float(
                np.max(np.abs([x["d_abs_mean"] for x in attack_rows]))
            ),
        },
        "matched_twins": {
            "frequency_auc": 0.5,
            "w1_counterfactual_score_auc": auc_rank(cf_w1, np.zeros_like(cf_w1)),
            "w1_raw_world_score_auc": auc_rank(w1_attack, w1_clean),
            "w1_paired_misordering_probability": paired_no_larger(w1_attack, w1_clean),
            "w1_positive_pair_fraction": float(np.mean(cf_w1 > 0)),
            "w1_mean_paired_gap": float(cf_w1.mean()),
            "js_counterfactual_score_auc": auc_rank(cf_js, np.zeros_like(cf_js)),
            "js_raw_world_score_auc": auc_rank(js_attack, js_clean),
            "js_paired_misordering_probability": paired_no_larger(js_attack, js_clean),
            "js_positive_pair_fraction": float(np.mean(cf_js > 0)),
            "js_mean_paired_gap": float(cf_js.mean()),
            "mean_counterfactual_score_auc": auc_rank(cf_mean, np.zeros_like(cf_mean)),
            "mean_raw_world_score_auc": auc_rank(mean_attack, mean_clean),
            "mean_paired_misordering_probability": paired_no_larger(mean_attack, mean_clean),
            "mean_positive_pair_fraction": float(np.mean(cf_mean > 0)),
            "mean_mean_paired_gap": float(cf_mean.mean()),
        },
        "invariants": {
            "exact_block_mean_preservation": True,
            "all_six_selected_ratings_change": True,
            "three_disjoint_sum_preserving_pairs": True,
            "frequency_exactly_matched_at_r8": True,
            "item_block_slot_exposure_pairwise_matched": True,
            "mean_based_paired_effect_exactly_zero": True,
        },
        "interpretation_guard": (
            "Stage 6 is a controlled counterfactual shape-distortion experiment "
            "on real Amazon blocks. It tests distribution-sensitive evidence "
            "under exact mean preservation; it is not historical fraud labeling."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def summarize(vals):
    vals = np.asarray(vals, dtype=float)
    mean = float(vals.mean())
    if len(vals) <= 1:
        return {"mean": mean, "ci_lower": mean, "ci_upper": mean}
    se = float(vals.std(ddof=1) / math.sqrt(len(vals)))
    crit = float(t.ppf(0.975, df=len(vals) - 1))
    return {
        "mean": mean,
        "ci_lower": mean - crit * se,
        "ci_upper": mean + crit * se,
    }


def aggregate(
    summaries,
    out_root,
    *,
    layout,
    config_path,
    raw_dataset_sha256,
    freeze,
    upstream_artifacts,
):
    scalar_paths = {
        "mean_d_w1": ("aggregate", "mean_d_w1"),
        "positive_d_w1_fraction": ("aggregate", "positive_d_w1_fraction"),
        "mean_d_js": ("aggregate", "mean_d_js"),
        "positive_d_js_fraction": ("aggregate", "positive_d_js_fraction"),
        "mean_d_abs_mean": ("aggregate", "mean_d_abs_mean"),
        "max_abs_d_abs_mean": ("aggregate", "max_abs_d_abs_mean"),
        "frequency_auc": ("matched_twins", "frequency_auc"),
        "w1_counterfactual_score_auc": ("matched_twins", "w1_counterfactual_score_auc"),
        "w1_raw_world_score_auc": ("matched_twins", "w1_raw_world_score_auc"),
        "w1_paired_misordering_probability": (
            "matched_twins", "w1_paired_misordering_probability"
        ),
        "w1_positive_pair_fraction": ("matched_twins", "w1_positive_pair_fraction"),
        "w1_mean_paired_gap": ("matched_twins", "w1_mean_paired_gap"),
        "js_counterfactual_score_auc": ("matched_twins", "js_counterfactual_score_auc"),
        "js_raw_world_score_auc": ("matched_twins", "js_raw_world_score_auc"),
        "js_paired_misordering_probability": (
            "matched_twins", "js_paired_misordering_probability"
        ),
        "js_positive_pair_fraction": ("matched_twins", "js_positive_pair_fraction"),
        "js_mean_paired_gap": ("matched_twins", "js_mean_paired_gap"),
        "mean_counterfactual_score_auc": (
            "matched_twins", "mean_counterfactual_score_auc"
        ),
        "mean_raw_world_score_auc": ("matched_twins", "mean_raw_world_score_auc"),
        "mean_paired_misordering_probability": (
            "matched_twins", "mean_paired_misordering_probability"
        ),
        "mean_positive_pair_fraction": ("matched_twins", "mean_positive_pair_fraction"),
        "mean_mean_paired_gap": ("matched_twins", "mean_mean_paired_gap"),
    }

    metrics = {}
    for name, (section, key) in scalar_paths.items():
        metrics[name] = summarize([s[section][key] for s in summaries])

    final = {
        "stage": "amazon_stage6_mean_preserving_shape_r8",
        "category": layout.category,
        "raw_dataset_sha256": raw_dataset_sha256,
        "selected_lambda": float(freeze["selected_lambda"]),
        "n_seeds": len(summaries),
        "seed_ids": [int(s["seed"]) for s in summaries],
        "reuse_r": 8,
        "n_items_per_seed": summaries[0]["n_items"],
        "metrics": metrics,
        "invariants": summaries[0]["invariants"],
        "interpretation_guard": summaries[0]["interpretation_guard"],
    }
    final["provenance"] = build_provenance(
        category=layout.category,
        config_path=config_path,
        raw_dataset_sha256=raw_dataset_sha256,
        selected_lambda=freeze["selected_lambda"],
        seed_ids=final["seed_ids"],
        upstream_artifacts=upstream_artifacts,
        repo_root=ROOT,
        shared_root=layout.shared_category_root,
    )
    out = out_root / layout.artifact_name("stage6_shape_summary")
    out.write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(ROOT / "configs" / "amazon_shape.yaml"),
    )
    ap.add_argument(
        "--shared-root",
        type=Path,
        help="Operational shared-root override; does not alter the config file.",
    )
    ap.add_argument(
        "--roles",
        default=None,
    )
    ap.add_argument(
        "--references",
        default=None,
    )
    ap.add_argument(
        "--freeze",
        default=None,
    )
    ap.add_argument(
        "--out-dir",
        default=None,
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed", type=int)
    g.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    layout = AmazonPathLayout.from_config(
        cfg,
        repo_root=ROOT,
        shared_root=args.shared_root,
    )
    shared = cfg["shared_data"]
    roles_path = (
        Path(args.roles).resolve() if args.roles
        else layout.resolve_shared_path(shared["roles_file"])
    )
    references_path = (
        Path(args.references).resolve() if args.references
        else layout.resolve_shared_path(shared["reference_histograms"])
    )
    freeze_path = (
        Path(args.freeze).resolve() if args.freeze
        else layout.resolve_shared_path(shared["reference_freeze"])
    )
    out_root = (
        Path(args.out_dir).resolve() if args.out_dir
        else layout.resolve_shared_path(shared["output_subdir"])
    )
    freeze = load_json(freeze_path)

    require_artifact_category(
        freeze,
        expected=cfg["category"],
        label="Stage-3 freeze",
        allow_legacy_home_missing=False,
    )
    if freeze["stage4_evidence_mode"] != "paired_counterfactual":
        raise RuntimeError("Stage-3 freeze does not authorize paired mode.")

    blocks = load_experimental_blocks(roles_path)
    refs = load_reference_table(references_path)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        if args.seed not in cfg["seed_ids"]:
            raise ValueError("Seed not in frozen Stage-6 seed list.")
        s = run_seed(
            int(args.seed), cfg, blocks, refs, freeze, out_root, args.overwrite
        )
        print(json.dumps({
            "aggregate": s["aggregate"],
            "matched_twins": s["matched_twins"],
            "feasible_item_universe": s["feasible_item_universe"],
        }, indent=2))
        print(f"Summary: {out_root / f'seed_{args.seed:03d}' / 'summary.json'}")
        return

    summaries = []
    for seed in cfg["seed_ids"]:
        print(f"\n=== Stage 6 seed {seed} ===")
        summaries.append(
            run_seed(
                int(seed), cfg, blocks, refs, freeze, out_root, args.overwrite
            )
        )

    raw_dataset_sha256 = resolve_raw_dataset_sha256(layout, freeze)
    final = aggregate(
        summaries,
        out_root,
        layout=layout,
        config_path=config_path,
        raw_dataset_sha256=raw_dataset_sha256,
        freeze=freeze,
        upstream_artifacts={
            "roles_file": roles_path,
            "reference_histograms": references_path,
            "reference_freeze": freeze_path,
        },
    )
    print("\nStage 6 complete.")
    for name, stat in final["metrics"].items():
        print(
            f"{name}: {stat['mean']:.6f} "
            f"[{stat['ci_lower']:.6f}, {stat['ci_upper']:.6f}]"
        )
    print(f"Summary: {out_root / layout.artifact_name('stage6_shape_summary')}")


if __name__ == "__main__":
    main()
