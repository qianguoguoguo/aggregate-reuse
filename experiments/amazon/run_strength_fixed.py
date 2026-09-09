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

from aggregate_reuse.amazon.strength_fixed import (
    auc_rank,
    fixed_nested_incidence,
    hist,
    load_experimental_blocks,
    load_json,
    load_reference_table,
    modification_counts,
    shrunk_reference,
    w1_counts,
)
from aggregate_reuse.amazon.paths import AmazonPathLayout
from aggregate_reuse.amazon.provenance import (
    build_provenance,
    require_artifact_category,
    resolve_raw_dataset_sha256,
)


def write_csv_gz(path, rows, fields):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build_common_seed(seed, cfg, blocks, refs, lam):
    L = int(cfg["n_items_per_seed"])
    strengths = [int(x) for x in cfg["strength_grid_k"]]
    max_k = max(strengths)
    rng = np.random.default_rng(seed + int(cfg["attack_seed_offset"]))

    # Common population: BOTH candidate blocks must support the largest k.
    feasible = sorted([
        asin for asin in blocks if asin in refs
        and sum(int(x["rating"]) != 5 for x in blocks[asin]["experimental_A"]) >= max_k
        and sum(int(x["rating"]) != 5 for x in blocks[asin]["experimental_B"]) >= max_k
    ])
    if len(feasible) < L:
        raise RuntimeError(
            f"Only {len(feasible):,} items have >= {max_k} non-five-star "
            f"reviews in both experimental blocks; need {L:,}."
        )

    chosen = sorted(rng.choice(feasible, size=L, replace=False).tolist())
    common = {}

    for asin in chosen:
        block_name = "experimental_A" if rng.random() < 0.5 else "experimental_B"
        block = blocks[asin][block_name]
        cand = [j for j, r in enumerate(block) if int(r["rating"]) != 5]

        # One common ordered list of 9 donor positions.
        donor_order = rng.choice(cand, size=max_k, replace=False).tolist()

        c, loo = refs[asin]
        q = shrunk_reference(c, loo, lam)
        clean = hist(block)

        common[asin] = {
            "block_name": block_name,
            "block": block,
            "donor_order": [int(x) for x in donor_order],
            "q": q,
            "clean_counts": clean,
            "clean_w1": w1_counts(clean, q),
        }

    return feasible, common


def run_seed(seed, cfg, blocks, refs, freeze, out_root, overwrite):
    strengths = [int(x) for x in cfg["strength_grid_k"]]
    r = int(cfg["reuse_r"])
    item_degree = int(cfg["fixed_item_degree"])
    lam = float(freeze["selected_lambda"])

    if max(strengths) != item_degree:
        raise ValueError(
            "For this fixed-identity design, fixed_item_degree must equal max(strength_grid_k)."
        )

    out_seed = out_root / f"seed_{seed:03d}"
    out_seed.mkdir(parents=True, exist_ok=True)
    summary_path = out_seed / "summary.json"
    if summary_path.exists() and not overwrite:
        print(f"Seed {seed}: already complete, skipping.")
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        require_artifact_category(
            existing,
            expected=cfg["category"],
            label=f"existing Stage-9 seed {seed}",
        )
        return existing

    feasible, common = build_common_seed(seed, cfg, blocks, refs, lam)
    asins = sorted(common)

    # ------------------------------------------------------------
    # ONE identity population and ONE account-item incidence for ALL k.
    # ------------------------------------------------------------
    incidence_seed = (
        seed * int(cfg["incidence_seed_multiplier"])
        + int(cfg["incidence_seed_offset"])
    )
    account_by_rank = fixed_nested_incidence(
        asins,
        item_degree=item_degree,
        reuse_r=r,
        rng=np.random.default_rng(incidence_seed),
    )

    all_accounts = sorted({a for ids in account_by_rank.values() for a in ids})
    M = len(all_accounts)
    expected_M = len(asins) * item_degree // r
    if M != expected_M:
        raise RuntimeError(f"Expected {expected_M} accounts, got {M}.")

    # Fixed exposure set for every account, used at every k.
    account_freq = defaultdict(int)
    for asin in asins:
        for a in account_by_rank[asin]:
            account_freq[a] += 1
    if len(account_freq) != M or any(v != r for v in account_freq.values()):
        raise RuntimeError("Fixed account exposure invariant failed.")

    # Verify balanced nested modified-exposure counts.
    mod_count_summary = {}
    previous_active_edges = None
    for k in strengths:
        counts = modification_counts(account_by_rank, k)
        vals = np.asarray([counts.get(a, 0) for a in all_accounts], dtype=int)

        active_edges = {
            (asin, rank, account_by_rank[asin][rank])
            for asin in asins
            for rank in range(k)
        }
        if previous_active_edges is not None and not previous_active_edges.issubset(active_edges):
            raise RuntimeError("Nested modified-edge invariant failed.")
        previous_active_edges = active_edges

        if vals.max() - vals.min() > 1:
            raise RuntimeError(
                f"Modified exposures are not balanced at k={k}: "
                f"min={vals.min()}, max={vals.max()}."
            )

        mod_count_summary[str(k)] = {
            "min_modified_exposures_per_account": int(vals.min()),
            "max_modified_exposures_per_account": int(vals.max()),
            "mean_modified_exposures_per_account": float(vals.mean()),
        }

    metrics = {}

    for k in strengths:
        d_by_item = {}
        item_rows = []

        for asin in asins:
            x = common[asin]
            attack = x["clean_counts"].copy()
            slots = []

            # k is ONLY the number of modified ratings.
            for donor_rank, j in enumerate(x["donor_order"][:k]):
                rev = x["block"][j]
                old = int(rev["rating"])
                if old == 5:
                    raise RuntimeError("Five-star donor selected.")
                attack[old - 1] -= 1
                attack[4] += 1
                slots.append({
                    "donor_rank": int(donor_rank),
                    "account_id": account_by_rank[asin][donor_rank],
                    "position": int(rev["position"]),
                    "source_line": int(rev["source_line"]),
                    "original_rating": old,
                    "replacement_rating": 5,
                })

            if len(slots) != k or attack.sum() != 30 or np.any(attack < 0):
                raise RuntimeError("Attack invariant failed.")

            attack_w1 = w1_counts(attack, x["q"])
            d = attack_w1 - x["clean_w1"]
            d_by_item[asin] = d

            # Record ALL nine fixed synthetic identities, not only modified ones.
            fixed_assignments = []
            for donor_rank, j in enumerate(x["donor_order"]):
                rev = x["block"][j]
                fixed_assignments.append({
                    "donor_rank": int(donor_rank),
                    "account_id": account_by_rank[asin][donor_rank],
                    "position": int(rev["position"]),
                    "source_line": int(rev["source_line"]),
                    "rating_in_clean_world": int(rev["rating"]),
                    "is_modified_at_k": bool(donor_rank < k),
                })

            item_rows.append({
                "asin": asin,
                "treatment_block": x["block_name"],
                "k": k,
                "fixed_item_degree": item_degree,
                "reuse_r": r,
                "modified_slots": json.dumps(slots),
                "fixed_assignments": json.dumps(fixed_assignments),
                "clean_counts": json.dumps(x["clean_counts"].astype(int).tolist()),
                "attack_counts": json.dumps(attack.astype(int).tolist()),
                "clean_w1": x["clean_w1"],
                "attack_w1": attack_w1,
                "d_cf": d,
            })

        # ------------------------------------------------------------
        # IMPORTANT: score the SAME account population on the SAME 8 item
        # exposures at every k.  Only d_cf changes with k.
        # ------------------------------------------------------------
        account_scores = defaultdict(float)
        for asin in asins:
            d = d_by_item[asin]
            for a in account_by_rank[asin]:
                account_scores[a] += d

        if set(account_scores) != set(all_accounts):
            raise RuntimeError("Account population changed across strength.")

        scores = np.asarray([account_scores[a] for a in all_accounts], float)
        zeros = np.zeros_like(scores)

        mean_d = float(np.mean(list(d_by_item.values())))
        mean_gap = float(scores.mean())
        pred = r * mean_d

        if abs(mean_gap - pred) > 1e-10:
            raise RuntimeError("Reuse law failed under fixed identity exposure.")

        mod_counts = modification_counts(account_by_rank, k)
        mod_vals = np.asarray([mod_counts.get(a, 0) for a in all_accounts], int)

        metrics[str(k)] = {
            "n_accounts": M,
            "fixed_item_degree": item_degree,
            "reuse_r": r,
            "mean_d_cf": mean_d,
            "positive_block_fraction": float(
                np.mean(np.asarray(list(d_by_item.values())) > 0)
            ),
            "counterfactual_score_auc": auc_rank(scores, zeros),
            "positive_pair_fraction": float(np.mean(scores > 0)),
            "paired_misordering_probability": float(np.mean(scores <= 0)),
            "mean_paired_score_gap": mean_gap,
            "predicted_mean_gap_r_times_mean_dcf": pred,
            "reuse_law_abs_error": abs(mean_gap - pred),
            "min_modified_exposures_per_account": int(mod_vals.min()),
            "max_modified_exposures_per_account": int(mod_vals.max()),
            "mean_modified_exposures_per_account": float(mod_vals.mean()),
        }

        write_csv_gz(
            out_seed / f"attack_world_k{k}.csv.gz",
            item_rows,
            list(item_rows[0].keys()),
        )

    summary = {
        "stage": "amazon_stage9_intervention_strength_fixed_identity",
        "category": cfg["category"],
        "seed": seed,
        "selected_lambda": lam,
        "reuse_r": r,
        "fixed_item_degree": item_degree,
        "n_items": len(asins),
        "n_accounts": M,
        "feasible_item_universe": len(feasible),
        "strength_grid_k": strengths,
        "metrics_by_k": metrics,
        "modified_exposure_balance": mod_count_summary,
        "invariants": {
            "same_items_across_strengths": True,
            "same_treatment_block_across_strengths": True,
            "same_reference_across_strengths": True,
            "nested_donor_slots_3_in_6_in_9": True,
            "same_identity_population_across_strengths": True,
            "same_account_item_incidence_across_strengths": True,
            "account_degree_exactly_r_across_strengths": True,
            "item_degree_fixed_at_9_across_strengths": True,
            "modified_edges_nested_across_strengths": True,
            "modified_exposure_count_balanced_within_one_across_accounts": True,
            "all_replacements_are_five": True,
        },
        "interpretation_guard": (
            "Only intervention strength k changes in the scored worlds. "
            "Items, treatment blocks, reference model, nine donor positions, "
            "synthetic identity population, account-item incidence, and per-account "
            "exposure (r=8) are fixed across k. The k=3 and k=6 manipulated "
            "positions are nested subsets of k=9. Modified exposures are balanced "
            "across identities: counts differ by at most one for each k."
        ),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def ci(vals):
    x = np.asarray(vals, float)
    m = float(x.mean())
    if len(x) == 1:
        return {"mean": m, "ci_lower": m, "ci_upper": m}
    se = float(x.std(ddof=1) / math.sqrt(len(x)))
    crit = float(t.ppf(0.975, df=len(x) - 1))
    return {
        "mean": m,
        "ci_lower": m - crit * se,
        "ci_upper": m + crit * se,
    }


def aggregate(
    ss,
    cfg,
    out_root,
    *,
    layout,
    config_path,
    raw_dataset_sha256,
    freeze,
    upstream_artifacts,
):
    final_metrics = {}
    for k in [int(x) for x in cfg["strength_grid_k"]]:
        rr = [s["metrics_by_k"][str(k)] for s in ss]
        final_metrics[str(k)] = {
            "n_accounts_per_seed": rr[0]["n_accounts"],
            "fixed_item_degree": rr[0]["fixed_item_degree"],
            "reuse_r": rr[0]["reuse_r"],
            "mean_d_cf": ci([x["mean_d_cf"] for x in rr]),
            "positive_block_fraction": ci(
                [x["positive_block_fraction"] for x in rr]
            ),
            "counterfactual_score_auc": ci(
                [x["counterfactual_score_auc"] for x in rr]
            ),
            "positive_pair_fraction": ci(
                [x["positive_pair_fraction"] for x in rr]
            ),
            "paired_misordering_probability": ci(
                [x["paired_misordering_probability"] for x in rr]
            ),
            "mean_paired_score_gap": ci(
                [x["mean_paired_score_gap"] for x in rr]
            ),
            "reuse_law_abs_error": ci(
                [x["reuse_law_abs_error"] for x in rr]
            ),
            "min_modified_exposures_per_account": rr[0][
                "min_modified_exposures_per_account"
            ],
            "max_modified_exposures_per_account": rr[0][
                "max_modified_exposures_per_account"
            ],
        }

    final = {
        "stage": "amazon_stage9_intervention_strength_fixed_identity",
        "category": layout.category,
        "raw_dataset_sha256": raw_dataset_sha256,
        "selected_lambda": float(freeze["selected_lambda"]),
        "n_seeds": len(ss),
        "seed_ids": [int(s["seed"]) for s in ss],
        "reuse_r": int(cfg["reuse_r"]),
        "fixed_item_degree": int(cfg["fixed_item_degree"]),
        "strength_grid_k": [int(x) for x in cfg["strength_grid_k"]],
        "metrics_by_k": final_metrics,
        "invariants": ss[0]["invariants"],
        "interpretation_guard": ss[0]["interpretation_guard"],
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
    p = out_root / layout.artifact_name("stage9_strength_summary")
    p.write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(ROOT / "configs" / "amazon_strength_fixed.yaml"),
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
    out = (
        Path(args.out_dir).resolve() if args.out_dir
        else layout.resolve_shared_path(shared["output_subdir"])
    )
    blocks = load_experimental_blocks(roles_path)
    refs = load_reference_table(references_path)
    freeze = load_json(freeze_path)
    require_artifact_category(
        freeze,
        expected=cfg["category"],
        label="Stage-3 freeze",
        allow_legacy_home_missing=False,
    )
    out.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        if args.seed not in cfg["seed_ids"]:
            raise ValueError("Seed not frozen.")
        s = run_seed(
            args.seed, cfg, blocks, refs, freeze, out, args.overwrite
        )
        print(json.dumps(s["metrics_by_k"], indent=2))
        print(f"Summary: {out / f'seed_{args.seed:03d}' / 'summary.json'}")
        return

    ss = []
    for seed in cfg["seed_ids"]:
        print(f"\n=== Fixed-identity Stage 9 seed {seed} ===")
        ss.append(
            run_seed(
                int(seed),
                cfg,
                blocks,
                refs,
                freeze,
                out,
                args.overwrite,
            )
        )

    raw_dataset_sha256 = resolve_raw_dataset_sha256(layout, freeze)
    final = aggregate(
        ss,
        cfg,
        out,
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
    print("\nFixed-identity Stage 9 complete.")
    print(json.dumps(final["metrics_by_k"], indent=2))
    print(
        "Summary:",
        out / layout.artifact_name("stage9_strength_summary"),
    )


if __name__ == "__main__":
    main()
