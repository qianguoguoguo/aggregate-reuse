#!/usr/bin/env python3
"""Assign fixed-attack synthetic identities and compute reuse attribution.

Phase 2J consumes Phase-2I attack worlds from:
  ../amazon_preprocess/stage4_attack/

It does NOT select items, blocks, donor slots, replacement ratings, or d_cf.
Those fields are frozen and hash-verified before identity assignment.

Outputs:
  ../amazon_preprocess/stage4_reuse/
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
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

from aggregate_reuse.amazon.primary_attack import (
    canonical_attack_hash,
    load_experimental_blocks,
)
from aggregate_reuse.amazon.reuse import regular_incidence, summarize_accounts
from aggregate_reuse.amazon.paths import AmazonPathLayout
from aggregate_reuse.amazon.provenance import (
    build_provenance,
    require_artifact_category,
    resolve_raw_dataset_sha256,
)


def read_attack_world(path: Path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "asin": r["asin"],
                "treatment_block": r["treatment_block"],
                "treated_positions": json.loads(r["treated_positions"]),
                "treated_source_lines": json.loads(r["treated_source_lines"]),
                "original_ratings": json.loads(r["original_ratings"]),
                "replacement_ratings": json.loads(r["replacement_ratings"]),
                "clean_counts": json.loads(r["clean_counts"]),
                "attack_counts": json.loads(r["attack_counts"]),
                "clean_w1": float(r["clean_w1"]),
                "attack_w1": float(r["attack_w1"]),
                "d_cf": float(r["d_cf"]),
            })
    return rows


def write_csv_gz(path, rows, fieldnames):
    path = Path(path)
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
        "freeze": layout.resolve_shared_path(shared["reference_freeze"]),
        "attack": layout.resolve_shared_path(shared["output_subdir"]),
        "reuse": layout.resolve_shared_path(shared["reuse_output_subdir"]),
    }


def participant_blocks_from_frozen_attack(attack_rows, blocks):
    """Recover original non-donor participants from the frozen treated block."""
    out = {}
    for row in attack_rows:
        asin = row["asin"]
        block_name = row["treatment_block"]
        if asin not in blocks:
            raise RuntimeError(f"Missing Stage-2 block for {asin}")

        clean_reviews = blocks[asin][block_name]
        donor_source_lines = set(int(x) for x in row["treated_source_lines"])

        # Verify the frozen donor slots are exactly the corresponding Stage-2 rows.
        stage2_by_source = {
            int(rev["source_line"]): rev for rev in clean_reviews
        }
        for source_line, old_rating, position in zip(
            row["treated_source_lines"],
            row["original_ratings"],
            row["treated_positions"],
        ):
            rev = stage2_by_source.get(int(source_line))
            if rev is None:
                raise RuntimeError(
                    f"{asin}: frozen donor source line missing from treated block"
                )
            if int(rev["rating"]) != int(old_rating):
                raise RuntimeError(f"{asin}: frozen donor rating mismatch")
            if int(rev["position"]) != int(position):
                raise RuntimeError(f"{asin}: frozen donor position mismatch")

        normal_users = [
            str(rev["user_id"])
            for rev in clean_reviews
            if int(rev["source_line"]) not in donor_source_lines
        ]
        if len(normal_users) != 24:
            raise RuntimeError(f"{asin}: expected 24 normal participants")

        out[asin] = {
            "normal_users": normal_users,
            "treated_positions": list(row["treated_positions"]),
            "treated_source_lines": list(row["treated_source_lines"]),
        }
    return out


def run_seed(seed, cfg, blocks, attack_root, reuse_root, overwrite=False):
    m = int(cfg["attack"]["manipulated_slots_per_item"])
    L = int(cfg["attack"]["n_items_per_seed"])
    reuse_grid = [int(x) for x in cfg["reuse"]["grid"]]

    attack_seed_dir = attack_root / f"seed_{seed:03d}"
    attack_path = attack_seed_dir / "attack_world.csv.gz"
    attack_summary_path = attack_seed_dir / "attack_summary.json"

    if not attack_path.exists() or not attack_summary_path.exists():
        raise FileNotFoundError(
            f"Phase-2I attack world missing for seed {seed}: {attack_seed_dir}"
        )

    attack_rows = read_attack_world(attack_path)
    attack_summary = json.loads(
        attack_summary_path.read_text(encoding="utf-8")
    )
    require_artifact_category(
        attack_summary,
        expected=cfg["category"],
        label=f"Stage-4 attack seed {seed}",
    )

    if len(attack_rows) != L:
        raise RuntimeError(f"Seed {seed}: expected {L} frozen attack rows")

    # Critical Phase-2J guard: recompute the attack hash BEFORE any identity work.
    attack_hash = canonical_attack_hash(attack_rows)
    if attack_hash != attack_summary["attack_world_sha256"]:
        raise RuntimeError(f"Seed {seed}: Phase-2I attack hash mismatch")

    participant_blocks = participant_blocks_from_frozen_attack(
        attack_rows, blocks
    )

    by_asin = {x["asin"]: x for x in attack_rows}
    asins = sorted(by_asin)
    d = np.asarray([by_asin[a]["d_cf"] for a in asins], dtype=float)

    seed_dir = reuse_root / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    summary_path = seed_dir / "reuse_summary.json"

    if summary_path.exists() and not overwrite:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        require_artifact_category(
            existing,
            expected=cfg["category"],
            label=f"existing reuse seed {seed}",
        )
        return existing

    reuse_rows = []
    identity_hashes = {}

    for r in reuse_grid:
        rng = np.random.default_rng(seed * 100_003 + r * 997 + 41)
        incidence = regular_incidence(asins, m, r, rng)

        account_score = defaultdict(float)
        account_freq = defaultdict(int)
        edge_rows = []

        for asin in asins:
            d_cf = float(by_asin[asin]["d_cf"])

            # Historical participants not replaced by synthetic identities.
            for user in participant_blocks[asin]["normal_users"]:
                account_score[user] += d_cf
                account_freq[user] += 1

            syn = incidence[asin]
            positions = participant_blocks[asin]["treated_positions"]
            source_lines = participant_blocks[asin]["treated_source_lines"]

            if len(syn) != m:
                raise RuntimeError("Synthetic item degree is not exactly m")

            for k in range(m):
                a = syn[k]
                account_score[a] += d_cf
                account_freq[a] += 1
                edge_rows.append({
                    "asin": asin,
                    "treatment_block": by_asin[asin]["treatment_block"],
                    "treated_position": positions[k],
                    "treated_source_line": source_lines[k],
                    "synthetic_account_id": a,
                    "d_cf": d_cf,
                })

        metrics = summarize_accounts(account_score, account_freq)
        c_scores = metrics["coalition_scores"]
        n_scores = metrics["normal_scores"]
        c_freq = metrics["coalition_freq"]
        n_freq = metrics["normal_freq"]

        M_expected = L * m // r
        if len(c_scores) != M_expected:
            raise RuntimeError(
                f"r={r}: coalition account count {len(c_scores)} != {M_expected}"
            )
        if not np.all(c_freq == r):
            raise RuntimeError(f"r={r}: coalition row degree is not exactly r")

        predicted_mean = r * float(d.mean())
        observed_mean = float(c_scores.mean())
        law_abs_error = abs(observed_mean - predicted_mean)

        total_expected = m * float(d.sum())
        total_observed = float(c_scores.sum())
        total_error = abs(total_observed - total_expected)
        tol = 1e-10 * max(1.0, abs(total_expected))
        if total_error > tol:
            raise RuntimeError(f"r={r}: score conservation failed")

        hh = hashlib.sha256()
        for e in edge_rows:
            hh.update(
                json.dumps(
                    e, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            )
            hh.update(b"\n")
        identity_hash = hh.hexdigest()
        identity_hashes[str(r)] = identity_hash

        write_csv_gz(
            seed_dir / f"identity_assignment_r{r}.csv.gz",
            edge_rows,
            list(edge_rows[0].keys()),
        )

        coalition_set = set(metrics["coalition_accounts"])
        account_rows = [
            {
                "account_id": a,
                "is_synthetic_coalition": int(a in coalition_set),
                "score": account_score[a],
                "frequency": account_freq[a],
            }
            for a in sorted(account_score)
        ]
        write_csv_gz(
            seed_dir / f"account_metrics_r{r}.csv.gz",
            account_rows,
            ["account_id", "is_synthetic_coalition", "score", "frequency"],
        )

        reuse_rows.append({
            "seed": seed,
            "reuse_r": r,
            "attack_world_sha256": attack_hash,
            "identity_assignment_sha256": identity_hash,
            "n_items": L,
            "m": m,
            "coalition_accounts": len(c_scores),
            "normal_accounts": len(n_scores),
            "mean_d_cf": float(d.mean()),
            "positive_d_cf_fraction": float((d > 0).mean()),
            "zero_d_cf_fraction": float((d == 0).mean()),
            "negative_d_cf_fraction": float((d < 0).mean()),
            "clean_w1_mean": float(
                np.mean([x["clean_w1"] for x in attack_rows])
            ),
            "attack_w1_mean": float(
                np.mean([x["attack_w1"] for x in attack_rows])
            ),
            "score_auc": metrics["score_auc"],
            "frequency_auc": metrics["frequency_auc"],
            "score_no_larger_probability":
                metrics["score_no_larger_probability"],
            "coalition_mean_score": observed_mean,
            "predicted_coalition_mean_score_r_times_mean_dcf":
                predicted_mean,
            "mean_reuse_law_abs_error": law_abs_error,
            "coalition_score_total": total_observed,
            "expected_coalition_score_total_m_times_sum_dcf":
                total_expected,
            "score_conservation_abs_error": total_error,
            "coalition_frequency_mean": float(c_freq.mean()),
            "normal_frequency_mean": float(n_freq.mean()),
        })

    # Strong fixed-attack guard across all reuse conditions.
    if len({x["attack_world_sha256"] for x in reuse_rows}) != 1:
        raise RuntimeError("Attack world changed across reuse values")

    with (seed_dir / "reuse_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(reuse_rows[0].keys()))
        w.writeheader()
        w.writerows(reuse_rows)

    summary = {
        "experiment": "amazon_primary_fixed_attack_reuse",
        "category": cfg["category"],
        "seed": seed,
        "attack_world_sha256": attack_hash,
        "identity_assignment_sha256_by_reuse": identity_hashes,
        "sampled_items": L,
        "m": m,
        "reuse_grid": reuse_grid,
        "invariants": {
            "phase2I_attack_hash_verified_before_identity_assignment": True,
            "same_attack_world_across_reuse": True,
            "coalition_row_degree_exactly_r": True,
            "item_column_degree_exactly_m": True,
            "no_duplicate_coalition_account_within_item": True,
            "score_conservation_verified": True,
        },
        "reuse_metrics": reuse_rows,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def aggregate_all(
    summaries,
    out_root,
    *,
    layout,
    config_path,
    raw_dataset_sha256,
    selected_lambda,
    upstream_artifacts,
):
    rows = []
    for s in summaries:
        rows.extend(s["reuse_metrics"])

    with (out_root / layout.artifact_name("stage4_reuse_seed_metrics")).open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    reuse_grid = sorted({int(x["reuse_r"]) for x in rows})
    aggregate = {}
    metrics_to_summarize = [
        "mean_d_cf",
        "positive_d_cf_fraction",
        "score_auc",
        "frequency_auc",
        "score_no_larger_probability",
        "coalition_mean_score",
        "mean_reuse_law_abs_error",
    ]

    for r in reuse_grid:
        rr = [x for x in rows if int(x["reuse_r"]) == r]
        out = {"n_seeds": len(rr)}
        for metric in metrics_to_summarize:
            vals = np.asarray([float(x[metric]) for x in rr], dtype=float)
            mean = float(vals.mean())
            se = float(vals.std(ddof=1) / math.sqrt(len(vals)))
            crit = float(t.ppf(0.975, df=len(vals)-1))
            out[metric] = {
                "mean": mean,
                "ci_lower": float(mean - crit*se),
                "ci_upper": float(mean + crit*se),
            }
        aggregate[str(r)] = out

    seed_visibility = np.asarray(
        [s["reuse_metrics"][0]["mean_d_cf"] for s in summaries],
        dtype=float,
    )
    vis_mean = float(seed_visibility.mean())
    se = float(seed_visibility.std(ddof=1) / math.sqrt(len(seed_visibility)))
    crit = float(t.ppf(0.975, df=len(seed_visibility)-1))

    global_summary = {
        "stage": "amazon_stage4_fixed_five_star_reuse",
        "category": layout.category,
        "raw_dataset_sha256": raw_dataset_sha256,
        "selected_lambda": float(selected_lambda),
        "n_seeds": len(summaries),
        "seed_ids": [int(s["seed"]) for s in summaries],
        "aggregate_visibility_mean_d_cf": {
            "mean": vis_mean,
            "ci_lower": float(vis_mean - crit*se),
            "ci_upper": float(vis_mean + crit*se),
        },
        "reuse_summary": aggregate,
        "interpretation_guard": (
            "Stage 4 tests a fixed paired aggregate five-star intervention. "
            "Across r, aggregate attack construction is unchanged; only the "
            "partition of manipulated slots into synthetic identities changes. "
            "This first reuse sweep does not match coalition and normal marginal "
            "activity; account-marginal matching is a later experiment."
        ),
    }
    global_summary["provenance"] = build_provenance(
        category=layout.category,
        config_path=config_path,
        raw_dataset_sha256=raw_dataset_sha256,
        selected_lambda=selected_lambda,
        seed_ids=global_summary["seed_ids"],
        upstream_artifacts=upstream_artifacts,
        repo_root=ROOT,
        shared_root=layout.shared_category_root,
    )
    (out_root / layout.artifact_name("stage4_reuse_summary")).write_text(
        json.dumps(global_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return global_summary


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
    ap.add_argument("--seed", type=int)
    ap.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_cfg(config_path)
    paths = shared_paths(cfg, shared_root=args.shared_root)

    if not paths["roles"].exists():
        raise FileNotFoundError(paths["roles"])
    if not paths["attack"].exists():
        raise FileNotFoundError(paths["attack"])
    if not paths["freeze"].exists():
        raise FileNotFoundError(paths["freeze"])

    freeze = json.loads(paths["freeze"].read_text(encoding="utf-8"))
    require_artifact_category(
        freeze,
        expected=cfg["category"],
        label="reference freeze",
        allow_legacy_home_missing=False,
    )
    raw_dataset_sha256 = resolve_raw_dataset_sha256(paths["layout"], freeze)

    blocks = load_experimental_blocks(paths["roles"])
    paths["reuse"].mkdir(parents=True, exist_ok=True)

    seeds = [int(x) for x in cfg["seeds"]]
    if args.seed is not None:
        if args.seed not in seeds:
            raise ValueError("Seed outside frozen list")
        summary = run_seed(
            args.seed, cfg, blocks, paths["attack"], paths["reuse"],
            overwrite=args.overwrite,
        )
        print(json.dumps(summary, indent=2))
        return 0

    summaries = []
    for seed in seeds:
        print(f"=== Fixed-attack reuse seed {seed} ===")
        summaries.append(
            run_seed(
                seed, cfg, blocks, paths["attack"], paths["reuse"],
                overwrite=args.overwrite,
            )
        )

    upstream_artifacts = {
        "stage2_roles": paths["roles"],
        "stage3_reference_freeze": paths["freeze"],
    }
    attack_aggregate = paths["attack"] / paths["layout"].artifact_name(
        "stage4_attack_summary"
    )
    if attack_aggregate.is_file():
        upstream_artifacts["stage4_attack_summary"] = attack_aggregate
    global_summary = aggregate_all(
        summaries,
        paths["reuse"],
        layout=paths["layout"],
        config_path=config_path,
        raw_dataset_sha256=raw_dataset_sha256,
        selected_lambda=freeze["selected_lambda"],
        upstream_artifacts=upstream_artifacts,
    )
    print("Fixed-attack reuse experiment complete.")
    print(json.dumps(global_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
