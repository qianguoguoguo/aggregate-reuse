#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.io import write_csv_gz
from aggregate_reuse.amazon.references import (
    exact_predictive_w1_baseline,
    load_reference_table,
    shrunk_reference,
)
from aggregate_reuse.amazon.twins import (
    auc_rank,
    load_attack_world,
    load_r8_assignment,
    paired_no_larger_probability,
)
from aggregate_reuse.amazon.paths import AmazonPathLayout
from aggregate_reuse.amazon.provenance import (
    build_provenance,
    require_artifact_category,
    resolve_raw_dataset_sha256,
)



def twin_id(coalition_id: str) -> str:
    if not coalition_id.startswith("syn_coal_r8_"):
        raise RuntimeError(f"Unexpected r=8 coalition ID: {coalition_id}")
    return coalition_id.replace("syn_coal_r8_", "syn_ctrl_r8_", 1)


def run_seed(
    *,
    seed: int,
    cfg,
    refs,
    freeze,
    attack_root: Path,
    reuse_root: Path,
    out_root: Path,
    overwrite: bool,
):
    r = int(cfg["reuse_r"])
    if r != 8:
        raise RuntimeError("Stage 5 is frozen to r=8.")

    attack_seed = attack_root / f"seed_{seed:03d}"
    reuse_seed = reuse_root / f"seed_{seed:03d}"

    attack_summary_path = attack_seed / "attack_summary.json"
    attack_path = attack_seed / "attack_world.csv.gz"
    reuse_summary_path = reuse_seed / "reuse_summary.json"
    assignment_path = reuse_seed / "identity_assignment_r8.csv.gz"

    for p in [
        attack_summary_path,
        attack_path,
        reuse_summary_path,
        assignment_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    out_seed = out_root / f"seed_{seed:03d}"
    out_seed.mkdir(parents=True, exist_ok=True)
    summary_path = out_seed / "summary.json"
    if summary_path.exists() and not overwrite:
        print(f"Seed {seed}: already complete, skipping.")
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        require_artifact_category(
            existing,
            expected=cfg["category"],
            label=f"existing Stage-5 seed {seed}",
        )
        return existing

    attack_summary = json.loads(
        attack_summary_path.read_text(encoding="utf-8")
    )
    reuse_summary = json.loads(
        reuse_summary_path.read_text(encoding="utf-8")
    )
    require_artifact_category(
        attack_summary, expected=cfg["category"], label="Stage-4 attack summary"
    )
    require_artifact_category(
        reuse_summary, expected=cfg["category"], label="Stage-4 reuse summary"
    )

    if attack_summary["experiment"] != "amazon_primary_attack":
        raise RuntimeError("Unexpected Phase-2I attack summary type.")
    if int(attack_summary["selected_lambda"]) != int(freeze["selected_lambda"]):
        raise RuntimeError("Phase-2I lambda differs from Stage-3 freeze.")
    if not all(bool(v) for v in attack_summary["invariants"].values()):
        raise RuntimeError("Phase-2I attack invariants did not all pass.")
    if not all(bool(v) for v in reuse_summary["invariants"].values()):
        raise RuntimeError("Phase-2J reuse invariants did not all pass.")

    attack = load_attack_world(attack_path)
    edges = load_r8_assignment(assignment_path)

    # Exact Phase-2I/2J handoff.
    expected_attack_hash = attack_summary["attack_world_sha256"]
    if reuse_summary["attack_world_sha256"] != expected_attack_hash:
        raise RuntimeError("Phase-2J attack hash differs from Phase-2I.")
    expected_identity_hash = (
        reuse_summary["identity_assignment_sha256_by_reuse"]["8"]
    )

    import hashlib
    hh = hashlib.sha256()
    for e in edges:
        payload = {
            "asin": e["asin"],
            "treatment_block": e["treatment_block"],
            "treated_position": int(e["treated_position"]),
            "treated_source_line": int(e["treated_source_line"]),
            "synthetic_account_id": e["synthetic_account_id"],
            "d_cf": float(e["d_cf"]),
        }
        hh.update(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        hh.update(b"\n")
    if hh.hexdigest() != expected_identity_hash:
        raise RuntimeError("Phase-2J r=8 identity-assignment hash mismatch.")

    # Map each attack slot to its aggregate record and validate every edge.
    slot_lookup = {}
    for asin, row in attack.items():
        positions = row["treated_positions"]
        source_lines = row["treated_source_lines"]
        if len(positions) != len(source_lines):
            raise RuntimeError(f"{asin}: malformed attack slot lists.")
        for pos, src in zip(positions, source_lines):
            key = (asin, row["treatment_block"], int(pos), int(src))
            if key in slot_lookup:
                raise RuntimeError(f"Duplicate attack slot: {key}")
            slot_lookup[key] = row

    if len(edges) != len(slot_lookup):
        raise RuntimeError(
            f"r=8 assignment has {len(edges)} edges, expected {len(slot_lookup)}."
        )

    seen_slots = set()
    account_edges = defaultdict(list)

    for e in edges:
        key = (
            e["asin"],
            e["treatment_block"],
            int(e["treated_position"]),
            int(e["treated_source_line"]),
        )
        if key not in slot_lookup:
            raise RuntimeError(f"Identity edge not found in attack world: {key}")
        if key in seen_slots:
            raise RuntimeError(f"Duplicate identity assignment for slot: {key}")
        seen_slots.add(key)

        row = slot_lookup[key]
        if abs(float(e["d_cf"]) - float(row["d_cf"])) > 1e-12:
            raise RuntimeError("d_cf differs between attack world and identity edge.")
        account_edges[e["synthetic_account_id"]].append((key, row))

    if seen_slots != set(slot_lookup):
        raise RuntimeError("Not every manipulated slot is assigned at r=8.")

    coalition_ids = sorted(account_edges)
    if not coalition_ids:
        raise RuntimeError("No r=8 coalition accounts found.")

    # Every r=8 account must have eight distinct item/block exposures.
    for a in coalition_ids:
        lst = account_edges[a]
        if len(lst) != r:
            raise RuntimeError(f"{a}: expected degree {r}, got {len(lst)}")
        item_block = [(k[0], k[1]) for k, _ in lst]
        if len(item_block) != len(set(item_block)):
            raise RuntimeError(f"{a}: duplicate item/block exposure.")

    lam = float(freeze["selected_lambda"])

    # Predictive finite-sample baseline per item, included only to show that
    # matched clean/attack score differences equal the counterfactual score.
    pred_baseline = {}
    for asin in attack:
        ref_counts, loo_probs = refs[asin]
        alpha, _ = shrunk_reference(ref_counts, loo_probs, lam)
        pred_baseline[asin] = exact_predictive_w1_baseline(alpha, 30)

    account_rows = []
    coalition_cf = []
    control_cf = []
    coalition_raw = []
    control_raw = []
    coalition_pred = []
    control_pred = []
    coalition_freq = []
    control_freq = []
    paired_gaps = []

    for coalition_id in coalition_ids:
        ctrl = twin_id(coalition_id)
        exposures = account_edges[coalition_id]

        cf_attack = 0.0
        raw_attack = 0.0
        raw_clean = 0.0
        centered_attack = 0.0
        centered_clean = 0.0

        item_set = []
        slot_set = []

        for key, row in exposures:
            asin = key[0]
            b = pred_baseline[asin]
            d = float(row["d_cf"])

            cf_attack += d
            raw_attack += float(row["attack_w1"])
            raw_clean += float(row["clean_w1"])
            centered_attack += float(row["attack_w1"]) - b
            centered_clean += float(row["clean_w1"]) - b
            item_set.append(asin)
            slot_set.append(
                f"{asin}|{key[1]}|{key[2]}|{key[3]}"
            )

        cf_control = 0.0
        gap = raw_attack - raw_clean

        # The same matched item set means the predictive baselines cancel.
        if abs((centered_attack - centered_clean) - gap) > 1e-10:
            raise RuntimeError("Predictive-baseline cancellation invariant failed.")
        if abs(gap - cf_attack) > 1e-10:
            raise RuntimeError("Matched raw-world gap does not equal sum d_cf.")

        coalition_cf.append(cf_attack)
        control_cf.append(cf_control)
        coalition_raw.append(raw_attack)
        control_raw.append(raw_clean)
        coalition_pred.append(centered_attack)
        control_pred.append(centered_clean)
        coalition_freq.append(r)
        control_freq.append(r)
        paired_gaps.append(gap)

        account_rows.append({
            "pair_id": coalition_id,
            "coalition_account_id": coalition_id,
            "control_account_id": ctrl,
            "frequency_each": r,
            "item_set_json": json.dumps(sorted(item_set)),
            "slot_set_json": json.dumps(sorted(slot_set)),
            "coalition_counterfactual_score": cf_attack,
            "control_counterfactual_score": cf_control,
            "coalition_raw_world_score": raw_attack,
            "control_raw_world_score": raw_clean,
            "coalition_predictive_centered_world_score": centered_attack,
            "control_predictive_centered_world_score": centered_clean,
            "paired_score_gap": gap,
        })

    coalition_cf = np.asarray(coalition_cf, dtype=float)
    control_cf = np.asarray(control_cf, dtype=float)
    coalition_raw = np.asarray(coalition_raw, dtype=float)
    control_raw = np.asarray(control_raw, dtype=float)
    coalition_pred = np.asarray(coalition_pred, dtype=float)
    control_pred = np.asarray(control_pred, dtype=float)
    coalition_freq = np.asarray(coalition_freq, dtype=float)
    control_freq = np.asarray(control_freq, dtype=float)
    paired_gaps = np.asarray(paired_gaps, dtype=float)

    # Exact matching checks.
    if not np.array_equal(coalition_freq, control_freq):
        raise RuntimeError("Frequency matching failed.")
    frequency_auc = auc_rank(coalition_freq, control_freq)
    if abs(frequency_auc - 0.5) > 1e-12:
        raise RuntimeError(f"Frequency AUC should be exactly 0.5, got {frequency_auc}")

    mean_d_cf = float(np.mean([row["d_cf"] for row in attack.values()]))
    predicted_mean_gap = r * mean_d_cf
    observed_mean_gap = float(paired_gaps.mean())
    reuse_law_error = abs(observed_mean_gap - predicted_mean_gap)

    # Total gap conservation: each block d_cf appears in six manipulated slots.
    m = int(attack_summary["m"])
    total_expected = m * float(np.sum([row["d_cf"] for row in attack.values()]))
    total_observed = float(paired_gaps.sum())
    total_error = abs(total_observed - total_expected)
    tol = 1e-10 * max(1.0, abs(total_expected))
    if total_error > tol:
        raise RuntimeError("Stage-5 score-gap conservation failed.")

    write_csv_gz(
        out_seed / "matched_twin_accounts.csv.gz",
        account_rows,
        list(account_rows[0].keys()),
    )

    # Compact edge-level twin manifest: same slots, clean vs attacked world.
    twin_edge_rows = []
    for coalition_id in coalition_ids:
        ctrl = twin_id(coalition_id)
        for key, row in account_edges[coalition_id]:
            twin_edge_rows.append({
                "pair_id": coalition_id,
                "coalition_account_id": coalition_id,
                "control_account_id": ctrl,
                "asin": key[0],
                "block": key[1],
                "position": key[2],
                "source_line": key[3],
                "clean_rating": row["original_ratings"][
                    row["treated_source_lines"].index(key[3])
                ],
                "attack_rating": 5,
                "clean_w1": row["clean_w1"],
                "attack_w1": row["attack_w1"],
                "d_cf": row["d_cf"],
            })
    write_csv_gz(
        out_seed / "matched_twin_edges.csv.gz",
        twin_edge_rows,
        list(twin_edge_rows[0].keys()),
    )

    summary = {
        "stage": "amazon_stage5_exact_matched_twins_r8",
        "category": cfg["category"],
        "seed": seed,
        "reuse_r": r,
        "selected_lambda": lam,
        "stage4_attack_world_sha256": expected_attack_hash,
        "n_pairs": len(coalition_ids),
        "n_edges_per_group": len(edges),
        "matching": {
            "frequency_exact": True,
            "item_set_pairwise_exact": True,
            "block_pairwise_exact": True,
            "slot_position_pairwise_exact": True,
            "source_line_pairwise_exact": True,
            "timestamp_pairwise_exact_via_same_source_slot": True,
            "only_difference": (
                "coalition member occupies attacked version of the slot; "
                "control twin occupies exact clean counterpart"
            ),
        },
        "metrics": {
            "frequency_auc": frequency_auc,
            "counterfactual_score_auc": auc_rank(coalition_cf, control_cf),
            "raw_world_score_auc": auc_rank(coalition_raw, control_raw),
            "predictive_centered_world_score_auc": auc_rank(
                coalition_pred, control_pred
            ),
            "paired_raw_misordering_probability": paired_no_larger_probability(
                coalition_raw, control_raw
            ),
            "paired_predictive_misordering_probability": paired_no_larger_probability(
                coalition_pred, control_pred
            ),
            "paired_gap_nonpositive_fraction": float(np.mean(paired_gaps <= 0)),
            "paired_gap_positive_fraction": float(np.mean(paired_gaps > 0)),
            "mean_paired_score_gap": observed_mean_gap,
            "predicted_mean_gap_r_times_mean_dcf": predicted_mean_gap,
            "reuse_law_abs_error": reuse_law_error,
            "total_paired_gap": total_observed,
            "expected_total_m_times_sum_dcf": total_expected,
            "total_gap_conservation_abs_error": total_error,
            "mean_d_cf": mean_d_cf,
        },
        "interpretation_guard": (
            "This is an exact counterfactual matched-twin experiment. It does "
            "not claim that the synthetic control identities are historical "
            "Amazon accounts. Their underlying clean actions, item contexts, "
            "positions, timestamps, and ratings are real; only identity labels "
            "are regrouped to create exact matched activity."
        ),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def summarize_metric(vals):
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
    out_root: Path,
    *,
    layout: AmazonPathLayout,
    config_path: Path,
    raw_dataset_sha256: str,
    freeze,
    upstream_artifacts,
):
    metric_names = list(summaries[0]["metrics"].keys())
    out_metrics = {}
    for name in metric_names:
        vals = [float(s["metrics"][name]) for s in summaries]
        out_metrics[name] = summarize_metric(vals)

    final = {
        "stage": "amazon_stage5_exact_matched_twins_r8",
        "category": layout.category,
        "raw_dataset_sha256": raw_dataset_sha256,
        "selected_lambda": float(freeze["selected_lambda"]),
        "n_seeds": len(summaries),
        "seed_ids": [int(s["seed"]) for s in summaries],
        "reuse_r": 8,
        "matching": summaries[0]["matching"],
        "metrics": out_metrics,
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
    (out_root / layout.artifact_name("stage5_twins_summary")).write_text(
        json.dumps(final, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(ROOT / "configs" / "amazon_twins.yaml"),
    )
    ap.add_argument(
        "--shared-root",
        type=Path,
        help="Operational shared-root override; does not alter the config file.",
    )
    ap.add_argument(
        "--attack-dir",
        default=None,
    )
    ap.add_argument(
        "--reuse-dir",
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
    attack_root = (
        Path(args.attack_dir).resolve() if args.attack_dir
        else layout.resolve_shared_path(shared["attack_dir"])
    )
    reuse_root = (
        Path(args.reuse_dir).resolve() if args.reuse_dir
        else layout.resolve_shared_path(shared["reuse_dir"])
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
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))

    require_artifact_category(
        freeze,
        expected=cfg["category"],
        label="Stage-3 freeze",
        allow_legacy_home_missing=False,
    )
    if freeze["stage4_evidence_mode"] != "paired_counterfactual":
        raise RuntimeError("Stage-3 freeze evidence mode mismatch.")

    refs = load_reference_table(references_path)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        if args.seed not in cfg["seeds"]:
            raise ValueError("Requested seed is not frozen in config.")
        s = run_seed(
            seed=int(args.seed),
            cfg=cfg,
            refs=refs,
            freeze=freeze,
            attack_root=attack_root,
            reuse_root=reuse_root,
            out_root=out_root,
            overwrite=args.overwrite,
        )
        print(json.dumps(s["metrics"], indent=2))
        print(f"Summary: {out_root / f'seed_{args.seed:03d}' / 'summary.json'}")
        return

    summaries = []
    for seed in cfg["seeds"]:
        print(f"\n=== Stage 5 seed {seed} ===")
        summaries.append(
            run_seed(
                seed=int(seed),
                cfg=cfg,
                refs=refs,
                freeze=freeze,
                attack_root=attack_root,
                reuse_root=reuse_root,
                out_root=out_root,
                overwrite=args.overwrite,
            )
        )

    upstream_artifacts = {
        "reference_histograms": references_path,
        "reference_freeze": freeze_path,
    }
    for label, path in {
        "stage4_attack_summary": attack_root / layout.artifact_name(
            "stage4_attack_summary"
        ),
        "stage4_reuse_summary": reuse_root / layout.artifact_name(
            "stage4_reuse_summary"
        ),
    }.items():
        if path.is_file():
            upstream_artifacts[label] = path
    raw_dataset_sha256 = resolve_raw_dataset_sha256(layout, freeze)
    final = aggregate(
        summaries,
        out_root,
        layout=layout,
        config_path=config_path,
        raw_dataset_sha256=raw_dataset_sha256,
        freeze=freeze,
        upstream_artifacts=upstream_artifacts,
    )
    print("\nStage 5 complete.")
    for name, stat in final["metrics"].items():
        print(
            f"{name}: {stat['mean']:.6f} "
            f"[{stat['ci_lower']:.6f}, {stat['ci_upper']:.6f}]"
        )
    print(f"Summary: {out_root / layout.artifact_name('stage5_twins_summary')}")


if __name__ == "__main__":
    main()
