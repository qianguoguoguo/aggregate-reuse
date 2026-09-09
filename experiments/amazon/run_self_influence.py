#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, gzip, json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import pearsonr, rankdata, spearmanr, t

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.references import load_reference_table, shrunk_reference
from aggregate_reuse.amazon.paths import AmazonPathLayout
from aggregate_reuse.amazon.provenance import (
    build_provenance,
    require_artifact_category,
    resolve_raw_dataset_sha256,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_attack_world(path):
    out = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            asin = str(row["asin"])
            treated_positions = [int(x) for x in json.loads(row["treated_positions"])]
            treated_source_lines = [int(x) for x in json.loads(row["treated_source_lines"])]
            original_ratings = [int(x) for x in json.loads(row["original_ratings"])]
            replacement_ratings = [int(x) for x in json.loads(row["replacement_ratings"])]
            if not (
                len(treated_positions)
                == len(treated_source_lines)
                == len(original_ratings)
                == len(replacement_ratings)
            ):
                raise RuntimeError(f"{asin}: inconsistent attack-manifest slot lengths.")
            slot_ratings = {
                (p, s): (ro, rr)
                for p, s, ro, rr in zip(
                    treated_positions,
                    treated_source_lines,
                    original_ratings,
                    replacement_ratings,
                )
            }
            out[asin] = {
                "asin": asin,
                "treatment_block": row["treatment_block"],
                "treated_positions": treated_positions,
                "treated_source_lines": treated_source_lines,
                "original_ratings": original_ratings,
                "replacement_ratings": replacement_ratings,
                "slot_ratings": slot_ratings,
                "clean_counts": np.asarray(json.loads(row["clean_counts"]), dtype=int),
                "attack_counts": np.asarray(json.loads(row["attack_counts"]), dtype=int),
                "d_cf": float(row["d_cf"]),
            }
    return out


def load_r8_assignment(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            rows.append({
                "synthetic_account_id": str(row["synthetic_account_id"]),
                "asin": str(row["asin"]),
                "treatment_block": row["treatment_block"],
                "treated_position": int(row["treated_position"]),
                "treated_source_line": int(row["treated_source_line"]),
                "d_cf": float(row["d_cf"]),
            })
    return rows


def load_selected_blocks(roles_path, attack):
    wanted = set(attack)
    blocks = {}
    with gzip.open(roles_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            asin = str(obj["asin"])
            if asin not in wanted:
                continue
            role = attack[asin]["treatment_block"]
            block = obj["blocks"][role]
            if len(block) != 30:
                raise RuntimeError(f"{asin}/{role}: expected 30 reviews.")
            blocks[asin] = block
            if len(blocks) == len(wanted):
                break
    if set(blocks) != wanted:
        raise RuntimeError("Missing selected blocks from roles file.")
    return blocks


def w1_counts(counts, q):
    p = np.asarray(counts, dtype=float)
    p /= p.sum()
    q = np.asarray(q, dtype=float)
    return float(np.abs(np.cumsum(p - q)[:-1]).sum())


def auc_rank(pos, neg):
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    vals = np.concatenate([pos, neg])
    ranks = rankdata(vals, method="average")
    n1, n0 = len(pos), len(neg)
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0))


def mean_t_ci(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": float("nan"), "ci_lower": float("nan"),
                "ci_upper": float("nan"), "n": 0}
    m = float(x.mean())
    if len(x) == 1:
        return {"mean": m, "ci_lower": m, "ci_upper": m, "n": 1}
    se = float(x.std(ddof=1) / math.sqrt(len(x)))
    crit = float(t.ppf(0.975, df=len(x)-1))
    return {"mean": m, "ci_lower": m-crit*se,
            "ci_upper": m+crit*se, "n": int(len(x))}


def loo_dcf(clean_counts, attack_counts, original_rating, replacement_rating, q):
    c = np.asarray(clean_counts, int).copy()
    a = np.asarray(attack_counts, int).copy()
    c[int(original_rating)-1] -= 1
    a[int(replacement_rating)-1] -= 1
    if c.sum() != 29 or a.sum() != 29 or np.any(c < 0) or np.any(a < 0):
        raise RuntimeError("Invalid 29-review LOO histogram.")
    cw = w1_counts(c, q)
    aw = w1_counts(a, q)
    return float(aw-cw), float(cw), float(aw)


def run_seed(seed, cfg, attack_root, reuse_root, roles_path, refs_path, freeze_path, out_root, overwrite):
    attack_seed_dir = Path(attack_root) / f"seed_{seed:03d}"
    reuse_seed_dir = Path(reuse_root) / f"seed_{seed:03d}"
    attack_path = attack_seed_dir / "attack_world.csv.gz"
    assign_path = reuse_seed_dir / "identity_assignment_r8.csv.gz"
    s4_path = attack_seed_dir / "attack_summary.json"
    reuse_summary_path = reuse_seed_dir / "reuse_summary.json"

    out_seed = Path(out_root) / f"seed_{seed:03d}"
    out_seed.mkdir(parents=True, exist_ok=True)
    summary_path = out_seed / "summary.json"
    if summary_path.exists() and not overwrite:
        existing = read_json(summary_path)
        require_artifact_category(
            existing,
            expected=cfg["category"],
            label=f"existing Stage-11 seed {seed}",
        )
        return existing

    s4 = read_json(s4_path)
    reuse_summary = read_json(reuse_summary_path)
    require_artifact_category(
        s4, expected=cfg["category"], label="Stage-4 attack summary"
    )
    require_artifact_category(
        reuse_summary, expected=cfg["category"], label="Stage-4 reuse summary"
    )
    if s4.get("experiment") != "amazon_primary_attack":
        raise RuntimeError("Unexpected Phase-2I attack summary type.")
    if int(s4["m"]) != int(cfg["primary_modified_ratings_k"]):
        raise RuntimeError("Phase-2I intervention strength mismatch.")
    if not all(bool(v) for v in s4["invariants"].values()):
        raise RuntimeError("Phase-2I attack invariants failed.")
    if not all(bool(v) for v in reuse_summary["invariants"].values()):
        raise RuntimeError("Phase-2J reuse invariants failed.")
    if reuse_summary["attack_world_sha256"] != s4["attack_world_sha256"]:
        raise RuntimeError("Phase-2J attack hash differs from Phase-2I.")
    if "8" not in reuse_summary["identity_assignment_sha256_by_reuse"]:
        raise RuntimeError("Phase-2J r=8 identity assignment missing.")

    attack = load_attack_world(attack_path)
    assignment = load_r8_assignment(assign_path)
    blocks = load_selected_blocks(roles_path, attack)

    refs = load_reference_table(Path(refs_path))
    freeze = read_json(freeze_path)
    require_artifact_category(
        freeze,
        expected=cfg["category"],
        label="Stage-3 freeze",
        allow_legacy_home_missing=False,
    )
    lam = float(freeze["selected_lambda"])

    q_by_asin = {}
    for asin, (c, loo) in refs.items():
        _, q = shrunk_reference(c, loo, lam)
        q_by_asin[asin] = q

    source_index = {}
    for asin, block in blocks.items():
        source_index[asin] = {
            (int(rev["position"]), int(rev["source_line"])): rev
            for rev in block
        }

    orig_score = defaultdict(float)
    loo_score = defaultdict(float)
    freq = defaultdict(int)
    exposure_rows = []

    for row in assignment:
        acc = row["synthetic_account_id"]
        asin = row["asin"]
        block_name = row["treatment_block"]
        if attack[asin]["treatment_block"] != block_name:
            raise RuntimeError("Treatment block mismatch.")

        key = (row["treated_position"], row["treated_source_line"])
        rev = source_index[asin].get(key)
        if rev is None:
            raise RuntimeError("Assigned source record not found.")

        if key not in attack[asin]["slot_ratings"]:
            raise RuntimeError(f"{asin}: assigned r=8 slot absent from attack manifest.")
        original_rating, replacement_rating = attack[asin]["slot_ratings"][key]

        if int(rev["rating"]) != int(original_rating):
            raise RuntimeError("Original rating mismatch between roles and attack manifest.")

        dloo, cw, aw = loo_dcf(
            attack[asin]["clean_counts"],
            attack[asin]["attack_counts"],
            original_rating,
            replacement_rating,
            q_by_asin[asin],
        )
        dorig = float(attack[asin]["d_cf"])
        if abs(dorig - row["d_cf"]) > 1e-12:
            raise RuntimeError("Stage-4 d_cf mismatch.")

        orig_score[acc] += dorig
        loo_score[acc] += dloo
        freq[acc] += 1
        exposure_rows.append({
            "synthetic_account_id": acc,
            "asin": asin,
            "treatment_block": block_name,
            "treated_position": row["treated_position"],
            "treated_source_line": row["treated_source_line"],
            "original_rating": int(original_rating),
            "replacement_rating": int(replacement_rating),
            "original_d_cf": dorig,
            "loo_d_cf": dloo,
            "clean_loo_w1": cw,
            "attack_loo_w1": aw,
        })

    accounts = sorted(orig_score)
    r = int(cfg["reuse_r"])
    if any(freq[a] != r for a in accounts):
        raise RuntimeError("Not every account has exactly r exposures.")

    orig = np.asarray([orig_score[a] for a in accounts], float)
    loo = np.asarray([loo_score[a] for a in accounts], float)
    zeros = np.zeros_like(orig)

    orig_gap = float(orig.mean())
    loo_gap = float(loo.mean())
    retained = float(loo_gap/orig_gap) if abs(orig_gap) > 1e-15 else float("nan")

    metrics = {
        "original_matched_twin_auc": auc_rank(orig, zeros),
        "loo_matched_twin_auc": auc_rank(loo, zeros),
        "original_mean_paired_gap": orig_gap,
        "loo_mean_paired_gap": loo_gap,
        "original_misordering_probability": float(np.mean(orig <= 0)),
        "loo_misordering_probability": float(np.mean(loo <= 0)),
        "retained_mean_gap_fraction": retained,
        "pearson_original_vs_loo_score": float(pearsonr(orig, loo).statistic),
        "spearman_original_vs_loo_score": float(spearmanr(orig, loo).statistic),
        "mean_direct_self_component": float((orig-loo).mean()),
        "median_direct_self_component": float(np.median(orig-loo)),
        "fraction_accounts_loo_score_positive": float(np.mean(loo > 0)),
    }

    with gzip.open(out_seed / "account_scores.csv.gz", "wt",
                   encoding="utf-8", newline="") as f:
        fields = ["synthetic_account_id", "frequency",
                  "original_counterfactual_score",
                  "loo_counterfactual_score", "direct_self_component"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for a in accounts:
            w.writerow({
                "synthetic_account_id": a,
                "frequency": freq[a],
                "original_counterfactual_score": orig_score[a],
                "loo_counterfactual_score": loo_score[a],
                "direct_self_component": orig_score[a]-loo_score[a],
            })

    with gzip.open(out_seed / "exposure_loo.csv.gz", "wt",
                   encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(exposure_rows[0].keys()))
        w.writeheader()
        w.writerows(exposure_rows)

    summary = {
        "stage": "amazon_stage11_leave_one_out_self_influence",
        "category": cfg["category"],
        "seed": seed,
        "source_stage4": {
            "attack_world_sha256": s4["attack_world_sha256"],
            "modified_ratings_k": int(cfg["primary_modified_ratings_k"]),
            "reuse_r": r,
            "selected_lambda": lam,
            "n_items": len(attack),
        },
        "population": {
            "n_synthetic_accounts": len(accounts),
            "n_exposures": len(exposure_rows),
            "exposures_per_account": r,
        },
        "metrics": metrics,
        "invariants": {
            "stage4_attack_world_reused_without_modification": True,
            "stage4_r8_identity_assignment_reused_without_modification": True,
            "same_reference_used_for_original_and_loo": True,
            "scored_review_removed_from_attack_and_clean_worlds": True,
            "loo_histograms_have_size_29": True,
            "all_accounts_have_exactly_r_exposures": True,
        },
        "interpretation_guard": (
            "Stage 11 is an offline diagnostic, not the deployable online score. "
            "Each scored synthetic account's own review is removed from both "
            "attacked and clean worlds before recomputing paired evidence."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True),
                            encoding="utf-8")
    return summary


def aggregate_all(
    seed_summaries,
    out_root,
    *,
    layout,
    config_path,
    raw_dataset_sha256,
    freeze,
    upstream_artifacts,
):
    keys = list(seed_summaries[0]["metrics"].keys())
    metrics = {
        k: mean_t_ci([s["metrics"][k] for s in seed_summaries]) for k in keys
    }
    final = {
        "stage": "amazon_stage11_leave_one_out_self_influence",
        "category": layout.category,
        "raw_dataset_sha256": raw_dataset_sha256,
        "selected_lambda": float(freeze["selected_lambda"]),
        "n_seeds": len(seed_summaries),
        "seed_ids": [s["seed"] for s in seed_summaries],
        "metrics": metrics,
        "population": {
            "n_synthetic_accounts_per_seed":
                seed_summaries[0]["population"]["n_synthetic_accounts"],
            "n_exposures_per_seed":
                seed_summaries[0]["population"]["n_exposures"],
            "exposures_per_account":
                seed_summaries[0]["population"]["exposures_per_account"],
        },
        "interpretation_guard": seed_summaries[0]["interpretation_guard"],
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
    p = Path(out_root) / layout.artifact_name(
        "stage11_self_influence_summary"
    )
    p.write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(
        ROOT / "configs" / "amazon_self_influence.yaml"))
    ap.add_argument("--shared-root", type=Path,
                    help="Operational shared-root override; does not alter the config file.")
    ap.add_argument("--attack-dir", default=None)
    ap.add_argument("--reuse-dir", default=None)
    ap.add_argument("--roles", default=None)
    ap.add_argument("--references", default=None)
    ap.add_argument("--freeze", default=None)
    ap.add_argument("--out-dir", default=None)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed", type=int)
    g.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    layout = AmazonPathLayout.from_config(
        cfg, repo_root=ROOT, shared_root=args.shared_root)
    shared = cfg["shared_data"]
    attack_root = (
        Path(args.attack_dir).resolve() if args.attack_dir
        else layout.resolve_shared_path(shared["attack_dir"])
    )
    reuse_root = (
        Path(args.reuse_dir).resolve() if args.reuse_dir
        else layout.resolve_shared_path(shared["reuse_dir"])
    )
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
    freeze = read_json(freeze_path)
    require_artifact_category(
        freeze,
        expected=cfg["category"],
        label="Stage-3 freeze",
        allow_legacy_home_missing=False,
    )
    seeds = [int(x) for x in cfg["seed_ids"]]
    out_root.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        s = run_seed(args.seed, cfg, attack_root, reuse_root, roles_path,
                     references_path, freeze_path, out_root, args.overwrite)
        print(json.dumps(s["metrics"], indent=2))
        return

    ss = []
    for seed in seeds:
        print(f"\n=== Stage 11 seed {seed} ===")
        ss.append(run_seed(seed, cfg, attack_root, reuse_root, roles_path,
                           references_path, freeze_path, out_root, args.overwrite))
    upstream_artifacts = {
        "roles_file": roles_path,
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
            artifact = read_json(path)
            require_artifact_category(
                artifact, expected=cfg["category"], label=label
            )
            upstream_artifacts[label] = path
    raw_dataset_sha256 = resolve_raw_dataset_sha256(layout, freeze)
    final = aggregate_all(
        ss,
        out_root,
        layout=layout,
        config_path=config_path,
        raw_dataset_sha256=raw_dataset_sha256,
        freeze=freeze,
        upstream_artifacts=upstream_artifacts,
    )
    print("\nStage 11 complete.")
    print(json.dumps(final["metrics"], indent=2))


if __name__ == "__main__":
    main()
