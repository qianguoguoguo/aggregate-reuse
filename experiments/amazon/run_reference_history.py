#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/"src"
if str(SRC) not in sys.path:
    sys.path.insert(0,str(SRC))

from aggregate_reuse.amazon.reference_history import (
    auc_rank,
    counts_from_reviews,
    exact_predictive_w1_baseline,
    load_attack_world,
    load_r8_assignment,
    load_roles,
    w1_counts,
)
from aggregate_reuse.amazon.paths import AmazonPathLayout
from aggregate_reuse.amazon.provenance import (
    build_provenance,
    require_artifact_category,
    resolve_raw_dataset_sha256,
)

def calibrate_reference_lengths(roles, ref_lengths, lambda_grid):
    asins=sorted(roles)
    result={}
    for nref in ref_lengths:
        item_counts={}
        global_counts=np.zeros(5,dtype=float)
        for asin in asins:
            ref_reviews=roles[asin]["reference"][:nref]
            if len(ref_reviews)!=nref:
                raise RuntimeError(f"{asin}: insufficient reference reviews.")
            c=counts_from_reviews(ref_reviews)
            item_counts[asin]=c
            global_counts += c

        loo_probs={}
        for asin in asins:
            loo=global_counts-item_counts[asin]
            loo_probs[asin]=loo/loo.sum()

        rows=[]
        for lam in lambda_grid:
            signed=[]
            for asin in asins:
                c=item_counts[asin].astype(float)
                alpha=c+float(lam)*loo_probs[asin]
                q=alpha/alpha.sum()
                b=exact_predictive_w1_baseline(alpha,30)
                for role in ["calibration_1","calibration_2"]:
                    obs=counts_from_reviews(roles[asin][role])
                    signed.append(w1_counts(obs,q)-b)
            mean=float(np.mean(signed))
            rows.append({
                "reference_length":int(nref),
                "lambda":float(lam),
                "predictive_signed_mean":mean,
                "predictive_abs_mean":abs(mean),
            })
        best=min(rows,key=lambda x:(x["predictive_abs_mean"],x["lambda"]))
        result[int(nref)]={
            "selected_lambda":float(best["lambda"]),
            "selected_calibration_row":best,
            "grid_rows":rows,
            "item_counts":item_counts,
            "loo_probs":loo_probs,
        }
        print(
            f"n_ref={nref}: lambda={best['lambda']:g}, "
            f"cal mean={best['predictive_signed_mean']:.8f}"
        )
    return result

def run_seed(seed, cfg, calibration, attack_root, reuse_root, twins_root, out_root):
    r=int(cfg["reuse_r"])
    attack_seed=attack_root/f"seed_{seed:03d}"
    reuse_seed=reuse_root/f"seed_{seed:03d}"
    attack=load_attack_world(attack_seed/"attack_world.csv.gz")
    edges=load_r8_assignment(reuse_seed/"identity_assignment_r8.csv.gz")

    attack_summary_path=attack_seed/"attack_summary.json"
    reuse_summary_path=reuse_seed/"reuse_summary.json"
    if not attack_summary_path.exists() or not reuse_summary_path.exists():
        raise FileNotFoundError(
            f"Missing Phase-2I/2J handoff summary for seed {seed}"
        )
    attack_summary=json.loads(
        attack_summary_path.read_text(encoding="utf-8")
    )
    reuse_summary=json.loads(
        reuse_summary_path.read_text(encoding="utf-8")
    )
    require_artifact_category(
        attack_summary, expected=cfg["category"], label="Stage-4 attack summary"
    )
    require_artifact_category(
        reuse_summary, expected=cfg["category"], label="Stage-4 reuse summary"
    )
    if reuse_summary["attack_world_sha256"] != attack_summary["attack_world_sha256"]:
        raise RuntimeError("Phase-2J attack hash differs from Phase-2I.")
    if not all(bool(v) for v in attack_summary["invariants"].values()):
        raise RuntimeError("Phase-2I attack invariants failed.")
    if not all(bool(v) for v in reuse_summary["invariants"].values()):
        raise RuntimeError("Phase-2J reuse invariants failed.")

    by_account=defaultdict(list)
    for e in edges:
        by_account[e["account"]].append(e["asin"])
    if any(len(v)!=r for v in by_account.values()):
        raise RuntimeError("r=8 degree invariant failed.")

    seed_rows=[]
    for nref in cfg["reference_lengths"]:
        cal=calibration[int(nref)]
        lam=cal["selected_lambda"]
        d_by_item={}
        for asin,a in attack.items():
            c=cal["item_counts"][asin].astype(float)
            loo=cal["loo_probs"][asin]
            alpha=c+lam*loo
            q=alpha/alpha.sum()
            d = w1_counts(a["attack_counts"],q)-w1_counts(a["clean_counts"],q)
            d_by_item[asin]=d

            if int(nref)==120:
                if abs(d-a["d_cf_stage4"])>1e-12:
                    raise RuntimeError(
                        f"n_ref=120 failed Stage-4 reproduction for {asin}: "
                        f"{d} vs {a['d_cf_stage4']}"
                    )

        scores=np.asarray([
            sum(d_by_item[asin] for asin in items)
            for _,items in sorted(by_account.items())
        ],dtype=float)
        zeros=np.zeros_like(scores)

        auc=auc_rank(scores,zeros)
        pos=float(np.mean(scores>0))
        mis=float(np.mean(scores<=0))
        mean_gap=float(scores.mean())
        mean_d=float(np.mean(list(d_by_item.values())))

        # Optional exact Stage-5 reproduction check at n_ref=120.
        if int(nref)==120:
            s5path=twins_root/f"seed_{seed:03d}"/"summary.json"
            if s5path.exists():
                s5=json.loads(s5path.read_text(encoding="utf-8"))
                require_artifact_category(
                    s5, expected=cfg["category"], label="Stage-5 seed summary"
                )
                target=float(s5["metrics"]["counterfactual_score_auc"])
                if abs(auc-target)>1e-12:
                    raise RuntimeError(
                        f"Stage-5 AUC reproduction failed: {auc} vs {target}"
                    )

        seed_rows.append({
            "category":cfg["category"],
            "seed":seed,
            "reference_length":int(nref),
            "selected_lambda":lam,
            "mean_d_cf":mean_d,
            "counterfactual_score_auc":auc,
            "positive_pair_fraction":pos,
            "paired_misordering_probability":mis,
            "mean_paired_score_gap":mean_gap,
        })
    return seed_rows

def ci(vals):
    x=np.asarray(vals,float); m=float(x.mean())
    if len(x)==1:
        return {"mean":m,"ci_lower":m,"ci_upper":m}
    se=float(x.std(ddof=1)/math.sqrt(len(x)))
    crit=float(t.ppf(0.975,df=len(x)-1))
    return {"mean":m,"ci_lower":m-crit*se,"ci_upper":m+crit*se}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default=str(ROOT/"configs"/"amazon_reference_history.yaml"))
    ap.add_argument("--shared-root",type=Path,
                    help="Operational shared-root override; does not alter the config file.")
    ap.add_argument("--roles",default=None)
    ap.add_argument("--attack-dir",default=None)
    ap.add_argument("--reuse-dir",default=None)
    ap.add_argument("--twins-dir",default=None)
    ap.add_argument("--out-dir",default=None)
    args=ap.parse_args()

    config_path=Path(args.config).resolve()
    cfg=yaml.safe_load(config_path.read_text(encoding="utf-8"))
    layout=AmazonPathLayout.from_config(
        cfg, repo_root=ROOT, shared_root=args.shared_root)
    shared=cfg["shared_data"]
    roles_path=(Path(args.roles).resolve() if args.roles
                else layout.resolve_shared_path(shared["roles_file"]))
    attack_root=(Path(args.attack_dir).resolve() if args.attack_dir
                 else layout.resolve_shared_path(shared["attack_dir"]))
    reuse_root=(Path(args.reuse_dir).resolve() if args.reuse_dir
                else layout.resolve_shared_path(shared["reuse_dir"]))
    twins_root=(Path(args.twins_dir).resolve() if args.twins_dir
                else layout.resolve_shared_path(shared["twins_dir"]))
    out=(Path(args.out_dir).resolve() if args.out_dir
         else layout.resolve_shared_path(shared["output_subdir"]))
    roles=load_roles(roles_path)
    print(f"Loaded {len(roles):,} Stage-2 items.")

    calibration=calibrate_reference_lengths(
        roles,cfg["reference_lengths"],cfg["lambda_grid"]
    )

    out.mkdir(parents=True,exist_ok=True)

    # Calibration grid output.
    grid=[]
    for nref in cfg["reference_lengths"]:
        grid.extend(calibration[int(nref)]["grid_rows"])
    calibration_path=out/layout.artifact_name("stage8_reference_history_calibration")
    with open(calibration_path,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(grid[0].keys()))
        w.writeheader(); w.writerows(grid)

    seed_rows=[]
    for seed in cfg["seed_ids"]:
        print(f"seed {seed}")
        seed_rows.extend(run_seed(
            int(seed), cfg, calibration,
            attack_root,
            reuse_root,
            twins_root,
            out
        ))

    seed_metrics_path=out/layout.artifact_name("stage8_reference_history_seed_metrics")
    with open(seed_metrics_path,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(seed_rows[0].keys()))
        w.writeheader(); w.writerows(seed_rows)

    metrics={}
    for nref in cfg["reference_lengths"]:
        rr=[x for x in seed_rows if x["reference_length"]==int(nref)]
        metrics[str(nref)]={
            "selected_lambda":calibration[int(nref)]["selected_lambda"],
            "calibration_predictive_signed_mean":calibration[int(nref)]["selected_calibration_row"]["predictive_signed_mean"],
            "mean_d_cf":ci([x["mean_d_cf"] for x in rr]),
            "counterfactual_score_auc":ci([x["counterfactual_score_auc"] for x in rr]),
            "positive_pair_fraction":ci([x["positive_pair_fraction"] for x in rr]),
            "paired_misordering_probability":ci([x["paired_misordering_probability"] for x in rr]),
            "mean_paired_score_gap":ci([x["mean_paired_score_gap"] for x in rr]),
        }

    summary={
        "stage":"amazon_stage8_reference_history_robustness",
        "category":cfg["category"],
        "n_seeds":len(cfg["seed_ids"]),
        "seed_ids":[int(x) for x in cfg["seed_ids"]],
        "reference_lengths":[int(x) for x in cfg["reference_lengths"]],
        "reuse_r":int(cfg["reuse_r"]),
        "metrics_by_reference_length":metrics,
        "invariants":{
            "stage4_attack_world_fixed":True,
            "stage4_r8_identity_assignment_fixed":True,
            "lambda_selected_from_calibration_only_for_each_reference_length":True,
            "n_ref_120_reproduces_stage4_item_dcf":True,
            "n_ref_120_reproduces_stage5_counterfactual_auc_when_stage5_seed_summaries_exist":True,
        },
        "interpretation_guard":(
            "Only the historical prefix used to estimate the item reference and "
            "its calibration-selected shrinkage parameter vary. Attack items, "
            "slots, replacement ratings, and r=8 identities are inherited unchanged."
        ),
    }
    primary_reference_length=120
    if primary_reference_length not in calibration:
        primary_reference_length=max(int(x) for x in cfg["reference_lengths"])
    primary_lambda=calibration[primary_reference_length]["selected_lambda"]
    upstream_artifacts={"roles_file":roles_path}
    upstream_data=[]
    for label,artifact_path in {
        "stage4_attack_summary":attack_root / layout.artifact_name(
            "stage4_attack_summary"
        ),
        "stage4_reuse_summary":reuse_root / layout.artifact_name(
            "stage4_reuse_summary"
        ),
        "stage5_twins_summary":twins_root / layout.artifact_name(
            "stage5_twins_summary"
        ),
    }.items():
        if artifact_path.is_file():
            artifact=json.loads(artifact_path.read_text(encoding="utf-8"))
            require_artifact_category(
                artifact, expected=cfg["category"], label=label
            )
            upstream_artifacts[label]=artifact_path
            upstream_data.append(artifact)
    if not upstream_data:
        first_seed=int(cfg["seed_ids"][0])
        attack_seed_summary=attack_root/f"seed_{first_seed:03d}"/"attack_summary.json"
        upstream_artifacts["stage4_attack_seed_summary"]=attack_seed_summary
    raw_dataset_sha256=resolve_raw_dataset_sha256(layout,*upstream_data)
    summary["raw_dataset_sha256"]=raw_dataset_sha256
    summary["selected_lambda"]=float(primary_lambda)
    summary["selected_lambda_scope"]=(
        f"primary reference length n_ref={primary_reference_length}; "
        "metrics_by_reference_length records every calibration-selected value"
    )
    summary["provenance"]=build_provenance(
        category=layout.category,
        config_path=config_path,
        raw_dataset_sha256=raw_dataset_sha256,
        selected_lambda=primary_lambda,
        seed_ids=summary["seed_ids"],
        upstream_artifacts=upstream_artifacts,
        repo_root=ROOT,
        shared_root=layout.shared_category_root,
    )
    path=out/layout.artifact_name("stage8_reference_history_summary")
    path.write_text(json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(summary["metrics_by_reference_length"],indent=2))
    print(f"Summary: {path}")

if __name__=="__main__":
    main()
