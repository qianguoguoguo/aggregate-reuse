#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, gzip, json, math, sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.complementarity import (
    aggregate_counterfactual_scores,
    auc_rank,
    coactivity_scores,
    degree_preserving_randomize,
    incidence_edges,
    load_attack_world,
    pooled_z,
    repeated_pair_summary,
    team_rich_incidence,
    validate_regular_incidence,
)
from aggregate_reuse.amazon.paths import AmazonPathLayout
from aggregate_reuse.amazon.provenance import (
    build_provenance,
    require_artifact_category,
    resolve_raw_dataset_sha256,
)

def write_edges(path, adj, prefix):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["account_id","asin"])
        for u,a in incidence_edges(adj):
            w.writerow([f"{prefix}_{u:06d}", a])

def run_seed(seed, cfg, attack_root, out_root, overwrite):
    r = int(cfg["reuse_r"]); m = int(cfg["item_degree_m"])
    L = int(cfg["n_items"]); M = int(cfg["n_accounts"])
    team_size = int(cfg["team_size"]); reps = int(cfg["team_repetitions"])
    swap_mult = int(cfg["swap_success_multiplier"])

    attack_dir = attack_root / f"seed_{seed:03d}"
    attack_summary_path = attack_dir / "attack_summary.json"
    attack_path = attack_dir / "attack_world.csv.gz"
    if not attack_summary_path.exists() or not attack_path.exists():
        raise FileNotFoundError(
            f"Missing Phase-2I attack artifacts for seed {seed}."
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
            label=f"existing Stage-7 seed {seed}",
        )
        return existing

    s4 = json.loads(attack_summary_path.read_text(encoding="utf-8"))
    require_artifact_category(
        s4, expected=cfg["category"], label="Stage-4 attack summary"
    )
    if s4["experiment"] != "amazon_primary_attack":
        raise RuntimeError("Unexpected Phase-2I attack summary.")
    if int(s4["sampled_items"]) != L or int(s4["m"]) != m:
        raise RuntimeError("Phase-2I attack dimensions differ from complementarity config.")
    if not all(bool(v) for v in s4["invariants"].values()):
        raise RuntimeError("Phase-2I attack invariants did not all pass.")

    attack = load_attack_world(attack_path)
    asins = sorted(attack)
    if len(asins) != L or L*m != M*r:
        raise RuntimeError("Frozen Stage-7 degree equation does not balance.")

    team = team_rich_incidence(
        asins, team_size=team_size, repetitions=reps,
        rng=np.random.default_rng(seed*100003 + 7001)
    )
    swaps = swap_mult * L * m
    rand1, sw1 = degree_preserving_randomize(
        team, rng=np.random.default_rng(seed*100003 + 7101),
        successful_swaps=swaps
    )
    rand2, sw2 = degree_preserving_randomize(
        team, rng=np.random.default_rng(seed*100003 + 7201),
        successful_swaps=swaps
    )
    for adj in [team, rand1, rand2]:
        validate_regular_incidence(adj, asins=asins, account_degree=r, item_degree=m)

    # Evidence-only branch: attacked + rand1 vs clean + identical rand1.
    agg_e_pos = aggregate_counterfactual_scores(rand1, attack)
    agg_e_neg = np.zeros(M)
    co_e_pos = coactivity_scores(rand1)
    co_e_neg = co_e_pos.copy()

    # Topology-only branch: clean + team vs clean + randomized.
    agg_t_pos = np.zeros(M); agg_t_neg = np.zeros(M)
    co_t_pos = coactivity_scores(team)
    co_t_neg = coactivity_scores(rand2)

    # Both-active branch: attacked + team vs clean + randomized.
    agg_b_pos = aggregate_counterfactual_scores(team, attack)
    agg_b_neg = np.zeros(M)
    co_b_pos = co_t_pos.copy()
    co_b_neg = co_t_neg.copy()

    # Primary mixed benchmark.
    agg_pos = np.concatenate([agg_e_pos, agg_t_pos])
    agg_neg = np.concatenate([agg_e_neg, agg_t_neg])
    co_pos = np.concatenate([co_e_pos, co_t_pos])
    co_neg = np.concatenate([co_e_neg, co_t_neg])

    za_pos, za_neg, anorm = pooled_z(agg_pos, agg_neg)
    zc_pos, zc_neg, cnorm = pooled_z(co_pos, co_neg)
    comb_pos = za_pos + zc_pos
    comb_neg = za_neg + zc_neg

    agg_auc = auc_rank(agg_pos, agg_neg)
    co_auc = auc_rank(co_pos, co_neg)
    comb_auc = auc_rank(comb_pos, comb_neg)
    gain = comb_auc - max(agg_auc, co_auc)

    # Both-active combination uses same normalization frozen by primary benchmark.
    asd = anorm["std"] if anorm["std"] else 1.0
    csd = cnorm["std"] if cnorm["std"] else 1.0
    both_comb_pos = (agg_b_pos-anorm["mean"])/asd + (co_b_pos-cnorm["mean"])/csd
    both_comb_neg = (agg_b_neg-anorm["mean"])/asd + (co_b_neg-cnorm["mean"])/csd

    freq = np.full(2*M, r, dtype=float)
    frequency_auc = auc_rank(freq, freq)
    if abs(frequency_auc-0.5) > 1e-12:
        raise RuntimeError("Frequency AUC invariant failed.")

    write_edges(out_seed/"team_incidence.csv.gz", team, "team")
    write_edges(out_seed/"random_incidence_1.csv.gz", rand1, "rand1")
    write_edges(out_seed/"random_incidence_2.csv.gz", rand2, "rand2")

    summary = {
        "stage": "amazon_stage7_complementarity",
        "category": cfg["category"],
        "seed": seed,
        "selected_lambda": float(s4["selected_lambda"]),
        "stage4_attack_world_sha256": s4["attack_world_sha256"],
        "dimensions": {
            "n_items":L, "item_degree_m":m, "account_degree_r":r,
            "n_accounts_per_branch":M, "team_size":team_size,
            "team_repetitions":reps,
        },
        "swap_diagnostics": {"random1":sw1, "random2":sw2},
        "incidence_diagnostics": {
            "team": repeated_pair_summary(team),
            "random1": repeated_pair_summary(rand1),
            "random2": repeated_pair_summary(rand2),
        },
        "branch_metrics": {
            "evidence_only": {
                "aggregate_auc": auc_rank(agg_e_pos, agg_e_neg),
                "coactivity_auc": auc_rank(co_e_pos, co_e_neg),
            },
            "topology_only": {
                "aggregate_auc": auc_rank(agg_t_pos, agg_t_neg),
                "coactivity_auc": auc_rank(co_t_pos, co_t_neg),
            },
            "both_active": {
                "aggregate_auc": auc_rank(agg_b_pos, agg_b_neg),
                "coactivity_auc": auc_rank(co_b_pos, co_b_neg),
                "combined_auc": auc_rank(both_comb_pos, both_comb_neg),
            },
        },
        "primary_mixed_benchmark": {
            "frequency_auc": frequency_auc,
            "aggregate_auc": agg_auc,
            "coactivity_auc": co_auc,
            "combined_auc": comb_auc,
            "combined_minus_best_single": gain,
            "normalization": {
                "aggregate": anorm, "coactivity": cnorm,
                "combined_rule": "z(aggregate)+z(coactivity)",
                "weights_trained": False,
            },
        },
        "invariants": {
            "all_account_degrees_equal_r": True,
            "all_item_degrees_equal_m": True,
            "frequency_auc_exactly_half": True,
            "evidence_only_branch_uses_identical_incidence_for_pos_neg": True,
            "topology_only_branch_has_zero_aggregate_treatment_effect": True,
            "stage4_attack_world_reused_without_modification": True,
        },
        "interpretation_guard": (
            "The primary mixed benchmark is a complementarity stress test, not "
            "an isolation experiment. Half of positive mechanisms carry aggregate "
            "evidence without repeated-team structure; half carry repeated-team "
            "structure without rating intervention. The combined score is untrained."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary

def summarize(vals):
    vals=np.asarray(vals,float); mean=float(vals.mean())
    if len(vals)==1:
        return {"mean":mean,"ci_lower":mean,"ci_upper":mean}
    se=float(vals.std(ddof=1)/math.sqrt(len(vals)))
    crit=float(t.ppf(0.975,df=len(vals)-1))
    return {"mean":mean,"ci_lower":mean-crit*se,"ci_upper":mean+crit*se}

def aggregate_all(
    ss,
    out_root,
    *,
    layout,
    config_path,
    raw_dataset_sha256,
    upstream_artifacts,
):
    def gp(o,*path):
        for p in path: o=o[p]
        return o
    paths = {
        "frequency_auc":("primary_mixed_benchmark","frequency_auc"),
        "aggregate_auc":("primary_mixed_benchmark","aggregate_auc"),
        "coactivity_auc":("primary_mixed_benchmark","coactivity_auc"),
        "combined_auc":("primary_mixed_benchmark","combined_auc"),
        "combined_minus_best_single":("primary_mixed_benchmark","combined_minus_best_single"),
        "evidence_only_aggregate_auc":("branch_metrics","evidence_only","aggregate_auc"),
        "evidence_only_coactivity_auc":("branch_metrics","evidence_only","coactivity_auc"),
        "topology_only_aggregate_auc":("branch_metrics","topology_only","aggregate_auc"),
        "topology_only_coactivity_auc":("branch_metrics","topology_only","coactivity_auc"),
        "both_active_aggregate_auc":("branch_metrics","both_active","aggregate_auc"),
        "both_active_coactivity_auc":("branch_metrics","both_active","coactivity_auc"),
        "both_active_combined_auc":("branch_metrics","both_active","combined_auc"),
    }
    metrics={k:summarize([float(gp(s,*p)) for s in ss]) for k,p in paths.items()}
    final={
        "stage":"amazon_stage7_complementarity",
        "category":layout.category,
        "raw_dataset_sha256":raw_dataset_sha256,
        "selected_lambda":float(ss[0]["selected_lambda"]),
        "n_seeds":len(ss),
        "seed_ids":[int(s["seed"]) for s in ss],
        "metrics":metrics,
        "dimensions":ss[0]["dimensions"],
        "invariants":ss[0]["invariants"],
        "interpretation_guard":ss[0]["interpretation_guard"],
    }
    final["provenance"] = build_provenance(
        category=layout.category,
        config_path=config_path,
        raw_dataset_sha256=raw_dataset_sha256,
        selected_lambda=final["selected_lambda"],
        seed_ids=final["seed_ids"],
        upstream_artifacts=upstream_artifacts,
        repo_root=ROOT,
        shared_root=layout.shared_category_root,
    )
    path=out_root/layout.artifact_name("stage7_complementarity_summary")
    path.write_text(json.dumps(final,indent=2,sort_keys=True),encoding="utf-8")
    return final

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default=str(ROOT/"configs"/"amazon_complementarity.yaml"))
    ap.add_argument("--shared-root",type=Path,
                    help="Operational shared-root override; does not alter the config file.")
    ap.add_argument("--attack-dir",default=None)
    ap.add_argument("--out-dir",default=None)
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed",type=int); g.add_argument("--all-seeds",action="store_true")
    ap.add_argument("--overwrite",action="store_true")
    args=ap.parse_args()

    config_path=Path(args.config).resolve()
    cfg=yaml.safe_load(config_path.read_text(encoding="utf-8"))
    layout=AmazonPathLayout.from_config(
        cfg, repo_root=ROOT, shared_root=args.shared_root)
    shared=cfg["shared_data"]
    s4=(Path(args.attack_dir).resolve() if args.attack_dir
        else layout.resolve_shared_path(shared["attack_dir"]))
    out=(Path(args.out_dir).resolve() if args.out_dir
         else layout.resolve_shared_path(shared["output_subdir"]))
    out.mkdir(parents=True,exist_ok=True)

    if args.seed is not None:
        if args.seed not in cfg["seed_ids"]: raise ValueError("Seed not frozen.")
        s=run_seed(args.seed,cfg,s4,out,args.overwrite)
        print(json.dumps({
            "branch_metrics":s["branch_metrics"],
            "primary_mixed_benchmark":s["primary_mixed_benchmark"],
            "incidence_diagnostics":s["incidence_diagnostics"],
        },indent=2))
        print(f"Summary: {out/f'seed_{args.seed:03d}'/'summary.json'}")
        return

    ss=[]
    for seed in cfg["seed_ids"]:
        print(f"\n=== Stage 7 seed {seed} ===")
        ss.append(run_seed(int(seed),cfg,s4,out,args.overwrite))
    attack_aggregate=s4/layout.artifact_name("stage4_attack_summary")
    upstream_artifacts={}
    upstream_data=[]
    if attack_aggregate.is_file():
        attack_aggregate_data=json.loads(attack_aggregate.read_text(encoding="utf-8"))
        require_artifact_category(
            attack_aggregate_data,
            expected=cfg["category"],
            label="Stage-4 attack aggregate",
        )
        upstream_artifacts["stage4_attack_summary"]=attack_aggregate
        upstream_data.append(attack_aggregate_data)
    else:
        first_seed_summary=s4/f"seed_{int(cfg['seed_ids'][0]):03d}"/"attack_summary.json"
        upstream_artifacts["stage4_attack_seed_summary"]=first_seed_summary
    raw_dataset_sha256=resolve_raw_dataset_sha256(layout,*upstream_data)
    final=aggregate_all(
        ss,
        out,
        layout=layout,
        config_path=config_path,
        raw_dataset_sha256=raw_dataset_sha256,
        upstream_artifacts=upstream_artifacts,
    )
    print("\nStage 7 complete.")
    for k,v in final["metrics"].items():
        print(f"{k}: {v['mean']:.6f} [{v['ci_lower']:.6f}, {v['ci_upper']:.6f}]")
    print(f"Summary: {out/layout.artifact_name('stage7_complementarity_summary')}")

if __name__=="__main__":
    main()
