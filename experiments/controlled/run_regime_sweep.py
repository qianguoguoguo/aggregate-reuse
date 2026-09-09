#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, platform, shutil, sys
from pathlib import Path

import numpy as np
import scipy
import sklearn
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.controlled.regime_sweep import (
    controlled_exposure_intermittency_regime,
)
from aggregate_reuse.metrics import percentile_bootstrap_ci
from aggregate_reuse.seeding import derive_stage_seed

EXPERIMENT = "controlled_exposure_intermittency_regime"
SUMMARY_METRICS = [
    "realized_R_exp", "realized_p_on", "mean_signed_increment",
    "mean_active_increment", "mean_inactive_increment", "mean_score_gap",
    "frequency_auc", "score_auc", "matched_pair_misordering",
]

def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def require_env(cfg, allow):
    required = cfg.get("required_environment", {})
    actual = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "pyyaml": yaml.__version__,
    }
    bad = {k:(str(required[k]),str(actual.get(k))) for k in required
           if str(required[k]) != str(actual.get(k))}
    if bad and not allow:
        msg = "\n".join(f"  {k}: expected {e}, found {a}"
                        for k,(e,a) in bad.items())
        raise SystemExit("Exact paper reproduction requires:\n"+msg)

def legacy_cfg(cfg):
    cc = dict(cfg["controlled_model"])
    rs = cfg["regime_sweep"]
    cc["campaign"] = {
        "p_on_grid": rs["p_on_grid"],
        "exposure_ratio_grid": rs["exposure_ratio_grid"],
        "principal_exposure_ratio": rs["principal_exposure_ratio"],
        "principal_p_on": rs["principal_p_on"],
        "scheduler": rs["scheduler"],
        "coalition_in_inactive_intervals":
            rs["coalition_in_inactive_intervals"],
    }
    cc["coalition_distributions"] = {"mean_shift": rs["coalition_distribution"]}
    cc["baseline_monte_carlo"] = {
        "draws_per_configuration":
            rs["legacy_compatibility"]["baseline_monte_carlo_draws_per_configuration"],
        "caching": True,
    }
    return {"controlled_model": cc}

def summarize_metric(rows, metric, cfg, boot_seed):
    vals = [float(r[metric]) for r in rows if math.isfinite(float(r[metric]))]
    if not vals:
        return {"mean":None,"ci_lower":None,"ci_upper":None,
                "n_finite":0,"n_total":len(rows),"status":"not_applicable"}
    if len(vals) != len(rows):
        raise RuntimeError(f"Partial non-finiteness for {metric}")
    ci = percentile_bootstrap_ci(
        vals,
        level=float(cfg["confidence_intervals"]["level"]),
        replicates=int(cfg["confidence_intervals"]["bootstrap_replicates"]),
        seed=boot_seed,
        statistic="mean",
    )
    return {"mean":ci["estimate"],"ci_lower":ci["lower"],
            "ci_upper":ci["upper"],"n_finite":len(vals),
            "n_total":len(rows),"status":"ok"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default=str(ROOT/"configs"/"controlled.yaml"))
    ap.add_argument("--only-seed",type=int)
    ap.add_argument("--clean",action="store_true")
    ap.add_argument("--allow-version-mismatch",action="store_true")
    args=ap.parse_args()

    cfg=load_config(args.config); require_env(cfg,args.allow_version_mismatch)
    seeds=list(range(int(cfg["seeds"]["start"]),int(cfg["seeds"]["stop"])+1))
    if args.only_seed is not None:
        if args.only_seed not in seeds: raise SystemExit("Seed outside frozen range")
        seeds=[args.only_seed]

    rs=cfg["regime_sweep"]; stage_name=rs["random_stage_name"]
    run_root=ROOT/"artifacts"/"results"/"controlled_sweep_runs"
    ps=ROOT/"artifacts"/"per_seed"; results=ROOT/"artifacts"/"results"
    fig=ROOT/"artifacts"/"figure_data"; manifests=ROOT/"artifacts"/"manifests"
    if args.clean and run_root.exists(): shutil.rmtree(run_root)
    for p in [run_root,ps,results,fig,manifests]: p.mkdir(parents=True,exist_ok=True)

    rows_all=[]; inv_all=[]
    lc=legacy_cfg(cfg)
    for seed in seeds:
        stage_seed=derive_stage_seed(seed,stage_name)
        rd=run_root/f"seed_{seed}"; sp=rd/"summary.json"
        rp=rd/"data"/"controlled_exposure_intermittency_rows.json"
        if sp.exists() and rp.exists():
            saved=json.loads(sp.read_text(encoding="utf-8"))
            metrics=saved["metrics"]; manifest=saved["manifest"]
            rows=json.loads(rp.read_text(encoding="utf-8"))
        else:
            if rd.exists(): shutil.rmtree(rd)
            rd.mkdir(parents=True)
            metrics,manifest=controlled_exposure_intermittency_regime(
                lc,seed,rd,stage_seed
            )
            rows=json.loads(rp.read_text(encoding="utf-8"))
            sp.write_text(json.dumps({"metrics":metrics,"manifest":manifest},
                                     indent=2,sort_keys=True),encoding="utf-8")
        if len(rows)!=70: raise RuntimeError(f"Seed {seed}: grid != 70")
        rows_all.extend(rows)
        inv_all.append({"seed":seed,**manifest["invariants"]})

    (ps/"controlled_sweep_per_seed_rows.json").write_text(
        json.dumps(rows_all,indent=2,sort_keys=True),encoding="utf-8")
    with (ps/"controlled_sweep_per_seed_rows.csv").open(
        "w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows_all[0].keys()))
        w.writeheader();w.writerows(rows_all)
    with (ps/"controlled_sweep_invariants.csv").open(
        "w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(inv_all[0].keys()))
        w.writeheader();w.writerows(inv_all)

    if len(seeds)!=30:
        print(f"Completed 70-cell sweep for seed {seeds[0]}")
        return 0

    cells={}
    for row in rows_all:
        key=(float(row["requested_R_exp"]),float(row["requested_p_on"]))
        cells.setdefault(key,[]).append(row)
    expected={(float(R),float(p)) for R in rs["exposure_ratio_grid"]
              for p in rs["p_on_grid"]}
    if set(cells)!=expected: raise RuntimeError("Grid mismatch")

    summary_rows=[]; nested={}
    for R,p_on in sorted(cells):
        cr=cells[(R,p_on)]
        if len(cr)!=30 or {int(r["master_seed"]) for r in cr}!=set(seeds):
            raise RuntimeError(f"Incomplete cell {(R,p_on)}")
        out={"requested_R_exp":R,"requested_p_on":p_on,"n_seeds":30}
        for m in SUMMARY_METRICS:
            bs=derive_stage_seed(0,f"controlled_sweep_bootstrap::{R}::{p_on}::{m}")
            s=summarize_metric(cr,m,cfg,bs)
            for k in ["mean","ci_lower","ci_upper","n_finite","status"]:
                out[f"{m}_{k}"]=s[k]
        summary_rows.append(out); nested.setdefault(str(R),{})[str(p_on)]=out

    principal=next(r for r in summary_rows
                   if r["requested_R_exp"]==float(rs["principal_exposure_ratio"])
                   and r["requested_p_on"]==float(rs["principal_p_on"]))
    matched=[r for r in summary_rows if r["requested_R_exp"]==1.0]
    result={
        "experiment":EXPERIMENT,"n_seeds":30,"seed_ids":seeds,
        "n_cells":len(summary_rows),
        "bootstrap_level":float(cfg["confidence_intervals"]["level"]),
        "bootstrap_replicates":int(cfg["confidence_intervals"]["bootstrap_replicates"]),
        "undefined_metric_policy":(
            "If a metric is non-finite for all seeds in a cell because the "
            "quantity is structurally undefined (e.g., inactive mean at "
            "p_on=1), its estimate and CI are null. Partial non-finiteness "
            "within a cell is treated as an error."
        ),
        "principal_cell":principal,
        "matched_exposure_curve_R_exp_1":matched,
        "cells":nested,
    }
    (results/"controlled_sweep_summary.json").write_text(
        json.dumps(result,indent=2,sort_keys=True),encoding="utf-8")
    with (fig/"controlled_regime_auc.csv").open(
        "w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(summary_rows[0].keys()))
        w.writeheader();w.writerows(summary_rows)

    all_inv=all(all(bool(v) for k,v in row.items() if k!="seed") for row in inv_all)
    (manifests/"controlled_sweep_manifest.json").write_text(
        json.dumps({
            "experiment":EXPERIMENT,"seeds":seeds,"n_cells":70,
            "config":str(Path(args.config).resolve().relative_to(ROOT)),
            "stage_seed_name":stage_name,"all_run_invariants_pass":all_inv,
        },indent=2,sort_keys=True),encoding="utf-8")
    if not all_inv: raise RuntimeError("Sweep invariant failed")
    print("Controlled regime sweep complete: 30 seeds x 70 cells")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
