#!/usr/bin/env python3
from __future__ import annotations
import csv,json,subprocess,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
VERIFY=ROOT/"tools"/"verify_results.py"

def compare(e,a,r):
    return subprocess.run([
        sys.executable,str(VERIFY),"compare","--expected",str(e),"--actual",str(a),
        "--rtol","1e-10","--atol","1e-12","--report",str(r)
    ]).returncode==0

def core():
    gdir=ROOT/"paper_results"/"expected"/"per_seed"/"controlled_sweep_core"
    adir=ROOT/"artifacts"/"results"/"controlled_sweep_runs"
    max_abs=0.;ok=True
    fields=["D","normal_n_t","normal_frequency","campaign_uniform"]
    for seed in range(27001,27031):
        g=np.load(gdir/f"seed_{seed}.npz")
        a=np.load(adir/f"seed_{seed}"/"data"/"controlled_exposure_intermittency_core.npz")
        for k in fields:
            if g[k].shape!=a[k].shape: ok=False;continue
            diff=float(np.max(np.abs(g[k].astype(float)-a[k].astype(float))))
            max_abs=max(max_abs,diff)
            if not np.allclose(g[k],a[k],rtol=1e-10,atol=1e-12): ok=False
    return ok,max_abs

def main():
    out=ROOT/"artifacts"/"verification";out.mkdir(parents=True,exist_ok=True)
    checks={}
    checks["per_seed_cell_rows"]=compare(
        ROOT/"paper_results"/"expected"/"per_seed"/"controlled_sweep_per_seed_rows.json",
        ROOT/"artifacts"/"per_seed"/"controlled_sweep_per_seed_rows.json",
        out/"controlled_sweep_rows_compare.json")
    checks["aggregate_summary"]=compare(
        ROOT/"paper_results"/"expected"/"source_snapshots"/"controlled_sweep_30seed_summary.json",
        ROOT/"artifacts"/"results"/"controlled_sweep_summary.json",
        out/"controlled_sweep_summary_compare.json")
    checks["figure_grid"]=compare(
        ROOT/"paper_results"/"expected"/"figure_data"/"controlled_regime_auc.csv",
        ROOT/"artifacts"/"figure_data"/"controlled_regime_auc.csv",
        out/"controlled_sweep_grid_compare.json")
    checks["per_seed_core_arrays"],max_abs=core()
    with (ROOT/"artifacts"/"per_seed"/"controlled_sweep_invariants.csv").open(
        "r",newline="",encoding="utf-8") as f: inv=list(csv.DictReader(f))
    b=[k for k in inv[0] if k!="seed"]
    checks["all_invariants"]=len(inv)==30 and all(
        row[k].strip().lower()=="true" for row in inv for k in b)
    status="PASS" if all(checks.values()) else "FAIL"
    report={"phase":"2F","status":status,"checks":checks,
            "max_absolute_core_array_difference":max_abs,
            "tolerances":{"rtol":1e-10,"atol":1e-12}}
    (out/"phase2F_report.json").write_text(
        json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    print("="*68);print("PHASE 2F CONTROLLED REGIME-SWEEP VERIFICATION");print("="*68)
    for k,v in checks.items(): print(f"{k:28s}: {'PASS' if v else 'FAIL'}")
    print(f"max core-array abs diff      : {max_abs:.3e}")
    print(f"OVERALL                      : {status}");print("="*68)
    return 0 if status=="PASS" else 1

if __name__=="__main__": raise SystemExit(main())
