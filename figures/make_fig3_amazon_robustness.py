#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, re
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_SIZE = (8.6, 4.6)
PNG_DPI = 400

def output_stems(prefix):
    if re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", prefix) is None:
        raise ValueError(f"Invalid figure stem prefix: {prefix!r}")
    return (
        f"fig3a_{prefix}_complementarity",
        f"fig3b_{prefix}_reference_history",
        f"fig3c_{prefix}_intervention_strength",
    )

def configure_plot_style():
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 17,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "figure.titlesize": 17,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 7.0,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

def load_rows(path):
    if not path.exists(): raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def rows_for_panel(rows, panel):
    return [r for r in rows if r["panel"] == panel]

def save_figure(fig, out_pdf):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=PNG_DPI)
    plt.close(fig)

def plot_complementarity(rows, out):
    rr = rows_for_panel(rows, "complementarity")
    order = ["Aggregate", "Co-activity", "Combined"]
    by = {str(r["x"]):r for r in rr}
    y = np.asarray([float(by[k]["mean"]) for k in order])
    lo = np.asarray([float(by[k]["ci_lower"]) for k in order])
    hi = np.asarray([float(by[k]["ci_upper"]) for k in order])
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = np.arange(len(order))
    ax.bar(x, y, yerr=np.vstack([y-lo,hi-y]), capsize=5,
           error_kw={"elinewidth":1.8,"capthick":1.8})
    ax.axhline(0.5, linestyle=":", linewidth=1.2)
    ax.set_ylim(0.48,max(0.91,float(np.max(hi))+0.03)); ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel(r"ROC--AUC"); ax.grid(axis="y", alpha=0.15)
    for i,v in enumerate(y):
        ax.text(i,v+0.010,f"{v:.3f}",ha="center",va="bottom",fontsize=16)
    fig.tight_layout(); save_figure(fig,out)

def plot_reference_history(rows, out):
    rr = sorted(rows_for_panel(rows,"reference_history"),key=lambda r:int(float(r["x"])))
    x=np.asarray([int(float(r["x"])) for r in rr])
    y=np.asarray([float(r["mean"]) for r in rr])
    lo=np.asarray([float(r["ci_lower"]) for r in rr])
    hi=np.asarray([float(r["ci_upper"]) for r in rr])
    fig,ax=plt.subplots(figsize=FIG_SIZE)
    ax.errorbar(x,y,yerr=np.vstack([y-lo,hi-y]),marker="o",linewidth=2.4,
                markersize=8.0,capsize=5,elinewidth=1.8,capthick=1.8)
    ax.axhline(0.5,linestyle=":",linewidth=1.2)
    ax.set_ylim(0.48,max(0.78,float(np.max(hi))+0.02))
    ax.set_xticks(x); ax.set_xlabel(r"Reference history $n_{\mathrm{ref}}$")
    ax.set_ylabel(r"Matched-twin ROC--AUC"); ax.grid(alpha=0.15)
    for x0,y0 in zip(x,y):
        ax.text(x0,y0+0.010,f"{y0:.3f}",ha="center",va="bottom",fontsize=16)
    fig.tight_layout(); save_figure(fig,out)

def plot_intervention_strength(rows, out):
    rr=sorted(rows_for_panel(rows,"intervention_strength"),key=lambda r:int(float(r["x"])))
    x=np.asarray([int(float(r["x"])) for r in rr])
    y=np.asarray([float(r["mean"]) for r in rr])
    lo=np.asarray([float(r["ci_lower"]) for r in rr])
    hi=np.asarray([float(r["ci_upper"]) for r in rr])
    d=np.asarray([float(r["aux_value"]) for r in rr])
    fig,ax=plt.subplots(figsize=FIG_SIZE)
    ax.errorbar(x,y,yerr=np.vstack([y-lo,hi-y]),marker="o",linewidth=2.4,
                markersize=8.0,capsize=5,elinewidth=1.8,capthick=1.8)
    ax.axhline(0.5,linestyle=":",linewidth=1.2)
    ax.set_ylim(min(0.12,float(np.min(lo))-0.03),max(0.90,float(np.max(hi))+0.03))
    ax.set_xticks(x); ax.set_xlabel(r"Modified reviews per block $k$")
    ax.set_ylabel(r"Matched-twin ROC--AUC"); ax.grid(alpha=0.15)
    offsets={3:(0.10,0.035,"left","bottom"),6:(0.10,0.045,"left","bottom"),9:(-0.10,-0.045,"right","top")}
    for x0,y0,d0 in zip(x,y,d):
        dx,dy,ha,va=offsets[int(x0)]
        ax.text(x0+dx,y0+dy,rf"$\bar{{d}}_{{\mathrm{{cf}}}}={d0:.3f}$",
                ha=ha,va=va,fontsize=16)
    fig.tight_layout(); save_figure(fig,out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default=str(ROOT/"artifacts"/"figure_data"/"fig3_amazon_robustness_figure_data.csv"))
    ap.add_argument("--out-dir",default=str(ROOT/"artifacts"/"figures"))
    ap.add_argument("--stem-prefix",default="amazon")
    args=ap.parse_args()
    configure_plot_style()
    rows=load_rows(Path(args.data)); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    stems=output_stems(args.stem_prefix)
    files=[out/f"{stem}.pdf" for stem in stems]
    plot_complementarity(rows,files[0]); plot_reference_history(rows,files[1]); plot_intervention_strength(rows,files[2])
    for p in files:
        print(f"Wrote: {p}"); print(f"Wrote: {p.with_suffix('.png')}")

if __name__=="__main__":
    main()
