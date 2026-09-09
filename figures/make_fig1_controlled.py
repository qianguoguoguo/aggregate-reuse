#!/usr/bin/env python3
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "artifacts" / "figure_data"
OUT_DIR = ROOT / "artifacts" / "figures"
FIG_SIZE = (8.6, 4.6)
PNG_DPI = 400

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
        "lines.markersize": 4.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def read_csv(name):
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save_figure(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=PNG_DPI)
    plt.close(fig)

def make_fig1a():
    rows = read_csv("controlled_null_trajectory.csv")
    x = np.asarray([float(r["horizon"]) for r in rows])
    raw = np.asarray([float(r["raw_mean"]) for r in rows])
    raw_lo = np.asarray([float(r["raw_ci_lower"]) for r in rows])
    raw_hi = np.asarray([float(r["raw_ci_upper"]) for r in rows])
    cen = np.asarray([float(r["centered_mean"]) for r in rows])
    cen_lo = np.asarray([float(r["centered_ci_lower"]) for r in rows])
    cen_hi = np.asarray([float(r["centered_ci_upper"]) for r in rows])

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(x, raw, label=r"Raw $W_1$", lw=2.0)
    ax.fill_between(x, raw_lo, raw_hi, alpha=0.18)
    ax.plot(x, cen, label=r"Centered $W_1-b_W$", lw=2.0)
    ax.fill_between(x, cen_lo, cen_hi, alpha=0.18)
    ax.axhline(0.0, color="black", linewidth=0.7, linestyle=":")
    ax.set_xlabel(r"Interval")
    ax.set_ylabel(r"Cumulative evidence")
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.margins(x=0)
    ax.grid(alpha=0.15)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98),
              borderaxespad=0.0, frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig1a_controlled_null")

def make_fig1b():
    rows = read_csv("controlled_theory_trajectory.csv")
    x = np.asarray([float(r["horizon"]) for r in rows])
    emp = np.asarray([float(r["empirical_gap_mean"]) for r in rows])
    emp_lo = np.asarray([float(r["empirical_gap_ci_lower"]) for r in rows])
    emp_hi = np.asarray([float(r["empirical_gap_ci_upper"]) for r in rows])
    pred = np.asarray([float(r["predicted_gap_mean"]) for r in rows])
    err = np.asarray([float(r["pair_error_mean"]) for r in rows])

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    line_emp, = ax.plot(x, emp, label=r"Empirical score gap", lw=2.4)
    ax.fill_between(x, emp_lo, emp_hi, alpha=0.18)
    line_pred, = ax.plot(x, pred, linestyle="--",
                         label=r"Predicted $t\widehat{\Delta}$", lw=2.4)
    ax.set_xlabel(r"Interval")
    ax.set_ylabel(r"Coalition--normal score gap")
    x_min, x_max = float(np.min(x)), float(np.max(x))
    x_pad = 0.02 * (x_max - x_min)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.grid(alpha=0.15)

    ax2 = ax.twinx()
    positive_idx = np.flatnonzero(err > 0)
    line_err, = ax2.plot(
        x, err, linestyle=":", linewidth=2.2, marker="o",
        markersize=6.5, markeredgewidth=0.0, markevery=positive_idx,
        label=r"Index-paired non-win rate", zorder=4,
    )
    ax2.set_ylabel(r"Index-paired non-win rate")
    y2_min, y2_max = -0.04, 1.02
    ax2.set_ylim(y2_min, y2_max)
    clip_rect = Rectangle(
        (x_min - x_pad, 0.0), (x_max - x_min) + 2*x_pad, y2_max,
        transform=ax2.transData
    )
    line_err.set_clip_path(clip_rect)
    ax.legend(handles=[line_emp, line_pred, line_err],
              loc="upper left", frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig1b_controlled_theory")

def make_fig1c():
    rows = read_csv("controlled_regime_auc.csv")
    rs = sorted({float(r["requested_R_exp"]) for r in rows})
    ps = sorted({float(r["requested_p_on"]) for r in rows})
    R = {v:i for i,v in enumerate(rs)}
    P = {v:j for j,v in enumerate(ps)}
    auc = np.full((len(rs), len(ps)), np.nan)
    for row in rows:
        auc[R[float(row["requested_R_exp"])], P[float(row["requested_p_on"])]] = float(row["score_auc_mean"])

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    im = ax.imshow(auc, aspect="auto", origin="lower", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels([f"{p:g}" for p in ps])
    ax.set_yticks(range(len(rs)))
    ax.set_yticklabels([f"{r:g}" for r in rs])
    ax.set_xlabel(r"$p_{\mathrm{on}}$")
    ax.set_ylabel(r"$R_{\mathrm{exp}}$")
    if 1.0 in R:
        ax.axhline(R[1.0], linewidth=1.0, linestyle="--")
    if 1.0 in R and 0.2 in P:
        ax.plot(P[0.2], R[1.0], marker="o", markersize=3.8,
                markerfacecolor="none", markeredgewidth=1.0)
    for idx, label in enumerate(ax.get_xticklabels()):
        label.set_visible(idx % 2 == 1 or idx == len(ps)-1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(r"Score ROC--AUC", labelpad=6, fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    fig.tight_layout()
    save_figure(fig, "fig1c_controlled_regime")

def main():
    configure_plot_style()
    make_fig1a()
    make_fig1b()
    make_fig1c()
    for stem in ["fig1a_controlled_null","fig1b_controlled_theory","fig1c_controlled_regime"]:
        print(f"Wrote: {OUT_DIR / (stem+'.pdf')}")
        print(f"Wrote: {OUT_DIR / (stem+'.png')}")

if __name__ == "__main__":
    main()
