#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_SIZE = (8.6, 4.6)
PNG_DPI = 400


def output_stems(prefix: str) -> tuple[str, str, str]:
    """Return safe category-specific stems while preserving Amazon defaults."""
    if re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", prefix) is None:
        raise ValueError(f"Invalid figure stem prefix: {prefix!r}")
    return (
        f"fig2a_{prefix}_reuse",
        f"fig2b_{prefix}_matched",
        f"fig2c_{prefix}_shape",
    )


def configure_plot_style() -> None:
    """Match the typography and source-canvas geometry used by Figure 1."""
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


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(x: str) -> float:
    return float(x)


def save_figure(fig: plt.Figure, out_pdf: Path) -> None:
    """Save PDF and PNG while preserving the nominal 8.6 x 4.6 canvas."""
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_pdf.with_suffix(".png")

    # Do not use bbox_inches='tight': it changes the saved aspect/geometry.
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=PNG_DPI)
    plt.close(fig)


def save_reuse(rows: list[dict[str, str]], out: Path) -> None:
    score = sorted(
        [r for r in rows if r["panel"] == "a_reuse" and r["metric"] == "score_auc"],
        key=lambda r: int(r["x"]),
    )
    freq = sorted(
        [r for r in rows if r["panel"] == "a_reuse" and r["metric"] == "frequency_auc"],
        key=lambda r: int(r["x"]),
    )

    x = np.asarray([int(r["x"]) for r in score])
    ys = np.asarray([f(r["mean"]) for r in score])
    yf = np.asarray([f(r["mean"]) for r in freq])

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        x,
        ys,
        marker="o",
        linewidth=2.4,
        markersize=7.0,
        label=r"Evidence score",
    )
    ax.plot(
        x,
        yf,
        marker="s",
        linestyle="--",
        linewidth=2.4,
        markersize=7.0,
        label=r"Frequency",
    )
    ax.axhline(0.5, linewidth=1.2, linestyle=":")

    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_ylim(0.45, 1.03)
    ax.set_xlabel(r"Identity reuse $r$")
    ax.set_ylabel(r"ROC--AUC")
    ax.grid(alpha=0.15)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        borderaxespad=0.0,
        frameon=False,
    )

    fig.tight_layout()
    save_figure(fig, out)


def save_matched(rows: list[dict[str, str]], out: Path) -> None:
    rr = [r for r in rows if r["panel"] == "b_matched"]
    order = ["Frequency", "Counterfactual", "Raw world", "Predictive"]
    by = {r["x"]: r for r in rr}

    y = np.asarray([f(by[k]["mean"]) for k in order])
    lo = np.asarray([f(by[k]["ci_lower"]) for k in order])
    hi = np.asarray([f(by[k]["ci_upper"]) for k in order])
    err = np.vstack([y - lo, hi - y])

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = np.arange(len(order))

    ax.bar(
        x,
        y,
        yerr=err,
        capsize=5,
        error_kw={"elinewidth": 1.8, "capthick": 1.8},
    )
    ax.axhline(0.5, linewidth=1.2, linestyle=":")
    ax.set_ylim(0.45, 0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.set_ylabel(r"ROC--AUC")
    ax.grid(axis="y", alpha=0.15)

    fig.tight_layout()
    save_figure(fig, out)


def save_shape(rows: list[dict[str, str]], out: Path) -> None:
    rr = [r for r in rows if r["panel"] == "c_shape"]
    order = ["Frequency", "Mean", r"$W_1$", "JS"]
    by = {r["x"]: r for r in rr}

    # Lookup keys remain the literal CSV strings even though W1 is typeset as W_1.
    csv_order = ["Frequency", "Mean", "W1", "JS"]
    y = np.asarray([f(by[k]["mean"]) for k in csv_order])
    lo = np.asarray([f(by[k]["ci_lower"]) for k in csv_order])
    hi = np.asarray([f(by[k]["ci_upper"]) for k in csv_order])
    err = np.vstack([y - lo, hi - y])

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = np.arange(len(order))

    ax.bar(
        x,
        y,
        yerr=err,
        capsize=5,
        error_kw={"elinewidth": 1.8, "capthick": 1.8},
    )
    ax.axhline(0.5, linewidth=1.2, linestyle=":")
    ax.set_ylim(0.45, 1.01)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel(r"ROC--AUC")
    ax.grid(axis="y", alpha=0.15)

    fig.tight_layout()
    save_figure(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default=str(
            ROOT / "artifacts" / "figure_data" / "amazon_main_figure_data.csv"
        ),
    )
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "artifacts" / "figures"),
    )
    ap.add_argument(
        "--stem-prefix",
        default="amazon",
        help="Filename component used between the panel number and metric name.",
    )
    args = ap.parse_args()

    configure_plot_style()

    rows = load_rows(Path(args.data))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reuse_stem, matched_stem, shape_stem = output_stems(args.stem_prefix)

    save_reuse(rows, out_dir / f"{reuse_stem}.pdf")
    save_matched(rows, out_dir / f"{matched_stem}.pdf")
    save_shape(rows, out_dir / f"{shape_stem}.pdf")

    for stem in (reuse_stem, matched_stem, shape_stem):
        print(f"Wrote: {out_dir / (stem + '.pdf')}")
        print(f"Wrote: {out_dir / (stem + '.png')}")


if __name__ == "__main__":
    main()
