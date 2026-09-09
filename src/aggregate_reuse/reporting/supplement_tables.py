"""Canonical numerical supplement-table extraction and rendering.

Every numerical value is read from a canonical CSV/JSON output.  This module
contains formatting rules and table structure only; it contains no frozen
scientific result constants.

Static/specification tables are intentionally out of scope for Phase 2S.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f3(x: float) -> str:
    return f"{float(x):.3f}"


def f4(x: float) -> str:
    return f"{float(x):.4f}"


def f4_signed(x: float) -> str:
    return f"{float(x):.4f}"


def fint(x: float | int) -> str:
    return f"{int(round(float(x))):,}"


def table_reuse(stage4_summary: Dict[str, Any]):
    rows = []
    for r in ["1", "2", "4", "8", "16"]:
        x = stage4_summary["reuse_summary"][r]
        rows.append([
            r,
            f3(x["score_auc"]["mean"]),
            f3(x["frequency_auc"]["mean"]),
        ])
    return rows


def table_twins(stage5_summary: Dict[str, Any]):
    m = stage5_summary["metrics"]
    return {
        "Frequency ROC--AUC": f3(m["frequency_auc"]["mean"]),
        "Counterfactual ROC--AUC": f3(m["counterfactual_score_auc"]["mean"]),
        "Raw-world ROC--AUC": f3(m["raw_world_score_auc"]["mean"]),
        "Posterior-predictive ROC--AUC":
            f3(m["predictive_centered_world_score_auc"]["mean"]),
        "Matched-pair misordering":
            f3(m["paired_gap_nonpositive_fraction"]["mean"]),
        "Mean paired score gap": f"{m['mean_paired_score_gap']['mean']:.4f}",
    }


def table_shape(stage6_summary: Dict[str, Any]):
    m = stage6_summary["metrics"]
    return [
        ["Frequency", f3(m["frequency_auc"]["mean"]), "1.000"],
        ["Mean deviation",
         f3(m["mean_counterfactual_score_auc"]["mean"]),
         f3(m["mean_paired_misordering_probability"]["mean"])],
        ["Wasserstein--1",
         f3(m["w1_counterfactual_score_auc"]["mean"]),
         f3(m["w1_paired_misordering_probability"]["mean"])],
        ["Jensen--Shannon",
         f3(m["js_counterfactual_score_auc"]["mean"]),
         f3(m["js_paired_misordering_probability"]["mean"])],
    ]


def table_complementarity(stage7_summary: Dict[str, Any]):
    m = stage7_summary["metrics"]
    return [
        ["Evidence only",
         f3(m["evidence_only_aggregate_auc"]["mean"]),
         f3(m["evidence_only_coactivity_auc"]["mean"]),
         "--"],
        ["Topology only",
         f3(m["topology_only_aggregate_auc"]["mean"]),
         f3(m["topology_only_coactivity_auc"]["mean"]),
         "--"],
        ["Mixed",
         f3(m["aggregate_auc"]["mean"]),
         f3(m["coactivity_auc"]["mean"]),
         f3(m["combined_auc"]["mean"])],
    ]


def table_reference_history(stage8_summary: Dict[str, Any]):
    out = []
    for nref in ["60", "90", "120"]:
        x = stage8_summary["metrics_by_reference_length"][nref]
        out.append([
            nref,
            f"{float(x['selected_lambda']):.0f}",
            f3(x["counterfactual_score_auc"]["mean"]),
            f3(x["paired_misordering_probability"]["mean"]),
        ])
    return out


def table_strength(stage9_summary: Dict[str, Any]):
    out = []
    for k in ["3", "6", "9"]:
        x = stage9_summary["metrics_by_k"][k]
        out.append([
            k,
            f4_signed(x["mean_d_cf"]["mean"]),
            f3(x["positive_block_fraction"]["mean"]),
            f3(x["counterfactual_score_auc"]["mean"]),
            f3(x["paired_misordering_probability"]["mean"]),
        ])
    return out


def table_k6_population(stage5_summary, stage9_summary, population_audit):
    p = population_audit["population_results"]
    return [
        [
            "Primary",
            fint(p["k6_feasible_population"]["population_size"]),
            f4(stage5_summary["metrics"]["mean_d_cf"]["mean"]),
            f3(stage5_summary["metrics"]["counterfactual_score_auc"]["mean"]),
        ],
        [
            "Strength sensitivity",
            fint(p["k9_feasible_population"]["population_size"]),
            f4(stage9_summary["metrics_by_k"]["6"]["mean_d_cf"]["mean"]),
            f3(stage9_summary["metrics_by_k"]["6"]["counterfactual_score_auc"]["mean"]),
        ],
    ]


def table_self_influence(stage11_summary: Dict[str, Any]):
    m = stage11_summary["metrics"]
    return [
        ["ROC--AUC",
         f3(m["original_matched_twin_auc"]["mean"]),
         f3(m["loo_matched_twin_auc"]["mean"])],
        ["Mean paired gap",
         f3(m["original_mean_paired_gap"]["mean"]),
         f3(m["loo_mean_paired_gap"]["mean"])],
        ["Misordering probability",
         f3(m["original_misordering_probability"]["mean"]),
         f3(m["loo_misordering_probability"]["mean"])],
    ]


def table_one_world(stage10_summary: Dict[str, Any]):
    m = stage10_summary["metrics"]

    channels = [
        ("Frequency", "frequency"),
        ("Raw W1", "raw_w1"),
        ("Predictive W1", "predictive_centered_w1"),
    ]

    full = []
    for label, key in channels:
        x = m[key]["all"]
        full.append((label, {
            "Median background percentile":
                f"{x['planted_background_percentile_median']['mean']:.5f}",
            "Planted in top 1%": f3(x["top_0p01_planted_capture"]["mean"]),
            "Planted in top 5%": f3(x["top_0p05_planted_capture"]["mean"]),
            "Accounts inspected for 50%": fint(x["burden_0p5_accounts"]["mean"]),
            "Inspection fraction for 50%":
                f"{x['burden_0p5_fraction_of_universe']['mean']:.5f}",
        }))

    freq8 = []
    for label, key in channels:
        x = m[key]["freq_eq_8"]
        freq8.append((label, {
            "Median background percentile":
                f3(x["planted_background_percentile_median"]["mean"]),
            "25th--75th percentile":
                f"{x['planted_background_percentile_q25']['mean']:.3f}--"
                f"{x['planted_background_percentile_q75']['mean']:.3f}",
            "Above background 95th percentile":
                f3(x["fraction_planted_above_background_p95"]["mean"]),
            "Above background 99th percentile":
                f3(x["fraction_planted_above_background_p99"]["mean"]),
            "Background accounts above planted median":
                (
                    "0"
                    if abs(float(
                        x["background_accounts_strictly_above_planted_median_score"]["mean"]
                    )) < 0.5
                    else f"{float(x['background_accounts_strictly_above_planted_median_score']['mean']):.1f}"
                ),
        }))

    full_rows = []
    row_names = [
        "Median background percentile",
        "Planted in top 1%",
        "Planted in top 5%",
        "Accounts inspected for 50%",
        "Inspection fraction for 50%",
    ]
    fdict = {label: vals for label, vals in full}
    for row in row_names:
        full_rows.append([
            row,
            fdict["Frequency"][row],
            fdict["Raw W1"][row],
            fdict["Predictive W1"][row],
        ])

    freq8_rows = []
    row_names = [
        "Median background percentile",
        "25th--75th percentile",
        "Above background 95th percentile",
        "Above background 99th percentile",
        "Background accounts above planted median",
    ]
    fdict = {label: vals for label, vals in freq8}
    for row in row_names:
        freq8_rows.append([
            row,
            fdict["Frequency"][row],
            fdict["Raw W1"][row],
            fdict["Predictive W1"][row],
        ])

    return {"full": full_rows, "freq_eq_8": freq8_rows}


def table_controlled_grid(rows: List[Dict[str, str]]):
    by_R: Dict[float, Dict[float, float]] = {}
    for row in rows:
        R = float(row["requested_R_exp"])
        p = float(row["requested_p_on"])
        by_R.setdefault(R, {})[p] = float(row["score_auc_mean"])

    out = {}
    for R in sorted(by_R):
        key = f"{R:.2f}"
        out[key] = [f3(by_R[R][p]) for p in sorted(by_R[R])]
    return out


def tex_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def render_simple_tabular(headers: List[str], rows: List[List[str]],
                          alignment: str | None = None) -> str:
    if alignment is None:
        alignment = "l" + "c" * (len(headers) - 1)
    lines = [
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(x) for x in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def render_key_value_tabular(rows: Dict[str, str]) -> str:
    return render_simple_tabular(
        ["Metric", "Value"],
        [[k, v] for k, v in rows.items()],
        alignment="lc",
    )
