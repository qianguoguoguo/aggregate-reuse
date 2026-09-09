#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.references import (
    exact_plugin_w1_baseline,
    exact_predictive_w1_baseline,
    item_cluster_bootstrap_mean,
    load_reference_table,
    load_stage3_blocks,
    shrunk_reference,
    w1_hist_1to5,
)
from aggregate_reuse.amazon.paths import AmazonPathLayout
from aggregate_reuse.amazon.provenance import (
    build_provenance,
    discover_raw_dataset_sha256,
    logical_path,
    require_artifact_category,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(ROOT / "configs" / "amazon_reference.yaml"),
    )
    ap.add_argument(
        "--shared-root",
        type=Path,
        help="Operational shared-root override; does not alter the config file.",
    )
    ap.add_argument(
        "--roles",
        default=None,
    )
    ap.add_argument(
        "--references",
        default=None,
    )
    ap.add_argument(
        "--out-dir",
        default=None,
    )
    ap.add_argument("--bootstrap-replicates", type=int, default=None)
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the shared canonical reference-calibration outputs.",
    )
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    layout = AmazonPathLayout.from_config(
        cfg,
        repo_root=ROOT,
        shared_root=args.shared_root,
    )
    shared = cfg["shared_data"]
    roles_path = (
        Path(args.roles).resolve()
        if args.roles is not None
        else layout.resolve_shared_path(shared["roles_file"])
    )
    refs_path = (
        Path(args.references).resolve()
        if args.references is not None
        else layout.resolve_shared_path(shared["reference_histograms"])
    )
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir is not None
        else layout.resolve_shared_path(shared["output_subdir"])
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    lambda_grid = [float(value) for value in cfg["lambda_grid"]]
    bootstrap_replicates = (
        int(args.bootstrap_replicates)
        if args.bootstrap_replicates is not None
        else int(cfg["bootstrap"]["replicates"])
    )
    raw_dataset_sha256 = discover_raw_dataset_sha256(layout)

    stage2_summary_path = layout.artifact("stage2_summary")
    if stage2_summary_path.is_file():
        stage2_summary = json.loads(
            stage2_summary_path.read_text(encoding="utf-8")
        )
        require_artifact_category(
            stage2_summary,
            expected=layout.category,
            label="Stage-2 summary",
        )

    outputs = {
        "calibration": out_dir / layout.artifact_name("stage3_lambda_calibration"),
        "holdout": out_dir / layout.artifact_name("stage3_holdout_validation"),
        "item_holdout": out_dir / layout.artifact_name(
            "stage3_holdout_item_metrics"
        ),
        "chronology": out_dir / layout.artifact_name(
            "stage3_selected_lambda_chronology"
        ),
        "freeze": out_dir / layout.artifact_name("stage3_reference_freeze"),
        "summary": out_dir / layout.artifact_name("stage3_summary"),
    }

    if not args.overwrite:
        for p in outputs.values():
            if p.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing Stage-3 output: {p}\n"
                    "Use --overwrite to regenerate the canonical shared reference outputs."
                )

    refs = load_reference_table(refs_path)
    blocks = load_stage3_blocks(roles_path)

    if set(refs) != set(blocks):
        raise RuntimeError("Reference and Stage-2 item universes differ.")

    asins = sorted(refs)
    n_items = len(asins)

    ref_counts = np.stack([refs[a][0] for a in asins])
    loo_probs = np.stack([refs[a][1] for a in asins])

    cal = np.stack([
        np.stack([
            blocks[a]["calibration_1"],
            blocks[a]["calibration_2"],
        ])
        for a in asins
    ])

    hold = np.stack([
        np.stack([
            blocks[a]["holdout_1"],
            blocks[a]["holdout_2"],
        ])
        for a in asins
    ])

    calibration_rows = []
    cached = {}

    def baselines_for(i, lam):
        key = (i, float(lam))
        if key not in cached:
            alpha, q = shrunk_reference(ref_counts[i], loo_probs[i], lam)
            cached[key] = (
                alpha,
                q,
                exact_plugin_w1_baseline(q, 30),
                exact_predictive_w1_baseline(alpha, 30),
            )
        return cached[key]

    print("Stage 3A: reference-shrinkage selection on calibration blocks only.")

    all_cal_metrics = {}

    for lam in lambda_grid:
        raw_vals = np.empty((n_items, 2), dtype=float)
        plugin_vals = np.empty((n_items, 2), dtype=float)
        pred_vals = np.empty((n_items, 2), dtype=float)

        for i in range(n_items):
            _, q, b_plugin, b_pred = baselines_for(i, lam)
            for b in range(2):
                raw = w1_hist_1to5(cal[i, b], q)
                raw_vals[i, b] = raw
                plugin_vals[i, b] = raw - b_plugin
                pred_vals[i, b] = raw - b_pred

        row = {
            "lambda": float(lam),
            "raw_mean": float(raw_vals.mean()),
            "plugin_signed_mean": float(plugin_vals.mean()),
            "plugin_abs_mean": float(abs(plugin_vals.mean())),
            "plugin_positive_fraction": float((plugin_vals > 0).mean()),
            "predictive_signed_mean": float(pred_vals.mean()),
            "predictive_abs_mean": float(abs(pred_vals.mean())),
            "predictive_positive_fraction": float((pred_vals > 0).mean()),
        }
        calibration_rows.append(row)
        all_cal_metrics[float(lam)] = (raw_vals, plugin_vals, pred_vals)

        print(
            f"  lambda={lam:g}: predictive mean="
            f"{row['predictive_signed_mean']:.8f}"
        )

    best = min(
        calibration_rows,
        key=lambda r: (r["predictive_abs_mean"], r["lambda"]),
    )
    selected_lambda = float(best["lambda"])

    with open(outputs["calibration"], "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(calibration_rows[0].keys()))
        writer.writeheader()
        writer.writerows(calibration_rows)

    print(f"\nFrozen reference lambda = {selected_lambda:g}")

    # ---------------------------------------------------------------
    # Stage 3B: untouched holdout as a DIAGNOSTIC, not a tuning gate.
    # ---------------------------------------------------------------
    raw_h = np.empty((n_items, 2), dtype=float)
    plugin_h = np.empty((n_items, 2), dtype=float)
    pred_h = np.empty((n_items, 2), dtype=float)

    item_rows = []

    for i, asin in enumerate(asins):
        _, q, b_plugin, b_pred = baselines_for(i, selected_lambda)
        for b in range(2):
            raw = w1_hist_1to5(hold[i, b], q)
            raw_h[i, b] = raw
            plugin_h[i, b] = raw - b_plugin
            pred_h[i, b] = raw - b_pred

        item_rows.append({
            "asin": asin,
            "lambda": selected_lambda,
            "holdout_1_raw_w1": raw_h[i, 0],
            "holdout_2_raw_w1": raw_h[i, 1],
            "holdout_1_plugin_signed": plugin_h[i, 0],
            "holdout_2_plugin_signed": plugin_h[i, 1],
            "holdout_1_predictive_signed": pred_h[i, 0],
            "holdout_2_predictive_signed": pred_h[i, 1],
        })

    with open(outputs["item_holdout"], "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(item_rows[0].keys()))
        writer.writeheader()
        writer.writerows(item_rows)

    holdout_rows = []
    for method, vals in [
        ("raw_w1", raw_h),
        ("plugin_centered_w1", plugin_h),
        ("predictive_centered_w1", pred_h),
    ]:
        boot = item_cluster_bootstrap_mean(
            vals,
            replicates=bootstrap_replicates,
            seed=int(cfg["bootstrap"]["seed"]),
            level=float(cfg["bootstrap"]["level"]),
        )
        holdout_rows.append({
            "method": method,
            **boot,
            "positive_fraction_all_blocks": float((vals > 0).mean()),
            "holdout_1_mean": float(vals[:, 0].mean()),
            "holdout_2_mean": float(vals[:, 1].mean()),
        })

    with open(outputs["holdout"], "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(holdout_rows[0].keys()))
        writer.writeheader()
        writer.writerows(holdout_rows)

    # ---------------------------------------------------------------
    # Chronological diagnostic at selected lambda:
    # calibration_1, calibration_2, holdout_1, holdout_2.
    # ---------------------------------------------------------------
    _, cal_plugin, cal_pred = all_cal_metrics[selected_lambda]

    chronology_rows = [
        {
            "role": "calibration_1",
            "predictive_signed_mean": float(cal_pred[:, 0].mean()),
            "plugin_signed_mean": float(cal_plugin[:, 0].mean()),
        },
        {
            "role": "calibration_2",
            "predictive_signed_mean": float(cal_pred[:, 1].mean()),
            "plugin_signed_mean": float(cal_plugin[:, 1].mean()),
        },
        {
            "role": "holdout_1",
            "predictive_signed_mean": float(pred_h[:, 0].mean()),
            "plugin_signed_mean": float(plugin_h[:, 0].mean()),
        },
        {
            "role": "holdout_2",
            "predictive_signed_mean": float(pred_h[:, 1].mean()),
            "plugin_signed_mean": float(plugin_h[:, 1].mean()),
        },
    ]

    with open(outputs["chronology"], "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(chronology_rows[0].keys()))
        writer.writeheader()
        writer.writerows(chronology_rows)

    pred_holdout = next(
        r for r in holdout_rows if r["method"] == "predictive_centered_w1"
    )

    zero_inside = (
        pred_holdout["ci_lower"] <= 0.0 <= pred_holdout["ci_upper"]
    )

    drift_diagnostic = {
        "predictive_centered_holdout_mean": pred_holdout["mean"],
        "ci_lower": pred_holdout["ci_lower"],
        "ci_upper": pred_holdout["ci_upper"],
        "zero_inside_95pct_ci": bool(zero_inside),
        "holdout_1_mean": pred_holdout["holdout_1_mean"],
        "holdout_2_mean": pred_holdout["holdout_2_mean"],
        "interpretation": (
            "Absolute posterior-predictive centering retains statistically "
            "detectable temporal drift on real Amazon traffic."
            if not zero_inside else
            "No statistically detectable residual drift under this diagnostic."
        ),
    }

    # ---------------------------------------------------------------
    # Canonical handoff to Stage 4.
    # ---------------------------------------------------------------
    freeze = {
        "schema_version": 1,
        "category": layout.category,
        "raw_dataset_sha256": raw_dataset_sha256,
        "selected_lambda": selected_lambda,
        "lambda_grid": lambda_grid,
        "reference_formula": "(c_i + lambda * H_-i) / (120 + lambda)",
        "reference_source_positions": [1, 120],
        "selection_source_roles": ["calibration_1", "calibration_2"],
        "holdout_used_for_selection": False,
        "experimental_A_B_used_for_selection": False,
        "stage4_evidence_mode": "paired_counterfactual",
        "stage4_evidence_definition": (
            "d_cf(i,t) = W1(H_attack(i,t), Hhat_i) "
            "- W1(H_clean(i,t), Hhat_i)"
        ),
        "baseline_cancellation_identity": (
            "[(W1_attack - b_i) - (W1_clean - b_i)] "
            "= W1_attack - W1_clean"
        ),
        "stage4_invariant_requirement": (
            "Across identity-reuse conditions, item/block set, treated slots, "
            "timestamps, replacement ratings, clean histograms, attacked "
            "histograms, d_cf, and total manipulated volume must remain fixed."
        ),
        "holdout_role": (
            "diagnostic only; never used to retune lambda or construct attacks"
        ),
    }
    freeze["provenance"] = build_provenance(
        category=layout.category,
        config_path=config_path,
        raw_dataset_sha256=raw_dataset_sha256,
        selected_lambda=selected_lambda,
        seed_ids=[],
        upstream_artifacts={
            "stage2_roles": roles_path,
            "stage2_reference_histograms": refs_path,
        },
        repo_root=ROOT,
        shared_root=layout.shared_category_root,
    )
    outputs["freeze"].write_text(
        json.dumps(freeze, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary = {
        "stage": "amazon_stage3_reference_calibration_and_drift_diagnostic",
        "category": layout.category,
        "raw_dataset_sha256": raw_dataset_sha256,
        "n_items": n_items,
        "lambda_grid": lambda_grid,
        "selection_rule": (
            "Minimize absolute mean predictive signed increment over "
            "calibration_1 and calibration_2; break ties toward smaller lambda."
        ),
        "selected_lambda": selected_lambda,
        "selected_calibration_row": best,
        "chronological_selected_lambda_diagnostic": chronology_rows,
        "absolute_null_diagnostic": drift_diagnostic,
        "holdout_validation": {
            row["method"]: {
                k: v for k, v in row.items() if k != "method"
            }
            for row in holdout_rows
        },
        "posterior_predictive_model": (
            "Dirichlet(alpha_i(lambda)) with alpha_i = item reference counts "
            "+ lambda * leave-one-item-out domain probabilities."
        ),
        "predictive_baseline": (
            "Exact Dirichlet-multinomial predictive W1 expectation from "
            "Beta-Binomial cumulative marginals."
        ),
        "semantic_decision": {
            "absolute_centering_used_as_primary_amazon_stage4_evidence": False,
            "reason": (
                "Untouched holdout exhibits residual temporal drift under "
                "absolute posterior-predictive centering."
            ),
            "stage4_evidence_mode": "paired_counterfactual",
        },
        "data_usage_guards": {
            "lambda_selection_uses_only_calibration_1_2": True,
            "holdout_used_for_lambda_selection": False,
            "experimental_A_B_used_in_stage3": False,
            "attack_generated_in_stage3": False,
            "lambda_retuned_after_holdout": False,
        },
        "outputs": {
            k: logical_path(
                v, repo_root=ROOT, shared_root=layout.shared_category_root
            )
            for k, v in outputs.items() if k != "summary"
        },
    }
    summary["provenance"] = build_provenance(
        category=layout.category,
        config_path=config_path,
        raw_dataset_sha256=raw_dataset_sha256,
        selected_lambda=selected_lambda,
        seed_ids=[],
        upstream_artifacts={
            "stage2_roles": roles_path,
            "stage2_reference_histograms": refs_path,
        },
        repo_root=ROOT,
        shared_root=layout.shared_category_root,
    )

    outputs["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\nStage 3 revised canonical run complete.")
    print(f"Selected lambda: {selected_lambda:g}")
    print(
        "Predictive holdout mean: "
        f"{drift_diagnostic['predictive_centered_holdout_mean']:.8f}"
    )
    print(
        "95% CI: "
        f"[{drift_diagnostic['ci_lower']:.8f}, "
        f"{drift_diagnostic['ci_upper']:.8f}]"
    )
    print(
        "Zero inside 95% CI: "
        f"{drift_diagnostic['zero_inside_95pct_ci']}"
    )
    print("Stage-4 evidence mode: paired_counterfactual")
    print(f"Freeze file: {outputs['freeze']}")
    print(f"Summary: {outputs['summary']}")


if __name__ == "__main__":
    main()
