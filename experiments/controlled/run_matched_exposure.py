#!/usr/bin/env python3
"""Run the controlled matched-exposure theory-validation experiment.

Phase 2E migrates ONLY the principal matched-exposure condition:
  R_exp = 1
  p_on = 0.2

Outputs:
  artifacts/per_seed/controlled_matched_per_seed.{json,csv}
  artifacts/per_seed/controlled_matched_invariants.csv
  artifacts/results/controlled_matched_summary.json
  artifacts/figure_data/controlled_theory_trajectory.csv
  artifacts/manifests/controlled_matched_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy
import sklearn
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.controlled.matched_exposure import (
    controlled_matched_exposure_divergence,
)
from aggregate_reuse.metrics import percentile_bootstrap_ci
from aggregate_reuse.seeding import derive_stage_seed


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def require_paper_environment(cfg: dict, allow_mismatch: bool) -> None:
    required = cfg.get("required_environment", {})
    actual = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "pyyaml": yaml.__version__,
    }
    bad = {
        k: (str(required[k]), str(actual.get(k)))
        for k in required
        if str(required[k]) != str(actual.get(k))
    }
    if bad and not allow_mismatch:
        details = "\n".join(
            f"  {k}: expected {e}, found {a}" for k, (e, a) in bad.items()
        )
        raise SystemExit(
            "Exact paper reproduction requires the frozen environment:\n"
            + details
            + "\nUse --allow-version-mismatch only for migration/debug checks."
        )


def scalar_summary(values, name, cfg):
    ci = percentile_bootstrap_ci(
        np.asarray(values, dtype=float),
        level=float(cfg["confidence_intervals"]["level"]),
        replicates=int(cfg["confidence_intervals"]["bootstrap_replicates"]),
        seed=derive_stage_seed(0, "bootstrap::" + name),
        statistic="mean",
    )
    return {
        "mean": ci["estimate"],
        "ci_lower": ci["lower"],
        "ci_upper": ci["upper"],
        "n_seeds": len(values),
    }


def curve_bootstrap(arr, name, cfg, batch=250):
    x = np.asarray(arr, dtype=float)
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError("curve array must be finite seeds x horizon")
    n, T = x.shape
    reps = int(cfg["confidence_intervals"]["bootstrap_replicates"])
    level = float(cfg["confidence_intervals"]["level"])
    rng = np.random.default_rng(derive_stage_seed(0, "curve_bootstrap::" + name))
    boot = np.empty((reps, T), dtype=np.float32)

    for start in range(0, reps, batch):
        b = min(batch, reps - start)
        idx = rng.integers(0, n, size=(b, n))
        boot[start:start+b] = (
            x[idx].mean(axis=1, dtype=np.float64).astype(np.float32)
        )

    alpha = 1.0 - level
    return (
        x.mean(axis=0),
        np.quantile(boot, alpha/2.0, axis=0),
        np.quantile(boot, 1.0-alpha/2.0, axis=0),
    )


def make_legacy_config(cfg: dict) -> dict:
    """Adapter only; preserves the exact historical function input schema."""
    cc = cfg["controlled_model"]
    mc = cfg["matched_exposure"]
    return {
        "controlled_model": {
            **cc,
            "campaign": {
                "principal_p_on": mc["campaign"]["principal_p_on"],
                "principal_exposure_ratio":
                    mc["campaign"]["principal_exposure_ratio"],
            },
            "coalition_distributions": {
                "mean_shift": mc["coalition_distribution"],
            },
        }
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(ROOT / "configs" / "controlled.yaml"),
    )
    ap.add_argument("--only-seed", type=int)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--allow-version-mismatch", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    require_paper_environment(cfg, args.allow_version_mismatch)

    seeds = list(range(int(cfg["seeds"]["start"]), int(cfg["seeds"]["stop"]) + 1))
    if args.only_seed is not None:
        if args.only_seed not in seeds:
            raise SystemExit("Requested seed is outside the frozen range.")
        seeds = [args.only_seed]

    exp_cfg = cfg["matched_exposure"]
    stage_name = exp_cfg["random_stage_name"]

    run_root = ROOT / "artifacts" / "results" / "controlled_matched_runs"
    per_seed_dir = ROOT / "artifacts" / "per_seed"
    result_dir = ROOT / "artifacts" / "results"
    figure_dir = ROOT / "artifacts" / "figure_data"
    manifest_dir = ROOT / "artifacts" / "manifests"

    if args.clean and run_root.exists():
        shutil.rmtree(run_root)

    for p in [run_root, per_seed_dir, result_dir, figure_dir, manifest_dir]:
        p.mkdir(parents=True, exist_ok=True)

    rows = []
    inv_rows = []
    empirical = []
    predicted = []
    pair_error = []

    legacy_cfg = make_legacy_config(cfg)

    for seed in seeds:
        stage_seed = derive_stage_seed(seed, stage_name)
        run_dir = run_root / f"seed_{seed}"
        summary_path = run_dir / "summary.json"
        data_path = run_dir / "data" / "controlled_matched_exposure_divergence.npz"

        if summary_path.exists() and data_path.exists():
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = saved["metrics"]
            manifest = saved["manifest"]
        else:
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True)

            metrics, manifest = controlled_matched_exposure_divergence(
                legacy_cfg, seed, run_dir, stage_seed
            )
            summary_path.write_text(
                json.dumps(
                    {"metrics": metrics, "manifest": manifest},
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

        rows.append(metrics)
        inv_rows.append({"seed": seed, **manifest["invariants"]})

        z = np.load(data_path)
        empirical.append(np.asarray(z["empirical_gap"], dtype=np.float64))
        predicted.append(np.asarray(z["predicted_gap"], dtype=np.float64))
        pair_error.append(np.asarray(z["pair_error"], dtype=np.float64))

    (per_seed_dir / "controlled_matched_per_seed.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )

    fields = sorted({k for row in rows for k in row})
    with (per_seed_dir / "controlled_matched_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    with (per_seed_dir / "controlled_matched_invariants.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields_i = list(inv_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields_i)
        w.writeheader()
        w.writerows(inv_rows)

    if len(seeds) != 30:
        print(f"Completed matched-exposure migration check for seed {seeds[0]}.")
        return 0

    exp = "controlled_matched_exposure_divergence"
    metric_names = [
        "realized_R_exp",
        "realized_p_on",
        "frequency_auc",
        "score_auc",
        "empirical_final_gap",
        "predicted_final_gap",
        "Delta_hat",
        "empirical_gap_slope",
        "slope_relative_error",
        "final_pair_misordering",
    ]
    summary = {
        "experiment": exp,
        "n_seeds": 30,
        "metrics": {
            m: scalar_summary([row[m] for row in rows], f"{exp}::{m}", cfg)
            for m in metric_names
        },
    }
    (result_dir / "controlled_matched_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    em, el, eu = curve_bootstrap(
        np.stack(empirical), "matched_empirical_gap", cfg
    )
    pm, pl, pu = curve_bootstrap(
        np.stack(predicted), "matched_predicted_gap", cfg
    )
    qm, ql, qu = curve_bootstrap(
        np.stack(pair_error), "matched_pair_error", cfg
    )

    with (figure_dir / "controlled_theory_trajectory.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow([
            "horizon",
            "empirical_gap_mean",
            "empirical_gap_ci_lower",
            "empirical_gap_ci_upper",
            "predicted_gap_mean",
            "predicted_gap_ci_lower",
            "predicted_gap_ci_upper",
            "pair_error_mean",
            "pair_error_ci_lower",
            "pair_error_ci_upper",
        ])
        for i in range(len(em)):
            w.writerow([
                i+1,
                em[i], el[i], eu[i],
                pm[i], pl[i], pu[i],
                qm[i], ql[i], qu[i],
            ])

    all_inv = all(
        all(bool(v) for k, v in row.items() if k != "seed")
        for row in inv_rows
    )
    manifest = {
        "experiment": "controlled_matched_exposure",
        "seeds": seeds,
        "config": str(config_path.relative_to(ROOT)),
        "stage_seed_name": stage_name,
        "all_run_invariants_pass": all_inv,
        "outputs": {
            "per_seed_json": "artifacts/per_seed/controlled_matched_per_seed.json",
            "per_seed_csv": "artifacts/per_seed/controlled_matched_per_seed.csv",
            "invariants_csv": "artifacts/per_seed/controlled_matched_invariants.csv",
            "summary": "artifacts/results/controlled_matched_summary.json",
            "figure_data":
                "artifacts/figure_data/controlled_theory_trajectory.csv",
        },
    }
    (manifest_dir / "controlled_matched_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    if not all_inv:
        raise RuntimeError("At least one matched-exposure invariant failed.")

    print("Controlled matched exposure complete: 30/30 seeds.")
    print("All invariants: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
