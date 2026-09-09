#!/usr/bin/env python3
"""Run the controlled normal-only null-calibration experiment.

Phase 2D public entry point. This is the ONLY controlled experiment migrated
in this phase.

Outputs:
  artifacts/per_seed/controlled_null_per_seed.{json,csv}
  artifacts/results/controlled_null_summary.json
  artifacts/figure_data/controlled_null_trajectory.csv
  artifacts/manifests/controlled_null_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import platform
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.controlled.null_calibration import controlled_null_centering
from aggregate_reuse.metrics import percentile_bootstrap_ci
from aggregate_reuse.seeding import derive_stage_seed


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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

    for s in range(0, reps, batch):
        b = min(batch, reps - s)
        idx = rng.integers(0, n, size=(b, n))
        boot[s:s+b] = x[idx].mean(axis=1, dtype=np.float64).astype(np.float32)

    alpha = 1.0 - level
    return (
        x.mean(axis=0),
        np.quantile(boot, alpha / 2.0, axis=0),
        np.quantile(boot, 1.0 - alpha / 2.0, axis=0),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "controlled.yaml"))
    ap.add_argument("--only-seed", type=int, default=None)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--allow-version-mismatch", action="store_true",
                    help="Debug only: run even if paper package versions differ.")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    import scipy
    import sklearn
    required = cfg.get("required_environment", {})
    actual_env = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "pyyaml": yaml.__version__,
    }
    mismatches = {
        k: (str(required[k]), str(actual_env[k]))
        for k in required
        if str(required[k]) != str(actual_env.get(k))
    }
    if mismatches and not args.allow_version_mismatch:
        details = "\n".join(
            f"  {k}: expected {e}, found {a}"
            for k, (e, a) in mismatches.items()
        )
        raise SystemExit(
            "Exact paper reproduction requires the frozen environment:\n"
            + details
            + "\nUse --allow-version-mismatch only for migration/debug checks."
        )

    seeds = list(range(int(cfg["seeds"]["start"]), int(cfg["seeds"]["stop"]) + 1))
    if args.only_seed is not None:
        if args.only_seed not in seeds:
            raise SystemExit(f"Seed {args.only_seed} is not in frozen seed range.")
        seeds = [args.only_seed]

    run_root = ROOT / "artifacts" / "results" / "controlled_null_runs"
    per_seed_dir = ROOT / "artifacts" / "per_seed"
    result_dir = ROOT / "artifacts" / "results"
    figure_data_dir = ROOT / "artifacts" / "figure_data"
    manifest_dir = ROOT / "artifacts" / "manifests"

    if args.clean and run_root.exists():
        shutil.rmtree(run_root)

    for p in [run_root, per_seed_dir, result_dir, figure_data_dir, manifest_dir]:
        p.mkdir(parents=True, exist_ok=True)

    rows = []
    raw_curves = []
    signed_curves = []
    invariant_rows = []

    # Adapter to preserve the exact historical config shape expected by the
    # validated implementation.
    legacy_cfg = {
        "controlled_model": cfg["controlled_model"],
    }

    for seed in seeds:
        stage_seed = derive_stage_seed(
            seed, cfg["implementation"]["random_stage_name"]
        )
        run_dir = run_root / f"seed_{seed}"
        summary_path = run_dir / "summary.json"
        data_path = run_dir / "data" / "controlled_null_centering.npz"

        # Resume support is operational only: an already completed seed is
        # reused verbatim.  --clean removes the whole run root before starting.
        if summary_path.exists() and data_path.exists():
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = saved["metrics"]
            manifest = saved["manifest"]
        else:
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True)

            metrics, manifest = controlled_null_centering(
                legacy_cfg, seed, run_dir, stage_seed
            )
            summary_path.write_text(
                json.dumps({"metrics": metrics, "manifest": manifest},
                           indent=2, sort_keys=True),
                encoding="utf-8",
            )

        rows.append(metrics)
        invariant_rows.append({
            "seed": seed,
            **manifest["invariants"],
        })

        z = np.load(data_path)
        raw_curves.append(np.asarray(z["cumulative_raw"], dtype=np.float64))
        signed_curves.append(np.asarray(z["cumulative_signed"], dtype=np.float64))

    # Always emit per-seed outputs.
    (per_seed_dir / "controlled_null_per_seed.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    fields = sorted({k for row in rows for k in row})
    with (per_seed_dir / "controlled_null_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    with (per_seed_dir / "controlled_null_invariants.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields_inv = list(invariant_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields_inv)
        w.writeheader()
        w.writerows(invariant_rows)

    # A one-seed debug run stops here by design.
    if len(seeds) != 30:
        print(f"Completed controlled null calibration for seed {seeds[0]}.")
        return 0

    metric_names = [
        "mean_matched_baseline",
        "mean_raw_w1",
        "mean_signed_increment",
        "raw_cumulative_slope",
        "signed_cumulative_slope",
        "final_raw_cumulative",
        "final_signed_cumulative",
    ]
    exp = "controlled_null_centering"
    summary = {
        "experiment": exp,
        "n_seeds": 30,
        "metrics": {
            m: scalar_summary(
                [r[m] for r in rows],
                f"{exp}::{m}",
                cfg,
            )
            for m in metric_names
        },
    }

    (result_dir / "controlled_null_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    raw_mean, raw_lo, raw_hi = curve_bootstrap(
        np.stack(raw_curves), "null_raw", cfg
    )
    signed_mean, signed_lo, signed_hi = curve_bootstrap(
        np.stack(signed_curves), "null_signed", cfg
    )

    with (figure_data_dir / "controlled_null_trajectory.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow([
            "horizon",
            "raw_mean", "raw_ci_lower", "raw_ci_upper",
            "centered_mean", "centered_ci_lower", "centered_ci_upper",
        ])
        for i in range(len(raw_mean)):
            w.writerow([
                i + 1,
                raw_mean[i], raw_lo[i], raw_hi[i],
                signed_mean[i], signed_lo[i], signed_hi[i],
            ])

    all_invariants = all(
        all(bool(v) for k, v in row.items() if k != "seed")
        for row in invariant_rows
    )
    run_manifest = {
        "experiment": "controlled_null_calibration",
        "seeds": seeds,
        "config": str(config_path.relative_to(ROOT)),
        "implementation_stage_seed_name":
            cfg["implementation"]["random_stage_name"],
        "baseline_method": cfg["implementation"]["baseline_method"],
        "all_run_invariants_pass": all_invariants,
        "outputs": {
            "per_seed_json": "artifacts/per_seed/controlled_null_per_seed.json",
            "per_seed_csv": "artifacts/per_seed/controlled_null_per_seed.csv",
            "invariants_csv": "artifacts/per_seed/controlled_null_invariants.csv",
            "summary": "artifacts/results/controlled_null_summary.json",
            "figure_data": "artifacts/figure_data/controlled_null_trajectory.csv",
        },
    }
    (manifest_dir / "controlled_null_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if not all_invariants:
        raise RuntimeError("At least one controlled-null invariant failed.")

    print("Controlled null calibration complete: 30/30 seeds.")
    print("All invariants: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
