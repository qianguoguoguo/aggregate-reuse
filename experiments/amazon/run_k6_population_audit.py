#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.primary_attack import (
    load_experimental_blocks,
    load_json,
    load_reference_table,
)
from aggregate_reuse.amazon.population_audit import (
    exact_population_expected_mean,
)
from aggregate_reuse.amazon.paths import AmazonPathLayout
from aggregate_reuse.amazon.provenance import (
    build_provenance,
    require_artifact_category,
    resolve_raw_dataset_sha256,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(ROOT / "configs" / "amazon_k6_population_audit.yaml"),
    )
    ap.add_argument(
        "--shared-root",
        type=Path,
        help="Operational shared-root override; does not alter the config file.",
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
    roles_path = layout.resolve_shared_path(shared["roles_file"])
    refs_path = layout.resolve_shared_path(shared["reference_histograms"])
    freeze_path = layout.resolve_shared_path(shared["reference_freeze"])
    out = layout.resolve_shared_path(shared["output_subdir"])
    out.mkdir(parents=True, exist_ok=True)

    blocks = load_experimental_blocks(roles_path)
    refs = load_reference_table(refs_path)
    freeze = load_json(freeze_path)
    require_artifact_category(
        freeze,
        expected=cfg["category"],
        label="Stage-3 freeze",
        allow_legacy_home_missing=False,
    )

    lam = float(freeze["selected_lambda"])
    k = int(cfg["k"])

    rows = []
    population_results = {}

    for name, threshold in cfg["population_rules"].items():
        print(f"Computing exact k=6 population expectation: {name}")
        result = exact_population_expected_mean(
            blocks=blocks,
            refs=refs,
            lam=lam,
            threshold=int(threshold),
            k=k,
        )
        population_results[name] = {
            "population_size": int(result["population_size"]),
            "exact_expected_mean_d_cf": float(
                result["exact_expected_mean_d_cf"]
            ),
            "threshold_nonfive_each_block": int(threshold),
        }
        rows.append({
            "population": name,
            "threshold_nonfive_each_block": int(threshold),
            "population_size": int(result["population_size"]),
            "exact_expected_mean_d_cf": float(
                result["exact_expected_mean_d_cf"]
            ),
        })

    with (out / layout.artifact_name("stage9_k6_population_expectations")).open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "experiment": "amazon_same_k6_population_audit",
        "category": cfg["category"],
        "k": k,
        "selected_lambda": lam,
        "population_results": population_results,
        "invariants": {
            "same_intervention_k6_in_both_populations": True,
            "same_reference_lambda": True,
            "only_feasibility_population_rule_changes": True,
            "exact_population_expectation_uses_no_monte_carlo": True,
            "treatment_block_integrated_exactly_as_half_A_half_B": True,
            "uniform_six_donor_selection_integrated_exactly": True,
        },
        "interpretation_guard": (
            "This is a deterministic population-selection audit. It isolates "
            "the effect of changing only the feasibility population while "
            "holding the intervention at k=6. It does not attempt to reproduce "
            "legacy exploratory finite-sample RNG outputs whose exact protocol "
            "was not retained."
        ),
        "legacy_empirical_values": {
            "status": "not_used_as_reproduction_targets",
            "reason": (
                "The trimmed legacy repository retained only rounded values and "
                "no exact finite-sample protocol. A dedicated diagnostic found "
                "no exact match among retained Stage-4/Stage-9 conventions."
            ),
        },
    }
    raw_dataset_sha256 = resolve_raw_dataset_sha256(layout, freeze)
    summary["raw_dataset_sha256"] = raw_dataset_sha256
    summary["provenance"] = build_provenance(
        category=layout.category,
        config_path=config_path,
        raw_dataset_sha256=raw_dataset_sha256,
        selected_lambda=lam,
        seed_ids=[],
        upstream_artifacts={
            "roles_file": roles_path,
            "reference_histograms": refs_path,
            "reference_freeze": freeze_path,
        },
        repo_root=ROOT,
        shared_root=layout.shared_category_root,
    )

    (out / layout.artifact_name("stage9_k6_population_summary")).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
