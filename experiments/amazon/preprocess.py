#!/usr/bin/env python3
"""Build frozen Amazon preprocessing artifacts for a configured category.

The config names a raw dataset one directory above the repository and a
category-isolated output root.

Phases:
  1. eligibility scan + user-item count table;
  2. earliest-review user-item dedup + chronological first 300;
  3. frozen role split + leave-one-item-out reference tables;
  4. k=6/k=9 experimental-block feasibility audit.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.scan import scan_category
from aggregate_reuse.amazon.paths import (
    HOME_AND_KITCHEN,
    AmazonPathLayout,
    clean_isolated_category_root,
)
from aggregate_reuse.amazon.preprocessing import (
    connect_work_db,
    finalize_stage1,
    ingest_preliminary_eligible_reviews,
    load_preliminary_eligible_asins,
)
from aggregate_reuse.amazon.provenance import (
    build_provenance,
    external_path_label,
    logical_path,
    require_artifact_category,
    sha256_file,
)
from aggregate_reuse.amazon.roles import (
    ROLE_RANGES,
    STAR_RATINGS,
    load_stage1_items,
    mean_from_counts,
    probs_from_counts,
    rating_counts,
    split_reviews,
    validate_stage1_item,
)


def load_cfg(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def run_stage2(stage1_corpus, layout, provenance):
    out_dir = layout.stage_dir("stage2")
    out_dir.mkdir(parents=True, exist_ok=True)
    roles_path = layout.artifact("stage2_roles")
    manifest_path = layout.artifact("stage2_manifest")
    refs_path = layout.artifact("stage2_reference_histograms")
    summary_path = layout.artifact("stage2_summary")

    per_item_ref = {}
    global_ref = Counter()
    n_items = 0

    for _, obj in load_stage1_items(stage1_corpus):
        asin, reviews = validate_stage1_item(obj)
        if asin in per_item_ref:
            raise RuntimeError(f"Duplicate ASIN: {asin}")
        blocks = split_reviews(reviews)
        counts = rating_counts(blocks["reference"])
        if sum(counts.values()) != 120:
            raise RuntimeError(f"{asin}: reference size != 120")
        per_item_ref[asin] = counts
        global_ref.update(counts)
        n_items += 1

    expected_global_n = n_items * 120
    if sum(global_ref.values()) != expected_global_n:
        raise RuntimeError("Global reference-prefix invariant failed.")

    global_probs = probs_from_counts(
        {r: int(global_ref.get(r, 0)) for r in STAR_RATINGS}
    )
    global_mean = mean_from_counts(
        {r: int(global_ref.get(r, 0)) for r in STAR_RATINGS}
    )

    ref_fields = ["asin", "reference_n", "reference_mean"]
    for r in STAR_RATINGS:
        ref_fields += [f"reference_count_{r}", f"reference_prob_{r}"]
    ref_fields += ["loo_domain_n", "loo_domain_mean"]
    for r in STAR_RATINGS:
        ref_fields += [f"loo_domain_count_{r}", f"loo_domain_prob_{r}"]

    with refs_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ref_fields)
        writer.writeheader()
        for asin in sorted(per_item_ref):
            item = per_item_ref[asin]
            item_probs = probs_from_counts(item)
            loo_counts = {
                r: int(global_ref[r] - item[r]) for r in STAR_RATINGS
            }
            loo_n = sum(loo_counts.values())
            if loo_n != (n_items - 1) * 120:
                raise RuntimeError(f"{asin}: LOO size invariant failed.")
            loo_probs = probs_from_counts(loo_counts)
            row = {
                "asin": asin,
                "reference_n": 120,
                "reference_mean": mean_from_counts(item),
                "loo_domain_n": loo_n,
                "loo_domain_mean": mean_from_counts(loo_counts),
            }
            for r in STAR_RATINGS:
                row[f"reference_count_{r}"] = item[r]
                row[f"reference_prob_{r}"] = item_probs[r]
                row[f"loo_domain_count_{r}"] = loo_counts[r]
                row[f"loo_domain_prob_{r}"] = loo_probs[r]
            writer.writerow(row)

    manifest_fields = ["asin", "first_timestamp", "last_timestamp"]
    for role in ROLE_RANGES:
        manifest_fields += [
            f"{role}_start_position",
            f"{role}_end_position",
            f"{role}_n",
            f"{role}_mean",
        ]
        for r in STAR_RATINGS:
            manifest_fields.append(f"{role}_count_{r}")

    role_rating_totals = {role: Counter() for role in ROLE_RANGES}
    role_review_totals = Counter()
    written_items = 0

    with gzip.open(
        roles_path, "wt", encoding="utf-8", compresslevel=6
    ) as jout, manifest_path.open(
        "w", encoding="utf-8", newline=""
    ) as mf:
        mw = csv.DictWriter(mf, fieldnames=manifest_fields)
        mw.writeheader()

        for _, obj in load_stage1_items(stage1_corpus):
            asin, reviews = validate_stage1_item(obj)
            blocks = split_reviews(reviews)

            role_obj = {
                "asin": asin,
                "role_ranges": {
                    role: {"start": lo, "end": hi}
                    for role, (lo, hi) in ROLE_RANGES.items()
                },
                "blocks": blocks,
            }
            jout.write(json.dumps(role_obj, separators=(",", ":")) + "\n")

            row = {
                "asin": asin,
                "first_timestamp": int(reviews[0]["timestamp"]),
                "last_timestamp": int(reviews[-1]["timestamp"]),
            }
            for role, block in blocks.items():
                lo, hi = ROLE_RANGES[role]
                counts = rating_counts(block)
                row[f"{role}_start_position"] = lo
                row[f"{role}_end_position"] = hi
                row[f"{role}_n"] = len(block)
                row[f"{role}_mean"] = mean_from_counts(counts)
                for r in STAR_RATINGS:
                    row[f"{role}_count_{r}"] = counts[r]
                role_rating_totals[role].update(counts)
                role_review_totals[role] += len(block)

            mw.writerow(row)
            written_items += 1

    if written_items != n_items:
        raise RuntimeError("Stage2 pass item counts differ.")

    expected_role_sizes = {
        role: (hi - lo + 1) * n_items
        for role, (lo, hi) in ROLE_RANGES.items()
    }
    for role, expected in expected_role_sizes.items():
        if role_review_totals[role] != expected:
            raise RuntimeError(f"{role}: role-size invariant failed.")

    summary = {
        "stage": "amazon_stage2_frozen_roles_and_references",
        "category": layout.category,
        "raw_dataset_sha256": provenance["raw_dataset_sha256"],
        "provenance": provenance,
        "input_stage1_corpus": logical_path(
            stage1_corpus, repo_root=ROOT, shared_root=layout.shared_category_root
        ),
        "n_items": n_items,
        "role_ranges": {
            role: {"start": lo, "end": hi, "n_per_item": hi-lo+1}
            for role, (lo, hi) in ROLE_RANGES.items()
        },
        "global_reference_prefix": {
            "n": expected_global_n,
            "mean": global_mean,
            "counts": {str(r): int(global_ref[r]) for r in STAR_RATINGS},
            "probs": {str(r): global_probs[r] for r in STAR_RATINGS},
        },
        "leave_one_item_out_reference": {
            "construction": (
                "For item i, subtract its positions 1--120 rating counts from "
                "the global positions 1--120 pool and normalize the remainder."
            ),
            "n_per_item": (n_items - 1) * 120,
            "uses_only_reference_prefixes": True,
        },
        "role_review_totals": {
            role: int(role_review_totals[role]) for role in ROLE_RANGES
        },
        "role_rating_totals": {
            role: {
                str(r): int(role_rating_totals[role].get(r, 0))
                for r in STAR_RATINGS
            }
            for role in ROLE_RANGES
        },
        "data_usage_guards": {
            "lambda_selected_in_stage2": False,
            "attack_generated_in_stage2": False,
            "calibration_used_for_reference_estimation": False,
            "experimental_blocks_used_for_reference_estimation": False,
            "holdout_used_for_reference_estimation": False,
            "holdout_used_for_tuning": False,
        },
        "outputs": {
            "roles": logical_path(
                roles_path, repo_root=ROOT, shared_root=layout.shared_category_root
            ),
            "manifest": logical_path(
                manifest_path, repo_root=ROOT, shared_root=layout.shared_category_root
            ),
            "references": logical_path(
                refs_path, repo_root=ROOT, shared_root=layout.shared_category_root
            ),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def feasibility_audit(stage2_manifest):
    n = k6 = k9 = 0
    with Path(stage2_manifest).open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            n += 1
            a_non5 = 30 - int(row["experimental_A_count_5"])
            b_non5 = 30 - int(row["experimental_B_count_5"])
            k6 += int(a_non5 >= 6 and b_non5 >= 6)
            k9 += int(a_non5 >= 9 and b_non5 >= 9)
    return {
        "stage2_items": n,
        "both_experimental_blocks_feasible_k6": k6,
        "both_experimental_blocks_feasible_k9": k9,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(ROOT / "configs" / "amazon_preprocess.yaml"),
    )
    ap.add_argument(
        "--shared-root",
        type=Path,
        help=(
            "Operational output-root override for a clean-room reproduction; "
            "the loaded config and its provenance hash remain unchanged."
        ),
    )
    ap.add_argument("--clean", action="store_true")
    ap.add_argument(
        "--skip-scan",
        action="store_true",
        help="Reuse an existing counts CSV/scan JSON.",
    )
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    config_path = Path(args.config).resolve()
    layout = AmazonPathLayout.from_config(
        cfg,
        repo_root=ROOT,
        shared_root=args.shared_root,
    )
    raw = layout.raw_dataset
    if raw is None:
        raise ValueError("Preprocessing config must define dataset.")
    if not raw.exists():
        raise FileNotFoundError(
            f"Expected raw dataset beside repository: {raw}"
        )

    raw_dataset_sha256 = sha256_file(raw)
    out = layout.shared_category_root
    scan_dir = layout.stage_dir("scan")
    stage1_dir = layout.stage_dir("stage1")
    stage2_dir = layout.stage_dir("stage2")
    work_db = out / "_work_stage1.sqlite"

    if args.clean and out.exists():
        if layout.category == HOME_AND_KITCHEN:
            legacy_local = (ROOT / "artifacts" / "amazon_preprocess").resolve()
            if out != legacy_local:
                raise ValueError(
                    f"Refusing to clean unexpected Home preprocessing root: {out}"
                )
            shutil.rmtree(out)
        else:
            clean_isolated_category_root(
                out,
                shared_root=out.parent,
                category=layout.category,
            )
    for p in [scan_dir, stage1_dir, stage2_dir]:
        p.mkdir(parents=True, exist_ok=True)

    scan_json = layout.artifact("scan_summary")
    counts_csv = layout.artifact("scan_item_counts")

    if not args.skip_scan:
        scan_summary = scan_category(
            raw,
            scan_json,
            counts_csv,
            category=cfg["category"],
            progress_every=int(
                cfg["implementation"]["scan_progress_every"]
            ),
        )
    else:
        if not scan_json.exists() or not counts_csv.exists():
            raise FileNotFoundError("--skip-scan requires existing scan outputs")
        scan_summary = json.loads(scan_json.read_text(encoding="utf-8"))
        require_artifact_category(
            scan_summary,
            expected=layout.category,
            label="reused scan summary",
        )

    scan_summary["category"] = layout.category
    scan_summary["input_path"] = external_path_label(raw)
    scan_summary["raw_dataset_sha256"] = raw_dataset_sha256
    scan_summary["provenance"] = build_provenance(
        category=layout.category,
        config_path=config_path,
        raw_dataset_sha256=raw_dataset_sha256,
        selected_lambda=None,
        seed_ids=[],
        repo_root=ROOT,
        shared_root=out,
    )
    scan_json.write_text(
        json.dumps(scan_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    preliminary = load_preliminary_eligible_asins(
        counts_csv,
        threshold=int(cfg["eligibility"]["minimum_usable_unique_user_reviews"]),
    )

    corpus = layout.artifact("stage1_corpus")
    manifest1 = layout.artifact("stage1_manifest")
    excluded = layout.artifact("stage1_excluded")
    summary1_path = layout.artifact("stage1_summary")

    conn = connect_work_db(work_db)
    try:
        ingest_stats = ingest_preliminary_eligible_reviews(
            reviews_gz=raw,
            preliminary_eligible=preliminary,
            conn=conn,
            progress_every=int(
                cfg["implementation"]["ingest_progress_every"]
            ),
            batch_size=int(cfg["implementation"]["sqlite_batch_size"]),
        )
        final_stats = finalize_stage1(
            conn=conn,
            preliminary_eligible=preliminary,
            out_jsonl_gz=corpus,
            out_manifest_csv=manifest1,
            out_excluded_csv=excluded,
            threshold=int(
                cfg["eligibility"]["minimum_usable_unique_user_reviews"]
            ),
        )
    finally:
        conn.close()
        if work_db.exists():
            work_db.unlink()

    summary1 = {
        "stage": "amazon_stage1_canonical_first300",
        "category": cfg["category"],
        "raw_dataset_sha256": raw_dataset_sha256,
        "provenance": build_provenance(
            category=layout.category,
            config_path=config_path,
            raw_dataset_sha256=raw_dataset_sha256,
            selected_lambda=None,
            seed_ids=[],
            upstream_artifacts={
                "scan_summary": scan_json,
                "scan_item_counts": counts_csv,
            },
            repo_root=ROOT,
            shared_root=out,
        ),
        "rating_policy": {
            "accepted": [1, 2, 3, 4, 5],
            "rating_zero": "explicitly skipped",
            "other_nonstandard_ratings": "skipped and counted",
        },
        "deduplication_policy": (
            "For each (user_id, asin), keep the earliest usable review by "
            "timestamp; break exact timestamp ties by source line number."
        ),
        "chronological_order": "(timestamp, source_line)",
        "input_reviews": external_path_label(raw),
        "preliminary_counts_csv": logical_path(
            counts_csv, repo_root=ROOT, shared_root=out
        ),
        "ingest": ingest_stats,
        "final": final_stats,
        "outputs": {
            "corpus": logical_path(corpus, repo_root=ROOT, shared_root=out),
            "manifest": logical_path(manifest1, repo_root=ROOT, shared_root=out),
            "excluded": logical_path(excluded, repo_root=ROOT, shared_root=out),
        },
    }
    summary1_path.write_text(
        json.dumps(summary1, indent=2, sort_keys=True), encoding="utf-8"
    )

    summary2 = run_stage2(
        corpus,
        layout,
        build_provenance(
            category=layout.category,
            config_path=config_path,
            raw_dataset_sha256=raw_dataset_sha256,
            selected_lambda=None,
            seed_ids=[],
            upstream_artifacts={
                "stage1_corpus": corpus,
                "stage1_manifest": manifest1,
                "stage1_summary": summary1_path,
            },
            repo_root=ROOT,
            shared_root=out,
        ),
    )
    audit = feasibility_audit(
        layout.artifact("stage2_manifest")
    )
    audit.update({
        "category": layout.category,
        "raw_dataset_sha256": raw_dataset_sha256,
        "duplicate_user_item_records_removed":
            scan_summary["duplicate_user_item_records"],
        "usable_records_after_user_item_dedup":
            scan_summary["usable_records_after_user_item_dedup"],
        "eligible_items_ge_300_after_dedup":
            scan_summary["threshold_counts_after_dedup"]["300"],
    })
    audit["provenance"] = build_provenance(
        category=layout.category,
        config_path=config_path,
        raw_dataset_sha256=raw_dataset_sha256,
        selected_lambda=None,
        seed_ids=[],
        upstream_artifacts={
            "scan_summary": scan_json,
            "stage1_summary": summary1_path,
            "stage2_summary": layout.artifact("stage2_summary"),
        },
        repo_root=ROOT,
        shared_root=out,
    )
    layout.artifact("preprocess_audit").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("Amazon preprocessing complete.")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
