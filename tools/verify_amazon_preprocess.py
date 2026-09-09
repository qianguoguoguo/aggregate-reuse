#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "paper_results" / "expected" / "amazon_preprocess"
SHARED = ROOT.parent / "amazon_preprocess"
LOCAL = ROOT / "artifacts" / "amazon_preprocess"
CATEGORY = "home_and_kitchen"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _drop_additive_provenance(obj):
    """Remove metadata added after the immutable scientific preprocessing gold.

    The Phase-1/2 category-aware repository added portable raw-data and upstream
    provenance to preprocessing summaries.  Those fields are verified
    separately below because some upstream byte hashes (notably gzip files)
    may legitimately change across fresh rebuilds while their scientific
    manifests remain byte-identical.
    """
    obj = json.loads(json.dumps(obj))
    obj.pop("raw_dataset_sha256", None)
    obj.pop("provenance", None)
    return obj


def normalized_scan(obj):
    obj = _drop_additive_provenance(obj)
    obj.pop("input_path", None)
    return obj


def normalized_stage1(obj):
    obj = _drop_additive_provenance(obj)
    for k in ["input_reviews", "preliminary_counts_csv"]:
        obj.pop(k, None)
    obj.pop("outputs", None)
    return obj


def normalized_stage2(obj):
    obj = _drop_additive_provenance(obj)
    obj.pop("input_stage1_corpus", None)
    obj.pop("outputs", None)
    return obj


def json_equal(a, b):
    return a == b


def csv_bytes_equal(a, b):
    return Path(a).read_bytes() == Path(b).read_bytes()


def expected_raw_sha256() -> str:
    candidates = [
        ROOT / "artifacts" / "verification" / "phase3_smoke_home_and_kitchen_windows_seed0_full.json",
        ROOT / "artifacts" / "verification" / "phase3_smoke_home_and_kitchen_windows_seed0_core.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        value = obj.get("inputs", {}).get("raw_dataset_sha256")
        if isinstance(value, str) and len(value) == 64:
            return value
    raise RuntimeError(
        "Cannot locate the frozen Home_and_Kitchen raw SHA-256 in the Phase-3 verification manifests."
    )


def provenance_matches(obj, expected_sha: str) -> bool:
    provenance = obj.get("provenance")
    if not isinstance(provenance, dict):
        return False
    return (
        obj.get("category") == CATEGORY
        and obj.get("raw_dataset_sha256") == expected_sha
        and provenance.get("category") == CATEGORY
        and provenance.get("raw_dataset_sha256") == expected_sha
        and provenance.get("config", {}).get("path") == "configs/amazon_preprocess.yaml"
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--actual-root",
        type=Path,
        help=(
            "Preprocessing root to verify. For a fresh Phase-4 rebuild use "
            "artifacts/amazon_preprocess. If omitted, the staged local root is "
            "preferred when present, otherwise ../amazon_preprocess is used."
        ),
    )
    return ap.parse_args()


def main():
    args = parse_args()
    if args.actual_root is not None:
        act = args.actual_root
        if not act.is_absolute():
            act = ROOT / act
        act = act.resolve()
    else:
        act = LOCAL if LOCAL.exists() else SHARED
        act = act.resolve()
    if not act.is_dir():
        raise FileNotFoundError(act)

    out = ROOT / "artifacts" / "verification"
    out.mkdir(parents=True, exist_ok=True)

    checks = {}

    gold_scan = json.loads((GOLD/"home_and_kitchen_scan.json").read_text())
    act_scan = json.loads(
        (act/"scan"/"home_and_kitchen_scan.json").read_text()
    )
    checks["scan_summary"] = json_equal(
        normalized_scan(gold_scan),
        normalized_scan(act_scan),
    )

    gold_stage1 = json.loads(
        (GOLD/"home_and_kitchen_stage1_summary.json").read_text()
    )
    act_stage1 = json.loads(
        (act/"stage1"/"home_and_kitchen_stage1_summary.json").read_text()
    )
    checks["stage1_summary"] = json_equal(
        normalized_stage1(gold_stage1),
        normalized_stage1(act_stage1),
    )

    gold_stage2 = json.loads(
        (GOLD/"home_and_kitchen_stage2_summary.json").read_text()
    )
    act_stage2 = json.loads(
        (act/"stage2"/"home_and_kitchen_stage2_summary.json").read_text()
    )
    checks["stage2_summary"] = json_equal(
        normalized_stage2(gold_stage2),
        normalized_stage2(act_stage2),
    )

    checks["stage1_manifest"] = csv_bytes_equal(
        GOLD/"home_and_kitchen_first300_manifest.csv",
        act/"stage1"/"home_and_kitchen_first300_manifest.csv",
    )
    checks["stage1_excluded"] = csv_bytes_equal(
        GOLD/"home_and_kitchen_stage1_excluded.csv",
        act/"stage1"/"home_and_kitchen_stage1_excluded.csv",
    )
    checks["stage2_manifest"] = csv_bytes_equal(
        GOLD/"home_and_kitchen_stage2_manifest.csv",
        act/"stage2"/"home_and_kitchen_stage2_manifest.csv",
    )
    checks["reference_histograms"] = csv_bytes_equal(
        GOLD/"home_and_kitchen_reference_histograms.csv",
        act/"stage2"/"home_and_kitchen_reference_histograms.csv",
    )

    audit_gold = json.loads(
        (GOLD/"amazon_preprocess_gold_audit.json").read_text()
    )
    audit_actual = json.loads(
        (act/"amazon_preprocess_audit.json").read_text()
    )
    checks["headline_audit"] = all(
        audit_actual[k] == audit_gold[k]
        for k in [
            "duplicate_user_item_records_removed",
            "usable_records_after_user_item_dedup",
            "eligible_items_ge_300_after_dedup",
            "stage2_items",
            "both_experimental_blocks_feasible_k6",
            "both_experimental_blocks_feasible_k9",
        ]
    )

    counts = act/"scan"/"home_and_kitchen_item_counts.csv"
    checks["item_counts_sha256"] = (
        counts.exists()
        and sha256(counts) == audit_gold["item_counts_csv_sha256"]
    )

    expected_sha = expected_raw_sha256()
    checks["portable_provenance"] = all(
        provenance_matches(obj, expected_sha)
        for obj in (act_scan, act_stage1, act_stage2)
    )
    checks["audit_raw_sha256"] = (
        audit_actual.get("category") == CATEGORY
        and audit_actual.get("raw_dataset_sha256") == expected_sha
        and isinstance(audit_actual.get("provenance"), dict)
        and audit_actual["provenance"].get("raw_dataset_sha256") == expected_sha
    )

    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "phase": "2G",
        "status": status,
        "actual_root": str(act.relative_to(ROOT)) if ROOT in act.parents else act.name,
        "checks": checks,
        "headline": {
            "duplicate_user_item_records_removed":
                audit_actual.get("duplicate_user_item_records_removed"),
            "eligible_items_ge_300_after_dedup":
                audit_actual.get("eligible_items_ge_300_after_dedup"),
            "both_experimental_blocks_feasible_k6":
                audit_actual.get("both_experimental_blocks_feasible_k6"),
            "both_experimental_blocks_feasible_k9":
                audit_actual.get("both_experimental_blocks_feasible_k9"),
        },
    }
    (out/"phase2G_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("="*70)
    print("PHASE 2G AMAZON PREPROCESSING VERIFICATION")
    print("="*70)
    print(f"actual_root                 : {act}")
    for k,v in checks.items():
        print(f"{k:28s}: {'PASS' if v else 'FAIL'}")
    print(f"OVERALL                      : {status}")
    print("="*70)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
