#!/usr/bin/env python3
"""Copy only the frozen Stage-2/3 Amazon inputs needed by Phase-3 smoke runs."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for item in (ROOT, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from aggregate_reuse.amazon.paths import AmazonPathLayout
from aggregate_reuse.amazon.provenance import (
    discover_raw_dataset_sha256,
    require_artifact_category,
    sha256_file,
)
from tools._phase3_smoke_common import logical_sha256


def config_path(category: str) -> Path:
    if category == "home_and_kitchen":
        return ROOT / "configs" / "amazon_primary.yaml"
    return ROOT / "configs" / "electronics" / "amazon_primary.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", choices=["home_and_kitchen", "electronics"], required=True)
    ap.add_argument("--source-root", type=Path, required=True,
                    help="Existing category root containing canonical stage2/ and stage3/ inputs.")
    ap.add_argument("--dest-root", type=Path, required=True,
                    help="Clean category root to copy and use on both Windows and Linux.")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument(
        "--raw-dataset",
        type=Path,
        default=None,
        help=(
            "Optional raw category file used only to recover/verify its SHA-256 "
            "when legacy preprocessing metadata does not already record it."
        ),
    )
    args = ap.parse_args()

    source = args.source_root.resolve()
    dest = args.dest_root.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source == dest or source in dest.parents or dest in source.parents:
        raise RuntimeError("Source and destination must be separate category roots.")
    if dest.exists() and args.clean:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    cfg_path = config_path(args.category)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    src_layout = AmazonPathLayout.from_config(cfg, repo_root=ROOT, shared_root=source)
    dst_layout = AmazonPathLayout.from_config(cfg, repo_root=ROOT, shared_root=dest)
    shared = cfg["shared_data"]

    # Phase-3 downstream runs require the raw-dataset digest for provenance,
    # even though they do not reread the raw dataset.  New category artifacts
    # carry this digest directly, while the legacy Home pipeline may only be
    # able to recover it from the original scan input.  Resolve it while the
    # full canonical preprocessing root is still available, then create a
    # sanitized provenance-only scan stub in the portable Phase-3 bundle.
    raw_hash = None
    if args.raw_dataset is not None:
        raw_path = args.raw_dataset.resolve()
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        raw_hash = sha256_file(raw_path)
    else:
        try:
            raw_hash = discover_raw_dataset_sha256(src_layout)
        except RuntimeError:
            dataset = cfg.get("dataset") or {}
            filename = dataset.get("filename") if isinstance(dataset, dict) else None
            candidates = []
            if filename:
                # Canonical legacy layout normally keeps the raw file beside
                # amazon_preprocess/.  The repository-parent candidate covers
                # category-aware clean-room layouts.
                candidates.extend([source.parent / filename, ROOT.parent / filename])
            for candidate in candidates:
                candidate = candidate.resolve()
                if candidate.is_file():
                    raw_hash = sha256_file(candidate)
                    break
    if raw_hash is None:
        raise RuntimeError(
            "Could not recover the raw dataset SHA-256 from the canonical "
            "preprocessing root. Re-run this command with --raw-dataset PATH "
            "pointing to the category JSONL.GZ file."
        )

    items = {
        "roles": shared["roles_file"],
        "reference_histograms": shared["reference_histograms"],
        "reference_freeze": shared["reference_freeze"],
    }
    manifest = {"schema_version":1,"category":args.category,"files":{}}
    for label, relative in items.items():
        src = src_layout.resolve_shared_path(relative)
        dst = dst_layout.resolve_shared_path(relative)
        if not src.is_file():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        manifest["files"][label] = {
            "relative_path": Path(relative).as_posix(),
            "logical_sha256": logical_sha256(dst),
            "bytes": dst.stat().st_size,
        }

    freeze = json.loads(dst_layout.resolve_shared_path(shared["reference_freeze"]).read_text(encoding="utf-8"))
    require_artifact_category(freeze, expected=args.category, label="copied reference freeze", allow_legacy_home_missing=False)
    manifest["selected_lambda"] = freeze.get("selected_lambda")
    manifest["raw_dataset_sha256"] = raw_hash

    # Downstream provenance discovery looks for canonical preprocessing
    # summaries.  Write a tiny portable scan stub containing only category and
    # digest; do not copy legacy workstation paths into the smoke bundle.
    scan_stub = dst_layout.artifact("scan_summary")
    scan_stub.parent.mkdir(parents=True, exist_ok=True)
    scan_stub.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "category": args.category,
                "raw_dataset_sha256": raw_hash,
                "phase3_provenance_only": True,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    manifest["files"]["raw_provenance"] = {
        "relative_path": scan_stub.relative_to(dest).as_posix(),
        "logical_sha256": logical_sha256(scan_stub),
        "bytes": scan_stub.stat().st_size,
    }

    out = dest / "phase3_input_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PHASE 3 INPUT PREPARATION: PASS\nManifest: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
