from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from tools._phase3_smoke_common import canonical_csv_fingerprint, compare_reports, logical_sha256, split_summary


def test_logical_sha_ignores_gzip_container_metadata(tmp_path: Path):
    a = tmp_path / "a.csv.gz"
    b = tmp_path / "b.csv.gz"
    payload = "x,y\n1,2\n"
    with gzip.GzipFile(filename=str(a), mode="wb", mtime=1) as f:
        f.write(payload.encode())
    with gzip.GzipFile(filename=str(b), mode="wb", mtime=999999) as f:
        f.write(payload.encode())
    assert logical_sha256(a) == logical_sha256(b)


def test_structural_fingerprint_ignores_unselected_float_columns(tmp_path: Path):
    a = tmp_path / "a.csv.gz"
    b = tmp_path / "b.csv.gz"
    for path, value in [(a, "0.1000000000001"), (b, "0.1000000000002")]:
        with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["asin", "slots", "d_cf"])
            w.writeheader(); w.writerow({"asin":"A", "slots":json.dumps([1,2]), "d_cf":value})
    fa = canonical_csv_fingerprint(a, fields=["asin","slots"], json_fields=["slots"])
    fb = canonical_csv_fingerprint(b, fields=["asin","slots"], json_fields=["slots"])
    assert fa == fb


def test_split_summary_separates_exact_and_float_and_ignores_paths():
    exact, numeric = split_summary({
        "category":"electronics", "seed":0, "auc":0.75,
        "invariants":{"ok":True}, "output_path":"/tmp/example/file",
        "provenance":{"platform":"windows"},
    })
    assert exact["category"] == "electronics"
    assert exact["seed"] == 0
    assert exact["invariants.ok"] is True
    assert "output_path" not in exact
    assert "provenance.platform" not in exact
    assert numeric["auc"] == 0.75


def test_compare_reports_accepts_tiny_float_drift():
    base = {
        "schema_version":1,"category":"electronics","seed":0,"scope":"core",
        "platform":{"system":"windows"},
        "inputs":{"x":"same"},"structural_fingerprints":{"s":{"sha256":"abc"}},
        "exact_summary":{"n":2},"numeric_summary":{"auc":0.75},
    }
    other = json.loads(json.dumps(base)); other["platform"]={"system":"linux"}; other["numeric_summary"]["auc"] += 5e-12
    assert compare_reports(base, other)["status"] == "PASS"


def test_compare_reports_rejects_structural_difference():
    base = {
        "schema_version":1,"category":"electronics","seed":0,"scope":"core",
        "platform":{"system":"windows"},
        "inputs":{"x":"same"},"structural_fingerprints":{"s":{"sha256":"abc"}},
        "exact_summary":{},"numeric_summary":{},
    }
    other = json.loads(json.dumps(base)); other["platform"]={"system":"linux"}; other["structural_fingerprints"]["s"]["sha256"]="def"
    result = compare_reports(base, other)
    assert result["status"] == "FAIL"
    assert result["failure_count"] >= 1
