from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

DEFAULT_RTOL = 1e-10
DEFAULT_ATOL = 1e-12

_ABS_WINDOWS = re.compile(r"^[A-Za-z]:[\\/]")
_ABS_POSIX = re.compile(r"^/(?:home|Users)/")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def logical_sha256(path: Path) -> str:
    """Hash logical text payload, ignoring gzip container metadata/newlines."""
    path = Path(path)
    h = hashlib.sha256()
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    mode = "rt"
    with opener(path, mode, encoding="utf-8", errors="strict", newline=None) as f:
        for line in f:
            # Normalize platform newline serialization without changing content.
            h.update(line.rstrip("\r\n").encode("utf-8"))
            h.update(b"\n")
    return h.hexdigest()


def _open_csv(path: Path):
    path = Path(path)
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def canonical_csv_fingerprint(
    path: Path,
    *,
    fields: Iterable[str] | None = None,
    json_fields: Iterable[str] = (),
) -> dict[str, Any]:
    """Hash a row set after selecting only discrete/structural columns."""
    json_fields = set(json_fields)
    payloads: list[str] = []
    with _open_csv(path) as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        selected = header if fields is None else list(fields)
        missing = [x for x in selected if x not in header]
        if missing:
            raise RuntimeError(f"{path}: missing structural fields {missing}")
        for row in reader:
            obj: dict[str, Any] = {}
            for key in selected:
                value: Any = row[key]
                if key in json_fields:
                    value = json.loads(value)
                obj[key] = value
            payloads.append(
                json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            )
    payloads.sort()
    h = hashlib.sha256()
    for payload in payloads:
        h.update(payload.encode("utf-8"))
        h.update(b"\n")
    return {
        "rows": len(payloads),
        "fields": selected,
        "sha256": h.hexdigest(),
    }


def _skip_summary_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in {
        "provenance", "environment", "software", "software_versions",
        "upstream_artifacts", "upstream_data", "inputs", "outputs",
        "generated_at", "generated_at_utc", "validated_at", "validated_at_utc",
    }:
        return True
    return lowered.endswith(("_path", "_file", "_dir", "_root", "_sha256"))


def split_summary(value: Any) -> tuple[dict[str, Any], dict[str, float]]:
    """Flatten a summary into exact non-floats and tolerant floating values."""
    exact: dict[str, Any] = {}
    numeric: dict[str, float] = {}

    def walk(obj: Any, path: str, key_name: str | None = None) -> None:
        if key_name is not None and _skip_summary_key(key_name):
            return
        if isinstance(obj, dict):
            for key in sorted(obj):
                walk(obj[key], f"{path}.{key}" if path else key, key)
            return
        if isinstance(obj, list):
            for i, item in enumerate(obj):
                walk(item, f"{path}[{i}]", None)
            return
        if isinstance(obj, bool) or obj is None or isinstance(obj, int):
            exact[path] = obj
            return
        if isinstance(obj, float):
            if math.isnan(obj):
                # In the full-background ranking, freq_eq_8 conditions on a
                # subgroup whose frequency is constant by construction.  Pearson
                # and Spearman score-frequency correlations are therefore
                # mathematically undefined.  Canonicalize only those expected
                # undefined correlations to JSON null so Windows/Linux manifests
                # remain standards-compliant and exactly comparable.
                leaf = path.rsplit(".", 1)[-1]
                expected_undefined = (
                    ".freq_eq_8." in f".{path}."
                    and leaf in {
                        "pearson_score_frequency_all",
                        "pearson_score_frequency_background",
                        "spearman_score_frequency_all",
                        "spearman_score_frequency_background",
                    }
                )
                if expected_undefined:
                    exact[path] = None
                    return
                raise RuntimeError(f"Unexpected NaN at {path}")
            if not math.isfinite(obj):
                raise RuntimeError(f"Non-finite float at {path}: {obj}")
            numeric[path] = float(obj)
            return
        if isinstance(obj, str):
            # Machine paths are deliberately excluded from cross-platform exactness.
            if _ABS_WINDOWS.match(obj) or _ABS_POSIX.match(obj):
                return
            exact[path] = obj
            return
        exact[path] = obj

    walk(value, "")
    return exact, numeric


def load_summary(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected JSON object")
    return value


def compare_reports(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    numeric_differences: list[dict[str, Any]] = []

    for key in ("schema_version", "category", "seed", "scope"):
        if left.get(key) != right.get(key):
            failures.append({"kind": "metadata", "path": key, "left": left.get(key), "right": right.get(key)})

    for section in ("inputs", "structural_fingerprints", "exact_summary"):
        lval = left.get(section, {})
        rval = right.get(section, {})
        if lval != rval:
            # Give path-level diagnostics for dictionaries.
            keys = sorted(set(lval) | set(rval)) if isinstance(lval, dict) and isinstance(rval, dict) else []
            if keys:
                for key in keys:
                    if lval.get(key) != rval.get(key):
                        failures.append({"kind": "exact", "path": f"{section}.{key}", "left": lval.get(key), "right": rval.get(key)})
            else:
                failures.append({"kind": "exact", "path": section, "left": lval, "right": rval})

    lnum = left.get("numeric_summary", {})
    rnum = right.get("numeric_summary", {})
    for key in sorted(set(lnum) | set(rnum)):
        if key not in lnum or key not in rnum:
            failures.append({"kind": "numeric_missing", "path": key, "left": lnum.get(key), "right": rnum.get(key)})
            continue
        a = float(lnum[key])
        b = float(rnum[key])
        abs_diff = abs(a - b)
        rel_diff = abs_diff / max(abs(a), abs(b), 1e-300)
        item = {"path": key, "left": a, "right": b, "abs_diff": abs_diff, "rel_diff": rel_diff}
        if abs_diff:
            numeric_differences.append(item)
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            failures.append({"kind": "numeric", **item})

    numeric_differences.sort(key=lambda x: x["abs_diff"], reverse=True)
    return {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "rtol": rtol,
        "atol": atol,
        "failure_count": len(failures),
        "failures": failures,
        "nonzero_numeric_difference_count": len(numeric_differences),
        "largest_numeric_differences": numeric_differences[:25],
        "left_platform": left.get("platform"),
        "right_platform": right.get("platform"),
        "category": left.get("category"),
        "seed": left.get("seed"),
        "scope": left.get("scope"),
    }
