"""Shared provenance and category guards for Amazon experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import re
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .paths import HOME_AND_KITCHEN, AmazonPathLayout


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENVIRONMENT_DISTRIBUTIONS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "pyyaml": "PyYAML",
}


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def environment_versions() -> dict[str, str]:
    """Return the frozen runtime components recorded by Amazon summaries."""
    result = {"python": platform.python_version()}
    for label, distribution in _ENVIRONMENT_DISTRIBUTIONS.items():
        result[label] = version(distribution)
    return result


def _validate_sha256(value: str | None, *, label: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).lower()
    if not _SHA256.fullmatch(normalized):
        raise ValueError(f"{label} must be a 64-character SHA-256 digest.")
    return normalized


def logical_path(
    path: str | Path,
    *,
    repo_root: str | Path,
    shared_root: str | Path | None = None,
) -> str:
    resolved = Path(path).resolve()
    repository = Path(repo_root).resolve()
    shared = Path(shared_root).resolve() if shared_root is not None else None
    if shared is not None:
        try:
            relative = resolved.relative_to(shared)
            return f"shared_data/{relative.as_posix()}"
        except ValueError:
            pass
    try:
        return resolved.relative_to(repository).as_posix()
    except ValueError:
        return f"external/{resolved.name}"


def external_path_label(path: str | Path) -> str:
    """Return a portable label for a file intentionally outside the repository."""
    return f"external/{Path(path).name}"


# Backward-compatible private alias used by older callers.
def _logical_path(
    path: str | Path,
    *,
    repo_root: str | Path,
    shared_root: str | Path | None = None,
) -> str:
    return logical_path(
        path,
        repo_root=repo_root,
        shared_root=shared_root,
    )


def build_provenance(
    *,
    category: str,
    config_path: str | Path,
    raw_dataset_sha256: str | None,
    selected_lambda: float | int | None,
    seed_ids: Sequence[int],
    upstream_artifacts: Mapping[str, str | Path] | None = None,
    repo_root: str | Path,
    shared_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build one portable provenance block for an aggregate summary.

    Paths are stored relative to the repository or shared category root so an
    independent reproduction can use a different absolute checkout location.
    Direct upstream files are hashed from their actual resolved paths.
    """
    config = Path(config_path).resolve()
    repository = Path(repo_root).resolve()
    shared = Path(shared_root).resolve() if shared_root is not None else None
    raw_hash = _validate_sha256(
        raw_dataset_sha256,
        label="raw_dataset_sha256",
    )

    seeds: list[int] = []
    for seed in seed_ids:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed_ids must contain integers, got {seed!r}.")
        seeds.append(int(seed))
    if len(seeds) != len(set(seeds)):
        raise ValueError("seed_ids must not contain duplicates.")

    upstream = {}
    for label, source in sorted((upstream_artifacts or {}).items()):
        path = Path(source).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        upstream[label] = {
            "path": _logical_path(
                path,
                repo_root=repository,
                shared_root=shared,
            ),
            "sha256": sha256_file(path),
        }

    return {
        "category": category,
        "raw_dataset_sha256": raw_hash,
        "config": {
            "path": _logical_path(
                config,
                repo_root=repository,
                shared_root=None,
            ),
            "sha256": sha256_file(config),
        },
        "selected_lambda": (
            None if selected_lambda is None else float(selected_lambda)
        ),
        "seed_ids": seeds,
        "environment_versions": environment_versions(),
        "upstream_artifacts": upstream,
    }


def raw_dataset_sha256_from_artifact(
    artifact: Mapping[str, Any],
    *,
    label: str,
) -> str:
    """Read and validate a raw-dataset digest propagated by an artifact."""
    value = artifact.get("raw_dataset_sha256")
    provenance = artifact.get("provenance")
    if value is None and isinstance(provenance, Mapping):
        value = provenance.get("raw_dataset_sha256")
    normalized = _validate_sha256(value, label=f"{label} raw dataset SHA-256")
    if normalized is None:
        raise RuntimeError(f"{label} does not record raw_dataset_sha256.")
    return normalized


def discover_raw_dataset_sha256(layout: AmazonPathLayout) -> str:
    """Find the propagated raw digest, with a legacy Home scan fallback."""
    candidates = [
        ("preprocessing audit", "preprocess_audit"),
        ("Stage-2 summary", "stage2_summary"),
        ("scan summary", "scan_summary"),
    ]
    scan_summary: Mapping[str, Any] | None = None
    for label, artifact_name in candidates:
        path = layout.artifact(artifact_name)
        if not path.is_file():
            continue
        artifact = json.loads(path.read_text(encoding="utf-8"))
        require_artifact_category(
            artifact,
            expected=layout.category,
            label=label,
        )
        try:
            return raw_dataset_sha256_from_artifact(artifact, label=label)
        except RuntimeError:
            if artifact_name == "scan_summary":
                scan_summary = artifact

    # Released Home scan summaries predate raw digest recording but retain the
    # exact input path. Hash that file once when migrating a legacy pipeline.
    if scan_summary is not None:
        input_path = scan_summary.get("input_path")
        if input_path is not None and Path(input_path).is_file():
            return sha256_file(input_path)
        # Sanitized legacy summaries retain only a logical external label.
        # When the category profile binds the raw dataset, hash that file
        # instead of requiring the historical workstation path.
        if layout.raw_dataset is not None and layout.raw_dataset.is_file():
            return sha256_file(layout.raw_dataset)
    raise RuntimeError(
        f"No raw dataset SHA-256 is available for category {layout.category!r}. "
        "Run category preprocessing first."
    )


def resolve_raw_dataset_sha256(
    layout: AmazonPathLayout,
    *artifacts: Mapping[str, Any],
) -> str:
    """Use propagated provenance first, then the preprocessing registry."""
    for index, artifact in enumerate(artifacts, start=1):
        try:
            return raw_dataset_sha256_from_artifact(
                artifact,
                label=f"upstream artifact {index}",
            )
        except RuntimeError:
            continue
    return discover_raw_dataset_sha256(layout)


def require_artifact_category(
    artifact: Mapping[str, Any],
    *,
    expected: str,
    label: str,
    allow_legacy_home_missing: bool = True,
) -> None:
    """Reject cross-category inputs before an experiment consumes them."""
    actual = artifact.get("category")
    provenance = artifact.get("provenance")
    if actual is None and isinstance(provenance, Mapping):
        actual = provenance.get("category")
    if (
        actual is None
        and allow_legacy_home_missing
        and expected == HOME_AND_KITCHEN
    ):
        return
    if actual != expected:
        raise RuntimeError(
            f"{label} category mismatch: expected {expected!r}, got {actual!r}."
        )
