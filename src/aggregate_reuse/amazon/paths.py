"""Category-aware path handling for the Amazon experiment pipeline.

The Home-and-Kitchen release predates category namespaces, so its downstream
artifacts live directly under ``../amazon_preprocess``. New categories use a
dedicated root such as ``../amazon_preprocess/electronics``. This module keeps
that legacy layout stable while giving every stage one validated resolver.

Stage-level files are category-prefixed for new categories. A few historical
Home-and-Kitchen aggregate files have generic names; those exact legacy names
are retained for Home-and-Kitchen, while new categories receive a category
prefix. Per-seed filenames remain generic because seed directories are already
isolated below the category root.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


HOME_AND_KITCHEN = "home_and_kitchen"
_CATEGORY_SLUG = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


STAGE_SUBDIRECTORIES = MappingProxyType({
    "scan": "scan",
    "stage1": "stage1",
    "stage2": "stage2",
    "stage3": "stage3",
    "stage4_attack": "stage4_attack",
    "stage4_reuse": "stage4_reuse",
    "stage5_twins": "stage5_twins",
    "stage6_shape": "stage6_shape",
    "stage7_complementarity": "stage7_complementarity",
    "stage8_reference_history": "stage8_reference_history",
    "stage9_strength_fixed": "stage9_strength_fixed",
    "stage9_k6_population_audit": "stage9_k6_population_audit",
    "stage10_full_background": "stage10_full_background",
    "stage11_self_influence": "stage11_self_influence",
})


@dataclass(frozen=True)
class _ArtifactSpec:
    stage: str | None
    filename_template: str
    home_legacy_filename: str | None = None

    def filename(self, category_prefix: str) -> str:
        if (
            category_prefix == HOME_AND_KITCHEN
            and self.home_legacy_filename is not None
        ):
            return self.home_legacy_filename
        return self.filename_template.format(prefix=category_prefix)


_ARTIFACT_SPECS = MappingProxyType({
    # Preprocessing: scan, canonical first 300, roles, and audit.
    "scan_summary": _ArtifactSpec("scan", "{prefix}_scan.json"),
    "scan_item_counts": _ArtifactSpec(
        "scan", "{prefix}_item_counts.csv"
    ),
    "stage1_corpus": _ArtifactSpec(
        "stage1", "{prefix}_first300.items.jsonl.gz"
    ),
    "stage1_manifest": _ArtifactSpec(
        "stage1", "{prefix}_first300_manifest.csv"
    ),
    "stage1_excluded": _ArtifactSpec(
        "stage1", "{prefix}_stage1_excluded.csv"
    ),
    "stage1_summary": _ArtifactSpec(
        "stage1", "{prefix}_stage1_summary.json"
    ),
    "stage2_roles": _ArtifactSpec(
        "stage2", "{prefix}_roles.items.jsonl.gz"
    ),
    "stage2_manifest": _ArtifactSpec(
        "stage2", "{prefix}_stage2_manifest.csv"
    ),
    "stage2_reference_histograms": _ArtifactSpec(
        "stage2", "{prefix}_reference_histograms.csv"
    ),
    "stage2_summary": _ArtifactSpec(
        "stage2", "{prefix}_stage2_summary.json"
    ),
    "preprocess_audit": _ArtifactSpec(
        None,
        "{prefix}_amazon_preprocess_audit.json",
        home_legacy_filename="amazon_preprocess_audit.json",
    ),
    # Stage 3 reference calibration.
    "stage3_lambda_calibration": _ArtifactSpec(
        "stage3", "{prefix}_lambda_calibration.csv"
    ),
    "stage3_holdout_validation": _ArtifactSpec(
        "stage3", "{prefix}_holdout_validation.csv"
    ),
    "stage3_holdout_item_metrics": _ArtifactSpec(
        "stage3", "{prefix}_holdout_item_metrics.csv"
    ),
    "stage3_selected_lambda_chronology": _ArtifactSpec(
        "stage3", "{prefix}_selected_lambda_chronology.csv"
    ),
    "stage3_reference_freeze": _ArtifactSpec(
        "stage3", "{prefix}_reference_freeze.json"
    ),
    "stage3_summary": _ArtifactSpec(
        "stage3", "{prefix}_stage3_summary.json"
    ),
    # Stages 4--11. Generic historical aggregate names get a prefix only for
    # new category namespaces; already-prefixed Home names use the template.
    "stage4_attack_seed_metrics": _ArtifactSpec(
        "stage4_attack",
        "{prefix}_primary_attack_seed_metrics.csv",
        home_legacy_filename="primary_attack_seed_metrics.csv",
    ),
    "stage4_attack_summary": _ArtifactSpec(
        "stage4_attack",
        "{prefix}_primary_attack_summary.json",
        home_legacy_filename="primary_attack_summary.json",
    ),
    "stage4_reuse_seed_metrics": _ArtifactSpec(
        "stage4_reuse", "{prefix}_stage4_seed_metrics.csv"
    ),
    "stage4_reuse_summary": _ArtifactSpec(
        "stage4_reuse", "{prefix}_stage4_summary.json"
    ),
    "stage5_twins_summary": _ArtifactSpec(
        "stage5_twins", "{prefix}_stage5_summary.json"
    ),
    "stage6_shape_summary": _ArtifactSpec(
        "stage6_shape", "{prefix}_stage6_summary.json"
    ),
    "stage7_complementarity_summary": _ArtifactSpec(
        "stage7_complementarity", "{prefix}_stage7_summary.json"
    ),
    "stage8_reference_history_calibration": _ArtifactSpec(
        "stage8_reference_history",
        "{prefix}_reference_history_calibration.csv",
        home_legacy_filename="reference_history_calibration.csv",
    ),
    "stage8_reference_history_seed_metrics": _ArtifactSpec(
        "stage8_reference_history",
        "{prefix}_reference_history_seed_metrics.csv",
        home_legacy_filename="reference_history_seed_metrics.csv",
    ),
    "stage8_reference_history_summary": _ArtifactSpec(
        "stage8_reference_history",
        "{prefix}_stage8_refhistory_summary.json",
    ),
    "stage9_strength_summary": _ArtifactSpec(
        "stage9_strength_fixed",
        "{prefix}_stage9_strength_fixed_identity_summary.json",
    ),
    "stage9_k6_population_expectations": _ArtifactSpec(
        "stage9_k6_population_audit",
        "{prefix}_k6_population_exact_expectations.csv",
        home_legacy_filename="k6_population_exact_expectations.csv",
    ),
    "stage9_k6_population_summary": _ArtifactSpec(
        "stage9_k6_population_audit",
        "{prefix}_k6_population_audit_summary.json",
    ),
    "stage10_full_background_summary": _ArtifactSpec(
        "stage10_full_background",
        "{prefix}_stage10_ranking_full_background_summary.json",
    ),
    "stage11_self_influence_summary": _ArtifactSpec(
        "stage11_self_influence",
        "{prefix}_stage11_self_influence_loo_summary.json",
    ),
})


def repository_root() -> Path:
    """Return the repository root containing ``src/`` and ``configs/``."""
    return Path(__file__).resolve().parents[3]


def _validate_category(category: Any) -> str:
    if not isinstance(category, str) or not _CATEGORY_SLUG.fullmatch(category):
        raise ValueError(
            "Category must be a lowercase underscore-delimited slug; "
            f"got {category!r}."
        )
    return category


def _resolve_from_repository(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _validate_leaf_filename(filename: str | Path, *, label: str) -> str:
    path = Path(filename)
    if (
        path.is_absolute()
        or path.name != str(path)
        or str(path) in {"", ".", ".."}
    ):
        raise ValueError(f"{label} must be one filename, got {filename!r}.")
    return str(path)


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _resolve_raw_dataset(
    config: Mapping[str, Any],
    *,
    root: Path,
) -> Path | None:
    dataset = config.get("dataset")
    if dataset is None:
        return None
    dataset = _require_mapping(dataset, label="dataset")
    location = dataset.get("location")
    if location != "parent_of_repository":
        raise ValueError(
            "dataset.location must be 'parent_of_repository'; "
            f"got {location!r}."
        )
    filename = _validate_leaf_filename(
        dataset.get("filename", ""),
        label="dataset.filename",
    )
    return (root.parent / filename).resolve()


def _resolve_shared_root(
    config: Mapping[str, Any],
    *,
    root: Path,
    category: str,
) -> Path:
    shared_data = config.get("shared_data")
    if shared_data is None:
        # The released Home preprocessing command historically stages output
        # inside the repository before publishing it to ../amazon_preprocess.
        if (
            category == HOME_AND_KITCHEN
            and config.get("experiment") == "amazon_preprocess"
        ):
            return (root / "artifacts" / "amazon_preprocess").resolve()
        raise ValueError("Config must define shared_data.root.")

    shared_data = _require_mapping(shared_data, label="shared_data")
    configured_root = shared_data.get("root")
    if not isinstance(configured_root, (str, Path)) or not str(configured_root):
        raise ValueError("shared_data.root must be a non-empty path.")
    return _resolve_from_repository(root, configured_root)


@dataclass(frozen=True)
class AmazonPathLayout:
    """Resolved, category-aware paths for one Amazon configuration."""

    repository_root: Path
    category: str
    shared_category_root: Path
    raw_dataset: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repository_root",
            Path(self.repository_root).resolve(),
        )
        object.__setattr__(self, "category", _validate_category(self.category))
        object.__setattr__(
            self,
            "shared_category_root",
            Path(self.shared_category_root).resolve(),
        )
        if self.raw_dataset is not None:
            object.__setattr__(
                self,
                "raw_dataset",
                Path(self.raw_dataset).resolve(),
            )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        repo_root: str | Path | None = None,
        shared_root: str | Path | None = None,
    ) -> AmazonPathLayout:
        """Resolve one loaded YAML config relative to the repository root.

        ``shared_root`` is an operational output-root override.  It is kept
        separate from the loaded configuration so clean-room reproductions can
        use the exact frozen scientific config (and therefore the same config
        SHA-256) while writing to a distinct actual-artifact namespace.
        """
        config = _require_mapping(config, label="config")
        category = _validate_category(config.get("category"))
        root = (
            repository_root()
            if repo_root is None
            else Path(repo_root).resolve()
        )
        return cls(
            repository_root=root,
            category=category,
            shared_category_root=(
                _resolve_shared_root(config, root=root, category=category)
                if shared_root is None
                else _resolve_from_repository(root, shared_root)
            ),
            raw_dataset=_resolve_raw_dataset(config, root=root),
        )

    @property
    def category_prefix(self) -> str:
        """Return the validated prefix used by category-scoped filenames."""
        return self.category

    def resolve_shared_path(self, relative_path: str | Path) -> Path:
        """Resolve a config path and reject absolute or escaping paths."""
        relative = Path(relative_path)
        if relative.is_absolute():
            raise ValueError(
                f"Shared-data paths must be relative, got {relative_path!r}."
            )
        resolved = (self.shared_category_root / relative).resolve()
        root = self.shared_category_root.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(
                f"Shared-data path escapes category root {root}: {resolved}"
            )
        return resolved

    def stage_dir(self, stage: str) -> Path:
        """Return a canonical stage directory below the category root."""
        try:
            subdirectory = STAGE_SUBDIRECTORIES[stage]
        except KeyError as exc:
            known = ", ".join(sorted(STAGE_SUBDIRECTORIES))
            raise KeyError(
                f"Unknown Amazon stage {stage!r}; expected one of {known}"
            ) from exc
        return self.resolve_shared_path(subdirectory)

    def category_filename(self, basename: str | Path) -> str:
        """Prefix one leaf filename with this layout's category slug."""
        leaf = _validate_leaf_filename(basename, label="basename")
        return f"{self.category_prefix}_{leaf}"

    def artifact_name(self, artifact: str) -> str:
        """Return the category-correct filename for a registered artifact."""
        try:
            spec = _ARTIFACT_SPECS[artifact]
        except KeyError as exc:
            known = ", ".join(sorted(_ARTIFACT_SPECS))
            raise KeyError(
                f"Unknown Amazon artifact {artifact!r}; expected one of {known}"
            ) from exc
        return spec.filename(self.category_prefix)

    def artifact(self, artifact: str) -> Path:
        """Return the full path for a registered stage-level artifact."""
        spec = _ARTIFACT_SPECS.get(artifact)
        if spec is None:
            # Delegate to artifact_name for the detailed error message.
            self.artifact_name(artifact)
            raise AssertionError("unreachable")
        parent = (
            self.shared_category_root
            if spec.stage is None
            else self.stage_dir(spec.stage)
        )
        return parent / spec.filename(self.category_prefix)

    def seed_dir(self, stage: str, seed: int) -> Path:
        """Return ``seed_NNN`` below a canonical stage directory."""
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"Seed must be a non-negative integer, got {seed!r}.")
        return self.stage_dir(stage) / f"seed_{seed:03d}"

    def per_seed_file(
        self,
        stage: str,
        seed: int,
        filename: str | Path,
    ) -> Path:
        """Return one generic per-seed file inside an isolated seed directory."""
        leaf = _validate_leaf_filename(filename, label="per-seed filename")
        return self.seed_dir(stage, seed) / leaf


def amazon_paths_from_config(
    config: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    shared_root: str | Path | None = None,
) -> AmazonPathLayout:
    """Functional wrapper around :meth:`AmazonPathLayout.from_config`."""
    return AmazonPathLayout.from_config(
        config,
        repo_root=repo_root,
        shared_root=shared_root,
    )


def isolated_category_root(
    shared_root: str | Path,
    category: str,
) -> Path:
    """Return the one allowed isolated root for ``category``.

    The returned path is always an immediate child of ``shared_root``. This
    deliberately excludes the shared root itself, sibling categories, stage
    directories, and paths reached through an escaping symlink.
    """
    category = _validate_category(category)
    parent = Path(shared_root).resolve()
    candidate = (parent / category).resolve()
    if candidate.parent != parent:
        raise ValueError(
            f"Category root must be an immediate child of {parent}: {candidate}"
        )
    return candidate


def clean_isolated_category_root(
    path: str | Path,
    *,
    shared_root: str | Path,
    category: str,
) -> bool:
    """Remove exactly one validated category namespace, if it exists.

    Returns ``True`` when a directory was removed and ``False`` when the
    validated category directory did not exist. Files and broader or sibling
    paths are rejected rather than removed.
    """
    expected = isolated_category_root(shared_root, category)
    requested = Path(path).resolve()
    if requested != expected:
        raise ValueError(
            f"Refusing to clean {requested}; the only allowed target is {expected}."
        )
    if not requested.exists():
        return False
    if not requested.is_dir():
        raise ValueError(f"Refusing to clean non-directory target: {requested}")

    shutil.rmtree(requested)
    return True
