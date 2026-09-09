"""Invariant-only verification for the Electronics Amazon reproduction.

This module deliberately does not import or read ``paper_results/expected``.
It validates the generated Electronics run against its declared protocol,
upstream provenance, and internal conservation laws.  The public command-line
entry point is :mod:`tools.verify_electronics`.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterator

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.paths import AmazonPathLayout
from aggregate_reuse.amazon.population_audit import exact_population_expected_mean
from aggregate_reuse.amazon.primary_attack import (
    canonical_attack_hash,
    load_experimental_blocks,
    load_reference_table,
)
from aggregate_reuse.amazon.provenance import (
    environment_versions,
    external_path_label,
    logical_path,
)
from aggregate_reuse.amazon.shape import canonical_attack_hash as shape_attack_hash

from _verify_common import DEFAULT_ATOL, DEFAULT_RTOL, float_close


CATEGORY = "electronics"
EXPECTED_SEEDS = tuple(range(30))
REUSE_GRID = (1, 2, 4, 8, 16)
STRENGTH_GRID = (3, 6, 9)
REFERENCE_LENGTHS = (60, 90, 120)

VERIFICATION_SCOPE = (
    "input_provenance",
    "schemas",
    "row_counts",
    "seed_sets",
    "exact_discrete_fields",
    "hashes",
    "conservation_laws",
    "category_consistency",
    "declared_floating_point_tolerances",
)

CONFIG_FILES = (
    "amazon_preprocess.yaml",
    "amazon_reference.yaml",
    "amazon_primary.yaml",
    "amazon_twins.yaml",
    "amazon_shape.yaml",
    "amazon_complementarity.yaml",
    "amazon_reference_history.yaml",
    "amazon_strength_fixed.yaml",
    "amazon_k6_population_audit.yaml",
    "amazon_full_background.yaml",
    "amazon_self_influence.yaml",
)

ATTACK_HEADER = (
    "asin", "treatment_block", "treated_positions", "treated_source_lines",
    "original_ratings", "replacement_ratings", "clean_counts",
    "attack_counts", "clean_w1", "attack_w1", "d_cf",
)
ATTACK_METRICS_HEADER = (
    "seed", "attack_world_sha256", "n_items", "m", "mean_d_cf",
    "positive_d_cf_fraction", "zero_d_cf_fraction",
    "negative_d_cf_fraction", "clean_w1_mean", "attack_w1_mean",
)
REUSE_METRICS_HEADER = (
    "seed", "reuse_r", "attack_world_sha256",
    "identity_assignment_sha256", "n_items", "m", "coalition_accounts",
    "normal_accounts", "mean_d_cf", "positive_d_cf_fraction",
    "zero_d_cf_fraction", "negative_d_cf_fraction", "clean_w1_mean",
    "attack_w1_mean", "score_auc", "frequency_auc",
    "score_no_larger_probability", "coalition_mean_score",
    "predicted_coalition_mean_score_r_times_mean_dcf",
    "mean_reuse_law_abs_error", "coalition_score_total",
    "expected_coalition_score_total_m_times_sum_dcf",
    "score_conservation_abs_error", "coalition_frequency_mean",
    "normal_frequency_mean",
)
IDENTITY_HEADER = (
    "asin", "treatment_block", "treated_position", "treated_source_line",
    "synthetic_account_id", "d_cf",
)
ACCOUNT_METRICS_HEADER = (
    "account_id", "is_synthetic_coalition", "score", "frequency",
)
TWIN_ACCOUNTS_HEADER = (
    "pair_id", "coalition_account_id", "control_account_id",
    "frequency_each", "item_set_json", "slot_set_json",
    "coalition_counterfactual_score", "control_counterfactual_score",
    "coalition_raw_world_score", "control_raw_world_score",
    "coalition_predictive_centered_world_score",
    "control_predictive_centered_world_score", "paired_score_gap",
)
TWIN_EDGES_HEADER = (
    "pair_id", "coalition_account_id", "control_account_id", "asin",
    "block", "position", "source_line", "clean_rating", "attack_rating",
    "clean_w1", "attack_w1", "d_cf",
)
SHAPE_ATTACK_HEADER = (
    "asin", "treatment_block", "slots", "clean_counts", "attack_counts",
    "clean_mean", "attack_mean", "clean_w1", "attack_w1", "d_w1",
    "clean_js", "attack_js", "d_js", "clean_abs_mean",
    "attack_abs_mean", "d_abs_mean",
)
SHAPE_IDENTITY_HEADER = (
    "account_id", "asin", "block", "position", "source_line",
    "original_rating", "replacement_rating", "d_w1", "d_js",
    "d_abs_mean",
)
SHAPE_TWINS_HEADER = (
    "pair_id", "coalition_account_id", "control_account_id",
    "frequency_each", "w1_attack_score", "w1_clean_score",
    "w1_paired_gap", "js_attack_score", "js_clean_score", "js_paired_gap",
    "abs_mean_attack_score", "abs_mean_clean_score", "abs_mean_paired_gap",
)
INCIDENCE_HEADER = ("account_id", "asin")
STRENGTH_HEADER = (
    "asin", "treatment_block", "k", "fixed_item_degree", "reuse_r",
    "modified_slots", "fixed_assignments", "clean_counts", "attack_counts",
    "clean_w1", "attack_w1", "d_cf",
)
FULL_BACKGROUND_HEADER = (
    "account_type", "account_id", "frequency", "raw_w1_score",
    "predictive_centered_w1_score", "coactivity_score", "combined_score",
)
SELF_ACCOUNT_HEADER = (
    "synthetic_account_id", "frequency", "original_counterfactual_score",
    "loo_counterfactual_score", "direct_self_component",
)
SELF_EXPOSURE_HEADER = (
    "synthetic_account_id", "asin", "treatment_block", "treated_position",
    "treated_source_line", "original_rating", "replacement_rating",
    "original_d_cf", "loo_d_cf", "clean_loo_w1", "attack_loo_w1",
)

AGGREGATE_KEYS = {
    "reference": {
        "absolute_null_diagnostic", "category",
        "chronological_selected_lambda_diagnostic", "data_usage_guards",
        "holdout_validation", "lambda_grid", "n_items", "outputs",
        "posterior_predictive_model", "predictive_baseline", "provenance",
        "raw_dataset_sha256", "selected_calibration_row", "selected_lambda",
        "selection_rule", "semantic_decision", "stage",
    },
    "primary_attack": {
        "aggregate_visibility_mean_d_cf", "all_attack_invariants_pass",
        "category", "experiment", "identity_assignment_performed",
        "n_seeds", "provenance", "raw_dataset_sha256", "seed_ids",
        "selected_lambda",
    },
    "primary_reuse": {
        "aggregate_visibility_mean_d_cf", "category", "interpretation_guard",
        "n_seeds", "provenance", "raw_dataset_sha256", "reuse_summary",
        "seed_ids", "selected_lambda", "stage",
    },
    "matched_twins": {
        "category", "interpretation_guard", "matching", "metrics", "n_seeds",
        "provenance", "raw_dataset_sha256", "reuse_r", "seed_ids",
        "selected_lambda", "stage",
    },
    "shape": {
        "category", "interpretation_guard", "invariants", "metrics",
        "n_items_per_seed", "n_seeds", "provenance", "raw_dataset_sha256",
        "reuse_r", "seed_ids", "selected_lambda", "stage",
    },
    "complementarity": {
        "category", "dimensions", "interpretation_guard", "invariants",
        "metrics", "n_seeds", "provenance", "raw_dataset_sha256",
        "seed_ids", "selected_lambda", "stage",
    },
    "reference_history": {
        "category", "interpretation_guard", "invariants",
        "metrics_by_reference_length", "n_seeds", "provenance",
        "raw_dataset_sha256", "reference_lengths", "reuse_r", "seed_ids",
        "selected_lambda", "selected_lambda_scope", "stage",
    },
    "strength_fixed": {
        "category", "fixed_item_degree", "interpretation_guard", "invariants",
        "metrics_by_k", "n_seeds", "provenance", "raw_dataset_sha256",
        "reuse_r", "seed_ids", "selected_lambda", "stage",
        "strength_grid_k",
    },
    "k6_population_audit": {
        "category", "experiment", "interpretation_guard", "invariants", "k",
        "legacy_empirical_values", "population_results", "provenance",
        "raw_dataset_sha256", "selected_lambda",
    },
    "full_background": {
        "activity_strata", "category", "inspection_burden_targets",
        "interpretation_guard", "metrics", "n_seeds", "population",
        "provenance", "raw_dataset_sha256", "seed_ids", "selected_lambda",
        "stage", "top_fractions",
    },
    "self_influence": {
        "category", "interpretation_guard", "metrics", "n_seeds",
        "population", "provenance", "raw_dataset_sha256", "seed_ids",
        "selected_lambda", "stage",
    },
}

SEED_KEYS = {
    "primary_attack": {
        "aggregate_visibility", "attack_world_sha256", "category",
        "evidence_mode", "experiment", "feasible_item_universe",
        "invariants", "m", "sampled_items", "seed", "selected_lambda",
    },
    "primary_reuse": {
        "attack_world_sha256", "category", "experiment",
        "identity_assignment_sha256_by_reuse", "invariants", "m",
        "reuse_grid", "reuse_metrics", "sampled_items", "seed",
    },
    "matched_twins": {
        "category", "interpretation_guard", "matching", "metrics",
        "n_edges_per_group", "n_pairs", "reuse_r", "seed",
        "selected_lambda", "stage", "stage4_attack_world_sha256",
    },
    "shape": {
        "aggregate", "attack_world_sha256", "category",
        "feasible_item_universe", "interpretation_guard", "invariants", "m",
        "matched_twins", "n_account_pairs", "n_items", "reuse_r", "seed",
        "selected_lambda", "stage",
    },
    "complementarity": {
        "branch_metrics", "category", "dimensions", "incidence_diagnostics",
        "interpretation_guard", "invariants", "primary_mixed_benchmark",
        "seed", "selected_lambda", "stage", "stage4_attack_world_sha256",
        "swap_diagnostics",
    },
    "strength_fixed": {
        "category", "feasible_item_universe", "fixed_item_degree",
        "interpretation_guard", "invariants", "metrics_by_k",
        "modified_exposure_balance", "n_accounts", "n_items", "reuse_r",
        "seed", "selected_lambda", "stage", "strength_grid_k",
    },
    "full_background": {
        "category", "interpretation_guard", "invariants", "metrics",
        "population", "seed", "source_stage4", "stage", "stream",
    },
    "self_influence": {
        "category", "interpretation_guard", "invariants", "metrics",
        "population", "seed", "source_stage4", "stage",
    },
}


class VerificationError(RuntimeError):
    """Raised when an exact Electronics verification requirement fails."""


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path, *, keys: set[str] | None = None) -> dict[str, Any]:
    source = Path(path)
    ensure(source.is_file(), f"missing JSON artifact: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    ensure(isinstance(value, dict), f"JSON root is not an object: {source}")
    if keys is not None:
        ensure(set(value) == keys, f"JSON schema mismatch: {source}")
    return value


@contextmanager
def csv_rows(
    path: str | Path,
    expected_header: tuple[str, ...],
) -> Iterator[csv.DictReader]:
    source = Path(path)
    ensure(source.is_file(), f"missing CSV artifact: {source}")
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        ensure(tuple(reader.fieldnames or ()) == expected_header,
               f"CSV header mismatch: {source}")
        yield reader


def read_csv(path: str | Path, expected_header: tuple[str, ...]) -> list[dict[str, str]]:
    with csv_rows(path, expected_header) as reader:
        return list(reader)


def ensure_close(actual: float, expected: float, label: str) -> None:
    ensure(
        float_close(actual, expected, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL),
        f"floating-point mismatch for {label}: {actual!r} != {expected!r}",
    )


def ensure_exact_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    ensure(set(value) == keys, f"schema mismatch for {label}")


def ensure_true_flags(value: dict[str, Any], label: str) -> None:
    flags = {key: item for key, item in value.items() if isinstance(item, bool)}
    ensure(flags, f"no boolean checks found in {label}")
    ensure(all(flags.values()), f"false invariant or matching flag in {label}")


def ensure_category(value: dict[str, Any], label: str) -> None:
    ensure(value.get("category") == CATEGORY, f"category mismatch in {label}")
    ensure(
        "home_and_kitchen" not in json.dumps(value, sort_keys=True).lower(),
        f"Home-and-Kitchen reference found in {label}",
    )


def expected_seed_names() -> set[str]:
    return {f"seed_{seed:03d}" for seed in EXPECTED_SEEDS}


def ensure_seed_directories(stage_root: str | Path) -> None:
    source = Path(stage_root)
    ensure(source.is_dir(), f"missing stage directory: {source}")
    actual = {path.name for path in source.iterdir() if path.is_dir()}
    ensure(actual == expected_seed_names(), f"seed directory set mismatch: {source}")


def ensure_seed_files(seed_root: str | Path, expected_files: set[str]) -> None:
    source = Path(seed_root)
    actual = {path.name for path in source.iterdir() if path.is_file()}
    ensure(actual == expected_files, f"per-seed file set mismatch: {source}")


def canonical_identity_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def parse_attack_rows(path: str | Path) -> list[dict[str, Any]]:
    json_fields = (
        "treated_positions", "treated_source_lines", "original_ratings",
        "replacement_ratings", "clean_counts", "attack_counts",
    )
    result: list[dict[str, Any]] = []
    with csv_rows(path, ATTACK_HEADER) as reader:
        for raw in reader:
            row = {key: json.loads(raw[key]) for key in json_fields}
            row.update({
                "asin": raw["asin"],
                "treatment_block": raw["treatment_block"],
                "clean_w1": float(raw["clean_w1"]),
                "attack_w1": float(raw["attack_w1"]),
                "d_cf": float(raw["d_cf"]),
            })
            result.append(row)
    return result


def parse_shape_rows(path: str | Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    float_fields = (
        "clean_mean", "attack_mean", "clean_w1", "attack_w1", "d_w1",
        "clean_js", "attack_js", "d_js", "clean_abs_mean",
        "attack_abs_mean", "d_abs_mean",
    )
    with csv_rows(path, SHAPE_ATTACK_HEADER) as reader:
        for raw in reader:
            row: dict[str, Any] = {
                "asin": raw["asin"],
                "treatment_block": raw["treatment_block"],
                "slots": json.loads(raw["slots"]),
                "clean_counts": json.loads(raw["clean_counts"]),
                "attack_counts": json.loads(raw["attack_counts"]),
            }
            row.update({key: float(raw[key]) for key in float_fields})
            result.append(row)
    return result


@dataclass(frozen=True)
class VerificationContext:
    repo_root: Path
    config_dir: Path
    shared_root: Path
    report_dir: Path
    raw_path: Path
    raw_sha256: str
    n_items: int
    selected_lambda: float | None

    def config_path(self, name: str) -> Path:
        return self.config_dir / name

    def stage(self, name: str) -> Path:
        return self.shared_root / name


def build_context(
    *,
    repo_root: str | Path = ROOT,
    config_dir: str | Path | None = None,
    shared_root: str | Path | None = None,
    report_dir: str | Path | None = None,
) -> VerificationContext:
    repo = Path(repo_root).resolve()
    configs = (
        (repo / "configs" / "electronics").resolve()
        if config_dir is None else Path(config_dir).resolve()
    )
    preprocess_cfg = yaml.safe_load(
        (configs / "amazon_preprocess.yaml").read_text(encoding="utf-8")
    )
    layout = AmazonPathLayout.from_config(
        preprocess_cfg,
        repo_root=repo,
        shared_root=shared_root,
    )
    shared = (
        layout.shared_category_root
        if shared_root is None else Path(shared_root).resolve()
    )
    raw_path = layout.raw_dataset
    ensure(raw_path is not None, "Electronics preprocessing config has no raw dataset")
    audit = load_json(shared / "electronics_amazon_preprocess_audit.json")
    stage2 = load_json(shared / "stage2" / "electronics_stage2_summary.json")
    freeze_path = shared / "stage3" / "electronics_reference_freeze.json"
    freeze = load_json(freeze_path) if freeze_path.is_file() else None
    reports = (
        (repo / "artifacts" / "electronics" / "verification").resolve()
        if report_dir is None else Path(report_dir).resolve()
    )
    return VerificationContext(
        repo_root=repo,
        config_dir=configs,
        shared_root=shared,
        report_dir=reports,
        raw_path=raw_path,
        raw_sha256=str(audit["raw_dataset_sha256"]),
        n_items=int(stage2["n_items"]),
        selected_lambda=(
            float(freeze["selected_lambda"]) if freeze is not None else None
        ),
    )


@dataclass
class StageAudit:
    stage: str
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, str]] = field(default_factory=list)

    def capture(self, name: str, check: Callable[[], Any]) -> None:
        ensure(name not in self.checks, f"duplicate verifier check name: {name}")
        try:
            detail = check()
        except Exception as exc:  # report failures instead of losing the audit
            self.checks[name] = False
            self.failures.append({
                "check": name,
                "error_type": type(exc).__name__,
                "message": str(exc),
            })
        else:
            self.checks[name] = True
            if detail is not None:
                self.details[name] = detail

    @property
    def status(self) -> str:
        return "PASS" if self.checks and all(self.checks.values()) else "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "category": CATEGORY,
            "verification_mode": "invariant_only",
            "gold_comparison_performed": False,
            "independent_gold_reproduction_claimed": False,
            "stage": self.stage,
            "status": self.status,
            "verification_scope": list(VERIFICATION_SCOPE),
            "tolerances": {"rtol": DEFAULT_RTOL, "atol": DEFAULT_ATOL},
            "checks": self.checks,
            "details": self.details,
            "failures": self.failures,
        }


def verify_provenance(
    summary: dict[str, Any],
    ctx: VerificationContext,
    *,
    config_name: str,
    seed_ids: list[int],
    selected_lambda: float | None,
) -> None:
    ensure_category(summary, f"{config_name} summary")
    ensure(summary.get("raw_dataset_sha256") == ctx.raw_sha256,
           f"raw digest mismatch in {config_name}")
    provenance = summary.get("provenance")
    ensure(isinstance(provenance, dict), f"missing provenance in {config_name}")
    ensure_category(provenance, f"{config_name} provenance")
    ensure(provenance.get("raw_dataset_sha256") == ctx.raw_sha256,
           f"provenance raw digest mismatch in {config_name}")
    ensure(provenance.get("seed_ids") == seed_ids,
           f"provenance seed set mismatch in {config_name}")
    if selected_lambda is None:
        ensure(provenance.get("selected_lambda") is None,
               f"unexpected selected lambda in {config_name}")
    else:
        ensure_close(float(provenance.get("selected_lambda")), selected_lambda,
                     f"{config_name} provenance lambda")
    config = provenance.get("config", {})
    expected_logical = f"configs/electronics/{config_name}"
    ensure(config.get("path") == expected_logical,
           f"config provenance path mismatch in {config_name}")
    config_path = ctx.config_path(config_name)
    ensure(sha256_file(config_path) == config.get("sha256"),
           f"config hash mismatch in {config_name}")
    ensure(provenance.get("environment_versions") == environment_versions(),
           f"environment provenance mismatch in {config_name}")
    upstream = provenance.get("upstream_artifacts")
    ensure(isinstance(upstream, dict), f"missing upstream provenance in {config_name}")
    for label, item in upstream.items():
        logical = item.get("path", "")
        ensure(logical.startswith("shared_data/"),
               f"non-Electronics upstream path for {config_name}:{label}")
        relative = logical.removeprefix("shared_data/")
        source = (ctx.shared_root / relative).resolve()
        ensure(ctx.shared_root == source or ctx.shared_root in source.parents,
               f"upstream path escapes Electronics root: {source}")
        ensure(source.is_file(), f"missing upstream artifact: {source}")
        ensure(sha256_file(source) == item.get("sha256"),
               f"upstream hash mismatch for {config_name}:{label}")


def aggregate_path(ctx: VerificationContext, stage: str) -> Path:
    paths = {
        "reference": ctx.stage("stage3") / "electronics_stage3_summary.json",
        "primary_attack": ctx.stage("stage4_attack") / "electronics_primary_attack_summary.json",
        "primary_reuse": ctx.stage("stage4_reuse") / "electronics_stage4_summary.json",
        "matched_twins": ctx.stage("stage5_twins") / "electronics_stage5_summary.json",
        "shape": ctx.stage("stage6_shape") / "electronics_stage6_summary.json",
        "complementarity": ctx.stage("stage7_complementarity") / "electronics_stage7_summary.json",
        "reference_history": ctx.stage("stage8_reference_history") / "electronics_stage8_refhistory_summary.json",
        "strength_fixed": ctx.stage("stage9_strength_fixed") / "electronics_stage9_strength_fixed_identity_summary.json",
        "k6_population_audit": ctx.stage("stage9_k6_population_audit") / "electronics_k6_population_audit_summary.json",
        "full_background": ctx.stage("stage10_full_background") / "electronics_stage10_ranking_full_background_summary.json",
        "self_influence": ctx.stage("stage11_self_influence") / "electronics_stage11_self_influence_loo_summary.json",
    }
    return paths[stage]


def verify_inputs(ctx: VerificationContext) -> StageAudit:
    audit = StageAudit("inputs")

    def configurations() -> dict[str, Any]:
        actual = {path.name for path in ctx.config_dir.glob("*.yaml")}
        ensure(actual == set(CONFIG_FILES), "Electronics config set mismatch")
        hashes = {}
        for name in CONFIG_FILES:
            path = ctx.config_path(name)
            text = path.read_text(encoding="utf-8")
            cfg = yaml.safe_load(text)
            ensure(cfg.get("category") == CATEGORY, f"category mismatch in {name}")
            ensure("home_and_kitchen" not in text.lower(),
                   f"Home-and-Kitchen reference in {name}")
            layout = AmazonPathLayout.from_config(
                cfg,
                repo_root=ctx.repo_root,
                shared_root=ctx.shared_root,
            )
            ensure(layout.shared_category_root == ctx.shared_root,
                   f"shared root mismatch in {name}")
            hashes[name] = sha256_file(path)
        preprocess = yaml.safe_load(
            ctx.config_path("amazon_preprocess.yaml").read_text(encoding="utf-8")
        )
        ensure(preprocess["dataset"] == {
            "filename": "Electronics.jsonl.gz",
            "location": "parent_of_repository",
        }, "raw dataset binding mismatch")
        return {"config_count": len(hashes), "config_sha256": hashes}

    def raw_dataset() -> dict[str, Any]:
        ensure(ctx.raw_path.is_file(), f"missing raw dataset: {ctx.raw_path}")
        ensure(ctx.raw_path.name == "Electronics.jsonl.gz",
               "unexpected raw dataset filename")
        actual = sha256_file(ctx.raw_path)
        ensure(actual == ctx.raw_sha256, "raw Electronics SHA-256 mismatch")
        return {
            "path": external_path_label(ctx.raw_path),
            "size_bytes": ctx.raw_path.stat().st_size,
            "sha256": actual,
        }

    def summaries_and_provenance() -> dict[str, Any]:
        schemas = {
            "scan/electronics_scan.json": {
                "blank_lines", "category", "duplicate_user_item_records",
                "input_path", "invalid_rating", "malformed_json",
                "missing_required_fields", "provenance", "raw_dataset_sha256",
                "threshold_counts_after_dedup", "threshold_counts_before_dedup",
                "total_lines", "unique_items_after_dedup",
                "unique_items_before_dedup", "unique_reviewers_valid_records",
                "usable_records_after_user_item_dedup",
                "usable_reviews_per_item_after_dedup_quantiles",
                "valid_records_before_user_item_dedup",
            },
            "stage1/electronics_stage1_summary.json": {
                "category", "chronological_order", "deduplication_policy",
                "final", "ingest", "input_reviews", "outputs",
                "preliminary_counts_csv", "provenance", "rating_policy",
                "raw_dataset_sha256", "stage",
            },
            "stage2/electronics_stage2_summary.json": {
                "category", "data_usage_guards", "global_reference_prefix",
                "input_stage1_corpus", "leave_one_item_out_reference", "n_items",
                "outputs", "provenance", "raw_dataset_sha256", "role_ranges",
                "role_rating_totals", "role_review_totals", "stage",
            },
            "electronics_amazon_preprocess_audit.json": {
                "both_experimental_blocks_feasible_k6",
                "both_experimental_blocks_feasible_k9", "category",
                "duplicate_user_item_records_removed",
                "eligible_items_ge_300_after_dedup", "provenance",
                "raw_dataset_sha256", "stage2_items",
                "usable_records_after_user_item_dedup",
            },
        }
        loaded = {}
        for relative, keys in schemas.items():
            value = load_json(ctx.shared_root / relative, keys=keys)
            ensure_category(value, relative)
            verify_provenance(
                value,
                ctx,
                config_name="amazon_preprocess.yaml",
                seed_ids=[],
                selected_lambda=None,
            )
            loaded[relative] = value
        stage1 = loaded["stage1/electronics_stage1_summary.json"]
        stage2 = loaded["stage2/electronics_stage2_summary.json"]
        prep = loaded["electronics_amazon_preprocess_audit.json"]
        ensure(stage1["final"]["final_eligible_items"] == ctx.n_items,
               "Stage-1 item count mismatch")
        ensure(stage1["final"]["exported_reviews"] == ctx.n_items * 300,
               "Stage-1 review conservation mismatch")
        ensure(stage2["n_items"] == prep["stage2_items"] == ctx.n_items,
               "Stage-2/audit item count mismatch")
        ensure(prep["eligible_items_ge_300_after_dedup"] == ctx.n_items,
               "preprocessing eligible count mismatch")
        ensure_true_flags(
            {key: not value for key, value in stage2["data_usage_guards"].items()},
            "Stage-2 data-usage guards",
        )
        return {
            "summary_count": len(loaded),
            "n_items": ctx.n_items,
            "k6_feasible_items": prep["both_experimental_blocks_feasible_k6"],
            "k9_feasible_items": prep["both_experimental_blocks_feasible_k9"],
        }

    def canonical_rows() -> dict[str, Any]:
        stage1_manifest_header = (
            "asin", "usable_unique_user_reviews", "exported_reviews",
            "first_timestamp", "last_timestamp", "first_source_line",
            "last_source_line", "rating_1", "rating_2", "rating_3",
            "rating_4", "rating_5",
        )
        roles = (
            "reference", "calibration_1", "calibration_2", "experimental_A",
            "experimental_B", "holdout_1", "holdout_2",
        )
        stage2_manifest_header = ["asin", "first_timestamp", "last_timestamp"]
        for role in roles:
            stage2_manifest_header.extend([
                f"{role}_start_position", f"{role}_end_position", f"{role}_n",
                f"{role}_mean", *[f"{role}_count_{rating}" for rating in range(1, 6)],
            ])
        reference_header = ["asin", "reference_n", "reference_mean"]
        for rating in range(1, 6):
            reference_header.extend([
                f"reference_count_{rating}", f"reference_prob_{rating}",
            ])
        reference_header.extend(["loo_domain_n", "loo_domain_mean"])
        for rating in range(1, 6):
            reference_header.extend([
                f"loo_domain_count_{rating}", f"loo_domain_prob_{rating}",
            ])

        manifest1 = read_csv(
            ctx.stage("stage1") / "electronics_first300_manifest.csv",
            stage1_manifest_header,
        )
        manifest2 = read_csv(
            ctx.stage("stage2") / "electronics_stage2_manifest.csv",
            tuple(stage2_manifest_header),
        )
        references = read_csv(
            ctx.stage("stage2") / "electronics_reference_histograms.csv",
            tuple(reference_header),
        )
        ensure(len(manifest1) == len(manifest2) == len(references) == ctx.n_items,
               "preprocessing CSV row count mismatch")
        asin1 = {row["asin"] for row in manifest1}
        asin2 = {row["asin"] for row in manifest2}
        asin_ref = {row["asin"] for row in references}
        ensure(len(asin1) == ctx.n_items and asin1 == asin2 == asin_ref,
               "preprocessing ASIN universe mismatch")
        for row in manifest1:
            ensure(int(row["exported_reviews"]) == 300,
                   "Stage-1 exported review count is not 300")
            ensure(sum(int(row[f"rating_{rating}"]) for rating in range(1, 6)) == 300,
                   "Stage-1 rating histogram does not sum to 300")
        for row in manifest2:
            for role in roles:
                expected_n = 120 if role == "reference" else 30
                ensure(int(row[f"{role}_n"]) == expected_n,
                       f"role length mismatch for {role}")
                ensure(sum(int(row[f"{role}_count_{rating}"])
                           for rating in range(1, 6)) == expected_n,
                       f"role histogram mismatch for {role}")
        for row in references:
            ensure(int(row["reference_n"]) == 120,
                   "reference histogram length mismatch")
            ensure(sum(int(row[f"reference_count_{rating}"])
                       for rating in range(1, 6)) == 120,
                   "reference counts do not sum to 120")
            ensure_close(
                sum(float(row[f"reference_prob_{rating}"])
                    for rating in range(1, 6)),
                1.0,
                "reference probability sum",
            )

        corpus_asins = set()
        with gzip.open(
            ctx.stage("stage1") / "electronics_first300.items.jsonl.gz",
            "rt", encoding="utf-8",
        ) as handle:
            for line in handle:
                item = json.loads(line)
                ensure(set(item) == {"asin", "reviews"},
                       "Stage-1 JSONL schema mismatch")
                ensure(len(item["reviews"]) == 300,
                       "Stage-1 JSONL item does not contain 300 reviews")
                corpus_asins.add(item["asin"])
        ensure(corpus_asins == asin1, "Stage-1 JSONL ASIN universe mismatch")

        role_asins = set()
        expected_role_sizes = {role: (120 if role == "reference" else 30)
                               for role in roles}
        with gzip.open(
            ctx.stage("stage2") / "electronics_roles.items.jsonl.gz",
            "rt", encoding="utf-8",
        ) as handle:
            for line in handle:
                item = json.loads(line)
                ensure(set(item) == {"asin", "blocks", "role_ranges"},
                       "Stage-2 roles JSONL schema mismatch")
                ensure(set(item["blocks"]) == set(roles),
                       "Stage-2 role set mismatch")
                for role, expected_n in expected_role_sizes.items():
                    block = item["blocks"][role]
                    ensure(len(block) == expected_n,
                           f"Stage-2 block length mismatch for {role}")
                    ensure(all(set(review) == {
                        "position", "rating", "source_line", "timestamp", "user_id"
                    } for review in block), f"Stage-2 review schema mismatch for {role}")
                    ensure(all(1 <= int(review["rating"]) <= 5 for review in block),
                           f"invalid Stage-2 rating for {role}")
                role_asins.add(item["asin"])
        ensure(role_asins == asin1, "Stage-2 roles ASIN universe mismatch")
        return {
            "stage1_rows": len(manifest1),
            "stage2_rows": len(manifest2),
            "reference_rows": len(references),
            "stage1_exported_reviews": ctx.n_items * 300,
        }

    audit.capture("configuration_category_isolation", configurations)
    audit.capture("raw_dataset_sha256", raw_dataset)
    audit.capture("preprocessing_schemas_and_provenance", summaries_and_provenance)
    audit.capture("preprocessing_rows_and_conservation", canonical_rows)
    return audit


def verify_reference(ctx: VerificationContext) -> StageAudit:
    audit = StageAudit("reference")
    root = ctx.stage("stage3")

    def schema_and_provenance() -> dict[str, Any]:
        summary = load_json(aggregate_path(ctx, "reference"), keys=AGGREGATE_KEYS["reference"])
        freeze_keys = {
            "baseline_cancellation_identity", "category",
            "experimental_A_B_used_for_selection", "holdout_role",
            "holdout_used_for_selection", "lambda_grid", "provenance",
            "raw_dataset_sha256", "reference_formula",
            "reference_source_positions", "schema_version", "selected_lambda",
            "selection_source_roles", "stage4_evidence_definition",
            "stage4_evidence_mode", "stage4_invariant_requirement",
        }
        freeze = load_json(root / "electronics_reference_freeze.json", keys=freeze_keys)
        verify_provenance(summary, ctx, config_name="amazon_reference.yaml",
                          seed_ids=[], selected_lambda=ctx.selected_lambda)
        verify_provenance(freeze, ctx, config_name="amazon_reference.yaml",
                          seed_ids=[], selected_lambda=ctx.selected_lambda)
        ensure(summary["selected_lambda"] == freeze["selected_lambda"] == 20.0,
               "reference lambda mismatch")
        ensure(freeze["selection_source_roles"] == ["calibration_1", "calibration_2"],
               "reference selection-role mismatch")
        ensure(freeze["reference_source_positions"] == [1, 120],
               "reference prefix mismatch")
        ensure(freeze["experimental_A_B_used_for_selection"] is False and
               freeze["holdout_used_for_selection"] is False,
               "forbidden data used for reference selection")
        guards = summary["data_usage_guards"]
        ensure(guards == {
            "attack_generated_in_stage3": False,
            "experimental_A_B_used_in_stage3": False,
            "holdout_used_for_lambda_selection": False,
            "lambda_retuned_after_holdout": False,
            "lambda_selection_uses_only_calibration_1_2": True,
        }, "reference data-usage guard mismatch")
        return {"selected_lambda": ctx.selected_lambda, "n_items": summary["n_items"]}

    def rows_and_selection() -> dict[str, Any]:
        calibration_header = (
            "lambda", "raw_mean", "plugin_signed_mean", "plugin_abs_mean",
            "plugin_positive_fraction", "predictive_signed_mean",
            "predictive_abs_mean", "predictive_positive_fraction",
        )
        holdout_header = (
            "method", "mean", "ci_lower", "ci_upper", "n_items",
            "replicates", "level", "positive_fraction_all_blocks",
            "holdout_1_mean", "holdout_2_mean",
        )
        item_header = (
            "asin", "lambda", "holdout_1_raw_w1", "holdout_2_raw_w1",
            "holdout_1_plugin_signed", "holdout_2_plugin_signed",
            "holdout_1_predictive_signed", "holdout_2_predictive_signed",
        )
        chronology_header = (
            "role", "predictive_signed_mean", "plugin_signed_mean",
        )
        calibration = read_csv(root / "electronics_lambda_calibration.csv",
                               calibration_header)
        holdout = read_csv(root / "electronics_holdout_validation.csv", holdout_header)
        items = read_csv(root / "electronics_holdout_item_metrics.csv", item_header)
        chronology = read_csv(root / "electronics_selected_lambda_chronology.csv",
                              chronology_header)
        ensure(len(calibration) == 7, "reference calibration row count mismatch")
        expected_grid = [0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
        ensure([float(row["lambda"]) for row in calibration] == expected_grid,
               "reference lambda grid mismatch")
        best = min(calibration, key=lambda row: (
            float(row["predictive_abs_mean"]), float(row["lambda"])
        ))
        ensure(float(best["lambda"]) == ctx.selected_lambda,
               "selected lambda is not the calibration optimum")
        ensure(len(holdout) == 3 and {row["method"] for row in holdout} == {
            "raw_w1", "plugin_centered_w1", "predictive_centered_w1"
        }, "holdout summary schema/rows mismatch")
        ensure(all(int(row["n_items"]) == ctx.n_items and
                   int(row["replicates"]) == 10000 for row in holdout),
               "holdout count mismatch")
        ensure(len(items) == ctx.n_items and
               len({row["asin"] for row in items}) == ctx.n_items,
               "holdout item universe mismatch")
        ensure(all(float(row["lambda"]) == ctx.selected_lambda for row in items),
               "holdout item lambda mismatch")
        ensure(len(chronology) == 4 and [row["role"] for row in chronology] == [
            "calibration_1", "calibration_2", "holdout_1", "holdout_2"
        ], "reference chronology mismatch")
        summary = load_json(aggregate_path(ctx, "reference"))
        selected = summary["selected_calibration_row"]
        for key, value in best.items():
            ensure_close(float(selected[key]), float(value),
                         f"reference selected row {key}")
        return {
            "calibration_rows": len(calibration),
            "holdout_rows": len(holdout),
            "holdout_item_rows": len(items),
            "selected_lambda": ctx.selected_lambda,
        }

    audit.capture("schemas_category_and_provenance", schema_and_provenance)
    audit.capture("row_counts_selection_and_tolerances", rows_and_selection)
    return audit


def verify_primary_attack(ctx: VerificationContext) -> StageAudit:
    audit = StageAudit("primary_attack")
    root = ctx.stage("stage4_attack")

    def seed_set_and_files() -> dict[str, Any]:
        ensure_seed_directories(root)
        for seed in EXPECTED_SEEDS:
            ensure_seed_files(root / f"seed_{seed:03d}", {
                "attack_summary.json", "attack_world.csv.gz"
            })
        return {"seed_ids": list(EXPECTED_SEEDS), "seed_directories": 30}

    def aggregate() -> dict[str, Any]:
        summary = load_json(aggregate_path(ctx, "primary_attack"),
                            keys=AGGREGATE_KEYS["primary_attack"])
        verify_provenance(summary, ctx, config_name="amazon_primary.yaml",
                          seed_ids=list(EXPECTED_SEEDS),
                          selected_lambda=ctx.selected_lambda)
        ensure(summary["n_seeds"] == 30 and summary["seed_ids"] == list(EXPECTED_SEEDS),
               "primary-attack aggregate seed set mismatch")
        ensure(summary["all_attack_invariants_pass"] is True,
               "primary-attack aggregate invariant failed")
        ensure(summary["identity_assignment_performed"] is False,
               "identity assignment leaked into attack construction")
        return {"aggregate_summary_sha256": sha256_file(aggregate_path(ctx, "primary_attack"))}

    def per_seed() -> dict[str, Any]:
        metrics = read_csv(root / "electronics_primary_attack_seed_metrics.csv",
                           ATTACK_METRICS_HEADER)
        ensure(len(metrics) == 30 and {int(row["seed"]) for row in metrics} == set(EXPECTED_SEEDS),
               "primary-attack seed metrics mismatch")
        by_seed = {int(row["seed"]): row for row in metrics}
        means = []
        total_rows = 0
        for seed in EXPECTED_SEEDS:
            directory = root / f"seed_{seed:03d}"
            summary = load_json(directory / "attack_summary.json",
                                keys=SEED_KEYS["primary_attack"])
            ensure_category(summary, f"primary attack seed {seed}")
            ensure(summary["seed"] == seed and summary["sampled_items"] == 2000 and
                   summary["m"] == 6 and summary["selected_lambda"] == ctx.selected_lambda,
                   f"primary attack discrete mismatch for seed {seed}")
            ensure_true_flags(summary["invariants"], f"primary attack seed {seed}")
            rows = parse_attack_rows(directory / "attack_world.csv.gz")
            ensure(len(rows) == 2000 and len({row["asin"] for row in rows}) == 2000,
                   f"primary attack row count mismatch for seed {seed}")
            for row in rows:
                ensure(len(row["treated_positions"]) ==
                       len(row["treated_source_lines"]) ==
                       len(row["original_ratings"]) ==
                       len(row["replacement_ratings"]) == 6,
                       f"primary attack slot count mismatch for seed {seed}")
                ensure(len(set(row["treated_source_lines"])) == 6,
                       f"duplicate primary donor slot for seed {seed}")
                ensure(all(int(value) != 5 for value in row["original_ratings"]),
                       f"five-star primary donor for seed {seed}")
                ensure(row["replacement_ratings"] == [5] * 6,
                       f"non-five primary replacement for seed {seed}")
                ensure(sum(row["clean_counts"]) == sum(row["attack_counts"]) == 30,
                       f"primary histogram conservation failed for seed {seed}")
            digest = canonical_attack_hash(rows)
            ensure(digest == summary["attack_world_sha256"],
                   f"primary canonical attack hash mismatch for seed {seed}")
            metric = by_seed[seed]
            ensure(metric["attack_world_sha256"] == digest and
                   int(metric["n_items"]) == 2000 and int(metric["m"]) == 6,
                   f"primary seed metric discrete mismatch for seed {seed}")
            for field in (
                "mean_d_cf", "positive_d_cf_fraction", "zero_d_cf_fraction",
                "negative_d_cf_fraction", "clean_w1_mean", "attack_w1_mean",
            ):
                ensure_close(float(metric[field]),
                             float(summary["aggregate_visibility"][field.replace(
                                 "mean_d_cf", "mean_d_cf"
                             ).replace("positive_d_cf_fraction", "positive_fraction")
                               .replace("zero_d_cf_fraction", "zero_fraction")
                               .replace("negative_d_cf_fraction", "negative_fraction")]),
                             f"primary seed {seed} {field}")
            means.append(float(metric["mean_d_cf"]))
            total_rows += len(rows)
        global_summary = load_json(aggregate_path(ctx, "primary_attack"))
        ensure_close(global_summary["aggregate_visibility_mean_d_cf"]["mean"],
                     fmean(means), "primary aggregate mean d_cf")
        return {
            "attack_rows": total_rows,
            "manipulated_slots": total_rows * 6,
            "canonical_hashes_recomputed": 30,
        }

    audit.capture("seed_sets_and_expected_files", seed_set_and_files)
    audit.capture("aggregate_schema_category_and_provenance", aggregate)
    audit.capture("per_seed_rows_hashes_invariants_and_tolerances", per_seed)
    return audit


def verify_primary_reuse(ctx: VerificationContext) -> StageAudit:
    audit = StageAudit("primary_reuse")
    root = ctx.stage("stage4_reuse")
    attack_root = ctx.stage("stage4_attack")

    expected_files = {"reuse_metrics.csv", "reuse_summary.json"}
    expected_files.update({f"identity_assignment_r{reuse}.csv.gz" for reuse in REUSE_GRID})
    expected_files.update({f"account_metrics_r{reuse}.csv.gz" for reuse in REUSE_GRID})

    def seed_set_and_files() -> dict[str, Any]:
        ensure_seed_directories(root)
        for seed in EXPECTED_SEEDS:
            ensure_seed_files(root / f"seed_{seed:03d}", expected_files)
        return {"seed_ids": list(EXPECTED_SEEDS), "files_per_seed": len(expected_files)}

    def aggregate() -> dict[str, Any]:
        summary = load_json(aggregate_path(ctx, "primary_reuse"),
                            keys=AGGREGATE_KEYS["primary_reuse"])
        verify_provenance(summary, ctx, config_name="amazon_primary.yaml",
                          seed_ids=list(EXPECTED_SEEDS),
                          selected_lambda=ctx.selected_lambda)
        ensure(summary["n_seeds"] == 30 and summary["seed_ids"] == list(EXPECTED_SEEDS),
               "primary-reuse aggregate seed set mismatch")
        ensure(set(summary["reuse_summary"]) == {str(value) for value in REUSE_GRID},
               "primary-reuse aggregate grid mismatch")
        ensure(all(summary["reuse_summary"][str(value)]["n_seeds"] == 30
                   for value in REUSE_GRID), "primary-reuse aggregate count mismatch")
        return {"reuse_grid": list(REUSE_GRID)}

    def per_seed() -> dict[str, Any]:
        global_rows = read_csv(root / "electronics_stage4_seed_metrics.csv",
                               REUSE_METRICS_HEADER)
        expected_keys = {(seed, reuse) for seed in EXPECTED_SEEDS for reuse in REUSE_GRID}
        ensure(len(global_rows) == 150 and
               {(int(row["seed"]), int(row["reuse_r"])) for row in global_rows} == expected_keys,
               "primary-reuse aggregate seed/reuse rows mismatch")
        global_by_key = {(int(row["seed"]), int(row["reuse_r"])): row
                         for row in global_rows}
        total_edges = 0
        total_account_rows = 0
        means_by_reuse: dict[int, list[float]] = defaultdict(list)
        for seed in EXPECTED_SEEDS:
            directory = root / f"seed_{seed:03d}"
            summary = load_json(directory / "reuse_summary.json",
                                keys=SEED_KEYS["primary_reuse"])
            ensure_category(summary, f"primary reuse seed {seed}")
            ensure(summary["seed"] == seed and summary["sampled_items"] == 2000 and
                   summary["m"] == 6 and summary["reuse_grid"] == list(REUSE_GRID),
                   f"primary-reuse discrete mismatch for seed {seed}")
            ensure_true_flags(summary["invariants"], f"primary reuse seed {seed}")
            attack_summary = load_json(
                attack_root / f"seed_{seed:03d}" / "attack_summary.json"
            )
            ensure(summary["attack_world_sha256"] == attack_summary["attack_world_sha256"],
                   f"primary-reuse attack handoff mismatch for seed {seed}")
            attack_rows = parse_attack_rows(
                attack_root / f"seed_{seed:03d}" / "attack_world.csv.gz"
            )
            attack_by_asin = {row["asin"]: row for row in attack_rows}
            local_metrics = read_csv(directory / "reuse_metrics.csv", REUSE_METRICS_HEADER)
            ensure(len(local_metrics) == 5 and
                   [int(row["reuse_r"]) for row in local_metrics] == list(REUSE_GRID),
                   f"primary-reuse local metric rows mismatch for seed {seed}")
            for metric in local_metrics:
                reuse = int(metric["reuse_r"])
                assignment_rows: list[dict[str, Any]] = []
                with csv_rows(directory / f"identity_assignment_r{reuse}.csv.gz",
                              IDENTITY_HEADER) as reader:
                    for raw in reader:
                        assignment_rows.append({
                            "asin": raw["asin"],
                            "treatment_block": raw["treatment_block"],
                            "treated_position": int(raw["treated_position"]),
                            "treated_source_line": int(raw["treated_source_line"]),
                            "synthetic_account_id": raw["synthetic_account_id"],
                            "d_cf": float(raw["d_cf"]),
                        })
                ensure(len(assignment_rows) == 12000,
                       f"identity edge count mismatch for seed {seed}, r={reuse}")
                item_degree = Counter(row["asin"] for row in assignment_rows)
                account_degree = Counter(row["synthetic_account_id"]
                                         for row in assignment_rows)
                ensure(len(item_degree) == 2000 and set(item_degree.values()) == {6},
                       f"item degree mismatch for seed {seed}, r={reuse}")
                ensure(len(account_degree) == 12000 // reuse and
                       set(account_degree.values()) == {reuse},
                       f"account degree mismatch for seed {seed}, r={reuse}")
                ensure(len({(row["asin"], row["synthetic_account_id"])
                            for row in assignment_rows}) == 12000,
                       f"duplicate account/item edge for seed {seed}, r={reuse}")
                for row in assignment_rows:
                    attack = attack_by_asin[row["asin"]]
                    ensure_close(row["d_cf"], attack["d_cf"],
                                 f"reuse d_cf seed {seed}, r={reuse}")
                    slot = (row["treated_position"], row["treated_source_line"])
                    ensure(slot in set(zip(attack["treated_positions"],
                                           attack["treated_source_lines"])),
                           f"reuse slot mismatch for seed {seed}, r={reuse}")
                identity_hash = canonical_identity_hash(assignment_rows)
                ensure(identity_hash ==
                       summary["identity_assignment_sha256_by_reuse"][str(reuse)] ==
                       metric["identity_assignment_sha256"],
                       f"identity hash mismatch for seed {seed}, r={reuse}")

                with csv_rows(directory / f"account_metrics_r{reuse}.csv.gz",
                              ACCOUNT_METRICS_HEADER) as reader:
                    account_rows = list(reader)
                expected_accounts = int(metric["coalition_accounts"]) + int(metric["normal_accounts"])
                ensure(len(account_rows) == expected_accounts and
                       len({row["account_id"] for row in account_rows}) == expected_accounts,
                       f"account metric row mismatch for seed {seed}, r={reuse}")
                coalition = [row for row in account_rows
                             if int(row["is_synthetic_coalition"]) == 1]
                ensure(len(coalition) == 12000 // reuse and
                       all(int(row["frequency"]) == reuse for row in coalition),
                       f"coalition account mismatch for seed {seed}, r={reuse}")
                coalition_total = sum(float(row["score"]) for row in coalition)
                ensure_close(coalition_total, float(metric["coalition_score_total"]),
                             f"reuse score conservation seed {seed}, r={reuse}")
                ensure_close(float(metric["coalition_mean_score"]),
                             reuse * float(metric["mean_d_cf"]),
                             f"reuse law seed {seed}, r={reuse}")
                ensure_close(float(metric["coalition_score_total"]),
                             float(metric["expected_coalition_score_total_m_times_sum_dcf"]),
                             f"reuse total conservation seed {seed}, r={reuse}")
                global_metric = global_by_key[(seed, reuse)]
                for field in REUSE_METRICS_HEADER:
                    if field in {"seed", "reuse_r", "attack_world_sha256",
                                 "identity_assignment_sha256", "n_items", "m",
                                 "coalition_accounts", "normal_accounts"}:
                        ensure(global_metric[field] == metric[field],
                               f"reuse exact aggregate field mismatch: {field}")
                    else:
                        ensure_close(float(global_metric[field]), float(metric[field]),
                                     f"reuse aggregate seed {seed}, r={reuse}, {field}")
                means_by_reuse[reuse].append(float(metric["mean_d_cf"]))
                total_edges += len(assignment_rows)
                total_account_rows += len(account_rows)
        aggregate_summary = load_json(aggregate_path(ctx, "primary_reuse"))
        for reuse, values in means_by_reuse.items():
            ensure_close(
                aggregate_summary["reuse_summary"][str(reuse)]["mean_d_cf"]["mean"],
                fmean(values), f"aggregate reuse mean d_cf r={reuse}",
            )
        return {
            "seed_reuse_rows": len(global_rows),
            "identity_edges": total_edges,
            "account_metric_rows": total_account_rows,
            "identity_hashes_recomputed": 150,
        }

    audit.capture("seed_sets_and_expected_files", seed_set_and_files)
    audit.capture("aggregate_schema_category_and_provenance", aggregate)
    audit.capture("degrees_hashes_rows_and_conservation", per_seed)
    return audit


def verify_matched_twins(ctx: VerificationContext) -> StageAudit:
    audit = StageAudit("matched_twins")
    root = ctx.stage("stage5_twins")
    attack_root = ctx.stage("stage4_attack")

    def seed_set_and_files() -> dict[str, Any]:
        ensure_seed_directories(root)
        expected = {
            "matched_twin_accounts.csv.gz", "matched_twin_edges.csv.gz",
            "summary.json",
        }
        for seed in EXPECTED_SEEDS:
            ensure_seed_files(root / f"seed_{seed:03d}", expected)
        return {"seed_ids": list(EXPECTED_SEEDS), "pairs_per_seed": 1500}

    def aggregate() -> dict[str, Any]:
        summary = load_json(aggregate_path(ctx, "matched_twins"),
                            keys=AGGREGATE_KEYS["matched_twins"])
        verify_provenance(summary, ctx, config_name="amazon_twins.yaml",
                          seed_ids=list(EXPECTED_SEEDS),
                          selected_lambda=ctx.selected_lambda)
        ensure(summary["n_seeds"] == 30 and summary["seed_ids"] == list(EXPECTED_SEEDS)
               and summary["reuse_r"] == 8, "matched-twin aggregate mismatch")
        ensure_true_flags(summary["matching"], "matched-twin aggregate")
        ensure(summary["metrics"]["frequency_auc"]["mean"] == 0.5,
               "matched-twin aggregate frequency AUC is not 0.5")
        return {"reuse_r": 8}

    def per_seed() -> dict[str, Any]:
        total_pairs = 0
        total_edges = 0
        auc_values = []
        for seed in EXPECTED_SEEDS:
            directory = root / f"seed_{seed:03d}"
            summary = load_json(directory / "summary.json",
                                keys=SEED_KEYS["matched_twins"])
            ensure_category(summary, f"matched twins seed {seed}")
            ensure(summary["seed"] == seed and summary["reuse_r"] == 8 and
                   summary["n_pairs"] == 1500 and
                   summary["n_edges_per_group"] == 12000 and
                   summary["selected_lambda"] == ctx.selected_lambda,
                   f"matched-twin discrete mismatch for seed {seed}")
            ensure_true_flags(summary["matching"], f"matched twins seed {seed}")
            attack_summary = load_json(
                attack_root / f"seed_{seed:03d}" / "attack_summary.json"
            )
            ensure(summary["stage4_attack_world_sha256"] ==
                   attack_summary["attack_world_sha256"],
                   f"matched-twin attack hash mismatch for seed {seed}")
            accounts = read_csv(directory / "matched_twin_accounts.csv.gz",
                                TWIN_ACCOUNTS_HEADER)
            edges = read_csv(directory / "matched_twin_edges.csv.gz", TWIN_EDGES_HEADER)
            ensure(len(accounts) == 1500 and len(edges) == 12000,
                   f"matched-twin row count mismatch for seed {seed}")
            ensure(len({row["pair_id"] for row in accounts}) == 1500,
                   f"duplicate matched-twin pair for seed {seed}")
            counts = Counter(row["pair_id"] for row in edges)
            ensure(set(counts) == {row["pair_id"] for row in accounts} and
                   set(counts.values()) == {8},
                   f"matched-twin exposure degree mismatch for seed {seed}")
            items: dict[str, set[str]] = defaultdict(set)
            slots: dict[str, set[tuple[str, str, str, str]]] = defaultdict(set)
            gaps: dict[str, float] = defaultdict(float)
            for row in edges:
                ensure(int(row["clean_rating"]) != 5 and
                       int(row["attack_rating"]) == 5,
                       f"matched-twin rating mismatch for seed {seed}")
                items[row["pair_id"]].add(row["asin"])
                slots[row["pair_id"]].add((
                    row["asin"], row["block"], row["position"], row["source_line"]
                ))
                gaps[row["pair_id"]] += float(row["d_cf"])
            for row in accounts:
                pair = row["pair_id"]
                ensure(row["pair_id"] == row["coalition_account_id"] and
                       int(row["frequency_each"]) == 8,
                       f"matched-twin identity/frequency mismatch for seed {seed}")
                ensure(len(json.loads(row["item_set_json"])) ==
                       len(json.loads(row["slot_set_json"])) == 8 and
                       len(items[pair]) == len(slots[pair]) == 8,
                       f"matched-twin exact exposure mismatch for seed {seed}")
                ensure(float(row["control_counterfactual_score"]) == 0.0,
                       f"matched-twin control score is not zero for seed {seed}")
                ensure_close(float(row["paired_score_gap"]), gaps[pair],
                             f"matched-twin paired gap seed {seed}")
            metrics = summary["metrics"]
            ensure(metrics["frequency_auc"] == 0.5,
                   f"matched-twin frequency AUC mismatch for seed {seed}")
            ensure_close(metrics["mean_paired_score_gap"],
                         8 * metrics["mean_d_cf"],
                         f"matched-twin reuse law seed {seed}")
            ensure_close(metrics["total_paired_gap"],
                         metrics["expected_total_m_times_sum_dcf"],
                         f"matched-twin total conservation seed {seed}")
            auc_values.append(float(metrics["counterfactual_score_auc"]))
            total_pairs += len(accounts)
            total_edges += len(edges)
        aggregate_summary = load_json(aggregate_path(ctx, "matched_twins"))
        ensure_close(aggregate_summary["metrics"]["counterfactual_score_auc"]["mean"],
                     fmean(auc_values), "matched-twin aggregate AUC")
        return {"pairs": total_pairs, "edges": total_edges}

    audit.capture("seed_sets_and_expected_files", seed_set_and_files)
    audit.capture("aggregate_schema_category_and_provenance", aggregate)
    audit.capture("exact_matching_hash_handoffs_and_conservation", per_seed)
    return audit


def verify_shape(ctx: VerificationContext) -> StageAudit:
    audit = StageAudit("shape")
    root = ctx.stage("stage6_shape")

    def seed_set_and_files() -> dict[str, Any]:
        ensure_seed_directories(root)
        expected = {
            "shape_attack_world.csv.gz", "shape_identity_assignment_r8.csv.gz",
            "shape_matched_twin_accounts.csv.gz", "summary.json",
        }
        for seed in EXPECTED_SEEDS:
            ensure_seed_files(root / f"seed_{seed:03d}", expected)
        return {"seed_ids": list(EXPECTED_SEEDS), "items_per_seed": 2000}

    def aggregate() -> dict[str, Any]:
        summary = load_json(aggregate_path(ctx, "shape"), keys=AGGREGATE_KEYS["shape"])
        verify_provenance(summary, ctx, config_name="amazon_shape.yaml",
                          seed_ids=list(EXPECTED_SEEDS),
                          selected_lambda=ctx.selected_lambda)
        ensure(summary["n_seeds"] == 30 and summary["seed_ids"] == list(EXPECTED_SEEDS)
               and summary["n_items_per_seed"] == 2000 and summary["reuse_r"] == 8,
               "shape aggregate discrete mismatch")
        ensure_true_flags(summary["invariants"], "shape aggregate")
        ensure(summary["metrics"]["mean_d_abs_mean"]["mean"] == 0.0 and
               summary["metrics"]["max_abs_d_abs_mean"]["mean"] == 0.0,
               "shape aggregate mean preservation failed")
        return {"reuse_r": 8, "n_items_per_seed": 2000}

    def per_seed() -> dict[str, Any]:
        total_rows = 0
        total_edges = 0
        w1_auc = []
        for seed in EXPECTED_SEEDS:
            directory = root / f"seed_{seed:03d}"
            summary = load_json(directory / "summary.json", keys=SEED_KEYS["shape"])
            ensure_category(summary, f"shape seed {seed}")
            ensure(summary["seed"] == seed and summary["n_items"] == 2000 and
                   summary["m"] == 6 and summary["n_account_pairs"] == 1500 and
                   summary["reuse_r"] == 8 and
                   summary["selected_lambda"] == ctx.selected_lambda,
                   f"shape discrete mismatch for seed {seed}")
            ensure_true_flags(summary["invariants"], f"shape seed {seed}")
            rows = parse_shape_rows(directory / "shape_attack_world.csv.gz")
            ensure(len(rows) == 2000 and len({row["asin"] for row in rows}) == 2000,
                   f"shape row count mismatch for seed {seed}")
            attack_by_asin = {row["asin"]: row for row in rows}
            for row in rows:
                ensure(len(row["slots"]) == 6 and
                       len({slot["source_line"] for slot in row["slots"]}) == 6,
                       f"shape slot mismatch for seed {seed}")
                ensure(all(slot["original_rating"] != slot["replacement_rating"]
                           for slot in row["slots"]),
                       f"shape unchanged slot for seed {seed}")
                ensure(sum(row["clean_counts"]) == sum(row["attack_counts"]) == 30,
                       f"shape histogram conservation failed for seed {seed}")
                ensure(row["clean_mean"] == row["attack_mean"] and
                       row["d_abs_mean"] == 0.0,
                       f"shape exact mean preservation failed for seed {seed}")
            ensure(shape_attack_hash(rows) == summary["attack_world_sha256"],
                   f"shape canonical hash mismatch for seed {seed}")
            assignments = read_csv(directory / "shape_identity_assignment_r8.csv.gz",
                                   SHAPE_IDENTITY_HEADER)
            ensure(len(assignments) == 12000,
                   f"shape assignment count mismatch for seed {seed}")
            item_degree = Counter(row["asin"] for row in assignments)
            account_degree = Counter(row["account_id"] for row in assignments)
            ensure(len(item_degree) == 2000 and set(item_degree.values()) == {6} and
                   len(account_degree) == 1500 and set(account_degree.values()) == {8},
                   f"shape degree mismatch for seed {seed}")
            for assignment in assignments:
                attack = attack_by_asin[assignment["asin"]]
                slot_keys = {
                    (str(slot["position"]), str(slot["source_line"]))
                    for slot in attack["slots"]
                }
                ensure((assignment["position"], assignment["source_line"]) in slot_keys,
                       f"shape assignment slot mismatch for seed {seed}")
                ensure(int(assignment["original_rating"]) !=
                       int(assignment["replacement_rating"]),
                       f"shape assignment rating mismatch for seed {seed}")
                ensure(float(assignment["d_abs_mean"]) == 0.0,
                       f"shape assignment mean effect is nonzero for seed {seed}")
            twins = read_csv(directory / "shape_matched_twin_accounts.csv.gz",
                             SHAPE_TWINS_HEADER)
            ensure(len(twins) == 1500 and all(
                int(row["frequency_each"]) == 8 and
                float(row["abs_mean_paired_gap"]) == 0.0 for row in twins
            ), f"shape matched-twin mismatch for seed {seed}")
            ensure(summary["aggregate"]["mean_d_abs_mean"] == 0.0 and
                   summary["aggregate"]["max_abs_d_abs_mean"] == 0.0 and
                   summary["matched_twins"]["frequency_auc"] == 0.5,
                   f"shape summary conservation mismatch for seed {seed}")
            w1_auc.append(float(summary["matched_twins"]["w1_counterfactual_score_auc"]))
            total_rows += len(rows)
            total_edges += len(assignments)
        aggregate_summary = load_json(aggregate_path(ctx, "shape"))
        ensure_close(aggregate_summary["metrics"]["w1_counterfactual_score_auc"]["mean"],
                     fmean(w1_auc), "shape aggregate W1 AUC")
        return {"attack_rows": total_rows, "identity_edges": total_edges,
                "canonical_hashes_recomputed": 30}

    audit.capture("seed_sets_and_expected_files", seed_set_and_files)
    audit.capture("aggregate_schema_category_and_provenance", aggregate)
    audit.capture("mean_preservation_degrees_hashes_and_tolerances", per_seed)
    return audit


def verify_complementarity(ctx: VerificationContext) -> StageAudit:
    audit = StageAudit("complementarity")
    root = ctx.stage("stage7_complementarity")
    attack_root = ctx.stage("stage4_attack")
    incidence_files = (
        "random_incidence_1.csv.gz", "random_incidence_2.csv.gz",
        "team_incidence.csv.gz",
    )

    def seed_set_and_files() -> dict[str, Any]:
        ensure_seed_directories(root)
        expected = {*incidence_files, "summary.json"}
        for seed in EXPECTED_SEEDS:
            ensure_seed_files(root / f"seed_{seed:03d}", expected)
        return {"seed_ids": list(EXPECTED_SEEDS), "incidence_files": 90}

    def aggregate() -> dict[str, Any]:
        summary = load_json(aggregate_path(ctx, "complementarity"),
                            keys=AGGREGATE_KEYS["complementarity"])
        verify_provenance(summary, ctx, config_name="amazon_complementarity.yaml",
                          seed_ids=list(EXPECTED_SEEDS),
                          selected_lambda=ctx.selected_lambda)
        ensure(summary["n_seeds"] == 30 and summary["seed_ids"] == list(EXPECTED_SEEDS),
               "complementarity aggregate seed mismatch")
        ensure_true_flags(summary["invariants"], "complementarity aggregate")
        ensure(summary["dimensions"] == {
            "account_degree_r": 8, "item_degree_m": 6,
            "n_accounts_per_branch": 1500, "n_items": 2000,
            "team_repetitions": 8, "team_size": 6,
        }, "complementarity aggregate dimensions mismatch")
        return {"dimensions": summary["dimensions"]}

    def per_seed() -> dict[str, Any]:
        total_edges = 0
        combined_auc = []
        for seed in EXPECTED_SEEDS:
            directory = root / f"seed_{seed:03d}"
            summary = load_json(directory / "summary.json",
                                keys=SEED_KEYS["complementarity"])
            ensure_category(summary, f"complementarity seed {seed}")
            ensure(summary["seed"] == seed and
                   summary["selected_lambda"] == ctx.selected_lambda,
                   f"complementarity discrete mismatch for seed {seed}")
            ensure_true_flags(summary["invariants"], f"complementarity seed {seed}")
            attack = load_json(attack_root / f"seed_{seed:03d}" / "attack_summary.json")
            ensure(summary["stage4_attack_world_sha256"] == attack["attack_world_sha256"],
                   f"complementarity attack hash mismatch for seed {seed}")
            ensure(summary["dimensions"] == {
                "account_degree_r": 8, "item_degree_m": 6,
                "n_accounts_per_branch": 1500, "n_items": 2000,
                "team_repetitions": 8, "team_size": 6,
            }, f"complementarity dimensions mismatch for seed {seed}")
            for name in incidence_files:
                rows = read_csv(directory / name, INCIDENCE_HEADER)
                ensure(len(rows) == 12000 and
                       len({(row["account_id"], row["asin"]) for row in rows}) == 12000,
                       f"complementarity edge mismatch for seed {seed}: {name}")
                item_degree = Counter(row["asin"] for row in rows)
                account_degree = Counter(row["account_id"] for row in rows)
                ensure(len(item_degree) == 2000 and set(item_degree.values()) == {6} and
                       len(account_degree) == 1500 and set(account_degree.values()) == {8},
                       f"complementarity degree mismatch for seed {seed}: {name}")
                total_edges += len(rows)
            ensure(summary["primary_mixed_benchmark"]["frequency_auc"] == 0.5 and
                   summary["branch_metrics"]["topology_only"]["aggregate_auc"] == 0.5 and
                   summary["branch_metrics"]["evidence_only"]["coactivity_auc"] == 0.5,
                   f"complementarity null constraint mismatch for seed {seed}")
            combined_auc.append(float(summary["primary_mixed_benchmark"]["combined_auc"]))
        aggregate_summary = load_json(aggregate_path(ctx, "complementarity"))
        ensure_close(aggregate_summary["metrics"]["combined_auc"]["mean"],
                     fmean(combined_auc), "complementarity aggregate combined AUC")
        return {"incidence_edges": total_edges, "incidence_files": 90}

    audit.capture("seed_sets_and_expected_files", seed_set_and_files)
    audit.capture("aggregate_schema_category_and_provenance", aggregate)
    audit.capture("incidence_degrees_hash_handoffs_and_null_constraints", per_seed)
    return audit


def verify_reference_history(ctx: VerificationContext) -> StageAudit:
    audit = StageAudit("reference_history")
    root = ctx.stage("stage8_reference_history")

    def files_schema_and_provenance() -> dict[str, Any]:
        expected = {
            "electronics_reference_history_calibration.csv",
            "electronics_reference_history_seed_metrics.csv",
            "electronics_stage8_refhistory_summary.json",
        }
        ensure({path.name for path in root.iterdir() if path.is_file()} == expected,
               "reference-history file set mismatch")
        summary = load_json(aggregate_path(ctx, "reference_history"),
                            keys=AGGREGATE_KEYS["reference_history"])
        verify_provenance(summary, ctx,
                          config_name="amazon_reference_history.yaml",
                          seed_ids=list(EXPECTED_SEEDS),
                          selected_lambda=ctx.selected_lambda)
        ensure(summary["n_seeds"] == 30 and summary["seed_ids"] == list(EXPECTED_SEEDS)
               and summary["reference_lengths"] == list(REFERENCE_LENGTHS) and
               summary["reuse_r"] == 8,
               "reference-history aggregate discrete mismatch")
        ensure_true_flags(summary["invariants"], "reference-history aggregate")
        return {"reference_lengths": list(REFERENCE_LENGTHS)}

    def rows_and_handoffs() -> dict[str, Any]:
        calibration_header = (
            "reference_length", "lambda", "predictive_signed_mean",
            "predictive_abs_mean",
        )
        metric_header = (
            "category", "seed", "reference_length", "selected_lambda",
            "mean_d_cf", "counterfactual_score_auc", "positive_pair_fraction",
            "paired_misordering_probability", "mean_paired_score_gap",
        )
        calibration = read_csv(
            root / "electronics_reference_history_calibration.csv",
            calibration_header,
        )
        metrics = read_csv(
            root / "electronics_reference_history_seed_metrics.csv",
            metric_header,
        )
        lambda_grid = (0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
        ensure(len(calibration) == 21 and {
            (int(row["reference_length"]), float(row["lambda"]))
            for row in calibration
        } == {(length, value) for length in REFERENCE_LENGTHS for value in lambda_grid},
               "reference-history calibration grid mismatch")
        selected = {}
        for length in REFERENCE_LENGTHS:
            rows = [row for row in calibration
                    if int(row["reference_length"]) == length]
            best = min(rows, key=lambda row: (
                float(row["predictive_abs_mean"]), float(row["lambda"])
            ))
            selected[length] = float(best["lambda"])
        ensure(selected == {60: 20.0, 90: 20.0, 120: 20.0},
               "reference-history selected lambda mismatch")
        expected_keys = {(seed, length) for seed in EXPECTED_SEEDS
                         for length in REFERENCE_LENGTHS}
        ensure(len(metrics) == 90 and {
            (int(row["seed"]), int(row["reference_length"])) for row in metrics
        } == expected_keys and {row["category"] for row in metrics} == {CATEGORY},
               "reference-history seed rows mismatch")
        by_key = {(int(row["seed"]), int(row["reference_length"])): row
                  for row in metrics}
        for seed in EXPECTED_SEEDS:
            attack = load_json(
                ctx.stage("stage4_attack") / f"seed_{seed:03d}" / "attack_summary.json"
            )
            twins = load_json(
                ctx.stage("stage5_twins") / f"seed_{seed:03d}" / "summary.json"
            )
            endpoint = by_key[(seed, 120)]
            ensure_close(float(endpoint["mean_d_cf"]),
                         attack["aggregate_visibility"]["mean_d_cf"],
                         f"reference-history Stage-4 endpoint seed {seed}")
            ensure_close(float(endpoint["counterfactual_score_auc"]),
                         twins["metrics"]["counterfactual_score_auc"],
                         f"reference-history Stage-5 endpoint seed {seed}")
        aggregate_summary = load_json(aggregate_path(ctx, "reference_history"))
        for length in REFERENCE_LENGTHS:
            values = [float(by_key[(seed, length)]["counterfactual_score_auc"])
                      for seed in EXPECTED_SEEDS]
            ensure_close(
                aggregate_summary["metrics_by_reference_length"][str(length)]
                                 ["counterfactual_score_auc"]["mean"],
                fmean(values), f"reference-history aggregate AUC n_ref={length}",
            )
        return {"calibration_rows": 21, "seed_reference_rows": 90}

    audit.capture("files_schema_category_and_provenance", files_schema_and_provenance)
    audit.capture("rows_selection_hash_handoffs_and_tolerances", rows_and_handoffs)
    return audit


def verify_strength_fixed(ctx: VerificationContext) -> StageAudit:
    audit = StageAudit("strength_fixed")
    root = ctx.stage("stage9_strength_fixed")

    def seed_set_and_files() -> dict[str, Any]:
        ensure_seed_directories(root)
        expected = {"summary.json", *{
            f"attack_world_k{strength}.csv.gz" for strength in STRENGTH_GRID
        }}
        for seed in EXPECTED_SEEDS:
            ensure_seed_files(root / f"seed_{seed:03d}", expected)
        return {"seed_ids": list(EXPECTED_SEEDS), "strength_grid": list(STRENGTH_GRID)}

    def aggregate() -> dict[str, Any]:
        summary = load_json(aggregate_path(ctx, "strength_fixed"),
                            keys=AGGREGATE_KEYS["strength_fixed"])
        verify_provenance(summary, ctx, config_name="amazon_strength_fixed.yaml",
                          seed_ids=list(EXPECTED_SEEDS),
                          selected_lambda=ctx.selected_lambda)
        ensure(summary["n_seeds"] == 30 and summary["seed_ids"] == list(EXPECTED_SEEDS)
               and summary["strength_grid_k"] == list(STRENGTH_GRID) and
               summary["fixed_item_degree"] == 9 and summary["reuse_r"] == 8,
               "fixed-strength aggregate discrete mismatch")
        ensure_true_flags(summary["invariants"], "fixed-strength aggregate")
        return {"fixed_item_degree": 9, "reuse_r": 8}

    def per_seed() -> dict[str, Any]:
        total_attack_rows = 0
        auc_by_strength: dict[int, list[float]] = defaultdict(list)
        fixed_fields = (
            "donor_rank", "account_id", "position", "source_line",
            "rating_in_clean_world",
        )
        for seed in EXPECTED_SEEDS:
            directory = root / f"seed_{seed:03d}"
            summary = load_json(directory / "summary.json",
                                keys=SEED_KEYS["strength_fixed"])
            ensure_category(summary, f"fixed strength seed {seed}")
            ensure(summary["seed"] == seed and summary["n_items"] == 2000 and
                   summary["n_accounts"] == 2250 and summary["reuse_r"] == 8 and
                   summary["fixed_item_degree"] == 9 and
                   summary["strength_grid_k"] == list(STRENGTH_GRID) and
                   summary["selected_lambda"] == ctx.selected_lambda,
                   f"fixed-strength discrete mismatch for seed {seed}")
            ensure_true_flags(summary["invariants"], f"fixed strength seed {seed}")
            worlds: dict[int, dict[str, dict[str, str]]] = {}
            for strength in STRENGTH_GRID:
                rows = read_csv(directory / f"attack_world_k{strength}.csv.gz",
                                STRENGTH_HEADER)
                ensure(len(rows) == 2000 and len({row["asin"] for row in rows}) == 2000,
                       f"fixed-strength row count mismatch seed {seed}, k={strength}")
                ensure(all(int(row["k"]) == strength and
                           int(row["fixed_item_degree"]) == 9 and
                           int(row["reuse_r"]) == 8 for row in rows),
                       f"fixed-strength row metadata mismatch seed {seed}, k={strength}")
                worlds[strength] = {row["asin"]: row for row in rows}
                total_attack_rows += len(rows)
            accounts = Counter()
            for asin in worlds[9]:
                rows = [worlds[strength][asin] for strength in STRENGTH_GRID]
                ensure(len({row["treatment_block"] for row in rows}) == 1 and
                       len({row["clean_counts"] for row in rows}) == 1,
                       f"fixed-strength block mismatch seed {seed}, item {asin}")
                fixed = [json.loads(row["fixed_assignments"]) for row in rows]
                projected = [
                    [tuple(entry[field] for field in fixed_fields) for entry in group]
                    for group in fixed
                ]
                ensure(projected[0] == projected[1] == projected[2] and
                       len(projected[0]) == 9,
                       f"fixed identity mismatch seed {seed}, item {asin}")
                for strength, group in zip(STRENGTH_GRID, fixed):
                    ensure([entry["is_modified_at_k"] for entry in group] ==
                           [rank < strength for rank in range(9)],
                           f"fixed-strength activation mismatch seed {seed}, item {asin}")
                modified = [json.loads(row["modified_slots"]) for row in rows]
                ensure([len(group) for group in modified] == list(STRENGTH_GRID),
                       f"fixed-strength modified count mismatch seed {seed}, item {asin}")
                ids = [[entry["account_id"] for entry in group] for group in modified]
                ensure(ids[0] == ids[1][:3] and ids[1] == ids[2][:6],
                       f"fixed-strength nesting mismatch seed {seed}, item {asin}")
                for strength, group in zip(STRENGTH_GRID, modified):
                    ensure([entry["donor_rank"] for entry in group] == list(range(strength)),
                           f"fixed-strength donor rank mismatch seed {seed}, item {asin}")
                    ensure(all(entry["original_rating"] != 5 and
                               entry["replacement_rating"] == 5 for entry in group),
                           f"fixed-strength rating mismatch seed {seed}, item {asin}")
                accounts.update(entry["account_id"] for entry in fixed[2])
            ensure(len(accounts) == 2250 and set(accounts.values()) == {8},
                   f"fixed-strength account degree mismatch for seed {seed}")
            for strength in STRENGTH_GRID:
                metrics = summary["metrics_by_k"][str(strength)]
                ensure_close(metrics["mean_paired_score_gap"],
                             8 * metrics["mean_d_cf"],
                             f"fixed-strength reuse law seed {seed}, k={strength}")
                ensure(metrics["fixed_item_degree"] == 9 and metrics["reuse_r"] == 8 and
                       metrics["n_accounts"] == 2250,
                       f"fixed-strength metric metadata seed {seed}, k={strength}")
                auc_by_strength[strength].append(
                    float(metrics["counterfactual_score_auc"])
                )
        aggregate_summary = load_json(aggregate_path(ctx, "strength_fixed"))
        for strength, values in auc_by_strength.items():
            ensure_close(
                aggregate_summary["metrics_by_k"][str(strength)]
                                 ["counterfactual_score_auc"]["mean"],
                fmean(values), f"fixed-strength aggregate AUC k={strength}",
            )
        return {"attack_rows": total_attack_rows, "fixed_accounts_per_seed": 2250}

    audit.capture("seed_sets_and_expected_files", seed_set_and_files)
    audit.capture("aggregate_schema_category_and_provenance", aggregate)
    audit.capture("fixed_identity_nesting_degrees_and_conservation", per_seed)
    return audit


def verify_k6_population_audit(ctx: VerificationContext) -> StageAudit:
    audit = StageAudit("k6_population_audit")
    root = ctx.stage("stage9_k6_population_audit")

    def schema_and_provenance() -> dict[str, Any]:
        expected = {
            "electronics_k6_population_exact_expectations.csv",
            "electronics_k6_population_audit_summary.json",
        }
        ensure({path.name for path in root.iterdir() if path.is_file()} == expected,
               "k6 population-audit file set mismatch")
        summary = load_json(aggregate_path(ctx, "k6_population_audit"),
                            keys=AGGREGATE_KEYS["k6_population_audit"])
        verify_provenance(summary, ctx,
                          config_name="amazon_k6_population_audit.yaml",
                          seed_ids=[], selected_lambda=ctx.selected_lambda)
        ensure(summary["k"] == 6 and summary["selected_lambda"] == ctx.selected_lambda,
               "k6 population-audit metadata mismatch")
        ensure_true_flags(summary["invariants"], "k6 population audit")
        ensure(summary["legacy_empirical_values"]["status"] ==
               "not_used_as_reproduction_targets",
               "legacy empirical values became verification targets")
        return {"k": 6}

    def exact_recomputation() -> dict[str, Any]:
        header = (
            "population", "threshold_nonfive_each_block", "population_size",
            "exact_expected_mean_d_cf",
        )
        rows = read_csv(root / "electronics_k6_population_exact_expectations.csv",
                        header)
        ensure(len(rows) == 2 and {row["population"] for row in rows} == {
            "k6_feasible_population", "k9_feasible_population"
        }, "k6 population-audit row mismatch")
        cfg = yaml.safe_load(
            ctx.config_path("amazon_k6_population_audit.yaml").read_text(encoding="utf-8")
        )
        blocks = load_experimental_blocks(
            ctx.shared_root / cfg["shared_data"]["roles_file"]
        )
        references = load_reference_table(
            ctx.shared_root / cfg["shared_data"]["reference_histograms"]
        )
        by_name = {row["population"]: row for row in rows}
        summary = load_json(aggregate_path(ctx, "k6_population_audit"))
        results = {}
        for name, threshold in cfg["population_rules"].items():
            recomputed = exact_population_expected_mean(
                blocks=blocks,
                refs=references,
                lam=ctx.selected_lambda,
                threshold=int(threshold),
                k=6,
            )
            row = by_name[name]
            reported = summary["population_results"][name]
            ensure(int(row["threshold_nonfive_each_block"]) == int(threshold) ==
                   reported["threshold_nonfive_each_block"],
                   f"k6 population threshold mismatch: {name}")
            ensure(int(row["population_size"]) == recomputed["population_size"] ==
                   reported["population_size"],
                   f"k6 population size mismatch: {name}")
            ensure_close(float(row["exact_expected_mean_d_cf"]),
                         recomputed["exact_expected_mean_d_cf"],
                         f"k6 exact expectation CSV: {name}")
            ensure_close(reported["exact_expected_mean_d_cf"],
                         recomputed["exact_expected_mean_d_cf"],
                         f"k6 exact expectation summary: {name}")
            results[name] = recomputed
        return {"exact_population_results": results}

    audit.capture("files_schema_category_and_provenance", schema_and_provenance)
    audit.capture("exact_population_recomputation_and_tolerances", exact_recomputation)
    return audit


def _nested_keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _nested_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _nested_keys(item)


def verify_full_background(ctx: VerificationContext) -> StageAudit:
    audit = StageAudit("full_background")
    root = ctx.stage("stage10_full_background")
    attack_root = ctx.stage("stage4_attack")
    reuse_root = ctx.stage("stage4_reuse")

    def seed_set_and_files() -> dict[str, Any]:
        ensure_seed_directories(root)
        for seed in EXPECTED_SEEDS:
            ensure_seed_files(root / f"seed_{seed:03d}", {
                "account_scores.csv.gz", "summary.json"
            })
        return {"seed_ids": list(EXPECTED_SEEDS), "files_per_seed": 2}

    def aggregate() -> dict[str, Any]:
        summary = load_json(aggregate_path(ctx, "full_background"),
                            keys=AGGREGATE_KEYS["full_background"])
        verify_provenance(summary, ctx,
                          config_name="amazon_full_background.yaml",
                          seed_ids=list(EXPECTED_SEEDS),
                          selected_lambda=ctx.selected_lambda)
        ensure(summary["n_seeds"] == 30 and summary["seed_ids"] == list(EXPECTED_SEEDS),
               "full-background aggregate seed mismatch")
        ensure(summary["activity_strata"] == ["all", "freq_eq_8", "freq_7_9"],
               "full-background activity strata mismatch")
        forbidden = {"auc", "precision", "recall", "fpr", "false_positive_rate"}
        ensure(not (set(key.lower() for key in _nested_keys(summary["metrics"])) & forbidden),
               "forbidden classification metric in full-background aggregate")
        return {"activity_strata": summary["activity_strata"]}

    def per_seed() -> dict[str, Any]:
        total_rows = 0
        totals = []
        for seed in EXPECTED_SEEDS:
            directory = root / f"seed_{seed:03d}"
            summary = load_json(directory / "summary.json",
                                keys=SEED_KEYS["full_background"])
            ensure_category(summary, f"full background seed {seed}")
            ensure(summary["seed"] == seed, f"full-background seed mismatch: {seed}")
            ensure_true_flags(summary["invariants"], f"full background seed {seed}")
            attack = load_json(attack_root / f"seed_{seed:03d}" / "attack_summary.json")
            source = summary["source_stage4"]
            ensure(source == {
                "attack_world_sha256": attack["attack_world_sha256"],
                "modified_ratings_k": 6,
                "n_attacked_items": 2000,
                "reuse_r": 8,
                "selected_lambda": ctx.selected_lambda,
            }, f"full-background Stage-4 handoff mismatch for seed {seed}")
            ensure(summary["stream"] == {
                "block_roles": ["experimental_A", "experimental_B"],
                "blocks_per_item": 2,
                "n_eligible_items": ctx.n_items,
                "n_scored_blocks": ctx.n_items * 2,
                "n_scored_reviews": ctx.n_items * 60,
            }, f"full-background stream conservation mismatch for seed {seed}")
            population = summary["population"]
            ensure(population["n_planted_accounts"] == 1500 and
                   population["n_total_accounts"] ==
                   population["n_historical_background_accounts"] + 1500,
                   f"full-background population mismatch for seed {seed}")
            with csv_rows(
                reuse_root / f"seed_{seed:03d}" / "identity_assignment_r8.csv.gz",
                IDENTITY_HEADER,
            ) as reader:
                planted_expected = {row["synthetic_account_id"] for row in reader}
            ensure(len(planted_expected) == 1500,
                   f"full-background planted handoff size mismatch for seed {seed}")
            ids = set()
            planted = set()
            historical = 0
            rows = 0
            with csv_rows(directory / "account_scores.csv.gz",
                          FULL_BACKGROUND_HEADER) as reader:
                for row in reader:
                    rows += 1
                    account_id = row["account_id"]
                    ensure(account_id not in ids,
                           f"duplicate full-background account for seed {seed}")
                    ids.add(account_id)
                    ensure(row["coactivity_score"] == row["combined_score"] == "",
                           f"forbidden full-background score for seed {seed}")
                    if row["account_type"] == "planted":
                        planted.add(account_id)
                        ensure(int(row["frequency"]) == 8,
                               f"planted full-background frequency mismatch seed {seed}")
                    else:
                        ensure(row["account_type"] == "historical_background",
                               f"unknown full-background account type seed {seed}")
                        historical += 1
            ensure(rows == population["n_total_accounts"] and
                   planted == planted_expected and historical ==
                   population["n_historical_background_accounts"],
                   f"full-background account row conservation mismatch seed {seed}")
            forbidden = {"auc", "precision", "recall", "fpr", "false_positive_rate"}
            ensure(not (set(key.lower() for key in _nested_keys(summary["metrics"])) & forbidden),
                   f"forbidden classification metric seed {seed}")
            total_rows += rows
            totals.append(float(population["n_total_accounts"]))
        aggregate_summary = load_json(aggregate_path(ctx, "full_background"))
        ensure_close(aggregate_summary["population"]["n_total_accounts"]["mean"],
                     fmean(totals), "full-background aggregate account count")
        return {"unique_account_rows": total_rows, "seed_files_streamed": 30}

    audit.capture("seed_sets_and_expected_files", seed_set_and_files)
    audit.capture("aggregate_schema_category_provenance_and_metric_guard", aggregate)
    audit.capture("rows_handoffs_population_and_unlabeled_background", per_seed)
    return audit


def verify_self_influence(ctx: VerificationContext) -> StageAudit:
    audit = StageAudit("self_influence")
    root = ctx.stage("stage11_self_influence")
    attack_root = ctx.stage("stage4_attack")
    reuse_root = ctx.stage("stage4_reuse")
    twins_root = ctx.stage("stage5_twins")

    def seed_set_and_files() -> dict[str, Any]:
        ensure_seed_directories(root)
        for seed in EXPECTED_SEEDS:
            ensure_seed_files(root / f"seed_{seed:03d}", {
                "account_scores.csv.gz", "exposure_loo.csv.gz", "summary.json"
            })
        return {"seed_ids": list(EXPECTED_SEEDS), "accounts_per_seed": 1500}

    def aggregate() -> dict[str, Any]:
        summary = load_json(aggregate_path(ctx, "self_influence"),
                            keys=AGGREGATE_KEYS["self_influence"])
        verify_provenance(summary, ctx,
                          config_name="amazon_self_influence.yaml",
                          seed_ids=list(EXPECTED_SEEDS),
                          selected_lambda=ctx.selected_lambda)
        ensure(summary["n_seeds"] == 30 and summary["seed_ids"] == list(EXPECTED_SEEDS),
               "self-influence aggregate seed mismatch")
        ensure(summary["population"] == {
            "n_synthetic_accounts_per_seed": 1500,
            "exposures_per_account": 8,
            "n_exposures_per_seed": 12000,
        }, "self-influence aggregate population mismatch")
        return {"population": summary["population"]}

    def per_seed() -> dict[str, Any]:
        total_accounts = 0
        total_exposures = 0
        loo_auc = []
        for seed in EXPECTED_SEEDS:
            directory = root / f"seed_{seed:03d}"
            summary = load_json(directory / "summary.json",
                                keys=SEED_KEYS["self_influence"])
            ensure_category(summary, f"self influence seed {seed}")
            ensure(summary["seed"] == seed and summary["population"] == {
                "exposures_per_account": 8,
                "n_exposures": 12000,
                "n_synthetic_accounts": 1500,
            }, f"self-influence population mismatch for seed {seed}")
            ensure_true_flags(summary["invariants"], f"self influence seed {seed}")
            attack = load_json(attack_root / f"seed_{seed:03d}" / "attack_summary.json")
            ensure(summary["source_stage4"] == {
                "attack_world_sha256": attack["attack_world_sha256"],
                "modified_ratings_k": 6,
                "n_items": 2000,
                "reuse_r": 8,
                "selected_lambda": ctx.selected_lambda,
            }, f"self-influence Stage-4 handoff mismatch for seed {seed}")
            twins = load_json(twins_root / f"seed_{seed:03d}" / "summary.json")
            ensure_close(summary["metrics"]["original_matched_twin_auc"],
                         twins["metrics"]["counterfactual_score_auc"],
                         f"self-influence original AUC seed {seed}")
            exposures = read_csv(directory / "exposure_loo.csv.gz",
                                 SELF_EXPOSURE_HEADER)
            accounts = read_csv(directory / "account_scores.csv.gz",
                                SELF_ACCOUNT_HEADER)
            ensure(len(exposures) == 12000 and len(accounts) == 1500,
                   f"self-influence row count mismatch for seed {seed}")
            degree = Counter(row["synthetic_account_id"] for row in exposures)
            ensure(len(degree) == 1500 and set(degree.values()) == {8} and
                   len({(row["synthetic_account_id"], row["asin"])
                        for row in exposures}) == 12000,
                   f"self-influence degree mismatch for seed {seed}")
            ensure(all(int(row["original_rating"]) != 5 and
                       int(row["replacement_rating"]) == 5 for row in exposures),
                   f"self-influence rating mismatch for seed {seed}")
            original: dict[str, float] = defaultdict(float)
            loo: dict[str, float] = defaultdict(float)
            for row in exposures:
                original[row["synthetic_account_id"]] += float(row["original_d_cf"])
                loo[row["synthetic_account_id"]] += float(row["loo_d_cf"])
            ensure(len({row["synthetic_account_id"] for row in accounts}) == 1500,
                   f"duplicate self-influence account for seed {seed}")
            for row in accounts:
                account_id = row["synthetic_account_id"]
                ensure(int(row["frequency"]) == 8,
                       f"self-influence frequency mismatch for seed {seed}")
                ensure_close(float(row["original_counterfactual_score"]),
                             original[account_id],
                             f"self-influence original sum seed {seed}")
                ensure_close(float(row["loo_counterfactual_score"]), loo[account_id],
                             f"self-influence LOO sum seed {seed}")
                ensure_close(float(row["direct_self_component"]),
                             original[account_id] - loo[account_id],
                             f"self-influence direct component seed {seed}")
            with csv_rows(
                reuse_root / f"seed_{seed:03d}" / "identity_assignment_r8.csv.gz",
                IDENTITY_HEADER,
            ) as reader:
                assigned = {
                    (row["synthetic_account_id"], row["asin"],
                     row["treated_position"], row["treated_source_line"])
                    for row in reader
                }
            observed = {
                (row["synthetic_account_id"], row["asin"],
                 row["treated_position"], row["treated_source_line"])
                for row in exposures
            }
            ensure(observed == assigned,
                   f"self-influence exposure handoff mismatch for seed {seed}")
            loo_auc.append(float(summary["metrics"]["loo_matched_twin_auc"]))
            total_accounts += len(accounts)
            total_exposures += len(exposures)
        aggregate_summary = load_json(aggregate_path(ctx, "self_influence"))
        ensure_close(aggregate_summary["metrics"]["loo_matched_twin_auc"]["mean"],
                     fmean(loo_auc), "self-influence aggregate LOO AUC")
        return {"accounts": total_accounts, "exposures": total_exposures}

    audit.capture("seed_sets_and_expected_files", seed_set_and_files)
    audit.capture("aggregate_schema_category_and_provenance", aggregate)
    audit.capture("rows_handoffs_loo_conservation_and_tolerances", per_seed)
    return audit


STAGE_ORDER = (
    "inputs",
    "reference",
    "primary_attack",
    "primary_reuse",
    "matched_twins",
    "shape",
    "complementarity",
    "reference_history",
    "strength_fixed",
    "k6_population_audit",
    "full_background",
    "self_influence",
)

STAGE_VERIFIERS: dict[str, Callable[[VerificationContext], StageAudit]] = {
    "inputs": verify_inputs,
    "reference": verify_reference,
    "primary_attack": verify_primary_attack,
    "primary_reuse": verify_primary_reuse,
    "matched_twins": verify_matched_twins,
    "shape": verify_shape,
    "complementarity": verify_complementarity,
    "reference_history": verify_reference_history,
    "strength_fixed": verify_strength_fixed,
    "k6_population_audit": verify_k6_population_audit,
    "full_background": verify_full_background,
    "self_influence": verify_self_influence,
}


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_verification(
    ctx: VerificationContext,
    *,
    stages: tuple[str, ...] = STAGE_ORDER,
) -> tuple[int, dict[str, Any] | None]:
    unknown = set(stages) - set(STAGE_VERIFIERS)
    ensure(not unknown, f"unknown Electronics verifier stages: {sorted(unknown)}")
    ctx.report_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, StageAudit] = {}
    report_entries: dict[str, dict[str, Any]] = {}
    for stage in stages:
        result = STAGE_VERIFIERS[stage](ctx)
        results[stage] = result
        report_path = ctx.report_dir / f"{stage}_report.json"
        write_json(report_path, result.to_dict())
        report_entries[stage] = {
            "path": logical_path(
                report_path,
                repo_root=ctx.repo_root,
                shared_root=ctx.shared_root,
            ),
            "sha256": sha256_file(report_path),
            "status": result.status,
        }

    print("=" * 76)
    print("ELECTRONICS INVARIANT-ONLY VERIFICATION")
    print("=" * 76)
    for stage in stages:
        print(f"{stage:28s}: {results[stage].status}")

    consolidated = None
    if stages == STAGE_ORDER:
        status = "PASS" if all(result.status == "PASS" for result in results.values()) else "FAIL"
        consolidated = {
            "schema_version": 1,
            "category": CATEGORY,
            "verification_mode": "invariant_only",
            "gold_comparison_performed": False,
            "independent_gold_reproduction_claimed": False,
            "status": status,
            "meaning": (
                "Internal Electronics execution validity only; this is not an "
                "independent reproduction against frozen gold."
            ),
            "tolerances": {"rtol": DEFAULT_RTOL, "atol": DEFAULT_ATOL},
            "seed_ids": list(EXPECTED_SEEDS),
            "checks": {stage: results[stage].status == "PASS" for stage in STAGE_ORDER},
            "stage_reports": report_entries,
        }
        final_path = ctx.report_dir / "electronics_final_report.json"
        write_json(final_path, consolidated)
        print("=" * 76)
        print(f"ELECTRONICS FINAL STATUS: {status}")
        return (0 if status == "PASS" else 1), consolidated

    status = "PASS" if all(result.status == "PASS" for result in results.values()) else "FAIL"
    print("=" * 76)
    label = ",".join(stages).upper()
    print(f"ELECTRONICS {label} STATUS: {status}")
    return (0 if status == "PASS" else 1), None


def _must_reject(check: Callable[[], Any], label: str) -> None:
    try:
        check()
    except VerificationError:
        return
    raise AssertionError(f"negative control was not rejected: {label}")


def run_selftest() -> int:
    """Exercise structural and numerical negative controls without real data."""

    ensure_category({"category": CATEGORY, "value": 1}, "selftest category")
    ensure_true_flags({"invariant": True}, "selftest invariant")
    ensure_close(0.5 + 5e-13, 0.5, "selftest tolerated drift")
    _must_reject(
        lambda: ensure_category({"category": "home_and_kitchen"}, "perturbed category"),
        "category mismatch",
    )
    _must_reject(
        lambda: ensure_true_flags({"invariant": False}, "perturbed invariant"),
        "false invariant",
    )
    _must_reject(
        lambda: ensure_close(0.5001, 0.5, "perturbed numeric result"),
        "scientific numeric change",
    )

    with TemporaryDirectory(prefix="electronics-verifier-selftest-") as temp:
        stage = Path(temp) / "stage"
        stage.mkdir()
        for seed in EXPECTED_SEEDS:
            (stage / f"seed_{seed:03d}").mkdir()
        ensure_seed_directories(stage)
        (stage / "seed_029").rmdir()
        _must_reject(lambda: ensure_seed_directories(stage), "missing seed")

        artifact = Path(temp) / "artifact.json"
        artifact.write_text('{"category":"electronics"}\n', encoding="utf-8")
        digest = sha256_file(artifact)
        artifact.write_text('{"category":"electronics","changed":true}\n',
                            encoding="utf-8")
        _must_reject(
            lambda: ensure(sha256_file(artifact) == digest, "perturbed hash accepted"),
            "hash mutation",
        )

    print("ELECTRONICS VERIFIER SELFTEST: PASS")
    return 0
