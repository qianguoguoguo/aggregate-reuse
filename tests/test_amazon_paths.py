from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.paths import (
    AmazonPathLayout,
    STAGE_SUBDIRECTORIES,
    amazon_paths_from_config,
    repository_root,
)


ELECTRONICS_CONFIGS = [
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
]

AGGREGATE_ARTIFACTS = [
    "preprocess_audit",
    "stage3_summary",
    "stage4_attack_summary",
    "stage4_reuse_summary",
    "stage5_twins_summary",
    "stage6_shape_summary",
    "stage7_complementarity_summary",
    "stage8_reference_history_summary",
    "stage9_strength_summary",
    "stage9_k6_population_summary",
    "stage10_full_background_summary",
    "stage11_self_influence_summary",
]


def load_config(name: str, *, electronics: bool = False) -> dict:
    directory = ROOT / "configs"
    if electronics:
        directory /= "electronics"
    return yaml.safe_load((directory / name).read_text(encoding="utf-8"))


def test_repository_root_resolves_from_installed_module_location():
    assert repository_root() == ROOT


def test_electronics_preprocess_paths_match_phase3_examples():
    cfg = load_config("amazon_preprocess.yaml", electronics=True)
    paths = amazon_paths_from_config(cfg, repo_root=ROOT)
    shared = (ROOT.parent / "amazon_preprocess" / "electronics").resolve()

    assert paths.repository_root == ROOT
    assert paths.category == "electronics"
    assert paths.category_prefix == "electronics"
    assert paths.raw_dataset == (ROOT.parent / "Electronics.jsonl.gz").resolve()
    assert paths.shared_category_root == shared
    assert paths.artifact("scan_item_counts") == (
        shared / "scan" / "electronics_item_counts.csv"
    )
    assert paths.artifact("stage1_corpus") == (
        shared / "stage1" / "electronics_first300.items.jsonl.gz"
    )
    assert paths.artifact("stage2_roles") == (
        shared / "stage2" / "electronics_roles.items.jsonl.gz"
    )
    assert paths.artifact("stage2_reference_histograms") == (
        shared / "stage2" / "electronics_reference_histograms.csv"
    )
    assert paths.artifact("stage3_reference_freeze") == (
        shared / "stage3" / "electronics_reference_freeze.json"
    )


@pytest.mark.parametrize("config_name", ELECTRONICS_CONFIGS)
def test_every_electronics_config_resolves_to_the_isolated_root(config_name):
    cfg = load_config(config_name, electronics=True)
    paths = AmazonPathLayout.from_config(cfg, repo_root=ROOT)
    expected = (ROOT.parent / "amazon_preprocess" / "electronics").resolve()

    assert paths.category == "electronics"
    assert paths.shared_category_root == expected
    for stage in STAGE_SUBDIRECTORIES:
        stage_dir = paths.stage_dir(stage)
        assert stage_dir.parent == expected or expected in stage_dir.parents


def test_no_electronics_stage_resolves_to_a_home_stage_directory():
    home = AmazonPathLayout.from_config(
        load_config("amazon_primary.yaml"),
        repo_root=ROOT,
    )
    electronics = AmazonPathLayout.from_config(
        load_config("amazon_primary.yaml", electronics=True),
        repo_root=ROOT,
    )
    home_stage_directories = {
        home.stage_dir(stage) for stage in STAGE_SUBDIRECTORIES
    }

    for stage in STAGE_SUBDIRECTORIES:
        electronics_stage = electronics.stage_dir(stage)
        assert electronics_stage not in home_stage_directories
        assert not any(
            home_stage in electronics_stage.parents
            for home_stage in home_stage_directories
        )


def test_electronics_stage_level_aggregates_receive_category_prefixes():
    paths = AmazonPathLayout.from_config(
        load_config("amazon_primary.yaml", electronics=True),
        repo_root=ROOT,
    )
    shared = paths.shared_category_root

    assert paths.artifact("preprocess_audit") == (
        shared / "electronics_amazon_preprocess_audit.json"
    )
    assert paths.artifact("stage4_attack_summary") == (
        shared
        / "stage4_attack"
        / "electronics_primary_attack_summary.json"
    )
    assert paths.artifact("stage8_reference_history_seed_metrics") == (
        shared
        / "stage8_reference_history"
        / "electronics_reference_history_seed_metrics.csv"
    )
    assert paths.artifact("stage9_k6_population_expectations") == (
        shared
        / "stage9_k6_population_audit"
        / "electronics_k6_population_exact_expectations.csv"
    )
    assert all(
        paths.artifact_name(artifact).startswith("electronics_")
        for artifact in AGGREGATE_ARTIFACTS
    )


def test_per_seed_files_remain_generic_inside_electronics_namespace():
    paths = AmazonPathLayout.from_config(
        load_config("amazon_primary.yaml", electronics=True),
        repo_root=ROOT,
    )
    shared = paths.shared_category_root

    assert paths.per_seed_file(
        "stage4_attack", 0, "attack_world.csv.gz"
    ) == shared / "stage4_attack" / "seed_000" / "attack_world.csv.gz"
    assert paths.per_seed_file(
        "stage4_reuse", 29, "reuse_summary.json"
    ) == shared / "stage4_reuse" / "seed_029" / "reuse_summary.json"


def test_home_and_electronics_seed_directories_do_not_collide():
    home = AmazonPathLayout.from_config(
        load_config("amazon_primary.yaml"),
        repo_root=ROOT,
    )
    electronics = AmazonPathLayout.from_config(
        load_config("amazon_primary.yaml", electronics=True),
        repo_root=ROOT,
    )

    for seed in [0, 29]:
        home_dir = home.seed_dir("stage4_attack", seed)
        electronics_dir = electronics.seed_dir("stage4_attack", seed)
        assert home_dir != electronics_dir
        assert home.stage_dir("stage4_attack") not in electronics_dir.parents
        assert electronics.shared_category_root not in home_dir.parents


def test_home_downstream_paths_and_legacy_aggregate_names_do_not_change():
    paths = AmazonPathLayout.from_config(
        load_config("amazon_primary.yaml"),
        repo_root=ROOT,
    )
    shared = (ROOT.parent / "amazon_preprocess").resolve()

    assert paths.shared_category_root == shared
    assert paths.artifact("stage2_roles") == (
        shared / "stage2" / "home_and_kitchen_roles.items.jsonl.gz"
    )
    assert paths.artifact("stage4_attack_summary") == (
        shared / "stage4_attack" / "primary_attack_summary.json"
    )
    assert paths.artifact("stage4_attack_seed_metrics") == (
        shared / "stage4_attack" / "primary_attack_seed_metrics.csv"
    )
    assert paths.artifact("stage4_reuse_summary") == (
        shared
        / "stage4_reuse"
        / "home_and_kitchen_stage4_summary.json"
    )
    assert paths.artifact("stage8_reference_history_calibration") == (
        shared
        / "stage8_reference_history"
        / "reference_history_calibration.csv"
    )


def test_home_preprocess_keeps_repository_local_staging_behavior():
    paths = AmazonPathLayout.from_config(
        load_config("amazon_preprocess.yaml"),
        repo_root=ROOT,
    )
    local = (ROOT / "artifacts" / "amazon_preprocess").resolve()

    assert paths.raw_dataset == (
        ROOT.parent / "Home_and_Kitchen.jsonl.gz"
    ).resolve()
    assert paths.shared_category_root == local
    assert paths.artifact("scan_summary") == (
        local / "scan" / "home_and_kitchen_scan.json"
    )
    assert paths.artifact("preprocess_audit") == (
        local / "amazon_preprocess_audit.json"
    )


@pytest.mark.parametrize(
    "relative_path",
    ["../stage2", "stage2/../../stage3", "/tmp/outside.csv"],
)
def test_shared_paths_cannot_escape_category_root(relative_path):
    paths = AmazonPathLayout.from_config(
        load_config("amazon_primary.yaml", electronics=True),
        repo_root=ROOT,
    )
    with pytest.raises(ValueError, match="relative|escapes category root"):
        paths.resolve_shared_path(relative_path)


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        # Creating directory symlinks on Windows may require Developer Mode
        # or elevated privileges.  That is an environment limitation, not a
        # failure of the path-safety logic being tested.
        if sys.platform == "win32" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink creation requires Developer Mode or elevation")
        raise


def test_shared_path_rejects_symlink_escape(tmp_path):
    shared = tmp_path / "amazon_preprocess" / "electronics"
    outside = tmp_path / "outside"
    shared.mkdir(parents=True)
    outside.mkdir()
    _symlink_or_skip(shared / "stage2", outside)
    paths = AmazonPathLayout(
        repository_root=tmp_path,
        category="electronics",
        shared_category_root=shared,
    )

    with pytest.raises(ValueError, match="escapes category root"):
        paths.resolve_shared_path("stage2/electronics_roles.items.jsonl.gz")


@pytest.mark.parametrize("seed", [-1, True, 1.5, "0"])
def test_seed_directory_rejects_invalid_seed(seed):
    paths = AmazonPathLayout.from_config(
        load_config("amazon_primary.yaml", electronics=True),
        repo_root=ROOT,
    )
    with pytest.raises(ValueError, match="non-negative integer"):
        paths.seed_dir("stage4_attack", seed)


def test_unknown_stage_and_artifact_are_rejected():
    paths = AmazonPathLayout.from_config(
        load_config("amazon_primary.yaml", electronics=True),
        repo_root=ROOT,
    )
    with pytest.raises(KeyError, match="Unknown Amazon stage"):
        paths.stage_dir("stage12")
    with pytest.raises(KeyError, match="Unknown Amazon artifact"):
        paths.artifact("stage12_summary")


@pytest.mark.parametrize("filename", ["../Electronics.jsonl.gz", "/raw.jsonl.gz"])
def test_raw_dataset_filename_must_be_one_leaf(filename):
    cfg = {
        "category": "electronics",
        "dataset": {
            "filename": filename,
            "location": "parent_of_repository",
        },
        "shared_data": {"root": "../amazon_preprocess/electronics"},
    }
    with pytest.raises(ValueError, match="one filename"):
        AmazonPathLayout.from_config(cfg, repo_root=ROOT)


def test_unknown_dataset_location_is_rejected():
    cfg = {
        "category": "electronics",
        "dataset": {
            "filename": "Electronics.jsonl.gz",
            "location": "download_directory",
        },
        "shared_data": {"root": "../amazon_preprocess/electronics"},
    }
    with pytest.raises(ValueError, match="parent_of_repository"):
        AmazonPathLayout.from_config(cfg, repo_root=ROOT)
