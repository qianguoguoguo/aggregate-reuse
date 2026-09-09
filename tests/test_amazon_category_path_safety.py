from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.paths import (
    clean_isolated_category_root,
    isolated_category_root,
)


def test_electronics_root_is_an_immediate_shared_child(tmp_path):
    shared = tmp_path / "amazon_preprocess"
    shared.mkdir()

    assert isolated_category_root(shared, "electronics") == (
        shared / "electronics"
    ).resolve()


def test_clean_electronics_preserves_home_and_kitchen_siblings(tmp_path):
    shared = tmp_path / "amazon_preprocess"
    home_stage = shared / "stage2"
    electronics = shared / "electronics"
    home_stage.mkdir(parents=True)
    electronics.mkdir()
    home_sentinel = home_stage / "home_and_kitchen_stage2_summary.json"
    electronics_sentinel = electronics / "electronics_scan.json"
    home_sentinel.write_text("home", encoding="utf-8")
    electronics_sentinel.write_text("electronics", encoding="utf-8")

    removed = clean_isolated_category_root(
        electronics,
        shared_root=shared,
        category="electronics",
    )

    assert removed is True
    assert not electronics.exists()
    assert home_sentinel.read_text(encoding="utf-8") == "home"


@pytest.mark.parametrize(
    "requested_suffix",
    [Path("."), Path("stage2"), Path("electronics") / "stage1"],
)
def test_clean_rejects_broad_sibling_and_nested_targets(
    tmp_path,
    requested_suffix,
):
    shared = tmp_path / "amazon_preprocess"
    (shared / "stage2").mkdir(parents=True)
    (shared / "electronics" / "stage1").mkdir(parents=True)

    with pytest.raises(ValueError, match="only allowed target"):
        clean_isolated_category_root(
            shared / requested_suffix,
            shared_root=shared,
            category="electronics",
        )


@pytest.mark.parametrize(
    "category",
    ["", "Electronics", "../electronics", "electronics/stage1", "electronics-2"],
)
def test_invalid_category_slugs_are_rejected(tmp_path, category):
    with pytest.raises(ValueError, match="lowercase underscore-delimited slug"):
        isolated_category_root(tmp_path / "amazon_preprocess", category)


def test_clean_missing_electronics_root_is_a_noop(tmp_path):
    shared = tmp_path / "amazon_preprocess"
    shared.mkdir()

    assert clean_isolated_category_root(
        shared / "electronics",
        shared_root=shared,
        category="electronics",
    ) is False


def test_clean_rejects_non_directory_target(tmp_path):
    shared = tmp_path / "amazon_preprocess"
    shared.mkdir()
    electronics = shared / "electronics"
    electronics.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="non-directory"):
        clean_isolated_category_root(
            electronics,
            shared_root=shared,
            category="electronics",
        )


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


def test_category_root_rejects_symlink_escape(tmp_path):
    shared = tmp_path / "amazon_preprocess"
    outside = tmp_path / "outside"
    shared.mkdir()
    outside.mkdir()
    _symlink_or_skip(shared / "electronics", outside)

    with pytest.raises(ValueError, match="immediate child"):
        isolated_category_root(shared, "electronics")
