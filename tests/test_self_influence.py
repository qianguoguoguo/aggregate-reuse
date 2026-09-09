from __future__ import annotations

import runpy
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]


def ns():
    return runpy.run_path(
        str(ROOT/"experiments"/"amazon"/"run_self_influence.py"),
        run_name="not_main",
    )


def test_config_frozen():
    cfg=yaml.safe_load(
        (ROOT/"configs"/"amazon_self_influence.yaml").read_text()
    )
    assert cfg["primary_modified_ratings_k"]==6
    assert cfg["reuse_r"]==8
    assert cfg["shared_data"]["attack_dir"]=="stage4_attack"
    assert cfg["shared_data"]["reuse_dir"]=="stage4_reuse"
    assert cfg["shared_data"]["output_subdir"]=="stage11_self_influence"


def test_loo_histograms_are_29():
    f=ns()["loo_dcf"]
    clean=np.array([2,4,6,8,10])
    attack=np.array([1,4,6,8,11])
    q=np.ones(5)/5
    d,cw,aw=f(clean,attack,1,5,q)
    assert np.isfinite(d)
    assert np.isfinite(cw)
    assert np.isfinite(aw)


def test_runner_reuses_frozen_attack_and_assignment():
    text=(ROOT/"experiments"/"amazon"/"run_self_influence.py").read_text()
    assert "build_attack_world" not in text
    assert "regular_incidence" not in text
    assert "stage4_attack" in text
    assert "stage4_reuse" in text


def test_interpretation_guard_is_offline():
    text=(ROOT/"experiments"/"amazon"/"run_self_influence.py").read_text()
    assert "offline diagnostic, not the deployable online score" in text
