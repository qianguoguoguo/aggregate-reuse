from __future__ import annotations

import runpy
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]


def load_runner_namespace():
    return runpy.run_path(
        str(ROOT/"experiments"/"amazon"/"run_full_background_ranking.py"),
        run_name="not_main",
    )


def test_config_frozen():
    cfg=yaml.safe_load(
        (ROOT/"configs"/"amazon_full_background.yaml").read_text()
    )
    assert cfg["n_attacked_items_per_seed"]==2000
    assert cfg["primary_modified_ratings_k"]==6
    assert cfg["reuse_r"]==8
    assert cfg["activity_strata"]==["all","freq_eq_8","freq_7_9"]
    assert cfg["top_fractions"]==[0.001,0.005,0.01,0.05]
    assert cfg["inspection_burden_targets"]==[0.25,0.50,0.75]
    assert cfg["shared_data"]["attack_dir"]=="stage4_attack"
    assert cfg["shared_data"]["reuse_dir"]=="stage4_reuse"


def test_background_percentile_midrank():
    ns=load_runner_namespace()
    f=ns["background_percentiles"]
    out=f([2.0],[1.0,2.0,3.0])
    assert out.tolist()==[0.5]


def test_expected_top_capture_ties_is_fractional():
    ns=load_runner_namespace()
    f=ns["expected_top_capture"]
    r=f([1,1,1,1],[True,True,False,False],0.5)
    assert r["cutoff_accounts"]==2
    assert r["planted_capture"]==0.5


def test_runner_reuses_attack_and_identity_without_reconstruction():
    text=(
        ROOT/"experiments"/"amazon"/"run_full_background_ranking.py"
    ).read_text()
    assert "build_attack_world" not in text
    assert "regular_incidence" not in text
    assert "stage4_attack" in text
    assert "stage4_reuse" in text


def test_no_fraud_auc_against_historical_background():
    text=(
        ROOT/"experiments"/"amazon"/"run_full_background_ranking.py"
    ).read_text()
    assert "historical_accounts_not_treated_as_verified_negatives" in text
    assert "no_auc_precision_fpr_or_recall_against_historical_background_reported" in text
