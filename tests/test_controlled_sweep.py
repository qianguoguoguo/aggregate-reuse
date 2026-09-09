from __future__ import annotations
import sys
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/"src"
if str(SRC) not in sys.path:sys.path.insert(0,str(SRC))
from aggregate_reuse.controlled.campaign import requested_k_on
from aggregate_reuse.controlled.regime_sweep import _balanced_scores_and_frequency
from aggregate_reuse.seeding import derive_stage_seed

def test_regime_grid_is_frozen():
    c=yaml.safe_load((ROOT/"configs"/"controlled.yaml").read_text())["regime_sweep"]
    assert c["p_on_grid"]==[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    assert c["exposure_ratio_grid"]==[0.25,0.5,0.75,1.0,1.25,1.5,2.0]

def test_all_grid_points_feasible():
    c=yaml.safe_load((ROOT/"configs"/"controlled.yaml").read_text())
    for R in c["regime_sweep"]["exposure_ratio_grid"]:
        for p in c["regime_sweep"]["p_on_grid"]:
            k=requested_k_on(R_exp=R,p_normal=0.04,N_coalition=2000,p_on=p)
            assert 1<=k<=2000

def test_balanced_score_allocator():
    I=np.array([1,0,1,1,0,1],dtype=np.int8);d=np.arange(1.,7.)
    s,f=_balanced_scores_and_frequency(I,2,8,d)
    assert int(f.sum())==int(I.sum())*2
    assert f.max()-f.min()<=1
    assert np.isclose(s.sum(),(d[I==1]*2).sum())

def test_stage_seed():
    assert derive_stage_seed(27001,"controlled_exposure_intermittency_regime")==662276841
