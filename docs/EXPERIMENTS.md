# Experiments

Canonical controlled experiments:
- `experiments/controlled/run_null_calibration.py`
- `experiments/controlled/run_matched_exposure.py`
- `experiments/controlled/run_regime_sweep.py`

Canonical Amazon experiments:
- `experiments/amazon/preprocess.py`
- `experiments/amazon/calibrate_reference.py`
- `experiments/amazon/run_primary_attack.py`
- `experiments/amazon/run_primary_reuse.py`
- `experiments/amazon/run_matched_twins.py`
- `experiments/amazon/run_shape.py`
- `experiments/amazon/run_complementarity.py`
- `experiments/amazon/run_reference_history.py`
- `experiments/amazon/run_strength_fixed.py`
- `experiments/amazon/run_k6_population_audit.py`
- `experiments/amazon/run_full_background_ranking.py`
- `experiments/amazon/run_self_influence.py`

Reporting:
- `experiments/reporting/generate_supplement_tables.py`
- `experiments/reporting/generate_static_tables.py`

Temporary migration diagnostics and obsolete placeholder configs are not part
of the canonical pipeline.
