.PHONY: check-env verify-gold verifier-selftest controlled controlled-null controlled-matched controlled-sweep
.PHONY: amazon-preprocess publish-home-preprocess amazon-reference amazon-primary-attack amazon-primary-reuse amazon-matched-twins amazon-shape amazon-complementarity amazon-reference-history amazon-strength-fixed amazon-k6-population-audit amazon-full-background amazon-self-influence amazon-downstream
.PHONY: numerical-tables static-tables cross-category-table tables home-amazon-figure-data verify-figure-data figures electronics-figures all-figures tests final-check
.PHONY: verify-raw-amazon-inputs clean-caches reproduce-amazon-both-from-raw reporting-from-current-results reproduce-final-from-raw

check-env:
	python tools/check_environment.py

verify-gold:
	python tools/verify_results.py integrity --expected-dir paper_results/expected --report artifacts/verification/gold_integrity.json

verifier-selftest:
	python tools/verify_results.py selftest --expected-dir paper_results/expected

controlled-null: check-env
	python experiments/controlled/run_null_calibration.py --clean
	python tools/verify_controlled_null.py

controlled-matched: check-env
	python experiments/controlled/run_matched_exposure.py --clean
	python tools/verify_controlled_matched.py

controlled-sweep: check-env
	python experiments/controlled/run_regime_sweep.py --clean
	python tools/verify_controlled_sweep.py

controlled: controlled-null controlled-matched controlled-sweep

verify-raw-amazon-inputs:
	python tools/verify_raw_amazon_inputs.py

amazon-preprocess:
	python experiments/amazon/preprocess.py --clean
	python tools/verify_amazon_preprocess.py
	@echo "Home preprocessing verified in artifacts/amazon_preprocess. Run make publish-home-preprocess before Home downstream targets."

publish-home-preprocess:
	python tools/publish_home_preprocess.py --clean

amazon-reference:
	python experiments/amazon/calibrate_reference.py --overwrite
	python tools/verify_amazon_reference.py

amazon-primary-attack:
	python experiments/amazon/run_primary_attack.py --all-seeds --overwrite
	python tools/verify_primary_attack.py

amazon-primary-reuse:
	python experiments/amazon/run_primary_reuse.py --all-seeds --overwrite
	python tools/verify_primary_reuse.py

amazon-matched-twins:
	python experiments/amazon/run_matched_twins.py --all-seeds --overwrite
	python tools/verify_matched_twins.py

amazon-shape:
	python experiments/amazon/run_shape.py --all-seeds --overwrite
	python tools/verify_amazon_shape.py

amazon-complementarity:
	python experiments/amazon/run_complementarity.py --all-seeds --overwrite
	python tools/verify_complementarity.py

amazon-reference-history:
	python experiments/amazon/run_reference_history.py
	python tools/verify_reference_history.py

amazon-strength-fixed:
	python experiments/amazon/run_strength_fixed.py --all-seeds --overwrite
	python tools/verify_strength_fixed.py

amazon-k6-population-audit:
	python experiments/amazon/run_k6_population_audit.py
	python tools/verify_k6_population_audit.py

amazon-full-background:
	python experiments/amazon/run_full_background_ranking.py --all-seeds --overwrite
	python tools/verify_full_background_ranking.py

amazon-self-influence:
	python experiments/amazon/run_self_influence.py --all-seeds --overwrite
	python tools/verify_self_influence.py

amazon-downstream: amazon-reference amazon-primary-attack amazon-primary-reuse amazon-matched-twins amazon-shape amazon-complementarity amazon-reference-history amazon-strength-fixed amazon-k6-population-audit amazon-full-background amazon-self-influence

numerical-tables:
	python experiments/reporting/generate_supplement_tables.py
	python tools/verify_supplement_tables.py

static-tables:
	python experiments/reporting/generate_static_tables.py
	python tools/verify_reporting_spec.py

cross-category-table:
	python experiments/reporting/generate_cross_category_table.py
	python tools/verify_cross_category_table.py

tables: numerical-tables static-tables cross-category-table

home-amazon-figure-data:
	python experiments/reporting/generate_amazon_figure_data.py --config configs/amazon_preprocess.yaml --shared-root ../amazon_preprocess --out-dir artifacts/figure_data --report artifacts/verification/home_amazon_figure_data_source_report.json

verify-figure-data:
	python tools/verify_figure_data.py

figures: verify-figure-data
	python figures/make_fig1_controlled.py
	python figures/make_fig2_amazon_main.py
	python figures/make_fig3_amazon_robustness.py

tests:
	python -m pytest -q

clean-caches:
	python tools/clean_release_caches.py

final-check:
	python tools/final_release_check.py

# BEGIN PHASE1 CATEGORY-AWARE ELECTRONICS
# -----------------------------------------------------------------------------
# Category-aware Electronics pipeline. These targets use the same scientific
# runners as Home-and-Kitchen and change only the frozen category configs.
# -----------------------------------------------------------------------------
.PHONY: electronics-preprocess electronics-reference electronics-primary-attack electronics-primary-reuse
.PHONY: electronics-matched-twins electronics-shape electronics-complementarity electronics-reference-history
.PHONY: electronics-strength-fixed electronics-k6-population-audit electronics-full-background electronics-self-influence
.PHONY: electronics-downstream electronics-verifier-selftest electronics-verify electronics-figure-data
.PHONY: electronics-verify-figure-data electronics-reporting
ELECTRONICS_CONFIG_DIR := configs/electronics
ELECTRONICS_SHARED_ROOT := ../amazon_preprocess/electronics

# The raw Electronics.jsonl.gz file is expected one directory above the repo,
# as declared by configs/electronics/amazon_preprocess.yaml.
electronics-preprocess:
	python experiments/amazon/preprocess.py --config $(ELECTRONICS_CONFIG_DIR)/amazon_preprocess.yaml --clean

electronics-reference:
	python experiments/amazon/calibrate_reference.py --config $(ELECTRONICS_CONFIG_DIR)/amazon_reference.yaml --overwrite

electronics-primary-attack:
	python experiments/amazon/run_primary_attack.py --config $(ELECTRONICS_CONFIG_DIR)/amazon_primary.yaml --all-seeds --overwrite

electronics-primary-reuse:
	python experiments/amazon/run_primary_reuse.py --config $(ELECTRONICS_CONFIG_DIR)/amazon_primary.yaml --all-seeds --overwrite

electronics-matched-twins:
	python experiments/amazon/run_matched_twins.py --config $(ELECTRONICS_CONFIG_DIR)/amazon_twins.yaml --all-seeds --overwrite

electronics-shape:
	python experiments/amazon/run_shape.py --config $(ELECTRONICS_CONFIG_DIR)/amazon_shape.yaml --all-seeds --overwrite

electronics-complementarity:
	python experiments/amazon/run_complementarity.py --config $(ELECTRONICS_CONFIG_DIR)/amazon_complementarity.yaml --all-seeds --overwrite

electronics-reference-history:
	python experiments/amazon/run_reference_history.py --config $(ELECTRONICS_CONFIG_DIR)/amazon_reference_history.yaml

electronics-strength-fixed:
	python experiments/amazon/run_strength_fixed.py --config $(ELECTRONICS_CONFIG_DIR)/amazon_strength_fixed.yaml --all-seeds --overwrite

electronics-k6-population-audit:
	python experiments/amazon/run_k6_population_audit.py --config $(ELECTRONICS_CONFIG_DIR)/amazon_k6_population_audit.yaml

electronics-full-background:
	python experiments/amazon/run_full_background_ranking.py --config $(ELECTRONICS_CONFIG_DIR)/amazon_full_background.yaml --all-seeds --overwrite

electronics-self-influence:
	python experiments/amazon/run_self_influence.py --config $(ELECTRONICS_CONFIG_DIR)/amazon_self_influence.yaml --all-seeds --overwrite

electronics-downstream: electronics-reference electronics-primary-attack electronics-primary-reuse electronics-matched-twins electronics-shape electronics-complementarity electronics-reference-history electronics-strength-fixed electronics-k6-population-audit electronics-full-background electronics-self-influence

electronics-verifier-selftest:
	python tools/verify_electronics.py --selftest

electronics-verify:
	python tools/verify_electronics.py

electronics-figure-data:
	python experiments/reporting/generate_amazon_figure_data.py --config $(ELECTRONICS_CONFIG_DIR)/amazon_preprocess.yaml

electronics-verify-figure-data:
	python tools/verify_electronics_figure_data.py --config $(ELECTRONICS_CONFIG_DIR)/amazon_preprocess.yaml

electronics-figures: electronics-figure-data electronics-verify-figure-data
	python figures/make_fig2_amazon_main.py --data artifacts/electronics/figure_data/amazon_main_figure_data.csv --out-dir artifacts/electronics/figures --stem-prefix electronics
	python figures/make_fig3_amazon_robustness.py --data artifacts/electronics/figure_data/amazon_robustness_figure_data.csv --out-dir artifacts/electronics/figures --stem-prefix electronics

electronics-reporting: electronics-figures
	python experiments/reporting/generate_supplement_tables.py --category electronics --shared-root $(ELECTRONICS_SHARED_ROOT) --amazon-only --out-dir artifacts/electronics/tables --data-dir artifacts/electronics/table_data
	python experiments/reporting/generate_static_tables.py --config-dir $(ELECTRONICS_CONFIG_DIR) --amazon-only --out-dir artifacts/electronics/tables --data-dir artifacts/electronics/table_data
	python tools/verify_electronics_reporting.py
# END PHASE1 CATEGORY-AWARE ELECTRONICS

all-figures: figures electronics-figures

# Fresh dual-category Amazon rebuild from the two raw category files.  This is
# intentionally sequential: Home is staged inside the repo, published to the
# shared root, then Electronics is rebuilt in its isolated subdirectory.
reproduce-amazon-both-from-raw:
	$(MAKE) verify-raw-amazon-inputs
	$(MAKE) amazon-preprocess
	$(MAKE) publish-home-preprocess
	$(MAKE) amazon-downstream
	$(MAKE) electronics-preprocess
	$(MAKE) electronics-downstream
	$(MAKE) electronics-verifier-selftest
	$(MAKE) electronics-verify

# Regenerate all reporting artifacts that depend on the current Home and
# Electronics experiment outputs.  The six Electronics panels remain six
# separate PDF/PNG files; no composite plot is introduced.
reporting-from-current-results:
	$(MAKE) home-amazon-figure-data
	$(MAKE) electronics-reporting
	$(MAKE) numerical-tables
	$(MAKE) static-tables
	$(MAKE) cross-category-table
	$(MAKE) figures

# Strongest clean-room reproduction target: controlled experiments + both raw
# Amazon categories + all reported tables/figures + consolidated verification.
# The final cleanup/scan is repeated after pytest because pytest recreates
# bytecode caches that are intentionally excluded from the anonymous release.
reproduce-final-from-raw:
	$(MAKE) controlled
	$(MAKE) reproduce-amazon-both-from-raw
	$(MAKE) reporting-from-current-results
	$(MAKE) clean-caches
	$(MAKE) final-check
	$(MAKE) clean-caches
	python tools/verify_results.py integrity --expected-dir paper_results/expected --report artifacts/verification/gold_integrity_final.json
	$(MAKE) clean-caches
	python tools/anonymity_scan.py --root . --report artifacts/verification/final_anonymity_after_reproduction.json
