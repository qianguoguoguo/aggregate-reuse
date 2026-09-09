from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def test_release_requirements_include_frozen_sklearn():
 assert 'scikit-learn==1.9.0' in (ROOT/'requirements-paper.txt').read_text(encoding='utf-8')
def test_environment_config_and_checker_include_sklearn():
 cfg=yaml.safe_load((ROOT/'configs/controlled.yaml').read_text(encoding='utf-8'))
 assert cfg['required_environment']['sklearn']=='1.9.0'
 assert '"sklearn": "1.9.0"' in (ROOT/'tools/check_environment.py').read_text(encoding='utf-8')
def test_integrated_supplement_verifier_allows_static_fragments():
 t=(ROOT/'tools/verify_supplement_tables.py').read_text(encoding='utf-8')
 assert 'EXPECTED_TEX.issubset(tex_files)' in t
 assert 'tex_files == EXPECTED_TEX' not in t
def test_no_migration_phase_targets_in_makefile():
 t=(ROOT/'Makefile').read_text(encoding='utf-8').lower()
 for x in [f'phase2{c}' for c in 'abcdefghijklmnopqrstuv']:
  assert x not in t
def test_final_release_checker_exists():
 assert (ROOT/'tools/final_release_check.py').exists()
