# Reproducibility contract

A successful reproduction requires: (1) all scientific outputs to match frozen gold under their declared verifier tolerances; (2) reporting artifacts to be derived from canonical outputs/configs rather than hand-transcribed; and (3) `paper_results/expected/` to remain immutable.

Exact controlled-result reproduction requires the environment checked by `tools/check_environment.py`. Large Amazon intermediates are kept outside the repository in sibling `../amazon_preprocess/`. After the complete pipeline has been generated, run `python tools/final_release_check.py`.
