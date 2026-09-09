# Aggregate Reuse Reproducibility Artifact

Public reproducibility artifact for:

**Principled Detection of Coordinated Manipulation from Aggregate Distortion and Account Reuse**

Authors: Qian Guo, Yidan Hu, and Rui Zhang

Repository: [https://github.com/qianguoguoguo/aggregate-reuse](https://github.com/qianguoguoguo/aggregate-reuse)

This artifact reproduces the controlled experiments, the primary
`Home_and_Kitchen` Amazon evaluation, the `Electronics` cross-category
replication, Figures 1–3, the supporting figure panels, and all 17 appendix
tables.

The standalone supplement prepared for conference review is not distributed
in this public repository. Its content is incorporated as Appendices A–I of
the arXiv paper. The scientific code, frozen configurations, expected
numerical results, and verification logic are unchanged from the frozen
conference-review artifact.


## 1. Full clean-room reproduction

The same Python orchestration backend is used on Windows and Linux. Windows
uses `RUN_PHASE4_WINDOWS.cmd`; Linux uses `RUN_PHASE4_LINUX.sh`.
**Windows reproduction does not require `make`.**

### Windows

Install **Python 3.12.10**, then from the repository root:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python tools\check_environment.py
```

Require:

```text
STATUS: PASS
```

Place the two Amazon raw review files one directory above the repository as
shown in Section 3. Then run:

```bat
RUN_PHASE4_WINDOWS.cmd verify-raw
RUN_PHASE4_WINDOWS.cmd reproduce-final-from-raw
```

The complete run must end with:

```text
FINAL STATUS: PASS
```

### Linux

Verify that the interpreter is exactly Python 3.12.10:

```bash
python3.12 --version
```

Create a clean environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/check_environment.py
```

Require:

```text
STATUS: PASS
```

Place the same two raw review files one directory above the repository, then:

```bash
bash RUN_PHASE4_LINUX.sh verify-raw
bash RUN_PHASE4_LINUX.sh reproduce-final-from-raw
```

The complete run must end with:

```text
FINAL STATUS: PASS
```

The Linux wrapper calls the same Python orchestration backend as Windows. The
historical backend filename `tools/run_phase4_windows.py` is retained for
provenance; its orchestration logic is platform-independent.

## 2. Frozen scientific environment

```text
Python        3.12.10
NumPy         2.5.1
SciPy         1.18.0
scikit-learn  1.9.0
PyYAML        6.0.3
```

Before running experiments, verify the immutable expected-result package:

```bash
python tools/verify_results.py integrity \
  --expected-dir paper_results/expected \
  --report artifacts/verification/gold_integrity.json

python tools/verify_results.py selftest \
  --expected-dir paper_results/expected
```

On Windows the same commands may be entered on one line with backslashes
replaced by Windows command continuation as desired.

The self-test must end with:

```text
VERIFIER SELFTEST: PASS
```

## 3. Amazon Reviews 2023 data

The Amazon experiments use the public **Amazon Reviews 2023** release from the
McAuley Lab at UC San Diego.

Primary category:

```text
Home_and_Kitchen
```

Cross-category replication:

```text
Electronics
```

Official dataset page:

<https://amazon-reviews-2023.github.io/>

Download the **review** files, not the item-metadata files, and keep the exact
filenames:

```text
Home_and_Kitchen.jsonl.gz
Electronics.jsonl.gz
```

Required layout:

```text
parent/
├── Home_and_Kitchen.jsonl.gz
├── Electronics.jsonl.gz
└── aggregate-reuse/
```

The raw-input verifier checks these frozen SHA-256 identities:

```text
Home_and_Kitchen
49fe8d03284ce71c1a8da720b15290d288e3694f335bcac4a66e2c8aa4ad5965

Electronics
17b2c5f3736d4c0cb874859076436ebd4513f4c2396c528440a63204084e6a28
```

Generated Amazon preprocessing is kept outside the repository in the sibling
`amazon_preprocess/` tree. It is not part of the GitHub artifact.

## 4. One-command workflow

Use:

| Operating system | Command |
|---|---|
| Windows | `RUN_PHASE4_WINDOWS.cmd reproduce-final-from-raw` |
| Linux | `bash RUN_PHASE4_LINUX.sh reproduce-final-from-raw` |

The workflow performs:

```text
environment verification
→ controlled experiments
→ raw Amazon identity verification
→ Home_and_Kitchen preprocessing
→ Home_and_Kitchen downstream experiments and verification
→ Electronics preprocessing
→ Electronics downstream experiments and verification
→ Home + Electronics reporting
→ Figures 1--3
→ six separate Electronics figure files used in Appendix H
→ all 17 appendix tables
→ consolidated release verification
→ final frozen-gold integrity
→ metadata-hygiene scan
```

All randomized Amazon experiments use the 30 primary seeds
`0,1,...,29`.

## 5. Staged commands

Use the same target with the platform-specific front-end:

| Stage | Windows | Linux |
|---|---|---|
| Verify raw inputs | `RUN_PHASE4_WINDOWS.cmd verify-raw` | `bash RUN_PHASE4_LINUX.sh verify-raw` |
| Controlled experiments | `RUN_PHASE4_WINDOWS.cmd controlled` | `bash RUN_PHASE4_LINUX.sh controlled` |
| Home preprocessing | `RUN_PHASE4_WINDOWS.cmd home-preprocess` | `bash RUN_PHASE4_LINUX.sh home-preprocess` |
| Home downstream | `RUN_PHASE4_WINDOWS.cmd home-downstream` | `bash RUN_PHASE4_LINUX.sh home-downstream` |
| Electronics preprocessing | `RUN_PHASE4_WINDOWS.cmd electronics-preprocess` | `bash RUN_PHASE4_LINUX.sh electronics-preprocess` |
| Electronics downstream | `RUN_PHASE4_WINDOWS.cmd electronics-downstream` | `bash RUN_PHASE4_LINUX.sh electronics-downstream` |
| Both Amazon categories | `RUN_PHASE4_WINDOWS.cmd amazon-both-from-raw` | `bash RUN_PHASE4_LINUX.sh amazon-both-from-raw` |
| Reporting | `RUN_PHASE4_WINDOWS.cmd reporting` | `bash RUN_PHASE4_LINUX.sh reporting` |
| Final checker | `RUN_PHASE4_WINDOWS.cmd final-check` | `bash RUN_PHASE4_LINUX.sh final-check` |

The `Makefile` is retained as an additional Unix convenience, but the Linux
wrapper above is preferred because it follows the exact same orchestration
backend as Windows.

## 6. Electronics-only resume points

If Home has already completed and a failure occurs in Electronics, use an
Electronics resume point rather than returning to Home:

```text
resume-after-electronics-preprocess
resume-after-electronics-reference
resume-after-electronics-primary-attack
resume-after-electronics-primary-reuse
resume-after-electronics-matched-twins
resume-after-electronics-shape
resume-after-electronics-complementarity
resume-after-electronics-reference-history
resume-after-electronics-strength
resume-after-electronics-k6-audit
resume-after-electronics-full-background
resume-after-electronics-self-influence
```

Example:

```bat
RUN_PHASE4_WINDOWS.cmd resume-after-electronics-shape
```

or on Linux:

```bash
bash RUN_PHASE4_LINUX.sh resume-after-electronics-shape
```

These Electronics resume targets do not rerun Home.

## 7. Reporting outputs

Main-paper figures:

```text
artifacts/figures/
```

Electronics figure files used in Appendix H:

```text
artifacts/electronics/figures/
```

The six Electronics PDFs are:

```text
fig2a_electronics_reuse.pdf
fig2b_electronics_matched.pdf
fig2c_electronics_shape.pdf
fig3a_electronics_complementarity.pdf
fig3b_electronics_reference_history.pdf
fig3c_electronics_intervention_strength.pdf
```

Cross-category appendix table:

```text
artifacts/tables/supp_electronics.tex
```

The arXiv appendices contain 17 registered tables.

## 8. Final verification

Run:

```bash
python tools/final_release_check.py
```

or:

```bat
RUN_PHASE4_WINDOWS.cmd final-check
```

or:

```bash
bash RUN_PHASE4_LINUX.sh final-check
```

Required final line:

```text
FINAL STATUS: PASS
```

The compact report is:

```text
artifacts/verification/final_release_report.json
```

Final frozen-gold integrity:

```bash
python tools/verify_results.py integrity \
  --expected-dir paper_results/expected \
  --report artifacts/verification/gold_integrity_final.json
```

Required:

```text
differences = []
failures    = 0
status      = PASS
```

## 9. Relationship to the arXiv paper

The standalone supplement used for conference review has been incorporated
into Appendices A–I of the arXiv paper and is therefore not distributed in
this public repository.

The corresponding appendix source is available at:

[`paper/appendix.tex`](paper/appendix.tex)

Historical internal filenames containing `supplement` or `supp_` are retained
to preserve the validated computational pipeline. They generate the tables and
figures now presented in the appendices.


## 10. Included and excluded material

Included:

- scientific implementation and experiment entry points;
- frozen configurations;
- Windows and Linux orchestration front-ends;
- verification tools and regression tests;
- immutable expected-result gold, including the compact binary NPZ objects
  consumed by the controlled verifiers;
- canonical figure data;
- rendered main-paper and Electronics appendix figures;
- generated appendix-table fragments and their compact data;
- documentation;
- compact final verification reports; and
- paper/appendix.tex.

Not included:

- `Home_and_Kitchen.jsonl.gz`;
- `Electronics.jsonl.gz`;
- `amazon_preprocess/`;
- `phase3-inputs/`;
- development Git history;
- virtual environments and caches;
- bytecode;
- patch/rollback directories; or
- large experiment intermediates.

## 11. Final package integrity

The GitHub artifact was independently reproduced on both Windows 11 and Linux, with the complete reproduction workflow ending in `FINAL STATUS: PASS` on both platforms. After this validation, the release contents were frozen and a package-wide SHA-256 manifest was generated with:

```bash
python FINALIZE_PACKAGE_HASHES.py .
```

The resulting manifest is included at:

```text
PACKAGE_MANIFEST_SHA256.txt
```

It records the SHA-256 identities of the frozen release contents and provides a package-level integrity reference for the final artifact.

The manifest was generated only after independent cross-platform validation and immediately before the final artifact commit and release tag.

## 12. Additional documentation

See:

```text
docs/REPRODUCIBILITY.md
docs/DATA.md
docs/EXPERIMENTS.md
docs/CATEGORY_AWARE_AMAZON.md
```

for additional implementation and provenance details.
