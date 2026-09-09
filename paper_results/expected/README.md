# Step 1B — Frozen Gold Numerical Results

This package is the pre-refactor numerical ground truth for the WSDM paper and
supplement.

## Frozen inputs

Source repository: `aggregate_reuse_repro - Copy(1).zip`  
SHA-256: `4536cdfedcac1336ac59963344df6e55c9ebb75232be663161daf5427e51682a`

Source supplement: `supplement.tex`  
SHA-256: `d2f4b5858fda048303384149ae7607cbd02c49d7bb0bb02b963d8e45de2314c7`

## Contents

- `gold_results.json` — master full-precision verification payload.
- `supplement_table_gold.json` — full-precision table data plus the exact
  display strings currently printed in the supplement.
- `reported_prose_gold.json` — important result values reported outside tables.
- `source_hashes.csv` — SHA-256 provenance for each accepted source artifact.
- `source_snapshots/` — compact copies of the accepted result summaries.
- `figure_data/` — current numerical source data for Figures 1–3.

## Stage-2 feasibility audit

Both experimental blocks feasible for k=6: **21,197**  
Both experimental blocks feasible for k=9: **13,268**

Manifest columns used: `None`

## Known gaps to close during refactoring

1. The auxiliary k=6 feasibility-population audit in the supplement reports
   `(0.0761, 0.739)`, `(-0.0004, 0.509)`, and exact expectations
   `0.07637`, `0.00079`, but the trimmed repository contains no dedicated
   compact result artifact for that audit.
2. Separate compact per-seed result vectors are not retained for
   Stages 5/6/7/9/10/11. The clean repository should emit them.
3. The reported preprocessing removal count **786,064** is not stored in the
   retained Stage-2 summary. The final preprocessing report should record it.

These are provenance/output-format gaps, not discrepancies in the frozen
reported results.

## Rule for subsequent phases

Do not edit this gold package. Refactored code must write results elsewhere and
be compared against these values/hashes.
