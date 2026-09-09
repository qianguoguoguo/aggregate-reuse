# Phase 3: cross-platform, cross-category smoke validation

Phase 3 validates the merged category-aware code before the expensive full
Windows/Linux reproductions. It intentionally reuses already-frozen Stage-2
and Stage-3 category inputs and regenerates one downstream seed in an isolated
`_smoke/` namespace. Canonical result directories are never overwritten.

## What this phase tests

For each Amazon category, run seed 0 on Windows and Linux using the same
logical Stage-2 roles, Stage-2 reference histograms, and Stage-3 reference
freeze. The smoke harness verifies:

- category routing and isolation;
- exact discrete attack construction;
- exact identity assignments at every reuse value;
- exact matched-twin exposures;
- exact mean-preserving shape construction;
- exact complementarity incidence graphs;
- exact nested fixed-identity strength construction;
- all embedded stage invariants;
- floating-point agreement under `rtol=1e-10`, `atol=1e-12`;
- anonymity of newly generated smoke artifacts.

PDF/PNG bytes are not compared. This phase compares scientific structure and
numerical outputs, not renderer metadata.

## Why Stage 2/3 are reused

Full Amazon preprocessing and reference calibration scan the complete category
corpus and are not cheap smoke tests. Phase 3 isolates downstream cross-platform
behavior by requiring the Windows and Linux smoke runs to start from identical
logical Stage-2/3 inputs. The later full reproduction phase reruns preprocessing
and calibration independently on both operating systems.


## Prepare identical Stage-2/3 inputs for both operating systems

To isolate downstream platform differences, prepare one portable input root per
category and copy that same folder to Windows and Linux:

```bash
python tools/prepare_phase3_inputs.py \
  --category home_and_kitchen \
  --source-root /data/canonical_home \
  --dest-root /data/phase3_home \
  --clean
```

and similarly for Electronics. The helper copies only the three inputs needed
by all downstream smoke stages: the Stage-2 roles corpus, Stage-2 reference
histograms, and Stage-3 reference freeze. It writes a compact
`phase3_input_manifest.json` with logical content hashes.

Copy the resulting category root unchanged to the other operating system. The
Windows/Linux smoke manifests must report the same logical input hashes before
any downstream result is compared.

## Core smoke

From the repository root:

```bash
python tools/run_phase3_smoke.py \
  --category home_and_kitchen \
  --shared-root /data/phase3_home \
  --seed 0 --scope core --clean
```

and:

```bash
python tools/run_phase3_smoke.py \
  --category electronics \
  --shared-root /data/phase3_electronics \
  --seed 0 --scope core --clean
```

The category root must contain the canonical `stage2/` and `stage3/` inputs.
Generated data are written only below:

```text
<shared-root>/_smoke/phase3_seed0/
```

The repository receives only logs, temporary smoke configs, and a compact
manifest under `artifacts/phase3_smoke/` and `artifacts/verification/`.

## Full one-seed smoke

After the core Windows/Linux comparison passes, repeat with `--scope full` to
also exercise reference-history sensitivity, the deterministic k=6 population
audit, and the one-world full-background ranking experiment.

## Compare Windows and Linux

Copy the two compact manifest JSON files onto either machine and run:

```bash
python tools/compare_phase3_smoke.py windows.json linux.json \
  --report artifacts/verification/phase3_compare_home.json
```

A passing comparison requires exact equality of all structural fingerprints
and exact non-floating summary values. Derived floats may differ only within:

```text
rtol = 1e-10
atol = 1e-12
```

The comparison report lists the largest nonzero numerical differences even
when they remain within tolerance.

## Acceptance order

1. Home core: Windows vs Linux PASS.
2. Electronics core: Windows vs Linux PASS.
3. Home full one-seed: Windows vs Linux PASS.
4. Electronics full one-seed: Windows vs Linux PASS.
5. Proceed to independent full preprocessing/reference and 30-seed runs.
