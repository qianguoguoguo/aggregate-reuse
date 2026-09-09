# Category-aware Amazon pipeline

The merged repository uses the same Amazon experiment implementations for
`home_and_kitchen` and `electronics`. Category identity, raw filename, shared
artifact root, and category-specific expected values are supplied by YAML
configuration rather than by source edits.

## Category configurations

Home-and-Kitchen retains the historical paper configs at `configs/amazon_*.yaml`
for backward compatibility. Electronics uses the corresponding files under
`configs/electronics/`.

The raw review files are expected one directory above the repository:

```text
../Home_and_Kitchen.jsonl.gz
../Electronics.jsonl.gz
```

Operational outputs are isolated as follows:

```text
../amazon_preprocess/                 # Home-and-Kitchen legacy root
../amazon_preprocess/electronics/     # Electronics root
```

The `--shared-root` option may be used for a clean-room reproduction without
changing the frozen config bytes or config SHA-256.

## Run Electronics

```bash
make electronics-preprocess
make electronics-downstream
make electronics-verifier-selftest
make electronics-verify
make electronics-reporting
```

The same stages can be run individually with the `electronics-*` Make targets.
All category-aware Amazon entry points also accept an explicit `--config` and,
where applicable, `--shared-root`.

## Run Home-and-Kitchen

The original Home targets remain available:

```bash
make amazon-preprocess
make amazon-downstream
```

For a new symmetric clean-room run, pass an operational shared root directly
to the category-aware entry points. The historical paper artifact layout and
frozen Home regression files remain unchanged by this compatibility-first
merge.

## Verification contract

- Category, schemas, seeds, integer counts, retained identifiers, discrete
  construction fields, and portable structural hashes are checked exactly.
- Derived floating-point quantities use `rtol=1e-10` and `atol=1e-12` in the
  cross-platform verifiers.
- Electronics invariant verification does not compare against Home-and-Kitchen
  results or assume Home-specific counts, selected values, or AUCs.
- Paper-specific Home regression verifiers remain separate from generic or
  Electronics verification.

## Phase-1 scope

This merge adds category-aware orchestration, provenance, reporting, and
cross-platform comparison. It does not yet sanitize previously generated
absolute paths, publish Electronics gold, or assert final Windows/Linux
reproduction. Those tasks belong to the subsequent anonymity and validation
phases.
