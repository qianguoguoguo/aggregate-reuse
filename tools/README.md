# Verification tools

`verify_results.py` is the generic numerical/provenance verifier introduced in
Phase 2B. It contains no scientific experiment implementation.

Examples:

```bash
make verify-gold
make verifier-selftest
```

Later phases will use its JSON/CSV comparison functions to compare newly
migrated experiment outputs against immutable gold.
