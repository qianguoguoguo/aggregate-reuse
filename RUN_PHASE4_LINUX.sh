#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  bash RUN_PHASE4_LINUX.sh TARGET

Common targets:
  verify-raw
  controlled
  home-preprocess
  home-downstream
  electronics-preprocess
  electronics-downstream
  amazon-both-from-raw
  reporting
  final-check
  reproduce-final-from-raw

Electronics-only resume targets:
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
EOF
  exit 2
fi

# The backend keeps its historical filename, but it is implemented in Python
# with pathlib/subprocess/sys.executable and is used identically on both OSes.
exec "${PYTHON:-python}" tools/run_phase4_windows.py "$@"
