#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  printf '%s\n' \
    'Usage: bash scripts/run_v3_server.sh <command>' \
    '' \
    'Audit and planning commands (no model fitting):' \
    '  scientific-audit   Validate the Evidence Tree and write the current status report.' \
    '  training-readiness Fail closed unless every scientific and data gate is resolved.' \
    '  tests              Run the focused V3 scientific-audit and run-grid tests.' \
    '  status             Print the last written audit report, if present.' \
    '  prepare-grid       Generate the locked 420-run V3 server config grid.' \
    '  plan               Verify the locked grid and print the execution plan.' \
    '  verify-grid        Re-hash every generated config against the run grid.' \
    '' \
    'Training remains fail-closed until training_authorized=true and training-readiness passes.'
}

command="${1:-}"
case "$command" in
  scientific-audit)
    python scripts/v3_scientific_audit.py \
      --require-structure \
      --output v3/registry/scientific_audit_status.json
    ;;
  training-readiness)
    python scripts/v3_scientific_audit.py \
      --require-training-ready \
      --output v3/registry/scientific_audit_status.json
    ;;
  tests)
    python -m pytest tests/test_v3_scientific_audit.py tests/test_v3_run_grid.py -q
    ;;
  status)
    if [[ ! -f v3/registry/scientific_audit_status.json ]]; then
      printf '%s\n' 'No audit report exists. Run scientific-audit first.' >&2
      exit 1
    fi
    python -m json.tool v3/registry/scientific_audit_status.json
    ;;
  prepare-grid)
    python scripts/generate_v3_run_grid.py --prepare-configs
    ;;
  plan)
    python scripts/generate_v3_run_grid.py --plan
    ;;
  verify-grid)
    python scripts/generate_v3_run_grid.py --verify
    ;;
  train|training|run)
    printf '%s\n' \
      'V3 training is intentionally fail-closed.' \
      'Current state: run grid generated; training_authorized remains false.' \
      'Resolve scientific/data gates, set training_authorized=true under PI review,' \
      'then re-run training-readiness before any GPU fitting.' >&2
    exit 2
    ;;
  -h|--help|help|'')
    usage
    ;;
  *)
    printf 'Unknown command: %s\n\n' "$command" >&2
    usage >&2
    exit 64
    ;;
esac
