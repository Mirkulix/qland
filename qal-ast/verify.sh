#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 qal-ast/validate.py qal-ast/simple.json qal-ast/types.json qal-ast/tests.json qal-ast/test_only.json >/dev/null
if python3 qal-ast/validate.py qal-ast/bad.json >/dev/null 2>&1; then
  echo "bad.json unexpectedly valid" >&2; exit 1
fi
python3 qal-ast/lower.py qal-ast/simple.json --check >/dev/null
echo "ast: schema v1, 3 good docs, bad rejected, simple lowers+checks"
