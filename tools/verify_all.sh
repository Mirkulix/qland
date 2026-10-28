#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export QAL_BIN="${QAL_BIN:-$PWD/qal/target/release/qal}"
python3 bench/conformance/run.py >/tmp/qal-conformance.json
python3 qal-registry/selftest.py >/tmp/qal-registry-selftest.json
python3 tools/contract_diff.py --base HEAD >/tmp/qal-contract-diff.json
./bench/sim/verify.sh >/tmp/qal-sim-verify.txt
echo "conformance: $(python3 -c 'import json;print(len(json.load(open("/tmp/qal-conformance.json"))["results"]))') cases"
echo "registry selftest: $(python3 -c 'import json;print(len(json.load(open("/tmp/qal-registry-selftest.json"))["results"]))') cases"
echo "contract diff: $(python3 -c 'import json;print(len(json.load(open("/tmp/qal-contract-diff.json"))["findings"]))') findings"
cat /tmp/qal-sim-verify.txt
