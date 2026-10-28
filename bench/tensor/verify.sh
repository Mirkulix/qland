#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
QAL="${QAL_BIN:-$PWD/qal/target/release/qal}"
python3 bench/tensor/server.py --port 8901 >/tmp/qal-tensor-server.log 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null || true; rm -f /tmp/qal-tensor-combined.qal' EXIT
sleep 1
cat bench/tensor/host.qal bench/tensor/main.qal > /tmp/qal-tensor-combined.qal
"$QAL" test bench/tensor/host.qal >/dev/null
OUT="$("$QAL" run /tmp/qal-tensor-combined.qal --allow io,net:127.0.0.1:8901)"
echo "$OUT" | grep -q '"stdout":"19.0,22.0,43.0,50.0\\n"'
"$QAL" check bench/tensor/native.qal >/dev/null
NATIVE="$("$QAL" run bench/tensor/native.qal --allow io,tensor)"
echo "$NATIVE" | grep -q '"stdout":"19.0,22.0,43.0,50.0\\n"'
echo "tensor: NumPy sidecar 2x2 -> 19.0,22.0,43.0,50.0; native !tensor ok"
