#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if ! python3 -c 'import mlx.core' 2>/dev/null; then
  echo "mlx: not installed, skipped"
  exit 0
fi
QAL="${QAL_BIN:-$PWD/qal/target/release/qal}"
python3 bench/tensor/server_mlx.py --port 8901 >/tmp/qal-mlx-server.log 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null || true; rm -f /tmp/qal-tensor-mlx-combined.qal' EXIT
sleep 1.5
cat bench/tensor/host.qal bench/tensor/main.qal > /tmp/qal-tensor-mlx-combined.qal
OUT="$("$QAL" run /tmp/qal-tensor-mlx-combined.qal --allow io,net:127.0.0.1:8901)"
echo "$OUT" | grep -q '"stdout":"19.0,22.0,43.0,50.0\\n"'
python3 bench/tensor/bench_mlx.py --n 512 --iters 10 >/tmp/qal-mlx-bench.json
python3 - <<'PY'
import json
d=json.load(open('/tmp/qal-mlx-bench.json'))
print(f"tensor-mlx: 2x2 via MLX/Metal, {d['mlx_gflops']} GFLOPS vs numpy {d['numpy_gflops']} ({d['speedup']}x)")
PY
