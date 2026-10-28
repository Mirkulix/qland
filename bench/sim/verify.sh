#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
QAL="${QAL_BIN:-$PWD/qal/target/release/qal}"
REPO="$PWD"
TMP="$(mktemp -d /tmp/qal-sim-verify.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

run_case() {
  local name="$1"; shift
  local dir="$TMP/$name"; mkdir -p "$dir"; cd "$dir"
  "$QAL" init >/dev/null
  for pkg in "$@"; do "$QAL" import "$pkg" >/dev/null; done
}

# queue package
run_case queue
"$QAL" define "$REPO/bench/sim/queue.qal" >/dev/null
"$QAL" test push_size pop_order pop_empty >/dev/null

# tensor package
run_case tensor
"$QAL" define "$REPO/bench/sim/tensor.qal" >/dev/null
"$QAL" test identity_shape identity_matmul matmul_2x2 >/dev/null

# numerics package
run_case numerics
"$QAL" define "$REPO/bench/sim/numerics.qal" >/dev/null
"$QAL" test square_integral linear_integral >/dev/null

# two-phase commit
run_case twopc
"$QAL" define "$REPO/bench/sim/twopc.qal" >/dev/null
"$QAL" test commit_once abort_noop >/dev/null

# idempotent cross-shard transfers
run_case once
"$QAL" define "$REPO/bench/sim/idempotent.qal" >/dev/null
"$QAL" test replay_is_noop first_debits >/dev/null

# cross-shard transfer primitive
run_case cross
"$QAL" define "$REPO/bench/sim/cross.qal" >/dev/null
"$QAL" test cross_conserves cross_debits >/dev/null

# packaged simulation core
run_case sim_package "$REPO/qal/packages/sim_core.qalpkg"
"$QAL" test transfer_conserves self_transfer_is_noop >/dev/null

# 100-agent simulation
run_case sim
"$QAL" define "$REPO/bench/sim/sim.qal" >/dev/null
OUT="$("$QAL" run main --allow io,rand)"
echo "$OUT" | grep -q 'total=100000'

# queue-driven simulation
run_case sim_queue "$REPO/qal/packages/event_queue.qalpkg"
"$QAL" define "$REPO/bench/sim/sim_queue.qal" >/dev/null
OUT="$("$QAL" run main --allow io,rand)"
echo "$OUT" | grep -q 'total=100000.*events=1000'

# sharded simulation (small version of the million-agent run)
"$REPO/bench/sim/shard.py" --shards 10 --agents 1000 --steps 100 >/dev/null

# dashboard
run_case dashboard
"$QAL" define "$REPO/bench/sim/dashboard.qal" >/dev/null
"$QAL" test total_100 >/dev/null
printf '[{"method":"GET","path":"/metrics"}]\n' > reqs.json
"$QAL" request handle reqs.json --allow io | grep -q '"total":100000'

echo "simulation: queue 3, tensor 3, numerics 2, twopc 2, once 2, cross 2, sim_package 2, sim 100000, queue-sim 1000 events, shards 10, dashboard 200"
