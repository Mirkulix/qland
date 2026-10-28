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

# automatic failover and catch-up
"$REPO/bench/sim/cluster_auto.py" >/dev/null

# continuous leader loop
"$REPO/bench/sim/cluster_loop.py" >/dev/null

# leader retry / follower catch-up
"$REPO/bench/sim/leader_replicate.py" >/dev/null

# 3-node log replication
"$REPO/bench/sim/replicate_run.py" >/dev/null

# 3-node QAL cluster
"$REPO/bench/sim/cluster_run.py" >/dev/null

# file-backed Raft node endpoint
run_case node
"$QAL" define "$REPO/bench/sim/node.qal" >/dev/null
printf '[{"method":"POST","path":"/heartbeat","body":"hb:1:2:100"},{"method":"GET","path":"/state"}]\n' > reqs.json
"$QAL" request handle --init setup --allow fs:. reqs.json | grep -q '"leader":1'

# wire format
run_case wire
"$QAL" define "$REPO/bench/sim/wire.qal" >/dev/null
"$QAL" test heartbeat_roundtrip heartbeat_bad vote_roundtrip >/dev/null

# cluster tick (heartbeat + timeout + election)
run_case cluster
"$QAL" define "$REPO/bench/sim/cluster.qal" >/dev/null
"$QAL" test heartbeat_keeps_follower timeout_elects_single_node >/dev/null

# heartbeat/timeout
run_case heartbeat
"$QAL" define "$REPO/bench/sim/heartbeat.qal" >/dev/null
"$QAL" test heartbeat_resets stale_term_ignored timeout_promotes fresh_heartbeat_stays >/dev/null

# snapshot installation
run_case install
"$QAL" define "$REPO/bench/sim/install.qal" >/dev/null
"$QAL" test install_keeps_newer install_advances_commit install_can_advance_empty >/dev/null

# log compaction
run_case snapshot
"$QAL" define "$REPO/bench/sim/snapshot.qal" >/dev/null
"$QAL" test compact_keeps_suffix compact_beyond_commit_is_refused compact_after_all >/dev/null

# raft replication
run_case repl
"$QAL" define "$REPO/bench/sim/repl.qal" >/dev/null
"$QAL" test majority_commits all_three_append divergent_follower_rejected >/dev/null

# raft log matching
run_case raft
"$QAL" define "$REPO/bench/sim/raft.qal" >/dev/null
"$QAL" test append_after_zero reject_term_mismatch advance_commits advance_cannot_skip >/dev/null

# leader election
run_case election
"$QAL" define "$REPO/bench/sim/election.qal" >/dev/null
"$QAL" test majority_elects duplicate_vote_counts_once minority_does_not_elect >/dev/null

# majority quorum
run_case quorum
"$QAL" define "$REPO/bench/sim/quorum.qal" >/dev/null
"$QAL" test majority_wins single_wins tie_has_no_quorum empty_has_no_quorum >/dev/null

# conflict resolution policies
run_case resolve
"$QAL" define "$REPO/bench/sim/resolve.qal" >/dev/null
"$QAL" test min_resolves_to_lower max_resolves_to_higher reject_keeps_conflict >/dev/null

# conflict-aware merge
run_case conflict
"$QAL" define "$REPO/bench/sim/conflict.qal" >/dev/null
"$QAL" test same_payload_no_conflict different_amount_conflict disjoint_merges >/dev/null

# journal merge
run_case merge
"$QAL" define "$REPO/bench/sim/merge.qal" >/dev/null
"$QAL" test merge_dedup merge_idempotent merge_keeps_both >/dev/null

# durable transaction journal
run_case journal
"$QAL" define "$REPO/bench/sim/journal.qal" >/dev/null
"$QAL" test duplicate_is_noop roundtrip_lines >/dev/null
"$QAL" run main --allow io,fs:/tmp -- /tmp/qal-journal-verify.txt >/dev/null

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

echo "simulation: queue 3, tensor 3, numerics 2, auto 1, loop10 1, leader1 1, repl3 1, cluster3 1, node 1, wire 3, cluster 2, heartbeat 4, install 3, snapshot 3, repl 3, raft 4, election 3, quorum 4, resolve 3, conflict 3, merge 3, journal 2, twopc 2, once 2, cross 2, sim_package 2, sim 100000, queue-sim 1000 events, shards 10, dashboard 200"
