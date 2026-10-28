# QAL 100-agent cash simulation

`sim.qal` is a discrete, auditable simulation:

- 100 agents, each starting with 1000 cash, in a `Map[Int, Agent]`;
- 1000 randomized transfer steps (`rand_int`), effect `!rand`;
- pure `transfer` with `ens total(r) == total(sim)`;
- `main` prints the total and final tick.

## Measured

```text
qal check sim.qal  -> {"ok":true}
qal run sim.qal --allow io,rand
  -> agents=100 total=100000 tick=989
qal test sim.qal
  -> transfer_conserves passed
  -> self_transfer_is_noop passed
```

## The contract caught a real bug

The first run failed with:

```json
{"kind":"contract_violation","def":"transfer","path":"ens[0]",
 "message":"`ens total(r) == total(sim)` of `transfer` is false",
 "values":{"from":91,"to":91,"amount":50,"tick":19}}
```

A self-transfer (`from == to`) subtracted from the source and added the old
destination balance, creating 50 cash. Python tests would only catch this if
they happened to test `from == to`; the QAL postcondition caught it immediately.
After the fix:

```qal
= if from == to then sim else match map_get(sim.agents, from) { ... }
```

the run conserves 100,000 exactly.

## What this demonstrates

- QAL can express a discrete event/agent simulation;
- the invariant is checked at runtime, not just in tests;
- deterministic replay is possible with a fixed `rand` seed;
- the same pattern scales to scheduler/event-queue libraries and host numerics.

It is **not** a high-performance simulator: the interpreter, immutable maps and
recursion are fine for 100 agents/1000 steps, not millions.

## Event queue package

`queue.qal` defines a deterministic event queue over `Map[Int, Event]`:

- `enqueue(q, at, kind, payload)` keys by `at * 1_000_000 + seq`,
- `pop(q)` sorts keys, removes the first event and returns `Option[Popped]`,
- tests: `push_size`, `pop_order`, `pop_empty` all pass.

It is exported as `qal/packages/event_queue.qalpkg` (`qal_event_queue`, 10
definitions) and installs through the signed registry:

```text
installed qal_event_queue d8e8b761…
qal query --name enqueue -> fn(Queue, Int, Str, Str) -> Queue
qal test push_size pop_order pop_empty -> 3 passed
```

The queue is QAL-owned logic; time and worker execution stay in the host.

## Tensor package

`tensor.qal` is a row-major `Tensor { rows, cols, data }` package:

- `at`, `dot`, `matmul`, `identity`;
- `matmul` has `ens r.rows == a.rows && r.cols == b.cols`;
- tests: identity shape, identity multiplication and the 2x2 product (43).

Exported as `qal/packages/tensor.qalpkg` (`qal_tensor`, 11 definitions), signed
and installed through the registry:

```text
qal query --name matmul -> fn(Tensor, Tensor) -> Tensor
qal test identity_shape identity_matmul matmul_2x2 -> 3 passed
```

This is the QAL-owned shape/contract layer; a GPU/NumPy host would implement the
inner multiplication later.

## Numerics package

`numerics.qal` adds `integrate(f, a, b, n)` (left Riemann sum) with
`req n > 0 && n <= 1_000_000 && a <= b`, plus `square`, `linear` and `absf`.
Tests: integral of x^2 on [0,1] and x on [0,2], both within 0.01.

Exported as `qal/packages/numerics.qalpkg` (`qal_numerics`, 6 definitions),
signed and installable:

```text
qal query --name integrate
  -> fn(fn(Float) -> Float, Float, Float, Int) -> Float
qal test square_integral linear_integral -> 2 passed
```

QAL owns the contract and the loop shape; a host GPU/NumPy implementation can
replace the inner numerical work later.

## Queue-driven simulation

`sim_queue.qal` imports `qal_event_queue`, schedules 1000 tick events and processes
them in order; each event triggers a randomized transfer.

```sh
qal init
qal import qal/packages/event_queue.qalpkg
qal define bench/sim/sim_queue.qal
qal run main --allow io,rand
```

Verified:

```text
agents=100 total=100000 events=1000 tick=1000
```

The queue is the scheduler; `transfer` keeps the conservation contract.
`add_events` carries a bounded-range precondition; the prover reports `transfer`'s
map-based conservation as `unknown` (checked at runtime), which is honest.

## Dashboard endpoint

`dashboard.qal` serves both the dashboard shell and its metrics:

- `GET /` → HTML page that fetches `/metrics` and renders the JSON;
- `GET /metrics` → `{"agents":100,"total":100000,"tick":1000,
  "invariant":"total == 100000","source":"qal"}`.

```sh
qal define bench/sim/dashboard.qal
qal request handle reqs.json --allow io
```

Verified:

```text
GET /        -> 200 text/html
GET /metrics -> 200 application/json, total=100000
qal test total_100 -> passed
```

The HTML/JS is a shell; QAL owns the metric values, the invariant and the JSON
contract. Charts/updates remain a browser concern.

## Scaling limit (measured)

Same `sim.qal`, varying `init(agents)` and `run(..., steps)` on the release
binary, 256 MiB Wasm memory limit:

| agents | steps | result | wall | total |
|---:|---:|---|---:|---:|
| 100 | 1,000 | ok | 0.24 s | 100,000 |
| 500 | 5,000 | ok | 0.50 s | 500,000 |
| 1,000 | 5,000 | ok | 0.76 s | 1,000,000 |
| 2,000 | 1,000 | ok | 0.43 s | 2,000,000 |
| 5,000 | 1,000 | ok | 0.75 s | 5,000,000 |
| 2,000 | 5,000 | **OOM** (256 MiB) | 0.80 s | – |
| 1,000 | 10,000 | **OOM** (256 MiB) | 0.89 s | – |

The wall-clock time is not the problem; the **persistent immutable map** grows
until the execution's 256 MiB limit. So QAL's simulation package is good for
thousands of agents/steps, not millions. A million-agent runtime needs native
mutable state, sharding and a different memory model; QAL should define the
rules/contracts, not interpret every agent.

The core (without `main`) is also `qal/packages/sim_core.qalpkg` (`qal_sim`),
10 definitions; signed install verified, `transfer_conserves` and
`self_transfer_is_noop` pass after import.

## Million-agent sharding

`shard.qal` is one independent shard; `shard.py` runs many processes so each has
its own 256 MiB budget.

```sh
./bench/sim/shard.py --shards 100 --agents 10000 --steps 100
# {"shards":100,"agents":10000,"steps":100,"total_agents":1000000,
#  "jobs":8,"total":1000000000,"expected":1000000000,"seconds":8.51}
```

- **1,000,000 agents**, 100 steps each, **8.51 s** (8 parallel shard processes),
- total conserved: **1,000,000,000**,
- each shard is a fresh QAL execution, so the persistent-map OOM is avoided.

Caveat: shards are independent; transfers do not cross shard boundaries. A
production sharded runtime must handle cross-shard messages explicitly.

## Cross-shard primitive

`cross.qal` (package `qal_sim_cross`, 10 defs) moves cash between two shard
states in one call:

```qal
fn transfer_between(x: Pair, from: Int, to: Int, amount: Int) -> Pair
  ens total(r.a) + total(r.b) == total(x.a) + total(x.b)
```

Verified: `cross_conserves` and `cross_debits` pass; signed registry install
SHA-256 `1672d02a…`. The host still owns coordination/ordering; QAL owns the
two-shard conservation contract.

## Exactly-once cross-shard transfers

`idempotent.qal` (package `qal_sim_once`, 12 defs) wraps a transfer in a `Txn`
with a `seen` map:

```qal
fn once(t: Txn, id: Int, from: Int, to: Int, amount: Int) -> Txn
```

A replay with the same `id` returns the unchanged state. Tested:
`replay_is_noop` (balance stays 700 after two calls) and `first_debits`; total
conservation holds. Signed install SHA-256 `99547aa9…`.

## Two-phase commit

`twopc.qal` (package `qal_sim_2pc`, 14 defs) models prepare/commit/abort over a
two-shard `Coord`:

- `prepare` validates the source and records "prepared",
- `commit` applies the transfer exactly once and records "committed",
- `abort` marks a non-committed transaction "aborted".

Tests `commit_once` (700 after commit, total 2000) and `abort_noop` (1000 after
abort) pass. Signed install SHA-256 `8f050e15…`.

## Durable transaction journal

`journal.qal` (package `qal_sim_journal`, 12 defs) appends `id:kind:amount`
entries, ignores duplicate ids, serializes/loads through `write_file`/
`read_file` with `!fs`. Verified: `duplicate_is_noop`, `roundtrip_lines`, and a
write/read run (`1:debit:5` on disk). Signed install SHA-256 `2d1fe93a…`.

## Journal merge

`merge.qal` (package `qal_sim_merge`, 7 defs) unions two journals by transaction
id; a duplicate is a no-op. Tests `merge_dedup`, `merge_idempotent`,
`merge_keeps_both` pass. Signed install SHA-256 `36d07639…`. This is the
replication primitive for a distributed journal; conflict resolution/ordering
stay with the host.

## Conflict-aware merge

`conflict.qal` (package `qal_sim_conflict`, 12 defs) returns both the merged log
and the list of same-id/different-payload conflicts:

```qal
type Merged = { log: List[Entry], conflicts: List[Conflict] }
fn merge_checked(a, b) -> Merged
```

Tests `same_payload_no_conflict`, `different_amount_conflict`, `disjoint_merges`
pass. Signed install SHA-256 `18df143d…`. The host decides how to resolve a
reported conflict; QAL never silently overwrites.

## Conflict resolution policies

`resolve.qal` (package `qal_sim_resolve`, 13 defs) adds deterministic policies:

- `"min"` / `"max"` pick the lower/higher amount,
- `"reject"` keeps the conflict and leaves the original entry,
- `merge_policy(a, b, policy) -> Merged`.

Tests `min_resolves_to_lower`, `max_resolves_to_higher`,
`reject_keeps_conflict` pass. Signed install SHA-256 `2e7f32f8…`. The policy name
is the host's choice; QAL makes the result deterministic and auditable.

## Majority quorum

`quorum.qal` (package `qal_sim_quorum`, 8 defs) counts replica votes and returns
a value only with a strict majority:

```qal
fn quorum(votes: List[Vote], n: Int) -> Option[Str]
```

Tests `majority_wins` (2/3), `single_wins`, `tie_has_no_quorum` (1/2),
`empty_has_no_quorum` pass. Signed install SHA-256 `c2480194…`. The host
collects votes and retries; QAL decides what counts as a quorum.

## Leader election

`election.qal` (package `qal_sim_election`, 8 defs) counts votes for a candidate
in a term:

```qal
fn vote(b: Ballot, node: Int) -> Ballot
fn leader(b: Ballot) -> Option[Int]
```

A leader needs a strict majority of the cluster; duplicate votes count once.
Tests `majority_elects`, `duplicate_vote_counts_once`,
`minority_does_not_elect` pass. Signed install SHA-256 `8308521d…`. The host
sends heartbeats/retries; QAL decides when a candidate has won.

## Raft log

`raft.qal` (package `qal_sim_raft`, 9 defs) implements log matching and commit
index:

```qal
fn append(log, prev_index, prev_term, e) -> RaftLog
fn advance(log, index) -> RaftLog
```

Tests `append_after_zero`, `reject_term_mismatch`, `advance_commits`,
`advance_cannot_skip` pass. Signed install SHA-256 `2b72f5de…`. The host owns
transport, timeouts and voting; QAL owns the log/commit rules.

## Raft replication

`repl.qal` (package `qal_sim_repl`, 12 defs) appends a leader command to all
matching followers and commits it once a strict majority has it:

```qal
fn replicate(c: Cluster, e: LogEntry) -> Cluster
```

Tests `majority_commits`, `all_three_append`, `divergent_follower_rejected`
pass. Signed install SHA-256 `c5dc86dc…`. Transport, retries and leader
heartbeats stay with the host.

## Log compaction

`snapshot.qal` (package `qal_sim_snapshot`, 10 defs) compacts a committed
prefix into a snapshot and keeps the suffix:

```qal
fn compact(log: RaftLog, index: Int, state: Str) -> Compacted
```

Tests `compact_keeps_suffix`, `compact_beyond_commit_is_refused`,
`compact_after_all` pass. Signed install SHA-256 `cd726b4d…`. Snapshots are
QAL data; durable storage/transport stay with the host.

## Snapshot installation

`install.qal` (package `qal_sim_install`, 8 defs) installs a snapshot: entries at
or below `last_index` are dropped, newer ones stay, and the commit index advances
to at least `last_index`:

```qal
fn install(snap: Snapshot, log: RaftLog) -> RaftLog
```

Tests `install_keeps_newer`, `install_advances_commit`,
`install_can_advance_empty` pass. Signed install SHA-256 `e3b84086…`.

## Heartbeats and timeouts

`heartbeat.qal` (package `qal_sim_heartbeat`, 9 defs) tracks a follower:

- a heartbeat at an equal/newer term resets to follower and records the leader,
- an older term is ignored,
- no heartbeat for `timeout` promotes to candidate and increments the term.

Tests `heartbeat_resets`, `stale_term_ignored`, `timeout_promotes`,
`fresh_heartbeat_stays` pass. Signed install SHA-256 `77e9fe7a…`.

## Cluster tick

`cluster.qal` (package `qal_sim_cluster`, 14 defs) composes heartbeat, timeout and
election into one `tick`:

```qal
fn tick(c: ClusterState, hb: Heartbeat, now: Int, timeout: Int) -> ClusterState
```

A heartbeat keeps the follower; a timeout promotes to candidate, votes for
itself, and a single-node cluster elects it. Tests `heartbeat_keeps_follower`
and `timeout_elects_single_node` pass. Signed install SHA-256 `934e87b8…`.

## Wire format

`wire.qal` (package `qal_sim_wire`, 9 defs) encodes/decodes Raft messages:

- `hb:<leader>:<term>:<at>`,
- `rv:<term>:<candidate>:<last_index>:<last_term>`.

Tests `heartbeat_roundtrip`, `heartbeat_bad`, `vote_roundtrip` pass. Signed
install SHA-256 `9744e680…`. The host owns sockets/TLS; QAL owns the payload
contract.
