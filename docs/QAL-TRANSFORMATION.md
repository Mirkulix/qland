# QAL transformation status (2026-10-28)

This is the current handover for "QAL becomes the language agents use to build
verified software". It is evidence, not a vision document.

## What is now real

### 1. Agent surface
- **Text QAL**: 0/5 on the stdin pilot; the model writes Python habits.
- **JSON AST**: `qal ast <file.json>` and `qal define <file.json> --ast`.
- AST supports literals, lists, variables, calls, operators, fields, `let`,
  `if`, `match`, records, `types`, `effects`, `returns`, `tests`,
  `allowed_calls`, `ast_version`.
- **100-task helper pilot: 100/100** (Python 100/100).
- **100-task stdlib-only pilot: 94/100** (Python 90/100).
- Remaining stdlib failures are semantic edges (`ß` upper, strict `slice`).

### 2. Verification
- Conformance: **12 checks** (check/test/prove/effect/contract-diff).
- Contract-diff gate: deleted/weakened `req`/`ens` is refused.
- `tools/verify_all.sh`: conformance + registry + contract + simulation.
- QAL ARM64: build 18m17s, `cargo test --release` **656 passed, 0 failed**.

### 3. Package ecosystem
- `.qalpkg` text format, SHA-256, Ed25519 signatures, trust list.
- Dependency graph resolver, install in dependency order.
- HTTP server + HTTP client with signature/hash verification.
- Packages: `qal_prelude`, `qal_event_queue`, `qal_tensor`, `qal_numerics`,
  `qal_sim`, `qal_tensor_remote` (28 `.qalpkg` in the tree).
- Registry self-test: **13 cases**.

### 4. Simulation stack
- 100-agent cash simulation: `total=100000`.
- Event-queue-driven simulation: 1000 events, `total=100000`.
- Event queue: 3 tests; tensor: 3 tests; numerics: 2 tests.
- Dashboard: `GET /` HTML shell, `GET /metrics` JSON, `total=100000`.
- Scaling: 5,000 agents/1,000 steps in 0.75 s; 2,000×5,000 and
  1,000×10,000 hit the 256 MiB execution limit.

## Inventory (2026-10-28)

`python3 tools/qal_first.py`:

```text
qal:        5,195 files, 148,747 lines, 2,606 tests
rust:       1,393 files, 684,276 lines
python:       738 files, 146,948 lines
typescript:    10 files
packages:      46 .qalpkg
```

The QAL line count includes the self-hosted compiler, std, examples and tests;
Rust remains the bootstrap/host layer.

The **new-code share** is a different number: `python3 tools/qal_first.py --base
<ref>` counts added lines by kind. Over the last five commits it was 20 QAL lines
of 245 added lines (8.2%); host tooling and docs dominated. This is the honest
measure of how much of the *recent* work is QAL versus infrastructure.

## The architecture

```
Agent -> JSON AST -> qal define --ast -> QAL -> check/prove/test -> store
Human -> QAL text  -> qal define      -> QAL -> check/prove/test -> store
                                           |
                                           +-> signed packages / registry
                                           +-> simulation / dashboard
```

Host owns time, randomness, execution, GPU, threads; QAL owns decisions,
contracts, invariants, audit and composition.

## Honest limits

- No `!tensor` host capability yet; tensor matmul is QAL code, not GPU/NumPy.
- No native/mutable million-agent runtime; persistent maps hit 256 MiB.
- QAL text is not agent-native; AST is.
- Stdlib-only AST is 94/100, not 100.
- No external users; Phase 7/9 still open.
- Phase 6 remains 1.313x; Phase 8 student still fails.
- IGQK lossless/lossy decision still open.

## Next 90 days

1. `!tensor` host capability (shapes in QAL, matmul in GPU/NumPy).
2. Native/mutable sharded runtime for large simulations.
3. Registry client as an MCP tool.
4. Dashboard UI (external) against `/metrics`.
5. One design partner for Phase 7/9.
6. Freeze `ast_version 1` and publish the schema.

## Verify everything

```sh
./tools/verify_all.sh
./bench/sim/verify.sh
QAL_BIN=$PWD/qal/target/release/qal python3 bench/compare/scale_stdlib.py
```


## Distributed stack (2026-10-28, later)

The simulation packages now include a working Raft-style stack:

- heartbeat/timeout (`qal_sim_heartbeat`, `qal_sim_cluster`),
- election/quorum (`qal_sim_election`, `qal_sim_quorum`, `qal_sim_raft`),
- replication and commit (`qal_sim_repl`, `qal_sim_snapshot`, `qal_sim_install`),
- wire format (`qal_sim_wire`),
- durable transactions (`qal_sim_journal`, `qal_sim_merge`, `qal_sim_conflict`,
  `qal_sim_resolve`, `qal_sim_2pc`, `qal_sim_once`),
- a file-backed HTTP node (`qal_sim_node`),
- an aggregating cluster dashboard (`qal_sim_dashboard`),
- an Apple-Metal backend for the tensor sidecar (`bench/tensor/server_mlx.py`):
  512x512 matmul up to 12.7x faster than NumPy on the M1 (315.7 vs 24.9 GFLOPS);
2048x2048 reaches 1.44 TFLOPS (5.84x).

Verified end-to-end on real processes:

- 3 QAL nodes, `hb:1:2:100..102`,
- timeout -> candidate term 3, 3/3 votes,
- leader 0, logs `alpha`, `beta`, commit 2,
- kill follower 2, restart and catch up,
- kill leader 0, monitor elects node 1 term 2, old leader catches up,
- snapshot prefix transfer to a fresh follower,
- `/api/metrics` per node and a QAL cluster dashboard over three nodes.

`/metrics` is reserved by `qal serve`; QAL routes use `/api/metrics`.
