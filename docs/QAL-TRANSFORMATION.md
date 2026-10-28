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
- Packages: `qal_prelude`, `qal_event_queue`, `qal_tensor`, `qal_numerics`.
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
qal:        5,175 files, 147,784 lines, 2,558 tests
rust:       1,393 files, 684,112 lines
python:       727 files, 146,250 lines
typescript:    10 files
packages:      27 .qalpkg
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
