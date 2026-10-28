# QAL checker: split CHECK_FILE via FORK/JOIN (R3 → R0)

**Spec:** `docs/superpowers/specs/2026-10-16-qal-phase5-design.md` §4.3 (Teilen über Fork/Join), §14.2 (Rückfall-Leiter).

## Measured state (2026-10-20)

- `qal/selfhost/tools/gen/tests/full_fixpoint.rs` (ignored) runs parse + check + codegen of `selfhost/qalc.qal` through the QAL backend.
- `Qal.parse` succeeds and equals the Rust parse.
- `Qal.check` fails: `E0951` wrapping `out_of_memory (E0504)`, limit 1 GiB per execution, work item `main`.
- Roadmap row 5 is ◐ at level R3 ("Self-Hosting ohne Prüfer"): QAL frontend and QAL codegen, Rust checker, mixed fixpoint holds (`mixed_bootstrap_fixpoint`).

## Why it runs out of memory

`qc_check_stored` (`selfhost/src/qc_check.qal`, near the end) builds the globals once (`qc_build_globals`), then `qc_check_defs_loop` checks every definition in one execution. QAL has no GC (bump allocator), so the inference garbage of all ≈ 2 000 definitions accumulates. Then monomorphisation runs in the same execution.

## Design

1. **Phase A (main item):** parse, `qc_build_globals`. On diagnostics, return them. Otherwise answer `Fork` with
   - `continuation` = encoded globals plus the definition count;
   - `items` = one `QALQ` request per chunk of definitions (chunk size chosen so a chunk stays well below 1 GiB; start with 64 and measure). Each item carries the encoded globals and the definition range.
2. **Phase B (per item, fresh execution):** decode the globals, run `qc_check_defs_loop` on the range only, and return the typed functions, generics, tests and diagnostics as a `QALT` fragment.
3. **Join:** `Request::Join { continuation, results }` concatenates the fragments in item order, which gives the same order as today. If any diagnostics exist, return them (same order as the single-pass loop). Otherwise run polymorphic-recursion check → monomorphise → validate. If mono itself exceeds the limit, fork it the same way by root groups.
4. **Host:** nothing to do. `init_qal_backend` (`qal/crates/qal-run/src/selfhost.rs`) installs a driver that runs every request through `drive`, which already follows `Fork` rounds and sends `Join`. `Qal.check` only ever sees the final answer. The whole change is on the QAL side.

## Needed pieces

- Encoder/decoder for `QcGlobals` in QAL (types, signatures, environment), mirrored by a Rust codec only if the host must inspect it (it should not; the payload is opaque to the host).
- Determinism: the fragments must concatenate to exactly the single-pass result. Diagnostics order must match the Rust reference (differential tests in `checker_diff.rs`).

## Tests

- `checker_diff.rs` stays green (small programs never fork, or fork into one item).
- A new differential test that forces a tiny chunk size (for example 2) on the corpus and compares with the single-pass result.
- Un-ignore `full_fixpoint.rs`: Stage 1 == Stage 2 == Stage 3 WASM.
- Record wall time of the self-translation (criterion 5: ≤ 120 s, R1 allows up to 600 s with measurements).

## Process

QAL compiler sources change through agents over MCP, with task cards and hidden tests, as in the 5b–5d ledgers. Full verification (tests, 0 warnings, clippy restriction) before pushing to `Mac`. Roadmap row 5 becomes ☑ only when the un-ignored fixpoint test passes.
