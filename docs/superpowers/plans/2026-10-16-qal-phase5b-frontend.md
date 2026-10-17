# Implementation Plan: QAL Phase 5b — Frontend & Identity in QAL

**Goal:** Implement the self-hosted QAL compiler frontend and canonical identity modules (`qc_ast`, `qc_lex`, `qc_parse`, `qc_print`, `qc_canon`, `qc_proto`, `qc_main`) in QAL, wire `Backend::Qal` and `Backend::Both` in `qal-compiler`, and verify byte-identical results against the Rust reference across the entire corpus and 10,000 generated programs.

---

### Step 1: Rust Reference Server Completion for Phase 5b Ops
- In `qal-compiler/src/proto.rs`:
  - Implement `Request::Define` in `serve_request`:
    - Parses input text.
    - Decodes stored state into `Program` using `program_of`.
    - Combines definitions.
    - Checks combined program with `qal_check::check_stored`.
    - Computes hashes and canonical forms with `qal_core::canon::hash_program`.
    - Produces `Prepared` with `typed: QALT`, `canons`, `groups`, and `sig_old`.
    - Returns `Response::Done(encode_prepared(&prepared))`.
  - Verify `serve` handles `PARSE`, `PRINT`, `SIGNATURES`, and `DEFINE`.

---

### Step 2: Implementation of `qalc` in QAL (`qal/selfhost/qalc.qal`)
- Author `qal/selfhost/qalc.qal` (and export as `qal/selfhost/qalc.qalpkg`):
  1. `qc_ast`: AST types and helpers.
  2. `qc_lex`: Bytes-based lexer using `bytes_at`, `char_code`, handling numbers, strings, symbols, keywords, `std.<module>.<name>`.
  3. `qc_parse`: Recursive-descent parser bounded by `MAX_DEPTH = 256`.
  4. `qc_print`: Canonical printer producing source text.
  5. `qc_canon`: Canonical LEB128 encoding, De Bruijn indices, reference resolution, group labelling, `sha256` hashing, signature text, and canonical decoding.
  6. `qc_proto`: Decoding `QALQ` requests (ops 1, 2, 3, 8), encoding `QALR` responses.
  7. `qc_main`: Pure entry point `fn qc_main(req: Bytes) -> Bytes`.

---

### Step 3: Wire `Backend::Qal` and `Backend::Both` in `qal-compiler`
- In `qal-compiler/src/lib.rs`:
  - Provide a driver hook `pub type QalDriver = Box<dyn Fn(&[u8]) -> Result<Vec<u8>, Diagnostic> + Send + Sync>;`.
  - Implement `struct Qal` implementing `Backend`:
    - `parse`, `print`, `print_expr`, `hash` call the QAL compiler module via `driver`.
    - `check`, `compile` fall back to Rust (until 5c/5d).
  - Implement `struct Both` implementing `Backend`:
    - Calls both Rust and QAL, compares results byte-for-byte.
    - On mismatch, returns `Diagnostic::compiler_divergence` (E0950).
  - Update `select()` to return `Rust`, `Qal`, or `Both`.
- Wire `qal-run` to supply the driver running `qalc.wasm` (compiled from `qalc.qal` via `CompileOptions::compiler()`).
- In `qal-cli`:
  - Support `--compiler rust|qal|both` and `QAL_COMPILER`.

---

### Step 4: Verification & Test Harness
- Add `qal/selfhost/tools/gen/tests/frontend_diff.rs`:
  - Run differential tests across all 783 corpus programs for:
    - Text parsing (`PARSE`)
    - Canonical printing (`PRINT`)
    - Canonical hashing & forms (`hash`, `SIGNATURES`, `DEFINE`)
    - Store decoding
  - Run 10,000 generated programs with 0 divergence.
  - Run lint test checking no non-tail recursion over collections.
- Pass full workspace tests (`cargo test --offline --workspace`).
- Pass strict clippy restrictions.
- Commit and prepare for Phase 5c.
