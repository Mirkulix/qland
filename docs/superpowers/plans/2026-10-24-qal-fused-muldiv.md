# Fused multiply-divide — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** An `Int` expression `(e1 * e2 * … * en) / d` (a division whose left operand is a chain of one or more `*`) is evaluated as one exact operation. The product is computed in 128 bits and divided with truncation toward zero, like `/` today. It traps (int_overflow) only if the product does not fit in 128 bits or the quotient does not fit in 64 bits. Division by zero traps as today. `%` is unchanged.

**Why:** In the pressure run of the living program (`bench/living/results-pressure-2026-10-24/report.md`), `price * qty * 95 / 100` overflowed for amounts of 9 digits. Warnings let it through in 1 of 2 runs, because an agent can ignore any warning. Rust code using `u128` had no such bug. This plan removes the bug class for the common business form "multiply, then scale down".

**Architecture:** the code generator recognises the pattern in the typed AST (`TExpr` `Bin(Div, l, r)` with `l` a `Bin(Mul, …)` chain, all `Int`). It evaluates the chain's operands and the divisor left to right, in source order, and calls a new runtime helper `Rt::MulDiv(n)` (n = number of factors). The helper multiplies in 128 bits (two `i64` limbs, from 32-bit partial products), checks that the product fits, divides by the 64-bit divisor (sign handled separately, truncation toward zero), and checks that the quotient fits in `i64`. Both compilers emit byte-identical modules: the Rust generator (`crates/qal-wasm/src/gen.rs`, `rt.rs`) and the self-hosted one (`selfhost/src/qc_wasm.qal`, `qc_rt.qal`). The prover (`crates/qal-prove/src/vc.rs`) models the fused form exactly: no trap condition on the intermediate products, only the 128-bit product bound and the quotient bound.

**Spec of the operation:** for factors `f1..fn` (n ≥ 2, or n = 1 with `f1 * f2` counting as the chain `f1, f2`; a single factor `x / d` is plain `/`) and divisor `d`: the mathematical product P = f1·…·fn; trap `int_overflow` if |P| ≥ 2^127; trap `div_by_zero` if d = 0; Q = trunc(P / d); trap `int_overflow` if Q ∉ [−2^63, 2^63−1]; the result is Q. The trap's site and `values` are those of the `/` (left: P is not representable, report `left` as the first factor and `right` as d; keep the existing site format).

**Global constraints:** zero warnings; the bootstrap fixpoint holds (stage 0 == stage 1 == stage 2, the existing tests in `qal/selfhost/tools/gen/tests` and `qal/tests` check it); the seed `qal/selfhost/qalc.wasm` and `.sha256` are rebuilt with `cargo run --release -p qal-cli --features reference -- selfhost bootstrap --out selfhost/qalc.wasm`; the corpus is re-extracted (`python3 qal/selfhost/tools/extract_corpus.py`, pins in `qal/selfhost/tools/gen/tests/corpus.rs`); never `git add -A`.

## Tasks

### Task 1 — Rust code generator and runtime
- [ ] Failing tests in `crates/qal-wasm` (or `qal-cli/tests/cli.rs`, whichever holds end-to-end arithmetic tests): `999999999 * 999999999 * 95 / 100 == 949999998100000000`; `-999999999 * 999999999 * 95 / 100 == -949999998100000000` (truncation toward zero); `i64::MAX * 2 / 2 == i64::MAX`; `i64::MAX * 4 / 2` traps int_overflow (quotient too large); `x * y / 0` traps div_by_zero; a plain `a * b` that overflows still traps; `a * b % c` unchanged; evaluation order: `f() * g() / h()` with effects prints in order f, g, h.
- [ ] Implement the pattern match in `gen.rs` (only when every operand is `Int`) and `Rt::MulDiv(n)` in `rt.rs`.
- [ ] Commit.

### Task 2 — Prover
- [ ] Failing test in `crates/qal-prove/src/tests.rs`: `fn total(price: Int, qty: Int) -> Int req price >= 0 && price <= 999999999 && qty >= 0 && qty <= 999999999 ens r >= 0 = price * qty * 95 / 100` is proven, with no trap obligation left. `fn big(a: Int) -> Int req a >= 0 ens r >= 0 = a * a * a * a * a / 1` is unknown or refuted on the 128-bit product bound, never proven.
- [ ] Model the fused form in `vc.rs`.
- [ ] Commit.

### Task 3 — Self-hosted compiler and bootstrap
- [ ] Mirror Task 1 in `selfhost/src/qc_wasm.qal` and `qc_rt.qal`. Rebuild `qalc.qal` (concatenation of the modules: see `selfhost/README` or the build step in the tests). Rebuild the seed and verify the fixpoint and the stage-0/stage-1 module equality tests.
- [ ] Commit.

### Task 4 — Reference and property-test warnings
- [ ] Reference line (`crates/qal-cli/src/reference.md`, Types line on Int): "`a * b * … / c` is computed exactly (128-bit intermediate): it traps only if the result does not fit." Raise the pin in `tests/mcp.rs`.
- [ ] E0805 (`unbounded_product`, `ops.rs`) is not emitted for a function whose products all sit in a fused `… / d` (it can only trap when the result is out of range).
- [ ] Re-extract the corpus, full test suite, commit.
