# QAL Specification (working draft)

Status: **working draft**, 2026-10-28. This document starts the separation of the
language from any one implementation. It is not yet a complete grammar; it is
normative for the principles and for `bench/conformance/`.

## 1. What QAL is

QAL is a deterministic, capability-scoped language for definitions that carry
their own evidence: type, effects, contracts, tests and content hash.

A QAL **definition** is not just a function body. It is:

```
name + type + effects + contracts + body + tests + hash + origin
```

## 2. Normative principles

1. **Determinism.** The same definition, inputs, grants and seed produce the same
   result. No ambient time, randomness or I/O.
2. **Explicit effects.** A definition declares `!io`, `!fs`, `!db`, `!net`,
   `!model`, `!time`, `!rand`; the host grants them explicitly.
3. **Contracts are part of the type.** `req`/`ens` are checked; `qal prove`
   reports `proven`, `refuted` or `unknown`.
4. **Content addressing.** Every stored definition has a SHA-256 hash; renaming
   does not change the hash.
5. **Packages are text.** `.qalpkg` is the compiler's canonical text format; its
   checksum covers everything after the `# package` line.
6. **Agent-readable diagnostics.** Errors are JSON with `code`, `kind`, `def`,
   `path`, `message`, `hints` (and `counterexample` where applicable).
7. **No silent success.** A missing proof, a failed test, a stale review or a
   hash mismatch is an error, not a warning that can be ignored.

## 3. Implementation split

- **Stage 0:** Rust pipeline (`qal-text`, `qal-check`, `qal-wasm`, canon, store).
- **Stage 1+:** the self-hosted QAL compiler (`qal/selfhost/qalc.qal`).
- The bootstrap fixpoint is evidence that the language can describe itself.

The specification, not Rust, is the authority when implementations disagree.

## 4. Conformance

An implementation may call itself QAL only if it passes
`bench/conformance/run.py`:

- valid check/test,
- type mismatch `E0203`,
- failing test,
- proven contract,
- refuted contract with counterexample.

The suite grows with every language change. A bug fix without a conformance case
is incomplete.

## 5. Packages and registry

- `qal export` writes `.qalpkg`;
- `qal import` verifies the header checksum and every definition hash;
- `qal-registry/registry.py` stores packages by SHA-256 and re-verifies before
  serving.

## 6. Versioning

- `qal --version` reports the implementation version.
- The spec has its own version; breaking changes require a migration note and a
  new conformance case.
- Silent semantic changes are forbidden.

## 7. Not yet specified

- full grammar and typing rules,
- float/array/concurrency semantics,
- FFI/host ABI,
- package dependency resolution,
- formal verification of the compiler.

These are open; this draft exists so they can be specified deliberately instead
of accreting in code.
