# QAL-First Policy

The language becomes self-sustaining only if it is used to build itself and its
ecosystem. This policy makes that concrete.

## Rule

**New internal capabilities are built in QAL first.** A host language is allowed
only where QAL cannot run:

- UI/frontend (TypeScript/HTML),
- ML training/numerics (Python/GPU),
- hardware/drivers (Rust/C),
- the WASM runtime and OS adapters (Rust).

Everything else — tools, services, workflows, benchmarks, package tooling,
diagnostics — is QAL.

## Exceptions

An exception must be recorded in the change description with:

1. why QAL cannot do it,
2. what the host boundary is,
3. which QAL contract covers the boundary,
4. a follow-up to move it into QAL if possible.

## Evidence

Every service or tool added under this policy must have:

- declared effects,
- `req`/`ens` where the logic is not trivial,
- tests and at least one property test,
- a `.qalpkg` when reusable,
- a conformance case if it changes language behavior.

## Language changes

A language change needs:

1. a written problem from real usage,
2. a conformance case,
3. a benchmark or migration note,
4. approval in the spec.

No feature without a use case. No use case without a test.

## Registry

Reusable verified definitions are published through `qal-registry/registry.py`
and imported by hash. Copy-paste is not composition.
