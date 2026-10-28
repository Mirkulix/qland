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

## Contract changes

A change that deletes or weakens a `req`/`ens` is **not** a normal edit:

1. it needs a new or updated test that fails without the contract,
2. it needs a `qal prove` result showing what remains proven,
3. it needs an explicit review note (which obligation was removed and why),
4. it must not be merged while every existing test still passes only because
   the obligation disappeared.

The 134-mutant study (`bench/mutation/expanded.md`) found that test-only
validation cannot catch contract deletion: removing a contract removes an
obligation, not a behavior. Proof/diff review is mandatory for that class.

## Contract diff gate

`tools/contract_diff.py` implements the rule:

```sh
python3 tools/contract_diff.py --base origin/Mac      # changed *.qal files
python3 tools/contract_diff.py --before old.qal --after new.qal
```

It exits 1 when a `req`/`ens` line is deleted or a comparison is weakened
(`>=`→`>`, `<=`→`<`, `==`→`!=`). `--allow-contract-change` still reports the
finding but returns 0 for an explicitly acknowledged change.

Fixture evidence (`tools/fixtures/`):

- identical after → exit 0,
- removed `req a >= 0` → exit 1,
- weakened `ens r >= a` → `ens r > a` → exit 1.

The gate is deliberately not a proof: it forces a human/agent to add a test,
a proof result and a review note before the contract disappears.
