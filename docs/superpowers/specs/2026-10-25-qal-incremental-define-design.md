# QAL: incremental define — design

**Date:** 2026-10-25
**Status:** implemented, approaches A and C combined (2026-10-26):

- **Compiler side:** the self-hosted compiler has the operation `CHECK_DELTA` (op 11,
  `qc_proto.qal`, `qc_declare` in `qc_check.qal`), with the bootstrap fixpoint re-established.
- **Host side:** the store's cache and assembly live in `qal/crates/qal-store/src/incremental.rs`.
- **Still open:** approach B.
**Why:** the 1-turn verification loop takes 0.46 s on a 67-function program, but about 2 s at 1520
definitions (`bench/loop/report.md`, sections "Scaling" and "Why the rest needs the self-hosted
compiler"). Of those 2 s, about 1.6 s are spent checking and hashing the **whole** codebase in the
self-hosted compiler on every define.

## Measured facts this design rests on

At 1520 definitions:

- **Two full passes per define.** `CHECK_FILE` (check) and `DEFINE` with empty state (hash) each
  receive the whole program as freshly printed text, 400 KB. Printing costs 78 ms, so it is not
  the bottleneck.
- **Where the check's 1.4 s go:** a single-core main call of 780 ms (parsing and splitting), fork
  rounds of 14 and 27 items (725 + 258 ms on 8 cores), and joins of 230 ms.
- **`DEFINE` with the real store state** (§4.5 of the Phase-5 design) gives identical hashes and
  takes just as long (1.58 s): it re-checks everything.
- **A host-side cache of work items cannot help.** Items hold about 100 definitions with their
  context, so one change alters several of them; the main call is not an item at all.

## Goal

A define costs time in proportion to what it changes (the defined names and their transitive
dependents), not to the size of the codebase. Target: under 0.5 s for a typical define at
1520 definitions. The results must be byte-identical to today's: hashes, canonical forms,
diagnostics, typed form.

## Non-goals

- No change to the store's semantics (replacement rule, relinking, E0401/E0405).
- No second checker in Rust in the release binary: the self-hosted compiler stays the only
  checker.

## Approach (to be decided in review)

**A. Signature environment from the store (preferred).** A new op, `CHECK_DELTA`:

- **Input:** the signatures of every stored definition (types, function signatures with
  contracts' types and effects; already available through `SIGNATURES`) plus the text of the
  definitions to check, which are the input and its transitive dependents.
- **Work:** the checker builds its global environment from the signatures without checking any
  bodies, then checks only the given bodies (top-level definitions have complete signatures, so a
  body depends only on the signatures of its dependencies — the Phase-5 design §4.3 already relies
  on this for its work items).
- **Output:** the typed form of the checked definitions only, plus their canons (the hash part of
  `DEFINE`).
- **Host side:** the typed program is assembled from a cache of per-definition typed forms, kept
  in a store table keyed by definition hash, plus the new ones.

  **Open question:** the whole-program monomorphisation (`qc_mono`). Instances of generic functions
  are shared across the program, so per-definition typed forms must keep generic calls unresolved,
  with monomorphisation as a separate, cheaper pass over the assembled program, or instances must
  be keyed by (generic hash, type arguments).

**B. Persistent server state.** The MCP server is long-lived in real use, so the checked program
could stay in memory between defines, with the compiler receiving only deltas. This is cheaper to
build, but does nothing for the CLI or for the benchmark harness, which starts a server per call,
and it puts state into a protocol that is stateless today.

**C. Reuse the last define's results on the host (implemented).** No change to the self-hosted
compiler, its protocol or the bootstrap:

- After a define, the store keeps its checked and hashed program in `.qal/typed.cache`. The key is
  a fingerprint of the codebase (names, hashes, local names) and the compiler's module hash.
- **Check:** a define of the same codebase checks a reduced program. The changed definitions (the
  input and its transitive dependents), the generic functions and the definitions that call an
  instance stay in full; every other function becomes its signature with body `todo()`, every
  other test `true`. The typed forms of the unchanged definitions are taken from the cache with
  their calls renumbered: by name for functions, by generic name and type arguments for
  instances. This answers the open question above: instances never have to be keyed, because what
  produces them is checked again.
- **Hash:** a definition's hash depends only on what it refers to. Only the changed definitions
  and their transitive dependencies are hashed. The dependencies of the input are over-approximated
  by every identifier in its text that names a stored definition, plus the type of every variant it
  names. The canons of the rest come from the cache.
- **Fallback:** anything uncertain falls back to the full path: no cache for this codebase, a
  diagnostic, a name or instance missing on either side, a changed signature of a kept function.
  `QAL_INCREMENTAL=off` always takes the full path; `QAL_INCREMENTAL=verify` takes both and reports
  whether they agree.

**As built, A + C:** `CHECK_DELTA` takes three things:

- the text;
- the names written now;
- the names of the functions that count only by their signature (*declared*).

**How a declared function is checked.** The compiler reduces it to its signature:

- its parameters are renamed `qc_p0`, `qc_p1`, ...;
- its contracts are dropped;
- its body becomes `todo()`.

The Rust reference does the same (`FnDef::declared`). Their bodies are neither resolved nor
checked, so a parameter named `todo` can no longer shadow anything.

**What the host sends:**

- every type;
- the changed definitions;
- the generics and the callers of instances;
- the functions those refer to, declared.

**What the host does with the result:**

- **Assembly:** it assembles the full typed program from the result and the cache.
- **Instances:** these are not keyed. Whatever produces them is checked again, which answers the
  open question of approach A.

**Measured:**

- **Bootstrap:** stage 1 = stage 2 = stage 3, seed `1a86ae72…`.
- **Seed and reference:** `check_delta_of_the_seed_equals_the_reference` (`qal-run/tests/selfhost.rs`).
- **Differential:** 318 stores, 2424 defines. 1788 incremental checks and 1788 incremental hashes,
  all identical to the full path.

## Verification

- **Differential:** for every corpus program and 10 000 generated programs, an incremental define
  after a random edit gives the same hashes, canons, diagnostics and typed form as today's full
  path. Today's path stays available as the oracle (`QAL_DEFINE=full`).
- **Bootstrap fixpoint:** stage 1 == stage 2 == stage 3, byte-identical, with a rebuilt seed.
- **Measurement:** `bench/loop/scale.py` (1520 definitions) and `sweep.py` (67 functions) before
  and after.

## Size and risk

- **Size:** `qc_check` (2135 lines QAL), `qc_mono` (803), `qc_tform` (1338) and `qc_proto` (679)
  are affected; on the host side `qal-store/src/define.rs` and the protocol.
- **Estimate:** several days.
- **Main risk:** a divergence in hashes. The differential test against the full path is the gate
  before anything is switched on.
