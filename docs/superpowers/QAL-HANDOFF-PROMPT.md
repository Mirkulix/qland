# Handoff prompt: continue the QAL project

Copy everything below the line into a new AI session (Claude Code or a comparable coding agent) that has access to this repository.

---

You are taking over an existing software project. Read this whole prompt before you start, and check every claim against the repository (`git log`, the files named here). The repository is the source of truth, not this text.

## The project

**QAL (QLANG Agent Language)** is a programming language designed for AI agents, not for humans. The owner's thesis is that languages built for humans make agents less effective, and that an AI-native language makes agents more effective. Human readability is explicitly not a goal. The owner rejects transpiling to existing languages ("we want something new") but accepted a new concept on existing foundations: WebAssembly/wasmtime, SQLite and Z3.

Core ideas:
- **Graph-native, content-addressed code.** Definitions are stored in a SQLite store (`.qal/store.db`), not in files. Each definition has a SHA-256 hash of its canonical form. Renaming changes no hash, and equal logic gives equal hashes. `define` replaces a definition and relinks its dependents automatically.
- **Agents work through operations**, exposed as an MCP server (`qal mcp`) with tools such as `qal_define`, `qal_test`, `qal_run`, `qal_call`, `qal_view`, `qal_query`, `qal_rename`, `qal_delete`, `qal_history`, `qal_build`, `qal_prove` and `qal_request`, plus the team tools. All results and diagnostics are JSON.
- **Effects are capabilities.** `!io`, `!fs`, `!net`, `!db`, `!time` and `!rand` must be declared and granted (`--allow`).
- **Contracts** (`req`/`ens`) are checked at run time and proven statically with Z3 where possible. A refutation counts only with a counterexample confirmed by running the code. The prover must never report `proven` for something that can fail.
- **Direct compilation to WebAssembly**, run in wasmtime with fuel, memory and time limits.
- **Rust is only the bootstrap.** The end goal is self-hosting: a QAL compiler written in QAL.

## Where things are

- Repository: `/Users/aleksandarbarisic/Workspace/qland` (GitHub `Mirkulix/qland`), working branch **`Mac`**, PR #2 against `main`.
- Rust workspace: `qal/`. Crates: `qal-core` (AST, typed form, canonical form/hashing, diagnostics, QALT), `qal-text` (lexer/parser/printer), `qal-check` (type inference, effects, generics/monomorphisation, exhaustiveness), `qal-wasm` (codegen and runtime in WASM), `qal-run` (wasmtime host, effects, limits, `selfhost::drive`), `qal-store` (SQLite store, packages, team mode), `qal-cli` (CLI and MCP server), `qal-prove` (Z3 proofs and property tests), `qal-serve` (HTTP service runtime) and `qal-compiler` (the compiler facade every host crate must use).
- Standard library, written by agents over MCP: `qal/std/std.qalpkg` (modules `std.str`, `std.url`, `std.json`, `std.http`, `std.route`, `std.db`; 227 definitions). It is embedded in the binary and auto-imported into projects.
- Example backend: `qal/examples/todo/` (Todo REST API written by an agent, `todo.qalpkg`, scenarios).
- Benchmark (phase 1e): `bench/` (`report.md`, `discussion.md`).
- Specs: `docs/superpowers/specs/`. The overall design is `2026-10-15-qal-design.md`. The **roadmap with the status of every phase** is `2026-10-15-qal-roadmap.md`. Phase specs: `…-phase2-design.md`, `…-phase3-design.md`, `2026-10-16-qal-phase4-design.md` and `2026-10-16-qal-phase5-design.md`.
- Plans: `docs/superpowers/plans/`.
- Local, git-ignored working notes: `.superpowers/sdd/` (ledgers, reviews, reports per phase; see `phase4/progress.md` and `phase5/progress.md`), and the local stores `qal/std/.qal` and `qal/examples/todo/.qal`.

## Status (check it with `git log` and the roadmap)

Done, reviewed and pushed to `Mac` (head `08928a82` at handoff, 548 tests passing, 0 warnings):
- **Phase 1** (1a–1e): core pipeline, data types, memory and effects, the graph store, the MCP server, and the benchmark.
  - The benchmark compared QAL via MCP with Rust via cargo, 20 tasks × 2 runs. QAL was worse at first (39/40 solved, 69.5k tokens per solution, against 40/40 and 60.1k). The causes found were missing closures, sequences that could not discard values, and built-ins that were not values. All three were fixed in Phase 2.
- **Phase 2:** generics (monomorphisation), closures, `Map`/`Bytes`, Z3 proofs, property tests.
- **Phase 3:** `net`/`db` effects with security hardening (SSRF, SQL literals, HTTP smuggling, fs races), `qal serve`, the std library (written by agents, all hidden tests passed on the first attempt), the Todo API (47/47 hidden scenarios) and the HTTP harness.
- **Phase 4a:** team identity, roles, Ed25519-signed change records, `qal verify`, store schema 6, `.qal/team-log.db`.
- **Phase 4b:** change sets, leases, stale-base check E0904, `todo()` stubs, multi-process concurrency tests.
- **Phase 5a:** self-hosting foundation (`qal-compiler` facade, typed form QALT, QALQ/QALR protocol, new built-ins, measurement gate), and a stricter prover rule (`qal-prove/4`). Any user-function or closure call in a contract, any indirect call and any depth cut make the result `unknown`.

In progress:
- **Phase 4c** (review, merge, rebase in the team) is on branch **`p4c`** (pushed; worktree `.worktrees/p4c`).
  - Most of it works.
  - The last commit `ce9b7fc2` is a `wip(4c)` commit (keeping conflict marks across rebases) and is **not compiled or tested**.
  - A full workspace test run is missing.
  - Status and exact next steps: `.superpowers/sdd/phase4/4c-report.md`.

Not started:
- **4d:** a demo in which a team of 6 agents builds the service `shop`, measured against a solo agent.
- **5b–5e:** the compiler in QAL (frontend, codegen, checker, bootstrap fixpoint stage 2 == stage 3). The 5a measurement gate says work must be split per definition and function, because a whole-program check would need about 0.4–1.7 GB against a 1 GiB limit. Spec §14 has a fallback ladder.

## How the work was done (keep doing it this way)

- **Per sub-phase:** spec → implementation in a separate git worktree on its own branch → full verification → independent whole-branch review → fix round → scoped re-review → fast-forward `Mac` → push.
  - Full verification means `cargo test --offline --workspace --no-fail-fast` all green, with the total checked yourself, `cargo build --workspace --all-targets` with 0 warnings, and this clippy restriction command clean:
    `cargo clippy --offline --workspace --lib --bins -- -A clippy::all -D clippy::unwrap_used -D clippy::expect_used -D clippy::panic -D clippy::unreachable -D clippy::todo -D clippy::unimplemented`
- **QAL code (std, examples, the later compiler) is written by agents through the MCP server only,** never by hand.
  - Each agent gets a task card with public tests.
  - Hidden acceptance tests are kept back and run by the controller on a throwaway copy of the store.
  - Provenance is checked: every definition's history agent must be an agent name, never `cli`.
- **Rules that hold everywhere:**
  - No panic/unwrap/expect on any CLI or MCP path.
  - Existing hashes never change.
  - Non-team stores behave as before.
  - New diagnostic codes go into `diag::KNOWN`.
  - New QAL test sources need `extract_corpus.py` to be re-run (a corpus guard test checks this).
  - New dependencies must be pinned exactly and must be at least two weeks old. Offline builds.
- **Store schema numbers:** 3 = proofs, 4 = meta, 5 = delete, 6 = Phase 4 team tables.
- **Known load-sensitive tests:** a few timing tests can fail under heavy parallel builds. Re-run them in isolation before you treat a failure as real.

## Rules from the owner (binding)

- The owner delegates design decisions ("decide yourself"). Make a decision, justify it briefly and record it in the phase ledger. Do not ask multiple-choice questions.
- The owner paused the work and will say when to continue. **Do not resume until the owner says so.**
- **Usage limit:** stop all work and push everything when the weekly usage limit reaches **65 %**. It was 54 % at handoff.
- **Models:** use the cheaper model (Sonnet) for implementation and re-reviews, and the strongest model (Opus) only for one final review per sub-phase.
- Never touch or commit the owner's uncommitted `qlang/web/index.html`. The untracked `.codex/` and `AGENTS.md` are not part of this work. Stage explicit paths only, never `git add -A`. No force-push.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. PR descriptions end with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- The PR bot `ecc-tools` keeps asking for "Checks: read and write" permission. That is the owner's account decision. Do not change it, and do not add scanner workflows unasked.

## Next steps once the owner says "weiter" (continue)

1. In `.worktrees/p4c`: compile and test `ce9b7fc2`, fix it, run the full verification, and reword the WIP commit.
2. Run an independent review of 4c, then a fix round, then a re-review. Fast-forward `Mac` and push.
3. Phase 4d (demo and measurement, set up like `bench/run.py`). Then mark roadmap row 4 done with evidence.
4. Phases 5b → 5c → 5d → 5e as in spec §15, built with the team procedure from Phase 4, with differential tests against the Rust pipeline. Before 5b, add a decoded-size budget for compiler responses (parked item M-A in `phase5/progress.md`).
5. Check the usage limit after every sub-phase. At 65 %, stop and push.
