# Phase 6b — Erfahrungsgedächtnis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The store remembers corrections (a failed `define` followed by a successful `define` of the same name), answers `qal_recall`, and puts the best-known correction into the hint of a diagnostic whose code has corrections.

**Architecture:** Two new tables in `.qal/store.db` (schema version 7): `attempts` (failed definitions, written on a failed `define`, bounded) and `corrections` (code, path, before, after, hashes, agent, time), written on the next successful `define` of an attempted name. A `memory` module in `qal-store` owns both; `Store::recall` reads them; `define_with` records attempts and corrections and decorates diagnostics with the memory's hint. The MCP server gets `qal_recall`; the CLI gets `qal recall` and `qal corrections import`; a bench tool extracts the seed corpus from the benchmark transcripts.

**Tech Stack:** Rust (rusqlite, serde), Python (bench tool). No new crates.

**Spec:** `docs/superpowers/specs/2026-10-23-qal-weg-design.md` §5 "Phase 6b — Erfahrungsgedächtnis".

## Global Constraints

- Zero compiler warnings, zero panics (CLAUDE.md). Never `git add -A`.
- The store's write path stays all-or-nothing for definitions: recording attempts and corrections happens in its own short transaction and never changes the result of a `define` (a failed memory write is ignored, like a failed proof save: `try_save_proofs`).
- Every reply is bounded: `recall` returns at most 5 corrections and at most 20 definitions; a `before`/`after` excerpt is at most 600 bytes (cut at a line boundary, `…` appended).
- The reference (`crates/qal-cli/src/reference.md`) gains at most one line; the pin in `tests/mcp.rs` (`<= 75 lines, <= 9973 bytes`) is raised to the new size in the same commit.
- The tool lists in `tests/mcp.rs:189` and `tests/team.rs:297-298` name every tool exactly; `qal_recall` is a read tool (team: `reads`).
- Work in the worktree, build with `CARGO_TARGET_DIR` outside the repository (the living-program experiment copies `qal/target/release/qal` of the main checkout and must not see this work until it is merged).

## Review Focus

1. A diagnostic whose `def` is empty (parse errors, store errors) records no attempt and gets no memory hint: no panic, no empty-name rows.
2. A `define` of several names where only one had a failed attempt records exactly one correction.
3. Two agents in one store: attempts are matched by name and agent, so one agent's failure never becomes another agent's correction.
4. An attempt older than the retention window (2 hours) or beyond the cap (last 20 per name) is dropped, never matched.
5. `recall` with no filter is refused (E0701 `bad_arguments`), with an unknown code returns empty lists, not an error.

---

### Task 1: Schema version 7 — `attempts` and `corrections`

**Files:**
- Modify: `qal/crates/qal-store/src/schema.rs` (SCHEMA_VERSION 7, tables, `upgrade` 6 -> 7)
- Create: `qal/crates/qal-store/src/memory.rs` (row types, insert/select helpers)
- Modify: `qal/crates/qal-store/src/lib.rs` (`mod memory;`)
- Test: `qal/crates/qal-store/src/tests.rs`

**Interfaces:**
- Produces: `memory::Attempt { name, code, path, message, text, agent, at }`, `memory::Correction { id, code, path, message, before, after, after_hash, agent, at }`,
  `memory::record_attempts(tx, diags: &[Diagnostic], input: &Program, src: &str, agent: &str) -> Result<(), Vec<Diagnostic>>` (one row per diagnostic with a non-empty `def` that names a definition of `input`; `text` = that definition printed with `qal_compiler::print_program` of a one-definition program, else the raw `src` when `input` has no such definition; keeps the last 20 attempts per name, drops attempts older than 7200 s),
  `memory::record_corrections(tx, defined: &[Defined], after_text: &dyn Fn(&str) -> Option<String>, agent: &str) -> Result<usize, Vec<Diagnostic>>` (for each `defined` with status `added`/`replaced` that has attempts by `agent`: one correction from the newest attempt, `after` = `after_text(name)`; deletes that name's attempts of `agent`),
  `memory::recall(tx, q: &RecallQuery) -> Result<Vec<Correction>, Vec<Diagnostic>>` (newest first, identical (before, after) pairs collapsed with a `count`, at most 5),
  `memory::hint_for(tx, code: &str) -> Result<Option<String>, Vec<Diagnostic>>` (the most frequent correction of `code`: `"earlier fix of <code> (<n>x): <first differing line of before> -> <first differing line of after>"`, each side cut to 120 chars).

- [ ] **Step 1: Write the failing tests** in `tests.rs`:

```rust
#[test]
fn a_failed_define_followed_by_a_successful_one_is_a_correction() {
    let mut s = store();
    err(&mut s, "fn f(n: Int) -> Int = { let x = n + 1; x }");            // E0201 (unknown name after `;`)
    ok(&mut s, "fn f(n: Int) -> Int = let x = n + 1 in x");
    let r = s.recall(&RecallQuery { code: Some("E0201".into()), ..RecallQuery::default() }).unwrap();
    assert_eq!(r.corrections.len(), 1);
    let c = &r.corrections[0];
    assert!(c.before.contains("let x = n + 1;") && c.after.contains("let x = n + 1 in x") && c.count == 1, "{c:?}");
    assert_eq!(c.after_hash, hash(&mut s, "f"));
}

#[test]
fn attempts_match_by_name_and_agent_and_are_bounded() {
    let mut s = store();
    assert!(s.define("fn g() -> Int = h()", "a1").is_err());             // unknown name
    ok_as(&mut s, "fn g() -> Int = 1", "a2");                              // another agent: no correction
    assert!(s.recall(&RecallQuery { code: Some("E0201".into()), ..RecallQuery::default() }).unwrap().corrections.is_empty());
    for i in 0..25 { assert!(s.define(&format!("fn k() -> Int = x{i}"), "a1").is_err()); }
    assert_eq!(count(&s, "attempts"), 20);                                 // last 20 per name
}

#[test]
fn recall_needs_a_filter_and_hints_carry_the_memory() {
    let mut s = store();
    assert_eq!(s.recall(&RecallQuery::default()).unwrap_err()[0].code, "E0701");
    err(&mut s, "fn f(n: Int) -> Int = { let x = n + 1; x }");
    ok(&mut s, "fn f(n: Int) -> Int = let x = n + 1 in x");
    let d = err(&mut s, "fn q(n: Int) -> Int = { let y = n; y }");
    assert!(d[0].hints.iter().any(|h| h.starts_with("earlier fix of E0201 (1x):")), "{d:#?}");
}
```
(`ok_as` is `ok` with an agent argument; `count` exists.) Check the diagnostic code of an unknown local name after `;` in the first test by reading `crates/qal-check/src/infer.rs` (E0201 with the sequence-let hint); adjust the expected code if the checker reports another.

- [ ] **Step 2: Run them**: `cargo test -p qal-store memory_ correction_ attempts_ recall_` — Expected: compile errors (no `RecallQuery`, `recall`).
- [ ] **Step 3: Implement** schema v7 (`CREATE TABLE attempts (id INTEGER PRIMARY KEY, name TEXT NOT NULL, agent TEXT NOT NULL, code TEXT NOT NULL, path TEXT NOT NULL, message TEXT NOT NULL, text TEXT NOT NULL, at INTEGER NOT NULL); CREATE INDEX attempts_by_name ON attempts(name, agent); CREATE TABLE corrections (id INTEGER PRIMARY KEY, code TEXT NOT NULL, path TEXT NOT NULL, message TEXT NOT NULL, before TEXT NOT NULL, after TEXT NOT NULL, after_hash TEXT NOT NULL, agent TEXT NOT NULL, at INTEGER NOT NULL); CREATE INDEX corrections_by_code ON corrections(code);`), `upgrade` 6 -> 7, `memory.rs` with the functions above, `RecallQuery { code: Option<String>, name: Option<String>, type_pattern: Option<String> }` and `Recall { corrections: Vec<Correction>, definitions: Vec<Entry> }` in `ops.rs` (definitions: `query` with the name/type filters when given, at most 20), `Store::recall`. Hook into `lib.rs::define_with`: on `Err(diags)` record attempts (own transaction, errors ignored) and add `hint_for(code)` to each diagnostic (once per code); on `Ok(defined)` record corrections (own transaction, errors ignored). The `after_text` closure prints the stored definition (`ops::text` of `snap.decode_name`).
- [ ] **Step 4: Run** `cargo test -p qal-store` — Expected: all pass, including the 112 existing.
- [ ] **Step 5: Commit** `feat(store): schema 7 — attempts and corrections, recall, diagnostics with memory`.

### Task 2: MCP `qal_recall`, CLI `qal recall`, reference line

**Files:**
- Modify: `qal/crates/qal-cli/src/mcp.rs` (TOOL_NAMES, SPACE_TOOLS, `tool_params` `"qal_recall" => &["code", "name", "type", "change"]`, dispatch, schema entry with title "Recall earlier fixes"), `ops.rs` (`pub fn recall(ctx, q) -> Reply` with `Output.recall: Option<Recall>`), `main.rs` (`qal recall [--code <E...>] [--name <prefix>] [--type <pattern>]`, usage lines), `reference.md` (one line after the Workflow line: "A diagnostic whose code was fixed before carries the fix in its hints; qal_recall {\"code\": \"E0201\"} (or name/type) lists earlier corrections and matching definitions.")
- Test: `qal/crates/qal-cli/tests/mcp.rs` (tool list, pin), `tests/team.rs` (reads list)

**Interfaces:**
- Consumes: `Store::recall(&RecallQuery) -> Result<Recall, Vec<Diagnostic>>` (Task 1).
- Produces: tool `qal_recall` with `inputSchema` properties `code`, `name`, `type` (all strings, none required; the server answers E0701 when all are absent, from the store).

- [ ] **Step 1: Failing test** `recall_answers_with_the_earlier_fix` in `tests/mcp.rs` (through `both`): define a failing source (E0201 as in Task 1), then the fix, then `qal_recall {"code": "E0201"}` has one correction with `count` 1 and `before`/`after`; a second failing define of another name has a hint starting with `earlier fix of E0201`; `qal_recall {}` is an error with E0701. Update the exact tool lists in `tests/mcp.rs:189` and `tests/team.rs:297-298`.
- [ ] **Step 2: Run** `cargo test -p qal-cli --features reference --test mcp recall` — Expected: FAIL (unknown tool).
- [ ] **Step 3: Implement** as listed; raise the reference pin to the new byte count.
- [ ] **Step 4: Run** `cargo test -p qal-cli --features reference --test mcp` and `--test team` — Expected: all pass; `cargo build -p qal-cli --features reference` — zero warnings.
- [ ] **Step 5: Commit** `feat(mcp): qal_recall and the recall command`.

### Task 3: Seed corpus from the benchmark transcripts

**Files:**
- Create: `bench/corrections.py` (`python3 bench/corrections.py <transcript.jsonl>... --out corrections.json`): for each transcript, every failed `qal_define` tool result (isError true with `diagnostics`) followed later in the same transcript by a successful `qal_define` whose `defined` names include the diagnostic's `def`: one record `{code, path, message, before, after, agent}` with `before` = the failing definition's source text (the heredoc of the failing call, cut to the definition starting at `fn <def>`/`type <def>`/`test <def>` until the next definition start) and `after` = the same cut of the successful call's source.
- Create: `bench/test_corrections.py` (a synthetic transcript with one failed and one successful define yields one record with the right `before`/`after`; a failed define never fixed yields none).
- Modify: `qal/crates/qal-cli/src/main.rs` (`qal corrections import <file.json>`: inserts the records as corrections with `after_hash` empty and `agent` `corpus`), `qal/crates/qal-store/src/lib.rs` (`Store::import_corrections(&[CorrectionInput]) -> Result<usize, Vec<Diagnostic>>`).
- Test: `qal/crates/qal-store/src/tests.rs` (`imported_corrections_answer_recall`).

- [ ] **Step 1: Failing tests** (`bench/test_corrections.py`, the store test).
- [ ] **Step 2: Run them** — Expected: FAIL (module/function missing).
- [ ] **Step 3: Implement**; then run the tool over the transcripts of `bench/results.jsonl` and `bench/rerun-2026-10-22/results.jsonl` (agent ids in the records; transcripts under `~/.claude/projects/-Users-aleksandarbarisic-Workspace-qland/*/subagents/` and `bench/analyze.transcript_path`) into `bench/analysis/corrections-2026-10-23.json`; print how many records per code.
- [ ] **Step 4: Run** `python3 bench/test_corrections.py`, `cargo test -p qal-store` — Expected: pass.
- [ ] **Step 5: Commit** `feat(memory): seed corpus of corrections from the benchmark transcripts`.

### Task 4: Acceptance test of the three error classes

**Files:**
- Test: `qal/crates/qal-cli/tests/mcp.rs` (`recall_knows_the_three_error_classes_of_the_token_analysis`)

- [ ] **Step 1:** In a fresh project, import `bench/analysis/corrections-2026-10-23.json` (through `Store::import_corrections`, the file is read by the test), then assert `qal_recall` for `E0201` returns a correction whose `after` contains ` in ` (local name after `;` fixed with `let … in`), for the `String`-vs-`Str` class (find its code in the corpus) a correction whose `after` contains `Str`, and for the built-in-as-value class a correction (code from the corpus) with a closure in `after`. If a class has no record in the corpus, the test states so with `eprintln!` and asserts the other two: the corpus is what it is.
- [ ] **Step 2: Run** — Expected: pass. **Step 3: Commit** `test(memory): the three error classes recall their fixes`.

### Task 5: Documentation

- [ ] Roadmap row 6b (`docs/superpowers/specs/2026-10-15-qal-roadmap.md`): ◐ with what was built; the path design §5 Phase 6b gets a line "Stand 2026-10-23: 1–3 gebaut (Schema 7, qal_recall, Hinweise mit Gedächtnis), Abnahme im Rerun offen".
- [ ] Commit `docs(p6b): memory built, acceptance pending the rerun`.
