# QAL Autonomous Agent Playbook (v1.0)
> **The Official Operating Manual for Large Language Models & Autonomous AI Agents**  
> *Target Architecture: QAL Content-Addressed Provable Substrate & Z3 Verification Engine*

---

## 1. Executive Summary & Core Paradigm Shift

This document defines the operational directives for AI agents interacting with the **QAL (Quantitative Assurance Language)** system.

| Traditional Programming (Python/TS/Rust) | The QAL Autonomous Agent Paradigm |
| :--- | :--- |
| Code is stored in unstructured text files (`.py`, `.ts`). | Code is stored as **semantic, content-addressed records in SQLite** (`.qal/store.db`). |
| Agents edit files by replacing line ranges (fragile, offset-sensitive). | Agents manipulate **individual, typed definitions** (`fn`, `type`, `test`) atomically via MCP tools. |
| Bugs are caught by running integration tests or in production. | Invariants are **proven statically at compile time** by the embedded Z3 SMT theorem prover. |
| Team collaboration creates Git merge conflicts on neighboring lines. | Merges operate on definition level: **zero line-based merge conflicts**, disjoint definitions merge automatically; overlapping leases trigger `E0901 lease_conflict`. |
| Functions have unrestricted access to system calls and disk. | **Zero-Trust Algebraic Effect System**: pure functions cannot touch I/O, disk, or network unless declared (`!io`, `!fs`, `!db`, `!net`). |

---

## 2. The 12 Native MCP Tools

QAL provides exactly **12 native Model Context Protocol (MCP) tools**:

1. **`qal_define`**: Atomically add or replace types, functions, or tests. Triggers Z3 contract prover.
2. **`qal_prove`**: Audit formal proof status of stored functions (`proven`, `refuted`, `unknown`).
3. **`qal_test`**: Run unit tests and execute 100 synthesized property-based fuzzing runs per contract.
4. **`qal_call`**: Execute any stored function in the isolated WASM sandbox with JSON arguments.
5. **`qal_run`**: Execute an application entry point (`fn main(args: List[Str]) -> Int !...`).
6. **`qal_view`**: Read exact definition source code, signature, hash, and direct dependencies (`deps`).
7. **`qal_query`**: Search definition index by name prefix, signature pattern, or dependency usage.
8. **`qal_request`**: Test HTTP service handlers (`fn handle(HttpRequest) -> HttpResponse !...`).
9. **`qal_rename`**: Rename an existing definition without altering its hash or breaking dependents.
10. **`qal_delete`**: Delete unused definitions (fails with `has_dependents` if referenced).
11. **`qal_history`**: Query the immutable audit trail (actions, agent signatures, timestamps, SHA-256 hashes).
12. **`qal_build`**: Compile a definition into a standalone WebAssembly binary artifact.

---

## 3. Cognitive Loop & Self-Healing Protocol

Follow this deterministic loop when authoring or modifying code:

```
[1. Inspect] ──► Call qal_query / qal_view to inspect existing definitions and types.
                     │
[2. Specify] ──► Formulate contracts first: req (preconditions) & ens (postconditions).
                     │
[3. Submit] ───► Call qal_define with definition source and agent ID.
                     │
[4. Diagnose] ─► Evaluate compiler output:
                     ├─ Status "proven": Obligation verified by Z3.
                     ├─ Status "contract_refuted (E0801)":
                     │    Read `counterexample` values in the JSON diagnostic.
                     │    Analyze why the postcondition fails for these values.
                     │    Adjust req or the function body, then resubmit via qal_define.
                     └─ Status "alias_reference (E0404)":
                          A single definition refers to two distinct names that share
                          the identical definition hash. Use only one alias name.
```

---

## 4. QAL Language Reference & Syntax Rules

### Types & Data Structures
* Primitive Types: `Int` (64-bit signed; overflow and division-by-zero trap deterministically), `Float`, `Bool`, `Str`, `Bytes`, `Unit` (`()`).
* Compound Types:
  * Records: `type Account = { id: Int, owner: Str, balance: Int }`
  * Algebraic Data Types (Enums): `type Shape = Circle(Int) | Rect(Int, Int) | Point`
  * Generics: `type Pair[A, B] = { fst: A, snd: B }`
  * Prelude Types: `Option[T] = Some(x) | None`, `Result[T, E] = Ok(x) | Err(e)`.

### Control Flow (Expressions Only)
* Conditions: `if condition then a else b` (`else` is required).
* Pattern Matching: Must be exhaustive.
* Local Bindings: `let var = expr in body`
* Sequencing: `{ statement1; statement2; result_expr }`
* Loops / Recursion: No loops exist. Tail calls are optimized into loops by the WASM compiler. Functions with `ens` do not make tail calls.

### Algebraic Effects
Declare required system capabilities explicitly after the return type:
* `!io`: Terminal I/O (`print`, `read_line`)
* `!fs`: Project filesystem access (`read_file`, `write_file`)
* `!db`: SQLite queries (`db_exec`, `db_query`)
* `!net`: Outgoing HTTP requests (`http`)
* `!time`: Clock access (`now_ms`)
* `!rand`: Random integer generation (`rand_int`)

---

## 5. Diagnostic Codes Reference

| Code | Kind | Cause & Agent Action |
| :--- | :--- | :--- |
| **`E0801`** | `contract_refuted` | Z3 found an input assignment violating `ens`. Read `counterexample` in JSON diagnostic; strengthen `req` or fix edge-cases in the body. |
| **`E0802`** | `contract_unproven` | Contract cannot be proven statically. Simplify to linear integer relationships (`Int`/`Bool`). |
| **`E0501`** | `contract_violation` | Runtime precondition failed in `qal_call`. Ensure arguments satisfy callee's `req`. |
| **`E0502`** | `int_overflow` | 64-bit integer overflow trapped at runtime. Add realistic upper bounds in `req`. |
| **`E0404`** | `alias_reference` | A definition refers to multiple names that share the identical definition hash. Reference only one alias name. |
| **`E0201`** | `unknown_name` | Identifier not found in store or prelude. Use `qal_query` to locate exact names. |
| **`E0301`** | `type_mismatch` | Argument or return type conflict. Inspect signature via `qal_view`. |
| **`E0901`** | `lease_conflict` | Another agent holds an active lease on this definition name. |

---
*Playbook maintained by QAL Autonomous Core. Grounded in formal tests and verified compiler behavior.*
