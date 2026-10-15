# QAL Phase 1b — Data Types, Memory and Effects Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Phase 1a pipeline to the full Phase 1 language: `Float`, `Str`, `List[T]`, `Option`/`Result`, records and enums with exhaustively checked `match`, sequences `{ a; b }`, the built-in functions of spec §3.5, a bump allocator in linear memory, effects as capability-gated host imports, and `qal run` for `main(args: List[Str]) -> Int` with `--allow io,fs:./data`.

**Architecture:** The checker now returns a typed, name-resolved program (`qal_core::tast::TProgram`): every expression carries its inferred type (local inference by unification), locals are numbered, calls are resolved to functions, built-ins, variants or fn values. Code generation consumes only that typed program, compiles per entry point (only reachable code), computes all function indices in a module builder, and lowers built-ins to a runtime library written directly in WebAssembly. The runtime links only the host-function groups the caller granted and reports every failure in the diagnostic format, including the values involved.

**Tech Stack:** Rust 1.97 (edition 2021), `wasm-encoder =0.259.0`, `wasmparser =0.259.0` (tests only), `wasmtime =49.0.1` (cranelift, tail calls, fuel, store limits), `serde`/`serde_json` 1. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-15-qal-design.md` (§3.1–3.6, §4, §7, §8). Previous sub-plan: `docs/superpowers/plans/2026-10-15-qal-phase1a-core-pipeline.md`.

**Sub-plan context:** Phase 1 is delivered as five sub-plans: 1a core pipeline (done), **1b data types + memory + effects (this plan)**, 1c canonical form + hashing + SQLite store, 1d MCP server + full CLI operations, 1e benchmark.

**How this plan was made:** every file below was first implemented and tested in a scratch copy of the repository (121 tests passing, zero compiler warnings) and then reviewed by an independent reviewer whose confirmed findings (symlink escape in `fs`, i32 overflow in `join`, reserved export names, fn values inside compared records, JSON field order, non-UTF-8 arguments, NaN identity) are fixed and pinned by tests; the code blocks in this plan were generated from those verified files.

## Global Constraints

- Workspace lives at `qland/qal/`. Dependency versions stay exactly as in 1a; no dependency is added.
- Code generation emits WebAssembly directly; never generate source code of another language.
- `Int` is 64-bit; overflow, `MIN / -1` and `-MIN` are `int_overflow`, never silent wrap-around. Division/remainder by zero is `div_by_zero`. `Float` is IEEE 754 binary64 without traps (`1.0 / 0.0` is infinity); `%` is Int-only.
- Effects: `io`, `fs`, `time`, `rand`. Rule: effects(callee) ⊆ effects(caller); calling a fn value needs the effects in its type; `map`/`filter`/`fold` need the effects of their function argument. Contracts and tests are effect-free.
- The generated module imports `qal_host.fail` plus only the host functions its reachable code calls (spec §8: "nur die Effekte des Einstiegspunkts" — this is stricter). The runtime refuses to start a function whose declared effects were not granted (`permission_denied`).
- Memory: bump allocator, no freeing, at most 256 MiB (4096 pages) per execution; exceeding it is `out_of_memory`.
- Inside `ens`, the function result is named `r` (`qal_core::ast::RESULT_VAR`); a parameter named `r` in a function with `ens` is an error.
- Entry point: `fn main(args: List[Str]) -> Int` (any effects); its result is the exit code.
- Diagnostic JSON fields: `code, kind, def, path, message, expected?, found?, hints`. Runtime errors (spec §7) use `code, kind, def, path, message, contract?, values?, hints`. All messages are English. `path` addresses AST nodes; parse errors use `line:col`.
- Checked programs only: nothing is compiled while any diagnostic exists.
- Builds and tests must be warning-free (`cargo build --workspace --all-targets`).

## Design Decisions

Decided here (the user delegates design decisions); each is reflected in the code below.

1. **Typed AST between checker and codegen.** `qal_check::check` returns `TProgram`; codegen never re-infers types. One source of type truth; codegen becomes pure lowering.
2. **Local inference by unification.** `None`, `[]`, `Ok(x)` get type variables that later uses fix; variables still unknown at the end default to `Unit` (their representation never matters). Optional annotation `let x: T = v in b` for the rare ambiguous case.
3. **Name case is syntax.** Types and variants start uppercase, functions/locals/fields/effects lowercase. This makes `Name { ... }` (record literal) and `Name(...)` (variant) unambiguous for the parser and resolution trivial. As in Rust, a bare record literal is not allowed directly in a `match` scrutinee.
4. **Uniform 8-byte slots** for list elements, record fields and variant payloads (i64 as is, f64 bit pattern, i32 zero-extended). Each list built-in is one runtime routine instead of one per element type; only `map`/`filter`/`fold` are specialised per machine representation (at most 3×3 variants).
5. **Payload-free variants are static** (`None`, `Empty`): a shared 8-byte block per tag in the data segment, no allocation.
6. **Str `len`/`slice` count Unicode scalar values**, consistent with `chars`; `slice` out of range is the runtime error `index`.
7. **Float text conversion via two pure host helpers** (`float_to_str`, `str_to_float`): shortest round-trip float formatting in hand-written wasm is error-prone, and these helpers grant no system access. Everything else (including `to_int`/`from_int`) is wasm.
8. **Two built-ins added beyond §3.5:** `int_to_float(Int) -> Float` and `float_to_int(Float) -> Option[Int]` (truncating; `None` for NaN/±inf/out of range). Without them `Float` is unusable because there are no implicit conversions.
9. **Built-in preconditions are runtime contract violations:** `split` needs a non-empty separator, `rand_int(lo, hi)` needs `lo <= hi` and is inclusive on both ends. `print` appends a newline.
10. **Compile per entry point.** `qal_wasm::compile(&TProgram, &Entry)` emits only reachable code, so a module for a pure `main` has no effect imports at all (success criterion 5).
11. **Runtime errors are diagnostics:** codes `E0501`–`E0510`, new kind `permission_denied`, `def` is always a definition name (never `test:x`), `values` carries parameters (and `r`) for contracts, operands for arithmetic, arguments for built-in preconditions, requested bytes for `out_of_memory`.
12. **`fs` confinement:** `--allow fs:<dir>` (repeatable) grants a directory; paths are canonicalised (symlinks and `..` resolved) and must lie inside a granted directory, otherwise the call returns `Err("permission denied: ...")` as data.
13. **`ens` disables tail calls in that function** (as in 1a): the postcondition must be checked per call against that call's parameters. A deep tail-recursive loop therefore belongs in a helper without `ens`; the `stack_overflow` hint says so.
14. **Smaller known limits (documented, not fixed):** the literal `-9223372036854775808` does not lex (write `-9223372036854775807 - 1`); `to_float` accepts Rust's float syntax (`inf`, `NaN`, `.5`); captured stdout is not size-limited beyond the instruction budget; variant names are global across enums.
15. **Known limitation (documented, not fixed):** `push` and `++` copy their input (persistent values on a bump heap), so building an n-element list with `push` in a loop costs O(n²) memory and ends in a clean `out_of_memory` around n ≈ 8000 at 256 MiB. Prefer `map`/`filter`/`fold`/`range`/`split`/`concat`. Fixing this needs a different list representation and belongs with memory management (spec Phase 3) or a dedicated follow-up.

Deferred items from the 1a final review, and where they are done: runtime errors with values in diagnostic format (Tasks 4–5); computed instead of fixed wasm function indices (Task 4, `module.rs`); `BinOp::is_comparison` instead of `prec == 3` (Tasks 1–2); parameter `r` with `ens` rejected (Task 3); `def` normalised to definition names (Task 5); let-value errors still check the body (Task 3); arguments checked despite arity errors (Task 3); `10abc` rejected by a separator check (Task 2); keyword in expression position reports "expected an expression" (Task 2); runtime tests for `ens` violations and overflow (Task 5).

## Review Focus

1. **Lists or strings built with `push`/`++` in a loop** (quadratic copying, Decision 15): must end in `out_of_memory` with a hint naming `push`, never crash or hang — pinned in Task 5 `quadratic_push_hits_the_memory_limit_cleanly`.
2. **Input larger than the memory limit** (`read_line`/`read_file` of a huge input): must be `out_of_memory` attributed to the entry definition, not a host panic — pinned in Task 5 `huge_input_is_out_of_memory_not_a_crash`.
3. **Output printed before a runtime error:** the error result must still carry everything printed so far — pinned in Task 5 `stdout_survives_runtime_errors` and Task 6 `run_refuses_ungranted_effects`.
4. **Very deep return values** (long linked enums returned from `qal call`): the JSON must be truncated with `"..."`, not overflow the host stack — pinned in Task 5 `deep_values_are_truncated_in_json`.
5. **Exit codes outside 0..=255** (negative or large `main` results): the process exits 1 while the JSON keeps the real value — pinned in Task 6 `exit_codes_outside_0_to_255_exit_1_but_keep_the_value`.

The 1a Review Focus tests (integer boundaries, timeouts/stack overflow, deep tail recursion, `ens` in recursive functions, malformed input) remain and are carried over unchanged or extended in Tasks 2, 4 and 5.

## File Structure

| File | Responsibility |
|---|---|
| `qal/crates/qal-core/src/ast.rs` | Source AST: types (incl. `Var` for inference), expressions, patterns, `type`/`fn`/`test` definitions |
| `qal/crates/qal-core/src/tast.rs` | Typed program (checker output, codegen input), `Builtin`, variant tags, type lookups |
| `qal/crates/qal-core/src/diag.rs` | Diagnostics (+ `non_exhaustive`, `missing_field`) |
| `qal/crates/qal-text/src/lexer.rs` | Tokens incl. floats, strings with escapes, number-separator check |
| `qal/crates/qal-text/src/parser.rs` | Full 1b grammar (module doc), depth bound for every recursive construct |
| `qal/crates/qal-text/src/printer.rs` | Canonical text, round-trip safe for all constructs |
| `qal/crates/qal-check/src/lib.rs` | Program-level checks (types, duplicates, signatures, `main`), `check` / `check_program` |
| `qal/crates/qal-check/src/unify.rs` | Type variables and unification (fn values may have fewer effects) |
| `qal/crates/qal-check/src/infer.rs` | Expression and pattern inference, deferred checks, final type resolution |
| `qal/crates/qal-check/src/builtins.rs` | Typing of built-in calls |
| `qal/crates/qal-check/src/exhaust.rs` | Match exhaustiveness with a missing-pattern witness |
| `qal/crates/qal-check/src/tests.rs` | Checker tests |
| `qal/crates/qal-wasm/src/lib.rs` | `compile(&TProgram, &Entry)`, metadata (`FuncMeta`, `Site`), failure kinds, host import names |
| `qal/crates/qal-wasm/src/asm.rs` | Instruction-building helpers, value representation, slot conversion |
| `qal/crates/qal-wasm/src/module.rs` | Module builder: computed indices, imports, static data, `emit_fail`, structural equality |
| `qal/crates/qal-wasm/src/rt.rs` | Runtime library in wasm: allocator, checked arithmetic, Str/List built-ins |
| `qal/crates/qal-wasm/src/gen.rs` | Lowering of typed expressions, contracts, `match` |
| `qal/crates/qal-wasm/tests/codegen.rs` | End-to-end codegen tests with a minimal host |
| `qal/crates/qal-run/src/lib.rs` | `Runner`: permissions, invocation, value passing, failure → `RunError` |
| `qal/crates/qal-run/src/host.rs` | `Permissions`, host functions per effect group |
| `qal/crates/qal-run/src/value.rs` | `Value`, JSON form, decoding guest memory |
| `qal/crates/qal-run/src/tests.rs` | Runtime tests |
| `qal/crates/qal-cli/src/lib.rs` | `load`, per-entry runner, `--allow` option parsing, CLI argument conversion |
| `qal/crates/qal-cli/src/main.rs` | `qal check / fmt / call / test / run` |
| `qal/crates/qal-cli/tests/cli.rs` | Binary-level tests |
| `qal/examples/age.qal`, `qal/examples/wordcount.qal` | The spec §4 example and an `fs` example |

**Build order note:** Task 1 changes the AST that every other crate uses, so from Task 1 until Task 6 the crates *downstream* of the task's crate do not compile. Each task therefore runs only its own crate's tests (`cargo test -p <crate>`), which build only the crate and its already-updated dependencies. The whole workspace builds again after Task 6.

---

### Task 1: Core data model (`qal-core`)

**Files:**
- Modify (full rewrite): `qal/crates/qal-core/src/ast.rs`
- Create: `qal/crates/qal-core/src/tast.rs`
- Modify: `qal/crates/qal-core/src/diag.rs`, `qal/crates/qal-core/src/lib.rs`
- Test: inline modules in `ast.rs`, `tast.rs`, `diag.rs`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `ast::Type` (no longer `Copy`; `Clone, PartialEq, Eq, Hash, Display` printing source syntax): `Int, Float, Bool, Unit, Str, List(Box<Type>), Option(Box<Type>), Result(Box<Type>, Box<Type>), Named(String), Fn(Vec<Type>, Box<Type>, Vec<String>), Var(u32)`; constructors `Type::list/option/result`; `Type::BUILTIN_NAMES`. `Type::name()` is gone — use `to_string()`.
  - `ast::BinOp` gains `Concat` (`++`, precedence 4) and `is_comparison()`.
  - `ast::Expr` adds `Float(f64), Str(String), List(Vec<Expr>), Record(String, Vec<(String, Expr)>), Field(Box<Expr>, String), Match(Box<Expr>, Vec<Arm>), Seq(Vec<Expr>)`; `Let` becomes `Let(String, Option<Type>, Box<Expr>, Box<Expr>)`. `Expr` is `PartialEq` only (contains `f64`).
  - `ast::{Arm { pat, guard: Option<Expr>, body }, Pat::{Wild, Bind, Int, Str, Bool, Unit, Variant(String, Vec<Pat>), Record(String, Vec<(String, Pat)>)}, Variant { name, fields: Vec<Type> }, TypeBody::{Record, Enum}, TypeDef { name, body }}`; `Def` gains `Type(TypeDef)`.
  - `tast::{TProgram { types, fns: Vec<TFn>, tests: Vec<TTest> }, TFn { def: FnDef, locals: Vec<Type>, req, ens, result_local: Option<u32>, body }, TTest { name, locals, body }, TExpr { ty, kind }, TKind, TArm, TPat, Builtin}` and `TAG_NONE = 0, TAG_SOME = 1, TAG_OK = 0, TAG_ERR = 1`; lookups `type_def, fn_index, variant_names, variant_fields, record_fields, is_record`; `Builtin::{name, effect, NAMES}`.
  - `Diagnostic::non_exhaustive(def, path, missing)` (E0205) and `Diagnostic::missing_field(def, path, record, field)` (E0206, kind `type_mismatch`).

- [ ] **Step 1: Write the failing tests**

Append to `qal/crates/qal-core/src/ast.rs` (it has no test module yet):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn types_display_as_source_syntax() {
        let t = Type::Fn(vec![Type::list(Type::Str), Type::Named("User".into())], Box::new(Type::result(Type::Int, Type::Str)), vec!["io".into(), "fs".into()]);
        assert_eq!(t.to_string(), "fn(List[Str], User) -> Result[Int, Str] !io !fs");
        assert_eq!(Type::option(Type::Float).to_string(), "Option[Float]");
        let inner = Type::Fn(vec![], Box::new(Type::Int), vec!["io".into()]);
        assert_eq!(Type::Fn(vec![], Box::new(inner), vec![]).to_string(), "fn() -> (fn() -> Int !io)");
    }

    #[test]
    fn comparison_classification() {
        assert!(BinOp::Le.is_comparison());
        assert!(!BinOp::Concat.is_comparison());
        assert_eq!(BinOp::Concat.precedence(), BinOp::Add.precedence());
    }
}
```

Create `qal/crates/qal-core/src/tast.rs` containing only its test module for now:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Variant;

    #[test]
    fn variant_and_record_lookup() {
        let p = TProgram {
            types: vec![
                TypeDef { name: "E".into(), body: TypeBody::Enum(vec![Variant { name: "A".into(), fields: vec![] }, Variant { name: "B".into(), fields: vec![Type::Int, Type::Str] }]) },
                TypeDef { name: "R".into(), body: TypeBody::Record(vec![("x".into(), Type::Float)]) },
            ],
            ..TProgram::default()
        };
        assert_eq!(p.variant_names(&Type::Named("E".into())), vec!["A".to_string(), "B".to_string()]);
        assert_eq!(p.variant_fields(&Type::Named("E".into()), 1), vec![Type::Int, Type::Str]);
        assert_eq!(p.variant_fields(&Type::result(Type::Int, Type::Str), TAG_ERR), vec![Type::Str]);
        assert_eq!(p.variant_fields(&Type::option(Type::Int), TAG_NONE), vec![]);
        assert_eq!(p.record_fields(&Type::Named("R".into())), vec![("x".to_string(), Type::Float)]);
        assert!(p.is_record(&Type::Named("R".into())) && !p.is_record(&Type::Named("E".into())));
        assert_eq!(Builtin::StrLen.name(), "len");
        assert_eq!(Builtin::WriteFile.effect(), Some("fs"));
        assert!(Builtin::NAMES.contains(&Builtin::RandInt.name()));
    }
}
```

Replace the `#[cfg(test)] mod tests` block at the end of `qal/crates/qal-core/src/diag.rs` with:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_without_empty_optionals() {
        let d = Diagnostic::type_mismatch("f", "body.then", "Int", "Bool").hint("use if");
        let json = serde_json::to_value(&d).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "code": "E0203", "kind": "type_mismatch", "def": "f", "path": "body.then",
                "message": "expected Int, found Bool", "expected": "Int", "found": "Bool",
                "hints": ["use if"]
            })
        );
        let p = serde_json::to_value(Diagnostic::parse(3, 7, "unexpected `)`")).unwrap();
        assert!(p.get("expected").is_none());
        assert_eq!(p["path"], "3:7");
    }

    #[test]
    fn match_and_record_diagnostics() {
        let d = Diagnostic::non_exhaustive("f", "body", "Some(_)");
        assert_eq!((d.code, d.kind, d.message.as_str()), ("E0205", "non_exhaustive_match", "match does not cover `Some(_)`"));
        assert_eq!(d.hints, vec!["add an arm `Some(_) => ...` or a final `_ => ...` arm".to_string()]);
        let d = Diagnostic::missing_field("f", "body", "User", "id");
        assert_eq!((d.code, d.kind), ("E0206", "type_mismatch"));
    }
}
```

Replace `qal/crates/qal-core/src/lib.rs` with:

```rust
//! QAL core: the program graph (AST), the typed program and structured diagnostics.

pub mod ast;
pub mod diag;
pub mod tast;
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd qal && cargo test -p qal-core`
Expected: FAIL to compile (`Type::list`, `BinOp::Concat`, `Builtin`, `TProgram`, `Diagnostic::non_exhaustive` not found).

- [ ] **Step 3: Write the implementation**

Replace `qal/crates/qal-core/src/ast.rs` with:

```rust
//! Abstract syntax of QAL (Phase 1b): the program as written, before name resolution.

use std::fmt;

/// Name of the result variable inside `ens` contracts.
pub const RESULT_VAR: &str = "r";

/// Value types. `Var` only exists while the checker infers types; checked output never contains it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Int,
    Float,
    Bool,
    Unit,
    Str,
    List(Box<Type>),
    Option(Box<Type>),
    Result(Box<Type>, Box<Type>),
    /// A record or enum declared with `type`.
    Named(String),
    /// `fn(A, B) -> R !e`: a reference to a top-level function (no captures).
    Fn(Vec<Type>, Box<Type>, Vec<String>),
    /// Inference variable (checker-internal).
    Var(u32),
}

impl Type {
    pub fn list(t: Type) -> Type {
        Type::List(Box::new(t))
    }

    pub fn option(t: Type) -> Type {
        Type::Option(Box::new(t))
    }

    pub fn result(t: Type, e: Type) -> Type {
        Type::Result(Box::new(t), Box::new(e))
    }

    /// Type names that cannot be declared with `type`.
    pub const BUILTIN_NAMES: &'static [&'static str] = &["Int", "Float", "Bool", "Unit", "Str", "List", "Option", "Result"];
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Int => write!(f, "Int"),
            Type::Float => write!(f, "Float"),
            Type::Bool => write!(f, "Bool"),
            Type::Unit => write!(f, "Unit"),
            Type::Str => write!(f, "Str"),
            Type::List(t) => write!(f, "List[{t}]"),
            Type::Option(t) => write!(f, "Option[{t}]"),
            Type::Result(t, e) => write!(f, "Result[{t}, {e}]"),
            Type::Named(n) => write!(f, "{n}"),
            Type::Fn(ps, r, effs) => {
                let ps: Vec<String> = ps.iter().map(|p| p.to_string()).collect();
                // A function-typed result is parenthesized so its effects stay distinguishable from ours.
                match **r {
                    Type::Fn(..) => write!(f, "fn({}) -> ({r})", ps.join(", "))?,
                    _ => write!(f, "fn({}) -> {r}", ps.join(", "))?,
                }
                for e in effs {
                    write!(f, " !{e}")?;
                }
                Ok(())
            }
            Type::Var(_) => write!(f, "_"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    /// `++`: Str and List concatenation.
    Concat,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

impl BinOp {
    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::Concat => "++",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::And => "&&",
            BinOp::Or => "||",
        }
    }

    /// Binding strength: higher binds tighter.
    pub fn precedence(self) -> u8 {
        match self {
            BinOp::Or => 1,
            BinOp::And => 2,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => 3,
            BinOp::Add | BinOp::Sub | BinOp::Concat => 4,
            BinOp::Mul | BinOp::Div | BinOp::Rem => 5,
        }
    }

    /// Comparisons are non-associative: `a < b < c` is a parse error.
    pub fn is_comparison(self) -> bool {
        matches!(self, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Int(i64),
    /// Always finite and non-negative; negative values are `Un(Neg, Float)`.
    Float(f64),
    Bool(bool),
    Str(String),
    Unit,
    Var(String),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Un(UnOp, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    /// `let name (: type)? = value in body`
    Let(String, Option<Type>, Box<Expr>, Box<Expr>),
    /// `name(args)`: function, built-in, variant constructor or fn-typed local.
    Call(String, Vec<Expr>),
    /// `[a, b, c]`
    List(Vec<Expr>),
    /// `Name { field: value, ... }`
    Record(String, Vec<(String, Expr)>),
    /// `target.field`
    Field(Box<Expr>, String),
    /// `match scrutinee { arms }`
    Match(Box<Expr>, Vec<Arm>),
    /// `{ a; b; c }`: evaluates all, discards the (Unit) values of all but the last.
    Seq(Vec<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Arm {
    pub pat: Pat,
    pub guard: Option<Expr>,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pat {
    /// `_`
    Wild,
    /// lowercase name: binds the value
    Bind(String),
    Int(i64),
    Str(String),
    Bool(bool),
    Unit,
    /// `Name` or `Name(p, ...)`
    Variant(String, Vec<Pat>),
    /// `Name { field: p, other }`; fields not mentioned match anything.
    Record(String, Vec<(String, Pat)>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FnDef {
    pub name: String,
    pub params: Vec<Param>,
    pub ret: Type,
    pub effects: Vec<String>,
    pub req: Vec<Expr>,
    pub ens: Vec<Expr>,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TestDef {
    pub name: String,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub name: String,
    pub fields: Vec<Type>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeBody {
    Record(Vec<(String, Type)>),
    Enum(Vec<Variant>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeDef {
    pub name: String,
    pub body: TypeBody,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Def {
    Type(TypeDef),
    Fn(FnDef),
    Test(TestDef),
}

impl Def {
    pub fn name(&self) -> &str {
        match self {
            Def::Type(t) => &t.name,
            Def::Fn(f) => &f.name,
            Def::Test(t) => &t.name,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Program {
    pub defs: Vec<Def>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn types_display_as_source_syntax() {
        let t = Type::Fn(vec![Type::list(Type::Str), Type::Named("User".into())], Box::new(Type::result(Type::Int, Type::Str)), vec!["io".into(), "fs".into()]);
        assert_eq!(t.to_string(), "fn(List[Str], User) -> Result[Int, Str] !io !fs");
        assert_eq!(Type::option(Type::Float).to_string(), "Option[Float]");
        let inner = Type::Fn(vec![], Box::new(Type::Int), vec!["io".into()]);
        assert_eq!(Type::Fn(vec![], Box::new(inner), vec![]).to_string(), "fn() -> (fn() -> Int !io)");
    }

    #[test]
    fn comparison_classification() {
        assert!(BinOp::Le.is_comparison());
        assert!(!BinOp::Concat.is_comparison());
        assert_eq!(BinOp::Concat.precedence(), BinOp::Add.precedence());
    }
}
```

Replace `qal/crates/qal-core/src/tast.rs` with:

```rust
//! Typed, name-resolved program: the checker's output and the code generator's input.
//!
//! Every expression carries its fully inferred type (never `Type::Var`). Locals are numbered per
//! definition (`Local(id)`; parameters are ids `0..n`), functions are indices into `TProgram::fns`,
//! variants are tags and record fields are declaration-order indices.

use crate::ast::{BinOp, FnDef, Type, TypeBody, TypeDef, UnOp};

/// Variant tags of the built-in sum types.
pub const TAG_NONE: u32 = 0;
pub const TAG_SOME: u32 = 1;
pub const TAG_OK: u32 = 0;
pub const TAG_ERR: u32 = 1;

/// Built-in functions (spec §3.5, plus `int_to_float` / `float_to_int`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Builtin {
    StrLen,
    ListLen,
    Slice,
    Split,
    Join,
    Trim,
    Contains,
    StartsWith,
    ToInt,
    ToFloat,
    FromInt,
    FromFloat,
    Chars,
    Get,
    Push,
    Concat,
    Map,
    Filter,
    Fold,
    Range,
    Reverse,
    SortInt,
    IntToFloat,
    FloatToInt,
    Print,
    ReadLine,
    ReadFile,
    WriteFile,
    NowMs,
    RandInt,
}

impl Builtin {
    /// Source name. `StrLen` and `ListLen` are both `len`.
    pub fn name(self) -> &'static str {
        match self {
            Builtin::StrLen | Builtin::ListLen => "len",
            Builtin::Slice => "slice",
            Builtin::Split => "split",
            Builtin::Join => "join",
            Builtin::Trim => "trim",
            Builtin::Contains => "contains",
            Builtin::StartsWith => "starts_with",
            Builtin::ToInt => "to_int",
            Builtin::ToFloat => "to_float",
            Builtin::FromInt => "from_int",
            Builtin::FromFloat => "from_float",
            Builtin::Chars => "chars",
            Builtin::Get => "get",
            Builtin::Push => "push",
            Builtin::Concat => "concat",
            Builtin::Map => "map",
            Builtin::Filter => "filter",
            Builtin::Fold => "fold",
            Builtin::Range => "range",
            Builtin::Reverse => "reverse",
            Builtin::SortInt => "sort_int",
            Builtin::IntToFloat => "int_to_float",
            Builtin::FloatToInt => "float_to_int",
            Builtin::Print => "print",
            Builtin::ReadLine => "read_line",
            Builtin::ReadFile => "read_file",
            Builtin::WriteFile => "write_file",
            Builtin::NowMs => "now_ms",
            Builtin::RandInt => "rand_int",
        }
    }

    /// The effect a call needs (map/filter/fold need the effects of their function argument).
    pub fn effect(self) -> Option<&'static str> {
        match self {
            Builtin::Print | Builtin::ReadLine => Some("io"),
            Builtin::ReadFile | Builtin::WriteFile => Some("fs"),
            Builtin::NowMs => Some("time"),
            Builtin::RandInt => Some("rand"),
            _ => None,
        }
    }

    /// All source names of built-in functions (reserved: user functions cannot use them).
    pub const NAMES: &'static [&'static str] = &[
        "len", "slice", "split", "join", "trim", "contains", "starts_with", "to_int", "to_float", "from_int", "from_float",
        "chars", "get", "push", "concat", "map", "filter", "fold", "range", "reverse", "sort_int", "int_to_float",
        "float_to_int", "print", "read_line", "read_file", "write_file", "now_ms", "rand_int",
    ];
}

#[derive(Debug, Clone, PartialEq)]
pub struct TExpr {
    pub ty: Type,
    pub kind: TKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TKind {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Unit,
    Local(u32),
    /// A top-level function used as a value (index into `TProgram::fns`).
    FnRef(usize),
    /// `path` addresses the node for runtime failures (overflow, division by zero).
    Bin(BinOp, Box<TExpr>, Box<TExpr>, String),
    Un(UnOp, Box<TExpr>, String),
    If(Box<TExpr>, Box<TExpr>, Box<TExpr>),
    Let(u32, Box<TExpr>, Box<TExpr>),
    /// Direct call of `TProgram::fns[index]`.
    Call(usize, Vec<TExpr>),
    /// Call through a fn-typed value.
    CallIndirect(Box<TExpr>, Vec<TExpr>),
    Builtin(Builtin, Vec<TExpr>, String),
    List(Vec<TExpr>),
    /// Variant of the enum/Option/Result type `ty`, with payload values.
    Variant(u32, Vec<TExpr>),
    /// Record of type `ty`; fields in declaration order.
    Record(Vec<TExpr>),
    /// Field access by declaration-order index.
    Field(Box<TExpr>, u32),
    Match(Box<TExpr>, Vec<TArm>),
    Seq(Vec<TExpr>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TArm {
    pub pat: TPat,
    pub guard: Option<TExpr>,
    pub body: TExpr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TPat {
    Wild,
    Bind(u32),
    Int(i64),
    Str(String),
    Bool(bool),
    Unit,
    Variant(u32, Vec<TPat>),
    /// (field index, pattern) for the mentioned fields.
    Record(Vec<(u32, TPat)>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TFn {
    /// The source definition (names, signature, contract text).
    pub def: FnDef,
    /// Types of all locals by id; parameters first.
    pub locals: Vec<Type>,
    pub req: Vec<TExpr>,
    pub ens: Vec<TExpr>,
    /// Local id of the result variable `r` (only when `ens` is non-empty).
    pub result_local: Option<u32>,
    pub body: TExpr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TTest {
    pub name: String,
    pub locals: Vec<Type>,
    pub body: TExpr,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TProgram {
    pub types: Vec<TypeDef>,
    pub fns: Vec<TFn>,
    pub tests: Vec<TTest>,
}

impl TProgram {
    pub fn type_def(&self, name: &str) -> Option<&TypeDef> {
        self.types.iter().find(|t| t.name == name)
    }

    pub fn fn_index(&self, name: &str) -> Option<usize> {
        self.fns.iter().position(|f| f.def.name == name)
    }

    /// Variant names of a sum type, by tag. Empty for non-sum types.
    pub fn variant_names(&self, ty: &Type) -> Vec<String> {
        match ty {
            Type::Option(_) => vec!["None".into(), "Some".into()],
            Type::Result(..) => vec!["Ok".into(), "Err".into()],
            Type::Named(n) => match self.type_def(n).map(|d| &d.body) {
                Some(TypeBody::Enum(vs)) => vs.iter().map(|v| v.name.clone()).collect(),
                _ => vec![],
            },
            _ => vec![],
        }
    }

    /// Payload types of variant `tag` of the sum type `ty`.
    pub fn variant_fields(&self, ty: &Type, tag: u32) -> Vec<Type> {
        match (ty, tag) {
            (Type::Option(_), TAG_NONE) => vec![],
            (Type::Option(t), _) => vec![(**t).clone()],
            (Type::Result(t, _), TAG_OK) => vec![(**t).clone()],
            (Type::Result(_, e), _) => vec![(**e).clone()],
            (Type::Named(n), _) => match self.type_def(n).map(|d| &d.body) {
                Some(TypeBody::Enum(vs)) => vs[tag as usize].fields.clone(),
                _ => panic!("`{n}` is not an enum"),
            },
            _ => panic!("{ty} is not a sum type"),
        }
    }

    /// Field names and types of the record type `ty`.
    pub fn record_fields(&self, ty: &Type) -> Vec<(String, Type)> {
        match ty {
            Type::Named(n) => match self.type_def(n).map(|d| &d.body) {
                Some(TypeBody::Record(fs)) => fs.clone(),
                _ => panic!("`{n}` is not a record"),
            },
            _ => panic!("{ty} is not a record"),
        }
    }

    pub fn is_record(&self, ty: &Type) -> bool {
        matches!(ty, Type::Named(n) if matches!(self.type_def(n).map(|d| &d.body), Some(TypeBody::Record(_))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Variant;

    #[test]
    fn variant_and_record_lookup() {
        let p = TProgram {
            types: vec![
                TypeDef { name: "E".into(), body: TypeBody::Enum(vec![Variant { name: "A".into(), fields: vec![] }, Variant { name: "B".into(), fields: vec![Type::Int, Type::Str] }]) },
                TypeDef { name: "R".into(), body: TypeBody::Record(vec![("x".into(), Type::Float)]) },
            ],
            ..TProgram::default()
        };
        assert_eq!(p.variant_names(&Type::Named("E".into())), vec!["A".to_string(), "B".to_string()]);
        assert_eq!(p.variant_fields(&Type::Named("E".into()), 1), vec![Type::Int, Type::Str]);
        assert_eq!(p.variant_fields(&Type::result(Type::Int, Type::Str), TAG_ERR), vec![Type::Str]);
        assert_eq!(p.variant_fields(&Type::option(Type::Int), TAG_NONE), vec![]);
        assert_eq!(p.record_fields(&Type::Named("R".into())), vec![("x".to_string(), Type::Float)]);
        assert!(p.is_record(&Type::Named("R".into())) && !p.is_record(&Type::Named("E".into())));
        assert_eq!(Builtin::StrLen.name(), "len");
        assert_eq!(Builtin::WriteFile.effect(), Some("fs"));
        assert!(Builtin::NAMES.contains(&Builtin::RandInt.name()));
    }
}
```

Replace `qal/crates/qal-core/src/diag.rs` with:

```rust
//! Structured diagnostics: every compiler error is data, not prose.

use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Diagnostic {
    pub code: &'static str,
    pub kind: &'static str,
    pub def: String,
    pub path: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub found: Option<String>,
    pub hints: Vec<String>,
}

impl Diagnostic {
    pub fn new(code: &'static str, kind: &'static str, def: &str, path: &str, message: impl Into<String>) -> Self {
        Diagnostic {
            code,
            kind,
            def: def.to_string(),
            path: path.to_string(),
            message: message.into(),
            expected: None,
            found: None,
            hints: Vec::new(),
        }
    }

    pub fn expected_found(mut self, expected: impl Into<String>, found: impl Into<String>) -> Self {
        self.expected = Some(expected.into());
        self.found = Some(found.into());
        self
    }

    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        self.hints.push(hint.into());
        self
    }

    pub fn parse(line: usize, col: usize, message: impl Into<String>) -> Self {
        Diagnostic::new("E0100", "parse", "", &format!("{line}:{col}"), message)
    }

    pub fn unknown_name(def: &str, path: &str, name: &str) -> Self {
        Diagnostic::new("E0201", "unknown_name", def, path, format!("unknown name `{name}`"))
    }

    pub fn duplicate_name(def: &str, path: &str, name: &str) -> Self {
        Diagnostic::new("E0202", "duplicate_name", def, path, format!("`{name}` is defined more than once"))
    }

    pub fn type_mismatch(def: &str, path: &str, expected: &str, found: &str) -> Self {
        Diagnostic::new("E0203", "type_mismatch", def, path, format!("expected {expected}, found {found}"))
            .expected_found(expected, found)
    }

    pub fn arity(def: &str, path: &str, callee: &str, expected: usize, found: usize) -> Self {
        Diagnostic::new(
            "E0204",
            "type_mismatch",
            def,
            path,
            format!("`{callee}` takes {expected} argument(s), {found} given"),
        )
        .expected_found(expected.to_string(), found.to_string())
    }

    pub fn effect_violation(def: &str, path: &str, callee: &str, missing: &[String]) -> Self {
        Diagnostic::new(
            "E0301",
            "effect_violation",
            def,
            path,
            format!("call to `{callee}` needs effect(s) {} not declared here", missing.join(", ")),
        )
        .hint(format!("add `!{}` to the signature of `{def}`", missing.join(" !")))
    }

    pub fn non_exhaustive(def: &str, path: &str, missing: &str) -> Self {
        Diagnostic::new("E0205", "non_exhaustive_match", def, path, format!("match does not cover `{missing}`"))
            .hint(format!("add an arm `{missing} => ...` or a final `_ => ...` arm"))
    }

    pub fn missing_field(def: &str, path: &str, record: &str, field: &str) -> Self {
        Diagnostic::new("E0206", "type_mismatch", def, path, format!("`{record}` needs field `{field}`"))
            .hint(format!("add `{field}: ...` to the `{record} {{ ... }}` literal"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_without_empty_optionals() {
        let d = Diagnostic::type_mismatch("f", "body.then", "Int", "Bool").hint("use if");
        let json = serde_json::to_value(&d).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "code": "E0203", "kind": "type_mismatch", "def": "f", "path": "body.then",
                "message": "expected Int, found Bool", "expected": "Int", "found": "Bool",
                "hints": ["use if"]
            })
        );
        let p = serde_json::to_value(Diagnostic::parse(3, 7, "unexpected `)`")).unwrap();
        assert!(p.get("expected").is_none());
        assert_eq!(p["path"], "3:7");
    }

    #[test]
    fn match_and_record_diagnostics() {
        let d = Diagnostic::non_exhaustive("f", "body", "Some(_)");
        assert_eq!((d.code, d.kind, d.message.as_str()), ("E0205", "non_exhaustive_match", "match does not cover `Some(_)`"));
        assert_eq!(d.hints, vec!["add an arm `Some(_) => ...` or a final `_ => ...` arm".to_string()]);
        let d = Diagnostic::missing_field("f", "body", "User", "id");
        assert_eq!((d.code, d.kind), ("E0206", "type_mismatch"));
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd qal && cargo test -p qal-core 2>&1 | grep -E "test result|warning"`
Expected: `test result: ok. 5 passed`, no warnings. (Downstream crates do not compile yet — see the build order note.)

- [ ] **Step 5: Commit**

```bash
git add qal/crates/qal-core
git commit -m "feat(qal): 1b core data model — full types, patterns, typed program, new diagnostics

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Text form for the full language (`qal-text`)

**Files:**
- Modify (full rewrites): `qal/crates/qal-text/src/lexer.rs`, `qal/crates/qal-text/src/parser.rs`, `qal/crates/qal-text/src/printer.rs`
- Modify: `qal/crates/qal-text/src/lib.rs`
- Test: inline modules in all three files

**Interfaces:**
- Consumes: Task 1 `ast` types, `BinOp::is_comparison`, `Diagnostic::parse`.
- Produces: `Tok` gains `Float(f64)` and `Str(String)` (`Tok` is `PartialEq` only); new symbols `++ => [ ] { } ; . |`; keywords `type`, `match`. `parse_program` accepts the grammar in the parser's module doc; `print_program`, `print_expr`, and new `print_pat`, `print_str` (exported from `qal_text`: `print_expr`, `print_pat`, `print_program`). Round trip `parse_program(print_program(p)) == p` holds for every construct. Decided details: number literals must end at a separator (`10abc` is one error), a keyword where an expression is expected reports `expected an expression, found ...`, float literals must be finite, string escapes are `\n \t \r \0 \\ \" \u{HEX}`, a function-typed return type is printed in parentheses (`fn f() -> (fn() -> Int) !io`) so effects stay unambiguous.

- [ ] **Step 1: Write the failing tests**

Replace the `#[cfg(test)] mod tests` block of `qal/crates/qal-text/src/lexer.rs` with:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn toks(src: &str) -> Vec<Tok> {
        lex(src).unwrap().into_iter().map(|t| t.tok).collect()
    }

    #[test]
    fn lexes_function_header() {
        assert_eq!(
            toks("fn f(x: Int) -> Int"),
            vec![
                Tok::Ident("fn".into()),
                Tok::Ident("f".into()),
                Tok::Sym("("),
                Tok::Ident("x".into()),
                Tok::Sym(":"),
                Tok::Ident("Int".into()),
                Tok::Sym(")"),
                Tok::Sym("->"),
                Tok::Ident("Int".into()),
                Tok::Eof,
            ]
        );
    }

    #[test]
    fn lexes_two_char_operators_and_skips_comments() {
        assert_eq!(
            toks("a <= b # comment\n&& c != 10"),
            vec![
                Tok::Ident("a".into()),
                Tok::Sym("<="),
                Tok::Ident("b".into()),
                Tok::Sym("&&"),
                Tok::Ident("c".into()),
                Tok::Sym("!="),
                Tok::Int(10),
                Tok::Eof,
            ]
        );
    }

    #[test]
    fn tracks_positions() {
        let t = lex("fn\n  x").unwrap();
        assert_eq!((t[1].line, t[1].col), (2, 3));
    }

    #[test]
    fn eof_column_accounts_for_trailing_comment() {
        let t = lex("x # c").unwrap();
        let eof = t.last().unwrap();
        assert_eq!((eof.line, eof.col), (1, 6));
    }

    #[test]
    fn rejects_oversized_integer_and_unknown_characters() {
        let e = lex("99999999999999999999").unwrap_err();
        assert_eq!(e.kind, "parse");
        assert_eq!(e.path, "1:1");
        let e = lex("x @ y").unwrap_err();
        assert_eq!(e.path, "1:3");
        let e = lex("größe").unwrap_err();
        assert_eq!(e.kind, "parse");
    }

    #[test]
    fn lexes_floats_and_new_symbols() {
        assert_eq!(
            toks("1.5 2e3 7.25E-2 x.y ++ => [ ] { } ; |"),
            vec![
                Tok::Float(1.5),
                Tok::Float(2000.0),
                Tok::Float(0.0725),
                Tok::Ident("x".into()),
                Tok::Sym("."),
                Tok::Ident("y".into()),
                Tok::Sym("++"),
                Tok::Sym("=>"),
                Tok::Sym("["),
                Tok::Sym("]"),
                Tok::Sym("{"),
                Tok::Sym("}"),
                Tok::Sym(";"),
                Tok::Sym("|"),
                Tok::Eof,
            ]
        );
        // `1.` is not a float: the dot is a separate token.
        assert_eq!(toks("1.x"), vec![Tok::Int(1), Tok::Sym("."), Tok::Ident("x".into()), Tok::Eof]);
    }

    #[test]
    fn number_literals_need_a_separator() {
        let e = lex("x = 10abc").unwrap_err();
        assert_eq!(e.path, "1:5");
        assert!(e.message.contains("invalid number literal `10abc`"), "{}", e.message);
        assert!(lex("1.5e").unwrap_err().message.contains("`1.5e`"));
        assert!(lex("1e999").unwrap_err().message.contains("out of range"));
    }

    #[test]
    fn lexes_strings_with_escapes_and_utf8() {
        assert_eq!(toks(r#""a\"b\\c\n\u{1F600}" "größe""#), vec![Tok::Str("a\"b\\c\n😀".into()), Tok::Str("größe".into()), Tok::Eof]);
        let t = lex("\"a\nb\" x").unwrap();
        assert_eq!((t[1].line, t[1].col), (2, 4));
        let e = lex("x \"abc").unwrap_err();
        assert_eq!((e.path.as_str(), e.message.as_str()), ("1:3", "unterminated string literal"));
        assert!(lex(r#""\q""#).unwrap_err().message.contains("invalid escape `\\q`"));
        assert!(lex(r#""\u{110000}""#).unwrap_err().message.contains("invalid escape"));
    }
}
```

Replace the `#[cfg(test)] mod tests` block of `qal/crates/qal-text/src/parser.rs` with:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn int(n: i64) -> Box<Expr> {
        Box::new(Expr::Int(n))
    }
    fn var(s: &str) -> Box<Expr> {
        Box::new(Expr::Var(s.into()))
    }
    fn body(src: &str) -> Expr {
        let p = parse_program(&format!("test t = {src}")).unwrap();
        let Def::Test(t) = &p.defs[0] else { panic!() };
        t.body.clone()
    }

    #[test]
    fn parses_function_with_contracts_and_effects() {
        let p = parse_program("fn f(x: Int, b: Bool) -> Int !io req x >= 0 ens r > x = x + 1").unwrap();
        let Def::Fn(f) = &p.defs[0] else { panic!("expected fn") };
        assert_eq!(f.name, "f");
        assert_eq!(f.params, vec![Param { name: "x".into(), ty: Type::Int }, Param { name: "b".into(), ty: Type::Bool }]);
        assert_eq!(f.ret, Type::Int);
        assert_eq!(f.effects, vec!["io".to_string()]);
        assert_eq!(f.req, vec![Expr::Bin(BinOp::Ge, var("x"), int(0))]);
        assert_eq!(f.ens, vec![Expr::Bin(BinOp::Gt, var("r"), var("x"))]);
        assert_eq!(f.body, Expr::Bin(BinOp::Add, var("x"), int(1)));
    }

    #[test]
    fn respects_precedence_and_left_associativity() {
        let sum = Expr::Bin(BinOp::Sub, Box::new(Expr::Bin(BinOp::Add, int(1), Box::new(Expr::Bin(BinOp::Mul, int(2), int(3))))), int(4));
        let cmp = Expr::Bin(BinOp::Eq, Box::new(sum), int(3));
        assert_eq!(body("1 + 2 * 3 - 4 == 3 || false"), Expr::Bin(BinOp::Or, Box::new(cmp), Box::new(Expr::Bool(false))));
        let s = |x: &str| Box::new(Expr::Str(x.into()));
        assert_eq!(body(r#""a" ++ "b" == "ab""#), Expr::Bin(BinOp::Eq, Box::new(Expr::Bin(BinOp::Concat, s("a"), s("b"))), s("ab")));
    }

    #[test]
    fn parses_let_if_calls_unary_and_unit() {
        let p = parse_program("fn g() -> Unit = ()\ntest t = let y = -f(1, 2) in if !true then y else y").unwrap();
        assert_eq!(p.defs.len(), 2);
        let Def::Test(t) = &p.defs[1] else { panic!() };
        let call = Expr::Call("f".into(), vec![Expr::Int(1), Expr::Int(2)]);
        let neg = Expr::Un(UnOp::Neg, Box::new(call));
        let cond = Expr::Un(UnOp::Not, Box::new(Expr::Bool(true)));
        let iff = Expr::If(Box::new(cond), var("y"), var("y"));
        assert_eq!(t.body, Expr::Let("y".into(), None, Box::new(neg), Box::new(iff)));
    }

    #[test]
    fn parses_type_definitions_and_type_syntax() {
        let p = parse_program(
            "type User = { id: Int, name: Str, }\n\
             type Err = Empty | NotNumber(Str) | Pair(Int, List[Float])\n\
             type One = | Only\n\
             fn f(u: User, k: fn(Int) -> Option[Int] !io, r2: Result[Unit, Str]) -> List[Err] = []",
        )
        .unwrap();
        assert_eq!(
            p.defs[0],
            Def::Type(TypeDef { name: "User".into(), body: TypeBody::Record(vec![("id".into(), Type::Int), ("name".into(), Type::Str)]) })
        );
        let Def::Type(TypeDef { body: TypeBody::Enum(vs), .. }) = &p.defs[1] else { panic!() };
        assert_eq!(vs.iter().map(|v| (v.name.as_str(), v.fields.len())).collect::<Vec<_>>(), vec![("Empty", 0), ("NotNumber", 1), ("Pair", 2)]);
        assert_eq!(vs[2].fields[1], Type::list(Type::Float));
        let Def::Type(TypeDef { body: TypeBody::Enum(one), .. }) = &p.defs[2] else { panic!() };
        assert_eq!(one.len(), 1);
        let Def::Fn(f) = &p.defs[3] else { panic!() };
        assert_eq!(f.params[1].ty, Type::Fn(vec![Type::Int], Box::new(Type::option(Type::Int)), vec!["io".into()]));
        assert_eq!(f.params[2].ty, Type::result(Type::Unit, Type::Str));
        assert_eq!(f.ret, Type::list(Type::Named("Err".into())));
    }

    #[test]
    fn parses_data_expressions() {
        assert_eq!(body("[1, 2,]"), Expr::List(vec![Expr::Int(1), Expr::Int(2)]));
        assert_eq!(body("[]"), Expr::List(vec![]));
        assert_eq!(body("{ print(\"a\"); 1 }"), Expr::Seq(vec![Expr::Call("print".into(), vec![Expr::Str("a".into())]), Expr::Int(1)]));
        assert_eq!(
            body("User { id: 1, name: \"a\" }.name"),
            Expr::Field(Box::new(Expr::Record("User".into(), vec![("id".into(), Expr::Int(1)), ("name".into(), Expr::Str("a".into()))])), "name".into())
        );
        assert_eq!(body("-a.b.c"), Expr::Un(UnOp::Neg, Box::new(Expr::Field(Box::new(Expr::Field(var("a"), "b".into())), "c".into()))));
        assert_eq!(body("let x: Option[Int] = None in x"), Expr::Let("x".into(), Some(Type::option(Type::Int)), var("None"), var("x")));
        assert_eq!(body("1.5"), Expr::Float(1.5));
    }

    #[test]
    fn parses_match_with_patterns_and_guards() {
        let e = body("match get(xs, 0) { None => 0, Some(n) if n < 0 => -1, Some(_) => 1, }");
        let Expr::Match(scrut, arms) = e else { panic!() };
        assert_eq!(*scrut, Expr::Call("get".into(), vec![Expr::Var("xs".into()), Expr::Int(0)]));
        assert_eq!(arms.len(), 3);
        assert_eq!(arms[0].pat, Pat::Variant("None".into(), vec![]));
        assert_eq!(arms[1].pat, Pat::Variant("Some".into(), vec![Pat::Bind("n".into())]));
        assert_eq!(arms[1].guard, Some(Expr::Bin(BinOp::Lt, var("n"), int(0))));
        assert_eq!(arms[2].pat, Pat::Variant("Some".into(), vec![Pat::Wild]));
        let Expr::Match(_, arms) = body("match u { User { id: 1, name } => name, _ => \"\" }") else { panic!() };
        assert_eq!(arms[0].pat, Pat::Record("User".into(), vec![("id".into(), Pat::Int(1)), ("name".into(), Pat::Bind("name".into()))]));
        let Expr::Match(_, arms) = body("match x { -1 => 0, \"s\" => 1, true => 2, () => 3 }") else { panic!() };
        let pats: Vec<Pat> = arms.into_iter().map(|a| a.pat).collect();
        assert_eq!(pats, vec![Pat::Int(-1), Pat::Str("s".into()), Pat::Bool(true), Pat::Unit]);
    }

    #[test]
    fn record_literal_is_not_parsed_in_match_scrutinee() {
        // `match Empty { ... }`: the braces open the arms, not a record literal.
        let Expr::Match(scrut, arms) = body("match Empty { Empty => 1, _ => 2 }") else { panic!() };
        assert_eq!(*scrut, Expr::Var("Empty".into()));
        assert_eq!(arms.len(), 2);
        // Inside parentheses record literals are allowed again.
        let Expr::Match(scrut, _) = body("match (P { x: 1 }) { _ => 1 }") else { panic!() };
        assert_eq!(*scrut, Expr::Record("P".into(), vec![("x".into(), Expr::Int(1))]));
    }

    #[test]
    fn reports_errors_with_position() {
        let e = parse_program("fn f(x: Int) -> Int = (x + 1").unwrap_err();
        assert_eq!(e.kind, "parse");
        assert_eq!(e.path, "1:29");
        assert!(e.message.contains("expected `)`"), "{}", e.message);
        let e = parse_program("fn if() -> Int = 1").unwrap_err();
        assert!(e.message.contains("expected a name"), "{}", e.message);
        let e = parse_program("test t = 1 < 2 < 3").unwrap_err();
        assert!(e.message.contains("chained"), "{}", e.message);
        let e = parse_program("fn f() -> 3 = 1").unwrap_err();
        assert!(e.message.contains("expected a type"), "{}", e.message);
    }

    #[test]
    fn keyword_in_expression_position_is_expected_an_expression() {
        let e = parse_program("fn f() -> Int = then").unwrap_err();
        assert_eq!(e.message, "expected an expression, found `then`");
        let e = parse_program("test t = 1 + if true then 1 else 2 == 2").unwrap_err();
        assert!(e.message.starts_with("expected an expression, found `if` (wrap"), "{}", e.message);
    }

    #[test]
    fn enforces_name_case() {
        assert!(parse_program("fn F() -> Int = 1").unwrap_err().message.contains("function names start with a lowercase letter"));
        assert!(parse_program("type user = { a: Int }").unwrap_err().message.contains("type names start with an uppercase letter"));
        assert!(parse_program("type U = a | B").unwrap_err().message.contains("variant names start with an uppercase letter"));
        assert!(parse_program("test t = let X = 1 in X").unwrap_err().message.contains("variable names"));
        assert!(parse_program("test t = {}").unwrap_err().message.contains("expected an expression, found `}`"));
        assert!(parse_program("test t = match x { }").unwrap_err().message.contains("at least one arm"));
    }

    /// Debug-build parser frames are large; give deep-but-legal inputs more room than a 2 MiB test thread.
    fn parse_big_stack(src: String) -> Result<Program, Diagnostic> {
        std::thread::Builder::new().stack_size(64 << 20).spawn(move || parse_program(&src)).unwrap().join().unwrap()
    }

    fn rejects_deep(src: String) {
        let e = parse_big_stack(src).unwrap_err();
        assert_eq!(e.kind, "parse");
        assert!(e.message.contains("nested too deeply (max 256 levels)"), "{}", e.message);
    }

    #[test]
    fn rejects_deeply_nested_parens() {
        let n = 10_000;
        rejects_deep(format!("test t = {}1{}", "(".repeat(n), ")".repeat(n)));
    }

    #[test]
    fn rejects_long_operator_chain() {
        let mut src = String::from("test t = 1");
        for _ in 0..10_000 {
            src.push_str(" + 1");
        }
        rejects_deep(src);
    }

    #[test]
    fn rejects_long_unary_chain() {
        rejects_deep(format!("test t = {}1", "-".repeat(10_000)));
    }

    #[test]
    fn rejects_deeply_nested_lets() {
        rejects_deep(format!("test t = {}1", "let x = 1 in ".repeat(10_000)));
    }

    #[test]
    fn rejects_deeply_nested_data_and_patterns() {
        rejects_deep(format!("test t = {}1{}", "[".repeat(10_000), "]".repeat(10_000)));
        rejects_deep(format!("test t = {}1{}", "{".repeat(10_000), "}".repeat(10_000)));
        rejects_deep(format!("test t = x{}", ".f".repeat(10_000)));
        rejects_deep(format!("test t = match x {{ {}_{} => 1 }}", "Some(".repeat(10_000), ")".repeat(10_000)));
        rejects_deep(format!("fn f(x: {}Int{}) -> Int = 1", "List[".repeat(10_000), "]".repeat(10_000)));
    }

    #[test]
    fn accepts_moderate_chains_and_nesting() {
        let mut src = String::from("test t = 1");
        for _ in 0..200 {
            src.push_str(" + 1");
        }
        parse_big_stack(src).unwrap();
        parse_big_stack(format!("test t = {}1{}", "(".repeat(200), ")".repeat(200))).unwrap();
    }

    #[test]
    fn empty_source_is_empty_program() {
        assert_eq!(parse_program("  # nothing\n").unwrap(), Program::default());
    }
}
```

Replace the `#[cfg(test)] mod tests` block of `qal/crates/qal-text/src/printer.rs` with:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_program;

    fn roundtrip(src: &str) {
        let p = parse_program(src).unwrap();
        let printed = print_program(&p);
        let reparsed = parse_program(&printed).unwrap_or_else(|e| panic!("reparse failed: {e:?}\n{printed}"));
        assert_eq!(reparsed, p, "printed:\n{printed}");
    }

    #[test]
    fn prints_canonical_form() {
        let p = parse_program("fn  add(a:Int,b:Int)->Int !io req a>=0 = a+b*2").unwrap();
        assert_eq!(print_program(&p), "fn add(a: Int, b: Int) -> Int !io\n  req a >= 0\n= a + b * 2\n");
    }

    #[test]
    fn keeps_needed_parentheses_only() {
        let p = parse_program("test t = (1 + 2) * 3 - (4 - 5) == ((6))").unwrap();
        assert_eq!(print_program(&p), "test t = (1 + 2) * 3 - (4 - 5) == 6\n");
    }

    #[test]
    fn prints_data_definitions_and_match() {
        let src = "type   E =|A|B( Int,Str )\ntype R={x:Float,}\n\
                   fn f(e: E) -> Str = match e {A=>\"a\\n\",B(n,s) if n>0=>s,_=>\"\"}";
        let p = parse_program(src).unwrap();
        assert_eq!(
            print_program(&p),
            "type E = A | B(Int, Str)\n\ntype R = { x: Float }\n\n\
             fn f(e: E) -> Str\n= match e { A => \"a\\n\", B(n, s) if n > 0 => s, _ => \"\" }\n"
        );
    }

    #[test]
    fn roundtrips_all_constructs() {
        roundtrip("fn f(x: Int) -> Int req x > 0 ens r >= 1 = if x == 1 then 1 else x * f(x - 1)");
        roundtrip("test t = let a = (if true then 1 else 2) in a + (let b = 3 in b)");
        roundtrip("test t = -(-1) == 1 && !(1 < 2) || (1 == 1) == true");
        roundtrip("fn u() -> Unit !io !fs = ()");
        roundtrip("test t = 1 - (2 - 3) - 4 * (5 % 2) / 1");
        roundtrip("type User = { id: Int, tags: List[Str] }\ntype T = Leaf | Node(T, Int, T)");
        roundtrip("fn g(k: fn(Int, Str) -> Option[Int] !io, r2: Result[Unit, Str]) -> List[Float] = [1.5, 0.1, 1e300, 2.5e-7]");
        roundtrip("fn f() -> (fn(Int) -> Int) !io = g\nfn h() -> fn(Int) -> Int !io = g\nfn k(x: fn() -> (fn() -> Int !io)) -> Int = 1");
        roundtrip(r#"test t = "q\"\\\n\t\r\0\u{7}é" ++ "x" == "y""#);
        roundtrip("test t = { print(\"a\"); { 1 }; 2 } == (User { id: 1, n: [] }).id");
        roundtrip("test t = (-a).b.c == -a.b.c && (a + b).c == 1");
        roundtrip("test t = let x: Option[List[Int]] = None in match x { Some(_) => 0, None => 1 }");
        roundtrip("test t = match (P { x: 1 }) { P { x: 1, y } => y, P { x: n } if n < -1 => -1, _ => match y { -3 => 1, () => 2 } }");
        roundtrip("test t = match (P { x: 1 }.x + 1) { _ => (match a { _ => 1 }) + 1 }");
        roundtrip("test t = match f(P { x: 1 }) { \"s\" => true, _ => false }");
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd qal && cargo test -p qal-text`
Expected: FAIL to compile (the 1a parser and printer do not know the new AST: `Expr::Let` has four fields, `Type` is not `Copy`, `Tok::Float` / `Tok::Str` missing).

- [ ] **Step 3: Write the implementation**

Replace `qal/crates/qal-text/src/lexer.rs` with:

```rust
//! Lexer: source text -> tokens with line/column positions.

use qal_core::diag::Diagnostic;

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Ident(String),
    Int(i64),
    Float(f64),
    /// String literal with escapes already resolved.
    Str(String),
    /// Punctuation and operators, e.g. "(", "->", "==", "&&".
    Sym(&'static str),
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub tok: Tok,
    pub line: usize,
    pub col: usize,
}

pub const KEYWORDS: &[&str] =
    &["fn", "type", "test", "let", "in", "if", "then", "else", "match", "req", "ens", "true", "false"];

/// Longest symbols first so that "->" wins over "-" and "++" over "+".
const SYMBOLS: &[&str] = &[
    "->", "=>", "==", "!=", "<=", ">=", "&&", "||", "++", "(", ")", "[", "]", "{", "}", ",", ";", ":", ".", "|", "=", "<",
    ">", "+", "-", "*", "/", "%", "!",
];

pub fn lex(src: &str) -> Result<Vec<Token>, Diagnostic> {
    let chars: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let (mut i, mut line, mut col) = (0usize, 1usize, 1usize);
    while i < chars.len() {
        let c = chars[i];
        if c == '\n' {
            i += 1;
            line += 1;
            col = 1;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            col += 1;
            continue;
        }
        if c == '#' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
                col += 1;
            }
            continue;
        }
        let (start_line, start_col) = (line, col);
        if c.is_ascii_digit() {
            let start = i;
            let digits = |i: &mut usize| {
                while *i < chars.len() && chars[*i].is_ascii_digit() {
                    *i += 1;
                }
            };
            digits(&mut i);
            let mut is_float = false;
            if i + 1 < chars.len() && chars[i] == '.' && chars[i + 1].is_ascii_digit() {
                is_float = true;
                i += 1;
                digits(&mut i);
            }
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                let sign = usize::from(i + 1 < chars.len() && (chars[i + 1] == '+' || chars[i + 1] == '-'));
                if i + 1 + sign < chars.len() && chars[i + 1 + sign].is_ascii_digit() {
                    is_float = true;
                    i += 1 + sign;
                    digits(&mut i);
                }
            }
            // A literal must end at a separator: `10abc` is one bad token, not `10` followed by `abc`.
            if i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let text: String = chars[start..i].iter().collect();
                return Err(Diagnostic::parse(start_line, start_col, format!("invalid number literal `{text}`")));
            }
            let text: String = chars[start..i].iter().collect();
            col += i - start;
            let tok = if is_float {
                match text.parse::<f64>() {
                    Ok(v) if v.is_finite() => Tok::Float(v),
                    _ => return Err(Diagnostic::parse(start_line, start_col, format!("float literal `{text}` is out of range"))),
                }
            } else {
                Tok::Int(text.parse::<i64>().map_err(|_| {
                    Diagnostic::parse(start_line, start_col, format!("integer literal `{text}` does not fit in Int (64-bit)"))
                })?)
            };
            out.push(Token { tok, line: start_line, col: start_col });
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            col += i - start;
            let text: String = chars[start..i].iter().collect();
            out.push(Token { tok: Tok::Ident(text), line: start_line, col: start_col });
            continue;
        }
        if c == '"' {
            let s = string(&chars, &mut i, &mut line, &mut col)?;
            out.push(Token { tok: Tok::Str(s), line: start_line, col: start_col });
            continue;
        }
        let rest: String = chars[i..chars.len().min(i + 2)].iter().collect();
        match SYMBOLS.iter().find(|s| rest.starts_with(**s)) {
            Some(sym) => {
                i += sym.len();
                col += sym.len();
                out.push(Token { tok: Tok::Sym(sym), line: start_line, col: start_col });
            }
            None => return Err(Diagnostic::parse(start_line, start_col, format!("unexpected character `{c}`"))),
        }
    }
    out.push(Token { tok: Tok::Eof, line, col });
    Ok(out)
}

/// Lexes a string literal starting at the opening quote. Escapes: `\n \t \r \0 \\ \" \u{HEX}`.
fn string(chars: &[char], i: &mut usize, line: &mut usize, col: &mut usize) -> Result<String, Diagnostic> {
    let (start_line, start_col) = (*line, *col);
    let mut s = String::new();
    *i += 1;
    *col += 1;
    loop {
        let Some(&c) = chars.get(*i) else {
            return Err(Diagnostic::parse(start_line, start_col, "unterminated string literal"));
        };
        let (esc_line, esc_col) = (*line, *col);
        *i += 1;
        *col += 1;
        match c {
            '"' => return Ok(s),
            '\n' => {
                *line += 1;
                *col = 1;
                s.push('\n');
            }
            '\\' => {
                let bad = |what: &str| Diagnostic::parse(esc_line, esc_col, format!("invalid escape `\\{what}` in string literal"));
                let Some(&e) = chars.get(*i) else {
                    return Err(Diagnostic::parse(start_line, start_col, "unterminated string literal"));
                };
                *i += 1;
                *col += 1;
                match e {
                    'n' => s.push('\n'),
                    't' => s.push('\t'),
                    'r' => s.push('\r'),
                    '0' => s.push('\0'),
                    '\\' => s.push('\\'),
                    '"' => s.push('"'),
                    'u' => {
                        if chars.get(*i) != Some(&'{') {
                            return Err(bad("u"));
                        }
                        let start = *i + 1;
                        let mut end = start;
                        while end < chars.len() && chars[end].is_ascii_hexdigit() && end - start < 6 {
                            end += 1;
                        }
                        let hex: String = chars[start..end].iter().collect();
                        if chars.get(end) != Some(&'}') || hex.is_empty() {
                            return Err(bad(&format!("u{{{hex}")));
                        }
                        let ch = u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32).ok_or_else(|| bad(&format!("u{{{hex}}}")))?;
                        s.push(ch);
                        *col += end + 1 - *i;
                        *i = end + 1;
                    }
                    other => return Err(bad(&other.to_string())),
                }
            }
            c => s.push(c),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(src: &str) -> Vec<Tok> {
        lex(src).unwrap().into_iter().map(|t| t.tok).collect()
    }

    #[test]
    fn lexes_function_header() {
        assert_eq!(
            toks("fn f(x: Int) -> Int"),
            vec![
                Tok::Ident("fn".into()),
                Tok::Ident("f".into()),
                Tok::Sym("("),
                Tok::Ident("x".into()),
                Tok::Sym(":"),
                Tok::Ident("Int".into()),
                Tok::Sym(")"),
                Tok::Sym("->"),
                Tok::Ident("Int".into()),
                Tok::Eof,
            ]
        );
    }

    #[test]
    fn lexes_two_char_operators_and_skips_comments() {
        assert_eq!(
            toks("a <= b # comment\n&& c != 10"),
            vec![
                Tok::Ident("a".into()),
                Tok::Sym("<="),
                Tok::Ident("b".into()),
                Tok::Sym("&&"),
                Tok::Ident("c".into()),
                Tok::Sym("!="),
                Tok::Int(10),
                Tok::Eof,
            ]
        );
    }

    #[test]
    fn tracks_positions() {
        let t = lex("fn\n  x").unwrap();
        assert_eq!((t[1].line, t[1].col), (2, 3));
    }

    #[test]
    fn eof_column_accounts_for_trailing_comment() {
        let t = lex("x # c").unwrap();
        let eof = t.last().unwrap();
        assert_eq!((eof.line, eof.col), (1, 6));
    }

    #[test]
    fn rejects_oversized_integer_and_unknown_characters() {
        let e = lex("99999999999999999999").unwrap_err();
        assert_eq!(e.kind, "parse");
        assert_eq!(e.path, "1:1");
        let e = lex("x @ y").unwrap_err();
        assert_eq!(e.path, "1:3");
        let e = lex("größe").unwrap_err();
        assert_eq!(e.kind, "parse");
    }

    #[test]
    fn lexes_floats_and_new_symbols() {
        assert_eq!(
            toks("1.5 2e3 7.25E-2 x.y ++ => [ ] { } ; |"),
            vec![
                Tok::Float(1.5),
                Tok::Float(2000.0),
                Tok::Float(0.0725),
                Tok::Ident("x".into()),
                Tok::Sym("."),
                Tok::Ident("y".into()),
                Tok::Sym("++"),
                Tok::Sym("=>"),
                Tok::Sym("["),
                Tok::Sym("]"),
                Tok::Sym("{"),
                Tok::Sym("}"),
                Tok::Sym(";"),
                Tok::Sym("|"),
                Tok::Eof,
            ]
        );
        // `1.` is not a float: the dot is a separate token.
        assert_eq!(toks("1.x"), vec![Tok::Int(1), Tok::Sym("."), Tok::Ident("x".into()), Tok::Eof]);
    }

    #[test]
    fn number_literals_need_a_separator() {
        let e = lex("x = 10abc").unwrap_err();
        assert_eq!(e.path, "1:5");
        assert!(e.message.contains("invalid number literal `10abc`"), "{}", e.message);
        assert!(lex("1.5e").unwrap_err().message.contains("`1.5e`"));
        assert!(lex("1e999").unwrap_err().message.contains("out of range"));
    }

    #[test]
    fn lexes_strings_with_escapes_and_utf8() {
        assert_eq!(toks(r#""a\"b\\c\n\u{1F600}" "größe""#), vec![Tok::Str("a\"b\\c\n😀".into()), Tok::Str("größe".into()), Tok::Eof]);
        let t = lex("\"a\nb\" x").unwrap();
        assert_eq!((t[1].line, t[1].col), (2, 4));
        let e = lex("x \"abc").unwrap_err();
        assert_eq!((e.path.as_str(), e.message.as_str()), ("1:3", "unterminated string literal"));
        assert!(lex(r#""\q""#).unwrap_err().message.contains("invalid escape `\\q`"));
        assert!(lex(r#""\u{110000}""#).unwrap_err().message.contains("invalid escape"));
    }
}
```

Replace `qal/crates/qal-text/src/parser.rs` with:

```rust
//! Recursive-descent parser: tokens -> Program.
//!
//! Grammar (Phase 1b). LOWER = name starting with a lowercase letter or `_`, UPPER = with an uppercase letter.
//!   program  := def*
//!   def      := "fn" LOWER "(" params ")" "->" type effect* contract* "=" expr
//!             | "test" LOWER "=" expr
//!             | "type" UPPER "=" ( "{" fields "}" | "|"? variant ("|" variant)* )
//!   params   := (LOWER ":" type ("," LOWER ":" type)* ","?)?
//!   fields   := (LOWER ":" type ("," LOWER ":" type)* ","?)?
//!   variant  := UPPER ("(" type ("," type)* ","? ")")?
//!   type     := "Int" | "Float" | "Bool" | "Unit" | "Str" | "List" "[" type "]" | "Option" "[" type "]"
//!             | "Result" "[" type "," type "]" | UPPER | "fn" "(" (type ("," type)*)? ")" "->" type effect*
//!             | "(" type ")"
//!   effect   := "!" LOWER
//!   contract := ("req" | "ens") expr
//!   expr     := "let" LOWER (":" type)? "=" expr "in" expr
//!             | "if" expr "then" expr "else" expr
//!             | "match" expr "{" arm ("," arm)* ","? "}"      (no `Name {` record literal directly in the scrutinee)
//!             | binary
//!   arm      := pattern ("if" expr)? "=>" expr
//!   pattern  := "_" | LOWER | "-"? INT | STR | "true" | "false" | "(" ")"
//!             | UPPER ("(" pattern ("," pattern)* ","? ")")? | UPPER "{" (LOWER (":" pattern)?),* "}"
//!   binary   := precedence climbing over BinOp::precedence (comparisons non-associative)
//!   unary    := ("-" | "!") unary | postfix
//!   postfix  := atom ("." LOWER)*
//!   atom     := INT | FLOAT | STR | "true" | "false" | "(" ")" | "(" expr ")" | "[" (expr ("," expr)* ","?)? "]"
//!             | "{" expr (";" expr)* "}" | UPPER "{" (LOWER ":" expr),* "}" | IDENT | IDENT "(" args ")"

use crate::lexer::{lex, Tok, Token, KEYWORDS};
use qal_core::ast::*;
use qal_core::diag::Diagnostic;

/// Maximum AST height (and parser nesting). Every later stage only sees parser-built ASTs and
/// recurses over them, so bounding this here bounds recursion everywhere.
const MAX_DEPTH: usize = 256;

pub fn parse_program(src: &str) -> Result<Program, Diagnostic> {
    let tokens = lex(src)?;
    let mut p = Parser { tokens, pos: 0, depth: 0, height: 0, no_struct: false };
    let mut defs = Vec::new();
    while p.peek() != &Tok::Eof {
        defs.push(p.def()?);
    }
    Ok(Program { defs })
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    /// Current recursion depth of expr()/unary()/pattern()/ty().
    depth: usize,
    /// Height of the AST most recently built.
    height: usize,
    /// True while parsing a match scrutinee: `Name {` is not a record literal there.
    no_struct: bool,
}

fn is_upper(s: &str) -> bool {
    s.starts_with(|c: char| c.is_ascii_uppercase())
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.tokens[self.pos].tok
    }

    fn peek_at(&self, n: usize) -> &Tok {
        &self.tokens[(self.pos + n).min(self.tokens.len() - 1)].tok
    }

    fn next(&mut self) -> Token {
        let t = self.tokens[self.pos].clone();
        if t.tok != Tok::Eof {
            self.pos += 1;
        }
        t
    }

    fn err_here(&self, message: impl Into<String>) -> Diagnostic {
        let t = &self.tokens[self.pos];
        Diagnostic::parse(t.line, t.col, message)
    }

    #[cold]
    fn too_deep(&self) -> Diagnostic {
        self.err_here(format!("expression nested too deeply (max {MAX_DEPTH} levels)"))
    }

    fn enter(&mut self) -> Result<(), Diagnostic> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.too_deep());
        }
        Ok(())
    }

    /// Records a new AST node whose tallest child has height `child`.
    fn node(&mut self, child: usize) -> Result<(), Diagnostic> {
        self.height = child + 1;
        if self.height > MAX_DEPTH {
            return Err(self.too_deep());
        }
        Ok(())
    }

    fn describe(tok: &Tok) -> String {
        match tok {
            Tok::Ident(s) => format!("`{s}`"),
            Tok::Int(n) => format!("`{n}`"),
            Tok::Float(x) => format!("`{x:?}`"),
            Tok::Str(s) => format!("{s:?}"),
            Tok::Sym(s) => format!("`{s}`"),
            Tok::Eof => "end of input".to_string(),
        }
    }

    fn is_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Tok::Ident(s) if s == kw)
    }

    fn is_sym(&self, sym: &str) -> bool {
        matches!(self.peek(), Tok::Sym(s) if *s == sym)
    }

    fn expect_kw(&mut self, kw: &str) -> Result<(), Diagnostic> {
        if self.is_kw(kw) {
            self.next();
            Ok(())
        } else {
            Err(self.err_here(format!("expected `{kw}`, found {}", Self::describe(self.peek()))))
        }
    }

    fn expect_sym(&mut self, sym: &str) -> Result<(), Diagnostic> {
        if self.is_sym(sym) {
            self.next();
            Ok(())
        } else {
            Err(self.err_here(format!("expected `{sym}`, found {}", Self::describe(self.peek()))))
        }
    }

    fn ident(&mut self) -> Result<String, Diagnostic> {
        match self.peek().clone() {
            Tok::Ident(s) if !KEYWORDS.contains(&s.as_str()) => {
                self.next();
                Ok(s)
            }
            other => Err(self.err_here(format!("expected a name, found {}", Self::describe(&other)))),
        }
    }

    /// A name for a function, test, parameter, local, field or effect.
    fn lower(&mut self, what: &str) -> Result<String, Diagnostic> {
        if matches!(self.peek(), Tok::Ident(s) if is_upper(s)) {
            return Err(self.err_here(format!("{what} names start with a lowercase letter, found {}", Self::describe(self.peek()))));
        }
        self.ident()
    }

    /// A name for a type or variant.
    fn upper(&mut self, what: &str) -> Result<String, Diagnostic> {
        if matches!(self.peek(), Tok::Ident(s) if !is_upper(s)) {
            return Err(self.err_here(format!("{what} names start with an uppercase letter, found {}", Self::describe(self.peek()))));
        }
        self.ident()
    }

    /// Parses `open item (sep item)* sep? close`, calling `item` for each element.
    fn list_of<T>(&mut self, close: &str, mut item: impl FnMut(&mut Self) -> Result<T, Diagnostic>) -> Result<Vec<T>, Diagnostic> {
        let mut out = Vec::new();
        while !self.is_sym(close) {
            out.push(item(self)?);
            if !self.is_sym(close) {
                self.expect_sym(",")?;
            }
        }
        self.expect_sym(close)?;
        Ok(out)
    }

    fn ty(&mut self) -> Result<Type, Diagnostic> {
        self.enter()?;
        let t = self.ty_inner();
        self.depth -= 1;
        t
    }

    fn ty_inner(&mut self) -> Result<Type, Diagnostic> {
        let name = match self.peek().clone() {
            Tok::Sym("(") => {
                self.next();
                let t = self.ty()?;
                self.expect_sym(")")?;
                return Ok(t);
            }
            Tok::Ident(s) if s == "fn" => {
                self.next();
                self.expect_sym("(")?;
                let params = self.list_of(")", |p| p.ty())?;
                self.expect_sym("->")?;
                let ret = self.ty()?;
                let effects = self.effects()?;
                return Ok(Type::Fn(params, Box::new(ret), effects));
            }
            Tok::Ident(s) if is_upper(&s) => s,
            other => return Err(self.err_here(format!("expected a type (e.g. Int, Str, List[Int]), found {}", Self::describe(&other)))),
        };
        self.next();
        let arg = |p: &mut Self| -> Result<Type, Diagnostic> {
            p.expect_sym("[")?;
            let t = p.ty()?;
            Ok(t)
        };
        Ok(match name.as_str() {
            "Int" => Type::Int,
            "Float" => Type::Float,
            "Bool" => Type::Bool,
            "Unit" => Type::Unit,
            "Str" => Type::Str,
            "List" | "Option" => {
                let t = arg(self)?;
                self.expect_sym("]")?;
                if name == "List" { Type::list(t) } else { Type::option(t) }
            }
            "Result" => {
                let t = arg(self)?;
                self.expect_sym(",")?;
                let e = self.ty()?;
                self.expect_sym("]")?;
                Type::result(t, e)
            }
            _ => Type::Named(name),
        })
    }

    fn effects(&mut self) -> Result<Vec<String>, Diagnostic> {
        let mut effects = Vec::new();
        while self.is_sym("!") {
            self.next();
            effects.push(self.lower("effect")?);
        }
        Ok(effects)
    }

    fn def(&mut self) -> Result<Def, Diagnostic> {
        if self.is_kw("fn") {
            self.next();
            self.fn_def().map(Def::Fn)
        } else if self.is_kw("test") {
            self.next();
            let name = self.lower("test")?;
            self.expect_sym("=")?;
            let body = self.expr()?;
            Ok(Def::Test(TestDef { name, body }))
        } else if self.is_kw("type") {
            self.next();
            self.type_def().map(Def::Type)
        } else {
            Err(self.err_here(format!("expected `fn`, `type` or `test`, found {}", Self::describe(self.peek()))))
        }
    }

    fn type_def(&mut self) -> Result<TypeDef, Diagnostic> {
        let name = self.upper("type")?;
        self.expect_sym("=")?;
        if self.is_sym("{") {
            self.next();
            let fields = self.list_of("}", |p| {
                let f = p.lower("field")?;
                p.expect_sym(":")?;
                Ok((f, p.ty()?))
            })?;
            return Ok(TypeDef { name, body: TypeBody::Record(fields) });
        }
        if self.is_sym("|") {
            self.next();
        }
        let mut variants = Vec::new();
        loop {
            let vname = self.upper("variant")?;
            let fields = if self.is_sym("(") {
                self.next();
                self.list_of(")", |p| p.ty())?
            } else {
                Vec::new()
            };
            variants.push(Variant { name: vname, fields });
            if !self.is_sym("|") {
                break;
            }
            self.next();
        }
        Ok(TypeDef { name, body: TypeBody::Enum(variants) })
    }

    fn fn_def(&mut self) -> Result<FnDef, Diagnostic> {
        let name = self.lower("function")?;
        self.expect_sym("(")?;
        let params = self.list_of(")", |p| {
            let pname = p.lower("parameter")?;
            p.expect_sym(":")?;
            Ok(Param { name: pname, ty: p.ty()? })
        })?;
        self.expect_sym("->")?;
        let ret = self.ty()?;
        let effects = self.effects()?;
        let (mut req, mut ens) = (Vec::new(), Vec::new());
        loop {
            if self.is_kw("req") {
                self.next();
                req.push(self.expr()?);
            } else if self.is_kw("ens") {
                self.next();
                ens.push(self.expr()?);
            } else {
                break;
            }
        }
        self.expect_sym("=")?;
        let body = self.expr()?;
        Ok(FnDef { name, params, ret, effects, req, ens, body })
    }

    fn expr(&mut self) -> Result<Expr, Diagnostic> {
        self.enter()?;
        let e = self.expr_inner();
        self.depth -= 1;
        e
    }

    fn expr_inner(&mut self) -> Result<Expr, Diagnostic> {
        if self.is_kw("let") {
            self.let_expr()
        } else if self.is_kw("if") {
            self.if_expr()
        } else if self.is_kw("match") {
            self.match_expr()
        } else {
            self.binary(1)
        }
    }

    // let/if/match/call/paren live in their own functions so the frames on the common recursion path stay small.
    fn let_expr(&mut self) -> Result<Expr, Diagnostic> {
        self.next();
        let name = self.lower("variable")?;
        let ann = if self.is_sym(":") {
            self.next();
            Some(self.ty()?)
        } else {
            None
        };
        self.expect_sym("=")?;
        let value = self.expr()?;
        self.expect_kw("in")?;
        let value_h = self.height;
        let body = self.expr()?;
        self.node(value_h.max(self.height))?;
        Ok(Expr::Let(name, ann, Box::new(value), Box::new(body)))
    }

    fn if_expr(&mut self) -> Result<Expr, Diagnostic> {
        self.next();
        let c = self.expr()?;
        let mut h = self.height;
        self.expect_kw("then")?;
        let t = self.expr()?;
        h = h.max(self.height);
        self.expect_kw("else")?;
        let e = self.expr()?;
        self.node(h.max(self.height))?;
        Ok(Expr::If(Box::new(c), Box::new(t), Box::new(e)))
    }

    fn match_expr(&mut self) -> Result<Expr, Diagnostic> {
        self.next();
        let saved = std::mem::replace(&mut self.no_struct, true);
        let scrutinee = self.expr();
        self.no_struct = saved;
        let scrutinee = scrutinee?;
        let mut h = self.height;
        self.expect_sym("{")?;
        let arms = self.list_of("}", |p| {
            let pat = p.pattern()?;
            h = h.max(p.height);
            let guard = if p.is_kw("if") {
                p.next();
                let g = p.expr()?;
                h = h.max(p.height);
                Some(g)
            } else {
                None
            };
            p.expect_sym("=>")?;
            let body = p.expr()?;
            h = h.max(p.height);
            Ok(Arm { pat, guard, body })
        })?;
        if arms.is_empty() {
            return Err(self.err_here("a match needs at least one arm"));
        }
        self.node(h)?;
        Ok(Expr::Match(Box::new(scrutinee), arms))
    }

    fn pattern(&mut self) -> Result<Pat, Diagnostic> {
        self.enter()?;
        let p = self.pattern_inner();
        self.depth -= 1;
        p
    }

    fn pattern_inner(&mut self) -> Result<Pat, Diagnostic> {
        self.height = 1;
        match self.peek().clone() {
            Tok::Int(n) => {
                self.next();
                Ok(Pat::Int(n))
            }
            Tok::Sym("-") if matches!(self.peek_at(1), Tok::Int(_)) => {
                self.next();
                let Tok::Int(n) = self.next().tok else { unreachable!() };
                Ok(Pat::Int(-n))
            }
            Tok::Str(s) => {
                self.next();
                Ok(Pat::Str(s))
            }
            Tok::Sym("(") if self.peek_at(1) == &Tok::Sym(")") => {
                self.next();
                self.next();
                Ok(Pat::Unit)
            }
            Tok::Ident(s) if s == "true" || s == "false" => {
                self.next();
                Ok(Pat::Bool(s == "true"))
            }
            Tok::Ident(s) if s == "_" => {
                self.next();
                Ok(Pat::Wild)
            }
            Tok::Ident(s) if is_upper(&s) => {
                self.next();
                if self.is_sym("(") {
                    self.next();
                    let mut h = 0;
                    let ps = self.list_of(")", |p| {
                        let x = p.pattern()?;
                        h = h.max(p.height);
                        Ok(x)
                    })?;
                    self.node(h)?;
                    Ok(Pat::Variant(s, ps))
                } else if self.is_sym("{") {
                    self.next();
                    let mut h = 0;
                    let fs = self.list_of("}", |p| {
                        let f = p.lower("field")?;
                        let x = if p.is_sym(":") {
                            p.next();
                            p.pattern()?
                        } else {
                            Pat::Bind(f.clone())
                        };
                        h = h.max(p.height);
                        Ok((f, x))
                    })?;
                    self.node(h)?;
                    Ok(Pat::Record(s, fs))
                } else {
                    Ok(Pat::Variant(s, vec![]))
                }
            }
            Tok::Ident(_) => Ok(Pat::Bind(self.lower("variable")?)),
            other => Err(self.err_here(format!("expected a pattern, found {}", Self::describe(&other)))),
        }
    }

    fn peek_binop(&self) -> Option<BinOp> {
        let op = match self.peek() {
            Tok::Sym("+") => BinOp::Add,
            Tok::Sym("-") => BinOp::Sub,
            Tok::Sym("*") => BinOp::Mul,
            Tok::Sym("/") => BinOp::Div,
            Tok::Sym("%") => BinOp::Rem,
            Tok::Sym("++") => BinOp::Concat,
            Tok::Sym("==") => BinOp::Eq,
            Tok::Sym("!=") => BinOp::Ne,
            Tok::Sym("<") => BinOp::Lt,
            Tok::Sym("<=") => BinOp::Le,
            Tok::Sym(">") => BinOp::Gt,
            Tok::Sym(">=") => BinOp::Ge,
            Tok::Sym("&&") => BinOp::And,
            Tok::Sym("||") => BinOp::Or,
            _ => return None,
        };
        Some(op)
    }

    fn binary(&mut self, min_prec: u8) -> Result<Expr, Diagnostic> {
        let mut lhs = self.unary()?;
        while let Some(op) = self.peek_binop() {
            let prec = op.precedence();
            if prec < min_prec {
                break;
            }
            self.next();
            let lhs_h = self.height;
            let rhs = self.binary(prec + 1)?;
            self.node(lhs_h.max(self.height))?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
            if op.is_comparison() && self.peek_binop().is_some_and(BinOp::is_comparison) {
                return Err(self.chained());
            }
        }
        Ok(lhs)
    }

    #[cold]
    fn chained(&self) -> Diagnostic {
        self.err_here("comparisons cannot be chained; use `&&`")
    }

    fn unary(&mut self) -> Result<Expr, Diagnostic> {
        let op = if self.is_sym("-") {
            UnOp::Neg
        } else if self.is_sym("!") {
            UnOp::Not
        } else {
            return self.postfix();
        };
        self.next();
        self.enter()?;
        let inner = self.unary()?;
        self.depth -= 1;
        self.node(self.height)?;
        Ok(Expr::Un(op, Box::new(inner)))
    }

    fn postfix(&mut self) -> Result<Expr, Diagnostic> {
        let mut e = self.atom()?;
        while self.is_sym(".") {
            self.next();
            let field = self.lower("field")?;
            self.node(self.height)?;
            e = Expr::Field(Box::new(e), field);
        }
        Ok(e)
    }

    fn atom(&mut self) -> Result<Expr, Diagnostic> {
        self.height = 1;
        match self.peek().clone() {
            Tok::Int(n) => {
                self.next();
                Ok(Expr::Int(n))
            }
            Tok::Float(x) => {
                self.next();
                Ok(Expr::Float(x))
            }
            Tok::Str(s) => {
                self.next();
                Ok(Expr::Str(s))
            }
            Tok::Ident(s) if s == "true" || s == "false" => {
                self.next();
                Ok(Expr::Bool(s == "true"))
            }
            Tok::Sym("(") => self.nested(Self::paren),
            Tok::Sym("[") => self.nested(Self::list_lit),
            Tok::Sym("{") => self.nested(Self::seq),
            Tok::Ident(s) if KEYWORDS.contains(&s.as_str()) => Err(self.expected_expr(&Tok::Ident(s))),
            Tok::Ident(s) if is_upper(&s) && !self.no_struct && self.peek_at(1) == &Tok::Sym("{") => self.nested(Self::record_lit),
            Tok::Ident(_) => self.nested(Self::name_or_call),
            other => Err(self.expected_expr(&other)),
        }
    }

    /// Runs `f` with record literals allowed again (inside delimiters the match-scrutinee restriction ends).
    fn nested(&mut self, f: fn(&mut Self) -> Result<Expr, Diagnostic>) -> Result<Expr, Diagnostic> {
        let saved = std::mem::replace(&mut self.no_struct, false);
        let e = f(self);
        self.no_struct = saved;
        e
    }

    #[cold]
    fn expected_expr(&self, found: &Tok) -> Diagnostic {
        let mut msg = format!("expected an expression, found {}", Self::describe(found));
        if let Tok::Ident(k) = found {
            if k == "if" || k == "let" || k == "match" {
                msg.push_str(&format!(" (wrap the `{k}` expression in parentheses here)"));
            }
        }
        self.err_here(msg)
    }

    fn paren(&mut self) -> Result<Expr, Diagnostic> {
        self.next();
        if self.is_sym(")") {
            self.next();
            self.height = 1;
            return Ok(Expr::Unit);
        }
        let e = self.expr()?;
        self.expect_sym(")")?;
        Ok(e)
    }

    fn list_lit(&mut self) -> Result<Expr, Diagnostic> {
        self.next();
        let mut h = 0;
        let items = self.list_of("]", |p| {
            let e = p.expr()?;
            h = h.max(p.height);
            Ok(e)
        })?;
        self.node(h)?;
        Ok(Expr::List(items))
    }

    fn seq(&mut self) -> Result<Expr, Diagnostic> {
        self.next();
        let mut items = vec![self.expr()?];
        let mut h = self.height;
        while self.is_sym(";") {
            self.next();
            items.push(self.expr()?);
            h = h.max(self.height);
        }
        self.expect_sym("}")?;
        self.node(h)?;
        Ok(Expr::Seq(items))
    }

    fn record_lit(&mut self) -> Result<Expr, Diagnostic> {
        let name = self.upper("type")?;
        self.expect_sym("{")?;
        let mut h = 0;
        let fields = self.list_of("}", |p| {
            let f = p.lower("field")?;
            p.expect_sym(":")?;
            let e = p.expr()?;
            h = h.max(p.height);
            Ok((f, e))
        })?;
        self.node(h)?;
        Ok(Expr::Record(name, fields))
    }

    fn name_or_call(&mut self) -> Result<Expr, Diagnostic> {
        let name = self.ident()?;
        if !self.is_sym("(") {
            self.height = 1;
            return Ok(Expr::Var(name));
        }
        self.next();
        let mut h = 0;
        let args = self.list_of(")", |p| {
            let e = p.expr()?;
            h = h.max(p.height);
            Ok(e)
        })?;
        self.node(h)?;
        Ok(Expr::Call(name, args))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(n: i64) -> Box<Expr> {
        Box::new(Expr::Int(n))
    }
    fn var(s: &str) -> Box<Expr> {
        Box::new(Expr::Var(s.into()))
    }
    fn body(src: &str) -> Expr {
        let p = parse_program(&format!("test t = {src}")).unwrap();
        let Def::Test(t) = &p.defs[0] else { panic!() };
        t.body.clone()
    }

    #[test]
    fn parses_function_with_contracts_and_effects() {
        let p = parse_program("fn f(x: Int, b: Bool) -> Int !io req x >= 0 ens r > x = x + 1").unwrap();
        let Def::Fn(f) = &p.defs[0] else { panic!("expected fn") };
        assert_eq!(f.name, "f");
        assert_eq!(f.params, vec![Param { name: "x".into(), ty: Type::Int }, Param { name: "b".into(), ty: Type::Bool }]);
        assert_eq!(f.ret, Type::Int);
        assert_eq!(f.effects, vec!["io".to_string()]);
        assert_eq!(f.req, vec![Expr::Bin(BinOp::Ge, var("x"), int(0))]);
        assert_eq!(f.ens, vec![Expr::Bin(BinOp::Gt, var("r"), var("x"))]);
        assert_eq!(f.body, Expr::Bin(BinOp::Add, var("x"), int(1)));
    }

    #[test]
    fn respects_precedence_and_left_associativity() {
        let sum = Expr::Bin(BinOp::Sub, Box::new(Expr::Bin(BinOp::Add, int(1), Box::new(Expr::Bin(BinOp::Mul, int(2), int(3))))), int(4));
        let cmp = Expr::Bin(BinOp::Eq, Box::new(sum), int(3));
        assert_eq!(body("1 + 2 * 3 - 4 == 3 || false"), Expr::Bin(BinOp::Or, Box::new(cmp), Box::new(Expr::Bool(false))));
        let s = |x: &str| Box::new(Expr::Str(x.into()));
        assert_eq!(body(r#""a" ++ "b" == "ab""#), Expr::Bin(BinOp::Eq, Box::new(Expr::Bin(BinOp::Concat, s("a"), s("b"))), s("ab")));
    }

    #[test]
    fn parses_let_if_calls_unary_and_unit() {
        let p = parse_program("fn g() -> Unit = ()\ntest t = let y = -f(1, 2) in if !true then y else y").unwrap();
        assert_eq!(p.defs.len(), 2);
        let Def::Test(t) = &p.defs[1] else { panic!() };
        let call = Expr::Call("f".into(), vec![Expr::Int(1), Expr::Int(2)]);
        let neg = Expr::Un(UnOp::Neg, Box::new(call));
        let cond = Expr::Un(UnOp::Not, Box::new(Expr::Bool(true)));
        let iff = Expr::If(Box::new(cond), var("y"), var("y"));
        assert_eq!(t.body, Expr::Let("y".into(), None, Box::new(neg), Box::new(iff)));
    }

    #[test]
    fn parses_type_definitions_and_type_syntax() {
        let p = parse_program(
            "type User = { id: Int, name: Str, }\n\
             type Err = Empty | NotNumber(Str) | Pair(Int, List[Float])\n\
             type One = | Only\n\
             fn f(u: User, k: fn(Int) -> Option[Int] !io, r2: Result[Unit, Str]) -> List[Err] = []",
        )
        .unwrap();
        assert_eq!(
            p.defs[0],
            Def::Type(TypeDef { name: "User".into(), body: TypeBody::Record(vec![("id".into(), Type::Int), ("name".into(), Type::Str)]) })
        );
        let Def::Type(TypeDef { body: TypeBody::Enum(vs), .. }) = &p.defs[1] else { panic!() };
        assert_eq!(vs.iter().map(|v| (v.name.as_str(), v.fields.len())).collect::<Vec<_>>(), vec![("Empty", 0), ("NotNumber", 1), ("Pair", 2)]);
        assert_eq!(vs[2].fields[1], Type::list(Type::Float));
        let Def::Type(TypeDef { body: TypeBody::Enum(one), .. }) = &p.defs[2] else { panic!() };
        assert_eq!(one.len(), 1);
        let Def::Fn(f) = &p.defs[3] else { panic!() };
        assert_eq!(f.params[1].ty, Type::Fn(vec![Type::Int], Box::new(Type::option(Type::Int)), vec!["io".into()]));
        assert_eq!(f.params[2].ty, Type::result(Type::Unit, Type::Str));
        assert_eq!(f.ret, Type::list(Type::Named("Err".into())));
    }

    #[test]
    fn parses_data_expressions() {
        assert_eq!(body("[1, 2,]"), Expr::List(vec![Expr::Int(1), Expr::Int(2)]));
        assert_eq!(body("[]"), Expr::List(vec![]));
        assert_eq!(body("{ print(\"a\"); 1 }"), Expr::Seq(vec![Expr::Call("print".into(), vec![Expr::Str("a".into())]), Expr::Int(1)]));
        assert_eq!(
            body("User { id: 1, name: \"a\" }.name"),
            Expr::Field(Box::new(Expr::Record("User".into(), vec![("id".into(), Expr::Int(1)), ("name".into(), Expr::Str("a".into()))])), "name".into())
        );
        assert_eq!(body("-a.b.c"), Expr::Un(UnOp::Neg, Box::new(Expr::Field(Box::new(Expr::Field(var("a"), "b".into())), "c".into()))));
        assert_eq!(body("let x: Option[Int] = None in x"), Expr::Let("x".into(), Some(Type::option(Type::Int)), var("None"), var("x")));
        assert_eq!(body("1.5"), Expr::Float(1.5));
    }

    #[test]
    fn parses_match_with_patterns_and_guards() {
        let e = body("match get(xs, 0) { None => 0, Some(n) if n < 0 => -1, Some(_) => 1, }");
        let Expr::Match(scrut, arms) = e else { panic!() };
        assert_eq!(*scrut, Expr::Call("get".into(), vec![Expr::Var("xs".into()), Expr::Int(0)]));
        assert_eq!(arms.len(), 3);
        assert_eq!(arms[0].pat, Pat::Variant("None".into(), vec![]));
        assert_eq!(arms[1].pat, Pat::Variant("Some".into(), vec![Pat::Bind("n".into())]));
        assert_eq!(arms[1].guard, Some(Expr::Bin(BinOp::Lt, var("n"), int(0))));
        assert_eq!(arms[2].pat, Pat::Variant("Some".into(), vec![Pat::Wild]));
        let Expr::Match(_, arms) = body("match u { User { id: 1, name } => name, _ => \"\" }") else { panic!() };
        assert_eq!(arms[0].pat, Pat::Record("User".into(), vec![("id".into(), Pat::Int(1)), ("name".into(), Pat::Bind("name".into()))]));
        let Expr::Match(_, arms) = body("match x { -1 => 0, \"s\" => 1, true => 2, () => 3 }") else { panic!() };
        let pats: Vec<Pat> = arms.into_iter().map(|a| a.pat).collect();
        assert_eq!(pats, vec![Pat::Int(-1), Pat::Str("s".into()), Pat::Bool(true), Pat::Unit]);
    }

    #[test]
    fn record_literal_is_not_parsed_in_match_scrutinee() {
        // `match Empty { ... }`: the braces open the arms, not a record literal.
        let Expr::Match(scrut, arms) = body("match Empty { Empty => 1, _ => 2 }") else { panic!() };
        assert_eq!(*scrut, Expr::Var("Empty".into()));
        assert_eq!(arms.len(), 2);
        // Inside parentheses record literals are allowed again.
        let Expr::Match(scrut, _) = body("match (P { x: 1 }) { _ => 1 }") else { panic!() };
        assert_eq!(*scrut, Expr::Record("P".into(), vec![("x".into(), Expr::Int(1))]));
    }

    #[test]
    fn reports_errors_with_position() {
        let e = parse_program("fn f(x: Int) -> Int = (x + 1").unwrap_err();
        assert_eq!(e.kind, "parse");
        assert_eq!(e.path, "1:29");
        assert!(e.message.contains("expected `)`"), "{}", e.message);
        let e = parse_program("fn if() -> Int = 1").unwrap_err();
        assert!(e.message.contains("expected a name"), "{}", e.message);
        let e = parse_program("test t = 1 < 2 < 3").unwrap_err();
        assert!(e.message.contains("chained"), "{}", e.message);
        let e = parse_program("fn f() -> 3 = 1").unwrap_err();
        assert!(e.message.contains("expected a type"), "{}", e.message);
    }

    #[test]
    fn keyword_in_expression_position_is_expected_an_expression() {
        let e = parse_program("fn f() -> Int = then").unwrap_err();
        assert_eq!(e.message, "expected an expression, found `then`");
        let e = parse_program("test t = 1 + if true then 1 else 2 == 2").unwrap_err();
        assert!(e.message.starts_with("expected an expression, found `if` (wrap"), "{}", e.message);
    }

    #[test]
    fn enforces_name_case() {
        assert!(parse_program("fn F() -> Int = 1").unwrap_err().message.contains("function names start with a lowercase letter"));
        assert!(parse_program("type user = { a: Int }").unwrap_err().message.contains("type names start with an uppercase letter"));
        assert!(parse_program("type U = a | B").unwrap_err().message.contains("variant names start with an uppercase letter"));
        assert!(parse_program("test t = let X = 1 in X").unwrap_err().message.contains("variable names"));
        assert!(parse_program("test t = {}").unwrap_err().message.contains("expected an expression, found `}`"));
        assert!(parse_program("test t = match x { }").unwrap_err().message.contains("at least one arm"));
    }

    /// Debug-build parser frames are large; give deep-but-legal inputs more room than a 2 MiB test thread.
    fn parse_big_stack(src: String) -> Result<Program, Diagnostic> {
        std::thread::Builder::new().stack_size(64 << 20).spawn(move || parse_program(&src)).unwrap().join().unwrap()
    }

    fn rejects_deep(src: String) {
        let e = parse_big_stack(src).unwrap_err();
        assert_eq!(e.kind, "parse");
        assert!(e.message.contains("nested too deeply (max 256 levels)"), "{}", e.message);
    }

    #[test]
    fn rejects_deeply_nested_parens() {
        let n = 10_000;
        rejects_deep(format!("test t = {}1{}", "(".repeat(n), ")".repeat(n)));
    }

    #[test]
    fn rejects_long_operator_chain() {
        let mut src = String::from("test t = 1");
        for _ in 0..10_000 {
            src.push_str(" + 1");
        }
        rejects_deep(src);
    }

    #[test]
    fn rejects_long_unary_chain() {
        rejects_deep(format!("test t = {}1", "-".repeat(10_000)));
    }

    #[test]
    fn rejects_deeply_nested_lets() {
        rejects_deep(format!("test t = {}1", "let x = 1 in ".repeat(10_000)));
    }

    #[test]
    fn rejects_deeply_nested_data_and_patterns() {
        rejects_deep(format!("test t = {}1{}", "[".repeat(10_000), "]".repeat(10_000)));
        rejects_deep(format!("test t = {}1{}", "{".repeat(10_000), "}".repeat(10_000)));
        rejects_deep(format!("test t = x{}", ".f".repeat(10_000)));
        rejects_deep(format!("test t = match x {{ {}_{} => 1 }}", "Some(".repeat(10_000), ")".repeat(10_000)));
        rejects_deep(format!("fn f(x: {}Int{}) -> Int = 1", "List[".repeat(10_000), "]".repeat(10_000)));
    }

    #[test]
    fn accepts_moderate_chains_and_nesting() {
        let mut src = String::from("test t = 1");
        for _ in 0..200 {
            src.push_str(" + 1");
        }
        parse_big_stack(src).unwrap();
        parse_big_stack(format!("test t = {}1{}", "(".repeat(200), ")".repeat(200))).unwrap();
    }

    #[test]
    fn empty_source_is_empty_program() {
        assert_eq!(parse_program("  # nothing\n").unwrap(), Program::default());
    }
}
```

Replace `qal/crates/qal-text/src/printer.rs` with:

```rust
//! Canonical printer: Program -> text. `parse_program(print_program(p)) == p`.

use qal_core::ast::*;

pub fn print_program(p: &Program) -> String {
    let parts: Vec<String> = p.defs.iter().map(print_def).collect();
    let mut out = parts.join("\n\n");
    if !out.is_empty() {
        out.push('\n');
    }
    out
}

fn print_def(d: &Def) -> String {
    match d {
        Def::Test(t) => format!("test {} = {}", t.name, print_expr(&t.body)),
        Def::Type(t) => match &t.body {
            TypeBody::Record(fields) => {
                let fs: Vec<String> = fields.iter().map(|(n, ty)| format!("{n}: {ty}")).collect();
                format!("type {} = {{ {} }}", t.name, fs.join(", "))
            }
            TypeBody::Enum(variants) => {
                let vs: Vec<String> = variants
                    .iter()
                    .map(|v| {
                        if v.fields.is_empty() {
                            v.name.clone()
                        } else {
                            let fs: Vec<String> = v.fields.iter().map(|f| f.to_string()).collect();
                            format!("{}({})", v.name, fs.join(", "))
                        }
                    })
                    .collect();
                format!("type {} = {}", t.name, vs.join(" | "))
            }
        },
        Def::Fn(f) => {
            let params: Vec<String> = f.params.iter().map(|p| format!("{}: {}", p.name, p.ty)).collect();
            let mut s = match f.ret {
                // `fn f() -> fn() -> Int !io` would give `!io` to the result type; parenthesize it.
                Type::Fn(..) => format!("fn {}({}) -> ({})", f.name, params.join(", "), f.ret),
                _ => format!("fn {}({}) -> {}", f.name, params.join(", "), f.ret),
            };
            for e in &f.effects {
                s.push_str(&format!(" !{e}"));
            }
            for c in &f.req {
                s.push_str(&format!("\n  req {}", print_expr(c)));
            }
            for c in &f.ens {
                s.push_str(&format!("\n  ens {}", print_expr(c)));
            }
            s.push_str(&format!("\n= {}", print_expr(&f.body)));
            s
        }
    }
}

/// Precedence of an expression's outermost construct (higher binds tighter).
fn prec(e: &Expr) -> u8 {
    match e {
        Expr::Let(..) | Expr::If(..) | Expr::Match(..) => 0,
        Expr::Bin(op, ..) => op.precedence(),
        Expr::Un(..) => 6,
        Expr::Field(..) => 8,
        _ => 9,
    }
}

fn wrap(e: &Expr, needs_parens: bool) -> String {
    let s = print_expr(e);
    if needs_parens {
        format!("({s})")
    } else {
        s
    }
}

/// True if `e` contains a record literal outside of any delimiter, which a match scrutinee must not.
fn has_bare_record(e: &Expr) -> bool {
    match e {
        Expr::Record(..) => true,
        Expr::Bin(_, l, r) => has_bare_record(l) || has_bare_record(r),
        Expr::Un(_, x) | Expr::Field(x, _) => has_bare_record(x),
        Expr::If(c, t, f) => has_bare_record(c) || has_bare_record(t) || has_bare_record(f),
        Expr::Let(_, _, v, b) => has_bare_record(v) || has_bare_record(b),
        Expr::Match(s, arms) => {
            has_bare_record(s) || arms.iter().any(|a| a.guard.as_ref().is_some_and(has_bare_record) || has_bare_record(&a.body))
        }
        _ => false,
    }
}

pub fn print_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            '\0' => out.push_str("\\0"),
            c if c.is_control() => out.push_str(&format!("\\u{{{:x}}}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

pub fn print_pat(p: &Pat) -> String {
    match p {
        Pat::Wild => "_".into(),
        Pat::Bind(n) => n.clone(),
        Pat::Int(n) => n.to_string(),
        Pat::Str(s) => print_str(s),
        Pat::Bool(b) => b.to_string(),
        Pat::Unit => "()".into(),
        Pat::Variant(n, ps) if ps.is_empty() => n.clone(),
        Pat::Variant(n, ps) => {
            let ps: Vec<String> = ps.iter().map(print_pat).collect();
            format!("{n}({})", ps.join(", "))
        }
        Pat::Record(n, fs) => {
            let fs: Vec<String> = fs
                .iter()
                .map(|(f, p)| match p {
                    Pat::Bind(b) if b == f => f.clone(),
                    _ => format!("{f}: {}", print_pat(p)),
                })
                .collect();
            format!("{n} {{ {} }}", fs.join(", "))
        }
    }
}

fn comma(items: &[Expr]) -> String {
    items.iter().map(print_expr).collect::<Vec<_>>().join(", ")
}

pub fn print_expr(e: &Expr) -> String {
    match e {
        Expr::Int(n) => n.to_string(),
        Expr::Float(x) => format!("{x:?}"),
        Expr::Bool(b) => b.to_string(),
        Expr::Str(s) => print_str(s),
        Expr::Unit => "()".to_string(),
        Expr::Var(v) => v.clone(),
        Expr::Call(f, args) => format!("{f}({})", comma(args)),
        Expr::List(items) => format!("[{}]", comma(items)),
        Expr::Seq(items) => format!("{{ {} }}", items.iter().map(print_expr).collect::<Vec<_>>().join("; ")),
        Expr::Record(n, fs) => {
            let fs: Vec<String> = fs.iter().map(|(f, v)| format!("{f}: {}", print_expr(v))).collect();
            format!("{n} {{ {} }}", fs.join(", "))
        }
        Expr::Field(t, f) => format!("{}.{f}", wrap(t, prec(t) < 8)),
        Expr::Un(op, x) => {
            let sym = match op {
                UnOp::Neg => "-",
                UnOp::Not => "!",
            };
            format!("{sym}{}", wrap(x, prec(x) < 6))
        }
        Expr::Bin(op, l, r) => {
            let p = op.precedence();
            // Left-associative: the left side may share our precedence, the right may not.
            // Comparisons are non-associative: neither side may share it.
            let left_parens = prec(l) < p || (op.is_comparison() && prec(l) == p);
            let right_parens = prec(r) <= p;
            format!("{} {} {}", wrap(l, left_parens), op.symbol(), wrap(r, right_parens))
        }
        Expr::If(c, t, f) => format!("if {} then {} else {}", print_expr(c), print_expr(t), print_expr(f)),
        Expr::Let(n, ann, v, b) => match ann {
            Some(t) => format!("let {n}: {t} = {} in {}", print_expr(v), print_expr(b)),
            None => format!("let {n} = {} in {}", print_expr(v), print_expr(b)),
        },
        Expr::Match(s, arms) => {
            let arms: Vec<String> = arms
                .iter()
                .map(|a| match &a.guard {
                    Some(g) => format!("{} if {} => {}", print_pat(&a.pat), print_expr(g), print_expr(&a.body)),
                    None => format!("{} => {}", print_pat(&a.pat), print_expr(&a.body)),
                })
                .collect();
            format!("match {} {{ {} }}", wrap(s, has_bare_record(s)), arms.join(", "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_program;

    fn roundtrip(src: &str) {
        let p = parse_program(src).unwrap();
        let printed = print_program(&p);
        let reparsed = parse_program(&printed).unwrap_or_else(|e| panic!("reparse failed: {e:?}\n{printed}"));
        assert_eq!(reparsed, p, "printed:\n{printed}");
    }

    #[test]
    fn prints_canonical_form() {
        let p = parse_program("fn  add(a:Int,b:Int)->Int !io req a>=0 = a+b*2").unwrap();
        assert_eq!(print_program(&p), "fn add(a: Int, b: Int) -> Int !io\n  req a >= 0\n= a + b * 2\n");
    }

    #[test]
    fn keeps_needed_parentheses_only() {
        let p = parse_program("test t = (1 + 2) * 3 - (4 - 5) == ((6))").unwrap();
        assert_eq!(print_program(&p), "test t = (1 + 2) * 3 - (4 - 5) == 6\n");
    }

    #[test]
    fn prints_data_definitions_and_match() {
        let src = "type   E =|A|B( Int,Str )\ntype R={x:Float,}\n\
                   fn f(e: E) -> Str = match e {A=>\"a\\n\",B(n,s) if n>0=>s,_=>\"\"}";
        let p = parse_program(src).unwrap();
        assert_eq!(
            print_program(&p),
            "type E = A | B(Int, Str)\n\ntype R = { x: Float }\n\n\
             fn f(e: E) -> Str\n= match e { A => \"a\\n\", B(n, s) if n > 0 => s, _ => \"\" }\n"
        );
    }

    #[test]
    fn roundtrips_all_constructs() {
        roundtrip("fn f(x: Int) -> Int req x > 0 ens r >= 1 = if x == 1 then 1 else x * f(x - 1)");
        roundtrip("test t = let a = (if true then 1 else 2) in a + (let b = 3 in b)");
        roundtrip("test t = -(-1) == 1 && !(1 < 2) || (1 == 1) == true");
        roundtrip("fn u() -> Unit !io !fs = ()");
        roundtrip("test t = 1 - (2 - 3) - 4 * (5 % 2) / 1");
        roundtrip("type User = { id: Int, tags: List[Str] }\ntype T = Leaf | Node(T, Int, T)");
        roundtrip("fn g(k: fn(Int, Str) -> Option[Int] !io, r2: Result[Unit, Str]) -> List[Float] = [1.5, 0.1, 1e300, 2.5e-7]");
        roundtrip("fn f() -> (fn(Int) -> Int) !io = g\nfn h() -> fn(Int) -> Int !io = g\nfn k(x: fn() -> (fn() -> Int !io)) -> Int = 1");
        roundtrip(r#"test t = "q\"\\\n\t\r\0\u{7}é" ++ "x" == "y""#);
        roundtrip("test t = { print(\"a\"); { 1 }; 2 } == (User { id: 1, n: [] }).id");
        roundtrip("test t = (-a).b.c == -a.b.c && (a + b).c == 1");
        roundtrip("test t = let x: Option[List[Int]] = None in match x { Some(_) => 0, None => 1 }");
        roundtrip("test t = match (P { x: 1 }) { P { x: 1, y } => y, P { x: n } if n < -1 => -1, _ => match y { -3 => 1, () => 2 } }");
        roundtrip("test t = match (P { x: 1 }.x + 1) { _ => (match a { _ => 1 }) + 1 }");
        roundtrip("test t = match f(P { x: 1 }) { \"s\" => true, _ => false }");
    }
}
```

Replace `qal/crates/qal-text/src/lib.rs` with:

```rust
//! QAL text form: a token-dense view of the program graph for LLM agents.

pub mod lexer;
pub mod parser;
pub mod printer;

pub use parser::parse_program;
pub use printer::{print_expr, print_pat, print_program};
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd qal && cargo test -p qal-text 2>&1 | grep -E "test result|warning"`
Expected: `test result: ok. 29 passed`, no warnings.

- [ ] **Step 5: Commit**

```bash
git add qal/crates/qal-text
git commit -m "feat(qal): 1b text form — floats, strings, lists, records, enums, match, sequences

Rejects 10abc at the lexer, reports keywords in expression position as
'expected an expression', and keeps the parse/print round trip for every construct.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Checker with inference, resolution and exhaustiveness (`qal-check`)

**Files:**
- Modify (full rewrite): `qal/crates/qal-check/src/lib.rs`
- Create: `qal/crates/qal-check/src/unify.rs`, `qal/crates/qal-check/src/infer.rs`, `qal/crates/qal-check/src/builtins.rs`, `qal/crates/qal-check/src/exhaust.rs`
- Test: `qal/crates/qal-check/src/tests.rs`, inline modules in `unify.rs` and `exhaust.rs`

**Interfaces:**
- Consumes: Task 1 `ast`, `tast`, `Diagnostic`; dev: `qal_text::parse_program`.
- Produces: `qal_check::check(&Program) -> Result<TProgram, Vec<Diagnostic>>` (the typed program; every `TExpr::ty` fully resolved, `len` resolved to `Builtin::StrLen`/`ListLen`), `qal_check::check_program(&Program) -> Vec<Diagnostic>` (unchanged signature), `KNOWN_EFFECTS`, `RESULT_VAR`, `Sig { params, ret, effects }` with `fn_type()`. Paths extend the 1a scheme: `items[i]`, `fields.<name>`, `target`, `scrutinee`, `arms[i].pattern|guard|body`, `params`, `ret`, `effects[i]`, `variants[i].fields[j]`. Name resolution order for `x` / `x(...)`: local, top-level fn, variant, built-in. Built-in function names, built-in type names and the runtime exports `memory`/`qal_alloc` are reserved.

- [ ] **Step 1: Write the failing tests**

Create `qal/crates/qal-check/src/tests.rs`:

```rust
use super::*;
use qal_core::tast::{TKind, TPat};
use qal_text::parse_program;

fn check_src(src: &str) -> Vec<Diagnostic> {
    check_program(&parse_program(src).unwrap())
}

fn typed(src: &str) -> TProgram {
    check(&parse_program(src).unwrap()).unwrap_or_else(|d| panic!("unexpected diagnostics: {d:#?}"))
}

fn kinds(src: &str) -> Vec<(&'static str, String, String)> {
    check_src(src).into_iter().map(|d| (d.kind, d.def, d.path)).collect()
}

fn one(src: &str) -> Diagnostic {
    let d = check_src(src);
    assert_eq!(d.len(), 1, "{d:#?}");
    d.into_iter().next().unwrap()
}

#[test]
fn accepts_well_typed_program() {
    let src = "fn fact(n: Int) -> Int req n >= 0 ens r >= 1 = if n <= 1 then 1 else n * fact(n - 1)\n\
               test fact_5 = fact(5) == 120\n\
               fn noop() -> Unit = ()\n\
               test unit_eq = noop() == ()";
    assert_eq!(check_src(src), vec![]);
}

#[test]
fn reports_type_mismatch_with_path() {
    let d = one("fn f(x: Int) -> Int = if x then 1 else 2");
    assert_eq!((d.kind, d.path.as_str()), ("type_mismatch", "body.cond"));
    assert_eq!((d.expected.as_deref(), d.found.as_deref()), (Some("Bool"), Some("Int")));
    assert_eq!(d.hints, vec!["compare explicitly, e.g. `x != 0`".to_string()]);
}

#[test]
fn reports_unknown_names_arity_and_duplicates() {
    assert_eq!(kinds("fn f() -> Int = y"), vec![("unknown_name", "f".into(), "body".into())]);
    assert_eq!(kinds("fn f() -> Int = g()"), vec![("unknown_name", "f".into(), "body".into())]);
    let d = check_src("fn g(a: Int) -> Int = a\nfn f() -> Int = g(1, 2)");
    assert_eq!((d[0].code, d[0].kind), ("E0204", "type_mismatch"));
    assert_eq!(kinds("fn f() -> Int = 1\nfn f() -> Int = 2")[0].0, "duplicate_name");
    assert_eq!(kinds("fn f(a: Int, a: Int) -> Int = a")[0], ("duplicate_name", "f".into(), "params[1]".into()));
}

#[test]
fn arguments_are_checked_even_when_arity_is_wrong() {
    let k = kinds("fn g(a: Int) -> Int = a\nfn f() -> Int = g(1, true && 1)");
    assert_eq!(k, vec![("type_mismatch", "f".into(), "body".into()), ("type_mismatch", "f".into(), "body.args[1].right".into())]);
    let k = kinds("fn f() -> Int = len(1 + true, 2)");
    assert_eq!(k, vec![("type_mismatch", "f".into(), "body".into()), ("type_mismatch", "f".into(), "body.args[0].right".into())]);
}

#[test]
fn let_scoping_shadowing_and_annotations() {
    assert_eq!(check_src("fn f() -> Bool = let x = 1 in let x = true in x"), vec![]);
    assert_eq!(kinds("fn f() -> Int = (let x = 1 in x) + x"), vec![("unknown_name", "f".into(), "body.right".into())]);
    assert_eq!(check_src("fn f() -> Int = let x: Option[Int] = None in match x { Some(n) => n, None => 0 }"), vec![]);
    assert_eq!(kinds("fn f() -> Int = let x: Int = true in x")[0], ("type_mismatch", "f".into(), "body.value".into()));
}

#[test]
fn let_value_errors_still_check_the_body() {
    let k = kinds("fn f() -> Int = let x = nope in x + true");
    assert_eq!(k, vec![("unknown_name", "f".into(), "body.value".into()), ("type_mismatch", "f".into(), "body.body.right".into())]);
}

#[test]
fn contracts_must_be_bool_and_see_result() {
    assert_eq!(kinds("fn f(x: Int) -> Int req x = x"), vec![("type_mismatch", "f".into(), "req[0]".into())]);
    assert_eq!(kinds("fn f(x: Int) -> Int req r > 0 = x"), vec![("unknown_name", "f".into(), "req[0].left".into())]);
    assert_eq!(check_src("fn f(x: Int) -> Bool ens r == (x > 0) = x > 0"), vec![]);
}

#[test]
fn parameter_named_r_is_rejected_when_ens_exists() {
    let d = one("fn f(r: Int) -> Int ens r > 0 = r");
    assert_eq!((d.kind, d.path.as_str()), ("duplicate_name", "params[0]"));
    assert_eq!(check_src("fn f(r: Int) -> Int req r > 0 = r"), vec![]);
}

#[test]
fn effects_must_be_declared_and_known() {
    let src = "fn say() -> Unit !io = ()\nfn pure_caller() -> Unit = say()\nfn ok() -> Unit !io = say()";
    let d = one(src);
    assert_eq!((d.kind, d.def.as_str(), d.code), ("effect_violation", "pure_caller", "E0301"));
    assert_eq!(d.hints, vec!["add `!io` to the signature of `pure_caller`".to_string()]);
    assert_eq!(kinds("fn f() -> Unit !net = ()")[0], ("unknown_name", "f".into(), "effects[0]".into()));
    assert_eq!(kinds("fn say() -> Unit !io = ()\ntest t = say() == ()")[0].0, "effect_violation");
    assert_eq!(kinds("fn say() -> Unit !io = ()\nfn f() -> Int !io req say() == () = 1")[0].0, "effect_violation");
}

#[test]
fn builtin_effects_and_function_values() {
    assert_eq!(kinds("fn f() -> Unit = print(\"x\")"), vec![("effect_violation", "f".into(), "body".into())]);
    assert_eq!(check_src("fn f(p: Str) -> Result[Str, Str] !fs = read_file(p)"), vec![]);
    let src = "fn shout(s: Str) -> Str !io = { print(s); s }\n\
               fn all(xs: List[Str]) -> List[Str] = map(xs, shout)";
    let d = one(src);
    assert_eq!((d.kind, d.path.as_str()), ("effect_violation", "body.args[1]"));
    let src = "fn twice(f: fn(Int) -> Int, x: Int) -> Int = f(f(x))\nfn inc(x: Int) -> Int = x + 1\ntest t = twice(inc, 1) == 3";
    assert_eq!(check_src(src), vec![]);
    let d = one("fn run(f: fn() -> Unit !io) -> Unit = f()");
    assert_eq!(d.kind, "effect_violation");
    let d = one("fn hi() -> Unit !io = print(\"hi\")\nfn take(f: fn() -> Unit) -> Unit = f()\nfn g() -> Unit = take(hi)");
    assert_eq!((d.kind, d.path.as_str()), ("type_mismatch", "body.args[0]"));
    assert_eq!(kinds("fn f() -> Int = len")[0].0, "unknown_name");
    assert_eq!(kinds("fn len(s: Str) -> Int = 1")[0].0, "duplicate_name");
}

#[test]
fn tests_must_be_bool() {
    assert_eq!(kinds("test t = 1"), vec![("type_mismatch", "t".into(), "body".into())]);
}

#[test]
fn infers_generic_builtins_and_literals() {
    let src = "fn double(x: Int) -> Int = x * 2\n\
               fn add(a: Int, b: Int) -> Int = a + b\n\
               fn positive(x: Int) -> Bool = x > 0\n\
               test t = fold(map(filter(range(-2, 3), positive), double), 0, add) == 6\n\
               test u = len(\"héllo\") == 5 && len([1, 2]) == 2 && get([1.5], 0) == Some(1.5)\n\
               test v = [] == concat([], []) && reverse([\"a\"]) ++ [\"b\"] == [\"a\", \"b\"]\n\
               test w = to_int(\"12\") == Some(12) && from_float(int_to_float(2)) == \"2.0\"";
    let tp = typed(src);
    let TKind::Bin(_, l, _, _) = &tp.tests[1].body.kind else { panic!() };
    let TKind::Bin(_, ll, _, _) = &l.kind else { panic!() };
    let TKind::Bin(_, len_call, _, _) = &ll.kind else { panic!() };
    assert!(matches!(len_call.kind, TKind::Builtin(Builtin::StrLen, ..)));
    assert_eq!(check_src("fn f() -> Int = len(true)")[0].expected.as_deref(), Some("Str or List[_]"));
}

#[test]
fn float_arithmetic_and_no_mixing() {
    assert_eq!(check_src("fn avg(a: Float, b: Float) -> Float = (a + b) / 2.0\ntest t = avg(1.0, 2.0) > 1.4"), vec![]);
    let d = one("fn f(a: Float) -> Float = a + 1");
    assert_eq!((d.path.as_str(), d.expected.as_deref(), d.found.as_deref()), ("body.right", Some("Float"), Some("Int")));
    assert_eq!(d.hints, vec!["convert with `int_to_float(n)`".to_string()]);
    assert_eq!(kinds("fn f(a: Float) -> Float = a % 2.0")[0].2, "body.left");
    assert_eq!(one("fn f(a: Str) -> Str = -a").expected.as_deref(), Some("Int or Float"));
}

#[test]
fn concat_works_on_str_and_list_only() {
    assert_eq!(check_src("test t = \"a\" ++ \"b\" == \"ab\" && [1] ++ [2] == [1, 2]"), vec![]);
    assert_eq!(one("test t = 1 ++ 2 == 3").expected.as_deref(), Some("Str or List[_]"));
    assert_eq!(one("test t = \"a\" ++ 1 == \"a\"").path, "body.left.right");
}

#[test]
fn records_and_fields() {
    let src = "type User = { id: Int, name: Str }\n\
               fn mk() -> User = User { name: \"a\", id: 1 }\n\
               test t = mk().name == \"a\" && mk() == User { id: 1, name: \"a\" }";
    let tp = typed(src);
    let TKind::Record(fields) = &tp.fns[0].body.kind else { panic!() };
    assert_eq!(fields[0].kind, TKind::Int(1), "fields are stored in declaration order");
    let ty = "type User = { id: Int, name: Str }\n";
    let d = one(&format!("{ty}fn f() -> User = User {{ id: 1 }}"));
    assert_eq!((d.code, d.message.as_str()), ("E0206", "`User` needs field `name`"));
    assert_eq!(kinds(&format!("{ty}fn f() -> User = User {{ id: 1, name: \"\", age: 3 }}"))[0], ("unknown_name", "f".into(), "body.fields.age".into()));
    assert_eq!(kinds(&format!("{ty}fn f(u: User) -> Int = u.age"))[0], ("unknown_name", "f".into(), "body".into()));
    assert_eq!(one(&format!("{ty}fn f(u: Int) -> Int = u.id")).expected.as_deref(), Some("a record"));
    assert_eq!(kinds("fn f(u: Usr) -> Int = 1")[0], ("unknown_name", "f".into(), "params[0]".into()));
}

#[test]
fn enums_variants_and_option_result() {
    let src = "type Err = Empty | NotNumber(Str) | Negative(Int)\n\
               fn parse_age(s: Str) -> Result[Int, Err]\n\
                 ens match r { Ok(n) => n >= 0, _ => true }\n\
               = if len(trim(s)) == 0 then Err(Empty)\n\
                 else match to_int(trim(s)) { None => Err(NotNumber(s)), Some(n) if n < 0 => Err(Negative(n)), Some(n) => Ok(n) }\n\
               test parse_age_ok = parse_age(\" 42 \") == Ok(42)\n\
               test parse_age_neg = parse_age(\"-1\") == Err(Negative(-1))";
    assert_eq!(check_src(src), vec![]);
    assert_eq!(kinds("type E = A | B(Int)\nfn f() -> E = B")[0], ("type_mismatch", "f".into(), "body".into()));
    assert_eq!(kinds("type E = A | B(Int)\nfn f() -> E = B(1, 2)")[0].0, "type_mismatch");
    assert_eq!(kinds("type E = A | Some")[0].0, "duplicate_name");
    assert_eq!(kinds("type E = A\ntype F = A")[0].0, "duplicate_name");
    assert_eq!(kinds("type Int = A")[0].0, "duplicate_name");
    let d = one("fn f() -> Int = Some(1)");
    assert_eq!(d.hints, vec!["unpack the Option with `match v { Some(x) => ..., None => ... }`".to_string()]);
}

#[test]
fn match_patterns_bind_and_type_check() {
    let src = "type P = { x: Int, y: Int }\n\
               fn f(p: P) -> Int = match p { P { x: 0, y } => y, P { x, y: 1 } => x, _ => 0 }";
    let tp = typed(src);
    let TKind::Match(_, arms) = &tp.fns[0].body.kind else { panic!() };
    assert_eq!(arms[0].pat, TPat::Record(vec![(0, TPat::Int(0)), (1, TPat::Bind(1))]));
    assert_eq!(tp.fns[0].locals, vec![Type::Named("P".into()), Type::Int, Type::Int]);
    assert_eq!(kinds("fn f(x: Int) -> Int = match x { \"a\" => 1, _ => 2 }")[0], ("type_mismatch", "f".into(), "body.arms[0].pattern".into()));
    assert_eq!(kinds("fn f(x: Int) -> Int = match x { 1 => true, _ => 2 }")[0], ("type_mismatch", "f".into(), "body.arms[1].body".into()));
    assert_eq!(kinds("fn f(x: Option[Int]) -> Int = match x { Some(a, b) => 1, _ => 2 }")[0].0, "type_mismatch");
    assert_eq!(kinds("fn f(x: Option[Int]) -> Int = match x { Some(a) if a => 1, _ => 2 }")[0].2, "body.arms[0].guard");
    assert_eq!(kinds("type T = A(Int, Int)\nfn f(x: T) -> Int = match x { A(a, a) => a }")[0].0, "duplicate_name");
}

#[test]
fn non_exhaustive_match_is_reported_with_a_witness() {
    let d = one("fn f(x: Option[Int]) -> Int = match x { Some(n) => n }");
    assert_eq!((d.kind, d.code, d.path.as_str(), d.message.as_str()), ("non_exhaustive_match", "E0205", "body", "match does not cover `None`"));
    let d = one("type E = A | B(Bool)\nfn f(x: E) -> Int = match x { A => 1, B(true) => 2 }");
    assert_eq!(d.message, "match does not cover `B(false)`");
    let d = one("fn f(x: Option[Int]) -> Int = match x { None => 0, Some(n) if n > 0 => n }");
    assert_eq!(d.message, "match does not cover `Some(_)`");
    let d = one("fn f(x: Int) -> Int = match x { 0 => 1, 1 => 2 }");
    assert_eq!(d.message, "match does not cover `_`");
    assert_eq!(check_src("fn f(x: Bool, y: Int) -> Int = match x { true => y, false => 0 }"), vec![]);
}

#[test]
fn sequences_discard_only_unit() {
    assert_eq!(check_src("fn f() -> Int !io = { print(\"a\"); print(\"b\"); 3 }"), vec![]);
    let d = one("fn f() -> Int = { 1; 2 }");
    assert_eq!((d.path.as_str(), d.expected.as_deref()), ("body.items[0]", Some("Unit")));
    assert!(d.hints[0].starts_with("only Unit values can be discarded"));
}

#[test]
fn main_signature_is_checked() {
    assert_eq!(check_src("fn main(args: List[Str]) -> Int !io = { print(\"hi\"); 0 }"), vec![]);
    let k = kinds("fn main() -> Unit = ()");
    assert_eq!(k, vec![("type_mismatch", "main".into(), "params".into()), ("type_mismatch", "main".into(), "ret".into())]);
}

#[test]
fn functions_cannot_be_compared() {
    let d = one("fn a() -> Int = 1\ntest t = a == a");
    assert_eq!((d.path.as_str(), d.hints[0].as_str()), ("body", "functions cannot be compared"));
    let d = one("type R = { f: fn(Int) -> Int, n: Int }\ntype L = Nil | Cons(R, L)\nfn eq(a: L, b: L) -> Bool = a == b");
    assert_eq!(d.hints, vec!["functions cannot be compared".to_string()]);
    assert_eq!(check_src("type L = Nil | Cons(Int, L)\nfn eq(a: L, b: L) -> Bool = a == b"), vec![]);
}

#[test]
fn runtime_export_names_are_reserved() {
    assert_eq!(kinds("fn memory() -> Int = 1")[0].0, "duplicate_name");
    assert_eq!(one("fn qal_alloc() -> Int = 1").hints, vec!["`qal_alloc` is reserved by the runtime".to_string()]);
}
```

Create `qal/crates/qal-check/src/unify.rs` containing only its test module for now:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binds_and_resolves_variables() {
        let mut s = Subst::default();
        let a = s.fresh();
        assert!(s.unify(&Type::list(a.clone()), &Type::list(Type::Int)));
        assert_eq!(s.resolve(&Type::option(a.clone())), Type::option(Type::Int));
        assert!(!s.unify(&a, &Type::Str));
        let b = s.fresh();
        assert_eq!(s.finish(&Type::list(b)), Type::list(Type::Unit));
    }

    #[test]
    fn rejects_infinite_types() {
        let mut s = Subst::default();
        let a = s.fresh();
        assert!(!s.unify(&a, &Type::list(a.clone())));
    }

    #[test]
    fn function_values_may_have_fewer_effects() {
        let mut s = Subst::default();
        let pure = Type::Fn(vec![Type::Int], Box::new(Type::Int), vec![]);
        let io = Type::Fn(vec![Type::Int], Box::new(Type::Int), vec!["io".into()]);
        assert!(s.unify(&io, &pure));
        assert!(!s.unify(&pure, &io));
    }
}
```

Create `qal/crates/qal-check/src/exhaust.rs` containing only its test module for now:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use qal_core::ast::{TypeDef, Variant};

    fn prog() -> TProgram {
        TProgram {
            types: vec![
                TypeDef {
                    name: "E".into(),
                    body: TypeBody::Enum(vec![
                        Variant { name: "A".into(), fields: vec![] },
                        Variant { name: "B".into(), fields: vec![Type::Bool, Type::Int] },
                    ]),
                },
                TypeDef { name: "R".into(), body: TypeBody::Record(vec![("x".into(), Type::Bool), ("y".into(), Type::Int)]) },
            ],
            ..TProgram::default()
        }
    }

    fn check(ty: Type, rows: &[TPat]) -> Option<String> {
        missing_pattern(&prog(), &ty, &rows.iter().collect::<Vec<_>>())
    }

    #[test]
    fn option_and_result() {
        let opt = Type::option(Type::Int);
        assert_eq!(check(opt.clone(), &[TPat::Variant(1, vec![TPat::Wild])]), Some("None".into()));
        assert_eq!(check(opt.clone(), &[TPat::Variant(0, vec![])]), Some("Some(_)".into()));
        assert_eq!(check(opt.clone(), &[TPat::Variant(0, vec![]), TPat::Variant(1, vec![TPat::Bind(0)])]), None);
        assert_eq!(check(opt, &[TPat::Variant(0, vec![]), TPat::Variant(1, vec![TPat::Int(3)])]), Some("Some(_)".into()));
        let res = Type::result(Type::Bool, Type::Str);
        assert_eq!(check(res, &[TPat::Variant(0, vec![TPat::Bool(true)]), TPat::Variant(1, vec![TPat::Wild])]), Some("Ok(false)".into()));
    }

    #[test]
    fn user_enums_records_and_nesting() {
        let e = Type::Named("E".into());
        assert_eq!(check(e.clone(), &[TPat::Variant(0, vec![])]), Some("B(_, _)".into()));
        assert_eq!(check(e.clone(), &[TPat::Variant(0, vec![]), TPat::Variant(1, vec![TPat::Bool(true), TPat::Wild])]), Some("B(false, _)".into()));
        assert_eq!(check(e, &[TPat::Wild]), None);
        let r = Type::Named("R".into());
        assert_eq!(check(r.clone(), &[TPat::Record(vec![(0, TPat::Bool(true))])]), Some("R { x: false, y: _ }".into()));
        assert_eq!(check(r, &[TPat::Record(vec![(0, TPat::Bool(true))]), TPat::Record(vec![(0, TPat::Bool(false))])]), None);
        let nested = Type::option(Type::option(Type::Unit));
        assert_eq!(check(nested, &[TPat::Variant(0, vec![]), TPat::Variant(1, vec![TPat::Variant(1, vec![TPat::Unit])])]), Some("Some(None)".into()));
    }

    #[test]
    fn infinite_types_need_a_wildcard() {
        assert_eq!(check(Type::Int, &[TPat::Int(0), TPat::Int(1)]), Some("_".into()));
        assert_eq!(check(Type::Str, &[TPat::Str("a".into()), TPat::Bind(0)]), None);
        assert_eq!(check(Type::Bool, &[TPat::Bool(true), TPat::Bool(false)]), None);
        assert_eq!(check(Type::Unit, &[TPat::Unit]), None);
        assert_eq!(check(Type::Bool, &[]), Some("false".into()));
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd qal && cargo test -p qal-check`
Expected: FAIL to compile (the 1a checker uses `Type: Copy`, `Type::name()` and the old `Expr::Let`; `check`, `Subst`, `missing_pattern` do not exist).

- [ ] **Step 3: Write the unification module**

Replace `qal/crates/qal-check/src/unify.rs` with:

```rust
//! Type variables and unification for local type inference inside one definition.

use qal_core::ast::Type;

#[derive(Debug, Default)]
pub struct Subst {
    /// Binding of each inference variable `Type::Var(i)`.
    vars: Vec<Option<Type>>,
}

impl Subst {
    pub fn fresh(&mut self) -> Type {
        self.vars.push(None);
        Type::Var(self.vars.len() as u32 - 1)
    }

    /// Replaces bound variables (deeply). Unbound variables stay.
    pub fn resolve(&self, t: &Type) -> Type {
        match t {
            Type::Var(v) => match &self.vars[*v as usize] {
                Some(b) => self.resolve(b),
                None => t.clone(),
            },
            Type::List(x) => Type::list(self.resolve(x)),
            Type::Option(x) => Type::option(self.resolve(x)),
            Type::Result(a, b) => Type::result(self.resolve(a), self.resolve(b)),
            Type::Fn(ps, r, e) => Type::Fn(ps.iter().map(|p| self.resolve(p)).collect(), Box::new(self.resolve(r)), e.clone()),
            _ => t.clone(),
        }
    }

    /// Like `resolve`, but unbound variables become `Unit` (their representation never matters).
    pub fn finish(&self, t: &Type) -> Type {
        match self.resolve(t) {
            Type::Var(_) => Type::Unit,
            Type::List(x) => Type::list(self.finish(&x)),
            Type::Option(x) => Type::option(self.finish(&x)),
            Type::Result(a, b) => Type::result(self.finish(&a), self.finish(&b)),
            Type::Fn(ps, r, e) => Type::Fn(ps.iter().map(|p| self.finish(p)).collect(), Box::new(self.finish(&r)), e),
            t => t,
        }
    }

    fn occurs(&self, v: u32, t: &Type) -> bool {
        match self.resolve(t) {
            Type::Var(w) => v == w,
            Type::List(x) | Type::Option(x) => self.occurs(v, &x),
            Type::Result(a, b) => self.occurs(v, &a) || self.occurs(v, &b),
            Type::Fn(ps, r, _) => ps.iter().any(|p| self.occurs(v, p)) || self.occurs(v, &r),
            _ => false,
        }
    }

    /// Makes `expected` and `found` equal. A function value may have fewer effects than expected.
    /// On failure, bindings made so far are kept (the caller reports a diagnostic and stops).
    pub fn unify(&mut self, expected: &Type, found: &Type) -> bool {
        let (a, b) = (self.resolve(expected), self.resolve(found));
        match (&a, &b) {
            (Type::Var(x), Type::Var(y)) if x == y => true,
            (Type::Var(x), t) | (t, Type::Var(x)) => {
                if self.occurs(*x, t) {
                    return false;
                }
                self.vars[*x as usize] = Some(t.clone());
                true
            }
            (Type::List(x), Type::List(y)) | (Type::Option(x), Type::Option(y)) => self.unify(x, y),
            (Type::Result(a1, b1), Type::Result(a2, b2)) => self.unify(a1, a2) && self.unify(b1, b2),
            (Type::Fn(p1, r1, e1), Type::Fn(p2, r2, e2)) => {
                p1.len() == p2.len()
                    && p1.iter().zip(p2).all(|(x, y)| self.unify(x, y))
                    && self.unify(r1, r2)
                    && e2.iter().all(|e| e1.contains(e))
            }
            _ => a == b,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binds_and_resolves_variables() {
        let mut s = Subst::default();
        let a = s.fresh();
        assert!(s.unify(&Type::list(a.clone()), &Type::list(Type::Int)));
        assert_eq!(s.resolve(&Type::option(a.clone())), Type::option(Type::Int));
        assert!(!s.unify(&a, &Type::Str));
        let b = s.fresh();
        assert_eq!(s.finish(&Type::list(b)), Type::list(Type::Unit));
    }

    #[test]
    fn rejects_infinite_types() {
        let mut s = Subst::default();
        let a = s.fresh();
        assert!(!s.unify(&a, &Type::list(a.clone())));
    }

    #[test]
    fn function_values_may_have_fewer_effects() {
        let mut s = Subst::default();
        let pure = Type::Fn(vec![Type::Int], Box::new(Type::Int), vec![]);
        let io = Type::Fn(vec![Type::Int], Box::new(Type::Int), vec!["io".into()]);
        assert!(s.unify(&io, &pure));
        assert!(!s.unify(&pure, &io));
    }
}
```

- [ ] **Step 4: Write the exhaustiveness module**

Replace `qal/crates/qal-check/src/exhaust.rs` with:

```rust
//! Match exhaustiveness (usefulness algorithm after Maranget): finds a value no arm covers.
//!
//! Guarded arms do not count: a guard may be false. Int, Float, Str, List and fn values have
//! infinitely many constructors, so only a wildcard or binding covers them completely.

use qal_core::ast::{Type, TypeBody};
use qal_core::tast::{TPat, TProgram};

#[derive(Debug, Clone, PartialEq)]
enum Key {
    Tag(u32),
    Bool(bool),
    Unit,
    Record,
    Lit(String),
}

/// A pattern reduced to constructor form.
#[derive(Debug, Clone)]
enum P {
    Wild,
    C(Key, Vec<P>),
}

struct Ctor {
    key: Key,
    /// Display: variant name, `true`, `()`, or the record type name.
    name: String,
    args: Vec<Type>,
    /// Field names, for records.
    fields: Vec<String>,
}

/// A missing value, printed in pattern syntax.
enum W {
    Wild,
    C(String, Vec<W>, Vec<String>, bool),
}

fn show(w: &W) -> String {
    match w {
        W::Wild => "_".into(),
        W::C(name, args, _, false) if args.is_empty() => name.clone(),
        W::C(name, args, _, false) => format!("{name}({})", args.iter().map(show).collect::<Vec<_>>().join(", ")),
        W::C(name, args, fields, true) => {
            let fs: Vec<String> = fields.iter().zip(args).map(|(f, a)| format!("{f}: {}", show(a))).collect();
            format!("{name} {{ {} }}", fs.join(", "))
        }
    }
}

/// Returns a pattern (as text) for a value of type `ty` that none of `rows` matches, if any.
pub fn missing_pattern(tp: &TProgram, ty: &Type, rows: &[&TPat]) -> Option<String> {
    let rows: Vec<Vec<P>> = rows.iter().map(|p| vec![reduce(tp, p, ty)]).collect();
    missing(tp, &rows, std::slice::from_ref(ty)).map(|w| show(&w[0]))
}

/// Constructors of `ty`, or None if it has infinitely many.
fn ctors(tp: &TProgram, ty: &Type) -> Option<Vec<Ctor>> {
    let simple = |key, name: &str| Ctor { key, name: name.into(), args: vec![], fields: vec![] };
    match ty {
        Type::Bool => Some(vec![simple(Key::Bool(false), "false"), simple(Key::Bool(true), "true")]),
        Type::Unit => Some(vec![simple(Key::Unit, "()")]),
        Type::Option(_) | Type::Result(..) => Some(variants(tp, ty)),
        Type::Named(n) => match tp.type_def(n).map(|d| &d.body) {
            Some(TypeBody::Enum(_)) => Some(variants(tp, ty)),
            Some(TypeBody::Record(fs)) => Some(vec![Ctor {
                key: Key::Record,
                name: n.clone(),
                args: fs.iter().map(|(_, t)| t.clone()).collect(),
                fields: fs.iter().map(|(f, _)| f.clone()).collect(),
            }]),
            None => None,
        },
        _ => None,
    }
}

fn variants(tp: &TProgram, ty: &Type) -> Vec<Ctor> {
    tp.variant_names(ty)
        .into_iter()
        .enumerate()
        .map(|(tag, name)| Ctor { key: Key::Tag(tag as u32), name, args: tp.variant_fields(ty, tag as u32), fields: vec![] })
        .collect()
}

fn reduce(tp: &TProgram, p: &TPat, ty: &Type) -> P {
    match p {
        TPat::Wild | TPat::Bind(_) => P::Wild,
        TPat::Int(n) => P::C(Key::Lit(n.to_string()), vec![]),
        TPat::Str(s) => P::C(Key::Lit(format!("{s:?}")), vec![]),
        TPat::Bool(b) => P::C(Key::Bool(*b), vec![]),
        TPat::Unit => P::C(Key::Unit, vec![]),
        TPat::Variant(tag, ps) => {
            let tys = tp.variant_fields(ty, *tag);
            P::C(Key::Tag(*tag), ps.iter().zip(&tys).map(|(p, t)| reduce(tp, p, t)).collect())
        }
        TPat::Record(fs) => {
            let all = tp.record_fields(ty);
            let args = all
                .iter()
                .enumerate()
                .map(|(i, (_, t))| match fs.iter().find(|(f, _)| *f as usize == i) {
                    Some((_, p)) => reduce(tp, p, t),
                    None => P::Wild,
                })
                .collect();
            P::C(Key::Record, args)
        }
    }
}

fn head(row: &[P]) -> Option<&Key> {
    match &row[0] {
        P::C(k, _) => Some(k),
        P::Wild => None,
    }
}

/// Rows that match constructor `key` (with `arity` arguments), with the head replaced by its arguments.
fn specialize(rows: &[Vec<P>], key: &Key, arity: usize) -> Vec<Vec<P>> {
    rows.iter()
        .filter_map(|row| {
            let args = match &row[0] {
                P::C(k, args) if k == key => args.clone(),
                P::C(..) => return None,
                P::Wild => vec![P::Wild; arity],
            };
            Some(args.into_iter().chain(row[1..].iter().cloned()).collect())
        })
        .collect()
}

fn missing(tp: &TProgram, rows: &[Vec<P>], tys: &[Type]) -> Option<Vec<W>> {
    if tys.is_empty() {
        return rows.is_empty().then(Vec::new);
    }
    let heads: Vec<&Key> = rows.iter().filter_map(|r| head(r)).collect();
    let all = ctors(tp, &tys[0]);
    if let Some(all) = &all {
        if !all.is_empty() && all.iter().all(|c| heads.contains(&&c.key)) {
            for c in all {
                let sub = specialize(rows, &c.key, c.args.len());
                let sub_tys: Vec<Type> = c.args.iter().chain(&tys[1..]).cloned().collect();
                if let Some(mut w) = missing(tp, &sub, &sub_tys) {
                    let rest = w.split_off(c.args.len());
                    let mut out = vec![W::C(c.name.clone(), w, c.fields.clone(), c.key == Key::Record)];
                    out.extend(rest);
                    return Some(out);
                }
            }
            return None;
        }
    }
    let default: Vec<Vec<P>> = rows.iter().filter(|r| head(r).is_none()).map(|r| r[1..].to_vec()).collect();
    let rest = missing(tp, &default, &tys[1..])?;
    let first = match all.as_ref().and_then(|all| all.iter().find(|c| !heads.contains(&&c.key))) {
        Some(c) => W::C(c.name.clone(), c.args.iter().map(|_| W::Wild).collect(), c.fields.clone(), c.key == Key::Record),
        None => W::Wild,
    };
    Some(std::iter::once(first).chain(rest).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use qal_core::ast::{TypeDef, Variant};

    fn prog() -> TProgram {
        TProgram {
            types: vec![
                TypeDef {
                    name: "E".into(),
                    body: TypeBody::Enum(vec![
                        Variant { name: "A".into(), fields: vec![] },
                        Variant { name: "B".into(), fields: vec![Type::Bool, Type::Int] },
                    ]),
                },
                TypeDef { name: "R".into(), body: TypeBody::Record(vec![("x".into(), Type::Bool), ("y".into(), Type::Int)]) },
            ],
            ..TProgram::default()
        }
    }

    fn check(ty: Type, rows: &[TPat]) -> Option<String> {
        missing_pattern(&prog(), &ty, &rows.iter().collect::<Vec<_>>())
    }

    #[test]
    fn option_and_result() {
        let opt = Type::option(Type::Int);
        assert_eq!(check(opt.clone(), &[TPat::Variant(1, vec![TPat::Wild])]), Some("None".into()));
        assert_eq!(check(opt.clone(), &[TPat::Variant(0, vec![])]), Some("Some(_)".into()));
        assert_eq!(check(opt.clone(), &[TPat::Variant(0, vec![]), TPat::Variant(1, vec![TPat::Bind(0)])]), None);
        assert_eq!(check(opt, &[TPat::Variant(0, vec![]), TPat::Variant(1, vec![TPat::Int(3)])]), Some("Some(_)".into()));
        let res = Type::result(Type::Bool, Type::Str);
        assert_eq!(check(res, &[TPat::Variant(0, vec![TPat::Bool(true)]), TPat::Variant(1, vec![TPat::Wild])]), Some("Ok(false)".into()));
    }

    #[test]
    fn user_enums_records_and_nesting() {
        let e = Type::Named("E".into());
        assert_eq!(check(e.clone(), &[TPat::Variant(0, vec![])]), Some("B(_, _)".into()));
        assert_eq!(check(e.clone(), &[TPat::Variant(0, vec![]), TPat::Variant(1, vec![TPat::Bool(true), TPat::Wild])]), Some("B(false, _)".into()));
        assert_eq!(check(e, &[TPat::Wild]), None);
        let r = Type::Named("R".into());
        assert_eq!(check(r.clone(), &[TPat::Record(vec![(0, TPat::Bool(true))])]), Some("R { x: false, y: _ }".into()));
        assert_eq!(check(r, &[TPat::Record(vec![(0, TPat::Bool(true))]), TPat::Record(vec![(0, TPat::Bool(false))])]), None);
        let nested = Type::option(Type::option(Type::Unit));
        assert_eq!(check(nested, &[TPat::Variant(0, vec![]), TPat::Variant(1, vec![TPat::Variant(1, vec![TPat::Unit])])]), Some("Some(None)".into()));
    }

    #[test]
    fn infinite_types_need_a_wildcard() {
        assert_eq!(check(Type::Int, &[TPat::Int(0), TPat::Int(1)]), Some("_".into()));
        assert_eq!(check(Type::Str, &[TPat::Str("a".into()), TPat::Bind(0)]), None);
        assert_eq!(check(Type::Bool, &[TPat::Bool(true), TPat::Bool(false)]), None);
        assert_eq!(check(Type::Unit, &[TPat::Unit]), None);
        assert_eq!(check(Type::Bool, &[]), Some("false".into()));
    }
}
```

- [ ] **Step 5: Write expression inference**

Create `qal/crates/qal-check/src/infer.rs`:

```rust
//! Type inference for one definition: expressions, patterns, and the final resolution pass.

use crate::exhaust::missing_pattern;
use crate::unify::Subst;
use crate::{validate_type, Globals};
use qal_core::ast::*;
use qal_core::diag::Diagnostic;
use qal_core::tast::*;

/// Checks that need fully inferred types; run by `finish`.
enum Deferred {
    /// `==`/`!=` operand type (functions cannot be compared).
    Eq(String, Type),
    /// `++` operand type (must be Str or List).
    Concat(String, Type),
    /// Match scrutinee type and the unguarded arm patterns.
    Exhaust(String, Type, Vec<TPat>),
}

pub(crate) struct Cx<'a> {
    pub g: &'a Globals,
    pub def: &'a str,
    /// Effects the current expression may use (empty for contracts and tests).
    pub effects: &'a [String],
    pub diags: &'a mut Vec<Diagnostic>,
    /// Number of diagnostics when this definition started; more means it has errors.
    pub start: usize,
    pub sub: Subst,
    /// Type of each local by id.
    locals: Vec<Type>,
    /// Visible names, innermost last.
    scope: Vec<(String, u32)>,
    deferred: Vec<Deferred>,
}

impl<'a> Cx<'a> {
    pub fn new(g: &'a Globals, def: &'a str, effects: &'a [String], diags: &'a mut Vec<Diagnostic>) -> Self {
        let start = diags.len();
        Cx { g, def, effects, diags, start, sub: Subst::default(), locals: Vec::new(), scope: Vec::new(), deferred: Vec::new() }
    }

    pub fn bind(&mut self, name: &str, ty: Type) -> u32 {
        let id = self.locals.len() as u32;
        self.locals.push(ty);
        self.scope.push((name.to_string(), id));
        id
    }

    pub fn unbind(&mut self) {
        self.scope.pop();
    }

    fn lookup(&self, name: &str) -> Option<u32> {
        self.scope.iter().rev().find(|(n, _)| n == name).map(|(_, id)| *id)
    }

    fn has_errors(&self) -> bool {
        self.diags.len() > self.start
    }

    /// An expression whose type is unknown because of an error already reported.
    pub fn poison(&mut self) -> TExpr {
        TExpr { ty: self.sub.fresh(), kind: TKind::Unit }
    }

    pub fn mismatch(&mut self, path: &str, expected: &Type, found: &Type) {
        let (e, f) = (self.sub.finish(expected), self.sub.finish(found));
        let mut d = Diagnostic::type_mismatch(self.def, path, &e.to_string(), &f.to_string());
        match (&e, &f) {
            (Type::Bool, Type::Int) => d = d.hint("compare explicitly, e.g. `x != 0`"),
            (Type::Option(x), y) if **x == *y => d = d.hint("wrap the value: `Some(...)`"),
            (x, Type::Option(y)) if *x == **y => d = d.hint("unpack the Option with `match v { Some(x) => ..., None => ... }`"),
            (x, Type::Result(y, _)) if *x == **y => d = d.hint("unpack the Result with `match v { Ok(x) => ..., Err(e) => ... }`"),
            (Type::Float, Type::Int) => d = d.hint("convert with `int_to_float(n)`"),
            (Type::Int, Type::Float) => d = d.hint("convert with `float_to_int(x)` (returns Option[Int])"),
            (Type::Str, Type::Int) => d = d.hint("convert with `from_int(n)`"),
            _ => {}
        }
        self.diags.push(d);
    }

    /// Unifies and reports a mismatch at `path`. Returns false on mismatch.
    pub fn unify_at(&mut self, path: &str, expected: &Type, found: &Type) -> bool {
        if self.sub.unify(expected, found) {
            true
        } else {
            self.mismatch(path, expected, found);
            false
        }
    }

    pub fn expect(&mut self, e: &Expr, want: &Type, path: &str) -> TExpr {
        let t = self.infer(e, path);
        self.unify_at(path, want, &t.ty);
        t
    }

    pub fn effect_check(&mut self, path: &str, callee: &str, needed: &[String]) {
        let missing: Vec<String> = needed.iter().filter(|e| !self.effects.contains(e)).cloned().collect();
        if !missing.is_empty() {
            self.diags.push(Diagnostic::effect_violation(self.def, path, callee, &missing));
        }
    }

    pub fn infer_all(&mut self, args: &[Expr], path: &str) -> Vec<TExpr> {
        args.iter().enumerate().map(|(i, a)| self.infer(a, &format!("{path}.args[{i}]"))).collect()
    }

    /// Checks `args` against `params`; on an arity mismatch reports it and still checks every argument.
    pub fn args(&mut self, callee: &str, args: &[Expr], params: &[Type], path: &str) -> Vec<TExpr> {
        if args.len() != params.len() {
            self.diags.push(Diagnostic::arity(self.def, path, callee, params.len(), args.len()));
            return self.infer_all(args, path);
        }
        args.iter().zip(params).enumerate().map(|(i, (a, t))| self.expect(a, t, &format!("{path}.args[{i}]"))).collect()
    }

    pub fn infer(&mut self, e: &Expr, path: &str) -> TExpr {
        let (ty, kind) = match e {
            Expr::Int(n) => (Type::Int, TKind::Int(*n)),
            Expr::Float(x) => (Type::Float, TKind::Float(*x)),
            Expr::Bool(b) => (Type::Bool, TKind::Bool(*b)),
            Expr::Str(s) => (Type::Str, TKind::Str(s.clone())),
            Expr::Unit => (Type::Unit, TKind::Unit),
            Expr::Var(v) => return self.var(v, path),
            Expr::Call(name, args) => return self.call(name, args, path),
            Expr::Un(UnOp::Not, x) => (Type::Bool, TKind::Un(UnOp::Not, Box::new(self.expect(x, &Type::Bool, &format!("{path}.operand"))), path.into())),
            Expr::Un(UnOp::Neg, x) => {
                let x = self.infer(x, &format!("{path}.operand"));
                let t = self.numeric(&format!("{path}.operand"), &x.ty);
                (t, TKind::Un(UnOp::Neg, Box::new(x), path.into()))
            }
            Expr::Bin(op, l, r) => return self.binary(*op, l, r, path),
            Expr::If(c, t, f) => {
                let c = self.expect(c, &Type::Bool, &format!("{path}.cond"));
                let t = self.infer(t, &format!("{path}.then"));
                let f = self.expect(f, &t.ty, &format!("{path}.else"));
                (t.ty.clone(), TKind::If(Box::new(c), Box::new(t), Box::new(f)))
            }
            Expr::Let(name, ann, v, body) => {
                let v = match ann {
                    Some(t) if validate_type(self.g, t, self.def, &format!("{path}.type"), self.diags) => {
                        self.expect(v, t, &format!("{path}.value"))
                    }
                    _ => self.infer(v, &format!("{path}.value")),
                };
                // Even if the value had errors, the body is checked with the (possibly unknown) type.
                let id = self.bind(name, v.ty.clone());
                let b = self.infer(body, &format!("{path}.body"));
                self.unbind();
                (b.ty.clone(), TKind::Let(id, Box::new(v), Box::new(b)))
            }
            Expr::List(items) => {
                let elem = self.sub.fresh();
                let items: Vec<TExpr> = items.iter().enumerate().map(|(i, x)| self.expect(x, &elem, &format!("{path}.items[{i}]"))).collect();
                (Type::list(elem), TKind::List(items))
            }
            Expr::Seq(items) => {
                let n = items.len();
                let mut out = Vec::new();
                for (i, x) in items.iter().enumerate() {
                    let p = format!("{path}.items[{i}]");
                    let t = self.infer(x, &p);
                    if i + 1 < n && !self.sub.unify(&Type::Unit, &t.ty) {
                        self.mismatch(&p, &Type::Unit, &t.ty);
                        self.diags.last_mut().expect("just pushed").hints =
                            vec!["only Unit values can be discarded in `{ a; b }`; bind the value with `let` or handle it with `match`".into()];
                    }
                    out.push(t);
                }
                (out.last().expect("parser: non-empty").ty.clone(), TKind::Seq(out))
            }
            Expr::Record(name, fields) => return self.record(name, fields, path),
            Expr::Field(target, field) => return self.field(target, field, path),
            Expr::Match(s, arms) => return self.match_expr(s, arms, path),
        };
        TExpr { ty, kind }
    }

    /// The type of a numeric operand: Int or Float (an unknown type defaults to Int).
    fn numeric(&mut self, path: &str, t: &Type) -> Type {
        match self.sub.resolve(t) {
            Type::Int => Type::Int,
            Type::Float => Type::Float,
            Type::Var(_) => {
                self.sub.unify(&Type::Int, t);
                Type::Int
            }
            other => {
                self.diags.push(Diagnostic::type_mismatch(self.def, path, "Int or Float", &self.sub.finish(&other).to_string()));
                self.sub.fresh()
            }
        }
    }

    fn binary(&mut self, op: BinOp, l: &Expr, r: &Expr, path: &str) -> TExpr {
        let (lp, rp) = (format!("{path}.left"), format!("{path}.right"));
        let (l, r, ty) = match op {
            BinOp::And | BinOp::Or => (self.expect(l, &Type::Bool, &lp), self.expect(r, &Type::Bool, &rp), Type::Bool),
            BinOp::Rem => (self.expect(l, &Type::Int, &lp), self.expect(r, &Type::Int, &rp), Type::Int),
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                let l = self.infer(l, &lp);
                let r = self.infer(r, &rp);
                let t = self.numeric(&lp, &l.ty);
                self.unify_at(&rp, &t, &r.ty);
                (l, r, if op.is_comparison() { Type::Bool } else { t })
            }
            BinOp::Eq | BinOp::Ne => {
                let l = self.infer(l, &lp);
                let r = self.expect(r, &l.ty, &rp);
                self.deferred.push(Deferred::Eq(path.into(), l.ty.clone()));
                (l, r, Type::Bool)
            }
            BinOp::Concat => {
                let l = self.infer(l, &lp);
                let r = self.expect(r, &l.ty, &rp);
                self.deferred.push(Deferred::Concat(lp, l.ty.clone()));
                let t = l.ty.clone();
                (l, r, t)
            }
        };
        TExpr { ty, kind: TKind::Bin(op, Box::new(l), Box::new(r), path.into()) }
    }

    fn var(&mut self, v: &str, path: &str) -> TExpr {
        if let Some(id) = self.lookup(v) {
            return TExpr { ty: self.locals[id as usize].clone(), kind: TKind::Local(id) };
        }
        if let Some((idx, sig)) = self.g.fns.get(v) {
            return TExpr { ty: sig.fn_type(), kind: TKind::FnRef(*idx) };
        }
        if let Some((ty, tag, fields)) = self.variant(v) {
            if !fields.is_empty() {
                self.diags.push(Diagnostic::arity(self.def, path, v, fields.len(), 0));
                return self.poison();
            }
            return TExpr { ty, kind: TKind::Variant(tag, vec![]) };
        }
        let mut d = Diagnostic::unknown_name(self.def, path, v);
        if Builtin::NAMES.contains(&v) {
            d = d.hint(format!("built-in `{v}` cannot be used as a value; wrap it in a fn"));
        }
        self.diags.push(d);
        self.poison()
    }

    /// Looks up a variant by name: (its sum type, with fresh variables for Option/Result), tag, payload types.
    fn variant(&mut self, name: &str) -> Option<(Type, u32, Vec<Type>)> {
        let (a, b) = (self.sub.fresh(), self.sub.fresh());
        Some(match name {
            "None" => (Type::option(a), TAG_NONE, vec![]),
            "Some" => (Type::option(a.clone()), TAG_SOME, vec![a]),
            "Ok" => (Type::result(a.clone(), b), TAG_OK, vec![a]),
            "Err" => (Type::result(a, b.clone()), TAG_ERR, vec![b]),
            _ => {
                let (enum_name, tag, fields) = self.g.variants.get(name)?;
                (Type::Named(enum_name.clone()), *tag, fields.clone())
            }
        })
    }

    fn call(&mut self, name: &str, args: &[Expr], path: &str) -> TExpr {
        if let Some(id) = self.lookup(name) {
            let callee = TExpr { ty: self.locals[id as usize].clone(), kind: TKind::Local(id) };
            let Type::Fn(params, ret, effs) = self.sub.resolve(&callee.ty) else {
                let found = self.sub.finish(&callee.ty).to_string();
                self.diags.push(Diagnostic::type_mismatch(self.def, path, "a function", &found).hint(format!("`{name}` is a local value of type {found}")));
                self.infer_all(args, path);
                return self.poison();
            };
            let args = self.args(name, args, &params, path);
            self.effect_check(path, name, &effs);
            return TExpr { ty: *ret, kind: TKind::CallIndirect(Box::new(callee), args) };
        }
        if let Some((idx, sig)) = self.g.fns.get(name) {
            let (idx, sig) = (*idx, sig.clone());
            let args = self.args(name, args, &sig.params, path);
            self.effect_check(path, name, &sig.effects);
            return TExpr { ty: sig.ret, kind: TKind::Call(idx, args) };
        }
        if let Some((ty, tag, fields)) = self.variant(name) {
            let args = self.args(name, args, &fields, path);
            return TExpr { ty, kind: TKind::Variant(tag, args) };
        }
        if let Some(e) = self.builtin(name, args, path) {
            return e;
        }
        self.diags.push(Diagnostic::unknown_name(self.def, path, name));
        self.infer_all(args, path);
        self.poison()
    }

    fn record(&mut self, name: &str, fields: &[(String, Expr)], path: &str) -> TExpr {
        let decl = match self.g.tp.type_def(name).map(|d| &d.body) {
            Some(TypeBody::Record(fs)) => fs.clone(),
            other => {
                let mut d = Diagnostic::unknown_name(self.def, path, name);
                if other.is_some() {
                    d = Diagnostic::type_mismatch(self.def, path, "a record type", name)
                        .hint(format!("`{name}` is an enum; construct one of its variants instead"));
                }
                self.diags.push(d);
                for (f, e) in fields {
                    self.infer(e, &format!("{path}.fields.{f}"));
                }
                return self.poison();
            }
        };
        let mut values: Vec<Option<TExpr>> = vec![None; decl.len()];
        for (f, e) in fields {
            let p = format!("{path}.fields.{f}");
            match decl.iter().position(|(n, _)| n == f) {
                Some(i) if values[i].is_some() => {
                    self.diags.push(Diagnostic::duplicate_name(self.def, &p, f));
                    self.infer(e, &p);
                }
                Some(i) => values[i] = Some(self.expect(e, &decl[i].1, &p)),
                None => {
                    let names: Vec<&str> = decl.iter().map(|(n, _)| n.as_str()).collect();
                    self.diags.push(Diagnostic::unknown_name(self.def, &p, f).hint(format!("fields of `{name}`: {}", names.join(", "))));
                    self.infer(e, &p);
                }
            }
        }
        let mut out = Vec::new();
        for (v, (f, _)) in values.into_iter().zip(&decl) {
            match v {
                Some(v) => out.push(v),
                None => self.diags.push(Diagnostic::missing_field(self.def, path, name, f)),
            }
        }
        TExpr { ty: Type::Named(name.into()), kind: TKind::Record(out) }
    }

    fn field(&mut self, target: &Expr, field: &str, path: &str) -> TExpr {
        let t = self.infer(target, &format!("{path}.target"));
        let ty = self.sub.resolve(&t.ty);
        let fields = match &ty {
            Type::Named(n) => match self.g.tp.type_def(n).map(|d| &d.body) {
                Some(TypeBody::Record(fs)) => Some(fs.clone()),
                _ => None,
            },
            _ => None,
        };
        let Some(fields) = fields else {
            if !matches!(ty, Type::Var(_)) || !self.has_errors() {
                let mut d = Diagnostic::type_mismatch(self.def, path, "a record", &self.sub.finish(&ty).to_string());
                if matches!(ty, Type::Var(_)) {
                    d = d.hint("the type is not known yet here; annotate it, e.g. `let x: User = ... in x.field`");
                }
                self.diags.push(d);
            }
            return self.poison();
        };
        match fields.iter().position(|(n, _)| n == field) {
            Some(i) => TExpr { ty: fields[i].1.clone(), kind: TKind::Field(Box::new(t), i as u32) },
            None => {
                let names: Vec<&str> = fields.iter().map(|(n, _)| n.as_str()).collect();
                self.diags.push(Diagnostic::unknown_name(self.def, path, field).hint(format!("fields of `{ty}`: {}", names.join(", "))));
                self.poison()
            }
        }
    }

    fn match_expr(&mut self, s: &Expr, arms: &[Arm], path: &str) -> TExpr {
        let s = self.infer(s, &format!("{path}.scrutinee"));
        let result = self.sub.fresh();
        let mut tarms = Vec::new();
        let mut rows = Vec::new();
        for (i, a) in arms.iter().enumerate() {
            let depth = self.scope.len();
            let mut bound = Vec::new();
            let pat = self.pattern(&a.pat, &s.ty, &format!("{path}.arms[{i}].pattern"), &mut bound);
            let guard = a.guard.as_ref().map(|g| self.expect(g, &Type::Bool, &format!("{path}.arms[{i}].guard")));
            let body = self.expect(&a.body, &result, &format!("{path}.arms[{i}].body"));
            self.scope.truncate(depth);
            if guard.is_none() {
                rows.push(pat.clone());
            }
            tarms.push(TArm { pat, guard, body });
        }
        self.deferred.push(Deferred::Exhaust(path.into(), s.ty.clone(), rows));
        TExpr { ty: result, kind: TKind::Match(Box::new(s), tarms) }
    }

    fn pattern(&mut self, p: &Pat, ty: &Type, path: &str, bound: &mut Vec<String>) -> TPat {
        let lit = |cx: &mut Self, t: Type, pat: TPat| {
            cx.unify_at(path, ty, &t);
            pat
        };
        match p {
            Pat::Wild => TPat::Wild,
            Pat::Bind(n) => {
                if bound.contains(n) {
                    self.diags.push(Diagnostic::duplicate_name(self.def, path, n));
                }
                bound.push(n.clone());
                TPat::Bind(self.bind(n, ty.clone()))
            }
            Pat::Int(n) => lit(self, Type::Int, TPat::Int(*n)),
            Pat::Str(s) => lit(self, Type::Str, TPat::Str(s.clone())),
            Pat::Bool(b) => lit(self, Type::Bool, TPat::Bool(*b)),
            Pat::Unit => lit(self, Type::Unit, TPat::Unit),
            Pat::Variant(name, ps) => {
                let Some((vty, tag, fields)) = self.variant(name) else {
                    self.diags.push(Diagnostic::unknown_name(self.def, path, name));
                    return TPat::Wild;
                };
                if !self.unify_at(path, ty, &vty) {
                    return TPat::Wild;
                }
                if ps.len() != fields.len() {
                    self.diags.push(Diagnostic::arity(self.def, path, name, fields.len(), ps.len()));
                    return TPat::Wild;
                }
                let sub = ps.iter().zip(&fields).enumerate().map(|(i, (p, t))| self.pattern(p, t, &format!("{path}.args[{i}]"), bound)).collect();
                TPat::Variant(tag, sub)
            }
            Pat::Record(name, fs) => {
                let decl = match self.g.tp.type_def(name).map(|d| &d.body) {
                    Some(TypeBody::Record(decl)) => decl.clone(),
                    _ => {
                        self.diags.push(Diagnostic::unknown_name(self.def, path, name));
                        return TPat::Wild;
                    }
                };
                if !self.unify_at(path, ty, &Type::Named(name.clone())) {
                    return TPat::Wild;
                }
                let mut out = Vec::new();
                for (f, p) in fs {
                    let fp = format!("{path}.fields.{f}");
                    match decl.iter().position(|(n, _)| n == f) {
                        Some(i) => out.push((i as u32, self.pattern(p, &decl[i].1, &fp, bound))),
                        None => self.diags.push(Diagnostic::unknown_name(self.def, &fp, f)),
                    }
                }
                TPat::Record(out)
            }
        }
    }

    /// Resolves all types, runs deferred checks, and returns (local types, expressions);
    /// None if this definition has errors.
    pub fn finish(mut self, mut exprs: Vec<TExpr>) -> Option<(Vec<Type>, Vec<TExpr>)> {
        if self.has_errors() {
            return None;
        }
        for d in std::mem::take(&mut self.deferred) {
            match d {
                Deferred::Eq(path, t) => {
                    let t = self.sub.finish(&t);
                    if contains_fn(&self.g.tp, &t, &mut Vec::new()) {
                        self.diags.push(Diagnostic::type_mismatch(self.def, &path, "a comparable type", &t.to_string()).hint("functions cannot be compared"));
                    }
                }
                Deferred::Concat(path, t) => {
                    let t = self.sub.resolve(&t);
                    if !matches!(t, Type::Str | Type::List(_)) {
                        self.diags.push(Diagnostic::type_mismatch(self.def, &path, "Str or List[_]", &self.sub.finish(&t).to_string()));
                    }
                }
                Deferred::Exhaust(path, t, rows) => {
                    let t = self.sub.finish(&t);
                    if let Some(w) = missing_pattern(&self.g.tp, &t, &rows.iter().collect::<Vec<_>>()) {
                        self.diags.push(Diagnostic::non_exhaustive(self.def, &path, &w));
                    }
                }
            }
        }
        for e in &mut exprs {
            self.zonk(e);
        }
        if self.has_errors() {
            return None;
        }
        let locals = self.locals.iter().map(|t| self.sub.finish(t)).collect();
        Some((locals, exprs))
    }

    /// Replaces inference variables by their final types and picks `len` by argument type.
    fn zonk(&mut self, e: &mut TExpr) {
        e.ty = self.sub.finish(&e.ty);
        match &mut e.kind {
            TKind::Int(_) | TKind::Float(_) | TKind::Bool(_) | TKind::Str(_) | TKind::Unit | TKind::Local(_) | TKind::FnRef(_) => {}
            TKind::Un(_, x, _) | TKind::Field(x, _) => self.zonk(x),
            TKind::Bin(_, l, r, _) | TKind::Let(_, l, r) => {
                self.zonk(l);
                self.zonk(r);
            }
            TKind::If(c, t, f) => {
                self.zonk(c);
                self.zonk(t);
                self.zonk(f);
            }
            TKind::Call(_, xs) | TKind::List(xs) | TKind::Variant(_, xs) | TKind::Record(xs) | TKind::Seq(xs) => {
                xs.iter_mut().for_each(|x| self.zonk(x));
            }
            TKind::CallIndirect(f, xs) => {
                self.zonk(f);
                xs.iter_mut().for_each(|x| self.zonk(x));
            }
            TKind::Builtin(b, xs, path) => {
                xs.iter_mut().for_each(|x| self.zonk(x));
                if matches!(b, Builtin::StrLen | Builtin::ListLen) {
                    match &xs[0].ty {
                        Type::Str => *b = Builtin::StrLen,
                        Type::List(_) => *b = Builtin::ListLen,
                        other => {
                            let d = Diagnostic::type_mismatch(self.def, &format!("{path}.args[0]"), "Str or List[_]", &other.to_string());
                            self.diags.push(d);
                        }
                    }
                }
            }
            TKind::Match(s, arms) => {
                self.zonk(s);
                for a in arms {
                    if let Some(g) = &mut a.guard {
                        self.zonk(g);
                    }
                    self.zonk(&mut a.body);
                }
            }
        }
    }
}

/// True if a value of type `t` can contain a function (`seen` guards recursive types).
fn contains_fn(tp: &TProgram, t: &Type, seen: &mut Vec<String>) -> bool {
    match t {
        Type::Fn(..) => true,
        Type::List(x) | Type::Option(x) => contains_fn(tp, x, seen),
        Type::Result(a, b) => contains_fn(tp, a, seen) || contains_fn(tp, b, seen),
        Type::Named(n) if !seen.contains(n) => {
            seen.push(n.clone());
            match tp.type_def(n).map(|d| &d.body) {
                Some(TypeBody::Record(fs)) => fs.iter().any(|(_, f)| contains_fn(tp, f, seen)),
                Some(TypeBody::Enum(vs)) => vs.iter().any(|v| v.fields.iter().any(|f| contains_fn(tp, f, seen))),
                None => false,
            }
        }
        _ => false,
    }
}
```

- [ ] **Step 6: Write built-in typing**

Create `qal/crates/qal-check/src/builtins.rs`:

```rust
//! Typing of built-in function calls (spec §3.5).

use crate::infer::Cx;
use qal_core::ast::{Expr, Type};
use qal_core::diag::Diagnostic;
use qal_core::tast::{Builtin, TExpr, TKind};

impl Cx<'_> {
    /// Types a call of built-in `name`; None if `name` is not a built-in.
    pub fn builtin(&mut self, name: &str, args: &[Expr], path: &str) -> Option<TExpr> {
        use Builtin as B;
        let (a, b) = (self.sub.fresh(), self.sub.fresh());
        let fun = |ps: Vec<Type>, r: Type| Type::Fn(ps, Box::new(r), vec![]);
        let list = Type::list;
        let (builtin, params, ret) = match name {
            "len" => return Some(self.len(args, path)),
            "slice" => (B::Slice, vec![Type::Str, Type::Int, Type::Int], Type::Str),
            "split" => (B::Split, vec![Type::Str, Type::Str], list(Type::Str)),
            "join" => (B::Join, vec![list(Type::Str), Type::Str], Type::Str),
            "trim" => (B::Trim, vec![Type::Str], Type::Str),
            "contains" => (B::Contains, vec![Type::Str, Type::Str], Type::Bool),
            "starts_with" => (B::StartsWith, vec![Type::Str, Type::Str], Type::Bool),
            "to_int" => (B::ToInt, vec![Type::Str], Type::option(Type::Int)),
            "to_float" => (B::ToFloat, vec![Type::Str], Type::option(Type::Float)),
            "from_int" => (B::FromInt, vec![Type::Int], Type::Str),
            "from_float" => (B::FromFloat, vec![Type::Float], Type::Str),
            "chars" => (B::Chars, vec![Type::Str], list(Type::Str)),
            "get" => (B::Get, vec![list(a.clone()), Type::Int], Type::option(a)),
            "push" => (B::Push, vec![list(a.clone()), a.clone()], list(a)),
            "concat" => (B::Concat, vec![list(a.clone()), list(a.clone())], list(a)),
            "map" => (B::Map, vec![list(a.clone()), fun(vec![a], b.clone())], list(b)),
            "filter" => (B::Filter, vec![list(a.clone()), fun(vec![a.clone()], Type::Bool)], list(a)),
            "fold" => (B::Fold, vec![list(a.clone()), b.clone(), fun(vec![b.clone(), a], b.clone())], b),
            "range" => (B::Range, vec![Type::Int, Type::Int], list(Type::Int)),
            "reverse" => (B::Reverse, vec![list(a.clone())], list(a)),
            "sort_int" => (B::SortInt, vec![list(Type::Int)], list(Type::Int)),
            "int_to_float" => (B::IntToFloat, vec![Type::Int], Type::Float),
            "float_to_int" => (B::FloatToInt, vec![Type::Float], Type::option(Type::Int)),
            "print" => (B::Print, vec![Type::Str], Type::Unit),
            "read_line" => (B::ReadLine, vec![], Type::option(Type::Str)),
            "read_file" => (B::ReadFile, vec![Type::Str], Type::result(Type::Str, Type::Str)),
            "write_file" => (B::WriteFile, vec![Type::Str, Type::Str], Type::result(Type::Unit, Type::Str)),
            "now_ms" => (B::NowMs, vec![], Type::Int),
            "rand_int" => (B::RandInt, vec![Type::Int, Type::Int], Type::Int),
            _ => return None,
        };
        if args.len() != params.len() {
            self.diags.push(Diagnostic::arity(self.def, path, name, params.len(), args.len()));
            self.infer_all(args, path);
            return Some(self.poison());
        }
        let mut targs = Vec::new();
        for (i, (arg, p)) in args.iter().zip(&params).enumerate() {
            let ap = format!("{path}.args[{i}]");
            targs.push(match p {
                Type::Fn(fps, fr, _) => self.fn_arg(name, arg, fps, fr, &ap),
                _ => self.expect(arg, p, &ap),
            });
        }
        if let Some(e) = builtin.effect() {
            self.effect_check(path, name, &[e.to_string()]);
        }
        Some(TExpr { ty: ret, kind: TKind::Builtin(builtin, targs, path.into()) })
    }

    /// A function argument of map/filter/fold: its effects become effects of the call.
    fn fn_arg(&mut self, callee: &str, arg: &Expr, params: &[Type], ret: &Type, path: &str) -> TExpr {
        let t = self.infer(arg, path);
        let effects = match self.sub.resolve(&t.ty) {
            Type::Fn(_, _, effs) => effs,
            _ => vec![],
        };
        let want = Type::Fn(params.to_vec(), Box::new(ret.clone()), effects.clone());
        if self.unify_at(path, &want, &t.ty) {
            self.effect_check(path, callee, &effects);
        }
        t
    }

    /// `len` works on Str and List; the variant is fixed once the argument type is known.
    fn len(&mut self, args: &[Expr], path: &str) -> TExpr {
        if args.len() != 1 {
            self.diags.push(Diagnostic::arity(self.def, path, "len", 1, args.len()));
            self.infer_all(args, path);
            return self.poison();
        }
        let ap = format!("{path}.args[0]");
        let x = self.infer(&args[0], &ap);
        let b = match self.sub.resolve(&x.ty) {
            Type::Str => Builtin::StrLen,
            Type::List(_) | Type::Var(_) => Builtin::ListLen,
            other => {
                let found = self.sub.finish(&other).to_string();
                self.diags.push(Diagnostic::type_mismatch(self.def, &ap, "Str or List[_]", &found));
                return self.poison();
            }
        };
        TExpr { ty: Type::Int, kind: TKind::Builtin(b, vec![x], path.into()) }
    }
}
```

- [ ] **Step 7: Write the program-level checker**

Replace `qal/crates/qal-check/src/lib.rs` with:

```rust
//! QAL checker: names, types, effects, contracts and match exhaustiveness.
//! Only checked programs may be compiled; `check` returns the typed program the code generator needs.

mod builtins;
mod exhaust;
mod infer;
mod unify;

use infer::Cx;
use qal_core::ast::*;
use qal_core::diag::Diagnostic;
use qal_core::tast::{Builtin, TFn, TProgram, TTest};
use std::collections::{HashMap, HashSet};

/// Effects known in Phase 1.
pub const KNOWN_EFFECTS: &[&str] = &["io", "fs", "time", "rand"];

pub use qal_core::ast::RESULT_VAR;

/// Names the compiled module exports itself; a user function cannot take them.
const RESERVED_EXPORTS: &[&str] = &["memory", "qal_alloc"];

/// Variant names that belong to the built-in `Option` and `Result`.
const BUILTIN_VARIANTS: &[&str] = &["None", "Some", "Ok", "Err"];

#[derive(Debug, Clone, PartialEq)]
pub struct Sig {
    pub params: Vec<Type>,
    pub ret: Type,
    pub effects: Vec<String>,
}

impl Sig {
    pub fn fn_type(&self) -> Type {
        Type::Fn(self.params.clone(), Box::new(self.ret.clone()), self.effects.clone())
    }
}

/// Program-wide names, known before any body is checked.
pub(crate) struct Globals {
    /// Declared types (for lookups); `fns`/`tests` stay empty.
    pub tp: TProgram,
    /// Function name -> (index into `TProgram::fns`, signature).
    pub fns: HashMap<String, (usize, Sig)>,
    /// User variant name -> (enum name, tag, payload types).
    pub variants: HashMap<String, (String, u32, Vec<Type>)>,
}

/// Checks a whole program. An empty result means the program may be compiled.
pub fn check_program(p: &Program) -> Vec<Diagnostic> {
    match check(p) {
        Ok(_) => Vec::new(),
        Err(d) => d,
    }
}

/// Checks a whole program and returns its typed form, or all diagnostics.
pub fn check(p: &Program) -> Result<TProgram, Vec<Diagnostic>> {
    let mut diags = Vec::new();
    let g = globals(p, &mut diags);
    let mut out = TProgram { types: g.tp.types.clone(), ..TProgram::default() };
    for d in &p.defs {
        match d {
            Def::Type(_) => {}
            Def::Fn(f) => {
                if let Some(tf) = check_fn(f, &g, &mut diags) {
                    out.fns.push(tf);
                }
            }
            Def::Test(t) => {
                let mut cx = Cx::new(&g, &t.name, &[], &mut diags);
                let body = cx.expect(&t.body, &Type::Bool, "body");
                if let Some((locals, mut exprs)) = cx.finish(vec![body]) {
                    out.tests.push(TTest { name: t.name.clone(), locals, body: exprs.remove(0) });
                }
            }
        }
    }
    if diags.is_empty() {
        Ok(out)
    } else {
        Err(diags)
    }
}

fn globals(p: &Program, diags: &mut Vec<Diagnostic>) -> Globals {
    let mut g = Globals { tp: TProgram::default(), fns: HashMap::new(), variants: HashMap::new() };
    for d in &p.defs {
        let Def::Type(t) = d else { continue };
        if Type::BUILTIN_NAMES.contains(&t.name.as_str()) {
            diags.push(Diagnostic::duplicate_name(&t.name, "", &t.name).hint(format!("`{}` is a built-in type", t.name)));
            continue;
        }
        if g.tp.type_def(&t.name).is_some() {
            diags.push(Diagnostic::duplicate_name(&t.name, "", &t.name));
            continue;
        }
        match &t.body {
            TypeBody::Record(fields) => {
                let mut seen = HashSet::new();
                for (f, _) in fields {
                    if !seen.insert(f) {
                        diags.push(Diagnostic::duplicate_name(&t.name, &format!("fields.{f}"), f));
                    }
                }
            }
            TypeBody::Enum(vs) => {
                for (tag, v) in vs.iter().enumerate() {
                    if BUILTIN_VARIANTS.contains(&v.name.as_str()) || g.variants.contains_key(&v.name) {
                        let mut d = Diagnostic::duplicate_name(&t.name, &format!("variants[{tag}]"), &v.name);
                        if BUILTIN_VARIANTS.contains(&v.name.as_str()) {
                            d = d.hint(format!("`{}` belongs to the built-in Option/Result", v.name));
                        }
                        diags.push(d);
                    } else {
                        g.variants.insert(v.name.clone(), (t.name.clone(), tag as u32, v.fields.clone()));
                    }
                }
            }
        }
        g.tp.types.push(t.clone());
    }
    for t in g.tp.types.clone() {
        match &t.body {
            TypeBody::Record(fields) => {
                for (f, ty) in fields {
                    validate_type(&g, ty, &t.name, &format!("fields.{f}"), diags);
                }
            }
            TypeBody::Enum(vs) => {
                for (tag, v) in vs.iter().enumerate() {
                    for (i, ty) in v.fields.iter().enumerate() {
                        validate_type(&g, ty, &t.name, &format!("variants[{tag}].fields[{i}]"), diags);
                    }
                }
            }
        }
    }
    let mut seen = HashSet::new();
    let mut index = 0;
    for d in &p.defs {
        if matches!(d, Def::Type(_)) {
            continue;
        }
        if !seen.insert(d.name().to_string()) {
            diags.push(Diagnostic::duplicate_name(d.name(), "", d.name()));
        } else if Builtin::NAMES.contains(&d.name()) {
            diags.push(Diagnostic::duplicate_name(d.name(), "", d.name()).hint(format!("`{}` is a built-in function", d.name())));
        } else if RESERVED_EXPORTS.contains(&d.name()) {
            diags.push(Diagnostic::duplicate_name(d.name(), "", d.name()).hint(format!("`{}` is reserved by the runtime", d.name())));
        }
        if let Def::Fn(f) = d {
            g.fns.entry(f.name.clone()).or_insert_with(|| {
                (index, Sig { params: f.params.iter().map(|p| p.ty.clone()).collect(), ret: f.ret.clone(), effects: f.effects.clone() })
            });
            index += 1;
        }
    }
    g
}

/// Reports unknown type names and effects inside a written type.
pub(crate) fn validate_type(g: &Globals, t: &Type, def: &str, path: &str, diags: &mut Vec<Diagnostic>) -> bool {
    match t {
        Type::Named(n) => {
            if g.tp.type_def(n).is_some() {
                return true;
            }
            let mut d = Diagnostic::unknown_name(def, path, n);
            if g.variants.contains_key(n) || BUILTIN_VARIANTS.contains(&n.as_str()) {
                d = d.hint(format!("`{n}` is a variant, not a type"));
            }
            diags.push(d);
            false
        }
        Type::List(x) | Type::Option(x) => validate_type(g, x, def, path, diags),
        Type::Result(a, b) => validate_type(g, a, def, path, diags) & validate_type(g, b, def, path, diags),
        Type::Fn(ps, r, effs) => {
            // No short-circuit: report every unknown name.
            let mut ok = validate_type(g, r, def, path, diags);
            for p in ps {
                ok &= validate_type(g, p, def, path, diags);
            }
            for e in effs {
                if !KNOWN_EFFECTS.contains(&e.as_str()) {
                    diags.push(Diagnostic::unknown_name(def, path, e).hint(format!("known effects: {}", KNOWN_EFFECTS.join(", "))));
                    ok = false;
                }
            }
            ok
        }
        _ => true,
    }
}

fn check_fn(f: &FnDef, g: &Globals, diags: &mut Vec<Diagnostic>) -> Option<TFn> {
    let before = diags.len();
    let mut names = HashSet::new();
    for (i, p) in f.params.iter().enumerate() {
        let path = format!("params[{i}]");
        if !names.insert(p.name.as_str()) {
            diags.push(Diagnostic::duplicate_name(&f.name, &path, &p.name));
        }
        if p.name == RESULT_VAR && !f.ens.is_empty() {
            diags.push(
                Diagnostic::duplicate_name(&f.name, &path, &p.name)
                    .hint("inside `ens`, `r` is the result; rename the parameter".to_string()),
            );
        }
        validate_type(g, &p.ty, &f.name, &path, diags);
    }
    validate_type(g, &f.ret, &f.name, "ret", diags);
    for (i, e) in f.effects.iter().enumerate() {
        if !KNOWN_EFFECTS.contains(&e.as_str()) {
            diags.push(
                Diagnostic::unknown_name(&f.name, &format!("effects[{i}]"), e)
                    .hint(format!("known effects: {}", KNOWN_EFFECTS.join(", "))),
            );
        }
    }
    if f.name == "main" {
        let entry_hint = "the entry point must be `fn main(args: List[Str]) -> Int` (any effects)";
        if f.params.len() != 1 || f.params[0].ty != Type::list(Type::Str) {
            let found: Vec<String> = f.params.iter().map(|p| p.ty.to_string()).collect();
            diags.push(Diagnostic::type_mismatch(&f.name, "params", "(List[Str])", &format!("({})", found.join(", "))).hint(entry_hint));
        }
        if f.ret != Type::Int {
            diags.push(Diagnostic::type_mismatch(&f.name, "ret", "Int", &f.ret.to_string()).hint(entry_hint));
        }
    }

    let mut cx = Cx::new(g, &f.name, &[], diags);
    cx.start = before;
    for p in &f.params {
        cx.bind(&p.name, p.ty.clone());
    }
    let mut exprs = Vec::new();
    for (i, c) in f.req.iter().enumerate() {
        exprs.push(cx.expect(c, &Type::Bool, &format!("req[{i}]")));
    }
    let result_local = (!f.ens.is_empty()).then(|| cx.bind(RESULT_VAR, f.ret.clone()));
    for (i, c) in f.ens.iter().enumerate() {
        exprs.push(cx.expect(c, &Type::Bool, &format!("ens[{i}]")));
    }
    if result_local.is_some() {
        cx.unbind();
    }
    cx.effects = &f.effects;
    exprs.push(cx.expect(&f.body, &f.ret, "body"));
    let (locals, mut exprs) = cx.finish(exprs)?;
    let body = exprs.pop().expect("body");
    let ens = exprs.split_off(f.req.len());
    Some(TFn { def: f.clone(), locals, req: exprs, ens, result_local, body })
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `cd qal && cargo test -p qal-check 2>&1 | grep -E "test result|warning"`
Expected: `test result: ok. 28 passed`, no warnings.

- [ ] **Step 9: Commit**

```bash
git add qal/crates/qal-check
git commit -m "feat(qal): checker returns a typed program — inference, resolution, effects, exhaustive match

Also: parameter r with ens is an error, let-value errors still check the body,
and arguments are checked even when the arity is wrong.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Code generation with memory and runtime library (`qal-wasm`)

**Files:**
- Modify (full rewrite): `qal/crates/qal-wasm/src/lib.rs`
- Create: `qal/crates/qal-wasm/src/asm.rs`, `qal/crates/qal-wasm/src/module.rs`, `qal/crates/qal-wasm/src/rt.rs`, `qal/crates/qal-wasm/src/gen.rs`
- Test: `qal/crates/qal-wasm/tests/codegen.rs` (full rewrite)

**Interfaces:**
- Consumes: Task 1 `tast` (input must come from `qal_check::check`); dev: `qal_text::parse_program`, `qal_check::check`, `wasmparser::validate`, `wasmtime`.
- Produces:
  - `qal_wasm::compile(&TProgram, &Entry) -> CompiledModule`; `Entry::{Fn(String), Tests, All}`.
  - `CompiledModule { wasm, funcs: Vec<FuncMeta>, types: Vec<TypeDef>, imports: Vec<&'static str> }` with `func(export)`.
  - `FuncMeta { name, export, is_test, params: Vec<(String, Type)>, ret: Type, effects, req: Vec<Expr>, ens: Vec<Expr>, sites: Vec<Site> }` — reachable fns in program order, then tests (`test:<name>`).
  - `Site { path, label, contract: Option<usize>, values: Vec<(String, Type)> }`.
  - Host imports (module `HOST_MODULE = "qal_host"`, names in `qal_wasm::host`): `fail(kind, func, site, vals_ptr)`, `float_to_str(f64) -> i32`, `str_to_float(i32) -> (i32, f64)`, `io.print(i32)`, `io.read_line() -> i32` (-1 = end), `fs.read_file(i32) -> (i32 ok, i32 str)`, `fs.write_file(i32, i32) -> (i32 ok, i32 err)`, `time.now_ms() -> i64`, `rand.rand_int(i64, i64) -> i64`; `host::signature(name)`, `host::effect(name)`.
  - Failure kinds `FAIL_REQ = 1, FAIL_ENS = 2, FAIL_OVERFLOW = 3, FAIL_DIV_ZERO = 4, FAIL_INDEX = 5, FAIL_BUILTIN = 6, FAIL_OOM = 7` (OOM: `func = site = -1`, requested size as i64 at `OOM_SCRATCH = 0`); `fail`'s `vals_ptr` points to one 8-byte slot per `Site::values` entry.
  - Exports: each emitted fn by name, tests as `test:<name>`, `memory`, `qal_alloc(size: i64) -> i32`. Layout constants `TAG_TABLE = 8`, `HEAP_GLOBAL = 0`, `MAX_PAGES = 4096`; `Repr::{I32, I64, F64}`.

- [ ] **Step 1: Write the failing tests**

Replace `qal/crates/qal-wasm/tests/codegen.rs` with:

```rust
//! End-to-end codegen tests: text -> check -> wasm -> execute in wasmtime with a minimal host.

use qal_wasm::{compile, host, CompiledModule, Entry, FAIL_BUILTIN, FAIL_DIV_ZERO, FAIL_ENS, FAIL_INDEX, FAIL_OOM, FAIL_OVERFLOW, FAIL_REQ, HOST_MODULE};
use wasmtime::{Caller, Config, Engine, Extern, Instance, Linker, Memory, Module, Store, Val};

/// (kind, func, site, vals_ptr) recorded by `fail`.
type Failure = (i32, i32, i32, i32);

fn build(src: &str, entry: Entry) -> CompiledModule {
    let p = qal_text::parse_program(src).expect("parse");
    let tp = qal_check::check(&p).unwrap_or_else(|d| panic!("check failed: {d:#?}"));
    let m = compile(&tp, &entry);
    wasmparser::validate(&m.wasm).expect("valid wasm");
    m
}

fn memory(c: &mut Caller<'_, Option<Failure>>) -> Memory {
    match c.get_export("memory") {
        Some(Extern::Memory(m)) => m,
        _ => panic!("no memory export"),
    }
}

fn read_str(c: &mut Caller<'_, Option<Failure>>, ptr: i32) -> String {
    let mem = memory(c);
    let data = mem.data(&c);
    let len = u32::from_le_bytes(data[ptr as usize..ptr as usize + 4].try_into().unwrap()) as usize;
    String::from_utf8(data[ptr as usize + 4..ptr as usize + 4 + len].to_vec()).unwrap()
}

fn new_str(c: &mut Caller<'_, Option<Failure>>, s: &str) -> i32 {
    let alloc = c.get_export("qal_alloc").and_then(Extern::into_func).unwrap().typed::<i64, i32>(&c).unwrap();
    let p = alloc.call(&mut *c, 4 + s.len() as i64).unwrap();
    let mem = memory(c);
    mem.write(&mut *c, p as usize, &(s.len() as u32).to_le_bytes()).unwrap();
    mem.write(&mut *c, p as usize + 4, s.as_bytes()).unwrap();
    p
}

struct Run {
    store: Store<Option<Failure>>,
    instance: Instance,
}

fn instantiate(m: &CompiledModule) -> Run {
    let mut config = Config::new();
    config.wasm_tail_call(true);
    let engine = Engine::new(&config).unwrap();
    let module = Module::new(&engine, &m.wasm).unwrap();
    let mut store: Store<Option<Failure>> = Store::new(&engine, None);
    let mut linker = Linker::new(&engine);
    linker
        .func_wrap(HOST_MODULE, host::FAIL, |mut c: Caller<'_, Option<Failure>>, k: i32, f: i32, s: i32, v: i32| {
            *c.data_mut() = Some((k, f, s, v));
            Err::<(), _>(wasmtime::Error::msg("qal failure"))
        })
        .unwrap();
    linker
        .func_wrap(HOST_MODULE, host::FLOAT_TO_STR, |mut c: Caller<'_, Option<Failure>>, x: f64| new_str(&mut c, &format!("{x:?}")))
        .unwrap();
    linker
        .func_wrap(HOST_MODULE, host::STR_TO_FLOAT, |mut c: Caller<'_, Option<Failure>>, p: i32| match read_str(&mut c, p).parse::<f64>() {
            Ok(x) => (1, x),
            Err(_) => (0, 0.0),
        })
        .unwrap();
    let instance = linker.instantiate(&mut store, &module).unwrap();
    Run { store, instance }
}

impl Run {
    fn call(&mut self, export: &str, args: &[Val]) -> Result<Val, Failure> {
        let func = self.instance.get_func(&mut self.store, export).unwrap_or_else(|| panic!("no export {export}"));
        let mut results = vec![Val::I32(0)];
        match func.call(&mut self.store, args, &mut results) {
            Ok(()) => Ok(results[0]),
            Err(e) => Err(self.store.data().unwrap_or_else(|| panic!("unexpected trap: {e:?}"))),
        }
    }

    fn slot(&mut self, ptr: i32, k: usize) -> i64 {
        let mem = self.instance.get_memory(&mut self.store, "memory").unwrap();
        let at = ptr as usize + 8 * k;
        i64::from_le_bytes(mem.data(&self.store)[at..at + 8].try_into().unwrap())
    }
}

fn call_i64(m: &CompiledModule, export: &str, args: &[i64]) -> Result<i64, Failure> {
    let mut r = instantiate(m);
    let args: Vec<Val> = args.iter().map(|a| Val::I64(*a)).collect();
    r.call(export, &args).map(|v| match v {
        Val::I64(v) => v,
        Val::I32(v) => v as i64,
        other => panic!("unexpected {other:?}"),
    })
}

/// Runs every test definition and asserts each returns true.
fn all_tests_pass(src: &str) {
    let m = build(src, Entry::Tests);
    let mut r = instantiate(&m);
    for f in m.funcs.iter().filter(|f| f.is_test) {
        let result = r.call(&f.export, &[]).map(|v| v.unwrap_i32());
        assert_eq!(result, Ok(1), "test `{}` failed", f.name);
    }
}

#[test]
fn arithmetic_and_precedence() {
    let m = build("fn f(a: Int, b: Int) -> Int = a + b * 2 - a / b % 3", Entry::All);
    assert_eq!(call_i64(&m, "f", &[7, 2]), Ok(7 + 2 * 2 - (7 / 2) % 3));
    assert_eq!(call_i64(&m, "f", &[-7, 2]), Ok(-7 + 4 - (-7 / 2) % 3));
}

#[test]
fn booleans_if_let_and_unit() {
    all_tests_pass(
        "fn max(a: Int, b: Int) -> Int = if a > b then a else b\n\
         fn both(a: Bool, b: Bool) -> Bool = a && !b || false\n\
         fn twice(x: Int) -> Int = let y = x + x in let y = y + 1 in y\n\
         fn u() -> Unit = ()\n\
         test t1 = max(3, 9) == 9 && max(-1, -5) == -1\n\
         test t2 = both(true, false) && !both(true, true)\n\
         test t3 = twice(4) == 9 && u() == ()",
    );
}

#[test]
fn integer_overflow_is_an_error_not_a_wrap() {
    let m = build(
        "fn add(a: Int, b: Int) -> Int = a + b\nfn sub(a: Int, b: Int) -> Int = a - b\n\
         fn mul(a: Int, b: Int) -> Int = a * b\nfn div(a: Int, b: Int) -> Int = a / b\n\
         fn rem(a: Int, b: Int) -> Int = a % b\nfn neg(a: Int) -> Int = -a",
        Entry::All,
    );
    let (max, min) = (i64::MAX, i64::MIN);
    let kind = |r: Result<i64, Failure>| r.map_err(|f| f.0);
    assert_eq!(kind(call_i64(&m, "add", &[max, 1])), Err(FAIL_OVERFLOW));
    assert_eq!(kind(call_i64(&m, "sub", &[min, 1])), Err(FAIL_OVERFLOW));
    assert_eq!(kind(call_i64(&m, "mul", &[min, -1])), Err(FAIL_OVERFLOW));
    assert_eq!(kind(call_i64(&m, "mul", &[-1, min])), Err(FAIL_OVERFLOW));
    assert_eq!(kind(call_i64(&m, "mul", &[1 << 32, 1 << 32])), Err(FAIL_OVERFLOW));
    assert_eq!(kind(call_i64(&m, "div", &[min, -1])), Err(FAIL_OVERFLOW));
    assert_eq!(kind(call_i64(&m, "div", &[1, 0])), Err(FAIL_DIV_ZERO));
    assert_eq!(kind(call_i64(&m, "rem", &[1, 0])), Err(FAIL_DIV_ZERO));
    assert_eq!(kind(call_i64(&m, "neg", &[min])), Err(FAIL_OVERFLOW));
    assert_eq!(call_i64(&m, "rem", &[min, -1]), Ok(0));
    assert_eq!(call_i64(&m, "add", &[max - 1, 1]), Ok(max));
}

#[test]
fn failures_carry_site_and_operand_values() {
    let m = build("fn f(a: Int, b: Int) -> Int = a + (a * b)", Entry::All);
    let mut r = instantiate(&m);
    let (kind, func, site, vals) = r.call("f", &[Val::I64(2), Val::I64(i64::MAX)]).unwrap_err();
    assert_eq!((kind, func), (FAIL_OVERFLOW, 0));
    let s = &m.funcs[0].sites[site as usize];
    assert_eq!((s.path.as_str(), s.label.as_str()), ("body.right", "*"));
    assert_eq!((r.slot(vals, 0), r.slot(vals, 1)), (2, i64::MAX));
}

#[test]
fn contracts_are_checked_at_runtime_with_values() {
    let m = build(
        "fn sqrt_floor(n: Int) -> Int\n  req n >= 0\n  ens r * r <= n\n  ens (r + 1) * (r + 1) > n\n= go(n, 0)\n\
         fn go(n: Int, i: Int) -> Int = if (i + 1) * (i + 1) > n then i else go(n, i + 1)\n\
         fn liar(n: Int) -> Int ens r > n = n",
        Entry::All,
    );
    assert_eq!(call_i64(&m, "sqrt_floor", &[17]), Ok(4));
    let mut r = instantiate(&m);
    let (kind, _, site, vals) = r.call("sqrt_floor", &[Val::I64(-4)]).unwrap_err();
    assert_eq!(kind, FAIL_REQ);
    assert_eq!(m.funcs[0].sites[site as usize].contract, Some(0));
    assert_eq!(r.slot(vals, 0), -4);
    let (kind, func, site, vals) = r.call("liar", &[Val::I64(5)]).unwrap_err();
    assert_eq!(kind, FAIL_ENS);
    let s = &m.funcs[func as usize].sites[site as usize];
    assert_eq!(s.values.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(), vec!["n", "r"]);
    assert_eq!((r.slot(vals, 0), r.slot(vals, 1)), (5, 5));
}

#[test]
fn ens_in_recursive_function_checks_each_call() {
    // fact(1) = 1 violates `r > n` only for n = 1; the inner call must be checked, not just the outer.
    let m = build("fn fact(n: Int) -> Int ens r > n = if n <= 1 then 1 else n * fact(n - 1)", Entry::All);
    assert_eq!(call_i64(&m, "fact", &[3]).map_err(|f| f.0), Err(FAIL_ENS));
}

#[test]
fn deep_tail_recursion_does_not_overflow_the_stack() {
    let m = build(
        "fn count(n: Int, acc: Int) -> Int = if n == 0 then acc else count(n - 1, acc + 1)\n\
         fn walk(n: Int) -> Int = match n { 0 => 0, _ => { walk(n - 1) } }",
        Entry::All,
    );
    assert_eq!(call_i64(&m, "count", &[1_000_000, 0]), Ok(1_000_000));
    assert_eq!(call_i64(&m, "walk", &[1_000_000]), Ok(0));
}

#[test]
fn floats() {
    all_tests_pass(
        "fn avg(a: Float, b: Float) -> Float = (a + b) / 2.0\n\
         test arith = avg(1.0, 2.0) == 1.5 && -avg(1.0, 1.0) < 0.0 && 1.0 / 0.0 > 1e308\n\
         test convert = int_to_float(3) == 3.0 && float_to_int(2.9) == Some(2) && float_to_int(-2.9) == Some(-2)\n\
         test convert_range = float_to_int(1e19) == None && float_to_int(0.0 / 0.0) == None && float_to_int(-9.2e18) != None\n\
         test text = from_float(1.5) == \"1.5\" && to_float(\"2.25\") == Some(2.25) && to_float(\"x\") == None\n\
         test nan = 0.0 / 0.0 != 0.0 / 0.0\n\
         test nan_in_values = (let l = [0.0 / 0.0] in l != l) && (let o = Some(0.0 / 0.0) in o != o)",
    );
}

#[test]
fn strings() {
    all_tests_pass(
        "test lit = \"héllo\" == \"héllo\" && \"a\" != \"b\" && \"\" == \"\"\n\
         test cat = \"ab\" ++ \"\" ++ \"cd\" == \"abcd\"\n\
         test length = len(\"héllo\") == 5 && len(\"\") == 0\n\
         test slicing = slice(\"héllo\", 1, 3) == \"él\" && slice(\"abc\", 0, 3) == \"abc\" && slice(\"abc\", 2, 2) == \"\"\n\
         test splitting = split(\"a,b,,c\", \",\") == [\"a\", \"b\", \"\", \"c\"] && split(\"abc\", \"--\") == [\"abc\"] && split(\"a--b\", \"--\") == [\"a\", \"b\"]\n\
         test joining = join([\"a\", \"b\", \"c\"], \", \") == \"a, b, c\" && join([], \"x\") == \"\" && join([\"z\"], \"-\") == \"z\"\n\
         test trimming = trim(\" \\t a b \\n\") == \"a b\" && trim(\"   \") == \"\" && trim(\"x\") == \"x\"\n\
         test search = contains(\"hello\", \"ll\") && !contains(\"hello\", \"lo!\") && contains(\"a\", \"\") && starts_with(\"hello\", \"he\") && !starts_with(\"h\", \"he\")\n\
         test ints = to_int(\"42\") == Some(42) && to_int(\"-7\") == Some(-7) && to_int(\"+3\") == Some(3) && to_int(\"\") == None && to_int(\"-\") == None && to_int(\"1x\") == None && to_int(\" 1\") == None\n\
         test int_limits = to_int(\"9223372036854775807\") == Some(9223372036854775807) && to_int(\"-9223372036854775808\") == Some(-9223372036854775807 - 1) && to_int(\"9223372036854775808\") == None\n\
         test from_ints = from_int(0) == \"0\" && from_int(-120) == \"-120\" && from_int(-9223372036854775807 - 1) == \"-9223372036854775808\"\n\
         test char_list = chars(\"aé😀\") == [\"a\", \"é\", \"😀\"] && chars(\"\") == []",
    );
}

#[test]
fn lists_and_function_values() {
    all_tests_pass(
        "fn double(x: Int) -> Int = x * 2\n\
         fn positive(x: Int) -> Bool = x > 0\n\
         fn add(a: Int, b: Int) -> Int = a + b\n\
         fn shout(s: Str) -> Str = s ++ \"!\"\n\
         fn half(x: Int) -> Float = int_to_float(x) / 2.0\n\
         fn apply(f: fn(Int) -> Int, x: Int) -> Int = f(x)\n\
         fn pick(b: Bool) -> fn(Int) -> Int = if b then double else negate\n\
         fn negate(x: Int) -> Int = -x\n\
         test basics = len([1, 2, 3]) == 3 && get([1, 2], 1) == Some(2) && get([1], 1) == None && get([1], -1) == None\n\
         test building = push([1], 2) == [1, 2] && concat([1], [2, 3]) == [1, 2, 3] && [1] ++ [] == [1] && reverse([1, 2, 3]) == [3, 2, 1]\n\
         test ranges = range(0, 4) == [0, 1, 2, 3] && range(3, 3) == [] && range(5, 1) == [] && len(range(-2, 2)) == 4\n\
         test sorting = sort_int([3, -1, 2, 2, 0]) == [-1, 0, 2, 2, 3] && sort_int([]) == [] && sort_int(reverse(range(0, 1000))) == range(0, 1000)\n\
         test higher_order = map([1, 2], double) == [2, 4] && filter(range(-3, 4), positive) == [1, 2, 3] && fold([1, 2, 3], 10, add) == 16\n\
         test mixed_reprs = map([\"a\"], shout) == [\"a!\"] && map([1, 3], half) == [0.5, 1.5]\n\
         test fn_values = apply(double, 5) == 10 && (let f = pick(false) in f(3)) == -3 && apply(pick(true), 1) == 2\n\
         test nested = [[1], []] == [[1], []] && [[1]] != [[2]] && get([[\"x\"]], 0) == Some([\"x\"])",
    );
}

#[test]
fn records_enums_and_match() {
    all_tests_pass(
        "type User = { id: Int, name: Str, tags: List[Str] }\n\
         type Shape = Circle(Float) | Rect(Float, Float) | Empty\n\
         type Tree = Leaf | Node(Tree, Int, Tree)\n\
         fn area(s: Shape) -> Float = match s { Circle(r) => 3.0 * r * r, Rect(w, h) => w * h, Empty => 0.0 }\n\
         fn sum(t: Tree) -> Int = match t { Leaf => 0, Node(l, v, r) => sum(l) + v + sum(r) }\n\
         fn classify(n: Int) -> Str = match n { 0 => \"zero\", -1 => \"minus one\", x if x > 0 => \"pos\", _ => \"neg\" }\n\
         fn greet(u: User) -> Str = match u { User { name: \"root\" } => \"admin\", User { id, name } => name ++ from_int(id) }\n\
         fn first(xs: List[Int]) -> Int = match get(xs, 0) { Some(n) => n, None => -1 }\n\
         fn word(s: Str) -> Int = match s { \"one\" => 1, \"two\" => 2, _ => 0 }\n\
         fn both(a: Bool, b: Bool) -> Int = match a { true => match b { true => 3, false => 2 }, false => 1 }\n\
         test records = User { id: 1, name: \"a\", tags: [] }.name == \"a\" && User { id: 1, name: \"a\", tags: [\"x\"] } == User { tags: [\"x\"], name: \"a\", id: 1 }\n\
         test record_ne = User { id: 1, name: \"a\", tags: [] } != User { id: 2, name: \"a\", tags: [] }\n\
         test enums = area(Rect(2.0, 3.0)) == 6.0 && area(Empty) == 0.0 && Circle(1.0) == Circle(1.0) && Circle(1.0) != Rect(1.0, 1.0) && Empty == Empty\n\
         test trees = sum(Node(Node(Leaf, 1, Leaf), 2, Node(Leaf, 3, Leaf))) == 6 && Node(Leaf, 1, Leaf) == Node(Leaf, 1, Leaf)\n\
         test literals = classify(0) == \"zero\" && classify(-1) == \"minus one\" && classify(5) == \"pos\" && classify(-5) == \"neg\"\n\
         test record_patterns = greet(User { id: 7, name: \"root\", tags: [] }) == \"admin\" && greet(User { id: 7, name: \"bob\", tags: [] }) == \"bob7\"\n\
         test options = first([4, 5]) == 4 && first([]) == -1 && Some(Some(1)) == Some(Some(1)) && Ok(1) != Err(\"x\")\n\
         test strings_and_bools = word(\"two\") == 2 && word(\"three\") == 0 && both(true, false) == 2 && both(false, true) == 1",
    );
}

#[test]
fn sequences_evaluate_left_to_right() {
    all_tests_pass("fn f(x: Int) -> Int = { (); (); x + 1 }\ntest t = f(1) == 2 && { (); true }");
}

#[test]
fn builtin_preconditions_fail_with_values() {
    let m = build("fn cut() -> Str = slice(\"héllo\", 2, 9)\nfn parts() -> List[Str] = split(\"abc\", \"\")", Entry::All);
    let mut r = instantiate(&m);
    let (kind, func, site, vals) = r.call("cut", &[]).unwrap_err();
    assert_eq!(kind, FAIL_INDEX);
    assert_eq!(m.funcs[func as usize].sites[site as usize].label, "slice");
    assert_eq!((r.slot(vals, 0), r.slot(vals, 1), r.slot(vals, 2)), (2, 9, 5));
    let (kind, _, _, _) = r.call("parts", &[]).unwrap_err();
    assert_eq!(kind, FAIL_BUILTIN);
}

#[test]
fn exhausting_memory_is_out_of_memory() {
    let m = build("fn big(n: Int) -> Int = len(range(0, n))", Entry::All);
    let mut r = instantiate(&m);
    let (kind, func, site, ptr) = r.call("big", &[Val::I64(40_000_000)]).unwrap_err();
    assert_eq!((kind, func, site), (FAIL_OOM, -1, -1));
    assert_eq!(r.slot(ptr, 0), 8 + 8 * 40_000_000);
    assert_eq!(call_i64(&m, "big", &[1_000_000]), Ok(1_000_000));
}

#[test]
fn huge_join_is_out_of_memory_not_a_wrapped_length() {
    // 4 Mi empty strings joined by a 1 KiB separator need 4 GiB: the length must not wrap in i32.
    let src = "fn empty(i: Int) -> Str = \"\"\n\
               fn rep(s: Str, n: Int) -> Str = if n == 0 then s else rep(s ++ s, n - 1)\n\
               fn big() -> Int = len(join(map(range(0, 4194305), empty), rep(\"a\", 10)))";
    let m = build(src, Entry::All);
    assert_eq!(call_i64(&m, "big", &[]).map_err(|f| f.0), Err(FAIL_OOM));
}

#[test]
fn only_needed_host_functions_are_imported() {
    let pure = build("fn f(x: Int) -> Int = x\nfn g(x: Float) -> Str = from_float(x)", Entry::Fn("f".into()));
    assert_eq!(pure.imports, vec![host::FAIL]);
    assert_eq!(pure.funcs.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), vec!["f"]);
    let io = build(
        "fn main(args: List[Str]) -> Int !io !fs = { print(\"hi\"); helper() }\nfn helper() -> Int !io = { print(\"x\"); 0 }\n\
         fn unused(p: Str) -> Result[Str, Str] !fs = read_file(p)",
        Entry::Fn("main".into()),
    );
    assert_eq!(io.imports, vec![host::FAIL, host::PRINT]);
    let imports: Vec<String> = wasmparser::Parser::new(0)
        .parse_all(&io.wasm)
        .filter_map(|p| match p.unwrap() {
            wasmparser::Payload::ImportSection(s) => Some(s.into_imports().map(|i| i.unwrap().name.to_string()).collect::<Vec<_>>()),
            _ => None,
        })
        .flatten()
        .collect();
    assert_eq!(imports, vec!["fail", "io.print"]);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd qal && cargo test -p qal-wasm`
Expected: FAIL to compile (`Entry`, `host`, `FAIL_INDEX`, `HOST_MODULE` not found; the 1a codegen does not compile against the new AST).

- [ ] **Step 3: Write the assembly helpers**

Create `qal/crates/qal-wasm/src/asm.rs`:

```rust
//! Small helpers for writing WebAssembly function bodies by hand.

use qal_core::ast::Type;
use wasm_encoder::{Function, Instruction as I, MemArg, ValType};

/// Machine representation of a QAL value: Int = I64, Float = F64, everything else = I32
/// (Bool/Unit as 0/1, all heap values and function references as pointers/table indices).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Repr {
    I32,
    I64,
    F64,
}

impl Repr {
    pub fn of(t: &Type) -> Repr {
        match t {
            Type::Int => Repr::I64,
            Type::Float => Repr::F64,
            _ => Repr::I32,
        }
    }

    pub fn val(self) -> ValType {
        match self {
            Repr::I32 => ValType::I32,
            Repr::I64 => ValType::I64,
            Repr::F64 => ValType::F64,
        }
    }
}

pub fn val(t: &Type) -> ValType {
    Repr::of(t).val()
}

pub fn m8(offset: u64) -> MemArg {
    MemArg { offset, align: 0, memory_index: 0 }
}

pub fn m32(offset: u64) -> MemArg {
    MemArg { offset, align: 2, memory_index: 0 }
}

pub fn m64(offset: u64) -> MemArg {
    MemArg { offset, align: 3, memory_index: 0 }
}

/// Converts the value on the stack to its 8-byte slot form (i64).
pub fn to_slot(r: Repr) -> Option<I<'static>> {
    match r {
        Repr::I32 => Some(I::I64ExtendI32U),
        Repr::I64 => None,
        Repr::F64 => Some(I::I64ReinterpretF64),
    }
}

/// Converts an i64 slot on the stack back to the value representation.
pub fn from_slot(r: Repr) -> Option<I<'static>> {
    match r {
        Repr::I32 => Some(I::I32WrapI64),
        Repr::I64 => None,
        Repr::F64 => Some(I::F64ReinterpretI64),
    }
}

/// A function body under construction: parameters are locals `0..n`, extra locals follow.
pub struct Asm {
    n_params: u32,
    locals: Vec<ValType>,
    pub ins: Vec<I<'static>>,
}

impl Asm {
    pub fn new(n_params: usize) -> Asm {
        Asm { n_params: n_params as u32, locals: Vec::new(), ins: Vec::new() }
    }

    pub fn local(&mut self, v: ValType) -> u32 {
        self.locals.push(v);
        self.n_params + self.locals.len() as u32 - 1
    }

    pub fn i(&mut self, ins: I<'static>) {
        self.ins.push(ins);
    }

    pub fn all<const N: usize>(&mut self, ins: [I<'static>; N]) {
        self.ins.extend(ins);
    }

    pub fn opt(&mut self, ins: Option<I<'static>>) {
        self.ins.extend(ins);
    }

    pub fn finish(mut self) -> Function {
        self.ins.push(I::End);
        let mut f = Function::new(self.locals.iter().map(|v| (1, *v)));
        for ins in &self.ins {
            f.instruction(ins);
        }
        f
    }
}
```

- [ ] **Step 4: Write the module builder**

Create `qal/crates/qal-wasm/src/module.rs`:

```rust
//! Module builder: index spaces (types, imports, functions), static data, and lazily generated helpers.
//!
//! Function indices are computed, never fixed: imports come first, then user definitions, then
//! runtime-library and equality helpers in the order they are first needed.

use crate::asm::{from_slot, m32, m64, to_slot, Asm, Repr};
use crate::rt::Rt;
use crate::{host, MAX_PAGES, TAG_TABLE};
use qal_core::ast::{Type, TypeBody};
use qal_core::tast::TProgram;
use std::borrow::Cow;
use std::collections::HashMap;
use wasm_encoder::{
    BlockType, CodeSection, ConstExpr, DataSection, ElementSection, Elements, EntityType, ExportKind, ExportSection, Function,
    FunctionSection, GlobalSection, GlobalType, ImportSection, Instruction as I, MemoryType, Module, RefType, TableSection,
    TableType, TypeSection, ValType,
};

/// Where a runtime failure's `func` / `site` argument comes from.
#[derive(Clone, Copy)]
pub enum Arg {
    Const(i32),
    Local(u32),
}

impl Arg {
    pub fn push(self, a: &mut Asm) {
        a.i(match self {
            Arg::Const(c) => I::I32Const(c),
            Arg::Local(l) => I::LocalGet(l),
        });
    }
}

pub struct Mb<'p> {
    pub p: &'p TProgram,
    types: Vec<(Vec<ValType>, Vec<ValType>)>,
    imports: Vec<(&'static str, u32)>,
    funcs: Vec<(u32, Option<Function>)>,
    rt: HashMap<Rt, u32>,
    eqs: HashMap<Type, u32>,
    strs: HashMap<String, u32>,
    data: Vec<u8>,
}

impl<'p> Mb<'p> {
    /// `imports`: host functions (names from `crate::host`) this module needs; `fail` must be among them.
    pub fn new(p: &'p TProgram, imports: &[&'static str]) -> Mb<'p> {
        let mut mb = Mb { p, types: Vec::new(), imports: Vec::new(), funcs: Vec::new(), rt: HashMap::new(), eqs: HashMap::new(), strs: HashMap::new(), data: Vec::new() };
        for name in imports {
            let (params, results) = host::signature(name);
            let ty = mb.ty(params, results);
            mb.imports.push((name, ty));
        }
        // Static data: OOM scratch slot, then one 8-byte block per tag for payload-free variants.
        let max_tags = p
            .types
            .iter()
            .map(|t| match &t.body {
                TypeBody::Enum(vs) => vs.len(),
                TypeBody::Record(_) => 0,
            })
            .max()
            .unwrap_or(0)
            .max(2);
        mb.data = vec![0; TAG_TABLE as usize];
        for tag in 0..max_tags as u32 {
            mb.data.extend(tag.to_le_bytes());
            mb.data.extend([0; 4]);
        }
        mb
    }

    pub fn ty(&mut self, params: &[ValType], results: &[ValType]) -> u32 {
        let key = (params.to_vec(), results.to_vec());
        match self.types.iter().position(|t| *t == key) {
            Some(i) => i as u32,
            None => {
                self.types.push(key);
                self.types.len() as u32 - 1
            }
        }
    }

    /// Function index of an imported host function.
    pub fn import(&self, name: &str) -> u32 {
        self.imports.iter().position(|(n, _)| *n == name).unwrap_or_else(|| panic!("host import `{name}` not declared")) as u32
    }

    /// Declares a function and returns its index; its body is supplied later with `define`.
    pub fn declare(&mut self, params: &[ValType], results: &[ValType]) -> u32 {
        let ty = self.ty(params, results);
        self.funcs.push((ty, None));
        (self.imports.len() + self.funcs.len() - 1) as u32
    }

    pub fn define(&mut self, index: u32, f: Function) {
        self.funcs[index as usize - self.imports.len()].1 = Some(f);
    }

    /// Index of a runtime-library helper, generating it on first use.
    pub fn rt(&mut self, key: Rt) -> u32 {
        if let Some(i) = self.rt.get(&key) {
            return *i;
        }
        let (params, results) = key.signature();
        let index = self.declare(&params, &results);
        self.rt.insert(key, index);
        let body = key.body(self, params.len());
        self.define(index, body);
        index
    }

    /// Address of a static string `[len: i32][bytes]`.
    pub fn str_addr(&mut self, s: &str) -> u32 {
        if let Some(a) = self.strs.get(s) {
            return *a;
        }
        while !self.data.len().is_multiple_of(8) {
            self.data.push(0);
        }
        let addr = self.data.len() as u32;
        self.data.extend((s.len() as u32).to_le_bytes());
        self.data.extend(s.as_bytes());
        self.strs.insert(s.to_string(), addr);
        addr
    }

    /// Address of the shared value of a variant without payload.
    pub fn tag_addr(&self, tag: u32) -> i32 {
        (TAG_TABLE + 8 * tag) as i32
    }

    /// Emits a runtime failure: copies `vals` (local, repr) into a fresh block of slots, calls
    /// `qal_host.fail(kind, func, site, vals_ptr)` and traps.
    pub fn emit_fail(&mut self, a: &mut Asm, kind: i32, func: Arg, site: Arg, vals: &[(u32, Repr)]) {
        let ptr = a.local(ValType::I32);
        if vals.is_empty() {
            a.all([I::I32Const(0), I::LocalSet(ptr)]);
        } else {
            let alloc = self.rt(Rt::Alloc);
            a.all([I::I64Const(8 * vals.len() as i64), I::Call(alloc), I::LocalSet(ptr)]);
            for (k, (local, r)) in vals.iter().enumerate() {
                a.all([I::LocalGet(ptr), I::LocalGet(*local)]);
                a.opt(to_slot(*r));
                a.i(I::I64Store(m64(8 * k as u64)));
            }
        }
        a.i(I::I32Const(kind));
        func.push(a);
        site.push(a);
        a.all([I::LocalGet(ptr), I::Call(self.import(host::FAIL)), I::Unreachable]);
    }

    /// With two values of type `t` on the stack, leaves 1 if they are equal, else 0.
    pub fn emit_eq(&mut self, a: &mut Asm, t: &Type) {
        match Repr::of(t) {
            Repr::I64 => a.i(I::I64Eq),
            Repr::F64 => a.i(I::F64Eq),
            Repr::I32 => match t {
                Type::Bool | Type::Unit | Type::Fn(..) => a.i(I::I32Eq),
                _ => {
                    let f = self.eq_fn(t);
                    a.i(I::Call(f));
                }
            },
        }
    }

    /// Structural equality function `(a: i32, b: i32) -> i32` for a heap type.
    fn eq_fn(&mut self, t: &Type) -> u32 {
        if *t == Type::Str {
            return self.rt(Rt::StrEq);
        }
        if let Some(i) = self.eqs.get(t) {
            return *i;
        }
        let index = self.declare(&[ValType::I32, ValType::I32], &[ValType::I32]);
        self.eqs.insert(t.clone(), index);
        let mut a = Asm::new(2);
        // Identical pointers are equal (also covers shared payload-free variants) — unless a Float
        // inside could be NaN, which is never equal to itself.
        if !self.has_float(t, &mut Vec::new()) {
            a.all([I::LocalGet(0), I::LocalGet(1), I::I32Eq, I::If(BlockType::Empty), I::I32Const(1), I::Return, I::End]);
        }
        let field = |mb: &mut Self, a: &mut Asm, ft: &Type, offset: u64| {
            for side in [0, 1] {
                a.all([I::LocalGet(side), I::I64Load(m64(offset))]);
                a.opt(from_slot(Repr::of(ft)));
            }
            mb.emit_eq(a, ft);
            a.all([I::I32Eqz, I::If(BlockType::Empty), I::I32Const(0), I::Return, I::End]);
        };
        match t {
            Type::List(elem) => {
                let (n, k) = (a.local(ValType::I32), a.local(ValType::I32));
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalTee(n), I::LocalGet(1), I::I32Load(m32(0)), I::I32Ne]);
                a.all([I::If(BlockType::Empty), I::I32Const(0), I::Return, I::End]);
                a.all([I::Block(BlockType::Empty), I::Loop(BlockType::Empty)]);
                a.all([I::LocalGet(k), I::LocalGet(n), I::I32GeU, I::BrIf(1)]);
                for side in [0, 1] {
                    a.all([I::LocalGet(side), I::LocalGet(k), I::I32Const(3), I::I32Shl, I::I32Add, I::I64Load(m64(8))]);
                    a.opt(from_slot(Repr::of(elem)));
                }
                self.emit_eq(&mut a, elem);
                a.all([I::I32Eqz, I::If(BlockType::Empty), I::I32Const(0), I::Return, I::End]);
                a.all([I::LocalGet(k), I::I32Const(1), I::I32Add, I::LocalSet(k), I::Br(0), I::End, I::End]);
            }
            _ if self.p.is_record(t) => {
                for (j, (_, ft)) in self.p.record_fields(t).iter().enumerate() {
                    field(self, &mut a, ft, 8 * j as u64);
                }
            }
            _ => {
                let tag = a.local(ValType::I32);
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalTee(tag), I::LocalGet(1), I::I32Load(m32(0)), I::I32Ne]);
                a.all([I::If(BlockType::Empty), I::I32Const(0), I::Return, I::End]);
                for v in 0..self.p.variant_names(t).len() as u32 {
                    let fields = self.p.variant_fields(t, v);
                    if fields.is_empty() {
                        continue;
                    }
                    a.all([I::LocalGet(tag), I::I32Const(v as i32), I::I32Eq, I::If(BlockType::Empty)]);
                    for (j, ft) in fields.iter().enumerate() {
                        field(self, &mut a, ft, 8 + 8 * j as u64);
                    }
                    a.i(I::End);
                }
            }
        }
        a.i(I::I32Const(1));
        self.define(index, a.finish());
        index
    }

    /// True if a value of type `t` can contain a Float (`seen` guards recursive types).
    fn has_float(&self, t: &Type, seen: &mut Vec<String>) -> bool {
        match t {
            Type::Float => true,
            Type::List(x) | Type::Option(x) => self.has_float(x, seen),
            Type::Result(a, b) => self.has_float(a, seen) || self.has_float(b, seen),
            Type::Named(n) if !seen.contains(n) => {
                seen.push(n.clone());
                match self.p.type_def(n).map(|d| &d.body) {
                    Some(TypeBody::Record(fs)) => fs.iter().any(|(_, f)| self.has_float(f, seen)),
                    Some(TypeBody::Enum(vs)) => vs.iter().any(|v| v.fields.iter().any(|f| self.has_float(f, seen))),
                    None => false,
                }
            }
            _ => false,
        }
    }

    /// Assembles the module. `exports`: (name, function index); `table`: function indices for `call_indirect`.
    pub fn finish(mut self, exports: &[(String, u32)], table: &[u32]) -> Vec<u8> {
        let alloc = self.rt(Rt::Alloc);
        let mut types = TypeSection::new();
        for (p, r) in &self.types {
            types.ty().function(p.iter().copied(), r.iter().copied());
        }
        let mut imports = ImportSection::new();
        for (name, ty) in &self.imports {
            imports.import(crate::HOST_MODULE, name, EntityType::Function(*ty));
        }
        let mut functions = FunctionSection::new();
        let mut codes = CodeSection::new();
        for (ty, body) in &self.funcs {
            functions.function(*ty);
            codes.function(body.as_ref().expect("every declared function is defined"));
        }
        let mut tables = TableSection::new();
        tables.table(TableType { element_type: RefType::FUNCREF, table64: false, minimum: table.len() as u64, maximum: Some(table.len() as u64), shared: false });
        while !self.data.len().is_multiple_of(8) {
            self.data.push(0);
        }
        let heap_start = self.data.len() as u64;
        let mut memories = wasm_encoder::MemorySection::new();
        memories.memory(MemoryType { minimum: heap_start / 65536 + 1, maximum: Some(MAX_PAGES), memory64: false, shared: false, page_size_log2: None });
        let mut globals = GlobalSection::new();
        globals.global(GlobalType { val_type: ValType::I32, mutable: true, shared: false }, &ConstExpr::i32_const(heap_start as i32));
        let mut export = ExportSection::new();
        for (name, idx) in exports {
            export.export(name, ExportKind::Func, *idx);
        }
        export.export("memory", ExportKind::Memory, 0);
        export.export("qal_alloc", ExportKind::Func, alloc);
        let mut elements = ElementSection::new();
        if !table.is_empty() {
            elements.active(Some(0), &ConstExpr::i32_const(0), Elements::Functions(Cow::Borrowed(table)));
        }
        let mut data = DataSection::new();
        data.active(0, &ConstExpr::i32_const(0), self.data.iter().copied());
        let mut module = Module::new();
        module
            .section(&types)
            .section(&imports)
            .section(&functions)
            .section(&tables)
            .section(&memories)
            .section(&globals)
            .section(&export)
            .section(&elements)
            .section(&codes)
            .section(&data);
        module.finish()
    }
}
```

- [ ] **Step 5: Write the runtime library**

Create `qal/crates/qal-wasm/src/rt.rs`:

```rust
//! Runtime library: allocator, checked arithmetic and the built-in Str/List functions, written
//! directly in WebAssembly. Each helper is generated once, on first use.
//!
//! Memory layout (all blocks 8-byte aligned, see `crate` docs):
//!   Str  = [len: i32][UTF-8 bytes]          List = [len: i32][pad][slot; len]
//!   variant = [tag: i32][pad][slot; fields]  record = [slot; fields]
//! A slot is 8 bytes: i64 as is, f64 bit pattern, i32 zero-extended.

use crate::asm::{from_slot, m32, m64, m8, to_slot, Asm, Repr};
use crate::module::{Arg, Mb};
use crate::{host, FAIL_BUILTIN, FAIL_DIV_ZERO, FAIL_INDEX, FAIL_OOM, FAIL_OVERFLOW, HEAP_GLOBAL, OOM_SCRATCH};
use qal_core::ast::BinOp;
use qal_core::tast::{TAG_ERR, TAG_NONE, TAG_OK, TAG_SOME};
use wasm_encoder::{BlockType, Function, Instruction as I, ValType};

use ValType::{F64, I32, I64};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rt {
    /// (size: i64) -> ptr; fails with out_of_memory beyond the memory limit.
    Alloc,
    /// Checked Int `+ - * / %`: (a: i64, b: i64, func: i32, site: i32) -> i64.
    Arith(BinOp),
    /// (tag: i32, slot: i64) -> ptr of a one-field variant.
    Variant1,
    /// (ptr or -1) -> Option[Str]
    OptFromPtr,
    /// (ok: i32, str: i32) -> Result[Str, Str]
    ResultFrom,
    /// (ok: i32, err: i32) -> Result[Unit, Str]
    ResultUnitFrom,
    /// (len: i32) -> fresh Str with uninitialized bytes
    StrNew,
    /// (s) -> number of chars (i64)
    StrLen,
    StrConcat,
    /// (a_ptr, b_ptr, n) -> 1 if the n bytes are equal
    BytesEq,
    StrEq,
    /// (s, from_byte, to_byte) -> new Str
    StrSub,
    /// (s, char_index: i64) -> byte offset of that char (or byte length)
    CharToByte,
    /// (s, start: i64, end: i64, func, site) -> Str; fails with `index`
    StrSlice,
    /// (s, sub, from_byte) -> byte offset or -1
    StrFind,
    StrContains,
    StrStartsWith,
    /// (s, sep, func, site) -> List[Str]; fails if sep is empty
    StrSplit,
    StrJoin,
    StrTrim,
    /// (s) -> Option[Int]
    StrToInt,
    /// (n: i64) -> Str
    IntToStr,
    StrChars,
    /// (n: i32) -> fresh List with uninitialized slots
    ListNew,
    /// (xs, i: i64) -> Option[slot]
    ListGet,
    /// (xs, slot: i64) -> List
    ListPush,
    ListConcat,
    ListReverse,
    /// (a: i64, b: i64) -> List[Int]
    ListRange,
    ListSortInt,
    /// (base, root, end): heap sift-down over i64 elements
    Sift,
    /// (x: f64) -> Option[Int]
    FloatToInt,
    /// (lo: i64, hi: i64, func, site) -> i64
    RandChecked,
    /// (xs, f) -> List; f: (in) -> out
    Map(Repr, Repr),
    /// (xs, f) -> List; f: (in) -> Bool
    Filter(Repr),
    /// (xs, init: acc, f) -> acc; f: (acc, elem) -> acc
    Fold(Repr, Repr),
}

/// Pushes the address of slot `k` of list `xs` (use with offset 8).
fn slot_addr(a: &mut Asm, xs: u32, k: u32) {
    a.all([I::LocalGet(xs), I::LocalGet(k), I::I32Const(3), I::I32Shl, I::I32Add]);
}

/// `block { loop { <cond> br_if 1 ... } }` opener: exits when `k >= n` (unsigned i32).
fn loop_until(a: &mut Asm, k: u32, n: u32) {
    a.all([I::Block(BlockType::Empty), I::Loop(BlockType::Empty), I::LocalGet(k), I::LocalGet(n), I::I32GeU, I::BrIf(1)]);
}

/// Closes a `loop_until` loop, incrementing `k` by one.
fn loop_next(a: &mut Asm, k: u32) {
    a.all([I::LocalGet(k), I::I32Const(1), I::I32Add, I::LocalSet(k), I::Br(0), I::End, I::End]);
}

fn incr(a: &mut Asm, l: u32) {
    a.all([I::LocalGet(l), I::I32Const(1), I::I32Add, I::LocalSet(l)]);
}

impl Rt {
    pub fn signature(self) -> (Vec<ValType>, Vec<ValType>) {
        let (p, r): (&[ValType], &[ValType]) = match self {
            Rt::Alloc => (&[I64], &[I32]),
            Rt::Arith(_) => (&[I64, I64, I32, I32], &[I64]),
            Rt::Variant1 => (&[I32, I64], &[I32]),
            Rt::OptFromPtr | Rt::StrNew | Rt::StrTrim | Rt::StrToInt | Rt::StrChars | Rt::ListNew | Rt::ListReverse | Rt::ListSortInt => (&[I32], &[I32]),
            Rt::ResultFrom | Rt::ResultUnitFrom | Rt::StrConcat | Rt::StrEq | Rt::StrContains | Rt::StrStartsWith | Rt::StrJoin | Rt::ListConcat => (&[I32, I32], &[I32]),
            Rt::StrLen => (&[I32], &[I64]),
            Rt::BytesEq | Rt::StrSub | Rt::StrFind => (&[I32, I32, I32], &[I32]),
            Rt::CharToByte => (&[I32, I64], &[I32]),
            Rt::StrSlice => (&[I32, I64, I64, I32, I32], &[I32]),
            Rt::StrSplit => (&[I32, I32, I32, I32], &[I32]),
            Rt::IntToStr => (&[I64], &[I32]),
            Rt::ListGet | Rt::ListPush => (&[I32, I64], &[I32]),
            Rt::ListRange => (&[I64, I64], &[I32]),
            Rt::Sift => (&[I32, I32, I32], &[]),
            Rt::FloatToInt => (&[F64], &[I32]),
            Rt::RandChecked => (&[I64, I64, I32, I32], &[I64]),
            Rt::Map(..) | Rt::Filter(_) => (&[I32, I32], &[I32]),
            Rt::Fold(acc, _) => return (vec![I32, acc.val(), I32], vec![acc.val()]),
        };
        (p.to_vec(), r.to_vec())
    }

    pub fn body(self, m: &mut Mb, n_params: usize) -> Function {
        let mut a = Asm::new(n_params);
        match self {
            Rt::Alloc => alloc(m, &mut a),
            Rt::Arith(op) => arith(m, &mut a, op),
            Rt::Variant1 => {
                let p = a.local(I32);
                a.all([I::I64Const(16), I::Call(m.rt(Rt::Alloc)), I::LocalTee(p), I::LocalGet(0), I::I32Store(m32(0))]);
                a.all([I::LocalGet(p), I::LocalGet(1), I::I64Store(m64(8)), I::LocalGet(p)]);
            }
            Rt::OptFromPtr => {
                a.all([I::LocalGet(0), I::I32Const(0), I::I32LtS, I::If(BlockType::Result(I32)), I::I32Const(m.tag_addr(TAG_NONE)), I::Else]);
                a.all([I::I32Const(TAG_SOME as i32), I::LocalGet(0), I::I64ExtendI32U, I::Call(m.rt(Rt::Variant1)), I::End]);
            }
            Rt::ResultFrom | Rt::ResultUnitFrom => {
                // ok ? Ok(x) : Err(str); for Result[Unit, Str] the Ok payload is Unit (0).
                a.all([I::I32Const(TAG_OK as i32), I::I32Const(TAG_ERR as i32), I::LocalGet(0), I::Select]);
                if self == Rt::ResultFrom {
                    a.i(I::LocalGet(1));
                } else {
                    a.all([I::I32Const(0), I::LocalGet(1), I::LocalGet(0), I::Select]);
                }
                a.all([I::I64ExtendI32U, I::Call(m.rt(Rt::Variant1))]);
            }
            Rt::StrNew => {
                let p = a.local(I32);
                a.all([I::LocalGet(0), I::I64ExtendI32U, I::I64Const(4), I::I64Add, I::Call(m.rt(Rt::Alloc)), I::LocalTee(p)]);
                a.all([I::LocalGet(0), I::I32Store(m32(0)), I::LocalGet(p)]);
            }
            Rt::StrLen => {
                let (n, i, c) = (a.local(I32), a.local(I32), a.local(I64));
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalSet(n)]);
                loop_until(&mut a, i, n);
                // count bytes that are not UTF-8 continuation bytes (10xxxxxx)
                a.all([I::LocalGet(0), I::LocalGet(i), I::I32Add, I::I32Load8U(m8(4)), I::I32Const(0xC0), I::I32And, I::I32Const(0x80), I::I32Ne]);
                a.all([I::I64ExtendI32U, I::LocalGet(c), I::I64Add, I::LocalSet(c)]);
                loop_next(&mut a, i);
                a.i(I::LocalGet(c));
            }
            Rt::StrConcat => {
                let (la, lb, p) = (a.local(I32), a.local(I32), a.local(I32));
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalSet(la), I::LocalGet(1), I::I32Load(m32(0)), I::LocalSet(lb)]);
                a.all([I::LocalGet(la), I::LocalGet(lb), I::I32Add, I::Call(m.rt(Rt::StrNew)), I::LocalSet(p)]);
                a.all([I::LocalGet(p), I::I32Const(4), I::I32Add, I::LocalGet(0), I::I32Const(4), I::I32Add, I::LocalGet(la), copy()]);
                a.all([I::LocalGet(p), I::I32Const(4), I::I32Add, I::LocalGet(la), I::I32Add, I::LocalGet(1), I::I32Const(4), I::I32Add, I::LocalGet(lb), copy()]);
                a.i(I::LocalGet(p));
            }
            Rt::BytesEq => {
                let i = a.local(I32);
                loop_until(&mut a, i, 2);
                a.all([I::LocalGet(0), I::LocalGet(i), I::I32Add, I::I32Load8U(m8(0)), I::LocalGet(1), I::LocalGet(i), I::I32Add, I::I32Load8U(m8(0)), I::I32Ne]);
                a.all([I::If(BlockType::Empty), I::I32Const(0), I::Return, I::End]);
                loop_next(&mut a, i);
                a.i(I::I32Const(1));
            }
            Rt::StrEq => {
                a.all([I::LocalGet(0), I::LocalGet(1), I::I32Eq, I::If(BlockType::Empty), I::I32Const(1), I::Return, I::End]);
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalGet(1), I::I32Load(m32(0)), I::I32Ne, I::If(BlockType::Empty), I::I32Const(0), I::Return, I::End]);
                a.all([I::LocalGet(0), I::I32Const(4), I::I32Add, I::LocalGet(1), I::I32Const(4), I::I32Add, I::LocalGet(0), I::I32Load(m32(0)), I::Call(m.rt(Rt::BytesEq))]);
            }
            Rt::StrSub => {
                let p = a.local(I32);
                a.all([I::LocalGet(2), I::LocalGet(1), I::I32Sub, I::Call(m.rt(Rt::StrNew)), I::LocalSet(p)]);
                a.all([I::LocalGet(p), I::I32Const(4), I::I32Add, I::LocalGet(0), I::I32Const(4), I::I32Add, I::LocalGet(1), I::I32Add, I::LocalGet(2), I::LocalGet(1), I::I32Sub, copy()]);
                a.i(I::LocalGet(p));
            }
            Rt::CharToByte => {
                let (n, i, c) = (a.local(I32), a.local(I32), a.local(I64));
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalSet(n)]);
                loop_until(&mut a, i, n);
                a.all([I::LocalGet(0), I::LocalGet(i), I::I32Add, I::I32Load8U(m8(4)), I::I32Const(0xC0), I::I32And, I::I32Const(0x80), I::I32Ne, I::If(BlockType::Empty)]);
                a.all([I::LocalGet(c), I::LocalGet(1), I::I64Eq, I::If(BlockType::Empty), I::LocalGet(i), I::Return, I::End]);
                a.all([I::LocalGet(c), I::I64Const(1), I::I64Add, I::LocalSet(c), I::End]);
                loop_next(&mut a, i);
                a.i(I::LocalGet(n));
            }
            Rt::StrSlice => {
                let n = a.local(I64);
                a.all([I::LocalGet(0), I::Call(m.rt(Rt::StrLen)), I::LocalSet(n)]);
                a.all([I::LocalGet(1), I::I64Const(0), I::I64LtS, I::LocalGet(2), I::LocalGet(1), I::I64LtS, I::I32Or, I::LocalGet(2), I::LocalGet(n), I::I64GtS, I::I32Or]);
                a.i(I::If(BlockType::Empty));
                m.emit_fail(&mut a, FAIL_INDEX, Arg::Local(3), Arg::Local(4), &[(1, Repr::I64), (2, Repr::I64), (n, Repr::I64)]);
                a.i(I::End);
                let c2b = m.rt(Rt::CharToByte);
                a.all([I::LocalGet(0), I::LocalGet(0), I::LocalGet(1), I::Call(c2b), I::LocalGet(0), I::LocalGet(2), I::Call(c2b), I::Call(m.rt(Rt::StrSub))]);
            }
            Rt::StrFind => {
                let (ls, lsub, i) = (a.local(I32), a.local(I32), a.local(I32));
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalSet(ls), I::LocalGet(1), I::I32Load(m32(0)), I::LocalSet(lsub), I::LocalGet(2), I::LocalSet(i)]);
                a.all([I::Block(BlockType::Empty), I::Loop(BlockType::Empty)]);
                a.all([I::LocalGet(i), I::LocalGet(lsub), I::I32Add, I::LocalGet(ls), I::I32GtU, I::BrIf(1)]);
                a.all([I::LocalGet(0), I::I32Const(4), I::I32Add, I::LocalGet(i), I::I32Add, I::LocalGet(1), I::I32Const(4), I::I32Add, I::LocalGet(lsub), I::Call(m.rt(Rt::BytesEq))]);
                a.all([I::If(BlockType::Empty), I::LocalGet(i), I::Return, I::End]);
                incr(&mut a, i);
                a.all([I::Br(0), I::End, I::End, I::I32Const(-1)]);
            }
            Rt::StrContains => {
                a.all([I::LocalGet(0), I::LocalGet(1), I::I32Const(0), I::Call(m.rt(Rt::StrFind)), I::I32Const(-1), I::I32Ne]);
            }
            Rt::StrStartsWith => {
                a.all([I::LocalGet(1), I::I32Load(m32(0)), I::LocalGet(0), I::I32Load(m32(0)), I::I32GtU, I::If(BlockType::Empty), I::I32Const(0), I::Return, I::End]);
                a.all([I::LocalGet(0), I::I32Const(4), I::I32Add, I::LocalGet(1), I::I32Const(4), I::I32Add, I::LocalGet(1), I::I32Load(m32(0)), I::Call(m.rt(Rt::BytesEq))]);
            }
            Rt::StrSplit => split(m, &mut a),
            Rt::StrJoin => join(m, &mut a),
            Rt::StrTrim => {
                let (s, e, c) = (a.local(I32), a.local(I32), a.local(I32));
                let is_ws = |a: &mut Asm| {
                    a.all([I::LocalTee(c), I::I32Const(32), I::I32Eq, I::LocalGet(c), I::I32Const(9), I::I32Eq, I::I32Or]);
                    a.all([I::LocalGet(c), I::I32Const(10), I::I32Eq, I::I32Or, I::LocalGet(c), I::I32Const(13), I::I32Eq, I::I32Or]);
                };
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalSet(e)]);
                a.all([I::Block(BlockType::Empty), I::Loop(BlockType::Empty), I::LocalGet(s), I::LocalGet(e), I::I32GeU, I::BrIf(1)]);
                a.all([I::LocalGet(0), I::LocalGet(s), I::I32Add, I::I32Load8U(m8(4))]);
                is_ws(&mut a);
                a.all([I::I32Eqz, I::BrIf(1)]);
                incr(&mut a, s);
                a.all([I::Br(0), I::End, I::End]);
                a.all([I::Block(BlockType::Empty), I::Loop(BlockType::Empty), I::LocalGet(e), I::LocalGet(s), I::I32LeU, I::BrIf(1)]);
                a.all([I::LocalGet(0), I::LocalGet(e), I::I32Add, I::I32Load8U(m8(3))]);
                is_ws(&mut a);
                a.all([I::I32Eqz, I::BrIf(1), I::LocalGet(e), I::I32Const(1), I::I32Sub, I::LocalSet(e), I::Br(0), I::End, I::End]);
                a.all([I::LocalGet(0), I::LocalGet(s), I::LocalGet(e), I::Call(m.rt(Rt::StrSub))]);
            }
            Rt::StrToInt => str_to_int(m, &mut a),
            Rt::IntToStr => int_to_str(m, &mut a),
            Rt::StrChars => {
                let (xs, nb, i, k, cl) = (a.local(I32), a.local(I32), a.local(I32), a.local(I32), a.local(I32));
                a.all([I::LocalGet(0), I::Call(m.rt(Rt::StrLen)), I::I32WrapI64, I::Call(m.rt(Rt::ListNew)), I::LocalSet(xs)]);
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalSet(nb)]);
                loop_until(&mut a, i, nb);
                // UTF-8 sequence length from the lead byte: 1 + (b >= 0xC0) + (b >= 0xE0) + (b >= 0xF0)
                let lead = [I::LocalGet(0), I::LocalGet(i), I::I32Add, I::I32Load8U(m8(4))];
                a.i(I::I32Const(1));
                for bound in [0xC0, 0xE0, 0xF0] {
                    a.all(lead.clone());
                    a.all([I::I32Const(bound), I::I32GeU, I::I32Add]);
                }
                a.i(I::LocalSet(cl));
                slot_addr(&mut a, xs, k);
                a.all([I::LocalGet(0), I::LocalGet(i), I::LocalGet(i), I::LocalGet(cl), I::I32Add, I::Call(m.rt(Rt::StrSub)), I::I64ExtendI32U, I::I64Store(m64(8))]);
                incr(&mut a, k);
                a.all([I::LocalGet(i), I::LocalGet(cl), I::I32Add, I::LocalSet(i), I::Br(0), I::End, I::End]);
                a.i(I::LocalGet(xs));
            }
            Rt::ListNew => {
                let p = a.local(I32);
                a.all([I::LocalGet(0), I::I64ExtendI32U, I::I64Const(3), I::I64Shl, I::I64Const(8), I::I64Add, I::Call(m.rt(Rt::Alloc)), I::LocalTee(p)]);
                a.all([I::LocalGet(0), I::I32Store(m32(0)), I::LocalGet(p)]);
            }
            Rt::ListGet => {
                a.all([I::LocalGet(1), I::I64Const(0), I::I64LtS, I::LocalGet(1), I::LocalGet(0), I::I32Load(m32(0)), I::I64ExtendI32U, I::I64GeS, I::I32Or]);
                a.all([I::If(BlockType::Result(I32)), I::I32Const(m.tag_addr(TAG_NONE)), I::Else]);
                a.all([I::I32Const(TAG_SOME as i32), I::LocalGet(0), I::LocalGet(1), I::I32WrapI64, I::I32Const(3), I::I32Shl, I::I32Add, I::I64Load(m64(8))]);
                a.all([I::Call(m.rt(Rt::Variant1)), I::End]);
            }
            Rt::ListPush => {
                let (n, p) = (a.local(I32), a.local(I32));
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalTee(n), I::I32Const(1), I::I32Add, I::Call(m.rt(Rt::ListNew)), I::LocalSet(p)]);
                a.all([I::LocalGet(p), I::I32Const(8), I::I32Add, I::LocalGet(0), I::I32Const(8), I::I32Add, I::LocalGet(n), I::I32Const(3), I::I32Shl, copy()]);
                slot_addr(&mut a, p, n);
                a.all([I::LocalGet(1), I::I64Store(m64(8)), I::LocalGet(p)]);
            }
            Rt::ListConcat => {
                let (na, nb, p) = (a.local(I32), a.local(I32), a.local(I32));
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalSet(na), I::LocalGet(1), I::I32Load(m32(0)), I::LocalSet(nb)]);
                a.all([I::LocalGet(na), I::LocalGet(nb), I::I32Add, I::Call(m.rt(Rt::ListNew)), I::LocalSet(p)]);
                a.all([I::LocalGet(p), I::I32Const(8), I::I32Add, I::LocalGet(0), I::I32Const(8), I::I32Add, I::LocalGet(na), I::I32Const(3), I::I32Shl, copy()]);
                slot_addr(&mut a, p, na);
                a.all([I::I32Const(8), I::I32Add, I::LocalGet(1), I::I32Const(8), I::I32Add, I::LocalGet(nb), I::I32Const(3), I::I32Shl, copy(), I::LocalGet(p)]);
            }
            Rt::ListReverse => {
                let (n, p, k) = (a.local(I32), a.local(I32), a.local(I32));
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalTee(n), I::Call(m.rt(Rt::ListNew)), I::LocalSet(p)]);
                loop_until(&mut a, k, n);
                slot_addr(&mut a, p, k);
                a.all([I::LocalGet(0), I::LocalGet(n), I::LocalGet(k), I::I32Sub, I::I32Const(3), I::I32Shl, I::I32Add, I::I64Load(m64(0)), I::I64Store(m64(8))]);
                loop_next(&mut a, k);
                a.i(I::LocalGet(p));
            }
            Rt::ListRange => {
                let (count, p, k) = (a.local(I64), a.local(I32), a.local(I32));
                a.all([I::LocalGet(1), I::LocalGet(0), I::I64LeS, I::If(BlockType::Empty), I::I32Const(0), I::Call(m.rt(Rt::ListNew)), I::Return, I::End]);
                // b - a may exceed i64::MAX; as unsigned it is exact. Absurd sizes are clamped so alloc reports out_of_memory.
                a.all([I::LocalGet(1), I::LocalGet(0), I::I64Sub, I::LocalSet(count)]);
                a.all([I::I64Const(1 << 40), I::LocalGet(count), I::I64Const(3), I::I64Shl, I::I64Const(8), I::I64Add, I::LocalGet(count), I::I64Const(1 << 32), I::I64GtU, I::Select]);
                a.all([I::Call(m.rt(Rt::Alloc)), I::LocalTee(p), I::LocalGet(count), I::I32WrapI64, I::I32Store(m32(0))]);
                let n = a.local(I32);
                a.all([I::LocalGet(count), I::I32WrapI64, I::LocalSet(n)]);
                loop_until(&mut a, k, n);
                slot_addr(&mut a, p, k);
                a.all([I::LocalGet(0), I::LocalGet(k), I::I64ExtendI32U, I::I64Add, I::I64Store(m64(8))]);
                loop_next(&mut a, k);
                a.i(I::LocalGet(p));
            }
            Rt::ListSortInt => sort_int(m, &mut a),
            Rt::Sift => sift(&mut a),
            Rt::FloatToInt => {
                a.all([I::LocalGet(0), I::LocalGet(0), I::F64Ne, I::LocalGet(0), I::F64Const(9223372036854775808.0.into()), I::F64Ge, I::I32Or]);
                a.all([I::LocalGet(0), I::F64Const((-9223372036854775808.0).into()), I::F64Lt, I::I32Or]);
                a.all([I::If(BlockType::Result(I32)), I::I32Const(m.tag_addr(TAG_NONE)), I::Else]);
                a.all([I::I32Const(TAG_SOME as i32), I::LocalGet(0), I::I64TruncF64S, I::Call(m.rt(Rt::Variant1)), I::End]);
            }
            Rt::RandChecked => {
                a.all([I::LocalGet(0), I::LocalGet(1), I::I64GtS, I::If(BlockType::Empty)]);
                m.emit_fail(&mut a, FAIL_BUILTIN, Arg::Local(2), Arg::Local(3), &[(0, Repr::I64), (1, Repr::I64)]);
                a.all([I::End, I::LocalGet(0), I::LocalGet(1), I::Call(m.import(host::RAND_INT))]);
            }
            Rt::Map(from, to) => {
                let (n, p, k) = (a.local(I32), a.local(I32), a.local(I32));
                let ty = m.ty(&[from.val()], &[to.val()]);
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalTee(n), I::Call(m.rt(Rt::ListNew)), I::LocalSet(p)]);
                loop_until(&mut a, k, n);
                slot_addr(&mut a, p, k);
                slot_addr(&mut a, 0, k);
                a.i(I::I64Load(m64(8)));
                a.opt(from_slot(from));
                a.all([I::LocalGet(1), I::CallIndirect { type_index: ty, table_index: 0 }]);
                a.opt(to_slot(to));
                a.i(I::I64Store(m64(8)));
                loop_next(&mut a, k);
                a.i(I::LocalGet(p));
            }
            Rt::Filter(elem) => {
                let (n, p, k, kept, v) = (a.local(I32), a.local(I32), a.local(I32), a.local(I32), a.local(I64));
                let ty = m.ty(&[elem.val()], &[I32]);
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalTee(n), I::Call(m.rt(Rt::ListNew)), I::LocalSet(p)]);
                loop_until(&mut a, k, n);
                slot_addr(&mut a, 0, k);
                a.all([I::I64Load(m64(8)), I::LocalTee(v)]);
                a.opt(from_slot(elem));
                a.all([I::LocalGet(1), I::CallIndirect { type_index: ty, table_index: 0 }, I::If(BlockType::Empty)]);
                slot_addr(&mut a, p, kept);
                a.all([I::LocalGet(v), I::I64Store(m64(8))]);
                incr(&mut a, kept);
                a.i(I::End);
                loop_next(&mut a, k);
                // The fresh list was sized for all elements; shrink its length to the kept ones.
                a.all([I::LocalGet(p), I::LocalGet(kept), I::I32Store(m32(0)), I::LocalGet(p)]);
            }
            Rt::Fold(acc, elem) => {
                let (n, k) = (a.local(I32), a.local(I32));
                let ty = m.ty(&[acc.val(), elem.val()], &[acc.val()]);
                a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalSet(n)]);
                loop_until(&mut a, k, n);
                a.i(I::LocalGet(1));
                slot_addr(&mut a, 0, k);
                a.i(I::I64Load(m64(8)));
                a.opt(from_slot(elem));
                a.all([I::LocalGet(2), I::CallIndirect { type_index: ty, table_index: 0 }, I::LocalSet(1)]);
                loop_next(&mut a, k);
                a.i(I::LocalGet(1));
            }
        }
        a.finish()
    }
}

fn copy() -> I<'static> {
    I::MemoryCopy { src_mem: 0, dst_mem: 0 }
}

/// Bump allocator over the exported memory; grows memory page-wise up to its maximum.
fn alloc(m: &mut Mb, a: &mut Asm) {
    let (p, end, have) = (a.local(I32), a.local(I64), a.local(I64));
    a.all([I::GlobalGet(HEAP_GLOBAL), I::LocalTee(p), I::I64ExtendI32U, I::LocalGet(0), I::I64Add, I::I64Const(7), I::I64Add, I::I64Const(-8), I::I64And, I::LocalSet(end)]);
    a.all([I::MemorySize(0), I::I64ExtendI32U, I::I64Const(16), I::I64Shl, I::LocalSet(have)]);
    a.all([I::LocalGet(end), I::LocalGet(have), I::I64GtU, I::If(BlockType::Empty)]);
    // pages needed = ceil((end - have) / 64 KiB); more than 4 GiB can never be granted
    a.all([I::LocalGet(end), I::I64Const(0xFFFF_FFFF), I::I64GtU, I::If(BlockType::Result(I32)), I::I32Const(-1), I::Else]);
    a.all([I::LocalGet(end), I::LocalGet(have), I::I64Sub, I::I64Const(65535), I::I64Add, I::I64Const(16), I::I64ShrU, I::I32WrapI64, I::MemoryGrow(0), I::End]);
    a.all([I::I32Const(-1), I::I32Eq, I::If(BlockType::Empty)]);
    a.all([I::I32Const(OOM_SCRATCH as i32), I::LocalGet(0), I::I64Store(m64(0))]);
    a.all([I::I32Const(FAIL_OOM), I::I32Const(-1), I::I32Const(-1), I::I32Const(OOM_SCRATCH as i32), I::Call(m.import(host::FAIL)), I::Unreachable]);
    a.all([I::End, I::End]);
    a.all([I::LocalGet(end), I::I32WrapI64, I::GlobalSet(HEAP_GLOBAL), I::LocalGet(p)]);
}

/// Checked Int arithmetic. Params: a = 0, b = 1, func = 2, site = 3; local r = 4.
fn arith(m: &mut Mb, a: &mut Asm, op: BinOp) {
    let r = a.local(I64);
    let vals = [(0, Repr::I64), (1, Repr::I64)];
    let fail_if = |m: &mut Mb, a: &mut Asm, kind: i32| {
        a.i(I::If(BlockType::Empty));
        m.emit_fail(a, kind, Arg::Local(2), Arg::Local(3), &vals);
        a.i(I::End);
    };
    match op {
        BinOp::Add | BinOp::Sub => {
            a.all([I::LocalGet(0), I::LocalGet(1), if op == BinOp::Add { I::I64Add } else { I::I64Sub }, I::LocalSet(r)]);
            // add overflows iff ((a ^ r) & (b ^ r)) < 0; sub iff ((a ^ b) & (a ^ r)) < 0
            if op == BinOp::Add {
                a.all([I::LocalGet(0), I::LocalGet(r), I::I64Xor, I::LocalGet(1), I::LocalGet(r), I::I64Xor]);
            } else {
                a.all([I::LocalGet(0), I::LocalGet(1), I::I64Xor, I::LocalGet(0), I::LocalGet(r), I::I64Xor]);
            }
            a.all([I::I64And, I::I64Const(0), I::I64LtS]);
            fail_if(m, a, FAIL_OVERFLOW);
            a.i(I::LocalGet(r));
        }
        BinOp::Mul => {
            // a == -1 && b == MIN overflows (and would trap in the division check below)
            a.all([I::LocalGet(0), I::I64Const(-1), I::I64Eq, I::LocalGet(1), I::I64Const(i64::MIN), I::I64Eq, I::I32And]);
            fail_if(m, a, FAIL_OVERFLOW);
            a.all([I::LocalGet(0), I::LocalGet(1), I::I64Mul, I::LocalSet(r)]);
            // a != 0 && r / a != b
            a.all([I::LocalGet(0), I::I64Const(0), I::I64Ne, I::If(BlockType::Empty)]);
            a.all([I::LocalGet(r), I::LocalGet(0), I::I64DivS, I::LocalGet(1), I::I64Ne]);
            fail_if(m, a, FAIL_OVERFLOW);
            a.all([I::End, I::LocalGet(r)]);
        }
        BinOp::Div | BinOp::Rem => {
            a.all([I::LocalGet(1), I::I64Eqz]);
            fail_if(m, a, FAIL_DIV_ZERO);
            if op == BinOp::Div {
                a.all([I::LocalGet(0), I::I64Const(i64::MIN), I::I64Eq, I::LocalGet(1), I::I64Const(-1), I::I64Eq, I::I32And]);
                fail_if(m, a, FAIL_OVERFLOW);
                a.all([I::LocalGet(0), I::LocalGet(1), I::I64DivS]);
            } else {
                // wasm i64.rem_s(MIN, -1) is defined as 0, no trap
                a.all([I::LocalGet(0), I::LocalGet(1), I::I64RemS]);
            }
        }
        _ => unreachable!("not an arithmetic operator: {op:?}"),
    }
}

/// split(s, sep). Params: s = 0, sep = 1, func = 2, site = 3.
fn split(m: &mut Mb, a: &mut Asm) {
    let (lsep, count, pos, f, xs, k) = (a.local(I32), a.local(I32), a.local(I32), a.local(I32), a.local(I32), a.local(I32));
    let find = m.rt(Rt::StrFind);
    let sub = m.rt(Rt::StrSub);
    a.all([I::LocalGet(1), I::I32Load(m32(0)), I::LocalTee(lsep), I::I32Eqz, I::If(BlockType::Empty)]);
    m.emit_fail(a, FAIL_BUILTIN, Arg::Local(2), Arg::Local(3), &[(1, Repr::I32)]);
    a.i(I::End);
    // pass 1: count parts
    a.all([I::I32Const(1), I::LocalSet(count)]);
    a.all([I::Block(BlockType::Empty), I::Loop(BlockType::Empty)]);
    a.all([I::LocalGet(0), I::LocalGet(1), I::LocalGet(pos), I::Call(find), I::LocalTee(f), I::I32Const(0), I::I32LtS, I::BrIf(1)]);
    incr(a, count);
    a.all([I::LocalGet(f), I::LocalGet(lsep), I::I32Add, I::LocalSet(pos), I::Br(0), I::End, I::End]);
    // pass 2: cut
    a.all([I::LocalGet(count), I::Call(m.rt(Rt::ListNew)), I::LocalSet(xs), I::I32Const(0), I::LocalSet(pos)]);
    a.all([I::Block(BlockType::Empty), I::Loop(BlockType::Empty)]);
    a.all([I::LocalGet(0), I::LocalGet(1), I::LocalGet(pos), I::Call(find), I::LocalTee(f), I::I32Const(0), I::I32LtS, I::BrIf(1)]);
    slot_addr(a, xs, k);
    a.all([I::LocalGet(0), I::LocalGet(pos), I::LocalGet(f), I::Call(sub), I::I64ExtendI32U, I::I64Store(m64(8))]);
    incr(a, k);
    a.all([I::LocalGet(f), I::LocalGet(lsep), I::I32Add, I::LocalSet(pos), I::Br(0), I::End, I::End]);
    slot_addr(a, xs, k);
    a.all([I::LocalGet(0), I::LocalGet(pos), I::LocalGet(0), I::I32Load(m32(0)), I::Call(sub), I::I64ExtendI32U, I::I64Store(m64(8)), I::LocalGet(xs)]);
}

/// join(xs, sep). Params: xs = 0, sep = 1.
fn join(m: &mut Mb, a: &mut Asm) {
    let (n, total, k, p, at, e, lsep) = (a.local(I32), a.local(I64), a.local(I32), a.local(I32), a.local(I32), a.local(I32), a.local(I32));
    a.all([I::LocalGet(1), I::I32Load(m32(0)), I::LocalSet(lsep)]);
    a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalTee(n), I::I32Eqz, I::If(BlockType::Empty), I::I32Const(0), I::Call(m.rt(Rt::StrNew)), I::Return, I::End]);
    // total = lsep * (n - 1) + sum of element lengths, in i64 so that huge results cannot wrap
    a.all([I::LocalGet(lsep), I::I64ExtendI32U, I::LocalGet(n), I::I32Const(1), I::I32Sub, I::I64ExtendI32U, I::I64Mul, I::LocalSet(total)]);
    loop_until(a, k, n);
    slot_addr(a, 0, k);
    a.all([I::I64Load(m64(8)), I::I32WrapI64, I::I32Load(m32(0)), I::I64ExtendI32U, I::LocalGet(total), I::I64Add, I::LocalSet(total)]);
    loop_next(a, k);
    // Beyond 2 GiB the i32 length cannot be stored; such a request always exceeds the memory limit.
    a.all([I::LocalGet(total), I::I64Const(0x7FFF_0000), I::I64GtU, I::If(BlockType::Empty), I::LocalGet(total), I::Call(m.rt(Rt::Alloc)), I::Drop, I::End]);
    a.all([I::LocalGet(total), I::I32WrapI64, I::Call(m.rt(Rt::StrNew)), I::LocalTee(p), I::I32Const(4), I::I32Add, I::LocalSet(at), I::I32Const(0), I::LocalSet(k)]);
    loop_until(a, k, n);
    a.all([I::LocalGet(k), I::If(BlockType::Empty)]);
    a.all([I::LocalGet(at), I::LocalGet(1), I::I32Const(4), I::I32Add, I::LocalGet(lsep), copy(), I::LocalGet(at), I::LocalGet(lsep), I::I32Add, I::LocalSet(at), I::End]);
    slot_addr(a, 0, k);
    a.all([I::I64Load(m64(8)), I::I32WrapI64, I::LocalSet(e)]);
    a.all([I::LocalGet(at), I::LocalGet(e), I::I32Const(4), I::I32Add, I::LocalGet(e), I::I32Load(m32(0)), copy()]);
    a.all([I::LocalGet(at), I::LocalGet(e), I::I32Load(m32(0)), I::I32Add, I::LocalSet(at)]);
    loop_next(a, k);
    a.i(I::LocalGet(p));
}

/// to_int(s): optional sign, then decimal digits only; None on anything else or on overflow.
/// Accumulates negatively so that i64::MIN parses.
fn str_to_int(m: &mut Mb, a: &mut Asm) {
    let (n, i, neg, acc, d) = (a.local(I32), a.local(I32), a.local(I32), a.local(I64), a.local(I64));
    let none = m.tag_addr(TAG_NONE);
    let ret_none = [I::I32Const(none), I::Return];
    a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalTee(n), I::I32Eqz, I::If(BlockType::Empty)]);
    a.all(ret_none.clone());
    a.i(I::End);
    let first = [I::LocalGet(0), I::I32Load8U(m8(4))];
    a.all(first.clone());
    a.all([I::I32Const(b'-' as i32), I::I32Eq, I::If(BlockType::Empty), I::I32Const(1), I::LocalSet(neg), I::I32Const(1), I::LocalSet(i), I::End]);
    a.all(first);
    a.all([I::I32Const(b'+' as i32), I::I32Eq, I::If(BlockType::Empty), I::I32Const(1), I::LocalSet(i), I::End]);
    a.all([I::LocalGet(i), I::LocalGet(n), I::I32Eq, I::If(BlockType::Empty)]);
    a.all(ret_none.clone());
    a.i(I::End);
    loop_until(a, i, n);
    a.all([I::LocalGet(0), I::LocalGet(i), I::I32Add, I::I32Load8U(m8(4)), I::I32Const(48), I::I32Sub, I::I64ExtendI32U, I::LocalTee(d), I::I64Const(9), I::I64GtU]);
    // acc < MIN / 10 means acc * 10 overflows; acc * 10 < MIN + d means acc * 10 - d overflows
    a.all([I::LocalGet(acc), I::I64Const(i64::MIN / 10), I::I64LtS, I::I32Or, I::If(BlockType::Empty)]);
    a.all(ret_none.clone());
    a.i(I::End);
    a.all([I::LocalGet(acc), I::I64Const(10), I::I64Mul, I::LocalTee(acc), I::I64Const(i64::MIN), I::LocalGet(d), I::I64Add, I::I64LtS, I::If(BlockType::Empty)]);
    a.all(ret_none.clone());
    a.i(I::End);
    a.all([I::LocalGet(acc), I::LocalGet(d), I::I64Sub, I::LocalSet(acc)]);
    loop_next(a, i);
    a.all([I::LocalGet(neg), I::I32Eqz, I::If(BlockType::Empty)]);
    a.all([I::LocalGet(acc), I::I64Const(i64::MIN), I::I64Eq, I::If(BlockType::Empty)]);
    a.all(ret_none);
    a.all([I::End, I::I64Const(0), I::LocalGet(acc), I::I64Sub, I::LocalSet(acc), I::End]);
    a.all([I::I32Const(TAG_SOME as i32), I::LocalGet(acc), I::Call(m.rt(Rt::Variant1))]);
}

/// from_int(n): decimal text. Works on the non-positive value so that i64::MIN needs no special case.
fn int_to_str(m: &mut Mb, a: &mut Asm) {
    let (neg, v, t, digits, len, p, pos) = (a.local(I32), a.local(I64), a.local(I64), a.local(I32), a.local(I32), a.local(I32), a.local(I32));
    a.all([I::LocalGet(0), I::I64Const(0), I::I64LtS, I::LocalTee(neg)]);
    a.all([I::If(BlockType::Result(I64)), I::LocalGet(0), I::Else, I::I64Const(0), I::LocalGet(0), I::I64Sub, I::End, I::LocalTee(v), I::LocalSet(t)]);
    a.all([I::I32Const(1), I::LocalSet(digits)]);
    a.all([I::Block(BlockType::Empty), I::Loop(BlockType::Empty)]);
    a.all([I::LocalGet(t), I::I64Const(10), I::I64DivS, I::LocalTee(t), I::I64Eqz, I::BrIf(1)]);
    incr(a, digits);
    a.all([I::Br(0), I::End, I::End]);
    a.all([I::LocalGet(digits), I::LocalGet(neg), I::I32Add, I::LocalTee(len), I::Call(m.rt(Rt::StrNew)), I::LocalSet(p)]);
    a.all([I::LocalGet(len), I::I32Const(1), I::I32Sub, I::LocalSet(pos), I::LocalGet(v), I::LocalSet(t)]);
    a.all([I::Loop(BlockType::Empty)]);
    a.all([I::LocalGet(p), I::LocalGet(pos), I::I32Add, I::I32Const(48), I::I64Const(0), I::LocalGet(t), I::I64Const(10), I::I64RemS, I::I64Sub, I::I32WrapI64, I::I32Add, I::I32Store8(m8(4))]);
    a.all([I::LocalGet(pos), I::I32Const(1), I::I32Sub, I::LocalSet(pos)]);
    a.all([I::LocalGet(t), I::I64Const(10), I::I64DivS, I::LocalTee(t), I::I64Const(0), I::I64Ne, I::BrIf(0), I::End]);
    a.all([I::LocalGet(neg), I::If(BlockType::Empty), I::LocalGet(p), I::I32Const(b'-' as i32), I::I32Store8(m8(4)), I::End, I::LocalGet(p)]);
}

/// sort_int(xs): heapsort on a copy (O(n log n), no extra memory).
fn sort_int(m: &mut Mb, a: &mut Asm) {
    let (n, p, base, i) = (a.local(I32), a.local(I32), a.local(I32), a.local(I32));
    let sift = m.rt(Rt::Sift);
    a.all([I::LocalGet(0), I::I32Load(m32(0)), I::LocalTee(n), I::Call(m.rt(Rt::ListNew)), I::LocalTee(p), I::I32Const(8), I::I32Add, I::LocalTee(base)]);
    a.all([I::LocalGet(0), I::I32Const(8), I::I32Add, I::LocalGet(n), I::I32Const(3), I::I32Shl, copy()]);
    // heapify: i = n/2 - 1 down to 0
    a.all([I::LocalGet(n), I::I32Const(1), I::I32ShrU, I::I32Const(1), I::I32Sub, I::LocalSet(i)]);
    a.all([I::Block(BlockType::Empty), I::Loop(BlockType::Empty), I::LocalGet(i), I::I32Const(0), I::I32LtS, I::BrIf(1)]);
    a.all([I::LocalGet(base), I::LocalGet(i), I::LocalGet(n), I::Call(sift), I::LocalGet(i), I::I32Const(1), I::I32Sub, I::LocalSet(i), I::Br(0), I::End, I::End]);
    // pop: i = n - 1 down to 1: swap(0, i); sift(0, i)
    let t = a.local(I64);
    a.all([I::LocalGet(n), I::I32Const(1), I::I32Sub, I::LocalSet(i)]);
    a.all([I::Block(BlockType::Empty), I::Loop(BlockType::Empty), I::LocalGet(i), I::I32Const(0), I::I32LeS, I::BrIf(1)]);
    a.all([I::LocalGet(base), I::I64Load(m64(0)), I::LocalSet(t)]);
    a.all([I::LocalGet(base), I::LocalGet(base), I::LocalGet(i), I::I32Const(3), I::I32Shl, I::I32Add, I::I64Load(m64(0)), I::I64Store(m64(0))]);
    a.all([I::LocalGet(base), I::LocalGet(i), I::I32Const(3), I::I32Shl, I::I32Add, I::LocalGet(t), I::I64Store(m64(0))]);
    a.all([I::LocalGet(base), I::I32Const(0), I::LocalGet(i), I::Call(sift), I::LocalGet(i), I::I32Const(1), I::I32Sub, I::LocalSet(i), I::Br(0), I::End, I::End]);
    a.i(I::LocalGet(p));
}

/// Sift-down for a max-heap of i64 at `base` (params: base = 0, root = 1, end = 2).
fn sift(a: &mut Asm) {
    let (child, t) = (a.local(I32), a.local(I64));
    let at = |a: &mut Asm, idx: u32| a.all([I::LocalGet(0), I::LocalGet(idx), I::I32Const(3), I::I32Shl, I::I32Add]);
    a.i(I::Loop(BlockType::Empty));
    a.all([I::LocalGet(1), I::I32Const(1), I::I32Shl, I::I32Const(1), I::I32Add, I::LocalTee(child), I::LocalGet(2), I::I32GeS, I::If(BlockType::Empty), I::Return, I::End]);
    // pick the larger child
    a.all([I::LocalGet(child), I::I32Const(1), I::I32Add, I::LocalGet(2), I::I32LtS, I::If(BlockType::Empty)]);
    at(a, child);
    a.i(I::I64Load(m64(0)));
    at(a, child);
    a.all([I::I64Load(m64(8)), I::I64LtS, I::If(BlockType::Empty), I::LocalGet(child), I::I32Const(1), I::I32Add, I::LocalSet(child), I::End, I::End]);
    at(a, 1);
    a.i(I::I64Load(m64(0)));
    at(a, child);
    a.all([I::I64Load(m64(0)), I::I64GeS, I::If(BlockType::Empty), I::Return, I::End]);
    // swap root and child, continue at child
    at(a, 1);
    a.all([I::I64Load(m64(0)), I::LocalSet(t)]);
    at(a, 1);
    at(a, child);
    a.all([I::I64Load(m64(0)), I::I64Store(m64(0))]);
    at(a, child);
    a.all([I::LocalGet(t), I::I64Store(m64(0)), I::LocalGet(child), I::LocalSet(1), I::Br(0), I::End]);
}
```

- [ ] **Step 6: Write expression lowering**

Create `qal/crates/qal-wasm/src/gen.rs`:

```rust
//! Lowering of typed expressions to WebAssembly instructions.

use crate::asm::{from_slot, m32, m64, to_slot, val, Asm, Repr};
use crate::module::{Arg, Mb};
use crate::rt::Rt;
use crate::{host, Site, FAIL_ENS, FAIL_REQ};
use qal_core::ast::{BinOp, Type, UnOp, RESULT_VAR};
use qal_core::tast::*;
use wasm_encoder::{BlockType, Function, Instruction as I, ValType};

pub struct FnGen<'g, 'p> {
    m: &'g mut Mb<'p>,
    /// Index of this function in `CompiledModule::funcs` (the `func` argument of `fail`).
    meta: i32,
    /// Wasm function index of each `TProgram::fns` entry (None if not emitted).
    fn_index: &'g [Option<u32>],
    a: Asm,
    pub sites: Vec<Site>,
}

impl<'g, 'p> FnGen<'g, 'p> {
    /// `locals`: types of all locals by id; the first `n_params` are the parameters.
    pub fn new(m: &'g mut Mb<'p>, meta: u32, fn_index: &'g [Option<u32>], locals: &[Type], n_params: usize) -> Self {
        let mut a = Asm::new(n_params);
        for t in &locals[n_params..] {
            a.local(val(t));
        }
        FnGen { m, meta: meta as i32, fn_index, a, sites: Vec::new() }
    }

    fn site(&mut self, path: &str, label: &str, values: &[(&str, Type)], contract: Option<usize>) -> i32 {
        self.sites.push(Site {
            path: path.into(),
            label: label.into(),
            contract,
            values: values.iter().map(|(n, t)| (n.to_string(), t.clone())).collect(),
        });
        self.sites.len() as i32 - 1
    }

    pub fn fn_body(mut self, f: &TFn) -> (Function, Vec<Site>) {
        let mut vals: Vec<(String, Type, u32)> = f.def.params.iter().enumerate().map(|(i, p)| (p.name.clone(), p.ty.clone(), i as u32)).collect();
        for (i, c) in f.req.iter().enumerate() {
            self.contract(c, FAIL_REQ, &format!("req[{i}]"), i, &vals);
        }
        match f.result_local {
            None => {
                self.expr(&f.body, true);
            }
            Some(r) => {
                // No tail calls here: `ens` must be checked against this call's params after the body returns.
                self.expr(&f.body, false);
                self.a.i(I::LocalSet(r));
                vals.push((RESULT_VAR.into(), f.def.ret.clone(), r));
                for (i, c) in f.ens.iter().enumerate() {
                    self.contract(c, FAIL_ENS, &format!("ens[{i}]"), i, &vals);
                }
                self.a.i(I::LocalGet(r));
            }
        }
        (self.a.finish(), self.sites)
    }

    pub fn test_body(mut self, t: &TTest) -> (Function, Vec<Site>) {
        self.expr(&t.body, false);
        (self.a.finish(), self.sites)
    }

    fn contract(&mut self, c: &TExpr, kind: i32, path: &str, index: usize, vals: &[(String, Type, u32)]) {
        let named: Vec<(&str, Type)> = vals.iter().map(|(n, t, _)| (n.as_str(), t.clone())).collect();
        let site = self.site(path, path, &named, Some(index));
        self.expr(c, false);
        self.a.all([I::I32Eqz, I::If(BlockType::Empty)]);
        let locals: Vec<(u32, Repr)> = vals.iter().map(|(_, t, l)| (*l, Repr::of(t))).collect();
        self.m.emit_fail(&mut self.a, kind, Arg::Const(self.meta), Arg::Const(site), &locals);
        self.a.i(I::End);
    }

    fn call_rt(&mut self, rt: Rt) {
        let f = self.m.rt(rt);
        self.a.i(I::Call(f));
    }

    /// Pushes `func, site` for a runtime helper that can fail.
    fn site_args(&mut self, site: i32) {
        self.a.all([I::I32Const(self.meta), I::I32Const(site)]);
    }

    fn temp(&mut self, r: Repr) -> u32 {
        self.a.local(r.val())
    }

    /// Emits `e`, leaving its value on the stack. `tail`: `e` is the function's result.
    fn expr(&mut self, e: &TExpr, tail: bool) {
        match &e.kind {
            TKind::Int(n) => self.a.i(I::I64Const(*n)),
            TKind::Float(x) => self.a.i(I::F64Const((*x).into())),
            TKind::Bool(b) => self.a.i(I::I32Const(*b as i32)),
            TKind::Unit => self.a.i(I::I32Const(0)),
            TKind::Str(s) => {
                let addr = self.m.str_addr(s);
                self.a.i(I::I32Const(addr as i32));
            }
            TKind::Local(id) => self.a.i(I::LocalGet(*id)),
            TKind::FnRef(i) => {
                let table = self.table_index(*i);
                self.a.i(I::I32Const(table as i32));
            }
            TKind::Un(UnOp::Not, x, _) => {
                self.expr(x, false);
                self.a.i(I::I32Eqz);
            }
            TKind::Un(UnOp::Neg, x, path) => {
                if x.ty == Type::Float {
                    self.expr(x, false);
                    self.a.i(I::F64Neg);
                } else {
                    let site = self.site(path, "-", &[("zero", Type::Int), ("operand", Type::Int)], None);
                    self.a.i(I::I64Const(0));
                    self.expr(x, false);
                    self.site_args(site);
                    self.call_rt(Rt::Arith(BinOp::Sub));
                }
            }
            TKind::Bin(op, l, r, path) => self.binary(*op, l, r, path),
            TKind::If(c, t, f) => {
                self.expr(c, false);
                self.a.i(I::If(BlockType::Result(val(&e.ty))));
                self.expr(t, tail);
                self.a.i(I::Else);
                self.expr(f, tail);
                self.a.i(I::End);
            }
            TKind::Let(id, v, body) => {
                self.expr(v, false);
                self.a.i(I::LocalSet(*id));
                self.expr(body, tail);
            }
            TKind::Seq(items) => {
                let (last, init) = items.split_last().expect("non-empty sequence");
                for x in init {
                    self.expr(x, false);
                    self.a.i(I::Drop);
                }
                self.expr(last, tail);
            }
            TKind::Call(i, args) => {
                for x in args {
                    self.expr(x, false);
                }
                let f = self.fn_index[*i].expect("callee is reachable");
                self.a.i(if tail { I::ReturnCall(f) } else { I::Call(f) });
            }
            TKind::CallIndirect(f, args) => {
                for x in args {
                    self.expr(x, false);
                }
                self.expr(f, false);
                let Type::Fn(ps, r, _) = &f.ty else { unreachable!("checked: callee is a function") };
                let params: Vec<ValType> = ps.iter().map(val).collect();
                let type_index = self.m.ty(&params, &[val(r)]);
                self.a.i(if tail { I::ReturnCallIndirect { type_index, table_index: 0 } } else { I::CallIndirect { type_index, table_index: 0 } });
            }
            TKind::List(items) => {
                let p = self.temp(Repr::I32);
                self.a.i(I::I32Const(items.len() as i32));
                self.call_rt(Rt::ListNew);
                self.a.i(I::LocalSet(p));
                self.store_slots(p, 8, items);
                self.a.i(I::LocalGet(p));
            }
            TKind::Variant(tag, xs) if xs.is_empty() => {
                let addr = self.m.tag_addr(*tag);
                self.a.i(I::I32Const(addr));
            }
            TKind::Variant(tag, xs) => {
                let p = self.temp(Repr::I32);
                self.alloc(8 + 8 * xs.len() as i64, p);
                self.a.all([I::LocalGet(p), I::I32Const(*tag as i32), I::I32Store(m32(0))]);
                self.store_slots(p, 8, xs);
                self.a.i(I::LocalGet(p));
            }
            TKind::Record(xs) => {
                let p = self.temp(Repr::I32);
                self.alloc(8 * xs.len() as i64, p);
                self.store_slots(p, 0, xs);
                self.a.i(I::LocalGet(p));
            }
            TKind::Field(x, i) => {
                self.expr(x, false);
                self.a.i(I::I64Load(m64(8 * *i as u64)));
                self.a.opt(from_slot(Repr::of(&e.ty)));
            }
            TKind::Match(s, arms) => self.match_expr(s, arms, &e.ty, tail),
            TKind::Builtin(b, args, path) => self.builtin(*b, args, path, &e.ty),
        }
    }

    fn table_index(&self, fn_idx: usize) -> u32 {
        // Emitted fns occupy table slots in the same order as their wasm indices.
        let wasm = self.fn_index[fn_idx].expect("referenced function is reachable");
        self.fn_index.iter().flatten().filter(|w| **w < wasm).count() as u32
    }

    fn alloc(&mut self, size: i64, into: u32) {
        self.a.i(I::I64Const(size));
        self.call_rt(Rt::Alloc);
        self.a.i(I::LocalSet(into));
    }

    /// Stores each value as a slot at `p + offset + 8k`.
    fn store_slots(&mut self, p: u32, offset: u64, xs: &[TExpr]) {
        for (k, x) in xs.iter().enumerate() {
            self.a.i(I::LocalGet(p));
            self.expr(x, false);
            self.a.opt(to_slot(Repr::of(&x.ty)));
            self.a.i(I::I64Store(m64(offset + 8 * k as u64)));
        }
    }

    fn binary(&mut self, op: BinOp, l: &TExpr, r: &TExpr, path: &str) {
        match op {
            BinOp::And => {
                self.expr(l, false);
                self.a.i(I::If(BlockType::Result(ValType::I32)));
                self.expr(r, false);
                self.a.all([I::Else, I::I32Const(0), I::End]);
                return;
            }
            BinOp::Or => {
                self.expr(l, false);
                self.a.all([I::If(BlockType::Result(ValType::I32)), I::I32Const(1), I::Else]);
                self.expr(r, false);
                self.a.i(I::End);
                return;
            }
            _ => {}
        }
        let float = l.ty == Type::Float;
        let int_site = |g: &mut Self| g.site(path, op.symbol(), &[("left", Type::Int), ("right", Type::Int)], None);
        let site = (!float && matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem)).then(|| int_site(self));
        self.expr(l, false);
        self.expr(r, false);
        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => match site {
                Some(site) => {
                    self.site_args(site);
                    self.call_rt(Rt::Arith(op));
                }
                None => self.a.i(match op {
                    BinOp::Add => I::F64Add,
                    BinOp::Sub => I::F64Sub,
                    BinOp::Mul => I::F64Mul,
                    _ => I::F64Div,
                }),
            },
            BinOp::Lt => self.a.i(if float { I::F64Lt } else { I::I64LtS }),
            BinOp::Le => self.a.i(if float { I::F64Le } else { I::I64LeS }),
            BinOp::Gt => self.a.i(if float { I::F64Gt } else { I::I64GtS }),
            BinOp::Ge => self.a.i(if float { I::F64Ge } else { I::I64GeS }),
            BinOp::Eq | BinOp::Ne => {
                self.m.emit_eq(&mut self.a, &l.ty);
                if op == BinOp::Ne {
                    self.a.i(I::I32Eqz);
                }
            }
            BinOp::Concat => self.call_rt(if l.ty == Type::Str { Rt::StrConcat } else { Rt::ListConcat }),
            BinOp::And | BinOp::Or => unreachable!("handled above"),
        }
    }

    fn builtin(&mut self, b: Builtin, args: &[TExpr], path: &str, ret: &Type) {
        use Builtin as B;
        for x in args {
            self.expr(x, false);
        }
        let elem = |t: &Type| match t {
            Type::List(x) => Repr::of(x),
            _ => unreachable!("checked: list argument"),
        };
        match b {
            B::StrLen => self.call_rt(Rt::StrLen),
            B::ListLen => self.a.all([I::I32Load(m32(0)), I::I64ExtendI32U]),
            B::Slice => {
                let site = self.site(path, "slice", &[("start", Type::Int), ("end", Type::Int), ("len", Type::Int)], None);
                self.site_args(site);
                self.call_rt(Rt::StrSlice);
            }
            B::Split => {
                let site = self.site(path, "split: req sep != \"\"", &[("sep", Type::Str)], None);
                self.site_args(site);
                self.call_rt(Rt::StrSplit);
            }
            B::Join => self.call_rt(Rt::StrJoin),
            B::Trim => self.call_rt(Rt::StrTrim),
            B::Contains => self.call_rt(Rt::StrContains),
            B::StartsWith => self.call_rt(Rt::StrStartsWith),
            B::ToInt => self.call_rt(Rt::StrToInt),
            B::FromInt => self.call_rt(Rt::IntToStr),
            B::Chars => self.call_rt(Rt::StrChars),
            B::ToFloat => {
                // host returns (ok, value)
                let (ok, v) = (self.temp(Repr::I32), self.temp(Repr::F64));
                self.a.all([I::Call(self.m.import(host::STR_TO_FLOAT)), I::LocalSet(v), I::LocalSet(ok), I::LocalGet(ok)]);
                self.a.all([I::If(BlockType::Result(ValType::I32)), I::I32Const(TAG_SOME as i32), I::LocalGet(v), I::I64ReinterpretF64]);
                self.call_rt(Rt::Variant1);
                self.a.all([I::Else, I::I32Const(self.m.tag_addr(TAG_NONE)), I::End]);
            }
            B::FromFloat => self.a.i(I::Call(self.m.import(host::FLOAT_TO_STR))),
            B::Get => self.call_rt(Rt::ListGet),
            B::Push => {
                self.a.opt(to_slot(Repr::of(&args[1].ty)));
                self.call_rt(Rt::ListPush);
            }
            B::Concat => self.call_rt(Rt::ListConcat),
            B::Map => self.call_rt(Rt::Map(elem(&args[0].ty), elem(ret))),
            B::Filter => self.call_rt(Rt::Filter(elem(&args[0].ty))),
            B::Fold => self.call_rt(Rt::Fold(Repr::of(ret), elem(&args[0].ty))),
            B::Range => self.call_rt(Rt::ListRange),
            B::Reverse => self.call_rt(Rt::ListReverse),
            B::SortInt => self.call_rt(Rt::ListSortInt),
            B::IntToFloat => self.a.i(I::F64ConvertI64S),
            B::FloatToInt => self.call_rt(Rt::FloatToInt),
            B::Print => self.a.all([I::Call(self.m.import(host::PRINT)), I::I32Const(0)]),
            B::ReadLine => {
                self.a.i(I::Call(self.m.import(host::READ_LINE)));
                self.call_rt(Rt::OptFromPtr);
            }
            B::ReadFile => {
                self.a.i(I::Call(self.m.import(host::READ_FILE)));
                self.call_rt(Rt::ResultFrom);
            }
            B::WriteFile => {
                self.a.i(I::Call(self.m.import(host::WRITE_FILE)));
                self.call_rt(Rt::ResultUnitFrom);
            }
            B::NowMs => self.a.i(I::Call(self.m.import(host::NOW_MS))),
            B::RandInt => {
                let site = self.site(path, "rand_int: req lo <= hi", &[("lo", Type::Int), ("hi", Type::Int)], None);
                self.site_args(site);
                self.call_rt(Rt::RandChecked);
            }
        }
    }

    /// `block $done { (block $next_i { test pattern i (br_if $next_i on mismatch); guard; body; br $done })* unreachable }`
    fn match_expr(&mut self, s: &TExpr, arms: &[TArm], ty: &Type, tail: bool) {
        let scrut = self.temp(Repr::of(&s.ty));
        self.expr(s, false);
        self.a.i(I::LocalSet(scrut));
        self.a.i(I::Block(BlockType::Result(val(ty))));
        for arm in arms {
            self.a.i(I::Block(BlockType::Empty));
            // depth 0 = $next (this arm's block), 1 = $done
            self.pattern(&arm.pat, scrut, &s.ty, 0);
            if let Some(g) = &arm.guard {
                self.expr(g, false);
                self.a.all([I::I32Eqz, I::BrIf(0)]);
            }
            self.expr(&arm.body, tail);
            self.a.all([I::Br(1), I::End]);
        }
        // Unreachable for checked programs: the last unguarded arms cover every value.
        self.a.all([I::Unreachable, I::End]);
    }

    /// Tests `value` (in local `v`, of type `t`) against `p`, binding variables;
    /// branches to the block at `depth` (relative to the current nesting) on mismatch.
    fn pattern(&mut self, p: &TPat, v: u32, t: &Type, depth: u32) {
        match p {
            TPat::Wild => {}
            TPat::Bind(id) => self.a.all([I::LocalGet(v), I::LocalSet(*id)]),
            TPat::Int(n) => self.a.all([I::LocalGet(v), I::I64Const(*n), I::I64Ne, I::BrIf(depth)]),
            TPat::Bool(b) => self.a.all([I::LocalGet(v), I::I32Const(*b as i32), I::I32Ne, I::BrIf(depth)]),
            TPat::Unit => {}
            TPat::Str(s) => {
                let addr = self.m.str_addr(s);
                self.a.all([I::LocalGet(v), I::I32Const(addr as i32)]);
                self.call_rt(Rt::StrEq);
                self.a.all([I::I32Eqz, I::BrIf(depth)]);
            }
            TPat::Variant(tag, ps) => {
                self.a.all([I::LocalGet(v), I::I32Load(m32(0)), I::I32Const(*tag as i32), I::I32Ne, I::BrIf(depth)]);
                let fields = self.m.p.variant_fields(t, *tag);
                for (k, (sub, ft)) in ps.iter().zip(&fields).enumerate() {
                    self.sub_pattern(sub, v, 8 + 8 * k as u64, ft, depth);
                }
            }
            TPat::Record(fs) => {
                let all = self.m.p.record_fields(t);
                for (i, sub) in fs {
                    self.sub_pattern(sub, v, 8 * *i as u64, &all[*i as usize].1, depth);
                }
            }
        }
    }

    fn sub_pattern(&mut self, p: &TPat, v: u32, offset: u64, t: &Type, depth: u32) {
        if matches!(p, TPat::Wild) {
            return;
        }
        let x = self.temp(Repr::of(t));
        self.a.all([I::LocalGet(v), I::I64Load(m64(offset))]);
        self.a.opt(from_slot(Repr::of(t)));
        self.a.i(I::LocalSet(x));
        self.pattern(p, x, t, depth);
    }
}
```

- [ ] **Step 7: Write the public API and entry-point compilation**

Replace `qal/crates/qal-wasm/src/lib.rs` with:

```rust
//! QAL -> WebAssembly code generation from the checked, typed program (`qal_check::check`).
//!
//! Module layout:
//!   imports   `qal_host.*` (see [`host`]): `fail` always; effect functions only if the compiled
//!             code calls them (so a pure program has no system access at all)
//!   functions user fns reachable from the entry, then tests, then runtime/equality helpers
//!             (all indices computed by the module builder)
//!   table     the emitted user fns, for `call_indirect` through fn values
//!   memory    exported as `memory`, at most 4096 pages (256 MiB); bump allocator exported as `qal_alloc(size: i64) -> i32`
//!   data      0..8 OOM scratch slot, 8.. one block per variant tag (shared payload-free variants), then string literals
//!
//! Values: Int = i64, Float = f64, Bool/Unit = i32, everything else = i32 pointer (fn values: table index).
//! Heap blocks: Str = [len: i32][bytes], List = [len: i32][pad][8-byte slots], variant = [tag: i32][pad][slots],
//! record = [slots]. Payload-free variants (`None`, `Empty`) point into the static tag table.

mod asm;
mod gen;
mod module;
mod rt;

use gen::FnGen;
use module::Mb;
use qal_core::ast::{Expr, Type, TypeDef};
use qal_core::tast::{Builtin, TExpr, TKind, TProgram};
use std::collections::BTreeSet;

pub use asm::Repr;

/// Failure kinds passed to `qal_host.fail`.
pub const FAIL_REQ: i32 = 1;
pub const FAIL_ENS: i32 = 2;
pub const FAIL_OVERFLOW: i32 = 3;
pub const FAIL_DIV_ZERO: i32 = 4;
pub const FAIL_INDEX: i32 = 5;
/// Precondition of a built-in function (e.g. `split` with an empty separator).
pub const FAIL_BUILTIN: i32 = 6;
/// Memory limit exceeded; `func` and `site` are -1, the requested size (i64) is at `OOM_SCRATCH`.
pub const FAIL_OOM: i32 = 7;

/// Byte address of the slot holding the size of a failed allocation.
pub const OOM_SCRATCH: u32 = 0;
/// Byte address of the static variant-tag table.
pub const TAG_TABLE: u32 = 8;
/// Global index of the heap pointer.
pub const HEAP_GLOBAL: u32 = 0;
/// Maximum memory: 4096 pages of 64 KiB = 256 MiB.
pub const MAX_PAGES: u64 = 4096;

pub const HOST_MODULE: &str = "qal_host";

/// Host functions in module `qal_host`. Effect functions are named `<effect>.<name>`.
pub mod host {
    use wasm_encoder::ValType::{self, F64, I32, I64};

    /// `fail(kind, func, site, vals_ptr)`: records a structured failure, then traps.
    pub const FAIL: &str = "fail";
    /// `(x: f64) -> Str` (pure; formatting floats is not worth hand-written wasm)
    pub const FLOAT_TO_STR: &str = "float_to_str";
    /// `(s: Str) -> (ok: i32, value: f64)` (pure)
    pub const STR_TO_FLOAT: &str = "str_to_float";
    pub const PRINT: &str = "io.print";
    /// `() -> Str ptr, or -1 at end of input`
    pub const READ_LINE: &str = "io.read_line";
    /// `(path) -> (ok, Str content or error message)`
    pub const READ_FILE: &str = "fs.read_file";
    /// `(path, content) -> (ok, error message or 0)`
    pub const WRITE_FILE: &str = "fs.write_file";
    pub const NOW_MS: &str = "time.now_ms";
    /// `(lo, hi) -> uniform Int in [lo, hi]` (the caller ensures lo <= hi)
    pub const RAND_INT: &str = "rand.rand_int";

    pub fn signature(name: &str) -> (&'static [ValType], &'static [ValType]) {
        match name {
            FAIL => (&[I32, I32, I32, I32], &[]),
            FLOAT_TO_STR => (&[F64], &[I32]),
            STR_TO_FLOAT => (&[I32], &[I32, F64]),
            PRINT => (&[I32], &[]),
            READ_LINE => (&[], &[I32]),
            READ_FILE => (&[I32], &[I32, I32]),
            WRITE_FILE => (&[I32, I32], &[I32, I32]),
            NOW_MS => (&[], &[I64]),
            RAND_INT => (&[I64, I64], &[I64]),
            _ => panic!("unknown host function `{name}`"),
        }
    }

    /// The effect a host function belongs to (None for `fail` and the pure helpers).
    pub fn effect(name: &str) -> Option<&str> {
        name.split_once('.').map(|(e, _)| e)
    }
}

/// A place in a function that can fail at runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct Site {
    /// AST path inside the definition, e.g. `req[0]` or `body.left`.
    pub path: String,
    /// What failed: the contract (`req[0]`), the operator (`+`) or the built-in precondition.
    pub label: String,
    /// Index into `req`/`ens` for contract failures.
    pub contract: Option<usize>,
    /// Names and types of the values passed to `fail` (in slot order).
    pub values: Vec<(String, Type)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncMeta {
    /// Definition name (without `test:` prefix).
    pub name: String,
    /// Export name: the def name, or `test:<name>` for tests.
    pub export: String,
    pub is_test: bool,
    pub params: Vec<(String, Type)>,
    pub ret: Type,
    pub effects: Vec<String>,
    pub req: Vec<Expr>,
    pub ens: Vec<Expr>,
    /// Indexed by the `site` argument of `qal_host.fail`.
    pub sites: Vec<Site>,
}

#[derive(Debug, Clone)]
pub struct CompiledModule {
    pub wasm: Vec<u8>,
    /// Indexed by the `func` argument of `qal_host.fail`.
    pub funcs: Vec<FuncMeta>,
    /// Declared types, for decoding values.
    pub types: Vec<TypeDef>,
    /// Host functions the module imports.
    pub imports: Vec<&'static str>,
}

impl CompiledModule {
    pub fn func(&self, export: &str) -> Option<&FuncMeta> {
        self.funcs.iter().find(|f| f.export == export)
    }
}

/// What a module is compiled for. Only code reachable from the entry is emitted, so the
/// module imports exactly the host functions that code can call.
#[derive(Debug, Clone, PartialEq)]
pub enum Entry {
    /// One function (e.g. `main`).
    Fn(String),
    /// All tests.
    Tests,
    /// All functions and tests.
    All,
}

pub fn compile(p: &TProgram, entry: &Entry) -> CompiledModule {
    let (fn_roots, with_tests): (Vec<usize>, bool) = match entry {
        Entry::Fn(name) => (p.fn_index(name).into_iter().collect(), false),
        Entry::Tests => (vec![], true),
        Entry::All => ((0..p.fns.len()).collect(), true),
    };
    let tests: Vec<usize> = if with_tests { (0..p.tests.len()).collect() } else { vec![] };
    let mut reached = BTreeSet::new();
    let mut builtins = BTreeSet::new();
    let mut stack = fn_roots;
    for t in &tests {
        walk(&p.tests[*t].body, &mut stack, &mut builtins);
    }
    while let Some(i) = stack.pop() {
        if reached.insert(i) {
            let f = &p.fns[i];
            for e in f.req.iter().chain(&f.ens).chain([&f.body]) {
                walk(e, &mut stack, &mut builtins);
            }
        }
    }

    let mut imports = vec![host::FAIL];
    for b in &builtins {
        let name = match b {
            Builtin::FromFloat => host::FLOAT_TO_STR,
            Builtin::ToFloat => host::STR_TO_FLOAT,
            Builtin::Print => host::PRINT,
            Builtin::ReadLine => host::READ_LINE,
            Builtin::ReadFile => host::READ_FILE,
            Builtin::WriteFile => host::WRITE_FILE,
            Builtin::NowMs => host::NOW_MS,
            Builtin::RandInt => host::RAND_INT,
            _ => continue,
        };
        imports.push(name);
    }

    let mut mb = Mb::new(p, &imports);
    let mut fn_index = vec![None; p.fns.len()];
    let mut metas = Vec::new();
    let mut exports = Vec::new();
    for &i in &reached {
        let f = &p.fns[i];
        let params: Vec<_> = f.def.params.iter().map(|p| asm::val(&p.ty)).collect();
        let idx = mb.declare(&params, &[asm::val(&f.def.ret)]);
        fn_index[i] = Some(idx);
        exports.push((f.def.name.clone(), idx));
        metas.push(FuncMeta {
            name: f.def.name.clone(),
            export: f.def.name.clone(),
            is_test: false,
            params: f.def.params.iter().map(|p| (p.name.clone(), p.ty.clone())).collect(),
            ret: f.def.ret.clone(),
            effects: f.def.effects.clone(),
            req: f.def.req.clone(),
            ens: f.def.ens.clone(),
            sites: vec![],
        });
    }
    let table: Vec<u32> = fn_index.iter().flatten().copied().collect();
    let mut test_index = Vec::new();
    for &t in &tests {
        let name = &p.tests[t].name;
        let idx = mb.declare(&[], &[wasm_encoder::ValType::I32]);
        test_index.push(idx);
        exports.push((format!("test:{name}"), idx));
        metas.push(FuncMeta {
            name: name.clone(),
            export: format!("test:{name}"),
            is_test: true,
            params: vec![],
            ret: Type::Bool,
            effects: vec![],
            req: vec![],
            ens: vec![],
            sites: vec![],
        });
    }

    let mut meta = 0;
    for &i in &reached {
        let f = &p.fns[i];
        let (body, sites) = FnGen::new(&mut mb, meta, &fn_index, &f.locals, f.def.params.len()).fn_body(f);
        mb.define(fn_index[i].expect("declared"), body);
        metas[meta as usize].sites = sites;
        meta += 1;
    }
    for (&t, &idx) in tests.iter().zip(&test_index) {
        let test = &p.tests[t];
        let (body, sites) = FnGen::new(&mut mb, meta, &fn_index, &test.locals, 0).test_body(test);
        mb.define(idx, body);
        metas[meta as usize].sites = sites;
        meta += 1;
    }

    CompiledModule { wasm: mb.finish(&exports, &table), funcs: metas, types: p.types.clone(), imports }
}

/// Collects called/referenced functions and used built-ins.
fn walk(e: &TExpr, fns: &mut Vec<usize>, builtins: &mut BTreeSet<Builtin>) {
    match &e.kind {
        TKind::Int(_) | TKind::Float(_) | TKind::Bool(_) | TKind::Str(_) | TKind::Unit | TKind::Local(_) => {}
        TKind::FnRef(i) => fns.push(*i),
        TKind::Un(_, x, _) | TKind::Field(x, _) => walk(x, fns, builtins),
        TKind::Bin(_, l, r, _) | TKind::Let(_, l, r) => {
            walk(l, fns, builtins);
            walk(r, fns, builtins);
        }
        TKind::If(c, t, f) => {
            for x in [c, t, f] {
                walk(x, fns, builtins);
            }
        }
        TKind::Call(i, xs) => {
            fns.push(*i);
            xs.iter().for_each(|x| walk(x, fns, builtins));
        }
        TKind::Builtin(b, xs, _) => {
            builtins.insert(*b);
            xs.iter().for_each(|x| walk(x, fns, builtins));
        }
        TKind::CallIndirect(f, xs) => {
            walk(f, fns, builtins);
            xs.iter().for_each(|x| walk(x, fns, builtins));
        }
        TKind::List(xs) | TKind::Variant(_, xs) | TKind::Record(xs) | TKind::Seq(xs) => xs.iter().for_each(|x| walk(x, fns, builtins)),
        TKind::Match(s, arms) => {
            walk(s, fns, builtins);
            for a in arms {
                if let Some(g) = &a.guard {
                    walk(g, fns, builtins);
                }
                walk(&a.body, fns, builtins);
            }
        }
    }
}
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `cd qal && cargo test -p qal-wasm 2>&1 | grep -E "test result|warning"`
Expected: `test result: ok. 16 passed` for `tests/codegen.rs`, no warnings.

- [ ] **Step 9: Commit**

```bash
git add qal/crates/qal-wasm
git commit -m "feat(qal): wasm codegen for data types — bump allocator, Str/List runtime, match, fn values

Function indices are computed by a module builder; modules are compiled per entry
point and import only the host functions their reachable code calls. Failures pass
their site and involved values to qal_host.fail.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Runtime with permissions, effects and structured errors (`qal-run`)

**Files:**
- Modify (full rewrite): `qal/crates/qal-run/src/lib.rs`
- Create: `qal/crates/qal-run/src/host.rs`, `qal/crates/qal-run/src/value.rs`
- Test: `qal/crates/qal-run/src/tests.rs`, inline modules in `host.rs` and `value.rs`

**Interfaces:**
- Consumes: Task 4 `CompiledModule`, `FuncMeta`, `Site`, `host` names, `FAIL_*`, `OOM_SCRATCH`; Task 1 `TProgram` lookups (for decoding); `qal_text::print_expr` (contract text); dev: `qal_check::check`, `qal_wasm::{compile, Entry}`.
- Produces:
  - `Runner::new(CompiledModule) -> Result<Runner, String>`, builders `with_fuel(u64)`, `with_memory_limit(bytes)`, `with_permissions(Permissions)`, `with_stdin(bytes)`; `meta()`; `call(export, &[Value]) -> Result<Value, RunError>`; `invoke(export, &[Value]) -> Outcome { result, stdout }`; `run_main(&[String]) -> Outcome`; `run_tests() -> Vec<TestResult>`. Constants `DEFAULT_FUEL`, `DEFAULT_MEMORY_LIMIT = 256 MiB`.
  - `Permissions { io, time, rand, fs: Vec<PathBuf> }` with `parse("io,fs:./data")`, `allows(effect)`, `check_path(path, for_write)`.
  - `Value::{Int, Float, Bool, Unit, Str, List, Record(name, fields), Variant(name, payload)}` (JSON: Unit = `null`, record = object, payload-free variant = `"Name"`, else `{"Name": [...]}`, non-finite floats as strings); `Values` (ordered JSON object).
  - `RunError { code, kind, def, path, message, contract: Option<String>, values: Values, hints }` with `RunError::new(kind, def, message)` (normalises `test:x` to `x`); kinds and codes: `contract_violation E0501, int_overflow E0502, div_by_zero E0503, out_of_memory E0504, index E0505, stack_overflow E0506, timeout E0507, permission_denied E0508, bad_args E0509, unknown_function E0510, trap E0599`.

- [ ] **Step 1: Write the failing tests**

Create `qal/crates/qal-run/src/tests.rs`:

```rust
use super::*;
use qal_wasm::Entry;
use serde_json::json;

fn compile(src: &str, entry: Entry) -> CompiledModule {
    let p = qal_text::parse_program(src).unwrap();
    let tp = qal_check::check(&p).unwrap_or_else(|d| panic!("{d:#?}"));
    qal_wasm::compile(&tp, &entry)
}

fn runner(src: &str) -> Runner {
    Runner::new(compile(src, Entry::All)).unwrap()
}

fn main_runner(src: &str, allow: &str) -> Runner {
    Runner::new(compile(src, Entry::Fn("main".into()))).unwrap().with_permissions(Permissions::parse(allow).unwrap())
}

fn str_list(xs: &[&str]) -> Value {
    Value::List(xs.iter().map(|s| Value::Str(s.to_string())).collect())
}

#[test]
fn calls_with_typed_values() {
    let r = runner(
        "fn pick(c: Bool, a: Int, b: Int) -> Int = if c then a else b\nfn u() -> Unit = ()\nfn neg(b: Bool) -> Bool = !b\n\
         fn greet(name: Str, xs: List[Str]) -> Str = name ++ \":\" ++ join(xs, \",\")\nfn half(x: Float) -> Float = x / 2.0",
    );
    assert_eq!(r.call("pick", &[Value::Bool(false), Value::Int(1), Value::Int(2)]), Ok(Value::Int(2)));
    assert_eq!(r.call("u", &[]), Ok(Value::Unit));
    assert_eq!(r.call("neg", &[Value::Bool(true)]), Ok(Value::Bool(false)));
    assert_eq!(r.call("greet", &[Value::Str("é".into()), str_list(&["a", "b"])]), Ok(Value::Str("é:a,b".into())));
    assert_eq!(r.call("half", &[Value::Float(3.0)]), Ok(Value::Float(1.5)));
}

#[test]
fn returns_structured_values() {
    let r = runner(
        "type P = { x: Int, tags: List[Str] }\ntype E = A | B(P, Option[Int])\n\
         fn mk() -> E = B(P { x: 1, tags: [\"t\"] }, None)\nfn parse(s: Str) -> Result[Int, Str] = match to_int(s) { Some(n) => Ok(n), None => Err(s) }",
    );
    let v = r.call("mk", &[]).unwrap();
    assert_eq!(serde_json::to_value(&v).unwrap(), json!({"B": [{"x": 1, "tags": ["t"]}, "None"]}));
    assert_eq!(serde_json::to_value(r.call("parse", &[Value::Str("x".into())]).unwrap()).unwrap(), json!({"Err": ["x"]}));
}

#[test]
fn rejects_bad_arguments_and_unknown_functions() {
    let r = runner("fn f(a: Int) -> Int = a\ntest t = true");
    assert_eq!(r.call("f", &[]).unwrap_err().kind, "bad_args");
    let e = r.call("f", &[Value::Bool(true)]).unwrap_err();
    assert_eq!((e.kind, e.code, e.message.as_str()), ("bad_args", "E0509", "argument 0 (`a`): expected Int, got true"));
    let e = r.call("g", &[]).unwrap_err();
    assert_eq!((e.kind, e.def.as_str()), ("unknown_function", "g"));
    assert_eq!(r.call("test:nope", &[]).unwrap_err().def, "nope", "def is the definition name, not the export");
}

#[test]
fn contract_violation_is_a_diagnostic_with_values() {
    let r = runner("fn half(n: Int, label: Str) -> Int req n % 2 == 0 = n / 2");
    let e = r.call("half", &[Value::Int(3), Value::Str("x".into())]).unwrap_err();
    assert_eq!(
        serde_json::to_value(&e).unwrap(),
        json!({"code": "E0501", "kind": "contract_violation", "def": "half", "path": "req[0]",
            "message": "`req n % 2 == 0` of `half` is false", "contract": "req[0]: n % 2 == 0",
            "values": {"n": 3, "label": "x"},
            "hints": ["the caller must ensure `n % 2 == 0` before calling `half`"]})
    );
}

#[test]
fn ens_violation_reports_parameters_and_result() {
    let r = runner("fn inc(n: Int) -> Int ens r > n = if n > 5 then n else n + 1");
    assert_eq!(r.call("inc", &[Value::Int(1)]), Ok(Value::Int(2)));
    let e = r.call("inc", &[Value::Int(9)]).unwrap_err();
    assert_eq!((e.kind, e.path.as_str(), e.contract.as_deref()), ("contract_violation", "ens[0]", Some("ens[0]: r > n")));
    assert_eq!(serde_json::to_value(&e.values).unwrap(), json!({"n": 9, "r": 9}));
}

#[test]
fn arithmetic_failures_name_def_path_and_operands() {
    let r = runner("fn inner(a: Int) -> Int = a / 0\nfn outer() -> Int = inner(1) + 1\nfn big(x: Int) -> Int = 2 * (x * x)");
    let e = r.call("outer", &[]).unwrap_err();
    assert_eq!((e.kind, e.code, e.def.as_str(), e.path.as_str()), ("div_by_zero", "E0503", "inner", "body"));
    assert_eq!(serde_json::to_value(&e.values).unwrap(), json!({"left": 1, "right": 0}));
    let e = r.call("big", &[Value::Int(4_000_000_000)]).unwrap_err();
    assert_eq!((e.kind, e.path.as_str()), ("int_overflow", "body.right"));
    assert_eq!(serde_json::to_value(&e.values).unwrap(), json!({"left": 4_000_000_000i64, "right": 4_000_000_000i64}));
}

#[test]
fn builtin_failures_are_reported() {
    let r = runner("fn cut(s: Str) -> Str = slice(s, 1, 10)\nfn parts(s: Str) -> List[Str] = split(s, \"\")");
    let e = r.call("cut", &[Value::Str("abc".into())]).unwrap_err();
    assert_eq!((e.kind, e.code, e.def.as_str(), e.path.as_str()), ("index", "E0505", "cut", "body"));
    assert_eq!(serde_json::to_value(&e.values).unwrap(), json!({"start": 1, "end": 10, "len": 3}));
    let e = r.call("parts", &[Value::Str("abc".into())]).unwrap_err();
    assert_eq!((e.kind, e.contract.as_deref()), ("contract_violation", Some("split: req sep != \"\"")));
    let r = main_runner("fn main(args: List[Str]) -> Int !rand = rand_int(5, 1)", "rand");
    let e = r.run_main(&[]).result.unwrap_err();
    assert_eq!(serde_json::to_value(&e.values).unwrap(), json!({"lo": 5, "hi": 1}));
}

#[test]
fn memory_limit_reports_out_of_memory() {
    let r = runner("fn big(n: Int) -> Int = len(range(0, n))").with_memory_limit(1 << 20);
    assert_eq!(r.call("big", &[Value::Int(1000)]), Ok(Value::Int(1000)));
    let e = r.call("big", &[Value::Int(1_000_000)]).unwrap_err();
    assert_eq!((e.kind, e.code, e.def.as_str()), ("out_of_memory", "E0504", "big"));
    assert_eq!(serde_json::to_value(&e.values).unwrap(), json!({"requested_bytes": 8_000_008, "limit_bytes": 1 << 20}));
}

#[test]
fn infinite_loop_times_out() {
    let r = runner("fn spin(n: Int) -> Int = spin(n)").with_fuel(10_000_000);
    assert_eq!(r.call("spin", &[Value::Int(0)]).unwrap_err().kind, "timeout");
}

#[test]
fn deep_non_tail_recursion_reports_stack_overflow() {
    let r = runner("fn down(n: Int) -> Int = if n == 0 then 0 else 1 + down(n - 1)");
    let e = r.call("down", &[Value::Int(100_000_000)]).unwrap_err();
    assert_eq!(e.kind, "stack_overflow");
    assert!(e.hints[0].contains("tail-recursive"));
}

#[test]
fn runs_tests_in_order() {
    let r = Runner::new(compile("test a = 1 == 1\ntest b = 1 == 2\ntest c = 1 / 0 == 0", Entry::Tests)).unwrap();
    let res = r.run_tests();
    assert_eq!(res.iter().map(|t| (t.name.as_str(), t.passed)).collect::<Vec<_>>(), vec![("a", true), ("b", false), ("c", false)]);
    let e = res[2].error.as_ref().unwrap();
    assert_eq!((e.kind, e.def.as_str()), ("div_by_zero", "c"));
}

#[test]
fn main_gets_args_and_prints() {
    let src = "fn main(args: List[Str]) -> Int !io = match get(args, 0) {\n\
                 None => { print(\"usage: hello <name>\"); 2 },\n\
                 Some(n) => { print(\"hello \" ++ n); print(from_int(len(args))); 0 },\n\
               }";
    let r = main_runner(src, "io");
    assert_eq!(r.run_main(&["ada".into(), "x".into()]), Outcome { result: Ok(Value::Int(0)), stdout: "hello ada\n2\n".into() });
    assert_eq!(r.run_main(&[]), Outcome { result: Ok(Value::Int(2)), stdout: "usage: hello <name>\n".into() });
}

#[test]
fn read_line_reads_stdin_until_end() {
    let src = "fn main(args: List[Str]) -> Int !io = loop(0)\n\
               fn loop(total: Int) -> Int !io = match read_line() {\n\
                 None => { print(from_int(total)); 0 },\n\
                 Some(line) => loop(total + (match to_int(trim(line)) { Some(n) => n, None => 0 })),\n\
               }";
    let r = main_runner(src, "io").with_stdin("1\n2\r\nx\n39");
    assert_eq!(r.run_main(&[]).stdout, "42\n");
}

#[test]
fn effects_must_be_granted() {
    let r = main_runner("fn main(args: List[Str]) -> Int !io = { print(\"hi\"); 0 }", "");
    let out = r.run_main(&[]);
    let e = out.result.unwrap_err();
    assert_eq!((e.kind, e.code, e.def.as_str()), ("permission_denied", "E0508", "main"));
    assert_eq!(e.hints, vec!["grant them with `--allow io`".to_string()]);
    assert_eq!(out.stdout, "");
    let r = main_runner("fn main(args: List[Str]) -> Int !fs !time = 0", "time");
    assert_eq!(r.run_main(&[]).result.unwrap_err().hints, vec!["grant them with `--allow fs:<dir>`".to_string()]);
}

#[test]
fn pure_programs_import_no_effects() {
    let m = compile("fn main(args: List[Str]) -> Int = len(args)\nfn other() -> Unit !io = print(\"x\")", Entry::Fn("main".into()));
    assert_eq!(m.imports, vec![qal_wasm::host::FAIL]);
    let r = Runner::new(m).unwrap();
    assert_eq!(r.run_main(&["a".into()]).result, Ok(Value::Int(1)));
}

#[test]
fn fs_is_confined_to_granted_directories() {
    let root = std::env::temp_dir().join(format!("qal-run-fs-{}", std::process::id()));
    let data = root.join("data");
    std::fs::create_dir_all(&data).unwrap();
    std::fs::write(root.join("secret.txt"), "top secret").unwrap();
    let src = "fn main(args: List[Str]) -> Int !fs !io = {\n\
                 print(show(write_file(dir(args) ++ \"/out.txt\", \"written\")));\n\
                 print(show_read(read_file(dir(args) ++ \"/out.txt\")));\n\
                 print(show_read(read_file(dir(args) ++ \"/../secret.txt\")));\n\
                 print(show(write_file(dir(args) ++ \"/../evil.txt\", \"x\")));\n\
                 0 }\n\
               fn dir(args: List[Str]) -> Str = match get(args, 0) { Some(d) => d, None => \".\" }\n\
               fn show(r: Result[Unit, Str]) -> Str = match r { Ok(_) => \"ok\", Err(e) => e }\n\
               fn show_read(r: Result[Str, Str]) -> Str = match r { Ok(s) => s, Err(e) => e }";
    let r = main_runner(src, &format!("io,fs:{}", data.display()));
    let out = r.run_main(&[data.to_string_lossy().into_owned()]);
    assert_eq!(out.result, Ok(Value::Int(0)));
    let lines: Vec<&str> = out.stdout.lines().collect();
    assert_eq!(lines[0], "ok");
    assert_eq!(lines[1], "written");
    assert!(lines[2].starts_with("permission denied:"), "{}", lines[2]);
    assert!(lines[3].starts_with("permission denied:"), "{}", lines[3]);
    assert!(!root.join("evil.txt").exists());
}

#[test]
fn time_and_rand() {
    let src = "fn main(args: List[Str]) -> Int !time !rand = if now_ms() > 1700000000000 && in_range(100) then 0 else 1\n\
               fn in_range(n: Int) -> Bool !rand = if n == 0 then true else (let x = rand_int(1, 6) in x >= 1 && x <= 6) && in_range(n - 1)";
    assert_eq!(main_runner(src, "time,rand").run_main(&[]).result, Ok(Value::Int(0)));
    let full = main_runner("fn main(args: List[Str]) -> Int !rand = rand_int(-9223372036854775807 - 1, 9223372036854775807)", "rand");
    assert!(full.run_main(&[]).result.is_ok());
}

#[test]
fn quadratic_push_hits_the_memory_limit_cleanly() {
    // push copies the list, so building n elements this way allocates ~4n² bytes.
    let src = "fn collect(i: Int, n: Int, acc: List[Int]) -> List[Int] = if i == n then acc else collect(i + 1, n, push(acc, i))\n\
               fn build(n: Int) -> Int = len(collect(0, n, []))";
    let r = runner(src).with_memory_limit(1 << 20);
    assert_eq!(r.call("build", &[Value::Int(100)]), Ok(Value::Int(100)));
    let e = r.call("build", &[Value::Int(5_000)]).unwrap_err();
    assert_eq!((e.kind, e.def.as_str()), ("out_of_memory", "build"));
    assert!(e.hints[0].contains("`push`"), "{:?}", e.hints);
}

#[test]
fn huge_input_is_out_of_memory_not_a_crash() {
    let src = "fn main(args: List[Str]) -> Int !io = match read_line() { Some(l) => len(l), None => 0 }";
    let r = main_runner(src, "io").with_memory_limit(1 << 20).with_stdin(vec![b'x'; 2 << 20]);
    let e = r.run_main(&[]).result.unwrap_err();
    assert_eq!((e.kind, e.def.as_str()), ("out_of_memory", "main"));
}

#[test]
fn stdout_survives_runtime_errors() {
    let out = main_runner("fn main(args: List[Str]) -> Int !io = { print(\"before\"); 1 / (len(args) - len(args)) }", "io").run_main(&[]);
    assert_eq!(out.stdout, "before\n");
    assert_eq!(out.result.unwrap_err().kind, "div_by_zero");
}

#[test]
fn deep_values_are_truncated_in_json() {
    let src = "type Chain = End | Link(Int, Chain)\n\
               fn chain(n: Int, acc: Chain) -> Chain = if n == 0 then acc else chain(n - 1, Link(n, acc))\n\
               fn long() -> Chain = chain(100000, End)";
    let v = runner(src).call("long", &[]).unwrap();
    let json = serde_json::to_string(&v).unwrap();
    assert!(json.starts_with(r#"{"Link":[1,{"Link":[2,"#), "{}", &json[..40]);
    assert!(json.contains(r#""...""#));
}
```

Create `qal/crates/qal-run/src/value.rs` containing only its test module for now:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_as_plain_json() {
        let v = vec![
            Value::Int(-3),
            Value::Bool(true),
            Value::Unit,
            Value::Float(1.5),
            Value::Float(f64::NAN),
            Value::Str("a".into()),
            Value::List(vec![Value::Int(1)]),
            Value::Variant("None".into(), vec![]),
            Value::Variant("Some".into(), vec![Value::Int(2)]),
            Value::Record("P".into(), vec![("y".into(), Value::Int(1)), ("x".into(), Value::Int(2))]),
        ];
        assert_eq!(serde_json::to_string(&v).unwrap(), r#"[-3,true,null,1.5,"NaN","a",[1],"None",{"Some":[2]},{"y":1,"x":2}]"#);
        let vals = Values(vec![("n".into(), Value::Int(3))]);
        assert_eq!(serde_json::to_string(&vals).unwrap(), r#"{"n":3}"#);
    }

    #[test]
    fn decodes_from_memory_layout() {
        let tp = TProgram::default();
        // Str at 0: "hi"; List[Int] at 8: [7]; Some(5) at 24
        let mut mem = vec![0u8; 48];
        mem[0..4].copy_from_slice(&2u32.to_le_bytes());
        mem[4..6].copy_from_slice(b"hi");
        mem[8..12].copy_from_slice(&1u32.to_le_bytes());
        mem[16..24].copy_from_slice(&7i64.to_le_bytes());
        mem[24..28].copy_from_slice(&1u32.to_le_bytes());
        mem[32..40].copy_from_slice(&5i64.to_le_bytes());
        assert_eq!(decode(&tp, &mem, &Type::Str, 0), Value::Str("hi".into()));
        assert_eq!(decode(&tp, &mem, &Type::list(Type::Int), 8), Value::List(vec![Value::Int(7)]));
        assert_eq!(decode(&tp, &mem, &Type::option(Type::Int), 24), Value::Variant("Some".into(), vec![Value::Int(5)]));
        assert_eq!(decode(&tp, &mem, &Type::Str, 1000), Value::Str("<invalid Str at 1000>".into()));
    }
}
```

Create `qal/crates/qal-run/src/host.rs` containing only its test module for now:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_permissions() {
        let p = Permissions::parse("io, rand").unwrap();
        assert!(p.allows("io") && p.allows("rand") && !p.allows("fs") && !p.allows("time"));
        let dir = std::env::temp_dir();
        let p = Permissions::parse(&format!("fs:{}", dir.display())).unwrap();
        assert!(p.allows("fs"));
        assert!(Permissions::parse("fs").unwrap_err().contains("needs a directory"));
        assert!(Permissions::parse("net").unwrap_err().contains("unknown effect `net`"));
        assert!(Permissions::parse("fs:/definitely/not/here").is_err());
        assert_eq!(Permissions::parse("").unwrap(), Permissions::default());
    }

    #[test]
    fn fs_paths_must_stay_inside_granted_directories() {
        let root = std::env::temp_dir().join(format!("qal-perm-{}", std::process::id()));
        let inside = root.join("data");
        std::fs::create_dir_all(&inside).unwrap();
        std::fs::write(root.join("secret.txt"), "x").unwrap();
        std::fs::write(inside.join("a.txt"), "a").unwrap();
        let p = Permissions::parse(&format!("fs:{}", inside.display())).unwrap();
        assert!(p.check_path(&inside.join("a.txt").to_string_lossy(), false).is_ok());
        assert!(p.check_path(&inside.join("new.txt").to_string_lossy(), true).is_ok());
        let escape = inside.join("../secret.txt");
        assert!(p.check_path(&escape.to_string_lossy(), false).unwrap_err().starts_with("permission denied"));
        assert!(p.check_path(&root.join("other.txt").to_string_lossy(), true).unwrap_err().starts_with("permission denied"));
        #[cfg(unix)]
        {
            let link = inside.join("link.txt");
            let _ = std::fs::remove_file(&link);
            std::os::unix::fs::symlink(root.join("created-outside.txt"), &link).unwrap();
            assert!(p.check_path(&link.to_string_lossy(), true).unwrap_err().contains("dangling symbolic link"));
            let back = inside.join("back.txt");
            let _ = std::fs::remove_file(&back);
            std::os::unix::fs::symlink(root.join("secret.txt"), &back).unwrap();
            assert!(p.check_path(&back.to_string_lossy(), false).unwrap_err().starts_with("permission denied"));
        }
    }

    #[test]
    fn random_numbers_stay_in_range() {
        let mut s = 1;
        for _ in 0..1000 {
            let span = 6u64;
            let v = 1 + (next_random(&mut s) % span) as i64;
            assert!((1..=6).contains(&v));
        }
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd qal && cargo test -p qal-run`
Expected: FAIL to compile (`Permissions`, `Outcome`, `Value::Str`, `with_memory_limit` not found).

- [ ] **Step 3: Write value decoding and JSON form**

Replace `qal/crates/qal-run/src/value.rs` with:

```rust
//! QAL values on the host side: decoding from and encoding into guest memory, and JSON form.

use qal_core::ast::Type;
use qal_core::tast::TProgram;
use serde::ser::{SerializeMap, Serializer};
use serde::Serialize;

/// Values deeper than this are shown as `"..."` (guards the host stack against huge recursive values).
const MAX_DEPTH: usize = 200;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    /// JSON `null`.
    Unit,
    Str(String),
    List(Vec<Value>),
    /// Record type name and fields in declaration order; JSON object.
    Record(String, Vec<(String, Value)>),
    /// Variant name and payload; JSON `"Name"` without payload, else `{"Name": [payload...]}`.
    Variant(String, Vec<Value>),
}

impl Serialize for Value {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Value::Int(n) => s.serialize_i64(*n),
            Value::Float(x) if x.is_finite() => s.serialize_f64(*x),
            Value::Float(x) => s.serialize_str(&format!("{x:?}")),
            Value::Bool(b) => s.serialize_bool(*b),
            Value::Unit => s.serialize_unit(),
            Value::Str(t) => s.serialize_str(t),
            Value::List(xs) => xs.serialize(s),
            Value::Record(_, fields) => {
                let mut m = s.serialize_map(Some(fields.len()))?;
                for (k, v) in fields {
                    m.serialize_entry(k, v)?;
                }
                m.end()
            }
            Value::Variant(name, xs) if xs.is_empty() => s.serialize_str(name),
            Value::Variant(name, xs) => {
                let mut m = s.serialize_map(Some(1))?;
                m.serialize_entry(name, xs)?;
                m.end()
            }
        }
    }
}

/// Named values (contract parameters, operands); serialized as a JSON object in order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Values(pub Vec<(String, Value)>);

impl Values {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Serialize for Values {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            m.serialize_entry(k, v)?;
        }
        m.end()
    }
}

fn u32_at(mem: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(mem.get(at..at + 4)?.try_into().ok()?))
}

pub fn slot_at(mem: &[u8], at: usize) -> Option<i64> {
    Some(i64::from_le_bytes(mem.get(at..at + 8)?.try_into().ok()?))
}

pub fn read_str(mem: &[u8], ptr: u32) -> Option<String> {
    let len = u32_at(mem, ptr as usize)? as usize;
    let bytes = mem.get(ptr as usize + 4..ptr as usize + 4 + len)?;
    Some(String::from_utf8_lossy(bytes).into_owned())
}

/// Decodes a value of type `ty` from its slot form (see `qal_wasm` layout).
pub fn decode(tp: &TProgram, mem: &[u8], ty: &Type, slot: i64) -> Value {
    decode_at(tp, mem, ty, slot, 0)
}

fn decode_at(tp: &TProgram, mem: &[u8], ty: &Type, slot: i64, depth: usize) -> Value {
    if depth > MAX_DEPTH {
        return Value::Str("...".into());
    }
    let ptr = slot as u32;
    let bad = || Value::Str(format!("<invalid {ty} at {ptr}>"));
    match ty {
        Type::Int => Value::Int(slot),
        Type::Float => Value::Float(f64::from_bits(slot as u64)),
        Type::Bool => Value::Bool(slot != 0),
        Type::Unit | Type::Var(_) => Value::Unit,
        Type::Fn(..) => Value::Str(format!("<fn #{ptr}>")),
        Type::Str => read_str(mem, ptr).map(Value::Str).unwrap_or_else(bad),
        Type::List(elem) => {
            let Some(len) = u32_at(mem, ptr as usize) else { return bad() };
            let mut out = Vec::new();
            for k in 0..len as usize {
                let Some(s) = slot_at(mem, ptr as usize + 8 + 8 * k) else { return bad() };
                out.push(decode_at(tp, mem, elem, s, depth + 1));
            }
            Value::List(out)
        }
        _ if tp.is_record(ty) => {
            let Type::Named(name) = ty else { unreachable!() };
            let mut fields = Vec::new();
            for (k, (f, ft)) in tp.record_fields(ty).iter().enumerate() {
                let Some(s) = slot_at(mem, ptr as usize + 8 * k) else { return bad() };
                fields.push((f.clone(), decode_at(tp, mem, ft, s, depth + 1)));
            }
            Value::Record(name.clone(), fields)
        }
        _ => {
            let Some(tag) = u32_at(mem, ptr as usize) else { return bad() };
            let names = tp.variant_names(ty);
            let Some(name) = names.get(tag as usize) else { return bad() };
            let mut payload = Vec::new();
            for (k, ft) in tp.variant_fields(ty, tag).iter().enumerate() {
                let Some(s) = slot_at(mem, ptr as usize + 8 + 8 * k) else { return bad() };
                payload.push(decode_at(tp, mem, ft, s, depth + 1));
            }
            Value::Variant(name.clone(), payload)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_as_plain_json() {
        let v = vec![
            Value::Int(-3),
            Value::Bool(true),
            Value::Unit,
            Value::Float(1.5),
            Value::Float(f64::NAN),
            Value::Str("a".into()),
            Value::List(vec![Value::Int(1)]),
            Value::Variant("None".into(), vec![]),
            Value::Variant("Some".into(), vec![Value::Int(2)]),
            Value::Record("P".into(), vec![("y".into(), Value::Int(1)), ("x".into(), Value::Int(2))]),
        ];
        assert_eq!(serde_json::to_string(&v).unwrap(), r#"[-3,true,null,1.5,"NaN","a",[1],"None",{"Some":[2]},{"y":1,"x":2}]"#);
        let vals = Values(vec![("n".into(), Value::Int(3))]);
        assert_eq!(serde_json::to_string(&vals).unwrap(), r#"{"n":3}"#);
    }

    #[test]
    fn decodes_from_memory_layout() {
        let tp = TProgram::default();
        // Str at 0: "hi"; List[Int] at 8: [7]; Some(5) at 24
        let mut mem = vec![0u8; 48];
        mem[0..4].copy_from_slice(&2u32.to_le_bytes());
        mem[4..6].copy_from_slice(b"hi");
        mem[8..12].copy_from_slice(&1u32.to_le_bytes());
        mem[16..24].copy_from_slice(&7i64.to_le_bytes());
        mem[24..28].copy_from_slice(&1u32.to_le_bytes());
        mem[32..40].copy_from_slice(&5i64.to_le_bytes());
        assert_eq!(decode(&tp, &mem, &Type::Str, 0), Value::Str("hi".into()));
        assert_eq!(decode(&tp, &mem, &Type::list(Type::Int), 8), Value::List(vec![Value::Int(7)]));
        assert_eq!(decode(&tp, &mem, &Type::option(Type::Int), 24), Value::Variant("Some".into(), vec![Value::Int(5)]));
        assert_eq!(decode(&tp, &mem, &Type::Str, 1000), Value::Str("<invalid Str at 1000>".into()));
    }
}
```

- [ ] **Step 4: Write permissions and host functions**

Replace `qal/crates/qal-run/src/host.rs` with:

```rust
//! Host functions (`qal_host.*`) and the permission model: only granted effect groups are linked,
//! and `fs` only reaches files inside granted directories.

use crate::value::read_str;
use qal_wasm::{host, HOST_MODULE};
use std::io::BufRead;
use std::path::{Path, PathBuf};
use wasmtime::{Caller, Extern, Linker, Memory, StoreLimits};

/// Effect permissions for one execution, e.g. parsed from `--allow io,fs:./data`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Permissions {
    pub io: bool,
    pub time: bool,
    pub rand: bool,
    /// Canonical directories `fs` may read and write below. Empty = no `fs`.
    pub fs: Vec<PathBuf>,
}

impl Permissions {
    /// Parses a comma-separated list: `io`, `time`, `rand`, `fs:<dir>` (repeatable). Directories must exist.
    pub fn parse(spec: &str) -> Result<Permissions, String> {
        let mut p = Permissions::default();
        for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            match item {
                "io" => p.io = true,
                "time" => p.time = true,
                "rand" => p.rand = true,
                "fs" => return Err("`fs` needs a directory: `fs:<dir>`, e.g. `fs:./data`".into()),
                _ => match item.strip_prefix("fs:") {
                    Some(dir) => {
                        let canonical = Path::new(dir).canonicalize().map_err(|e| format!("fs:{dir}: {e}"))?;
                        if !canonical.is_dir() {
                            return Err(format!("fs:{dir}: not a directory"));
                        }
                        p.fs.push(canonical);
                    }
                    None => return Err(format!("unknown effect `{item}` (known: io, fs:<dir>, time, rand)")),
                },
            }
        }
        Ok(p)
    }

    pub fn allows(&self, effect: &str) -> bool {
        match effect {
            "io" => self.io,
            "time" => self.time,
            "rand" => self.rand,
            "fs" => !self.fs.is_empty(),
            _ => false,
        }
    }

    /// Resolves `path` (relative to the working directory) and checks it lies inside a granted directory.
    /// For writing, the file itself may not exist yet, but its directory must.
    pub fn check_path(&self, path: &str, for_write: bool) -> Result<PathBuf, String> {
        let p = Path::new(path);
        let resolved = if for_write && !p.exists() {
            // A dangling symlink "does not exist" but writing would follow it, possibly outside.
            if std::fs::symlink_metadata(p).is_ok() {
                return Err(format!("permission denied: `{path}` is a dangling symbolic link"));
            }
            let parent = match p.parent() {
                Some(d) if !d.as_os_str().is_empty() => d,
                _ => Path::new("."),
            };
            let name = p.file_name().ok_or_else(|| format!("`{path}` is not a file path"))?;
            parent.canonicalize().map_err(|e| format!("{path}: {e}"))?.join(name)
        } else {
            p.canonicalize().map_err(|e| format!("{path}: {e}"))?
        };
        if self.fs.iter().any(|root| resolved.starts_with(root)) {
            Ok(resolved)
        } else {
            Err(format!("permission denied: `{path}` is outside the allowed directories"))
        }
    }
}

/// A failure reported by the guest through `qal_host.fail`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Failure {
    pub kind: i32,
    pub func: i32,
    pub site: i32,
    pub vals: i32,
}

pub struct HostState {
    pub failure: Option<Failure>,
    pub stdout: Vec<u8>,
    pub stdin: Box<dyn BufRead + Send>,
    pub perms: Permissions,
    pub rng: u64,
    pub limits: StoreLimits,
}

fn memory(c: &mut Caller<'_, HostState>) -> Result<Memory, wasmtime::Error> {
    match c.get_export("memory") {
        Some(Extern::Memory(m)) => Ok(m),
        _ => Err(wasmtime::Error::msg("module has no memory export")),
    }
}

fn guest_str(c: &mut Caller<'_, HostState>, ptr: i32) -> Result<String, wasmtime::Error> {
    let mem = memory(c)?;
    read_str(mem.data(&c), ptr as u32).ok_or_else(|| wasmtime::Error::msg("invalid string pointer"))
}

/// Allocates a Str in guest memory via the exported allocator (which may fail with out_of_memory).
pub fn new_str(c: &mut Caller<'_, HostState>, s: &str) -> Result<i32, wasmtime::Error> {
    let alloc = c.get_export("qal_alloc").and_then(Extern::into_func).ok_or_else(|| wasmtime::Error::msg("no qal_alloc"))?;
    let p = alloc.typed::<i64, i32>(&c)?.call(&mut *c, 4 + s.len() as i64)?;
    let mem = memory(c)?;
    mem.write(&mut *c, p as usize, &(s.len() as u32).to_le_bytes())?;
    mem.write(&mut *c, p as usize + 4, s.as_bytes())?;
    Ok(p)
}

/// splitmix64
fn next_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Defines `fail`, the pure helpers, and the functions of every granted effect.
pub fn link(linker: &mut Linker<HostState>, perms: &Permissions) -> Result<(), wasmtime::Error> {
    linker.func_wrap(HOST_MODULE, host::FAIL, |mut c: Caller<'_, HostState>, kind: i32, func: i32, site: i32, vals: i32| {
        c.data_mut().failure = Some(Failure { kind, func, site, vals });
        Err::<(), _>(wasmtime::Error::msg("qal failure"))
    })?;
    linker.func_wrap(HOST_MODULE, host::FLOAT_TO_STR, |mut c: Caller<'_, HostState>, x: f64| new_str(&mut c, &format!("{x:?}")))?;
    linker.func_wrap(HOST_MODULE, host::STR_TO_FLOAT, |mut c: Caller<'_, HostState>, p: i32| {
        Ok(match guest_str(&mut c, p)?.parse::<f64>() {
            Ok(x) => (1, x),
            Err(_) => (0, 0.0),
        })
    })?;
    if perms.io {
        linker.func_wrap(HOST_MODULE, host::PRINT, |mut c: Caller<'_, HostState>, p: i32| {
            let s = guest_str(&mut c, p)?;
            let out = &mut c.data_mut().stdout;
            out.extend_from_slice(s.as_bytes());
            out.push(b'\n');
            Ok(())
        })?;
        linker.func_wrap(HOST_MODULE, host::READ_LINE, |mut c: Caller<'_, HostState>| {
            let mut buf = Vec::new();
            let n = c.data_mut().stdin.read_until(b'\n', &mut buf).map_err(wasmtime::Error::msg)?;
            if n == 0 {
                return Ok(-1);
            }
            if buf.ends_with(b"\n") {
                buf.pop();
                if buf.ends_with(b"\r") {
                    buf.pop();
                }
            }
            new_str(&mut c, &String::from_utf8_lossy(&buf))
        })?;
    }
    if !perms.fs.is_empty() {
        linker.func_wrap(HOST_MODULE, host::READ_FILE, |mut c: Caller<'_, HostState>, p: i32| {
            let path = guest_str(&mut c, p)?;
            let result = c.data().perms.check_path(&path, false).and_then(|full| {
                let bytes = std::fs::read(&full).map_err(|e| format!("{path}: {e}"))?;
                String::from_utf8(bytes).map_err(|_| format!("{path}: not valid UTF-8"))
            });
            let (ok, text) = match result {
                Ok(s) => (1, s),
                Err(e) => (0, e),
            };
            Ok((ok, new_str(&mut c, &text)?))
        })?;
        linker.func_wrap(HOST_MODULE, host::WRITE_FILE, |mut c: Caller<'_, HostState>, p: i32, content: i32| {
            let path = guest_str(&mut c, p)?;
            let text = guest_str(&mut c, content)?;
            let result = c.data().perms.check_path(&path, true).and_then(|full| std::fs::write(&full, text).map_err(|e| format!("{path}: {e}")));
            match result {
                Ok(()) => Ok((1, 0)),
                Err(e) => Ok((0, new_str(&mut c, &e)?)),
            }
        })?;
    }
    if perms.time {
        linker.func_wrap(HOST_MODULE, host::NOW_MS, || {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
        })?;
    }
    if perms.rand {
        linker.func_wrap(HOST_MODULE, host::RAND_INT, |mut c: Caller<'_, HostState>, lo: i64, hi: i64| {
            let span = (hi as u64).wrapping_sub(lo as u64).wrapping_add(1);
            let r = next_random(&mut c.data_mut().rng);
            let offset = if span == 0 { r } else { r % span };
            lo.wrapping_add(offset as i64)
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_permissions() {
        let p = Permissions::parse("io, rand").unwrap();
        assert!(p.allows("io") && p.allows("rand") && !p.allows("fs") && !p.allows("time"));
        let dir = std::env::temp_dir();
        let p = Permissions::parse(&format!("fs:{}", dir.display())).unwrap();
        assert!(p.allows("fs"));
        assert!(Permissions::parse("fs").unwrap_err().contains("needs a directory"));
        assert!(Permissions::parse("net").unwrap_err().contains("unknown effect `net`"));
        assert!(Permissions::parse("fs:/definitely/not/here").is_err());
        assert_eq!(Permissions::parse("").unwrap(), Permissions::default());
    }

    #[test]
    fn fs_paths_must_stay_inside_granted_directories() {
        let root = std::env::temp_dir().join(format!("qal-perm-{}", std::process::id()));
        let inside = root.join("data");
        std::fs::create_dir_all(&inside).unwrap();
        std::fs::write(root.join("secret.txt"), "x").unwrap();
        std::fs::write(inside.join("a.txt"), "a").unwrap();
        let p = Permissions::parse(&format!("fs:{}", inside.display())).unwrap();
        assert!(p.check_path(&inside.join("a.txt").to_string_lossy(), false).is_ok());
        assert!(p.check_path(&inside.join("new.txt").to_string_lossy(), true).is_ok());
        let escape = inside.join("../secret.txt");
        assert!(p.check_path(&escape.to_string_lossy(), false).unwrap_err().starts_with("permission denied"));
        assert!(p.check_path(&root.join("other.txt").to_string_lossy(), true).unwrap_err().starts_with("permission denied"));
        #[cfg(unix)]
        {
            let link = inside.join("link.txt");
            let _ = std::fs::remove_file(&link);
            std::os::unix::fs::symlink(root.join("created-outside.txt"), &link).unwrap();
            assert!(p.check_path(&link.to_string_lossy(), true).unwrap_err().contains("dangling symbolic link"));
            let back = inside.join("back.txt");
            let _ = std::fs::remove_file(&back);
            std::os::unix::fs::symlink(root.join("secret.txt"), &back).unwrap();
            assert!(p.check_path(&back.to_string_lossy(), false).unwrap_err().starts_with("permission denied"));
        }
    }

    #[test]
    fn random_numbers_stay_in_range() {
        let mut s = 1;
        for _ in 0..1000 {
            let span = 6u64;
            let v = 1 + (next_random(&mut s) % span) as i64;
            assert!((1..=6).contains(&v));
        }
    }
}
```

- [ ] **Step 5: Write the runner**

Replace `qal/crates/qal-run/src/lib.rs` with:

```rust
//! QAL runtime: executes compiled modules in wasmtime with capability-gated host functions and
//! reports every failure as data in the diagnostic format (code, kind, def, path, message, values, hints).

mod host;
mod value;

pub use host::Permissions;
pub use value::{Value, Values};

use host::{Failure, HostState};
use qal_core::ast::Type;
use qal_core::tast::TProgram;
use qal_text::print_expr;
use qal_wasm::{CompiledModule, FuncMeta, FAIL_BUILTIN, FAIL_DIV_ZERO, FAIL_ENS, FAIL_INDEX, FAIL_OOM, FAIL_OVERFLOW, FAIL_REQ, OOM_SCRATCH};
use serde::Serialize;
use std::io::Cursor;
use wasmtime::{Config, Engine, Linker, Memory, Module, Store, StoreLimitsBuilder, Trap, Val};

/// Instruction budget per call; exhausting it reports `timeout` instead of hanging.
pub const DEFAULT_FUEL: u64 = 2_000_000_000;
/// Memory limit per execution (spec §8).
pub const DEFAULT_MEMORY_LIMIT: usize = 256 << 20;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RunError {
    pub code: &'static str,
    /// contract_violation | int_overflow | div_by_zero | out_of_memory | index | stack_overflow | timeout
    /// | permission_denied | bad_args | unknown_function | trap
    pub kind: &'static str,
    /// Name of the definition where the failure happened (never an export name like `test:x`).
    pub def: String,
    /// AST path inside `def` (e.g. `req[0]`, `body.left`); empty if unknown.
    pub path: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contract: Option<String>,
    /// Values involved: parameters (and `r`) for contracts, operands for arithmetic, arguments for built-ins.
    #[serde(skip_serializing_if = "Values::is_empty")]
    pub values: Values,
    pub hints: Vec<String>,
}

impl RunError {
    pub fn new(kind: &'static str, def: &str, message: impl Into<String>) -> RunError {
        let code = match kind {
            "contract_violation" => "E0501",
            "int_overflow" => "E0502",
            "div_by_zero" => "E0503",
            "out_of_memory" => "E0504",
            "index" => "E0505",
            "stack_overflow" => "E0506",
            "timeout" => "E0507",
            "permission_denied" => "E0508",
            "bad_args" => "E0509",
            "unknown_function" => "E0510",
            _ => "E0599",
        };
        RunError {
            code,
            kind,
            def: def.strip_prefix("test:").unwrap_or(def).to_string(),
            path: String::new(),
            message: message.into(),
            contract: None,
            values: Values::default(),
            hints: Vec::new(),
        }
    }

    fn at(mut self, path: &str) -> RunError {
        self.path = path.to_string();
        self
    }

    fn hint(mut self, h: impl Into<String>) -> RunError {
        self.hints.push(h.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TestResult {
    pub name: String,
    pub passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RunError>,
}

/// Result of one execution plus everything the program printed.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub result: Result<Value, RunError>,
    pub stdout: String,
}

pub struct Runner {
    engine: Engine,
    module: Module,
    meta: CompiledModule,
    types: TProgram,
    fuel: u64,
    memory_limit: usize,
    perms: Permissions,
    stdin: Option<Vec<u8>>,
}

impl Runner {
    pub fn new(meta: CompiledModule) -> Result<Runner, String> {
        let mut config = Config::new();
        config.wasm_tail_call(true);
        config.consume_fuel(true);
        let engine = Engine::new(&config).map_err(|e| e.to_string())?;
        let module = Module::new(&engine, &meta.wasm).map_err(|e| e.to_string())?;
        let types = TProgram { types: meta.types.clone(), ..TProgram::default() };
        Ok(Runner { engine, module, meta, types, fuel: DEFAULT_FUEL, memory_limit: DEFAULT_MEMORY_LIMIT, perms: Permissions::default(), stdin: None })
    }

    pub fn with_fuel(mut self, fuel: u64) -> Runner {
        self.fuel = fuel;
        self
    }

    pub fn with_memory_limit(mut self, bytes: usize) -> Runner {
        self.memory_limit = bytes;
        self
    }

    pub fn with_permissions(mut self, perms: Permissions) -> Runner {
        self.perms = perms;
        self
    }

    /// Feeds `read_line` from these bytes instead of the process's stdin.
    pub fn with_stdin(mut self, input: impl Into<Vec<u8>>) -> Runner {
        self.stdin = Some(input.into());
        self
    }

    pub fn meta(&self) -> &CompiledModule {
        &self.meta
    }

    /// Calls the function exported as `export` (a fn name, or `test:<name>`).
    pub fn call(&self, export: &str, args: &[Value]) -> Result<Value, RunError> {
        self.invoke(export, args).result
    }

    /// Runs the entry point `main(args: List[Str]) -> Int`; the result is the exit code.
    pub fn run_main(&self, args: &[String]) -> Outcome {
        let list = Value::List(args.iter().map(|a| Value::Str(a.clone())).collect());
        self.invoke("main", &[list])
    }

    /// Runs every test definition in program order.
    pub fn run_tests(&self) -> Vec<TestResult> {
        self.meta
            .funcs
            .iter()
            .filter(|f| f.is_test)
            .map(|f| match self.call(&f.export, &[]) {
                Ok(Value::Bool(passed)) => TestResult { name: f.name.clone(), passed, error: None },
                Ok(_) => unreachable!("tests are checked to return Bool"),
                Err(e) => TestResult { name: f.name.clone(), passed: false, error: Some(e) },
            })
            .collect()
    }

    pub fn invoke(&self, export: &str, args: &[Value]) -> Outcome {
        let stdin: Box<dyn std::io::BufRead + Send> = match &self.stdin {
            Some(bytes) => Box::new(Cursor::new(bytes.clone())),
            None => Box::new(std::io::BufReader::new(std::io::stdin())),
        };
        let state = HostState {
            failure: None,
            stdout: Vec::new(),
            stdin,
            perms: self.perms.clone(),
            rng: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1),
            limits: StoreLimitsBuilder::new().memory_size(self.memory_limit).build(),
        };
        let mut store = Store::new(&self.engine, state);
        store.limiter(|s| &mut s.limits);
        let result = self.invoke_in(&mut store, export, args);
        let stdout = String::from_utf8_lossy(&store.data().stdout).into_owned();
        Outcome { result, stdout }
    }

    fn invoke_in(&self, store: &mut Store<HostState>, export: &str, args: &[Value]) -> Result<Value, RunError> {
        let meta = self.meta.func(export).ok_or_else(|| {
            RunError::new("unknown_function", export, format!("no function `{export}` in this module"))
                .hint("only functions reachable from the compiled entry point can be called")
        })?;
        let missing: Vec<&String> = meta.effects.iter().filter(|e| !self.perms.allows(e)).collect();
        if !missing.is_empty() {
            let list: Vec<&str> = missing.iter().map(|s| s.as_str()).collect();
            let flag = list.iter().map(|e| if *e == "fs" { "fs:<dir>" } else { e }).collect::<Vec<_>>().join(",");
            return Err(RunError::new("permission_denied", &meta.name, format!("`{}` needs effect(s) {} which were not granted", meta.name, list.join(", ")))
                .hint(format!("grant them with `--allow {flag}`")));
        }
        if meta.params.len() != args.len() {
            return Err(RunError::new("bad_args", &meta.name, format!("`{}` takes {} argument(s), {} given", meta.name, meta.params.len(), args.len())));
        }
        store.set_fuel(self.fuel).map_err(|e| RunError::new("trap", &meta.name, e.to_string()))?;
        let mut linker = Linker::new(&self.engine);
        host::link(&mut linker, &self.perms).map_err(|e| RunError::new("trap", &meta.name, e.to_string()))?;
        let instance = linker.instantiate(&mut *store, &self.module).map_err(|e| RunError::new("trap", &meta.name, e.to_string()))?;
        let mem = instance.get_memory(&mut *store, "memory").expect("memory export");
        let mut params = Vec::new();
        for (i, (a, (pname, t))) in args.iter().zip(&meta.params).enumerate() {
            let slot = match self.encode(store, &instance, t, a) {
                Ok(slot) => slot,
                Err(_) if store.data().failure.is_some() => {
                    let f = store.data().failure.expect("checked");
                    return Err(self.describe(store, mem, &meta.name, f));
                }
                Err(m) => return Err(RunError::new("bad_args", &meta.name, format!("argument {i} (`{pname}`): {m}"))),
            };
            params.push(match t {
                Type::Int => Val::I64(slot),
                Type::Float => Val::F64(slot as u64),
                _ => Val::I32(slot as i32),
            });
        }
        let func = instance.get_func(&mut *store, export).ok_or_else(|| RunError::new("unknown_function", export, format!("no export `{export}`")))?;
        let mut results = [Val::I32(0)];
        match func.call(&mut *store, &params, &mut results) {
            Ok(()) => {
                let slot = match results[0] {
                    Val::I64(v) => v,
                    Val::F64(bits) => bits as i64,
                    Val::I32(v) => v as u32 as i64,
                    ref other => return Err(RunError::new("trap", &meta.name, format!("unexpected result {other:?}"))),
                };
                Ok(value::decode(&self.types, mem.data(&*store), &meta.ret, slot))
            }
            Err(e) => Err(match store.data().failure {
                Some(f) => self.describe(store, mem, &meta.name, f),
                None => match e.downcast_ref::<Trap>() {
                    Some(Trap::StackOverflow) => RunError::new("stack_overflow", &meta.name, "call stack exhausted")
                        .hint("make the recursion tail-recursive: the recursive call must be the last thing the function does (functions with `ens` never use tail calls)"),
                    Some(Trap::OutOfFuel) => RunError::new("timeout", &meta.name, format!("instruction budget of {} exhausted", self.fuel))
                        .hint("look for unbounded recursion"),
                    _ => RunError::new("trap", &meta.name, e.to_string()),
                },
            }),
        }
    }

    /// Writes an argument into guest memory (Str and List[...] are allocated) and returns its slot.
    fn encode(&self, store: &mut Store<HostState>, instance: &wasmtime::Instance, t: &Type, v: &Value) -> Result<i64, String> {
        Ok(match (t, v) {
            (Type::Int, Value::Int(n)) => *n,
            (Type::Float, Value::Float(x)) => x.to_bits() as i64,
            (Type::Bool, Value::Bool(b)) => *b as i64,
            (Type::Unit, Value::Unit) => 0,
            (Type::Str, Value::Str(s)) => {
                let p = self.alloc(store, instance, 4 + s.len() as i64)?;
                let mem = instance.get_memory(&mut *store, "memory").expect("memory export");
                mem.write(&mut *store, p as usize, &(s.len() as u32).to_le_bytes()).map_err(|e| e.to_string())?;
                mem.write(&mut *store, p as usize + 4, s.as_bytes()).map_err(|e| e.to_string())?;
                p as i64
            }
            (Type::List(elem), Value::List(xs)) => {
                let mut slots = Vec::new();
                for x in xs {
                    slots.push(self.encode(store, instance, elem, x)?);
                }
                let p = self.alloc(store, instance, 8 + 8 * xs.len() as i64)?;
                let mem = instance.get_memory(&mut *store, "memory").expect("memory export");
                mem.write(&mut *store, p as usize, &(xs.len() as u32).to_le_bytes()).map_err(|e| e.to_string())?;
                for (k, s) in slots.iter().enumerate() {
                    mem.write(&mut *store, p as usize + 8 + 8 * k, &s.to_le_bytes()).map_err(|e| e.to_string())?;
                }
                p as i64
            }
            (t, v) => return Err(format!("expected {t}, got {}", serde_json::to_string(v).unwrap_or_default())),
        })
    }

    fn alloc(&self, store: &mut Store<HostState>, instance: &wasmtime::Instance, size: i64) -> Result<i32, String> {
        let f = instance.get_typed_func::<i64, i32>(&mut *store, "qal_alloc").map_err(|e| e.to_string())?;
        f.call(&mut *store, size).map_err(|e| e.to_string())
    }

    fn values(&self, store: &Store<HostState>, mem: Memory, names: &[(String, Type)], ptr: i32) -> Values {
        let data = mem.data(store);
        Values(
            names
                .iter()
                .enumerate()
                .map(|(k, (n, t))| {
                    let slot = value::slot_at(data, ptr as usize + 8 * k).unwrap_or(0);
                    (n.clone(), value::decode(&self.types, data, t, slot))
                })
                .collect(),
        )
    }

    /// Turns a failure reported through `qal_host.fail` into a RunError with values read from guest memory.
    fn describe(&self, store: &Store<HostState>, mem: Memory, entry: &str, f: Failure) -> RunError {
        if f.kind == FAIL_OOM {
            let requested = value::slot_at(mem.data(store), OOM_SCRATCH as usize);
            let mut e = RunError::new("out_of_memory", entry, format!("out of memory: the limit is {} bytes per execution", self.memory_limit))
                .hint("memory is never freed in Phase 1; avoid building large intermediate lists or strings in loops (`push` and `++` copy)");
            if let Some(r) = requested {
                e.values = Values(vec![("requested_bytes".into(), Value::Int(r)), ("limit_bytes".into(), Value::Int(self.memory_limit as i64))]);
            }
            return e;
        }
        let meta: &FuncMeta = &self.meta.funcs[f.func as usize];
        let site = &meta.sites[f.site as usize];
        let values = self.values(store, mem, &site.values, f.vals);
        let def = meta.name.as_str();
        let mut e = match f.kind {
            FAIL_REQ | FAIL_ENS => {
                let (label, list) = if f.kind == FAIL_REQ { ("req", &meta.req) } else { ("ens", &meta.ens) };
                let index = site.contract.expect("contract site");
                let text = print_expr(&list[index]);
                let mut e = RunError::new("contract_violation", def, format!("`{label} {text}` of `{def}` is false"));
                e.contract = Some(format!("{label}[{index}]: {text}"));
                if f.kind == FAIL_REQ {
                    e.hint(format!("the caller must ensure `{text}` before calling `{def}`"))
                } else {
                    e.hint(format!("`{def}` returned a result that violates its postcondition; fix the body or the contract"))
                }
            }
            FAIL_OVERFLOW => RunError::new("int_overflow", def, format!("Int overflow in `{}` (64-bit)", site.label))
                .hint("check the operand range before the operation"),
            FAIL_DIV_ZERO => RunError::new("div_by_zero", def, format!("division by zero in `{}`", site.label))
                .hint("check the divisor first, e.g. `if b == 0 then ... else a / b`"),
            FAIL_INDEX => RunError::new("index", def, format!("`{}` index out of range", site.label))
                .hint("valid character indices satisfy 0 <= start <= end <= len(s)"),
            FAIL_BUILTIN => {
                let mut e = RunError::new("contract_violation", def, format!("precondition of a built-in was violated: `{}`", site.label));
                e.contract = Some(site.label.clone());
                e
            }
            other => RunError::new("trap", def, format!("unknown failure kind {other}")),
        };
        e.values = values;
        e.at(&site.path)
    }
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd qal && cargo test -p qal-run 2>&1 | grep -E "test result|warning"`
Expected: `test result: ok. 26 passed`, no warnings.

- [ ] **Step 7: Commit**

```bash
git add qal/crates/qal-run
git commit -m "feat(qal): runtime with effect permissions, fs confinement and diagnostic-format errors

Runtime errors carry code, path, the values involved and hints; def is always a
definition name. Adds runtime tests for ens violations and Int overflow.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: `qal run`, `--allow` and examples (`qal-cli`)

**Files:**
- Modify (full rewrites): `qal/crates/qal-cli/src/lib.rs`, `qal/crates/qal-cli/src/main.rs`, `qal/crates/qal-cli/tests/cli.rs`
- Modify: `qal/crates/qal-cli/Cargo.toml` (adds the workspace's existing `serde` so output is serialized in declaration order; cargo updates `qal/Cargo.lock` accordingly)
- Create: `qal/examples/age.qal` (spec §4 example, verbatim apart from the comment line), `qal/examples/wordcount.qal`
- Test: `qal/crates/qal-cli/tests/cli.rs`, inline module in `lib.rs`

**Interfaces:**
- Consumes: everything above.
- Produces: `qal_cli::{load(&str) -> Result<Loaded, Vec<Diagnostic>>, Loaded { program, typed }, Loaded::runner(&Entry, Permissions) -> Runner, take_options(&[String]) -> Result<(Permissions, Vec<String>), String>, parse_args(&TProgram, name, &[String]) -> Result<Vec<Value>, RunError>}`. CLI (module doc of `main.rs`): `qal run <file> [--allow <spec>] [--] [args..]` → `{"exit_code":n,"stdout":".."}`; `qal call` now also accepts `--allow` and `Str`/`Float` arguments and reports `stdout`.

- [ ] **Step 1: Write the failing tests and examples**

Replace `qal/crates/qal-cli/Cargo.toml` with:

```toml
[package]
name = "qal-cli"
version.workspace = true
edition.workspace = true
license.workspace = true

[[bin]]
name = "qal"
path = "src/main.rs"

[dependencies]
qal-core = { workspace = true }
qal-text = { workspace = true }
qal-check = { workspace = true }
qal-wasm = { workspace = true }
qal-run = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
```

Replace `qal/crates/qal-cli/tests/cli.rs` with:

```rust
//! Drives the real `qal` binary and checks its JSON output.

use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn example(name: &str) -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples").join(name).to_string_lossy().into_owned()
}

fn temp_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qal-cli-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn temp_file(name: &str, content: &str) -> String {
    let path = temp_dir().join(name);
    std::fs::write(&path, content).unwrap();
    path.to_string_lossy().into_owned()
}

fn qal(args: &[&str]) -> (i32, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_qal")).args(args).stdin(Stdio::null()).output().unwrap();
    (out.status.code().unwrap(), String::from_utf8(out.stdout).unwrap())
}

fn qal_json(args: &[&str]) -> (i32, Value) {
    let (code, out) = qal(args);
    (code, serde_json::from_str(&out).unwrap_or_else(|e| panic!("not JSON ({e}): {out}")))
}

#[test]
fn check_ok_and_call() {
    let math = example("math.qal");
    assert_eq!(qal_json(&["check", &math]), (0, json!({"ok": true})));
    assert_eq!(qal_json(&["call", &math, "gcd", "48", "18"]), (0, json!({"ok": 6, "stdout": ""})));
    assert_eq!(qal_json(&["call", &math, "is_prime", "97"]), (0, json!({"ok": true, "stdout": ""})));
}

#[test]
fn call_reports_contract_violation_and_bad_args() {
    let math = example("math.qal");
    let (code, v) = qal_json(&["call", &math, "fact", "-1"]);
    assert_eq!(code, 1);
    assert_eq!(v["error"]["kind"], "contract_violation");
    assert_eq!(v["error"]["contract"], "req[0]: n >= 0");
    assert_eq!(v["error"]["values"], json!({"n": -1}));
    let (code, v) = qal_json(&["call", &math, "gcd", "x", "1"]);
    assert_eq!((code, v["error"]["kind"].as_str()), (1, Some("bad_args")));
    let (code, v) = qal_json(&["call", &math, "nope"]);
    assert_eq!((code, v["error"]["kind"].as_str()), (1, Some("unknown_function")));
}

#[test]
fn call_with_strings_and_effects() {
    let src = temp_file("say.qal", "fn say(s: Str) -> Int !io = { print(s ++ \"!\"); len(s) }");
    assert_eq!(qal_json(&["call", &src, "--allow", "io", "say", "héllo"]), (0, json!({"ok": 5, "stdout": "héllo!\n"})));
    let (code, v) = qal_json(&["call", &src, "say", "x"]);
    assert_eq!((code, v["error"]["kind"].as_str()), (1, Some("permission_denied")));
}

#[test]
fn output_keeps_declaration_order() {
    let src = temp_file("order.qal", "type P = { z: Int, a: Str }\nfn mk(s: Str) -> P = P { z: 1, a: s }\nfn chk(z: Int, a: Int) -> Int req z < a = 0");
    assert_eq!(qal(&["call", &src, "mk", "x"]), (0, "{\"ok\":{\"z\":1,\"a\":\"x\"},\"stdout\":\"\"}\n".to_string()));
    let (_, out) = qal(&["call", &src, "chk", "2", "1"]);
    assert!(out.contains(r#""values":{"z":2,"a":1}"#), "{out}");
}

#[test]
fn non_utf8_arguments_do_not_crash() {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let src = temp_file("argc.qal", "fn main(args: List[Str]) -> Int = len(args)");
        let out = Command::new(env!("CARGO_BIN_EXE_qal"))
            .args([std::ffi::OsStr::new("run"), std::ffi::OsStr::new(&src), std::ffi::OsStr::from_bytes(b"\xff")])
            .output()
            .unwrap();
        assert_eq!(String::from_utf8(out.stdout).unwrap(), "{\"exit_code\":1,\"stdout\":\"\"}\n");
    }
}

#[test]
fn test_command_summarizes() {
    let (code, v) = qal_json(&["test", &example("math.qal")]);
    assert_eq!(code, 0);
    assert_eq!((v["passed"].as_u64(), v["failed"].as_u64()), (Some(4), Some(0)));
    let failing = temp_file("failing.qal", "test good = true\ntest bad = 1 == 2");
    let (code, v) = qal_json(&["test", &failing]);
    assert_eq!((code, v["failed"].as_u64()), (1, Some(1)));
}

#[test]
fn run_passes_args_and_returns_exit_code() {
    let age = example("age.qal");
    assert_eq!(qal_json(&["run", &age, "--allow", "io", "--", " 42"]), (0, json!({"exit_code": 0, "stdout": "ok 42\n"})));
    assert_eq!(qal_json(&["run", &age, "--allow", "io", "abc"]), (1, json!({"exit_code": 1, "stdout": "invalid\n"})));
    assert_eq!(qal_json(&["run", &age, "--allow", "io"]), (2, json!({"exit_code": 2, "stdout": "usage: age <n>\n"})));
    let (code, v) = qal_json(&["test", &age]);
    assert_eq!((code, v["passed"].as_u64()), (0, Some(2)));
}

#[test]
fn run_refuses_ungranted_effects() {
    let (code, v) = qal_json(&["run", &example("age.qal"), "42"]);
    assert_eq!(code, 1);
    assert_eq!(v["error"]["kind"], "permission_denied");
    assert_eq!(v["error"]["hints"], json!(["grant them with `--allow io`"]));
    assert_eq!(v["stdout"], "");
}

#[test]
fn run_reads_files_only_in_granted_directories() {
    let dir = temp_dir().join("wc");
    std::fs::create_dir_all(&dir).unwrap();
    let text = dir.join("text.txt");
    std::fs::write(&text, "one two\nthree\n").unwrap();
    let allow = format!("io,fs:{}", dir.display());
    let wc = example("wordcount.qal");
    assert_eq!(qal_json(&["run", &wc, "--allow", &allow, &text.to_string_lossy()]), (0, json!({"exit_code": 0, "stdout": "2 3 14\n"})));
    let (code, v) = qal_json(&["run", &wc, "--allow", &allow, &example("math.qal")]);
    assert_eq!(code, 1);
    assert!(v["stdout"].as_str().unwrap().starts_with("error: permission denied"), "{v}");
}

#[test]
fn exit_codes_outside_0_to_255_exit_1_but_keep_the_value() {
    let src = temp_file("exit.qal", "fn main(args: List[Str]) -> Int = match get(args, 0) { Some(a) => match to_int(a) { Some(n) => n, None => 0 }, None => 0 }");
    assert_eq!(qal_json(&["run", &src, "300"]), (1, json!({"exit_code": 300, "stdout": ""})));
    assert_eq!(qal_json(&["run", &src, "--", "-1"]), (1, json!({"exit_code": -1, "stdout": ""})));
    assert_eq!(qal_json(&["run", &src, "7"]), (7, json!({"exit_code": 7, "stdout": ""})));
}

#[test]
fn run_without_main_is_an_error() {
    let (code, v) = qal_json(&["run", &example("math.qal")]);
    assert_eq!((code, v["error"]["kind"].as_str()), (1, Some("unknown_function")));
}

#[test]
fn diagnostics_are_json() {
    let bad = temp_file("bad.qal", "fn f(x: Int) -> Bool = x + 1");
    let (code, v) = qal_json(&["check", &bad]);
    assert_eq!(code, 1);
    assert_eq!(v["diagnostics"][0]["kind"], "type_mismatch");
    assert_eq!(v["diagnostics"][0]["path"], "body");
    let broken = temp_file("broken.qal", "fn f( -> Int = 1");
    let (code, v) = qal_json(&["call", &broken, "f"]);
    assert_eq!((code, v["diagnostics"][0]["kind"].as_str()), (1, Some("parse")));
    let partial = temp_file("partial.qal", "fn f(x: Option[Int]) -> Int = match x { Some(n) => n }");
    let (_, v) = qal_json(&["check", &partial]);
    assert_eq!(v["diagnostics"][0]["kind"], "non_exhaustive_match");
}

#[test]
fn fmt_prints_canonical_text() {
    let messy = temp_file("messy.qal", "fn   f(a:Int)->Int=a*(2+3)");
    assert_eq!(qal(&["fmt", &messy]), (0, "fn f(a: Int) -> Int\n= a * (2 + 3)\n".to_string()));
}

#[test]
fn usage_and_io_errors_exit_2() {
    assert_eq!(qal(&[]).0, 2);
    assert_eq!(qal(&["check", "/nonexistent/x.qal"]).0, 2);
    assert_eq!(qal(&["frobnicate", &example("math.qal")]).0, 2);
    assert_eq!(qal(&["run", &example("age.qal"), "--allow", "net"]).0, 2);
    assert_eq!(qal(&["call", &example("math.qal")]).0, 2);
}

#[test]
fn check_rejects_deeply_nested_input_with_json_diagnostic() {
    let src = format!("test t = {}1{} == 1\n", "(".repeat(10_000), ")".repeat(10_000));
    let file = temp_file("deep.qal", &src);
    let (code, v) = qal_json(&["check", &file]);
    assert_eq!(code, 1);
    assert_eq!(v["diagnostics"][0]["kind"], "parse");
    assert!(v["diagnostics"][0]["message"].as_str().unwrap().contains("nested too deeply"), "{v}");
}
```

Create `qal/examples/age.qal`:

```text
# Spec §4 example: parse an age from the command line.

type Err = Empty | NotNumber(Str) | Negative(Int)

fn parse_age(s: Str) -> Result[Int, Err]
  ens match r { Ok(n) => n >= 0, _ => true }
= if len(trim(s)) == 0 then Err(Empty)
  else match to_int(trim(s)) {
    None => Err(NotNumber(s)),
    Some(n) if n < 0 => Err(Negative(n)),
    Some(n) => Ok(n),
  }

test parse_age_ok = parse_age(" 42 ") == Ok(42)
test parse_age_neg = parse_age("-1") == Err(Negative(-1))

fn main(args: List[Str]) -> Int !io
= match get(args, 0) {
    None => { print("usage: age <n>"); 2 },
    Some(a) => match parse_age(a) {
      Ok(n) => { print("ok " ++ from_int(n)); 0 },
      Err(_) => { print("invalid"); 1 },
    },
  }
```

Create `qal/examples/wordcount.qal`:

```text
# Counts lines, words and characters of a file: `qal run examples/wordcount.qal --allow io,fs:examples examples/age.qal`

type Counts = { lines: Int, words: Int, chars: Int }

fn count(text: Str) -> Counts
= Counts {
    lines: len(filter(split(text, "\n"), non_empty)) ,
    words: fold(split(text, "\n"), 0, add_words),
    chars: len(text),
  }

fn non_empty(s: Str) -> Bool = len(trim(s)) > 0

fn add_words(acc: Int, line: Str) -> Int = acc + len(filter(split(line, " "), non_empty))

test counts_words = count("a b\n\nc  d e\n") == Counts { lines: 2, words: 5, chars: 12 }

fn main(args: List[Str]) -> Int !io !fs
= match get(args, 0) {
    None => { print("usage: wordcount <file>"); 2 },
    Some(path) => match read_file(path) {
      Err(e) => { print("error: " ++ e); 1 },
      Ok(text) => {
        let c = count(text) in print(from_int(c.lines) ++ " " ++ from_int(c.words) ++ " " ++ from_int(c.chars));
        0
      },
    },
  }
```

Append to `qal/crates/qal-cli/src/lib.rs` (1a has no test module there):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn strings(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn options_are_taken_from_the_front() {
        let (p, rest) = take_options(&strings(&["--allow", "io", "--allow", "time", "a", "--allow"])).unwrap();
        assert!(p.io && p.time && !p.rand);
        assert_eq!(rest, strings(&["a", "--allow"]));
        let (p, rest) = take_options(&strings(&["--", "--allow", "x"])).unwrap();
        assert_eq!((p, rest), (Permissions::default(), strings(&["--allow", "x"])));
        assert!(take_options(&strings(&["--allow"])).is_err());
        assert!(take_options(&strings(&["--allow", "net"])).unwrap_err().contains("unknown effect"));
    }

    #[test]
    fn parses_cli_arguments_by_signature() {
        let l = load("fn f(a: Int, b: Float, c: Str, d: Bool) -> Int = a\nfn g(xs: List[Int]) -> Int = 0").unwrap();
        let v = parse_args(&l.typed, "f", &strings(&["-3", "2.5", "hi", "true"])).unwrap();
        assert_eq!(v, vec![Value::Int(-3), Value::Float(2.5), Value::Str("hi".into()), Value::Bool(true)]);
        assert_eq!(parse_args(&l.typed, "f", &strings(&["x", "1", "", "true"])).unwrap_err().kind, "bad_args");
        assert!(parse_args(&l.typed, "g", &strings(&["[]"])).unwrap_err().message.contains("cannot be passed"));
        assert_eq!(parse_args(&l.typed, "h", &[]).unwrap_err().kind, "unknown_function");
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd qal && cargo test -p qal-cli`
Expected: FAIL to compile (the 1a CLI uses the removed `qal_wasm::compile(&Program)` and `Value::Unit(())`; `take_options` does not exist).

- [ ] **Step 3: Write the pipeline library**

Replace `qal/crates/qal-cli/src/lib.rs` with:

```rust
//! The QAL pipeline: text -> parse -> check -> compile (per entry point) -> runner.

use qal_core::ast::{Program, Type};
use qal_core::diag::Diagnostic;
use qal_core::tast::TProgram;
use qal_run::{Permissions, RunError, Runner, Value};
use qal_wasm::Entry;

pub struct Loaded {
    pub program: Program,
    pub typed: TProgram,
}

/// Parses and checks `src`. Any diagnostic means nothing can be compiled.
pub fn load(src: &str) -> Result<Loaded, Vec<Diagnostic>> {
    let program = qal_text::parse_program(src).map_err(|d| vec![d])?;
    let typed = qal_check::check(&program)?;
    Ok(Loaded { program, typed })
}

impl Loaded {
    /// Compiles only the code reachable from `entry` and prepares a runner with `perms`.
    pub fn runner(&self, entry: &Entry, perms: Permissions) -> Runner {
        Runner::new(qal_wasm::compile(&self.typed, entry)).expect("checked programs compile to valid wasm").with_permissions(perms)
    }
}

/// Splits leading options off `args`: `--allow <spec>` (repeatable) and an optional `--` separator.
pub fn take_options(args: &[String]) -> Result<(Permissions, Vec<String>), String> {
    let mut spec = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--allow" => {
                let value = args.get(i + 1).ok_or("`--allow` needs a value, e.g. `--allow io,fs:./data`")?;
                spec.push(value.clone());
                i += 2;
            }
            "--" => {
                i += 1;
                break;
            }
            _ => break,
        }
    }
    Ok((Permissions::parse(&spec.join(","))?, args[i..].to_vec()))
}

/// Converts CLI strings to values according to the signature of fn `name` (Int, Float, Bool, Unit, Str).
pub fn parse_args(typed: &TProgram, name: &str, raw: &[String]) -> Result<Vec<Value>, RunError> {
    let bad = |message: String| RunError::new("bad_args", name, message);
    let f = typed.fns.iter().find(|f| f.def.name == name).ok_or_else(|| RunError::new("unknown_function", name, format!("no function `{name}`")))?;
    if f.def.params.len() != raw.len() {
        return Err(bad(format!("`{name}` takes {} argument(s), {} given", f.def.params.len(), raw.len())));
    }
    f.def
        .params
        .iter()
        .zip(raw)
        .map(|(p, s)| match &p.ty {
            Type::Int => s.parse::<i64>().map(Value::Int).map_err(|_| bad(format!("`{}` must be Int, got `{s}`", p.name))),
            Type::Float => s.parse::<f64>().map(Value::Float).map_err(|_| bad(format!("`{}` must be Float, got `{s}`", p.name))),
            Type::Bool => match s.as_str() {
                "true" => Ok(Value::Bool(true)),
                "false" => Ok(Value::Bool(false)),
                _ => Err(bad(format!("`{}` must be Bool (true/false), got `{s}`", p.name))),
            },
            Type::Unit => match s.as_str() {
                "()" => Ok(Value::Unit),
                _ => Err(bad(format!("`{}` must be Unit `()`, got `{s}`", p.name))),
            },
            Type::Str => Ok(Value::Str(s.clone())),
            t => Err(bad(format!("`{}` has type {t}, which cannot be passed on the command line", p.name))),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn options_are_taken_from_the_front() {
        let (p, rest) = take_options(&strings(&["--allow", "io", "--allow", "time", "a", "--allow"])).unwrap();
        assert!(p.io && p.time && !p.rand);
        assert_eq!(rest, strings(&["a", "--allow"]));
        let (p, rest) = take_options(&strings(&["--", "--allow", "x"])).unwrap();
        assert_eq!((p, rest), (Permissions::default(), strings(&["--allow", "x"])));
        assert!(take_options(&strings(&["--allow"])).is_err());
        assert!(take_options(&strings(&["--allow", "net"])).unwrap_err().contains("unknown effect"));
    }

    #[test]
    fn parses_cli_arguments_by_signature() {
        let l = load("fn f(a: Int, b: Float, c: Str, d: Bool) -> Int = a\nfn g(xs: List[Int]) -> Int = 0").unwrap();
        let v = parse_args(&l.typed, "f", &strings(&["-3", "2.5", "hi", "true"])).unwrap();
        assert_eq!(v, vec![Value::Int(-3), Value::Float(2.5), Value::Str("hi".into()), Value::Bool(true)]);
        assert_eq!(parse_args(&l.typed, "f", &strings(&["x", "1", "", "true"])).unwrap_err().kind, "bad_args");
        assert!(parse_args(&l.typed, "g", &strings(&["[]"])).unwrap_err().message.contains("cannot be passed"));
        assert_eq!(parse_args(&l.typed, "h", &[]).unwrap_err().kind, "unknown_function");
    }
}
```

- [ ] **Step 4: Write the binary**

Replace `qal/crates/qal-cli/src/main.rs` with:

```rust
//! `qal` command line. Every output is JSON on stdout, except `fmt` which prints canonical text.
//!
//!   qal check <file>                                 -> {"ok":true} | {"diagnostics":[...]}
//!   qal fmt <file>                                   -> canonical text
//!   qal call <file> [--allow <spec>] <fn> [args..]   -> {"ok":<value>,"stdout":".."} | {"error":{...},"stdout":".."} | {"diagnostics":[...]}
//!   qal test <file>                                  -> {"tests":[...],"passed":n,"failed":m}
//!   qal run <file> [--allow <spec>] [--] [args..]    -> {"exit_code":n,"stdout":".."} | {"error":{...},"stdout":".."} | {"diagnostics":[...]}
//!
//! `<spec>` grants effects: `io`, `time`, `rand`, `fs:<dir>` (comma-separated, `--allow` repeatable).
//! Exit codes: 0 success; for `run` the program's exit code (1 if outside 0..=255);
//! 1 diagnostics / runtime error / failed tests; 2 usage or I/O error.

use qal_cli::{load, parse_args, take_options};
use qal_core::diag::Diagnostic;
use qal_run::{RunError, TestResult, Value};
use qal_wasm::Entry;
use serde::Serialize;
use std::process::ExitCode;

const USAGE: &str = "usage: qal <check|fmt|test> <file>\n       qal call <file> [--allow <spec>] <fn> [args...]\n       qal run <file> [--allow <spec>] [--] [args...]";

/// One JSON output line. Serialized directly (not via `serde_json::Value`) so record fields and
/// contract values keep their declaration order.
#[derive(Serialize, Default)]
struct Out<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    ok: Option<&'a Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exit_code: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a RunError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stdout: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diagnostics: Option<&'a [Diagnostic]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tests: Option<&'a [TestResult]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    passed: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    failed: Option<usize>,
}

fn usage() -> ExitCode {
    eprintln!("{USAGE}");
    ExitCode::from(2)
}

fn emit(out: Out, code: u8) -> ExitCode {
    println!("{}", serde_json::to_string(&out).expect("output serializes"));
    ExitCode::from(code)
}

fn main() -> ExitCode {
    // Arguments that are not valid UTF-8 are converted lossily instead of panicking.
    let args: Vec<String> = std::env::args_os().skip(1).map(|a| a.to_string_lossy().into_owned()).collect();
    let (Some(cmd), Some(file)) = (args.first(), args.get(1)) else {
        return usage();
    };
    if !["check", "fmt", "call", "test", "run"].contains(&cmd.as_str()) {
        return usage();
    }
    let src = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read {file}: {e}");
            return ExitCode::from(2);
        }
    };
    if cmd == "fmt" {
        return match qal_text::parse_program(&src) {
            Ok(p) => {
                print!("{}", qal_text::print_program(&p));
                ExitCode::SUCCESS
            }
            Err(d) => emit(Out { diagnostics: Some(&[d]), ..Out::default() }, 1),
        };
    }
    let (perms, rest) = match take_options(&args[2..]) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let loaded = match load(&src) {
        Ok(l) => l,
        Err(d) => return emit(Out { diagnostics: Some(&d), ..Out::default() }, 1),
    };
    match cmd.as_str() {
        "check" => emit(Out { ok: Some(&Value::Bool(true)), ..Out::default() }, 0),
        "test" => {
            let tests = loaded.runner(&Entry::Tests, perms).run_tests();
            let passed = tests.iter().filter(|t| t.passed).count();
            let failed = tests.len() - passed;
            emit(Out { tests: Some(&tests), passed: Some(passed), failed: Some(failed), ..Out::default() }, u8::from(failed > 0))
        }
        "call" => {
            let Some(name) = rest.first() else {
                return usage();
            };
            let vals = match parse_args(&loaded.typed, name, &rest[1..]) {
                Ok(v) => v,
                Err(e) => return emit(Out { error: Some(&e), ..Out::default() }, 1),
            };
            let out = loaded.runner(&Entry::Fn(name.clone()), perms).invoke(name, &vals);
            match &out.result {
                Ok(v) => emit(Out { ok: Some(v), stdout: Some(&out.stdout), ..Out::default() }, 0),
                Err(e) => emit(Out { error: Some(e), stdout: Some(&out.stdout), ..Out::default() }, 1),
            }
        }
        "run" => {
            if loaded.typed.fn_index("main").is_none() {
                let e = RunError::new("unknown_function", "main", "no entry point `main`");
                return emit(Out { error: Some(&e), ..Out::default() }, 1);
            }
            let out = loaded.runner(&Entry::Fn("main".into()), perms).run_main(&rest);
            match &out.result {
                Ok(Value::Int(code)) => {
                    emit(Out { exit_code: Some(*code), stdout: Some(&out.stdout), ..Out::default() }, u8::try_from(*code).unwrap_or(1))
                }
                Ok(other) => unreachable!("main is checked to return Int, got {other:?}"),
                Err(e) => emit(Out { error: Some(e), stdout: Some(&out.stdout), ..Out::default() }, 1),
            }
        }
        _ => usage(),
    }
}
```

- [ ] **Step 5: Run the whole workspace**

Run: `cd qal && cargo test --workspace 2>&1 | grep "test result" && cargo build --workspace --all-targets 2>&1 | grep -c warning`
Expected: every line `test result: ok.` — 121 tests in total (qal-core 5, qal-text 29, qal-check 28, qal-wasm 16, qal-run 26, qal-cli 17); warning count `0`.

- [ ] **Step 6: Smoke-test by hand**

Run: `cd qal && cargo run -q -p qal-cli -- run examples/age.qal --allow io " 42"`
Expected output: `{"exit_code":0,"stdout":"ok 42\n"}` (process exit code 0)

Run: `cd qal && cargo run -q -p qal-cli -- run examples/age.qal " 42"`
Expected: `{"error":{"code":"E0508",...,"kind":"permission_denied",...},"stdout":""}` (exit code 1)

Run: `cd qal && cargo run -q -p qal-cli -- run examples/wordcount.qal --allow io,fs:examples examples/age.qal`
Expected output: `{"exit_code":0,"stdout":"20 118 712\n"}`

Run: `cd qal && cargo run -q -p qal-cli -- call examples/math.qal fact -1`
Expected: an `error` with `"kind":"contract_violation"`, `"contract":"req[0]: n >= 0"`, `"values":{"n":-1}`, `"path":"req[0]"`.

- [ ] **Step 7: Commit**

```bash
git add qal/Cargo.lock qal/crates/qal-cli qal/examples
git commit -m "feat(qal): qal run with --allow, Str/Float CLI arguments, spec §4 example

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Done when

- `cd qal && cargo test --workspace` passes (121 tests) with zero warnings.
- `qal run examples/age.qal --allow io " 42"` prints `{"exit_code":0,"stdout":"ok 42\n"}` — the spec §4 example runs unchanged.
- A pure `main` compiles to a module whose only import is `qal_host.fail` (Task 4 `only_needed_host_functions_are_imported`, Task 5 `pure_programs_import_no_effects`), and a `main` with ungranted effects is refused (`permission_denied`).
- The Review Focus tests above exist and pass.
