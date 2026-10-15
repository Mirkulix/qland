# QAL Phase 1a — Core Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working QAL toolchain for the pure Int/Bool subset: text form → parse → check (types, effects, contracts) → WebAssembly → execution with structured JSON results, driven by a `qal` CLI.

**Architecture:** A new Cargo workspace `qland/qal/` with six small crates in a strict dependency chain: `qal-core` (AST, diagnostics) ← `qal-text` (lexer, parser, canonical printer) ← `qal-check` (type/effect/contract checking) ← `qal-wasm` (direct WASM codegen via `wasm-encoder`) ← `qal-run` (wasmtime host, failure reporting, fuel limit) ← `qal-cli` (pipeline + binary). Every compiler and runtime error is a JSON-serializable struct.

**Tech Stack:** Rust 1.97 (edition 2021), `wasm-encoder =0.259.0`, `wasmparser =0.259.0` (tests only), `wasmtime =49.0.1` (cranelift, tail calls, fuel), `serde`/`serde_json` 1.

**Spec:** `docs/superpowers/specs/2026-10-15-qal-design.md`

**Sub-plan context:** Phase 1 of the spec is delivered as five sub-plans, each leaving working software: **1a core pipeline (this plan)**, 1b data types (Float, Str, List, Option/Result, Records/Enums, sequences `{ a; b }`, built-ins, `match`, memory) + effects with capability-gated host functions, 1c canonical form + hashing + SQLite store, 1d MCP server + full CLI operations, 1e benchmark. Each later sub-plan is written after the previous one lands, against the real code.

## Global Constraints

- Workspace lives at `qland/qal/` (separate from the LLVM-heavy `qlang` workspace).
- Exact dependency versions: `wasm-encoder = "=0.259.0"`, `wasmparser = "=0.259.0"`, `wasmtime = { version = "=49.0.1", default-features = false, features = ["cranelift", "runtime", "std"] }`, `serde = { version = "1", features = ["derive"] }`, `serde_json = "1"`. No other dependencies.
- Code generation emits WebAssembly directly; never generate source code of another language.
- `Int` is 64-bit; overflow, `MIN / -1` and `-MIN` are errors (`int_overflow`), never silent wrap-around. Division/remainder by zero is `div_by_zero`.
- Effects known in Phase 1: `io`, `fs`, `time`, `rand`. Rule: effects(callee) ⊆ effects(caller). Contracts and tests are effect-free.
- Inside `ens`, the function result is named `r` (`qal_core::ast::RESULT_VAR`).
- Diagnostic JSON fields: `code, kind, def, path, message, expected?, found?, hints`. Diagnostic and runtime messages are written in English (the agents' strongest language). `path` addresses AST nodes (e.g. `body.then.args[0]`); parse errors use `line:col`.
- Checked programs only: nothing is compiled while any diagnostic exists.
- Builds and tests must be warning-free.

## Review Focus

1. **Integer boundaries** (`i64::MAX + 1`, `MIN * -1`, `-1 * MIN`, `MIN / -1`, `-MIN`, `MIN % -1`): must report `int_overflow` (or `0` for `MIN % -1`), never wrap or crash the host — pinned in Task 6 `integer_overflow_is_an_error_not_a_wrap`.
2. **Runaway programs** (infinite recursion, very deep non-tail recursion): must return `timeout` / `stack_overflow` as data, not hang or abort — pinned in Task 7 `infinite_loop_times_out`, `deep_non_tail_recursion_reports_stack_overflow`.
3. **Deep tail recursion** (the only loop construct in 1a): one million iterations must complete — pinned in Task 6 `deep_tail_recursion_does_not_overflow_the_stack`.
4. **Contracts on recursive functions**: `ens` must be checked per call against that call's parameters (so tail calls are disabled in functions with `ens`) — pinned in Task 6 `ens_in_recursive_function_checks_each_call`.
5. **Malformed input** (empty file, non-ASCII identifiers, oversized integer literals, unclosed parentheses, keywords as names, chained comparisons): must yield a `parse` diagnostic with position, never a panic — pinned in Task 2 `rejects_oversized_integer_and_unknown_characters` and Task 3 `reports_errors_with_position`, `empty_source_is_empty_program`.

## File Structure

| File | Responsibility |
|---|---|
| `qal/Cargo.toml` | Workspace, pinned dependency versions |
| `qal/.gitignore` | Ignore `target/` |
| `qal/crates/qal-core/src/ast.rs` | Types, expressions, definitions, `RESULT_VAR` |
| `qal/crates/qal-core/src/diag.rs` | `Diagnostic` struct + constructors per error kind |
| `qal/crates/qal-text/src/lexer.rs` | Source → tokens with positions |
| `qal/crates/qal-text/src/parser.rs` | Tokens → `Program` (grammar in module doc) |
| `qal/crates/qal-text/src/printer.rs` | `Program` → canonical text (round-trip safe) |
| `qal/crates/qal-check/src/lib.rs` | Names, types, effects, contracts |
| `qal/crates/qal-wasm/src/lib.rs` | `Program` → WASM bytes + per-function metadata |
| `qal/crates/qal-wasm/tests/codegen.rs` | End-to-end execution tests with a minimal wasmtime harness |
| `qal/crates/qal-run/src/lib.rs` | `Runner`: typed calls, failure → `RunError`, fuel, tests |
| `qal/crates/qal-cli/src/lib.rs` | `load` (whole pipeline), CLI argument conversion |
| `qal/crates/qal-cli/src/main.rs` | `qal check / fmt / call / test` |
| `qal/crates/qal-cli/tests/cli.rs` | Binary-level tests on JSON output |
| `qal/examples/math.qal` | Example program used by CLI tests |

Note vs. spec §10: the checker lives in its own crate `qal-check` (not inside `qal-core`) so its tests can use the parser without a dependency cycle.

---

### Task 1: Workspace, AST and diagnostics (`qal-core`)

**Files:**
- Create: `qal/Cargo.toml`, `qal/.gitignore`
- Create: `qal/crates/qal-core/Cargo.toml`, `qal/crates/qal-core/src/lib.rs`, `qal/crates/qal-core/src/ast.rs`, `qal/crates/qal-core/src/diag.rs`
- Test: inline module in `diag.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: `qal_core::ast::{Type, BinOp, UnOp, Expr, Param, FnDef, TestDef, Def, Program, RESULT_VAR}` with `Type::name()`, `BinOp::symbol()`, `BinOp::precedence()` (Or 1, And 2, comparisons 3, Add/Sub 4, Mul/Div/Rem 5), `Def::name()`; `qal_core::diag::Diagnostic` (fields `code: &'static str, kind: &'static str, def: String, path: String, message: String, expected: Option<String>, found: Option<String>, hints: Vec<String>`) with constructors `new, parse(line, col, msg), unknown_name(def, path, name), duplicate_name(def, path, name), type_mismatch(def, path, expected, found), arity(def, path, callee, expected, found), effect_violation(def, path, callee, missing: &[String])` and builders `expected_found`, `hint`.

- [ ] **Step 1: Create the workspace files**

`qal/Cargo.toml`:

```toml
[workspace]
members = [
    "crates/qal-core",
]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "MIT"

[workspace.dependencies]
qal-core = { path = "crates/qal-core" }
qal-text = { path = "crates/qal-text" }
qal-check = { path = "crates/qal-check" }
qal-wasm = { path = "crates/qal-wasm" }
qal-run = { path = "crates/qal-run" }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
wasm-encoder = "=0.259.0"
wasmparser = "=0.259.0"
wasmtime = { version = "=49.0.1", default-features = false, features = ["cranelift", "runtime", "std"] }
```

`qal/.gitignore`:

```text
target/
```

`qal/crates/qal-core/Cargo.toml`:

```toml
[package]
name = "qal-core"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
serde = { workspace = true }
serde_json = { workspace = true }
```

`qal/crates/qal-core/src/lib.rs`:

```rust
//! QAL core: the program graph (AST) and structured diagnostics.

pub mod ast;
pub mod diag;
```

`qal/crates/qal-core/src/ast.rs` (plain data types, no logic to test beyond what later tasks exercise):

```rust
//! Abstract syntax of QAL (Phase 1a subset).

/// Name of the result variable inside `ens` contracts.
pub const RESULT_VAR: &str = "r";

/// Value types available in Phase 1a.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Int,
    Bool,
    Unit,
}

impl Type {
    pub fn name(self) -> &'static str {
        match self {
            Type::Int => "Int",
            Type::Bool => "Bool",
            Type::Unit => "Unit",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
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
            BinOp::Add | BinOp::Sub => 4,
            BinOp::Mul | BinOp::Div | BinOp::Rem => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Int(i64),
    Bool(bool),
    Unit,
    Var(String),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Un(UnOp, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    /// `let name = value in body`
    Let(String, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
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
pub enum Def {
    Fn(FnDef),
    Test(TestDef),
}

impl Def {
    pub fn name(&self) -> &str {
        match self {
            Def::Fn(f) => &f.name,
            Def::Test(t) => &t.name,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Program {
    pub defs: Vec<Def>,
}
```

- [ ] **Step 2: Write the failing test**

Create `qal/crates/qal-core/src/diag.rs` with only the test module:

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
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd qal && cargo test -p qal-core`
Expected: compile error — `cannot find type Diagnostic` / `use of undeclared type Diagnostic`.

- [ ] **Step 4: Write the implementation**

Insert above the test module in `diag.rs`:

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
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd qal && cargo test -p qal-core`
Expected: `serializes_without_empty_optionals ... ok`, no warnings.

- [ ] **Step 6: Commit**

```bash
git add qal/Cargo.toml qal/.gitignore qal/crates/qal-core
git commit -m "feat(qal): workspace, AST and structured diagnostics

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2: Lexer (`qal-text`)

**Files:**
- Create: `qal/crates/qal-text/Cargo.toml`, `qal/crates/qal-text/src/lib.rs`, `qal/crates/qal-text/src/lexer.rs`
- Test: inline module in `lexer.rs`

**Interfaces:**
- Consumes: `qal_core::diag::Diagnostic::parse`.
- Produces: `qal_text::lexer::{Tok, Token, KEYWORDS, lex}`; `lex(&str) -> Result<Vec<Token>, Diagnostic>`; `Tok::{Ident(String), Int(i64), Sym(&'static str), Eof}`; `Token { tok, line, col }` (1-based). Keywords are lexed as `Ident` and recognised by the parser. `#` starts a line comment.

- [ ] **Step 0: Register the crate in the workspace**

In `qal/Cargo.toml`, add `"crates/qal-text",` to `members` so it reads:

```toml
members = [
    "crates/qal-core",
    "crates/qal-text",
]
```

- [ ] **Step 0b: Create the crate skeleton**

`qal/crates/qal-text/Cargo.toml`:

```toml
[package]
name = "qal-text"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
qal-core = { workspace = true }
```

`qal/crates/qal-text/src/lib.rs` (grows in Tasks 3 and 4):

```rust
//! QAL text form: a token-dense view of the program graph for LLM agents.

pub mod lexer;
```

- [ ] **Step 1: Write the failing tests**

Create `qal/crates/qal-text/src/lexer.rs` containing only the test module for now:

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
    fn rejects_oversized_integer_and_unknown_characters() {
        let e = lex("99999999999999999999").unwrap_err();
        assert_eq!(e.kind, "parse");
        assert_eq!(e.path, "1:1");
        let e = lex("x @ y").unwrap_err();
        assert_eq!(e.path, "1:3");
        let e = lex("größe").unwrap_err();
        assert_eq!(e.kind, "parse");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd qal && cargo test -p qal-text lexer`
Expected: compile error — `cannot find function lex` / `cannot find type Tok`

- [ ] **Step 3: Write the implementation**

Insert this above the test module in `qal/crates/qal-text/src/lexer.rs` (the file then is implementation followed by tests):

```rust
//! Lexer: source text -> tokens with line/column positions.

use qal_core::diag::Diagnostic;

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Ident(String),
    Int(i64),
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

pub const KEYWORDS: &[&str] = &["fn", "test", "let", "in", "if", "then", "else", "req", "ens", "true", "false"];

/// Longest symbols first so that "->" wins over "-".
const SYMBOLS: &[&str] = &[
    "->", "==", "!=", "<=", ">=", "&&", "||", "(", ")", ",", ":", "=", "<", ">", "+", "-", "*", "/", "%", "!",
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
            }
            continue;
        }
        let (start_line, start_col) = (line, col);
        if c.is_ascii_digit() {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            let text: String = chars[start..i].iter().collect();
            col += i - start;
            let value = text
                .parse::<i64>()
                .map_err(|_| Diagnostic::parse(start_line, start_col, format!("integer literal `{text}` does not fit in Int (64-bit)")))?;
            out.push(Token { tok: Tok::Int(value), line: start_line, col: start_col });
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd qal && cargo test -p qal-text`
Expected: all tests PASS, no warnings.

- [ ] **Step 5: Commit**

```bash
git add qal/Cargo.toml qal/crates/qal-text
git commit -m "feat(qal): lexer with positions and parse diagnostics

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 3: Parser (`qal-text`)

**Files:**
- Modify: `qal/crates/qal-text/src/lib.rs`
- Create: `qal/crates/qal-text/src/parser.rs`
- Test: inline module in `parser.rs`

**Interfaces:**
- Consumes: `lexer::{lex, Tok, Token, KEYWORDS}`; all `qal_core::ast` types; `Diagnostic::parse`.
- Produces: `qal_text::parse_program(&str) -> Result<Program, Diagnostic>`. Syntax: `fn name(p: T, ...) -> T !eff* (req e | ens e)* = expr`, `test name = expr`, `let x = e in e`, `if c then a else b`, `f(args)`, `()`, unary `-`/`!`; comparisons are non-associative.

- [ ] **Step 0: Export the parser**

Replace `qal/crates/qal-text/src/lib.rs` with:

```rust
//! QAL text form: a token-dense view of the program graph for LLM agents.

pub mod lexer;
pub mod parser;

pub use parser::parse_program;
```

- [ ] **Step 1: Write the failing tests**

Create `qal/crates/qal-text/src/parser.rs` containing only the test module for now:

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
        let p = parse_program("test t = 1 + 2 * 3 - 4 == 3 || false").unwrap();
        let Def::Test(t) = &p.defs[0] else { panic!() };
        let sum = Expr::Bin(BinOp::Sub, Box::new(Expr::Bin(BinOp::Add, int(1), Box::new(Expr::Bin(BinOp::Mul, int(2), int(3))))), int(4));
        let cmp = Expr::Bin(BinOp::Eq, Box::new(sum), int(3));
        assert_eq!(t.body, Expr::Bin(BinOp::Or, Box::new(cmp), Box::new(Expr::Bool(false))));
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
        assert_eq!(t.body, Expr::Let("y".into(), Box::new(neg), Box::new(iff)));
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
        let e = parse_program("fn f() -> Float = 1").unwrap_err();
        assert!(e.message.contains("expected a type"), "{}", e.message);
    }

    #[test]
    fn empty_source_is_empty_program() {
        assert_eq!(parse_program("  # nothing\n").unwrap(), Program::default());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd qal && cargo test -p qal-text parser`
Expected: compile error — `cannot find function parse_program`

- [ ] **Step 3: Write the implementation**

Insert this above the test module in `qal/crates/qal-text/src/parser.rs` (the file then is implementation followed by tests):

```rust
//! Recursive-descent parser: tokens -> Program.
//!
//! Grammar (Phase 1a):
//!   program  := def*
//!   def      := "fn" IDENT "(" params ")" "->" type effect* contract* "=" expr
//!             | "test" IDENT "=" expr
//!   params   := (IDENT ":" type ("," IDENT ":" type)* ","?)?
//!   type     := "Int" | "Bool" | "Unit"
//!   effect   := "!" IDENT
//!   contract := ("req" | "ens") expr
//!   expr     := "let" IDENT "=" expr "in" expr
//!             | "if" expr "then" expr "else" expr
//!             | binary
//!   binary   := precedence climbing over BinOp::precedence (comparisons non-associative)
//!   unary    := ("-" | "!") unary | atom
//!   atom     := INT | "true" | "false" | "(" ")" | "(" expr ")" | IDENT | IDENT "(" args ")"

use crate::lexer::{lex, Tok, Token, KEYWORDS};
use qal_core::ast::*;
use qal_core::diag::Diagnostic;

pub fn parse_program(src: &str) -> Result<Program, Diagnostic> {
    let tokens = lex(src)?;
    let mut p = Parser { tokens, pos: 0 };
    let mut defs = Vec::new();
    while p.peek() != &Tok::Eof {
        defs.push(p.def()?);
    }
    Ok(Program { defs })
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.tokens[self.pos].tok
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

    fn describe(tok: &Tok) -> String {
        match tok {
            Tok::Ident(s) => format!("`{s}`"),
            Tok::Int(n) => format!("`{n}`"),
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

    fn ty(&mut self) -> Result<Type, Diagnostic> {
        let t = match self.peek() {
            Tok::Ident(s) if s == "Int" => Type::Int,
            Tok::Ident(s) if s == "Bool" => Type::Bool,
            Tok::Ident(s) if s == "Unit" => Type::Unit,
            other => return Err(self.err_here(format!("expected a type (Int, Bool, Unit), found {}", Self::describe(other)))),
        };
        self.next();
        Ok(t)
    }

    fn def(&mut self) -> Result<Def, Diagnostic> {
        if self.is_kw("fn") {
            self.next();
            self.fn_def().map(Def::Fn)
        } else if self.is_kw("test") {
            self.next();
            let name = self.ident()?;
            self.expect_sym("=")?;
            let body = self.expr()?;
            Ok(Def::Test(TestDef { name, body }))
        } else {
            Err(self.err_here(format!("expected `fn` or `test`, found {}", Self::describe(self.peek()))))
        }
    }

    fn fn_def(&mut self) -> Result<FnDef, Diagnostic> {
        let name = self.ident()?;
        self.expect_sym("(")?;
        let mut params = Vec::new();
        while !self.is_sym(")") {
            let pname = self.ident()?;
            self.expect_sym(":")?;
            let ty = self.ty()?;
            params.push(Param { name: pname, ty });
            if !self.is_sym(")") {
                self.expect_sym(",")?;
            }
        }
        self.expect_sym(")")?;
        self.expect_sym("->")?;
        let ret = self.ty()?;
        let mut effects = Vec::new();
        while self.is_sym("!") {
            self.next();
            effects.push(self.ident()?);
        }
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
        if self.is_kw("let") {
            self.next();
            let name = self.ident()?;
            self.expect_sym("=")?;
            let value = self.expr()?;
            self.expect_kw("in")?;
            let body = self.expr()?;
            return Ok(Expr::Let(name, Box::new(value), Box::new(body)));
        }
        if self.is_kw("if") {
            self.next();
            let c = self.expr()?;
            self.expect_kw("then")?;
            let t = self.expr()?;
            self.expect_kw("else")?;
            let e = self.expr()?;
            return Ok(Expr::If(Box::new(c), Box::new(t), Box::new(e)));
        }
        self.binary(1)
    }

    fn peek_binop(&self) -> Option<BinOp> {
        let op = match self.peek() {
            Tok::Sym("+") => BinOp::Add,
            Tok::Sym("-") => BinOp::Sub,
            Tok::Sym("*") => BinOp::Mul,
            Tok::Sym("/") => BinOp::Div,
            Tok::Sym("%") => BinOp::Rem,
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
            let rhs = self.binary(prec + 1)?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
            if prec == 3 {
                if let Some(next) = self.peek_binop() {
                    if next.precedence() == 3 {
                        return Err(self.err_here("comparisons cannot be chained; use `&&`"));
                    }
                }
            }
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<Expr, Diagnostic> {
        if self.is_sym("-") {
            self.next();
            return Ok(Expr::Un(UnOp::Neg, Box::new(self.unary()?)));
        }
        if self.is_sym("!") {
            self.next();
            return Ok(Expr::Un(UnOp::Not, Box::new(self.unary()?)));
        }
        self.atom()
    }

    fn atom(&mut self) -> Result<Expr, Diagnostic> {
        match self.peek().clone() {
            Tok::Int(n) => {
                self.next();
                Ok(Expr::Int(n))
            }
            Tok::Ident(s) if s == "true" => {
                self.next();
                Ok(Expr::Bool(true))
            }
            Tok::Ident(s) if s == "false" => {
                self.next();
                Ok(Expr::Bool(false))
            }
            Tok::Sym("(") => {
                self.next();
                if self.is_sym(")") {
                    self.next();
                    return Ok(Expr::Unit);
                }
                let e = self.expr()?;
                self.expect_sym(")")?;
                Ok(e)
            }
            Tok::Ident(_) => {
                let name = self.ident()?;
                if self.is_sym("(") {
                    self.next();
                    let mut args = Vec::new();
                    while !self.is_sym(")") {
                        args.push(self.expr()?);
                        if !self.is_sym(")") {
                            self.expect_sym(",")?;
                        }
                    }
                    self.expect_sym(")")?;
                    Ok(Expr::Call(name, args))
                } else {
                    Ok(Expr::Var(name))
                }
            }
            other => Err(self.err_here(format!("expected an expression, found {}", Self::describe(&other)))),
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd qal && cargo test -p qal-text`
Expected: all tests PASS, no warnings.

- [ ] **Step 5: Commit**

```bash
git add qal/crates/qal-text
git commit -m "feat(qal): recursive-descent parser for the 1a subset

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 4: Canonical printer (`qal-text`)

**Files:**
- Modify: `qal/crates/qal-text/src/lib.rs`
- Create: `qal/crates/qal-text/src/printer.rs`
- Test: inline module in `printer.rs`

**Interfaces:**
- Consumes: `qal_core::ast`; `parse_program` (tests).
- Produces: `qal_text::print_program(&Program) -> String` (defs separated by a blank line, trailing newline; contracts on their own lines indented two spaces, body on a line starting `= `) and `qal_text::print_expr(&Expr) -> String` (minimal parentheses). Guarantee: `parse_program(&print_program(&p)) == Ok(p)`.

- [ ] **Step 0: Export the printer**

Replace `qal/crates/qal-text/src/lib.rs` with its final form:

```rust
//! QAL text form: a token-dense view of the program graph for LLM agents.

pub mod lexer;
pub mod parser;
pub mod printer;

pub use parser::parse_program;
pub use printer::{print_expr, print_program};
```

- [ ] **Step 1: Write the failing tests**

Create `qal/crates/qal-text/src/printer.rs` containing only the test module for now:

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
    fn roundtrips_all_constructs() {
        roundtrip("fn f(x: Int) -> Int req x > 0 ens r >= 1 = if x == 1 then 1 else x * f(x - 1)");
        roundtrip("test t = let a = (if true then 1 else 2) in a + (let b = 3 in b)");
        roundtrip("test t = -(-1) == 1 && !(1 < 2) || (1 == 1) == true");
        roundtrip("fn u() -> Unit !io !fs = ()");
        roundtrip("test t = 1 - (2 - 3) - 4 * (5 % 2) / 1");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd qal && cargo test -p qal-text printer`
Expected: compile error — `cannot find function print_program`

- [ ] **Step 3: Write the implementation**

Insert this above the test module in `qal/crates/qal-text/src/printer.rs` (the file then is implementation followed by tests):

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
        Def::Fn(f) => {
            let params: Vec<String> = f.params.iter().map(|p| format!("{}: {}", p.name, p.ty.name())).collect();
            let mut s = format!("fn {}({}) -> {}", f.name, params.join(", "), f.ret.name());
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
        Expr::Let(..) | Expr::If(..) => 0,
        Expr::Bin(op, ..) => op.precedence(),
        Expr::Un(..) => 6,
        _ => 7,
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

pub fn print_expr(e: &Expr) -> String {
    match e {
        Expr::Int(n) => n.to_string(),
        Expr::Bool(b) => b.to_string(),
        Expr::Unit => "()".to_string(),
        Expr::Var(v) => v.clone(),
        Expr::Call(f, args) => {
            let a: Vec<String> = args.iter().map(print_expr).collect();
            format!("{f}({})", a.join(", "))
        }
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
            // Comparisons (3) are non-associative: neither side may share it.
            let left_parens = prec(l) < p || (p == 3 && prec(l) == 3);
            let right_parens = prec(r) <= p;
            format!("{} {} {}", wrap(l, left_parens), op.symbol(), wrap(r, right_parens))
        }
        Expr::If(c, t, f) => format!("if {} then {} else {}", print_expr(c), print_expr(t), print_expr(f)),
        Expr::Let(n, v, b) => format!("let {n} = {} in {}", print_expr(v), print_expr(b)),
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd qal && cargo test -p qal-text`
Expected: all tests PASS, no warnings.

- [ ] **Step 5: Commit**

```bash
git add qal/crates/qal-text
git commit -m "feat(qal): canonical printer with round-trip guarantee

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 5: Type, effect and contract checker (`qal-check`)

**Files:**
- Create: `qal/crates/qal-check/Cargo.toml`, `qal/crates/qal-check/src/lib.rs`
- Test: inline module in `lib.rs`

**Interfaces:**
- Consumes: `qal_core::ast`, `qal_core::diag::Diagnostic` constructors; `qal_text::parse_program` (dev-dependency, tests only).
- Produces: `qal_check::check_program(&Program) -> Vec<Diagnostic>` (empty = compilable), `qal_check::KNOWN_EFFECTS`, `qal_check::Sig`, re-export `qal_check::RESULT_VAR`. Paths used in diagnostics: `body`, `req[i]`, `ens[i]`, `params[i]`, `effects[i]`, and suffixes `.cond .then .else .value .body .left .right .operand .args[i]`.

- [ ] **Step 0: Register the crate in the workspace**

In `qal/Cargo.toml`, add `"crates/qal-check",` to `members` so it reads:

```toml
members = [
    "crates/qal-core",
    "crates/qal-text",
    "crates/qal-check",
]
```

`qal/crates/qal-check/Cargo.toml`:

```toml
[package]
name = "qal-check"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
qal-core = { workspace = true }

[dev-dependencies]
qal-text = { workspace = true }
```

- [ ] **Step 1: Write the failing tests**

Create `qal/crates/qal-check/src/lib.rs` with only the test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use qal_text::parse_program;

    fn check(src: &str) -> Vec<Diagnostic> {
        check_program(&parse_program(src).unwrap())
    }

    fn kinds(src: &str) -> Vec<(&'static str, String, String)> {
        check(src).into_iter().map(|d| (d.kind, d.def, d.path)).collect()
    }

    #[test]
    fn accepts_well_typed_program() {
        let src = "fn fact(n: Int) -> Int req n >= 0 ens r >= 1 = if n <= 1 then 1 else n * fact(n - 1)\n\
                   test fact_5 = fact(5) == 120\n\
                   fn noop() -> Unit = ()\n\
                   test unit_eq = noop() == ()";
        assert_eq!(check(src), vec![]);
    }

    #[test]
    fn reports_type_mismatch_with_path() {
        let d = check("fn f(x: Int) -> Int = if x then 1 else 2");
        assert_eq!(d.len(), 1);
        assert_eq!((d[0].kind, d[0].path.as_str()), ("type_mismatch", "body.cond"));
        assert_eq!((d[0].expected.as_deref(), d[0].found.as_deref()), (Some("Bool"), Some("Int")));
        assert_eq!(d[0].hints, vec!["compare explicitly, e.g. `x != 0`".to_string()]);
    }

    #[test]
    fn reports_unknown_names_arity_and_duplicates() {
        assert_eq!(kinds("fn f() -> Int = y"), vec![("unknown_name", "f".into(), "body".into())]);
        assert_eq!(kinds("fn f() -> Int = g()"), vec![("unknown_name", "f".into(), "body".into())]);
        let d = check("fn g(a: Int) -> Int = a\nfn f() -> Int = g(1, 2)");
        assert_eq!((d[0].code, d[0].kind), ("E0204", "type_mismatch"));
        assert_eq!(kinds("fn f() -> Int = 1\nfn f() -> Int = 2")[0].0, "duplicate_name");
        assert_eq!(kinds("fn f(a: Int, a: Int) -> Int = a")[0], ("duplicate_name", "f".into(), "params[1]".into()));
    }

    #[test]
    fn let_scoping_and_shadowing() {
        assert_eq!(check("fn f() -> Bool = let x = 1 in let x = true in x"), vec![]);
        assert_eq!(kinds("fn f() -> Int = (let x = 1 in x) + x"), vec![("unknown_name", "f".into(), "body.right".into())]);
    }

    #[test]
    fn contracts_must_be_bool_and_see_result() {
        assert_eq!(kinds("fn f(x: Int) -> Int req x = x"), vec![("type_mismatch", "f".into(), "req[0]".into())]);
        assert_eq!(kinds("fn f(x: Int) -> Int req r > 0 = x"), vec![("unknown_name", "f".into(), "req[0].left".into())]);
        assert_eq!(check("fn f(x: Int) -> Bool ens r == (x > 0) = x > 0"), vec![]);
    }

    #[test]
    fn effects_must_be_declared_and_known() {
        let src = "fn say() -> Unit !io = ()\nfn pure_caller() -> Unit = say()\nfn ok() -> Unit !io = say()";
        let d = check(src);
        assert_eq!(d.len(), 1);
        assert_eq!((d[0].kind, d[0].def.as_str(), d[0].code), ("effect_violation", "pure_caller", "E0301"));
        assert_eq!(d[0].hints, vec!["add `!io` to the signature of `pure_caller`".to_string()]);
        assert_eq!(kinds("fn f() -> Unit !net = ()")[0], ("unknown_name", "f".into(), "effects[0]".into()));
        assert_eq!(kinds("fn say() -> Unit !io = ()\ntest t = say() == ()")[0].0, "effect_violation");
        assert_eq!(kinds("fn say() -> Unit !io = ()\nfn f() -> Int !io req say() == () = 1")[0].0, "effect_violation");
    }

    #[test]
    fn tests_must_be_bool() {
        assert_eq!(kinds("test t = 1"), vec![("type_mismatch", "t".into(), "body".into())]);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd qal && cargo test -p qal-check`
Expected: compile error — `cannot find function check_program`.

- [ ] **Step 3: Write the implementation**

Insert above the test module:

```rust
//! QAL checker: names, types, effects and contracts. Only checked programs may be compiled.

use qal_core::ast::*;
use qal_core::diag::Diagnostic;
use std::collections::{HashMap, HashSet};

/// Effects known in Phase 1.
pub const KNOWN_EFFECTS: &[&str] = &["io", "fs", "time", "rand"];

pub use qal_core::ast::RESULT_VAR;

#[derive(Debug, Clone, PartialEq)]
pub struct Sig {
    pub params: Vec<Type>,
    pub ret: Type,
    pub effects: Vec<String>,
}

/// Checks a whole program. An empty result means the program may be compiled.
pub fn check_program(p: &Program) -> Vec<Diagnostic> {
    let mut diags = Vec::new();
    let mut sigs: HashMap<String, Sig> = HashMap::new();
    let mut seen = HashSet::new();
    for d in &p.defs {
        if !seen.insert(d.name().to_string()) {
            diags.push(Diagnostic::duplicate_name(d.name(), "", d.name()));
        }
        if let Def::Fn(f) = d {
            sigs.entry(f.name.clone()).or_insert_with(|| Sig {
                params: f.params.iter().map(|p| p.ty).collect(),
                ret: f.ret,
                effects: f.effects.clone(),
            });
        }
    }
    for d in &p.defs {
        match d {
            Def::Fn(f) => check_fn(f, &sigs, &mut diags),
            Def::Test(t) => {
                let mut cx = Cx { def: &t.name, sigs: &sigs, effects: &[], diags: &mut diags };
                let mut env = Vec::new();
                cx.expect(&t.body, Type::Bool, &mut env, "body");
            }
        }
    }
    diags
}

fn check_fn(f: &FnDef, sigs: &HashMap<String, Sig>, diags: &mut Vec<Diagnostic>) {
    let mut names = HashSet::new();
    for (i, p) in f.params.iter().enumerate() {
        if !names.insert(p.name.as_str()) {
            diags.push(Diagnostic::duplicate_name(&f.name, &format!("params[{i}]"), &p.name));
        }
    }
    for (i, e) in f.effects.iter().enumerate() {
        if !KNOWN_EFFECTS.contains(&e.as_str()) {
            diags.push(
                Diagnostic::unknown_name(&f.name, &format!("effects[{i}]"), e)
                    .hint(format!("known effects: {}", KNOWN_EFFECTS.join(", "))),
            );
        }
    }
    let params: Vec<(String, Type)> = f.params.iter().map(|p| (p.name.clone(), p.ty)).collect();
    let mut cx = Cx { def: &f.name, sigs, effects: &[], diags };
    for (i, c) in f.req.iter().enumerate() {
        let mut env = params.clone();
        cx.expect(c, Type::Bool, &mut env, &format!("req[{i}]"));
    }
    for (i, c) in f.ens.iter().enumerate() {
        let mut env = params.clone();
        env.push((RESULT_VAR.to_string(), f.ret));
        cx.expect(c, Type::Bool, &mut env, &format!("ens[{i}]"));
    }
    cx.effects = &f.effects;
    let mut env = params;
    cx.expect(&f.body, f.ret, &mut env, "body");
}

struct Cx<'a> {
    def: &'a str,
    sigs: &'a HashMap<String, Sig>,
    /// Effects the current expression may use (empty for contracts and tests).
    effects: &'a [String],
    diags: &'a mut Vec<Diagnostic>,
}

impl Cx<'_> {
    fn expect(&mut self, e: &Expr, want: Type, env: &mut Vec<(String, Type)>, path: &str) {
        if let Some(got) = self.infer(e, env, path) {
            if got != want {
                let mut d = Diagnostic::type_mismatch(self.def, path, want.name(), got.name());
                if want == Type::Bool && got == Type::Int {
                    d = d.hint("compare explicitly, e.g. `x != 0`");
                }
                self.diags.push(d);
            }
        }
    }

    /// Returns None if an error was already reported for `e` (avoids cascades).
    fn infer(&mut self, e: &Expr, env: &mut Vec<(String, Type)>, path: &str) -> Option<Type> {
        match e {
            Expr::Int(_) => Some(Type::Int),
            Expr::Bool(_) => Some(Type::Bool),
            Expr::Unit => Some(Type::Unit),
            Expr::Var(v) => match env.iter().rev().find(|(n, _)| n == v) {
                Some((_, t)) => Some(*t),
                None => {
                    let mut d = Diagnostic::unknown_name(self.def, path, v);
                    if self.sigs.contains_key(v) {
                        d = d.hint(format!("`{v}` is a function; call it as `{v}(...)`"));
                    }
                    self.diags.push(d);
                    None
                }
            },
            Expr::Un(op, x) => {
                let t = match op {
                    UnOp::Neg => Type::Int,
                    UnOp::Not => Type::Bool,
                };
                self.expect(x, t, env, &format!("{path}.operand"));
                Some(t)
            }
            Expr::Bin(op, l, r) => {
                let (lp, rp) = (format!("{path}.left"), format!("{path}.right"));
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => {
                        self.expect(l, Type::Int, env, &lp);
                        self.expect(r, Type::Int, env, &rp);
                        Some(Type::Int)
                    }
                    BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        self.expect(l, Type::Int, env, &lp);
                        self.expect(r, Type::Int, env, &rp);
                        Some(Type::Bool)
                    }
                    BinOp::And | BinOp::Or => {
                        self.expect(l, Type::Bool, env, &lp);
                        self.expect(r, Type::Bool, env, &rp);
                        Some(Type::Bool)
                    }
                    BinOp::Eq | BinOp::Ne => {
                        if let Some(lt) = self.infer(l, env, &lp) {
                            self.expect(r, lt, env, &rp);
                        } else {
                            self.infer(r, env, &rp);
                        }
                        Some(Type::Bool)
                    }
                }
            }
            Expr::If(c, t, f) => {
                self.expect(c, Type::Bool, env, &format!("{path}.cond"));
                let tt = self.infer(t, env, &format!("{path}.then"));
                match tt {
                    Some(tt) => {
                        self.expect(f, tt, env, &format!("{path}.else"));
                        Some(tt)
                    }
                    None => self.infer(f, env, &format!("{path}.else")),
                }
            }
            Expr::Let(name, v, body) => {
                let vt = self.infer(v, env, &format!("{path}.value"))?;
                env.push((name.clone(), vt));
                let bt = self.infer(body, env, &format!("{path}.body"));
                env.pop();
                bt
            }
            Expr::Call(name, args) => {
                let Some(sig) = self.sigs.get(name) else {
                    self.diags.push(Diagnostic::unknown_name(self.def, path, name));
                    for (i, a) in args.iter().enumerate() {
                        self.infer(a, env, &format!("{path}.args[{i}]"));
                    }
                    return None;
                };
                if sig.params.len() != args.len() {
                    self.diags.push(Diagnostic::arity(self.def, path, name, sig.params.len(), args.len()));
                } else {
                    for (i, (a, t)) in args.iter().zip(&sig.params).enumerate() {
                        self.expect(a, *t, env, &format!("{path}.args[{i}]"));
                    }
                }
                let missing: Vec<String> = sig.effects.iter().filter(|e| !self.effects.contains(e)).cloned().collect();
                if !missing.is_empty() {
                    self.diags.push(Diagnostic::effect_violation(self.def, path, name, &missing));
                }
                Some(sig.ret)
            }
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd qal && cargo test -p qal-check`
Expected: 7 tests PASS, no warnings.

- [ ] **Step 5: Commit**

```bash
git add qal/Cargo.toml qal/crates/qal-check
git commit -m "feat(qal): checker for names, types, effects and contracts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 6: WebAssembly code generation (`qal-wasm`)

**Files:**
- Create: `qal/crates/qal-wasm/Cargo.toml`, `qal/crates/qal-wasm/src/lib.rs`
- Test: `qal/crates/qal-wasm/tests/codegen.rs`

**Interfaces:**
- Consumes: `qal_core::ast` (including `RESULT_VAR`); dev: `qal_text::parse_program`, `qal_check::check_program`, `wasmparser::validate`, `wasmtime`.
- Produces: `qal_wasm::compile(&Program) -> CompiledModule` (input must be checked); `CompiledModule { wasm: Vec<u8>, funcs: Vec<FuncMeta> }` with `func(&self, export: &str) -> Option<&FuncMeta>`; `FuncMeta { name, export, is_test, params: Vec<Type>, ret: Type, req: Vec<Expr>, ens: Vec<Expr> }` — fns first in program order, then tests (export `test:<name>`); host import `qal_host.fail(kind: i32, func: i32, index: i32)` where `func` indexes `funcs`; constants `FAIL_REQ = 1, FAIL_ENS = 2, FAIL_OVERFLOW = 3, FAIL_DIV_ZERO = 4`. Value mapping: Int → i64, Bool/Unit → i32. Tail calls use `return_call` (requires wasmtime `wasm_tail_call(true)`), except in functions that have `ens`.

- [ ] **Step 0: Register the crate in the workspace**

In `qal/Cargo.toml`, add `"crates/qal-wasm",` to `members` so it reads:

```toml
members = [
    "crates/qal-core",
    "crates/qal-text",
    "crates/qal-check",
    "crates/qal-wasm",
]
```

`qal/crates/qal-wasm/Cargo.toml`:

```toml
[package]
name = "qal-wasm"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
qal-core = { workspace = true }
wasm-encoder = { workspace = true }

[dev-dependencies]
qal-text = { workspace = true }
qal-check = { workspace = true }
wasmparser = { workspace = true }
wasmtime = { workspace = true }
```

- [ ] **Step 1: Write the failing tests**

Create `qal/crates/qal-wasm/tests/codegen.rs`:

```rust
//! End-to-end codegen tests: text -> check -> wasm -> execute in wasmtime.

use qal_wasm::{compile, CompiledModule, FAIL_DIV_ZERO, FAIL_ENS, FAIL_OVERFLOW, FAIL_REQ};
use wasmtime::{Caller, Config, Engine, Instance, Linker, Module, Store, Val};

fn build(src: &str) -> CompiledModule {
    let p = qal_text::parse_program(src).expect("parse");
    let diags = qal_check::check_program(&p);
    assert!(diags.is_empty(), "check failed: {diags:?}");
    let m = compile(&p);
    wasmparser::validate(&m.wasm).expect("valid wasm");
    m
}

/// Calls `export` with i64 args. Ok(i64 result, or i32 widened) or Err((kind, func, index)) from `fail`.
fn call(m: &CompiledModule, export: &str, args: &[i64]) -> Result<i64, (i32, i32, i32)> {
    let mut config = Config::new();
    config.wasm_tail_call(true);
    let engine = Engine::new(&config).unwrap();
    let module = Module::new(&engine, &m.wasm).unwrap();
    let mut store: Store<Option<(i32, i32, i32)>> = Store::new(&engine, None);
    let mut linker = Linker::new(&engine);
    linker
        .func_wrap("qal_host", "fail", |mut c: Caller<'_, Option<(i32, i32, i32)>>, k: i32, f: i32, i: i32| {
            *c.data_mut() = Some((k, f, i));
            Err::<(), _>(wasmtime::Error::msg("qal failure"))
        })
        .unwrap();
    let instance: Instance = linker.instantiate(&mut store, &module).unwrap();
    let func = instance.get_func(&mut store, export).expect("export");
    let params: Vec<Val> = args.iter().map(|a| Val::I64(*a)).collect();
    let mut results = vec![Val::I64(0)];
    match func.call(&mut store, &params, &mut results) {
        Ok(()) => Ok(match results[0] {
            Val::I64(v) => v,
            Val::I32(v) => v as i64,
            _ => panic!("unexpected result"),
        }),
        Err(e) => Err(store.data().unwrap_or_else(|| panic!("unexpected trap: {e:?}"))),
    }
}

#[test]
fn arithmetic_and_precedence() {
    let m = build("fn f(a: Int, b: Int) -> Int = a + b * 2 - a / b % 3");
    assert_eq!(call(&m, "f", &[7, 2]), Ok(7 + 2 * 2 - (7 / 2) % 3));
    assert_eq!(call(&m, "f", &[-7, 2]), Ok(-7 + 4 - (-7 / 2) % 3));
}

#[test]
fn booleans_if_let_and_unit() {
    let m = build(
        "fn max(a: Int, b: Int) -> Int = if a > b then a else b\n\
         fn both(a: Bool, b: Bool) -> Bool = a && !b || false\n\
         fn twice(x: Int) -> Int = let y = x + x in let y = y + 1 in y\n\
         fn u() -> Unit = ()\n\
         test unit_eq = u() == ()",
    );
    assert_eq!(call(&m, "max", &[3, 9]), Ok(9));
    assert_eq!(call(&m, "twice", &[5]), Ok(11));
    assert_eq!(call(&m, "test:unit_eq", &[]), Ok(1));
}

#[test]
fn short_circuit_skips_right_side() {
    // If `&&` / `||` evaluated their right side, 1 / 0 would fail.
    let m = build("test and_sc = !(false && 1 / 0 == 0)\ntest or_sc = true || 1 / 0 == 0");
    assert_eq!(call(&m, "test:and_sc", &[]), Ok(1));
    assert_eq!(call(&m, "test:or_sc", &[]), Ok(1));
}

#[test]
fn recursion_and_tests() {
    let m = build("fn fact(n: Int) -> Int = if n <= 1 then 1 else n * fact(n - 1)\ntest t5 = fact(5) == 120\ntest bad = fact(3) == 7");
    assert_eq!(call(&m, "fact", &[20]), Ok(2432902008176640000));
    assert_eq!(call(&m, "test:t5", &[]), Ok(1));
    assert_eq!(call(&m, "test:bad", &[]), Ok(0));
}

#[test]
fn integer_overflow_is_an_error_not_a_wrap() {
    let m = build(
        "fn add(a: Int, b: Int) -> Int = a + b\nfn sub(a: Int, b: Int) -> Int = a - b\n\
         fn mul(a: Int, b: Int) -> Int = a * b\nfn div(a: Int, b: Int) -> Int = a / b\n\
         fn rem(a: Int, b: Int) -> Int = a % b\nfn neg(a: Int) -> Int = -a",
    );
    let (max, min) = (i64::MAX, i64::MIN);
    assert_eq!(call(&m, "add", &[max, 1]), Err((FAIL_OVERFLOW, 0, 0)));
    assert_eq!(call(&m, "add", &[max, -1]), Ok(max - 1));
    assert_eq!(call(&m, "sub", &[min, 1]), Err((FAIL_OVERFLOW, 1, 0)));
    assert_eq!(call(&m, "mul", &[min, -1]), Err((FAIL_OVERFLOW, 2, 0)));
    assert_eq!(call(&m, "mul", &[-1, min]), Err((FAIL_OVERFLOW, 2, 0)));
    assert_eq!(call(&m, "mul", &[1 << 32, 1 << 31]), Err((FAIL_OVERFLOW, 2, 0)));
    assert_eq!(call(&m, "mul", &[-3, 4]), Ok(-12));
    assert_eq!(call(&m, "div", &[min, -1]), Err((FAIL_OVERFLOW, 3, 0)));
    assert_eq!(call(&m, "div", &[1, 0]), Err((FAIL_DIV_ZERO, 3, 0)));
    assert_eq!(call(&m, "rem", &[1, 0]), Err((FAIL_DIV_ZERO, 4, 0)));
    assert_eq!(call(&m, "rem", &[min, -1]), Ok(0));
    assert_eq!(call(&m, "neg", &[min]), Err((FAIL_OVERFLOW, 5, 0)));
}

#[test]
fn contracts_are_checked_at_runtime() {
    let m = build(
        "fn sqrt_floor(n: Int) -> Int\n  req n >= 0\n  ens r * r <= n\n  ens (r + 1) * (r + 1) > n\n= go(n, 0)\n\
         fn go(n: Int, i: Int) -> Int = if (i + 1) * (i + 1) > n then i else go(n, i + 1)\n\
         fn liar(n: Int) -> Int ens r > n = n",
    );
    assert_eq!(call(&m, "sqrt_floor", &[17]), Ok(4));
    assert_eq!(call(&m, "sqrt_floor", &[-1]), Err((FAIL_REQ, 0, 0)));
    assert_eq!(call(&m, "liar", &[3]), Err((FAIL_ENS, 2, 0)));
}

#[test]
fn ens_in_recursive_function_checks_each_call() {
    // Inner call fact(1) returns 1 which violates `r > n` (1 > 1 is false): reported at index 0.
    let m = build("fn fact(n: Int) -> Int ens r > n = if n <= 1 then 1 else n * fact(n - 1)");
    assert_eq!(call(&m, "fact", &[4]), Err((FAIL_ENS, 0, 0)));
}

#[test]
fn deep_tail_recursion_does_not_overflow_the_stack() {
    let m = build("fn count(n: Int, acc: Int) -> Int = if n == 0 then acc else count(n - 1, acc + 1)");
    assert_eq!(call(&m, "count", &[1_000_000, 0]), Ok(1_000_000));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Create an empty `qal/crates/qal-wasm/src/lib.rs`, then run: `cd qal && cargo test -p qal-wasm`
Expected: compile error — `unresolved imports qal_wasm::compile, qal_wasm::CompiledModule ...`.

- [ ] **Step 3: Write the implementation**

`qal/crates/qal-wasm/src/lib.rs`:

```rust
//! QAL -> WebAssembly code generation. Input must have passed `qal_check::check_program`.
//!
//! Module layout (function index space):
//!   0      import `qal_host.fail(kind: i32, func: i32, index: i32)` — records a structured failure, then traps
//!   1..=5  checked Int helpers: add, sub, mul, div, rem — `(a: i64, b: i64, func: i32) -> i64`
//!   6..    user definitions in program order (fns, then tests in the same order); exported by name,
//!          tests as `test:<name>`
//! Value representation: Int = i64, Bool = i32 (0/1), Unit = i32 (always 0).

use qal_core::ast::*;
use wasm_encoder::{
    BlockType, CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection, ImportSection,
    Instruction as I, Module, TypeSection, ValType,
};

/// Failure kinds passed to `qal_host.fail`.
pub const FAIL_REQ: i32 = 1;
pub const FAIL_ENS: i32 = 2;
pub const FAIL_OVERFLOW: i32 = 3;
pub const FAIL_DIV_ZERO: i32 = 4;

const HOST_FAIL: u32 = 0;
const ADD: u32 = 1;
const SUB: u32 = 2;
const MUL: u32 = 3;
const DIV: u32 = 4;
const REM: u32 = 5;
const FIRST_USER: u32 = 6;

#[derive(Debug, Clone, PartialEq)]
pub struct FuncMeta {
    /// Definition name (without `test:` prefix).
    pub name: String,
    /// Export name: the def name, or `test:<name>` for tests.
    pub export: String,
    pub is_test: bool,
    pub params: Vec<Type>,
    pub ret: Type,
    pub req: Vec<Expr>,
    pub ens: Vec<Expr>,
}

#[derive(Debug, Clone)]
pub struct CompiledModule {
    pub wasm: Vec<u8>,
    /// Indexed by the `func` argument of `qal_host.fail`.
    pub funcs: Vec<FuncMeta>,
}

impl CompiledModule {
    pub fn func(&self, export: &str) -> Option<&FuncMeta> {
        self.funcs.iter().find(|f| f.export == export)
    }
}

fn val(t: Type) -> ValType {
    match t {
        Type::Int => ValType::I64,
        Type::Bool | Type::Unit => ValType::I32,
    }
}

pub fn compile(p: &Program) -> CompiledModule {
    let mut funcs = Vec::new();
    for d in &p.defs {
        if let Def::Fn(f) = d {
            funcs.push(FuncMeta {
                name: f.name.clone(),
                export: f.name.clone(),
                is_test: false,
                params: f.params.iter().map(|p| p.ty).collect(),
                ret: f.ret,
                req: f.req.clone(),
                ens: f.ens.clone(),
            });
        }
    }
    for d in &p.defs {
        if let Def::Test(t) = d {
            funcs.push(FuncMeta {
                name: t.name.clone(),
                export: format!("test:{}", t.name),
                is_test: true,
                params: vec![],
                ret: Type::Bool,
                req: vec![],
                ens: vec![],
            });
        }
    }

    let mut types = TypeSection::new();
    let mut imports = ImportSection::new();
    let mut functions = FunctionSection::new();
    let mut exports = ExportSection::new();
    let mut codes = CodeSection::new();

    // type 0: fail(i32, i32, i32); type 1: helper(i64, i64, i32) -> i64
    types.ty().function([ValType::I32, ValType::I32, ValType::I32], []);
    types.ty().function([ValType::I64, ValType::I64, ValType::I32], [ValType::I64]);
    imports.import("qal_host", "fail", EntityType::Function(0));
    for helper in [ADD, SUB, MUL, DIV, REM] {
        functions.function(1);
        codes.function(&helper_body(helper));
    }

    let defs: Vec<&Def> = p.defs.iter().filter(|d| matches!(d, Def::Fn(_))).chain(p.defs.iter().filter(|d| matches!(d, Def::Test(_)))).collect();
    for (i, d) in defs.iter().enumerate() {
        let meta = &funcs[i];
        let type_idx = 2 + i as u32;
        types.ty().function(meta.params.iter().map(|t| val(*t)), [val(meta.ret)]);
        functions.function(type_idx);
        exports.export(&meta.export, ExportKind::Func, FIRST_USER + i as u32);
        let body = match d {
            Def::Fn(f) => FnGen::new(&funcs, i as u32, &f.params).fn_body(f),
            Def::Test(t) => FnGen::new(&funcs, i as u32, &[]).test_body(&t.body),
        };
        codes.function(&body);
    }

    let mut module = Module::new();
    module.section(&types).section(&imports).section(&functions).section(&exports).section(&codes);
    CompiledModule { wasm: module.finish(), funcs }
}

/// Emits `fail(kind, func, index); unreachable`.
fn emit_fail(out: &mut Vec<I<'static>>, kind: i32, func: FuncRef, index: i32) {
    out.push(I::I32Const(kind));
    match func {
        FuncRef::Const(f) => out.push(I::I32Const(f)),
        FuncRef::Local(l) => out.push(I::LocalGet(l)),
    }
    out.push(I::I32Const(index));
    out.push(I::Call(HOST_FAIL));
    out.push(I::Unreachable);
}

#[derive(Clone, Copy)]
enum FuncRef {
    Const(i32),
    Local(u32),
}

/// Helper params: a = 0, b = 1, func = 2; local r = 3 (i64).
fn helper_body(which: u32) -> Function {
    let (a, b, func, r) = (0, 1, 2, 3);
    let mut f = Function::new([(1, ValType::I64)]);
    let mut out: Vec<I<'static>> = Vec::new();
    let site = FuncRef::Local(func);
    let fail_if = |out: &mut Vec<I<'static>>, kind: i32| {
        out.push(I::If(BlockType::Empty));
        emit_fail(out, kind, site, 0);
        out.push(I::End);
    };
    match which {
        ADD | SUB => {
            out.extend([I::LocalGet(a), I::LocalGet(b), if which == ADD { I::I64Add } else { I::I64Sub }, I::LocalSet(r)]);
            // add overflows iff ((a ^ r) & (b ^ r)) < 0; sub iff ((a ^ b) & (a ^ r)) < 0
            if which == ADD {
                out.extend([I::LocalGet(a), I::LocalGet(r), I::I64Xor, I::LocalGet(b), I::LocalGet(r), I::I64Xor]);
            } else {
                out.extend([I::LocalGet(a), I::LocalGet(b), I::I64Xor, I::LocalGet(a), I::LocalGet(r), I::I64Xor]);
            }
            out.extend([I::I64And, I::I64Const(0), I::I64LtS]);
            fail_if(&mut out, FAIL_OVERFLOW);
            out.push(I::LocalGet(r));
        }
        MUL => {
            // a == -1 && b == MIN overflows (and would trap in the check below)
            out.extend([I::LocalGet(a), I::I64Const(-1), I::I64Eq, I::LocalGet(b), I::I64Const(i64::MIN), I::I64Eq, I::I32And]);
            fail_if(&mut out, FAIL_OVERFLOW);
            out.extend([I::LocalGet(a), I::LocalGet(b), I::I64Mul, I::LocalSet(r)]);
            // a != 0 && r / a != b
            out.extend([I::LocalGet(a), I::I64Const(0), I::I64Ne]);
            out.push(I::If(BlockType::Empty));
            out.extend([I::LocalGet(r), I::LocalGet(a), I::I64DivS, I::LocalGet(b), I::I64Ne]);
            fail_if(&mut out, FAIL_OVERFLOW);
            out.push(I::End);
            out.push(I::LocalGet(r));
        }
        DIV | REM => {
            out.extend([I::LocalGet(b), I::I64Eqz]);
            fail_if(&mut out, FAIL_DIV_ZERO);
            if which == DIV {
                out.extend([I::LocalGet(a), I::I64Const(i64::MIN), I::I64Eq, I::LocalGet(b), I::I64Const(-1), I::I64Eq, I::I32And]);
                fail_if(&mut out, FAIL_OVERFLOW);
                out.extend([I::LocalGet(a), I::LocalGet(b), I::I64DivS]);
            } else {
                // wasm i64.rem_s(MIN, -1) is defined as 0, no trap
                out.extend([I::LocalGet(a), I::LocalGet(b), I::I64RemS]);
            }
        }
        _ => unreachable!("unknown helper"),
    }
    out.push(I::End);
    for ins in &out {
        f.instruction(ins);
    }
    f
}

struct FnGen<'a> {
    funcs: &'a [FuncMeta],
    func_id: u32,
    /// (name, local index, type); searched from the end for shadowing.
    env: Vec<(String, u32, Type)>,
    n_params: u32,
    extra_locals: Vec<ValType>,
    out: Vec<I<'static>>,
}

impl<'a> FnGen<'a> {
    fn new(funcs: &'a [FuncMeta], func_id: u32, params: &[Param]) -> Self {
        let env = params.iter().enumerate().map(|(i, p)| (p.name.clone(), i as u32, p.ty)).collect();
        FnGen { funcs, func_id, env, n_params: params.len() as u32, extra_locals: Vec::new(), out: Vec::new() }
    }

    fn new_local(&mut self, t: Type) -> u32 {
        self.extra_locals.push(val(t));
        self.n_params + self.extra_locals.len() as u32 - 1
    }

    fn finish(mut self) -> Function {
        self.out.push(I::End);
        let mut f = Function::new(self.extra_locals.iter().map(|v| (1, *v)));
        for ins in &self.out {
            f.instruction(ins);
        }
        f
    }

    fn fn_body(mut self, f: &FnDef) -> Function {
        for (i, c) in f.req.iter().enumerate() {
            self.check_contract(c, FAIL_REQ, i as i32);
        }
        if f.ens.is_empty() {
            self.expr(&f.body, true);
        } else {
            // No tail calls here: `ens` must be checked against this call's params after the body returns.
            self.expr(&f.body, false);
            let r = self.new_local(f.ret);
            self.out.push(I::LocalSet(r));
            self.env.push((RESULT_VAR.to_string(), r, f.ret));
            for (i, c) in f.ens.iter().enumerate() {
                self.check_contract(c, FAIL_ENS, i as i32);
            }
            self.out.push(I::LocalGet(r));
        }
        self.finish()
    }

    fn test_body(mut self, body: &Expr) -> Function {
        self.expr(body, false);
        self.finish()
    }

    fn check_contract(&mut self, c: &Expr, kind: i32, index: i32) {
        self.expr(c, false);
        self.out.push(I::I32Eqz);
        self.out.push(I::If(BlockType::Empty));
        emit_fail(&mut self.out, kind, FuncRef::Const(self.func_id as i32), index);
        self.out.push(I::End);
    }

    fn helper(&mut self, idx: u32) {
        self.out.push(I::I32Const(self.func_id as i32));
        self.out.push(I::Call(idx));
    }

    /// Emits `e`, leaving its value on the stack. Returns its type.
    fn expr(&mut self, e: &Expr, tail: bool) -> Type {
        match e {
            Expr::Int(n) => {
                self.out.push(I::I64Const(*n));
                Type::Int
            }
            Expr::Bool(b) => {
                self.out.push(I::I32Const(*b as i32));
                Type::Bool
            }
            Expr::Unit => {
                self.out.push(I::I32Const(0));
                Type::Unit
            }
            Expr::Var(v) => {
                let (_, idx, t) = self.env.iter().rev().find(|(n, _, _)| n == v).expect("checked: variable in scope").clone();
                self.out.push(I::LocalGet(idx));
                t
            }
            Expr::Un(UnOp::Neg, x) => {
                self.out.push(I::I64Const(0));
                self.expr(x, false);
                self.helper(SUB);
                Type::Int
            }
            Expr::Un(UnOp::Not, x) => {
                self.expr(x, false);
                self.out.push(I::I32Eqz);
                Type::Bool
            }
            Expr::Bin(BinOp::And, l, r) => {
                self.expr(l, false);
                self.out.push(I::If(BlockType::Result(ValType::I32)));
                self.expr(r, false);
                self.out.push(I::Else);
                self.out.push(I::I32Const(0));
                self.out.push(I::End);
                Type::Bool
            }
            Expr::Bin(BinOp::Or, l, r) => {
                self.expr(l, false);
                self.out.push(I::If(BlockType::Result(ValType::I32)));
                self.out.push(I::I32Const(1));
                self.out.push(I::Else);
                self.expr(r, false);
                self.out.push(I::End);
                Type::Bool
            }
            Expr::Bin(op, l, r) => {
                let lt = self.expr(l, false);
                self.expr(r, false);
                match op {
                    BinOp::Add => self.helper(ADD),
                    BinOp::Sub => self.helper(SUB),
                    BinOp::Mul => self.helper(MUL),
                    BinOp::Div => self.helper(DIV),
                    BinOp::Rem => self.helper(REM),
                    BinOp::Lt => self.out.push(I::I64LtS),
                    BinOp::Le => self.out.push(I::I64LeS),
                    BinOp::Gt => self.out.push(I::I64GtS),
                    BinOp::Ge => self.out.push(I::I64GeS),
                    BinOp::Eq | BinOp::Ne => {
                        let eq = op == &BinOp::Eq;
                        match lt {
                            Type::Int => self.out.push(if eq { I::I64Eq } else { I::I64Ne }),
                            Type::Bool => self.out.push(if eq { I::I32Eq } else { I::I32Ne }),
                            Type::Unit => {
                                self.out.extend([I::Drop, I::Drop, I::I32Const(eq as i32)]);
                            }
                        }
                    }
                    BinOp::And | BinOp::Or => unreachable!("handled above"),
                }
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => Type::Int,
                    _ => Type::Bool,
                }
            }
            Expr::If(c, t, f) => {
                self.expr(c, false);
                let start = self.out.len();
                self.out.push(I::If(BlockType::Empty)); // patched below once the branch type is known
                let ty = self.expr(t, tail);
                self.out[start] = I::If(BlockType::Result(val(ty)));
                self.out.push(I::Else);
                self.expr(f, tail);
                self.out.push(I::End);
                ty
            }
            Expr::Let(name, v, body) => {
                let vt = self.expr(v, false);
                let local = self.new_local(vt);
                self.out.push(I::LocalSet(local));
                self.env.push((name.clone(), local, vt));
                let bt = self.expr(body, tail);
                self.env.pop();
                bt
            }
            Expr::Call(name, args) => {
                for a in args {
                    self.expr(a, false);
                }
                let idx = self.funcs.iter().position(|f| !f.is_test && &f.name == name).expect("checked: callee exists");
                let ret = self.funcs[idx].ret;
                let wasm_idx = FIRST_USER + idx as u32;
                self.out.push(if tail { I::ReturnCall(wasm_idx) } else { I::Call(wasm_idx) });
                ret
            }
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd qal && cargo test -p qal-wasm`
Expected: 8 tests PASS (first build compiles wasmtime and takes a few minutes), no warnings.

- [ ] **Step 5: Commit**

```bash
git add qal/Cargo.toml qal/crates/qal-wasm
git commit -m "feat(qal): direct WebAssembly codegen with checked arithmetic, contracts and tail calls

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 7: Runtime (`qal-run`)

**Files:**
- Create: `qal/crates/qal-run/Cargo.toml`, `qal/crates/qal-run/src/lib.rs`
- Test: inline module in `lib.rs`

**Interfaces:**
- Consumes: `qal_wasm::{CompiledModule, FAIL_*}`, `qal_text::print_expr`, `qal_core::ast::Type`; dev: `qal_check`.
- Produces: `qal_run::Runner::new(CompiledModule) -> Result<Runner, String>`, `Runner::with_fuel(u64) -> Runner`, `Runner::call(&self, export: &str, args: &[Value]) -> Result<Value, RunError>`, `Runner::run_tests(&self) -> Vec<TestResult>`; `Value::{Int(i64), Bool(bool), Unit(())}` (serializes to plain JSON: number / bool / null); `RunError { kind: &'static str, def: String, message: String, contract: Option<String> }` with kinds `contract_violation, int_overflow, div_by_zero, stack_overflow, timeout, bad_args, unknown_function, trap`; `TestResult { name, passed, error: Option<RunError> }`; `DEFAULT_FUEL = 2_000_000_000`.

- [ ] **Step 0: Register the crate in the workspace**

In `qal/Cargo.toml`, add `"crates/qal-run",` to `members` so it reads:

```toml
members = [
    "crates/qal-core",
    "crates/qal-text",
    "crates/qal-check",
    "crates/qal-wasm",
    "crates/qal-run",
]
```

`qal/crates/qal-run/Cargo.toml`:

```toml
[package]
name = "qal-run"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
qal-core = { workspace = true }
qal-text = { workspace = true }
qal-wasm = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
wasmtime = { workspace = true }

[dev-dependencies]
qal-check = { workspace = true }
```

- [ ] **Step 1: Write the failing tests**

Create `qal/crates/qal-run/src/lib.rs` with only the test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn runner(src: &str) -> Runner {
        let p = qal_text::parse_program(src).unwrap();
        assert!(qal_check::check_program(&p).is_empty());
        Runner::new(qal_wasm::compile(&p)).unwrap()
    }

    #[test]
    fn calls_with_typed_values() {
        let r = runner("fn pick(c: Bool, a: Int, b: Int) -> Int = if c then a else b\nfn u() -> Unit = ()\nfn neg(b: Bool) -> Bool = !b");
        assert_eq!(r.call("pick", &[Value::Bool(false), Value::Int(1), Value::Int(2)]), Ok(Value::Int(2)));
        assert_eq!(r.call("u", &[]), Ok(Value::Unit(())));
        assert_eq!(r.call("neg", &[Value::Bool(true)]), Ok(Value::Bool(false)));
    }

    #[test]
    fn rejects_bad_arguments_and_unknown_functions() {
        let r = runner("fn f(a: Int) -> Int = a");
        assert_eq!(r.call("f", &[]).unwrap_err().kind, "bad_args");
        assert_eq!(r.call("f", &[Value::Bool(true)]).unwrap_err().kind, "bad_args");
        assert_eq!(r.call("g", &[]).unwrap_err().kind, "unknown_function");
    }

    #[test]
    fn contract_violation_names_def_and_contract() {
        let r = runner("fn half(n: Int) -> Int req n % 2 == 0 = n / 2");
        let e = r.call("half", &[Value::Int(3)]).unwrap_err();
        assert_eq!(e.kind, "contract_violation");
        assert_eq!(e.def, "half");
        assert_eq!(e.contract.as_deref(), Some("req[0]: n % 2 == 0"));
        assert_eq!(
            serde_json::to_value(&e).unwrap(),
            serde_json::json!({"kind": "contract_violation", "def": "half",
                "message": "`req n % 2 == 0` of `half` is false", "contract": "req[0]: n % 2 == 0"})
        );
    }

    #[test]
    fn arithmetic_failures_name_the_def() {
        let r = runner("fn inner(a: Int) -> Int = a / 0\nfn outer() -> Int = inner(1) + 1");
        let e = r.call("outer", &[]).unwrap_err();
        assert_eq!((e.kind, e.def.as_str()), ("div_by_zero", "inner"));
    }

    #[test]
    fn infinite_loop_times_out() {
        let r = runner("fn spin(n: Int) -> Int = spin(n)").with_fuel(10_000_000);
        assert_eq!(r.call("spin", &[Value::Int(0)]).unwrap_err().kind, "timeout");
    }

    #[test]
    fn deep_non_tail_recursion_reports_stack_overflow() {
        let r = runner("fn down(n: Int) -> Int = if n == 0 then 0 else 1 + down(n - 1)");
        assert_eq!(r.call("down", &[Value::Int(100_000_000)]).unwrap_err().kind, "stack_overflow");
    }

    #[test]
    fn runs_tests_in_order() {
        let r = runner("test a = 1 == 1\ntest b = 1 == 2\ntest c = 1 / 0 == 0");
        let res = r.run_tests();
        assert_eq!(res.iter().map(|t| (t.name.as_str(), t.passed)).collect::<Vec<_>>(), vec![("a", true), ("b", false), ("c", false)]);
        assert_eq!(res[2].error.as_ref().unwrap().kind, "div_by_zero");
    }

    #[test]
    fn values_serialize_as_plain_json() {
        assert_eq!(serde_json::to_string(&[Value::Int(-3), Value::Bool(true), Value::Unit(())]).unwrap(), "[-3,true,null]");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd qal && cargo test -p qal-run`
Expected: compile error — `cannot find type Runner` / `cannot find type Value`.

- [ ] **Step 3: Write the implementation**

Insert above the test module:

```rust
//! QAL runtime: executes compiled modules in wasmtime and reports failures as data.

use qal_core::ast::Type;
use qal_text::print_expr;
use qal_wasm::{CompiledModule, FAIL_DIV_ZERO, FAIL_ENS, FAIL_OVERFLOW, FAIL_REQ};
use serde::Serialize;
use wasmtime::{Caller, Config, Engine, Linker, Module, Store, Trap, Val};

/// Instruction budget per call; exhausting it reports `timeout` instead of hanging.
pub const DEFAULT_FUEL: u64 = 2_000_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum Value {
    Int(i64),
    Bool(bool),
    /// Serialized as JSON `null`.
    Unit(()),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RunError {
    /// contract_violation | int_overflow | div_by_zero | stack_overflow | timeout | bad_args | unknown_function | trap
    pub kind: &'static str,
    pub def: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contract: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TestResult {
    pub name: String,
    pub passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RunError>,
}

pub struct Runner {
    engine: Engine,
    module: Module,
    meta: CompiledModule,
    fuel: u64,
}

impl Runner {
    pub fn new(meta: CompiledModule) -> Result<Runner, String> {
        let mut config = Config::new();
        config.wasm_tail_call(true);
        config.consume_fuel(true);
        let engine = Engine::new(&config).map_err(|e| e.to_string())?;
        let module = Module::new(&engine, &meta.wasm).map_err(|e| e.to_string())?;
        Ok(Runner { engine, module, meta, fuel: DEFAULT_FUEL })
    }

    pub fn with_fuel(mut self, fuel: u64) -> Runner {
        self.fuel = fuel;
        self
    }

    /// Calls the function exported as `export` (a fn name, or `test:<name>`).
    pub fn call(&self, export: &str, args: &[Value]) -> Result<Value, RunError> {
        let err = |kind, message: String| RunError { kind, def: export.to_string(), message, contract: None };
        let meta = self.meta.func(export).ok_or_else(|| err("unknown_function", format!("no function `{export}`")))?;
        if meta.params.len() != args.len() {
            return Err(err("bad_args", format!("`{export}` takes {} argument(s), {} given", meta.params.len(), args.len())));
        }
        let mut params = Vec::new();
        for (i, (a, t)) in args.iter().zip(&meta.params).enumerate() {
            params.push(match (a, t) {
                (Value::Int(v), Type::Int) => Val::I64(*v),
                (Value::Bool(b), Type::Bool) => Val::I32(*b as i32),
                (Value::Unit(()), Type::Unit) => Val::I32(0),
                _ => return Err(err("bad_args", format!("argument {i} must be {}", t.name()))),
            });
        }

        let mut store: Store<Option<(i32, i32, i32)>> = Store::new(&self.engine, None);
        store.set_fuel(self.fuel).map_err(|e| err("trap", e.to_string()))?;
        let mut linker = Linker::new(&self.engine);
        linker
            .func_wrap("qal_host", "fail", |mut c: Caller<'_, Option<(i32, i32, i32)>>, kind: i32, func: i32, index: i32| {
                *c.data_mut() = Some((kind, func, index));
                Err::<(), _>(wasmtime::Error::msg("qal failure"))
            })
            .map_err(|e| err("trap", e.to_string()))?;
        let instance = linker.instantiate(&mut store, &self.module).map_err(|e| err("trap", e.to_string()))?;
        let func = instance.get_func(&mut store, export).ok_or_else(|| err("unknown_function", format!("no export `{export}`")))?;
        let mut results = [Val::I32(0)];
        match func.call(&mut store, &params, &mut results) {
            Ok(()) => Ok(match (meta.ret, &results[0]) {
                (Type::Int, Val::I64(v)) => Value::Int(*v),
                (Type::Bool, Val::I32(v)) => Value::Bool(*v != 0),
                (Type::Unit, _) => Value::Unit(()),
                (t, v) => return Err(err("trap", format!("result {v:?} does not match {}", t.name()))),
            }),
            Err(e) => Err(match *store.data() {
                Some(failure) => self.describe_failure(failure),
                None => match e.downcast_ref::<Trap>() {
                    Some(Trap::StackOverflow) => err("stack_overflow", "call stack exhausted (non-tail recursion too deep?)".into()),
                    Some(Trap::OutOfFuel) => err("timeout", format!("instruction budget of {} exhausted", self.fuel)),
                    _ => err("trap", e.to_string()),
                },
            }),
        }
    }

    fn describe_failure(&self, (kind, func, index): (i32, i32, i32)) -> RunError {
        let meta = &self.meta.funcs[func as usize];
        let def = meta.name.clone();
        match kind {
            FAIL_REQ | FAIL_ENS => {
                let (label, list) = if kind == FAIL_REQ { ("req", &meta.req) } else { ("ens", &meta.ens) };
                let text = print_expr(&list[index as usize]);
                RunError {
                    kind: "contract_violation",
                    message: format!("`{label} {text}` of `{def}` is false"),
                    contract: Some(format!("{label}[{index}]: {text}")),
                    def,
                }
            }
            FAIL_OVERFLOW => RunError { kind: "int_overflow", message: format!("Int overflow in `{def}`"), def, contract: None },
            FAIL_DIV_ZERO => RunError { kind: "div_by_zero", message: format!("division by zero in `{def}`"), def, contract: None },
            other => RunError { kind: "trap", message: format!("unknown failure kind {other}"), def, contract: None },
        }
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
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd qal && cargo test -p qal-run`
Expected: 8 tests PASS, no warnings.

- [ ] **Step 5: Commit**

```bash
git add qal/Cargo.toml qal/crates/qal-run
git commit -m "feat(qal): wasmtime runtime with structured failures and fuel limit

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 8: Pipeline and `qal` CLI (`qal-cli`)

**Files:**
- Create: `qal/crates/qal-cli/Cargo.toml`, `qal/crates/qal-cli/src/lib.rs`, `qal/crates/qal-cli/src/main.rs`, `qal/examples/math.qal`
- Test: `qal/crates/qal-cli/tests/cli.rs`

**Interfaces:**
- Consumes: `qal_text::{parse_program, print_program}`, `qal_check::check_program`, `qal_wasm::compile`, `qal_run::{Runner, Value, RunError}`.
- Produces: `qal_cli::load(&str) -> Result<Loaded, Vec<Diagnostic>>` with `Loaded { program: Program, runner: Runner }`; `qal_cli::parse_args(&Program, name: &str, raw: &[String]) -> Result<Vec<Value>, RunError>`; binary `qal` with `check | fmt | call | test` (JSON on stdout; exit 0 ok, 1 diagnostics/runtime error/failed tests, 2 usage or I/O error). Sub-plan 1d builds the MCP server on `load`.

- [ ] **Step 0: Register the crate in the workspace**

In `qal/Cargo.toml`, add `"crates/qal-cli",` to `members` so it reads:

```toml
members = [
    "crates/qal-core",
    "crates/qal-text",
    "crates/qal-check",
    "crates/qal-wasm",
    "crates/qal-run",
    "crates/qal-cli",
]
```

`qal/crates/qal-cli/Cargo.toml`:

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
serde_json = { workspace = true }
```

`qal/examples/math.qal`:

```text
# Phase 1a example: pure Int/Bool functions with contracts and tests.

fn fact(n: Int) -> Int
  req n >= 0
  ens r >= 1
= if n <= 1 then 1 else n * fact(n - 1)

fn gcd(a: Int, b: Int) -> Int
  req a >= 0 && b >= 0
= if b == 0 then a else gcd(b, a % b)

fn is_prime(n: Int) -> Bool = n >= 2 && no_divisor(n, 2)

fn no_divisor(n: Int, d: Int) -> Bool = if d * d > n then true else n % d != 0 && no_divisor(n, d + 1)

test fact_5 = fact(5) == 120
test gcd_48_18 = gcd(48, 18) == 6
test prime_97 = is_prime(97)
test not_prime_91 = !is_prime(91)
```

- [ ] **Step 1: Write the failing tests**

Create `qal/crates/qal-cli/tests/cli.rs`:

```rust
//! Drives the real `qal` binary and checks its JSON output.

use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::Command;

fn example() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/math.qal").to_string_lossy().into_owned()
}

fn temp_file(name: &str, content: &str) -> String {
    let dir = std::env::temp_dir().join(format!("qal-cli-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, content).unwrap();
    path.to_string_lossy().into_owned()
}

fn qal(args: &[&str]) -> (i32, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_qal")).args(args).output().unwrap();
    (out.status.code().unwrap(), String::from_utf8(out.stdout).unwrap())
}

fn qal_json(args: &[&str]) -> (i32, Value) {
    let (code, out) = qal(args);
    (code, serde_json::from_str(&out).unwrap_or_else(|e| panic!("not JSON ({e}): {out}")))
}

#[test]
fn check_ok_and_call() {
    assert_eq!(qal_json(&["check", &example()]), (0, json!({"ok": true})));
    assert_eq!(qal_json(&["call", &example(), "gcd", "48", "18"]), (0, json!({"ok": 6})));
    assert_eq!(qal_json(&["call", &example(), "is_prime", "97"]), (0, json!({"ok": true})));
}

#[test]
fn call_reports_contract_violation_and_bad_args() {
    let (code, v) = qal_json(&["call", &example(), "fact", "-1"]);
    assert_eq!(code, 1);
    assert_eq!(v["error"]["kind"], "contract_violation");
    assert_eq!(v["error"]["contract"], "req[0]: n >= 0");
    let (code, v) = qal_json(&["call", &example(), "gcd", "x", "1"]);
    assert_eq!((code, v["error"]["kind"].as_str()), (1, Some("bad_args")));
    let (code, v) = qal_json(&["call", &example(), "nope"]);
    assert_eq!((code, v["error"]["kind"].as_str()), (1, Some("unknown_function")));
}

#[test]
fn test_command_summarizes() {
    let (code, v) = qal_json(&["test", &example()]);
    assert_eq!(code, 0);
    assert_eq!((v["passed"].as_u64(), v["failed"].as_u64()), (Some(4), Some(0)));
    let failing = temp_file("failing.qal", "test good = true\ntest bad = 1 == 2");
    let (code, v) = qal_json(&["test", &failing]);
    assert_eq!((code, v["failed"].as_u64()), (1, Some(1)));
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
    assert_eq!(qal(&["frobnicate", &example()]).0, 2);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Create `qal/crates/qal-cli/src/main.rs` containing `fn main() {}` and an empty `src/lib.rs`, then run: `cd qal && cargo test -p qal-cli`
Expected: tests FAIL — `not JSON (EOF while parsing a value)` / exit-code assertions fail.

- [ ] **Step 3: Write the implementation**

`qal/crates/qal-cli/src/lib.rs`:

```rust
//! The QAL pipeline as one call: text -> parse -> check -> compile -> runner.

use qal_core::ast::{Def, Program, Type};
use qal_core::diag::Diagnostic;
use qal_run::{RunError, Runner, Value};

pub struct Loaded {
    pub program: Program,
    pub runner: Runner,
}

/// Parses, checks and compiles `src`. Any diagnostic means nothing is compiled.
pub fn load(src: &str) -> Result<Loaded, Vec<Diagnostic>> {
    let program = qal_text::parse_program(src).map_err(|d| vec![d])?;
    let diags = qal_check::check_program(&program);
    if !diags.is_empty() {
        return Err(diags);
    }
    let runner = Runner::new(qal_wasm::compile(&program)).expect("checked programs compile to valid wasm");
    Ok(Loaded { program, runner })
}

/// Converts CLI strings to values according to the signature of fn `name`.
pub fn parse_args(program: &Program, name: &str, raw: &[String]) -> Result<Vec<Value>, RunError> {
    let bad = |message: String| RunError { kind: "bad_args", def: name.to_string(), message, contract: None };
    let f = program
        .defs
        .iter()
        .find_map(|d| match d {
            Def::Fn(f) if f.name == name => Some(f),
            _ => None,
        })
        .ok_or_else(|| RunError { kind: "unknown_function", def: name.to_string(), message: format!("no function `{name}`"), contract: None })?;
    if f.params.len() != raw.len() {
        return Err(bad(format!("`{name}` takes {} argument(s), {} given", f.params.len(), raw.len())));
    }
    f.params
        .iter()
        .zip(raw)
        .map(|(p, s)| match p.ty {
            Type::Int => s.parse::<i64>().map(Value::Int).map_err(|_| bad(format!("`{}` must be Int, got `{s}`", p.name))),
            Type::Bool => match s.as_str() {
                "true" => Ok(Value::Bool(true)),
                "false" => Ok(Value::Bool(false)),
                _ => Err(bad(format!("`{}` must be Bool (true/false), got `{s}`", p.name))),
            },
            Type::Unit => match s.as_str() {
                "()" => Ok(Value::Unit(())),
                _ => Err(bad(format!("`{}` must be Unit `()`, got `{s}`", p.name))),
            },
        })
        .collect()
}
```

`qal/crates/qal-cli/src/main.rs`:

```rust
//! `qal` command line. Every output is JSON on stdout, except `fmt` which prints canonical text.
//!
//!   qal check <file>               -> {"ok":true} | {"diagnostics":[...]}
//!   qal fmt <file>                 -> canonical text
//!   qal call <file> <fn> [args..]  -> {"ok":<value>} | {"error":{...}} | {"diagnostics":[...]}
//!   qal test <file>                -> {"tests":[...],"passed":n,"failed":m}
//!
//! Exit codes: 0 success, 1 diagnostics / runtime error / failed tests, 2 usage or I/O error.

use qal_cli::{load, parse_args};
use serde_json::json;
use std::process::ExitCode;

const USAGE: &str = "usage: qal <check|fmt|call|test> <file> [fn] [args...]";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(cmd), Some(file)) = (args.first(), args.get(1)) else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let src = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read {file}: {e}");
            return ExitCode::from(2);
        }
    };
    let emit = |v: serde_json::Value, ok: bool| {
        println!("{v}");
        if ok { ExitCode::SUCCESS } else { ExitCode::from(1) }
    };
    match cmd.as_str() {
        "fmt" => match qal_text::parse_program(&src) {
            Ok(p) => {
                print!("{}", qal_text::print_program(&p));
                ExitCode::SUCCESS
            }
            Err(d) => emit(json!({ "diagnostics": [d] }), false),
        },
        "check" => match load(&src) {
            Ok(_) => emit(json!({ "ok": true }), true),
            Err(d) => emit(json!({ "diagnostics": d }), false),
        },
        "call" => {
            let Some(name) = args.get(2) else {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            };
            let loaded = match load(&src) {
                Ok(l) => l,
                Err(d) => return emit(json!({ "diagnostics": d }), false),
            };
            let result = parse_args(&loaded.program, name, &args[3..]).and_then(|vals| loaded.runner.call(name, &vals));
            match result {
                Ok(v) => emit(json!({ "ok": v }), true),
                Err(e) => emit(json!({ "error": e }), false),
            }
        }
        "test" => match load(&src) {
            Ok(l) => {
                let tests = l.runner.run_tests();
                let passed = tests.iter().filter(|t| t.passed).count();
                let failed = tests.len() - passed;
                emit(json!({ "tests": tests, "passed": passed, "failed": failed }), failed == 0)
            }
            Err(d) => emit(json!({ "diagnostics": d }), false),
        },
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `cd qal && cargo test --workspace && cargo build --workspace 2>&1 | grep -c warning`
Expected: 6 CLI tests PASS; whole workspace 42 tests PASS; warning count `0`.

- [ ] **Step 5: Smoke-test by hand**

Run: `cd qal && cargo run -q -p qal-cli -- call examples/math.qal fact 10`
Expected output: `{"ok":3628800}`

Run: `cd qal && cargo run -q -p qal-cli -- test examples/math.qal`
Expected: JSON with `"passed":4,"failed":0`.

- [ ] **Step 6: Commit**

```bash
git add qal/Cargo.toml qal/crates/qal-cli qal/examples
git commit -m "feat(qal): qal CLI — check, fmt, call, test with JSON output

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Done when

- `cd qal && cargo test --workspace` passes (42 tests) with zero warnings.
- `qal call examples/math.qal fact -1` reports `contract_violation` with `"contract":"req[0]: n >= 0"`.
- The Review Focus tests above exist and pass.
