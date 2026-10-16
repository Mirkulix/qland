//! The compiler facade (self-hosting spec §9): the one crate through which the host crates
//! (`qal-store`, `qal-run`, `qal-prove`, `qal-serve`, `qal-cli`) parse, print, check, hash and
//! compile QAL programs.
//!
//! The work is done by a [`Backend`]. Phase 5a has one, [`Rust`], which calls the Rust pipeline
//! (`qal-text`, `qal-check`, `qal_core::canon`, `qal-wasm`): the reference. From 5b on a `Qal`
//! backend runs the self-hosted compiler module through the `QALQ`/`QALR` protocol ([`proto`]);
//! the free functions below then dispatch on the selected backend instead of [`Rust`], and
//! `both` compares the two byte for byte. The host crates do not change again.
//!
//! The data types the host reads (AST, typed program, diagnostics, canonical forms, compiled
//! modules) are re-exported here, so a host crate needs no other compiler crate.

pub mod proto;

use qal_core::ast::{Expr, Program};
use qal_core::canon::{HashError, Hashed};
use qal_core::diag::Diagnostic;
use qal_core::tast::TProgram;

pub use qal_check::{KNOWN_EFFECTS, MAX_INSTANCES};
pub use qal_wasm::{
    host, CompileOptions, CompiledModule, Entry, FuncMeta, Site, COMPILER_MAX_PAGES, FAIL_BUILTIN, FAIL_DIV_ZERO, FAIL_ENS, FAIL_INDEX, FAIL_OOM, FAIL_OVERFLOW, FAIL_REQ, FAIL_TODO, HOST_MODULE, MAX_MEMORY_PAGES,
    MAX_PAGES, OOM_SCRATCH,
};

/// Rules for names (not compilation): the `std.` namespace, keywords and length limits.
pub mod lexer {
    pub use qal_text::lexer::{is_std_name, last_segment, std_parts, KEYWORDS, MAX_NAME_LEN, STD_PREFIX};
}

/// One implementation of the compiler operations (the `QALQ` operations of spec §4.2, as Rust
/// calls). Every method is deterministic: equal inputs give byte-identical outputs.
pub trait Backend: Send + Sync {
    /// `rust` or `qal`.
    fn name(&self) -> &'static str;
    /// Text -> program (`PARSE`).
    fn parse(&self, src: &str) -> Result<Program, Diagnostic>;
    /// Program -> canonical text (`PRINT`).
    fn print(&self, p: &Program) -> String;
    /// One expression as text (contract texts in runtime errors and proof reports).
    fn print_expr(&self, e: &Expr) -> String;
    /// Names, types, effects, contracts and exhaustiveness; the typed, monomorphic program
    /// (`TYPED`, `CHECK_FILE`). `is_new` tells which definitions are written now (see
    /// `qal_check::check_stored`).
    fn check(&self, p: &Program, is_new: &dyn Fn(&str) -> bool) -> Result<TProgram, Vec<Diagnostic>>;
    /// Canonical forms and hashes of every definition (the hash part of `DEFINE`).
    fn hash(&self, p: &Program) -> Result<Hashed, HashError>;
    /// The WebAssembly module of the code reachable from `entry` (`BUILD`).
    fn compile(&self, p: &TProgram, entry: &Entry, opts: &CompileOptions) -> Result<CompiledModule, String>;
}

/// The Rust pipeline: the reference implementation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Rust;

impl Backend for Rust {
    fn name(&self) -> &'static str {
        "rust"
    }

    fn parse(&self, src: &str) -> Result<Program, Diagnostic> {
        qal_text::parse_program(src)
    }

    fn print(&self, p: &Program) -> String {
        qal_text::print_program(p)
    }

    fn print_expr(&self, e: &Expr) -> String {
        qal_text::print_expr(e)
    }

    fn check(&self, p: &Program, is_new: &dyn Fn(&str) -> bool) -> Result<TProgram, Vec<Diagnostic>> {
        qal_check::check_stored(p, is_new)
    }

    fn hash(&self, p: &Program) -> Result<Hashed, HashError> {
        qal_core::canon::hash_program(p)
    }

    fn compile(&self, p: &TProgram, entry: &Entry, opts: &CompileOptions) -> Result<CompiledModule, String> {
        qal_wasm::compile_with(p, entry, opts)
    }
}

static RUST: Rust = Rust;

/// The backend the free functions use. Phase 5a: always [`Rust`].
pub fn backend() -> &'static dyn Backend {
    &RUST
}

/// The backend named `name` (`rust`; `qal` and `both` exist from 5b on).
pub fn select(name: &str) -> Result<&'static dyn Backend, String> {
    match name {
        "rust" => Ok(&RUST),
        "qal" | "both" => Err(format!("the compiler backend `{name}` is not available yet (self-hosting Phase 5b); use `rust`")),
        _ => Err(format!("unknown compiler backend `{name}`; expected rust, qal or both")),
    }
}

/// Parses a whole program.
pub fn parse_program(src: &str) -> Result<Program, Diagnostic> {
    backend().parse(src)
}

/// The canonical text of a program.
pub fn print_program(p: &Program) -> String {
    backend().print(p)
}

/// The canonical text of an expression.
pub fn print_expr(e: &Expr) -> String {
    backend().print_expr(e)
}

/// Checks a whole program (every definition is new) and returns its typed form.
pub fn check(p: &Program) -> Result<TProgram, Vec<Diagnostic>> {
    backend().check(p, &|_| true)
}

/// Checks a stored codebase: `is_new` tells which definitions are written now.
pub fn check_stored(p: &Program, is_new: &dyn Fn(&str) -> bool) -> Result<TProgram, Vec<Diagnostic>> {
    backend().check(p, is_new)
}

/// Canonical forms and hashes of a program's definitions.
pub fn hash_program(p: &Program) -> Result<Hashed, HashError> {
    backend().hash(p)
}

/// Compiles the code reachable from `entry` with the default options (user programs).
pub fn compile(p: &TProgram, entry: &Entry) -> Result<CompiledModule, String> {
    backend().compile(p, entry, &CompileOptions::default())
}

/// Compiles with options (compiler runs use [`CompileOptions::compiler`]).
pub fn compile_with(p: &TProgram, entry: &Entry, opts: &CompileOptions) -> Result<CompiledModule, String> {
    backend().compile(p, entry, opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rust_backend_is_the_reference_pipeline() {
        let src = "fn f(x: Int) -> Int = x + 1\ntest t = f(1) == 2";
        let p = parse_program(src).unwrap();
        assert_eq!(p, qal_text::parse_program(src).unwrap());
        assert_eq!(print_program(&p), qal_text::print_program(&p));
        let tp = check(&p).unwrap();
        assert_eq!(tp, qal_check::check(&p).unwrap());
        assert_eq!(hash_program(&p).unwrap(), qal_core::canon::hash_program(&p).unwrap());
        assert_eq!(compile(&tp, &Entry::All).unwrap().wasm, qal_wasm::compile(&tp, &Entry::All).unwrap().wasm);
        assert_eq!(backend().name(), "rust");
        assert_eq!(select("rust").map(|b| b.name()), Ok("rust"));
        assert!(select("qal").is_err() && select("both").is_err() && select("x").err().is_some_and(|e| e.contains("unknown")));
    }

    #[test]
    fn host_crates_depend_only_on_the_facade() {
        let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        for host in ["qal-store", "qal-run", "qal-prove", "qal-serve", "qal-cli"] {
            let manifest = std::fs::read_to_string(crates.join(host).join("Cargo.toml")).unwrap();
            assert!(manifest.contains("qal-compiler = { workspace = true }"), "{host}");
            for compiler in ["qal-text", "qal-check", "qal-wasm"] {
                assert!(!manifest.contains(compiler), "{host} depends on {compiler} directly");
            }
        }
    }

    #[test]
    fn compile_options_bound_the_memory() {
        let tp = check(&parse_program("fn f(x: Int) -> Int = x").unwrap()).unwrap();
        let pages = |o: CompileOptions| compile_with(&tp, &Entry::All, &o).map(|m| m.wasm.len());
        assert!(pages(CompileOptions::compiler()).is_ok() && pages(CompileOptions::default()).is_ok());
        for bad in [0, MAX_MEMORY_PAGES + 1, u64::MAX] {
            assert!(pages(CompileOptions { memory_max_pages: bad }).unwrap_err().contains("memory_max_pages"), "{bad}");
        }
        assert_eq!((MAX_PAGES, COMPILER_MAX_PAGES), (4096, 16384));
    }
}
