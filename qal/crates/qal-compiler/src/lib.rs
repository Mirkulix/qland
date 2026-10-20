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

pub use qal_core::{KNOWN_EFFECTS, MAX_INSTANCES};
pub use qal_core::module::{
    host, CompileOptions, CompiledModule, Entry, FuncMeta, Site, COMPILER_MAX_PAGES, FAIL_BUILTIN, FAIL_DIV_ZERO,
    FAIL_ENS, FAIL_INDEX, FAIL_OOM, FAIL_OVERFLOW, FAIL_REQ, FAIL_TODO, HOST_MODULE, MAX_MEMORY_PAGES, MAX_PAGES,
    OOM_SCRATCH,
};

/// Rules for names (not compilation): the `std.` namespace, keywords and length limits.
pub mod lexer {
    pub use qal_core::lexer::{is_std_name, last_segment, std_parts, KEYWORDS, MAX_NAME_LEN, STD_PREFIX};
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
#[cfg(feature = "reference")]
#[derive(Debug, Clone, Copy, Default)]
pub struct Rust;

#[cfg(feature = "reference")]
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

pub type Driver = Box<dyn Fn(&[u8]) -> Result<Vec<u8>, Diagnostic> + Send + Sync>;

static QAL_DRIVER: std::sync::RwLock<Option<Driver>> = std::sync::RwLock::new(None);

/// Sets the driver that invokes the self-hosted compiler module (e.g. via wasmtime in `qal-run`).
pub fn set_qal_driver(d: Driver) {
    if let Ok(mut lock) = QAL_DRIVER.write() {
        *lock = Some(d);
    }
}

/// True if a driver for the self-hosted compiler module has been installed.
pub fn has_qal_driver() -> bool {
    QAL_DRIVER.read().map(|l| l.is_some()).unwrap_or(false)
}

/// Invokes the configured QAL compiler driver (or the Rust reference server if none is set) on a raw `QALQ` request.
pub fn call_qal(req: &[u8]) -> Result<Vec<u8>, Diagnostic> {
    call_driver(req)
}

fn call_driver(req: &[u8]) -> Result<Vec<u8>, Diagnostic> {
    if let Ok(guard) = QAL_DRIVER.read() {
        if let Some(ref d) = *guard {
            return d(req);
        }
    }
    #[cfg(feature = "reference")]
    return Ok(proto::serve(req));
    #[cfg(not(feature = "reference"))]
    Err(Diagnostic::compiler_failed("qal", "DRIVER", "main", "no QAL compiler driver has been registered"))
}

/// The self-hosted QAL compiler backend running through `QALQ`/`QALR` protocol.
#[derive(Debug, Clone, Copy, Default)]
pub struct Qal;

impl Backend for Qal {
    fn name(&self) -> &'static str {
        "qal"
    }

    fn parse(&self, src: &str) -> Result<Program, Diagnostic> {
        let req = proto::Request::Parse { text: src.to_string() }.encode();
        let resp_bytes = call_driver(&req)?;
        match proto::Response::decode(&resp_bytes) {
            Ok(proto::Response::Done(payload)) => proto::decode_program(&payload).map_err(|e| Diagnostic::compiler_failed("qal", "PARSE", "main", &e)),
            Ok(proto::Response::Diagnostics(ds)) => Err(ds.into_iter().next().unwrap_or_else(|| Diagnostic::compiler_failed("qal", "PARSE", "main", "empty diagnostics"))),
            Ok(proto::Response::Internal(msg)) => Err(Diagnostic::compiler_failed("qal", "PARSE", "main", &msg)),
            Ok(proto::Response::Fork { .. }) => Err(Diagnostic::compiler_failed("qal", "PARSE", "main", "unexpected fork")),
            Err(e) => Err(Diagnostic::bad_compiler_response(&e)),
        }
    }

    fn print(&self, p: &Program) -> String {
        qal_core::print_program(p)
    }

    fn print_expr(&self, e: &Expr) -> String {
        qal_core::print_expr(e)
    }

    fn check(&self, p: &Program, _is_new: &dyn Fn(&str) -> bool) -> Result<TProgram, Vec<Diagnostic>> {
        let text = self.print(p);
        let req = proto::Request::CheckFile {
            text,
            entry: None,
            options: proto::Options::default(),
        }
        .encode();
        let resp_bytes = call_driver(&req).map_err(|d| vec![d])?;
        match proto::Response::decode(&resp_bytes) {
            Ok(proto::Response::Done(payload)) => match proto::decode_checked(&payload) {
                Ok(proto::Checked::Typed(bytes)) => qal_core::tform::decode(&bytes).map_err(|e| vec![Diagnostic::compiler_failed("qal", "CHECK", "main", &e)]),
                Ok(proto::Checked::Module(_)) => Err(vec![Diagnostic::compiler_failed("qal", "CHECK", "main", "expected typed, got module")]),
                Err(e) => Err(vec![Diagnostic::bad_compiler_response(&e)]),
            },
            Ok(proto::Response::Diagnostics(ds)) => Err(ds),
            Ok(proto::Response::Internal(msg)) => Err(vec![Diagnostic::compiler_failed("qal", "CHECK", "main", &msg)]),
            Ok(proto::Response::Fork { .. }) => Err(vec![Diagnostic::compiler_failed("qal", "CHECK", "main", "unexpected fork")]),
            Err(e) => Err(vec![Diagnostic::bad_compiler_response(&e)]),
        }
    }

    fn hash(&self, p: &Program) -> Result<Hashed, HashError> {

        let st = proto::StoreState::default();
        let text = self.print(p);
        let req = proto::Request::Define { state: st, text, options: Default::default() }.encode();
        let resp_bytes = call_driver(&req).map_err(|d| HashError::Internal(d.message))?;
        match proto::Response::decode(&resp_bytes) {
            Ok(proto::Response::Done(payload)) => match proto::decode_prepared(&payload) {
                Ok(prep) => {
                    let mut defs = Vec::new();
                    for c in prep.canons {
                        defs.push(qal_core::canon::Canon {
                            name: c.name,
                            kind: c.kind,
                            hash: c.hash,
                            canonical: c.canonical,
                            signature: c.signature,
                            effects: c.effects,
                            local_names: c.local_names,
                            refs: c.refs,
                            deps: c.deps,
                            group: c.group,
                        });
                    }
                    let groups = prep.groups.into_iter().collect();
                    Ok(Hashed { defs, groups })
                }
                Err(e) => Err(HashError::Internal(e)),
            },
            Ok(proto::Response::Diagnostics(ds)) => {
                let msgs: Vec<String> = ds.into_iter().map(|d| format!("{}: {}", d.code, d.message)).collect();
                Err(HashError::Internal(format!("diagnostics from DEFINE: {}", msgs.join("; "))))
            }
            Ok(proto::Response::Internal(msg)) => Err(HashError::Internal(msg)),
            Ok(other) => Err(HashError::Internal(format!("unexpected response from DEFINE: {other:?}"))),
            Err(e) => Err(HashError::Internal(format!("cannot decode response from DEFINE: {e}"))),
        }
    }

    fn compile(&self, p: &TProgram, entry: &Entry, opts: &CompileOptions) -> Result<CompiledModule, String> {
        let st = proto::StoreState {
            entries: Vec::new(),
            groups: vec![(qal_core::canon::Hash([0; 32]), qal_core::tform::encode(p))],
        };
        let req = proto::Request::Build {
            state: st,
            entry: entry.clone(),
            options: proto::Options { memory_max_pages: opts.memory_max_pages },
        }
        .encode();
        let resp_bytes = call_driver(&req).map_err(|d| d.message)?;
        match proto::Response::decode(&resp_bytes) {
            Ok(proto::Response::Done(payload)) => proto::decode_module(&payload),
            Ok(proto::Response::Internal(msg)) => Err(msg),
            _ => Err("bad response from BUILD".into()),
        }
    }
}

/// The differential `both` backend running Rust and QAL in parallel, comparing outputs byte for byte.
#[cfg(feature = "reference")]
#[derive(Debug, Clone, Copy, Default)]
pub struct Both;

#[cfg(feature = "reference")]
impl Backend for Both {
    fn name(&self) -> &'static str {
        "both"
    }

    fn parse(&self, src: &str) -> Result<Program, Diagnostic> {
        let r1 = Rust.parse(src);
        let r2 = Qal.parse(src);
        match (&r1, &r2) {
            (Ok(p1), Ok(p2)) if p1 == p2 => r1,
            (Err(d1), Err(d2)) if (d1.code, d1.kind) == (d2.code, d2.kind) => r1,
            _ => Err(Diagnostic::compiler_divergence("PARSE", &format!("Rust: {r1:?}, Qal: {r2:?}"))),
        }
    }

    fn print(&self, p: &Program) -> String {
        let s1 = Rust.print(p);
        let s2 = Qal.print(p);
        if s1 != s2 {
            eprintln!("compiler divergence in print:\n--- Rust:\n{s1}\n--- Qal:\n{s2}");
        }
        s1
    }

    fn print_expr(&self, e: &Expr) -> String {
        let s1 = Rust.print_expr(e);
        let s2 = Qal.print_expr(e);
        if s1 != s2 {
            eprintln!("compiler divergence in print_expr:\n--- Rust:\n{s1}\n--- Qal:\n{s2}");
        }
        s1
    }

    fn check(&self, p: &Program, is_new: &dyn Fn(&str) -> bool) -> Result<TProgram, Vec<Diagnostic>> {
        let r1 = Rust.check(p, is_new);
        let r2 = Qal.check(p, is_new);
        match (&r1, &r2) {
            (Ok(tp1), Ok(tp2)) => {
                let b1 = qal_core::tform::encode(tp1);
                let b2 = qal_core::tform::encode(tp2);
                if b1 != b2 {
                    return Err(vec![Diagnostic::compiler_divergence("CHECK", "Rust and Qal produced different QALT bytes")]);
                }
                r1
            }
            (Err(ds1), Err(ds2)) => {
                let c1: Vec<(&str, &str)> = ds1.iter().map(|d| (d.code, d.kind)).collect();
                let c2: Vec<(&str, &str)> = ds2.iter().map(|d| (d.code, d.kind)).collect();
                if c1 != c2 {
                    return Err(vec![Diagnostic::compiler_divergence("CHECK", &format!("diagnostics mismatch: Rust={c1:?}, Qal={c2:?}"))]);
                }
                r1
            }
            _ => Err(vec![Diagnostic::compiler_divergence("CHECK", &format!("Rust: {r1:?}, Qal: {r2:?}"))]),
        }
    }

    fn hash(&self, p: &Program) -> Result<Hashed, HashError> {
        let h1 = Rust.hash(p)?;
        let h2 = Qal.hash(p)?;
        if h1 != h2 {
            return Err(HashError::Internal("compiler divergence in hash".into()));
        }
        Ok(h1)
    }

    fn compile(&self, p: &TProgram, entry: &Entry, opts: &CompileOptions) -> Result<CompiledModule, String> {
        let m1 = Rust.compile(p, entry, opts)?;
        let m2 = Qal.compile(p, entry, opts)?;
        if m1.wasm != m2.wasm
            || m1.funcs != m2.funcs
            || m1.types != m2.types
            || m1.imports != m2.imports
            || m1.table_len != m2.table_len
            || m1.memory_max_pages != m2.memory_max_pages
        {
            return Err("compiler divergence in compile".into());
        }
        Ok(m1)
    }
}

#[cfg(feature = "reference")]
static RUST: Rust = Rust;
static QAL: Qal = Qal;
#[cfg(feature = "reference")]
static BOTH: Both = Both;

#[cfg(feature = "reference")]
static CURRENT_BACKEND: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0); // 0=Rust, 1=Qal, 2=Both
#[cfg(not(feature = "reference"))]
#[cfg(feature = "reference")]
static CURRENT_BACKEND: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0); // 0=Rust, 1=Qal, 2=Both
#[cfg(not(feature = "reference"))]
static CURRENT_BACKEND: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(1); // 1=Qal

/// Sets the default backend for free functions (`rust`, `qal`, `both`).
pub fn set_backend(name: &str) -> Result<(), String> {
    match name {
        #[cfg(feature = "reference")]
        "rust" => { CURRENT_BACKEND.store(0, std::sync::atomic::Ordering::SeqCst); Ok(()) }
        "qal" => { CURRENT_BACKEND.store(1, std::sync::atomic::Ordering::SeqCst); Ok(()) }
        #[cfg(feature = "reference")]
        "both" => { CURRENT_BACKEND.store(2, std::sync::atomic::Ordering::SeqCst); Ok(()) }
        #[cfg(not(feature = "reference"))]
        "rust" | "both" => Err(format!("compiler backend `{name}` is not available without feature `reference`")),
        _ => Err(format!("unknown compiler backend `{name}`; expected rust, qal or both")),
    }
}

/// The backend the free functions use.
pub fn backend() -> &'static dyn Backend {
    if let Ok(env) = std::env::var("QAL_COMPILER") {
        match env.as_str() {
            "qal" => return &QAL,
            #[cfg(feature = "reference")]
            "both" => return &BOTH,
            #[cfg(feature = "reference")]
            "rust" => return &RUST,
            _ => {}
        }
    }
    match CURRENT_BACKEND.load(std::sync::atomic::Ordering::SeqCst) {
        #[cfg(feature = "reference")]
        0 => &RUST,
        1 => &QAL,
        #[cfg(feature = "reference")]
        2 => &BOTH,
        #[cfg(feature = "reference")]
        _ => &RUST,
        #[cfg(not(feature = "reference"))]
        _ => &QAL,
    }
}

/// The backend named `name` (`rust`, `qal`, `both`).
pub fn select(name: &str) -> Result<&'static dyn Backend, String> {
    match name {
        #[cfg(feature = "reference")]
        "rust" => Ok(&RUST),
        "qal" => Ok(&QAL),
        #[cfg(feature = "reference")]
        "both" => Ok(&BOTH),
        #[cfg(not(feature = "reference"))]
        "rust" | "both" => Err(format!("compiler backend `{name}` is not available without feature `reference`")),
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
    #[allow(unused_imports)]
    use super::*;

    #[test]
    #[cfg(feature = "reference")]
    fn the_rust_backend_is_the_reference_pipeline() {
        let src = "fn f(x: Int) -> Int = x + 1\ntest t = f(1) == 2";
        let rust = select("rust").unwrap();
        let p = rust.parse(src).unwrap();
        assert_eq!(p, qal_text::parse_program(src).unwrap());
        assert_eq!(rust.print(&p), qal_text::print_program(&p));
        let tp = rust.check(&p, &|_| true).unwrap();
        assert_eq!(tp, qal_check::check(&p).unwrap());
        assert_eq!(rust.hash(&p).unwrap(), qal_core::canon::hash_program(&p).unwrap());
        assert_eq!(rust.compile(&tp, &Entry::All, &CompileOptions::default()).unwrap().wasm, qal_wasm::compile(&tp, &Entry::All).unwrap().wasm);
        assert_eq!(select("rust").map(|b| b.name()), Ok("rust"));
        assert_eq!(select("qal").map(|b| b.name()), Ok("qal"));
        assert_eq!(select("both").map(|b| b.name()), Ok("both"));
        assert!(select("x").err().is_some_and(|e| e.contains("unknown")));
    }

    #[test]
    #[cfg(feature = "reference")]
    fn qal_and_both_backends_match_reference() {
        let src = "fn add(x: Int, y: Int) -> Int = x + y\ntest t = add(2, 3) == 5";
        let rust = select("rust").unwrap();
        let qal = select("qal").unwrap();
        let both = select("both").unwrap();

        let p_rust = rust.parse(src).unwrap();
        let p_qal = qal.parse(src).unwrap();
        let p_both = both.parse(src).unwrap();
        assert_eq!(p_rust, p_qal);
        assert_eq!(p_rust, p_both);

        assert_eq!(rust.print(&p_rust), qal.print(&p_qal));
        assert_eq!(rust.print(&p_rust), both.print(&p_both));

        assert_eq!(rust.hash(&p_rust).unwrap(), qal.hash(&p_qal).unwrap());
        assert_eq!(rust.hash(&p_rust).unwrap(), both.hash(&p_both).unwrap());
    }

    #[test]
    fn host_crates_depend_only_on_the_facade() {
        let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        for host in ["qal-store", "qal-run", "qal-prove", "qal-serve", "qal-cli"] {
            let manifest = std::fs::read_to_string(crates.join(host).join("Cargo.toml")).unwrap();
            assert!(manifest.contains("qal-compiler = { workspace = true"), "{host}");
            for compiler in ["qal-text", "qal-check", "qal-wasm"] {
                assert!(!manifest.contains(compiler), "{host} depends on {compiler} directly");
            }
        }
    }

    #[test]
    #[cfg(feature = "reference")]
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
