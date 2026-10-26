//! WebAssembly module representations and compiler options (shared between host and compiler).

use crate::ast::{Expr, Type, TypeDef};

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
/// `todo()` was evaluated (a stub, Phase 4 §11).
pub const FAIL_TODO: i32 = 8;

/// Byte address of the slot holding the size of a failed allocation.
pub const OOM_SCRATCH: u32 = 0;
/// Byte address of the static variant-tag table.
pub const TAG_TABLE: u32 = 8;
/// Global index of the heap pointer.
pub const HEAP_GLOBAL: u32 = 0;
/// Maximum memory of user programs: 4096 pages of 64 KiB = 256 MiB.
pub const MAX_PAGES: u64 = 4096;
/// Maximum memory of compiler runs (self-hosting spec §6.3): 16 384 pages = 1 GiB.
pub const COMPILER_MAX_PAGES: u64 = 16384;
/// The largest `memory_max_pages` a module may declare.
pub const MAX_MEMORY_PAGES: u64 = COMPILER_MAX_PAGES;

/// Options of compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompileOptions {
    /// The module's memory maximum in 64 KiB pages: [`MAX_PAGES`] for user programs,
    /// [`COMPILER_MAX_PAGES`] for compiler runs; at most [`MAX_MEMORY_PAGES`].
    pub memory_max_pages: u64,
}

impl Default for CompileOptions {
    fn default() -> CompileOptions {
        CompileOptions { memory_max_pages: MAX_PAGES }
    }
}

impl CompileOptions {
    /// The options of compiler runs (`Limits::compiler()` in `qal-run`).
    pub fn compiler() -> CompileOptions {
        CompileOptions { memory_max_pages: COMPILER_MAX_PAGES }
    }
}

pub const HOST_MODULE: &str = "qal_host";

/// Host functions in module `qal_host`. Effect functions are named `<effect>.<name>`.
pub mod host {
    /// `fail(kind, func, site, vals_ptr)`: records a structured failure, then traps.
    pub const FAIL: &str = "fail";
    /// `(x: f64) -> Str` (pure; formatting floats is not worth hand-written wasm)
    pub const FLOAT_TO_STR: &str = "float_to_str";
    /// `(s: Str) -> (ok: i32, value: f64)` (pure)
    pub const STR_TO_FLOAT: &str = "str_to_float";
    /// `(b: Bytes) -> 1 if valid UTF-8, else 0` (pure)
    pub const UTF8_VALID: &str = "utf8_valid";
    /// `(b: Bytes) -> Bytes` of the 32-byte SHA-256 digest (pure; Phase 5a)
    pub const SHA256: &str = "sha256";
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
    /// `() -> Str ptr` of the rest of the input (empty at end of input)
    pub const READ_ALL: &str = "io.read_all";
    /// `(call: HttpCall ptr) -> Result[HttpResponse, NetError] ptr` (the host builds the result)
    pub const NET_HTTP: &str = "net.http";
    /// `(sql: Str ptr, params: List[SqlValue] ptr) -> Result[Int, DbError] ptr`
    pub const DB_EXEC: &str = "db.exec";
    /// `(sql: Str ptr, params: List[SqlValue] ptr) -> Result[DbRows, DbError] ptr`
    pub const DB_QUERY: &str = "db.query";
    /// `(prompt: Str ptr, schema: Str ptr) -> Result[ModelResponse, ModelError] ptr`
    pub const MODEL_ASK: &str = "model.ask";
    /// Every effect host function (the index is reported for a denied call: append only).
    pub const EFFECT_FNS: [&str; 11] = [PRINT, READ_LINE, READ_FILE, WRITE_FILE, NOW_MS, RAND_INT, READ_ALL, NET_HTTP, DB_EXEC, DB_QUERY, MODEL_ASK];

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ValType {
        I32,
        I64,
        F64,
    }

    /// Parameter and result types of a host function (None for an unknown name).
    pub fn signature(name: &str) -> Option<(&'static [ValType], &'static [ValType])> {
        use ValType::*;
        Some(match name {
            FAIL => (&[I32, I32, I32, I32], &[]),
            FLOAT_TO_STR => (&[F64], &[I32]),
            STR_TO_FLOAT => (&[I32], &[I32, F64]),
            UTF8_VALID | SHA256 => (&[I32], &[I32]),
            PRINT => (&[I32], &[]),
            NET_HTTP => (&[I32], &[I32]),
            DB_EXEC | DB_QUERY | MODEL_ASK => (&[I32, I32], &[I32]),
            READ_LINE | READ_ALL => (&[], &[I32]),
            READ_FILE => (&[I32], &[I32, I32]),
            WRITE_FILE => (&[I32, I32], &[I32, I32]),
            NOW_MS => (&[], &[I64]),
            RAND_INT => (&[I64, I64], &[I64]),
            _ => return None,
        })
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

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledModule {
    pub wasm: Vec<u8>,
    /// Indexed by the `func` argument of `qal_host.fail`.
    pub funcs: Vec<FuncMeta>,
    /// Declared types, for decoding values.
    pub types: Vec<TypeDef>,
    /// Host functions the module imports.
    pub imports: Vec<&'static str>,
    /// Elements of the module's function table (its minimum and maximum size).
    pub table_len: usize,
    /// The memory maximum the module declares, in 64 KiB pages.
    pub memory_max_pages: u64,
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
    /// Several functions (e.g. a service's handler and its init function).
    Fns(Vec<String>),
    /// All tests.
    Tests,
    /// All functions and tests.
    All,
}
