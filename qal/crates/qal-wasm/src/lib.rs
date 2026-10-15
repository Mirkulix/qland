//! QAL -> WebAssembly code generation from the checked, typed program (`qal_check::check`).
//!
//! Module layout:
//!   imports   `qal_host.*` (see [`host`]): `fail` always; effect functions only if code reachable
//!             from the entry uses them (so a pure program has no system access at all)
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
    pub use wasm_encoder::ValType;
    use ValType::{F64, I32, I64};

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
    /// Every effect host function.
    pub const EFFECT_FNS: [&str; 6] = [PRINT, READ_LINE, READ_FILE, WRITE_FILE, NOW_MS, RAND_INT];

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
