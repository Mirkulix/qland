//! QAL -> WebAssembly code generation from the checked, typed program (`qal_check::check`).
//!
//! Module layout:
//!   imports   `qal_host.*` (see [`host`]): `fail` always; effect functions only if code reachable
//!             from the entry uses them (so a pure program has no system access at all)
//!   functions user fns reachable from the entry, then tests, then runtime/equality helpers
//!             (all indices computed by the module builder)
//!   table     closure bodies and wrappers of fns used as values, for `call_indirect`
//!   memory    exported as `memory`, at most 4096 pages (256 MiB; compiler runs: 16 384 pages = 1 GiB,
//!             `CompileOptions::memory_max_pages`); bump allocator exported as `qal_alloc(size: i64) -> i32`
//!   data      0..8 OOM scratch slot, 8.. one block per variant tag (shared payload-free variants), then string literals
//!
//! Values: Int = i64, Float = f64, Bool/Unit = i32, everything else = i32 pointer.
//! Fn values are closures: a pointer to `[table slot: i32][pad][captured slots]`; a call passes the
//! closure pointer as an extra first argument (`env`). A top-level fn used as a value gets a static
//! closure whose slot holds a wrapper that ignores `env`.
//! Heap blocks: Str = [len: i32][bytes], List = [len: i32][pad][8-byte slots], variant = [tag: i32][pad][slots],
//! record = [slots]. Payload-free variants (`None`, `Empty`) point into the static tag table.
//! Bytes = like Str; Map = [len: i32][pad][(key slot, value slot)…] sorted by the key order
//! (`Mb::cmp_fn`), so equal maps have equal layouts.

mod asm;
mod gen;
mod module;
mod rt;

use gen::FnGen;
use module::Mb;
use qal_core::ast::Type;
use qal_core::tast::{Builtin, TExpr, TKind, TProgram};
use std::collections::BTreeSet;

pub use asm::Repr;
pub use qal_core::module::{
    CompileOptions, CompiledModule, Entry, FuncMeta, Site, COMPILER_MAX_PAGES, FAIL_BUILTIN, FAIL_DIV_ZERO,
    FAIL_ENS, FAIL_INDEX, FAIL_OOM, FAIL_OVERFLOW, FAIL_REQ, FAIL_TODO, HEAP_GLOBAL, HOST_MODULE, MAX_MEMORY_PAGES,
    MAX_PAGES, OOM_SCRATCH, TAG_TABLE,
};

/// Host functions in module `qal_host`. Effect functions are named `<effect>.<name>`.
pub mod host {
    pub use qal_core::module::host::*;
}

/// Compiles the code reachable from `entry`. `p` must have passed the checker; an `Err` is an
/// internal compiler error (a broken invariant), never a property of the program.
pub fn compile(p: &TProgram, entry: &Entry) -> Result<CompiledModule, String> {
    compile_with(p, entry, &CompileOptions::default())
}

/// [`compile`] with options (the memory maximum). An `Err` is an internal compiler error or an
/// option out of range.
pub fn compile_with(p: &TProgram, entry: &Entry, opts: &CompileOptions) -> Result<CompiledModule, String> {
    if opts.memory_max_pages == 0 || opts.memory_max_pages > MAX_MEMORY_PAGES {
        return Err(format!("memory_max_pages must be between 1 and {MAX_MEMORY_PAGES}, not {}", opts.memory_max_pages));
    }
    let (fn_roots, with_tests): (Vec<usize>, bool) = match entry {
        Entry::Fn(name) => (p.fn_index(name).into_iter().collect(), false),
        Entry::Fns(names) => (names.iter().filter_map(|n| p.fn_index(n)).collect(), false),
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
        let Some(f) = p.fns.get(i) else {
            return Err(format!("internal compiler error: reference to function {i}, which does not exist"));
        };
        if reached.insert(i) {
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
            Builtin::BytesToStr => host::UTF8_VALID,
            Builtin::Sha256 => host::SHA256,
            Builtin::Print => host::PRINT,
            Builtin::ReadLine => host::READ_LINE,
            Builtin::ReadAll => host::READ_ALL,
            Builtin::ReadFile => host::READ_FILE,
            Builtin::WriteFile => host::WRITE_FILE,
            Builtin::NowMs => host::NOW_MS,
            Builtin::RandInt => host::RAND_INT,
            Builtin::Http => host::NET_HTTP,
            Builtin::DbExec => host::DB_EXEC,
            Builtin::DbQuery => host::DB_QUERY,
            Builtin::ModelAsk => host::MODEL_ASK,
            Builtin::TensorMatmul => host::TENSOR_MATMUL,
            _ => continue,
        };
        imports.push(name);
    }

    let mut mb = Mb::new(p, &imports);
    mb.max_pages = opts.memory_max_pages;
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
        match fn_index[i] {
            Some(idx) => mb.define(idx, body),
            None => mb.bug(format!("function `{}` was not declared", f.def.name)),
        }
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

    let (wasm, table_len) = mb.finish(&exports)?;
    Ok(CompiledModule { wasm, funcs: metas, types: p.types.clone(), imports, table_len, memory_max_pages: opts.memory_max_pages })
}

/// Collects called/referenced functions and used built-ins.
fn walk(e: &TExpr, fns: &mut Vec<usize>, builtins: &mut BTreeSet<Builtin>) {
    match &e.kind {
        TKind::Int(_) | TKind::Float(_) | TKind::Bool(_) | TKind::Str(_) | TKind::Unit | TKind::Local(_) => {}
        TKind::FnRef(i) => fns.push(*i),
        TKind::Lambda(l) => walk(&l.body, fns, builtins),
        // Checked programs are monomorphic; code generation reports a generic node as a bug.
        TKind::CallG(..) | TKind::FnRefG(..) => {}
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
        TKind::List(xs) | TKind::Variant(_, xs) | TKind::Seq(xs) => xs.iter().for_each(|x| walk(x, fns, builtins)),
        TKind::Record(fs) => fs.iter().for_each(|(_, x)| walk(x, fns, builtins)),
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
