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
