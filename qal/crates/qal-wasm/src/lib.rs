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
