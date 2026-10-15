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
            TKind::Seq(items) => match items.split_last() {
                Some((last, init)) => {
                    for x in init {
                        self.expr(x, false);
                        self.a.i(I::Drop);
                    }
                    self.expr(last, tail);
                }
                // An empty sequence is `()` (the checker types it as Unit).
                None => self.a.i(I::I32Const(0)),
            },
            TKind::Call(i, args) => {
                for x in args {
                    self.expr(x, false);
                }
                match self.fn_index.get(*i).copied().flatten() {
                    Some(f) => self.a.i(if tail { I::ReturnCall(f) } else { I::Call(f) }),
                    None => {
                        self.m.bug(format!("callee {i} is not reachable"));
                        self.a.i(I::Unreachable);
                    }
                }
            }
            TKind::CallIndirect(f, args) => {
                for x in args {
                    self.expr(x, false);
                }
                self.expr(f, false);
                let Type::Fn(ps, r, _) = &f.ty else {
                    self.m.bug(format!("indirect callee of type {} is not a function", f.ty));
                    self.a.i(I::Unreachable);
                    return;
                };
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
            TKind::Record(fs) => {
                let p = self.temp(Repr::I32);
                self.alloc(8 * fs.len() as i64, p);
                // Evaluated in source order, each stored at its declared slot.
                for (i, x) in fs {
                    self.a.i(I::LocalGet(p));
                    self.expr(x, false);
                    self.a.opt(to_slot(Repr::of(&x.ty)));
                    self.a.i(I::I64Store(m64(8 * *i as u64)));
                }
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
        let Some(wasm) = self.fn_index.get(fn_idx).copied().flatten() else {
            self.m.bug(format!("referenced function {fn_idx} is not reachable"));
            return 0;
        };
        self.fn_index.iter().flatten().filter(|w| **w < wasm).count() as u32
    }

    /// Element representation of a list type.
    fn elem(&self, t: &Type) -> Repr {
        match t {
            Type::List(x) => Repr::of(x),
            other => {
                self.m.bug(format!("list built-in applied to {other}"));
                Repr::I64
            }
        }
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
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge if l.ty == Type::Str => {
                // byte-wise comparison of UTF-8 = comparison by Unicode scalar values
                self.call_rt(Rt::StrCmp);
                self.a.i(I::I32Const(0));
                self.a.i(match op {
                    BinOp::Lt => I::I32LtS,
                    BinOp::Le => I::I32LeS,
                    BinOp::Gt => I::I32GtS,
                    _ => I::I32GeS,
                });
            }
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
            BinOp::And | BinOp::Or => self.m.bug("`&&`/`||` reached the strict operators"),
        }
    }

    fn builtin(&mut self, b: Builtin, args: &[TExpr], path: &str, ret: &Type) {
        use Builtin as B;
        for x in args {
            self.expr(x, false);
        }
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
            B::Map => self.call_rt(Rt::Map(self.elem(&args[0].ty), self.elem(ret))),
            B::Filter => self.call_rt(Rt::Filter(self.elem(&args[0].ty))),
            B::Fold => self.call_rt(Rt::Fold(Repr::of(ret), self.elem(&args[0].ty))),
            B::Range => self.call_rt(Rt::ListRange),
            B::Reverse => self.call_rt(Rt::ListReverse),
            B::SortInt => self.call_rt(Rt::ListSort(false)),
            B::SortStr => self.call_rt(Rt::ListSort(true)),
            B::CharCode => {
                let site = self.site(path, "char_code: req s != \"\"", &[("s", Type::Str)], None);
                self.site_args(site);
                self.call_rt(Rt::CharCode);
            }
            B::FromCharCode => {
                let site = self.site(path, "from_char_code: req 0 <= n <= 1114111 && !(55296 <= n <= 57343)", &[("n", Type::Int)], None);
                self.site_args(site);
                self.call_rt(Rt::FromCharCode);
            }
            B::ToUpper => self.call_rt(Rt::StrCase(true)),
            B::ToLower => self.call_rt(Rt::StrCase(false)),
            B::Replace => {
                let site = self.site(path, "replace: req from != \"\"", &[("from", Type::Str)], None);
                self.site_args(site);
                self.call_rt(Rt::StrReplace);
            }
            B::IndexOf => self.call_rt(Rt::StrIndexOf),
            B::IntToFloat => self.a.i(I::F64ConvertI64S),
            B::FloatToInt => self.call_rt(Rt::FloatToInt),
            B::Print => self.a.all([I::Call(self.m.import(host::PRINT)), I::I32Const(0)]),
            B::ReadLine => {
                self.a.i(I::Call(self.m.import(host::READ_LINE)));
                self.call_rt(Rt::OptFromPtr);
            }
            B::ReadAll => self.a.i(I::Call(self.m.import(host::READ_ALL))),
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
