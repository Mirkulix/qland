//! Type inference for one definition: expressions, patterns, and the final resolution pass.

use crate::exhaust::missing_pattern;
use crate::unify::{Subst, MAX_TYPE_DEPTH, MAX_TYPE_SIZE};
use crate::{resolve_type, Globals, Sig};
use qal_core::ast::*;
use qal_core::diag::Diagnostic;
use qal_core::tast::*;
use std::collections::BTreeSet;

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
    /// Type parameters of the definition (rigid inside it).
    pub tps: &'a [String],
    /// Effects used by the bodies of the closures being checked, innermost last: inside a closure,
    /// effects are collected into its type instead of being checked against `effects`.
    frames: Vec<BTreeSet<String>>,
    /// Declared result type while checking a fn body (for effect hints).
    pub ret: Option<&'a Type>,
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
        Cx { g, def, effects, tps: &[], frames: Vec::new(), ret: None, diags, start, sub: Subst::default(), locals: Vec::new(), scope: Vec::new(), deferred: Vec::new() }
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
        if std::mem::take(&mut self.sub.too_deep) {
            self.diags.push(Diagnostic::type_too_deep(self.def, path, MAX_TYPE_DEPTH));
            return;
        }
        if std::mem::take(&mut self.sub.too_large) {
            self.diags.push(Diagnostic::type_too_large(self.def, path, MAX_TYPE_SIZE));
            return;
        }
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
        // A closure checked against a known function type takes its parameter types from it
        // (bidirectional), so `fn(x) => x * 2.0` where `fn(Float) -> Float` is expected works.
        // (Kept out of line: `expect` is on the recursion path of every nested expression.)
        let t = match e {
            Expr::Lambda(params, body) => self.lambda_against(params, body, want, path),
            _ => self.infer(e, path),
        };
        self.unify_at(path, want, &t.ty);
        t
    }

    #[inline(never)]
    fn lambda_against(&mut self, params: &[(String, Option<Type>)], body: &Expr, want: &Type, path: &str) -> TExpr {
        match self.sub.resolve(want) {
            Type::Fn(ps, _, _) if ps.len() == params.len() => self.lambda(params, body, path, Some(&ps)),
            _ => self.lambda(params, body, path, None),
        }
    }

    pub fn effect_check(&mut self, path: &str, callee: &str, needed: &[String]) {
        if let Some(frame) = self.frames.last_mut() {
            frame.extend(needed.iter().cloned());
            return;
        }
        let missing: Vec<String> = needed.iter().filter(|e| !self.effects.contains(e)).cloned().collect();
        if missing.is_empty() {
            return;
        }
        let mut d = Diagnostic::effect_violation(self.def, path, callee, &missing);
        if let Some(ret @ Type::Fn(..)) = self.ret {
            // `-> fn(..) -> T !e` would give the effect to the returned function: show the parenthesized form.
            let effs: String = self.effects.iter().chain(&missing).map(|e| format!(" !{e}")).collect();
            d.hints = vec![format!("add `!{}` to the signature of `{}`: `-> ({ret}){effs}`", missing.join(" !"), self.def)];
        }
        self.diags.push(d);
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
                let ann = ann.as_ref().and_then(|t| resolve_type(self.g, t, self.tps, self.def, &format!("{path}.type"), self.diags));
                let v = match ann {
                    Some(t) => self.expect(v, &t, &format!("{path}.value")),
                    None => self.infer(v, &format!("{path}.value")),
                };
                // Even if the value had errors, the body is checked with the (possibly unknown) type.
                let id = self.bind(name, v.ty.clone());
                let b = self.infer(body, &format!("{path}.body"));
                self.unbind();
                (b.ty.clone(), TKind::Let(id, Box::new(v), Box::new(b)))
            }
            Expr::List(items) => {
                let elem = self.sub.fresh();
                let ty = self.sub.wrap(Type::list(elem.clone()));
                let items: Vec<TExpr> = items.iter().enumerate().map(|(i, x)| self.expect(x, &elem, &format!("{path}.items[{i}]"))).collect();
                (ty, TKind::List(items))
            }
            // Only built by hand (the parser needs at least one item): evaluates to `()`.
            Expr::Seq(items) if items.is_empty() => (Type::Unit, TKind::Unit),
            Expr::Seq(items) => {
                let n = items.len();
                let mut out = Vec::new();
                for (i, x) in items.iter().enumerate() {
                    let p = format!("{path}.items[{i}]");
                    let t = self.infer(x, &p);
                    if i + 1 < n && !self.sub.unify(&Type::Unit, &t.ty) {
                        self.mismatch(&p, &Type::Unit, &t.ty);
                        if let Some(d) = self.diags.last_mut() {
                            d.hints = vec!["only Unit values can be discarded in `{ a; b }`; bind the value with `let` or handle it with `match`".into()];
                        }
                    }
                    out.push(t);
                }
                (out.last().map_or(Type::Unit, |t| t.ty.clone()), TKind::Seq(out))
            }
            Expr::Record(name, fields) => return self.record(name, fields, path),
            Expr::Field(target, field) => return self.field(target, field, path),
            Expr::Match(s, arms) => return self.match_expr(s, arms, path),
            Expr::Lambda(params, body) => return self.lambda(params, body, path, None),
        };
        TExpr { ty, kind }
    }

    /// `fn(params) => body`: parameter types are annotated or inferred; the body's effects become
    /// the effects of the closure's type (they are checked where the closure is called).
    pub(crate) fn lambda(&mut self, params: &[(String, Option<Type>)], body: &Expr, path: &str, expected: Option<&[Type]>) -> TExpr {
        let depth = self.scope.len();
        let (mut ids, mut ptys) = (Vec::new(), Vec::new());
        for (i, (n, ann)) in params.iter().enumerate() {
            let pp = format!("{path}.params[{i}]");
            if params[..i].iter().any(|(m, _)| m == n) {
                self.diags.push(Diagnostic::duplicate_name(self.def, &pp, n));
            }
            let t = match (ann, expected) {
                (Some(a), _) => resolve_type(self.g, a, self.tps, self.def, &pp, self.diags).unwrap_or_else(|| self.sub.fresh()),
                // The closure's parameter takes the expected parameter type; effect subtyping is
                // still checked when the closure's type is unified with the expected one.
                (None, Some(ps)) => ps[i].clone(),
                (None, None) => self.sub.fresh(),
            };
            ids.push(self.bind(n, t.clone()));
            ptys.push(t);
        }
        self.frames.push(BTreeSet::new());
        let saved_ret = self.ret.take();
        let body = self.infer(body, &format!("{path}.body"));
        self.ret = saved_ret;
        let effs: Vec<String> = self.frames.pop().unwrap_or_default().into_iter().collect();
        self.scope.truncate(depth);
        let ty = self.sub.wrap(Type::Fn(ptys, Box::new(body.ty.clone()), effs));
        TExpr { ty, kind: TKind::Lambda(Box::new(TLambda { params: ids, captures: Vec::new(), body })) }
    }

    /// Fresh type arguments for a generic signature: (type arguments, parameter types, result type).
    fn instantiate(&mut self, sig: &Sig) -> (Vec<Type>, Vec<Type>, Type) {
        let targs: Vec<Type> = sig.tparams.iter().map(|_| self.sub.fresh()).collect();
        let params = sig.params.iter().map(|p| self.sub.wrap(p.instantiate(&sig.tparams, &targs))).collect();
        let ret = self.sub.wrap(sig.ret.instantiate(&sig.tparams, &targs));
        (targs, params, ret)
    }

    /// The type of a declared type applied to fresh arguments, with its body instantiated.
    fn fresh_decl(&mut self, name: &str) -> Option<(Type, TypeBody)> {
        let d = self.g.type_def(name)?;
        if d.params.is_empty() {
            return Some((Type::Named(name.into()), d.body.clone()));
        }
        let args: Vec<Type> = d.params.iter().map(|_| self.sub.fresh()).collect();
        let ty = Type::App(name.into(), args);
        let body = self.g.tp.decl(&ty)?;
        Some((self.sub.wrap(ty), body))
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

    /// The type of an ordered operand (`<` `<=` `>` `>=`): Int, Float or Str (unknown defaults to Int).
    fn ordered(&mut self, path: &str, t: &Type) -> Type {
        match self.sub.resolve(t) {
            Type::Str => Type::Str,
            Type::Int | Type::Float | Type::Var(_) => self.numeric(path, t),
            other => {
                self.diags.push(Diagnostic::type_mismatch(self.def, path, "Int, Float or Str", &self.sub.finish(&other).to_string()));
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
                // An ordered operand whose type is still unknown takes the other side's (`x < "a"`).
                let left_unknown = matches!(self.sub.resolve(&l.ty), Type::Var(_)) && !matches!(self.sub.resolve(&r.ty), Type::Var(_));
                let t = if !op.is_comparison() {
                    self.numeric(&lp, &l.ty)
                } else if left_unknown {
                    let t = self.ordered(&rp, &r.ty);
                    self.unify_at(&lp, &t, &l.ty);
                    t
                } else {
                    self.ordered(&lp, &l.ty)
                };
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
            let (idx, sig) = (*idx, sig.clone());
            if sig.tparams.is_empty() {
                return TExpr { ty: sig.fn_type(), kind: TKind::FnRef(idx) };
            }
            let (targs, params, ret) = self.instantiate(&sig);
            let ty = self.sub.wrap(Type::Fn(params, Box::new(ret), sig.effects.clone()));
            return TExpr { ty, kind: TKind::FnRefG(idx, targs) };
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
            d.message = format!("built-in `{v}` is not a value");
            d = d.hint(format!("built-in `{v}` cannot be used as a value; wrap it in a fn, e.g. `fn my_{v}(...) = {v}(...)`, and pass `my_{v}`"));
        } else {
            d = d.hint("not a built-in or stored definition; built-ins are listed in the qal_define description; define it in the same call or check qal_query");
        }
        self.diags.push(d);
        self.poison()
    }

    /// Looks up a variant by name: (its sum type, with fresh variables for Option/Result), tag, payload types.
    fn variant(&mut self, name: &str) -> Option<(Type, u32, Vec<Type>)> {
        let (a, b) = (self.sub.fresh(), self.sub.fresh());
        Some(match name {
            "None" => (self.sub.wrap(Type::option(a)), TAG_NONE, vec![]),
            "Some" => (self.sub.wrap(Type::option(a.clone())), TAG_SOME, vec![a]),
            "Ok" => (self.sub.wrap(Type::result(a.clone(), b)), TAG_OK, vec![a]),
            "Err" => (self.sub.wrap(Type::result(a, b.clone())), TAG_ERR, vec![b]),
            _ => {
                let (enum_name, tag, _) = self.g.variants.get(name)?.clone();
                let (ty, body) = self.fresh_decl(&enum_name)?;
                let TypeBody::Enum(vs) = body else { return None };
                (ty, tag, vs[tag as usize].fields.clone())
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
            if !sig.tparams.is_empty() {
                let (targs, params, ret) = self.instantiate(&sig);
                let args = self.args(name, args, &params, path);
                self.effect_check(path, name, &sig.effects);
                return TExpr { ty: ret, kind: TKind::CallG(idx, targs, args) };
            }
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
        self.diags.push(Diagnostic::unknown_name(self.def, path, name).hint("not a built-in or stored definition; built-ins are listed in the qal_define description; define it in the same call or check qal_query"));
        self.infer_all(args, path);
        self.poison()
    }

    fn record(&mut self, name: &str, fields: &[(String, Expr)], path: &str) -> TExpr {
        let (rty, decl) = match self.fresh_decl(name) {
            Some((ty, TypeBody::Record(fs))) => (ty, fs),
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
        let mut out: Vec<(u32, TExpr)> = Vec::new();
        for (f, e) in fields {
            let p = format!("{path}.fields.{f}");
            match decl.iter().position(|(n, _)| n == f) {
                Some(i) if out.iter().any(|(k, _)| *k == i as u32) => {
                    self.diags.push(Diagnostic::duplicate_name(self.def, &p, f));
                    self.infer(e, &p);
                }
                Some(i) => out.push((i as u32, self.expect(e, &decl[i].1, &p))),
                None => {
                    let names: Vec<&str> = decl.iter().map(|(n, _)| n.as_str()).collect();
                    self.diags.push(Diagnostic::unknown_name(self.def, &p, f).hint(format!("fields of `{name}`: {}", names.join(", "))));
                    self.infer(e, &p);
                }
            }
        }
        for (i, (f, _)) in decl.iter().enumerate() {
            if !out.iter().any(|(k, _)| *k == i as u32) {
                self.diags.push(Diagnostic::missing_field(self.def, path, name, f));
            }
        }
        TExpr { ty: rty, kind: TKind::Record(out) }
    }

    fn field(&mut self, target: &Expr, field: &str, path: &str) -> TExpr {
        let t = self.infer(target, &format!("{path}.target"));
        let ty = self.sub.resolve(&t.ty);
        let fields = match self.g.tp.decl(&ty) {
            Some(TypeBody::Record(fs)) => Some(fs),
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
                let (rty, decl) = match self.fresh_decl(name) {
                    Some((rty, TypeBody::Record(decl))) => (rty, decl),
                    _ => {
                        self.diags.push(Diagnostic::unknown_name(self.def, path, name));
                        return TPat::Wild;
                    }
                };
                if !self.unify_at(path, ty, &rty) {
                    return TPat::Wild;
                }
                let mut out = Vec::new();
                let mut seen: Vec<&String> = Vec::new();
                for (f, p) in fs {
                    let fp = format!("{path}.fields.{f}");
                    if seen.contains(&f) {
                        self.diags.push(Diagnostic::duplicate_name(self.def, &fp, f));
                        continue;
                    }
                    seen.push(f);
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
                    if !matches!(t, Type::Str | Type::List(_) | Type::Bytes) {
                        self.diags.push(Diagnostic::type_mismatch(self.def, &path, "Str, Bytes or List[_]", &self.sub.finish(&t).to_string()));
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
            TKind::FnRefG(_, targs) => targs.iter_mut().for_each(|t| *t = self.sub.finish(t)),
            TKind::CallG(_, targs, xs) => {
                targs.iter_mut().for_each(|t| *t = self.sub.finish(t));
                xs.iter_mut().for_each(|x| self.zonk(x));
            }
            TKind::Lambda(l) => {
                self.zonk(&mut l.body);
                let mut used = BTreeSet::new();
                let mut bound: BTreeSet<u32> = l.params.iter().copied().collect();
                locals_of(&l.body, &mut used, &mut bound);
                l.captures = used.difference(&bound).copied().collect();
            }
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
            TKind::Call(_, xs) | TKind::List(xs) | TKind::Variant(_, xs) | TKind::Seq(xs) => {
                xs.iter_mut().for_each(|x| self.zonk(x));
            }
            TKind::Record(fs) => fs.iter_mut().for_each(|(_, x)| self.zonk(x)),
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

/// Locals a closure body uses (`used`, including the captures of nested closures) and binds itself (`bound`).
pub(crate) fn locals_of(e: &TExpr, used: &mut BTreeSet<u32>, bound: &mut BTreeSet<u32>) {
    fn pat(p: &TPat, bound: &mut BTreeSet<u32>) {
        match p {
            TPat::Bind(id) => {
                bound.insert(*id);
            }
            TPat::Variant(_, ps) => ps.iter().for_each(|p| pat(p, bound)),
            TPat::Record(fs) => fs.iter().for_each(|(_, p)| pat(p, bound)),
            _ => {}
        }
    }
    match &e.kind {
        TKind::Local(id) => {
            used.insert(*id);
        }
        TKind::Lambda(l) => {
            used.extend(l.captures.iter().copied());
        }
        TKind::Let(id, v, b) => {
            bound.insert(*id);
            locals_of(v, used, bound);
            locals_of(b, used, bound);
        }
        TKind::Match(s, arms) => {
            locals_of(s, used, bound);
            for a in arms {
                pat(&a.pat, bound);
                if let Some(g) = &a.guard {
                    locals_of(g, used, bound);
                }
                locals_of(&a.body, used, bound);
            }
        }
        _ => children(e).into_iter().for_each(|c| locals_of(c, used, bound)),
    }
}

/// The direct subexpressions of `e` (closure bodies excluded; patterns are not expressions).
pub fn children(e: &TExpr) -> Vec<&TExpr> {
    match &e.kind {
        TKind::Int(_) | TKind::Float(_) | TKind::Bool(_) | TKind::Str(_) | TKind::Unit | TKind::Local(_) | TKind::FnRef(_) | TKind::FnRefG(..) | TKind::Lambda(_) => vec![],
        TKind::Un(_, x, _) | TKind::Field(x, _) => vec![x],
        TKind::Bin(_, l, r, _) | TKind::Let(_, l, r) => vec![l, r],
        TKind::If(c, t, f) => vec![c, t, f],
        TKind::Call(_, xs) | TKind::CallG(_, _, xs) | TKind::List(xs) | TKind::Variant(_, xs) | TKind::Seq(xs) | TKind::Builtin(_, xs, _) => xs.iter().collect(),
        TKind::CallIndirect(f, xs) => std::iter::once(&**f).chain(xs.iter()).collect(),
        TKind::Record(fs) => fs.iter().map(|(_, x)| x).collect(),
        TKind::Match(s, arms) => std::iter::once(&**s).chain(arms.iter().flat_map(|a| a.guard.iter().chain([&a.body]))).collect(),
    }
}

/// Distinct declared types a type walk expands at most (see `contains_fn`).
pub(crate) const MAX_EXPANSIONS: usize = 256;

/// True if a value of type `t` can contain a function (`seen` guards recursive types).
pub(crate) fn contains_fn(tp: &TProgram, t: &Type, seen: &mut Vec<String>) -> bool {
    // Regular recursion has few distinct instances; more only arise from irregular types (E0212),
    // which are rejected anyway: answer conservatively instead of expanding forever.
    if seen.len() >= MAX_EXPANSIONS {
        return true;
    }
    match t {
        Type::Fn(..) => true,
        Type::List(x) | Type::Option(x) => contains_fn(tp, x, seen),
        Type::Result(a, b) | Type::Map(a, b) => contains_fn(tp, a, seen) || contains_fn(tp, b, seen),
        Type::Named(_) | Type::App(..) => {
            let key = t.to_string();
            if seen.contains(&key) {
                return false;
            }
            seen.push(key);
            match tp.decl(t) {
                Some(TypeBody::Record(fs)) => fs.iter().any(|(_, f)| contains_fn(tp, f, seen)),
                Some(TypeBody::Enum(vs)) => vs.iter().any(|v| v.fields.iter().any(|f| contains_fn(tp, f, seen))),
                None => false,
            }
        }
        _ => false,
    }
}
