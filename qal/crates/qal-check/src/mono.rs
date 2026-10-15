//! Monomorphisation: every generic function is compiled once per distinct list of type arguments
//! it is used with. Instances are generated on demand, starting from the monomorphic functions
//! and the tests; they are appended to `TProgram::fns` and named `f[Int, Str]`.
//!
//! Termination: a generic function may call the members of its own recursive group only with plain
//! type parameters (`polymorphic_recursion` rejects the rest), so every chain of instantiations is
//! finite. The total number of instances, the size and depth of type arguments and the expression nodes copied
//! into instances are capped as well ([`MAX_INSTANCES`], [`MAX_TYPE_SIZE`], [`MAX_TYPE_DEPTH`], [`MAX_INSTANCE_NODES`]),
//! which bounds the work for non-recursive nesting.

use crate::infer::{children, contains_fn};
use crate::unify::{MAX_TYPE_DEPTH, MAX_TYPE_SIZE};
use crate::MAX_INSTANCES;
use qal_core::ast::{BinOp, Type, TypeBody};
use qal_core::diag::Diagnostic;
use qal_core::tast::*;
use std::collections::HashMap;

/// Calls `f` on `e` and every expression inside it, closure bodies included.
fn visit<'e>(e: &'e TExpr, f: &mut dyn FnMut(&'e TExpr)) {
    f(e);
    if let TKind::Lambda(l) = &e.kind {
        visit(&l.body, f);
    }
    for c in children(e) {
        visit(c, f);
    }
}

fn fn_exprs(f: &TFn) -> impl Iterator<Item = &TExpr> {
    f.req.iter().chain(&f.ens).chain([&f.body])
}

fn show(targs: &[Type]) -> String {
    targs.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", ")
}

/// Nesting depth of `t` (called on types of at most `MAX_TYPE_SIZE` nodes).
fn depth(t: &Type) -> usize {
    1 + match t {
        Type::List(x) | Type::Option(x) => depth(x),
        Type::Result(a, b) | Type::Map(a, b) => depth(a).max(depth(b)),
        Type::App(_, args) => args.iter().map(depth).max().unwrap_or(0),
        Type::Fn(ps, r, _) => ps.iter().map(depth).max().unwrap_or(0).max(depth(r)),
        _ => 0,
    }
}

/// Rejects recursive calls of generic functions with type arguments other than plain type parameters.
pub(crate) fn polymorphic_recursion(tp: &TProgram, diags: &mut Vec<Diagnostic>) {
    let n = tp.generics.len();
    let mut calls: Vec<Vec<(usize, Vec<Type>)>> = vec![Vec::new(); n];
    for (g, f) in tp.generics.iter().enumerate() {
        for e in fn_exprs(f) {
            visit(e, &mut |x| match &x.kind {
                TKind::CallG(h, targs, _) | TKind::FnRefG(h, targs) => calls[g].push((*h, targs.clone())),
                _ => {}
            });
        }
    }
    let succ: Vec<Vec<usize>> = calls.iter().map(|cs| cs.iter().map(|(h, _)| *h).filter(|h| *h < n).collect()).collect();
    let comp = crate::scc(&succ);
    for (g, f) in tp.generics.iter().enumerate() {
        for (h, targs) in &calls[g] {
            let same_group = comp.get(*h).is_some_and(|c| *c == comp[g]);
            if same_group && !targs.iter().all(|t| matches!(t, Type::Param(_))) {
                diags.push(Diagnostic::polymorphic_recursion(&f.def.name, "", &tp.generics[*h].def.name, &show(targs)));
                return;
            }
        }
    }
}

/// Total expression nodes all instances may copy, each weighed by the size of the instance's type
/// arguments (bounds time and memory: an instance copies the generic body and substitutes the type
/// arguments into the type of every node, so 1000 instances of a large body, or a few at large
/// types, could otherwise take gigabytes).
pub(crate) const MAX_INSTANCE_NODES: usize = 500_000;

/// Expression nodes of a function (contracts, body, closure bodies).
fn fn_nodes(f: &TFn) -> usize {
    let mut n = 0;
    for e in fn_exprs(f) {
        visit(e, &mut |_| n += 1);
    }
    n
}

struct Mono<'a> {
    generics: &'a [TFn],
    /// Expression nodes of each generic function, and the total copied into instances so far.
    sizes: Vec<usize>,
    copied: usize,
    /// Index in `TProgram::fns` of the first instance.
    base: usize,
    instances: HashMap<(usize, Vec<Type>), usize>,
    new_fns: Vec<TFn>,
    queue: Vec<usize>,
    /// (def, instance name) when a limit was hit.
    failed: Option<(String, String)>,
}

impl Mono<'_> {
    fn instance(&mut self, caller: &str, g: usize, targs: &[Type]) -> usize {
        let key = (g, targs.to_vec());
        if let Some(i) = self.instances.get(&key) {
            return *i;
        }
        let gf = &self.generics[g];
        let name = format!("{}[{}]", gf.def.name, show(targs));
        // Every copied node carries a substituted type: weigh the nodes by the type arguments' size.
        let size = self.sizes.get(g).copied().unwrap_or(0).saturating_mul(1 + targs.iter().map(Type::size).sum::<usize>());
        // Inference keeps types within MAX_TYPE_DEPTH, but a chain of generic calls can nest type
        // arguments further at every step (below the size limit); code generation keeps per-type
        // helpers whose cost grows with the square of the depth.
        let too_big = |t: &Type| t.size() > MAX_TYPE_SIZE as usize || depth(t) > MAX_TYPE_DEPTH as usize;
        if self.instances.len() >= MAX_INSTANCES || targs.iter().any(too_big) || self.copied + size > MAX_INSTANCE_NODES {
            self.failed.get_or_insert((caller.to_string(), name));
            return 0;
        }
        let tps = &gf.def.tparams;
        let sub = |t: &Type| t.instantiate(tps, targs);
        let mut def = gf.def.clone();
        def.name = name;
        def.tparams = Vec::new();
        for (p, t) in def.params.iter_mut().zip(&gf.locals) {
            p.ty = sub(t);
        }
        def.ret = sub(&gf.body.ty);
        let f = TFn {
            def,
            locals: gf.locals.iter().map(sub).collect(),
            req: gf.req.iter().map(|e| subst_expr(e, tps, targs)).collect(),
            ens: gf.ens.iter().map(|e| subst_expr(e, tps, targs)).collect(),
            result_local: gf.result_local,
            body: subst_expr(&gf.body, tps, targs),
            instance: Some((g, targs.to_vec())),
        };
        self.copied += size;
        let k = self.new_fns.len();
        self.new_fns.push(f);
        self.queue.push(k);
        self.instances.insert(key, self.base + k);
        self.base + k
    }

    /// Replaces generic calls and references by calls of their instances.
    fn rewrite(&mut self, caller: &str, e: &mut TExpr) {
        match &mut e.kind {
            TKind::CallG(g, targs, args) => {
                for a in args.iter_mut() {
                    self.rewrite(caller, a);
                }
                let i = self.instance(caller, *g, targs);
                e.kind = TKind::Call(i, std::mem::take(args));
            }
            TKind::FnRefG(g, targs) => {
                let i = self.instance(caller, *g, targs);
                e.kind = TKind::FnRef(i);
            }
            TKind::Lambda(l) => self.rewrite(caller, &mut l.body),
            TKind::Un(_, x, _) | TKind::Field(x, _) => self.rewrite(caller, x),
            TKind::Bin(_, l, r, _) | TKind::Let(_, l, r) => {
                self.rewrite(caller, l);
                self.rewrite(caller, r);
            }
            TKind::If(c, t, f) => {
                self.rewrite(caller, c);
                self.rewrite(caller, t);
                self.rewrite(caller, f);
            }
            TKind::Call(_, xs) | TKind::List(xs) | TKind::Variant(_, xs) | TKind::Seq(xs) | TKind::Builtin(_, xs, _) => {
                xs.iter_mut().for_each(|x| self.rewrite(caller, x));
            }
            TKind::CallIndirect(f, xs) => {
                self.rewrite(caller, f);
                xs.iter_mut().for_each(|x| self.rewrite(caller, x));
            }
            TKind::Record(fs) => fs.iter_mut().for_each(|(_, x)| self.rewrite(caller, x)),
            TKind::Match(s, arms) => {
                self.rewrite(caller, s);
                for a in arms {
                    if let Some(g) = &mut a.guard {
                        self.rewrite(caller, g);
                    }
                    self.rewrite(caller, &mut a.body);
                }
            }
            TKind::Int(_) | TKind::Float(_) | TKind::Bool(_) | TKind::Str(_) | TKind::Unit | TKind::Local(_) | TKind::FnRef(_) => {}
        }
    }

    fn rewrite_fn(&mut self, f: &mut TFn) {
        let name = f.def.name.clone();
        for e in f.req.iter_mut().chain(f.ens.iter_mut()).chain([&mut f.body]) {
            self.rewrite(&name, e);
        }
    }
}

/// `e` with the type parameters `tps` replaced by `targs` in every type.
fn subst_expr(e: &TExpr, tps: &[String], targs: &[Type]) -> TExpr {
    let mut e = e.clone();
    subst_in(&mut e, tps, targs);
    e
}

fn subst_in(e: &mut TExpr, tps: &[String], targs: &[Type]) {
    e.ty = e.ty.instantiate(tps, targs);
    match &mut e.kind {
        TKind::CallG(_, ts, xs) => {
            ts.iter_mut().for_each(|t| *t = t.instantiate(tps, targs));
            xs.iter_mut().for_each(|x| subst_in(x, tps, targs));
        }
        TKind::FnRefG(_, ts) => ts.iter_mut().for_each(|t| *t = t.instantiate(tps, targs)),
        TKind::Lambda(l) => subst_in(&mut l.body, tps, targs),
        TKind::Un(_, x, _) | TKind::Field(x, _) => subst_in(x, tps, targs),
        TKind::Bin(_, l, r, _) | TKind::Let(_, l, r) => {
            subst_in(l, tps, targs);
            subst_in(r, tps, targs);
        }
        TKind::If(c, t, f) => {
            subst_in(c, tps, targs);
            subst_in(t, tps, targs);
            subst_in(f, tps, targs);
        }
        TKind::Call(_, xs) | TKind::List(xs) | TKind::Variant(_, xs) | TKind::Seq(xs) | TKind::Builtin(_, xs, _) => {
            xs.iter_mut().for_each(|x| subst_in(x, tps, targs));
        }
        TKind::CallIndirect(f, xs) => {
            subst_in(f, tps, targs);
            xs.iter_mut().for_each(|x| subst_in(x, tps, targs));
        }
        TKind::Record(fs) => fs.iter_mut().for_each(|(_, x)| subst_in(x, tps, targs)),
        TKind::Match(s, arms) => {
            subst_in(s, tps, targs);
            for a in arms {
                if let Some(g) = &mut a.guard {
                    subst_in(g, tps, targs);
                }
                subst_in(&mut a.body, tps, targs);
            }
        }
        TKind::Int(_) | TKind::Float(_) | TKind::Bool(_) | TKind::Str(_) | TKind::Unit | TKind::Local(_) | TKind::FnRef(_) => {}
    }
}

pub(crate) fn monomorphise(tp: &mut TProgram, diags: &mut Vec<Diagnostic>) {
    let generics = std::mem::take(&mut tp.generics);
    let mut fns = std::mem::take(&mut tp.fns);
    let sizes = generics.iter().map(fn_nodes).collect();
    let mut m = Mono { generics: &generics, sizes, copied: 0, base: fns.len(), instances: HashMap::new(), new_fns: Vec::new(), queue: Vec::new(), failed: None };
    for f in &mut fns {
        m.rewrite_fn(f);
    }
    for t in &mut tp.tests {
        let name = t.name.clone();
        m.rewrite(&name, &mut t.body);
    }
    while let Some(k) = m.queue.pop() {
        if m.failed.is_some() {
            break;
        }
        let mut f = m.new_fns[k].clone();
        m.rewrite_fn(&mut f);
        m.new_fns[k] = f;
    }
    if let Some((def, inst)) = m.failed.take() {
        diags.push(Diagnostic::instantiation_limit(&def, "", &inst, MAX_INSTANCES));
    }
    fns.extend(m.new_fns);
    tp.fns = fns;
    tp.generics = generics;
}

/// The first map key type inside `t` that is not comparable (contains a function), if any.
fn bad_map_key(tp: &TProgram, t: &Type, seen: &mut Vec<String>) -> Option<Type> {
    if seen.len() >= crate::infer::MAX_EXPANSIONS {
        return None;
    }
    match t {
        Type::Map(k, v) => {
            if contains_fn(tp, k, &mut Vec::new()) {
                return Some((**k).clone());
            }
            bad_map_key(tp, k, seen).or_else(|| bad_map_key(tp, v, seen))
        }
        Type::List(x) | Type::Option(x) => bad_map_key(tp, x, seen),
        Type::Result(a, b) => bad_map_key(tp, a, seen).or_else(|| bad_map_key(tp, b, seen)),
        Type::Fn(ps, r, _) => ps.iter().find_map(|p| bad_map_key(tp, p, seen)).or_else(|| bad_map_key(tp, r, seen)),
        Type::Named(_) | Type::App(..) => {
            let key = t.to_string();
            if seen.contains(&key) {
                return None;
            }
            seen.push(key);
            match tp.decl(t)? {
                TypeBody::Record(fs) => fs.iter().find_map(|(_, f)| bad_map_key(tp, f, seen)),
                TypeBody::Enum(vs) => vs.iter().flat_map(|v| v.fields.iter()).find_map(|f| bad_map_key(tp, f, seen)),
            }
        }
        _ => None,
    }
}

/// Checks that need concrete types: `==` inside instances (a type parameter instantiated with a
/// function type), and map keys everywhere (a key type inferred or instantiated as a function).
pub(crate) fn validate(tp: &TProgram, diags: &mut Vec<Diagnostic>) {
    let mut check = |def: &str, locals: &[Type], exprs: Vec<&TExpr>, instance: bool| {
        let mut seen = Vec::new();
        for t in locals {
            if let Some(k) = bad_map_key(tp, t, &mut seen) {
                diags.push(Diagnostic::type_mismatch(def, "", "a comparable key type", &k.to_string()).hint("map keys cannot contain functions"));
                return;
            }
        }
        for e in exprs {
            let mut found = None;
            visit(e, &mut |x| {
                if found.is_some() {
                    return;
                }
                if let Some(k) = bad_map_key(tp, &x.ty, &mut Vec::new()) {
                    found = Some(Diagnostic::type_mismatch(def, "", "a comparable key type", &k.to_string()).hint("map keys cannot contain functions"));
                } else if let TKind::Builtin(Builtin::Compare, args, path) = &x.kind {
                    if let Some(a) = args.first().filter(|a| contains_fn(tp, &a.ty, &mut Vec::new())) {
                        found = Some(Diagnostic::type_mismatch(def, path, "a comparable type", &a.ty.to_string()).hint("`compare` orders values without functions inside"));
                    }
                } else if let TKind::Bin(BinOp::Eq | BinOp::Ne, l, _, path) = &x.kind {
                    if instance && contains_fn(tp, &l.ty, &mut Vec::new()) {
                        found = Some(
                            Diagnostic::type_mismatch(def, path, "a comparable type", &l.ty.to_string())
                                .hint("functions cannot be compared; this generic function compares values of a type parameter"),
                        );
                    }
                }
            });
            if let Some(d) = found {
                diags.push(d);
                return;
            }
        }
    };
    for f in &tp.fns {
        check(&f.def.name, &f.locals, fn_exprs(f).collect(), f.instance.is_some());
    }
    for t in &tp.tests {
        check(&t.name, &t.locals, vec![&t.body], false);
    }
}
