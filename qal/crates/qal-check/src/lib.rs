//! QAL checker: names, types, effects, contracts and match exhaustiveness.
//! Only checked programs may be compiled; `check` returns the typed program the code generator needs.

mod builtins;
mod exhaust;
mod infer;
mod mono;
mod unify;

use infer::Cx;
use qal_core::ast::*;
use qal_core::diag::Diagnostic;
use qal_core::tast::{Builtin, TFn, TProgram, TTest};
use std::collections::{HashMap, HashSet};

/// Effects known in Phase 1.
pub const KNOWN_EFFECTS: &[&str] = &["io", "fs", "time", "rand"];

pub use qal_core::ast::RESULT_VAR;

/// Names the compiled module exports itself; a user function cannot take them.
const RESERVED_EXPORTS: &[&str] = &["memory", "qal_alloc"];

/// Variant names that belong to the built-in `Option` and `Result`.
const BUILTIN_VARIANTS: &[&str] = &["None", "Some", "Ok", "Err"];

#[derive(Debug, Clone, PartialEq)]
pub struct Sig {
    /// Type parameters (empty: monomorphic). `params` and `ret` use `Type::Param` for them.
    pub tparams: Vec<String>,
    pub params: Vec<Type>,
    pub ret: Type,
    pub effects: Vec<String>,
}

impl Sig {
    pub fn fn_type(&self) -> Type {
        Type::Fn(self.params.clone(), Box::new(self.ret.clone()), self.effects.clone())
    }
}

/// Maximum number of instances of generic functions per program.
pub const MAX_INSTANCES: usize = 1000;

/// Program-wide names, known before any body is checked.
pub(crate) struct Globals {
    /// Declared types (for lookups); `fns`/`tests` stay empty.
    pub tp: TProgram,
    /// Function name -> (index into `TProgram::fns`, or into `TProgram::generics` if the
    /// signature has type parameters; signature).
    pub fns: HashMap<String, (usize, Sig)>,
    /// User variant name -> (enum name, tag, payload types; `Type::Param` for the enum's parameters).
    pub variants: HashMap<String, (String, u32, Vec<Type>)>,
    /// Type name -> index into `tp.types` (a linear search per lookup made checking quadratic).
    pub type_index: HashMap<String, usize>,
}

impl Globals {
    pub fn type_def(&self, name: &str) -> Option<&TypeDef> {
        self.type_index.get(name).and_then(|&i| self.tp.types.get(i))
    }
}

/// Checks a whole program. An empty result means the program may be compiled.
pub fn check_program(p: &Program) -> Vec<Diagnostic> {
    match check(p) {
        Ok(_) => Vec::new(),
        Err(d) => d,
    }
}

/// Checks a whole program and returns its typed form, or all diagnostics.
///
/// Generic functions are checked once, with their type parameters as rigid types, and kept in
/// `TProgram::generics`; every instance reachable from a monomorphic function or a test is then
/// generated (monomorphisation), so `TProgram::fns` and the tests contain no generic calls.
pub fn check(p: &Program) -> Result<TProgram, Vec<Diagnostic>> {
    let mut diags = Vec::new();
    let g = globals(p, &mut diags);
    // An irregular recursive type has infinitely many instances: nothing below may expand it.
    if diags.iter().any(|d| d.kind == "irregular_recursion") {
        return Err(diags);
    }
    let mut out = TProgram { types: g.tp.types.clone(), ..TProgram::default() };
    for d in &p.defs {
        match d {
            Def::Type(_) => {}
            Def::Fn(f) => {
                if let Some(tf) = check_fn(f, &g, &mut diags) {
                    if f.tparams.is_empty() {
                        out.fns.push(tf);
                    } else {
                        out.generics.push(tf);
                    }
                }
            }
            Def::Test(t) => {
                let mut cx = Cx::new(&g, &t.name, &[], &mut diags);
                let body = cx.expect(&t.body, &Type::Bool, "body");
                if let Some((locals, mut exprs)) = cx.finish(vec![body]) {
                    out.tests.push(TTest { name: t.name.clone(), locals, body: exprs.remove(0) });
                }
            }
        }
    }
    if diags.is_empty() {
        mono::polymorphic_recursion(&out, &mut diags);
    }
    if diags.is_empty() {
        mono::monomorphise(&mut out, &mut diags);
    }
    if diags.is_empty() {
        mono::validate(&out, &mut diags);
    }
    if diags.is_empty() {
        Ok(out)
    } else {
        Err(diags)
    }
}

/// Type parameters of a definition: unique, and not the name of a declared or built-in type.
fn check_tparams(g: &Globals, def: &str, tps: &[String], diags: &mut Vec<Diagnostic>) {
    for (i, tp) in tps.iter().enumerate() {
        let path = format!("tparams[{i}]");
        if tps[..i].contains(tp) {
            diags.push(Diagnostic::duplicate_name(def, &path, tp));
        } else if Type::BUILTIN_NAMES.contains(&tp.as_str()) || g.type_def(tp).is_some() {
            diags.push(Diagnostic::duplicate_name(def, &path, tp).hint(format!("`{tp}` is already a type; choose another type parameter name")));
        }
    }
}

fn globals(p: &Program, diags: &mut Vec<Diagnostic>) -> Globals {
    let mut g = Globals { tp: TProgram::default(), fns: HashMap::new(), variants: HashMap::new(), type_index: HashMap::new() };
    for d in &p.defs {
        let Def::Type(t) = d else { continue };
        if Type::BUILTIN_NAMES.contains(&t.name.as_str()) {
            diags.push(Diagnostic::duplicate_name(&t.name, "", &t.name).hint(format!("`{}` is a built-in type", t.name)));
            continue;
        }
        if g.type_def(&t.name).is_some() {
            diags.push(Diagnostic::duplicate_name(&t.name, "", &t.name));
            continue;
        }
        match &t.body {
            TypeBody::Record(fields) => {
                let mut seen = HashSet::new();
                for (f, _) in fields {
                    if !seen.insert(f) {
                        diags.push(Diagnostic::duplicate_name(&t.name, &format!("fields.{f}"), f));
                    }
                }
            }
            TypeBody::Enum(vs) => {
                for (tag, v) in vs.iter().enumerate() {
                    if BUILTIN_VARIANTS.contains(&v.name.as_str()) || g.variants.contains_key(&v.name) {
                        let mut d = Diagnostic::duplicate_name(&t.name, &format!("variants[{tag}]"), &v.name);
                        if BUILTIN_VARIANTS.contains(&v.name.as_str()) {
                            d = d.hint(format!("`{}` belongs to the built-in Option/Result", v.name));
                        }
                        diags.push(d);
                    } else {
                        g.variants.insert(v.name.clone(), (t.name.clone(), tag as u32, v.fields.clone()));
                    }
                }
            }
        }
        g.type_index.insert(t.name.clone(), g.tp.types.len());
        g.tp.types.push(t.clone());
    }
    for k in 0..g.tp.types.len() {
        let t = g.tp.types[k].clone();
        check_tparams(&g, &t.name, &t.params, diags);
        let body = match &t.body {
            TypeBody::Record(fields) => TypeBody::Record(
                fields
                    .iter()
                    .map(|(f, ty)| (f.clone(), resolve_type(&g, ty, &t.params, &t.name, &format!("fields.{f}"), diags).unwrap_or(Type::Unit)))
                    .collect(),
            ),
            TypeBody::Enum(vs) => TypeBody::Enum(
                vs.iter()
                    .enumerate()
                    .map(|(tag, v)| Variant {
                        name: v.name.clone(),
                        fields: v
                            .fields
                            .iter()
                            .enumerate()
                            .map(|(i, ty)| resolve_type(&g, ty, &t.params, &t.name, &format!("variants[{tag}].fields[{i}]"), diags).unwrap_or(Type::Unit))
                            .collect(),
                    })
                    .collect(),
            ),
        };
        if let TypeBody::Enum(vs) = &body {
            for (tag, v) in vs.iter().enumerate() {
                if let Some(e) = g.variants.get_mut(&v.name) {
                    if e.0 == t.name && e.1 == tag as u32 {
                        e.2 = v.fields.clone();
                    }
                }
            }
        }
        g.tp.types[k].body = body;
    }
    irregular_recursion(&g.tp, diags);
    let mut seen = HashSet::new();
    let (mut index, mut generic_index) = (0, 0);
    for d in &p.defs {
        if matches!(d, Def::Type(_)) {
            continue;
        }
        if !seen.insert(d.name().to_string()) {
            diags.push(Diagnostic::duplicate_name(d.name(), "", d.name()));
        } else if Builtin::NAMES.contains(&d.name()) {
            diags.push(Diagnostic::builtin_name(d.name()));
        } else if RESERVED_EXPORTS.contains(&d.name()) {
            diags.push(Diagnostic::duplicate_name(d.name(), "", d.name()).hint(format!("`{}` is reserved by the runtime", d.name())));
        } else if d.name() == RESULT_VAR && matches!(d, Def::Fn(_)) {
            // A function named `r` could not be referred to inside `ens`, where `r` is the result.
            diags.push(Diagnostic::duplicate_name(d.name(), "", d.name()).hint("`r` names the result inside `ens`; choose another function name"));
        }
        if let Def::Fn(f) = d {
            // Signatures are resolved again (with diagnostics) when the function is checked.
            let resolve = |t: &Type| resolve_type(&g, t, &f.tparams, &f.name, "", &mut Vec::new()).unwrap_or(Type::Unit);
            let sig = Sig { tparams: f.tparams.clone(), params: f.params.iter().map(|p| resolve(&p.ty)).collect(), ret: resolve(&f.ret), effects: f.effects.clone() };
            let slot = if f.tparams.is_empty() { &mut index } else { &mut generic_index };
            let i = *slot;
            *slot += 1;
            g.fns.entry(f.name.clone()).or_insert((i, sig));
        }
    }
    g
}

/// Inside the recursive group of a generic type, uses of group members must pass plain type
/// parameters (regular recursion), so every type has finitely many instances.
fn irregular_recursion(tp: &TProgram, diags: &mut Vec<Diagnostic>) {
    fn refs(t: &Type, out: &mut Vec<(String, Type)>) {
        match t {
            Type::Named(n) => out.push((n.clone(), t.clone())),
            Type::App(n, args) => {
                out.push((n.clone(), t.clone()));
                args.iter().for_each(|a| refs(a, out));
            }
            Type::List(x) | Type::Option(x) => refs(x, out),
            Type::Result(a, b) | Type::Map(a, b) => {
                refs(a, out);
                refs(b, out);
            }
            Type::Fn(ps, r, _) => {
                ps.iter().for_each(|p| refs(p, out));
                refs(r, out);
            }
            _ => {}
        }
    }
    let field_types = |d: &TypeDef| -> Vec<Type> {
        match &d.body {
            TypeBody::Record(fs) => fs.iter().map(|(_, t)| t.clone()).collect(),
            TypeBody::Enum(vs) => vs.iter().flat_map(|v| v.fields.clone()).collect(),
        }
    };
    // Only a generic type can be applied irregularly.
    if tp.types.iter().all(|d| d.params.is_empty()) {
        return;
    }
    let n = tp.types.len();
    let index: HashMap<&str, usize> = tp.types.iter().enumerate().map(|(i, d)| (d.name.as_str(), i)).collect();
    let mut edges: Vec<Vec<(usize, Type)>> = vec![Vec::new(); n];
    for (i, d) in tp.types.iter().enumerate() {
        let mut out = Vec::new();
        for t in field_types(d) {
            refs(&t, &mut out);
        }
        edges[i] = out.into_iter().filter_map(|(m, t)| index.get(m.as_str()).map(|&j| (j, t))).collect();
    }
    let comp = scc(&edges.iter().map(|es| es.iter().map(|(j, _)| *j).collect()).collect::<Vec<_>>());
    for (i, d) in tp.types.iter().enumerate() {
        for (j, t) in &edges[i] {
            let same_group = comp[*j] == comp[i];
            if let Type::App(_, args) = t {
                if same_group && !args.iter().all(|a| matches!(a, Type::Param(_))) {
                    diags.push(Diagnostic::irregular_recursion(&d.name, "", &t.to_string()));
                }
            }
        }
    }
}

/// The strongly connected component of every node of a graph (`succ[i]`: the successors of node
/// `i`, all `< succ.len()`), as a component number per node: iterative Tarjan, O(nodes + edges)
/// time and no recursion, so long chains of definitions cannot overflow the stack.
pub(crate) fn scc(succ: &[Vec<usize>]) -> Vec<usize> {
    const NONE: usize = usize::MAX;
    let n = succ.len();
    let (mut index, mut low, mut comp) = (vec![NONE; n], vec![0; n], vec![NONE; n]);
    let (mut on_stack, mut stack) = (vec![false; n], Vec::new());
    let (mut next, mut comps) = (0, 0);
    for root in 0..n {
        if index[root] != NONE {
            continue;
        }
        // (node, position of the next successor to visit)
        let mut work = vec![(root, 0)];
        index[root] = next;
        low[root] = next;
        next += 1;
        stack.push(root);
        on_stack[root] = true;
        while let Some(&mut (v, ref mut k)) = work.last_mut() {
            if let Some(&w) = succ[v].get(*k) {
                *k += 1;
                if index[w] == NONE {
                    index[w] = next;
                    low[w] = next;
                    next += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    work.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(index[w]);
                }
                continue;
            }
            work.pop();
            if let Some(&(u, _)) = work.last() {
                low[u] = low[u].min(low[v]);
            }
            if low[v] == index[v] {
                while let Some(w) = stack.pop() {
                    on_stack[w] = false;
                    comp[w] = comps;
                    if w == v {
                        break;
                    }
                }
                comps += 1;
            }
        }
    }
    comp
}

/// Resolves a written type: type parameters (`tps`) become `Type::Param`, declared types must exist
/// with the right number of type arguments, effects must be known, and map keys comparable.
/// Reports every problem; None if there was one.
pub(crate) fn resolve_type(g: &Globals, t: &Type, tps: &[String], def: &str, path: &str, diags: &mut Vec<Diagnostic>) -> Option<Type> {
    let before = diags.len();
    let out = resolve_inner(g, t, tps, def, path, diags);
    (diags.len() == before).then_some(out)
}

fn resolve_inner(g: &Globals, t: &Type, tps: &[String], def: &str, path: &str, diags: &mut Vec<Diagnostic>) -> Type {
    let r = |t: &Type, diags: &mut Vec<Diagnostic>| resolve_inner(g, t, tps, def, path, diags);
    match t {
        Type::Named(n) | Type::Param(n) if tps.contains(n) => Type::Param(n.clone()),
        Type::Named(n) => match g.type_def(n) {
            Some(d) if d.params.is_empty() => t.clone(),
            Some(d) => {
                diags.push(Diagnostic::arity(def, path, n, d.params.len(), 0).hint(format!("write `{n}[{}]`", vec!["..."; d.params.len()].join(", "))));
                Type::Unit
            }
            None => {
                let mut d = Diagnostic::unknown_name(def, path, n);
                if g.variants.contains_key(n) || BUILTIN_VARIANTS.contains(&n.as_str()) {
                    d = d.hint(format!("`{n}` is a variant, not a type"));
                } else if let Some(h) = Type::foreign_name_hint(n) {
                    d = d.hint(h);
                }
                diags.push(d);
                Type::Unit
            }
        },
        Type::App(n, args) => {
            let args: Vec<Type> = args.iter().map(|a| r(a, diags)).collect();
            match g.type_def(n) {
                Some(d) if d.params.len() == args.len() => Type::App(n.clone(), args),
                Some(d) => {
                    diags.push(Diagnostic::arity(def, path, n, d.params.len(), args.len()));
                    Type::Unit
                }
                None => {
                    let mut d = Diagnostic::unknown_name(def, path, n);
                    if let Some(h) = Type::foreign_name_hint(n) {
                        d = d.hint(h);
                    }
                    diags.push(d);
                    Type::Unit
                }
            }
        }
        Type::Param(n) => {
            diags.push(Diagnostic::unknown_name(def, path, n));
            Type::Unit
        }
        Type::List(x) => Type::list(r(x, diags)),
        Type::Option(x) => Type::option(r(x, diags)),
        Type::Result(a, b) => {
            let a = r(a, diags);
            Type::result(a, r(b, diags))
        }
        Type::Map(k, v) => {
            let k = r(k, diags);
            if infer::contains_fn(&g.tp, &k, &mut Vec::new()) {
                diags.push(Diagnostic::type_mismatch(def, path, "a comparable key type", &k.to_string()).hint("map keys cannot contain functions"));
            }
            Type::map(k, r(v, diags))
        }
        Type::Fn(ps, ret, effs) => {
            // No short-circuit: report every unknown name.
            let ret = r(ret, diags);
            let ps: Vec<Type> = ps.iter().map(|p| r(p, diags)).collect();
            for e in effs {
                if !KNOWN_EFFECTS.contains(&e.as_str()) {
                    diags.push(Diagnostic::unknown_name(def, path, e).hint(format!("known effects: {}", KNOWN_EFFECTS.join(", "))));
                }
            }
            Type::Fn(ps, Box::new(ret), effs.clone())
        }
        _ => t.clone(),
    }
}

fn check_fn(f: &FnDef, g: &Globals, diags: &mut Vec<Diagnostic>) -> Option<TFn> {
    let before = diags.len();
    check_tparams(g, &f.name, &f.tparams, diags);
    let mut names = HashSet::new();
    let mut ptypes = Vec::new();
    for (i, p) in f.params.iter().enumerate() {
        let path = format!("params[{i}]");
        if !names.insert(p.name.as_str()) {
            diags.push(Diagnostic::duplicate_name(&f.name, &path, &p.name));
        }
        if p.name == RESULT_VAR && !f.ens.is_empty() {
            diags.push(
                Diagnostic::duplicate_name(&f.name, &path, &p.name)
                    .hint("inside `ens`, `r` is the result; rename the parameter".to_string()),
            );
        }
        ptypes.push(resolve_type(g, &p.ty, &f.tparams, &f.name, &path, diags).unwrap_or(Type::Unit));
    }
    let ret = resolve_type(g, &f.ret, &f.tparams, &f.name, "ret", diags).unwrap_or(Type::Unit);
    for (i, tp) in f.tparams.iter().enumerate() {
        let p = Type::Param(tp.clone());
        if !ptypes.iter().chain([&ret]).any(|t| mentions(t, &p)) {
            diags.push(Diagnostic::unused_type_param(&f.name, &format!("tparams[{i}]"), tp));
        }
    }
    for (i, e) in f.effects.iter().enumerate() {
        if !KNOWN_EFFECTS.contains(&e.as_str()) {
            diags.push(
                Diagnostic::unknown_name(&f.name, &format!("effects[{i}]"), e)
                    .hint(format!("known effects: {}", KNOWN_EFFECTS.join(", "))),
            );
        }
    }
    if f.name == "main" {
        let entry_hint = "the entry point must be `fn main(args: List[Str]) -> Int` (any effects)";
        if !f.tparams.is_empty() {
            diags.push(Diagnostic::type_mismatch(&f.name, "tparams", "no type parameters", &format!("[{}]", f.tparams.join(", "))).hint(entry_hint));
        }
        if f.params.len() != 1 || f.params[0].ty != Type::list(Type::Str) {
            let found: Vec<String> = f.params.iter().map(|p| p.ty.to_string()).collect();
            diags.push(Diagnostic::type_mismatch(&f.name, "params", "(List[Str])", &format!("({})", found.join(", "))).hint(entry_hint));
        }
        if f.ret != Type::Int {
            diags.push(Diagnostic::type_mismatch(&f.name, "ret", "Int", &f.ret.to_string()).hint(entry_hint));
        }
    }

    let mut cx = Cx::new(g, &f.name, &[], diags);
    cx.start = before;
    cx.tps = &f.tparams;
    for (p, t) in f.params.iter().zip(&ptypes) {
        cx.bind(&p.name, t.clone());
    }
    let mut exprs = Vec::new();
    for (i, c) in f.req.iter().enumerate() {
        exprs.push(cx.expect(c, &Type::Bool, &format!("req[{i}]")));
    }
    let result_local = (!f.ens.is_empty()).then(|| cx.bind(RESULT_VAR, ret.clone()));
    for (i, c) in f.ens.iter().enumerate() {
        exprs.push(cx.expect(c, &Type::Bool, &format!("ens[{i}]")));
    }
    if result_local.is_some() {
        cx.unbind();
    }
    cx.effects = &f.effects;
    cx.ret = Some(&ret);
    exprs.push(cx.expect(&f.body, &ret, "body"));
    let (locals, mut exprs) = cx.finish(exprs)?;
    let body = exprs.pop()?;
    let ens = exprs.split_off(f.req.len());
    Some(TFn { def: f.clone(), locals, req: exprs, ens, result_local, body, instance: None })
}

/// True if `t` mentions the type `p`.
fn mentions(t: &Type, p: &Type) -> bool {
    t == p
        || match t {
            Type::List(x) | Type::Option(x) => mentions(x, p),
            Type::Result(a, b) | Type::Map(a, b) => mentions(a, p) || mentions(b, p),
            Type::App(_, args) => args.iter().any(|a| mentions(a, p)),
            Type::Fn(ps, r, _) => ps.iter().any(|x| mentions(x, p)) || mentions(r, p),
            _ => false,
        }
}

#[cfg(test)]
mod tests;
