//! QAL checker: names, types, effects, contracts and match exhaustiveness.
//! Only checked programs may be compiled; `check` returns the typed program the code generator needs.

mod builtins;
mod exhaust;
mod infer;
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
    pub params: Vec<Type>,
    pub ret: Type,
    pub effects: Vec<String>,
}

impl Sig {
    pub fn fn_type(&self) -> Type {
        Type::Fn(self.params.clone(), Box::new(self.ret.clone()), self.effects.clone())
    }
}

/// Program-wide names, known before any body is checked.
pub(crate) struct Globals {
    /// Declared types (for lookups); `fns`/`tests` stay empty.
    pub tp: TProgram,
    /// Function name -> (index into `TProgram::fns`, signature).
    pub fns: HashMap<String, (usize, Sig)>,
    /// User variant name -> (enum name, tag, payload types).
    pub variants: HashMap<String, (String, u32, Vec<Type>)>,
}

/// Checks a whole program. An empty result means the program may be compiled.
pub fn check_program(p: &Program) -> Vec<Diagnostic> {
    match check(p) {
        Ok(_) => Vec::new(),
        Err(d) => d,
    }
}

/// Checks a whole program and returns its typed form, or all diagnostics.
pub fn check(p: &Program) -> Result<TProgram, Vec<Diagnostic>> {
    let mut diags = Vec::new();
    let g = globals(p, &mut diags);
    let mut out = TProgram { types: g.tp.types.clone(), ..TProgram::default() };
    for d in &p.defs {
        match d {
            Def::Type(_) => {}
            Def::Fn(f) => {
                if let Some(tf) = check_fn(f, &g, &mut diags) {
                    out.fns.push(tf);
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
        Ok(out)
    } else {
        Err(diags)
    }
}

fn globals(p: &Program, diags: &mut Vec<Diagnostic>) -> Globals {
    let mut g = Globals { tp: TProgram::default(), fns: HashMap::new(), variants: HashMap::new() };
    for d in &p.defs {
        let Def::Type(t) = d else { continue };
        if Type::BUILTIN_NAMES.contains(&t.name.as_str()) {
            diags.push(Diagnostic::duplicate_name(&t.name, "", &t.name).hint(format!("`{}` is a built-in type", t.name)));
            continue;
        }
        if g.tp.type_def(&t.name).is_some() {
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
        g.tp.types.push(t.clone());
    }
    for t in g.tp.types.clone() {
        match &t.body {
            TypeBody::Record(fields) => {
                for (f, ty) in fields {
                    validate_type(&g, ty, &t.name, &format!("fields.{f}"), diags);
                }
            }
            TypeBody::Enum(vs) => {
                for (tag, v) in vs.iter().enumerate() {
                    for (i, ty) in v.fields.iter().enumerate() {
                        validate_type(&g, ty, &t.name, &format!("variants[{tag}].fields[{i}]"), diags);
                    }
                }
            }
        }
    }
    let mut seen = HashSet::new();
    let mut index = 0;
    for d in &p.defs {
        if matches!(d, Def::Type(_)) {
            continue;
        }
        if !seen.insert(d.name().to_string()) {
            diags.push(Diagnostic::duplicate_name(d.name(), "", d.name()));
        } else if Builtin::NAMES.contains(&d.name()) {
            diags.push(Diagnostic::duplicate_name(d.name(), "", d.name()).hint(format!("`{}` is a built-in function", d.name())));
        } else if RESERVED_EXPORTS.contains(&d.name()) {
            diags.push(Diagnostic::duplicate_name(d.name(), "", d.name()).hint(format!("`{}` is reserved by the runtime", d.name())));
        }
        if let Def::Fn(f) = d {
            g.fns.entry(f.name.clone()).or_insert_with(|| {
                (index, Sig { params: f.params.iter().map(|p| p.ty.clone()).collect(), ret: f.ret.clone(), effects: f.effects.clone() })
            });
            index += 1;
        }
    }
    g
}

/// Reports unknown type names and effects inside a written type.
pub(crate) fn validate_type(g: &Globals, t: &Type, def: &str, path: &str, diags: &mut Vec<Diagnostic>) -> bool {
    match t {
        Type::Named(n) => {
            if g.tp.type_def(n).is_some() {
                return true;
            }
            let mut d = Diagnostic::unknown_name(def, path, n);
            if g.variants.contains_key(n) || BUILTIN_VARIANTS.contains(&n.as_str()) {
                d = d.hint(format!("`{n}` is a variant, not a type"));
            }
            diags.push(d);
            false
        }
        Type::List(x) | Type::Option(x) => validate_type(g, x, def, path, diags),
        Type::Result(a, b) => validate_type(g, a, def, path, diags) & validate_type(g, b, def, path, diags),
        Type::Fn(ps, r, effs) => {
            // No short-circuit: report every unknown name.
            let mut ok = validate_type(g, r, def, path, diags);
            for p in ps {
                ok &= validate_type(g, p, def, path, diags);
            }
            for e in effs {
                if !KNOWN_EFFECTS.contains(&e.as_str()) {
                    diags.push(Diagnostic::unknown_name(def, path, e).hint(format!("known effects: {}", KNOWN_EFFECTS.join(", "))));
                    ok = false;
                }
            }
            ok
        }
        _ => true,
    }
}

fn check_fn(f: &FnDef, g: &Globals, diags: &mut Vec<Diagnostic>) -> Option<TFn> {
    let before = diags.len();
    let mut names = HashSet::new();
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
        validate_type(g, &p.ty, &f.name, &path, diags);
    }
    validate_type(g, &f.ret, &f.name, "ret", diags);
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
    for p in &f.params {
        cx.bind(&p.name, p.ty.clone());
    }
    let mut exprs = Vec::new();
    for (i, c) in f.req.iter().enumerate() {
        exprs.push(cx.expect(c, &Type::Bool, &format!("req[{i}]")));
    }
    let result_local = (!f.ens.is_empty()).then(|| cx.bind(RESULT_VAR, f.ret.clone()));
    for (i, c) in f.ens.iter().enumerate() {
        exprs.push(cx.expect(c, &Type::Bool, &format!("ens[{i}]")));
    }
    if result_local.is_some() {
        cx.unbind();
    }
    cx.effects = &f.effects;
    cx.ret = Some(&f.ret);
    exprs.push(cx.expect(&f.body, &f.ret, "body"));
    let (locals, mut exprs) = cx.finish(exprs)?;
    let body = exprs.pop().expect("body");
    let ens = exprs.split_off(f.req.len());
    Some(TFn { def: f.clone(), locals, req: exprs, ens, result_local, body })
}

#[cfg(test)]
mod tests;
