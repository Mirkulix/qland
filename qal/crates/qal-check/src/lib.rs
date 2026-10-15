//! QAL checker: names, types, effects and contracts. Only checked programs may be compiled.

use qal_core::ast::*;
use qal_core::diag::Diagnostic;
use std::collections::{HashMap, HashSet};

/// Effects known in Phase 1.
pub const KNOWN_EFFECTS: &[&str] = &["io", "fs", "time", "rand"];

pub use qal_core::ast::RESULT_VAR;

#[derive(Debug, Clone, PartialEq)]
pub struct Sig {
    pub params: Vec<Type>,
    pub ret: Type,
    pub effects: Vec<String>,
}

/// Checks a whole program. An empty result means the program may be compiled.
pub fn check_program(p: &Program) -> Vec<Diagnostic> {
    let mut diags = Vec::new();
    let mut sigs: HashMap<String, Sig> = HashMap::new();
    let mut seen = HashSet::new();
    for d in &p.defs {
        if !seen.insert(d.name().to_string()) {
            diags.push(Diagnostic::duplicate_name(d.name(), "", d.name()));
        }
        if let Def::Fn(f) = d {
            sigs.entry(f.name.clone()).or_insert_with(|| Sig {
                params: f.params.iter().map(|p| p.ty).collect(),
                ret: f.ret,
                effects: f.effects.clone(),
            });
        }
    }
    for d in &p.defs {
        match d {
            Def::Fn(f) => check_fn(f, &sigs, &mut diags),
            Def::Test(t) => {
                let mut cx = Cx { def: &t.name, sigs: &sigs, effects: &[], diags: &mut diags };
                let mut env = Vec::new();
                cx.expect(&t.body, Type::Bool, &mut env, "body");
            }
        }
    }
    diags
}

fn check_fn(f: &FnDef, sigs: &HashMap<String, Sig>, diags: &mut Vec<Diagnostic>) {
    let mut names = HashSet::new();
    for (i, p) in f.params.iter().enumerate() {
        if !names.insert(p.name.as_str()) {
            diags.push(Diagnostic::duplicate_name(&f.name, &format!("params[{i}]"), &p.name));
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
    let params: Vec<(String, Type)> = f.params.iter().map(|p| (p.name.clone(), p.ty)).collect();
    let mut cx = Cx { def: &f.name, sigs, effects: &[], diags };
    for (i, c) in f.req.iter().enumerate() {
        let mut env = params.clone();
        cx.expect(c, Type::Bool, &mut env, &format!("req[{i}]"));
    }
    for (i, c) in f.ens.iter().enumerate() {
        let mut env = params.clone();
        env.push((RESULT_VAR.to_string(), f.ret));
        cx.expect(c, Type::Bool, &mut env, &format!("ens[{i}]"));
    }
    cx.effects = &f.effects;
    let mut env = params;
    cx.expect(&f.body, f.ret, &mut env, "body");
}

struct Cx<'a> {
    def: &'a str,
    sigs: &'a HashMap<String, Sig>,
    /// Effects the current expression may use (empty for contracts and tests).
    effects: &'a [String],
    diags: &'a mut Vec<Diagnostic>,
}

impl Cx<'_> {
    fn expect(&mut self, e: &Expr, want: Type, env: &mut Vec<(String, Type)>, path: &str) {
        if let Some(got) = self.infer(e, env, path) {
            if got != want {
                let mut d = Diagnostic::type_mismatch(self.def, path, want.name(), got.name());
                if want == Type::Bool && got == Type::Int {
                    d = d.hint("compare explicitly, e.g. `x != 0`");
                }
                self.diags.push(d);
            }
        }
    }

    /// Returns None if an error was already reported for `e` (avoids cascades).
    fn infer(&mut self, e: &Expr, env: &mut Vec<(String, Type)>, path: &str) -> Option<Type> {
        match e {
            Expr::Int(_) => Some(Type::Int),
            Expr::Bool(_) => Some(Type::Bool),
            Expr::Unit => Some(Type::Unit),
            Expr::Var(v) => match env.iter().rev().find(|(n, _)| n == v) {
                Some((_, t)) => Some(*t),
                None => {
                    let mut d = Diagnostic::unknown_name(self.def, path, v);
                    if self.sigs.contains_key(v) {
                        d = d.hint(format!("`{v}` is a function; call it as `{v}(...)`"));
                    }
                    self.diags.push(d);
                    None
                }
            },
            Expr::Un(op, x) => {
                let t = match op {
                    UnOp::Neg => Type::Int,
                    UnOp::Not => Type::Bool,
                };
                self.expect(x, t, env, &format!("{path}.operand"));
                Some(t)
            }
            Expr::Bin(op, l, r) => {
                let (lp, rp) = (format!("{path}.left"), format!("{path}.right"));
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => {
                        self.expect(l, Type::Int, env, &lp);
                        self.expect(r, Type::Int, env, &rp);
                        Some(Type::Int)
                    }
                    BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        self.expect(l, Type::Int, env, &lp);
                        self.expect(r, Type::Int, env, &rp);
                        Some(Type::Bool)
                    }
                    BinOp::And | BinOp::Or => {
                        self.expect(l, Type::Bool, env, &lp);
                        self.expect(r, Type::Bool, env, &rp);
                        Some(Type::Bool)
                    }
                    BinOp::Eq | BinOp::Ne => {
                        if let Some(lt) = self.infer(l, env, &lp) {
                            self.expect(r, lt, env, &rp);
                        } else {
                            self.infer(r, env, &rp);
                        }
                        Some(Type::Bool)
                    }
                }
            }
            Expr::If(c, t, f) => {
                self.expect(c, Type::Bool, env, &format!("{path}.cond"));
                let tt = self.infer(t, env, &format!("{path}.then"));
                match tt {
                    Some(tt) => {
                        self.expect(f, tt, env, &format!("{path}.else"));
                        Some(tt)
                    }
                    None => self.infer(f, env, &format!("{path}.else")),
                }
            }
            Expr::Let(name, v, body) => {
                let vt = self.infer(v, env, &format!("{path}.value"))?;
                env.push((name.clone(), vt));
                let bt = self.infer(body, env, &format!("{path}.body"));
                env.pop();
                bt
            }
            Expr::Call(name, args) => {
                let Some(sig) = self.sigs.get(name) else {
                    self.diags.push(Diagnostic::unknown_name(self.def, path, name));
                    for (i, a) in args.iter().enumerate() {
                        self.infer(a, env, &format!("{path}.args[{i}]"));
                    }
                    return None;
                };
                if sig.params.len() != args.len() {
                    self.diags.push(Diagnostic::arity(self.def, path, name, sig.params.len(), args.len()));
                } else {
                    for (i, (a, t)) in args.iter().zip(&sig.params).enumerate() {
                        self.expect(a, *t, env, &format!("{path}.args[{i}]"));
                    }
                }
                let missing: Vec<String> = sig.effects.iter().filter(|e| !self.effects.contains(e)).cloned().collect();
                if !missing.is_empty() {
                    self.diags.push(Diagnostic::effect_violation(self.def, path, name, &missing));
                }
                Some(sig.ret)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qal_text::parse_program;

    fn check(src: &str) -> Vec<Diagnostic> {
        check_program(&parse_program(src).unwrap())
    }

    fn kinds(src: &str) -> Vec<(&'static str, String, String)> {
        check(src).into_iter().map(|d| (d.kind, d.def, d.path)).collect()
    }

    #[test]
    fn accepts_well_typed_program() {
        let src = "fn fact(n: Int) -> Int req n >= 0 ens r >= 1 = if n <= 1 then 1 else n * fact(n - 1)\n\
                   test fact_5 = fact(5) == 120\n\
                   fn noop() -> Unit = ()\n\
                   test unit_eq = noop() == ()";
        assert_eq!(check(src), vec![]);
    }

    #[test]
    fn reports_type_mismatch_with_path() {
        let d = check("fn f(x: Int) -> Int = if x then 1 else 2");
        assert_eq!(d.len(), 1);
        assert_eq!((d[0].kind, d[0].path.as_str()), ("type_mismatch", "body.cond"));
        assert_eq!((d[0].expected.as_deref(), d[0].found.as_deref()), (Some("Bool"), Some("Int")));
        assert_eq!(d[0].hints, vec!["compare explicitly, e.g. `x != 0`".to_string()]);
    }

    #[test]
    fn reports_unknown_names_arity_and_duplicates() {
        assert_eq!(kinds("fn f() -> Int = y"), vec![("unknown_name", "f".into(), "body".into())]);
        assert_eq!(kinds("fn f() -> Int = g()"), vec![("unknown_name", "f".into(), "body".into())]);
        let d = check("fn g(a: Int) -> Int = a\nfn f() -> Int = g(1, 2)");
        assert_eq!((d[0].code, d[0].kind), ("E0204", "type_mismatch"));
        assert_eq!(kinds("fn f() -> Int = 1\nfn f() -> Int = 2")[0].0, "duplicate_name");
        assert_eq!(kinds("fn f(a: Int, a: Int) -> Int = a")[0], ("duplicate_name", "f".into(), "params[1]".into()));
    }

    #[test]
    fn let_scoping_and_shadowing() {
        assert_eq!(check("fn f() -> Bool = let x = 1 in let x = true in x"), vec![]);
        assert_eq!(kinds("fn f() -> Int = (let x = 1 in x) + x"), vec![("unknown_name", "f".into(), "body.right".into())]);
    }

    #[test]
    fn contracts_must_be_bool_and_see_result() {
        assert_eq!(kinds("fn f(x: Int) -> Int req x = x"), vec![("type_mismatch", "f".into(), "req[0]".into())]);
        assert_eq!(kinds("fn f(x: Int) -> Int req r > 0 = x"), vec![("unknown_name", "f".into(), "req[0].left".into())]);
        assert_eq!(check("fn f(x: Int) -> Bool ens r == (x > 0) = x > 0"), vec![]);
    }

    #[test]
    fn effects_must_be_declared_and_known() {
        let src = "fn say() -> Unit !io = ()\nfn pure_caller() -> Unit = say()\nfn ok() -> Unit !io = say()";
        let d = check(src);
        assert_eq!(d.len(), 1);
        assert_eq!((d[0].kind, d[0].def.as_str(), d[0].code), ("effect_violation", "pure_caller", "E0301"));
        assert_eq!(d[0].hints, vec!["add `!io` to the signature of `pure_caller`".to_string()]);
        assert_eq!(kinds("fn f() -> Unit !net = ()")[0], ("unknown_name", "f".into(), "effects[0]".into()));
        assert_eq!(kinds("fn say() -> Unit !io = ()\ntest t = say() == ()")[0].0, "effect_violation");
        assert_eq!(kinds("fn say() -> Unit !io = ()\nfn f() -> Int !io req say() == () = 1")[0].0, "effect_violation");
    }

    #[test]
    fn tests_must_be_bool() {
        assert_eq!(kinds("test t = 1"), vec![("type_mismatch", "t".into(), "body".into())]);
    }
}
