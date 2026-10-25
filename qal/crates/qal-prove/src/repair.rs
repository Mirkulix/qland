//! Req-directed generation: random inputs repaired towards a function's `req` before they run.
//!
//! A property test draws random inputs and discards those that fail `req`. For a `req` with several
//! conditions on record fields almost every input fails one of them (the mutation study of
//! 2026-10-25: 997 of 1000 inputs of `put_back`, all 1000 of `take_back`), and the property test
//! gives up. Here the `req` is read as a list of simple conditions (atoms) on paths into the
//! parameters, `c.taxes <= c.spent`, `qty <= o.qty - o.returned`, `o.open`, also through
//! Bool helpers (`cust_ok(c)`), map lookups (`match map_get(m, k) { Some(x) => p(x), None => true }`)
//! and conjunctions over collections (`fold(xs, true, fn(a, x) => a && p(x))`). A generated input
//! that violates an atom is changed at one path of it so that the atom holds; a few rounds settle
//! atoms that share paths.
//!
//! Only the chance of a run changes, not what a run means: every input still goes through the real
//! `req` and is discarded if it fails it. What cannot be read as atoms is left to chance as before.

use crate::gen::Rng;
use qal_core::ast::{BinOp, UnOp};
use qal_core::tast::{Builtin, TExpr, TFn, TKind, TPat, TProgram, TAG_NONE, TAG_SOME};
use qal_run::Value;
use std::collections::HashMap;

/// How deep Bool helpers are followed.
const MAX_HELPER_DEPTH: usize = 4;
/// Repair rounds per input: an atom repaired late can break one repaired earlier.
const ROUNDS: usize = 8;
/// Atoms read from one `req` at most (helpers can multiply them).
const MAX_ATOMS: usize = 256;

/// One step from a value to a part of it.
#[derive(Debug, Clone, PartialEq)]
enum Step {
    /// A record field by declaration index.
    Field(u32),
    /// The value of a map entry under a key.
    At(Box<Term>),
    /// Every element of a list, or every value of a map.
    Each,
}

/// A place in the arguments: parameter index and steps into it.
#[derive(Debug, Clone, PartialEq)]
struct Path {
    param: usize,
    steps: Vec<Step>,
}

/// An Int, Bool or Str expression over paths and literals.
#[derive(Debug, Clone, PartialEq)]
enum Term {
    Int(i64),
    Bool(bool),
    Str(String),
    Path(Path),
    Bin(BinOp, Box<Term>, Box<Term>),
    Neg(Box<Term>),
}

/// `l op r` with `op` a comparison.
#[derive(Debug, Clone, PartialEq)]
struct Atom {
    op: BinOp,
    l: Term,
    r: Term,
}

/// The atoms of a function's `req`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Plan {
    atoms: Vec<Atom>,
}

impl Plan {
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }
}

/// Reads the atoms of `f`'s `req`.
pub fn plan(tp: &TProgram, f: &TFn) -> Plan {
    plan_of(tp, f, &f.req)
}

/// Reads the atoms of the body of a Bool function `f` (an invariant, inv.rs).
pub fn plan_of_body(tp: &TProgram, f: &TFn) -> Plan {
    plan_of(tp, f, std::slice::from_ref(&f.body))
}

fn plan_of(tp: &TProgram, f: &TFn, conds: &[TExpr]) -> Plan {
    let env: HashMap<u32, Term> = (0..f.def.params.len()).map(|i| (i as u32, Term::Path(Path { param: i, steps: Vec::new() }))).collect();
    let mut atoms = Vec::new();
    for r in conds {
        collect(tp, r, &env, 0, &mut atoms);
    }
    atoms.truncate(MAX_ATOMS);
    Plan { atoms }
}

fn collect(tp: &TProgram, e: &TExpr, env: &HashMap<u32, Term>, depth: usize, out: &mut Vec<Atom>) {
    if out.len() >= MAX_ATOMS {
        return;
    }
    match &e.kind {
        TKind::Bin(BinOp::And, a, b, _) => {
            collect(tp, a, env, depth, out);
            collect(tp, b, env, depth, out);
        }
        TKind::Bin(op @ (BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne), a, b, _) => {
            if let (Some(l), Some(r)) = (term(a, env), term(b, env)) {
                out.push(Atom { op: *op, l, r });
            }
        }
        TKind::Un(UnOp::Not, a, _) => {
            if let Some(t @ Term::Path(_)) = term(a, env) {
                out.push(Atom { op: BinOp::Eq, l: t, r: Term::Bool(false) });
            }
        }
        TKind::Let(x, v, body) => {
            let t = term(v, env);
            let mut env = env.clone();
            match t {
                Some(t) => env.insert(*x, t),
                None => env.remove(x),
            };
            collect(tp, body, &env, depth, out);
        }
        TKind::Call(g, args) if depth < MAX_HELPER_DEPTH => {
            let Some(h) = tp.fns.get(*g) else { return };
            if h.def.effects.is_empty() && h.def.eparams.is_empty() && args.len() == h.def.params.len() {
                // A helper is read with its parameters bound to the caller's terms; an argument
                // that is no term leaves that parameter unknown (its atoms are dropped).
                let inner: HashMap<u32, Term> = args.iter().enumerate().filter_map(|(i, a)| term(a, env).map(|t| (i as u32, t))).collect();
                collect(tp, &h.body, &inner, depth + 1, out);
            }
        }
        TKind::Match(scrut, arms) => {
            // `match map_get(m, k) { Some(x) => p(x), None => true }`: p of the entry under k.
            if let TKind::Builtin(Builtin::MapGet, margs, _) = &scrut.kind {
                if let (Some(Term::Path(m)), Some(k)) = (margs.first().and_then(|m| term(m, env)), margs.get(1).and_then(|k| term(k, env))) {
                    let vacuous_none = arms.iter().any(|a| matches!(&a.pat, TPat::Variant(t, _) if *t == TAG_NONE) && matches!(a.body.kind, TKind::Bool(true)));
                    for arm in arms {
                        if let (TPat::Variant(t, ps), None, true) = (&arm.pat, &arm.guard, vacuous_none) {
                            if *t == TAG_SOME {
                                if let Some(TPat::Bind(x)) = ps.first() {
                                    let mut env = env.clone();
                                    let mut at = m.clone();
                                    at.steps.push(Step::At(Box::new(k.clone())));
                                    env.insert(*x, Term::Path(at));
                                    collect(tp, &arm.body, &env, depth, out);
                                }
                            }
                        }
                    }
                }
            }
        }
        TKind::Builtin(Builtin::Fold, fargs, _) if fargs.len() == 3 => {
            // `fold(xs, true, fn(acc, x) => acc && p(x))`: p of every element.
            let (TKind::Bool(true), TKind::Lambda(l)) = (&fargs[1].kind, &fargs[2].kind) else { return };
            let coll = match &fargs[0].kind {
                TKind::Builtin(Builtin::MapValues, m, _) => m.first().and_then(|m| term(m, env)),
                _ => term(&fargs[0], env),
            };
            let (Some(Term::Path(mut each)), [acc, x]) = (coll, l.params.as_slice()) else { return };
            let TKind::Bin(BinOp::And, a, p, _) = &l.body.kind else { return };
            if a.kind != TKind::Local(*acc) {
                return;
            }
            each.steps.push(Step::Each);
            let mut env = env.clone();
            env.insert(*x, Term::Path(each));
            collect(tp, p, &env, depth, out);
        }
        _ => {
            // A Bool path on its own: it must be true.
            if let Some(t @ Term::Path(_)) = term(e, env) {
                if e.ty == qal_core::ast::Type::Bool {
                    out.push(Atom { op: BinOp::Eq, l: t, r: Term::Bool(true) });
                }
            }
        }
    }
}

fn term(e: &TExpr, env: &HashMap<u32, Term>) -> Option<Term> {
    Some(match &e.kind {
        TKind::Int(n) => Term::Int(*n),
        TKind::Bool(b) => Term::Bool(*b),
        TKind::Str(s) => Term::Str(s.clone()),
        TKind::Local(x) => env.get(x)?.clone(),
        TKind::Field(r, i) => match term(r, env)? {
            Term::Path(mut p) => {
                p.steps.push(Step::Field(*i));
                Term::Path(p)
            }
            _ => return None,
        },
        TKind::Bin(op @ (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div), a, b, _) => Term::Bin(*op, Box::new(term(a, env)?), Box::new(term(b, env)?)),
        TKind::Un(UnOp::Neg, a, _) => Term::Neg(Box::new(term(a, env)?)),
        _ => return None,
    })
}

/// A concrete step: `Each` and `At` resolved against the arguments.
#[derive(Debug, Clone, Copy, PartialEq)]
enum CStep {
    Field(usize),
    Elem(usize),
    MapVal(usize),
}

type CPath = (usize, Vec<CStep>);

/// A term whose paths are concrete.
#[derive(Debug, Clone)]
enum CTerm {
    Int(i64),
    Bool(bool),
    Str(String),
    Path(CPath),
    Bin(BinOp, Box<CTerm>, Box<CTerm>),
    Neg(Box<CTerm>),
}

/// Changes `args` so that more atoms of `plan` hold.
pub fn repair(plan: &Plan, args: &mut [Value], rng: &mut Rng) {
    for _ in 0..ROUNDS {
        let mut changed = false;
        for atom in &plan.atoms {
            for (l, r) in instances(atom, args) {
                if holds(atom.op, &l, &r, args) == Some(false) && fix(atom.op, &l, &r, args, rng) {
                    changed = true;
                }
            }
        }
        if !changed {
            return;
        }
    }
}

/// The concrete versions of an atom: one per element of each collection its paths go through
/// (paths through the same collection take the same element).
fn instances(atom: &Atom, args: &[Value]) -> Vec<(CTerm, CTerm)> {
    let mut out = Vec::new();
    let mut choice: Vec<(Vec<Step>, usize, usize)> = Vec::new();
    expand(atom, args, &mut choice, &mut out);
    out
}

fn expand(atom: &Atom, args: &[Value], choice: &mut Vec<(Vec<Step>, usize, usize)>, out: &mut Vec<(CTerm, CTerm)>) {
    if out.len() >= 64 {
        return;
    }
    // The first `Each` without a chosen element: try every element of its collection.
    if let Some((param, prefix)) = first_open_each(&atom.l, choice).or_else(|| first_open_each(&atom.r, choice)) {
        let Some(n) = resolve(args, param, &prefix, choice).and_then(|(p, s)| at(args, &(p, s)).map(len_of)) else { return };
        for i in 0..n {
            choice.push((prefix.clone(), param, i));
            expand(atom, args, choice, out);
            choice.pop();
        }
        return;
    }
    if let (Some(l), Some(r)) = (concrete(&atom.l, args, choice), concrete(&atom.r, args, choice)) {
        out.push((l, r));
    }
}

fn len_of(v: &Value) -> usize {
    match v {
        Value::List(xs) => xs.len(),
        Value::Map(es) => es.len(),
        _ => 0,
    }
}

fn first_open_each(t: &Term, choice: &[(Vec<Step>, usize, usize)]) -> Option<(usize, Vec<Step>)> {
    match t {
        Term::Path(p) => {
            for (k, s) in p.steps.iter().enumerate() {
                if let Step::At(key) = s {
                    if let Some(found) = first_open_each(key, choice) {
                        return Some(found);
                    }
                }
                if *s == Step::Each && !choice.iter().any(|(pre, param, _)| *param == p.param && *pre == p.steps[..k]) {
                    return Some((p.param, p.steps[..k].to_vec()));
                }
            }
            None
        }
        Term::Bin(_, a, b) => first_open_each(a, choice).or_else(|| first_open_each(b, choice)),
        Term::Neg(a) => first_open_each(a, choice),
        _ => None,
    }
}

/// The concrete path of `steps` from parameter `param`, with chosen elements for `Each`.
fn resolve(args: &[Value], param: usize, steps: &[Step], choice: &[(Vec<Step>, usize, usize)]) -> Option<CPath> {
    let mut out = Vec::new();
    for (k, s) in steps.iter().enumerate() {
        let here = at(args, &(param, out.clone()))?;
        out.push(match s {
            Step::Field(i) => CStep::Field(*i as usize),
            Step::Each => {
                let i = choice.iter().find(|(pre, p, _)| *p == param && *pre == steps[..k])?.2;
                if matches!(here, Value::Map(_)) { CStep::MapVal(i) } else { CStep::Elem(i) }
            }
            Step::At(key) => {
                let key = value_of(&concrete(key, args, choice)?, args)?;
                let Value::Map(es) = here else { return None };
                CStep::MapVal(es.iter().position(|(k, _)| *k == key)?)
            }
        });
    }
    Some((param, out))
}

fn concrete(t: &Term, args: &[Value], choice: &[(Vec<Step>, usize, usize)]) -> Option<CTerm> {
    Some(match t {
        Term::Int(n) => CTerm::Int(*n),
        Term::Bool(b) => CTerm::Bool(*b),
        Term::Str(s) => CTerm::Str(s.clone()),
        Term::Path(p) => CTerm::Path(resolve(args, p.param, &p.steps, choice)?),
        Term::Bin(op, a, b) => CTerm::Bin(*op, Box::new(concrete(a, args, choice)?), Box::new(concrete(b, args, choice)?)),
        Term::Neg(a) => CTerm::Neg(Box::new(concrete(a, args, choice)?)),
    })
}

fn at<'a>(args: &'a [Value], (param, steps): &CPath) -> Option<&'a Value> {
    let mut v = args.get(*param)?;
    for s in steps {
        v = match (s, v) {
            (CStep::Field(i), Value::Record(_, fs)) => &fs.get(*i)?.1,
            (CStep::Elem(i), Value::List(xs)) => xs.get(*i)?,
            (CStep::MapVal(i), Value::Map(es)) => &es.get(*i)?.1,
            _ => return None,
        };
    }
    Some(v)
}

fn at_mut<'a>(args: &'a mut [Value], (param, steps): &CPath) -> Option<&'a mut Value> {
    let mut v = args.get_mut(*param)?;
    for s in steps {
        v = match (s, v) {
            (CStep::Field(i), Value::Record(_, fs)) => &mut fs.get_mut(*i)?.1,
            (CStep::Elem(i), Value::List(xs)) => xs.get_mut(*i)?,
            (CStep::MapVal(i), Value::Map(es)) => &mut es.get_mut(*i)?.1,
            _ => return None,
        };
    }
    Some(v)
}

fn value_of(t: &CTerm, args: &[Value]) -> Option<Value> {
    Some(match t {
        CTerm::Int(n) => Value::Int(*n),
        CTerm::Bool(b) => Value::Bool(*b),
        CTerm::Str(s) => Value::Str(s.clone()),
        CTerm::Path(p) => at(args, p)?.clone(),
        CTerm::Bin(..) | CTerm::Neg(_) => Value::Int(int(t, args)?),
    })
}

fn int(t: &CTerm, args: &[Value]) -> Option<i64> {
    match t {
        CTerm::Int(n) => Some(*n),
        CTerm::Path(p) => match at(args, p)? {
            Value::Int(n) => Some(*n),
            _ => None,
        },
        CTerm::Bin(op, a, b) => {
            let (a, b) = (int(a, args)?, int(b, args)?);
            match op {
                BinOp::Add => a.checked_add(b),
                BinOp::Sub => a.checked_sub(b),
                BinOp::Mul => a.checked_mul(b),
                // QAL's `/` truncates toward zero, as Rust's does; a zero divisor is no value.
                BinOp::Div => a.checked_div(b),
                _ => None,
            }
        }
        CTerm::Neg(a) => int(a, args)?.checked_neg(),
        _ => None,
    }
}

fn holds(op: BinOp, l: &CTerm, r: &CTerm, args: &[Value]) -> Option<bool> {
    if let (Some(a), Some(b)) = (int(l, args), int(r, args)) {
        return Some(match op {
            BinOp::Lt => a < b,
            BinOp::Le => a <= b,
            BinOp::Gt => a > b,
            BinOp::Ge => a >= b,
            BinOp::Eq => a == b,
            BinOp::Ne => a != b,
            _ => return None,
        });
    }
    let (a, b) = (value_of(l, args)?, value_of(r, args)?);
    match op {
        BinOp::Eq => Some(a == b),
        BinOp::Ne => Some(a != b),
        _ => None,
    }
}

/// Changes one path of the atom so that it holds; false if no path could be set.
fn fix(op: BinOp, l: &CTerm, r: &CTerm, args: &mut [Value], rng: &mut Rng) -> bool {
    let slack = rng.below(3) as i64;
    if let (Some(lv), Some(rv)) = (int(l, args), int(r, args)) {
        // The value the left side needs given the right one, and the reverse.
        let (want_l, want_r) = match op {
            BinOp::Ge => (rv.checked_add(slack), lv.checked_sub(slack)),
            BinOp::Gt => (rv.checked_add(1 + slack), lv.checked_sub(1 + slack)),
            BinOp::Le => (rv.checked_sub(slack), lv.checked_add(slack)),
            BinOp::Lt => (rv.checked_sub(1 + slack), lv.checked_add(1 + slack)),
            BinOp::Eq => (Some(rv), Some(lv)),
            BinOp::Ne => (rv.checked_add(1), lv.checked_add(1)),
            _ => (None, None),
        };
        // Either side may be changed: always the same one could undo another atom's repair every
        // round (`taxes >= 0` and `taxes <= spent` with a negative `spent`).
        let order = if rng.below(2) == 0 { [(l, want_l), (r, want_r)] } else { [(r, want_r), (l, want_l)] };
        for (side, want) in order {
            if let Some(want) = want {
                if let Some((p, v)) = solve(side, want, args) {
                    if let Some(slot @ Value::Int(_)) = at_mut(args, &p) {
                        *slot = Value::Int(v);
                        return true;
                    }
                }
            }
        }
        return false;
    }
    // Bool or Str equality: set the path side to the other side's value.
    if op != BinOp::Eq {
        return false;
    }
    for (side, other) in [(l, r), (r, l)] {
        if let (CTerm::Path(p), Some(v)) = (side, value_of(other, args)) {
            if let Some(slot) = at_mut(args, p) {
                if std::mem::discriminant(slot) == std::mem::discriminant(&v) {
                    *slot = v;
                    return true;
                }
            }
        }
    }
    false
}

/// A path inside `t` and the value it needs for `t` to equal `want` (the other paths kept).
fn solve(t: &CTerm, want: i64, args: &[Value]) -> Option<(CPath, i64)> {
    match t {
        CTerm::Path(p) => Some((p.clone(), want)),
        CTerm::Neg(a) => solve(a, want.checked_neg()?, args),
        CTerm::Bin(BinOp::Add, a, b) => solve(a, want.checked_sub(int(b, args)?)?, args).or_else(|| solve(b, want.checked_sub(int(a, args)?)?, args)),
        CTerm::Bin(BinOp::Sub, a, b) => solve(a, want.checked_add(int(b, args)?)?, args).or_else(|| solve(b, int(a, args)?.checked_sub(want)?, args)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn program(src: &str) -> TProgram {
        qal_compiler::check(&qal_compiler::parse_program(src).unwrap()).unwrap_or_else(|d| panic!("{d:#?}"))
    }

    fn holds_all(tp: &TProgram, f: &TFn, args: &[Value]) -> bool {
        plan(tp, f).atoms.iter().all(|a| instances(a, args).iter().all(|(l, r)| holds(a.op, l, r, args) == Some(true)))
    }

    #[test]
    fn repairs_fields_through_helpers_lookups_and_collections() {
        let tp = program(
            "type C = { spent: Int, taxes: Int, open: Bool }\n\
             fn ok(c: C) -> Bool = c.taxes >= 0 && c.taxes <= c.spent && c.open\n\
             fn at(m: Map[Str, C], k: Str) -> Bool = match map_get(m, k) { Some(x) => ok(x), None => true }\n\
             fn all(m: Map[Str, C]) -> Bool = fold(map_values(m), true, fn(a: Bool, c: C) => a && ok(c))\n\
             fn f(c: C, q: Int, m: Map[Str, C], k: Str) -> Int req ok(c) && q >= 1 && q <= c.spent - c.taxes && at(m, k) && all(m) = q",
        );
        let f = tp.fns.iter().find(|f| f.def.name == "f").unwrap();
        let p = plan(&tp, f);
        assert!(p.atoms.len() >= 8, "{p:?}");
        let c = |s: i64, t: i64, o: bool| Value::Record("C".into(), vec![("spent".into(), Value::Int(s)), ("taxes".into(), Value::Int(t)), ("open".into(), Value::Bool(o))]);
        let args = vec![c(-5, 9, false), Value::Int(-3), Value::Map(vec![(Value::Str("a".into()), c(1, 4, false)), (Value::Str("b".into()), c(-2, -1, true))]), Value::Str("a".into())];
        assert!(!holds_all(&tp, f, &args));
        // Atoms that share a path can undo each other's repair; across seeds nearly every input settles.
        let settled = (0..200).filter(|&seed| {
            let mut a = args.clone();
            repair(&p, &mut a, &mut Rng::new(seed));
            holds_all(&tp, f, &a)
        });
        let n = settled.count();
        assert!(n >= 180, "{n} of 200 inputs settled");
    }

    #[test]
    fn what_is_not_an_atom_is_left_alone() {
        let tp = program("fn f(xs: List[Int], n: Int) -> Int req len(xs) > n && n * n > 3 = n");
        let f = tp.fns.iter().find(|f| f.def.name == "f").unwrap();
        let p = plan(&tp, f);
        // `n * n > 3` is an atom (products are terms) but no path can be solved for; `len(xs)` is no term.
        let mut args = vec![Value::List(vec![]), Value::Int(1)];
        repair(&p, &mut args, &mut Rng::new(0));
        assert_eq!(args, vec![Value::List(vec![]), Value::Int(1)]);
    }
}

