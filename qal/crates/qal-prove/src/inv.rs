//! Type invariants: a pure `fn inv_T(x: T) -> Bool` states what every value of the record or enum
//! `T` keeps. The property tests then check it for every pure function that takes or returns a
//! `T`, also inside lists, maps, options, records and variants: inputs are repaired towards the
//! invariant and must satisfy it (else they are discarded, like inputs that fail `req`), and a
//! result that holds a `T` breaking it fails the property.
//!
//! Why: in the mutation study of 2026-10-25 the invariants of the order system caught three bugs
//! once they were written as `req`/`ens` on five functions, each with its frame conditions; agents
//! do not write that many contracts. One line per type says the same for every function at once.

use crate::gen::Rng;
use crate::repair::{self, Plan};
use qal_core::ast::{Type, TypeBody};
use qal_core::tast::TProgram;
use qal_run::{RunError, Runner, Value, Values};
use std::collections::{BTreeMap, BTreeSet};

/// Prefix of an invariant function's name: `inv_` and the type's name.
pub const PREFIX: &str = "inv_";

/// The invariants of a program: type name -> (function name, repair plan of its body).
#[derive(Debug, Clone, Default)]
pub struct Invariants {
    by_type: BTreeMap<String, (String, Plan)>,
}

impl Invariants {
    pub fn of(tp: &TProgram) -> Invariants {
        let mut by_type = BTreeMap::new();
        for f in &tp.fns {
            let Some(t) = f.def.name.strip_prefix(PREFIX) else { continue };
            if f.def.effects.is_empty() && f.def.params.len() == 1 && f.def.ret == Type::Bool && f.def.params[0].ty == Type::Named(t.to_string()) && tp.type_def(t).is_some() {
                by_type.insert(t.to_string(), (f.def.name.clone(), repair::plan_of_body(tp, f)));
            }
        }
        Invariants { by_type }
    }

    pub fn is_empty(&self) -> bool {
        self.by_type.is_empty()
    }

    pub fn is_invariant(&self, f: &str) -> bool {
        self.by_type.values().any(|(n, _)| n == f)
    }

    /// True if values of `ty` can hold a type with an invariant.
    pub fn reach(&self, tp: &TProgram, ty: &Type) -> bool {
        !self.is_empty() && reaches(tp, ty, &self.by_type, &mut BTreeSet::new())
    }

    /// Repairs every value with an invariant inside `args` towards it.
    pub fn repair(&self, args: &mut [Value], rng: &mut Rng) {
        for a in args {
            walk_mut(a, &mut |name, v| {
                if let Some((_, plan)) = self.by_type.get(name) {
                    repair::repair(plan, std::slice::from_mut(v), rng);
                }
            });
        }
    }

    /// The first value with an invariant inside `v` that breaks it: (type, invariant, value).
    /// An invariant that traps on a value counts as broken.
    pub fn broken(&self, runner: &Runner, v: &Value) -> Option<(String, String, Value)> {
        let mut found = None;
        walk(v, &mut |name, x| {
            if found.is_some() {
                return;
            }
            if let Some((f, _)) = self.by_type.get(name) {
                if !matches!(runner.call(f, std::slice::from_ref(x)), Ok(Value::Bool(true))) {
                    found = Some((name.to_string(), f.clone(), x.clone()));
                }
            }
        });
        found
    }

    /// The failure of a result that breaks an invariant.
    pub fn failure(def: &str, (ty, inv, value): (String, String, Value)) -> RunError {
        let mut e = RunError::new("contract_violation", def, format!("the result of `{def}` holds a `{ty}` that breaks `{inv}`"));
        e.path = "result".into();
        e.contract = Some(format!("{inv}({ty})"));
        e.values = Values(vec![(ty.to_lowercase(), value)]);
        e.hints.push(format!("every function that returns a `{ty}` must keep `{inv}`; fix the body, or the invariant if the rule changed"));
        e
    }
}

fn reaches(tp: &TProgram, ty: &Type, invs: &BTreeMap<String, (String, Plan)>, seen: &mut BTreeSet<String>) -> bool {
    match ty {
        Type::List(t) | Type::Option(t) => reaches(tp, t, invs, seen),
        Type::Result(a, b) | Type::Map(a, b) => reaches(tp, a, invs, seen) || reaches(tp, b, invs, seen),
        Type::Named(_) | Type::App(..) => {
            let key = ty.to_string();
            if invs.contains_key(&key) {
                return true;
            }
            if !seen.insert(key) {
                return false;
            }
            match tp.decl(ty) {
                Some(TypeBody::Record(fs)) => fs.iter().any(|(_, t)| reaches(tp, t, invs, seen)),
                Some(TypeBody::Enum(vs)) => vs.iter().any(|v| v.fields.iter().any(|t| reaches(tp, t, invs, seen))),
                _ => false,
            }
        }
        _ => false,
    }
}

/// Calls `f(type name, value)` for every record and variant value inside `v`, outermost first.
fn walk<'a>(v: &'a Value, f: &mut impl FnMut(&str, &'a Value)) {
    match v {
        Value::Record(name, fs) => {
            f(name, v);
            for (_, x) in fs {
                walk(x, f);
            }
        }
        Value::List(xs) => xs.iter().for_each(|x| walk(x, f)),
        Value::Map(es) => es.iter().for_each(|(k, x)| {
            walk(k, f);
            walk(x, f);
        }),
        Value::Variant(_, xs) => xs.iter().for_each(|x| walk(x, f)),
        _ => {}
    }
}

/// `walk` that may change the values; inner values first, so a record is repaired after its parts.
fn walk_mut(v: &mut Value, f: &mut impl FnMut(&str, &mut Value)) {
    match v {
        Value::Record(_, fs) => {
            for (_, x) in fs.iter_mut() {
                walk_mut(x, f);
            }
            let Value::Record(name, _) = v else { return };
            let name = name.clone();
            f(&name, v);
        }
        Value::List(xs) => xs.iter_mut().for_each(|x| walk_mut(x, f)),
        Value::Map(es) => es.iter_mut().for_each(|(_, x)| walk_mut(x, f)),
        Value::Variant(_, xs) => xs.iter_mut().for_each(|x| walk_mut(x, f)),
        _ => {}
    }
}
