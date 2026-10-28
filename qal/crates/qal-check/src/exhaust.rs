//! Match exhaustiveness (usefulness algorithm after Maranget): finds a value no arm covers.
//!
//! Guarded arms do not count: a guard may be false. Int, Float, Str, List and fn values have
//! infinitely many constructors, so only a wildcard or binding covers them completely.

use qal_core::ast::{Type, TypeBody};
use qal_core::tast::{TPat, TProgram};

#[derive(Debug, Clone, PartialEq)]
enum Key {
    Tag(u32),
    Bool(bool),
    Unit,
    Record,
    Lit(String),
}

/// A pattern reduced to constructor form.
#[derive(Debug, Clone)]
enum P {
    Wild,
    C(Key, Vec<P>),
}

struct Ctor {
    key: Key,
    /// Display: variant name, `true`, `()`, or the record type name.
    name: String,
    args: Vec<Type>,
    /// Field names, for records.
    fields: Vec<String>,
}

/// A missing value, printed in pattern syntax.
enum W {
    Wild,
    C(String, Vec<W>, Vec<String>, bool),
}

fn show(w: &W) -> String {
    match w {
        W::Wild => "_".into(),
        W::C(name, args, _, false) if args.is_empty() => name.clone(),
        W::C(name, args, _, false) => format!("{name}({})", args.iter().map(show).collect::<Vec<_>>().join(", ")),
        W::C(name, args, fields, true) => {
            let fs: Vec<String> = fields.iter().zip(args).map(|(f, a)| format!("{f}: {}", show(a))).collect();
            format!("{name} {{ {} }}", fs.join(", "))
        }
    }
}

/// Returns a pattern (as text) for a value of type `ty` that none of `rows` matches, if any.
pub fn missing_pattern(tp: &TProgram, ty: &Type, rows: &[&TPat]) -> Option<String> {
    let rows: Vec<Vec<P>> = rows.iter().map(|p| vec![reduce(tp, p, ty)]).collect();
    missing(tp, &rows, std::slice::from_ref(ty)).map(|w| show(&w[0]))
}

/// Constructors of `ty`, or None if it has infinitely many.
fn ctors(tp: &TProgram, ty: &Type) -> Option<Vec<Ctor>> {
    let simple = |key, name: &str| Ctor { key, name: name.into(), args: vec![], fields: vec![] };
    match ty {
        Type::Bool => Some(vec![simple(Key::Bool(false), "false"), simple(Key::Bool(true), "true")]),
        Type::Unit => Some(vec![simple(Key::Unit, "()")]),
        Type::Option(_) | Type::Result(..) => Some(variants(tp, ty)),
        Type::Named(n) | Type::App(n, _) => match tp.decl(ty) {
            Some(TypeBody::Enum(_)) => Some(variants(tp, ty)),
            Some(TypeBody::Record(fs)) => Some(vec![Ctor {
                key: Key::Record,
                name: n.clone(),
                args: fs.iter().map(|(_, t)| t.clone()).collect(),
                fields: fs.iter().map(|(f, _)| f.clone()).collect(),
            }]),
            None => None,
        },
        _ => None,
    }
}

fn variants(tp: &TProgram, ty: &Type) -> Vec<Ctor> {
    tp.variant_names(ty)
        .into_iter()
        .enumerate()
        .map(|(tag, name)| Ctor { key: Key::Tag(tag as u32), name, args: tp.variant_fields(ty, tag as u32), fields: vec![] })
        .collect()
}

fn reduce(tp: &TProgram, p: &TPat, ty: &Type) -> P {
    match p {
        TPat::Wild | TPat::Bind(_) => P::Wild,
        TPat::Int(n) => P::C(Key::Lit(n.to_string()), vec![]),
        TPat::Str(s) => P::C(Key::Lit(format!("{s:?}")), vec![]),
        TPat::Bool(b) => P::C(Key::Bool(*b), vec![]),
        TPat::Unit => P::C(Key::Unit, vec![]),
        TPat::Variant(tag, ps) => {
            let tys = tp.variant_fields(ty, *tag);
            P::C(Key::Tag(*tag), ps.iter().zip(&tys).map(|(p, t)| reduce(tp, p, t)).collect())
        }
        TPat::Record(fs) => {
            let all = tp.record_fields(ty);
            let args = all
                .iter()
                .enumerate()
                .map(|(i, (_, t))| match fs.iter().find(|(f, _)| *f as usize == i) {
                    Some((_, p)) => reduce(tp, p, t),
                    None => P::Wild,
                })
                .collect();
            P::C(Key::Record, args)
        }
    }
}

fn head(row: &[P]) -> Option<&Key> {
    match &row[0] {
        P::C(k, _) => Some(k),
        P::Wild => None,
    }
}

/// Rows that match constructor `key` (with `arity` arguments), with the head replaced by its arguments.
fn specialize(rows: &[Vec<P>], key: &Key, arity: usize) -> Vec<Vec<P>> {
    rows.iter()
        .filter_map(|row| {
            let args = match &row[0] {
                P::C(k, args) if k == key => args.clone(),
                P::C(..) => return None,
                P::Wild => vec![P::Wild; arity],
            };
            Some(args.into_iter().chain(row[1..].iter().cloned()).collect())
        })
        .collect()
}

fn missing(tp: &TProgram, rows: &[Vec<P>], tys: &[Type]) -> Option<Vec<W>> {
    if tys.is_empty() {
        return rows.is_empty().then(Vec::new);
    }
    let heads: Vec<&Key> = rows.iter().filter_map(|r| head(r)).collect();
    let all = ctors(tp, &tys[0]);
    if let Some(all) = &all {
        if all.is_empty() {
            // An enum without variants has no values: nothing in this column can be missing.
            return None;
        }
        if !all.is_empty() && all.iter().all(|c| heads.contains(&&c.key)) {
            for c in all {
                let sub = specialize(rows, &c.key, c.args.len());
                let sub_tys: Vec<Type> = c.args.iter().chain(&tys[1..]).cloned().collect();
                if let Some(mut w) = missing(tp, &sub, &sub_tys) {
                    let rest = w.split_off(c.args.len());
                    let mut out = vec![W::C(c.name.clone(), w, c.fields.clone(), c.key == Key::Record)];
                    out.extend(rest);
                    return Some(out);
                }
            }
            return None;
        }
    }
    let default: Vec<Vec<P>> = rows.iter().filter(|r| head(r).is_none()).map(|r| r[1..].to_vec()).collect();
    let rest = missing(tp, &default, &tys[1..])?;
    let first = match all.as_ref().and_then(|all| all.iter().find(|c| !heads.contains(&&c.key))) {
        Some(c) => W::C(c.name.clone(), c.args.iter().map(|_| W::Wild).collect(), c.fields.clone(), c.key == Key::Record),
        None => W::Wild,
    };
    Some(std::iter::once(first).chain(rest).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use qal_core::ast::{TypeDef, Variant};

    fn prog() -> TProgram {
        TProgram {
            types: vec![
                TypeDef {
                    name: "E".into(),
                    params: vec![],
                    body: TypeBody::Enum(vec![
                        Variant { name: "A".into(), fields: vec![] },
                        Variant { name: "B".into(), fields: vec![Type::Bool, Type::Int] },
                    ]),
                },
                TypeDef { name: "R".into(), params: vec![], body: TypeBody::Record(vec![("x".into(), Type::Bool), ("y".into(), Type::Int)]) },
            ],
            ..TProgram::default()
        }
    }

    fn check(ty: Type, rows: &[TPat]) -> Option<String> {
        missing_pattern(&prog(), &ty, &rows.iter().collect::<Vec<_>>())
    }

    #[test]
    fn option_and_result() {
        let opt = Type::option(Type::Int);
        assert_eq!(check(opt.clone(), &[TPat::Variant(1, vec![TPat::Wild])]), Some("None".into()));
        assert_eq!(check(opt.clone(), &[TPat::Variant(0, vec![])]), Some("Some(_)".into()));
        assert_eq!(check(opt.clone(), &[TPat::Variant(0, vec![]), TPat::Variant(1, vec![TPat::Bind(0)])]), None);
        assert_eq!(check(opt, &[TPat::Variant(0, vec![]), TPat::Variant(1, vec![TPat::Int(3)])]), Some("Some(_)".into()));
        let res = Type::result(Type::Bool, Type::Str);
        assert_eq!(check(res, &[TPat::Variant(0, vec![TPat::Bool(true)]), TPat::Variant(1, vec![TPat::Wild])]), Some("Ok(false)".into()));
    }

    #[test]
    fn user_enums_records_and_nesting() {
        let e = Type::Named("E".into());
        assert_eq!(check(e.clone(), &[TPat::Variant(0, vec![])]), Some("B(_, _)".into()));
        assert_eq!(check(e.clone(), &[TPat::Variant(0, vec![]), TPat::Variant(1, vec![TPat::Bool(true), TPat::Wild])]), Some("B(false, _)".into()));
        assert_eq!(check(e, &[TPat::Wild]), None);
        let r = Type::Named("R".into());
        assert_eq!(check(r.clone(), &[TPat::Record(vec![(0, TPat::Bool(true))])]), Some("R { x: false, y: _ }".into()));
        assert_eq!(check(r, &[TPat::Record(vec![(0, TPat::Bool(true))]), TPat::Record(vec![(0, TPat::Bool(false))])]), None);
        let nested = Type::option(Type::option(Type::Unit));
        assert_eq!(check(nested, &[TPat::Variant(0, vec![]), TPat::Variant(1, vec![TPat::Variant(1, vec![TPat::Unit])])]), Some("Some(None)".into()));
    }

    #[test]
    fn infinite_types_need_a_wildcard() {
        assert_eq!(check(Type::Int, &[TPat::Int(0), TPat::Int(1)]), Some("_".into()));
        assert_eq!(check(Type::Str, &[TPat::Str("a".into()), TPat::Bind(0)]), None);
        assert_eq!(check(Type::Bool, &[TPat::Bool(true), TPat::Bool(false)]), None);
        assert_eq!(check(Type::Unit, &[TPat::Unit]), None);
        assert_eq!(check(Type::Bool, &[]), Some("false".into()));
    }
}
