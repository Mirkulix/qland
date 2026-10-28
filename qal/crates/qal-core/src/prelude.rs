//! The prelude: built-in record and enum types for services (Phase 3 spec §2.1, §3.1, §3.2).
//!
//! Like `Option` and `Result`, these types exist in every program without being defined: their
//! names and variant names are reserved, and the host knows their layout (declaration order).
//! In canonical forms they are referenced by name, never by hash (see `canon`).
//!
//! Phase 5a (self-hosting spec §5.2) adds one generic prelude type, the append-only accumulator
//! `Accum[T]` (the spec's `Acc`, renamed because `bench/solutions/qal/03-csv-sum.qal` defines a
//! type `Acc`), used with `acc_list` to build lists in O(1) per element.

use crate::ast::{Type, TypeBody, TypeDef, Variant};

/// The service prelude type names (Phase 3 & 7), in declaration order.
pub const TYPE_NAMES: [&str; 10] = ["HttpHeader", "HttpRequest", "HttpResponse", "HttpCall", "NetError", "SqlValue", "DbRows", "DbError", "ModelResponse", "ModelError"];

/// The accumulator type of Phase 5a: `type Accum[T] = AccEnd | AccPush(Accum[T], T)`.
pub const ACCUM: &str = "Accum";

/// Every prelude type name, in declaration order (the service types, then `Accum`).
pub const ALL_TYPE_NAMES: [&str; 11] = ["HttpHeader", "HttpRequest", "HttpResponse", "HttpCall", "NetError", "SqlValue", "DbRows", "DbError", "ModelResponse", "ModelError", ACCUM];

fn named(n: &str) -> Type {
    Type::Named(n.to_string())
}

fn record(name: &str, fields: &[(&str, Type)]) -> TypeDef {
    TypeDef { name: name.into(), params: vec![], body: TypeBody::Record(fields.iter().map(|(f, t)| (f.to_string(), t.clone())).collect()) }
}

fn enumeration(name: &str, variants: &[(&str, &[Type])]) -> TypeDef {
    TypeDef { name: name.into(), params: vec![], body: TypeBody::Enum(variants.iter().map(|(v, fs)| Variant { name: v.to_string(), fields: fs.to_vec() }).collect()) }
}

/// The prelude type definitions:
///
/// ```text
/// type HttpHeader   = { name: Str, value: Str }
/// type HttpRequest  = { method: Str, path: Str, query: Str, headers: List[HttpHeader], body: Str }
/// type HttpResponse = { status: Int, headers: List[HttpHeader], body: Str }
/// type HttpCall     = { method: Str, url: Str, headers: List[HttpHeader], body: Str }
/// type NetError     = NetDenied(Str) | NetInvalid(Str) | NetConnect(Str) | NetTimeout | NetTooLarge
/// type SqlValue     = SqlNull | SqlInt(Int) | SqlFloat(Float) | SqlText(Str)
/// type DbRows       = { columns: List[Str], rows: List[List[SqlValue]] }
/// type DbError      = DbConstraint(Str) | DbSql(Str) | DbBusy | DbFull | DbTooLarge
/// type Accum[T]     = AccEnd | AccPush(Accum[T], T)
/// ```
pub fn types() -> Vec<TypeDef> {
    let headers = Type::list(named("HttpHeader"));
    vec![
        record("HttpHeader", &[("name", Type::Str), ("value", Type::Str)]),
        record("HttpRequest", &[("method", Type::Str), ("path", Type::Str), ("query", Type::Str), ("headers", headers.clone()), ("body", Type::Str)]),
        record("HttpResponse", &[("status", Type::Int), ("headers", headers.clone()), ("body", Type::Str)]),
        record("HttpCall", &[("method", Type::Str), ("url", Type::Str), ("headers", headers), ("body", Type::Str)]),
        enumeration("NetError", &[("NetDenied", &[Type::Str]), ("NetInvalid", &[Type::Str]), ("NetConnect", &[Type::Str]), ("NetTimeout", &[]), ("NetTooLarge", &[])]),
        enumeration("SqlValue", &[("SqlNull", &[]), ("SqlInt", &[Type::Int]), ("SqlFloat", &[Type::Float]), ("SqlText", &[Type::Str])]),
        record("DbRows", &[("columns", Type::list(Type::Str)), ("rows", Type::list(Type::list(named("SqlValue"))))]),
        enumeration("DbError", &[("DbConstraint", &[Type::Str]), ("DbSql", &[Type::Str]), ("DbBusy", &[]), ("DbFull", &[]), ("DbTooLarge", &[])]),
        record("ModelResponse", &[("text", Type::Str), ("model", Type::Str), ("in_tokens", Type::Int), ("out_tokens", Type::Int), ("cost_microcents", Type::Int)]),
        enumeration("ModelError", &[("ModelDenied", &[Type::Str]), ("ModelBudgetExceeded", &[Type::Int]), ("ModelFailed", &[Type::Str]), ("ModelInvalid", &[Type::Str]), ("ModelTimeout", &[])]),
        TypeDef {
            name: ACCUM.into(),
            params: vec!["T".into()],
            body: TypeBody::Enum(vec![
                Variant { name: "AccEnd".into(), fields: vec![] },
                Variant { name: "AccPush".into(), fields: vec![Type::App(ACCUM.into(), vec![named("T")]), named("T")] },
            ]),
        },
    ]
}

/// Number of type parameters of the prelude type `name` (None if it is not one).
pub fn arity(name: &str) -> Option<usize> {
    match name {
        ACCUM => Some(1),
        _ if TYPE_NAMES.contains(&name) => Some(0),
        _ => None,
    }
}

/// True if `name` is a prelude type.
pub fn is_type(name: &str) -> bool {
    ALL_TYPE_NAMES.contains(&name)
}

/// The prelude enum a variant name belongs to.
pub fn variant_owner(name: &str) -> Option<&'static str> {
    Some(match name {
        "NetDenied" | "NetInvalid" | "NetConnect" | "NetTimeout" | "NetTooLarge" => "NetError",
        "SqlNull" | "SqlInt" | "SqlFloat" | "SqlText" => "SqlValue",
        "DbConstraint" | "DbSql" | "DbBusy" | "DbFull" | "DbTooLarge" => "DbError",
        "ModelDenied" | "ModelBudgetExceeded" | "ModelFailed" | "ModelInvalid" | "ModelTimeout" => "ModelError",
        "AccEnd" | "AccPush" => ACCUM,
        _ => return None,
    })
}

/// True if `name` is a prelude type or a variant of a prelude enum (reserved in user code).
pub fn is_reserved(name: &str) -> bool {
    is_type(name) || variant_owner(name).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_variants_match_the_definitions() {
        let ts = types();
        assert_eq!(ts.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(), ALL_TYPE_NAMES.to_vec());
        assert_eq!(&ALL_TYPE_NAMES[..10], &TYPE_NAMES);
        for t in &ts {
            assert!(is_type(&t.name) && is_reserved(&t.name));
            if let TypeBody::Enum(vs) = &t.body {
                for v in vs {
                    assert_eq!(variant_owner(&v.name), Some(ALL_TYPE_NAMES.iter().copied().find(|n| *n == t.name).unwrap()), "{}", v.name);
                }
            }
        }
        let variants: usize = ts.iter().map(|t| if let TypeBody::Enum(vs) = &t.body { vs.len() } else { 0 }).sum();
        assert_eq!(variants, 21);
        assert_eq!((arity("Accum"), arity("DbRows"), arity("Option")), (Some(1), Some(0), None));
        for t in &ts {
            assert_eq!(arity(&t.name), Some(t.params.len()));
        }
        assert!(!is_reserved("Point") && variant_owner("Some").is_none() && !is_type("Option"));
    }
}
