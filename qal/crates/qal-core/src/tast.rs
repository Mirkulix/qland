//! Typed, name-resolved program: the checker's output and the code generator's input.
//!
//! Every expression carries its fully inferred type (never `Type::Var`). Locals are numbered per
//! definition (`Local(id)`; parameters are ids `0..n`), functions are indices into `TProgram::fns`,
//! variants are tags and record fields are declaration-order indices.

use crate::ast::{BinOp, FnDef, Type, TypeBody, TypeDef, UnOp};

/// Variant tags of the built-in sum types.
pub const TAG_NONE: u32 = 0;
pub const TAG_SOME: u32 = 1;
pub const TAG_OK: u32 = 0;
pub const TAG_ERR: u32 = 1;

/// Built-in functions (spec §3.5, plus `int_to_float` / `float_to_int`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Builtin {
    StrLen,
    ListLen,
    Slice,
    Split,
    Join,
    Trim,
    Contains,
    StartsWith,
    ToInt,
    ToFloat,
    FromInt,
    FromFloat,
    Chars,
    Get,
    Push,
    Concat,
    Map,
    Filter,
    Fold,
    Range,
    Reverse,
    SortInt,
    IntToFloat,
    FloatToInt,
    Print,
    ReadLine,
    ReadFile,
    WriteFile,
    NowMs,
    RandInt,
}

impl Builtin {
    /// Source name. `StrLen` and `ListLen` are both `len`.
    pub fn name(self) -> &'static str {
        match self {
            Builtin::StrLen | Builtin::ListLen => "len",
            Builtin::Slice => "slice",
            Builtin::Split => "split",
            Builtin::Join => "join",
            Builtin::Trim => "trim",
            Builtin::Contains => "contains",
            Builtin::StartsWith => "starts_with",
            Builtin::ToInt => "to_int",
            Builtin::ToFloat => "to_float",
            Builtin::FromInt => "from_int",
            Builtin::FromFloat => "from_float",
            Builtin::Chars => "chars",
            Builtin::Get => "get",
            Builtin::Push => "push",
            Builtin::Concat => "concat",
            Builtin::Map => "map",
            Builtin::Filter => "filter",
            Builtin::Fold => "fold",
            Builtin::Range => "range",
            Builtin::Reverse => "reverse",
            Builtin::SortInt => "sort_int",
            Builtin::IntToFloat => "int_to_float",
            Builtin::FloatToInt => "float_to_int",
            Builtin::Print => "print",
            Builtin::ReadLine => "read_line",
            Builtin::ReadFile => "read_file",
            Builtin::WriteFile => "write_file",
            Builtin::NowMs => "now_ms",
            Builtin::RandInt => "rand_int",
        }
    }

    /// The effect a call needs (map/filter/fold need the effects of their function argument).
    pub fn effect(self) -> Option<&'static str> {
        match self {
            Builtin::Print | Builtin::ReadLine => Some("io"),
            Builtin::ReadFile | Builtin::WriteFile => Some("fs"),
            Builtin::NowMs => Some("time"),
            Builtin::RandInt => Some("rand"),
            _ => None,
        }
    }

    /// All source names of built-in functions (reserved: user functions cannot use them).
    pub const NAMES: &'static [&'static str] = &[
        "len", "slice", "split", "join", "trim", "contains", "starts_with", "to_int", "to_float", "from_int", "from_float",
        "chars", "get", "push", "concat", "map", "filter", "fold", "range", "reverse", "sort_int", "int_to_float",
        "float_to_int", "print", "read_line", "read_file", "write_file", "now_ms", "rand_int",
    ];
}

#[derive(Debug, Clone, PartialEq)]
pub struct TExpr {
    pub ty: Type,
    pub kind: TKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TKind {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Unit,
    Local(u32),
    /// A top-level function used as a value (index into `TProgram::fns`).
    FnRef(usize),
    /// `path` addresses the node for runtime failures (overflow, division by zero).
    Bin(BinOp, Box<TExpr>, Box<TExpr>, String),
    Un(UnOp, Box<TExpr>, String),
    If(Box<TExpr>, Box<TExpr>, Box<TExpr>),
    Let(u32, Box<TExpr>, Box<TExpr>),
    /// Direct call of `TProgram::fns[index]`.
    Call(usize, Vec<TExpr>),
    /// Call through a fn-typed value.
    CallIndirect(Box<TExpr>, Vec<TExpr>),
    Builtin(Builtin, Vec<TExpr>, String),
    List(Vec<TExpr>),
    /// Variant of the enum/Option/Result type `ty`, with payload values.
    Variant(u32, Vec<TExpr>),
    /// Record of type `ty`: (declaration index, value) for every field, in source order
    /// (the order in which the values are evaluated).
    Record(Vec<(u32, TExpr)>),
    /// Field access by declaration-order index.
    Field(Box<TExpr>, u32),
    Match(Box<TExpr>, Vec<TArm>),
    Seq(Vec<TExpr>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TArm {
    pub pat: TPat,
    pub guard: Option<TExpr>,
    pub body: TExpr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TPat {
    Wild,
    Bind(u32),
    Int(i64),
    Str(String),
    Bool(bool),
    Unit,
    Variant(u32, Vec<TPat>),
    /// (field index, pattern) for the mentioned fields.
    Record(Vec<(u32, TPat)>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TFn {
    /// The source definition (names, signature, contract text).
    pub def: FnDef,
    /// Types of all locals by id; parameters first.
    pub locals: Vec<Type>,
    pub req: Vec<TExpr>,
    pub ens: Vec<TExpr>,
    /// Local id of the result variable `r` (only when `ens` is non-empty).
    pub result_local: Option<u32>,
    pub body: TExpr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TTest {
    pub name: String,
    pub locals: Vec<Type>,
    pub body: TExpr,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TProgram {
    pub types: Vec<TypeDef>,
    pub fns: Vec<TFn>,
    pub tests: Vec<TTest>,
}

impl TProgram {
    pub fn type_def(&self, name: &str) -> Option<&TypeDef> {
        self.types.iter().find(|t| t.name == name)
    }

    pub fn fn_index(&self, name: &str) -> Option<usize> {
        self.fns.iter().position(|f| f.def.name == name)
    }

    /// Variant names of a sum type, by tag. Empty for non-sum types.
    pub fn variant_names(&self, ty: &Type) -> Vec<String> {
        match ty {
            Type::Option(_) => vec!["None".into(), "Some".into()],
            Type::Result(..) => vec!["Ok".into(), "Err".into()],
            Type::Named(n) => match self.type_def(n).map(|d| &d.body) {
                Some(TypeBody::Enum(vs)) => vs.iter().map(|v| v.name.clone()).collect(),
                _ => vec![],
            },
            _ => vec![],
        }
    }

    /// Payload types of variant `tag` of the sum type `ty`.
    pub fn variant_fields(&self, ty: &Type, tag: u32) -> Vec<Type> {
        match (ty, tag) {
            (Type::Option(_), TAG_NONE) => vec![],
            (Type::Option(t), _) => vec![(**t).clone()],
            (Type::Result(t, _), TAG_OK) => vec![(**t).clone()],
            (Type::Result(_, e), _) => vec![(**e).clone()],
            (Type::Named(n), _) => match self.type_def(n).map(|d| &d.body) {
                Some(TypeBody::Enum(vs)) => vs[tag as usize].fields.clone(),
                _ => panic!("`{n}` is not an enum"),
            },
            _ => panic!("{ty} is not a sum type"),
        }
    }

    /// Field names and types of the record type `ty`.
    pub fn record_fields(&self, ty: &Type) -> Vec<(String, Type)> {
        match ty {
            Type::Named(n) => match self.type_def(n).map(|d| &d.body) {
                Some(TypeBody::Record(fs)) => fs.clone(),
                _ => panic!("`{n}` is not a record"),
            },
            _ => panic!("{ty} is not a record"),
        }
    }

    pub fn is_record(&self, ty: &Type) -> bool {
        matches!(ty, Type::Named(n) if matches!(self.type_def(n).map(|d| &d.body), Some(TypeBody::Record(_))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Variant;

    #[test]
    fn variant_and_record_lookup() {
        let p = TProgram {
            types: vec![
                TypeDef { name: "E".into(), body: TypeBody::Enum(vec![Variant { name: "A".into(), fields: vec![] }, Variant { name: "B".into(), fields: vec![Type::Int, Type::Str] }]) },
                TypeDef { name: "R".into(), body: TypeBody::Record(vec![("x".into(), Type::Float)]) },
            ],
            ..TProgram::default()
        };
        assert_eq!(p.variant_names(&Type::Named("E".into())), vec!["A".to_string(), "B".to_string()]);
        assert_eq!(p.variant_fields(&Type::Named("E".into()), 1), vec![Type::Int, Type::Str]);
        assert_eq!(p.variant_fields(&Type::result(Type::Int, Type::Str), TAG_ERR), vec![Type::Str]);
        assert_eq!(p.variant_fields(&Type::option(Type::Int), TAG_NONE), vec![]);
        assert_eq!(p.record_fields(&Type::Named("R".into())), vec![("x".to_string(), Type::Float)]);
        assert!(p.is_record(&Type::Named("R".into())) && !p.is_record(&Type::Named("E".into())));
        assert_eq!(Builtin::StrLen.name(), "len");
        assert_eq!(Builtin::WriteFile.effect(), Some("fs"));
        assert!(Builtin::NAMES.contains(&Builtin::RandInt.name()));
    }
}
