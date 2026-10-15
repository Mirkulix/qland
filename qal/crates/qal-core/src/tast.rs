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

/// Declares `Builtin` with its source names from one list, so `Builtin::ALL` cannot miss a variant.
macro_rules! builtins {
    ($($variant:ident => $name:literal,)*) => {
        /// Built-in functions (spec §3.5, plus `int_to_float` / `float_to_int`).
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum Builtin {
            $($variant,)*
        }

        impl Builtin {
            /// Every built-in, in declaration order.
            pub const ALL: &'static [Builtin] = &[$(Builtin::$variant,)*];

            /// Source name. `StrLen` and `ListLen` are both `len`.
            pub fn name(self) -> &'static str {
                match self {
                    $(Builtin::$variant => $name,)*
                }
            }
        }
    };
}

builtins! {
    StrLen => "len",
    ListLen => "len",
    Slice => "slice",
    Split => "split",
    Join => "join",
    Trim => "trim",
    Contains => "contains",
    StartsWith => "starts_with",
    ToInt => "to_int",
    ToFloat => "to_float",
    FromInt => "from_int",
    FromFloat => "from_float",
    Chars => "chars",
    Get => "get",
    Push => "push",
    Concat => "concat",
    Map => "map",
    Filter => "filter",
    Fold => "fold",
    Range => "range",
    Reverse => "reverse",
    SortInt => "sort_int",
    IntToFloat => "int_to_float",
    FloatToInt => "float_to_int",
    Print => "print",
    ReadLine => "read_line",
    ReadFile => "read_file",
    WriteFile => "write_file",
    NowMs => "now_ms",
    RandInt => "rand_int",
    ReadAll => "read_all",
    SortStr => "sort_str",
    CharCode => "char_code",
    FromCharCode => "from_char_code",
    ToUpper => "to_upper",
    ToLower => "to_lower",
    Replace => "replace",
    IndexOf => "index_of",
}

impl Builtin {
    /// The effect a call needs (map/filter/fold need the effects of their function argument).
    pub fn effect(self) -> Option<&'static str> {
        match self {
            Builtin::Print | Builtin::ReadLine | Builtin::ReadAll => Some("io"),
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
        "float_to_int", "print", "read_line", "read_file", "write_file", "now_ms", "rand_int", "read_all", "sort_str",
        "char_code", "from_char_code", "to_upper", "to_lower", "replace", "index_of",
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

    /// Payload types of variant `tag` of the sum type `ty` (empty if there is no such variant).
    pub fn variant_fields(&self, ty: &Type, tag: u32) -> Vec<Type> {
        match (ty, tag) {
            (Type::Option(_), TAG_NONE) => vec![],
            (Type::Option(t), _) => vec![(**t).clone()],
            (Type::Result(t, _), TAG_OK) => vec![(**t).clone()],
            (Type::Result(_, e), _) => vec![(**e).clone()],
            (Type::Named(n), _) => match self.type_def(n).map(|d| &d.body) {
                Some(TypeBody::Enum(vs)) => vs.get(tag as usize).map(|v| v.fields.clone()).unwrap_or_default(),
                _ => Vec::new(),
            },
            _ => Vec::new(),
        }
    }

    /// Field names and types of the record type `ty` (empty if it is not a record).
    pub fn record_fields(&self, ty: &Type) -> Vec<(String, Type)> {
        match ty {
            Type::Named(n) => match self.type_def(n).map(|d| &d.body) {
                Some(TypeBody::Record(fs)) => fs.clone(),
                _ => Vec::new(),
            },
            _ => Vec::new(),
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

    #[test]
    fn builtin_names_match_the_variants() {
        // `ALL` and `name()` come from one list (the `builtins!` macro); `NAMES` is written out
        // because user functions are checked against it, and must be exactly the set of names.
        let mut from_variants: Vec<&str> = Builtin::ALL.iter().map(|b| b.name()).collect();
        from_variants.sort_unstable();
        from_variants.dedup();
        let mut names = Builtin::NAMES.to_vec();
        names.sort_unstable();
        assert_eq!(names.len(), Builtin::NAMES.len(), "Builtin::NAMES has duplicates");
        assert_eq!(names, from_variants);
        assert_eq!(Builtin::ALL.len(), 38);
        assert_eq!(Builtin::ALL[0], Builtin::StrLen);
        assert_eq!(Builtin::ReadAll.effect(), Some("io"));
        assert_eq!((Builtin::SortStr.effect(), Builtin::Replace.effect()), (None, None));
    }
}
