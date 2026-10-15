//! Abstract syntax of QAL (Phase 1b): the program as written, before name resolution.

use std::fmt;

/// Name of the result variable inside `ens` contracts.
pub const RESULT_VAR: &str = "r";

/// Value types. `Var` only exists while the checker infers types; checked output never contains it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Int,
    Float,
    Bool,
    Unit,
    Str,
    List(Box<Type>),
    Option(Box<Type>),
    Result(Box<Type>, Box<Type>),
    /// A record or enum declared with `type`.
    Named(String),
    /// `fn(A, B) -> R !e`: a reference to a top-level function (no captures).
    Fn(Vec<Type>, Box<Type>, Vec<String>),
    /// Inference variable (checker-internal).
    Var(u32),
}

impl Type {
    pub fn list(t: Type) -> Type {
        Type::List(Box::new(t))
    }

    pub fn option(t: Type) -> Type {
        Type::Option(Box::new(t))
    }

    pub fn result(t: Type, e: Type) -> Type {
        Type::Result(Box::new(t), Box::new(e))
    }

    /// Type names that cannot be declared with `type`.
    pub const BUILTIN_NAMES: &'static [&'static str] = &["Int", "Float", "Bool", "Unit", "Str", "List", "Option", "Result"];
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Int => write!(f, "Int"),
            Type::Float => write!(f, "Float"),
            Type::Bool => write!(f, "Bool"),
            Type::Unit => write!(f, "Unit"),
            Type::Str => write!(f, "Str"),
            Type::List(t) => write!(f, "List[{t}]"),
            Type::Option(t) => write!(f, "Option[{t}]"),
            Type::Result(t, e) => write!(f, "Result[{t}, {e}]"),
            Type::Named(n) => write!(f, "{n}"),
            Type::Fn(ps, r, effs) => {
                let ps: Vec<String> = ps.iter().map(|p| p.to_string()).collect();
                // A function-typed result is parenthesized so its effects stay distinguishable from ours.
                match **r {
                    Type::Fn(..) => write!(f, "fn({}) -> ({r})", ps.join(", "))?,
                    _ => write!(f, "fn({}) -> {r}", ps.join(", "))?,
                }
                for e in effs {
                    write!(f, " !{e}")?;
                }
                Ok(())
            }
            Type::Var(_) => write!(f, "_"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    /// `++`: Str and List concatenation.
    Concat,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

impl BinOp {
    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::Concat => "++",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::And => "&&",
            BinOp::Or => "||",
        }
    }

    /// Binding strength: higher binds tighter.
    pub fn precedence(self) -> u8 {
        match self {
            BinOp::Or => 1,
            BinOp::And => 2,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => 3,
            BinOp::Add | BinOp::Sub | BinOp::Concat => 4,
            BinOp::Mul | BinOp::Div | BinOp::Rem => 5,
        }
    }

    /// Comparisons are non-associative: `a < b < c` is a parse error.
    pub fn is_comparison(self) -> bool {
        matches!(self, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Int(i64),
    /// Always finite and non-negative; negative values are `Un(Neg, Float)`.
    Float(f64),
    Bool(bool),
    Str(String),
    Unit,
    Var(String),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Un(UnOp, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    /// `let name (: type)? = value in body`
    Let(String, Option<Type>, Box<Expr>, Box<Expr>),
    /// `name(args)`: function, built-in, variant constructor or fn-typed local.
    Call(String, Vec<Expr>),
    /// `[a, b, c]`
    List(Vec<Expr>),
    /// `Name { field: value, ... }`
    Record(String, Vec<(String, Expr)>),
    /// `target.field`
    Field(Box<Expr>, String),
    /// `match scrutinee { arms }`
    Match(Box<Expr>, Vec<Arm>),
    /// `{ a; b; c }`: evaluates all, discards the (Unit) values of all but the last.
    Seq(Vec<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Arm {
    pub pat: Pat,
    pub guard: Option<Expr>,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pat {
    /// `_`
    Wild,
    /// lowercase name: binds the value
    Bind(String),
    Int(i64),
    Str(String),
    Bool(bool),
    Unit,
    /// `Name` or `Name(p, ...)`
    Variant(String, Vec<Pat>),
    /// `Name { field: p, other }`; fields not mentioned match anything.
    Record(String, Vec<(String, Pat)>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FnDef {
    pub name: String,
    pub params: Vec<Param>,
    pub ret: Type,
    pub effects: Vec<String>,
    pub req: Vec<Expr>,
    pub ens: Vec<Expr>,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TestDef {
    pub name: String,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub name: String,
    pub fields: Vec<Type>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeBody {
    Record(Vec<(String, Type)>),
    Enum(Vec<Variant>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeDef {
    pub name: String,
    pub body: TypeBody,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Def {
    Type(TypeDef),
    Fn(FnDef),
    Test(TestDef),
}

impl Def {
    pub fn name(&self) -> &str {
        match self {
            Def::Type(t) => &t.name,
            Def::Fn(f) => &f.name,
            Def::Test(t) => &t.name,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Program {
    pub defs: Vec<Def>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn types_display_as_source_syntax() {
        let t = Type::Fn(vec![Type::list(Type::Str), Type::Named("User".into())], Box::new(Type::result(Type::Int, Type::Str)), vec!["io".into(), "fs".into()]);
        assert_eq!(t.to_string(), "fn(List[Str], User) -> Result[Int, Str] !io !fs");
        assert_eq!(Type::option(Type::Float).to_string(), "Option[Float]");
        let inner = Type::Fn(vec![], Box::new(Type::Int), vec!["io".into()]);
        assert_eq!(Type::Fn(vec![], Box::new(inner), vec![]).to_string(), "fn() -> (fn() -> Int !io)");
    }

    #[test]
    fn comparison_classification() {
        assert!(BinOp::Le.is_comparison());
        assert!(!BinOp::Concat.is_comparison());
        assert_eq!(BinOp::Concat.precedence(), BinOp::Add.precedence());
    }
}
