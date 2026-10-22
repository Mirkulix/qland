//! QAL core: the program graph (AST), the typed program, canonical form and hashes, structured
//! diagnostics, the prelude types, the static SQL check and the typed form `QALT`.

pub mod ast;
pub mod canon;
pub mod diag;
pub mod lexer;
pub mod module;
pub mod prelude;
pub mod printer;
pub mod sql;
pub mod tast;
pub mod tform;

pub use printer::{print_expr, print_pat, print_program};

pub const KNOWN_EFFECTS: &[&str] = &["io", "fs", "time", "rand", "net", "db"];
pub const MAX_INSTANCES: usize = 1000;
