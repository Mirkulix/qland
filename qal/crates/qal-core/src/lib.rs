//! QAL core: the program graph (AST), the typed program, canonical form and hashes, structured
//! diagnostics, the prelude types, the static SQL check and the typed form `QALT`.

pub mod ast;
pub mod canon;
pub mod diag;
pub mod prelude;
pub mod sql;
pub mod tast;
pub mod tform;
