//! QAL text form: a token-dense view of the program graph for LLM agents.

pub mod lexer;
pub mod parser;
pub mod printer;

pub use parser::parse_program;
pub use printer::{print_expr, print_pat, print_program};
