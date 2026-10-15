//! The QAL pipeline as one call: text -> parse -> check -> compile -> runner.

use qal_core::ast::{Def, Program, Type};
use qal_core::diag::Diagnostic;
use qal_run::{RunError, Runner, Value};

pub struct Loaded {
    pub program: Program,
    pub runner: Runner,
}

/// Parses, checks and compiles `src`. Any diagnostic means nothing is compiled.
pub fn load(src: &str) -> Result<Loaded, Vec<Diagnostic>> {
    let program = qal_text::parse_program(src).map_err(|d| vec![d])?;
    let diags = qal_check::check_program(&program);
    if !diags.is_empty() {
        return Err(diags);
    }
    let runner = Runner::new(qal_wasm::compile(&program)).expect("checked programs compile to valid wasm");
    Ok(Loaded { program, runner })
}

/// Converts CLI strings to values according to the signature of fn `name`.
pub fn parse_args(program: &Program, name: &str, raw: &[String]) -> Result<Vec<Value>, RunError> {
    let bad = |message: String| RunError { kind: "bad_args", def: name.to_string(), message, contract: None };
    let f = program
        .defs
        .iter()
        .find_map(|d| match d {
            Def::Fn(f) if f.name == name => Some(f),
            _ => None,
        })
        .ok_or_else(|| RunError { kind: "unknown_function", def: name.to_string(), message: format!("no function `{name}`"), contract: None })?;
    if f.params.len() != raw.len() {
        return Err(bad(format!("`{name}` takes {} argument(s), {} given", f.params.len(), raw.len())));
    }
    f.params
        .iter()
        .zip(raw)
        .map(|(p, s)| match p.ty {
            Type::Int => s.parse::<i64>().map(Value::Int).map_err(|_| bad(format!("`{}` must be Int, got `{s}`", p.name))),
            Type::Bool => match s.as_str() {
                "true" => Ok(Value::Bool(true)),
                "false" => Ok(Value::Bool(false)),
                _ => Err(bad(format!("`{}` must be Bool (true/false), got `{s}`", p.name))),
            },
            Type::Unit => match s.as_str() {
                "()" => Ok(Value::Unit(())),
                _ => Err(bad(format!("`{}` must be Unit `()`, got `{s}`", p.name))),
            },
        })
        .collect()
}
