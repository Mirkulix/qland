//! The QAL pipeline: text -> parse -> check -> compile (per entry point) -> runner.

use qal_core::ast::{Program, Type};
use qal_core::diag::Diagnostic;
use qal_core::tast::TProgram;
use qal_run::{Permissions, RunError, Runner, Value};
use qal_wasm::Entry;

pub struct Loaded {
    pub program: Program,
    pub typed: TProgram,
}

/// Parses and checks `src`. Any diagnostic means nothing can be compiled.
pub fn load(src: &str) -> Result<Loaded, Vec<Diagnostic>> {
    let program = qal_text::parse_program(src).map_err(|d| vec![d])?;
    let typed = qal_check::check(&program)?;
    Ok(Loaded { program, typed })
}

impl Loaded {
    /// Compiles only the code reachable from `entry` and prepares a runner with `perms`.
    pub fn runner(&self, entry: &Entry, perms: Permissions) -> Runner {
        Runner::new(qal_wasm::compile(&self.typed, entry)).expect("checked programs compile to valid wasm").with_permissions(perms)
    }
}

/// Splits leading options off `args`: `--allow <spec>` (repeatable) and an optional `--` separator.
pub fn take_options(args: &[String]) -> Result<(Permissions, Vec<String>), String> {
    let mut spec = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--allow" => {
                let value = args.get(i + 1).ok_or("`--allow` needs a value, e.g. `--allow io,fs:./data`")?;
                spec.push(value.clone());
                i += 2;
            }
            "--" => {
                i += 1;
                break;
            }
            _ => break,
        }
    }
    Ok((Permissions::parse(&spec.join(","))?, args[i..].to_vec()))
}

/// Converts CLI strings to values according to the signature of fn `name` (Int, Float, Bool, Unit, Str).
pub fn parse_args(typed: &TProgram, name: &str, raw: &[String]) -> Result<Vec<Value>, RunError> {
    let bad = |message: String| RunError::new("bad_args", name, message);
    let f = typed.fns.iter().find(|f| f.def.name == name).ok_or_else(|| RunError::new("unknown_function", name, format!("no function `{name}`")))?;
    if f.def.params.len() != raw.len() {
        return Err(bad(format!("`{name}` takes {} argument(s), {} given", f.def.params.len(), raw.len())));
    }
    f.def
        .params
        .iter()
        .zip(raw)
        .map(|(p, s)| match &p.ty {
            Type::Int => s.parse::<i64>().map(Value::Int).map_err(|_| bad(format!("`{}` must be Int, got `{s}`", p.name))),
            Type::Float => s.parse::<f64>().map(Value::Float).map_err(|_| bad(format!("`{}` must be Float, got `{s}`", p.name))),
            Type::Bool => match s.as_str() {
                "true" => Ok(Value::Bool(true)),
                "false" => Ok(Value::Bool(false)),
                _ => Err(bad(format!("`{}` must be Bool (true/false), got `{s}`", p.name))),
            },
            Type::Unit => match s.as_str() {
                "()" => Ok(Value::Unit),
                _ => Err(bad(format!("`{}` must be Unit `()`, got `{s}`", p.name))),
            },
            Type::Str => Ok(Value::Str(s.clone())),
            t => Err(bad(format!("`{}` has type {t}, which cannot be passed on the command line", p.name))),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn options_are_taken_from_the_front() {
        let (p, rest) = take_options(&strings(&["--allow", "io", "--allow", "time", "a", "--allow"])).unwrap();
        assert!(p.io && p.time && !p.rand);
        assert_eq!(rest, strings(&["a", "--allow"]));
        let (p, rest) = take_options(&strings(&["--", "--allow", "x"])).unwrap();
        assert_eq!((p, rest), (Permissions::default(), strings(&["--allow", "x"])));
        assert!(take_options(&strings(&["--allow"])).is_err());
        assert!(take_options(&strings(&["--allow", "net"])).unwrap_err().contains("unknown effect"));
    }

    #[test]
    fn parses_cli_arguments_by_signature() {
        let l = load("fn f(a: Int, b: Float, c: Str, d: Bool) -> Int = a\nfn g(xs: List[Int]) -> Int = 0").unwrap();
        let v = parse_args(&l.typed, "f", &strings(&["-3", "2.5", "hi", "true"])).unwrap();
        assert_eq!(v, vec![Value::Int(-3), Value::Float(2.5), Value::Str("hi".into()), Value::Bool(true)]);
        assert_eq!(parse_args(&l.typed, "f", &strings(&["x", "1", "", "true"])).unwrap_err().kind, "bad_args");
        assert!(parse_args(&l.typed, "g", &strings(&["[]"])).unwrap_err().message.contains("cannot be passed"));
        assert_eq!(parse_args(&l.typed, "h", &[]).unwrap_err().kind, "unknown_function");
    }
}
