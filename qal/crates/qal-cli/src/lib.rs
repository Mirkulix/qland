//! The QAL pipeline: text or store -> parse -> check -> compile (per entry point) -> runner.

use qal_core::ast::{Program, Type};
use qal_core::diag::Diagnostic;
use qal_core::tast::TProgram;
use qal_run::{Permissions, RunError, Runner, Value};
use qal_store::Store;
use qal_wasm::Entry;
use std::collections::HashMap;

pub struct Loaded {
    pub program: Program,
    pub typed: TProgram,
}

/// Parses and checks `src`. Any diagnostic means nothing can be compiled.
pub fn load(src: &str) -> Result<Loaded, Vec<Diagnostic>> {
    let program = qal_text::parse_program(src).map_err(|d| vec![d])?;
    load_program(program)
}

/// Checks an already parsed (or stored) program.
pub fn load_program(program: Program) -> Result<Loaded, Vec<Diagnostic>> {
    let typed = qal_check::check(&program)?;
    Ok(Loaded { program, typed })
}

/// A command's target is a source file if it ends in `.qal`, otherwise a name in the store
/// (definition names never contain a dot).
pub fn is_file(target: &str) -> bool {
    target.ends_with(".qal")
}

/// The store of the project around the current directory.
pub fn open_store() -> Result<Store, Vec<Diagnostic>> {
    let cwd = std::env::current_dir().map_err(|e| vec![Diagnostic::store_error(format!("no current directory: {e}"))])?;
    Store::discover(&cwd)
}

/// Loads the whole codebase of the project's store.
pub fn load_store() -> Result<Loaded, Vec<Diagnostic>> {
    load_program(open_store()?.program()?)
}

/// Splits `--flag value` options (only those in `flags`, anywhere in `args`) from positional
/// arguments. Unknown `--options` are an error.
pub fn take_flags(args: &[String], flags: &[&str]) -> Result<(HashMap<String, String>, Vec<String>), String> {
    let mut found = HashMap::new();
    let mut rest = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if let Some(flag) = a.strip_prefix("--") {
            if !flags.contains(&flag) {
                return Err(format!("unknown option `{a}`"));
            }
            let value = args.get(i + 1).ok_or_else(|| format!("`{a}` needs a value"))?;
            if found.insert(flag.to_string(), value.clone()).is_some() {
                return Err(format!("`{a}` given twice"));
            }
            i += 2;
        } else {
            rest.push(a.clone());
            i += 1;
        }
    }
    Ok((found, rest))
}

/// True if `name` is a function `fn(List[Str]) -> Int` (with any effects), i.e. runnable by `qal run`.
pub fn is_entry_point(typed: &TProgram, name: &str) -> bool {
    typed.fns.iter().any(|f| f.def.name == name && f.def.ret == Type::Int && f.def.params.len() == 1 && f.def.params[0].ty == Type::list(Type::Str))
}

impl Loaded {
    /// Compiles only the code reachable from `entry` and prepares a runner with `perms`.
    pub fn runner(&self, entry: &Entry, perms: Permissions) -> Runner {
        // Task 4 turns both failures into a `compile_error` runtime error.
        Runner::new(qal_wasm::compile(&self.typed, entry).expect("checked programs compile")).expect("checked programs compile to valid wasm").with_permissions(perms)
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
    fn flags_are_taken_from_anywhere() {
        let (f, rest) = take_flags(&strings(&["a", "--agent", "x", "b"]), &["agent"]).unwrap();
        assert_eq!((f.get("agent").map(String::as_str), rest), (Some("x"), strings(&["a", "b"])));
        assert!(take_flags(&strings(&["--nope", "1"]), &["agent"]).unwrap_err().contains("unknown option"));
        assert!(take_flags(&strings(&["--agent"]), &["agent"]).unwrap_err().contains("needs a value"));
        assert!(take_flags(&strings(&["--agent", "a", "--agent", "b"]), &["agent"]).unwrap_err().contains("twice"));
        assert!(is_file("x/y.qal") && !is_file("main"));
    }

    #[test]
    fn entry_points_take_a_string_list_and_return_int() {
        let l = load("fn main(args: List[Str]) -> Int = 0\nfn app(a: List[Str]) -> Int !io = 1\nfn f(a: Int) -> Int = a").unwrap();
        assert!(is_entry_point(&l.typed, "main") && is_entry_point(&l.typed, "app"));
        assert!(!is_entry_point(&l.typed, "f") && !is_entry_point(&l.typed, "g"));
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
