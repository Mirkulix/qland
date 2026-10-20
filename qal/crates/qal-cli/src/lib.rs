//! The QAL pipeline: text or store -> parse -> check -> compile (per entry point) -> runner, plus
//! the operations shared by the command line (`main.rs`) and the MCP server ([`mcp`]).

pub mod lsp;
pub mod mcp;
pub mod ops;
pub mod secret_file;
pub mod team;

/// The standard library package (spec Phase 3 §4.3), embedded in the binary: `qal init` and every
/// write to a project store import it when the project does not have this version yet
/// (`meta.std_package`); the std project's own store (`qal init --std`) never does.
pub const STD_PACKAGE: &str = include_str!("../../../std/std.qalpkg");

use qal_core::ast::{Program, Type};
use qal_core::diag::Diagnostic;
use qal_core::tast::TProgram;
use qal_run::{Permissions, RunError, Runner, Value};
use qal_store::Store;
use qal_compiler::Entry;
use std::collections::HashMap;

pub struct Loaded {
    pub program: Program,
    pub typed: TProgram,
}

/// Parses and checks `src`. Any diagnostic means nothing can be compiled.
pub fn load(src: &str) -> Result<Loaded, Vec<Diagnostic>> {
    let program = qal_compiler::parse_program(src).map_err(|d| vec![d])?;
    let typed = qal_compiler::check(&program)?;
    Ok(Loaded { program, typed })
}

/// Checks a stored program. Stored definitions that take a prelude name (written before Phase 3)
/// keep working; see `qal_compiler::check_stored`.
pub fn load_program(program: Program) -> Result<Loaded, Vec<Diagnostic>> {
    let typed = qal_compiler::check_stored(&program, &|_| false)?;
    Ok(Loaded { program, typed })
}

/// A command's target is a source file if it ends in `.qal` or contains a path separator
/// (`./foo`, `dir/foo`), otherwise a name in the store (names never contain `/`; the only names
/// with `.` are `std.<module>.<name>`, which are names even if they end in `.qal`).
pub fn is_file(target: &str) -> bool {
    !qal_compiler::lexer::is_std_name(target) && (target.ends_with(".qal") || target.contains('/') || target.contains('\\'))
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

/// True if `name` is a function `fn(List[Str]) -> Int` (with any effects), i.e. runnable by `qal run`.
pub fn is_entry_point(typed: &TProgram, name: &str) -> bool {
    typed.fns.iter().any(|f| f.def.name == name && f.def.ret == Type::Int && f.def.params.len() == 1 && f.def.params[0].ty == Type::list(Type::Str))
}

impl Loaded {
    /// Compiles only the code reachable from `entry` and prepares a runner with `perms`.
    /// The caller has checked that `entry` names an existing function.
    pub fn runner(&self, entry: &Entry, perms: Permissions) -> Result<Runner, RunError> {
        let def = match entry {
            Entry::Fn(name) => name.as_str(),
            Entry::Tests | Entry::All | Entry::Fns(_) => "",
        };
        let compile_error = |m: String| {
            let mut e = RunError::new("compile_error", def, format!("the program cannot be compiled to WebAssembly: {m}"));
            e.hints.push("this is a limitation of the compiler, not an error in the program; try splitting very large functions or list literals".into());
            e
        };
        let module = qal_compiler::compile(&self.typed, entry).map_err(compile_error)?;
        Ok(Runner::new(module).map_err(compile_error)?.with_permissions(perms))
    }
}

/// Command-line arguments split into options, targets and program arguments.
#[derive(Debug, Default, PartialEq)]
pub struct Split {
    /// Values of each `--option` (in order; repeatable options have several).
    pub options: HashMap<String, Vec<String>>,
    /// Positional arguments that name what to work on (files, names).
    pub targets: Vec<String>,
    /// Everything after the targets: arguments for the program (`run`, `call`).
    pub rest: Vec<String>,
}

impl Split {
    /// The single value of `--name`, if given.
    pub fn one(&self, name: &str) -> Option<&str> {
        self.options.get(name).and_then(|v| v.first()).map(String::as_str)
    }
}

/// Splits `args` into `--option value` pairs (only the ones in `options`; `repeatable` ones may
/// occur several times), targets and program arguments. Options are accepted anywhere before the
/// program arguments start: those start after `--`, or once `targets(&targets_so_far)` targets
/// have been collected, at the first argument that is not a known option (so `run f.qal --verbose`
/// passes `--verbose` to the program). An unknown option before that is an error.
pub fn split_args(args: &[String], options: &[&str], repeatable: &[&str], targets: &dyn Fn(&[String]) -> usize) -> Result<Split, String> {
    let mut out = Split::default();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == "--" {
            out.rest = args[i + 1..].to_vec();
            return Ok(out);
        }
        let complete = out.targets.len() >= targets(&out.targets);
        if let Some(name) = a.strip_prefix("--").filter(|n| options.contains(n) || !complete) {
            if !options.contains(&name) {
                return Err(format!("unknown option `{a}`"));
            }
            let value = args.get(i + 1).ok_or_else(|| format!("`{a}` needs a value"))?;
            let values = out.options.entry(name.to_string()).or_default();
            if !values.is_empty() && !repeatable.contains(&name) {
                return Err(format!("`{a}` given twice"));
            }
            values.push(value.clone());
            i += 2;
            continue;
        }
        if !complete {
            out.targets.push(a.clone());
            i += 1;
            continue;
        }
        out.rest = args[i..].to_vec();
        return Ok(out);
    }
    Ok(out)
}

/// Parses `--allow` values (each a comma-separated list) into permissions.
pub fn permissions(split: &Split) -> Result<Permissions, String> {
    let spec = split.options.get("allow").map(|v| v.join(",")).unwrap_or_default();
    Permissions::parse(&spec)
}

/// The error for calling fn `name` that is not in `typed.fns`: a generic function cannot be called
/// directly (only its instances are compiled); the hint shows a monomorphic wrapper.
pub fn no_function(typed: &TProgram, name: &str) -> RunError {
    let mut e = RunError::new("unknown_function", name, format!("no function `{name}`"));
    if let Some(g) = typed.generics.iter().find(|f| f.def.name == name) {
        let tps = &g.def.tparams;
        let ints = vec![Type::Int; tps.len()];
        let params: Vec<String> = g.def.params.iter().zip(&g.locals).map(|(p, t)| format!("{}: {}", p.name, t.instantiate(tps, &ints))).collect();
        let args: Vec<&str> = g.def.params.iter().map(|p| p.name.as_str()).collect();
        let effects: String = g.def.effects.iter().map(|e| format!(" !{e}")).collect();
        e.message = format!("`{name}` is generic ([{}]) and cannot be called directly", tps.join(", "));
        e.hints.push(format!(
            "call it from a monomorphic wrapper with concrete types, e.g. `fn call_{name}({}) -> {}{effects} = {name}({})`, then call `call_{name}`",
            params.join(", "),
            g.body.ty.instantiate(tps, &ints),
            args.join(", ")
        ));
    }
    e
}

/// Converts CLI strings to values according to the signature of fn `name` (Int, Float, Bool, Unit, Str).
pub fn parse_args(typed: &TProgram, name: &str, raw: &[String]) -> Result<Vec<Value>, RunError> {
    let bad = |message: String| RunError::new("bad_args", name, message);
    let f = typed.fns.iter().find(|f| f.def.name == name).ok_or_else(|| no_function(typed, name))?;
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

/// Converts JSON arguments to values according to the signature of fn `name`, in the JSON form
/// results are reported in (`qal_run::from_json`): numbers, strings, Bool, `null` for Unit, arrays
/// for List and Bytes, `[[key, value], ...]` for Map, objects for records, `"None"` / `{"Some": [3]}`
/// for variants.
pub fn json_args(typed: &TProgram, name: &str, raw: &[serde_json::Value]) -> Result<Vec<Value>, RunError> {
    let f = typed.fns.iter().find(|f| f.def.name == name).ok_or_else(|| no_function(typed, name))?;
    if f.def.params.len() != raw.len() {
        return Err(RunError::new("bad_args", name, format!("`{name}` takes {} argument(s), {} given", f.def.params.len(), raw.len())));
    }
    f.def
        .params
        .iter()
        .zip(raw)
        .map(|(p, v)| qal_run::from_json(typed, &p.ty, v).map_err(|m| RunError::new("bad_args", name, format!("`{}`: {m}", p.name))))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    fn run_split(args: &[&str]) -> Result<Split, String> {
        // `run`-style: one target, then program arguments.
        split_args(&strings(args), &["allow"], &["allow"], &|_| 1)
    }

    #[test]
    fn options_are_accepted_before_and_after_the_target() {
        let s = run_split(&["--allow", "io", "main", "--allow", "time", "a", "--allow"]).unwrap();
        assert_eq!(s.targets, strings(&["main"]));
        assert_eq!(s.options["allow"], strings(&["io", "time"]));
        assert_eq!(s.rest, strings(&["a", "--allow"]));
        let p = permissions(&s).unwrap();
        assert!(p.io && p.time && !p.rand);
        let s = run_split(&["main", "--", "--allow", "x"]).unwrap();
        assert_eq!((s.targets, s.rest), (strings(&["main"]), strings(&["--allow", "x"])));
        assert!(s.options.is_empty());
        assert!(run_split(&["--allow"]).unwrap_err().contains("needs a value"));
        assert!(run_split(&["--bogus", "1", "main"]).unwrap_err().contains("unknown option"));
        // After the target, an unknown `--x` is the first program argument.
        let s = run_split(&["main", "--allow", "io", "--verbose", "--allow", "x"]).unwrap();
        assert_eq!((s.options["allow"].clone(), s.rest), (strings(&["io"]), strings(&["--verbose", "--allow", "x"])));
        assert!(permissions(&run_split(&["main", "--allow", "web"]).unwrap()).unwrap_err().contains("unknown effect"));
        assert!(permissions(&run_split(&["main", "--allow", "net"]).unwrap()).unwrap_err().contains("needs a host"));
    }

    #[test]
    fn targets_can_depend_on_the_first_one() {
        // `call`: a file needs a function name as second target, a stored name does not.
        let call = |args: &[&str]| split_args(&strings(args), &["allow"], &["allow"], &|t: &[String]| if t.first().is_some_and(|f| is_file(f)) { 2 } else { 1 }).unwrap();
        let s = call(&["--allow", "io", "x.qal", "--allow", "time", "f", "1", "2"]);
        assert_eq!((s.targets, s.rest), (strings(&["x.qal", "f"]), strings(&["1", "2"])));
        let s = call(&["f", "--allow", "io", "1"]);
        assert_eq!((s.targets, s.rest), (strings(&["f"]), strings(&["1"])));
    }

    #[test]
    fn non_repeatable_options_once() {
        let all = |_: &[String]| usize::MAX;
        let s = split_args(&strings(&["a", "--agent", "x", "b"]), &["agent"], &[], &all).unwrap();
        assert_eq!((s.one("agent"), &s.targets), (Some("x"), &strings(&["a", "b"])));
        assert!(split_args(&strings(&["--agent", "a", "--agent", "b"]), &["agent"], &[], &all).unwrap_err().contains("twice"));
    }

    #[test]
    fn the_embedded_standard_library_is_a_valid_package() {
        // Checksum and `# def` lines verified by `parse`; only `std.` names, text iff definitions.
        let p = qal_store::package::parse(STD_PACKAGE).unwrap();
        assert_eq!(p.name, "std");
        assert!(p.defs.iter().all(|d| qal_compiler::lexer::is_std_name(&d.name)));
        assert_eq!(p.text.trim().is_empty(), p.defs.is_empty());
        assert_eq!(p.std, None, "the std package does not refer to another std");
        let program = qal_compiler::parse_program(&p.text).unwrap();
        assert_eq!(program.defs.len(), p.defs.len(), "one definition per `# def` line");
    }

    #[test]
    fn files_end_in_qal_or_contain_a_path_separator() {
        assert!(is_file("x/y.qal") && is_file("y.qal") && is_file("./foo") && is_file("dir/foo") && is_file(".\\foo"));
        assert!(!is_file("main") && !is_file("parse_age"));
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

    #[test]
    fn converts_json_arguments_by_signature() {
        use serde_json::json;
        let l = load("type P = { x: Int }\nfn f(a: Int, b: Float, c: Str, d: Bool, e: Unit, xs: List[List[Int]]) -> Int = a\nfn g(p: P) -> Int = p.x").unwrap();
        let v = json_args(&l.typed, "f", &[json!(-3), json!(2), json!("hi"), json!(true), json!(null), json!([[1], []])]).unwrap();
        let list = Value::List(vec![Value::List(vec![Value::Int(1)]), Value::List(vec![])]);
        assert_eq!(v, vec![Value::Int(-3), Value::Float(2.0), Value::Str("hi".into()), Value::Bool(true), Value::Unit, list]);
        let e = json_args(&l.typed, "f", &[json!(1.5), json!(2), json!("hi"), json!(true), json!(null), json!([])]).unwrap_err();
        assert_eq!((e.kind, e.message.as_str()), ("bad_args", "`a`: expected Int, got 1.5"));
        assert_eq!(json_args(&l.typed, "g", &[json!({"x": 1})]).unwrap(), vec![Value::Record("P".into(), vec![("x".into(), Value::Int(1))])]);
        assert_eq!(json_args(&l.typed, "g", &[json!({"y": 1})]).unwrap_err().message, "`p`: P has no field `y`");
        assert!(json_args(&l.typed, "f", &[]).unwrap_err().message.contains("takes 6 argument(s), 0 given"));
        assert_eq!(json_args(&l.typed, "h", &[]).unwrap_err().kind, "unknown_function");
    }

    #[test]
    fn calling_a_generic_function_explains_how() {
        // Review finding M3: `qal_call` on a stored generic function said it does not exist.
        let l = load("type Pair[A, B] = { fst: A, snd: B }\nfn swap[A, B](p: Pair[A, B]) -> Pair[B, A] = Pair { fst: p.snd, snd: p.fst }").unwrap();
        for e in [json_args(&l.typed, "swap", &[]).unwrap_err(), parse_args(&l.typed, "swap", &[]).unwrap_err()] {
            assert_eq!((e.kind, e.message.as_str()), ("unknown_function", "`swap` is generic ([A, B]) and cannot be called directly"));
            assert_eq!(
                e.hints,
                ["call it from a monomorphic wrapper with concrete types, e.g. `fn call_swap(p: Pair[Int, Int]) -> Pair[Int, Int] = swap(p)`, then call `call_swap`"]
            );
        }
        assert_eq!(json_args(&l.typed, "nosuch", &[]).unwrap_err().message, "no function `nosuch`");
    }
}
