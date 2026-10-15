//! `qal` command line. Every output is JSON on stdout, except `fmt` which prints canonical text.
//!
//! Files (`<file>` ends in `.qal`):
//!   qal check <file>                                 -> {"ok":true} | {"diagnostics":[...]}
//!   qal fmt <file>                                   -> canonical text
//! The project's store (`.qal/store.db` in the current directory or a parent):
//!   qal init                                         -> {"ok":true,"store":"<path>"}
//!   qal define <file> [--agent <name>]               -> {"defined":[{name,kind,hash,status,old_hash?}]} | {"diagnostics":[...]}
//!   qal view <name|#hash>...                         -> {"views":[{name,kind,hash,signature,effects,text,deps}]}
//!   qal query [--name <prefix>] [--type <pattern>] [--uses <name|#hash>]
//!                                                    -> {"results":[{name,kind,hash,signature,effects}]}
//!   qal rename <old> <new> [--agent <name>]          -> {"renamed":{name,kind,hash}}
//!   qal history <name>                               -> {"history":[{action,name,from_name?,to_name?,old_hash?,new_hash,agent,at}]}
//! Running code from a file or from the store (`<target>` is a file or a name):
//!   qal call <file> [--allow <spec>] <fn> [args..]   -> {"ok":<value>,"stdout":".."} | {"error":{...},"stdout":".."} | {"diagnostics":[...]}
//!   qal call <fn> [--allow <spec>] [args..]          (store)
//!   qal test <file> | qal test [<name>...]           -> {"tests":[...],"passed":n,"failed":m}
//!   qal run <file> [--allow <spec>] [--] [args..]    -> {"exit_code":n,"stdout":".."} | {"error":{...},"stdout":".."} | {"diagnostics":[...]}
//!   qal run <name> [--allow <spec>] [--] [args..]    (store; `<name>` is any `fn(List[Str]) -> Int`)
//!
//! `<spec>` grants effects: `io`, `time`, `rand`, `fs:<dir>` (comma-separated, `--allow` repeatable).
//! `--agent` names who made a change in the history (default `cli`).
//! Exit codes: 0 success; for `run` the program's exit code (1 if outside 0..=255);
//! 1 diagnostics / runtime error / failed tests; 2 usage or I/O error (also: no or unreadable store).

use qal_cli::{is_entry_point, is_file, load, load_store, open_store, parse_args, take_flags, take_options, Loaded};
use qal_core::diag::Diagnostic;
use qal_run::{RunError, TestResult, Value};
use qal_store::{Defined, Entry as QueryEntry, HistoryEntry, Query, Renamed, Store, View};
use qal_wasm::Entry;
use serde::Serialize;
use std::process::ExitCode;

const USAGE: &str = "usage: qal <check|fmt> <file>\n       \
                     qal init | define <file> [--agent <name>] | view <name>... | rename <old> <new> [--agent <name>] | history <name>\n       \
                     qal query [--name <prefix>] [--type <pattern>] [--uses <name>]\n       \
                     qal call <file> [--allow <spec>] <fn> [args...] | call <fn> [--allow <spec>] [args...]\n       \
                     qal test [<file> | <name>...]\n       \
                     qal run <file|name> [--allow <spec>] [--] [args...]";

/// One JSON output line. Serialized directly (not via `serde_json::Value`) so record fields and
/// contract values keep their declaration order.
#[derive(Serialize, Default)]
struct Out<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    ok: Option<&'a Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    defined: Option<&'a [Defined]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    views: Option<&'a [View]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    results: Option<&'a [QueryEntry]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    renamed: Option<&'a Renamed>,
    #[serde(skip_serializing_if = "Option::is_none")]
    history: Option<&'a [HistoryEntry]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exit_code: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a RunError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stdout: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diagnostics: Option<&'a [Diagnostic]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tests: Option<&'a [TestResult]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    passed: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    failed: Option<usize>,
}

fn usage() -> ExitCode {
    eprintln!("{USAGE}");
    ExitCode::from(2)
}

fn bad_usage(message: &str) -> ExitCode {
    eprintln!("{message}");
    ExitCode::from(2)
}

fn emit(out: Out, code: u8) -> ExitCode {
    println!("{}", serde_json::to_string(&out).expect("output serializes"));
    ExitCode::from(code)
}

/// Diagnostics; exit 2 if the store itself failed (missing, unreadable, corrupt), else 1.
fn diagnostics(d: &[Diagnostic]) -> ExitCode {
    let code = if !d.is_empty() && d.iter().all(|d| d.kind == "store_error") { 2 } else { 1 };
    emit(Out { diagnostics: Some(d), ..Out::default() }, code)
}

fn main() -> ExitCode {
    // Arguments that are not valid UTF-8 are converted lossily instead of panicking.
    let args: Vec<String> = std::env::args_os().skip(1).map(|a| a.to_string_lossy().into_owned()).collect();
    let Some(cmd) = args.first() else {
        return usage();
    };
    match cmd.as_str() {
        "check" | "fmt" => file_command(cmd, &args[1..]),
        "init" | "define" | "view" | "query" | "rename" | "history" => store_command(cmd, &args[1..]),
        "call" | "test" | "run" => run_command(cmd, &args[1..]),
        _ => usage(),
    }
}

fn read(file: &str) -> Result<String, ExitCode> {
    std::fs::read_to_string(file).map_err(|e| bad_usage(&format!("cannot read {file}: {e}")))
}

fn file_command(cmd: &str, args: &[String]) -> ExitCode {
    let [file] = args else {
        return usage();
    };
    let src = match read(file) {
        Ok(s) => s,
        Err(code) => return code,
    };
    if cmd == "fmt" {
        return match qal_text::parse_program(&src) {
            Ok(p) => {
                print!("{}", qal_text::print_program(&p));
                ExitCode::SUCCESS
            }
            Err(d) => diagnostics(&[d]),
        };
    }
    match load(&src) {
        Ok(_) => emit(Out { ok: Some(&Value::Bool(true)), ..Out::default() }, 0),
        Err(d) => diagnostics(&d),
    }
}

fn store_command(cmd: &str, args: &[String]) -> ExitCode {
    let flags: &[&str] = match cmd {
        "define" | "rename" => &["agent"],
        "query" => &["name", "type", "uses"],
        _ => &[],
    };
    let (flags, rest) = match take_flags(args, flags) {
        Ok(x) => x,
        Err(e) => return bad_usage(&e),
    };
    let agent = flags.get("agent").map(String::as_str).unwrap_or("cli");
    if cmd == "init" {
        if !rest.is_empty() {
            return usage();
        }
        let cwd = match std::env::current_dir() {
            Ok(d) => d,
            Err(e) => return bad_usage(&format!("no current directory: {e}")),
        };
        return match Store::init(&cwd) {
            Ok(s) => emit(Out { ok: Some(&Value::Bool(true)), store: Some(&s.path().display().to_string()), ..Out::default() }, 0),
            Err(d) => diagnostics(&d),
        };
    }
    // Validate the arguments before touching the store.
    let src = match (cmd, rest.as_slice()) {
        ("define", [file]) => match read(file) {
            Ok(s) => Some(s),
            Err(code) => return code,
        },
        ("view", names) if !names.is_empty() => None,
        ("query", []) | ("rename", [_, _]) | ("history", [_]) => None,
        _ => return usage(),
    };
    let mut store = match open_store() {
        Ok(s) => s,
        Err(d) => return diagnostics(&d),
    };
    match cmd {
        "define" => match store.define(src.as_deref().unwrap_or_default(), agent) {
            Ok(d) => emit(Out { defined: Some(&d), ..Out::default() }, 0),
            Err(d) => diagnostics(&d),
        },
        "view" => match store.view(&rest) {
            Ok(v) => emit(Out { views: Some(&v), ..Out::default() }, 0),
            Err(d) => diagnostics(&d),
        },
        "query" => {
            let q = Query { name: flags.get("name").cloned(), type_pattern: flags.get("type").cloned(), uses: flags.get("uses").cloned() };
            match store.query(&q) {
                Ok(r) => emit(Out { results: Some(&r), ..Out::default() }, 0),
                Err(d) => diagnostics(&d),
            }
        }
        "rename" => match store.rename(&rest[0], &rest[1], agent) {
            Ok(r) => emit(Out { renamed: Some(&r), ..Out::default() }, 0),
            Err(d) => diagnostics(&d),
        },
        _ => match store.history(&rest[0]) {
            Ok(h) => emit(Out { history: Some(&h), ..Out::default() }, 0),
            Err(d) => diagnostics(&d),
        },
    }
}

fn run_command(cmd: &str, args: &[String]) -> ExitCode {
    // `<target>` is a `.qal` file, or (store) the function to call/run or the first test name.
    let from_file = args.first().is_some_and(|t| is_file(t));
    let loaded = if from_file {
        match read(&args[0]) {
            Ok(src) => load(&src),
            Err(code) => return code,
        }
    } else {
        if cmd != "test" && args.is_empty() {
            return usage();
        }
        load_store()
    };
    let opts_from = match (cmd, from_file) {
        ("test", true) => 1,
        ("test", false) => 0,
        (_, _) => 1,
    };
    let (perms, rest) = match take_options(&args[opts_from.min(args.len())..]) {
        Ok(x) => x,
        Err(e) => return bad_usage(&e),
    };
    let loaded = match loaded {
        Ok(l) => l,
        Err(d) => return diagnostics(&d),
    };
    match cmd {
        "test" => {
            if from_file && !rest.is_empty() {
                return usage();
            }
            test(&loaded, perms, &rest)
        }
        "call" => {
            let (name, raw) = if from_file {
                let Some(name) = rest.first() else {
                    return usage();
                };
                (name.clone(), &rest[1..])
            } else {
                (args[0].clone(), &rest[..])
            };
            call(&loaded, perms, &name, raw)
        }
        _ => {
            let name = if from_file { "main".to_string() } else { args[0].clone() };
            run(&loaded, perms, &name, &rest)
        }
    }
}

fn test(loaded: &Loaded, perms: qal_run::Permissions, names: &[String]) -> ExitCode {
    let unknown: Vec<Diagnostic> = names.iter().filter(|n| !loaded.typed.tests.iter().any(|t| &t.name == *n)).map(|n| Diagnostic::unknown_name(n, "", n)).collect();
    if !unknown.is_empty() {
        return diagnostics(&unknown);
    }
    let tests: Vec<TestResult> = loaded.runner(&Entry::Tests, perms).run_tests().into_iter().filter(|t| names.is_empty() || names.contains(&t.name)).collect();
    let passed = tests.iter().filter(|t| t.passed).count();
    let failed = tests.len() - passed;
    emit(Out { tests: Some(&tests), passed: Some(passed), failed: Some(failed), ..Out::default() }, u8::from(failed > 0))
}

fn call(loaded: &Loaded, perms: qal_run::Permissions, name: &str, raw: &[String]) -> ExitCode {
    let vals = match parse_args(&loaded.typed, name, raw) {
        Ok(v) => v,
        Err(e) => return emit(Out { error: Some(&e), ..Out::default() }, 1),
    };
    let out = loaded.runner(&Entry::Fn(name.to_string()), perms).invoke(name, &vals);
    match &out.result {
        Ok(v) => emit(Out { ok: Some(v), stdout: Some(&out.stdout), ..Out::default() }, 0),
        Err(e) => emit(Out { error: Some(e), stdout: Some(&out.stdout), ..Out::default() }, 1),
    }
}

fn run(loaded: &Loaded, perms: qal_run::Permissions, name: &str, args: &[String]) -> ExitCode {
    if !is_entry_point(&loaded.typed, name) {
        let e = if loaded.typed.fn_index(name).is_some() {
            let mut e = RunError::new("bad_args", name, format!("`{name}` is not an entry point"));
            e.hints.push("`qal run` needs a function `fn(List[Str]) -> Int`; use `qal call` for others".into());
            e
        } else {
            RunError::new("unknown_function", name, format!("no entry point `{name}`"))
        };
        return emit(Out { error: Some(&e), ..Out::default() }, 1);
    }
    let list = Value::List(args.iter().map(|a| Value::Str(a.clone())).collect());
    let out = loaded.runner(&Entry::Fn(name.to_string()), perms).invoke(name, &[list]);
    match &out.result {
        Ok(Value::Int(code)) => emit(Out { exit_code: Some(*code), stdout: Some(&out.stdout), ..Out::default() }, u8::try_from(*code).unwrap_or(1)),
        Ok(other) => {
            let e = RunError::new("trap", name, format!("entry point returned {other:?} instead of Int"));
            emit(Out { error: Some(&e), stdout: Some(&out.stdout), ..Out::default() }, 1)
        }
        Err(e) => emit(Out { error: Some(e), stdout: Some(&out.stdout), ..Out::default() }, 1),
    }
}
