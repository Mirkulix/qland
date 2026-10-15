//! `qal` command line. Every output is JSON on stdout, except `fmt` which prints canonical text.
//!
//!   qal check <file>                                 -> {"ok":true} | {"diagnostics":[...]}
//!   qal fmt <file>                                   -> canonical text
//!   qal call <file> [--allow <spec>] <fn> [args..]   -> {"ok":<value>,"stdout":".."} | {"error":{...},"stdout":".."} | {"diagnostics":[...]}
//!   qal test <file>                                  -> {"tests":[...],"passed":n,"failed":m}
//!   qal run <file> [--allow <spec>] [--] [args..]    -> {"exit_code":n,"stdout":".."} | {"error":{...},"stdout":".."} | {"diagnostics":[...]}
//!
//! `<spec>` grants effects: `io`, `time`, `rand`, `fs:<dir>` (comma-separated, `--allow` repeatable).
//! Exit codes: 0 success; for `run` the program's exit code (1 if outside 0..=255);
//! 1 diagnostics / runtime error / failed tests; 2 usage or I/O error.

use qal_cli::{load, parse_args, take_options};
use qal_core::diag::Diagnostic;
use qal_run::{RunError, TestResult, Value};
use qal_wasm::Entry;
use serde::Serialize;
use std::process::ExitCode;

const USAGE: &str = "usage: qal <check|fmt|test> <file>\n       qal call <file> [--allow <spec>] <fn> [args...]\n       qal run <file> [--allow <spec>] [--] [args...]";

/// One JSON output line. Serialized directly (not via `serde_json::Value`) so record fields and
/// contract values keep their declaration order.
#[derive(Serialize, Default)]
struct Out<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    ok: Option<&'a Value>,
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

fn emit(out: Out, code: u8) -> ExitCode {
    println!("{}", serde_json::to_string(&out).expect("output serializes"));
    ExitCode::from(code)
}

fn main() -> ExitCode {
    // Arguments that are not valid UTF-8 are converted lossily instead of panicking.
    let args: Vec<String> = std::env::args_os().skip(1).map(|a| a.to_string_lossy().into_owned()).collect();
    let (Some(cmd), Some(file)) = (args.first(), args.get(1)) else {
        return usage();
    };
    if !["check", "fmt", "call", "test", "run"].contains(&cmd.as_str()) {
        return usage();
    }
    let src = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read {file}: {e}");
            return ExitCode::from(2);
        }
    };
    if cmd == "fmt" {
        return match qal_text::parse_program(&src) {
            Ok(p) => {
                print!("{}", qal_text::print_program(&p));
                ExitCode::SUCCESS
            }
            Err(d) => emit(Out { diagnostics: Some(&[d]), ..Out::default() }, 1),
        };
    }
    let (perms, rest) = match take_options(&args[2..]) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let loaded = match load(&src) {
        Ok(l) => l,
        Err(d) => return emit(Out { diagnostics: Some(&d), ..Out::default() }, 1),
    };
    match cmd.as_str() {
        "check" => emit(Out { ok: Some(&Value::Bool(true)), ..Out::default() }, 0),
        "test" => {
            let tests = loaded.runner(&Entry::Tests, perms).run_tests();
            let passed = tests.iter().filter(|t| t.passed).count();
            let failed = tests.len() - passed;
            emit(Out { tests: Some(&tests), passed: Some(passed), failed: Some(failed), ..Out::default() }, u8::from(failed > 0))
        }
        "call" => {
            let Some(name) = rest.first() else {
                return usage();
            };
            let vals = match parse_args(&loaded.typed, name, &rest[1..]) {
                Ok(v) => v,
                Err(e) => return emit(Out { error: Some(&e), ..Out::default() }, 1),
            };
            let out = loaded.runner(&Entry::Fn(name.clone()), perms).invoke(name, &vals);
            match &out.result {
                Ok(v) => emit(Out { ok: Some(v), stdout: Some(&out.stdout), ..Out::default() }, 0),
                Err(e) => emit(Out { error: Some(e), stdout: Some(&out.stdout), ..Out::default() }, 1),
            }
        }
        "run" => {
            if loaded.typed.fn_index("main").is_none() {
                let e = RunError::new("unknown_function", "main", "no entry point `main`");
                return emit(Out { error: Some(&e), ..Out::default() }, 1);
            }
            let out = loaded.runner(&Entry::Fn("main".into()), perms).run_main(&rest);
            match &out.result {
                Ok(Value::Int(code)) => {
                    emit(Out { exit_code: Some(*code), stdout: Some(&out.stdout), ..Out::default() }, u8::try_from(*code).unwrap_or(1))
                }
                Ok(other) => unreachable!("main is checked to return Int, got {other:?}"),
                Err(e) => emit(Out { error: Some(e), stdout: Some(&out.stdout), ..Out::default() }, 1),
            }
        }
        _ => usage(),
    }
}
