//! `qal` command line. Every output is JSON on stdout, except `fmt` which prints canonical text.
//!
//!   qal check <file>               -> {"ok":true} | {"diagnostics":[...]}
//!   qal fmt <file>                 -> canonical text
//!   qal call <file> <fn> [args..]  -> {"ok":<value>} | {"error":{...}} | {"diagnostics":[...]}
//!   qal test <file>                -> {"tests":[...],"passed":n,"failed":m}
//!
//! Exit codes: 0 success, 1 diagnostics / runtime error / failed tests, 2 usage or I/O error.

use qal_cli::{load, parse_args};
use serde_json::json;
use std::process::ExitCode;

const USAGE: &str = "usage: qal <check|fmt|call|test> <file> [fn] [args...]";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(cmd), Some(file)) = (args.first(), args.get(1)) else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let src = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read {file}: {e}");
            return ExitCode::from(2);
        }
    };
    let emit = |v: serde_json::Value, ok: bool| {
        println!("{v}");
        if ok { ExitCode::SUCCESS } else { ExitCode::from(1) }
    };
    match cmd.as_str() {
        "fmt" => match qal_text::parse_program(&src) {
            Ok(p) => {
                print!("{}", qal_text::print_program(&p));
                ExitCode::SUCCESS
            }
            Err(d) => emit(json!({ "diagnostics": [d] }), false),
        },
        "check" => match load(&src) {
            Ok(_) => emit(json!({ "ok": true }), true),
            Err(d) => emit(json!({ "diagnostics": d }), false),
        },
        "call" => {
            let Some(name) = args.get(2) else {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            };
            let loaded = match load(&src) {
                Ok(l) => l,
                Err(d) => return emit(json!({ "diagnostics": d }), false),
            };
            let result = parse_args(&loaded.program, name, &args[3..]).and_then(|vals| loaded.runner.call(name, &vals));
            match result {
                Ok(v) => emit(json!({ "ok": v }), true),
                Err(e) => emit(json!({ "error": e }), false),
            }
        }
        "test" => match load(&src) {
            Ok(l) => {
                let tests = l.runner.run_tests();
                let passed = tests.iter().filter(|t| t.passed).count();
                let failed = tests.len() - passed;
                emit(json!({ "tests": tests, "passed": passed, "failed": failed }), failed == 0)
            }
            Err(d) => emit(json!({ "diagnostics": d }), false),
        },
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}
