//! `qal` command line. Every output is JSON on stdout, except `fmt` which prints canonical text.
//! Options may come before or after the file or name they apply to.
//!
//! Files (`<file>` ends in `.qal` or contains a `/`):
//!   qal check <file>                                 -> {"ok":true} | {"diagnostics":[...]}
//!   qal fmt <file>                                   -> canonical text
//! The project's store (`.qal/store.db` in the current directory or a parent):
//!   qal init                                         -> {"ok":true,"store":"<path>"}
//!   qal define <file> [--agent <name>]               -> {"defined":[{name,kind,hash,status,old_hash?}],"proofs":[...]?} | {"diagnostics":[...]}
//!                                                       (a function with a refuted contract is not stored: E0801)
//!   qal view <name|#hash>...                         -> {"views":[{name,kind,hash,signature,effects,text,deps}]}
//!   qal query [--name <prefix>] [--type <pattern>] [--uses <name|#hash>]
//!                                                    -> {"results":[{name,kind,hash,signature,effects}]}
//!   qal rename <old> <new> [--agent <name>]          -> {"renamed":{name,kind,hash}}
//!   qal history <name>                               -> {"history":[{action,name,from_name?,to_name?,old_hash?,new_hash,agent,at}]}
//!   qal build <name>                                 -> {"built":{name,hash,wasm,manifest,effects,bytes}} (paths relative to the project)
//! Running code from a file or from the store (`<target>` is a file or a name):
//!   qal call <file> [--allow <spec>] <fn> [args..]   -> {"ok":<value>,"stdout":".."} | {"error":{...},"stdout":".."} | {"diagnostics":[...]}
//!   qal call <fn> [--allow <spec>] [args..]          (store)
//!   qal test <file> | qal test [<name>...] [--seed <n>]
//!                                                    -> {"tests":[...],"properties":[...]?,"passed":n,"failed":m}
//!                                                       (names: tests, or functions for their property test)
//!   qal prove <file> | qal prove [<name>...]         -> {"proofs":[{def,obligation,path,contract,status,counterexample?,reason?}],
//!                                                        "proven":n,"refuted":m,"unknown":k} (exit 1 if refuted)
//!   qal run <file> [--allow <spec>] [--] [args..]    -> {"exit_code":n,"stdout":".."} | {"error":{...},"stdout":".."} | {"diagnostics":[...]}
//!   qal run <name> [--allow <spec>] [--] [args..]    (store; `<name>` is any `fn(List[Str]) -> Int`)
//! For agents:
//!   qal mcp [--root <dir>]                           MCP server on stdin/stdout (JSON-RPC 2.0); the store is <dir>/.qal/store.db
//!
//! `<spec>` grants effects: `io`, `time`, `rand`, `fs:<dir>` (comma-separated, `--allow` repeatable).
//! `--agent` names who made a change in the history (default `cli`).
//! Exit codes: 0 success; for `run` the program's exit code (1 if outside 0..=255);
//! 1 diagnostics / runtime error / failed tests; 2 usage or I/O error (also: no or unreadable store).

use qal_cli::ops::{self, CallArgs, Ctx, Reply, Source};
use qal_cli::{is_file, permissions, split_args, Split};
use qal_store::Query;
use std::io::Write;
use std::process::ExitCode;

const USAGE: &str = "usage: qal <check|fmt> <file>\n       \
                     qal init | define <file> [--agent <name>] | view <name>... | rename <old> <new> [--agent <name>] | history <name> | build <name>\n       \
                     qal query [--name <prefix>] [--type <pattern>] [--uses <name>]\n       \
                     qal call <file> [--allow <spec>] <fn> [args...] | call <fn> [--allow <spec>] [args...]\n       \
                     qal test [--allow <spec>] [--seed <n>] [<file> | <name>...] | prove [<file> | <name>...]\n       \
                     qal run <file|name> [--allow <spec>] [--] [args...]\n       \
                     qal mcp [--root <dir>]";

fn usage() -> ExitCode {
    bad_usage(USAGE)
}

fn bad_usage(message: &str) -> ExitCode {
    let _ = writeln!(std::io::stderr(), "{message}");
    ExitCode::from(2)
}

/// Prints the reply as one JSON line. A closed or failing stdout is not an error worth a panic:
/// the exit code still tells the result.
fn emit(reply: Reply) -> ExitCode {
    let line = serde_json::to_string(&reply.out).unwrap_or_else(|e| format!("{{\"diagnostics\":[{{\"code\":\"E0601\",\"kind\":\"store_error\",\"def\":\"\",\"path\":\"\",\"message\":\"cannot serialize the output: {e}\",\"hints\":[]}}]}}"));
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{line}").and_then(|_| out.flush());
    ExitCode::from(reply.code)
}

fn main() -> ExitCode {
    // Arguments that are not valid UTF-8 are converted lossily instead of panicking.
    let args: Vec<String> = std::env::args_os().skip(1).map(|a| a.to_string_lossy().into_owned()).collect();
    let Some(cmd) = args.first() else {
        return usage();
    };
    let rest = &args[1..];
    match cmd.as_str() {
        "check" | "fmt" => file_command(cmd, rest),
        "init" | "define" | "view" | "query" | "rename" | "history" | "build" => store_command(cmd, rest),
        "call" | "test" | "run" => run_command(cmd, rest),
        "prove" => prove_command(rest),
        "mcp" => mcp(rest),
        _ => usage(),
    }
}

fn read(file: &str) -> Result<String, ExitCode> {
    std::fs::read_to_string(file).map_err(|e| bad_usage(&format!("cannot read {file}: {e}")))
}

fn split(args: &[String], options: &[&str], repeatable: &[&str], targets: &dyn Fn(&[String]) -> usize) -> Result<Split, ExitCode> {
    split_args(args, options, repeatable, targets).map_err(|e| bad_usage(&e))
}

fn all(_: &[String]) -> usize {
    usize::MAX
}

fn ctx() -> Result<Ctx, ExitCode> {
    Ctx::cli().map_err(|d| emit(Reply::diagnostics(d)))
}

fn file_command(cmd: &str, args: &[String]) -> ExitCode {
    let s = match split(args, &[], &[], &all) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let [file] = s.targets.as_slice() else {
        return usage();
    };
    let src = match read(file) {
        Ok(s) => s,
        Err(code) => return code,
    };
    if cmd == "check" {
        return emit(ops::check(&src));
    }
    match qal_text::parse_program(&src) {
        Ok(p) => {
            let mut out = std::io::stdout().lock();
            let _ = write!(out, "{}", qal_text::print_program(&p)).and_then(|_| out.flush());
            ExitCode::SUCCESS
        }
        Err(d) => emit(Reply::diagnostics(vec![d])),
    }
}

fn store_command(cmd: &str, args: &[String]) -> ExitCode {
    let options: &[&str] = match cmd {
        "define" | "rename" => &["agent"],
        "query" => &["name", "type", "uses"],
        _ => &[],
    };
    // Validate the arguments before touching the store.
    let s = match split(args, options, &[], &all) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let agent = s.one("agent").unwrap_or("cli");
    let targets = s.targets.as_slice();
    match (cmd, targets) {
        ("init", []) => match std::env::current_dir() {
            Ok(dir) => emit(ops::init(&dir)),
            Err(e) => bad_usage(&format!("no current directory: {e}")),
        },
        ("define", [file]) => {
            let src = match read(file) {
                Ok(s) => s,
                Err(code) => return code,
            };
            match ctx() {
                Ok(c) => emit(ops::define(&c, &src, agent)),
                Err(code) => code,
            }
        }
        ("view", names) if !names.is_empty() => with_ctx(|c| ops::view(c, names)),
        ("query", []) => {
            let q = Query { name: s.one("name").map(str::to_string), type_pattern: s.one("type").map(str::to_string), uses: s.one("uses").map(str::to_string) };
            with_ctx(|c| ops::query(c, &q))
        }
        ("rename", [old, new]) => with_ctx(|c| ops::rename(c, old, new, agent)),
        ("history", [name]) => with_ctx(|c| ops::history(c, name)),
        ("build", [name]) => with_ctx(|c| ops::build(c, name)),
        _ => usage(),
    }
}

fn with_ctx(f: impl FnOnce(&Ctx) -> Reply) -> ExitCode {
    match ctx() {
        Ok(c) => emit(f(&c)),
        Err(code) => code,
    }
}

fn run_command(cmd: &str, args: &[String]) -> ExitCode {
    // Targets: `call` needs a file and a function, or a stored function; `run` one file or name;
    // `test` one file or any number of names (and no program arguments).
    let targets = |t: &[String]| match cmd {
        "call" if t.first().is_some_and(|f| is_file(f)) => 2,
        "test" => usize::MAX,
        _ => 1,
    };
    let options: &[&str] = if cmd == "test" { &["allow", "seed"] } else { &["allow"] };
    let s = match split(args, options, &["allow"], &targets) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let perms = match permissions(&s) {
        Ok(p) => p,
        Err(e) => return bad_usage(&e),
    };
    let seed = match s.one("seed").map(str::parse::<u64>) {
        None => 0,
        Some(Ok(n)) => n,
        Some(Err(_)) => return bad_usage("--seed takes a number"),
    };
    let from_file = s.targets.first().is_some_and(|t| is_file(t));
    if cmd == "test" && (!s.rest.is_empty() || (from_file && s.targets.len() > 1) || s.targets.iter().skip(1).any(|t| is_file(t))) {
        return usage();
    }
    if cmd != "test" && s.targets.len() < targets(&s.targets) {
        return usage();
    }
    let text = if from_file {
        match read(&s.targets[0]) {
            Ok(src) => Some(src),
            Err(code) => return code,
        }
    } else {
        None
    };
    let source = match &text {
        Some(src) => Source::Text(src),
        None => Source::Store,
    };
    let c = match ctx() {
        Ok(c) => c,
        Err(code) => return code,
    };
    emit(match cmd {
        "test" => ops::test(&c, &source, perms, if from_file { &[] } else { &s.targets }, seed),
        "call" => {
            let name = if from_file { &s.targets[1] } else { &s.targets[0] };
            ops::call(&c, &source, perms, name, CallArgs::Strings(&s.rest))
        }
        _ => {
            let name = if from_file { "main" } else { s.targets[0].as_str() };
            ops::run(&c, &source, perms, name, &s.rest)
        }
    })
}

fn prove_command(args: &[String]) -> ExitCode {
    let s = match split(args, &[], &[], &all) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let from_file = s.targets.first().is_some_and(|t| is_file(t));
    if from_file && s.targets.len() > 1 || s.targets.iter().skip(1).any(|t| is_file(t)) {
        return usage();
    }
    let text = if from_file {
        match read(&s.targets[0]) {
            Ok(src) => Some(src),
            Err(code) => return code,
        }
    } else {
        None
    };
    let source = match &text {
        Some(src) => Source::Text(src),
        None => Source::Store,
    };
    with_ctx(|c| ops::prove(c, &source, if from_file { &[] } else { &s.targets }))
}

fn mcp(args: &[String]) -> ExitCode {
    let s = match split(args, &["root"], &[], &all) {
        Ok(s) if s.targets.is_empty() => s,
        Ok(_) => return usage(),
        Err(code) => return code,
    };
    let root = match s.one("root") {
        Some(r) => std::path::PathBuf::from(r),
        None => match std::env::current_dir() {
            Ok(d) => d,
            Err(e) => return bad_usage(&format!("no current directory: {e}")),
        },
    };
    match qal_cli::mcp::serve_stdio(&root) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => bad_usage(&format!("qal mcp: {e}")),
    }
}
