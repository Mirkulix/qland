//! `qal` command line. Every output is JSON on stdout, except `fmt` which prints canonical text.
//! Options may come before or after the file or name they apply to.
//!
//! Files (`<file>` ends in `.qal` or contains a `/`):
//!   qal check <file>                                 -> {"ok":true} | {"diagnostics":[...]}
//!   qal fmt <file>                                   -> canonical text
//! The project's store (`.qal/store.db` in the current directory or a parent):
//!   qal init [--std]                                 -> {"ok":true,"store":"<path>","std_imported":"<hash>"?}
//!                                                       (imports the embedded std package; `--std`: the std project's
//!                                                       store, which defines only `std.` names and imports nothing later)
//! Every write (define, rename, import) first imports the embedded std package if the project lacks this
//! version (`std_imported` in its result); reads never write.
//!   qal define <file> [--agent <name>]               -> {"defined":[{name,kind,hash,status,old_hash?}],"proofs":[...]?} | {"diagnostics":[...]}
//!                                                       (a function with a refuted contract is not stored: E0801)
//!   qal view <name|#hash>...                         -> {"views":[{name,kind,hash,signature,effects,text,deps}]}
//!   qal query [--name <prefix>] [--type <pattern>] [--uses <name|#hash>]
//!                                                    -> {"results":[{name,kind,hash,signature,effects}]}
//!   qal rename <old> <new> [--agent <name>]          -> {"renamed":{name,kind,hash}}
//!   qal history <name>                               -> {"history":[{action,name,from_name?,to_name?,old_hash?,new_hash,agent,at}]}
//!   qal build <name>                                 -> {"built":{name,hash,wasm,manifest,effects,bytes}} (paths relative to the project)
//!   qal export <prefix> --out <file> [--name <pkg>]  -> {"exported":{package,hash,defs,file}} (names with the prefix and their dependencies)
//!   qal import <file>                                -> {"imported":{package,hash,defined}} | {"diagnostics":[...]} (E0243 package_mismatch)
//! Running code from a file or from the store (`<target>` is a file or a name):
//!   qal call <file> [--allow <spec>] <fn> [args..]   -> {"ok":<value>,"stdout":".."} | {"error":{...},"stdout":".."} | {"diagnostics":[...]}
//!   qal call <fn> [--allow <spec>] [args..]          (store)
//!   qal test <file> | qal test [<name>...] [--seed <n>]
//!                                                    -> {"tests":[...],"properties":[...]?,"passed":n,"failed":m}
//!                                                       (names: tests, or functions for their property test)
//!   qal prove <file> | qal prove [<name>...]         -> {"proofs":[{def,obligation,path,contract,status,counterexample?,reason?}],
//!                                                        "proven":n,"refuted":m,"unknown":k} (exit 1 if refuted)
//!                                                       (`proofs` and `warnings` are cut at 512 KiB each; `proofs_omitted`
//!                                                        and `warnings_omitted` count what was left out)
//!   qal run <file> [--allow <spec>] [--] [args..]    -> {"exit_code":n,"stdout":".."} | {"error":{...},"stdout":".."} | {"diagnostics":[...]}
//!   qal run <name> [--allow <spec>] [--] [args..]    (store; `<name>` is any `fn(List[Str]) -> Int`)
//! Services (a stored handler `fn(HttpRequest) -> HttpResponse`, default `handle`):
//!   qal serve [<handler>] [--listen 127.0.0.1:8080] [--allow <spec>] [--init <fn>] [--workers 4] [--queue 64]
//!             [--request-timeout <ms>] [--fuel <n>] [--memory <MiB>] [--max-body <KiB>] [--max-header-bytes <n>]
//!             [--max-response <KiB>] [--header-timeout <ms>] [--body-timeout <ms>] [--net-calls <n>]
//!             [--fs-write-budget <MiB>] [--db-max-rows <n>] [--db-max-mib <n>] [--debug-errors] [--log <file>]
//!                                                    HTTP/1.1 server; JSON log lines on stderr (or --log): listening, request, stopped
//!   qal request [<handler>] [--allow <spec>] [--init <fn>] <requests.json>
//!                                                    -> {"responses":[{status,headers,body,json?,db,stdout?,error?}]}
//! For agents:
//!   qal mcp [--root <dir>]                           MCP server on stdin/stdout (JSON-RPC 2.0); the store is <dir>/.qal/store.db
//!
//! `<spec>` grants effects: `io`, `time`, `rand`, `fs:<dir>`, `net:<host>[:port]`, `db`, `db:<name>`, `db:mem`
//! (comma-separated, `--allow` repeatable; `db` is <project>/.qal/db/main.sqlite).
//! `--agent` names who made a change in the history (default `cli`).
//! Exit codes: 0 success; for `run` the program's exit code (1 if outside 0..=255);
//! 1 diagnostics / runtime error / failed tests; 2 usage or I/O error (also: no or unreadable store).

use qal_cli::ops::{self, CallArgs, Ctx, Reply, Source};
use qal_cli::{is_file, permissions, split_args, Split};
use qal_store::Query;
use std::io::Write;
use std::process::ExitCode;
use std::sync::Arc;

const USAGE: &str = "usage: qal <check|fmt> <file>\n       \
                     qal init [--std] | define <file> [--agent <name>] | view <name>... | rename <old> <new> [--agent <name>] | history <name> | build <name>\n       \
                     qal export <prefix> --out <file> [--name <pkg>] | import <file>\n       \
                     qal query [--name <prefix>] [--type <pattern>] [--uses <name>]\n       \
                     qal call <file> [--allow <spec>] <fn> [args...] | call <fn> [--allow <spec>] [args...]\n       \
                     qal test [--allow <spec>] [--seed <n>] [<file> | <name>...] | prove [<file> | <name>...]\n       \
                     qal run <file|name> [--allow <spec>] [--] [args...]\n       \
                     qal serve [<handler>] [--listen <addr>] [--allow <spec>] [--init <fn>] [--workers <n>] [... see `qal serve --help`]\n       \
                     qal request [<handler>] [--allow <spec>] [--init <fn>] <requests.json>\n       \
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
        "init" | "define" | "view" | "query" | "rename" | "history" | "build" | "export" | "import" => store_command(cmd, rest),
        "call" | "test" | "run" => run_command(cmd, rest),
        "prove" => prove_command(rest),
        "mcp" => mcp(rest),
        "serve" => serve(rest),
        "request" => request(rest),
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

fn init(std: bool) -> ExitCode {
    match std::env::current_dir() {
        Ok(dir) => emit(ops::init(&dir, std)),
        Err(e) => bad_usage(&format!("no current directory: {e}")),
    }
}

fn store_command(cmd: &str, args: &[String]) -> ExitCode {
    // `--std` is a flag (the option parser only knows options with values).
    if cmd == "init" && matches!(args, [flag] if flag == "--std") {
        return init(true);
    }
    let options: &[&str] = match cmd {
        "define" | "rename" => &["agent"],
        "query" => &["name", "type", "uses"],
        "export" => &["out", "name"],
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
        ("init", []) => init(false),
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
        ("export", [prefix]) => match s.one("out") {
            Some(out) => with_ctx(|c| ops::export(c, prefix, std::path::Path::new(out), s.one("name"))),
            None => bad_usage("`qal export` needs `--out <file>`"),
        },
        ("import", [file]) => {
            let text = match read(file) {
                Ok(t) => t,
                Err(code) => return code,
            };
            with_ctx(|c| ops::import(c, &text))
        }
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

/// The options of `qal serve` with their defaults (spec Phase 3 §7.1).
const SERVE_OPTIONS: [&str; 18] = [
    "listen", "allow", "init", "workers", "queue", "request-timeout", "fuel", "memory", "max-body", "max-header-bytes", "max-response",
    "header-timeout", "body-timeout", "net-calls", "fs-write-budget", "db-max-rows", "db-max-mib", "log",
];

const SERVE_USAGE: &str = "usage: qal serve [<handler>] [--listen 127.0.0.1:8080] [--allow <spec>] [--init <fn>] [--workers 4 (1-64)] [--queue 64]\n       \
                           [--request-timeout 5000 (ms)] [--fuel 500000000] [--memory 64 (MiB, at most 256)] [--max-body 1024 (KiB)]\n       \
                           [--max-header-bytes 16384] [--max-response 8192 (KiB)] [--header-timeout 5000 (ms)] [--body-timeout 10000 (ms)]\n       \
                           [--net-calls 16] [--fs-write-budget 64 (MiB)] [--db-max-rows 10000] [--db-max-mib 1024] [--debug-errors] [--log <file>]";

/// A number option within `range` (default `default`).
fn number(s: &Split, name: &str, default: u64, range: std::ops::RangeInclusive<u64>) -> Result<u64, String> {
    match s.one(name) {
        None => Ok(default),
        Some(v) => v.parse::<u64>().ok().filter(|n| range.contains(n)).ok_or_else(|| format!("`--{name}` must be a number in {}..={}, got `{v}`", range.start(), range.end())),
    }
}

/// Writes a start failure (stderr and the log) and exits 1.
fn start_failed(log: &qal_serve::Log, error: &qal_run::RunError) -> ExitCode {
    log.line(&serde_json::json!({"event": "start_failed", "error": error}));
    ExitCode::from(1)
}

fn serve(args: &[String]) -> ExitCode {
    // `--debug-errors` is the only option without a value.
    let debug_errors = args.iter().any(|a| a == "--debug-errors");
    if args.iter().any(|a| a == "--help" || a == "-h") {
        return bad_usage(SERVE_USAGE);
    }
    let args: Vec<String> = args.iter().filter(|a| *a != "--debug-errors").cloned().collect();
    let s = match split(&args, &SERVE_OPTIONS, &["allow"], &all) {
        Ok(s) if s.targets.len() <= 1 && s.rest.is_empty() => s,
        Ok(_) => return bad_usage(SERVE_USAGE),
        Err(code) => return code,
    };
    let handler = s.targets.first().map_or("handle", String::as_str);
    let parsed = (|| -> Result<(qal_serve::ServiceConfig, qal_serve::ServerConfig), String> {
        let ms = |n: &str, d: u64| number(&s, n, d, 1..=3_600_000).map(std::time::Duration::from_millis);
        let workers = number(&s, "workers", 4, 1..=64)? as usize;
        let http = qal_serve::HttpLimits {
            max_header_bytes: number(&s, "max-header-bytes", 16 << 10, 1024..=1 << 20)? as usize,
            max_body: number(&s, "max-body", 1024, 0..=64 << 10)? as usize * 1024,
            header_timeout: ms("header-timeout", 5000)?,
            body_timeout: ms("body-timeout", 10_000)?,
            ..qal_serve::HttpLimits::default()
        };
        let budgets = qal_run::Limits {
            net_calls: number(&s, "net-calls", 16, 0..=10_000)? as usize,
            fs_write_bytes: number(&s, "fs-write-budget", 64, 0..=1 << 20)? as usize * (1 << 20),
            db_max_rows: number(&s, "db-max-rows", 10_000, 1..=10_000_000)? as usize,
            db_max_mib: number(&s, "db-max-mib", 1024, 1..=1 << 20)? as usize,
            ..qal_run::Limits::default()
        };
        let svc = qal_serve::ServiceConfig {
            entry: handler.to_string(),
            init: s.one("init").map(str::to_string),
            request_timeout: ms("request-timeout", 5000)?,
            fuel: number(&s, "fuel", 500_000_000, 1..=u64::MAX / 2)?,
            memory: number(&s, "memory", 64, 1..=256)? as usize * (1 << 20),
            max_response: number(&s, "max-response", 8192, 0..=256 << 10)? as usize * 1024,
            budgets,
            debug_errors,
            slots: workers + 2,
            ..qal_serve::ServiceConfig::default()
        };
        Ok((svc, qal_serve::ServerConfig { workers, queue: number(&s, "queue", 64, 1..=100_000)? as usize, http }))
    })();
    let (svc, srv) = match parsed {
        Ok(c) => c,
        Err(e) => return bad_usage(&e),
    };
    let perms = match permissions(&s) {
        Ok(p) if p.db == Some(qal_run::DbGrant::Memory) => return bad_usage("`db:mem` is a database per tool call; a server needs a file: `db` or `db:<name>`"),
        Ok(p) => p,
        Err(e) => return bad_usage(&e),
    };
    let log = Arc::new(match s.one("log") {
        Some(path) => match std::fs::OpenOptions::new().create(true).append(true).open(path) {
            Ok(f) => qal_serve::Log::new(Box::new(f)),
            Err(e) => return bad_usage(&format!("cannot open the log {path}: {e}")),
        },
        None => qal_serve::Log::stderr(),
    });
    let ctx = match ctx() {
        Ok(c) => c,
        Err(code) => return code,
    };
    let mut store = match ctx.open() {
        Ok(st) => st,
        Err(d) => return emit(Reply::diagnostics(d)),
    };
    // .qal/store.db -> the project, whose .qal/db holds the database files.
    let perms = match store.path().parent().and_then(std::path::Path::parent) {
        Some(project) if perms.db.is_some() => perms.with_db_dir(project.join(qal_store::STORE_DIR).join("db")),
        _ => perms,
    };
    let loaded = match store.program().and_then(qal_cli::load_program) {
        Ok(l) => l,
        Err(d) => return emit(Reply::diagnostics(d)),
    };
    let hash = store.query(&Query { name: Some(handler.to_string()), ..Query::default() }).ok().and_then(|rs| rs.into_iter().find(|e| e.name == handler)).map(|e| e.hash).unwrap_or_default();
    let service = match qal_serve::Service::new(&loaded.typed, perms, svc) {
        Ok(sv) => Arc::new(sv),
        Err(e) => return start_failed(&log, &e),
    };
    let mut workers = Vec::new();
    for _ in 0..srv.workers {
        match service.worker() {
            Ok(w) => workers.push(w),
            Err(e) => return start_failed(&log, &e),
        }
    }
    if let Some(first) = workers.first_mut() {
        if let Err(e) = service.init(first) {
            return start_failed(&log, &e);
        }
    }
    let listen = s.one("listen").unwrap_or("127.0.0.1:8080");
    let listener = match qal_serve::bind(listen) {
        Ok(l) => l,
        Err(e) => return start_failed(&log, &e),
    };
    let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
    if let Err(e) = qal_serve::on_shutdown_signals(&shutdown) {
        return start_failed(&log, &qal_run::RunError::new("listen_failed", handler, format!("cannot install the signal handlers: {e}")));
    }
    let addr = listener.local_addr().map(|a| a.to_string()).unwrap_or_default();
    log.line(&serde_json::json!({"event": "listening", "addr": addr, "entry": handler, "hash": hash, "workers": srv.workers, "pooling": service.pooled()}));
    let served = qal_serve::serve(listener, service, srv, log.clone(), shutdown, workers);
    log.line(&serde_json::json!({"event": "stopped", "served": served}));
    ExitCode::SUCCESS
}

fn request(args: &[String]) -> ExitCode {
    let s = match split(args, &["allow", "init"], &["allow"], &all) {
        Ok(s) if (1..=2).contains(&s.targets.len()) && s.rest.is_empty() => s,
        Ok(_) => return usage(),
        Err(code) => return code,
    };
    let (handler, file) = match s.targets.as_slice() {
        [file] => ("handle", file),
        [handler, file] => (handler.as_str(), file),
        _ => return usage(),
    };
    let perms = match permissions(&s) {
        Ok(p) => p,
        Err(e) => return bad_usage(&e),
    };
    let text = match read(file) {
        Ok(t) => t,
        Err(code) => return code,
    };
    let specs = match serde_json::from_str::<serde_json::Value>(&text).map_err(|e| format!("{file}: not JSON: {e}")).and_then(|v| {
        let list = match v {
            serde_json::Value::Array(xs) => xs,
            serde_json::Value::Object(mut m) => match m.remove("requests") {
                Some(serde_json::Value::Array(xs)) => xs,
                _ => return Err(format!("{file}: expected an array of requests or {{\"requests\": [...]}}")),
            },
            _ => return Err(format!("{file}: expected an array of requests or {{\"requests\": [...]}}")),
        };
        list.iter().enumerate().map(|(i, r)| ops::RequestSpec::from_json(r).map_err(|e| format!("{file}: request {i}: {e}"))).collect::<Result<Vec<_>, _>>()
    }) {
        Ok(specs) => specs,
        Err(e) => return bad_usage(&e),
    };
    with_ctx(|c| ops::request(c, &Source::Store, perms, handler, s.one("init"), &specs))
}
