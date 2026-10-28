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
//! `--breaking` (std project only): a name of the released std (the package embedded in this qal) may change its
//! signature (define), be deleted (delete) or renamed (rename); without it that is E0244 std_signature_change. Not offered
//! over MCP. The released std is the package embedded in this qal: until qal is rebuilt with the new package, every
//! later define of a name changed that way needs `--breaking` again.
//! Opening a store of an older schema upgrades it (any command; schema 5 since Phase 3b), after which older qal
//! binaries refuse it.
//! Every write (define, rename, delete, import) first imports the embedded std package if the project lacks this
//! version (`std_imported` in its result; if that fails, the
//! write still goes ahead with the previous std and `std_warnings` says why); reads never change the codebase.
//!   qal define <file> [--ast] [--agent <name>] [--breaking]  -> {"defined":[{name,kind,hash,status,old_hash?}],"proofs":[...]?} | {"diagnostics":[...]}
//!                                                       (a function with a refuted contract is not stored: E0801)
//!   qal view <name|#hash>...                         -> {"views":[{name,kind,hash,signature,effects,text,deps}]}
//!   qal query [--name <prefix>] [--type <pattern>] [--uses <name|#hash>] [--outline]
//!                                                    -> {"results":[{name,kind,hash,signature,effects}]}
//!   qal recall [--code <E...>] [--message <text>] [--name <prefix>] [--type <pattern>]
//!                                                    -> {"recall":{corrections:[{code,path,message,shape,before,after,count,...}],definitions:[...]}}
//!   qal corrections import <file.json>               -> {"corrections_imported":n} (the seed corpus of bench/corrections.py; stored with agent `corpus`)
//!   qal rename <old> <new> [--agent <name>] [--breaking] -> {"renamed":{name,kind,hash}}
//!   qal delete <name>... [--agent <name>] [--breaking] -> {"deleted":[{name,kind,hash}]} | {"diagnostics":[...]} (E0406 has_dependents:
//!                                                       names other stored definitions refer to; all or nothing)
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
//!   qal mcp [--root <dir>] [--session-file <file>]   MCP server on stdin/stdout (JSON-RPC 2.0); the store is <dir>/.qal/store.db
//!                                                       (a team store needs a session: --session-file or QAL_SESSION_FILE)
//! Teams (Phase 4a; the owner, on the command line; keys in $QAL_KEY_DIR, else ~/.config/qal/keys, never in the project):
//!   qal team init [--owner <id>]                     -> {"team":{owner,key_id,record,root,key_dir},"note":..} (genesis record)
//!   qal team add <id> --role <architect|implementer|verifier|reviewer>   -> {"member":{id,role,key_id,record}}
//!   qal team rekey <id> | qal team revoke <id>       -> {"member":{...}} | {"revoked":{...}}
//!   qal team session <id> [--ttl 24h] --out <file>   -> {"session":{id,member,expires_at,file}} (the token only in <file>, 0600)
//!   qal team export-trust --out <file>               -> {"trust":{file,owner,members}} (public keys only)
//!   qal team status | qal team stats [--json]        -> {"status":{...}} | {"stats":{...}} (never tokens)
//!   qal team sync-std                                -> {"std":{hash,status}} (signed import of the embedded std)
//!   qal verify [--trust <team.json>] [--from <seq>] [--head <hash>]
//!                                                    -> {"ok":true,"records":n,"members":m,"trust":"file"|"store","head":..,"signatures_from"?}
//!                                                       | {"ok":false,"error":{code,kind,seq,message},"checked":n} (exit 1)
//! In a team store the command line does not write code (define, rename, delete, import: E0903); agents do, over MCP.
//!
//! `<spec>` grants effects: `io`, `time`, `rand`, `fs:<dir>`, `net:<host>[:port]`, `db`, `db:<name>`, `db:mem`
//! (comma-separated, `--allow` repeatable; `db` is <project>/.qal/db/main.sqlite).
//! `--agent` names who made a change in the history (default `cli`).
//! Exit codes: 0 success; for `run` the program's exit code (1 if outside 0..=255);
//! 1 diagnostics / runtime error / failed tests; 2 usage or I/O error (also: no or unreadable store).

use qal_cli::ast;

use qal_cli::ops::{self, CallArgs, Ctx, Reply, Source};
use qal_cli::{is_file, permissions, split_args, Split};
use qal_store::{Query, RecallQuery};
use std::io::Write;
use std::process::ExitCode;
use std::sync::Arc;

const USAGE: &str = "usage: qal <check|fmt> <file>\n       \
                     qal init [--std] | define <file> [--ast] [--agent <name>] [--breaking] | view <name>... | rename <old> <new> [--agent <name>] [--breaking] | delete <name>... [--agent <name>] [--breaking] | history <name> | build <name>\n       \
                     qal export <prefix> --out <file> [--name <pkg>] | import <file|url> | install <file|url>\n       \
                     qal query [--name <prefix>] [--type <pattern>] [--uses <name>] [--outline]\n       \
                     qal recall [--code <E...>] [--message <text>] [--name <prefix>] [--type <pattern>]\n       \
                     qal corrections import <file.json>\n       \
                     qal call <file> [--allow <spec>] <fn> [args...] | call <fn> [--allow <spec>] [args...]\n       \
                     qal test [--allow <spec>] [--seed <n>] [<file> | <name>...] | prove [<file> | <name>...]\n       \
                     qal run <file|name> [--allow <spec>] [--] [args...]\n       \
                     qal serve [<handler>] [--listen <addr>] [--allow <spec>] [--init <fn>] [--workers <n>] [... see `qal serve --help`]\n       \
                     qal request [<handler>] [--allow <spec>] [--init <fn>] <requests.json>\n       \
                     qal mcp [--root <dir>] [--session-file <file>] | lsp\n       \
                     qal team <init [--owner <id>] | add <id> --role <role> | rekey <id> | revoke <id> | session <id> [--ttl 24h] --out <file> | export-trust --out <file> | status | stats | sync-std>\n       \
                     qal verify [--trust <team.json>] [--from <seq>] [--head <record hash>]";

fn usage() -> ExitCode {
    bad_usage(USAGE)
}

/// `--help`: the same text, but on stdout and with success (the usage error is for a bad call).
fn usage_ok() -> ExitCode {
    let _ = writeln!(std::io::stdout(), "{USAGE}");
    ExitCode::SUCCESS
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

fn extract_compiler_arg(args: &[String]) -> Result<(Option<String>, Vec<String>), String> {
    let mut compiler = None;
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--compiler" {
            if i + 1 >= args.len() {
                return Err("missing argument to --compiler".into());
            }
            compiler = Some(args[i + 1].clone());
            i += 2;
        } else if let Some(c) = args[i].strip_prefix("--compiler=") {
            compiler = Some(c.to_string());
            i += 1;
        } else {
            out.push(args[i].clone());
            i += 1;
        }
    }
    Ok((compiler, out))
}

fn main() -> ExitCode {
    // Arguments that are not valid UTF-8 are converted lossily instead of panicking.
    let raw_args: Vec<String> = std::env::args_os().skip(1).map(|a| a.to_string_lossy().into_owned()).collect();
    let (compiler_opt, args) = match extract_compiler_arg(&raw_args) {
        Ok(res) => res,
        Err(e) => return bad_usage(&e),
    };
    let Some(cmd) = args.first() else {
        return usage();
    };
    if cmd == "--help" || cmd == "-h" || cmd == "help" {
        return usage_ok();
    }
    if cmd == "--version" || cmd == "-v" || cmd == "version" {
        let _ = writeln!(std::io::stdout(), "qal {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    // The seed compiler is instantiated only when the QAL backend can run: always without the
    // reference compiler, otherwise only when `--compiler` or QAL_COMPILER asks for qal or both
    // (instantiating it costs seconds, which every other command must not pay).
    let wants_qal = |c: &str| c == "qal" || c == "both";
    let needs_seed = !cfg!(feature = "reference")
        || compiler_opt.as_deref().is_some_and(wants_qal)
        || std::env::var("QAL_COMPILER").is_ok_and(|c| wants_qal(&c));
    let fast_cmd = match cmd.as_str() {
        "mcp" | "lsp" | "verify" | "help" | "--help" | "-h" | "version" | "--version" | "-v" => true,
        // `qal team sync-std` imports the standard library, which checks it; the other team
        // commands never compile.
        "team" => args.get(1).map(String::as_str) != Some("sync-std"),
        _ => false,
    };
    if needs_seed && !fast_cmd && !qal_compiler::has_qal_driver() {
        let _ = qal_run::selfhost::init_seed_backend(qal_run::selfhost::default_jobs());
    }
    #[cfg(not(feature = "reference"))]
    if let Some(ref c) = compiler_opt {
        if let Err(e) = qal_compiler::set_backend(c) {
            return bad_usage(&e);
        }
    } else if std::env::var("QAL_COMPILER").is_err() {
        let _ = qal_compiler::set_backend("qal");
    }
    #[cfg(feature = "reference")]
    if let Some(ref c) = compiler_opt {
        if let Err(e) = qal_compiler::set_backend(c) {
            return bad_usage(&e);
        }
    }
    let rest = &args[1..];
    match cmd.as_str() {
        "check" | "fmt" => file_command(cmd, rest),
        "ast" => ast_command(rest),
        "init" | "define" | "view" | "query" | "recall" | "rename" | "delete" | "history" | "build" | "export" | "import" | "install" => store_command(cmd, rest),
        "call" | "test" | "run" => run_command(cmd, rest),
        "prove" => prove_command(rest),
        "corrections" => corrections(rest),
        "mcp" => mcp(rest),
        "lsp" => lsp(rest),
        "team" => team(rest),
        "verify" => verify(rest),
        "serve" => serve(rest),
        "request" => request(rest),
        "selfhost" => selfhost(rest),
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

fn ast_command(args: &[String]) -> ExitCode {
    let Some(file) = args.first() else { return bad_usage("usage: qal ast <file.json>"); };
    let text = match read(file) { Ok(t) => t, Err(e) => return e };
    let v: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => return bad_usage(&format!("invalid JSON: {e}")),
    };
    match ast::lower_main(&v) {
        Ok(src) => { let _ = write!(std::io::stdout(), "{src}"); ExitCode::SUCCESS }
        Err(e) => bad_usage(&format!("invalid AST: {e}")),
    }
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
    match qal_compiler::parse_program(&src) {
        Ok(p) => {
            let mut out = std::io::stdout().lock();
            let _ = write!(out, "{}", qal_compiler::print_program(&p)).and_then(|_| out.flush());
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
    // `--std` and `--breaking` are flags (the option parser only knows options with values).
    if cmd == "init" {
        return match args {
            [] => init(false),
            [flag] if flag == "--std" => init(true),
            _ => bad_usage("usage: qal init [--std]"),
        };
    }
    let breaking = matches!(cmd, "define" | "delete" | "rename") && args.iter().any(|a| a == "--breaking");
    let ast_flag = cmd == "define" && args.iter().any(|a| a == "--ast");
    let outline = cmd == "query" && args.iter().any(|a| a == "--outline");
    let without: Vec<String> = args.iter().filter(|a| !(breaking && *a == "--breaking") && !(outline && *a == "--outline") && !(ast_flag && *a == "--ast")).cloned().collect();
    let args = without.as_slice();
    let options: &[&str] = match cmd {
        "define" | "rename" | "delete" => &["agent"],
        "query" => &["name", "type", "uses"],
        "recall" => &["code", "message", "name", "type"],
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
        ("define", [file]) => {
            let src = if ast_flag {
                let text = match read(file) { Ok(s) => s, Err(code) => return code };
                let v: serde_json::Value = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => return bad_usage(&format!("invalid JSON AST: {e}")),
                };
                match ast::lower_main(&v) {
                    Ok(s) => s,
                    Err(e) => return bad_usage(&format!("invalid AST: {e}")),
                }
            } else {
                match read(file) { Ok(s) => s, Err(code) => return code }
            };
            match ctx() {
                Ok(c) => emit(ops::define(&c, &src, agent, breaking)),
                Err(code) => code,
            }
        }
        ("view", names) if !names.is_empty() => with_ctx(|c| ops::view(c, names)),
        ("query", []) => {
            let q = Query { name: s.one("name").map(str::to_string), type_pattern: s.one("type").map(str::to_string), uses: s.one("uses").map(str::to_string), outline };
            with_ctx(|c| ops::query(c, &q))
        }
        ("recall", []) => {
            let q = RecallQuery { code: s.one("code").map(str::to_string), message: s.one("message").map(str::to_string), name: s.one("name").map(str::to_string), type_pattern: s.one("type").map(str::to_string) };
            with_ctx(|c| ops::recall(c, &q))
        }
        ("rename", [old, new]) => with_ctx(|c| ops::rename(c, old, new, agent, breaking)),
        ("history", [name]) => with_ctx(|c| ops::history(c, name)),
        ("delete", names) if !names.is_empty() => with_ctx(|c| ops::delete(c, names, agent, breaking)),
        ("build", [name]) => with_ctx(|c| ops::build(c, name)),
        ("export", [prefix]) => match s.one("out") {
            Some(out) => with_ctx(|c| ops::export(c, prefix, std::path::Path::new(out), s.one("name"))),
            None => bad_usage("`qal export` needs `--out <file>`"),
        },
        ("import" | "install", [file_or_url]) => {
            let text = if file_or_url.starts_with("http://") || file_or_url.starts_with("https://") {
                match ureq::get(file_or_url).call() {
                    Ok(resp) => match resp.into_string() {
                        Ok(t) => t,
                        Err(e) => return bad_usage(&format!("failed to read response from {file_or_url}: {e}")),
                    },
                    Err(e) => return bad_usage(&format!("cannot download package from {file_or_url}: {e}")),
                }
            } else {
                match read(file_or_url) {
                    Ok(t) => t,
                    Err(code) => return code,
                }
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
    let replay = args.iter().any(|a| a == "--replay");
    let filtered: Vec<String> = args.iter().filter(|a| *a != "--replay").cloned().collect();
    let args = filtered.as_slice();
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
    let mut c = match ctx() {
        Ok(c) => c,
        Err(code) => return code,
    };
    c.replay = replay;
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
    let s = match split(args, &["root", "session-file"], &[], &all) {
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
    let session = s.one("session-file").map(std::path::PathBuf::from).or_else(|| std::env::var_os("QAL_SESSION_FILE").filter(|v| !v.is_empty()).map(std::path::PathBuf::from));
    match qal_cli::mcp::serve_stdio(&root, session) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => bad_usage(&format!("qal mcp: {e}")),
    }
}

fn lsp(_args: &[String]) -> ExitCode {
    let mut server = qal_cli::lsp::LspServer::new();
    match server.run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => bad_usage(&format!("qal lsp: {e}")),
    }
}

const TEAM_USAGE: &str = "usage: qal team init [--owner <id>] | add <id> --role <architect|implementer|verifier|reviewer> | rekey <id> | revoke <id>\n       \
                          qal team session <id> [--ttl 24h] --out <file> | export-trust --out <file> | status | stats [--json] | sync-std";

fn corrections(args: &[String]) -> ExitCode {
    match args {
        [sub, file] if sub == "import" => match read(file) {
            Ok(text) => with_ctx(|c| ops::import_corrections(c, &text)),
            Err(code) => code,
        },
        _ => bad_usage("usage: qal corrections import <file.json>"),
    }
}

fn team(args: &[String]) -> ExitCode {
    let Some(sub) = args.first() else {
        return bad_usage(TEAM_USAGE);
    };
    let rest: Vec<String> = args[1..].iter().filter(|a| !(sub == "stats" && *a == "--json")).cloned().collect();
    let options: &[&str] = match sub.as_str() {
        "init" => &["owner"],
        "add" => &["role"],
        "session" => &["ttl", "out"],
        "export-trust" => &["out"],
        _ => &[],
    };
    let s = match split(&rest, options, &[], &all) {
        Ok(s) => s,
        Err(_) => return bad_usage(TEAM_USAGE),
    };
    let c = match ctx() {
        Ok(c) => c,
        Err(code) => return code,
    };
    let out = s.one("out").map(std::path::Path::new);
    emit(match (sub.as_str(), s.targets.as_slice()) {
        ("init", []) => qal_cli::team::init(&c, s.one("owner").unwrap_or("owner")),
        ("add", [id]) => match s.one("role") {
            Some(role) => qal_cli::team::add(&c, id, role),
            None => return bad_usage("`qal team add` needs `--role <architect|implementer|verifier|reviewer>`"),
        },
        ("rekey", [id]) => qal_cli::team::rekey(&c, id),
        ("revoke", [id]) => qal_cli::team::revoke(&c, id),
        ("session", [id]) => match out {
            Some(out) => qal_cli::team::session(&c, id, s.one("ttl"), out),
            None => return bad_usage("`qal team session` needs `--out <file>` (the token is written only there)"),
        },
        ("export-trust", []) => match out {
            Some(out) => qal_cli::team::export_trust(&c, out),
            None => return bad_usage("`qal team export-trust` needs `--out <file>`"),
        },
        ("status", []) => qal_cli::team::status(&c),
        ("stats", []) => qal_cli::team::stats(&c),
        ("sync-std", []) => qal_cli::team::sync_std(&c),
        _ => return bad_usage(TEAM_USAGE),
    })
}

#[cfg(feature = "reference")]
const SELFHOST_USAGE: &str = "usage: qal selfhost bootstrap [--src selfhost/qalc.qal] [--out <seed.wasm>] [--jobs 4]";

/// `qal selfhost bootstrap`: Stage 1..3 and the fixpoint check; `--out` writes the Stage 3 seed.
fn selfhost(args: &[String]) -> ExitCode {
    #[cfg(not(feature = "reference"))]
    {
        let _ = args;
        bad_usage("selfhost bootstrap requires the reference Rust compiler (rebuild with --features reference)")
    }
    #[cfg(feature = "reference")]
    {
        let Some((sub, rest)) = args.split_first() else {
            return bad_usage(SELFHOST_USAGE);
        };
        if sub != "bootstrap" {
            return bad_usage(SELFHOST_USAGE);
        }
        let s = match split(rest, &["src", "out", "jobs"], &[], &all) {
            Ok(s) if s.targets.is_empty() => s,
            _ => return bad_usage(SELFHOST_USAGE),
        };
        let jobs = match s.one("jobs").map(str::parse::<usize>) {
            None => qal_run::selfhost::default_jobs(),
            Some(Ok(n)) if (1..=64).contains(&n) => n,
            Some(_) => return bad_usage("--jobs takes a number from 1 to 64"),
        };
        let src = match read(s.one("src").unwrap_or("selfhost/qalc.qal")) {
            Ok(t) => t,
            Err(code) => return code,
        };
        let (report, stage3) = match qal_cli::bootstrap::bootstrap(&src, jobs) {
            Ok(r) => r,
            Err(d) => return emit(Reply::diagnostics(vec![d])),
        };
        let fixpoint = report["ok"] == serde_json::Value::Bool(true);
        if let (true, Some(out)) = (fixpoint, s.one("out")) {
            if let Err(e) = std::fs::write(out, &stage3.wasm) {
                return bad_usage(&format!("cannot write {out}: {e}"));
            }
            if let Some(sha) = report["sha256"].as_str() {
                let _ = std::fs::write(format!("{out}.sha256"), format!("{sha}\n"));
            }
        }
        let line = report.to_string();
        let mut o = std::io::stdout().lock();
        let _ = writeln!(o, "{line}").and_then(|_| o.flush());
        if fixpoint { ExitCode::SUCCESS } else { ExitCode::from(1) }
    }
}

fn verify(args: &[String]) -> ExitCode {
    let s = match split(args, &["trust", "from", "head"], &[], &all) {
        Ok(s) if s.targets.is_empty() => s,
        _ => return bad_usage("usage: qal verify [--trust <team.json>] [--from <seq>] [--head <record hash>]"),
    };
    let from = match s.one("from").map(str::parse::<i64>) {
        None => 1,
        Some(Ok(n)) if n >= 1 => n,
        Some(_) => return bad_usage("--from takes a record number (1 or more)"),
    };
    let trust = match s.one("trust").map(read) {
        None => None,
        Some(Ok(t)) => Some(t),
        Some(Err(code)) => return code,
    };
    with_ctx(|c| qal_cli::team::verify(c, trust.as_deref(), from, s.one("head")))
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
    let mut listening = serde_json::json!({"event": "listening", "addr": addr, "entry": handler, "hash": hash, "workers": srv.workers, "pooling": service.pooled()});
    // Stubs the handler (or init) can reach: they answer with todo_reached (Phase 4 §11).
    let roots: Vec<usize> = [Some(handler), s.one("init")].into_iter().flatten().filter_map(|n| loaded.typed.fn_index(n)).collect();
    let stubs = loaded.typed.reachable_stubs(&roots);
    if !stubs.is_empty() {
        listening["warning"] = serde_json::json!(format!("the service reaches {} stub(s) (todo()): requests that reach them fail with todo_reached (E0540)", stubs.len()));
        listening["stubs"] = serde_json::json!(stubs);
    }
    log.line(&listening);
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
