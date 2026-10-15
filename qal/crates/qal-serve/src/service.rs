//! The service pipeline (spec Phase 3 §2): one request in, one response and one log entry out.
//!
//! [`Service::handle`] is everything between the HTTP parser and the HTTP writer: the request
//! becomes an `HttpRequest` value, the handler runs in a fresh instance of the prepared module
//! (inside one database transaction when it uses `db`), and its `HttpResponse` is validated and
//! completed (`content-length` and a default `content-type`, both not for 204/304; `x-request-id`). Traps and invalid
//! responses become 500/503 responses; nothing a handler does ends the process. `qal serve` and
//! `qal_request` both run exactly this pipeline.

use qal_core::ast::Type;
use qal_core::tast::TProgram;
use qal_run::{Db, ExecIn, Limits, Permissions, RunError, Runner, Value};
use qal_wasm::Entry;
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Limits and settings of a service (spec §2.5).
#[derive(Debug, Clone, PartialEq)]
pub struct ServiceConfig {
    /// Handler function, `fn(HttpRequest) -> HttpResponse` with any effects.
    pub entry: String,
    /// Optional init function, `fn() -> Result[Unit, Str]`, run once before serving.
    pub init: Option<String>,
    pub request_timeout: Duration,
    pub fuel: u64,
    /// Guest memory per request in bytes (also the host budget).
    pub memory: usize,
    /// Largest response body in bytes.
    pub max_response: usize,
    /// Printed output kept per request (the rest is counted).
    pub max_out: usize,
    pub budgets: Limits,
    /// Return the full runtime error in 500/503 bodies.
    pub debug_errors: bool,
    /// Instance slots of the pooling allocator (workers + 2).
    pub slots: usize,
}

impl Default for ServiceConfig {
    fn default() -> ServiceConfig {
        ServiceConfig {
            entry: "handle".into(),
            init: None,
            request_timeout: Duration::from_secs(5),
            fuel: 500_000_000,
            memory: 64 << 20,
            max_response: 8 << 20,
            max_out: 4 << 10,
            budgets: Limits::default(),
            debug_errors: false,
            slots: 6,
        }
    }
}

/// An HTTP request as the handler sees it (`HttpRequest`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub query: String,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

/// The response to send: status, complete headers (including `content-length` unless the status
/// has no body) and body.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl Response {
    /// A response made by the host (errors, rejections) with a JSON body.
    pub fn json(status: u16, body: String, id: u64) -> Response {
        let headers = vec![("content-type".into(), "application/json".into()), ("content-length".into(), body.len().to_string()), ("x-request-id".into(), id.to_string())];
        Response { status, headers, body }
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(n, _)| n.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

/// One line of the service log for a request (spec §2.6).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LogEntry {
    pub event: &'static str,
    pub id: u64,
    pub method: String,
    pub path: String,
    pub status: u16,
    pub ms: u64,
    pub fuel: u64,
    pub mem_bytes: usize,
    pub db: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub out: Option<String>,
    /// Bytes printed in total (more than `out` holds if it was cut).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub out_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RunError>,
}

/// What one request produced.
#[derive(Debug, Clone)]
pub struct Handled {
    pub response: Response,
    pub log: LogEntry,
}

/// Why a service cannot start.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StartError {
    pub error: RunError,
}

/// A worker's resources: its own database connection (opened lazily, reopened after a failure).
#[derive(Default)]
pub struct Worker {
    db: Option<Db>,
}

/// Effects a handler or init function may have.
const SERVICE_EFFECTS: [&str; 6] = ["io", "time", "rand", "db", "net", "fs"];

fn sig_text(params: &[Type], ret: &Type, effects: &[String]) -> String {
    let ps: Vec<String> = params.iter().map(|t| t.to_string()).collect();
    let effs: String = effects.iter().map(|e| format!(" !{e}")).collect();
    format!("fn({}) -> {ret}{effs}", ps.join(", "))
}

/// Checks that `name` is a function of the right shape before anything starts (`bad_entry`).
fn check_entry(typed: &TProgram, name: &str, params: &[Type], ret: &Type, what: &str) -> Result<Vec<String>, RunError> {
    let expected = sig_text(params, ret, &[]);
    let Some(f) = typed.fns.iter().find(|f| f.def.name == name) else {
        if typed.generics.iter().any(|g| g.def.name == name) {
            let mut e = RunError::new("bad_entry", name, format!("`{name}` is generic; the {what} must be a monomorphic function"));
            e.hints.push(format!("the {what} must be `{expected}` (any effects), without type parameters"));
            return Err(e);
        }
        let mut e = RunError::new("bad_entry", name, format!("there is no function `{name}` to use as the {what}"));
        e.hints.push(format!("the {what} must be `{expected}` (any effects)"));
        return Err(e);
    };
    let found: Vec<Type> = f.def.params.iter().map(|p| p.ty.clone()).collect();
    if found != params || f.def.ret != *ret {
        let mut e = RunError::new("bad_entry", name, format!("the {what} `{name}` must be `{expected}` (any effects), found `{}`", sig_text(&found, &f.def.ret, &f.def.effects)));
        e.hints.push(format!("define `fn {name}(...)` with exactly that signature, or choose another {what}"));
        return Err(e);
    }
    if let Some(bad) = f.def.effects.iter().find(|e| !SERVICE_EFFECTS.contains(&e.as_str())) {
        return Err(RunError::new("bad_entry", name, format!("the {what} `{name}` has the effect `{bad}`, which services do not support")));
    }
    Ok(f.def.effects.clone())
}

fn named(n: &str) -> Type {
    Type::Named(n.to_string())
}

fn headers_value(hs: &[(String, String)]) -> Value {
    Value::List(hs.iter().map(|(n, v)| Value::Record("HttpHeader".into(), vec![("name".into(), Value::Str(n.clone())), ("value".into(), Value::Str(v.clone()))])).collect())
}

/// The `HttpRequest` value of a request.
pub fn request_value(r: &Request) -> Value {
    Value::Record(
        "HttpRequest".into(),
        vec![
            ("method".into(), Value::Str(r.method.clone())),
            ("path".into(), Value::Str(r.path.clone())),
            ("query".into(), Value::Str(r.query.clone())),
            ("headers".into(), headers_value(&r.headers)),
            ("body".into(), Value::Str(r.body.clone())),
        ],
    )
}

/// Statuses whose responses carry no body (and no `content-length`).
pub fn bodiless(status: u16) -> bool {
    (100..200).contains(&status) || status == 204 || status == 304
}

fn is_token(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
}

/// A handler's `HttpResponse` checked against the rules of §2.6, or why it is invalid.
fn validate(v: &Value, max_body: usize) -> Result<Response, String> {
    let Value::Record(_, fields) = v else { return Err("the handler did not return an HttpResponse".into()) };
    let (Some(Value::Int(status)), Some(Value::List(hs)), Some(Value::Str(body))) = (field(fields, "status"), field(fields, "headers"), field(fields, "body")) else {
        return Err("malformed HttpResponse".into());
    };
    // 1xx are interim responses: sent as the final one they desynchronize clients and proxies
    // (101 would switch a proxy to a raw tunnel).
    let status = u16::try_from(*status).ok().filter(|s| (200..=599).contains(s)).ok_or_else(|| format!("status {status} is not a final status (200..599)"))?;
    if body.len() > max_body {
        return Err(format!("the body has {} bytes, more than the limit of {max_body}", body.len()));
    }
    if bodiless(status) && !body.is_empty() {
        return Err(format!("a {status} response cannot have a body"));
    }
    let mut headers = Vec::new();
    for h in hs {
        let Value::Record(_, f) = h else { return Err("malformed header".into()) };
        let (Some(Value::Str(n)), Some(Value::Str(val))) = (field(f, "name"), field(f, "value")) else { return Err("malformed header".into()) };
        if !is_token(n) {
            return Err(format!("`{}` is not a valid header name", n.escape_debug()));
        }
        if val.bytes().any(|b| b == b'\r' || b == b'\n' || b == 0) {
            return Err(format!("the value of header `{n}` contains CR, LF or NUL"));
        }
        let lower = n.to_ascii_lowercase();
        if matches!(lower.as_str(), "content-length" | "transfer-encoding" | "connection" | "upgrade") {
            return Err(format!("the handler may not set `{lower}` (the host does)"));
        }
        // The request id is always the host's.
        if lower != "x-request-id" {
            headers.push((lower, val.clone()));
        }
    }
    Ok(Response { status, headers, body: body.clone() })
}

fn field<'v>(fields: &'v [(String, Value)], name: &str) -> Option<&'v Value> {
    fields.iter().find(|(n, _)| n == name).map(|(_, v)| v)
}

pub struct Service {
    runner: Runner,
    cfg: ServiceConfig,
    next_id: AtomicU64,
    uses_db: bool,
}

impl Service {
    /// Checks the handler (and init function) and the grants, and prepares the module. Fails with
    /// `bad_entry`, `permission_denied` or `compile_error` before anything runs.
    pub fn new(typed: &TProgram, perms: Permissions, cfg: ServiceConfig) -> Result<Service, RunError> {
        let mut effects = check_entry(typed, &cfg.entry, &[named("HttpRequest")], &named("HttpResponse"), "handler")?;
        let mut entries = vec![cfg.entry.clone()];
        if let Some(init) = &cfg.init {
            effects.extend(check_entry(typed, init, &[], &Type::result(Type::Unit, Type::Str), "init function")?);
            entries.push(init.clone());
        }
        effects.sort();
        effects.dedup();
        let missing: Vec<&String> = effects.iter().filter(|e| !perms.allows(e)).collect();
        if !missing.is_empty() {
            let list: Vec<&str> = missing.iter().map(|s| s.as_str()).collect();
            let flag = list.iter().map(|e| match *e {
                "fs" => "fs:<dir>",
                "net" => "net:<host>",
                other => other,
            });
            return Err(RunError::new("permission_denied", &cfg.entry, format!("the service needs effect(s) {} which were not granted", list.join(", "))).hint(format!("grant them with `--allow {}`", flag.collect::<Vec<_>>().join(","))));
        }
        let compile_error = |m: String| RunError::new("compile_error", &cfg.entry, format!("the program cannot be compiled to WebAssembly: {m}"));
        let module = qal_wasm::compile(typed, &Entry::Fns(entries)).map_err(compile_error)?;
        let runner = Runner::prepared(module, perms, cfg.memory, cfg.slots)
            .map_err(compile_error)?
            .with_fuel(cfg.fuel)
            .with_timeout(cfg.request_timeout)
            .with_limits(cfg.budgets)
            .with_stdout_limit(cfg.max_out)
            .with_exact_results(cfg.max_response.saturating_add(1 << 20));
        let uses_db = effects.iter().any(|e| e == "db");
        Ok(Service { runner, cfg, next_id: AtomicU64::new(1), uses_db })
    }

    pub fn config(&self) -> &ServiceConfig {
        &self.cfg
    }

    /// True if executions use the pooling allocator.
    pub fn pooled(&self) -> bool {
        self.runner.pooled()
    }

    /// The prepared runner (for measurements).
    pub fn runner(&self) -> &Runner {
        &self.runner
    }

    /// A worker with its own database connection (opened now if the service uses `db`, so a
    /// broken database stops the start).
    pub fn worker(&self) -> Result<Worker, RunError> {
        let mut w = Worker::default();
        self.ensure_db(&mut w)?;
        Ok(w)
    }

    fn ensure_db(&self, w: &mut Worker) -> Result<(), RunError> {
        if !self.uses_db || w.db.is_some() {
            return Ok(());
        }
        let perms = self.runner.permissions();
        if let Some(grant) = &perms.db {
            let db = Db::open(grant, &perms.db_dir(), self.cfg.budgets.db_max_mib).map_err(|m| RunError::new("db_open", &self.cfg.entry, m))?;
            w.db = Some(db);
        }
        Ok(())
    }

    /// The next request id (monotonic per process).
    pub fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Runs the init function (if any) in its own instance and transaction: committed only on
    /// `Ok(())`. The error is a runtime error, or `Err(message)` as `init_failed`.
    pub fn init(&self, w: &mut Worker) -> Result<(), RunError> {
        let Some(init) = &self.cfg.init else { return Ok(()) };
        self.ensure_db(w)?;
        let ok = |r: &Result<Value, RunError>| matches!(r, Ok(Value::Variant(n, _)) if n == "Ok");
        let out = self.runner.execute(init, &[], ExecIn { db: w.db.take(), host_used: 0 }, &ok);
        w.db = out.db;
        match out.outcome.result {
            Ok(Value::Variant(n, _)) if n == "Ok" => Ok(()),
            Ok(Value::Variant(_, payload)) => {
                let m = match payload.first() {
                    Some(Value::Str(m)) => m.clone(),
                    _ => String::new(),
                };
                Err(RunError::new("init_failed", init, format!("`{init}` returned Err: {m}")))
            }
            Ok(other) => Err(RunError::new("trap", init, format!("`{init}` returned {other:?}"))),
            Err(e) => Err(e),
        }
    }

    /// Handles one request on worker `w` (spec §2.2, §2.6). `head_bytes` (the request line and
    /// headers as received) count against the host budget together with the body.
    pub fn handle(&self, w: &mut Worker, req: &Request, id: u64, head_bytes: usize) -> Handled {
        let start = Instant::now();
        let mut log = LogEntry { event: "request", id, method: req.method.clone(), path: req.path.clone(), status: 500, ms: 0, fuel: 0, mem_bytes: 0, db: "none", out: None, out_bytes: None, error: None };
        if let Err(e) = self.ensure_db(w) {
            return self.failed(log, e, start);
        }
        let max = self.cfg.max_response;
        let commit = |r: &Result<Value, RunError>| matches!(r, Ok(v) if validate(v, max).is_ok_and(|resp| resp.status < 400));
        let input = ExecIn { db: w.db.take(), host_used: head_bytes.saturating_add(req.body.len()) };
        let out = self.runner.execute(&self.cfg.entry, &[request_value(req)], input, &commit);
        w.db = out.db;
        log.fuel = out.fuel;
        log.mem_bytes = out.mem_bytes;
        log.db = out.outcome.db;
        if out.outcome.stdout_bytes > 0 {
            log.out = Some(out.outcome.stdout.clone());
            log.out_bytes = Some(out.outcome.stdout_bytes);
        }
        let result = out.outcome.result.and_then(|v| validate(&v, max).map_err(|m| RunError::new("bad_response", &self.cfg.entry, format!("invalid response: {m}"))));
        match result {
            Ok(mut resp) => {
                // A 204/304 has no body, so nothing to describe: no default content-type and no
                // content-length (RFC 9110 §8.3, §8.6).
                if !bodiless(resp.status) {
                    if resp.header("content-type").is_none() {
                        resp.headers.push(("content-type".into(), "text/plain; charset=utf-8".into()));
                    }
                    resp.headers.push(("content-length".into(), resp.body.len().to_string()));
                }
                resp.headers.push(("x-request-id".into(), id.to_string()));
                log.status = resp.status;
                log.ms = start.elapsed().as_millis() as u64;
                Handled { response: resp, log }
            }
            Err(e) => self.failed(log, e, start),
        }
    }

    /// The 500/503 response for a runtime error (`timeout` is 503).
    fn failed(&self, mut log: LogEntry, e: RunError, start: Instant) -> Handled {
        let (status, kind) = if e.kind == "timeout" { (503, "timeout") } else { (500, "internal_error") };
        let body = if self.cfg.debug_errors {
            serde_json::json!({"error": e, "id": log.id.to_string()}).to_string()
        } else {
            serde_json::json!({"error": {"kind": kind, "id": log.id.to_string()}}).to_string()
        };
        log.status = status;
        log.ms = start.elapsed().as_millis() as u64;
        log.error = Some(e);
        Handled { response: Response::json(status, body, log.id), log }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(status: i64, headers: &[(&str, &str)], body: &str) -> Value {
        Value::Record(
            "HttpResponse".into(),
            vec![
                ("status".into(), Value::Int(status)),
                ("headers".into(), headers_value(&headers.iter().map(|(n, v)| (n.to_string(), v.to_string())).collect::<Vec<_>>())),
                ("body".into(), Value::Str(body.into())),
            ],
        )
    }

    #[test]
    fn responses_are_validated() {
        assert_eq!(validate(&rec(200, &[("X-A", "1")], "hi"), 10).unwrap(), Response { status: 200, headers: vec![("x-a".into(), "1".into())], body: "hi".into() });
        let e = |v: Value| validate(&v, 10).unwrap_err();
        assert!(e(rec(99, &[], "")).contains("not a final status"));
        assert!(e(rec(600, &[], "")).contains("not a final status"));
        assert!(e(rec(100, &[], "")).contains("not a final status"), "found in review");
        assert!(e(rec(101, &[("upgrade", "websocket")], "")).contains("not a final status"));
        assert!(e(rec(200, &[], "01234567890")).contains("more than the limit"));
        assert!(e(rec(204, &[], "x")).contains("cannot have a body"));
        assert!(e(rec(200, &[("x", "a\r\nset-cookie: y")], "")).contains("CR, LF or NUL"));
        assert!(e(rec(200, &[("bad name", "v")], "")).contains("not a valid header name"));
        assert!(e(rec(200, &[("", "v")], "")).contains("not a valid header name"));
        for h in ["Content-Length", "transfer-encoding", "Connection", "Upgrade"] {
            assert!(e(rec(200, &[(h, "1")], "")).contains("may not set"), "{h}");
        }
        assert!(validate(&rec(200, &[("x-request-id", "mine")], ""), 10).unwrap().headers.is_empty(), "the host sets the request id");
        assert!(bodiless(204) && bodiless(304) && bodiless(101) && !bodiless(200));
    }
}
