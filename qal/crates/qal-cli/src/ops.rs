//! The operations of spec §9, shared by the command line and the MCP server: each returns one
//! [`Reply`] (a JSON object plus the CLI exit code), never panics, and reports every problem as
//! data (`diagnostics` for compile-time and store problems, `error` for runtime failures).

use crate::{is_entry_point, json_args, load, load_program, parse_args, Loaded};
use qal_core::canon::Hash;
use qal_core::diag::Diagnostic;
use qal_core::tast::TProgram;
use qal_prove::{PropOptions, PropReport, PropStatus, Report, Status};
use qal_run::{Permissions, RunError, TestResult, Value};
use qal_store::{Defined, Entry as QueryEntry, HistoryEntry, Query, Renamed, Store, Verify, View};
use qal_wasm::Entry;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// One operation's JSON result. Serialized directly (not via `serde_json::Value`) so record fields
/// and contract values keep their declaration order. Absent fields are omitted.
#[derive(Debug, Serialize, Default)]
pub struct Output {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ok: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defined: Option<Vec<Defined>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub views: Option<Vec<View>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results: Option<Vec<QueryEntry>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub renamed: Option<Renamed>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub history: Option<Vec<HistoryEntry>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub built: Option<Built>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RunError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stdout: Option<String>,
    /// Total size in bytes of a captured stdout that was cut to `Ctx::max_stdout`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stdout_truncated: Option<usize>,
    /// What happened to the database transaction of a run that used `db`: `commit` or `rollback`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub db: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Vec<Diagnostic>>,
    /// Contract proofs: one report per obligation (`define`: of the definitions it changed; `prove`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proofs: Option<Vec<Report>>,
    /// Number of reports left out of `proofs` to keep the result small ([`MAX_LISTED_BYTES`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proofs_omitted: Option<usize>,
    /// Proof results that are not `proven`, as diagnostics: E0801 `contract_refuted` (of a test, or in
    /// `prove`), E0802 `contract_unproven`. They do not make the operation fail.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<Diagnostic>>,
    /// Number of diagnostics left out of `warnings` to keep the result small ([`MAX_LISTED_BYTES`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings_omitted: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proven: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refuted: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unknown: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tests: Option<Vec<TestResult>>,
    /// Generated property tests of functions with contracts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<Vec<PropReport>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub passed: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failed: Option<usize>,
    /// Property tests that failed (`passed`/`failed` count the written tests).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failed_properties: Option<usize>,
    /// Tests not run because the time budget (`Ctx::test_budget`) was used up.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skipped: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exported: Option<Exported>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub imported: Option<qal_store::Imported>,
    /// Hash of the standard library package this write imported first (it was new to the project).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub std_imported: Option<String>,
    /// The results of `request`, one per request, in order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub responses: Option<Vec<RequestResult>>,
}

/// One request of `request` / `qal_request` (spec Phase 3 §7.2).
#[derive(Debug, Clone, PartialEq)]
pub struct RequestSpec {
    pub request: qal_serve::Request,
}

/// Most requests per `request` call.
pub const MAX_REQUESTS: usize = 50;

impl RequestSpec {
    /// Parses `{method?, path, query?, headers?, body? | json?}`. Header names are lowercased and
    /// hop-by-hop headers dropped, as the HTTP layer of `qal serve` does.
    pub fn from_json(v: &serde_json::Value) -> Result<RequestSpec, String> {
        use serde_json::Value as J;
        let J::Object(m) = v else { return Err("a request must be an object `{method, path, query, headers, body | json}`".into()) };
        if let Some(k) = m.keys().find(|k| !["method", "path", "query", "headers", "body", "json"].contains(&k.as_str())) {
            return Err(format!("unknown request field `{k}` (fields: method, path, query, headers, body, json)"));
        }
        let text = |k: &str, default: &str| -> Result<String, String> {
            match m.get(k) {
                None | Some(J::Null) => Ok(default.to_string()),
                Some(J::String(s)) => Ok(s.clone()),
                Some(_) => Err(format!("request field `{k}` must be a string")),
            }
        };
        let method = text("method", "GET")?;
        if method.is_empty() || !method.bytes().all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b)) {
            return Err(format!("`{method}` is not an HTTP method"));
        }
        let path = text("path", "")?;
        if !path.starts_with('/') || path.contains('?') || path.contains(char::is_whitespace) {
            return Err(format!("`path` must start with `/` and contain no `?` or whitespace (the query goes in `query`), got `{path}`"));
        }
        let query = text("query", "")?;
        let mut headers = Vec::new();
        match m.get("headers") {
            None | Some(J::Null) => {}
            Some(J::Array(hs)) => {
                for h in hs {
                    let (Some(J::String(n)), Some(J::String(val))) = (h.get("name"), h.get("value")) else { return Err("each header must be `{\"name\": \"...\", \"value\": \"...\"}`".into()) };
                    if n.is_empty() || !n.bytes().all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b)) {
                        return Err(format!("`{n}` is not a header name"));
                    }
                    if val.bytes().any(|b| b == b'\r' || b == b'\n' || b == 0) {
                        return Err(format!("the value of header `{n}` contains CR, LF or NUL"));
                    }
                    let n = n.to_ascii_lowercase();
                    if !["connection", "keep-alive", "transfer-encoding", "upgrade", "te", "trailer"].contains(&n.as_str()) {
                        headers.push((n, val.trim().to_string()));
                    }
                }
            }
            Some(_) => return Err("`headers` must be an array".into()),
        }
        // `json: null` is no JSON body (as in a GET).
        let body = match (m.get("body").filter(|b| !b.is_null()), m.get("json").filter(|j| !j.is_null())) {
            (Some(_), Some(_)) => return Err("give either `body` or `json`, not both".into()),
            (None, Some(j)) => {
                if !headers.iter().any(|(n, _)| n == "content-type") {
                    headers.push(("content-type".into(), "application/json".into()));
                }
                j.to_string()
            }
            _ => text("body", "")?,
        };
        match headers.iter().find(|(n, _)| n == "content-length") {
            Some((_, v)) if v.parse::<usize>().ok() != Some(body.len()) => return Err(format!("the `content-length` header says `{v}`, but the body has {} bytes", body.len())),
            Some(_) => {}
            None if !body.is_empty() => headers.push(("content-length".into(), body.len().to_string())),
            None => {}
        }
        Ok(RequestSpec { request: qal_serve::Request { method, path, query, headers, body } })
    }
}

/// The result of one request (spec Phase 3 §7.2).
#[derive(Debug, Clone, Serialize)]
pub struct RequestResult {
    pub status: u16,
    pub headers: Vec<qal_run::Value>,
    pub body: String,
    /// The body parsed, when the response is JSON.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json: Option<serde_json::Value>,
    /// `commit`, `rollback` or `none`.
    pub db: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stdout: Option<String>,
    /// The full runtime error of a failed request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RunError>,
}

/// The result of `build`: where the module and its manifest were written.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Built {
    pub name: String,
    pub hash: String,
    /// Paths relative to the project root.
    pub wasm: String,
    pub manifest: String,
    pub effects: Vec<String>,
    pub bytes: usize,
}

/// `<name>.manifest.json` next to `<name>.wasm` (spec §8).
#[derive(Debug, Serialize)]
struct Manifest<'a> {
    format: &'static str,
    name: &'a str,
    /// Hash of the definition in the store.
    hash: &'a str,
    signature: String,
    /// True for `fn(List[Str]) -> Int`, runnable with `qal run`.
    entry_point: bool,
    /// Effects the module needs granted (sorted).
    effects: &'a [String],
    /// Functions imported from module `qal_host`.
    imports: &'a [&'static str],
    /// Exported functions besides `memory` and `qal_alloc`.
    exports: Vec<String>,
    wasm_sha256: String,
    bytes: usize,
}

/// An operation's output and the CLI exit code that goes with it.
#[derive(Debug)]
pub struct Reply {
    pub out: Output,
    pub code: u8,
}

impl Reply {
    fn new(out: Output, code: u8) -> Reply {
        Reply { out, code }
    }

    /// Diagnostics; exit 2 if the store itself failed (missing, unreadable, corrupt), else 1.
    pub fn diagnostics(d: Vec<Diagnostic>) -> Reply {
        let code = if !d.is_empty() && d.iter().all(|d| d.kind == "store_error") { 2 } else { 1 };
        Reply::new(Output { diagnostics: Some(d), ..Output::default() }, code)
    }

    fn error(e: RunError) -> Reply {
        Reply::new(Output { error: Some(capped(e)), ..Output::default() }, 1)
    }

    /// True if the operation failed: diagnostics, a runtime error, or failed tests (a program
    /// that runs and returns a non-zero exit code did not fail).
    pub fn is_error(&self) -> bool {
        self.out.diagnostics.is_some() || self.out.error.is_some() || self.out.failed.is_some_and(|n| n > 0) || self.out.failed_properties.is_some() || self.out.refuted.is_some_and(|n| n > 0)
    }
}

/// Where the operations find the store and how programs run.
#[derive(Debug, Clone)]
pub struct Ctx {
    pub store: StoreAt,
    /// Input for `read_line`; None reads the process's stdin (CLI only: the MCP server's stdin
    /// is the protocol channel).
    pub stdin: Option<Vec<u8>>,
    /// Wall-clock limit per program execution.
    pub timeout: Option<Duration>,
    /// Wall-clock budget for all tests of one `test` call; the rest are reported as `skipped`.
    pub test_budget: Option<Duration>,
    /// Captured stdout longer than this is cut (at a character boundary).
    pub max_stdout: Option<usize>,
}

#[derive(Debug, Clone)]
pub enum StoreAt {
    /// The nearest `.qal/store.db` in this directory or its parents (CLI).
    Discover(PathBuf),
    /// `<root>/.qal/store.db`, created by the first `define` (MCP server).
    Root(PathBuf),
}

/// What `test`, `call` and `run` work on.
pub enum Source<'a> {
    /// Source text of a file (nothing is stored).
    Text(&'a str),
    /// The codebase in the store.
    Store,
}

/// Arguments of `call`: strings from the command line or JSON values from an agent.
pub enum CallArgs<'a> {
    Strings(&'a [String]),
    Json(&'a [serde_json::Value]),
}

impl Ctx {
    /// The CLI context: the store around the current directory, no limits beyond the runtime's.
    pub fn cli() -> Result<Ctx, Vec<Diagnostic>> {
        let cwd = std::env::current_dir().map_err(|e| vec![Diagnostic::store_error(format!("no current directory: {e}"))])?;
        Ok(Ctx { store: StoreAt::Discover(cwd), stdin: None, timeout: None, test_budget: None, max_stdout: None })
    }

    /// The store; `create` makes it under `Root` if it does not exist yet (only `define` does,
    /// so reading never leaves a store behind).
    pub fn open_store(&self, create: bool) -> Result<Store, Vec<Diagnostic>> {
        match &self.store {
            StoreAt::Discover(dir) => Store::discover(dir),
            StoreAt::Root(root) if create => Store::init(root),
            StoreAt::Root(root) => {
                let path = root.join(qal_store::STORE_DIR).join(qal_store::STORE_FILE);
                if !path.exists() {
                    return Err(vec![Diagnostic::store_error(format!("there is no QAL code in {} yet", root.display())).hint("define code with qal_define first; it creates the store")]);
                }
                Store::open(&path)
            }
        }
    }

    /// The existing store.
    pub fn open(&self) -> Result<Store, Vec<Diagnostic>> {
        self.open_store(false)
    }

    fn load(&self, source: &Source) -> Result<Loaded, Vec<Diagnostic>> {
        match source {
            Source::Text(src) => load(src),
            Source::Store => load_program(self.open()?.program()?),
        }
    }

    fn runner(&self, loaded: &Loaded, entry: &Entry, mut perms: Permissions) -> Result<qal_run::Runner, RunError> {
        // Database files belong to the project whose store the CLI works in.
        if let (Some(_), None, StoreAt::Discover(dir)) = (&perms.db, &perms.db_dir, &self.store) {
            if let Some(project) = qal_store::project_of(dir) {
                perms = perms.with_db_dir(project.join(qal_store::STORE_DIR).join("db"));
            }
        }
        let mut r = loaded.runner(entry, perms)?;
        if let Some(t) = self.timeout {
            r = r.with_timeout(t);
        }
        if let Some(input) = &self.stdin {
            r = r.with_stdin(input.clone());
        }
        if let Some(max) = self.max_stdout {
            r = r.with_stdout_limit(max);
        }
        Ok(r)
    }

    /// Sets `stdout` (cut to `max_stdout` at a character boundary) on `out`; `total` is the
    /// number of bytes the program printed.
    fn stdout(&self, out: &mut Output, stdout: String, total: usize) {
        let Some(max) = self.max_stdout else {
            out.stdout = Some(stdout);
            return;
        };
        let mut end = stdout.len().min(max);
        while !stdout.is_char_boundary(end) {
            end -= 1;
        }
        if end < stdout.len() || total > max {
            out.stdout_truncated = Some(total.max(stdout.len()));
        }
        out.stdout = Some(stdout[..end].to_string());
    }

    /// True for the MCP server's context (hints then name tools, not CLI commands).
    fn is_mcp(&self) -> bool {
        matches!(self.store, StoreAt::Root(_))
    }

    /// A hint for a name that is not in the codebase but was renamed in the store.
    fn renamed_hint(&self, source: &Source, name: &str) -> Option<String> {
        match source {
            Source::Store => self.open().ok()?.renamed_to(name).ok()?.map(|new| format!("`{name}` was renamed to `{new}`")),
            Source::Text(_) => None,
        }
    }
}

/// Creates `<dir>/.qal/store.db` (idempotent) and imports the embedded standard library into it.
/// `std`: the store of the std project instead (`meta.role = 'std'`, only `std.` names, no
/// automatic import); a new one is filled from the embedded package with its recorded agents.
pub fn init(dir: &Path, std: bool) -> Reply {
    let opened = if std {
        Store::init_std(dir).and_then(|mut s| {
            let fresh = s.query(&Query::default())?.is_empty() && s.meta(qal_store::package::META_STD_PACKAGE)?.is_none();
            let imported = if fresh { proving(|v| s.import_verified(crate::STD_PACKAGE, v)).0?.hash } else { String::new() };
            Ok((s, (!imported.is_empty()).then_some(imported)))
        })
    } else {
        Store::init(dir).and_then(|mut s| {
            let imported = with_std(&mut s)?;
            Ok((s, imported))
        })
    };
    match opened {
        Ok((s, std_imported)) => Reply::new(Output { ok: Some(Value::Bool(true)), store: Some(s.path().display().to_string()), std_imported, ..Output::default() }, 0),
        Err(d) => Reply::diagnostics(d),
    }
}

/// Imports the embedded standard library into the project's store if it does not have this
/// version yet (spec Phase 3 §4.3): before every write, in its own transaction, never in the std
/// project's store. Returns the package hash if it imported. Its contracts are proven as by any
/// import; the proof reports are not part of the write's result.
fn with_std(store: &mut Store) -> Result<Option<String>, Vec<Diagnostic>> {
    let (result, _) = proving(|v| store.ensure_std(crate::STD_PACKAGE, Some(v)));
    match result {
        Ok(imported) => Ok(imported.map(|i| i.hash)),
        Err(mut d) => {
            for x in &mut d {
                x.hints.push("the standard library of this qal could not be imported into the project first, so nothing was changed".into());
            }
            Err(d)
        }
    }
}

/// The store for a write (`create`: make it if missing), with the standard library imported.
fn open_for_write(ctx: &Ctx, create: bool) -> Result<(Store, Option<String>), Vec<Diagnostic>> {
    let mut store = ctx.open_store(create)?;
    let imported = with_std(&mut store)?;
    Ok((store, imported))
}

/// Parses and checks text without storing anything.
pub fn check(src: &str) -> Reply {
    match load(src) {
        Ok(_) => Reply::new(Output { ok: Some(Value::Bool(true)), ..Output::default() }, 0),
        Err(d) => Reply::diagnostics(d),
    }
}

/// Adds or replaces definitions; their contracts are proven before anything is stored. A function
/// with a refuted contract (its `ens`, or the `req` of a function it calls) is not stored; refuted
/// call sites in tests are reported (the test fails when run) but do not stop the definition.
pub fn define(ctx: &Ctx, src: &str, agent: &str) -> Reply {
    let (mut store, std_imported) = match open_for_write(ctx, true) {
        Ok(s) => s,
        Err(d) => return Reply::diagnostics(d),
    };
    let (result, reports) = proving(|verify| store.define_verified(src, agent, verify));
    match result {
        Ok(d) => {
            let reports = current_reports(&d, reports);
            let mut out = Output { defined: Some(d), std_imported, ..Output::default() };
            out.set_proofs(reports);
            Reply::new(out, 0)
        }
        Err(d) => Reply::diagnostics(d),
    }
}

/// Runs a store write with the verification step of `define` (the contracts of what it defines
/// are proven; a refuted one rejects the write) and returns its result with the proof reports.
/// Each report comes with the hash it was proven for (see [`current_reports`]).
fn proving<T>(write: impl FnOnce(&mut Verify) -> Result<T, Vec<Diagnostic>>) -> (Result<T, Vec<Diagnostic>>, Vec<(String, Report)>) {
    let opts = qal_prove::Options::default();
    // Reports with the hash they were proven for: a retried `define` may have verified an older
    // version first (review finding N4).
    let mut reports: Vec<(String, Report)> = Vec::new();
    let result = {
        let mut run = |tp: &TProgram, todo: &[(String, String)]| -> Result<Vec<(String, String)>, Vec<Diagnostic>> {
            let names: Vec<String> = todo.iter().map(|(n, _)| n.clone()).collect();
            let proofs = qal_prove::prove_defs(tp, Some(&names), &opts);
            let is_test = |n: &str| tp.tests.iter().any(|t| t.name == n);
            let refuted: Vec<Diagnostic> = proofs.iter().flat_map(|p| &p.reports).filter(|r| r.status == Status::Refuted && !is_test(&r.def)).filter_map(Report::diagnostic).collect();
            if !refuted.is_empty() {
                let (mut shown, omitted) = listed(refuted);
                if let (Some(n), Some(last)) = (omitted, shown.last_mut()) {
                    last.hints.push(format!("{n} more refuted obligations are not listed"));
                }
                return Err(shown);
            }
            let rows = todo.iter().filter_map(|(n, h)| proofs.iter().find(|p| p.def == *n)?.record.as_ref().map(|r| (h.clone(), record_json(r)))).collect();
            let hash_of = |n: &str| todo.iter().find(|(m, _)| m == n).map(|(_, h)| h.clone()).unwrap_or_default();
            reports.retain(|(h, r)| !todo.iter().any(|(n, _)| *n == r.def) && !h.is_empty());
            reports.extend(proofs.into_iter().flat_map(|p| {
                let h = hash_of(&p.def);
                p.reports.into_iter().map(move |r| (h.clone(), r))
            }));
            Ok(rows)
        };
        let mut verify = Verify { key: opts.key(), run: &mut run };
        write(&mut verify)
    };
    (result, reports)
}

/// The reports of the definitions as they were stored: a retried write may have verified an older
/// version first (review finding N4).
fn current_reports(defined: &[Defined], reports: Vec<(String, Report)>) -> Vec<Report> {
    let current: std::collections::HashSet<(&str, &str)> = defined.iter().map(|x| (x.name.as_str(), x.hash.as_str())).collect();
    reports.into_iter().filter(|(h, r)| current.contains(&(r.def.as_str(), h.as_str()))).map(|(_, r)| r).collect()
}

/// Bytes of JSON that `proofs` and `warnings` may each take in one result; the rest is counted in
/// `proofs_omitted` / `warnings_omitted` (a result must stay below [`crate::mcp::MAX_RESULT_BYTES`]).
pub const MAX_LISTED_BYTES: usize = 512 << 10;

/// The longest prefix of `items` within [`MAX_LISTED_BYTES`] of JSON, and how many were left out.
fn listed<T: Serialize>(items: Vec<T>) -> (Vec<T>, Option<usize>) {
    let total = items.len();
    let mut bytes = 0;
    let mut kept = Vec::new();
    for x in items {
        bytes += serde_json::to_vec(&x).map(|v| v.len() + 1).unwrap_or(0);
        if bytes > MAX_LISTED_BYTES {
            break;
        }
        kept.push(x);
    }
    let omitted = total - kept.len();
    (kept, (omitted > 0).then_some(omitted))
}

impl Output {
    /// Sets `proofs` (if there are any) and `warnings`, the diagnostic form of the results that are
    /// not `proven`, each within [`MAX_LISTED_BYTES`].
    fn set_proofs(&mut self, reports: Vec<Report>) {
        let w: Vec<Diagnostic> = reports.iter().filter_map(Report::diagnostic).collect();
        if !w.is_empty() {
            let (w, omitted) = listed(w);
            self.warnings = Some(w);
            self.warnings_omitted = omitted;
        }
        if !reports.is_empty() {
            let (r, omitted) = listed(reports);
            self.proofs = Some(r);
            self.proofs_omitted = omitted;
        }
    }

    /// Leaves out every proof report and warning (a committed `define` whose result would be too
    /// large otherwise): their numbers stay in `proofs_omitted` / `warnings_omitted`.
    pub fn omit_proofs(&mut self) {
        if let Some(p) = self.proofs.take() {
            self.proofs_omitted = Some(p.len() + self.proofs_omitted.unwrap_or(0));
        }
        if let Some(w) = self.warnings.take() {
            self.warnings_omitted = Some(w.len() + self.warnings_omitted.unwrap_or(0));
        }
    }
}

fn record_json(r: &qal_prove::Record) -> String {
    serde_json::to_string(r).unwrap_or_else(|_| "{\"results\":[]}".into())
}

/// Proves the contracts of the named functions and tests (all if `names` is empty). From the
/// store, results stored for the same definition and prover settings are reused; new ones are stored.
pub fn prove(ctx: &Ctx, source: &Source, names: &[String]) -> Reply {
    // From the store, the program and the hashes its results are stored under come from one read
    // (a `define` in between must not get the old definition's results under its new hash).
    let read = match source {
        Source::Text(src) => load(src).map(|l| (l, std::collections::HashMap::new())),
        Source::Store => ctx.open().and_then(|mut s| s.program_with_hashes()).and_then(|(p, h)| Ok((load_program(p)?, h))),
    };
    let (loaded, hashes) = match read {
        Ok(l) => l,
        Err(d) => return Reply::diagnostics(d),
    };
    let tp = &loaded.typed;
    let known = |n: &str| tp.fns.iter().any(|f| f.def.name == n && f.instance.is_none()) || tp.generics.iter().any(|f| f.def.name == n) || tp.tests.iter().any(|t| t.name == n);
    let unknown: Vec<Diagnostic> = names
        .iter()
        .filter(|n| !known(n))
        .map(|n| {
            let d = Diagnostic::unknown_name(n, "", n);
            match ctx.renamed_hint(source, n) {
                Some(h) => d.hint(h),
                None if tp.type_def(n).is_some() => d.hint(format!("`{n}` is a type; only functions and tests have contracts to prove")),
                None => d,
            }
        })
        .collect();
    if !unknown.is_empty() {
        return Reply::diagnostics(unknown);
    }
    let opts = qal_prove::Options::default();
    let only = (!names.is_empty()).then_some(names);
    let reports = match source {
        Source::Text(_) => qal_prove::prove(tp, only, &opts),
        Source::Store => match stored_proofs(ctx, tp, &hashes, only, &opts) {
            Ok(r) => r,
            Err(d) => return Reply::diagnostics(d),
        },
    };
    let count = |st: Status| reports.iter().filter(|r| r.status == st).count();
    let (proven, refuted, unknown) = (count(Status::Proven), count(Status::Refuted), count(Status::Unknown));
    let mut out = Output { proven: Some(proven), refuted: Some(refuted), unknown: Some(unknown), ..Output::default() };
    out.set_proofs(reports);
    out.proofs.get_or_insert_with(Vec::new);
    Reply::new(out, u8::from(refuted > 0))
}

/// Proof results of the stored definitions: reused per (hash, prover key) and shown with the
/// current names, computed and stored if missing. Showing and computing share one time budget.
/// `hashes` are the hashes of `tp`'s definitions, read with it.
fn stored_proofs(ctx: &Ctx, tp: &TProgram, hashes: &std::collections::HashMap<String, String>, only: Option<&[String]>, opts: &qal_prove::Options) -> Result<Vec<Report>, Vec<Diagnostic>> {
    let start = Instant::now();
    let deadline = start + opts.budget;
    let mut store = ctx.open()?;
    let key = opts.key();
    let targets: Vec<String> = tp
        .fns
        .iter()
        .filter(|f| f.instance.is_none())
        .map(|f| &f.def.name)
        .chain(tp.generics.iter().map(|f| &f.def.name))
        .chain(tp.tests.iter().map(|t| &t.name))
        .filter(|n| only.is_none_or(|ns| ns.contains(n)))
        .cloned()
        .collect();
    let mut cached: std::collections::HashMap<String, Vec<Report>> = std::collections::HashMap::new();
    let mut missing = Vec::new();
    for n in &targets {
        let stored = match hashes.get(n) {
            Some(h) => store.proofs(h, &key)?.and_then(|j| serde_json::from_str::<qal_prove::Record>(&j).ok()).and_then(|rec| qal_prove::render(tp, n, &rec, deadline)),
            None => None,
        };
        match stored {
            Some(rs) => {
                cached.insert(n.clone(), rs);
            }
            None => missing.push(n.clone()),
        }
    }
    if !missing.is_empty() {
        let left = qal_prove::Options { budget: opts.budget.saturating_sub(start.elapsed()), ..opts.clone() };
        for p in qal_prove::prove_defs(tp, Some(&missing), &left) {
            if let (Some(h), Some(rec)) = (hashes.get(&p.def), &p.record) {
                store.save_proofs(h, &key, &record_json(rec))?;
            }
            cached.insert(p.def, p.reports);
        }
    }
    Ok(targets.iter().flat_map(|n| cached.remove(n).unwrap_or_default()).collect())
}

/// Shows the named definitions. Repeated names count once; the views are built one by one under a
/// byte budget ([`crate::mcp::MAX_RESULT_BYTES`]), so a request can never make this build more than
/// it may return.
pub fn view(ctx: &Ctx, names: &[String]) -> Reply {
    let mut store = match ctx.open() {
        Ok(s) => s,
        Err(d) => return Reply::diagnostics(d),
    };
    let mut seen = std::collections::HashSet::new();
    let (mut views, mut diags, mut bytes) = (Vec::new(), Vec::new(), 0usize);
    for n in names.iter().filter(|n| seen.insert(n.as_str())) {
        match store.view(std::slice::from_ref(n)) {
            Ok(v) => {
                bytes += v.iter().map(|v| serde_json::to_vec(v).map_or(usize::MAX, |b| b.len())).fold(0, usize::saturating_add);
                if bytes > crate::mcp::MAX_RESULT_BYTES {
                    let limit = crate::mcp::MAX_RESULT_BYTES;
                    let d = Diagnostic::new("E0702", "result_too_large", "", "", format!("the views are more than {limit} bytes of JSON")).hint("view fewer or smaller definitions per call");
                    return Reply::diagnostics(vec![d]);
                }
                views.extend(v);
            }
            Err(d) => diags.extend(d),
        }
    }
    if diags.is_empty() {
        Reply::new(Output { views: Some(views), ..Output::default() }, 0)
    } else {
        Reply::diagnostics(diags)
    }
}

pub fn query(ctx: &Ctx, q: &Query) -> Reply {
    match ctx.open().and_then(|mut s| s.query(q)) {
        Ok(r) => Reply::new(Output { results: Some(r), ..Output::default() }, 0),
        Err(d) => Reply::diagnostics(d),
    }
}

pub fn rename(ctx: &Ctx, old: &str, new: &str, agent: &str) -> Reply {
    match open_for_write(ctx, false).and_then(|(mut s, std_imported)| Ok((s.rename(old, new, agent)?, std_imported))) {
        Ok((r, std_imported)) => Reply::new(Output { renamed: Some(r), std_imported, ..Output::default() }, 0),
        Err(d) => Reply::diagnostics(d),
    }
}

pub fn history(ctx: &Ctx, name: &str) -> Reply {
    match ctx.open().and_then(|mut s| s.history(name)) {
        Ok(h) => Reply::new(Output { history: Some(h), ..Output::default() }, 0),
        Err(d) => Reply::diagnostics(d),
    }
}

/// Runs the named tests and property tests (all if `names` is empty); `seed` varies the generated inputs.
pub fn test(ctx: &Ctx, source: &Source, perms: Permissions, names: &[String], seed: u64) -> Reply {
    let loaded = match ctx.load(source) {
        Ok(l) => l,
        Err(d) => return Reply::diagnostics(d),
    };
    let candidates: Vec<String> = qal_prove::props::candidates(&loaded.typed).into_iter().map(|i| loaded.typed.fns[i].def.name.clone()).collect();
    let mut unknown = Vec::new();
    for n in names {
        if loaded.typed.tests.iter().any(|t| &t.name == n) || candidates.contains(n) {
            continue;
        }
        let mut d = Diagnostic::unknown_name(n, "", n);
        if loaded.typed.fn_index(n).is_some() {
            d.message = format!("`{n}` is a function, not a test");
            let call = if ctx.is_mcp() { format!("`qal_call` with fn `{n}`") } else { format!("`qal call {n} ...`") };
            d = d.hint(format!("call it with {call}, or write a test: `test {n}_works = {n}(...) == ...`"));
            d = d.hint("a function gets a generated property test if it is pure, has an `ens` or calls a function with `req`, and takes no function values");
        } else if let Some(h) = ctx.renamed_hint(source, n) {
            d = d.hint(h);
        }
        unknown.push(d);
    }
    if !unknown.is_empty() {
        return Reply::diagnostics(unknown);
    }
    let runner = match ctx.runner(&loaded, &Entry::Tests, perms) {
        Ok(r) => r,
        Err(e) => return Reply::error(e),
    };
    let start = Instant::now();
    let mut tests = Vec::new();
    let mut skipped = Vec::new();
    let run_all = names.is_empty();
    for t in loaded.typed.tests.iter().filter(|t| run_all || names.contains(&t.name)) {
        if ctx.test_budget.is_some_and(|b| start.elapsed() > b) {
            skipped.push(t.name.clone());
            continue;
        }
        let mut r = runner.run_test(&format!("test:{}", t.name));
        r.error = r.error.map(capped);
        tests.push(r);
    }
    // Property tests: of every candidate, or of the functions named; with the rest of the time budget.
    let prop_names: Vec<String> = names.iter().filter(|n| candidates.contains(n)).cloned().collect();
    let mut properties = Vec::new();
    if run_all || !prop_names.is_empty() {
        let mut opts = PropOptions { seed, ..PropOptions::default() };
        if let (Some(budget), Some(timeout)) = (ctx.test_budget, ctx.timeout) {
            opts.budget = budget.saturating_sub(start.elapsed());
            opts.timeout = opts.timeout.min(timeout);
        }
        let only = (!run_all).then_some(prop_names.as_slice());
        match qal_prove::check_properties(&loaded.typed, only, &opts) {
            Ok(ps) => properties = ps,
            Err(e) => return Reply::error(e),
        }
    }
    let passed = tests.iter().filter(|t| t.passed).count();
    let failed = tests.len() - passed;
    let failed_properties = properties.iter().filter(|p| p.status == PropStatus::Failed).count();
    for p in &mut properties {
        p.error = p.error.take().map(capped);
    }
    let out = Output {
        tests: Some(tests),
        properties: (!properties.is_empty()).then_some(properties),
        passed: Some(passed),
        failed: Some(failed),
        failed_properties: (failed_properties > 0).then_some(failed_properties),
        skipped: (!skipped.is_empty()).then_some(skipped),
        ..Output::default()
    };
    Reply::new(out, u8::from(failed + failed_properties > 0))
}

/// The function `name` of `loaded`, or why it cannot be called (checked before compiling).
fn find_fn(ctx: &Ctx, source: &Source, loaded: &Loaded, name: &str) -> Result<(), RunError> {
    if loaded.typed.fn_index(name).is_some() {
        return Ok(());
    }
    // A generic function already has its hint.
    let mut e = crate::no_function(&loaded.typed, name);
    if !e.hints.is_empty() {
        return Err(e);
    }
    if loaded.typed.tests.iter().any(|t| t.name == name) {
        e.message = format!("`{name}` is a test, not a function");
        e.hints.push(if ctx.is_mcp() { format!("run it with `qal_test` and names [\"{name}\"]") } else { format!("run it with `qal test {name}`") });
    } else if let Some(h) = ctx.renamed_hint(source, name) {
        e.hints.push(h);
    }
    Err(e)
}

/// Calls any function with arguments converted by its signature.
pub fn call(ctx: &Ctx, source: &Source, perms: Permissions, name: &str, args: CallArgs) -> Reply {
    let loaded = match ctx.load(source) {
        Ok(l) => l,
        Err(d) => return Reply::diagnostics(d),
    };
    if let Err(e) = find_fn(ctx, source, &loaded, name) {
        return Reply::error(e);
    }
    let vals = match args {
        CallArgs::Strings(raw) => parse_args(&loaded.typed, name, raw),
        CallArgs::Json(raw) => json_args(&loaded.typed, name, raw),
    };
    let vals = match vals {
        Ok(v) => v,
        Err(e) => return Reply::error(e),
    };
    let runner = match ctx.runner(&loaded, &Entry::Fn(name.to_string()), perms) {
        Ok(r) => r,
        Err(e) => return Reply::error(e),
    };
    let outcome = runner.invoke(name, &vals);
    let mut out = Output { db: db_label(outcome.db), ..Output::default() };
    ctx.stdout(&mut out, outcome.stdout, outcome.stdout_bytes);
    match outcome.result {
        Ok(v) => {
            out.ok = Some(v);
            Reply::new(out, 0)
        }
        Err(e) => {
            out.error = Some(capped(e));
            Reply::new(out, 1)
        }
    }
}

/// Runs an entry point `fn(List[Str]) -> Int`; the exit code is the program's (1 if outside 0..=255).
pub fn run(ctx: &Ctx, source: &Source, perms: Permissions, name: &str, args: &[String]) -> Reply {
    let loaded = match ctx.load(source) {
        Ok(l) => l,
        Err(d) => return Reply::diagnostics(d),
    };
    if let Err(mut e) = find_fn(ctx, source, &loaded, name) {
        if e.message.starts_with("no function") {
            e.message = format!("no entry point `{name}`");
        }
        return Reply::error(e);
    }
    if !is_entry_point(&loaded.typed, name) {
        let mut e = RunError::new("bad_args", name, format!("`{name}` is not an entry point"));
        e.hints.push("`run` needs a function `fn(List[Str]) -> Int`; use `call` for others".into());
        return Reply::error(e);
    }
    let runner = match ctx.runner(&loaded, &Entry::Fn(name.to_string()), perms) {
        Ok(r) => r,
        Err(e) => return Reply::error(e),
    };
    let outcome = runner.run_entry(name, args);
    let mut out = Output { db: db_label(outcome.db), ..Output::default() };
    ctx.stdout(&mut out, outcome.stdout, outcome.stdout_bytes);
    match outcome.result {
        Ok(Value::Int(code)) => {
            out.exit_code = Some(code);
            Reply::new(out, u8::try_from(code).unwrap_or(1))
        }
        Ok(other) => {
            out.error = Some(RunError::new("trap", name, format!("entry point returned {other:?} instead of Int")));
            Reply::new(out, 1)
        }
        Err(e) => {
            out.error = Some(capped(e));
            Reply::new(out, 1)
        }
    }
}

/// Compiles the stored function `name` into `<project>/.qal/build/<name>.wasm` plus
/// `<name>.manifest.json` (required effects, imports, hash).
pub fn build(ctx: &Ctx, name: &str) -> Reply {
    let mut store = match ctx.open() {
        Ok(s) => s,
        Err(d) => return Reply::diagnostics(d),
    };
    let loaded = match store.program().and_then(load_program) {
        Ok(l) => l,
        Err(d) => return Reply::diagnostics(d),
    };
    if let Err(e) = find_fn(ctx, &Source::Store, &loaded, name) {
        return Reply::error(e);
    }
    let entry = match store.query(&Query { name: Some(name.to_string()), ..Query::default() }) {
        Ok(rs) => rs.into_iter().find(|e| e.name == name),
        Err(d) => return Reply::diagnostics(d),
    };
    let Some(entry) = entry else {
        return Reply::diagnostics(vec![Diagnostic::store_error(format!("`{name}` is in the codebase but has no stored entry"))]);
    };
    let module = match qal_wasm::compile(&loaded.typed, &Entry::Fn(name.to_string())) {
        Ok(m) => m,
        Err(m) => return Reply::error(RunError::new("compile_error", name, format!("the program cannot be compiled to WebAssembly: {m}"))),
    };
    let Some(meta) = module.func(name) else {
        return Reply::error(RunError::new("compile_error", name, format!("`{name}` is missing from its own module")));
    };
    // .qal/store.db -> the project root
    let Some(project) = store.path().parent().and_then(Path::parent) else {
        return Reply::diagnostics(vec![Diagnostic::store_error("the store has no project directory")]);
    };
    let dir = project.join(qal_store::STORE_DIR).join("build");
    let manifest = Manifest {
        format: "qal-build/1",
        name,
        hash: &entry.hash,
        signature: entry.signature.clone(),
        entry_point: is_entry_point(&loaded.typed, name),
        effects: &meta.effects,
        imports: &module.imports,
        exports: module.funcs.iter().filter(|f| !f.is_test).map(|f| f.export.clone()).collect(),
        wasm_sha256: Hash::of(&module.wasm).to_hex(),
        bytes: module.wasm.len(),
    };
    let json = match serde_json::to_string_pretty(&manifest) {
        Ok(j) => j + "\n",
        Err(e) => return Reply::diagnostics(vec![Diagnostic::store_error(format!("cannot write the manifest: {e}"))]),
    };
    let (wasm_rel, manifest_rel) = (format!(".qal/build/{name}.wasm"), format!(".qal/build/{name}.manifest.json"));
    let written = std::fs::create_dir_all(&dir)
        .and_then(|_| write_atomic(&project.join(&wasm_rel), &module.wasm))
        .and_then(|_| write_atomic(&project.join(&manifest_rel), json.as_bytes()));
    if let Err(e) = written {
        return Reply::diagnostics(vec![Diagnostic::store_error(format!("cannot write {}: {e}", dir.display()))]);
    }
    let built = Built { name: name.to_string(), hash: entry.hash, wasm: wasm_rel, manifest: manifest_rel, effects: meta.effects.clone(), bytes: module.wasm.len() };
    Reply::new(Output { built: Some(built), ..Output::default() }, 0)
}

/// The `db` field of a result: absent when no database was used.
fn db_label(db: &'static str) -> Option<&'static str> {
    (db != "none").then_some(db)
}

/// What `export` wrote.
#[derive(Debug, Clone, Serialize)]
pub struct Exported {
    pub package: String,
    pub hash: String,
    pub defs: usize,
    pub file: String,
}

/// Exports the names starting with `prefix` (and their dependencies) to the package file `out`;
/// the package is named `name`, by default after the file (`std/std.qalpkg` -> `std`).
pub fn export(ctx: &Ctx, prefix: &str, out: &Path, name: Option<&str>) -> Reply {
    let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let package = name.unwrap_or(stem);
    let exported = match ctx.open().and_then(|mut s| s.export(prefix, package)) {
        Ok(e) => e,
        Err(d) => return Reply::diagnostics(d),
    };
    if let Err(e) = out.parent().filter(|p| !p.as_os_str().is_empty()).map_or(Ok(()), std::fs::create_dir_all).and_then(|_| write_atomic(out, exported.file.as_bytes())) {
        return Reply::diagnostics(vec![Diagnostic::store_error(format!("cannot write {}: {e}", out.display()))]);
    }
    let e = Exported { package: exported.package, hash: exported.hash, defs: exported.defs, file: out.display().to_string() };
    Reply::new(Output { exported: Some(e), ..Output::default() }, 0)
}

/// Imports a package file into the store (atomically; every hash is checked and the contracts are
/// proven as by `define`: a refuted one rejects the import).
pub fn import(ctx: &Ctx, file: &str) -> Reply {
    let (mut store, std_imported) = match open_for_write(ctx, true) {
        Ok(s) => s,
        Err(d) => return Reply::diagnostics(d),
    };
    let (result, reports) = proving(|verify| store.import_verified(file, verify));
    match result {
        Ok(i) => {
            let reports = current_reports(&i.defined, reports);
            let mut out = Output { imported: Some(i), std_imported, ..Output::default() };
            out.set_proofs(reports);
            Reply::new(out, 0)
        }
        Err(d) => Reply::diagnostics(d),
    }
}

/// Runs requests through the service pipeline of `qal serve` without TCP: same limits, same
/// transaction rule, same response checks. `init` runs first; all requests share one database
/// (`db:mem`: a fresh one for this call).
pub fn request(ctx: &Ctx, source: &Source, mut perms: Permissions, handler: &str, init: Option<&str>, requests: &[RequestSpec]) -> Reply {
    if requests.len() > MAX_REQUESTS {
        let d = Diagnostic::new("E0701", "bad_arguments", "", "", format!("{} requests; at most {MAX_REQUESTS} per call", requests.len()));
        return Reply::diagnostics(vec![d]);
    }
    let loaded = match ctx.load(source) {
        Ok(l) => l,
        Err(d) => return Reply::diagnostics(d),
    };
    if let (Some(_), None, StoreAt::Discover(dir)) = (&perms.db, &perms.db_dir, &ctx.store) {
        if let Some(project) = qal_store::project_of(dir) {
            perms = perms.with_db_dir(project.join(qal_store::STORE_DIR).join("db"));
        }
    }
    let cfg = qal_serve::ServiceConfig { entry: handler.to_string(), init: init.map(str::to_string), slots: 2, max_out: ctx.max_stdout.unwrap_or(64 << 10), ..qal_serve::ServiceConfig::default() };
    let limits = qal_serve::HttpLimits::default();
    // Guest code needs a large native stack (the CLI's main thread has a small one).
    let run = || -> Result<Vec<RequestResult>, RunError> {
        let service = qal_serve::Service::new(&loaded.typed, perms, cfg)?;
        let mut worker = service.worker()?;
        service.init(&mut worker)?;
        Ok(requests
            .iter()
            .map(|spec| {
                let id = service.next_id();
                let req = &spec.request;
                if req.body.len() > limits.max_body {
                    let e = RunError::new("request_rejected", handler, format!("413 payload_too_large: the body has {} bytes, more than the limit of {}", req.body.len(), limits.max_body));
                    let resp = qal_serve::http::Rejection { status: 413, kind: "payload_too_large", message: e.message.clone() }.response(id);
                    return result_of(resp, "none", None, Some(e));
                }
                let mut handler_req = req.clone();
                if req.method == "HEAD" {
                    handler_req.method = "GET".into();
                }
                let h = service.handle(&mut worker, &handler_req, id, 0);
                let mut resp = h.response;
                if req.method == "HEAD" {
                    resp.body = String::new();
                }
                result_of(resp, h.log.db, h.log.out, h.log.error)
            })
            .collect())
    };
    let outcome = std::thread::scope(|s| match std::thread::Builder::new().name("qal-request".into()).stack_size(qal_run::GUEST_THREAD_STACK).spawn_scoped(s, run) {
        Ok(t) => t.join().unwrap_or_else(|_| Err(RunError::new("internal", handler, "the runtime panicked while running the requests"))),
        Err(e) => Err(RunError::new("trap", handler, format!("cannot start the execution thread: {e}"))),
    });
    match outcome {
        Ok(responses) => Reply::new(Output { responses: Some(responses), ..Output::default() }, 0),
        Err(e) => Reply::error(e),
    }
}

fn result_of(resp: qal_serve::Response, db: &'static str, stdout: Option<String>, error: Option<RunError>) -> RequestResult {
    let is_json = resp.header("content-type").is_some_and(|t| t.trim_start().to_ascii_lowercase().starts_with("application/json"));
    let json = if is_json { serde_json::from_str(&resp.body).ok() } else { None };
    let headers = resp.headers.iter().map(|(n, v)| Value::Record("HttpHeader".into(), vec![("name".into(), Value::Str(n.clone())), ("value".into(), Value::Str(v.clone()))])).collect();
    RequestResult { status: resp.status, headers, body: resp.body, json, db, stdout, error: error.map(capped) }
}

/// Largest JSON of the values of one runtime error; larger values are dropped (with a hint), so a
/// single huge contract argument cannot crowd out the rest of a report.
pub const MAX_ERROR_VALUES_BYTES: usize = 64 << 10;

fn capped(mut e: RunError) -> RunError {
    let size = serde_json::to_string(&e.values).map_or(usize::MAX, |s| s.len());
    if size > MAX_ERROR_VALUES_BYTES {
        e.values = qal_run::Values::default();
        e.hints.push(format!("the values involved ({size} bytes of JSON) are too large to show"));
    }
    e
}

/// Writes `bytes` to a temporary file next to `path`, then renames it into place.
fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}
