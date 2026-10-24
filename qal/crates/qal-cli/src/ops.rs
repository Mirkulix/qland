//! The operations of spec §9, shared by the command line and the MCP server: each returns one
//! [`Reply`] (a JSON object plus the CLI exit code), never panics, and reports every problem as
//! data (`diagnostics` for compile-time and store problems, `error` for runtime failures).

use crate::{is_entry_point, json_args, load, load_program, parse_args, Loaded};
use qal_core::canon::Hash;
use qal_core::diag::Diagnostic;
use qal_core::tast::{TExpr, TKind, TProgram};
use qal_prove::{PropOptions, PropReport, PropStatus, Report, Status};
use qal_run::{Permissions, RunError, TestResult, Value};
use qal_store::{CorrectionInput, Defined, Deleted, Entry as QueryEntry, HistoryEntry, Query, Recall, RecallQuery, Renamed, Store, Verify, View};
use qal_compiler::lexer::is_std_name;
use qal_compiler::Entry;
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
    /// The entries of `query`, or the cases of `run` with `cases`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results: Option<Results>,
    /// The earlier fixes and matching definitions of `recall`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall: Option<Recall>,
    /// How many corrections `corrections import` stored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub corrections_imported: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub renamed: Option<Renamed>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deleted: Option<Vec<Deleted>>,
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
    /// Why this write could not import the standard library of this qal first (it went ahead).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub std_warnings: Option<Vec<Diagnostic>>,
    /// The results of `request`, one per request, in order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub responses: Option<Vec<RequestResult>>,
    /// Fields of the team commands and of team mode (`notices`, `space`, `qal verify`'s result, ...).
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
    /// The operation failed without diagnostics (a failed `qal verify`, a case of `run` with
    /// `cases` that ended in a runtime error).
    #[serde(skip)]
    pub failure: bool,
    /// Failed `tests` do not fail the operation (`define` reports the tests of what it defined).
    #[serde(skip)]
    pub tests_informative: bool,
}

/// The `results` of an operation.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Results {
    Query(Vec<QueryEntry>),
    Run(Vec<RunResult>),
}

/// One case of `run` with `cases`: the program's arguments and its input (default: none).
#[derive(Debug, Clone, PartialEq)]
pub struct RunCase {
    pub args: Vec<String>,
    pub stdin: Option<String>,
}

/// Most cases per `run` call.
pub const MAX_CASES: usize = 64;

/// The result of one [`RunCase`]: its arguments, then what `run` reports for it. The input is not
/// echoed (replies stay small); the position in `results` identifies the case.
#[derive(Debug, Clone, Serialize)]
pub struct RunResult {
    pub args: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stdout: Option<String>,
    /// Total size in bytes of a captured stdout that was cut to `Ctx::max_stdout`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stdout_truncated: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RunError>,
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
    pub(crate) fn new(out: Output, code: u8) -> Reply {
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
    /// that runs and returns a non-zero exit code did not fail, but a case of `run` with `cases`
    /// that does fails the call; the failed tests of a `define` do not fail it).
    pub fn is_error(&self) -> bool {
        let failed_tests = !self.out.tests_informative && self.out.failed.is_some_and(|n| n > 0);
        self.out.failure || self.out.diagnostics.is_some() || self.out.error.is_some() || failed_tests || self.out.failed_properties.is_some() || self.out.refuted.is_some_and(|n| n > 0)
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
    /// Team mode: who signs the records of writes (an agent's session over MCP).
    pub signer: Option<std::sync::Arc<qal_store::team::Signer>>,
    /// Team mode: the space the operations work in (`main` or a change set); `build` writes to
    /// `.qal/build/<space>/`.
    pub space: Option<String>,
    /// Team mode: the agent's session (what it views is recorded as seen, §5.3).
    pub session: Option<String>,
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
        Ok(Ctx { store: StoreAt::Discover(cwd), stdin: None, timeout: None, test_budget: None, max_stdout: None, signer: None, space: None, session: None })
    }

    /// The store; `create` makes it under `Root` if it does not exist yet (only `define` does,
    /// so reading never leaves a store behind).
    pub fn open_store(&self, create: bool) -> Result<Store, Vec<Diagnostic>> {
        let mut store = self.open_any(create)?;
        if let Some(space) = &self.space {
            store.set_space(qal_store::Space::parse(space));
        }
        Ok(store)
    }

    fn open_any(&self, create: bool) -> Result<Store, Vec<Diagnostic>> {
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
        let (stdout, truncated) = self.cut_stdout(stdout, total);
        out.stdout = Some(stdout);
        out.stdout_truncated = truncated;
    }

    /// `stdout` cut to `max_stdout` at a character boundary, and the total size if it was cut.
    fn cut_stdout(&self, stdout: String, total: usize) -> (String, Option<usize>) {
        let Some(max) = self.max_stdout else {
            return (stdout, None);
        };
        let mut end = stdout.len().min(max);
        while !stdout.is_char_boundary(end) {
            end -= 1;
        }
        let truncated = (end < stdout.len() || total > max).then(|| total.max(stdout.len()));
        (stdout[..end].to_string(), truncated)
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
            s.set_release(crate::STD_PACKAGE)?;
            let fresh = s.query(&Query::default())?.is_empty() && s.meta(qal_store::package::META_STD_PACKAGE)?.is_none();
            let imported = if fresh { Some(proving(|v| s.import_verified(crate::STD_PACKAGE, v)).0?.hash) } else { None };
            Ok((s, StdImport { imported, warnings: None }))
        })
    } else {
        Store::init(dir).map(|mut s| {
            let std = with_std(&mut s);
            (s, std)
        })
    };
    match opened {
        Ok((s, std)) => {
            let mut out = Output { ok: Some(Value::Bool(true)), store: Some(s.path().display().to_string()), ..Output::default() };
            std.report(&mut out);
            Reply::new(out, 0)
        }
        Err(d) => Reply::diagnostics(d),
    }
}

/// What the automatic std import before a write did.
#[derive(Debug, Default)]
pub(crate) struct StdImport {
    /// The package hash, if it imported.
    imported: Option<String>,
    /// Why it could not import (the project keeps its previous std).
    warnings: Option<Vec<Diagnostic>>,
}

impl StdImport {
    pub(crate) fn report(self, out: &mut Output) {
        out.std_imported = self.imported;
        out.std_warnings = self.warnings;
    }

    /// A failed write: the std diagnostics follow the write's own (they may be the reason).
    fn failed(self, mut d: Vec<Diagnostic>) -> Reply {
        d.extend(self.warnings.unwrap_or_default());
        Reply::diagnostics(d)
    }
}

/// Imports the embedded standard library into the project's store if it does not have this
/// version yet (spec Phase 3 §4.3): before every write, in its own transaction, never in the std
/// project's store. Its contracts are proven as by any import; the proof reports are not part of
/// the write's result. A failed import never blocks the project (review finding C1): it changed
/// nothing, the project keeps its previous std, and the write goes ahead with a warning.
/// A team store never imports automatically (`qal team sync-std` does, signed).
pub(crate) fn with_std(store: &mut Store) -> StdImport {
    let (result, _) = proving(|v| store.ensure_std(crate::STD_PACKAGE, Some(v)));
    match result {
        Ok(imported) => StdImport { imported: imported.map(|i| i.hash), warnings: None },
        Err(mut d) => {
            let kept = store.meta(qal_store::package::META_STD_PACKAGE).ok().flatten().map_or("none".to_string(), |h| h.chars().take(12).collect());
            for x in &mut d {
                x.hints.push(format!("the standard library of this qal could not be imported, so the project keeps its previous one (std {kept}); this write went ahead; fix the cause above and any later write imports it"));
            }
            StdImport { imported: None, warnings: Some(d) }
        }
    }
}

/// The store for a write (`create`: make it if missing), with the standard library imported
/// (or a warning why not).
/// The writes of a team store are signed by `ctx.signer` (refused without one).
fn open_for_write(ctx: &Ctx, create: bool) -> Result<(Store, StdImport), Vec<Diagnostic>> {
    let mut store = ctx.open_store(create)?;
    let std = with_std(&mut store);
    store.set_signer(ctx.signer.clone());
    Ok((store, std))
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
/// `breaking` (CLI `--breaking`, std store only): released std names may change their signature.
pub fn define(ctx: &Ctx, src: &str, agent: &str, breaking: bool) -> Reply {
    let (mut store, std) = match open_for_write(ctx, true) {
        Ok(s) => s,
        Err(d) => return Reply::diagnostics(d),
    };
    store.set_breaking(breaking);
    let (result, reports) = proving(|verify| store.define_verified(src, agent, verify));
    match result {
        Ok(d) => {
            let reports = current_reports(&d, reports);
            let tests = tests_of(&mut store, &d);
            let fns: Vec<String> = d.iter().filter(|x| x.kind == "fn" && x.status != "relinked" && x.status != "unchanged").map(|x| x.name.clone()).collect();
            let mut out = Output { defined: Some(d), ..Output::default() };
            out.set_proofs(reports);
            std.report(&mut out);
            team_report(store.last_write(), &mut out);
            drop(store);
            run_tests_of_define(ctx, &tests, &mut out);
            run_properties_of_define(ctx, &fns, &mut out);
            Reply::new(out, 0)
        }
        Err(d) => std.failed(d),
    }
}

/// The stored tests that refer directly to a function this `define` wrote (as `query` with
/// `uses` finds them), and the tests it wrote itself; a dependent it only relinked does not count.
fn tests_of(store: &mut Store, defined: &[Defined]) -> Vec<String> {
    let mut tests: Vec<String> = Vec::new();
    for t in defined.iter().filter(|d| d.kind == "test" && d.status != "relinked") {
        if !tests.contains(&t.name) {
            tests.push(t.name.clone());
        }
    }
    for f in defined.iter().filter(|d| d.kind == "fn" && d.status != "relinked") {
        let Ok(users) = store.query(&Query { uses: Some(f.name.clone()), ..Query::default() }) else { continue };
        for u in users.into_iter().filter(|u| u.kind == "test") {
            if !tests.contains(&u.name) {
                tests.push(u.name);
            }
        }
    }
    tests
}

/// Runs `tests` as `test` does (same budgets, no property tests) and reports them in `out`.
/// Their failures do not undo or fail the `define`: the definitions are stored either way. If the
/// tests cannot run at all, `qal_test` reports why; the `define` stays as it is.
fn run_tests_of_define(ctx: &Ctx, tests: &[String], out: &mut Output) {
    if tests.is_empty() {
        return;
    }
    let r = test(ctx, &Source::Store, Permissions::default(), tests, 0);
    if r.out.tests.is_none() {
        return;
    }
    out.tests = r.out.tests;
    out.passed = r.out.passed;
    out.failed = r.out.failed;
    out.skipped = r.out.skipped;
    out.tests_informative = true;
}

/// Runs the generated property tests of the functions a `define` wrote (those that have one: pure,
/// with an `ens` or calling a function with `req`), as `test` does with their names. A failed
/// property is the warning E0803, runs that trapped (overflow, division by zero, ...) on inputs that
/// satisfy `req` are the warning E0804: the definitions stay stored and the reply is no error. Without
/// this, a contract proven under trapping arithmetic and a property nobody runs let an overflow
/// through (living program, pressure run of 2026-10-24).
fn run_properties_of_define(ctx: &Ctx, fns: &[String], out: &mut Output) {
    if fns.is_empty() {
        return;
    }
    let Ok(loaded) = ctx.load(&Source::Store) else { return };
    unbounded_products(&loaded.typed, fns, out);
    let names: Vec<String> = qal_prove::props::candidates(&loaded.typed).into_iter().map(|i| loaded.typed.fns[i].def.name.clone()).filter(|n| fns.contains(n)).collect();
    if names.is_empty() {
        return;
    }
    let mut opts = PropOptions::default();
    if let Some(t) = ctx.timeout {
        opts.timeout = opts.timeout.min(t);
    }
    let Ok(mut props) = qal_prove::check_properties(&loaded.typed, Some(&names), &opts) else { return };
    let warnings = out.warnings.get_or_insert_with(Vec::new);
    for p in &mut props {
        p.error = p.error.take().map(capped);
        p.trap = p.trap.take().map(capped);
        if p.status == PropStatus::Failed {
            let why = p.error.as_ref().map_or_else(String::new, |e| format!(": {}", e.message));
            warnings.push(Diagnostic::new("E0803", "property_failed", &p.def, "", format!("the property test of `{}` failed on a generated input{why}", p.def)).hint("the input is in `properties` (counterexample); fix the body or the contract"));
        }
        if p.traps > 0 {
            let what = p.trap.as_ref().map_or_else(|| "a trap".to_string(), |t| t.message.clone());
            let values = p.trap.as_ref().and_then(|t| serde_json::to_string(&t.values).ok()).unwrap_or_default();
            warnings.push(
                Diagnostic::new("E0804", "property_traps", &p.def, "", format!("`{}` trapped in {} of {} generated runs: {what} {values}", p.def, p.traps, p.runs + p.traps))
                    .hint(format!("if such inputs cannot occur, say so with a `req` on `{}` that bounds them (e.g. `req x >= 0 && x <= 1000000000`); otherwise compute so that it cannot trap", p.def)),
            );
        }
    }
    if warnings.is_empty() {
        out.warnings = None;
    }
    let existing = out.properties.get_or_insert_with(Vec::new);
    existing.extend(props);
}

/// The warning E0805 for a function that multiplies `Int` parameters no `req` mentions: a product of
/// unbounded values overflows 64 bits somewhere, and only a `req` can say where the domain ends (the
/// living program's pressure run: `price * qty * 95` with amounts of 9 digits). Read from the printed
/// body and contracts; a parameter counts as bounded when a `req` names it. A product that sits
/// in a fused `(a * b * …) / c` does not count: it is exact and only the quotient can overflow.
fn unbounded_products(tp: &TProgram, fns: &[String], out: &mut Output) {
    for f in tp.fns.iter().filter(|f| fns.contains(&f.def.name)) {
        let def = &f.def;
        if !has_plain_product(&f.body) {
            continue;
        }
        let body = qal_compiler::print_expr(&def.body);
        let reqs: Vec<String> = def.req.iter().map(qal_compiler::print_expr).collect();
        let mentions = |text: &str, name: &str| text.split(|c: char| !(c.is_alphanumeric() || c == '_')).any(|w| w == name);
        let unbounded: Vec<&str> = def.params.iter().filter(|p| p.ty == qal_core::ast::Type::Int && mentions(&body, &p.name) && !reqs.iter().any(|r| mentions(r, &p.name))).map(|p| p.name.as_str()).collect();
        if unbounded.is_empty() {
            continue;
        }
        let list = unbounded.iter().map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", ");
        let example = unbounded.iter().map(|n| format!("{n} >= 0 && {n} <= 1000000000")).collect::<Vec<_>>().join(" && ");
        out.warnings.get_or_insert_with(Vec::new).push(
            Diagnostic::new("E0805", "unbounded_product", &def.name, "", format!("`{}` multiplies Int values that depend on {list}, which no `req` bounds: some inputs overflow 64 bits", def.name))
                .hint(format!("state the domain, e.g. `req {example}`, with the task's real limits; the property test then checks every input inside it")),
        );
    }
}

/// True if `e` holds an Int `*` outside a fused `(a * b * …) / c` (whose factors may hold one).
fn has_plain_product(e: &TExpr) -> bool {
    if let TKind::Bin(op, l, r, _) = &e.kind {
        if let Some(factors) = TExpr::fused_factors(*op, l) {
            return factors.into_iter().chain([&**r]).any(has_plain_product);
        }
        if *op == qal_core::ast::BinOp::Mul && e.ty == qal_core::ast::Type::Int {
            return true;
        }
    }
    e.children().into_iter().any(has_plain_product)
}

/// What a write reports besides its result (Phase 4 §10.2): the leases it took (team mode) and the
/// stubs it defined, each stub also as the warning E0930 `stub_present`.
fn team_report(w: &qal_store::team::WriteReport, out: &mut Output) {
    if !w.leases.is_empty() {
        out.extra.insert("leases".into(), serde_json::to_value(&w.leases).unwrap_or(serde_json::Value::Null));
    }
    if w.stubs.is_empty() {
        return;
    }
    out.extra.insert("stubs".into(), serde_json::Value::from(w.stubs.clone()));
    let warnings = out.warnings.get_or_insert_with(Vec::new);
    // One E0930 per stub stands for all its unproven obligations (E0802 "this definition is a stub").
    warnings.retain(|d| !(d.code == "E0802" && w.stubs.contains(&d.def)));
    for n in &w.stubs {
        warnings.push(Diagnostic::new("E0930", "stub_present", n, "body", format!("`{n}` is a stub: calling it reaches `todo()` (runtime error todo_reached)")).hint(format!("replace the `todo()` in `{n}` with its implementation (keep the signature, and its users relink)")));
    }
}

/// Runs a store write with the verification step of `define` (the contracts of what it defines
/// are proven; a refuted one rejects the write) and returns its result with the proof reports.
/// Each report comes with the hash it was proven for (see [`current_reports`]).
pub(crate) fn proving<T>(write: impl FnOnce(&mut Verify) -> Result<T, Vec<Diagnostic>>) -> (Result<T, Vec<Diagnostic>>, Vec<(String, Report)>) {
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

/// Runs a merge (`write`) with its check (Phase 4 §7.2): the contracts of every function and test
/// whose hash is new to main are proven afresh (never from the `proofs` cache, 4a review N4) and
/// every test of the project runs (std's are std's own), within `budget` (30 s over MCP). A refuted
/// contract, a failed test or a used-up budget rejects the merge with E0912.
pub(crate) fn merging<T>(change: &str, budget: Duration, timeout: Option<Duration>, write: impl FnOnce(&mut qal_store::team::merge::MergeCheck) -> Result<T, Vec<Diagnostic>>) -> Result<T, Vec<Diagnostic>> {
    use qal_store::team::merge::{tests_failed, MergeChecked};
    let opts = qal_prove::Options::default();
    let mut run = |program: &qal_core::ast::Program, tp: &TProgram, changed: &[(String, String)]| -> Result<MergeChecked, Vec<Diagnostic>> {
        let start = Instant::now();
        let names: Vec<String> = changed.iter().map(|(n, _)| n.clone()).collect();
        let proofs = if names.is_empty() { Vec::new() } else { qal_prove::prove_defs(tp, Some(&names), &opts) };
        let is_test = |n: &str| tp.tests.iter().any(|t| t.name == n);
        let refuted: Vec<Diagnostic> = proofs.iter().flat_map(|p| &p.reports).filter(|r| r.status == Status::Refuted && !is_test(&r.def)).filter_map(Report::diagnostic).collect();
        if !refuted.is_empty() {
            let defs: std::collections::BTreeSet<&str> = refuted.iter().map(|d| d.def.as_str()).collect();
            let mut out = vec![tests_failed(change, format!("{} contract obligation(s) are refuted on main as it would be", refuted.len()), serde_json::json!({"refuted": defs}))];
            out.extend(listed(refuted).0);
            return Err(out);
        }
        let count = |st: Status| proofs.iter().flat_map(|p| &p.reports).filter(|r| r.status == st).count();
        let (proven, unknown) = (count(Status::Proven), count(Status::Unknown));
        let rows: Vec<(String, String)> = changed.iter().filter_map(|(n, h)| proofs.iter().find(|p| p.def == *n)?.record.as_ref().map(|r| (h.clone(), record_json(r)))).collect();
        // Every test of the new main view (the project's own; std tests itself).
        let loaded = Loaded { program: program.clone(), typed: tp.clone() };
        let mut runner = loaded.runner(&Entry::Tests, Permissions::default()).map_err(|e| vec![tests_failed(change, format!("the tests cannot run: {}", e.message), serde_json::json!({"error": e.kind}))])?;
        if let Some(t) = timeout {
            runner = runner.with_timeout(t);
        }
        let mut failed = Vec::new();
        let mut skipped = Vec::new();
        let mut passed = 0usize;
        for t in tp.tests.iter().filter(|t| !is_std_name(&t.name)) {
            if start.elapsed() > budget {
                skipped.push(t.name.clone());
                continue;
            }
            let r = runner.run_test(&format!("test:{}", t.name));
            if r.passed {
                passed += 1;
            } else {
                failed.push(serde_json::json!({"name": r.name, "error": r.error.map(|e| e.message)}));
            }
        }
        if !failed.is_empty() || !skipped.is_empty() {
            let what = if failed.is_empty() { format!("the test budget of {} s was used up before {} test(s) ran", budget.as_secs(), skipped.len()) } else { format!("{} test(s) fail on main as it would be", failed.len()) };
            return Err(vec![tests_failed(change, what, serde_json::json!({"failed": failed, "skipped": skipped, "passed": passed}))]);
        }
        let mut report = serde_json::Map::new();
        report.insert("tests".into(), serde_json::json!({"passed": passed}));
        report.insert("proofs".into(), serde_json::json!({"proven": proven, "unknown": unknown, "fresh": true}));
        Ok(MergeChecked { proofs: rows, report })
    };
    let mut check = qal_store::team::merge::MergeCheck { key: opts.key(), run: &mut run };
    write(&mut check)
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
    // In a project, "all" means the project's own definitions: std's are proven when named (or in
    // the std store), like `test`.
    let own_only = names.is_empty()
        && match source {
            Source::Store => match ctx.open().and_then(|s| s.is_std()) {
                Ok(is_std) => !is_std,
                Err(d) => return Reply::diagnostics(d),
            },
            Source::Text(_) => false,
        };
    let mut own: Vec<String> = Vec::new();
    if own_only {
        let fns = tp.fns.iter().filter(|f| f.instance.is_none()).map(|f| f.def.name.clone());
        own = fns.chain(tp.generics.iter().map(|f| f.def.name.clone())).chain(tp.tests.iter().map(|t| t.name.clone())).filter(|n| !is_std_name(n)).collect();
        own.sort();
        own.dedup();
    }
    let only = if own_only { Some(own.as_slice()) } else { (!names.is_empty()).then_some(names) };
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
    // In a team store `qal_prove` is a read for every role (spec §5.1): the proof cache is filled
    // only if the write lock is free at once, never waiting for (or failing on) a writer.
    let team = store.is_team()?;
    if !missing.is_empty() {
        let left = qal_prove::Options { budget: opts.budget.saturating_sub(start.elapsed()), ..opts.clone() };
        for p in qal_prove::prove_defs(tp, Some(&missing), &left) {
            if let (Some(h), Some(rec)) = (hashes.get(&p.def), &p.record) {
                if team {
                    store.try_save_proofs(h, &key, &record_json(rec));
                } else {
                    store.save_proofs(h, &key, &record_json(rec))?;
                }
            }
            cached.insert(p.def, p.reports);
        }
    }
    Ok(targets.iter().flat_map(|n| cached.remove(n).unwrap_or_default()).collect())
}

struct CountingWriter(usize);

impl std::io::Write for CountingWriter {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0 = self.0.saturating_add(buf.len());
        Ok(buf.len())
    }

    #[inline]
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Shows the named definitions. Repeated names count once; the views are built one by one under a
/// byte budget ([`crate::mcp::MAX_RESULT_BYTES`]), so a request can never make this build more than
/// it may return.
pub fn view(ctx: &Ctx, names: &[String]) -> Reply {
    let mut store = match ctx.open() {
        Ok(s) => s,
        Err(d) => return Reply::diagnostics(d),
    };
    let here = store.space().clone();
    let mut seen = std::collections::HashSet::new();
    let (mut views, mut diags, mut bytes) = (Vec::new(), Vec::new(), 0usize);
    for n in names.iter().filter(|n| seen.insert(n.as_str())) {
        // Team mode: `name@main` or `name@<change>` shows the name in another space.
        let (name, space) = match n.rsplit_once('@').filter(|_| ctx.space.is_some() && !n.starts_with('#')) {
            Some((name, sp)) => (name.to_string(), Some(qal_store::Space::parse(sp))),
            None => (n.clone(), None),
        };
        store.set_space(space.clone().unwrap_or_else(|| here.clone()));
        match store.view(std::slice::from_ref(&name)) {
            Ok(mut v) => {
                if let Some(session) = &ctx.session {
                    let pairs: Vec<(String, String)> = v.iter().filter(|_| !name.starts_with('#')).map(|x| (x.name.clone(), x.hash.clone())).collect();
                    // Best effort: a failure only means a later write may be refused as stale (E0904).
                    let _ = store.mark_seen(session, &store.space().clone(), &pairs);
                }
                if let Some(sp) = &space {
                    for x in &mut v {
                        x.space = Some(sp.id().to_string());
                    }
                }
                bytes += v.iter().map(|item| {
                    let mut count = CountingWriter(0);
                    serde_json::to_writer(&mut count, item).map_or(usize::MAX, |_| count.0)
                }).fold(0, usize::saturating_add);
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
        Ok(r) => Reply::new(Output { results: Some(Results::Query(r)), ..Output::default() }, 0),
        Err(d) => Reply::diagnostics(d),
    }
}

/// The experience memory: earlier corrections of a diagnostic code and definitions by name or signature.
pub fn recall(ctx: &Ctx, q: &RecallQuery) -> Reply {
    match ctx.open().and_then(|mut s| s.recall(q)) {
        Ok(r) => Reply::new(Output { recall: Some(r), ..Output::default() }, 0),
        Err(d) => Reply::diagnostics(d),
    }
}

/// Imports corrections (a JSON list of `{code, path?, message?, before, after}`, the seed corpus of
/// `bench/corrections.py`) into the memory of the store, all or nothing.
pub fn import_corrections(ctx: &Ctx, text: &str) -> Reply {
    let records: Vec<CorrectionInput> = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => {
            return Reply::diagnostics(vec![Diagnostic::new("E0701", "bad_arguments", "", "", format!("not a list of correction records: {e}"))
                .hint("each record is {\"code\": \"E0201\", \"before\": \"...\", \"after\": \"...\"} (path and message are optional)")])
        }
    };
    match ctx.open().and_then(|mut s| s.import_corrections(&records)) {
        Ok(n) => Reply::new(Output { corrections_imported: Some(n), ..Output::default() }, 0),
        Err(d) => Reply::diagnostics(d),
    }
}

/// `breaking` (CLI `--breaking`, std store only): released std names may be renamed.
pub fn rename(ctx: &Ctx, old: &str, new: &str, agent: &str, breaking: bool) -> Reply {
    let (mut store, std) = match open_for_write(ctx, false) {
        Ok(s) => s,
        Err(d) => return Reply::diagnostics(d),
    };
    store.set_breaking(breaking);
    match store.rename(old, new, agent) {
        Ok(r) => {
            let mut out = Output { renamed: Some(r), ..Output::default() };
            std.report(&mut out);
            Reply::new(out, 0)
        }
        Err(d) => std.failed(d),
    }
}

/// Removes names nothing else refers to (all or nothing).
/// `breaking` (CLI `--breaking`, std store only): released std names may be deleted.
pub fn delete(ctx: &Ctx, names: &[String], agent: &str, breaking: bool) -> Reply {
    let (mut store, std) = match open_for_write(ctx, false) {
        Ok(s) => s,
        Err(d) => return Reply::diagnostics(d),
    };
    store.set_breaking(breaking);
    match store.delete(names, agent) {
        Ok(d) => {
            let mut out = Output { deleted: Some(d), ..Output::default() };
            std.report(&mut out);
            Reply::new(out, 0)
        }
        Err(d) => std.failed(d),
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
    // In a project, "all" means the project's own tests: std's run when named (or in the std store).
    let own_only = run_all
        && match source {
            Source::Store => match ctx.open().and_then(|s| s.is_std()) {
                Ok(is_std) => !is_std,
                Err(d) => return Reply::diagnostics(d),
            },
            Source::Text(_) => false,
        };
    let selected = |n: &str| if run_all { !(own_only && is_std_name(n)) } else { names.iter().any(|m| m == n) };
    for t in loaded.typed.tests.iter().filter(|t| selected(&t.name)) {
        if ctx.test_budget.is_some_and(|b| start.elapsed() > b) {
            skipped.push(t.name.clone());
            continue;
        }
        let mut r = runner.run_test(&format!("test:{}", t.name));
        r.error = r.error.map(capped);
        tests.push(r);
    }
    // Property tests: of every candidate, or of the functions named; with the rest of the time budget.
    let prop_names: Vec<String> = if own_only { candidates.iter().filter(|n| !is_std_name(n)).cloned().collect() } else { names.iter().filter(|n| candidates.contains(n)).cloned().collect() };
    let mut properties = Vec::new();
    if (run_all && !own_only) || !prop_names.is_empty() {
        let mut opts = PropOptions { seed, ..PropOptions::default() };
        if let (Some(budget), Some(timeout)) = (ctx.test_budget, ctx.timeout) {
            opts.budget = budget.saturating_sub(start.elapsed());
            opts.timeout = opts.timeout.min(timeout);
        }
        let only = (!run_all || own_only).then_some(prop_names.as_slice());
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
    let runner = match entry_runner(ctx, source, perms, name) {
        Ok(r) => r,
        Err(reply) => return reply,
    };
    let outcome = runner.run_entry(name, args);
    let mut out = Output { db: db_label(outcome.db), ..Output::default() };
    ctx.stdout(&mut out, outcome.stdout, outcome.stdout_bytes);
    match exit_of(name, outcome.result) {
        Ok(code) => {
            out.exit_code = Some(code);
            Reply::new(out, u8::try_from(code).unwrap_or(1))
        }
        Err(e) => {
            out.error = Some(e);
            Reply::new(out, 1)
        }
    }
}

/// Runs an entry point once per case, each with its own arguments and input (a case without
/// `stdin` reads the context's, or nothing); the program is loaded and compiled once. `results`
/// are in the order of the cases; the call fails (exit 1) only if a case ended in a runtime
/// error. An exit code is data: a program whose specification says "1 when something was
/// rejected" ran correctly when it exits with 1.
pub fn run_cases(ctx: &Ctx, source: &Source, perms: Permissions, name: &str, cases: &[RunCase]) -> Reply {
    if cases.is_empty() || cases.len() > MAX_CASES {
        let d = Diagnostic::new("E0701", "bad_arguments", "", "", format!("{} cases; 1 to {MAX_CASES} per call", cases.len()));
        return Reply::diagnostics(vec![d]);
    }
    let mut runner = match entry_runner(ctx, source, perms, name) {
        Ok(r) => r,
        Err(reply) => return reply,
    };
    let mut results = Vec::with_capacity(cases.len());
    for case in cases {
        let input = case.stdin.as_ref().map(|s| s.as_bytes().to_vec()).or_else(|| ctx.stdin.clone()).unwrap_or_default();
        runner = runner.with_stdin(input);
        let outcome = runner.run_entry(name, &case.args);
        let (stdout, stdout_truncated) = ctx.cut_stdout(outcome.stdout, outcome.stdout_bytes);
        let (exit_code, error) = match exit_of(name, outcome.result) {
            Ok(code) => (Some(code), None),
            Err(e) => (None, Some(e)),
        };
        results.push(RunResult { args: case.args.clone(), exit_code, stdout: Some(stdout), stdout_truncated, error });
    }
    let failed = results.iter().any(|r| r.error.is_some());
    let out = Output { results: Some(Results::Run(results)), failure: failed, ..Output::default() };
    Reply::new(out, u8::from(failed))
}

/// The runner of the entry point `name` (`fn(List[Str]) -> Int`), or the reply why there is none.
fn entry_runner(ctx: &Ctx, source: &Source, perms: Permissions, name: &str) -> Result<qal_run::Runner, Reply> {
    let loaded = ctx.load(source).map_err(Reply::diagnostics)?;
    if let Err(mut e) = find_fn(ctx, source, &loaded, name) {
        if e.message.starts_with("no function") {
            e.message = format!("no entry point `{name}`");
        }
        return Err(Reply::error(e));
    }
    if !is_entry_point(&loaded.typed, name) {
        let mut e = RunError::new("bad_args", name, format!("`{name}` is not an entry point"));
        e.hints.push("`run` needs a function `fn(List[Str]) -> Int`; use `call` for others".into());
        return Err(Reply::error(e));
    }
    ctx.runner(&loaded, &Entry::Fn(name.to_string()), perms).map_err(Reply::error)
}

/// The exit code of an entry point's result, or its runtime error.
fn exit_of(name: &str, result: Result<Value, RunError>) -> Result<i64, RunError> {
    match result {
        Ok(Value::Int(code)) => Ok(code),
        Ok(other) => Err(RunError::new("trap", name, format!("entry point returned {other:?} instead of Int"))),
        Err(e) => Err(capped(e)),
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
    let module = match qal_compiler::compile(&loaded.typed, &Entry::Fn(name.to_string())) {
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
    // Team mode: one build directory per space (Phase 4 spec §10.2).
    let rel = match &ctx.space {
        Some(space) => format!(".qal/build/{space}"),
        None => ".qal/build".to_string(),
    };
    let dir = project.join(&rel);
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
    let (wasm_rel, manifest_rel) = (format!("{rel}/{name}.wasm"), format!("{rel}/{name}.manifest.json"));
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
    let (mut store, std) = match open_for_write(ctx, true) {
        Ok(s) => s,
        Err(d) => return Reply::diagnostics(d),
    };
    let (result, reports) = proving(|verify| store.import_verified(file, verify));
    match result {
        Ok(i) => {
            let reports = current_reports(&i.defined, reports);
            let mut out = Output { imported: Some(i), ..Output::default() };
            out.set_proofs(reports);
            std.report(&mut out);
            Reply::new(out, 0)
        }
        Err(d) => std.failed(d),
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
