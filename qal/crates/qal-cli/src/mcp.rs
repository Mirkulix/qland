//! `qal mcp`: a Model Context Protocol server over stdio (spec §9), implemented directly on
//! JSON-RPC 2.0 with serde_json.
//!
//! Transport: one JSON-RPC message per line on stdin, one response per line on stdout (nothing
//! else is ever written to stdout; diagnostics of the server itself go to stderr). Requests are
//! answered one at a time, in order.
//!
//! Protocol eras (both served by one process):
//! - legacy (`initialize` handshake, versions in [`LEGACY_VERSIONS`]): `initialize`,
//!   `notifications/initialized`, `ping`, `tools/list`, `tools/call`;
//! - modern ([`MODERN_VERSION`], stateless): every request carries
//!   `_meta["io.modelcontextprotocol/protocolVersion"]` and `.../clientCapabilities`;
//!   `server/discover`, `ping`, `tools/list`, `tools/call`; results carry `resultType`.
//!
//! Robustness: messages over [`MAX_MESSAGE_BYTES`] are rejected without being buffered, malformed
//! JSON or JSON-RPC is an error response (never a crash), every program run has a wall-clock
//! timeout, a fuel budget and a memory limit, captured output is cut to [`MAX_STDOUT_BYTES`], and a
//! panic inside a tool (a bug) is caught and answered as an internal error.

use crate::ops::{self, CallArgs, Ctx, Reply, Source, StoreAt};
use qal_core::diag::Diagnostic;
use qal_run::Permissions;
use qal_store::team::keys::KeyDir;
use qal_store::team::policy::{self, Role, Space};
use qal_store::team::{Identity, Notice, Signer, TeamLog};
use qal_store::{Query, RecallQuery, Store};
use std::sync::Arc;
use std::time::Instant;
use serde::Serialize;
use serde_json::{json, Map, Value as J};
use std::io::{BufRead, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Initialization-based protocol versions, newest first.
pub const LEGACY_VERSIONS: [&str; 4] = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];
/// The stateless protocol version (per-request `_meta`).
pub const MODERN_VERSION: &str = "2026-07-28";
/// Longest accepted message (one line), in bytes.
pub const MAX_MESSAGE_BYTES: usize = 4 << 20;
/// Longest accepted `source` of `qal_define`, in bytes.
pub const MAX_SOURCE_BYTES: usize = 1 << 20;
/// Wall-clock limit of one program execution.
pub const RUN_TIMEOUT: Duration = Duration::from_secs(10);
/// Wall-clock budget for the tests of one `qal_test` call.
pub const TEST_BUDGET: Duration = Duration::from_secs(30);
/// Captured program output returned to the agent, in bytes.
pub const MAX_STDOUT_BYTES: usize = 64 << 10;
/// Largest tool result (its JSON text), in bytes; a larger one is replaced by `result_too_large`.
pub const MAX_RESULT_BYTES: usize = 4 << 20;

const META_VERSION: &str = "io.modelcontextprotocol/protocolVersion";
const META_CAPABILITIES: &str = "io.modelcontextprotocol/clientCapabilities";
const META_CLIENT: &str = "io.modelcontextprotocol/clientInfo";
const META_SERVER: &str = "io.modelcontextprotocol/serverInfo";

const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const INTERNAL_ERROR: i64 = -32603;
const UNSUPPORTED_VERSION: i64 = -32022;

/// `reference.md`: the language reference, then the section „Team“ (Phase 4 §8) that team stores
/// add to `qal_define`'s description, per role.
const REFERENCE_FILE: &str = include_str!("reference.md");

/// Where the section „Team“ starts in `text` (a line `## Team`), else its end.
const fn team_at(text: &str) -> usize {
    let b = text.as_bytes();
    let marker = b"## Team\n";
    let mut i = 0;
    while i + marker.len() <= b.len() {
        if i == 0 || b[i - 1] == b'\n' {
            let mut j = 0;
            while j < marker.len() && b[i + j] == marker[j] {
                j += 1;
            }
            if j == marker.len() {
                return i;
            }
        }
        i += 1;
    }
    b.len()
}

/// `text` up to byte `at` (`from`: from it on), or all of it if that is not a character boundary.
const fn cut(text: &str, at: usize, from: bool) -> &str {
    let (head, tail) = text.as_bytes().split_at(at);
    match std::str::from_utf8(if from { tail } else { head }) {
        Ok(s) => s,
        Err(_) => text,
    }
}

/// The compact language reference (≤ 1 page), part of `qal_define`'s description.
pub const REFERENCE: &str = cut(REFERENCE_FILE, team_at(REFERENCE_FILE), false);

/// The section „Team“: lines `@<role> ...` for one role each, the other lines for every role.
pub const TEAM_REFERENCE: &str = cut(REFERENCE_FILE, team_at(REFERENCE_FILE), true);

const INSTRUCTIONS: &str = "QAL is a programming language for AI agents. Code lives in a checked, content-addressed store in this project \
(.qal/store.db), not in files: write it with qal_define (its description is the language reference), test it with qal_test, prove its contracts with qal_prove, run it with \
qal_run (entry point main(args: List[Str]) -> Int) or qal_call (any function). Every result is JSON; errors are diagnostics with a path and hints.";

/// A JSON-RPC error.
struct RpcError {
    code: i64,
    message: String,
    data: Option<J>,
}

fn rpc(code: i64, message: impl Into<String>) -> RpcError {
    RpcError { code, message: message.into(), data: None }
}

/// How a request is served: the legacy (session) or the modern (stateless) protocol.
#[derive(Clone, Copy, PartialEq)]
enum Era {
    Legacy,
    Modern,
}

pub struct Server {
    root: PathBuf,
    /// Client name from `initialize` (legacy), used as the default agent name.
    client: Option<String>,
    /// The session of a team store (Phase 4 spec §4.3), read once at start.
    session: Session,
}

/// What the server knows about its session: only the SHA-256 of the token is kept.
enum Session {
    /// No session file was given.
    None,
    /// The session file could not be used (why).
    Unusable(String),
    /// SHA-256 of the token.
    Token(String),
}

/// Longest `wait_s` of `qal_whoami` (below the 60 s timeout of the agents' client).
pub const MAX_WAIT_S: u64 = 45;
/// How often a waiting `qal_whoami` looks for new notices.
const POLL: Duration = Duration::from_millis(200);

/// The tools only a team store has.
const TEAM_TOOLS: [&str; 7] = ["qal_whoami", "qal_verify", "qal_change", "qal_lease", "qal_review", "qal_merge", "qal_rebase"];

/// The tools that work in a space (`change` argument, Phase 4 §10.2).
const SPACE_TOOLS: [&str; 13] = ["qal_define", "qal_rename", "qal_delete", "qal_view", "qal_query", "qal_history", "qal_test", "qal_prove", "qal_run", "qal_call", "qal_request", "qal_build", "qal_recall"];

#[derive(Serialize)]
struct Response<'a, R: Serialize> {
    jsonrpc: &'static str,
    id: &'a J,
    result: R,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    code: i64,
    message: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: &'a Option<J>,
}

#[derive(Serialize)]
struct ErrorResponse<'a> {
    jsonrpc: &'static str,
    id: &'a J,
    error: ErrorBody<'a>,
}

#[derive(Serialize)]
struct Text {
    #[serde(rename = "type")]
    kind: &'static str,
    text: String,
}

/// A `tools/call` result: the operation's JSON as text (for clients that only read `content`) and
/// as `structuredContent`.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolResult<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    result_type: Option<&'static str>,
    content: [Text; 1],
    structured_content: &'a ops::Output,
    is_error: bool,
    #[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
    meta: Option<J>,
}

fn server_info() -> J {
    json!({"name": "qal", "title": "QAL", "version": env!("CARGO_PKG_VERSION")})
}

fn to_line<T: Serialize>(v: &T) -> String {
    serde_json::to_string(v).unwrap_or_else(|e| {
        format!("{{\"jsonrpc\":\"2.0\",\"id\":null,\"error\":{{\"code\":{INTERNAL_ERROR},\"message\":\"cannot serialize the response: {}\"}}}}", e.to_string().replace(['"', '\\'], "'"))
    })
}

fn error_line(id: &J, e: &RpcError) -> String {
    to_line(&ErrorResponse { jsonrpc: "2.0", id, error: ErrorBody { code: e.code, message: &e.message, data: &e.data } })
}

fn ensure_compiler() {
    if !qal_compiler::has_qal_driver() {
        let wants_qal = |c: &str| c == "qal" || c == "both";
        let needs_seed = !cfg!(feature = "reference")
            || std::env::var("QAL_COMPILER").is_ok_and(|c| wants_qal(&c));
        if needs_seed {
            let _ = qal_run::selfhost::init_seed_backend(qal_run::selfhost::default_jobs());
        }
    }
}

fn tool_needs_compiler(name: &str) -> bool {
    matches!(name, "qal_define" | "qal_run" | "qal_test" | "qal_build" | "qal_prove" | "qal_request" | "qal_call" | "qal_merge" | "qal_rebase" | "qal_rename")
}

impl Server {

    pub fn new(root: &Path) -> std::io::Result<Server> {
        let root = root.canonicalize().map_err(|e| std::io::Error::new(e.kind(), format!("{}: {e}", root.display())))?;
        if !root.is_dir() {
            return Err(std::io::Error::other(format!("{} is not a directory", root.display())));
        }
        Ok(Server { root, client: None, session: Session::None })
    }

    /// A server whose agent uses the session in `file` (a team store; ignored by other stores).
    pub fn with_session(root: &Path, file: Option<&Path>) -> std::io::Result<Server> {
        let mut s = Server::new(root)?;
        s.session = match file {
            None => Session::None,
            Some(f) => match crate::secret_file::read_session(f) {
                Ok(token) => Session::Token(qal_store::team::token_hash(&token)),
                Err(why) => Session::Unusable(why),
            },
        };
        Ok(s)
    }

    /// The project's store if it is a team store. Fails closed: a store that exists but cannot be
    /// opened or read is an error, never treated as a store without team mode.
    fn team_store(&self) -> Result<Option<Store>, Vec<Diagnostic>> {
        let path = self.root.join(qal_store::STORE_DIR).join(qal_store::STORE_FILE);
        if !path.exists() {
            return Ok(None);
        }
        let store = Store::open(&path)?;
        Ok(store.is_team()?.then_some(store))
    }

    /// The agent's identity in a team store (E0902 without a valid session).
    fn identity(&self, store: &Store) -> Result<Identity, Diagnostic> {
        match &self.session {
            Session::None => Err(qal_store::team::no_session("this is a team store and the server has no session")),
            Session::Unusable(why) => Err(qal_store::team::no_session(why.clone())),
            Session::Token(h) => store.resolve_session(h),
        }
    }

    /// `tools/list` of a team store: the tools of the session's role (only `qal_whoami` without a
    /// valid session, which then explains why).
    fn team_tools(&self, store: &Store) -> J {
        let all = tools_with_team();
        let Ok(who) = self.identity(store) else {
            return J::Array(all.into_iter().filter(|t| t["name"] == "qal_whoami").collect());
        };
        let Some(role) = Role::parse(&who.role) else { return J::Array(Vec::new()) };
        let names = policy::tools_for(role);
        let mut out = Vec::new();
        for mut t in all.into_iter().filter(|t| t["name"].as_str().is_some_and(|n| names.contains(&n))) {
            if t["name"] == "qal_define" {
                let d = format!("{}\n\n{}", t["description"].as_str().unwrap_or(""), team_reference(role));
                t["description"] = J::from(d);
            }
            out.push(t);
        }
        J::Array(out)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Handles one message; returns the response line, or None for notifications and responses.
    pub fn handle(&mut self, line: &[u8]) -> Option<String> {
        match catch_unwind(AssertUnwindSafe(|| self.handle_inner(line))) {
            Ok(r) => r,
            Err(p) => {
                // Answer with the request's id when the message has a usable one.
                let id = serde_json::from_slice::<J>(line).ok().and_then(|m| m.get("id").filter(|id| id.is_string() || id.is_i64() || id.is_u64()).cloned()).unwrap_or(J::Null);
                Some(error_line(&id, &rpc(INTERNAL_ERROR, format!("internal error: {}", panic_message(&p)))))
            }
        }
    }

    fn handle_inner(&mut self, line: &[u8]) -> Option<String> {
        let msg: J = match serde_json::from_slice(line) {
            Ok(m) => m,
            Err(e) => return Some(error_line(&J::Null, &rpc(PARSE_ERROR, format!("parse error: {e}")))),
        };
        let Some(obj) = msg.as_object() else {
            let what = if msg.is_array() { "batches are not supported" } else { "a message must be a JSON object" };
            return Some(error_line(&J::Null, &rpc(INVALID_REQUEST, what)));
        };
        // MCP ids are strings or integers; other numbers could not be echoed exactly.
        let id = match obj.get("id") {
            None => None,
            Some(id @ J::String(_)) => Some(id.clone()),
            Some(id @ J::Number(n)) if n.is_i64() || n.is_u64() => Some(id.clone()),
            Some(_) => return Some(error_line(&J::Null, &rpc(INVALID_REQUEST, "`id` must be a string or an integer"))),
        };
        if obj.get("jsonrpc").and_then(J::as_str) != Some("2.0") {
            return Some(error_line(id.as_ref().unwrap_or(&J::Null), &rpc(INVALID_REQUEST, "`jsonrpc` must be \"2.0\"")));
        }
        let Some(method) = obj.get("method").and_then(J::as_str) else {
            // A response to a request we never sent (this server sends none): ignore it.
            if obj.contains_key("result") || obj.contains_key("error") {
                return None;
            }
            return Some(error_line(id.as_ref().unwrap_or(&J::Null), &rpc(INVALID_REQUEST, "`method` must be a string")));
        };
        let empty = Map::new();
        let params = match obj.get("params") {
            None => &empty,
            Some(J::Object(p)) => p,
            Some(_) => return id.map(|id| error_line(&id, &rpc(INVALID_PARAMS, "`params` must be an object"))),
        };
        // Notifications (`initialized`, `cancelled`, ...) need no answer; requests run to completion.
        let id = id?;
        Some(match self.request(method, params) {
            Ok(line) => line(&id),
            Err(e) => error_line(&id, &e),
        })
    }

    /// The era of a request from its `_meta`, validating the modern fields.
    fn era(params: &Map<String, J>) -> Result<Era, RpcError> {
        let meta = params.get("_meta").and_then(J::as_object);
        let Some(version) = meta.and_then(|m| m.get(META_VERSION)) else {
            return Ok(Era::Legacy);
        };
        let supported = || {
            let mut all = vec![MODERN_VERSION];
            all.extend(LEGACY_VERSIONS);
            all
        };
        match version.as_str() {
            Some(MODERN_VERSION) => {}
            Some(v) if LEGACY_VERSIONS.contains(&v) => return Ok(Era::Legacy),
            _ => {
                let mut e = rpc(UNSUPPORTED_VERSION, "Unsupported protocol version");
                e.data = Some(json!({"supported": supported(), "requested": version}));
                return Err(e);
            }
        }
        if !meta.is_some_and(|m| m.get(META_CAPABILITIES).is_some_and(J::is_object)) {
            return Err(rpc(INVALID_PARAMS, format!("`_meta` must contain `{META_CAPABILITIES}` (an object)")));
        }
        Ok(Era::Modern)
    }

    /// Dispatches a request; the result is a function from the id to the response line.
    #[allow(clippy::type_complexity)]
    fn request(&mut self, method: &str, params: &Map<String, J>) -> Result<Box<dyn FnOnce(&J) -> String>, RpcError> {
        let era = Self::era(params)?;
        let finish = move |mut result: Map<String, J>| -> Box<dyn FnOnce(&J) -> String> {
            if era == Era::Modern {
                result.insert("resultType".into(), J::from("complete"));
                result.insert("_meta".into(), json!({ META_SERVER: server_info() }));
            }
            Box::new(move |id: &J| to_line(&Response { jsonrpc: "2.0", id, result: J::Object(result) }))
        };
        let object = |v: J| match v {
            J::Object(m) => m,
            _ => Map::new(),
        };
        match method {
            "initialize" => {
                let requested = params.get("protocolVersion").and_then(J::as_str).unwrap_or("");
                let version = if LEGACY_VERSIONS.contains(&requested) { requested } else { LEGACY_VERSIONS[0] };
                self.client = params.get("clientInfo").and_then(|c| c.get("name")).and_then(J::as_str).map(str::to_string);
                Ok(finish(object(json!({
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": false}},
                    "serverInfo": server_info(),
                    "instructions": INSTRUCTIONS,
                }))))
            }
            "server/discover" => {
                if era != Era::Modern {
                    return Err(rpc(INVALID_PARAMS, format!("`server/discover` needs `_meta` with `{META_VERSION}` and `{META_CAPABILITIES}`")));
                }
                let mut versions = vec![MODERN_VERSION];
                versions.extend(LEGACY_VERSIONS);
                Ok(finish(object(json!({
                    "supportedVersions": versions,
                    "capabilities": {"tools": {"listChanged": false}},
                    "instructions": INSTRUCTIONS,
                }))))
            }
            "ping" => Ok(finish(Map::new())),
            "tools/list" => {
                let list = match self.team_store() {
                    Ok(Some(store)) => self.team_tools(&store),
                    Ok(None) => tools(),
                    // Fail closed: only the tool that reports the problem.
                    Err(_) => J::Array(tools_with_team().into_iter().filter(|t| t["name"] == "qal_whoami").collect()),
                };
                Ok(finish(object(json!({ "tools": list }))))
            }
            "tools/call" => {
                let Some(name) = params.get("name").and_then(J::as_str) else {
                    return Err(rpc(INVALID_PARAMS, "`name` must be a string"));
                };
                let empty = Map::new();
                let args = match params.get("arguments") {
                    None | Some(J::Null) => &empty,
                    Some(J::Object(a)) => a,
                    Some(_) => return Err(rpc(INVALID_PARAMS, "`arguments` must be an object")),
                };
                if !TOOL_NAMES.contains(&name) && !TEAM_TOOLS.contains(&name) {
                    return Err(rpc(INVALID_PARAMS, format!("Unknown tool: {name}")));
                }
                let client = match era {
                    Era::Modern => params.get("_meta").and_then(|m| m.get(META_CLIENT)).and_then(|c| c.get("name")).and_then(J::as_str).map(str::to_string),
                    Era::Legacy => self.client.clone(),
                };
                let mut reply = match catch_unwind(AssertUnwindSafe(|| self.call_tool(name, args, client.as_deref()))) {
                    Ok(r) => r,
                    Err(p) => return Err(rpc(INTERNAL_ERROR, format!("internal error in {name}: {}", panic_message(&p)))),
                };
                let modern = era == Era::Modern;
                // The output is serialized once; the response holds it twice (as text and as
                // structured content), so its size follows from the text.
                let mut text = to_line(&reply.out);
                let mut size = text.len() + to_line(&text).len();
                if size > MAX_RESULT_BYTES && reply.out.defined.is_some() {
                    // A `define` has committed: its result keeps `defined` and only counts the
                    // proof results it cannot show.
                    reply.out.omit_proofs();
                    text = to_line(&reply.out);
                    size = text.len() + to_line(&text).len();
                }
                if size > MAX_RESULT_BYTES {
                    let d = Diagnostic::new("E0702", "result_too_large", "", "", format!("the result of {name} is about {size} bytes of JSON, more than {MAX_RESULT_BYTES}"))
                        .hint("return or print less data, e.g. a summary or a prefix");
                    reply = Reply::diagnostics(vec![d]);
                    text = to_line(&reply.out);
                }
                Ok(Box::new(move |id: &J| {
                    let result = ToolResult {
                        result_type: modern.then_some("complete"),
                        content: [Text { kind: "text", text }],
                        structured_content: &reply.out,
                        is_error: reply.is_error(),
                        meta: modern.then(|| json!({ META_SERVER: server_info() })),
                    };
                    to_line(&Response { jsonrpc: "2.0", id, result })
                }))
            }
            _ => Err(rpc(METHOD_NOT_FOUND, format!("method not found: {method}"))),
        }
    }

    fn ctx(&self, stdin: Option<String>, team: Option<&TeamCall>) -> Ctx {
        Ctx {
            store: StoreAt::Root(self.root.clone()),
            // Never the process's stdin: that is the protocol channel.
            stdin: Some(stdin.unwrap_or_default().into_bytes()),
            timeout: Some(RUN_TIMEOUT),
            test_budget: Some(TEST_BUDGET),
            max_stdout: Some(MAX_STDOUT_BYTES),
            signer: team.and_then(|t| t.signer.clone()),
            space: team.map(|t| t.space.id().to_string()),
            session: team.map(|t| t.session.clone()),
        }
    }

    fn call_tool(&self, name: &str, args: &Map<String, J>, client: Option<&str>) -> Reply {
        let start = Instant::now();
        let team = match self.team_store() {
            Ok(t) => t,
            Err(d) => return Reply::diagnostics(d),
        };
        // The team's bookkeeping file (best effort: without it, no notices and no call log).
        let log = team.as_ref().and_then(|s| s.team_log().ok());
        let mut who = None;
        let mut pending = None;
        let mut space = qal_store::Space::Main;
        let result = match &team {
            None if TEAM_TOOLS.contains(&name) => Ok(Reply::diagnostics(vec![qal_store::team::not_a_team(&format!("`{name}`"))])),
            None if args.get("change").is_some_and(|c| !c.is_null()) => Ok(Reply::diagnostics(vec![qal_store::team::not_a_team("`change`")])),
            None => self.tool(name, args, client, None),
            Some(store) => match self.identity(store) {
                Err(d) => Ok(Reply::diagnostics(vec![d])),
                Ok(id) => {
                    pending = log.as_ref().and_then(|l| std_notice(store, l, &id));
                    let r = self.team_tool(store, log.as_ref(), &id, name, args, &mut pending, &mut space);
                    who = Some(id);
                    r
                }
            },
        };
        let mut reply = match result {
            Ok(r) => r,
            Err(message) => Reply::diagnostics(vec![Diagnostic::new("E0701", "bad_arguments", "", "", message).hint(format!("see the input schema of `{name}` in tools/list"))]),
        };
        // Hashes are for addressing old versions (qal_history, qal_view `#<hash>`): in 295 view,
        // history and query calls of the benchmark agents none used one, and two per defined name
        // made a define reply of 60 names 9 KB longer.
        for d in reply.out.defined.iter_mut().flatten() {
            d.hash.clear();
            d.old_hash = None;
        }
        if let Some(e) = &mut reply.out.error {
            for h in &mut e.hints {
                *h = allow_hint(h);
            }
            for s in [&mut e.def, &mut e.path, &mut e.message].into_iter().chain(e.hints.iter_mut()) {
                shorten(s);
            }
        }
        // Diagnostics echo what the agent sent (names, patterns): keep the real message readable.
        for d in reply.out.diagnostics.iter_mut().flatten() {
            for s in [&mut d.def, &mut d.path, &mut d.message].into_iter().chain(d.hints.iter_mut()) {
                shorten(s);
            }
            for s in d.expected.iter_mut().chain(d.found.iter_mut()).chain(d.dependents.iter_mut()) {
                shorten(s);
            }
        }
        if let Some(store) = &team {
            team_envelope(store, log.as_ref(), who.as_ref(), pending, name, &space, &mut reply, start);
        }
        reply
    }

    /// A tool call in a team store by the session's member: the matrix (E0903), the grants
    /// (E0915), the member's key for writes (E0920), then the tool itself, signed.
    #[allow(clippy::too_many_arguments)]
    fn team_tool(&self, store: &Store, log: Option<&TeamLog>, who: &Identity, name: &str, args: &Map<String, J>, pending: &mut Option<Notice>, space: &mut qal_store::Space) -> Result<Reply, String> {
        let Some(role) = Role::parse(&who.role) else {
            return Ok(Reply::diagnostics(vec![qal_store::team::no_session(format!("the member `{}` has no agent role", who.member))]));
        };
        // The space: the `change` argument, else the session's current change set, else main.
        let a = Args::new(name, args)?;
        let kind = if SPACE_TOOLS.contains(&name) {
            let chosen = a.string("change")?.or_else(|| who.change.clone());
            match chosen {
                None => Space::Main,
                Some(c) if c == qal_store::space::MAIN => Space::Main,
                Some(c) => match store.change(&c) {
                    Ok(Some(row)) => {
                        *space = qal_store::Space::Change(c);
                        row.kind_for(&who.member)
                    }
                    Ok(None) => return Ok(Reply::diagnostics(vec![qal_store::space::unknown_change(&c)])),
                    Err(d) => return Ok(Reply::diagnostics(d)),
                },
            }
        } else {
            Space::Main
        };
        // `qal_change` and `qal_lease` check their actions themselves (per change set).
        if !TEAM_ACTION_TOOLS.contains(&name) && !policy::allows(name, None, role, kind) {
            return Ok(Reply::diagnostics(vec![policy::forbidden(name, None, role, kind)]));
        }
        match name {
            "qal_whoami" => return whoami(store, log, who, &a, pending.take()),
            "qal_change" | "qal_lease" | "qal_review" | "qal_merge" | "qal_rebase" => {
                if matches!(name, "qal_merge" | "qal_rebase") {
                    ensure_compiler();
                }
                return Ok(team_action(store, who, name, &a));
            }
            "qal_verify" => {
                let a = Args::new(name, args)?;
                let from = match a.get("from") {
                    None => 1,
                    Some(J::Number(n)) => n.as_i64().filter(|n| *n >= 1).ok_or("`from` must be a record number (1 or more)")?,
                    Some(_) => return Err("`from` must be a record number (1 or more)".into()),
                };
                let head = a.string("head")?;
                return Ok(crate::team::verify(&self.ctx(None, None), None, from, head.as_deref()));
            }
            _ => {}
        }
        let mut io = false;
        if matches!(name, "qal_run" | "qal_call" | "qal_request" | "qal_define") {
            let a = Args::new(name, args)?;
            let wanted: Vec<String> = a.strings("allow")?.unwrap_or_default().iter().flat_map(|s| s.split(',').map(|x| x.trim().to_string()).collect::<Vec<_>>()).filter(|x| !x.is_empty()).collect();
            let allowed = match store.team_allow() {
                Ok(a) => a,
                Err(d) => return Ok(Reply::diagnostics(d)),
            };
            if let Some(x) = wanted.iter().find(|x| !allowed.contains(x)) {
                let d = Diagnostic::new("E0915", "grant_forbidden", "", "allow", format!("agents in this team may grant only {}, not `{x}`", allowed.join(", ")))
                    .hint("files, the network and real databases stay with the owner on the command line (qal run, qal serve); test with \"db:mem\"");
                return Ok(Reply::diagnostics(vec![d]));
            }
            io = allowed.iter().any(|x| x == "io");
        }
        let signer = if policy::WRITE_TOOLS.contains(&name) {
            match signer_of(store, who) {
                Ok(s) => Some(Arc::new(s)),
                Err(d) => return Ok(Reply::diagnostics(d)),
            }
        } else {
            None
        };
        self.tool(name, args, None, Some(&TeamCall { signer, member: who.member.clone(), session: who.session.clone(), space: space.clone(), io }))
    }

    fn tool(&self, name: &str, args: &Map<String, J>, client: Option<&str>, team: Option<&TeamCall>) -> Result<Reply, String> {
        if tool_needs_compiler(name) {
            ensure_compiler();
        }
        let a = Args::new(name, args)?;
        let agent = |a: &Args| -> Result<String, String> {
            // In a team store the member is the agent (the `agent` argument is ignored).
            if let Some(t) = team {
                a.string("agent")?;
                return Ok(t.member.clone());
            }
            Ok(match a.string("agent")? {
                Some(s) if s.is_empty() || s.chars().count() > 64 || s.chars().any(invisible) => return Err("`agent` must be 1 to 64 characters without control or formatting characters".into()),
                Some(s) => s,
                None => match client {
                    Some(c) => format!("mcp:{}", c.chars().filter(|c| !invisible(*c)).take(48).collect::<String>()),
                    None => "mcp".into(),
                },
            })
        };
        let ctx = self.ctx(a.string("stdin")?, team);
        // `io` is the call's own stdin and stdout (print, read_line, read_all): granted unless a team
        // store forbids it (pressure run 7: forgetting it cost a round 9 times in 12 stages); the
        // rest needs `allow`.
        let perms = |a: &Args| -> Result<Permissions, String> {
            let mut p = Permissions::parse_in(&a.strings("allow")?.unwrap_or_default(), &self.root)?;
            p.io |= team.is_none_or(|t| t.io);
            Ok(p)
        };
        Ok(match name {
            "qal_define" => {
                let src = a.required("source")?;
                if src.len() > MAX_SOURCE_BYTES {
                    return Err(format!("`source` is longer than {MAX_SOURCE_BYTES} bytes; define the code in several calls"));
                }
                // `cases`: run the entry point on them once the code is stored (the round of a
                // separate qal_run after every define); code that does not check is not run.
                let run = match a.cases()? {
                    Some(cases) => Some((a.string("entry")?.unwrap_or_else(|| "main".into()), perms(&a)?, cases)),
                    None => None,
                };
                let mut reply = ops::define(&ctx, &src, &agent(&a)?, false);
                if let Some((entry, perms, cases)) = run {
                    if reply.out.diagnostics.is_none() {
                        let r = ops::run_cases(&ctx, &Source::Store, perms, &entry, &cases);
                        reply.out.failure |= r.out.failure;
                        reply.out.results = r.out.results;
                        if let Some(d) = r.out.diagnostics {
                            reply.out.diagnostics = Some(d);
                        }
                        reply.code = reply.code.max(r.code);
                    }
                }
                reply
            }
            "qal_view" => {
                let names = a.names("names")?;
                if names.is_empty() {
                    return Err("`names` must list at least one name or `#<hash>`".into());
                }
                ops::view(&ctx, &names)
            }
            "qal_query" => ops::query(&ctx, &Query { name: a.string("name")?, type_pattern: a.string("type")?, uses: a.string("uses")?, outline: a.flag("outline")? }),
            "qal_recall" => ops::recall(&ctx, &RecallQuery { code: a.string("code")?, message: a.string("message")?, name: a.string("name")?, type_pattern: a.string("type")? }),
            "qal_test" => {
                let seed = match a.get("seed") {
                    None => 0,
                    Some(J::Number(n)) => n.as_u64().ok_or("`seed` must be a non-negative integer")?,
                    Some(_) => return Err("`seed` must be a non-negative integer".into()),
                };
                ops::test(&ctx, &Source::Store, Permissions::default(), &a.names("names")?, seed)
            }
            "qal_prove" => ops::prove(&ctx, &Source::Store, &a.names("names")?),
            "qal_run" => {
                let entry = a.string("entry")?.unwrap_or_else(|| "main".into());
                let perms = perms(&a)?;
                match a.cases()? {
                    Some(_) if a.get("args").is_some() || a.get("stdin").is_some() => return Err("with `cases`, give `args` and `stdin` in each case, not beside them".into()),
                    Some(cases) => ops::run_cases(&ctx, &Source::Store, perms, &entry, &cases),
                    None => ops::run(&ctx, &Source::Store, perms, &entry, &a.strings("args")?.unwrap_or_default()),
                }
            }
            "qal_call" => {
                let f = a.required("fn")?;
                let perms = perms(&a)?;
                let values = match a.get("args") {
                    None => Vec::new(),
                    Some(J::Array(xs)) => xs.clone(),
                    Some(_) => return Err("`args` must be an array of JSON values".into()),
                };
                ops::call(&ctx, &Source::Store, perms, &f, CallArgs::Json(&values))
            }
            "qal_build" => ops::build(&ctx, &a.required("entry")?),
            "qal_rename" => ops::rename(&ctx, &a.required("old")?, &a.required("new")?, &agent(&a)?, false),
            "qal_history" => ops::history(&ctx, &a.required("name")?),
            "qal_delete" => {
                let mut names = vec![a.required("name")?];
                names.extend(a.names("with")?);
                ops::delete(&ctx, &names, &agent(&a)?, false)
            }
            "qal_request" => {
                let handler = a.string("handler")?.unwrap_or_else(|| "handle".into());
                let perms = perms(&a)?;
                let list = match a.get("requests") {
                    Some(J::Array(xs)) => xs,
                    Some(_) => return Err("`requests` must be an array of requests".into()),
                    None => return Err("`requests` is required".into()),
                };
                if list.len() > ops::MAX_REQUESTS {
                    return Err(format!("`requests` lists {} requests; at most {} per call", list.len(), ops::MAX_REQUESTS));
                }
                let specs = list.iter().enumerate().map(|(i, r)| ops::RequestSpec::from_json(r).map_err(|e| format!("request {i}: {e}"))).collect::<Result<Vec<_>, _>>()?;
                ops::request(&ctx, &Source::Store, perms, &handler, a.string("init")?.as_deref(), &specs)
            }
            other => return Err(format!("unknown tool `{other}`")),
        })
    }
}

/// Characters that do not show (controls, zero-width and bidirectional formatting): they could make a
/// recorded agent name display as something else.
fn invisible(c: char) -> bool {
    // Controls plus the Unicode format characters (general category Cf, Unicode 15).
    c.is_control()
        || matches!(
            c,
            '\u{AD}'
                | '\u{600}'..='\u{605}'
                | '\u{61C}'
                | '\u{6DD}'
                | '\u{70F}'
                | '\u{890}'..='\u{891}'
                | '\u{8E2}'
                | '\u{180E}'
                | '\u{200B}'..='\u{200F}'
                | '\u{202A}'..='\u{202E}'
                | '\u{2060}'..='\u{2064}'
                | '\u{2066}'..='\u{206F}'
                | '\u{FEFF}'
                | '\u{FFF9}'..='\u{FFFB}'
                | '\u{110BD}'
                | '\u{110CD}'
                | '\u{13430}'..='\u{1343F}'
                | '\u{1BCA0}'..='\u{1BCA3}'
                | '\u{1D173}'..='\u{1D17A}'
                | '\u{E0001}'
                | '\u{E0020}'..='\u{E007F}'
        )
}

/// Longest text of a diagnostic field, in characters.
const MAX_FIELD_CHARS: usize = 300;
/// Most names a single call may list.
const MAX_NAMES: usize = 256;

/// Cuts a long text (typically echoed input) to [`MAX_FIELD_CHARS`] characters plus an ellipsis.
fn shorten(s: &mut String) {
    if let Some((i, _)) = s.char_indices().nth(MAX_FIELD_CHARS) {
        s.truncate(i);
        s.push_str("...");
    }
}

/// The runtime's permission hint names the CLI flag (`grant them with `--allow io,fs:<dir>``); an
/// agent grants effects with the `allow` argument instead.
fn allow_hint(hint: &str) -> String {
    for prefix in ["grant them with `--allow ", "grant it with `--allow "] {
        if let Some(rest) = hint.strip_prefix(prefix) {
            let spec = rest.trim_end_matches('`');
            // Agents test with a fresh in-memory database; the project's file needs an explicit "db".
            let items: Vec<String> = spec.split(',').map(|e| if e == "db" { "\"db:mem\"".to_string() } else { format!("\"{e}\"") }).collect();
            return format!("pass `\"allow\": [{}]` to qal_run, qal_call or qal_request", items.join(", "));
        }
    }
    hint.to_string()
}

/// Tool arguments with schema checks: unknown keys and wrong types are errors.
struct Args<'a> {
    map: &'a Map<String, J>,
}

impl<'a> Args<'a> {
    fn new(tool: &str, map: &'a Map<String, J>) -> Result<Args<'a>, String> {
        let known = tool_params(tool);
        if let Some(k) = map.keys().find(|k| !known.contains(&k.as_str())) {
            // `change` is listed only by the schemas of team stores.
            let shown: Vec<&str> = known.iter().copied().filter(|k| *k != "change").collect();
            return Err(format!("unknown argument `{k}` (`{tool}` takes: {})", shown.join(", ")));
        }
        Ok(Args { map })
    }

    fn get(&self, key: &str) -> Option<&J> {
        self.map.get(key).filter(|v| !v.is_null())
    }

    fn string(&self, key: &str) -> Result<Option<String>, String> {
        match self.get(key) {
            None => Ok(None),
            Some(J::String(s)) => Ok(Some(s.clone())),
            Some(_) => Err(format!("`{key}` must be a string")),
        }
    }

    fn flag(&self, key: &str) -> Result<bool, String> {
        match self.get(key) {
            None => Ok(false),
            Some(J::Bool(b)) => Ok(*b),
            Some(_) => Err(format!("`{key}` must be a boolean")),
        }
    }

    fn required(&self, key: &str) -> Result<String, String> {
        self.string(key)?.ok_or_else(|| format!("`{key}` is required"))
    }

    /// A list of names, at most [`MAX_NAMES`].
    fn names(&self, key: &str) -> Result<Vec<String>, String> {
        let names = self.strings(key)?.unwrap_or_default();
        if names.len() > MAX_NAMES {
            return Err(format!("`{key}` lists {} names; at most {MAX_NAMES} per call", names.len()));
        }
        Ok(names)
    }

    fn strings(&self, key: &str) -> Result<Option<Vec<String>>, String> {
        match self.get(key) {
            None => Ok(None),
            Some(J::Array(xs)) => xs.iter().map(|x| x.as_str().map(str::to_string).ok_or_else(|| format!("`{key}` must be an array of strings"))).collect::<Result<_, _>>().map(Some),
            Some(_) => Err(format!("`{key}` must be an array of strings")),
        }
    }

    /// The `cases` of `qal_run`: 1 to [`ops::MAX_CASES`] objects `{args?, stdin?}`.
    fn cases(&self) -> Result<Option<Vec<ops::RunCase>>, String> {
        let list = match self.get("cases") {
            None => return Ok(None),
            Some(J::Array(xs)) => xs,
            Some(_) => return Err("`cases` must be an array of cases `{args, stdin}`".into()),
        };
        if list.is_empty() || list.len() > ops::MAX_CASES {
            return Err(format!("`cases` lists {} cases; 1 to {} per call", list.len(), ops::MAX_CASES));
        }
        let case = |i: usize, c: &J| -> Result<ops::RunCase, String> {
            let J::Object(m) = c else { return Err(format!("case {i} must be an object `{{args, stdin}}`")) };
            if let Some(k) = m.keys().find(|k| !["args", "stdin"].contains(&k.as_str())) {
                return Err(format!("case {i}: unknown field `{k}` (fields: args, stdin)"));
            }
            let a = Args { map: m };
            Ok(ops::RunCase { args: a.strings("args").map_err(|e| format!("case {i}: {e}"))?.unwrap_or_default(), stdin: a.string("stdin").map_err(|e| format!("case {i}: {e}"))? })
        };
        list.iter().enumerate().map(|(i, c)| case(i, c)).collect::<Result<Vec<_>, _>>().map(Some)
    }
}

/// What a tool call in a team store adds to the context.
struct TeamCall {
    signer: Option<Arc<Signer>>,
    member: String,
    session: String,
    space: qal_store::Space,
    /// The team lets agents grant `io`: it is granted without `allow`, as outside a team.
    io: bool,
}

/// The tools whose rights depend on their action and change set (checked by the store).
const TEAM_ACTION_TOOLS: [&str; 5] = ["qal_change", "qal_lease", "qal_review", "qal_merge", "qal_rebase"];

/// The signer of the session's member (its key from the key directory; E0920 without one).
fn signer_of(store: &Store, who: &Identity) -> Result<Signer, Vec<Diagnostic>> {
    KeyDir::from_env().map_err(|d| vec![d]).and_then(|keys| store.member_signer(&keys, who))
}

/// `qal_change` and `qal_lease` (Phase 4 §10.1). Each store operation checks the rights of the
/// action for its change set and writes in one transaction.
fn team_action(store: &Store, who: &Identity, tool: &str, a: &Args) -> Reply {
    let mut store = match Store::open(store.path()) {
        Ok(s) => s,
        Err(d) => return Reply::diagnostics(d),
    };
    match team_action_inner(&mut store, who, tool, a) {
        Ok(Ok(v)) => {
            let mut out = ops::Output::default();
            if let J::Object(m) = v {
                out.extra = m;
            }
            Reply { out, code: 0 }
        }
        Ok(Err(d)) => Reply::diagnostics(d),
        Err(message) => Reply::diagnostics(vec![Diagnostic::new("E0701", "bad_arguments", "", "", message).hint(format!("see the input schema of `{tool}` in tools/list"))]),
    }
}

type Answer = Result<J, Vec<Diagnostic>>;

fn team_action_inner(store: &mut Store, who: &Identity, tool: &str, a: &Args) -> Result<Answer, String> {
    if matches!(tool, "qal_review" | "qal_merge" | "qal_rebase") {
        return review_merge_rebase(store, who, tool, a);
    }
    let action = a.required("action")?;
    // A role that may do this action nowhere is refused before anything else (E0903).
    if let Some(role) = Role::parse(&who.role) {
        if policy::rule(tool, Some(&action), role).is_some() && !policy::SPACES.into_iter().any(|sp| policy::allows(tool, Some(&action), role, sp)) {
            return Ok(Err(vec![policy::forbidden(tool, Some(&action), role, Space::Main)]));
        }
    }
    let change = |a: &Args| -> Result<Result<String, Vec<Diagnostic>>, String> {
        Ok(match a.string("change")?.or_else(|| who.change.clone()) {
            Some(c) => Ok(c),
            None => Err(vec![Diagnostic::new("E0701", "bad_arguments", "", "change", format!("`{tool}` with action `{action}` needs `change` (this session has no current change set)")).hint("qal_change {\"action\": \"list\"} shows the change sets")]),
        })
    };
    let signed = || signer_of(store, who);
    if tool == "qal_lease" {
        let names = a.names("names")?;
        if names.is_empty() {
            return Err("`names` must list at least one name".into());
        }
        let ttl = match a.get("ttl_s") {
            None => None,
            Some(J::Number(n)) => Some(n.as_i64().filter(|n| *n > 0).ok_or("`ttl_s` must be a positive number of seconds")?.saturating_mul(1000)),
            Some(_) => return Err("`ttl_s` must be a positive number of seconds".into()),
        };
        return Ok(match action.as_str() {
            "break" => signed().and_then(|s| store.lease_break(&s, &names)),
            "acquire" | "renew" | "release" => match change(a)? {
                Ok(c) => store.lease(&who.member, &action, &c, &names, ttl),
                Err(d) => Err(d),
            },
            other => return Err(format!("`action` must be acquire, renew, release or break, not `{other}`")),
        });
    }
    Ok(match action.as_str() {
        "list" => {
            let statuses = a.strings("status")?.unwrap_or_default();
            let mine = match a.get("mine") {
                None => false,
                Some(J::Bool(b)) => *b,
                Some(_) => return Err("`mine` must be true or false".into()),
            };
            let limit = match a.get("limit") {
                None => None,
                Some(J::Number(n)) => Some(n.as_u64().and_then(|n| usize::try_from(n).ok()).filter(|n| (1..=200).contains(n)).ok_or("`limit` must be an integer from 1 to 200")?),
                Some(_) => return Err("`limit` must be an integer from 1 to 200".into()),
            };
            store.change_list(&who.member, &statuses, mine, limit)
        }
        "open" => {
            let package = match a.get("package") {
                None => None,
                Some(J::Bool(b)) => Some(*b),
                Some(_) => return Err("`package` must be true or false".into()),
            };
            let args = qal_store::team::change::OpenArgs { title: a.required("title")?, description: a.string("description")?.unwrap_or_default(), scope: a.names("scope")?, assignee: a.string("assignee")?, package };
            signed().and_then(|s| store.change_open(&s, &args))
        }
        "show" | "diff" | "claim" | "submit" | "abandon" | "scope" => {
            let c = match change(a)? {
                Ok(c) => c,
                Err(d) => return Ok(Err(d)),
            };
            match action.as_str() {
                "show" => store.change_show(&c),
                "diff" => store.change_diff(&c),
                "claim" => signed().and_then(|s| store.change_claim(&s, &c)),
                "submit" => signed().and_then(|s| store.change_submit(&s, &c)),
                "abandon" => signed().and_then(|s| store.change_abandon(&s, &c)),
                _ => {
                    let (add, remove) = (a.names("add")?, a.names("remove")?);
                    signed().and_then(|s| store.change_scope(&s, &c, &add, &remove))
                }
            }
        }
        other => return Err(format!("`action` must be open, list, show, diff, claim, submit, abandon or scope, not `{other}`")),
    })
}

/// `qal_review`, `qal_merge` and `qal_rebase` (Phase 4 §7, §10.1) on `change` (default: the
/// session's current change set). The store checks the rights per change set; a role that may do
/// this nowhere is refused first (E0903).
fn review_merge_rebase(store: &mut Store, who: &Identity, tool: &str, a: &Args) -> Result<Answer, String> {
    let verdict = if tool == "qal_review" { Some(a.required("verdict")?) } else { None };
    if let Some(role) = Role::parse(&who.role) {
        let act = verdict.as_deref();
        if policy::rule(tool, act, role).is_some() && !policy::SPACES.into_iter().any(|sp| policy::allows(tool, act, role, sp)) {
            return Ok(Err(vec![policy::forbidden(tool, act, role, Space::Main)]));
        }
    }
    let comments = match a.get("comments") {
        None => Vec::new(),
        Some(J::Array(xs)) => xs
            .iter()
            .map(|c| serde_json::from_value::<qal_store::team::merge::Comment>(c.clone()).map_err(|_| "each of `comments` must be {\"name\"?: string, \"text\": string}".to_string()))
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => return Err("`comments` must be an array of {\"name\"?, \"text\"}".into()),
    };
    let Some(change) = a.string("change")?.or_else(|| who.change.clone()) else {
        return Ok(Err(vec![Diagnostic::new("E0701", "bad_arguments", "", "change", format!("`{tool}` needs `change` (this session has no current change set)")).hint("qal_change {\"action\": \"list\", \"status\": [\"submitted\"]} shows the change sets to review or merge")]));
    };
    let signer = match signer_of(store, who) {
        Ok(s) => s,
        Err(d) => return Ok(Err(d)),
    };
    Ok(match tool {
        "qal_review" => store.review(&signer, &change, verdict.as_deref().unwrap_or(""), &comments),
        "qal_merge" => ops::merging(&change, TEST_BUDGET, Some(RUN_TIMEOUT), |check| store.merge(&signer, &change, check)),
        _ => ops::proving(|v| store.rebase(&signer, &change, Some(v))).0,
    })
}

/// The first error code of a reply, for the call log.
fn error_code(reply: &Reply) -> Option<String> {
    if let Some(d) = reply.out.diagnostics.as_ref().and_then(|d| d.first()) {
        return Some(d.code.to_string());
    }
    if let Some(e) = &reply.out.error {
        return Some(e.kind.to_string());
    }
    reply.out.extra.get("error").and_then(|e| e.get("code")).and_then(J::as_str).map(str::to_string)
}

/// Team mode around every tool result: notices not given yet (at most 10, `notices_more` counts
/// the rest; `pending` first), the space, and the call log entry (spec §10.2-10.4). Nothing here
/// writes to `store.db` (spec §5.1): seen-marks and the log go to the team log file.
#[allow(clippy::too_many_arguments)]
fn team_envelope(store: &Store, log: Option<&TeamLog>, who: Option<&Identity>, pending: Option<Notice>, tool: &str, space: &qal_store::Space, reply: &mut Reply, start: Instant) {
    // A write lock held longer than the busy timeout: E0907 `store_busy`, retry (§5.1).
    for d in reply.out.diagnostics.iter_mut().flatten() {
        if qal_store::is_busy(d) {
            *d = Diagnostic::new("E0907", "store_busy", "", "", "another agent held the store's write lock for longer than 10 s").hint("retry the call; nothing was written");
        }
    }
    if let Some(w) = who {
        if tool != "qal_whoami" {
            let mut notices: Vec<Notice> = pending.into_iter().collect();
            let mut more = 0;
            if let Some(l) = log {
                if let Ok((n, m)) = store.take_notices(l, &w.member, qal_store::team::NOTICES_PER_REPLY.saturating_sub(notices.len())) {
                    notices.extend(n);
                    more = m;
                }
            }
            if !notices.is_empty() {
                reply.out.extra.insert("notices".into(), serde_json::to_value(&notices).unwrap_or(J::Null));
            }
            if more > 0 {
                reply.out.extra.insert("notices_more".into(), J::from(more));
            }
        }
        reply.out.extra.insert("space".into(), J::from(space.id()));
    }
    let code = error_code(reply);
    let call = qal_store::team::Call {
        member: who.map_or("", |w| w.member.as_str()),
        session: who.map_or("", |w| w.session.as_str()),
        tool,
        ok: !reply.is_error(),
        code: code.as_deref(),
        ms: i64::try_from(start.elapsed().as_millis()).unwrap_or(i64::MAX),
    };
    // The log is measurement only: a failure to write it does not change the answer.
    if let Some(l) = log {
        let _ = l.log_call(&call);
    }
}

/// The notice `std_outdated`, once per std version and member, when this qal embeds a newer
/// standard library than the team store has (the owner imports it with `qal team sync-std`). It
/// is not stored in `store.db` (a read writes nothing there): a flag in the team log remembers
/// that it was given, and it is returned with this call.
fn std_notice(store: &Store, log: &TeamLog, who: &Identity) -> Option<Notice> {
    let embedded = qal_store::package::parse(crate::STD_PACKAGE).ok()?.hash;
    let have = store.meta(qal_store::package::META_STD_PACKAGE).ok().flatten();
    if have.as_deref() == Some(embedded.as_str()) || store.is_std().unwrap_or(false) {
        return None;
    }
    if !log.flag_once(&who.member, &format!("std_outdated:{embedded}")).ok()? {
        return None;
    }
    let body = json!({"embedded": embedded, "store": have, "hint": "the owner imports the standard library of this qal with `qal team sync-std`"});
    let body = match body {
        J::Object(m) => m,
        _ => Map::new(),
    };
    let at = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(0));
    Some(Notice { id: 0, kind: "std_outdated".into(), at, body })
}

/// `qal_whoami {wait_s}`: who the session is, and the notices it was not given yet; with
/// `wait_s`, waits up to that long for a first notice (checking every 200 ms).
fn whoami(store: &Store, log: Option<&TeamLog>, who: &Identity, a: &Args, pending: Option<Notice>) -> Result<Reply, String> {
    let wait = match a.get("wait_s") {
        None => 0,
        Some(J::Number(n)) => n.as_u64().filter(|w| *w <= MAX_WAIT_S).ok_or(format!("`wait_s` must be an integer from 0 to {MAX_WAIT_S}"))?,
        Some(_) => return Err(format!("`wait_s` must be an integer from 0 to {MAX_WAIT_S}")),
    };
    let mut notices: Vec<Notice> = pending.into_iter().collect();
    let mut more = 0;
    if let Some(l) = log {
        let deadline = Instant::now() + Duration::from_secs(wait);
        while notices.is_empty() {
            match store.has_notices(l, &who.member) {
                Ok(true) => break,
                Ok(false) if Instant::now() >= deadline => break,
                Ok(false) => std::thread::sleep(POLL),
                Err(d) => return Ok(Reply::diagnostics(d)),
            }
        }
        match store.take_notices(l, &who.member, qal_store::team::NOTICES_PER_REPLY.saturating_sub(notices.len())) {
            Ok((n, m)) => {
                notices.extend(n);
                more = m;
            }
            Err(d) => return Ok(Reply::diagnostics(d)),
        }
    }
    let mut out = ops::Output::default();
    let leases = match &who.change {
        Some(c) => match store.leases(c) {
            Ok(l) => serde_json::to_value(l).unwrap_or(J::Null),
            Err(d) => return Ok(Reply::diagnostics(d)),
        },
        None => json!([]),
    };
    let mut fields = json!({"member": who.member, "role": who.role, "session": who.session, "expires_at": who.expires_at, "change": who.change, "leases": leases, "notices": notices});
    if more > 0 {
        fields["notices_more"] = J::from(more);
    }
    if let J::Object(m) = fields {
        out.extra = m;
    }
    Ok(Reply { out, code: 0 })
}

/// The ≤ 12-line „Team“ section of `qal_define`'s reference for a role (spec §8): from
/// `reference.md`, the role's own line and the lines for everyone.
pub fn team_reference(role: Role) -> String {
    let own = format!("@{} ", role.as_str());
    TEAM_REFERENCE
        .lines()
        .filter_map(|l| match l.strip_prefix('@') {
            None => Some(l),
            Some(_) => l.strip_prefix(own.as_str()),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The tool list including the team tools (`qal_whoami`, `qal_verify`, `qal_change`, `qal_lease`),
/// every space tool with the `change` argument.
fn tools_with_team() -> Vec<J> {
    let mut all = match tools() {
        J::Array(xs) => xs,
        _ => Vec::new(),
    };
    for t in &mut all {
        if t["name"].as_str().is_some_and(|n| SPACE_TOOLS.contains(&n)) {
            if let Some(J::Object(props)) = t.get_mut("inputSchema").and_then(|s| s.get_mut("properties")) {
                props.insert("change".into(), json!({"type": "string", "description": "The change set to work in (`main` for the main state, read-only). Default: your current change set (from open or claim), else main."}));
            }
        }
    }
    let read_only = json!({"readOnlyHint": true, "openWorldHint": false});
    let writes = json!({"readOnlyHint": false, "destructiveHint": false, "idempotentHint": false, "openWorldHint": false});
    let list = |d: &str| json!({"type": "array", "items": {"type": "string"}, "description": d});
    let text = |d: &str| json!({"type": "string", "description": d});
    all.push(json!({"name": "qal_change", "title": "Change sets (team)", "annotations": writes,
        "description": "Change sets are where code is written; main changes only by merge. open {title, description?, scope?} opens your own (a copy of main; it becomes your current change set); an architect opens work packages with `assignee` (or `package`: true for anyone), which implementers/verifiers `claim` (taking the leases of its scope). list {status?, mine?} is the board; show/diff {change?} show one (show includes each review with reviewer, verdict, comments, time; diff: name, origin, base, ours, main); submit/abandon {change?}; scope {change, add, remove} (architect). `scope` names the existing names an implementer may change (new names are always allowed). Notices tell others (change_opened, change_submitted).",
        "inputSchema": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["open", "list", "show", "diff", "claim", "submit", "abandon", "scope"]},
            "change": text("Change set id (default: your current one)."), "title": text("open: a short title."), "description": text("open: what is to be done."),
            "scope": list("open: existing names that may be changed."), "assignee": text("open (architect): the member the work package is for."),
            "package": {"type": "boolean", "description": "open (architect): a claimable work package (default: true with an assignee)."},
            "status": list("list: only these statuses (open, submitted, merged, abandoned; default open and submitted)."), "mine": {"type": "boolean", "description": "list: only yours (owned or assigned)."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200, "description": "list: at most this many (default 50; `more` counts the rest)."},
            "add": list("scope: names to add."), "remove": list("scope: names to remove.")},
          "required": ["action"], "additionalProperties": false}}));
    all.push(json!({"name": "qal_lease", "title": "Leases on names (team)", "annotations": writes,
        "description": "A lease reserves a name for one change set, so two agents do not change it at once; qal_define takes the leases of the names it writes by itself, so use this only to reserve names ahead. acquire is all or nothing (E0901 names the holders; nothing is taken then; never waits), renew extends your leases, release frees them; break (architect) removes another change set's lease (signed; its owner is told). Default TTL 900 s (60 s to 2 h).",
        "inputSchema": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["acquire", "renew", "release", "break"]}, "names": list("Definition names (up to 64)."),
            "ttl_s": {"type": "integer", "minimum": 60, "maximum": 7200, "description": "Lease time in seconds (default 900)."}, "change": text("Your change set (default: your current one).")},
          "required": ["action", "names"], "additionalProperties": false}}));
    all.push(json!({"name": "qal_review", "title": "Review a change set (team)", "annotations": writes,
        "description": "A signed verdict on a change set, bound to its current root (any later write makes it stale). Reviewers: approve or changes (changes sends it back to its owner, status open); verifiers: verified or failed (after qal_test/qal_prove/qal_request with `change`); architects: comment. Every verdict but comment needs the change set submitted. A merge needs a current approve and a current verified by members who do not own the change set.",
        "inputSchema": {"type": "object", "properties": {
            "change": text("Change set id (default: your current one)."),
            "verdict": {"type": "string", "enum": ["approve", "changes", "verified", "failed", "comment"]},
            "comments": {"type": "array", "maxItems": 32, "items": {"type": "object", "properties": {"name": text("The definition it is about."), "text": text("At most 2 KiB.")}, "required": ["text"], "additionalProperties": false}}},
          "required": ["verdict"], "additionalProperties": false}}));
    all.push(json!({"name": "qal_merge", "title": "Merge a change set into main (team)", "annotations": writes,
        "description": "Merges a submitted change set into main (reviewers and architects). Gate: a current approve and verified by non-owners, no conflict marks (else E0913/E0908). If main did not move since its base: fast_forward. Else each changed name is classified: changed in main too is a conflict (E0910: edit_edit, add_add, rename; then the owner runs qal_rebase); the rest is replayed onto main as one define (E0911 if it does not check, e.g. a signature it uses changed), its contracts are proven afresh and every test runs (E0912 if one fails). Result: kind, entries, rehashed (names whose hash in main differs from the change set's), tests, proofs; other change sets it affects get needs_rebase.",
        "inputSchema": {"type": "object", "properties": {"change": text("The submitted change set to merge.")}, "required": ["change"], "additionalProperties": false}}));
    all.push(json!({"name": "qal_rebase", "title": "Rebase a change set onto main (team)", "annotations": writes,
        "description": "Moves your change set's base to main as it is now: your definitions are replayed onto it in the change set. A name changed on both sides keeps your version and gets a conflict mark {base, theirs} (compare with qal_view [\"n\", \"n@main\"]; define n again to resolve it); a merge refuses marks. If your definitions do not check against the new main, nothing changes and the diagnostics name what to adapt. Clears needs_rebase.",
        "inputSchema": {"type": "object", "properties": {"change": text("Your change set (default: your current one).")}, "additionalProperties": false}}));
    all.push(json!({"name": "qal_whoami", "title": "Who am I (team)", "annotations": read_only,
        "description": "Your member id, role, session and unseen notices (change sets opened for you, merges, ...), marked seen once returned. With `wait_s` (up to 45) it waits for the first new notice instead of returning at once: use it to wait for work instead of polling.",
        "inputSchema": {"type": "object", "properties": {"wait_s": {"type": "integer", "minimum": 0, "maximum": MAX_WAIT_S, "description": "Seconds to wait for a new notice (default 0)."}}, "additionalProperties": false}}));
    all.push(json!({"name": "qal_verify", "title": "Verify the team's records", "annotations": read_only,
        "description": "Check the signed record chain of the team store: sequence and hashes, every Ed25519 signature, each record's rights, and that replaying all records gives exactly the current names. Result: {ok, records, members, trust, head} or {ok: false, error: {code, kind, seq, message}, checked}.",
        "inputSchema": {"type": "object", "properties": {"from": {"type": "integer", "minimum": 1, "description": "Check signatures from this record on (default 1; the result says when signatures were skipped)."}, "head": {"type": "string", "description": "A record hash the chain must contain (a head pinned earlier: finds a cut-off tail)."}}, "additionalProperties": false}}));
    all
}

const TOOL_NAMES: [&str; 13] = ["qal_define", "qal_test", "qal_prove", "qal_run", "qal_call", "qal_request", "qal_view", "qal_query", "qal_rename", "qal_delete", "qal_history", "qal_build", "qal_recall"];

fn tool_params(tool: &str) -> &'static [&'static str] {
    match tool {
        "qal_define" => &["source", "agent", "entry", "cases", "allow", "change"],
        "qal_view" => &["names", "change"],
        "qal_query" => &["name", "type", "uses", "outline", "change"],
        "qal_test" => &["names", "seed", "change"],
        "qal_prove" => &["names", "change"],
        "qal_run" => &["entry", "args", "stdin", "cases", "allow", "change"],
        "qal_call" => &["fn", "args", "stdin", "allow", "change"],
        "qal_build" => &["entry", "change"],
        "qal_rename" => &["old", "new", "agent", "change"],
        "qal_history" => &["name", "change"],
        "qal_recall" => &["code", "message", "name", "type", "change"],
        "qal_delete" => &["name", "with", "agent", "change"],
        "qal_request" => &["handler", "init", "allow", "requests", "change"],
        "qal_whoami" => &["wait_s"],
        "qal_verify" => &["from", "head"],
        "qal_change" => &["action", "change", "title", "description", "scope", "assignee", "package", "status", "mine", "limit", "add", "remove"],
        "qal_lease" => &["action", "names", "ttl_s", "change"],
        "qal_review" => &["change", "verdict", "comments"],
        "qal_merge" => &["change"],
        "qal_rebase" => &["change"],
        _ => &[],
    }
}

/// The tool list (deterministic order).
pub fn tools() -> J {
    let s = |d: &str| json!({"type": "string", "description": d});
    let list = |d: &str| json!({"type": "array", "items": {"type": "string"}, "description": d});
    let allow = list("Effects to grant: \"io\", \"time\", \"rand\", \"fs:<dir>\" (a directory inside the project, relative to its root; hidden entries like .git are never reachable), \"db:mem\" (a fresh in-memory SQLite database), \"db\" or \"db:<name>\" (the project's .qal/db/main.sqlite or <name>.sqlite), \"net:<host>[:port]\" (outgoing HTTP to that host; names must resolve to public addresses, IP literals like \"net:127.0.0.1:9000\" reach exactly that address). \"io\" (the call's own stdin and stdout) is always granted, the rest only when listed.");
    // The full list once (qal_run); the other tools that run code point to it.
    let allow_short = list("Effects to grant, as for qal_run (io is always granted).");
    let allow_request = allow_short.clone();
    let stdin = s("Text that read_line() and read_all() read (default: empty).");
    let agent = s("Who makes the change, recorded in the history (default: the MCP client's name).");
    let read_only = json!({"readOnlyHint": true, "openWorldHint": false});
    let writes = json!({"readOnlyHint": false, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false});
    let define = format!(
        "Add or replace QAL definitions (types, functions, tests) in the project's store. Nothing is stored unless all of them parse and type-check together with the stored code; then the result lists each name with its status (added, replaced, unchanged, relinked; qal_history has the hashes). Otherwise `diagnostics` say what to fix. Contracts are proven on the way (`proofs`): a function whose contract is refuted is not stored (the diagnostic has a counterexample). Then the stored tests that use a function it defined run (`tests`, `passed`, `failed`): a failing test does not undo the definition.\n\n{REFERENCE}"
    );
    json!([
        {"name": "qal_define", "title": "Define QAL code", "description": define, "annotations": writes,
         "inputSchema": {"type": "object", "properties": {"source": s("QAL source text with one or more definitions."), "agent": agent,
             "cases": {"type": "array", "minItems": 1, "maxItems": ops::MAX_CASES, "description": "Optional: once the code is stored, run the entry point on these cases, as qal_run does (`results`).",
                       "items": {"type": "object", "properties": {"args": {"type": "array", "items": {"type": "string"}}, "stdin": {"type": "string"}}, "additionalProperties": false}},
             "entry": s("Entry point for `cases` (default main)."), "allow": list("Effects to grant for `cases`, as for qal_run.")}, "required": ["source"], "additionalProperties": false}},
        {"name": "qal_test", "title": "Run QAL tests", "annotations": read_only,
         "description": "Run the stored tests and the generated property tests (all, or the named tests and functions; in a project, all means its own: `std.` tests run when named). A pure function with an `ens`, or that calls a function with `req`, gets a property test: 100 generated inputs that satisfy its `req`; any contract violation fails it, with a shrunk counterexample. Result: each test with passed true/false and the runtime error of a failing one, `properties` (passed, failed, gave_up, skipped), plus passed/failed counts.",
         "inputSchema": {"type": "object", "properties": {"names": list("Test or function names (default: all)."), "seed": {"type": "integer", "minimum": 0, "description": "Seed of the generated inputs (default 0)."}}, "additionalProperties": false}},
        {"name": "qal_prove", "title": "Prove QAL contracts", "annotations": read_only,
         "description": "Prove the `req`/`ens` contracts of stored functions and tests with Z3 (all, or the named ones; in a project, all means its own: `std.` ones when named): every `ens`, and the `req` of every called function at its call site. Status per obligation: proven, refuted (with a counterexample that really violates it), or unknown (with the reason: Int and Bool facts are modelled exactly, other values only approximately; the contract stays checked at runtime). Results are stored per definition hash.",
         "inputSchema": {"type": "object", "properties": {"names": list("Function or test names (default: all).")}, "additionalProperties": false}},
        {"name": "qal_run", "title": "Run a QAL program", "annotations": {"readOnlyHint": false, "destructiveHint": false, "openWorldHint": true},
         "description": "Run a stored entry point `fn(List[Str]) -> Int` (default `main`) with string arguments. Result: exit_code and stdout, or a runtime `error` (contract violation, overflow, timeout after 10 s, ...). Effects the program declares beyond io must be granted with `allow`. To try several inputs in one call, pass `cases` instead of `args`/`stdin`: `results` has one {args, exit_code, stdout, error} per case, in order; the call fails only if a case ends in a runtime error (an exit code is data).",
         "inputSchema": {"type": "object", "properties": {"entry": s("Entry point name (default main)."), "args": list("Program arguments."), "stdin": stdin.clone(),
             "cases": {"type": "array", "minItems": 1, "maxItems": ops::MAX_CASES, "description": "Runs, one per case, instead of `args`/`stdin`: {args (default []), stdin (default empty)}.",
                       "items": {"type": "object", "properties": {"args": {"type": "array", "items": {"type": "string"}}, "stdin": {"type": "string"}}, "additionalProperties": false}},
             "allow": allow.clone()}, "additionalProperties": false}},
        {"name": "qal_call", "title": "Call a QAL function", "annotations": {"readOnlyHint": false, "destructiveHint": false, "openWorldHint": true},
         "description": "Call any stored function with JSON arguments and get its result as JSON, both in the same form: numbers, strings, true/false, null for Unit, arrays for List and Bytes, [[key, value], ...] for Map, objects for records, `None` is \"None\", `Some(3)` is {\"Some\":[3]}. Function values cannot be passed.",
         "inputSchema": {"type": "object", "properties": {"fn": s("Function name."), "args": {"type": "array", "description": "Arguments in parameter order."}, "stdin": stdin, "allow": allow_short}, "required": ["fn"], "additionalProperties": false}},
        {"name": "qal_request", "title": "Send HTTP requests to a QAL service", "annotations": {"readOnlyHint": false, "destructiveHint": false, "openWorldHint": true},
         "description": "Run HTTP requests against a stored handler `fn(HttpRequest) -> HttpResponse` as `qal serve` would (one database transaction per request: committed for status 200-399, else rolled back). `init` (`fn() -> Result[Unit, Str]`) runs first; all requests share one database. Result: `responses` with status, headers, body, json, db (commit/rollback/none), stdout, and the runtime `error` of a failed request.",
         "inputSchema": {"type": "object", "properties": {
             "handler": s("Handler function (default handle)."),
             "init": s("Function run once before the requests."),
             "allow": allow_request,
             "requests": {"type": "array", "maxItems": 50, "description": "Requests in order: {method (default GET), path (starting with /, without ?), query (the part after ?), headers [{name, value}], body (text) or json (any JSON value: serialized as the body with content-type application/json)}.",
                          "items": {"type": "object"}}},
           "required": ["requests"], "additionalProperties": false}},
        {"name": "qal_view", "title": "View QAL definitions", "annotations": read_only,
         "description": "Show stored definitions as text with their hash, signature, effects and the signatures of their direct dependencies. Names, or `#<hash>` for old versions from the history.",
         "inputSchema": {"type": "object", "properties": {"names": list("Names or #<hash>.")}, "required": ["names"], "additionalProperties": false}},
        {"name": "qal_query", "title": "Search the QAL codebase", "annotations": read_only,
         "description": "List stored definitions {name, kind, hash, signature, effects}, filtered by name prefix, signature pattern (`_` = any one type, e.g. `fn(Str) -> _`) and/or direct use of a name or #hash. No filter lists everything. `outline: true` lists the project's own definitions (no `std.` unless `name` starts with `std.`, no hashes) with each one's `contracts` (req/ens), `uses`, `used_by` (functions) and `tests` (empty lists left out; a test is listed under the `tests` of what it uses): the view to read before a change, much smaller than qal_view.",
         "inputSchema": {"type": "object", "properties": {"name": s("Name prefix."), "type": s("Signature pattern, e.g. fn(List[Int]) -> Int."), "uses": s("Only definitions that refer to this name or #hash."), "outline": {"type": "boolean", "description": "Add contracts, uses, used_by and tests to every entry."}}, "additionalProperties": false}},
        {"name": "qal_rename", "title": "Rename a QAL definition", "annotations": writes,
         "description": "Rename a stored definition. No hash changes; everything that refers to it follows the new name; the history moves along.",
         "inputSchema": {"type": "object", "properties": {"old": s("Current name."), "new": s("New name."), "agent": agent.clone()}, "required": ["old", "new"], "additionalProperties": false}},
        {"name": "qal_delete", "title": "Delete a QAL definition", "annotations": {"readOnlyHint": false, "destructiveHint": true, "idempotentHint": false, "openWorldHint": false},
         "description": "Remove a stored name nothing else refers to (else `has_dependents` lists them: delete those too, or name them in `with`). The old version stays viewable by #hash; `std.` names cannot be deleted in a project.",
         "inputSchema": {"type": "object", "properties": {"name": s("Name to delete."), "with": list("More names deleted in the same call, e.g. the other members of a mutually recursive group (default: none)."), "agent": agent.clone()}, "required": ["name"], "additionalProperties": false}},
        {"name": "qal_history", "title": "QAL definition history", "annotations": read_only,
         "description": "Every recorded change of a name, oldest first: added, replaced, relinked, renamed, deleted, with hashes, agent and time.",
         "inputSchema": {"type": "object", "properties": {"name": s("Definition name.")}, "required": ["name"], "additionalProperties": false}},
        {"name": "qal_build", "title": "Build a QAL module", "annotations": writes,
         "description": "Compile a stored function into a standalone WebAssembly module `.qal/build/<entry>.wasm` plus `<entry>.manifest.json` (hash, signature, required effects, host imports).",
         "inputSchema": {"type": "object", "properties": {"entry": s("Function to build, usually main.")}, "required": ["entry"], "additionalProperties": false}},
        {"name": "qal_recall", "title": "Recall earlier fixes", "annotations": read_only,
         "description": "Earlier fixes of a diagnostic: with `code` and/or `message` the corrections (most frequent first, at most 5, with `before`/`after` excerpts); with `name` (prefix) and/or `type` (signature pattern) matching definitions (at most 20). At least one filter.",
         "inputSchema": {"type": "object", "properties": {"code": s("e.g. E0201."), "message": s("The diagnostic's message or a part of it."), "name": s("Name prefix."), "type": s("Signature pattern, e.g. fn(Str) -> _.")}, "additionalProperties": false}},
    ])
}

fn panic_message(p: &Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "a tool panicked".into())
}

/// One message read from the input.
enum Message {
    Line(Vec<u8>),
    /// Longer than the limit; its bytes were skipped up to the next newline.
    TooLong,
    Eof,
}

/// Reads up to the next newline without ever buffering more than `max` bytes.
fn read_message(input: &mut impl BufRead, max: usize) -> std::io::Result<Message> {
    let mut buf = Vec::new();
    let mut too_long = false;
    loop {
        let chunk = match input.fill_buf() {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        if chunk.is_empty() {
            return Ok(if too_long {
                Message::TooLong
            } else if buf.is_empty() {
                Message::Eof
            } else {
                Message::Line(buf)
            });
        }
        let (part, done) = match chunk.iter().position(|b| *b == b'\n') {
            Some(i) => (&chunk[..i], Some(i + 1)),
            None => (chunk, None),
        };
        if !too_long && buf.len() + part.len() <= max {
            buf.extend_from_slice(part);
        } else {
            too_long = true;
            buf = Vec::new();
        }
        let used = done.unwrap_or(part.len());
        input.consume(used);
        if done.is_some() {
            return Ok(if too_long { Message::TooLong } else { Message::Line(buf) });
        }
    }
}

/// Serves requests from `input` until it ends, writing one response line per request.
pub fn serve(root: &Path, input: impl BufRead, output: impl Write) -> std::io::Result<()> {
    serve_with(Server::new(root)?, input, output)
}

/// Serves requests with `server` (e.g. one with a session).
pub fn serve_with(mut server: Server, mut input: impl BufRead, mut output: impl Write) -> std::io::Result<()> {
    loop {
        let reply = match read_message(&mut input, MAX_MESSAGE_BYTES)? {
            Message::Eof => return Ok(()),
            Message::TooLong => Some(error_line(&J::Null, &rpc(INVALID_REQUEST, format!("message longer than {MAX_MESSAGE_BYTES} bytes")))),
            Message::Line(line) if line.iter().all(u8::is_ascii_whitespace) => None,
            Message::Line(line) => server.handle(&line),
        };
        if let Some(r) = reply {
            output.write_all(r.as_bytes())?;
            output.write_all(b"\n")?;
            output.flush()?;
        }
    }
}

/// `qal mcp`: serves stdin/stdout on a thread with a large stack (the checker and code generator
/// recurse over expressions up to the parser's nesting limit).
/// `session`: the agent's session file (a team store needs one; others ignore it).
pub fn serve_stdio(root: &Path, session: Option<PathBuf>) -> std::io::Result<()> {
    let server = Server::with_session(root, session.as_deref())?;
    let worker = std::thread::Builder::new().name("qal-mcp".into()).stack_size(256 << 20).spawn(move || {
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        serve_with(server, stdin.lock(), stdout.lock())
    })?;
    match worker.join() {
        Ok(r) => r,
        Err(p) => Err(std::io::Error::other(format!("the server stopped: {}", panic_message(&p)))),
    }
}
