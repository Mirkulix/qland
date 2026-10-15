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
use qal_store::Query;
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

/// The compact language reference (≤ 1 page), part of `qal_define`'s description.
pub const REFERENCE: &str = include_str!("reference.md");

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
}

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

impl Server {
    pub fn new(root: &Path) -> std::io::Result<Server> {
        let root = root.canonicalize().map_err(|e| std::io::Error::new(e.kind(), format!("{}: {e}", root.display())))?;
        if !root.is_dir() {
            return Err(std::io::Error::other(format!("{} is not a directory", root.display())));
        }
        Ok(Server { root, client: None })
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
            "tools/list" => Ok(finish(object(json!({ "tools": tools() })))),
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
                if !TOOL_NAMES.contains(&name) {
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

    fn ctx(&self, stdin: Option<String>) -> Ctx {
        Ctx {
            store: StoreAt::Root(self.root.clone()),
            // Never the process's stdin: that is the protocol channel.
            stdin: Some(stdin.unwrap_or_default().into_bytes()),
            timeout: Some(RUN_TIMEOUT),
            test_budget: Some(TEST_BUDGET),
            max_stdout: Some(MAX_STDOUT_BYTES),
        }
    }

    fn call_tool(&self, name: &str, args: &Map<String, J>, client: Option<&str>) -> Reply {
        let mut reply = match self.tool(name, args, client) {
            Ok(r) => r,
            Err(message) => Reply::diagnostics(vec![Diagnostic::new("E0701", "bad_arguments", "", "", message).hint(format!("see the input schema of `{name}` in tools/list"))]),
        };
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
        reply
    }

    fn tool(&self, name: &str, args: &Map<String, J>, client: Option<&str>) -> Result<Reply, String> {
        let a = Args::new(name, args)?;
        let agent = |a: &Args| -> Result<String, String> {
            Ok(match a.string("agent")? {
                Some(s) if s.is_empty() || s.chars().count() > 64 || s.chars().any(invisible) => return Err("`agent` must be 1 to 64 characters without control or formatting characters".into()),
                Some(s) => s,
                None => match client {
                    Some(c) => format!("mcp:{}", c.chars().filter(|c| !invisible(*c)).take(48).collect::<String>()),
                    None => "mcp".into(),
                },
            })
        };
        let ctx = self.ctx(a.string("stdin")?);
        let perms = |a: &Args| -> Result<Permissions, String> { Permissions::parse_in(&a.strings("allow")?.unwrap_or_default(), &self.root) };
        Ok(match name {
            "qal_define" => {
                let src = a.required("source")?;
                if src.len() > MAX_SOURCE_BYTES {
                    return Err(format!("`source` is longer than {MAX_SOURCE_BYTES} bytes; define the code in several calls"));
                }
                ops::define(&ctx, &src, &agent(&a)?)
            }
            "qal_view" => {
                let names = a.names("names")?;
                if names.is_empty() {
                    return Err("`names` must list at least one name or `#<hash>`".into());
                }
                ops::view(&ctx, &names)
            }
            "qal_query" => ops::query(&ctx, &Query { name: a.string("name")?, type_pattern: a.string("type")?, uses: a.string("uses")? }),
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
                ops::run(&ctx, &Source::Store, perms, &entry, &a.strings("args")?.unwrap_or_default())
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
            "qal_rename" => ops::rename(&ctx, &a.required("old")?, &a.required("new")?, &agent(&a)?),
            "qal_history" => ops::history(&ctx, &a.required("name")?),
            "qal_delete" => {
                let mut names = vec![a.required("name")?];
                names.extend(a.names("with")?);
                ops::delete(&ctx, &names, &agent(&a)?)
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
            return Err(format!("unknown argument `{k}` (`{tool}` takes: {})", known.join(", ")));
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
}

const TOOL_NAMES: [&str; 12] = ["qal_define", "qal_test", "qal_prove", "qal_run", "qal_call", "qal_request", "qal_view", "qal_query", "qal_rename", "qal_delete", "qal_history", "qal_build"];

fn tool_params(tool: &str) -> &'static [&'static str] {
    match tool {
        "qal_define" => &["source", "agent"],
        "qal_view" => &["names"],
        "qal_query" => &["name", "type", "uses"],
        "qal_test" => &["names", "seed"],
        "qal_prove" => &["names"],
        "qal_run" => &["entry", "args", "stdin", "allow"],
        "qal_call" => &["fn", "args", "stdin", "allow"],
        "qal_build" => &["entry"],
        "qal_rename" => &["old", "new", "agent"],
        "qal_history" => &["name"],
        "qal_delete" => &["name", "with", "agent"],
        "qal_request" => &["handler", "init", "allow", "requests"],
        _ => &[],
    }
}

/// The tool list (deterministic order).
pub fn tools() -> J {
    let s = |d: &str| json!({"type": "string", "description": d});
    let list = |d: &str| json!({"type": "array", "items": {"type": "string"}, "description": d});
    let allow = list("Effects to grant: \"io\", \"time\", \"rand\", \"fs:<dir>\" (a directory inside the project, relative to its root; hidden entries like .git are never reachable), \"db:mem\" (a fresh in-memory SQLite database), \"db\" or \"db:<name>\" (the project's .qal/db/main.sqlite or <name>.sqlite), \"net:<host>[:port]\" (outgoing HTTP to that host; names must resolve to public addresses, IP literals like \"net:127.0.0.1:9000\" reach exactly that address). Default: none.");
    let allow_request = allow.clone();
    let stdin = s("Text that read_line() and read_all() read (default: empty).");
    let agent = s("Who makes the change, recorded in the history (default: the MCP client's name).");
    let read_only = json!({"readOnlyHint": true, "openWorldHint": false});
    let writes = json!({"readOnlyHint": false, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false});
    let define = format!(
        "Add or replace QAL definitions (types, functions, tests) in the project's store. Nothing is stored unless all of them parse and type-check together with the stored code; then the result lists each name with its hash and status (added, replaced, unchanged, relinked). Otherwise `diagnostics` say what to fix. Contracts are proven on the way (`proofs`): a function whose contract is refuted is not stored (the diagnostic has a counterexample).\n\n{REFERENCE}"
    );
    json!([
        {"name": "qal_define", "title": "Define QAL code", "description": define, "annotations": writes,
         "inputSchema": {"type": "object", "properties": {"source": s("QAL source text with one or more definitions."), "agent": agent}, "required": ["source"], "additionalProperties": false}},
        {"name": "qal_test", "title": "Run QAL tests", "annotations": read_only,
         "description": "Run the stored tests and the generated property tests (all, or the named tests and functions; in a project, all means its own: `std.` tests run when named). A pure function with an `ens`, or that calls a function with `req`, gets a property test: 100 generated inputs that satisfy its `req`; any contract violation fails it, with a shrunk counterexample. Result: each test with passed true/false and the runtime error of a failing one, `properties` (passed, failed, gave_up, skipped), plus passed/failed counts.",
         "inputSchema": {"type": "object", "properties": {"names": list("Test or function names (default: all)."), "seed": {"type": "integer", "minimum": 0, "description": "Seed of the generated inputs (default 0)."}}, "additionalProperties": false}},
        {"name": "qal_prove", "title": "Prove QAL contracts", "annotations": read_only,
         "description": "Prove the `req`/`ens` contracts of stored functions and tests with Z3 (all, or the named ones): every `ens`, and the `req` of every called function at its call site. Status per obligation: proven, refuted (with a counterexample that really violates it), or unknown (with the reason: Int and Bool facts are modelled exactly, other values only approximately; the contract stays checked at runtime). Results are stored per definition hash.",
         "inputSchema": {"type": "object", "properties": {"names": list("Function or test names (default: all).")}, "additionalProperties": false}},
        {"name": "qal_run", "title": "Run a QAL program", "annotations": {"readOnlyHint": false, "destructiveHint": false, "openWorldHint": true},
         "description": "Run a stored entry point `fn(List[Str]) -> Int` (default `main`) with string arguments. Result: exit_code and stdout, or a runtime `error` (contract violation, overflow, timeout after 10 s, ...). Effects the program declares must be granted with `allow`.",
         "inputSchema": {"type": "object", "properties": {"entry": s("Entry point name (default main)."), "args": list("Program arguments."), "stdin": stdin.clone(), "allow": allow.clone()}, "additionalProperties": false}},
        {"name": "qal_call", "title": "Call a QAL function", "annotations": {"readOnlyHint": false, "destructiveHint": false, "openWorldHint": true},
         "description": "Call any stored function with JSON arguments and get its result as JSON, both in the same form: numbers, strings, true/false, null for Unit, arrays for List and Bytes, [[key, value], ...] for Map, objects for records, `None` is \"None\", `Some(3)` is {\"Some\":[3]}. Function values cannot be passed.",
         "inputSchema": {"type": "object", "properties": {"fn": s("Function name."), "args": {"type": "array", "description": "Arguments in parameter order."}, "stdin": stdin, "allow": allow}, "required": ["fn"], "additionalProperties": false}},
        {"name": "qal_request", "title": "Send HTTP requests to a QAL service", "annotations": {"readOnlyHint": false, "destructiveHint": false, "openWorldHint": true},
         "description": "Run HTTP requests against a stored handler `fn(HttpRequest) -> HttpResponse` exactly as `qal serve` would (same limits, same checks, one database transaction per request: committed for status 200-399, else rolled back), without a server. `init` (a `fn() -> Result[Unit, Str]`, e.g. creating tables) runs first; all requests share one database (\"db:mem\": a fresh one per call). Result: `responses`, one per request: status, headers, body, json (if the response is JSON), db (commit/rollback/none), stdout, and the full runtime `error` of a failed request (500/503).",
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
         "description": "List stored definitions {name, kind, hash, signature, effects}, filtered by name prefix, signature pattern (`_` = any one type, e.g. `fn(Str) -> _`) and/or direct use of a name or #hash. No filter lists everything.",
         "inputSchema": {"type": "object", "properties": {"name": s("Name prefix."), "type": s("Signature pattern, e.g. fn(List[Int]) -> Int."), "uses": s("Only definitions that refer to this name or #hash.")}, "additionalProperties": false}},
        {"name": "qal_rename", "title": "Rename a QAL definition", "annotations": writes,
         "description": "Rename a stored definition. No hash changes; everything that refers to it follows the new name; the history moves along.",
         "inputSchema": {"type": "object", "properties": {"old": s("Current name."), "new": s("New name."), "agent": agent.clone()}, "required": ["old", "new"], "additionalProperties": false}},
        {"name": "qal_delete", "title": "Delete a QAL definition", "annotations": {"readOnlyHint": false, "destructiveHint": true, "idempotentHint": false, "openWorldHint": false},
         "description": "Remove a stored name that no other stored definition refers to (otherwise `has_dependents` lists them: delete or redefine those first, or name them in `with`). The history keeps a `deleted` entry and the old version stays viewable by #hash; the name is gone from query, view, test and export and can be defined again. `std.` names cannot be deleted in a project.",
         "inputSchema": {"type": "object", "properties": {"name": s("Name to delete."), "with": list("More names deleted in the same call, e.g. the other members of a mutually recursive group (default: none)."), "agent": agent.clone()}, "required": ["name"], "additionalProperties": false}},
        {"name": "qal_history", "title": "QAL definition history", "annotations": read_only,
         "description": "Every recorded change of a name, oldest first: added, replaced, relinked, renamed, deleted, with hashes, agent and time.",
         "inputSchema": {"type": "object", "properties": {"name": s("Definition name.")}, "required": ["name"], "additionalProperties": false}},
        {"name": "qal_build", "title": "Build a QAL module", "annotations": writes,
         "description": "Compile a stored function into a standalone WebAssembly module `.qal/build/<entry>.wasm` plus `<entry>.manifest.json` (hash, signature, required effects, host imports).",
         "inputSchema": {"type": "object", "properties": {"entry": s("Function to build, usually main.")}, "required": ["entry"], "additionalProperties": false}},
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
pub fn serve(root: &Path, mut input: impl BufRead, mut output: impl Write) -> std::io::Result<()> {
    let mut server = Server::new(root)?;
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
pub fn serve_stdio(root: &Path) -> std::io::Result<()> {
    let root = root.to_path_buf();
    let worker = std::thread::Builder::new().name("qal-mcp".into()).stack_size(256 << 20).spawn(move || {
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        serve(&root, stdin.lock(), stdout.lock())
    })?;
    match worker.join() {
        Ok(r) => r,
        Err(p) => Err(std::io::Error::other(format!("the server stopped: {}", panic_message(&p)))),
    }
}
