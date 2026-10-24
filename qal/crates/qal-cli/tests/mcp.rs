//! Drives `qal mcp` as a real child process over stdio, like an MCP client does.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn example(name: &str) -> String {
    std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples").join(name)).unwrap()
}

struct Client {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    root: PathBuf,
    next_id: i64,
    /// Team mode (Phase 4 §17 regression): the architect's own change set the tools work in.
    change: Option<String>,
}

impl Client {
    /// A server on a fresh, empty project directory.
    fn start() -> Client {
        Client::start_at(Client::fresh_root())
    }

    fn fresh_root() -> PathBuf {
        let root = std::env::temp_dir().join(format!("qal-mcp-test-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::SeqCst)));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    /// A server on the project directory `root`.
    fn start_at(root: PathBuf) -> Client {
        let mut child = Command::new(env!("CARGO_BIN_EXE_qal"))
            .args(["mcp", "--root"])
            .arg(&root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Client { child, stdin, stdout, root, next_id: 0, change: None }
    }

    /// Phase 4 §17: a server on a fresh team store (`qal team init`), as the architect `arch-1`
    /// (keys and session outside the project), initialized and working in its own change set.
    fn architect() -> Client {
        let root = Client::fresh_root();
        let home = root.with_extension("home");
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(&home).unwrap();
        let keys = home.join("keys");
        let qal = |args: &[&str]| {
            let out = Command::new(env!("CARGO_BIN_EXE_qal")).args(args).current_dir(&root).env("QAL_KEY_DIR", &keys).env_remove("QAL_SESSION_FILE").stdin(Stdio::null()).output().unwrap();
            assert!(out.status.success(), "{args:?}: {}", String::from_utf8_lossy(&out.stdout));
        };
        let session = home.join("arch.session");
        qal(&["init"]);
        qal(&["team", "init", "--owner", "boss"]);
        qal(&["team", "add", "arch-1", "--role", "architect"]);
        qal(&["team", "session", "arch-1", "--out", session.to_str().unwrap()]);
        let mut child = Command::new(env!("CARGO_BIN_EXE_qal"))
            .args(["mcp", "--root"])
            .arg(&root)
            .arg("--session-file")
            .arg(&session)
            .env("QAL_KEY_DIR", &keys)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut c = Client { child, stdin, stdout, root, next_id: 0, change: None };
        c.initialize();
        let (err, v) = c.call("qal_change", json!({"action": "open", "title": "the MCP tests, as an architect"}));
        assert!(!err, "{v}");
        c.change = v["change"].as_str().map(str::to_string);
        c
    }

    /// The agent name the history shows for this client's writes.
    fn agent(&self, given: &str) -> String {
        if self.change.is_some() { "arch-1".into() } else { given.into() }
    }

    /// Where `qal_build` writes module `name`.
    fn build_path(&self, name: &str) -> PathBuf {
        let dir = self.root.join(".qal/build");
        match &self.change {
            Some(c) => dir.join(c).join(format!("{name}.wasm")),
            None => dir.join(format!("{name}.wasm")),
        }
    }

    fn send_raw(&mut self, line: &[u8]) {
        let stdin = self.stdin.as_mut().unwrap();
        stdin.write_all(line).unwrap();
        stdin.write_all(b"\n").unwrap();
        stdin.flush().unwrap();
    }

    fn read(&mut self) -> Value {
        let mut line = String::new();
        self.stdout.read_line(&mut line).unwrap();
        assert!(line.ends_with('\n') && !line[..line.len() - 1].contains('\n'), "one message per line: {line:?}");
        serde_json::from_str(&line).unwrap_or_else(|e| panic!("not JSON ({e}): {line}"))
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        self.send_raw(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string().as_bytes());
        let r = self.read();
        assert_eq!(r["id"], id, "{r}");
        assert_eq!(r["jsonrpc"], "2.0");
        r
    }

    fn initialize(&mut self) -> Value {
        let r = self.request("initialize", json!({"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test-client", "version": "1"}}));
        self.send_raw(br#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);
        r
    }

    /// Calls a tool; returns (isError, structuredContent) and checks that the text content is the same JSON.
    fn call(&mut self, tool: &str, args: Value) -> (bool, Value) {
        let r = self.request("tools/call", json!({"name": tool, "arguments": args}));
        let result = &r["result"];
        assert_eq!(result["content"][0]["type"], "text", "{r}");
        let text: Value = serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(text, result["structuredContent"]);
        let mut out = result["structuredContent"].clone();
        if let (Some(c), Value::Object(m)) = (&self.change, &mut out) {
            // Team mode adds the space and the notices to every result (Phase 4 §10.2).
            if tool != "qal_change" {
                assert_eq!(m.remove("space").as_ref().and_then(Value::as_str), Some(c.as_str()), "team results name their space");
            }
            m.remove("notices");
            m.remove("notices_more");
        }
        (result["isError"].as_bool().unwrap(), out)
    }

    /// Closes stdin and waits for the server to exit.
    fn finish(mut self) -> Option<i32> {
        drop(self.stdin.take());
        let start = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status.code();
            }
            assert!(start.elapsed() < Duration::from_secs(10), "server did not exit on EOF");
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn handshake_lists_the_tools_with_the_language_reference() {
    let mut c = Client::start();
    let r = c.initialize();
    assert_eq!(r["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(r["result"]["serverInfo"]["name"], "qal");
    assert!(r["result"]["capabilities"]["tools"].is_object());
    assert!(r["result"]["instructions"].as_str().unwrap().contains("qal_define"));
    assert!(r["result"].get("resultType").is_none(), "legacy results have no resultType");
    // The notification got no answer: the next line answers the ping.
    assert_eq!(c.request("ping", json!({}))["result"], json!({}));
    let tools = c.request("tools/list", json!({}))["result"]["tools"].clone();
    let names: Vec<&str> = tools.as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["qal_define", "qal_test", "qal_prove", "qal_run", "qal_call", "qal_request", "qal_view", "qal_query", "qal_rename", "qal_delete", "qal_history", "qal_build", "qal_recall"]);
    for t in tools.as_array().unwrap() {
        assert_eq!(t["inputSchema"]["type"], "object", "{t}");
        assert!(!t["description"].as_str().unwrap().is_empty());
    }
    let reference = tools[0]["description"].as_str().unwrap();
    assert!(reference.contains("QAL language reference") && reference.contains("alias_reference") && reference.contains("signature_change"));
    // "At most one page": a bounded size.
    // Phase 2 (generics, closures, maps, bytes, proofs), Phase 3a (a "Services" section of 12 lines)
    // Phase 3b (two lines on `std.` names), Phase 6a (one line on effect parameters) and Phase 6b
    // (one line on `qal_run` cases, one on the tests `qal_define` runs) and the deployment tests
    // of 2026-10-23 (the outline to read before a change) and Phase 6b (one line on `qal_recall`,
    // with its `message` filter), the rerun of 2026-10-24 (`cases` on qal_define) and the pressure run of
    // 2026-10-24 (the property tests define runs) and the fused multiply-divide (`a * b * ... / c`
    // exact, on the Int type) and pressure run 5 (E0806, a change without a test) add to it.
    assert!(qal_cli::mcp::REFERENCE.lines().count() <= 76 && qal_cli::mcp::REFERENCE.len() <= 10620, "{} lines, {} bytes", qal_cli::mcp::REFERENCE.lines().count(), qal_cli::mcp::REFERENCE.len());
    // An unknown version gets the newest legacy one.
    let r = c.request("initialize", json!({"protocolVersion": "1999-01-01", "capabilities": {}, "clientInfo": {"name": "x", "version": "1"}}));
    assert_eq!(r["result"]["protocolVersion"], "2025-11-25");
    assert_eq!(c.finish(), Some(0));
}

#[test]
fn an_agent_defines_tests_runs_and_inspects_code() {
    both(defines_tests_runs_and_inspects_code_in);
}

fn defines_tests_runs_and_inspects_code_in(c: &mut Client) {
    let (err, v) = c.call("qal_define", json!({"source": example("age.qal")}));
    assert!(!err, "{v}");
    assert_eq!(v["defined"].as_array().unwrap().len(), 5);
    assert!(c.root.join(".qal/store.db").is_file());
    let (err, v) = c.call("qal_test", json!({}));
    assert_eq!((err, v["passed"].as_u64(), v["failed"].as_u64()), (false, Some(2), Some(0)));
    // Effects must be granted explicitly.
    let (err, v) = c.call("qal_run", json!({"args": [" 42"]}));
    assert_eq!((err, v["error"]["kind"].as_str()), (true, Some("permission_denied")));
    assert_eq!(v["error"]["hints"], json!(["pass `\"allow\": [\"io\"]` to qal_run, qal_call or qal_request"]));
    let (err, v) = c.call("qal_run", json!({"args": [" 42"], "allow": ["io"]}));
    assert_eq!((err, v), (false, json!({"exit_code": 0, "stdout": "ok 42\n"})));
    // A non-zero exit code is a result, not a tool failure.
    let (err, v) = c.call("qal_run", json!({"entry": "main", "args": ["abc"], "allow": ["io"]}));
    assert_eq!((err, v["exit_code"].as_i64()), (false, Some(1)));
    let (err, v) = c.call("qal_call", json!({"fn": "parse_age", "args": [" 7 "]}));
    assert_eq!((err, v), (false, json!({"ok": {"Ok": [7]}, "stdout": ""})));
    let (err, v) = c.call("qal_call", json!({"fn": "parse_age", "args": [7]}));
    assert_eq!((err, v["error"]["kind"].as_str()), (true, Some("bad_args")));
    let (_, v) = c.call("qal_view", json!({"names": ["parse_age"]}));
    assert_eq!(v["views"][0]["deps"][0]["name"], "Err");
    let (_, v) = c.call("qal_query", json!({"type": "fn(Str) -> _"}));
    assert_eq!(v["results"][0]["name"], "parse_age");
    let (err, v) = c.call("qal_rename", json!({"old": "parse_age", "new": "parse_years"}));
    assert!(!err, "{v}");
    let (_, v) = c.call("qal_history", json!({"name": "parse_years"}));
    assert_eq!(v["history"][0]["agent"], c.agent("mcp:test-client"), "the client's name is the default agent (the member in a team)");
    assert_eq!(v["history"][1]["action"], "renamed");
    let (err, v) = c.call("qal_build", json!({"entry": "main"}));
    assert!(!err, "{v}");
    assert!(c.build_path("main").is_file());
    // Diagnostics are data.
    let (err, v) = c.call("qal_define", json!({"source": "fn f(x: Int) -> Bool = x + 1", "agent": "me"}));
    assert_eq!((err, v["diagnostics"][0]["kind"].as_str(), v["diagnostics"][0]["path"].as_str()), (true, Some("type_mismatch"), Some("body")));
    let (err, v) = c.call("qal_test", json!({"names": ["main"]}));
    assert_eq!((err, v["diagnostics"][0]["message"].as_str()), (true, Some("`main` is a function, not a test")));
    // Hints name MCP tools, never CLI commands.
    let hint = v["diagnostics"][0]["hints"][0].as_str().unwrap();
    assert!(hint.contains("qal_call") && !hint.contains("`qal call"), "{hint}");
    let (err, v) = c.call("qal_define", json!({"source": "test t_ok = 1 == 1"}));
    assert!(!err, "{v}");
    let (err, v) = c.call("qal_call", json!({"fn": "t_ok"}));
    let hint = v["error"]["hints"][0].as_str().unwrap();
    assert!(err && hint.contains("qal_test") && !hint.contains("`qal test"), "{hint}");
    let (err, v) = c.call("qal_define", json!({"source": "fn bad() -> Int = nosuchfn(1)"}));
    assert_eq!((err, v["diagnostics"][0]["kind"].as_str()), (true, Some("unknown_name")));
    assert!(v["diagnostics"][0]["hints"][0].as_str().unwrap().contains("qal_define"), "{v}");
    let (err, v) = c.call("qal_view", json!({"names": ["parse_age"]}));
    assert_eq!((err, v["diagnostics"][0]["hints"][0].as_str()), (true, Some("`parse_age` was renamed to `parse_years`")));

}

#[test]
fn programs_read_the_given_stdin_never_the_protocol_channel() {
    let mut c = Client::start();
    c.initialize();
    let src = "fn main(args: List[Str]) -> Int !io = count(0)\nfn count(n: Int) -> Int !io = match read_line() { Some(_) => count(n + 1), None => { print(from_int(n)); 0 } }";
    assert!(!c.call("qal_define", json!({"source": src})).0);
    let (_, v) = c.call("qal_run", json!({"allow": ["io"], "stdin": "a\nb\nc\n"}));
    assert_eq!(v["stdout"], "3\n");
    let (_, v) = c.call("qal_run", json!({"allow": ["io"]}));
    assert_eq!(v["stdout"], "0\n");
    let src = "fn wc(args: List[Str]) -> Int !io = let text = read_all() in { print(from_int(len(text))); len(split(text, \"\\n\")) }";
    assert!(!c.call("qal_define", json!({"source": src})).0);
    let (_, v) = c.call("qal_run", json!({"entry": "wc", "allow": ["io"], "stdin": "x\ny"}));
    assert_eq!((v["stdout"].as_str(), v["exit_code"].as_i64()), (Some("3\n"), Some(2)));
    // The server still answers: nothing of the protocol stream was consumed by the program.
    assert_eq!(c.request("ping", json!({}))["result"], json!({}));
}

#[test]
fn malformed_messages_get_error_responses_and_the_server_keeps_going() {
    let mut c = Client::start();
    c.send_raw(b"{not json");
    let r = c.read();
    assert_eq!((r["id"].clone(), r["error"]["code"].as_i64()), (Value::Null, Some(-32700)));
    c.send_raw(b"\xff\xfe");
    assert_eq!(c.read()["error"]["code"], -32700);
    c.send_raw(br#"[{"jsonrpc":"2.0","id":1,"method":"ping"}]"#);
    assert_eq!(c.read()["error"]["code"], -32600);
    c.send_raw(br#""just a string""#);
    assert_eq!(c.read()["error"]["code"], -32600);
    c.send_raw(br#"{"jsonrpc":"2.0","id":null,"method":"ping"}"#);
    assert_eq!(c.read()["error"]["code"], -32600);
    // Ids are strings or integers (others could not be echoed exactly).
    for id in ["1.5", "123456789012345678901234567890", "true", "{}"] {
        c.send_raw(format!(r#"{{"jsonrpc":"2.0","id":{id},"method":"ping"}}"#).as_bytes());
        let r = c.read();
        assert_eq!((r["id"].clone(), r["error"]["code"].as_i64()), (Value::Null, Some(-32600)), "{id}");
    }
    c.send_raw(br#"{"jsonrpc":"2.0","id":-9223372036854775808,"method":"ping"}"#);
    assert_eq!(c.read()["id"], json!(-9223372036854775808i64));
    c.send_raw(br#"{"jsonrpc":"2.0","id":"a\u0000b","method":"ping"}"#);
    assert_eq!(c.read()["id"], "a\u{0}b");
    c.send_raw(br#"{"jsonrpc":"1.0","id":7,"method":"ping"}"#);
    let r = c.read();
    assert_eq!((r["id"].as_i64(), r["error"]["code"].as_i64()), (Some(7), Some(-32600)));
    c.send_raw(br#"{"jsonrpc":"2.0","id":8,"method":"ping","params":[1]}"#);
    assert_eq!(c.read()["error"]["code"], -32602);
    // Unknown notifications and responses are ignored; blank lines too.
    c.send_raw(br#"{"jsonrpc":"2.0","method":"notifications/whatever","params":{}}"#);
    c.send_raw(br#"{"jsonrpc":"2.0","id":99,"result":{}}"#);
    c.send_raw(b"   ");
    let r = c.request("no/such/method", json!({}));
    assert_eq!(r["error"]["code"], -32601);
    let r = c.request("tools/call", json!({"name": "qal_nope", "arguments": {}}));
    assert_eq!((r["error"]["code"].as_i64(), r["error"]["message"].as_str()), (Some(-32602), Some("Unknown tool: qal_nope")));
    let r = c.request("tools/call", json!({"name": "qal_view", "arguments": "x"}));
    assert_eq!(r["error"]["code"], -32602);
    // Wrong tool arguments are tool errors the agent can fix.
    let (err, v) = c.call("qal_view", json!({"names": "main"}));
    assert_eq!((err, v["diagnostics"][0]["kind"].as_str()), (true, Some("bad_arguments")));
    let (err, v) = c.call("qal_define", json!({"src": "fn f() -> Int = 1"}));
    assert_eq!((err, v["diagnostics"][0]["message"].as_str()), (true, Some("unknown argument `src` (`qal_define` takes: source, agent, entry, cases, allow)")));
    let (err, v) = c.call("qal_define", json!({"source": "x".repeat((1 << 20) + 1)}));
    assert!(err && v["diagnostics"][0]["message"].as_str().unwrap().contains("longer than"));
    for agent in ["evil\u{202E}name", "a\u{61C}b", "tag\u{E0041}", ""] {
        let (err, v) = c.call("qal_define", json!({"source": "fn f() -> Int = 1", "agent": agent}));
        assert_eq!((err, v["diagnostics"][0]["kind"].as_str()), (true, Some("bad_arguments")), "{agent:?}");
    }
    // A message over the size limit is skipped without buffering it; the next one is served.
    let mut huge = br#"{"jsonrpc":"2.0","id":1,"method":"ping","params":{"x":""#.to_vec();
    huge.extend(std::iter::repeat_n(b'a', 5 << 20));
    huge.extend(br#""}}"#);
    c.send_raw(&huge);
    assert_eq!(c.read()["error"]["code"], -32600);
    assert_eq!(c.request("ping", json!({}))["result"], json!({}));
    assert_eq!(c.finish(), Some(0));
}

#[test]
fn many_requests_written_at_once_are_answered_in_order() {
    let mut c = Client::start();
    let mut batch = Vec::new();
    for id in 1..=200 {
        batch.extend(json!({"jsonrpc": "2.0", "id": id, "method": if id % 2 == 0 { "ping" } else { "tools/list" }}).to_string().into_bytes());
        batch.push(b'\n');
    }
    // Written from another thread so that a full pipe cannot deadlock the test.
    let mut stdin = c.stdin.take().unwrap();
    let writer = std::thread::spawn(move || {
        stdin.write_all(&batch).unwrap();
        stdin
    });
    for id in 1..=200 {
        assert_eq!(c.read()["id"], id);
    }
    c.stdin = Some(writer.join().unwrap());
    assert_eq!(c.finish(), Some(0));
}

#[test]
fn modern_requests_are_stateless_and_versioned() {
    let mut c = Client::start();
    let meta = json!({"io.modelcontextprotocol/protocolVersion": "2026-07-28", "io.modelcontextprotocol/clientCapabilities": {}, "io.modelcontextprotocol/clientInfo": {"name": "modern", "version": "1"}});
    let r = c.request("server/discover", json!({"_meta": meta}));
    assert_eq!(r["result"]["resultType"], "complete");
    assert_eq!(r["result"]["supportedVersions"][0], "2026-07-28");
    assert_eq!(r["result"]["_meta"]["io.modelcontextprotocol/serverInfo"]["name"], "qal");
    // No initialize needed.
    let r = c.request("tools/call", json!({"_meta": meta, "name": "qal_define", "arguments": {"source": "fn two() -> Int = 2"}}));
    assert_eq!((r["result"]["resultType"].as_str(), r["result"]["isError"].as_bool()), (Some("complete"), Some(false)));
    let r = c.request("tools/call", json!({"_meta": meta, "name": "qal_history", "arguments": {"name": "two"}}));
    assert_eq!(r["result"]["structuredContent"]["history"][0]["agent"], "mcp:modern");
    let r = c.request("tools/list", json!({"_meta": {"io.modelcontextprotocol/protocolVersion": "1900-01-01", "io.modelcontextprotocol/clientCapabilities": {}}}));
    assert_eq!(r["error"]["code"], -32022);
    assert_eq!(r["error"]["data"]["requested"], "1900-01-01");
    assert_eq!(r["error"]["data"]["supported"][0], "2026-07-28");
    let r = c.request("tools/list", json!({"_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28"}}));
    assert_eq!(r["error"]["code"], -32602, "clientCapabilities is required");
    let r = c.request("server/discover", json!({}));
    assert_eq!(r["error"]["code"], -32602);
}

#[test]
fn runaway_programs_time_out_and_the_server_survives() {
    let mut c = Client::start();
    c.initialize();
    let src = "fn spin(n: Int) -> Int = spin(n + 1)\nfn main(args: List[Str]) -> Int = spin(0)\n\
               fn shout(n: Int) -> Int !io = { print(\"0123456789abcdef\"); shout(n + 1) }\nfn loud(args: List[Str]) -> Int !io = shout(0)\n\
               fn grow(s: Str) -> Int = grow(s ++ s)\nfn hog(args: List[Str]) -> Int = grow(\"x\")\n\
               fn double(s: Str, n: Int) -> Str = if n == 0 then s else double(s ++ s, n - 1)\nfn huge() -> Str = double(\"x\", 25)\n\
               fn quotes(i: Int) -> Str = \"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\\\"\"\nfn many() -> List[Str] = map(range(0, 90000), quotes)\n\
               fn short(s: Str) -> Bool req len(s) < 10 = true\ntest big_value = short(double(\"y\", 22))\ntest fine = 1 == 1\n\
               fn talk(n: Int) -> Int !io = if n == 0 then 0 else { print(\"0123456789abcdef\"); talk(n - 1) }\nfn chatty(args: List[Str]) -> Int !io = talk(100000)\n\
               test forever = spin(0) == 1";
    assert!(!c.call("qal_define", json!({"source": src})).0);
    let start = Instant::now();
    let (err, v) = c.call("qal_run", json!({}));
    assert_eq!((err, v["error"]["kind"].as_str()), (true, Some("timeout")));
    // Output is capped for the agent; memory is limited per run.
    let (err, v) = c.call("qal_run", json!({"entry": "loud", "allow": ["io"]}));
    assert!(err, "{v}");
    assert_eq!(v["stdout"].as_str().unwrap().len(), 64 << 10);
    assert!(v["stdout_truncated"].as_u64().unwrap() > 64 << 10);
    let (err, v) = c.call("qal_run", json!({"entry": "hog"}));
    assert_eq!((err, v["error"]["kind"].as_str()), (true, Some("out_of_memory")));
    // Return values are capped too: one string of 2^25 bytes comes back as 256 KiB plus a marker.
    let (err, v) = c.call("qal_call", json!({"fn": "huge"}));
    assert!(!err, "{v}");
    let s = v["ok"].as_str().unwrap();
    assert!(s.len() < 300 << 10 && s.ends_with("...(+33292288 bytes)"), "{}", &s[s.len() - 30..]);
    // Much output from a program that then succeeds: cut for the agent, not an error.
    let (err, v) = c.call("qal_run", json!({"entry": "chatty", "allow": ["io"]}));
    assert_eq!((err, v["exit_code"].as_i64(), v["stdout"].as_str().map(str::len), v["stdout_truncated"].as_u64()), (false, Some(0), Some(64 << 10), Some(1_700_000)));
    // A result whose JSON would exceed 4 MiB (here 90 000 strings of escaped quotes) is replaced.
    let (err, v) = c.call("qal_call", json!({"fn": "many"}));
    assert_eq!((err, v["diagnostics"][0]["kind"].as_str()), (true, Some("result_too_large")));
    let (err, v) = c.call("qal_test", json!({"names": ["forever"]}));
    assert_eq!((err, v["tests"][0]["error"]["kind"].as_str()), (true, Some("timeout")));
    // One test's huge contract values are dropped; the rest of the report stays.
    let (_, v) = c.call("qal_test", json!({"names": ["big_value", "fine"]}));
    assert_eq!((v["passed"].as_u64(), v["tests"][0]["error"]["kind"].as_str()), (Some(1), Some("contract_violation")));
    assert!(v["tests"][0]["error"].get("values").is_none());
    assert!(v["tests"][0]["error"]["hints"][1].as_str().unwrap().contains("too large to show"), "{}", v["tests"][0]["error"]);
    assert!(start.elapsed() < Duration::from_secs(60), "{:?}", start.elapsed());
    assert_eq!(c.request("ping", json!({}))["result"], json!({}));
}

#[test]
fn file_access_is_confined_to_the_project_without_hidden_entries() {
    let mut c = Client::start();
    c.initialize();
    std::fs::create_dir_all(c.root.join("data")).unwrap();
    std::fs::write(c.root.join("data/in.txt"), "hello").unwrap();
    let src = "fn main(args: List[Str]) -> Int !fs !io = match get(args, 0) { None => 2, Some(p) => match read_file(p) { Ok(s) => { print(s); 0 }, Err(e) => { print(e); 1 } } }";
    assert!(!c.call("qal_define", json!({"source": src})).0);
    let (err, v) = c.call("qal_run", json!({"args": ["data/in.txt"], "allow": ["io", "fs:data"]}));
    assert_eq!((err, v), (false, json!({"exit_code": 0, "stdout": "hello\n"})));
    for bad in ["fs:..", "fs:/", "fs:.qal", "fs:data/../..", "net", "fs", "fs:", "fs: "] {
        let (err, v) = c.call("qal_run", json!({"args": ["x"], "allow": ["io", bad]}));
        assert_eq!((err, v["diagnostics"][0]["kind"].as_str()), (true, Some("bad_arguments")), "{bad}: {v}");
    }
    let (_, v) = c.call("qal_run", json!({"args": [".qal/store.db"], "allow": ["io", "fs:."]}));
    assert_eq!(v["stdout"], "permission denied: `.qal/store.db` is outside the allowed directories\n");
    let (_, v) = c.call("qal_run", json!({"args": ["/etc/passwd"], "allow": ["io", "fs:."]}));
    assert_eq!(v["exit_code"], 1);
}

#[test]
fn reading_tools_do_not_create_a_store() {
    let mut c = Client::start();
    c.initialize();
    for (tool, args) in [("qal_query", json!({})), ("qal_view", json!({"names": ["main"]})), ("qal_test", json!({})), ("qal_run", json!({})), ("qal_history", json!({"name": "x"}))] {
        let (err, v) = c.call(tool, args);
        assert_eq!((err, v["diagnostics"][0]["hints"][0].as_str()), (true, Some("define code with qal_define first; it creates the store")), "{tool}");
    }
    assert!(!c.root.join(".qal").exists());
    assert!(!c.call("qal_define", json!({"source": "fn one() -> Int = 1"})).0);
    assert!(c.root.join(".qal/store.db").is_file());
}

#[test]
fn the_repository_registers_the_server() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let config: Value = serde_json::from_str(&std::fs::read_to_string(repo.join(".mcp.json")).unwrap()).unwrap();
    let qal = &config["mcpServers"]["qal"];
    assert_eq!((qal["command"].as_str(), &qal["args"]), (Some("sh"), &json!(["qal/mcp.sh"])));
    let script = std::fs::read_to_string(repo.join("qal/mcp.sh")).unwrap();
    assert!(script.contains("cargo build --release") && script.contains(">&2") && script.contains("mcp --root"), "{script}");
}

#[test]
fn view_collapses_duplicates_and_is_bounded() {
    let mut c = Client::start();
    c.initialize();
    let big = |n: usize| format!("fn big{n}() -> Str = \"{}\"", "a".repeat(900_000));
    for n in 1..=5 {
        let (err, val) = c.call("qal_define", json!({"source": big(n)}));
        if err {
            panic!("failed on big({n}): {val}");
        }
    }
    let (err, v) = c.call("qal_view", json!({"names": ["big1", "big1", "big1"]}));
    assert!(!err && v["views"].as_array().unwrap().len() == 1, "duplicates collapse");
    let many: Vec<String> = (0..257).map(|i| format!("n{i}")).collect();
    let (err, v) = c.call("qal_view", json!({"names": many}));
    assert_eq!((err, v["diagnostics"][0]["kind"].as_str()), (true, Some("bad_arguments")));
    let names: Vec<&str> = (0..200).flat_map(|_| ["big1", "big2", "big3", "big4", "big5"]).take(256).collect();
    let start = Instant::now();
    let (err, v) = c.call("qal_view", json!({"names": names}));
    assert_eq!((err, v["diagnostics"][0]["kind"].as_str()), (true, Some("result_too_large")), "{}", v.to_string().len());
    assert!(start.elapsed() < Duration::from_secs(if cfg!(debug_assertions) { 10 } else { 2 }), "{:?}", start.elapsed());
    let (err, v) = c.call("qal_test", json!({"names": many_names(257)}));
    assert_eq!((err, v["diagnostics"][0]["kind"].as_str()), (true, Some("bad_arguments")));
}

fn many_names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("t{i}")).collect()
}

#[test]
fn echoed_user_input_in_diagnostics_is_truncated() {
    let mut c = Client::start();
    c.initialize();
    assert!(!c.call("qal_define", json!({"source": "fn one() -> Int = 1"})).0);
    let (err, v) = c.call("qal_view", json!({"names": ["x".repeat(100_000)]}));
    assert!(err, "{}", v.to_string().len());
    assert_eq!(v["diagnostics"][0]["kind"], "unknown_name");
    assert!(v.to_string().len() < 5_000, "{}", v.to_string().len());
    let (err, v) = c.call("qal_query", json!({"type": "(".repeat(2_000_000)}));
    assert!(err);
    assert_ne!(v["diagnostics"][0]["kind"], "result_too_large");
    assert!(v.to_string().len() < 5_000, "{}", v.to_string().len());
}

#[test]
fn a_missing_root_is_named_in_the_error() {
    let out = Command::new(env!("CARGO_BIN_EXE_qal")).args(["mcp", "--root", "/no/such/qal-root"]).stdin(Stdio::null()).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("/no/such/qal-root"), "{}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn reference_and_launcher_are_precise() {
    let r = qal_cli::mcp::REFERENCE;
    for needle in ["rand_int(lo, hi) -> Int !rand (both ends included; lo <= hi", "!io !fs", "narrowest", "fold(xs, 0, step)", "quadratic", "`r` cannot be a definition name", "no list patterns", "every field", "read_all() -> Str !io", "sort_str(xs)", "< <= > >= (Int, Float, Str)", "from_char_code(n) -> Str", "(not String)", "a `let` in a sequence item ends at its `;`", "not `map(xs, trim)`", "products of large inputs overflow and trap"] {
        assert!(r.contains(needle), "{needle}");
    }
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let script = std::fs::read_to_string(repo.join("qal/mcp.sh")).unwrap();
    assert!(script.contains("CDPATH=") && script.contains("build failed"), "{script}");
}

#[test]
fn a_committed_define_is_never_reported_as_too_large() {
    both(a_committed_define_is_never_reported_as_too_large_in);
}

fn a_committed_define_is_never_reported_as_too_large_in(c: &mut Client) {
    // Review finding I3: one proof report and one warning per call site made the result of a
    // committed `qal_define` exceed the result limit, so it was reported as an error.
    let req = format!("len(s) > 0 || s == \"{}\"", "a".repeat(4000));
    let src = format!("fn g(s: Str) -> Int req {req} = 1\nfn f(s: Str) -> Int = len([{}])", vec!["g(s)"; 300].join(", "));
    let (err, v) = c.call("qal_define", json!({"source": src}));
    assert!(!err, "{}", v.to_string().chars().take(500).collect::<String>());
    assert_eq!(v["defined"].as_array().map(Vec::len), Some(2));
    // The reply was cut somewhere: since the prover models Str, the first call site's `req` is
    // the only unproven one (the later ones follow from it), so the proof reports are what
    // overflows now, not the warnings.
    let omitted = v["warnings_omitted"].as_u64().unwrap_or(0) + v["proofs_omitted"].as_u64().unwrap_or(0);
    assert!(omitted > 0, "{}", v.to_string().chars().take(500).collect::<String>());
    assert!(v.to_string().len() < 3 << 20);
    assert!(!c.call("qal_view", json!({"names": ["f"]})).0);

}

#[test]
fn agents_get_proofs_counterexamples_and_property_tests() {
    both(get_proofs_counterexamples_and_property_tests_in);
}

fn get_proofs_counterexamples_and_property_tests_in(c: &mut Client) {
    let src = "fn abs(x: Int) -> Int ens r >= 0 = if x < 0 then -x else x\n\
               fn pos(x: Int) -> Int req x > 0 = x\n\
               fn head(xs: List[Int]) -> Int ens r >= 0 = match get(sort_int(xs), 0) { Some(x) => x, None => 0 }\n\
               test t = pos(abs(-3) + 1) == 4";
    let (err, v) = c.call("qal_define", json!({"source": src}));
    assert!(!err, "{v}");
    let statuses: Vec<(&str, &str)> = v["proofs"].as_array().unwrap().iter().map(|r| (r["def"].as_str().unwrap(), r["status"].as_str().unwrap())).collect();
    assert_eq!(statuses, [("abs", "proven"), ("head", "unknown"), ("t", "proven")]);
    assert_eq!((v["warnings"][0]["code"].as_str(), v["warnings"][0]["def"].as_str(), v["warnings"].as_array().map(Vec::len)), (Some("E0802"), Some("head"), Some(2)));
    // The define already runs the property test of `head`: the counterexample the proof could not decide.
    assert_eq!((v["warnings"][1]["code"].as_str(), v["warnings"][1]["def"].as_str()), (Some("E0803"), Some("head")), "{v}");
    // A refuted contract: nothing is stored, the diagnostic carries the counterexample.
    let (err, v) = c.call("qal_define", json!({"source": "fn dec(x: Int) -> Int ens r >= 0 = x - 1"}));
    assert!(err);
    assert_eq!((v["diagnostics"][0]["code"].as_str(), v["diagnostics"][0]["kind"].as_str()), (Some("E0801"), Some("contract_refuted")));
    assert!(v["diagnostics"][0]["counterexample"]["x"].as_i64().unwrap() <= 0, "{v}");
    assert_eq!(c.call("qal_view", json!({"names": ["dec"]})).1["diagnostics"][0]["kind"], "unknown_name");
    // qal_prove reuses the stored results.
    let (err, v) = c.call("qal_prove", json!({}));
    assert_eq!((err, v["proven"].as_u64(), v["refuted"].as_u64(), v["unknown"].as_u64()), (false, Some(2), Some(0), Some(1)));
    let (err, v) = c.call("qal_prove", json!({"names": ["nope"]}));
    assert_eq!((err, v["diagnostics"][0]["kind"].as_str()), (true, Some("unknown_name")));
    // The property test of `head` finds the counterexample the proof could not decide.
    let (err, v) = c.call("qal_test", json!({}));
    assert!(err, "{v}");
    assert_eq!((v["passed"].as_u64(), v["failed"].as_u64(), v["failed_properties"].as_u64()), (Some(1), Some(0), Some(1)));
    let props: Vec<(&str, &str)> = v["properties"].as_array().unwrap().iter().map(|p| (p["def"].as_str().unwrap(), p["status"].as_str().unwrap())).collect();
    assert_eq!(props, [("abs", "passed"), ("head", "failed")]);
    assert_eq!(v["properties"][1]["counterexample"], json!({"xs": [-1]}));
    let (_, v) = c.call("qal_test", json!({"names": ["head"], "seed": 7}));
    assert_eq!((v["properties"][0]["seed"].as_u64(), v["tests"].as_array().map(Vec::len)), (Some(7), Some(0)));
    assert_eq!(c.call("qal_test", json!({"seed": -1})).1["diagnostics"][0]["kind"], "bad_arguments");
    // Structured JSON arguments: maps, records, variants, bytes.
    let src = "type P = { name: Str, n: Int }\nfn total(m: Map[Str, P], b: Bytes, o: Option[Int]) -> Int = fold(map_values(m), bytes_len(b), fn(acc, p) => acc + p.n) + (match o { Some(k) => k, None => 0 })";
    assert!(!c.call("qal_define", json!({"source": src})).0);
    let (err, v) = c.call("qal_call", json!({"fn": "total", "args": [[["a", {"name": "x", "n": 2}], ["b", {"name": "y", "n": 3}]], [1, 2, 3], {"Some": [10]}]}));
    assert_eq!((err, &v["ok"]), (false, &json!(18)), "{v}");
    let (err, v) = c.call("qal_call", json!({"fn": "total", "args": [[], [300], "None"]}));
    assert_eq!((err, v["error"]["message"].as_str()), (true, Some("`b`: expected a byte (0..=255), got 300")));

}

#[test]
fn agents_use_an_in_memory_database_or_the_projects_file() {
    let mut c = Client::start();
    c.initialize();
    let src = "fn setup() -> Result[Int, DbError] !db = db_exec(\"CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, body TEXT NOT NULL)\", [])\n\
               fn note(b: Str) -> Int !db = match setup() { Ok(_) => match db_exec(\"INSERT INTO notes (body) VALUES (?)\", [SqlText(b)]) { Ok(n) => n, Err(_) => 0 - 1 }, Err(_) => 0 - 2 }\n\
               fn notes() -> Int !db = match db_query(\"SELECT count(*) FROM notes\", []) { Ok(r) => len(r.rows), Err(DbSql(m)) => 0 - len(m), Err(_) => 0 - 1 }";
    let (err, v) = c.call("qal_define", json!({"source": src}));
    assert!(!err, "{v}");
    let (err, v) = c.call("qal_call", json!({"fn": "note", "args": ["x"]}));
    assert_eq!((err, v["error"]["hints"].clone()), (true, json!(["pass `\"allow\": [\"db:mem\"]` to qal_run, qal_call or qal_request"])));
    let (err, v) = c.call("qal_call", json!({"fn": "note", "args": ["x"], "allow": ["db:mem"]}));
    assert_eq!((err, v), (false, json!({"ok": 1, "stdout": "", "db": "commit"})));
    assert!(!c.root.join(".qal/db").exists(), "db:mem leaves no file");
    let (err, v) = c.call("qal_call", json!({"fn": "note", "args": ["y"], "allow": ["db"]}));
    assert_eq!((err, v["ok"].clone()), (false, json!(1)));
    assert!(c.root.join(".qal/db/main.sqlite").is_file(), "`db` is the project's .qal/db/main.sqlite");
    let (_, v) = c.call("qal_call", json!({"fn": "notes", "allow": ["db"]}));
    assert_eq!(v["ok"], json!(1));
    let (err, v) = c.call("qal_call", json!({"fn": "notes", "allow": ["db", "db:mem"]}));
    assert!(err && v["diagnostics"][0]["message"].as_str().unwrap().contains("only one database"), "{v}");
    let (err, v) = c.call("qal_define", json!({"source": "fn bad(t: Str) -> Result[Int, DbError] !db = db_exec(\"DELETE FROM notes WHERE body = '\" ++ t ++ \"'\", [])"}));
    assert_eq!((err, v["diagnostics"][0]["code"].as_str()), (true, Some("E0241")));
}

fn fixture_app() -> String {
    std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../qal-serve/tests/fixtures/app.qal")).unwrap()
}

#[test]
fn agents_test_services_with_qal_request() {
    both(test_services_with_qal_request_in);
}

fn test_services_with_qal_request_in(c: &mut Client) {
    let (err, v) = c.call("qal_define", json!({"source": fixture_app()}));
    assert!(!err, "{v}");
    let requests = json!([
        {"method": "POST", "path": "/todos", "body": "milk"},
        {"method": "POST", "path": "/fail"},
        {"path": "/todos", "json": null},
        {"path": "/todos/1"},
        {"method": "HEAD", "path": "/todos"},
        {"path": "/trap"},
        {"path": "/print"},
        {"method": "PUT", "path": "/echo", "query": "a=1", "headers": [{"name": "X-Test", "value": " v "}, {"name": "Connection", "value": "close"}], "json": {"k": [1, 2]}},
    ]);
    let (err, v) = c.call("qal_request", json!({"init": "setup", "allow": ["db:mem", "io"], "requests": requests}));
    assert!(!err, "{v}");
    let r = v["responses"].as_array().unwrap();
    assert_eq!(r.len(), 8);
    assert_eq!((r[0]["status"].as_u64(), r[0]["db"].as_str(), r[0]["json"].clone()), (Some(201), Some("commit"), json!({"id": 1, "title": "milk"})));
    assert!(r[0]["headers"].as_array().unwrap().contains(&json!({"name": "location", "value": "/todos/1"})), "{}", r[0]);
    assert_eq!((r[1]["status"].as_u64(), r[1]["db"].as_str()), (Some(422), Some("rollback")));
    assert_eq!(r[2]["json"], json!([{"id": 1, "title": "milk"}]), "the 422 rolled its insert back; requests share the database");
    assert_eq!(r[3]["body"], r#"{"id":1,"title":"milk"}"#);
    assert_eq!((r[4]["status"].as_u64(), r[4]["body"].as_str()), (Some(200), Some("")));
    assert_eq!((r[5]["status"].as_u64(), r[5]["error"]["kind"].as_str(), r[5]["db"].as_str()), (Some(500), Some("int_overflow"), Some("rollback")));
    assert_eq!(r[6]["stdout"], "hello from the handler\n");
    assert_eq!(r[7]["body"], "PUT /echo?a=1 x-test,content-type,content-length {\"k\":[1,2]}", "lowercase names, no hop-by-hop headers");
    // A fresh in-memory database per call; the project's file only with "db".
    let (_, v) = c.call("qal_request", json!({"init": "setup", "allow": ["db:mem", "io"], "requests": [{"path": "/todos"}]}));
    assert_eq!(v["responses"][0]["json"], json!([]), "{v}");
    assert!(!c.root.join(".qal/db").exists());
    // Errors before anything runs.
    let (err, v) = c.call("qal_request", json!({"requests": [{"path": "/health"}]}));
    assert_eq!((err, v["error"]["kind"].as_str()), (true, Some("permission_denied")));
    assert_eq!(v["error"]["hints"], json!(["pass `\"allow\": [\"db:mem\", \"io\"]` to qal_run, qal_call or qal_request"]));
    let (err, v) = c.call("qal_request", json!({"handler": "text", "allow": ["db:mem", "io"], "requests": []}));
    assert_eq!((err, v["error"]["code"].as_str()), (true, Some("E0520")));
    let (err, v) = c.call("qal_request", json!({"init": "nope", "allow": ["db:mem", "io"], "requests": []}));
    assert_eq!((err, v["error"]["kind"].as_str()), (true, Some("bad_entry")));
    let (err, v) = c.call("qal_request", json!({"allow": ["db:mem", "io"], "requests": [{"path": "todos"}]}));
    assert!(err && v["diagnostics"][0]["message"].as_str().unwrap().contains("must start with `/`"), "{v}");
    let (err, v) = c.call("qal_request", json!({"allow": ["db:mem", "io"], "requests": [{"path": "/", "verb": "GET"}]}));
    assert!(err && v["diagnostics"][0]["message"].as_str().unwrap().contains("unknown request field `verb`"), "{v}");
    let many: Vec<Value> = (0..51).map(|_| json!({"path": "/health"})).collect();
    let (err, v) = c.call("qal_request", json!({"allow": ["db:mem", "io"], "requests": many}));
    assert!(err && v["diagnostics"][0]["message"].as_str().unwrap().contains("at most 50"), "{v}");
    let (err, v) = c.call("qal_request", json!({"allow": ["db:mem", "io"], "requests": [{"method": "POST", "path": "/todos", "body": "x".repeat((1 << 20) + 1)}]}));
    assert_eq!((err, v["responses"][0]["status"].as_u64(), v["responses"][0]["error"]["code"].as_str()), (false, Some(413), Some("E0800")));
    let reference = qal_cli::mcp::REFERENCE;
    for needle in ["fn handle(rq: HttpRequest) -> HttpResponse", "string literal with `?` placeholders", "committed only for status 200-399", "\"db:mem\"", "qal_query name \"std.\""] {
        assert!(reference.contains(needle), "{needle}");
    }

}

#[test]
fn the_std_project_is_its_own_mcp_root() {
    // `sh qal/mcp.sh std`: the root is the std project, whose store `qal init --std` made.
    let root = Client::fresh_root();
    let out = Command::new(env!("CARGO_BIN_EXE_qal")).arg("init").arg("--std").current_dir(&root).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stdout));
    let mut c = Client::start_at(root);
    c.initialize();
    let src = "type std.zz.Json = std.zz.Null | std.zz.Int(Int)\n\
               fn std.zz.int(n: Int) -> std.zz.Json = std.zz.Int(n)\n\
               test std.zz.t_int = match std.zz.int(1) { std.zz.Int(n) => n == 1, std.zz.Null => false }";
    let (err, v) = c.call("qal_define", json!({"source": src, "agent": "std-json-1"}));
    assert!(!err && v.get("std_imported").is_none(), "{v}");
    let (err, v) = c.call("qal_define", json!({"source": "fn helper() -> Int = 1"}));
    assert!(err && v["diagnostics"][0]["code"] == "E0240", "{v}");
    let (_, v) = c.call("qal_query", json!({"name": "std.zz."}));
    let names: Vec<&str> = v["results"].as_array().unwrap().iter().map(|e| e["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["std.zz.Json", "std.zz.int", "std.zz.t_int"]);
    let (_, v) = c.call("qal_history", json!({"name": "std.zz.int"}));
    assert_eq!(v["history"][0]["agent"], "std-json-1");
    let (err, v) = c.call("qal_test", json!({"names": ["std.zz.t_int"]}));
    assert!(!err && v["passed"] == 1, "{v}");
    let (_, v) = c.call("qal_view", json!({"names": ["std.zz.int"]}));
    assert_eq!(v["views"][0]["text"], "fn std.zz.int(n: Int) -> std.zz.Json\n= std.zz.Int(n)");
    assert_eq!(c.finish(), Some(0));
}

#[test]
fn a_projects_first_write_imports_the_std_package() {
    let mut c = Client::start();
    c.initialize();
    let std_hash = qal_store::package::parse(qal_cli::STD_PACKAGE).unwrap().hash;
    let (err, v) = c.call("qal_define", json!({"source": "fn f() -> Int = 1"}));
    assert!(!err && v["std_imported"] == std_hash.as_str(), "{v}");
    let (err, v) = c.call("qal_define", json!({"source": "fn g() -> Int = 2"}));
    assert!(!err && v.get("std_imported").is_none(), "{v}");
    let (err, v) = c.call("qal_define", json!({"source": "fn std.json.parse(s: Str) -> Int = 1"}));
    assert!(err && v["diagnostics"][0]["code"] == "E0240" && v["diagnostics"][0]["kind"] == "reserved_name", "{v}");
    let (err, v) = c.call("qal_rename", json!({"old": "g", "new": "std.json.g"}));
    assert!(err && v["diagnostics"][0]["code"] == "E0240", "{v}");
    assert_eq!(c.finish(), Some(0));
}

#[test]
fn agents_delete_names_nothing_refers_to() {
    both(delete_names_nothing_refers_to_in);
}

fn delete_names_nothing_refers_to_in(c: &mut Client) {
    assert!(!c.call("qal_define", json!({"source": "fn f(x: Int) -> Int = g(x)\nfn g(x: Int) -> Int = if x <= 0 then 0 else f(x - 1)\nfn h() -> Int = f(1)"})).0);
    let (err, v) = c.call("qal_delete", json!({"name": "f"}));
    assert!(err && v["diagnostics"][0]["kind"] == "has_dependents", "{v}");
    let (err, v) = c.call("qal_delete", json!({"name": "h", "agent": "cleaner"}));
    assert!(!err && v["deleted"][0]["name"] == "h", "{v}");
    // A mutually recursive group goes together.
    let (err, v) = c.call("qal_delete", json!({"name": "f", "with": ["g"]}));
    assert!(!err && v["deleted"].as_array().map(Vec::len) == Some(2), "{v}");
    let (_, v) = c.call("qal_history", json!({"name": "h"}));
    assert_eq!((v["history"][1]["action"].as_str(), v["history"][1]["agent"].as_str()), (Some("deleted"), Some(c.agent("cleaner").as_str())));
    let (err, v) = c.call("qal_delete", json!({"name": "std.json.parse"}));
    assert!(err && v["diagnostics"][0]["code"] == "E0240", "{v}");

}

#[test]
fn std_agents_cannot_rename_released_names() {
    // Re-review N1: over MCP there is no way to break the released std.
    let root = Client::fresh_root();
    let out = Command::new(env!("CARGO_BIN_EXE_qal")).arg("init").arg("--std").current_dir(&root).output().unwrap();
    assert!(out.status.success());
    let mut c = Client::start_at(root);
    c.initialize();
    let (err, v) = c.call("qal_rename", json!({"old": "std.url.Param", "new": "std.url.Param2"}));
    assert!(err && v["diagnostics"][0]["code"] == "E0244", "{v}");
    let (err, v) = c.call("qal_rename", json!({"old": "std.url.Param", "new": "std.url.Param2", "breaking": true}));
    assert!(err && v["diagnostics"][0]["kind"] == "bad_arguments", "{v}");
    let (err, v) = c.call("qal_delete", json!({"name": "std.db.test_first"}));
    assert!(err && v["diagnostics"][0]["code"] == "E0244", "{v}");
    // New names stay free.
    assert!(!c.call("qal_define", json!({"source": "fn std.zz.a() -> Int = 1", "agent": "std-zz-1"})).0);
    let (err, v) = c.call("qal_rename", json!({"old": "std.zz.a", "new": "std.zz.b"}));
    assert!(!err, "{v}");
    assert_eq!(c.finish(), Some(0));
}

#[test]
fn query_outline_is_the_compact_view_for_a_change() {
    both(query_outline_is_the_compact_view_for_a_change_in);
}

fn query_outline_is_the_compact_view_for_a_change_in(c: &mut Client) {
    let src = "fn inc(n: Int) -> Int req n >= 0 ens r > n = n + 1\nfn twice(n: Int) -> Int req n >= 0 = inc(inc(n))\ntest t_inc = inc(1) == 2";
    let (err, v) = c.call("qal_define", json!({"source": src}));
    assert!(!err, "{v}");
    // One call tells an agent what `inc` promises, needs, is relied on by and is covered by,
    // without the text of any body.
    let (err, v) = c.call("qal_query", json!({"name": "inc", "outline": true}));
    assert!(!err, "{v}");
    let e = &v["results"][0];
    assert_eq!((e["name"].as_str(), &e["contracts"], &e["uses"], &e["used_by"], &e["tests"]), (Some("inc"), &json!(["req n >= 0", "ens r > n"]), &json!([]), &json!(["twice"]), &json!(["t_inc"])), "{v}");
    // The outline is the project's own code, for reading: no `std.` names unless asked for, no hashes.
    let (_, v) = c.call("qal_query", json!({"outline": true}));
    let names: Vec<&str> = v["results"].as_array().unwrap().iter().map(|e| e["name"].as_str().unwrap()).collect();
    assert!(names == ["inc", "t_inc", "twice"] && v["results"][0].get("hash").is_none(), "{v}");
    let (_, v) = c.call("qal_query", json!({"outline": true, "name": "std."}));
    assert!(v["results"].as_array().is_some_and(|r| !r.is_empty() && r.iter().all(|e| e["name"].as_str().unwrap().starts_with("std."))), "{v}");
    // Without the flag the entries stay compact in the other direction: no outline fields.
    let (_, v) = c.call("qal_query", json!({"name": "inc"}));
    assert!(v["results"][0].get("contracts").is_none() && v["results"][0].get("used_by").is_none(), "{v}");
    let (err, v) = c.call("qal_query", json!({"outline": "yes"}));
    assert!(err && v["diagnostics"][0]["message"].as_str().unwrap().contains("`outline` must be a boolean"), "{v}");
}

#[test]
fn run_takes_a_list_of_cases() {
    both(run_takes_a_list_of_cases_in);
}

fn run_takes_a_list_of_cases_in(c: &mut Client) {
    let src = "fn main(args: List[Str]) -> Int !io = { print(read_all() ++ join(args, \",\")); if len(args) == 0 then 1 else 0 }";
    let (err, v) = c.call("qal_define", json!({"source": src}));
    assert!(!err, "{v}");
    // One call runs every case with its own arguments and input, in order.
    let (err, v) = c.call("qal_run", json!({"cases": [{"args": ["a"], "stdin": "x"}, {"stdin": "y"}], "allow": ["io"]}));
    assert_eq!(v["results"][0], json!({"args": ["a"], "exit_code": 0, "stdout": "xa\n"}), "{v}");
    assert_eq!(v["results"][1], json!({"args": [], "exit_code": 1, "stdout": "y\n"}), "{v}");
    // An exit code is data, not an error: a program whose exit code is part of its specification
    // (1 for "something was rejected") must not turn its own correct run into an error reply.
    assert!(!err, "{v}");
    let (err, v) = c.call("qal_run", json!({"cases": [{"args": ["a", "b"]}, {"args": ["c"], "stdin": ""}], "allow": ["io"]}));
    assert_eq!((err, v["results"][0]["stdout"].as_str(), v["results"][0].get("stdin")), (false, Some("a,b\n"), None), "{v}");
    // A case's runtime error is its result; the other cases still run.
    let (err, v) = c.call("qal_run", json!({"cases": [{"args": ["a"]}, {"args": ["b"]}]}));
    assert_eq!((err, v["results"][0]["error"]["kind"].as_str(), v["results"].as_array().map(Vec::len)), (true, Some("permission_denied"), Some(2)), "{v}");
    // With `cases`, the arguments and input belong to the cases.
    for bad in [json!({"cases": [{"args": ["a"]}], "args": ["z"]}), json!({"cases": [{}], "stdin": "z"}), json!({"cases": []}), json!({"cases": [{"argv": []}]}), json!({"cases": vec![json!({}); 65]})] {
        let (err, v) = c.call("qal_run", bad.clone());
        assert_eq!((err, v["diagnostics"][0]["code"].as_str()), (true, Some("E0701")), "{bad}: {v}");
    }
    // Without `cases`, `qal_run` is unchanged.
    let (err, v) = c.call("qal_run", json!({"args": ["q"], "stdin": "p", "allow": ["io"]}));
    assert_eq!((err, v), (false, json!({"exit_code": 0, "stdout": "pq\n"})));
}

#[test]
fn define_runs_the_property_tests_of_what_it_defines() {
    both(define_runs_the_property_tests_of_what_it_defines_in);
}

#[test]
fn define_warns_on_a_change_without_a_test() {
    both(define_warns_on_a_change_without_a_test_in);
}

fn define_warns_on_a_change_without_a_test_in(c: &mut Client) {
    // Pressure run 5 of the living program (2026-10-24): a change request changed `cancel`, the
    // agent wrote no test for the new rule, and a cancelled backorder vanished from the orders. The
    // Rust agents wrote a test for each new rule. A define that changes a stored function and
    // defines no test now says so (E0806); a first definition does not.
    let untested = |v: &serde_json::Value| v["warnings"].as_array().and_then(|ws| ws.iter().find(|w| w["code"] == "E0806")).cloned();
    let (err, v) = c.call("qal_define", json!({"source": "fn fee(n: Int) -> Int = n + 1"}));
    assert!(!err && untested(&v).is_none(), "{v}");
    let (err, v) = c.call("qal_define", json!({"source": "fn fee(n: Int) -> Int = n + 2"}));
    assert!(!err && v["defined"][0]["name"] == "fee", "{v}");
    let w = untested(&v).unwrap_or_default();
    assert!(w["def"] == "fee" && w["kind"] == "untested_change", "{v}");
    assert!(w["hints"].as_array().unwrap().iter().any(|h| h.as_str().unwrap().contains("test")), "{v}");
    // The same change with a test for it: no warning.
    let (err, v) = c.call("qal_define", json!({"source": "fn fee(n: Int) -> Int = n + 3\ntest fee_adds_three = fee(1) == 4"}));
    assert!(!err && untested(&v).is_none(), "{v}");
    // A define that only adds a test is no change.
    let (err, v) = c.call("qal_define", json!({"source": "test fee_zero = fee(0) == 3"}));
    assert!(!err && untested(&v).is_none(), "{v}");
}

fn define_runs_the_property_tests_of_what_it_defines_in(c: &mut Client) {
    // The pressure run of the living program (2026-10-24): `price * qty * 95` overflowed on large
    // inputs, the `ens` held whenever the function returned, and nobody ran qal_test. Now the
    // define reply carries the property test of what it defined and a warning for its traps.
    let (err, v) = c.call("qal_define", json!({"source": "fn total(price: Int, qty: Int) -> Int ens r == price * qty * 95 / 100 = price * qty * 95 / 100"}));
    assert!(!err, "{v}");
    let p = &v["properties"][0];
    assert!(p["def"] == "total" && p["traps"].as_u64().unwrap_or(0) > 0, "{v}");
    let w = v["warnings"].as_array().and_then(|ws| ws.iter().find(|w| w["code"] == "E0804")).cloned().unwrap_or_default();
    assert!(w["def"] == "total" && w["message"].as_str().unwrap().contains("overflow"), "{v}");
    assert!(w["hints"].as_array().unwrap().iter().any(|h| h.as_str().unwrap().contains("req")), "{v}");
    // `price * qty * 95 / 100` is one exact operation (2026-10-24, fused multiply-divide): only the
    // quotient can overflow, so there is no E0805 for its product.
    assert!(!v["warnings"].as_array().is_some_and(|ws| ws.iter().any(|w| w["code"] == "E0805")), "{v}");
    // The plain products of unbounded Int parameters are named (E0805): the domain lives in no `req`.
    // A parenthesised product inside the factors of a fused division is a plain one.
    for (src, names) in [("fn area(w: Int, h: Int) -> Int = w * h", ["w", "h"]), ("fn part(price: Int, qty: Int) -> Int = 95 * (price * qty) / 100", ["price", "qty"])] {
        let (err, v) = c.call("qal_define", json!({"source": src}));
        assert!(!err, "{v}");
        let w = v["warnings"].as_array().and_then(|ws| ws.iter().find(|w| w["code"] == "E0805")).cloned().unwrap_or_default();
        assert!(names.iter().all(|n| w["message"].as_str().unwrap_or("").contains(n)), "{src}: {v}");
    }
    // With a `req` that bounds the inputs to what fits, there is no trap and no warning.
    let (err, v) = c.call("qal_define", json!({"source": "fn total(price: Int, qty: Int) -> Int req price >= 0 && price <= 1000000 && qty >= 0 && qty <= 1000000 ens r == price * qty * 95 / 100 = price * qty * 95 / 100"}));
    assert!(!err && v["properties"][0]["traps"] == 0, "{v}");
    assert!(!v["warnings"].as_array().is_some_and(|ws| ws.iter().any(|w| w["code"] == "E0804" || w["code"] == "E0805")), "{v}");
    // The domain of the pressure run, 9-digit amounts: `price * qty * 95` does not fit in 64 bits,
    // but the fused division is exact, so no input inside the domain traps.
    let (err, v) = c.call("qal_define", json!({"source": "fn total(price: Int, qty: Int) -> Int req price >= 0 && price <= 999999999 && qty >= 0 && qty <= 999999999 ens r == price * qty * 95 / 100 = price * qty * 95 / 100"}));
    assert!(!err && v["properties"][0]["traps"] == 0, "{v}");
    assert!(!v["warnings"].as_array().is_some_and(|ws| ws.iter().any(|w| w["code"] == "E0804" || w["code"] == "E0805")), "{v}");
    // A bounded domain whose plain product still overflows: the trap shows up on inputs inside it.
    let (_, v) = c.call("qal_define", json!({"source": "fn total(price: Int, qty: Int) -> Int req price >= 0 && price <= 999999999 && qty >= 0 && qty <= 999999999 ens r == price * qty * 95 = price * qty * 95"}));
    assert!(v["warnings"].as_array().is_some_and(|ws| ws.iter().any(|w| w["code"] == "E0804" && w["def"] == "total")), "{v}");
    // A failing property is reported as a warning too; the definition stays stored and the call is not an error.
    // (A contract the prover can refute is rejected before that, E0801; maps it only approximates.)
    let (err, v) = c.call("qal_define", json!({"source": "fn small(m: Map[Int, Int]) -> Int ens r <= 1 = map_size(m)"}));
    assert!(!err && v["defined"][0]["name"] == "small", "{v}");
    assert!(v["warnings"].as_array().unwrap().iter().any(|w| w["code"] == "E0803" && w["def"] == "small"), "{v}");
}

#[test]
fn define_runs_cases_after_storing_the_code() {
    both(define_runs_cases_after_storing_the_code_in);
}

fn define_runs_cases_after_storing_the_code_in(c: &mut Client) {
    // One call defines the program and runs the examples: the round an agent spends on a separate
    // qal_run after every define.
    let src = "fn main(args: List[Str]) -> Int !io = { print(read_all() ++ \"!\"); if len(args) == 0 then 1 else 0 }";
    let (err, v) = c.call("qal_define", json!({"source": src, "cases": [{"args": ["a"], "stdin": "x"}, {"stdin": "y"}], "allow": ["io"]}));
    assert!(!err, "{v}");
    assert_eq!(v["defined"][0]["name"], "main", "{v}");
    assert_eq!(v["results"][0], json!({"args": ["a"], "exit_code": 0, "stdout": "x!\n"}), "{v}");
    assert_eq!(v["results"][1]["exit_code"], 1, "{v}");
    // A runtime error in a case makes the call an error; the definition is stored anyway.
    let (err, v) = c.call("qal_define", json!({"source": src, "cases": [{"stdin": "z"}]}));
    assert!(err && v["defined"][0]["status"] == "unchanged" && v["results"][0]["error"]["kind"] == "permission_denied", "{v}");
    // Code that does not check is not run.
    let (err, v) = c.call("qal_define", json!({"source": "fn main(args: List[Str]) -> Int = nope", "cases": [{}]}));
    assert!(err && v.get("results").is_none() && v["diagnostics"][0]["code"] == "E0201", "{v}");
    // Another entry point by `entry`.
    let (err, v) = c.call("qal_define", json!({"source": "fn other(args: List[Str]) -> Int = 7", "entry": "other", "cases": [{}]}));
    assert!(!err && v["results"][0]["exit_code"] == 7, "{v}");
}

#[test]
fn define_runs_the_tests_of_the_defined_names() {
    both(define_runs_the_tests_of_the_defined_names_in);
}

fn define_runs_the_tests_of_the_defined_names_in(c: &mut Client) {
    let (err, v) = c.call("qal_define", json!({"source": "fn f(x: Int) -> Int = x + 1\ntest t_f = f(1) == 2\nfn g() -> Int = 7\ntest t_g = g() == 7"}));
    assert!(!err, "{v}");
    assert_eq!((v["tests"].clone(), v["passed"].as_u64(), v["failed"].as_u64()), (json!([{"name": "t_f", "passed": true}, {"name": "t_g", "passed": true}]), Some(2), Some(0)), "{v}");
    // A new version of `f` runs the tests that use `f` (not `t_g`); a failing test keeps the definition.
    let (err, v) = c.call("qal_define", json!({"source": "fn f(x: Int) -> Int = x + 2"}));
    assert!(!err, "{v}");
    assert_eq!((v["tests"][0]["name"].as_str(), v["tests"][0]["passed"].as_bool(), v["tests"].as_array().map(Vec::len), v["failed"].as_u64()), (Some("t_f"), Some(false), Some(1), Some(1)), "{v}");
    assert!(v["defined"].as_array().unwrap().iter().any(|d| d["name"] == "f"), "{v}");
    let (_, v) = c.call("qal_call", json!({"fn": "f", "args": [1]}));
    assert_eq!(v["ok"], 3, "{v}");
    // A define of only a test runs that test (`f` is `x + 2` by now).
    let (err, v) = c.call("qal_define", json!({"source": "test t_g = f(1) == 3"}));
    assert!(!err, "{v}");
    assert_eq!((v["tests"].clone(), v["passed"].as_u64()), (json!([{"name": "t_g", "passed": true}]), Some(1)), "{v}");
    // Names no test uses, and types, run nothing.
    let (err, v) = c.call("qal_define", json!({"source": "fn h() -> Int = 1\ntype P = { x: Int }"}));
    assert!(!err && v.get("tests").is_none() && v.get("passed").is_none(), "{v}");
}

/// Runs `body` on a fresh project, and again (Phase 4 §17) as the architect of a team store in its
/// own change set: the same calls give the same results there.
fn both(body: fn(&mut Client)) {
    let mut c = Client::start();
    c.initialize();
    body(&mut c);
    assert_eq!(c.finish(), Some(0));
    let mut c = Client::architect();
    body(&mut c);
    assert_eq!(c.finish(), Some(0));
}

#[test]
fn recall_answers_with_the_earlier_fix() {
    both(recall_answers_with_the_earlier_fix_in);
}

fn recall_answers_with_the_earlier_fix_in(c: &mut Client) {
    let (err, v) = c.call("qal_define", json!({"source": "fn f(n: Int) -> Int = { let x = n + 1 in x; x }"}));
    assert!(err && v["diagnostics"][0]["code"] == "E0201", "{v}");
    let (err, v) = c.call("qal_define", json!({"source": "fn f(n: Int) -> Int = let x = n + 1 in { x; x }"}));
    assert!(!err, "{v}");
    let (err, v) = c.call("qal_recall", json!({"code": "E0201"}));
    assert!(!err, "{v}");
    let corrections = v["recall"]["corrections"].as_array().unwrap();
    assert_eq!(corrections.len(), 1, "{v}");
    assert_eq!(corrections[0]["count"], 1, "{v}");
    assert!(corrections[0]["before"].is_string() && corrections[0]["after"].is_string(), "{v}");
    // A failing define of another name carries the fix in its hints.
    let (err, v) = c.call("qal_define", json!({"source": "fn g(n: Int) -> Int = { let x = n + 2 in x; x }"}));
    assert!(err, "{v}");
    let hints = v["diagnostics"][0]["hints"].as_array().unwrap();
    assert!(hints.iter().any(|h| h.as_str().is_some_and(|h| h.starts_with("earlier fix of E0201"))), "{v}");
    // No filter is refused by the store.
    let (err, v) = c.call("qal_recall", json!({}));
    assert!(err && v["diagnostics"][0]["code"] == "E0701", "{v}");
}

/// Phase 6b acceptance: the corpus of the token analysis (`bench/analysis/corrections-2026-10-23.json`)
/// answers for its three error classes. The whole corpus is imported once into one project, as
/// `qal corrections import` does, and each class is recalled the way an agent holding the diagnostic
/// asks: by its code and message. A second import changes nothing.
#[test]
fn recall_knows_the_three_error_classes_of_the_token_analysis() {
    use qal_store::{CorrectionInput, Store};
    let corpus: Vec<CorrectionInput> = serde_json::from_str(
        &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../bench/analysis/corrections-2026-10-23.json")).unwrap(),
    )
    .unwrap();
    let root = Client::fresh_root();
    let mut store = Store::init(&root).unwrap();
    assert_eq!(store.import_corrections(&corpus).unwrap(), corpus.len());
    assert_eq!(store.import_corrections(&corpus).unwrap(), 0, "the corpus is stored once");
    drop(store);
    let mut c = Client::start_at(root);
    c.initialize();
    let mut recall = |code: &str, message: &str| -> Vec<Value> {
        let (err, v) = c.call("qal_recall", json!({"code": code, "message": message}));
        assert!(!err, "{v}");
        v["recall"]["corrections"].as_array().unwrap().clone()
    };
    let text = |c: &Value, k: &str| c[k].as_str().unwrap().to_string();

    // Class 1: a local name used after `;` is out of scope (E0201 unknown name). The fix binds it with
    // `let … in` and puts the uses in the block after it.
    let found = recall("E0201", "unknown name `x`");
    assert!(found.iter().all(|c| c["shape"] == "unknown name `_`"), "{found:?}");
    let spaced = |t: String| t.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(found.iter().any(|c| text(c, "before").contains(';') && spaced(text(c, "after")).contains(" in { ")), "{found:?}");

    // Class 2: `String` where `Str` is meant. The fix writes `Str`; the record with this very message comes first.
    let found = recall("E0201", "unknown name `String`");
    let first = &found[0];
    assert!(text(first, "before").contains("String") && text(first, "after").contains("Str") && !text(first, "after").contains("String"), "{found:?}");

    // Class 3: a built-in used as a value (E0201 "is not a value"). The fix replaces the bare built-in
    // by a helper or closure, so `after` differs from `before` where the built-in stood.
    let found = recall("E0201", "built-in `len` is not a value");
    assert!(!found.is_empty() && found.iter().all(|c| c["shape"] == "built-in `_` is not a value"), "{found:?}");
    assert!(
        found.iter().any(|c| {
            let (b, a) = (text(c, "before"), text(c, "after"));
            let name = text(c, "message").split('`').nth(1).unwrap().to_string();
            b.contains(&format!(", {name})")) && !a.contains(&format!(", {name})"))
        }),
        "{found:?}"
    );
    // By code alone the ranking still shows both shapes of E0201.
    let (err, v) = c.call("qal_recall", json!({"code": "E0201"}));
    assert!(!err, "{v}");
    let shapes: Vec<&str> = v["recall"]["corrections"].as_array().unwrap().iter().map(|c| c["shape"].as_str().unwrap()).collect();
    assert!(shapes.contains(&"unknown name `_`") && shapes.contains(&"built-in `_` is not a value"), "{v}");
    assert_eq!(c.finish(), Some(0));
}
