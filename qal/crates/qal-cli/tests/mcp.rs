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
}

impl Client {
    /// A server on a fresh, empty project directory.
    fn start() -> Client {
        let root = std::env::temp_dir().join(format!("qal-mcp-test-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::SeqCst)));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
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
        Client { child, stdin, stdout, root, next_id: 0 }
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
        (result["isError"].as_bool().unwrap(), result["structuredContent"].clone())
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
    assert_eq!(names, ["qal_define", "qal_test", "qal_run", "qal_call", "qal_view", "qal_query", "qal_rename", "qal_history", "qal_build"]);
    for t in tools.as_array().unwrap() {
        assert_eq!(t["inputSchema"]["type"], "object", "{t}");
        assert!(!t["description"].as_str().unwrap().is_empty());
    }
    let reference = tools[0]["description"].as_str().unwrap();
    assert!(reference.contains("QAL language reference") && reference.contains("alias_reference") && reference.contains("signature_change"));
    // "At most one page": a bounded size.
    assert!(qal_cli::mcp::REFERENCE.lines().count() <= 45 && qal_cli::mcp::REFERENCE.len() <= 6000, "{} lines, {} bytes", qal_cli::mcp::REFERENCE.lines().count(), qal_cli::mcp::REFERENCE.len());
    // An unknown version gets the newest legacy one.
    let r = c.request("initialize", json!({"protocolVersion": "1999-01-01", "capabilities": {}, "clientInfo": {"name": "x", "version": "1"}}));
    assert_eq!(r["result"]["protocolVersion"], "2025-11-25");
    assert_eq!(c.finish(), Some(0));
}

#[test]
fn an_agent_defines_tests_runs_and_inspects_code() {
    let mut c = Client::start();
    c.initialize();
    let (err, v) = c.call("qal_define", json!({"source": example("age.qal")}));
    assert!(!err, "{v}");
    assert_eq!(v["defined"].as_array().unwrap().len(), 5);
    assert!(c.root.join(".qal/store.db").is_file());
    let (err, v) = c.call("qal_test", json!({}));
    assert_eq!((err, v["passed"].as_u64(), v["failed"].as_u64()), (false, Some(2), Some(0)));
    // Effects must be granted explicitly.
    let (err, v) = c.call("qal_run", json!({"args": [" 42"]}));
    assert_eq!((err, v["error"]["kind"].as_str()), (true, Some("permission_denied")));
    assert_eq!(v["error"]["hints"], json!(["pass `\"allow\": [\"io\"]` to qal_run or qal_call"]));
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
    assert_eq!(v["history"][0]["agent"], "mcp:test-client", "the client's name is the default agent");
    assert_eq!(v["history"][1]["action"], "renamed");
    let (err, v) = c.call("qal_build", json!({"entry": "main"}));
    assert!(!err, "{v}");
    assert!(c.root.join(".qal/build/main.wasm").is_file());
    // Diagnostics are data.
    let (err, v) = c.call("qal_define", json!({"source": "fn f(x: Int) -> Bool = x + 1", "agent": "me"}));
    assert_eq!((err, v["diagnostics"][0]["kind"].as_str(), v["diagnostics"][0]["path"].as_str()), (true, Some("type_mismatch"), Some("body")));
    let (err, v) = c.call("qal_test", json!({"names": ["main"]}));
    assert_eq!((err, v["diagnostics"][0]["message"].as_str()), (true, Some("`main` is a function, not a test")));
    let (err, v) = c.call("qal_view", json!({"names": ["parse_age"]}));
    assert_eq!((err, v["diagnostics"][0]["hints"][0].as_str()), (true, Some("`parse_age` was renamed to `parse_years`")));
    assert_eq!(c.finish(), Some(0));
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
    assert_eq!((err, v["diagnostics"][0]["message"].as_str()), (true, Some("unknown argument `src` (`qal_define` takes: source, agent)")));
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
        assert!(!c.call("qal_define", json!({"source": big(n)})).0);
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
    assert!(start.elapsed() < Duration::from_secs(2), "{:?}", start.elapsed());
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
    for needle in ["rand_int(lo, hi) -> Int !rand (both ends included; lo <= hi", "!io !fs", "narrowest", "fold(xs, 0, step)", "quadratic", "`r` cannot be a definition name", "no list patterns", "every field"] {
        assert!(r.contains(needle), "{needle}");
    }
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let script = std::fs::read_to_string(repo.join("qal/mcp.sh")).unwrap();
    assert!(script.contains("CDPATH=") && script.contains("build failed"), "{script}");
}
