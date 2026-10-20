//! Phase 3c, spec Phase 3 §8 test layer 3: the Todo backend (`examples/todo/todo.qalpkg`, written
//! by an agent over MCP) served by the real `qal serve` process, driven by a raw TCP client.
//!
//! Every test makes a temporary project (`qal init`, `qal import todo.qalpkg`), starts
//! `qal serve --listen 127.0.0.1:0 --allow db,time --init setup` and reads the port from the
//! `listening` log line. The isolation and shutdown tests add the fixture handler
//! `fixtures/todo_isolation.qal` (test code) in front of `handle`.
//!
//! The `qal` binary is the one of this target directory (`cargo test --workspace` builds it;
//! alone: `cargo build -p qal-cli && cargo test -p qal-serve --test todo_http`), or `$QAL_BIN`.
#![cfg(unix)]

use serde_json::Value;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

const FIXTURE: &str = include_str!("fixtures/todo_isolation.qal");
static NEXT: AtomicUsize = AtomicUsize::new(0);

fn qal_bin() -> PathBuf {
    if let Some(p) = std::env::var_os("QAL_BIN") {
        return PathBuf::from(p);
    }
    // target/<profile>/deps/todo_http-<hash> -> target/<profile>/qal
    let exe = std::env::current_exe().unwrap();
    let bin = exe.parent().and_then(Path::parent).unwrap().join(format!("qal{}", std::env::consts::EXE_SUFFIX));
    assert!(bin.is_file(), "{} is missing: build it with `cargo build -p qal-cli` (or set QAL_BIN)", bin.display());
    bin
}

fn qal(dir: &Path, args: &[&str]) -> Value {
    let out = Command::new(qal_bin()).args(args).current_dir(dir).stdin(Stdio::null()).output().unwrap();
    assert!(out.status.success(), "qal {args:?}: {}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    serde_json::from_slice(&out.stdout).unwrap()
}

/// A temporary todo project; `fixture` also defines the misbehaving handler `isolation`.
fn project(fixture: bool) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qal-todo-http-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::SeqCst)));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    qal(&dir, &["init"]);
    let pkg = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/todo/todo.qalpkg");
    let imported = qal(&dir, &["import", pkg.to_str().unwrap()]);
    assert_eq!(imported["imported"]["package"], "todo");
    if fixture {
        let f = dir.join("isolation.qal");
        std::fs::write(&f, FIXTURE).unwrap();
        qal(&dir, &["define", f.to_str().unwrap(), "--agent", "todo-http-test"]);
    }
    dir
}

/// A running `qal serve` process.
struct Server {
    child: Child,
    addr: SocketAddr,
    log: Arc<Mutex<Vec<String>>>,
    reader: Option<std::thread::JoinHandle<()>>,
}

fn start(dir: &Path, extra: &[&str]) -> Server {
    let mut args = vec!["serve", "--listen", "127.0.0.1:0", "--allow", "db,time", "--init", "setup"];
    args.extend_from_slice(extra);
    let mut child = Command::new(qal_bin()).args(&args).current_dir(dir).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped()).spawn().unwrap();
    let stderr = child.stderr.take().unwrap();
    let log = Arc::new(Mutex::new(Vec::new()));
    let (tx, rx) = mpsc::channel();
    let lines = log.clone();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if let Ok(v) = serde_json::from_str::<Value>(&line) {
                if v["event"] == "listening" {
                    let _ = tx.send(v["addr"].as_str().unwrap_or("").to_string());
                }
            }
            lines.lock().unwrap().push(line);
        }
    });
    let addr = match rx.recv_timeout(Duration::from_secs(120)) {
        Ok(a) => a.parse().unwrap(),
        Err(_) => {
            let _ = child.kill();
            panic!("qal serve did not start: {:?}", log.lock().unwrap());
        }
    };
    Server { child, addr, log, reader: Some(reader) }
}

impl Server {
    fn pid(&self) -> u32 {
        self.child.id()
    }

    /// SIGTERM; waits for the exit and the end of the log. Returns the exit code.
    fn terminate(mut self) -> (Option<i32>, Vec<Value>) {
        let ok = Command::new("kill").args(["-TERM", &self.pid().to_string()]).status().unwrap().success();
        assert!(ok, "kill -TERM");
        let status = self.child.wait().unwrap();
        if let Some(r) = self.reader.take() {
            let _ = r.join();
        }
        (status.code(), self.events())
    }

    fn events(&self) -> Vec<Value> {
        self.log.lock().unwrap().iter().filter_map(|l| serde_json::from_str(l).ok()).collect()
    }

    fn rss_kib(&self) -> u64 {
        let out = Command::new("ps").args(["-o", "rss=", "-p", &self.pid().to_string()]).output().unwrap();
        String::from_utf8_lossy(&out.stdout).trim().parse().unwrap()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A parsed response.
#[derive(Debug)]
struct Resp {
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

impl Resp {
    fn header(&self, n: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k == n).map(|(_, v)| v.as_str())
    }
    fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or_else(|e| panic!("not JSON ({e}): {:?}", self.body))
    }
}

/// Reads one response (bytes read past it stay in `rest`).
fn read_resp(s: &mut TcpStream, rest: &mut Vec<u8>, head_only: bool) -> Option<Resp> {
    let mut chunk = [0u8; 8192];
    let end = loop {
        if let Some(i) = rest.windows(4).position(|w| w == b"\r\n\r\n") {
            break i;
        }
        let n = s.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        rest.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8(rest[..end].to_vec()).unwrap();
    rest.drain(..end + 4);
    let mut lines = head.split("\r\n");
    let status: u16 = lines.next()?.split(' ').nth(1)?.parse().ok()?;
    let headers: Vec<(String, String)> = lines.filter_map(|l| l.split_once(": ").map(|(a, b)| (a.to_ascii_lowercase(), b.to_string()))).collect();
    let len: usize = if head_only { 0 } else { headers.iter().find(|(k, _)| k == "content-length").map_or(0, |(_, v)| v.parse().unwrap()) };
    while rest.len() < len {
        let n = s.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        rest.extend_from_slice(&chunk[..n]);
    }
    let body = String::from_utf8(rest.drain(..len).collect()).unwrap();
    Some(Resp { status, headers, body })
}

fn connect(addr: SocketAddr) -> TcpStream {
    let s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
    s
}

fn request_bytes(method: &str, path: &str, json: Option<&str>) -> Vec<u8> {
    match json {
        Some(b) => format!("{method} {path} HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}", b.len()).into_bytes(),
        None => format!("{method} {path} HTTP/1.1\r\nHost: t\r\n\r\n").into_bytes(),
    }
}

/// A kept-alive connection.
struct Conn {
    s: TcpStream,
    rest: Vec<u8>,
}

impl Conn {
    fn new(addr: SocketAddr) -> Conn {
        Conn { s: connect(addr), rest: Vec::new() }
    }
    fn send(&mut self, method: &str, path: &str, json: Option<&str>) -> Resp {
        self.s.write_all(&request_bytes(method, path, json)).unwrap();
        read_resp(&mut self.s, &mut self.rest, method == "HEAD").unwrap_or_else(|| panic!("no response to {method} {path}"))
    }
}

/// One request on a fresh connection.
fn req(addr: SocketAddr, method: &str, path: &str, json: Option<&str>) -> Resp {
    Conn::new(addr).send(method, path, json)
}

fn raw(addr: SocketAddr, bytes: &[u8]) -> (Option<Resp>, bool) {
    let mut s = connect(addr);
    s.write_all(bytes).unwrap();
    let mut rest = Vec::new();
    let r = read_resp(&mut s, &mut rest, false);
    let closed = matches!(s.read(&mut [0u8; 16]), Ok(0));
    (r, closed)
}

/// `{"id":..,"title":..,"done":..,"created_at":<int>}` with the keys in this order.
fn assert_todo(body: &str, id: i64, title: &str, done: bool) -> i64 {
    let v: Value = serde_json::from_str(body).unwrap();
    let created = v["created_at"].as_i64().unwrap_or_else(|| panic!("created_at in {body}"));
    let want = format!("{{\"id\":{id},\"title\":{},\"done\":{done},\"created_at\":{created}}}", serde_json::to_string(title).unwrap());
    assert_eq!(body, want);
    created
}

#[test]
fn crud_over_one_kept_alive_connection() {
    let dir = project(false);
    let server = start(&dir, &[]);
    let mut c = Conn::new(server.addr);
    let h = c.send("GET", "/health", None);
    assert_eq!((h.status, h.body.as_str(), h.header("content-type")), (200, r#"{"ok":true}"#, Some("application/json")));
    let a = c.send("POST", "/todos", Some(r#"{"title":"  Buy milk  "}"#));
    assert_eq!((a.status, a.header("location"), a.header("connection")), (201, Some("/todos/1"), Some("keep-alive")));
    let created = assert_todo(&a.body, 1, "Buy milk", false);
    let b = c.send("POST", "/todos", Some(r#"{"title":"Tee kochen ☕","done":true}"#));
    assert_eq!(b.status, 201);
    assert_todo(&b.body, 2, "Tee kochen ☕", true);
    let g = c.send("GET", "/todos/1", None);
    assert_eq!(g.status, 200);
    assert_eq!(assert_todo(&g.body, 1, "Buy milk", false), created);
    let p = c.send("PATCH", "/todos/1", Some(r#"{"done":true}"#));
    assert_eq!(p.status, 200);
    assert_eq!(assert_todo(&p.body, 1, "Buy milk", true), created, "PATCH keeps created_at");
    let bad = c.send("PATCH", "/todos/1", Some(r#"{"title":"changed","done":"yes"}"#));
    assert_eq!((bad.status, bad.json()["error"]["code"].as_str()), (400, Some("invalid_type")));
    let l = c.send("GET", "/todos", None);
    let list = l.json();
    assert_eq!((l.status, list.as_array().map(Vec::len)), (200, Some(2)));
    assert_eq!((list[0]["title"].as_str(), list[0]["done"].as_bool()), (Some("Buy milk"), Some(true)), "the invalid PATCH changed nothing");
    let f = c.send("GET", "/todos?done=false", None);
    assert_eq!(f.body, "[]");
    let d = c.send("DELETE", "/todos/1", None);
    assert_eq!((d.status, d.body.as_str(), d.header("content-type"), d.header("content-length")), (204, "", None, None), "a 204 gets no default content-type");
    let gone = c.send("GET", "/todos/1", None);
    assert_eq!((gone.status, gone.json()["error"]["code"].as_str()), (404, Some("not_found")));
    let m = c.send("PUT", "/todos", None);
    assert_eq!((m.status, m.header("allow")), (405, Some("GET, HEAD, POST")));
    let head = c.send("HEAD", "/todos/2", None);
    assert_eq!((head.status, head.body.as_str()), (200, ""));
    assert!(head.header("content-length").is_some_and(|n| n != "0"));
    let (code, events) = server.terminate();
    assert_eq!(code, Some(0));
    let ids: Vec<u64> = events.iter().filter(|e| e["event"] == "request").map(|e| e["id"].as_u64().unwrap()).collect();
    assert_eq!(ids, (1..=12).collect::<Vec<u64>>(), "one connection, twelve requests, monotonic ids");
    assert_eq!(events.last().map(|e| e["event"].clone()), Some("stopped".into()));
}

#[test]
fn data_survives_a_sigterm_and_a_restart() {
    let dir = project(false);
    let server = start(&dir, &[]);
    let mut created = Vec::new();
    for t in ["one", "two", "three"] {
        let r = req(server.addr, "POST", "/todos", Some(&format!("{{\"title\":\"{t}\"}}")));
        assert_eq!(r.status, 201);
        created.push(r.body);
    }
    assert_eq!(req(server.addr, "DELETE", "/todos/2", None).status, 204);
    let (code, events) = server.terminate();
    assert_eq!(code, Some(0));
    assert_eq!(events.last().unwrap()["served"], 4);
    let again = start(&dir, &[]);
    let list = req(again.addr, "GET", "/todos", None);
    assert_eq!(list.body, format!("[{},{}]", created[0], created[2]), "same rows, same created_at");
    let next = req(again.addr, "POST", "/todos", Some(r#"{"title":"four"}"#));
    assert_eq!(next.header("location"), Some("/todos/4"), "ids continue after the restart");
}

#[test]
fn fifty_parallel_posts_get_fifty_ids() {
    let dir = project(false);
    let server = start(&dir, &[]);
    let addr = server.addr;
    let threads: Vec<_> = (0..50).map(|i| std::thread::spawn(move || req(addr, "POST", "/todos", Some(&format!("{{\"title\":\"t{i}\"}}"))))).collect();
    let mut ids: Vec<i64> = threads
        .into_iter()
        .map(|t| {
            let r = t.join().unwrap();
            assert_eq!(r.status, 201, "{}", r.body);
            r.json()["id"].as_i64().unwrap()
        })
        .collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids, (1..=50).collect::<Vec<i64>>());
    let list = req(addr, "GET", "/todos", None).json();
    assert_eq!(list.as_array().map(Vec::len), Some(50));
}

#[test]
fn the_host_enforces_its_limits() {
    let dir = project(false);
    let server = start(&dir, &[]);
    let addr = server.addr;
    let big_body = format!("POST /todos HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", (1 << 20) + 1);
    let cases: Vec<(Vec<u8>, u16, &str)> = vec![
        (big_body.into_bytes(), 413, "payload_too_large"),
        (format!("GET /todos HTTP/1.1\r\nHost: t\r\nX-Big: {}\r\n\r\n", "a".repeat(17 << 10)).into_bytes(), 431, "header_too_large"),
        (b"POST /todos HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\nE\r\n{\"title\":\"x\"}\r\n0\r\n\r\n".to_vec(), 411, "length_required"),
        (b"POST /todos HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: 14\r\n\r\n{\"title\":\"\xc3\x28\"}".to_vec(), 400, "bad_request"),
    ];
    for (bytes, status, kind) in cases {
        let (r, closed) = raw(addr, &bytes);
        let r = r.unwrap_or_else(|| panic!("no response for the {status} case"));
        assert_eq!(r.status, status, "{}", r.body);
        assert_eq!(r.json()["error"]["kind"], kind);
        assert!(closed, "{status}: the connection is closed after a rejection");
    }
    // Slowloris: a head that never ends is cut off after the default header timeout (5 s).
    let t = Instant::now();
    let mut s = connect(addr);
    s.write_all(b"GET /todos HTTP/1.1\r\nHost: t\r\nX-Slow: 1\r\n").unwrap();
    let mut rest = Vec::new();
    let r = read_resp(&mut s, &mut rest, false);
    let waited = t.elapsed();
    assert_eq!(r.map(|r| r.status), Some(408));
    assert!(matches!(s.read(&mut [0u8; 16]), Ok(0)), "closed after the timeout");
    assert!(waited >= Duration::from_millis(4500) && waited < Duration::from_secs(8), "closed after {waited:?}");
    assert_eq!(req(addr, "GET", "/todos", None).body, "[]", "no rejected request reached the handler");
    let rejected = server.events().iter().filter(|e| e["error"]["code"] == "E0800").count();
    assert!(rejected >= 5, "{rejected}");
}

#[test]
fn a_failing_request_does_not_affect_the_next() {
    let dir = project(true);
    // Fuel high enough that the wall-clock timeout (2 s) ends the endless recursion.
    let server = start(&dir, &["isolation", "--request-timeout", "2000", "--fuel", "1000000000000"]);
    let addr = server.addr;
    let mut c = Conn::new(addr);
    assert_eq!(c.send("POST", "/todos", Some(r#"{"title":"before"}"#)).status, 201, "the fixture serves the todo API too");
    let trap = req(addr, "GET", "/iso/trap", None);
    assert_eq!((trap.status, trap.json()["error"]["kind"].as_str()), (500, Some("internal_error")));
    assert_eq!(c.send("GET", "/health", None).body, r#"{"ok":true}"#, "the next request is fine");
    let t = Instant::now();
    let spin = req(addr, "GET", "/iso/spin", None);
    let took = t.elapsed();
    assert_eq!((spin.status, spin.json()["error"]["kind"].as_str()), (503, Some("timeout")));
    // The 2 s request timeout ends the spin; the upper bound only rules out a hang (slow CI runners).
    assert!(took >= Duration::from_millis(1900) && took < Duration::from_secs(20), "{took:?}");
    // A fresh connection: the kept-alive one may have idled out during the spin on a slow runner.
    assert_eq!(req(addr, "GET", "/todos", None).json().as_array().map(Vec::len), Some(1), "the server is still up");
    // About 60 MiB per request (a 30 MiB string built by doubling) within the 64 MiB limit, 20
    // times in a row: every request gets fresh memory.
    for i in 0..20 {
        let big = c.send("GET", "/iso/big", None);
        assert_eq!((big.status, big.body.as_str()), (200, "31457280"), "request {i}");
    }
    let (code, events) = server.terminate();
    assert_eq!(code, Some(0));
    let kinds: Vec<&str> = events.iter().filter_map(|e| e["error"]["kind"].as_str()).collect();
    assert_eq!(kinds, vec!["int_overflow", "timeout"]);
    let mem: Vec<u64> = events.iter().filter(|e| e["path"] == "/iso/big").map(|e| e["mem_bytes"].as_u64().unwrap()).collect();
    assert_eq!(mem.len(), 20);
    assert!(mem.iter().all(|&m| m >= 60 << 20), "{mem:?}");
    println!("isolation: /iso/big mem_bytes {} (limit 64 MiB), /iso/spin answered after {took:?}", mem[0]);
}

#[test]
fn sigterm_during_a_slow_request_answers_it_and_exits_0() {
    let dir = project(true);
    let server = start(&dir, &["isolation", "--fuel", "1000000000000", "--request-timeout", "60000"]);
    let addr = server.addr;
    // Calibrate: about 2 s of guest work.
    let mut n: u64 = 10_000_000;
    loop {
        let t = Instant::now();
        assert_eq!(req(addr, "GET", &format!("/iso/slow?{n}"), None).status, 200);
        let took = t.elapsed();
        if took >= Duration::from_millis(500) || n >= 10_000_000_000 {
            n = (n as f64 * 2000.0 / took.as_millis().max(1) as f64) as u64;
            break;
        }
        n *= 8;
    }
    // About 4 s of work: even if the load drops after calibrating, it still runs when SIGTERM comes.
    n *= 2;
    let (sent_tx, sent_rx) = std::sync::mpsc::channel();
    let slow = std::thread::spawn(move || {
        let mut s = connect(addr);
        s.write_all(&request_bytes("GET", &format!("/iso/slow?{n}"), None)).unwrap();
        let _ = sent_tx.send(());
        let mut rest = Vec::new();
        let r = read_resp(&mut s, &mut rest, false);
        (r, Instant::now())
    });
    sent_rx.recv().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    let signalled = Instant::now();
    let (code, events) = server.terminate();
    let (r, answered) = slow.join().unwrap();
    let r = r.expect("the running request is answered");
    assert_eq!((r.status, r.body), (200, n.to_string()));
    assert!(answered > signalled, "the request was still running when SIGTERM came");
    let took = answered - signalled;
    assert_eq!(code, Some(0));
    assert_eq!(events.last().unwrap()["event"], "stopped");
    println!("shutdown: slow request ({n} steps) answered {took:?} after SIGTERM");
}

/// 10 000 `GET /todos` over kept-alive connections: the RSS may grow at most 10 % after the
/// first 1 000 requests.
#[test]
fn memory_stays_flat_over_ten_thousand_requests() {
    let dir = project(false);
    let server = start(&dir, &[]);
    let addr = server.addr;
    for t in ["one", "two", "three", "four", "five"] {
        assert_eq!(req(addr, "POST", "/todos", Some(&format!("{{\"title\":\"{t}\"}}"))).status, 201);
    }
    let mut c = Conn::new(addr);
    let mut rss = Vec::new();
    let t = Instant::now();
    let mut at_1000 = Duration::ZERO;
    for i in 1..=10_000 {
        let r = c.send("GET", "/todos", None);
        assert_eq!(r.status, 200);
        if r.header("connection") == Some("close") {
            c = Conn::new(addr);
        }
        if i % 1000 == 0 {
            rss.push(server.rss_kib());
            if i == 1000 {
                at_1000 = t.elapsed();
            }
        }
    }
    let total = t.elapsed();
    let (first, last) = (rss[0], rss[rss.len() - 1]);
    let growth = (last as f64 - first as f64) / first as f64 * 100.0;
    println!(
        "rss: after 1000 {first} KiB, after 10000 {last} KiB, growth {growth:.2} % (every 1000: {rss:?}); 10000 requests in {total:?} ({:.0} req/s; first 1000 in {at_1000:?})",
        10_000.0 / total.as_secs_f64()
    );
    assert!(growth <= 10.0, "RSS grew {growth:.2} %: {rss:?}");
    let (code, _) = server.terminate();
    assert_eq!(code, Some(0));
}
