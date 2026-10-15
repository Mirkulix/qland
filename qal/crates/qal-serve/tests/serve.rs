//! The server over real TCP connections, with the fixture service `fixtures/app.qal`.

use qal_run::Permissions;
use qal_serve::{bind, serve, HttpLimits, Log, ServerConfig, Service, ServiceConfig};
use serde_json::Value;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const APP: &str = include_str!("fixtures/app.qal");
static NEXT: AtomicUsize = AtomicUsize::new(0);

fn typed() -> qal_core::tast::TProgram {
    qal_check::check(&qal_text::parse_program(APP).unwrap()).unwrap()
}

fn project() -> PathBuf {
    let d = std::env::temp_dir().join(format!("qal-serve-test-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::SeqCst)));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Collects log lines.
#[derive(Clone, Default)]
struct Lines(Arc<Mutex<Vec<u8>>>);

impl Write for Lines {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Lines {
    fn events(&self) -> Vec<Value> {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
    }
}

struct Running {
    addr: SocketAddr,
    shutdown: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<u64>>,
    log: Lines,
}

impl Running {
    fn stop(mut self) -> u64 {
        self.shutdown.store(true, Ordering::SeqCst);
        self.thread.take().unwrap().join().unwrap()
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn config(dir: &std::path::Path) -> (Permissions, ServiceConfig, ServerConfig) {
    let perms = Permissions::parse("db,io").unwrap().with_db_dir(dir.join(".qal/db"));
    let svc = ServiceConfig { init: Some("setup".into()), request_timeout: Duration::from_millis(1500), memory: 40 << 20, ..ServiceConfig::default() };
    let http = HttpLimits { header_timeout: Duration::from_millis(500), body_timeout: Duration::from_millis(500), ..HttpLimits::default() };
    (perms, svc, ServerConfig { workers: 4, queue: 64, http })
}

fn start_with(perms: Permissions, svc: ServiceConfig, srv: ServerConfig) -> Running {
    let slots = srv.workers + 2;
    let service = Arc::new(Service::new(&typed(), perms, ServiceConfig { slots, ..svc }).unwrap());
    let mut workers: Vec<_> = (0..srv.workers).map(|_| service.worker().unwrap()).collect();
    service.init(&mut workers[0]).unwrap();
    let listener = bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let log = Lines::default();
    let shutdown = Arc::new(AtomicBool::new(false));
    let (s, l) = (shutdown.clone(), Arc::new(Log::new(Box::new(log.clone()))));
    let thread = std::thread::spawn(move || serve(listener, service, srv, l, s, workers));
    Running { addr, shutdown, thread: Some(thread), log }
}

fn start(dir: &std::path::Path) -> Running {
    let (p, s, h) = config(dir);
    start_with(p, s, h)
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
}

/// Reads one response from `s` (bytes already read past it stay in `rest`).
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
    let headers: Vec<(String, String)> = lines.filter_map(|l| l.split_once(": ").map(|(a, b)| (a.to_string(), b.to_string()))).collect();
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
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    s
}

/// One request on a fresh connection.
fn req(addr: SocketAddr, method: &str, path: &str, body: &str) -> Resp {
    let mut s = connect(addr);
    write!(s, "{method} {path} HTTP/1.1\r\nHost: t\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
    read_resp(&mut s, &mut Vec::new(), method == "HEAD").unwrap()
}

fn raw(addr: SocketAddr, bytes: &[u8]) -> (Option<Resp>, bool) {
    let mut s = connect(addr);
    s.write_all(bytes).unwrap();
    let mut rest = Vec::new();
    let r = read_resp(&mut s, &mut rest, false);
    let mut more = [0u8; 16];
    let closed = matches!(s.read(&mut more), Ok(0));
    (r, closed)
}

#[test]
fn crud_over_keep_alive_and_pipelined_requests() {
    let dir = project();
    let server = start(&dir);
    let mut s = connect(server.addr);
    let mut rest = Vec::new();
    // Three requests written at once are answered in order on the same connection.
    s.write_all(b"POST /todos HTTP/1.1\r\nHost: t\r\nContent-Length: 4\r\n\r\nmilkPOST /todos HTTP/1.1\r\nHost: t\r\nContent-Length: 5\r\n\r\nbreadGET /todos HTTP/1.1\r\nHost: t\r\n\r\n").unwrap();
    let a = read_resp(&mut s, &mut rest, false).unwrap();
    assert_eq!((a.status, a.header("location"), a.body.as_str()), (201, Some("/todos/1"), r#"{"id":1,"title":"milk"}"#));
    assert_eq!(a.header("content-type"), Some("application/json"));
    assert_eq!(a.header("connection"), Some("keep-alive"));
    let b = read_resp(&mut s, &mut rest, false).unwrap();
    assert_eq!(b.status, 201);
    let c = read_resp(&mut s, &mut rest, false).unwrap();
    assert_eq!(c.body, r#"[{"id":1,"title":"milk"},{"id":2,"title":"bread"}]"#);
    let ids: Vec<u64> = [&a, &b, &c].iter().map(|r| r.header("x-request-id").unwrap().parse().unwrap()).collect();
    assert!(ids[0] < ids[1] && ids[1] < ids[2], "monotonic request ids {ids:?}");
    write!(s, "DELETE /todos/1 HTTP/1.1\r\nHost: t\r\n\r\n").unwrap();
    let d = read_resp(&mut s, &mut rest, false).unwrap();
    assert_eq!((d.status, d.header("content-length")), (204, None), "no content-length for 204");
    write!(s, "GET /todos/1 HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n").unwrap();
    let e = read_resp(&mut s, &mut rest, false).unwrap();
    assert_eq!((e.status, e.header("connection"), e.header("content-type")), (404, Some("close"), Some("text/plain; charset=utf-8")));
    assert_eq!(s.read(&mut [0u8; 8]).unwrap(), 0, "closed after `Connection: close`");
    // HEAD is a GET without the body.
    let h = req(server.addr, "HEAD", "/todos", "");
    assert_eq!((h.status, h.body.as_str()), (200, ""));
    assert_eq!(h.header("content-length"), Some(r#"[{"id":2,"title":"bread"}]"#.len().to_string().as_str()));
    let echo = req(server.addr, "PUT", "/echo?a=1&b=%20", "x");
    assert_eq!(echo.body, "PUT /echo?a=1&b=%20 host,content-length x");
    drop(s);
    let served = server.stop();
    assert!(served >= 7, "{served}");
}

#[test]
fn every_request_runs_in_a_fresh_instance() {
    let dir = project();
    let server = start(&dir);
    let trap = req(server.addr, "GET", "/trap", "");
    assert_eq!(trap.status, 500);
    let v: Value = serde_json::from_str(&trap.body).unwrap();
    assert_eq!(v["error"]["kind"], "internal_error");
    assert_eq!(v["error"]["id"].as_str(), trap.header("x-request-id"));
    assert_eq!(req(server.addr, "GET", "/health", "").body, "ok", "the next request is fine");
    let t = Instant::now();
    let spin = req(server.addr, "GET", "/spin", "");
    assert_eq!(spin.status, 503);
    assert!(t.elapsed() < Duration::from_secs(6), "{:?}", t.elapsed());
    assert_eq!(serde_json::from_str::<Value>(&spin.body).unwrap()["error"]["kind"], "timeout");
    // About 32 MiB per request within a 40 MiB limit, 20 times: memory is freed after each request.
    for _ in 0..20 {
        let big = req(server.addr, "GET", "/big", "");
        assert_eq!((big.status, big.body.as_str()), (200, "16777216"));
    }
    assert_eq!(req(server.addr, "GET", "/bad", "").status, 500, "CR/LF in a header value is a bad_response");
    assert_eq!(req(server.addr, "GET", "/print", "").body, "printed");
    let events = server.log.events();
    let kinds: Vec<&str> = events.iter().filter_map(|e| e["error"]["kind"].as_str()).collect();
    assert_eq!(kinds, vec!["int_overflow", "timeout", "bad_response"]);
    let printed = events.iter().find(|e| e["path"] == "/print").unwrap();
    assert_eq!(printed["out"], "hello from the handler\n");
    let big = events.iter().find(|e| e["path"] == "/big").unwrap();
    assert!(big["mem_bytes"].as_u64().unwrap() >= 32 << 20 && big["fuel"].as_u64().unwrap() > 0 && big["ms"].is_u64(), "{big}");
}

#[test]
fn writes_are_committed_only_for_successful_responses() {
    let dir = project();
    let server = start(&dir);
    assert_eq!(req(server.addr, "POST", "/todos", "kept").status, 201);
    assert_eq!(req(server.addr, "POST", "/fail", "").status, 422);
    assert_eq!(req(server.addr, "GET", "/todos", "").body, r#"[{"id":1,"title":"kept"}]"#, "the 4xx response rolled its insert back");
    let events = server.log.events();
    let dbs: Vec<&str> = events.iter().filter_map(|e| e["db"].as_str()).collect();
    assert_eq!(dbs, vec!["commit", "rollback", "commit"]);
    drop(server);
    // Persistence: a new server on the same database sees the data.
    let again = start(&dir);
    assert_eq!(req(again.addr, "GET", "/todos/1", "").body, r#"{"id":1,"title":"kept"}"#);
}

#[test]
fn concurrent_posts_get_distinct_ids() {
    let dir = project();
    let server = start(&dir);
    let addr = server.addr;
    let threads: Vec<_> = (0..50).map(|i| std::thread::spawn(move || req(addr, "POST", "/todos", &format!("t{i}")))).collect();
    let mut ids: Vec<String> = threads.into_iter().map(|t| t.join().unwrap()).map(|r| {
        assert_eq!(r.status, 201, "{}", r.body);
        r.header("location").unwrap().to_string()
    }).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 50);
    let list = req(addr, "GET", "/todos", "");
    assert_eq!(list.body.matches("\"id\"").count(), 50);
}

#[test]
fn the_host_refuses_bad_requests_and_closes() {
    let dir = project();
    let server = start(&dir);
    let big = format!("POST /todos HTTP/1.1\r\nHost: t\r\nContent-Length: {}\r\n\r\n", (1 << 20) + 1);
    let cases: Vec<(Vec<u8>, u16, &str)> = vec![
        (big.into_bytes(), 413, "payload_too_large"),
        (format!("GET / HTTP/1.1\r\nHost: t\r\nX: {}\r\n\r\n", "a".repeat(17 << 10)).into_bytes(), 431, "header_too_large"),
        (b"POST /todos HTTP/1.1\r\nHost: t\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nmilk\r\n0\r\n\r\n".to_vec(), 411, "length_required"),
        (b"POST /todos HTTP/1.1\r\nHost: t\r\nContent-Length: 2\r\n\r\n\xc3\x28".to_vec(), 400, "bad_request"),
    ];
    for (bytes, status, kind) in cases {
        let (r, closed) = raw(server.addr, &bytes);
        let r = r.unwrap();
        assert_eq!(r.status, status);
        assert_eq!(serde_json::from_str::<Value>(&r.body).unwrap()["error"]["kind"], kind);
        assert!(closed, "{status}: the connection is closed after a rejection");
    }
    // Slowloris: a head that never ends is cut off after the header timeout.
    let t = Instant::now();
    let mut s = connect(server.addr);
    s.write_all(b"GET / HTTP/1.1\r\nHost: t\r\n").unwrap();
    let r = read_resp(&mut s, &mut Vec::new(), false).unwrap();
    assert_eq!(r.status, 408);
    assert!(t.elapsed() < Duration::from_secs(4), "{:?}", t.elapsed());
    assert_eq!(req(server.addr, "GET", "/todos", "").body, "[]", "nothing was stored");
    let events = server.log.events();
    assert!(events.iter().filter(|e| e["error"]["code"] == "E0800").count() >= 5, "{events:?}");
}

#[test]
fn a_full_queue_answers_503_and_shutdown_finishes_running_requests() {
    let dir = project();
    let (p, mut svc, mut srv) = config(&dir);
    svc.request_timeout = Duration::from_secs(5);
    svc.fuel = u64::MAX / 4;
    srv.workers = 1;
    srv.queue = 1;
    let server = start_with(p, svc, srv);
    let addr = server.addr;
    // The only worker is busy for a while; one connection fits the queue; the next is refused.
    let slow = std::thread::spawn(move || req(addr, "GET", "/slow?100000000", ""));
    std::thread::sleep(Duration::from_millis(300));
    let mut queued = connect(addr);
    queued.write_all(b"GET /health HTTP/1.1\r\nHost: t\r\n\r\n").unwrap();
    std::thread::sleep(Duration::from_millis(100));
    let (r, _) = raw(addr, b"GET /health HTTP/1.1\r\nHost: t\r\n\r\n");
    let r = r.unwrap();
    assert_eq!((r.status, r.header("retry-after")), (503, Some("1")));
    assert_eq!(serde_json::from_str::<Value>(&r.body).unwrap()["error"]["kind"], "overloaded");
    // Shutdown while the slow request runs: it is still answered, and serve returns.
    server.shutdown.store(true, Ordering::SeqCst);
    let done = slow.join().unwrap();
    assert_eq!(done.status, 200);
    assert_eq!(read_resp(&mut queued, &mut Vec::new(), false).map(|r| r.status), Some(200), "queued connections are served");
    let log = server.log.clone();
    let served = server.stop();
    assert!(served >= 3, "{served}");
    assert!(log.events().iter().any(|e| e["error"]["code"] == "E0801"));
}

#[test]
fn services_are_checked_before_they_start() {
    let typed = typed();
    let perms = Permissions::parse("db,io").unwrap();
    let e = Service::new(&typed, perms.clone(), ServiceConfig { entry: "text".into(), ..ServiceConfig::default() }).err().unwrap();
    assert_eq!((e.kind, e.code), ("bad_entry", "E0520"));
    assert!(e.message.contains("must be `fn(HttpRequest) -> HttpResponse` (any effects), found `fn(Int, Str) -> HttpResponse`"), "{}", e.message);
    let e = Service::new(&typed, perms.clone(), ServiceConfig { entry: "nope".into(), ..ServiceConfig::default() }).err().unwrap();
    assert_eq!(e.kind, "bad_entry");
    let e = Service::new(&typed, perms.clone(), ServiceConfig { init: Some("list".into()), ..ServiceConfig::default() }).err().unwrap();
    assert!(e.message.contains("init function `list` must be `fn() -> Result[Unit, Str]`"), "{}", e.message);
    let e = Service::new(&typed, Permissions::parse("io").unwrap(), ServiceConfig::default()).err().unwrap();
    assert_eq!((e.kind, e.hints.clone()), ("permission_denied", vec!["grant them with `--allow db`".to_string()]));
    // An init function that returns Err stops the start.
    let src = format!("{APP}\nfn broken() -> Result[Unit, Str] = Err(\"no\")");
    let typed = qal_check::check(&qal_text::parse_program(&src).unwrap()).unwrap();
    let svc = Service::new(&typed, Permissions::parse("db:mem,io").unwrap(), ServiceConfig { init: Some("broken".into()), ..ServiceConfig::default() }).unwrap();
    let e = svc.init(&mut svc.worker().unwrap()).unwrap_err();
    assert_eq!((e.kind, e.code, e.message.as_str()), ("init_failed", "E0524", "`broken` returned Err: no"));
    assert_eq!(bind("256.0.0.1:1").unwrap_err().code, "E0802");
}

#[test]
fn debug_errors_return_the_runtime_error() {
    let dir = project();
    let (p, mut svc, h) = config(&dir);
    svc.debug_errors = true;
    let server = start_with(p, svc, h);
    let r = req(server.addr, "GET", "/trap", "");
    let v: Value = serde_json::from_str(&r.body).unwrap();
    assert_eq!((v["error"]["kind"].as_str(), v["error"]["code"].as_str()), (Some("int_overflow"), Some("E0502")));
}

/// Spec Phase 3 §2.2: instantiating and discarding the module per request must cost at most
/// 200 µs on average in a release build. Run with `cargo test --release -p qal-serve --test serve
/// instance_overhead -- --nocapture` to see the measurement.
#[test]
fn instance_overhead_meets_the_gate() {
    let svc = Service::new(&typed(), Permissions::parse("db:mem,io").unwrap(), ServiceConfig::default()).unwrap();
    let r = svc.runner();
    for _ in 0..100 {
        r.instantiate_only().unwrap();
    }
    let n = 2000;
    let t = Instant::now();
    for _ in 0..n {
        r.instantiate_only().unwrap();
    }
    let mean = t.elapsed() / n;
    println!("instance overhead: mean {mean:?} over {n} instantiations (pooling allocator: {})", svc.pooled());
    assert!(svc.pooled());
    let gate = if cfg!(debug_assertions) { Duration::from_millis(2) } else { Duration::from_micros(200) };
    assert!(mean <= gate, "mean {mean:?} > {gate:?}");
}

#[test]
fn kept_alive_clients_do_not_hold_workers_while_others_wait() {
    let dir = project();
    let (p, s, mut srv) = config(&dir);
    srv.workers = 1;
    srv.http.keepalive_idle = Duration::from_millis(800);
    let server = start_with(p, s, srv);
    let addr = server.addr;
    let stop = Arc::new(AtomicBool::new(false));
    let attacker = {
        let stop = stop.clone();
        std::thread::spawn(move || {
            let mut a = connect(addr);
            let mut rest = Vec::new();
            while !stop.load(Ordering::SeqCst) {
                if a.write_all(b"GET /health HTTP/1.1\r\nHost: t\r\n\r\n").is_err() || read_resp(&mut a, &mut rest, false).is_none() {
                    return true;
                }
                std::thread::sleep(Duration::from_millis(250));
            }
            false
        })
    };
    std::thread::sleep(Duration::from_millis(300));
    let t = Instant::now();
    let victim = req(addr, "GET", "/health", "");
    let waited = t.elapsed();
    stop.store(true, Ordering::SeqCst);
    assert_eq!(victim.body, "ok");
    // Without the rule the victim would wait until the attacker stops (never, here).
    assert!(waited < Duration::from_secs(2), "the victim waited {waited:?}");
    let _ = attacker.join();
}
