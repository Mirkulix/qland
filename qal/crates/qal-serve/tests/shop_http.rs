//! Phase 4d: the Shop backend (`examples/shop/shop.qalpkg`, authored by a 6-agent
//! team over MCP) served by the real `qal serve` process, driven by a raw TCP client.

#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use serde_json::Value;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn qal_bin() -> PathBuf {
    if let Some(p) = std::env::var_os("QAL_BIN") {
        return PathBuf::from(p);
    }
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

/// A temporary shop project with `shop.qalpkg` imported.
fn project() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qal-shop-http-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::SeqCst)));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    qal(&dir, &["init"]);
    let pkg = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/shop/shop.qalpkg");
    let imported = qal(&dir, &["import", pkg.to_str().unwrap()]);
    assert_eq!(imported["imported"]["package"], "shop");
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

/// Reads one response.
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

fn req(addr: SocketAddr, method: &str, path: &str, json: Option<&str>) -> Resp {
    Conn::new(addr).send(method, path, json)
}

#[test]
fn health_and_products_crud() {
    let dir = project();
    let server = start(&dir, &[]);
    let mut c = Conn::new(server.addr);

    // GET /health
    let h = c.send("GET", "/health", None);
    assert_eq!((h.status, h.header("content-type")), (200, Some("application/json")));
    assert_eq!(h.json()["ok"], true);

    // POST /products
    let p1 = c.send("POST", "/products", Some(r#"{"sku":"COFFEE-1","name":"Coffee Beans","price_cents":450,"stock":10}"#));
    assert_eq!((p1.status, p1.header("location")), (201, Some("/products/1")));
    let j1 = p1.json();
    assert_eq!((j1["id"].as_i64(), j1["sku"].as_str(), j1["price_cents"].as_i64(), j1["stock"].as_i64()), (Some(1), Some("COFFEE-1"), Some(450), Some(10)));

    // GET /products/1
    let g1 = c.send("GET", "/products/1", None);
    assert_eq!(g1.status, 200);
    assert_eq!(g1.json()["name"], "Coffee Beans");

    // PATCH /products/1
    let pt = c.send("PATCH", "/products/1", Some(r#"{"stock":15}"#));
    assert_eq!(pt.status, 200);
    assert_eq!(pt.json()["stock"], 15);

    // GET /products
    let list = c.send("GET", "/products", None);
    assert_eq!(list.status, 200);
    assert_eq!(list.json().as_array().map(Vec::len), Some(1));

    let (code, _) = server.terminate();
    assert_eq!(code, Some(0));
}

#[test]
fn order_creation_reservation_cancellation_and_reports() {
    let dir = project();
    let server = start(&dir, &[]);
    let mut c = Conn::new(server.addr);

    // Create 2 products
    let p1 = c.send("POST", "/products", Some(r#"{"sku":"BEANS-1","name":"Beans","price_cents":500,"stock":10}"#));
    assert_eq!(p1.status, 201);
    let p2 = c.send("POST", "/products", Some(r#"{"sku":"MUG-1","name":"Mug","price_cents":1000,"stock":5}"#));
    assert_eq!(p2.status, 201);

    // Create order with 2 beans and 1 mug
    let o1 = c.send("POST", "/orders", Some(r#"{"lines":[{"product_id":1,"qty":2},{"product_id":2,"qty":1}]}"#));
    assert_eq!((o1.status, o1.header("location")), (201, Some("/orders/1")));
    let oj = o1.json();
    assert_eq!(oj["status"], "confirmed");
    assert_eq!(oj["total_cents"], 2000); // 2*500 + 1*1000

    // Check report: stock should reflect reservation
    let rep = c.send("GET", "/reports/stock", None);
    assert_eq!(rep.status, 200);
    let rep_arr = rep.json();
    let r_beans = rep_arr.as_array().unwrap().iter().find(|x| x["sku"] == "BEANS-1").unwrap();
    assert_eq!(r_beans["current_stock"], 8);
    assert_eq!(r_beans["reserved_stock"], 2);

    // Cancel order
    let can = c.send("POST", "/orders/1/cancel", None);
    assert_eq!(can.status, 200);
    assert_eq!(can.json()["status"], "cancelled");

    // Stock should be restored
    let rep2 = c.send("GET", "/reports/stock", None);
    assert_eq!(rep2.status, 200);
    let rep_arr2 = rep2.json();
    let r_beans2 = rep_arr2.as_array().unwrap().iter().find(|x| x["sku"] == "BEANS-1").unwrap();
    assert_eq!(r_beans2["current_stock"], 10);
    assert_eq!(r_beans2["reserved_stock"], 0);

    let (code, _) = server.terminate();
    assert_eq!(code, Some(0));
}

#[test]
fn not_found_and_method_not_allowed() {
    let dir = project();
    let server = start(&dir, &[]);
    let mut c = Conn::new(server.addr);

    let nf = c.send("GET", "/nonexistent", None);
    assert_eq!(nf.status, 404);

    let mna = c.send("PUT", "/products", None);
    assert_eq!(mna.status, 405);

    let (code, _) = server.terminate();
    assert_eq!(code, Some(0));
}

#[test]
fn data_survives_server_restart() {
    let dir = project();
    let server = start(&dir, &[]);
    let r = req(server.addr, "POST", "/products", Some(r#"{"sku":"PERSIST-1","name":"Persistent Item","price_cents":100,"stock":50}"#));
    assert_eq!(r.status, 201);
    let (code, _) = server.terminate();
    assert_eq!(code, Some(0));

    let server2 = start(&dir, &[]);
    let r2 = req(server2.addr, "GET", "/products/1", None);
    assert_eq!(r2.status, 200);
    assert_eq!(r2.json()["sku"], "PERSIST-1");
    let (code2, _) = server2.terminate();
    assert_eq!(code2, Some(0));
}
