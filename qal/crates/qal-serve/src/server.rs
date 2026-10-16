//! The synchronous HTTP server (spec Phase 3 §2.3, §2.6): one accept thread, a bounded queue of
//! connections (full: `503` with `Retry-After: 1`), and worker threads with guest-sized stacks that
//! each serve one connection at a time (keep-alive, requests in order). JSON log lines go to the
//! log sink; a panic while handling a request is answered with 500 and the worker goes on;
//! shutdown stops accepting and lets running requests finish for up to 10 s.

use crate::http::{Conn, HttpLimits, Next, Rejection};
use crate::service::{Response, Service};
use qal_run::{RunError, GUEST_THREAD_STACK};
use serde::Serialize;
use serde_json::json;
use std::io::Write;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How long running requests may take to finish after a shutdown signal.
pub const SHUTDOWN_GRACE: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, PartialEq)]
pub struct ServerConfig {
    pub workers: usize,
    /// Accepted connections waiting for a worker.
    pub queue: usize,
    pub http: HttpLimits,
}

impl Default for ServerConfig {
    fn default() -> ServerConfig {
        ServerConfig { workers: 4, queue: 64, http: HttpLimits::default() }
    }
}

/// Where the JSON log lines go (stderr or a file); one line per event.
pub struct Log {
    out: Mutex<Box<dyn Write + Send>>,
}

impl Log {
    pub fn new(out: Box<dyn Write + Send>) -> Log {
        Log { out: Mutex::new(out) }
    }

    pub fn stderr() -> Log {
        Log::new(Box::new(std::io::stderr()))
    }

    /// Writes one event; a failing log never stops the server.
    pub fn line(&self, event: &impl Serialize) {
        if let (Ok(text), Ok(mut out)) = (serde_json::to_string(event), self.out.lock()) {
            let _ = writeln!(out, "{text}").and_then(|_| out.flush());
        }
    }
}

/// Binds the listening address (`listen_failed`, E0802).
pub fn bind(addr: &str) -> Result<TcpListener, RunError> {
    TcpListener::bind(addr).map_err(|e| RunError::new("listen_failed", "", format!("cannot listen on {addr}: {e}")).hint("choose another address with --listen, e.g. 127.0.0.1:0 for a free port"))
}

/// A host-made log entry (rejections, overload, internal errors).
fn host_event(id: u64, req: Option<(&str, &str)>, status: u16, error: RunError) -> serde_json::Value {
    let (method, path) = req.unwrap_or(("", ""));
    json!({"event": "request", "id": id, "method": method, "path": path, "status": status, "ms": 0, "fuel": 0, "mem_bytes": 0, "db": "none", "error": error})
}

struct Shared {
    service: Arc<Service>,
    cfg: ServerConfig,
    log: Arc<Log>,
    shutdown: Arc<AtomicBool>,
    served: AtomicU64,
    /// Connections waiting in the queue: while there are any, kept-alive connections are closed
    /// after their current request, so idle clients cannot hold every worker.
    pending: std::sync::atomic::AtomicUsize,
}

/// Serves connections from `listener` until `shutdown` is set; returns the number of responses
/// sent. The caller has run the service's init function already.
pub fn serve(listener: TcpListener, service: Arc<Service>, cfg: ServerConfig, log: Arc<Log>, shutdown: Arc<AtomicBool>, workers: Vec<crate::service::Worker>) -> u64 {
    let addr = listener.local_addr().ok();
    let shared = Arc::new(Shared { service, cfg, log, shutdown, served: AtomicU64::new(0), pending: std::sync::atomic::AtomicUsize::new(0) });
    let (tx, rx) = mpsc::sync_channel::<TcpStream>(shared.cfg.queue.max(1));
    let rx = Arc::new(Mutex::new(rx));
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let mut started = 0;
    for (k, worker) in workers.into_iter().enumerate() {
        let (shared, rx, done) = (shared.clone(), rx.clone(), done_tx.clone());
        let spawned = std::thread::Builder::new().name(format!("qal-worker-{k}")).stack_size(GUEST_THREAD_STACK).spawn(move || {
            work(&shared, &rx, worker);
            let _ = done.send(());
        });
        if spawned.is_ok() {
            started += 1;
        }
    }
    drop(done_tx);
    // Wakes the blocking `accept` once a shutdown is requested.
    let waker = {
        let shared = shared.clone();
        std::thread::Builder::new().name("qal-shutdown".into()).spawn(move || {
            while !shared.shutdown.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(25));
            }
            if let Some(a) = addr {
                let _ = TcpStream::connect_timeout(&local(a), Duration::from_millis(500));
            }
        })
    };
    accept(&listener, &shared, &tx);
    drop(tx);
    drop(listener);
    let deadline = Instant::now() + SHUTDOWN_GRACE;
    for _ in 0..started {
        if done_rx.recv_timeout(deadline.saturating_duration_since(Instant::now())).is_err() {
            break;
        }
    }
    if let Ok(w) = waker {
        let _ = w.join();
    }
    shared.served.load(Ordering::SeqCst)
}

/// An address to connect to for `a` (an unspecified listen address becomes loopback).
fn local(a: SocketAddr) -> SocketAddr {
    match a {
        SocketAddr::V4(v) if v.ip().is_unspecified() => SocketAddr::from(([127, 0, 0, 1], v.port())),
        SocketAddr::V6(v) if v.ip().is_unspecified() => SocketAddr::from((std::net::Ipv6Addr::LOCALHOST, v.port())),
        other => other,
    }
}

fn accept(listener: &TcpListener, shared: &Arc<Shared>, tx: &SyncSender<TcpStream>) {
    for conn in listener.incoming() {
        let stop = shared.shutdown.load(Ordering::SeqCst);
        // A connection accepted after the signal was still established before it (or is the
        // waker's, which reads as closed): it is answered, not dropped.
        if let Ok(stream) = conn {
            if !enqueue(shared, tx, stream) {
                return;
            }
        }
        if stop {
            // The connections the kernel queued before the signal are answered too.
            if listener.set_nonblocking(true).is_ok() {
                while let Ok((stream, _)) = listener.accept() {
                    if stream.set_nonblocking(false).is_err() || !enqueue(shared, tx, stream) {
                        break;
                    }
                }
            }
            return;
        }
    }
}

/// Hands a connection to the workers (503 if the queue is full); false once they are gone.
fn enqueue(shared: &Shared, tx: &SyncSender<TcpStream>, stream: TcpStream) -> bool {
    shared.pending.fetch_add(1, Ordering::SeqCst);
    match tx.try_send(stream) {
        Ok(()) => true,
        Err(TrySendError::Full(stream)) => {
            shared.pending.fetch_sub(1, Ordering::SeqCst);
            overloaded(shared, stream);
            true
        }
        Err(TrySendError::Disconnected(_)) => false,
    }
}

/// Answers a connection the queue has no room for.
fn overloaded(shared: &Shared, stream: TcpStream) {
    let id = shared.service.next_id();
    let mut resp = Response::json(503, json!({"error": {"kind": "overloaded"}}).to_string(), id);
    resp.headers.push(("retry-after".into(), "1".into()));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
    let mut conn = Conn::new(stream);
    let _ = conn.send(&resp, false, false);
    shared.served.fetch_add(1, Ordering::SeqCst);
    shared.log.line(&host_event(id, None, 503, RunError::new("overloaded", "", "the connection queue is full")));
}

fn work(shared: &Shared, rx: &Mutex<Receiver<TcpStream>>, mut worker: crate::service::Worker) {
    loop {
        let next = match rx.lock() {
            Ok(r) => r.recv(),
            Err(_) => return,
        };
        let Ok(stream) = next else { return };
        shared.pending.fetch_sub(1, Ordering::SeqCst);
        // The header deadline starts when a worker takes the connection, so time spent waiting
        // in the queue does not count against a client that sent its request promptly.
        let taken = Instant::now();
        // Requests are guarded inside `connection`; this catches a bug anywhere else in it.
        if catch_unwind(AssertUnwindSafe(|| connection(shared, &mut worker, stream, taken))).is_err() {
            worker = crate::service::Worker::default();
        }
    }
}

/// Serves the requests of one connection in order.
fn connection(shared: &Shared, worker: &mut crate::service::Worker, stream: TcpStream, taken: Instant) {
    let limits = &shared.cfg.http;
    let mut conn = Conn::new(stream);
    loop {
        let incoming = match conn.next(limits, taken) {
            Next::Request(i) => i,
            Next::Closed => return,
            Next::Reject(r) => return reject(shared, &mut conn, r),
        };
        let req = &incoming.request;
        let id = shared.service.next_id();
        let head_only = req.method == "HEAD";
        let mut handler_req = req.clone();
        if head_only {
            handler_req.method = "GET".into();
        }
        let handled = catch_unwind(AssertUnwindSafe(|| shared.service.handle(worker, &handler_req, id, incoming.head_bytes)));
        let (resp, log) = match handled {
            Ok(h) => (h.response, json!(h.log)),
            Err(p) => {
                // A bug in the host: answer, log, and continue with a fresh worker state.
                *worker = crate::service::Worker::default();
                let msg = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "panic".into());
                let body = json!({"error": {"kind": "internal_error", "id": id.to_string()}}).to_string();
                (Response::json(500, body, id), host_event(id, Some((&req.method, &req.path)), 500, RunError::new("internal", "", format!("the server panicked while handling the request: {msg}"))))
            }
        };
        let last = conn.answered + 1 >= limits.keepalive_requests;
        let keep = incoming.keep_alive && !last && !shared.shutdown.load(Ordering::SeqCst) && shared.pending.load(Ordering::SeqCst) == 0;
        // Logged before the response goes out: a client that has the response can rely on its
        // event being in the log (logging after sending raced with readers of the log).
        shared.log.line(&log);
        let sent = conn.send(&resp, head_only, keep);
        shared.served.fetch_add(1, Ordering::SeqCst);
        if sent.is_err() || !keep {
            return;
        }
    }
}

fn reject(shared: &Shared, conn: &mut Conn, r: Rejection) {
    let id = shared.service.next_id();
    let _ = conn.send(&r.response(id), false, false);
    shared.served.fetch_add(1, Ordering::SeqCst);
    shared.log.line(&host_event(id, None, r.status, RunError::new("request_rejected", "", format!("{} {}: {}", r.status, r.kind, r.message))));
}
