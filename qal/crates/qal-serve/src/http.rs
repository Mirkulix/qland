//! HTTP/1.1 and 1.0 on a TCP connection (spec Phase 3 §2.3, §2.5): reading requests (request line
//! and headers with `httparse`, bodies only with `Content-Length`) under size limits and deadlines,
//! and writing responses (always with `Content-Length`, except for statuses without a body).
//!
//! Everything ambiguous is refused and the connection closed: `Transfer-Encoding` (411 for
//! `chunked`, else 501), several `Content-Length` headers or a malformed one (400), a target that
//! is not origin-form (400), header names with whitespace before the colon or folded lines (the
//! parser refuses them). Requests on one connection are read and answered strictly in order.

use crate::service::{bodiless, Request, Response};
use std::io::{ErrorKind, Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

/// Limits of the HTTP layer.
#[derive(Debug, Clone, PartialEq)]
pub struct HttpLimits {
    /// Request line plus headers.
    pub max_header_bytes: usize,
    pub max_headers: usize,
    pub max_body: usize,
    /// Time for the request line and headers, from the connection's (or the request's first byte's) start.
    pub header_timeout: Duration,
    /// Time for the body, from the end of the headers.
    pub body_timeout: Duration,
    /// Requests per connection.
    pub keepalive_requests: usize,
    /// Idle time between requests on one connection.
    pub keepalive_idle: Duration,
}

impl Default for HttpLimits {
    fn default() -> HttpLimits {
        HttpLimits {
            max_header_bytes: 16 << 10,
            max_headers: 64,
            max_body: 1 << 20,
            header_timeout: Duration::from_secs(5),
            body_timeout: Duration::from_secs(10),
            keepalive_requests: 100,
            keepalive_idle: Duration::from_secs(5),
        }
    }
}

/// A request the host refuses without calling the handler.
#[derive(Debug, Clone, PartialEq)]
pub struct Rejection {
    pub status: u16,
    pub kind: &'static str,
    pub message: String,
}

impl Rejection {
    fn new(status: u16, message: impl Into<String>) -> Rejection {
        let kind = match status {
            400 => "bad_request",
            408 => "request_timeout",
            411 => "length_required",
            413 => "payload_too_large",
            431 => "header_too_large",
            501 => "not_implemented",
            _ => "rejected",
        };
        Rejection { status, kind, message: message.into() }
    }

    /// The response for this rejection (JSON body).
    pub fn response(&self, id: u64) -> Response {
        let body = serde_json::json!({"error": {"kind": self.kind, "message": self.message}}).to_string();
        Response::json(self.status, body, id)
    }
}

/// A request read from a connection.
#[derive(Debug, Clone, PartialEq)]
pub struct Incoming {
    pub request: Request,
    /// Size of the request line and headers as received.
    pub head_bytes: usize,
    /// The client allows the connection to stay open.
    pub keep_alive: bool,
}

/// What reading the next request gave.
#[derive(Debug, PartialEq)]
pub enum Next {
    Request(Box<Incoming>),
    /// The client closed the connection or stayed idle between requests: close without a response.
    Closed,
    /// Answer with this rejection, then close.
    Reject(Rejection),
}

/// Hop-by-hop headers: they describe this connection, not the request (removed for the handler).
const HOP_BY_HOP: [&str; 6] = ["connection", "keep-alive", "transfer-encoding", "upgrade", "te", "trailer"];

/// One client connection with the bytes read ahead (pipelined requests).
pub struct Conn {
    stream: TcpStream,
    buf: Vec<u8>,
    /// Responses written so far.
    pub answered: usize,
}

fn reason(status: u16) -> &'static str {
    match status {
        100 => "Continue",
        101 => "Switching Protocols",
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        204 => "No Content",
        301 => "Moved Permanently",
        302 => "Found",
        303 => "See Other",
        304 => "Not Modified",
        307 => "Temporary Redirect",
        308 => "Permanent Redirect",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        409 => "Conflict",
        411 => "Length Required",
        413 => "Content Too Large",
        415 => "Unsupported Media Type",
        422 => "Unprocessable Content",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Status",
    }
}

/// The bytes of a response (`head_only`: without the body, for HEAD).
pub fn encode_response(resp: &Response, head_only: bool, keep_alive: bool) -> Vec<u8> {
    let mut out = format!("HTTP/1.1 {} {}\r\n", resp.status, reason(resp.status));
    for (n, v) in &resp.headers {
        out.push_str(n);
        out.push_str(": ");
        out.push_str(v);
        out.push_str("\r\n");
    }
    out.push_str(if keep_alive { "connection: keep-alive\r\n\r\n" } else { "connection: close\r\n\r\n" });
    let mut bytes = out.into_bytes();
    if !head_only && !bodiless(resp.status) {
        bytes.extend_from_slice(resp.body.as_bytes());
    }
    bytes
}

impl Conn {
    pub fn new(stream: TcpStream) -> Conn {
        let _ = stream.set_nodelay(true);
        let _ = stream.set_write_timeout(Some(Duration::from_secs(10)));
        Conn { stream, buf: Vec::new(), answered: 0 }
    }

    /// Reads into the buffer until `deadline`. Ok(false): the peer closed; Err: timed out.
    fn fill(&mut self, deadline: Instant) -> Result<bool, ()> {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(());
        }
        let _ = self.stream.set_read_timeout(Some(left));
        let mut chunk = [0u8; 8192];
        loop {
            match self.stream.read(&mut chunk) {
                Ok(0) => return Ok(false),
                Ok(n) => {
                    self.buf.extend_from_slice(&chunk[..n]);
                    return Ok(true);
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => return Err(()),
                Err(_) => return Ok(false),
            }
        }
    }

    /// Reads the next request. `start` is when the connection was accepted (first request); for
    /// later requests the header deadline starts with their first byte, after at most
    /// `keepalive_idle` of waiting.
    pub fn next(&mut self, limits: &HttpLimits, start: Instant) -> Next {
        let first = self.answered == 0;
        if self.buf.is_empty() && !first {
            match self.fill(Instant::now() + limits.keepalive_idle) {
                Ok(true) => {}
                Ok(false) | Err(()) => return Next::Closed,
            }
        }
        let head_deadline = if first { start } else { Instant::now() } + limits.header_timeout;
        let head_len = loop {
            let mut headers = vec![httparse::EMPTY_HEADER; limits.max_headers];
            let mut req = httparse::Request::new(&mut headers);
            match req.parse(&self.buf) {
                Ok(httparse::Status::Complete(n)) => break n,
                Ok(httparse::Status::Partial) => {}
                Err(httparse::Error::TooManyHeaders) => return Next::Reject(Rejection::new(431, format!("more than {} header fields", limits.max_headers))),
                Err(e) => return Next::Reject(Rejection::new(400, format!("malformed request: {e}"))),
            }
            if self.buf.len() > limits.max_header_bytes {
                return Next::Reject(Rejection::new(431, format!("the request line and headers are larger than {} bytes", limits.max_header_bytes)));
            }
            match self.fill(head_deadline) {
                Ok(true) => {}
                // The client closed the connection before the request was complete.
                Ok(false) => return Next::Closed,
                Err(()) => return Next::Reject(Rejection::new(408, format!("the request line and headers did not arrive within {} ms", limits.header_timeout.as_millis()))),
            }
        };
        if head_len > limits.max_header_bytes {
            return Next::Reject(Rejection::new(431, format!("the request line and headers are larger than {} bytes", limits.max_header_bytes)));
        }
        let parsed = match self.head(limits, head_len) {
            Ok(p) => p,
            Err(r) => return Next::Reject(r),
        };
        let (mut incoming, body_len, expect) = parsed;
        self.buf.drain(..head_len);
        if expect && self.buf.len() < body_len {
            let _ = self.stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
        }
        let body_deadline = Instant::now() + limits.body_timeout;
        while self.buf.len() < body_len {
            match self.fill(body_deadline) {
                Ok(true) => {}
                Ok(false) => return Next::Closed,
                Err(()) => return Next::Reject(Rejection::new(408, format!("the body did not arrive within {} ms", limits.body_timeout.as_millis()))),
            }
        }
        let body: Vec<u8> = self.buf.drain(..body_len).collect();
        match String::from_utf8(body) {
            Ok(b) => incoming.request.body = b,
            Err(_) => return Next::Reject(Rejection::new(400, "the body is not valid UTF-8")),
        }
        Next::Request(Box::new(incoming))
    }

    /// Interprets the parsed head: the request without its body, the body length, and whether
    /// the client waits for `100 Continue`.
    fn head(&self, limits: &HttpLimits, head_len: usize) -> Result<(Incoming, usize, bool), Rejection> {
        let mut headers = vec![httparse::EMPTY_HEADER; limits.max_headers];
        let mut req = httparse::Request::new(&mut headers);
        if !matches!(req.parse(&self.buf), Ok(httparse::Status::Complete(_))) {
            return Err(Rejection::new(400, "malformed request"));
        }
        let method = req.method.unwrap_or("").to_string();
        let target = req.path.unwrap_or("");
        let version = req.version.unwrap_or(1);
        if !target.starts_with('/') {
            return Err(Rejection::new(400, "the request target must be a path starting with `/` (no absolute-form or `*`)"));
        }
        let (path, query) = match target.split_once('?') {
            Some((p, q)) => (p.to_string(), q.to_string()),
            None => (target.to_string(), String::new()),
        };
        let mut list = Vec::new();
        let mut length: Option<usize> = None;
        let (mut te, mut close, mut keep, mut expect, mut host) = (None::<String>, false, false, false, false);
        for h in req.headers.iter() {
            let name = h.name.to_ascii_lowercase();
            let value = std::str::from_utf8(h.value).map_err(|_| Rejection::new(400, format!("the value of header `{name}` is not UTF-8")))?.trim().to_string();
            match name.as_str() {
                "content-length" => {
                    if length.is_some() {
                        return Err(Rejection::new(400, "more than one Content-Length header"));
                    }
                    if value.is_empty() || value.len() > 18 || !value.bytes().all(|b| b.is_ascii_digit()) {
                        return Err(Rejection::new(400, format!("invalid Content-Length `{}`", value.chars().take(40).collect::<String>())));
                    }
                    length = Some(value.parse().map_err(|_| Rejection::new(400, "invalid Content-Length"))?);
                }
                "transfer-encoding" => te = Some(value.to_ascii_lowercase()),
                "connection" => {
                    for token in value.split(',').map(|t| t.trim().to_ascii_lowercase()) {
                        close |= token == "close";
                        keep |= token == "keep-alive";
                    }
                }
                "expect" => expect |= value.eq_ignore_ascii_case("100-continue"),
                "host" if host => return Err(Rejection::new(400, "more than one Host header")),
                "host" => host = true,
                _ => {}
            }
            if !HOP_BY_HOP.contains(&name.as_str()) {
                list.push((name, value));
            }
        }
        if let Some(te) = te {
            let last = te.rsplit(',').next().unwrap_or("").trim().to_string();
            return Err(if last == "chunked" { Rejection::new(411, "chunked request bodies are not supported: send Content-Length") } else { Rejection::new(501, format!("Transfer-Encoding `{te}` is not supported")) });
        }
        if version == 1 && !host {
            return Err(Rejection::new(400, "HTTP/1.1 requests need a Host header"));
        }
        let body_len = length.unwrap_or(0);
        if body_len > limits.max_body {
            return Err(Rejection::new(413, format!("the body has {body_len} bytes, more than the limit of {}", limits.max_body)));
        }
        let keep_alive = if version == 1 { !close } else { keep && !close };
        let request = Request { method, path, query, headers: list, body: String::new() };
        Ok((Incoming { request, head_bytes: head_len, keep_alive }, body_len, expect && version == 1))
    }

    /// Writes a response.
    pub fn send(&mut self, resp: &Response, head_only: bool, keep_alive: bool) -> std::io::Result<()> {
        self.answered += 1;
        self.stream.write_all(&encode_response(resp, head_only, keep_alive))?;
        self.stream.flush()
    }

    pub fn stream(&self) -> &TcpStream {
        &self.stream
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    /// A connection whose client side sends `chunks` (with `pause` between them) and keeps the
    /// socket open for `hold`.
    fn conn(chunks: Vec<Vec<u8>>, pause: Duration) -> (Conn, std::thread::JoinHandle<Vec<u8>>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let client = std::thread::spawn(move || {
            let mut s = TcpStream::connect(addr).unwrap();
            for c in chunks {
                s.write_all(&c).unwrap();
                std::thread::sleep(pause);
            }
            let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
            let mut out = Vec::new();
            let _ = s.read_to_end(&mut out);
            out
        });
        let (s, _) = l.accept().unwrap();
        (Conn::new(s), client)
    }

    fn limits() -> HttpLimits {
        HttpLimits { header_timeout: Duration::from_millis(400), body_timeout: Duration::from_millis(400), keepalive_idle: Duration::from_millis(300), ..HttpLimits::default() }
    }

    fn one(raw: &[u8]) -> Next {
        let (mut c, _client) = conn(vec![raw.to_vec()], Duration::ZERO);
        c.next(&limits(), Instant::now())
    }

    fn rejected(raw: &[u8]) -> (u16, String) {
        match one(raw) {
            Next::Reject(r) => (r.status, r.message),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn requests_are_parsed_with_lowercase_headers_and_no_hop_by_hop_headers() {
        let Next::Request(i) = one(b"POST /todos/1?done=true&x=%20 HTTP/1.1\r\nHost: a\r\nX-Thing:  v  \r\nConnection: keep-alive, Upgrade\r\nUpgrade: h2c\r\nContent-Length: 4\r\n\r\nbody") else { panic!() };
        assert_eq!(i.request.method, "POST");
        assert_eq!((i.request.path.as_str(), i.request.query.as_str()), ("/todos/1", "done=true&x=%20"));
        assert_eq!(i.request.headers, vec![("host".to_string(), "a".to_string()), ("x-thing".into(), "v".into()), ("content-length".into(), "4".into())]);
        assert_eq!((i.request.body.as_str(), i.keep_alive), ("body", true));
        let Next::Request(i) = one(b"GET / HTTP/1.0\r\n\r\n") else { panic!() };
        assert!(!i.keep_alive, "HTTP/1.0 closes unless keep-alive is asked for");
        let Next::Request(i) = one(b"GET / HTTP/1.1\r\nHost: a\r\nConnection: close\r\n\r\n") else { panic!() };
        assert!(!i.keep_alive);
    }

    #[test]
    fn ambiguous_or_oversized_requests_are_refused() {
        assert_eq!(rejected(b"POST / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n").0, 411);
        assert_eq!(rejected(b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 3\r\nTransfer-Encoding: chunked\r\n\r\n").0, 411, "CL + TE: smuggling");
        assert_eq!(rejected(b"POST / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: gzip\r\n\r\n").0, 501);
        assert_eq!(rejected(b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 3\r\nContent-Length: 3\r\n\r\nabc").0, 400);
        for cl in ["+3", "-1", "3 3", "0x3", "3,3", "99999999999999999999"] {
            let raw = format!("POST / HTTP/1.1\r\nHost: a\r\nContent-Length: {cl}\r\n\r\nabc");
            assert_eq!(rejected(raw.as_bytes()).0, 400, "{cl}");
        }
        assert_eq!(rejected(b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length : 3\r\n\r\nabc").0, 400, "space before the colon");
        assert_eq!(rejected(b"GET / HTTP/1.1\r\nHost: a\r\nX: a\r\n b\r\n\r\n").0, 400, "folded header");
        assert_eq!(rejected(b"GET http://evil/ HTTP/1.1\r\nHost: a\r\n\r\n").0, 400);
        assert_eq!(rejected(b"OPTIONS * HTTP/1.1\r\nHost: a\r\n\r\n").0, 400);
        assert_eq!(rejected(b"GET / HTTP/1.1\r\n\r\n").0, 400, "no Host");
        assert_eq!(rejected(b"GET / HTTP/1.1\r\nHost: a\r\nHost: b\r\n\r\n").0, 400, "two Hosts");
        assert_eq!(rejected(b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n").0, 400);
        assert_eq!(rejected(b"G@T / HTTP/1.1\r\nHost: a\r\n\r\n").0, 400);
        assert_eq!(rejected(b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 2\r\n\r\n\xff\xfe").0, 400);
        assert_eq!(rejected(b"GET / HTTP/1.1\r\nHost: a\r\nX-Bin: \xff\r\n\r\n").0, 400);
        let big = format!("POST / HTTP/1.1\r\nHost: a\r\nContent-Length: {}\r\n\r\n", (1 << 20) + 1);
        assert_eq!(rejected(big.as_bytes()).0, 413);
        let many: String = (0..65).map(|i| format!("X-{i}: v\r\n")).collect();
        assert_eq!(rejected(format!("GET / HTTP/1.1\r\nHost: a\r\n{many}\r\n").as_bytes()).0, 431);
        let long = format!("GET / HTTP/1.1\r\nHost: a\r\nX: {}\r\n\r\n", "v".repeat(17 << 10));
        assert_eq!(rejected(long.as_bytes()).0, 431);
    }

    #[test]
    fn slow_clients_time_out() {
        let t = Instant::now();
        let (mut c, _client) = conn(vec![b"GET / HTTP/1.1\r\nHost: a\r\n".to_vec(), b"X-Slow: 1\r\n".to_vec()], Duration::from_millis(250));
        let Next::Reject(r) = c.next(&limits(), Instant::now()) else { panic!() };
        assert_eq!(r.status, 408);
        assert!(t.elapsed() < Duration::from_secs(1));
        let (mut c, _client) = conn(vec![b"POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 10\r\n\r\nabc".to_vec()], Duration::from_millis(900));
        let Next::Reject(r) = c.next(&limits(), Instant::now()) else { panic!() };
        assert_eq!((r.status, r.message.contains("body")), (408, true));
    }

    #[test]
    fn pipelined_requests_are_read_in_order_and_idle_connections_close() {
        let two = b"GET /a HTTP/1.1\r\nHost: x\r\n\r\nPOST /b HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\nhiGET /c HTTP/1.1\r\nHost: x\r\n\r\n".to_vec();
        let (mut c, client) = conn(vec![two], Duration::ZERO);
        let mut paths = Vec::new();
        for _ in 0..3 {
            let Next::Request(i) = c.next(&limits(), Instant::now()) else { panic!() };
            paths.push((i.request.path.clone(), i.request.body.clone()));
            c.send(&Response::json(200, "{}".into(), 1), false, true).unwrap();
        }
        assert_eq!(paths, vec![("/a".into(), "".into()), ("/b".into(), "hi".into()), ("/c".into(), "".into())]);
        assert_eq!(c.next(&limits(), Instant::now()), Next::Closed, "idle after the last request");
        drop(c);
        let out = String::from_utf8(client.join().unwrap()).unwrap();
        assert_eq!(out.matches("HTTP/1.1 200 OK\r\n").count(), 3);
    }

    #[test]
    fn expect_continue_is_answered_before_the_body() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let client = std::thread::spawn(move || {
            let mut s = TcpStream::connect(addr).unwrap();
            s.write_all(b"PUT /x HTTP/1.1\r\nHost: a\r\nExpect: 100-continue\r\nContent-Length: 2\r\n\r\n").unwrap();
            let mut buf = [0u8; 25];
            s.read_exact(&mut buf).unwrap();
            s.write_all(b"ok").unwrap();
            String::from_utf8(buf.to_vec()).unwrap()
        });
        let (s, _) = l.accept().unwrap();
        let mut c = Conn::new(s);
        let Next::Request(i) = c.next(&limits(), Instant::now()) else { panic!() };
        assert_eq!(i.request.body, "ok");
        assert_eq!(client.join().unwrap(), "HTTP/1.1 100 Continue\r\n\r\n");
    }

    #[test]
    fn responses_always_carry_their_length() {
        let r = Response { status: 201, headers: vec![("content-type".into(), "text/plain".into()), ("content-length".into(), "2".into())], body: "hi".into() };
        assert_eq!(String::from_utf8(encode_response(&r, false, true)).unwrap(), "HTTP/1.1 201 Created\r\ncontent-type: text/plain\r\ncontent-length: 2\r\nconnection: keep-alive\r\n\r\nhi");
        assert!(String::from_utf8(encode_response(&r, true, false)).unwrap().ends_with("connection: close\r\n\r\n"), "HEAD: no body");
        let empty = Response { status: 204, headers: vec![], body: String::new() };
        assert_eq!(String::from_utf8(encode_response(&empty, false, true)).unwrap(), "HTTP/1.1 204 No Content\r\nconnection: keep-alive\r\n\r\n");
    }
}
