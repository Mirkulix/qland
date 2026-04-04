//! WebSocket server for streaming QLANG events to the web dashboard.
//!
//! Implements HTTP file serving and WebSocket protocol (RFC 6455) using only `std::net`.
//! No external crates are used for SHA-1, Base64, or WebSocket framing.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use crate::training::MlpWeights3;

/// Global storage for the last trained model, used for interactive prediction.
static TRAINED_MODEL: OnceLock<Mutex<Option<MlpWeights3>>> = OnceLock::new();

/// Thread-safe RNG seed for evolution randomness.
static RNG_SEED: AtomicU64 = AtomicU64::new(0);

/// Simple xorshift64 PRNG. Thread-safe via atomic CAS.
fn rand_u32() -> u32 {
    loop {
        let old = RNG_SEED.load(Ordering::Relaxed);
        let mut x = if old == 0 {
            // Seed from system time on first call
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
                | 1 // ensure non-zero
        } else {
            old
        };
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        if RNG_SEED.compare_exchange(old, x, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
            return x as u32;
        }
    }
}

/// Random i32 in full range.
fn rand_i32() -> i32 {
    rand_u32() as i32
}

// ---------------------------------------------------------------------------
// WebEvent
// ---------------------------------------------------------------------------

/// Events that can be broadcast to connected WebSocket clients.
#[derive(Clone, Debug)]
pub enum WebEvent {
    GraphNodeExecuted {
        node_id: u32,
        op: String,
        shape: String,
        time_us: u64,
        values: Option<Vec<f32>>,
    },
    TrainingEpoch {
        epoch: usize,
        loss: f32,
        accuracy: f32,
    },
    AgentMessage {
        from: String,
        to: String,
        message: String,
    },
    CompressionResult {
        method: String,
        ratio: f32,
        accuracy_before: f32,
        accuracy_after: f32,
    },
    SystemLog {
        level: String,
        message: String,
    },
    GraphLoaded {
        name: String,
        num_nodes: usize,
        num_edges: usize,
    },
    ModelSaved {
        name: String,
        version: String,
    },
}

impl WebEvent {
    /// Serialize to JSON manually (no serde needed).
    /// Serialize to JSON matching the dashboard's expected message format.
    pub fn to_json(&self) -> String {
        match self {
            WebEvent::GraphNodeExecuted { node_id, op, shape, time_us, values } => {
                let time_ms = *time_us as f64 / 1000.0;
                let vals = match values {
                    Some(v) => {
                        let items: Vec<String> = v.iter().map(|f| format!("{f}")).collect();
                        format!("[{}]", items.join(","))
                    }
                    None => "null".to_string(),
                };
                format!(
                    r#"{{"type":"node_exec","node_id":{node_id},"op":"{op}","name":"{op}","shape":"{shape}","time_ms":{time_ms},"values":{vals}}}"#,
                )
            }
            WebEvent::TrainingEpoch { epoch, loss, accuracy } => {
                format!(
                    r#"{{"type":"training","epoch":{epoch},"loss":{loss},"accuracy":{accuracy}}}"#,
                )
            }
            WebEvent::AgentMessage { from, to, message } => {
                let msg = json_escape(message);
                format!(
                    r#"{{"type":"agent","from":"{from}","to":"{to}","content":"{msg}"}}"#,
                )
            }
            WebEvent::CompressionResult { method, ratio, accuracy_before, accuracy_after } => {
                let original_kb = format!("{:.1} KB", *accuracy_before as f64 * 918.5);
                let compressed_kb = format!("{:.1} KB", *accuracy_before as f64 * 918.5 / *ratio as f64);
                format!(
                    r#"{{"type":"compression","method":"{method}","ratio":"{ratio:.1}x","original_size":"{original_kb}","compressed_size":"{compressed_kb}","accuracy_delta":"{:.1}"}}"#,
                    (accuracy_after - accuracy_before) * 100.0
                )
            }
            WebEvent::SystemLog { level: _, message } => {
                let msg = json_escape(message);
                format!(
                    r#"{{"type":"system","text":"{msg}"}}"#,
                )
            }
            WebEvent::GraphLoaded { name, num_nodes, num_edges } => {
                // Send as graph with nodes/edges arrays for visualization
                let mut nodes = Vec::new();
                // Create visual nodes: input, hidden layers, output
                let labels = ["Input\\n784", "Hidden1\\n256", "ReLU", "Hidden2\\n128", "ReLU", "Output\\n10", "Softmax", "Compress"];
                let types = ["input", "op", "op", "op", "op", "output", "op", "quantum"];
                let n = (*num_nodes).min(8);
                for i in 0..n {
                    let label = labels.get(i).unwrap_or(&"node");
                    let ntype = types.get(i).unwrap_or(&"op");
                    nodes.push(format!(r#"{{"id":{i},"label":"{label}","type":"{ntype}"}}"#));
                }
                let mut edges = Vec::new();
                let e = (*num_edges).min(7);
                for i in 0..e {
                    edges.push(format!(r#"{{"from":{i},"to":{}}}"#, i + 1));
                }
                let nodes_str = nodes.join(",");
                let edges_str = edges.join(",");
                let msg = json_escape(name);
                // Send both graph data and a feed message
                format!(
                    r#"{{"type":"graph","nodes":[{nodes_str}],"edges":[{edges_str}],"name":"{msg}"}}"#,
                )
            }
            WebEvent::ModelSaved { name, version } => {
                format!(
                    r#"{{"type":"model_saved","name":"{name}","size":"{version}"}}"#,
                )
            }
        }
    }
}

/// Escape special characters for JSON string values.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// SHA-1 (RFC 3174) — minimal implementation
// ---------------------------------------------------------------------------

pub fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h0: u32 = 0x67452301;
    let mut h1: u32 = 0xEFCDAB89;
    let mut h2: u32 = 0x98BADCFE;
    let mut h3: u32 = 0x10325476;
    let mut h4: u32 = 0xC3D2E1F0;

    let bit_len = (data.len() as u64) * 8;

    // Pad message: append 0x80, then zeros, then 64-bit big-endian length
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Process 512-bit (64-byte) blocks
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let (mut a, mut b, mut c, mut d, mut e) = (h0, h1, h2, h3, h4);

        for i in 0..80 {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A827999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1u32),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDCu32),
                _ => (b ^ c ^ d, 0xCA62C1D6u32),
            };

            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(w[i]);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut result = [0u8; 20];
    result[0..4].copy_from_slice(&h0.to_be_bytes());
    result[4..8].copy_from_slice(&h1.to_be_bytes());
    result[8..12].copy_from_slice(&h2.to_be_bytes());
    result[12..16].copy_from_slice(&h3.to_be_bytes());
    result[16..20].copy_from_slice(&h4.to_be_bytes());
    result
}

// ---------------------------------------------------------------------------
// Base64 encode
// ---------------------------------------------------------------------------

const BASE64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        out.push(BASE64_CHARS[((triple >> 18) & 0x3F) as usize] as char);
        out.push(BASE64_CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(BASE64_CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(BASE64_CHARS[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

// ---------------------------------------------------------------------------
// WebSocket accept key computation
// ---------------------------------------------------------------------------

const WS_MAGIC: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

pub fn compute_accept_key(client_key: &str) -> String {
    let mut input = String::with_capacity(client_key.len() + WS_MAGIC.len());
    input.push_str(client_key.trim());
    input.push_str(WS_MAGIC);
    let hash = sha1(input.as_bytes());
    base64_encode(&hash)
}

// ---------------------------------------------------------------------------
// WebSocket frame encoding / decoding
// ---------------------------------------------------------------------------

/// A decoded WebSocket frame.
#[derive(Debug, Clone)]
pub struct WsFrame {
    pub opcode: u8,
    pub payload: Vec<u8>,
}

impl WsFrame {
    /// Encode a frame for sending from server to client (unmasked).
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // FIN + opcode
        buf.push(0x80 | (self.opcode & 0x0F));
        let len = self.payload.len();
        if len < 126 {
            buf.push(len as u8);
        } else if len <= 0xFFFF {
            buf.push(126);
            buf.push((len >> 8) as u8);
            buf.push((len & 0xFF) as u8);
        } else {
            buf.push(127);
            buf.extend_from_slice(&(len as u64).to_be_bytes());
        }
        buf.extend_from_slice(&self.payload);
        buf
    }

    /// Decode a frame from a client stream (expects masked frames from client).
    pub fn decode(stream: &mut impl Read) -> std::io::Result<Self> {
        let mut header = [0u8; 2];
        stream.read_exact(&mut header)?;

        let opcode = header[0] & 0x0F;
        let masked = (header[1] & 0x80) != 0;
        let len_byte = header[1] & 0x7F;

        let payload_len: usize = if len_byte < 126 {
            len_byte as usize
        } else if len_byte == 126 {
            let mut buf = [0u8; 2];
            stream.read_exact(&mut buf)?;
            u16::from_be_bytes(buf) as usize
        } else {
            let mut buf = [0u8; 8];
            stream.read_exact(&mut buf)?;
            u64::from_be_bytes(buf) as usize
        };

        let mask_key = if masked {
            let mut key = [0u8; 4];
            stream.read_exact(&mut key)?;
            Some(key)
        } else {
            None
        };

        let mut payload = vec![0u8; payload_len];
        if payload_len > 0 {
            stream.read_exact(&mut payload)?;
        }

        // Unmask
        if let Some(key) = mask_key {
            for i in 0..payload.len() {
                payload[i] ^= key[i % 4];
            }
        }

        Ok(WsFrame { opcode, payload })
    }

    /// Create a text frame.
    pub fn text(msg: &str) -> Self {
        WsFrame {
            opcode: 0x1,
            payload: msg.as_bytes().to_vec(),
        }
    }

    /// Create a close frame.
    pub fn close() -> Self {
        WsFrame {
            opcode: 0x8,
            payload: Vec::new(),
        }
    }

    /// Create a pong frame with given payload.
    pub fn pong(payload: Vec<u8>) -> Self {
        WsFrame {
            opcode: 0xA,
            payload,
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

/// Read HTTP headers directly from a TcpStream without BufReader.
/// Reads byte-by-byte to avoid any buffering that could consume WebSocket frame data.
fn read_http_request(stream: &mut TcpStream) -> Option<(String, String, HashMap<String, String>)> {
    let mut buf = Vec::with_capacity(4096);
    let mut b = [0u8; 1];

    // Read until we find \r\n\r\n (end of HTTP headers)
    loop {
        match stream.read(&mut b) {
            Ok(0) => return None,
            Ok(_) => {
                buf.push(b[0]);
                if buf.len() >= 4 && &buf[buf.len() - 4..] == b"\r\n\r\n" {
                    break;
                }
                if buf.len() > 8192 {
                    return None; // headers too large
                }
            }
            Err(_) => return None,
        }
    }

    let header_str = std::str::from_utf8(&buf).ok()?;
    let mut lines = header_str.lines();

    let request_line = lines.next()?;
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        return None;
    }
    let method = parts[0].to_string();
    let path = parts[1].to_string();

    let mut headers = HashMap::new();
    for line in lines {
        if line.is_empty() {
            break;
        }
        if let Some((key, val)) = line.split_once(':') {
            headers.insert(key.trim().to_lowercase(), val.trim().to_string());
        }
    }

    Some((method, path, headers))
}

fn content_type_for(path: &str) -> &'static str {
    if path.ends_with(".html") {
        "text/html; charset=utf-8"
    } else if path.ends_with(".js") {
        "application/javascript; charset=utf-8"
    } else if path.ends_with(".css") {
        "text/css; charset=utf-8"
    } else if path.ends_with(".json") {
        "application/json; charset=utf-8"
    } else {
        "application/octet-stream"
    }
}

fn send_http_response(stream: &mut TcpStream, status: u16, status_text: &str, content_type: &str, body: &[u8]) {
    let header = format!(
        "HTTP/1.1 {status} {status_text}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

fn send_404(stream: &mut TcpStream) {
    send_http_response(stream, 404, "Not Found", "text/plain", b"404 Not Found");
}

// ---------------------------------------------------------------------------
// WebServer
// ---------------------------------------------------------------------------

/// Shared list of connected WebSocket client streams.
type Clients = Arc<Mutex<Vec<Arc<Mutex<TcpStream>>>>>;

/// A WebSocket + HTTP server for the QLANG dashboard.
pub struct WebServer {
    clients: Clients,
    web_root: String,
}

impl WebServer {
    /// Start the HTTP + WebSocket server on the given port.
    ///
    /// This blocks the calling thread. The returned `WebServerHandle` can be
    /// used from other threads to broadcast events.
    pub fn start(port: u16, web_root: String) -> std::io::Result<WebServerHandle> {
        // Try IPv6 dual-stack first (handles both IPv4 and IPv6 on macOS/Linux),
        // fall back to IPv4-only if IPv6 is not available.
        let listener = TcpListener::bind(format!("[::0]:{port}"))
            .or_else(|_| TcpListener::bind(format!("0.0.0.0:{port}")))?;
        let clients: Clients = Arc::new(Mutex::new(Vec::new()));
        let handle = WebServerHandle {
            clients: Arc::clone(&clients),
        };

        let server = WebServer {
            clients,
            web_root,
        };

        thread::spawn(move || {
            server.run(listener);
        });

        Ok(handle)
    }

    fn run(&self, listener: TcpListener) {
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let clients = Arc::clone(&self.clients);
                    let web_root = self.web_root.clone();
                    thread::spawn(move || {
                        handle_connection(stream, clients, &web_root);
                    });
                }
                Err(e) => {
                    eprintln!("[web_server] accept error: {e}");
                }
            }
        }
    }
}

/// Handle for broadcasting events to connected WebSocket clients.
#[derive(Clone)]
pub struct WebServerHandle {
    clients: Clients,
}

impl WebServerHandle {
    /// Broadcast a `WebEvent` to all connected WebSocket clients.
    pub fn broadcast(&self, event: WebEvent) {
        let json = event.to_json();
        let frame = WsFrame::text(&json).encode();
        let mut clients = self.clients.lock().unwrap();
        clients.retain(|client| {
            let mut stream = client.lock().unwrap();
            stream.write_all(&frame).is_ok()
        });
    }
}

fn handle_connection(mut stream: TcpStream, clients: Clients, web_root: &str) {
    let peer = stream.peer_addr().ok();

    // Disable Nagle's algorithm for low-latency WebSocket communication
    let _ = stream.set_nodelay(true);

    // Read HTTP headers directly (no BufReader to avoid buffering issues)
    let (method, path, headers) = match read_http_request(&mut stream) {
        Some(v) => v,
        None => return,
    };

    // Check for WebSocket upgrade on /ws
    if method == "GET"
        && path == "/ws"
        && headers.get("upgrade").map(|v| v.eq_ignore_ascii_case("websocket")).unwrap_or(false)
    {
        if let Some(key) = headers.get("sec-websocket-key") {
            let accept = compute_accept_key(key);

            // Build 101 response with all required headers
            let mut response = format!(
                "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n"
            );

            // Echo back Sec-WebSocket-Protocol if the browser requested one
            if let Some(protocol) = headers.get("sec-websocket-protocol") {
                // Use the first requested protocol
                if let Some(first) = protocol.split(',').next() {
                    response.push_str(&format!("Sec-WebSocket-Protocol: {}\r\n", first.trim()));
                }
            }

            response.push_str("\r\n");

            if stream.write_all(response.as_bytes()).is_err() {
                return;
            }
            let _ = stream.flush();

            if let Some(addr) = peer {
                eprintln!("[web_server] WebSocket connected: {addr}");
            }

            // Clone for broadcast (other threads write to this clone)
            let client = Arc::new(Mutex::new(stream.try_clone().unwrap()));
            {
                let mut list = clients.lock().unwrap();
                list.push(Arc::clone(&client));
            }

            // Read/write loop on the same stream — no clones, no BufReader
            handle_websocket(&mut stream, &clients, &client);

            // Remove client on disconnect
            {
                let mut list = clients.lock().unwrap();
                list.retain(|c| !Arc::ptr_eq(c, &client));
            }

            if let Some(addr) = peer {
                eprintln!("[web_server] WebSocket disconnected: {addr}");
            }
        }
        return;
    }

    // Regular HTTP file serving
    if method != "GET" {
        send_404(&mut stream);
        return;
    }

    let file_path = match path.as_str() {
        "/" => "index.html".to_string(),
        "/demo" => "demo.html".to_string(),
        p => {
            let p = p.trim_start_matches('/');
            // Security: reject path traversal
            if p.contains("..") {
                send_404(&mut stream);
                return;
            }
            p.to_string()
        }
    };

    // Only serve known file types
    if !(file_path.ends_with(".html")
        || file_path.ends_with(".js")
        || file_path.ends_with(".css")
        || file_path.ends_with(".json")
        || file_path.ends_with(".svg")
        || file_path.ends_with(".png")
        || file_path.ends_with(".ico"))
    {
        send_404(&mut stream);
        return;
    }

    let full_path = format!("{web_root}/{file_path}");
    match std::fs::read(&full_path) {
        Ok(body) => {
            let ct = content_type_for(&file_path);
            send_http_response(&mut stream, 200, "OK", ct, &body);
        }
        Err(_) => {
            send_404(&mut stream);
        }
    }
}

fn handle_websocket(
    stream: &mut TcpStream,
    clients: &Clients,
    _client: &Arc<Mutex<TcpStream>>,
) {
    eprintln!("[web_server] handle_websocket: entering read loop");
    loop {
        eprintln!("[web_server] handle_websocket: waiting for frame...");
        let frame = match WsFrame::decode(stream) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[web_server] WS decode error: {e}");
                break;
            }
        };

        match frame.opcode {
            0x1 => {
                // Text frame — parse as JSON command from dashboard
                if let Ok(text) = std::str::from_utf8(&frame.payload) {
                    eprintln!("[web_server] Received WS message: {}", &text[..text.len().min(200)]);
                    if let Ok(msg) = serde_json::from_str::<serde_json::Value>(text) {
                        match msg["type"].as_str() {
                            Some("list_models") => {
                                eprintln!("[web_server] → list_models request");
                                // Query Ollama for available models, respond on this client only
                                let ollama = crate::ollama::OllamaClient::from_env();
                                let models = ollama.list_models().unwrap_or_default();
                                let response = format!(
                                    r#"{{"type":"models_list","models":[{}]}}"#,
                                    models.iter().map(|m| format!("\"{}\"", json_escape(m))).collect::<Vec<_>>().join(",")
                                );
                                let resp_frame = WsFrame::text(&response).encode();
                                let _ = stream.write_all(&resp_frame);
                            }
                            Some("start_pipeline") => {
                                eprintln!("[web_server] → start_pipeline request!");
                                let task = msg["task"].as_str().unwrap_or("Train MNIST classifier").to_string();
                                let architect_model = msg["architect_model"].as_str().unwrap_or("deepseek-r1:1.5b").to_string();
                                let evaluator_model = msg["evaluator_model"].as_str().unwrap_or("llama3.2").to_string();
                                let epochs = msg["epochs"].as_u64().unwrap_or(15) as usize;
                                let dataset = msg["dataset"].as_str().unwrap_or("synthetic").to_string();
                                let compress = msg["compress"].as_bool().unwrap_or(true);

                                run_agent_pipeline(
                                    &task,
                                    &architect_model,
                                    &evaluator_model,
                                    epochs,
                                    &dataset,
                                    compress,
                                    clients,
                                );
                            }
                            Some("predict") => {
                                // Interactive model testing: receive 784 pixels, return prediction
                                let pixels: Vec<f32> = msg["pixels"].as_array()
                                    .map(|a| a.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
                                    .unwrap_or_default();

                                if pixels.len() == 784 {
                                    let model_store = TRAINED_MODEL.get_or_init(|| Mutex::new(None));
                                    let response = if let Some(model) = model_store.lock().unwrap().as_ref() {
                                        let probs = model.forward_single(&pixels);
                                        let (digit, _conf) = probs.iter().enumerate()
                                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                                            .unwrap();
                                        let conf_arr: Vec<String> = probs.iter().map(|p| format!("{:.6}", p)).collect();
                                        format!(
                                            r#"{{"type":"prediction","digit":{},"confidence":[{}]}}"#,
                                            digit, conf_arr.join(",")
                                        )
                                    } else {
                                        r#"{"type":"prediction","error":"Kein Modell trainiert. Bitte Pipeline zuerst starten."}"#.to_string()
                                    };
                                    let resp_frame = WsFrame::text(&response).encode();
                                    let _ = stream.write_all(&resp_frame);
                                } else {
                                    let resp = r#"{"type":"prediction","error":"Erwartet 784 Pixel-Werte."}"#;
                                    let resp_frame = WsFrame::text(resp).encode();
                                    let _ = stream.write_all(&resp_frame);
                                }
                            }
                            Some("start_evolution") => {
                                eprintln!("[web_server] -> start_evolution request!");
                                let population = msg["population"].as_u64().unwrap_or(10) as usize;
                                let generations = msg["generations"].as_u64().unwrap_or(5) as usize;
                                let epochs_per = msg["epochs_per"].as_u64().unwrap_or(3) as usize;
                                run_evolution(population, generations, epochs_per, clients);
                            }
                            _ => {}
                        }
                    }
                }
            }
            0x8 => {
                // Close frame — send close back
                let close = WsFrame::close().encode();
                let _ = stream.write_all(&close);
                break;
            }
            0x9 => {
                // Ping — respond with pong
                let pong = WsFrame::pong(frame.payload).encode();
                let _ = stream.write_all(&pong);
            }
            0xA => {
                // Pong — ignore
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Agent pipeline helpers
// ---------------------------------------------------------------------------

/// Broadcast a JSON message to all connected WebSocket clients.
/// Removes clients that have disconnected (write fails).
fn broadcast_to_clients(clients: &Clients, json: &str) {
    let frame = WsFrame::text(json).encode();
    let mut list = clients.lock().unwrap();
    list.retain(|c| {
        let mut s = c.lock().unwrap();
        s.write_all(&frame).is_ok()
    });
}

/// Extract a JSON object from an LLM response that may contain surrounding text.
fn extract_json_from_response(text: &str) -> String {
    if let Some(start) = text.find('{') {
        if let Some(end) = text.rfind('}') {
            if end >= start {
                return text[start..=end].to_string();
            }
        }
    }
    text.to_string()
}

/// Run the full AI agent pipeline in a background thread.
///
/// Steps:
/// 1. Ask AI architect to design model architecture
/// 2. Build MlpWeights3 with the designed architecture
/// 3. Load training data (synthetic or MNIST)
/// 4. Train the model, broadcasting progress each epoch
/// 5. IGQK ternary compression
/// 6. AI evaluator reviews results
fn run_agent_pipeline(
    task: &str,
    architect_model: &str,
    evaluator_model: &str,
    epochs: usize,
    dataset: &str,
    compress: bool,
    clients: &Clients,
) {
    let clients = Arc::clone(clients);
    let task = task.to_string();
    let architect_model = architect_model.to_string();
    let evaluator_model = evaluator_model.to_string();
    let dataset = dataset.to_string();

    thread::spawn(move || {
        eprintln!("[pipeline] ==============================");
        eprintln!("[pipeline] Starting agent pipeline");
        eprintln!("[pipeline] Task: {}", task);
        eprintln!("[pipeline] Architect: {}", architect_model);
        eprintln!("[pipeline] Evaluator: {}", evaluator_model);
        eprintln!("[pipeline] Epochs: {}, Dataset: {}", epochs, dataset);
        eprintln!("[pipeline] ==============================");

        // Step 1: Ask AI architect to design model
        eprintln!("[pipeline] Step 1: Calling Ollama ({})...", architect_model);
        broadcast_to_clients(&clients, &format!(
            r#"{{"type":"pipeline_status","step":1,"total_steps":6,"description":"Asking AI architect ({}) to design model..."}}"#,
            json_escape(&architect_model)
        ));
        broadcast_to_clients(&clients, &format!(
            r#"{{"type":"agent_thinking","agent":"Architekt","model":"{}","status":"thinking"}}"#,
            json_escape(&architect_model)
        ));

        let ollama = crate::ollama::OllamaClient::from_env();
        let prompt = format!(
            "You are a neural network architect. Task: {}. Design a small MLP for MNIST digit classification (784 inputs, 10 outputs). Reply with ONLY a JSON object: {{\"hidden1\": N, \"hidden2\": N, \"learning_rate\": 0.1, \"batch_size\": 32}}",
            task
        );

        eprintln!("[pipeline] Sending prompt to Ollama {}...", architect_model);
        let design_response = ollama.generate(&architect_model, &prompt, Some("Output valid JSON only."))
            .unwrap_or_else(|e| {
                eprintln!("[pipeline] ❌ Architect call failed: {e}");
                r#"{"hidden1": 128, "hidden2": 64, "learning_rate": 0.1, "batch_size": 32}"#.to_string()
            });
        eprintln!("[pipeline] ✅ Architect responded: {}", &design_response[..design_response.len().min(200)]);

        broadcast_to_clients(&clients, &format!(
            r#"{{"type":"agent_response","agent":"Architekt","model":"{}","response":"{}"}}"#,
            json_escape(&architect_model), json_escape(&design_response)
        ));

        // Parse design (with sensible defaults)
        let design: serde_json::Value = serde_json::from_str(&extract_json_from_response(&design_response))
            .unwrap_or(serde_json::json!({"hidden1": 128, "hidden2": 64, "learning_rate": 0.1, "batch_size": 32}));

        let hidden1 = design["hidden1"].as_u64().unwrap_or(128).min(512) as usize;
        let hidden2 = design["hidden2"].as_u64().unwrap_or(64).min(256) as usize;
        let lr = design["learning_rate"].as_f64().unwrap_or(0.1) as f32;
        let batch_size = design["batch_size"].as_u64().unwrap_or(64).min(256) as usize;

        // Step 2: Build model
        eprintln!("[pipeline] Step 2: Building model 784→{}→{}→10", hidden1, hidden2);
        broadcast_to_clients(&clients, &format!(
            r#"{{"type":"pipeline_status","step":2,"total_steps":6,"description":"Building model: 784\u2192{}\u2192{}\u219210"}}"#,
            hidden1, hidden2
        ));

        use crate::training::MlpWeights3;
        let mut model = MlpWeights3::new(784, hidden1, hidden2, 10);

        // Send graph visualization
        broadcast_to_clients(&clients, &format!(
            r#"{{"type":"graph","nodes":[{{"id":0,"label":"Input\\n784","type":"input"}},{{"id":1,"label":"Hidden1\\n{}","type":"op"}},{{"id":2,"label":"ReLU","type":"op"}},{{"id":3,"label":"Hidden2\\n{}","type":"op"}},{{"id":4,"label":"ReLU","type":"op"}},{{"id":5,"label":"Output\\n10","type":"output"}},{{"id":6,"label":"Softmax","type":"op"}},{{"id":7,"label":"IGQK","type":"quantum"}}],"edges":[{{"from":0,"to":1}},{{"from":1,"to":2}},{{"from":2,"to":3}},{{"from":3,"to":4}},{{"from":4,"to":5}},{{"from":5,"to":6}},{{"from":6,"to":7}}]}}"#,
            hidden1, hidden2
        ));

        // Step 3: Load data
        broadcast_to_clients(&clients, &format!(
            r#"{{"type":"pipeline_status","step":3,"total_steps":6,"description":"Loading {} data..."}}"#,
            json_escape(&dataset)
        ));

        use crate::mnist::MnistData;
        let data = if dataset == "mnist" {
            MnistData::download_and_load("data/mnist")
                .unwrap_or_else(|_| MnistData::synthetic(2000, 500))
        } else {
            MnistData::synthetic(2000, 500)
        };

        broadcast_to_clients(&clients, &format!(
            r#"{{"type":"system","text":"Loaded {} training samples, {} test samples"}}"#,
            data.n_train, data.n_test
        ));

        // Step 4: Train
        eprintln!("[pipeline] Step 4: Training {} epochs...", epochs);
        broadcast_to_clients(&clients, &format!(
            r#"{{"type":"pipeline_status","step":4,"total_steps":6,"description":"Training {} epochs (batch_size={}, lr={})..."}}"#,
            epochs, batch_size, lr
        ));

        let mut current_lr = lr;
        for epoch in 0..epochs {
            if epoch > 0 && epoch % 10 == 0 {
                current_lr *= 0.95;
            }

            let n_batches = data.n_train / batch_size;
            let mut epoch_loss = 0.0f32;
            for batch_idx in 0..n_batches {
                let (x, y) = data.train_batch(batch_idx * batch_size, batch_size);
                let loss = model.train_step_backprop(x, y, current_lr);
                epoch_loss += loss;
            }
            epoch_loss /= n_batches.max(1) as f32;

            // Calculate training accuracy
            let probs = model.forward(&data.train_images);
            let acc = model.accuracy(&probs, &data.train_labels);

            eprintln!("[pipeline]   Epoch {}/{}: loss={:.4}, acc={:.1}%", epoch + 1, epochs, epoch_loss, acc * 100.0);
            // Broadcast every epoch
            broadcast_to_clients(&clients, &format!(
                r#"{{"type":"training","epoch":{},"loss":{},"accuracy":{}}}"#,
                epoch + 1, epoch_loss, acc
            ));
        }

        // Test accuracy
        let test_probs = model.forward(&data.test_images);
        let test_acc = model.accuracy(&test_probs, &data.test_labels);

        broadcast_to_clients(&clients, &format!(
            r#"{{"type":"system","text":"Training complete. Test accuracy: {:.1}%"}}"#,
            test_acc * 100.0
        ));

        // Store model for interactive testing
        {
            let model_store = TRAINED_MODEL.get_or_init(|| Mutex::new(None));
            *model_store.lock().unwrap() = Some(model.clone());
            eprintln!("[pipeline] Model stored for interactive testing");
        }

        // Notify frontend that model is ready for testing
        broadcast_to_clients(&clients, r#"{"type":"model_ready","text":"Model ready for interactive testing"}"#);

        // Step 5: Compress
        eprintln!("[pipeline] Step 5: IGQK Compression...");
        if compress {
            broadcast_to_clients(&clients, &format!(
                r#"{{"type":"pipeline_status","step":5,"total_steps":6,"description":"IGQK Ternary Compression..."}}"#
            ));

            let compressed = model.compress_ternary();
            let comp_probs = compressed.forward(&data.test_images);
            let comp_acc = compressed.accuracy(&comp_probs, &data.test_labels);

            let total_params = model.param_count();
            let original_kb = total_params as f64 * 4.0 / 1024.0;
            let weight_count = model.w1.len() + model.w2.len() + model.w3.len();
            let ternary_bytes = (weight_count * 2 + 7) / 8;
            let bias_bytes = (model.b1.len() + model.b2.len() + model.b3.len()) * 4;
            let compressed_kb = (ternary_bytes + bias_bytes) as f64 / 1024.0;
            let ratio = original_kb / compressed_kb;

            broadcast_to_clients(&clients, &format!(
                r#"{{"type":"compression","method":"IGQK Ternary","ratio":"{:.1}x","original_size":"{:.1} KB","compressed_size":"{:.1} KB","accuracy_delta":"{:.1}"}}"#,
                ratio, original_kb, compressed_kb, (comp_acc - test_acc) * 100.0
            ));

            // Step 6: AI Evaluator
            broadcast_to_clients(&clients, &format!(
                r#"{{"type":"pipeline_status","step":6,"total_steps":6,"description":"AI Evaluator ({}) reviewing results..."}}"#,
                json_escape(&evaluator_model)
            ));
            broadcast_to_clients(&clients, &format!(
                r#"{{"type":"agent_thinking","agent":"Evaluator","model":"{}","status":"thinking"}}"#,
                json_escape(&evaluator_model)
            ));

            let eval_prompt = format!(
                "I trained a neural network for MNIST: 784\u{2192}{}\u{2192}{}\u{2192}10. Results: Test accuracy: {:.1}%, Compressed accuracy: {:.1}%, Compression: {:.1}x, Size: {:.1}KB\u{2192}{:.1}KB. Evaluate in 2 sentences.",
                hidden1, hidden2, test_acc * 100.0, comp_acc * 100.0, ratio, original_kb, compressed_kb
            );

            eprintln!("[pipeline] Step 6: Calling Evaluator ({})...", evaluator_model);
            let evaluation = ollama.generate(&evaluator_model, &eval_prompt, None)
                .unwrap_or_else(|e| {
                    eprintln!("[pipeline] ❌ Evaluator call failed: {e}");
                    "Evaluation unavailable.".to_string()
                });
            eprintln!("[pipeline] ✅ Evaluator: {}", &evaluation[..evaluation.len().min(200)]);

            broadcast_to_clients(&clients, &format!(
                r#"{{"type":"agent_response","agent":"Evaluator","model":"{}","response":"{}"}}"#,
                json_escape(&evaluator_model), json_escape(&evaluation)
            ));
        } else {
            // Skip compression, jump to step 6
            broadcast_to_clients(&clients, r#"{"type":"pipeline_status","step":5,"total_steps":6,"description":"Compression skipped."}"#);
        }

        broadcast_to_clients(&clients, r#"{"type":"pipeline_status","step":6,"total_steps":6,"description":"Pipeline complete!"}"#);
        eprintln!("[pipeline] ==============================");
        eprintln!("[pipeline] ✅ Pipeline complete!");
        eprintln!("[pipeline] ==============================");
    });
}

// ---------------------------------------------------------------------------
// Neuroevolution
// ---------------------------------------------------------------------------

/// Run evolutionary optimization of neural network architectures.
///
/// 1. Generate `pop_size` random architectures (different hidden layer sizes)
/// 2. Train each one briefly (`epochs_per` epochs)
/// 3. Evaluate fitness (accuracy)
/// 4. Select top 30%, mutate to refill population
/// 5. Repeat for `generations` generations
/// 6. Final: train best architecture with more epochs, store as TRAINED_MODEL
fn run_evolution(pop_size: usize, generations: usize, epochs_per: usize, clients: &Clients) {
    let clients = Arc::clone(clients);

    thread::spawn(move || {
        eprintln!("[evolution] ==============================");
        eprintln!("[evolution] Starting neuroevolution: pop={}, gen={}, epochs_per={}", pop_size, generations, epochs_per);
        eprintln!("[evolution] ==============================");

        broadcast_to_clients(&clients, r#"{"type":"system","text":"Neuroevolution gestartet..."}"#);

        // Generate synthetic data for quick evaluation
        use crate::mnist::MnistData;
        let data = MnistData::synthetic(1000, 200);

        // Initial population: random architectures (hidden1, hidden2, fitness)
        let mut population: Vec<(usize, usize, f32)> = (0..pop_size).map(|_| {
            let h1 = 32 + (rand_u32() % 224) as usize; // 32-256
            let h2 = 16 + (rand_u32() % 112) as usize;  // 16-128
            (h1, h2, 0.0)
        }).collect();

        for g in 0..generations {
            eprintln!("[evolution] Generation {}/{}", g + 1, generations);
            broadcast_to_clients(&clients, &format!(
                r#"{{"type":"evolution_generation","generation":{},"total_generations":{},"population_size":{}}}"#,
                g + 1, generations, population.len()
            ));

            // Train and evaluate each individual
            for i in 0..population.len() {
                let (h1, h2, _) = population[i];
                let mut model = MlpWeights3::new(784, h1, h2, 10);

                // Quick training
                let batch_size = 64usize;
                for _ep in 0..epochs_per {
                    let n_batches = data.n_train / batch_size;
                    for b in 0..n_batches {
                        let (x, y) = data.train_batch(b * batch_size, batch_size);
                        model.train_step_backprop(x, y, 0.1);
                    }
                }

                // Evaluate on test set
                let probs = model.forward(&data.test_images);
                let acc = model.accuracy(&probs, &data.test_labels);
                population[i].2 = acc;

                eprintln!("[evolution]   Individual {}/{}: {}x{} -> acc={:.1}%", i + 1, population.len(), h1, h2, acc * 100.0);
                broadcast_to_clients(&clients, &format!(
                    r#"{{"type":"evolution_individual","generation":{},"index":{},"hidden1":{},"hidden2":{},"accuracy":{}}}"#,
                    g + 1, i, h1, h2, acc
                ));
            }

            // Sort by fitness (descending accuracy)
            population.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

            let best = &population[0];
            eprintln!("[evolution]   Best: {}x{} acc={:.1}%", best.0, best.1, best.2 * 100.0);
            broadcast_to_clients(&clients, &format!(
                r#"{{"type":"evolution_best","generation":{},"hidden1":{},"hidden2":{},"accuracy":{}}}"#,
                g + 1, best.0, best.1, best.2
            ));

            // Selection & mutation: keep top 30%, mutate rest
            if g < generations - 1 {
                let survivors = (pop_size * 3 / 10).max(1);
                let mut next_gen: Vec<(usize, usize, f32)> = population[..survivors].to_vec();

                while next_gen.len() < pop_size {
                    let parent = &next_gen[rand_u32() as usize % survivors];
                    let h1 = ((parent.0 as i32) + (rand_i32() % 33 - 16)).max(16) as usize;
                    let h2 = ((parent.1 as i32) + (rand_i32() % 17 - 8)).max(8) as usize;
                    next_gen.push((h1.min(512), h2.min(256), 0.0));
                }

                population = next_gen;
            }
        }

        // Final training of the best architecture with more epochs
        let (best_h1, best_h2, best_acc) = population[0];
        eprintln!("[evolution] Final training: {}x{} (best acc={:.1}%)", best_h1, best_h2, best_acc * 100.0);
        broadcast_to_clients(&clients, &format!(
            r#"{{"type":"system","text":"Evolution komplett! Bestes Modell: {}x{} ({:.1}%). Finales Training..."}}"#,
            best_h1, best_h2, best_acc * 100.0
        ));

        // Train the winner properly
        let mut final_model = MlpWeights3::new(784, best_h1, best_h2, 10);
        let final_epochs = 10;
        let batch_size = 64usize;
        for epoch in 0..final_epochs {
            let n_batches = data.n_train / batch_size;
            for b in 0..n_batches {
                let (x, y) = data.train_batch(b * batch_size, batch_size);
                final_model.train_step_backprop(x, y, 0.1);
            }
            let probs = final_model.forward(&data.test_images);
            let acc = final_model.accuracy(&probs, &data.test_labels);
            broadcast_to_clients(&clients, &format!(
                r#"{{"type":"training","epoch":{},"loss":0,"accuracy":{}}}"#,
                epoch + 1, acc
            ));
        }

        let final_probs = final_model.forward(&data.test_images);
        let final_acc = final_model.accuracy(&final_probs, &data.test_labels);

        // Store as the active model for interactive prediction
        {
            let model_store = TRAINED_MODEL.get_or_init(|| Mutex::new(None));
            *model_store.lock().unwrap() = Some(final_model);
            eprintln!("[evolution] Model stored for interactive testing");
        }

        broadcast_to_clients(&clients, &format!(
            r#"{{"type":"evolution_complete","best_hidden1":{},"best_hidden2":{},"final_accuracy":{}}}"#,
            best_h1, best_h2, final_acc
        ));
        broadcast_to_clients(&clients, r#"{"type":"model_ready","text":"Evolution model ready for interactive testing"}"#);

        eprintln!("[evolution] ==============================");
        eprintln!("[evolution] Evolution complete! Best: {}x{}, acc={:.1}%", best_h1, best_h2, final_acc * 100.0);
        eprintln!("[evolution] ==============================");
    });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha1_empty() {
        let hash = sha1(b"");
        let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    }

    #[test]
    fn test_sha1_abc() {
        let hash = sha1(b"abc");
        let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "a9993e364706816aba3e25717850c26c9cd0d89d");
    }

    #[test]
    fn test_sha1_longer() {
        let hash = sha1(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "84983e441c3bd26ebaae4aa1f95129e5e54670f1");
    }

    #[test]
    fn test_base64_encode() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn test_websocket_accept_key() {
        // RFC 6455 Section 4.2.2 official test vector
        let key = "dGhlIHNhbXBsZSBub25jZQ==";
        let accept = compute_accept_key(key);
        assert_eq!(accept, "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");

        // Wikipedia WebSocket example
        let key2 = "x3JJHMbDL1EzLkh9GBhXDw==";
        let accept2 = compute_accept_key(key2);
        assert_eq!(accept2, "HSmrc0sMlYUkAGmm5OPpG2HaGWk=");
    }

    #[test]
    fn test_ws_frame_encode_short() {
        let frame = WsFrame::text("hello");
        let encoded = frame.encode();
        assert_eq!(encoded[0], 0x81); // FIN + text opcode
        assert_eq!(encoded[1], 5);    // payload length
        assert_eq!(&encoded[2..], b"hello");
    }

    #[test]
    fn test_ws_frame_encode_medium() {
        // 200-byte payload (uses 2-byte extended length)
        let payload = "x".repeat(200);
        let frame = WsFrame::text(&payload);
        let encoded = frame.encode();
        assert_eq!(encoded[0], 0x81);
        assert_eq!(encoded[1], 126);   // extended length marker
        assert_eq!(encoded[2], 0);     // length high byte
        assert_eq!(encoded[3], 200);   // length low byte
        assert_eq!(encoded.len(), 4 + 200);
    }

    #[test]
    fn test_ws_frame_decode_roundtrip() {
        // Simulate a masked client frame
        let payload = b"hello";
        let mask_key: [u8; 4] = [0x37, 0xfa, 0x21, 0x3d];
        let mut masked_payload = payload.to_vec();
        for i in 0..masked_payload.len() {
            masked_payload[i] ^= mask_key[i % 4];
        }

        let mut frame_data = Vec::new();
        frame_data.push(0x81); // FIN + text
        frame_data.push(0x80 | 5); // masked + length 5
        frame_data.extend_from_slice(&mask_key);
        frame_data.extend_from_slice(&masked_payload);

        let mut cursor = std::io::Cursor::new(frame_data);
        let decoded = WsFrame::decode(&mut cursor).unwrap();
        assert_eq!(decoded.opcode, 0x1);
        assert_eq!(decoded.payload, b"hello");
    }

    #[test]
    fn test_ws_frame_close() {
        let frame = WsFrame::close();
        let encoded = frame.encode();
        assert_eq!(encoded[0], 0x88); // FIN + close opcode
        assert_eq!(encoded[1], 0);
    }

    #[test]
    fn test_ws_frame_pong() {
        let frame = WsFrame::pong(b"pingdata".to_vec());
        let encoded = frame.encode();
        assert_eq!(encoded[0], 0x8A); // FIN + pong opcode
        assert_eq!(encoded[1], 8);
        assert_eq!(&encoded[2..], b"pingdata");
    }

    #[test]
    fn test_event_json_graph_node_executed() {
        let event = WebEvent::GraphNodeExecuted {
            node_id: 1,
            op: "MatMul".to_string(),
            shape: "[2,3]".to_string(),
            time_us: 42,
            values: Some(vec![1.0, 2.0]),
        };
        let json = event.to_json();
        assert!(json.contains("\"type\":\"node_exec\""));
        assert!(json.contains("\"node_id\":1"));
        assert!(json.contains("\"op\":\"MatMul\""));
        assert!(json.contains("\"time_ms\":"));
        assert!(json.contains("[1,2]"));
    }

    #[test]
    fn test_event_json_training_epoch() {
        let event = WebEvent::TrainingEpoch {
            epoch: 5,
            loss: 0.5,
            accuracy: 0.9,
        };
        let json = event.to_json();
        assert!(json.contains("\"type\":\"training\""));
        assert!(json.contains("\"epoch\":5"));
    }

    #[test]
    fn test_event_json_system_log_escape() {
        let event = WebEvent::SystemLog {
            level: "info".to_string(),
            message: "line1\nline2\"quoted\"".to_string(),
        };
        let json = event.to_json();
        assert!(json.contains("\\n"));
        assert!(json.contains("\\\"quoted\\\""));
    }

    #[test]
    fn test_event_json_agent_message() {
        let event = WebEvent::AgentMessage {
            from: "agent_a".to_string(),
            to: "agent_b".to_string(),
            message: "hello".to_string(),
        };
        let json = event.to_json();
        assert!(json.contains("\"type\":\"agent\""));
        assert!(json.contains("\"from\":\"agent_a\""));
    }

    #[test]
    fn test_event_json_compression_result() {
        let event = WebEvent::CompressionResult {
            method: "ternary".to_string(),
            ratio: 0.25,
            accuracy_before: 0.95,
            accuracy_after: 0.93,
        };
        let json = event.to_json();
        assert!(json.contains("\"type\":\"compression\""));
        assert!(json.contains("\"method\":\"ternary\""));
    }

    #[test]
    fn test_event_json_graph_loaded() {
        let event = WebEvent::GraphLoaded {
            name: "test_graph".to_string(),
            num_nodes: 10,
            num_edges: 15,
        };
        let json = event.to_json();
        assert!(json.contains("\"type\":\"graph\""));
        assert!(json.contains("\"nodes\":["));
    }

    #[test]
    fn test_event_json_model_saved() {
        let event = WebEvent::ModelSaved {
            name: "model1".to_string(),
            version: "v1.0".to_string(),
        };
        let json = event.to_json();
        assert!(json.contains("\"type\":\"model_saved\""));
        assert!(json.contains("\"size\":\"v1.0\""));
    }

    #[test]
    fn test_event_json_graph_node_no_values() {
        let event = WebEvent::GraphNodeExecuted {
            node_id: 2,
            op: "ReLU".to_string(),
            shape: "[4]".to_string(),
            time_us: 10,
            values: None,
        };
        let json = event.to_json();
        assert!(json.contains("\"values\":null"));
    }

    #[test]
    fn test_extract_json_from_response_clean() {
        let input = r#"{"hidden1": 128, "hidden2": 64}"#;
        assert_eq!(extract_json_from_response(input), input);
    }

    #[test]
    fn test_extract_json_from_response_with_prefix() {
        let input = r#"Here is the JSON: {"hidden1": 128, "hidden2": 64}"#;
        assert_eq!(
            extract_json_from_response(input),
            r#"{"hidden1": 128, "hidden2": 64}"#
        );
    }

    #[test]
    fn test_extract_json_from_response_with_suffix() {
        let input = r#"{"hidden1": 128} - this is my design"#;
        assert_eq!(
            extract_json_from_response(input),
            r#"{"hidden1": 128}"#
        );
    }

    #[test]
    fn test_extract_json_from_response_no_json() {
        let input = "no json here";
        assert_eq!(extract_json_from_response(input), input);
    }
}
