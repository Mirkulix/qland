//! QLANG Network Server — TCP-based graph exchange and remote execution.
//!
//! Provides a simple synchronous TCP server and client for exchanging
//! QLANG graphs between agents over the network. Uses JSON-over-TCP
//! as the wire format (length-prefixed JSON messages).
//!
//! Protocol: each message is sent as a 4-byte little-endian length prefix
//! followed by a JSON-encoded `Request` or `Response`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use qlang_core::graph::Graph;
use qlang_core::tensor::TensorData;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Unique identifier for a stored graph.
pub type GraphId = u64;

/// Compression method to apply to a graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CompressionMethod {
    /// Ternary weight compression ({-1, 0, +1}).
    Ternary,
    /// Low-rank approximation with target rank.
    LowRank { rank: usize },
    /// Sparsity-based compression with target sparsity (0.0 .. 1.0).
    Sparse { sparsity: f32 },
}

/// Summary information about a stored graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphInfo {
    pub id: GraphId,
    pub name: String,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub version: String,
    pub metadata: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Protocol messages
// ---------------------------------------------------------------------------

/// A request from client to server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Request {
    /// Submit a graph for storage. Returns the assigned GraphId.
    SubmitGraph(Graph),
    /// Execute a previously submitted graph with the given inputs.
    ExecuteGraph {
        graph_id: GraphId,
        inputs: HashMap<String, TensorData>,
    },
    /// Retrieve metadata about a stored graph.
    GetGraphInfo(GraphId),
    /// List all stored graph IDs.
    ListGraphs,
    /// Compress a stored graph using the specified method.
    /// Returns the GraphId of the newly created compressed graph.
    CompressGraph {
        graph_id: GraphId,
        method: CompressionMethod,
    },
}

/// A response from server to client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Response {
    /// A graph was successfully submitted.
    GraphSubmitted(GraphId),
    /// Graph execution completed.
    ExecutionResult {
        outputs: HashMap<String, TensorData>,
    },
    /// Graph info retrieved.
    GraphInfo(GraphInfo),
    /// List of all graph IDs.
    GraphList(Vec<GraphId>),
    /// A compressed copy of the graph was stored.
    GraphCompressed {
        original_id: GraphId,
        compressed_id: GraphId,
    },
    /// An error occurred while processing the request.
    Error(String),
}

// ---------------------------------------------------------------------------
// Wire format helpers (length-prefixed JSON)
// ---------------------------------------------------------------------------

/// Server-side error type.
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("graph not found: {0}")]
    GraphNotFound(GraphId),
    #[error("server error: {0}")]
    Other(String),
}

/// Write a length-prefixed JSON message to a stream.
pub fn write_message<W: Write, T: Serialize>(writer: &mut W, msg: &T) -> Result<(), ServerError> {
    let json = serde_json::to_vec(msg)?;
    let len = json.len() as u32;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&json)?;
    writer.flush()?;
    Ok(())
}

/// Read a length-prefixed JSON message from a stream.
pub fn read_message<R: Read, T: for<'de> Deserialize<'de>>(
    reader: &mut R,
) -> Result<T, ServerError> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;

    let mut json_buf = vec![0u8; len];
    reader.read_exact(&mut json_buf)?;

    let msg: T = serde_json::from_slice(&json_buf)?;
    Ok(msg)
}

// ---------------------------------------------------------------------------
// Graph store
// ---------------------------------------------------------------------------

/// Thread-safe in-memory store for QLANG graphs.
#[derive(Debug, Clone)]
pub struct GraphStore {
    inner: Arc<Mutex<GraphStoreInner>>,
}

#[derive(Debug)]
struct GraphStoreInner {
    graphs: HashMap<GraphId, Graph>,
    next_id: GraphId,
}

impl GraphStore {
    /// Create an empty graph store.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(GraphStoreInner {
                graphs: HashMap::new(),
                next_id: 0,
            })),
        }
    }

    /// Insert a graph and return its assigned ID.
    pub fn insert(&self, graph: Graph) -> GraphId {
        let mut inner = self.inner.lock().unwrap();
        let id = inner.next_id;
        inner.next_id += 1;
        inner.graphs.insert(id, graph);
        id
    }

    /// Retrieve a graph by ID.
    pub fn get(&self, id: GraphId) -> Option<Graph> {
        let inner = self.inner.lock().unwrap();
        inner.graphs.get(&id).cloned()
    }

    /// Get summary info for a graph.
    pub fn get_info(&self, id: GraphId) -> Option<GraphInfo> {
        let inner = self.inner.lock().unwrap();
        inner.graphs.get(&id).map(|g| GraphInfo {
            id,
            name: g.id.clone(),
            num_nodes: g.nodes.len(),
            num_edges: g.edges.len(),
            version: g.version.clone(),
            metadata: g.metadata.clone(),
        })
    }

    /// List all stored graph IDs.
    pub fn list(&self) -> Vec<GraphId> {
        let inner = self.inner.lock().unwrap();
        let mut ids: Vec<GraphId> = inner.graphs.keys().copied().collect();
        ids.sort();
        ids
    }
}

impl Default for GraphStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Request handler
// ---------------------------------------------------------------------------

/// Process a single request against the graph store.
///
/// Execution and compression are stubbed: execution returns an empty output
/// map, and compression stores a clone of the original graph (a real
/// implementation would call into `qlang-runtime`).
pub fn handle_request(store: &GraphStore, request: &Request) -> Response {
    match request {
        Request::SubmitGraph(graph) => {
            let id = store.insert(graph.clone());
            Response::GraphSubmitted(id)
        }
        Request::ExecuteGraph { graph_id, inputs } => {
            match store.get(*graph_id) {
                Some(graph) => {
                    // Stub: a real implementation would call qlang_runtime::executor::execute.
                    // We validate the graph and return an empty output set.
                    if let Err(errors) = graph.validate() {
                        return Response::Error(format!(
                            "graph validation failed: {:?}",
                            errors.iter().map(|e| e.to_string()).collect::<Vec<_>>()
                        ));
                    }
                    let _ = inputs; // acknowledged but unused in stub
                    Response::ExecutionResult {
                        outputs: HashMap::new(),
                    }
                }
                None => Response::Error(format!("graph not found: {graph_id}")),
            }
        }
        Request::GetGraphInfo(graph_id) => match store.get_info(*graph_id) {
            Some(info) => Response::GraphInfo(info),
            None => Response::Error(format!("graph not found: {graph_id}")),
        },
        Request::ListGraphs => Response::GraphList(store.list()),
        Request::CompressGraph { graph_id, method } => {
            match store.get(*graph_id) {
                Some(graph) => {
                    // Stub: a real implementation would apply the compression method
                    // (ternary quantization, low-rank decomposition, etc.)
                    // For now, store a copy with metadata noting the method.
                    let mut compressed = graph.clone();
                    compressed
                        .metadata
                        .insert("compressed_from".to_string(), graph_id.to_string());
                    compressed
                        .metadata
                        .insert("compression_method".to_string(), format!("{method:?}"));
                    let compressed_id = store.insert(compressed);
                    Response::GraphCompressed {
                        original_id: *graph_id,
                        compressed_id,
                    }
                }
                None => Response::Error(format!("graph not found: {graph_id}")),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/// A synchronous TCP server that accepts QLANG protocol requests.
///
/// Each incoming connection is handled on the calling thread (one request
/// per connection). For concurrent use, spawn a thread per connection
/// externally.
pub struct Server {
    listener: TcpListener,
    store: GraphStore,
}

impl Server {
    /// Bind to the given address (e.g. `"127.0.0.1:9100"`).
    pub fn bind(addr: &str) -> Result<Self, ServerError> {
        let listener = TcpListener::bind(addr)?;
        Ok(Self {
            listener,
            store: GraphStore::new(),
        })
    }

    /// Bind with a pre-existing graph store (useful for sharing state).
    pub fn bind_with_store(addr: &str, store: GraphStore) -> Result<Self, ServerError> {
        let listener = TcpListener::bind(addr)?;
        Ok(Self { listener, store })
    }

    /// Return the local address the server is bound to.
    pub fn local_addr(&self) -> io::Result<std::net::SocketAddr> {
        self.listener.local_addr()
    }

    /// Return a reference to the graph store.
    pub fn store(&self) -> &GraphStore {
        &self.store
    }

    /// Handle a single incoming connection (blocking).
    /// Reads one request, processes it, writes the response, then returns.
    pub fn handle_one(&self) -> Result<(), ServerError> {
        let (mut stream, _addr) = self.listener.accept()?;
        let request: Request = read_message(&mut stream)?;
        let response = handle_request(&self.store, &request);
        write_message(&mut stream, &response)?;
        Ok(())
    }

    /// Run the server loop, handling connections until an error occurs.
    /// Each connection handles exactly one request-response exchange.
    pub fn run(&self) -> Result<(), ServerError> {
        loop {
            if let Err(e) = self.handle_one() {
                // Log but continue for non-fatal I/O errors from clients
                eprintln!("qlang-server: connection error: {e}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// A synchronous TCP client for the QLANG server protocol.
///
/// Opens a new TCP connection for each request (connection-per-request).
pub struct Client {
    addr: String,
}

impl Client {
    /// Create a client that will connect to the given server address.
    pub fn new(addr: &str) -> Self {
        Self {
            addr: addr.to_string(),
        }
    }

    /// Send a raw request and receive the response.
    pub fn send(&self, request: &Request) -> Result<Response, ServerError> {
        let mut stream = TcpStream::connect(&self.addr)?;
        write_message(&mut stream, request)?;
        let response: Response = read_message(&mut stream)?;
        Ok(response)
    }

    /// Submit a graph and return its ID.
    pub fn submit_graph(&self, graph: Graph) -> Result<GraphId, ServerError> {
        match self.send(&Request::SubmitGraph(graph))? {
            Response::GraphSubmitted(id) => Ok(id),
            Response::Error(e) => Err(ServerError::Other(e)),
            other => Err(ServerError::Other(format!(
                "unexpected response: {other:?}"
            ))),
        }
    }

    /// Execute a graph with the given inputs.
    pub fn execute_graph(
        &self,
        graph_id: GraphId,
        inputs: HashMap<String, TensorData>,
    ) -> Result<HashMap<String, TensorData>, ServerError> {
        match self.send(&Request::ExecuteGraph { graph_id, inputs })? {
            Response::ExecutionResult { outputs } => Ok(outputs),
            Response::Error(e) => Err(ServerError::Other(e)),
            other => Err(ServerError::Other(format!(
                "unexpected response: {other:?}"
            ))),
        }
    }

    /// Get info about a stored graph.
    pub fn get_graph_info(&self, graph_id: GraphId) -> Result<GraphInfo, ServerError> {
        match self.send(&Request::GetGraphInfo(graph_id))? {
            Response::GraphInfo(info) => Ok(info),
            Response::Error(e) => Err(ServerError::Other(e)),
            other => Err(ServerError::Other(format!(
                "unexpected response: {other:?}"
            ))),
        }
    }

    /// List all graph IDs on the server.
    pub fn list_graphs(&self) -> Result<Vec<GraphId>, ServerError> {
        match self.send(&Request::ListGraphs)? {
            Response::GraphList(ids) => Ok(ids),
            Response::Error(e) => Err(ServerError::Other(e)),
            other => Err(ServerError::Other(format!(
                "unexpected response: {other:?}"
            ))),
        }
    }

    /// Compress a graph using the specified method.
    pub fn compress_graph(
        &self,
        graph_id: GraphId,
        method: CompressionMethod,
    ) -> Result<GraphId, ServerError> {
        match self.send(&Request::CompressGraph { graph_id, method })? {
            Response::GraphCompressed { compressed_id, .. } => Ok(compressed_id),
            Response::Error(e) => Err(ServerError::Other(e)),
            other => Err(ServerError::Other(format!(
                "unexpected response: {other:?}"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::graph::Graph;
    use qlang_core::ops::Op;
    use qlang_core::tensor::{Shape, TensorData, TensorType};

    /// Helper: create a simple valid graph.
    fn sample_graph(name: &str) -> Graph {
        let mut g = Graph::new(name);
        g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![TensorType::f32_vector(4)],
        );
        g.add_node(
            Op::Relu,
            vec![TensorType::f32_vector(4)],
            vec![TensorType::f32_vector(4)],
        );
        g.add_node(
            Op::Output { name: "y".into() },
            vec![TensorType::f32_vector(4)],
            vec![],
        );
        g.add_edge(0, 0, 1, 0, TensorType::f32_vector(4));
        g.add_edge(1, 0, 2, 0, TensorType::f32_vector(4));
        g
    }

    // ---- Serialization round-trip helpers ----

    /// Serialize to JSON, deserialize back, re-serialize, and assert the JSON strings match.
    fn assert_json_roundtrip<T: Serialize + for<'de> Deserialize<'de>>(value: &T) {
        let json1 = serde_json::to_string(value).unwrap();
        let decoded: T = serde_json::from_str(&json1).unwrap();
        let json2 = serde_json::to_string(&decoded).unwrap();
        assert_eq!(json1, json2);
    }

    // ---- Serialization round-trip tests ----

    #[test]
    fn request_submit_graph_roundtrip() {
        let req = Request::SubmitGraph(sample_graph("roundtrip"));
        assert_json_roundtrip(&req);
        // Also verify the JSON contains expected fields
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("SubmitGraph"));
        assert!(json.contains("roundtrip"));
    }

    #[test]
    fn request_execute_graph_roundtrip() {
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".into(),
            TensorData::from_f32(Shape::vector(4), &[1.0, 2.0, 3.0, 4.0]),
        );
        let req = Request::ExecuteGraph {
            graph_id: 42,
            inputs,
        };
        assert_json_roundtrip(&req);
    }

    #[test]
    fn request_get_graph_info_roundtrip() {
        let req = Request::GetGraphInfo(7);
        assert_json_roundtrip(&req);
    }

    #[test]
    fn request_list_graphs_roundtrip() {
        let req = Request::ListGraphs;
        assert_json_roundtrip(&req);
    }

    #[test]
    fn request_compress_graph_roundtrip() {
        assert_json_roundtrip(&Request::CompressGraph {
            graph_id: 3,
            method: CompressionMethod::Ternary,
        });
        assert_json_roundtrip(&Request::CompressGraph {
            graph_id: 5,
            method: CompressionMethod::LowRank { rank: 16 },
        });
        assert_json_roundtrip(&Request::CompressGraph {
            graph_id: 5,
            method: CompressionMethod::Sparse { sparsity: 0.9 },
        });
    }

    #[test]
    fn response_variants_roundtrip() {
        assert_json_roundtrip(&Response::GraphSubmitted(42));
        assert_json_roundtrip(&Response::ExecutionResult {
            outputs: HashMap::new(),
        });
        assert_json_roundtrip(&Response::GraphInfo(GraphInfo {
            id: 1,
            name: "test".into(),
            num_nodes: 3,
            num_edges: 2,
            version: "0.1".into(),
            metadata: HashMap::new(),
        }));
        assert_json_roundtrip(&Response::GraphList(vec![0, 1, 2]));
        assert_json_roundtrip(&Response::GraphCompressed {
            original_id: 0,
            compressed_id: 1,
        });
        assert_json_roundtrip(&Response::Error("something went wrong".into()));
    }

    // ---- Wire format tests ----

    #[test]
    fn write_read_message_roundtrip() {
        let req = Request::ListGraphs;
        let json_before = serde_json::to_string(&req).unwrap();

        let mut buf: Vec<u8> = Vec::new();
        write_message(&mut buf, &req).unwrap();

        // Verify length prefix is correct
        let len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        assert_eq!(len, buf.len() - 4);

        let mut cursor = io::Cursor::new(buf);
        let decoded: Request = read_message(&mut cursor).unwrap();
        let json_after = serde_json::to_string(&decoded).unwrap();
        assert_eq!(json_before, json_after);
    }

    #[test]
    fn write_read_response_roundtrip() {
        let resp = Response::GraphList(vec![10, 20, 30]);
        let json_before = serde_json::to_string(&resp).unwrap();

        let mut buf: Vec<u8> = Vec::new();
        write_message(&mut buf, &resp).unwrap();

        let mut cursor = io::Cursor::new(buf);
        let decoded: Response = read_message(&mut cursor).unwrap();
        let json_after = serde_json::to_string(&decoded).unwrap();
        assert_eq!(json_before, json_after);
    }

    // ---- Graph store tests ----

    #[test]
    fn graph_store_insert_and_get() {
        let store = GraphStore::new();
        let g = sample_graph("store_test");

        let id = store.insert(g.clone());
        assert_eq!(id, 0);

        let retrieved = store.get(id).unwrap();
        assert_eq!(retrieved.id, "store_test");
        assert_eq!(retrieved.nodes.len(), 3);
    }

    #[test]
    fn graph_store_list() {
        let store = GraphStore::new();
        store.insert(sample_graph("a"));
        store.insert(sample_graph("b"));
        store.insert(sample_graph("c"));

        let ids = store.list();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn graph_store_get_info() {
        let store = GraphStore::new();
        let id = store.insert(sample_graph("info_test"));

        let info = store.get_info(id).unwrap();
        assert_eq!(info.id, 0);
        assert_eq!(info.name, "info_test");
        assert_eq!(info.num_nodes, 3);
        assert_eq!(info.num_edges, 2);
    }

    #[test]
    fn graph_store_missing_graph() {
        let store = GraphStore::new();
        assert!(store.get(999).is_none());
        assert!(store.get_info(999).is_none());
    }

    // ---- Request handler tests ----

    #[test]
    fn handle_submit_graph() {
        let store = GraphStore::new();
        let resp = handle_request(&store, &Request::SubmitGraph(sample_graph("submit")));
        match resp {
            Response::GraphSubmitted(id) => assert_eq!(id, 0),
            other => panic!("expected GraphSubmitted, got {other:?}"),
        }
    }

    #[test]
    fn handle_list_graphs() {
        let store = GraphStore::new();
        store.insert(sample_graph("a"));
        store.insert(sample_graph("b"));

        let resp = handle_request(&store, &Request::ListGraphs);
        match resp {
            Response::GraphList(ids) => assert_eq!(ids, vec![0, 1]),
            other => panic!("expected GraphList, got {other:?}"),
        }
    }

    #[test]
    fn handle_get_graph_info() {
        let store = GraphStore::new();
        let id = store.insert(sample_graph("info"));
        let resp = handle_request(&store, &Request::GetGraphInfo(id));

        match resp {
            Response::GraphInfo(info) => {
                assert_eq!(info.name, "info");
                assert_eq!(info.num_nodes, 3);
            }
            other => panic!("expected GraphInfo, got {other:?}"),
        }
    }

    #[test]
    fn handle_get_graph_info_not_found() {
        let store = GraphStore::new();
        let resp = handle_request(&store, &Request::GetGraphInfo(999));
        match resp {
            Response::Error(msg) => assert!(msg.contains("not found")),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn handle_execute_graph() {
        let store = GraphStore::new();
        let id = store.insert(sample_graph("exec"));

        let resp = handle_request(
            &store,
            &Request::ExecuteGraph {
                graph_id: id,
                inputs: HashMap::new(),
            },
        );
        match resp {
            Response::ExecutionResult { outputs } => {
                // Stub returns empty outputs
                assert!(outputs.is_empty());
            }
            other => panic!("expected ExecutionResult, got {other:?}"),
        }
    }

    #[test]
    fn handle_execute_graph_not_found() {
        let store = GraphStore::new();
        let resp = handle_request(
            &store,
            &Request::ExecuteGraph {
                graph_id: 999,
                inputs: HashMap::new(),
            },
        );
        match resp {
            Response::Error(msg) => assert!(msg.contains("not found")),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn handle_compress_graph() {
        let store = GraphStore::new();
        let id = store.insert(sample_graph("compress_me"));

        let resp = handle_request(
            &store,
            &Request::CompressGraph {
                graph_id: id,
                method: CompressionMethod::Ternary,
            },
        );
        match resp {
            Response::GraphCompressed {
                original_id,
                compressed_id,
            } => {
                assert_eq!(original_id, id);
                assert_ne!(compressed_id, id);

                // The compressed graph should have metadata about its origin
                let info = store.get_info(compressed_id).unwrap();
                assert_eq!(info.name, "compress_me");
                let compressed_graph = store.get(compressed_id).unwrap();
                assert_eq!(
                    compressed_graph.metadata.get("compressed_from").unwrap(),
                    &id.to_string()
                );
            }
            other => panic!("expected GraphCompressed, got {other:?}"),
        }
    }

    #[test]
    fn handle_compress_graph_not_found() {
        let store = GraphStore::new();
        let resp = handle_request(
            &store,
            &Request::CompressGraph {
                graph_id: 999,
                method: CompressionMethod::LowRank { rank: 4 },
            },
        );
        match resp {
            Response::Error(msg) => assert!(msg.contains("not found")),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    // ---- TCP integration test ----

    #[test]
    fn tcp_client_server_roundtrip() {
        // Bind to port 0 to let the OS pick a free port.
        let server = Server::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap().to_string();

        // Spawn a thread to handle one request.
        let handle = std::thread::spawn(move || {
            server.handle_one().unwrap();
        });

        let client = Client::new(&addr);
        let id = client.submit_graph(sample_graph("tcp_test")).unwrap();
        assert_eq!(id, 0);

        handle.join().unwrap();
    }

    #[test]
    fn tcp_multiple_requests() {
        let store = GraphStore::new();
        let server = Server::bind_with_store("127.0.0.1:0", store).unwrap();
        let addr = server.local_addr().unwrap().to_string();

        // Handle 3 requests sequentially.
        let handle = std::thread::spawn(move || {
            for _ in 0..3 {
                server.handle_one().unwrap();
            }
            // Return the server so we can inspect the store
            server
        });

        let client = Client::new(&addr);

        // 1. Submit
        let id = client.submit_graph(sample_graph("multi")).unwrap();
        assert_eq!(id, 0);

        // 2. List
        let ids = client.list_graphs().unwrap();
        assert_eq!(ids, vec![0]);

        // 3. Info
        let info = client.get_graph_info(0).unwrap();
        assert_eq!(info.name, "multi");
        assert_eq!(info.num_nodes, 3);

        let server = handle.join().unwrap();
        assert_eq!(server.store().list(), vec![0]);
    }
}
