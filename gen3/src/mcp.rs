//! Native MCP (Model Context Protocol) Server for Gen-3 Substrate.
//!
//! Provides stdio JSON-RPC interface for Claude Code, DSH, and autonomous AI agents.

use std::io::{self, BufRead, Write};
use serde_json::{json, Value as JsonValue};
use crate::dag::Graph;
use crate::engine::{Engine, Value};
use crate::optimizer::Optimizer;
use crate::parser::Parser;
use crate::projection::Projection;
use crate::verifier::{Interval, Verifier};

pub struct McpServer;

impl McpServer {
    pub fn run_stdio() -> io::Result<()> {
        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut handle = stdout.lock();

        for line in stdin.lock().lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let req: JsonValue = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    let err_resp = json!({
                        "jsonrpc": "2.0",
                        "error": { "code": -32700, "message": format!("Parse error: {e}") },
                        "id": null
                    });
                    writeln!(handle, "{}", err_resp)?;
                    handle.flush()?;
                    continue;
                }
            };

            let id = req.get("id").cloned().unwrap_or(JsonValue::Null);
            let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");

            let resp = match method {
                "initialize" => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": { "tools": {} },
                        "serverInfo": {
                            "name": "tgs-engine",
                            "version": "0.1.0"
                        }
                    }
                }),
                "tools/list" => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "tools": [
                            {
                                "name": "tgs_verify",
                                "description": "Formally verify topological DAG and prove mathematical bounds ahead of execution (<1 µs).",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "graph": { "type": "string", "description": "Graph JSON string or SSA source code" },
                                        "input_bounds": { "type": "array", "description": "List of [min, max] pairs for input parameters" }
                                    },
                                    "required": ["graph"]
                                }
                            },
                            {
                                "name": "tgs_run",
                                "description": "Execute a verified DAG on the zero-alloc Dataflow Virtual Machine in nanoseconds.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "graph": { "type": "string", "description": "Graph JSON string or SSA source code" },
                                        "args": { "type": "array", "description": "Integer or float arguments" }
                                    },
                                    "required": ["graph", "args"]
                                }
                            },
                            {
                                "name": "tgs_optimize",
                                "description": "Optimize a DAG with algebraic identities, constant folding, CSE, and dead code pruning.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "graph": { "type": "string", "description": "Graph JSON string or SSA source code" }
                                    },
                                    "required": ["graph"]
                                }
                            },
                            {
                                "name": "tgs_project",
                                "description": "Decompile dense binary DAG into human-readable SSA pseudo-code and ASCII graph.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "graph": { "type": "string", "description": "Graph JSON string" }
                                    },
                                    "required": ["graph"]
                                }
                            }
                        ]
                    }
                }),
                "tools/call" => {
                    let params = req.get("params");
                    let tool_name = params.and_then(|p| p.get("name")).and_then(|n| n.as_str()).unwrap_or("");
                    let arguments = params.and_then(|p| p.get("arguments")).cloned().unwrap_or(json!({}));

                    let result_text = Self::handle_tool_call(tool_name, &arguments);
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": result_text
                                }
                            ]
                        }
                    })
                }
                _ => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": { "code": -32601, "message": format!("Method not found: {method}") }
                }),
            };

            writeln!(handle, "{}", resp)?;
            handle.flush()?;
        }

        Ok(())
    }

    fn resolve_graph(input: &str) -> Result<Graph, String> {
        // If it starts with '{', assume JSON
        if input.trim().starts_with('{') {
            serde_json::from_str::<Graph>(input).map_err(|e| format!("Invalid Graph JSON: {e}"))
        } else {
            // Assume SSA pseudo-code
            Parser::parse(input).map_err(|e| format!("Invalid SSA source: {e:?}"))
        }
    }

    fn handle_tool_call(name: &str, args: &JsonValue) -> String {
        match name {
            "tgs_verify" => {
                let g_str = args.get("graph").and_then(|g| g.as_str()).unwrap_or("");
                let graph = match Self::resolve_graph(g_str) {
                    Ok(g) => g,
                    Err(e) => return format!("Error: {e}"),
                };

                let bounds: Vec<Interval> = if let Some(arr) = args.get("input_bounds").and_then(|b| b.as_array()) {
                    arr.iter().filter_map(|pair| {
                        let p = pair.as_array()?;
                        if p.len() == 2 {
                            Some(Interval { min: p[0].as_i64()?, max: p[1].as_i64()? })
                        } else {
                            None
                        }
                    }).collect()
                } else {
                    vec![Interval::unbounded(); graph.input_types.len()]
                };

                match Verifier::verify(&graph, &bounds) {
                    Ok(b) => {
                        let mut out = format!("✅ VERIFIED: All {} nodes structurally sound and invariants satisfied!\n", graph.nodes.len());
                        for (i, inv) in b.iter().enumerate() {
                            out.push_str(&format!("  v{:02} -> Interval [{}, {}]\n", i, inv.min, inv.max));
                        }
                        out
                    }
                    Err(e) => format!("❌ VERIFICATION FAILED: {e:?}"),
                }
            }
            "tgs_run" => {
                let g_str = args.get("graph").and_then(|g| g.as_str()).unwrap_or("");
                let graph = match Self::resolve_graph(g_str) {
                    Ok(g) => g,
                    Err(e) => return format!("Error: {e}"),
                };

                let raw_args = args.get("args").and_then(|a| a.as_array());
                let vals: Vec<Value> = raw_args
                    .map(|arr| {
                        arr.iter()
                            .map(|v| {
                                if let Some(i) = v.as_i64() {
                                    Value::I64(i)
                                } else if let Some(f) = v.as_f64() {
                                    Value::F64(f)
                                } else if let Some(b) = v.as_bool() {
                                    Value::Bool(b)
                                } else {
                                    Value::Unit
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                match Engine::execute(&graph, &vals) {
                    Ok(res) => format!("Result: {res:?}"),
                    Err(e) => format!("Execution Error: {e}"),
                }
            }
            "tgs_optimize" => {
                let g_str = args.get("graph").and_then(|g| g.as_str()).unwrap_or("");
                let graph = match Self::resolve_graph(g_str) {
                    Ok(g) => g,
                    Err(e) => return format!("Error: {e}"),
                };

                let (opt_graph, stats) = Optimizer::optimize(&graph);
                format!(
                    "✅ OPTIMIZATION COMPLETE:\n  Initial Nodes: {}\n  Final Nodes: {}\n  Folded Constants: {}\n  Algebraic Reductions: {}\n  CSE Eliminations: {}\n  Dead Nodes Removed: {}\n\nHuman Projection:\n{}",
                    stats.initial_nodes, stats.final_nodes, stats.constants_folded, stats.algebraic_reductions, stats.cse_eliminations, stats.dead_nodes_removed,
                    Projection::decompile(&opt_graph)
                )
            }
            "tgs_project" => {
                let g_str = args.get("graph").and_then(|g| g.as_str()).unwrap_or("");
                let graph = match Self::resolve_graph(g_str) {
                    Ok(g) => g,
                    Err(e) => return format!("Error: {e}"),
                };
                format!("{}\n\n{}", Projection::visual_graph(&graph), Projection::decompile(&graph))
            }
            _ => format!("Unknown tool: {name}"),
        }
    }
}
