//! Language Server Protocol (LSP 3.17) implementation for QAL.
//! Enables IDE integration (VS Code, Cursor, Neovim) over JSON-RPC stdio.

use crate::load;
use qal_core::ast::Def;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{self, BufRead, Read, Write};

#[derive(Default)]
pub struct LspServer {
    documents: HashMap<String, String>,
}

impl LspServer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn run(&mut self) -> io::Result<()> {
        let stdin = io::stdin();
        let mut reader = io::BufReader::new(stdin.lock());
        let mut stdout = io::stdout().lock();

        loop {
            // Read headers
            let mut content_length: Option<usize> = None;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line)? == 0 {
                    return Ok(()); // EOF
                }
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    break; // Header boundary reached
                }
                if let Some(rest) = trimmed.strip_prefix("Content-Length:") {
                    if let Ok(len) = rest.trim().parse::<usize>() {
                        content_length = Some(len);
                    }
                }
            }

            let Some(len) = content_length else {
                continue;
            };

            let mut body = vec![0u8; len];
            reader.read_exact(&mut body)?;

            let msg: Value = match serde_json::from_slice(&body) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let id = msg.get("id").cloned();
            let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
            let params = msg.get("params").cloned().unwrap_or(Value::Null);

            match method {
                "initialize" => {
                    let result = json!({
                        "capabilities": {
                            "textDocumentSync": 1, // Full sync
                            "hoverProvider": true,
                            "documentSymbolProvider": true
                        },
                        "serverInfo": {
                            "name": "qal-lsp",
                            "version": "0.1.0"
                        }
                    });
                    if let Some(id) = id {
                        Self::send_response(&mut stdout, id, result)?;
                    }
                }
                "initialized" => {
                    // Client notification, no reply
                }
                "textDocument/didOpen" => {
                    if let Some(td) = params.get("textDocument") {
                        let uri = td.get("uri").and_then(|u| u.as_str()).unwrap_or("").to_string();
                        let text = td.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string();
                        self.documents.insert(uri.clone(), text.clone());
                        self.publish_diagnostics(&mut stdout, &uri, &text)?;
                    }
                }
                "textDocument/didChange" => {
                    if let (Some(td), Some(changes)) = (params.get("textDocument"), params.get("contentChanges")) {
                        let uri = td.get("uri").and_then(|u| u.as_str()).unwrap_or("").to_string();
                        if let Some(change) = changes.as_array().and_then(|arr| arr.last()) {
                            if let Some(new_text) = change.get("text").and_then(|t| t.as_str()) {
                                self.documents.insert(uri.clone(), new_text.to_string());
                                self.publish_diagnostics(&mut stdout, &uri, new_text)?;
                            }
                        }
                    }
                }
                "textDocument/hover" => {
                    let mut hover_text = None;
                    if let (Some(td), Some(pos)) = (params.get("textDocument"), params.get("position")) {
                        let uri = td.get("uri").and_then(|u| u.as_str()).unwrap_or("");
                        let line = pos.get("line").and_then(|l| l.as_u64()).unwrap_or(0) as usize;
                        let character = pos.get("character").and_then(|c| c.as_u64()).unwrap_or(0) as usize;

                        if let Some(doc) = self.documents.get(uri) {
                            hover_text = Self::compute_hover(doc, line, character);
                        }
                    }

                    if let Some(id) = id {
                        let res = match hover_text {
                            Some(markdown) => json!({
                                "contents": {
                                    "kind": "markdown",
                                    "value": markdown
                                }
                            }),
                            None => Value::Null,
                        };
                        Self::send_response(&mut stdout, id, res)?;
                    }
                }
                "textDocument/documentSymbol" => {
                    let mut symbols = Vec::new();
                    if let Some(td) = params.get("textDocument") {
                        let uri = td.get("uri").and_then(|u| u.as_str()).unwrap_or("");
                        if let Some(doc) = self.documents.get(uri) {
                            symbols = Self::compute_symbols(doc);
                        }
                    }
                    if let Some(id) = id {
                        Self::send_response(&mut stdout, id, json!(symbols))?;
                    }
                }
                "shutdown" => {
                    if let Some(id) = id {
                        Self::send_response(&mut stdout, id, Value::Null)?;
                    }
                }
                "exit" => {
                    return Ok(());
                }
                _ => {
                    if let Some(id) = id {
                        Self::send_response(&mut stdout, id, Value::Null)?;
                    }
                }
            }
        }
    }

    fn publish_diagnostics<W: Write>(&self, out: &mut W, uri: &str, text: &str) -> io::Result<()> {
        let mut lsp_diags = Vec::new();

        if let Err(diags) = load(text) {
            for d in diags {
                let (line, col, len) = Self::find_span_in_text(text, &d.def);
                let message = if d.hints.is_empty() {
                    d.message
                } else {
                    format!("{}\nHints: {}", d.message, d.hints.join(", "))
                };

                lsp_diags.push(json!({
                    "range": {
                        "start": { "line": line, "character": col },
                        "end": { "line": line, "character": col + len }
                    },
                    "severity": 1, // Error
                    "code": d.code,
                    "source": "qal",
                    "message": message
                }));
            }
        }

        let notification = json!({
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": uri,
                "diagnostics": lsp_diags
            }
        });

        Self::send_notification(out, notification)
    }

    fn find_span_in_text(text: &str, def_name: &str) -> (usize, usize, usize) {
        if !def_name.is_empty() {
            for (line_idx, line) in text.lines().enumerate() {
                if let Some(col_idx) = line.find(def_name) {
                    return (line_idx, col_idx, def_name.len());
                }
            }
        }
        (0, 0, 1)
    }

    fn compute_hover(doc: &str, target_line: usize, target_col: usize) -> Option<String> {
        let lines: Vec<&str> = doc.lines().collect();
        let line = lines.get(target_line)?;
        let ident = Self::ident_at(line, target_col)?;

        if let Ok(loaded) = load(doc) {
            for def in &loaded.program.defs {
                let name = def.name();
                if name == ident.as_str() {
                    match def {
                        Def::Fn(f) => {
                            let mut s = format!("```qal\nfn {}(", f.name);
                            for (i, p) in f.params.iter().enumerate() {
                                if i > 0 { s.push_str(", "); }
                                s.push_str(&format!("{}: {:?}", p.name, p.ty));
                            }
                            s.push_str(&format!(") -> {:?}\n```", f.ret));
                            if !f.req.is_empty() {
                                s.push_str("\n**Requires:**\n");
                                for req in &f.req {
                                    s.push_str(&format!("- `req {:?}`\n", req));
                                }
                            }
                            if !f.ens.is_empty() {
                                s.push_str("\n**Ensures:**\n");
                                for ens in &f.ens {
                                    s.push_str(&format!("- `ens {:?}`\n", ens));
                                }
                            }
                            return Some(s);
                        }
                        Def::Type(t) => {
                            return Some(format!("```qal\ntype {} = ...\n```\nDefined QAL type.", t.name));
                        }
                        Def::Test(t) => {
                            return Some(format!("```qal\ntest {}\n```\nDefined test.", t.name));
                        }
                    }
                }
            }
        }

        Some(format!("`{ident}` (QAL symbol)"))
    }

    fn compute_symbols(doc: &str) -> Vec<Value> {
        let mut symbols = Vec::new();
        if let Ok(loaded) = load(doc) {
            for (line_idx, line) in doc.lines().enumerate() {
                for def in &loaded.program.defs {
                    let name = def.name();
                    let kind = match def {
                        Def::Fn(_) => 12,   // Function
                        Def::Test(_) => 12, // Test
                        Def::Type(_) => 23, // Struct / Type
                    };
                    if line.contains(name) {
                        symbols.push(json!({
                            "name": name,
                            "kind": kind,
                            "range": {
                                "start": { "line": line_idx, "character": 0 },
                                "end": { "line": line_idx, "character": line.len() }
                            },
                            "selectionRange": {
                                "start": { "line": line_idx, "character": 0 },
                                "end": { "line": line_idx, "character": line.len() }
                            }
                        }));
                    }
                }
            }
        }
        symbols
    }

    fn ident_at(line: &str, col: usize) -> Option<String> {
        if col >= line.len() {
            return None;
        }
        let bytes = line.as_bytes();
        let mut start = col;
        while start > 0 && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_' || bytes[start - 1] == b'.') {
            start -= 1;
        }
        let mut end = col;
        while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_' || bytes[end] == b'.') {
            end += 1;
        }
        if start == end {
            None
        } else {
            Some(line[start..end].to_string())
        }
    }

    fn send_response<W: Write>(out: &mut W, id: Value, result: Value) -> io::Result<()> {
        let resp = json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result
        });
        Self::send(out, &resp)
    }

    fn send_notification<W: Write>(out: &mut W, notif: Value) -> io::Result<()> {
        Self::send(out, &notif)
    }

    fn send<W: Write>(out: &mut W, val: &Value) -> io::Result<()> {
        let body = serde_json::to_string(val).unwrap_or_default();
        write!(out, "Content-Length: {}\r\n\r\n{}", body.len(), body)?;
        out.flush()
    }
}
