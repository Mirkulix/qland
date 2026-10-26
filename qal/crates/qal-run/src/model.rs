//! Host implementation of the `model` effect (Phase 7: Software für KI-Systeme).
//!
//! Provides `model_ask(prompt, schema) -> Result[ModelResponse, ModelError] !model`.
//! Every call is:
//! 1. Recorded in SQLite (`model_calls`), so subsequent runs and tests are 100% deterministic and cost 0.
//! 2. Budget-tracked against `QAL_MODEL_BUDGET_EUR` (default 20 €) in `model_budget`.
//! 3. Authenticated via `ANTHROPIC_API_KEY` in `~/.config/qal/secrets.env` or the environment.

use std::path::{Path, PathBuf};
use std::time::Duration;
use rusqlite::Connection;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelResponse {
    pub text: String,
    pub model: String,
    pub in_tokens: i64,
    pub out_tokens: i64,
    pub cost_microcents: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelErr {
    Denied(String),
    BudgetExceeded(i64),
    Failed(String),
    Invalid(String),
    Timeout,
}

/// Computes SHA-256 hex string of bytes.
fn sha256_hex(b: &[u8]) -> String {
    use std::fmt::Write;
    let digest = qal_core::canon::Hash::of(b).0;
    let mut s = String::with_capacity(64);
    for byte in digest {
        let _ = write!(s, "{byte:02x}");
    }
    s
}

/// Returns the default path to the SQLite cache/store for model calls.
pub fn default_db_path() -> PathBuf {
    if let Ok(p) = std::env::var("QAL_MODEL_DB") {
        return PathBuf::from(p);
    }
    let mut dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    loop {
        let candidate = dir.join(".qal").join("store.db");
        if candidate.exists() {
            return candidate;
        }
        if !dir.pop() {
            break;
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let p = PathBuf::from(home).join(".config/qal/model_cache.sqlite");
        return p;
    }
    PathBuf::from(".qal/store.db")
}

/// Reads the API key and budget limit (in microcents) from environment or `~/.config/qal/secrets.env`.
pub fn load_secrets() -> (Option<String>, i64) {
    let mut key = std::env::var("ANTHROPIC_API_KEY").ok().filter(|s| !s.trim().is_empty());
    let mut budget_eur: i64 = std::env::var("QAL_MODEL_BUDGET_EUR").ok().and_then(|v| v.parse().ok()).unwrap_or(20);

    if let Some(home) = std::env::var_os("HOME") {
        let env_path = PathBuf::from(home).join(".config/qal/secrets.env");
        if let Ok(content) = std::fs::read_to_string(&env_path) {
            for line in content.lines() {
                let line = line.trim();
                if line.starts_with('#') || line.is_empty() {
                    continue;
                }
                if let Some((k, v)) = line.split_once('=') {
                    let k = k.trim();
                    let v = v.trim().trim_matches('"').trim_matches('\'');
                    if k == "ANTHROPIC_API_KEY" && key.is_none() && !v.is_empty() {
                        key = Some(v.to_string());
                    } else if k == "QAL_MODEL_BUDGET_EUR" {
                        if let Ok(b) = v.parse::<i64>() {
                            budget_eur = b;
                        }
                    }
                }
            }
        }
    }
    (key, budget_eur.saturating_mul(100).saturating_mul(10_000))
}

/// Opens or creates the SQLite database for caching model calls and tracking budget.
pub fn open_model_db(db_path: &Path) -> Result<Connection, rusqlite::Error> {
    if let Some(parent) = db_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let conn = Connection::open(db_path)?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS model_calls (
            prompt_hash TEXT NOT NULL,
            schema_hash TEXT NOT NULL,
            model TEXT NOT NULL,
            response_text TEXT NOT NULL,
            in_tokens INTEGER NOT NULL,
            out_tokens INTEGER NOT NULL,
            cost_microcents INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (prompt_hash, schema_hash, model)
        );
        CREATE TABLE IF NOT EXISTS model_budget (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            spent_microcents INTEGER NOT NULL DEFAULT 0
        );"
    )?;
    Ok(conn)
}

/// Checks if a call is already recorded in the SQLite database.
pub fn lookup_cached(conn: &Connection, prompt_hash: &str, schema_hash: &str, model: &str) -> Option<ModelResponse> {
    conn.query_row(
        "SELECT response_text, model, in_tokens, out_tokens, cost_microcents FROM model_calls WHERE prompt_hash = ?1 AND schema_hash = ?2 AND model = ?3",
        [prompt_hash, schema_hash, model],
        |r| Ok(ModelResponse {
            text: r.get(0)?,
            model: r.get(1)?,
            in_tokens: r.get(2)?,
            out_tokens: r.get(3)?,
            cost_microcents: r.get(4)?,
        })
    ).ok()
}

/// Records a model response and updates budget tracking.
pub fn record_call(conn: &Connection, prompt_hash: &str, schema_hash: &str, resp: &ModelResponse) -> Result<(), rusqlite::Error> {
    conn.execute(
        "INSERT INTO model_budget (id, spent_microcents) VALUES (1, ?1) ON CONFLICT(id) DO UPDATE SET spent_microcents = spent_microcents + ?1",
        [resp.cost_microcents],
    )?;
    conn.execute(
        "INSERT OR REPLACE INTO model_calls (prompt_hash, schema_hash, model, response_text, in_tokens, out_tokens, cost_microcents, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, datetime('now'))",
        rusqlite::params![prompt_hash, schema_hash, resp.model, resp.text, resp.in_tokens, resp.out_tokens, resp.cost_microcents],
    )?;
    Ok(())
}

/// Returns cumulative spent microcents.
pub fn spent_microcents(conn: &Connection) -> i64 {
    conn.query_row(
        "SELECT spent_microcents FROM model_budget WHERE id = 1",
        [],
        |r| r.get(0),
    ).unwrap_or(0)
}

/// Executes a model call: first checks cache, then budget, then performs the live request.
pub fn ask(
    prompt: &str,
    schema: &str,
    model_override: Option<&str>,
    db_path: Option<&Path>,
    replay_only: bool,
) -> Result<ModelResponse, ModelErr> {
    let model = model_override.unwrap_or("claude-haiku-5-5");
    let prompt_h = sha256_hex(prompt.as_bytes());
    let schema_h = sha256_hex(schema.as_bytes());

    // 1. Check SQLite cache
    let conn = db_path.and_then(|p| open_model_db(p).ok());
    if let Some(c) = &conn {
        if let Some(cached) = lookup_cached(c, &prompt_h, &schema_h, model) {
            return Ok(cached);
        }
    }

    if replay_only {
        return Err(ModelErr::Failed("no recorded model response found for replay".into()));
    }

    // 2. Check for Mock mode (for offline unit tests)
    if let Ok(mock_val) = std::env::var("QAL_MODEL_MOCK") {
        let resp = ModelResponse {
            text: mock_val,
            model: model.to_string(),
            in_tokens: 10,
            out_tokens: 20,
            cost_microcents: 11,
        };
        if let Some(c) = &conn {
            let _ = record_call(c, &prompt_h, &schema_h, &resp);
        }
        return Ok(resp);
    }

    // 3. Check secrets & budget
    let (api_key, budget_microcents) = load_secrets();
    let api_key = match api_key {
        Some(k) => k,
        None => return Err(ModelErr::Denied("ANTHROPIC_API_KEY missing in ~/.config/qal/secrets.env or environment".into())),
    };

    if let Some(c) = &conn {
        let spent = spent_microcents(c);
        if spent >= budget_microcents {
            return Err(ModelErr::BudgetExceeded(spent / 10_000));
        }
    }

    // 4. Live API Call via ureq
    let mut payload = json!({
        "model": "claude-haiku-4-5-20251001", // current Haiku endpoint alias
        "max_tokens": 4096,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    });

    if !schema.trim().is_empty() {
        payload["system"] = json!(format!("You must reply ONLY with a valid JSON object matching this schema:\n{}", schema.trim()));
    }

    let req_str = serde_json::to_string(&payload).map_err(|e| ModelErr::Failed(format!("failed to serialize request: {e}")))?;

    let res = ureq::post("https://api.anthropic.com/v1/messages")
        .set("x-api-key", &api_key)
        .set("anthropic-version", "2023-06-01")
        .set("content-type", "application/json")
        .timeout(Duration::from_secs(30))
        .send_string(&req_str);

    let res = match res {
        Ok(r) => r,
        Err(ureq::Error::Status(status, resp)) => {
            let err_body = resp.into_string().unwrap_or_default();
            return Err(ModelErr::Failed(format!("HTTP {status}: {err_body}")));
        }
        Err(ureq::Error::Transport(t)) => {
            let msg = t.to_string();
            if msg.contains("timed out") || msg.contains("Timeout") {
                return Err(ModelErr::Timeout);
            }
            return Err(ModelErr::Failed(format!("transport error: {t}")));
        }
    };

    let resp_str = res.into_string().map_err(|e| ModelErr::Failed(format!("failed to read response: {e}")))?;
    let body: Value = serde_json::from_str(&resp_str).map_err(|e| ModelErr::Failed(format!("failed to parse JSON response: {e}")))?;

    let text = body["content"]
        .as_array()
        .and_then(|arr| arr.iter().find(|b| b["type"] == "text"))
        .and_then(|b| b["text"].as_str())
        .ok_or_else(|| ModelErr::Failed("missing content.text in model response".into()))?
        .to_string();

    let in_tokens = body["usage"]["input_tokens"].as_i64().unwrap_or(0);
    let out_tokens = body["usage"]["output_tokens"].as_i64().unwrap_or(0);

    // Claude Haiku 5.5: $0.10 / M in, $0.50 / M out
    // 1 Euro/Dollar = 100 Cents = 1,000,000 Microcents
    // in: tokens * 100,000 / 1,000,000 = in_tokens / 10 microcents
    // out: tokens * 500,000 / 1,000,000 = out_tokens / 2 microcents
    let cost_microcents = (in_tokens / 10).saturating_add(out_tokens / 2);

    // If schema was required, verify that text is valid JSON
    if !schema.trim().is_empty() {
        if serde_json::from_str::<Value>(&text).is_err() {
            return Err(ModelErr::Invalid(format!("model output does not parse as JSON: {text}")));
        }
    }

    let resp = ModelResponse {
        text,
        model: model.to_string(),
        in_tokens,
        out_tokens,
        cost_microcents,
    };

    // 5. Record call in SQLite
    if let Some(c) = &conn {
        let _ = record_call(c, &prompt_h, &schema_h, &resp);
    }

    Ok(resp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashing_is_deterministic() {
        assert_eq!(sha256_hex(b"hello"), "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824");
    }

    #[test]
    fn mock_call_records_and_replays() {
        let tmp = std::env::temp_dir().join(format!("qal-test-model-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let db_file = tmp.join("model_test.sqlite");

        std::env::set_var("QAL_MODEL_MOCK", "{\"result\": 42}");
        let r1 = ask("what is the answer?", "{}", None, Some(&db_file), false).unwrap();
        assert_eq!(r1.text, "{\"result\": 42}");
        assert_eq!(r1.in_tokens, 10);
        assert_eq!(r1.out_tokens, 20);

        // Turn off mock mode; call should hit cache!
        std::env::remove_var("QAL_MODEL_MOCK");
        let r2 = ask("what is the answer?", "{}", None, Some(&db_file), false).unwrap();
        assert_eq!(r2.text, "{\"result\": 42}");

        // Replay only should succeed for cached call
        let r3 = ask("what is the answer?", "{}", None, Some(&db_file), true).unwrap();
        assert_eq!(r3.text, "{\"result\": 42}");

        // Replay only for unrecorded call fails
        let err = ask("unrecorded question", "", None, Some(&db_file), true).unwrap_err();
        assert!(matches!(err, ModelErr::Failed(_)));

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
