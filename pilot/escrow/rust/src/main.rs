use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use rusqlite::{params, Connection};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone)]
struct AppState {
    db: Arc<Mutex<Connection>>,
}

fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64
}

fn init_db(conn: &Connection) {
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY,
            owner TEXT NOT NULL,
            balance_cents INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS payouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_account INTEGER NOT NULL,
            target_iban TEXT NOT NULL,
            amount_cents INTEGER NOT NULL,
            requester TEXT NOT NULL,
            status TEXT NOT NULL,
            approvals TEXT NOT NULL,
            FOREIGN KEY(source_account) REFERENCES accounts(id)
        );
        CREATE TABLE IF NOT EXISTS audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            action TEXT NOT NULL,
            actor TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS treasury (
            key TEXT PRIMARY KEY,
            val INTEGER NOT NULL
        );
        INSERT OR IGNORE INTO treasury (key, val) VALUES ('escrow_pool', 0);
        INSERT OR IGNORE INTO treasury (key, val) VALUES ('total_disbursed', 0);
        INSERT OR IGNORE INTO treasury (key, val) VALUES ('total_deposited', 0);",
    )
    .unwrap();
}

#[derive(Deserialize)]
struct CreateAccountReq {
    id: i64,
    owner: String,
    initial_balance_cents: i64,
}

#[derive(Deserialize)]
struct CreatePayoutReq {
    source_account: i64,
    target_iban: String,
    amount_cents: i64,
    requester: String,
}

#[derive(Deserialize)]
struct ApprovePayoutReq {
    approver: String,
}

#[derive(Deserialize)]
struct CancelPayoutReq {
    reason: Option<String>,
}

async fn create_account(
    State(state): State<AppState>,
    Json(req): Json<CreateAccountReq>,
) -> Result<(StatusCode, Json<Value>), (StatusCode, Json<Value>)> {
    if req.initial_balance_cents < 0 {
        return Err((StatusCode::BAD_REQUEST, Json(json!({"error": "Initial balance cannot be negative"}))));
    }
    let conn = state.db.lock().unwrap();
    conn.execute_batch("BEGIN IMMEDIATE").map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))))?;
    
    let res = (|| -> Result<(), rusqlite::Error> {
        conn.execute("INSERT INTO accounts (id, owner, balance_cents) VALUES (?1, ?2, ?3)", params![req.id, req.owner, req.initial_balance_cents])?;
        conn.execute("UPDATE treasury SET val = val + ?1 WHERE key = 'total_deposited'", params![req.initial_balance_cents])?;
        let payload = json!({"account_id": req.id, "balance": req.initial_balance_cents}).to_string();
        conn.execute("INSERT INTO audit (ts, action, actor, payload) VALUES (?1, 'account_created', ?2, ?3)", params![now(), req.owner, payload])?;
        Ok(())
    })();

    match res {
        Ok(_) => {
            conn.execute_batch("COMMIT").unwrap();
            Ok((StatusCode::CREATED, Json(json!({
                "ok": true,
                "account": { "id": req.id, "owner": req.owner, "balance_cents": req.initial_balance_cents }
            }))))
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err((StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))))
        }
    }
}

async fn get_account(
    State(state): State<AppState>,
    Path(account_id): Path<i64>,
) -> Result<Json<Value>, StatusCode> {
    let conn = state.db.lock().unwrap();
    let mut stmt = conn.prepare("SELECT id, owner, balance_cents FROM accounts WHERE id = ?1").map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let mut rows = stmt.query(params![account_id]).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    if let Some(row) = rows.next().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)? {
        let id: i64 = row.get(0).unwrap();
        let owner: String = row.get(1).unwrap();
        let balance_cents: i64 = row.get(2).unwrap();
        Ok(Json(json!({ "account": { "id": id, "owner": owner, "balance_cents": balance_cents } })))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn create_payout(
    State(state): State<AppState>,
    Json(req): Json<CreatePayoutReq>,
) -> Result<(StatusCode, Json<Value>), (StatusCode, Json<Value>)> {
    if req.amount_cents <= 0 {
        return Err((StatusCode::BAD_REQUEST, Json(json!({"error": "Payout amount must be positive"}))));
    }
    let conn = state.db.lock().unwrap();
    conn.execute_batch("BEGIN IMMEDIATE").map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))))?;

    let balance: Option<i64> = conn
        .query_row("SELECT balance_cents FROM accounts WHERE id = ?1", params![req.source_account], |r| r.get(0))
        .ok();

    let Some(cur_balance) = balance else {
        let _ = conn.execute_batch("ROLLBACK");
        return Err((StatusCode::NOT_FOUND, Json(json!({"error": "Source account not found"}))));
    };

    if cur_balance < req.amount_cents {
        let _ = conn.execute_batch("ROLLBACK");
        return Err((StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"error": "Insufficient funds"}))));
    }

    let res = (|| -> Result<i64, rusqlite::Error> {
        conn.execute("UPDATE accounts SET balance_cents = balance_cents - ?1 WHERE id = ?2", params![req.amount_cents, req.source_account])?;
        conn.execute("UPDATE treasury SET val = val + ?1 WHERE key = 'escrow_pool'", params![req.amount_cents])?;
        conn.execute(
            "INSERT INTO payouts (source_account, target_iban, amount_cents, requester, status, approvals) VALUES (?1, ?2, ?3, ?4, 'pending', '[]')",
            params![req.source_account, req.target_iban, req.amount_cents, req.requester],
        )?;
        let id = conn.last_insert_rowid();
        let payload = json!({"payout_id": id, "amount": req.amount_cents}).to_string();
        conn.execute("INSERT INTO audit (ts, action, actor, payload) VALUES (?1, 'payout_created', ?2, ?3)", params![now(), req.requester, payload])?;
        Ok(id)
    })();

    match res {
        Ok(payout_id) => {
            conn.execute_batch("COMMIT").unwrap();
            Ok((StatusCode::CREATED, Json(json!({
                "ok": true,
                "payout": {
                    "id": payout_id,
                    "source_account": req.source_account,
                    "target_iban": req.target_iban,
                    "amount_cents": req.amount_cents,
                    "requester": req.requester,
                    "status": "pending",
                    "approvals": []
                }
            }))))
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))))
        }
    }
}

async fn get_payout(
    State(state): State<AppState>,
    Path(payout_id): Path<i64>,
) -> Result<Json<Value>, StatusCode> {
    let conn = state.db.lock().unwrap();
    let row = conn.query_row(
        "SELECT id, source_account, target_iban, amount_cents, requester, status, approvals FROM payouts WHERE id = ?1",
        params![payout_id],
        |r| {
            let approvals: String = r.get(6)?;
            let apps: Vec<String> = serde_json::from_str(&approvals).unwrap_or_default();
            Ok(json!({
                "payout": {
                    "id": r.get::<_, i64>(0)?,
                    "source_account": r.get::<_, i64>(1)?,
                    "target_iban": r.get::<_, String>(2)?,
                    "amount_cents": r.get::<_, i64>(3)?,
                    "requester": r.get::<_, String>(4)?,
                    "status": r.get::<_, String>(5)?,
                    "approvals": apps
                }
            }))
        },
    );
    match row {
        Ok(val) => Ok(Json(val)),
        Err(_) => Err(StatusCode::NOT_FOUND),
    }
}

async fn approve_payout(
    State(state): State<AppState>,
    Path(payout_id): Path<i64>,
    Json(req): Json<ApprovePayoutReq>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let conn = state.db.lock().unwrap();
    conn.execute_batch("BEGIN IMMEDIATE").map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))))?;

    let row = conn.query_row(
        "SELECT id, source_account, target_iban, amount_cents, requester, status, approvals FROM payouts WHERE id = ?1",
        params![payout_id],
        |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, String>(5)?,
                r.get::<_, String>(6)?,
            ))
        },
    );

    let (id, source_acc, iban, amount, requester, status, approvals_str) = match row {
        Ok(t) => t,
        Err(_) => {
            let _ = conn.execute_batch("ROLLBACK");
            return Err((StatusCode::NOT_FOUND, Json(json!({"error": "Payout not found"}))));
        }
    };

    if status != "pending" {
        let _ = conn.execute_batch("ROLLBACK");
        return Err((StatusCode::BAD_REQUEST, Json(json!({"error": "Payout is not pending"}))));
    }

    if req.approver == requester {
        let _ = conn.execute_batch("ROLLBACK");
        return Err((StatusCode::FORBIDDEN, Json(json!({"error": "Self-approval forbidden"}))));
    }

    let mut approvals: Vec<String> = serde_json::from_str(&approvals_str).unwrap_or_default();
    if approvals.contains(&req.approver) {
        let _ = conn.execute_batch("ROLLBACK");
        return Err((StatusCode::CONFLICT, Json(json!({"error": "Already approved by this user"}))));
    }

    approvals.push(req.approver.clone());
    let mut new_status = "pending";
    if amount <= 1_000_000 && approvals.len() >= 1 {
        new_status = "approved";
    } else if amount > 1_000_000 && approvals.len() >= 2 {
        new_status = "approved";
    }

    let new_apps_json = serde_json::to_string(&approvals).unwrap();
    conn.execute("UPDATE payouts SET status = ?1, approvals = ?2 WHERE id = ?3", params![new_status, new_apps_json, id]).unwrap();
    let payload = json!({"payout_id": id, "new_status": new_status}).to_string();
    conn.execute("INSERT INTO audit (ts, action, actor, payload) VALUES (?1, 'payout_approved', ?2, ?3)", params![now(), req.approver, payload]).unwrap();
    conn.execute_batch("COMMIT").unwrap();

    Ok(Json(json!({
        "ok": true,
        "payout": {
            "id": id,
            "source_account": source_acc,
            "target_iban": iban,
            "amount_cents": amount,
            "requester": requester,
            "status": new_status,
            "approvals": approvals
        }
    })))
}

async fn disburse_payout(
    State(state): State<AppState>,
    Path(payout_id): Path<i64>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let conn = state.db.lock().unwrap();
    conn.execute_batch("BEGIN IMMEDIATE").map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))))?;

    let row = conn.query_row(
        "SELECT id, source_account, amount_cents, status, approvals FROM payouts WHERE id = ?1",
        params![payout_id],
        |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?, r.get::<_, String>(3)?, r.get::<_, String>(4)?)),
    );

    let (id, source_acc, amount, status, apps_str) = match row {
        Ok(t) => t,
        Err(_) => {
            let _ = conn.execute_batch("ROLLBACK");
            return Err((StatusCode::NOT_FOUND, Json(json!({"error": "Payout not found"}))));
        }
    };

    if status == "disbursed" {
        let _ = conn.execute_batch("ROLLBACK");
        return Err((StatusCode::CONFLICT, Json(json!({"error": "Already disbursed"}))));
    }

    if status != "approved" {
        let _ = conn.execute_batch("ROLLBACK");
        return Err((StatusCode::BAD_REQUEST, Json(json!({"error": "Payout not approved"}))));
    }

    conn.execute("UPDATE payouts SET status = 'disbursed' WHERE id = ?1", params![id]).unwrap();
    conn.execute("UPDATE treasury SET val = val - ?1 WHERE key = 'escrow_pool'", params![amount]).unwrap();
    conn.execute("UPDATE treasury SET val = val + ?1 WHERE key = 'total_disbursed'", params![amount]).unwrap();
    let payload = json!({"payout_id": id, "amount": amount}).to_string();
    conn.execute("INSERT INTO audit (ts, action, actor, payload) VALUES (?1, 'payout_disbursed', 'system', ?2)", params![now(), payload]).unwrap();
    conn.execute_batch("COMMIT").unwrap();

    let apps: Vec<String> = serde_json::from_str(&apps_str).unwrap_or_default();
    Ok(Json(json!({
        "ok": true,
        "payout": {
            "id": id,
            "source_account": source_acc,
            "amount_cents": amount,
            "status": "disbursed",
            "approvals": apps
        }
    })))
}

async fn cancel_payout(
    State(state): State<AppState>,
    Path(payout_id): Path<i64>,
    Json(req): Json<CancelPayoutReq>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let conn = state.db.lock().unwrap();
    conn.execute_batch("BEGIN IMMEDIATE").map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))))?;

    let row = conn.query_row(
        "SELECT id, source_account, amount_cents, status, approvals FROM payouts WHERE id = ?1",
        params![payout_id],
        |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?, r.get::<_, String>(3)?, r.get::<_, String>(4)?)),
    );

    let (id, source_acc, amount, status, apps_str) = match row {
        Ok(t) => t,
        Err(_) => {
            let _ = conn.execute_batch("ROLLBACK");
            return Err((StatusCode::NOT_FOUND, Json(json!({"error": "Payout not found"}))));
        }
    };

    if status == "disbursed" {
        let _ = conn.execute_batch("ROLLBACK");
        return Err((StatusCode::CONFLICT, Json(json!({"error": "Cannot cancel disbursed payout"}))));
    }

    if status == "cancelled" {
        let _ = conn.execute_batch("ROLLBACK");
        return Err((StatusCode::BAD_REQUEST, Json(json!({"error": "Already cancelled"}))));
    }

    conn.execute("UPDATE accounts SET balance_cents = balance_cents + ?1 WHERE id = ?2", params![amount, source_acc]).unwrap();
    conn.execute("UPDATE treasury SET val = val - ?1 WHERE key = 'escrow_pool'", params![amount]).unwrap();
    conn.execute("UPDATE payouts SET status = 'cancelled' WHERE id = ?1", params![id]).unwrap();
    let reason = req.reason.unwrap_or_else(|| "No reason".into());
    let payload = json!({"payout_id": id, "reason": reason}).to_string();
    conn.execute("INSERT INTO audit (ts, action, actor, payload) VALUES (?1, 'payout_cancelled', 'risk_officer', ?2)", params![now(), payload]).unwrap();
    conn.execute_batch("COMMIT").unwrap();

    let apps: Vec<String> = serde_json::from_str(&apps_str).unwrap_or_default();
    Ok(Json(json!({
        "ok": true,
        "payout": {
            "id": id,
            "source_account": source_acc,
            "amount_cents": amount,
            "status": "cancelled",
            "approvals": apps
        }
    })))
}

async fn get_reconciliation(State(state): State<AppState>) -> Json<Value> {
    let conn = state.db.lock().unwrap();
    let accounts_sum: i64 = conn
        .query_row("SELECT COALESCE(SUM(balance_cents), 0) FROM accounts", [], |r| r.get(0))
        .unwrap_or(0);
    let mut stmt = conn.prepare("SELECT key, val FROM treasury").unwrap();
    let mut rows = stmt.query([]).unwrap();
    let mut escrow = 0i64;
    let mut disbursed = 0i64;
    let mut deposited = 0i64;
    while let Some(r) = rows.next().unwrap() {
        let k: String = r.get(0).unwrap();
        let v: i64 = r.get(1).unwrap();
        match k.as_str() {
            "escrow_pool" => escrow = v,
            "total_disbursed" => disbursed = v,
            "total_deposited" => deposited = v,
            _ => {}
        }
    }
    let current_total = accounts_sum + escrow + disbursed;
    let balanced = (current_total == deposited) && (accounts_sum >= 0) && (escrow >= 0);
    Json(json!({
        "balanced": balanced,
        "accounts_sum": accounts_sum,
        "escrow_pool": escrow,
        "total_disbursed": disbursed,
        "total_deposited": deposited,
        "current_total": current_total
    }))
}

async fn get_audit(State(state): State<AppState>) -> Json<Value> {
    let conn = state.db.lock().unwrap();
    let mut stmt = conn.prepare("SELECT id, ts, action, actor, payload FROM audit ORDER BY id ASC").unwrap();
    let rows = stmt
        .query_map([], |r| {
            Ok(json!({
                "id": r.get::<_, i64>(0)?,
                "ts": r.get::<_, i64>(1)?,
                "action": r.get::<_, String>(2)?,
                "actor": r.get::<_, String>(3)?,
                "payload": r.get::<_, String>(4)?
            }))
        })
        .unwrap()
        .collect::<Result<Vec<Value>, _>>()
        .unwrap();
    Json(json!({ "entries": rows }))
}

#[tokio::main]
async fn main() {
    let port = std::env::var("PORT").unwrap_or_else(|_| "8090".into());
    let db_path = std::env::var("DB_PATH").unwrap_or_else(|_| "escrow_rust.db".into());
    let conn = Connection::open(db_path).unwrap();
    init_db(&conn);

    let state = AppState {
        db: Arc::new(Mutex::new(conn)),
    };

    let app = Router::new()
        .route("/accounts", post(create_account))
        .route("/accounts/:id", get(get_account))
        .route("/payouts", post(create_payout))
        .route("/payouts/:id", get(get_payout))
        .route("/payouts/:id/approve", post(approve_payout))
        .route("/payouts/:id/disburse", post(disburse_payout))
        .route("/payouts/:id/cancel", post(cancel_payout))
        .route("/reconciliation", get(get_reconciliation))
        .route("/audit", get(get_audit))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await.unwrap();
    println!("Rust Escrow Service running on http://127.0.0.1:{port}");
    axum::serve(listener, app).await.unwrap();
}
