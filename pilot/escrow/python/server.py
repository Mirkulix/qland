import json
import sqlite3
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI(title="Internal Escrow & Payout Service (Python Reference)")

DB_PATH = "escrow_python.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    with conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                id INTEGER PRIMARY KEY,
                owner TEXT NOT NULL,
                balance_cents INTEGER NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS payouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_account INTEGER NOT NULL,
                target_iban TEXT NOT NULL,
                amount_cents INTEGER NOT NULL,
                requester TEXT NOT NULL,
                status TEXT NOT NULL,
                approvals TEXT NOT NULL,
                FOREIGN KEY(source_account) REFERENCES accounts(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                action TEXT NOT NULL,
                actor TEXT NOT NULL,
                payload TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS treasury (
                key TEXT PRIMARY KEY,
                val INTEGER NOT NULL
            )
        """)
        conn.execute("INSERT OR IGNORE INTO treasury (key, val) VALUES ('escrow_pool', 0)")
        conn.execute("INSERT OR IGNORE INTO treasury (key, val) VALUES ('total_disbursed', 0)")
        conn.execute("INSERT OR IGNORE INTO treasury (key, val) VALUES ('total_deposited', 0)")

init_db()

class CreateAccountReq(BaseModel):
    id: int
    owner: str
    initial_balance_cents: int

class CreatePayoutReq(BaseModel):
    source_account: int
    target_iban: str
    amount_cents: int
    requester: str

class ApprovePayoutReq(BaseModel):
    approver: str

class CancelPayoutReq(BaseModel):
    reason: Optional[str] = "No reason provided"

@app.post("/accounts", status_code=status.HTTP_201_CREATED)
def create_account(req: CreateAccountReq):
    if req.initial_balance_cents < 0:
        raise HTTPException(status_code=400, detail="Initial balance cannot be negative")
    conn = get_db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("INSERT INTO accounts (id, owner, balance_cents) VALUES (?, ?, ?)",
                     (req.id, req.owner, req.initial_balance_cents))
        conn.execute("UPDATE treasury SET val = val + ? WHERE key = 'total_deposited'", (req.initial_balance_cents,))
        conn.execute("INSERT INTO audit (ts, action, actor, payload) VALUES (?, 'account_created', ?, ?)",
                     (int(time.time()), req.owner, json.dumps({"account_id": req.id, "balance": req.initial_balance_cents})))
        conn.execute("COMMIT")
        return {"ok": True, "account": {"id": req.id, "owner": req.owner, "balance_cents": req.initial_balance_cents}}
    except sqlite3.IntegrityError:
        conn.execute("ROLLBACK")
        raise HTTPException(status_code=400, detail="Account already exists")
    except Exception as e:
        conn.execute("ROLLBACK")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/accounts/{account_id}")
def get_account(account_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Account not found")
    return {"account": {"id": row["id"], "owner": row["owner"], "balance_cents": row["balance_cents"]}}

@app.post("/payouts", status_code=status.HTTP_201_CREATED)
def create_payout(req: CreatePayoutReq):
    if req.amount_cents <= 0:
        raise HTTPException(status_code=400, detail="Payout amount must be positive")
    conn = get_db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        acc = conn.execute("SELECT balance_cents FROM accounts WHERE id = ?", (req.source_account,)).fetchone()
        if not acc:
            conn.execute("ROLLBACK")
            raise HTTPException(status_code=404, detail="Source account not found")
        if acc["balance_cents"] < req.amount_cents:
            conn.execute("ROLLBACK")
            raise HTTPException(status_code=422, detail="Insufficient funds")
        
        # Deduct from account, move to escrow
        conn.execute("UPDATE accounts SET balance_cents = balance_cents - ? WHERE id = ?",
                     (req.amount_cents, req.source_account))
        conn.execute("UPDATE treasury SET val = val + ? WHERE key = 'escrow_pool'", (req.amount_cents,))
        
        cur = conn.execute("""
            INSERT INTO payouts (source_account, target_iban, amount_cents, requester, status, approvals)
            VALUES (?, ?, ?, ?, 'pending', '[]')
        """, (req.source_account, req.target_iban, req.amount_cents, req.requester))
        payout_id = cur.lastrowid
        
        conn.execute("INSERT INTO audit (ts, action, actor, payload) VALUES (?, 'payout_created', ?, ?)",
                     (int(time.time()), req.requester, json.dumps({"payout_id": payout_id, "amount": req.amount_cents})))
        conn.execute("COMMIT")
        return {"ok": True, "payout": {
            "id": payout_id,
            "source_account": req.source_account,
            "target_iban": req.target_iban,
            "amount_cents": req.amount_cents,
            "requester": req.requester,
            "status": "pending",
            "approvals": []
        }}
    except HTTPException:
        raise
    except Exception as e:
        conn.execute("ROLLBACK")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/payouts/{payout_id}")
def get_payout(payout_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM payouts WHERE id = ?", (payout_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Payout not found")
    return {"payout": {
        "id": row["id"],
        "source_account": row["source_account"],
        "target_iban": row["target_iban"],
        "amount_cents": row["amount_cents"],
        "requester": row["requester"],
        "status": row["status"],
        "approvals": json.loads(row["approvals"])
    }}

@app.post("/payouts/{payout_id}/approve")
def approve_payout(payout_id: int, req: ApprovePayoutReq):
    conn = get_db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT * FROM payouts WHERE id = ?", (payout_id,)).fetchone()
        if not row:
            conn.execute("ROLLBACK")
            raise HTTPException(status_code=404, detail="Payout not found")
        if row["status"] != "pending":
            conn.execute("ROLLBACK")
            raise HTTPException(status_code=400, detail="Payout is not pending")
        
        # Invariant 4: No self approval
        if req.approver == row["requester"]:
            conn.execute("ROLLBACK")
            raise HTTPException(status_code=403, detail="Self-approval forbidden")
        
        approvals = json.loads(row["approvals"])
        if req.approver in approvals:
            conn.execute("ROLLBACK")
            raise HTTPException(status_code=409, detail="Already approved by this user")
        
        approvals.append(req.approver)
        
        # Multi-Sig Threshold: <= 1M cents (10k EUR) requires 1, > 1M requires 2
        amount = row["amount_cents"]
        new_status = "pending"
        if amount <= 1000000 and len(approvals) >= 1:
            new_status = "approved"
        elif amount > 1000000 and len(approvals) >= 2:
            new_status = "approved"
        
        conn.execute("UPDATE payouts SET status = ?, approvals = ? WHERE id = ?",
                     (new_status, json.dumps(approvals), payout_id))
        conn.execute("INSERT INTO audit (ts, action, actor, payload) VALUES (?, 'payout_approved', ?, ?)",
                     (int(time.time()), req.approver, json.dumps({"payout_id": payout_id, "new_status": new_status})))
        conn.execute("COMMIT")
        return {"ok": True, "payout": {
            "id": row["id"],
            "source_account": row["source_account"],
            "target_iban": row["target_iban"],
            "amount_cents": row["amount_cents"],
            "requester": row["requester"],
            "status": new_status,
            "approvals": approvals
        }}
    except HTTPException:
        raise
    except Exception as e:
        conn.execute("ROLLBACK")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/payouts/{payout_id}/disburse")
def disburse_payout(payout_id: int):
    conn = get_db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT * FROM payouts WHERE id = ?", (payout_id,)).fetchone()
        if not row:
            conn.execute("ROLLBACK")
            raise HTTPException(status_code=404, detail="Payout not found")
        if row["status"] == "disbursed":
            conn.execute("ROLLBACK")
            raise HTTPException(status_code=409, detail="Already disbursed")
        if row["status"] != "approved":
            conn.execute("ROLLBACK")
            raise HTTPException(status_code=400, detail=f"Payout not approved (current status: {row['status']})")
        
        amount = row["amount_cents"]
        conn.execute("UPDATE payouts SET status = 'disbursed' WHERE id = ?", (payout_id,))
        conn.execute("UPDATE treasury SET val = val - ? WHERE key = 'escrow_pool'", (amount,))
        conn.execute("UPDATE treasury SET val = val + ? WHERE key = 'total_disbursed'", (amount,))
        conn.execute("INSERT INTO audit (ts, action, actor, payload) VALUES (?, 'payout_disbursed', 'system', ?)",
                     (int(time.time()), json.dumps({"payout_id": payout_id, "amount": amount})))
        conn.execute("COMMIT")
        return {"ok": True, "payout": {
            "id": row["id"],
            "source_account": row["source_account"],
            "amount_cents": amount,
            "status": "disbursed",
            "approvals": json.loads(row["approvals"])
        }}
    except HTTPException:
        raise
    except Exception as e:
        conn.execute("ROLLBACK")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/payouts/{payout_id}/cancel")
def cancel_payout(payout_id: int, req: CancelPayoutReq):
    conn = get_db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT * FROM payouts WHERE id = ?", (payout_id,)).fetchone()
        if not row:
            conn.execute("ROLLBACK")
            raise HTTPException(status_code=404, detail="Payout not found")
        if row["status"] == "disbursed":
            conn.execute("ROLLBACK")
            raise HTTPException(status_code=409, detail="Cannot cancel disbursed payout")
        if row["status"] == "cancelled":
            conn.execute("ROLLBACK")
            raise HTTPException(status_code=400, detail="Already cancelled")
        
        amount = row["amount_cents"]
        # Refund source account, remove from escrow
        conn.execute("UPDATE accounts SET balance_cents = balance_cents + ? WHERE id = ?",
                     (amount, row["source_account"]))
        conn.execute("UPDATE treasury SET val = val - ? WHERE key = 'escrow_pool'", (amount,))
        conn.execute("UPDATE payouts SET status = 'cancelled' WHERE id = ?", (payout_id,))
        conn.execute("INSERT INTO audit (ts, action, actor, payload) VALUES (?, 'payout_cancelled', 'risk_officer', ?)",
                     (int(time.time()), json.dumps({"payout_id": payout_id, "reason": req.reason})))
        conn.execute("COMMIT")
        return {"ok": True, "payout": {
            "id": row["id"],
            "source_account": row["source_account"],
            "amount_cents": amount,
            "status": "cancelled",
            "approvals": json.loads(row["approvals"])
        }}
    except HTTPException:
        raise
    except Exception as e:
        conn.execute("ROLLBACK")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reconciliation")
def get_reconciliation():
    conn = get_db()
    accounts_sum = conn.execute("SELECT COALESCE(SUM(balance_cents), 0) AS total FROM accounts").fetchone()["total"]
    treasury_rows = {row["key"]: row["val"] for row in conn.execute("SELECT key, val FROM treasury").fetchall()}
    escrow = treasury_rows.get("escrow_pool", 0)
    disbursed = treasury_rows.get("total_disbursed", 0)
    deposited = treasury_rows.get("total_deposited", 0)
    
    current_total = accounts_sum + escrow + disbursed
    balanced = (current_total == deposited) and (accounts_sum >= 0) and (escrow >= 0)
    return {
        "balanced": balanced,
        "accounts_sum": accounts_sum,
        "escrow_pool": escrow,
        "total_disbursed": disbursed,
        "total_deposited": deposited,
        "current_total": current_total
    }

@app.get("/audit")
def get_audit():
    conn = get_db()
    rows = conn.execute("SELECT id, ts, action, actor, payload FROM audit ORDER BY id ASC").fetchall()
    return {"entries": [dict(r) for r in rows]}
