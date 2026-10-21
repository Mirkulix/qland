# Pilot Specification: Internal Financial Escrow & Multi-Sig Payout Engine (`escrow_payout`)

## 1. Context & Motivation

This pilot benchmarks **software assurance in a high-consequence domain**: internal treasury, escrow management, and multi-signature payout authorizations.

In standard web stacks (Python/FastAPI or Rust/Axum without formal methods), critical financial invariants (no negative balances, zero-sum preservation, separation of duties, atomic escrow locks) are typically enforced via runtime checks. When agents modify or refactor code, subtle omissions (e.g. forgetting to check self-approval, off-by-one comparisons, or non-atomic rollbacks on crash) easily escape to production.

In **QAL**, mathematical contracts (`req`/`ens`) verified by Z3, effect isolation (`!db` only where persistence occurs), immutable content addressing, and team cryptographic provenance guarantee that invalid states cannot be committed to `main`.

---

## 2. Domain Model & Invariants

### Formal Invariants
1. **Zero-Sum Preservation (Erhaltungssatz):**
   ```
   total_assets = sum(accounts.balance) + escrow_pool + total_disbursed == total_deposited
   ```
   Money can neither be created out of thin air nor lost through integer wrapping or partial failure.

2. **Non-Negativity:**
   ```
   balance_cents >= 0 (for all accounts at all times)
   escrow_pool >= 0
   ```

3. **Strictly Positive Operations:**
   ```
   amount_cents > 0
   ```

4. **Separation of Duties (Multi-Sig Governance):**
   - Requester cannot approve their own payout: `approver != requester`.
   - Idempotent approval: An approver can only sign once (`approver ∉ approvals`).
   - Tiered Approval Threshold:
     - Tier 1 (`amount_cents <= 1,000,000` = 10,000.00 €): Requires **1** authorized approval.
     - Tier 2 (`amount_cents > 1,000,000` > 10,000.00 €): Requires **2 distinct** authorized approvals.

5. **Atomic Escrow & Rollback:**
   - On payout creation: `amount_cents` is subtracted from `source_account` and added to `escrow_pool`.
   - On cancellation/rejection: `amount_cents` is subtracted from `escrow_pool` and refunded to `source_account`.
   - On disbursement: `amount_cents` is deducted from `escrow_pool` and recorded in `total_disbursed`.

---

## 3. HTTP API Endpoints

| Method | Endpoint | Description | Status Codes |
|---|---|---|---|
| `POST` | `/accounts` | Create account with initial deposit | `201 Created`, `400 Bad Request` |
| `GET` | `/accounts/:id` | Read account balance and owner | `200 OK`, `404 Not Found` |
| `POST` | `/payouts` | Create payout request (locks funds in escrow) | `201 Created`, `400 Bad Request`, `422 Insufficient Funds` |
| `GET` | `/payouts/:id` | Read payout request and approval chain | `200 OK`, `404 Not Found` |
| `POST` | `/payouts/:id/approve` | Add approver signature; evaluate threshold | `200 OK`, `400 Bad Request`, `403 Self-Approval`, `409 Already Approved` |
| `POST` | `/payouts/:id/disburse` | Execute disbursement of approved payout | `200 OK`, `400 Not Approved`, `404 Not Found` |
| `POST` | `/payouts/:id/cancel` | Cancel payout and refund source account | `200 OK`, `400 Already Disbursed`, `404 Not Found` |
| `GET` | `/reconciliation` | Full treasury balance check (proves Invariant 1) | `200 OK` |
| `GET` | `/audit` | Chronological audit log of all financial events | `200 OK` |

---

## 4. Evaluation Criteria & Benchmark Metrics

| Metric | Target / Measurement Unit | Rationale |
|---|---|---|
| **Escaped Defects / 1k LOC** | Count of logic flaws in production/test suite | The core assurance value proposition |
| **Time to First Deploy** | Seconds from spec to green HTTP test suite | Operational friction / agility |
| **Audit & Review Time** | Minutes for a human auditor to verify invariants | Governance cost |
| **Crash & Rollback Recovery** | Verified zero-sum preservation after `SIGKILL` | Crash resilience |
| **Total Cost** | LLM Tokens + Human intervention time | Total economic viability |
