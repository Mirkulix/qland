"""
Unified Automated Test Suite & Benchmark for Financial Escrow & Payout Service.
Tests 20 distinct scenarios (standard workflows, boundary conditions, and adversarial attacks)
against any HTTP server implementing the pilot specification.
"""

import os
import sys
import time
import requests
import unittest
from concurrent.futures import ThreadPoolExecutor

# Optional authentication injection for protected production gateways
_API_KEY = os.getenv("ESCROW_API_KEY", "")
if _API_KEY:
    _orig_get = requests.get
    _orig_post = requests.post
    def _patched_get(url, *args, **kwargs):
        headers = dict(kwargs.get("headers") or {})
        headers.setdefault("X-API-Key", _API_KEY)
        kwargs["headers"] = headers
        return _orig_get(url, *args, **kwargs)
    def _patched_post(url, *args, **kwargs):
        headers = dict(kwargs.get("headers") or {})
        headers.setdefault("X-API-Key", _API_KEY)
        kwargs["headers"] = headers
        return _orig_post(url, *args, **kwargs)
    requests.get = _patched_get
    requests.post = _patched_post

class EscrowServiceTestSuite(unittest.TestCase):
    BASE_URL = "http://127.0.0.1:8088"

    @classmethod
    def setUpClass(cls):
        # Allow overriding URL via env or command line
        if len(sys.argv) > 1 and sys.argv[1].startswith("http"):
            cls.BASE_URL = sys.argv[1]
        print(f"\n--- Testing Escrow Service against {cls.BASE_URL} ---")

    def url(self, path: str) -> str:
        return f"{self.BASE_URL.rstrip('/')}/{path.lstrip('/')}"

    # Scenario 1: Valid Account Creation
    def test_01_account_creation_valid(self):
        res = requests.post(self.url("/accounts"), json={"id": 1, "owner": "Alice", "initial_balance_cents": 5000000})
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertTrue(data.get("ok"))
        self.assertEqual(data["account"]["balance_cents"], 5000000)

    # Scenario 2: Negative Account Creation Rejected
    def test_02_account_creation_negative_rejected(self):
        res = requests.post(self.url("/accounts"), json={"id": 2, "owner": "Mallory", "initial_balance_cents": -100})
        self.assertIn(res.status_code, [400, 422])

    # Scenario 3: Payout Creation Locks Funds in Escrow
    def test_03_payout_creation_and_escrow_lock(self):
        requests.post(self.url("/accounts"), json={"id": 3, "owner": "Bob", "initial_balance_cents": 2000000})
        res = requests.post(self.url("/payouts"), json={
            "source_account": 3,
            "target_iban": "DE89370400440532013000",
            "amount_cents": 500000,
            "requester": "Bob"
        })
        self.assertEqual(res.status_code, 201)
        payout = res.json()["payout"]
        self.assertEqual(payout["status"], "pending")
        
        # Verify account balance dropped by 500,000 cents
        acc_res = requests.get(self.url("/accounts/3"))
        self.assertEqual(acc_res.status_code, 200)
        self.assertEqual(acc_res.json()["account"]["balance_cents"], 1500000)

    # Scenario 4: Insufficient Funds Rejected
    def test_04_insufficient_funds_rejected(self):
        requests.post(self.url("/accounts"), json={"id": 4, "owner": "Charlie", "initial_balance_cents": 100000})
        res = requests.post(self.url("/payouts"), json={
            "source_account": 4,
            "target_iban": "DE89370400440532013000",
            "amount_cents": 200000,
            "requester": "Charlie"
        })
        self.assertIn(res.status_code, [400, 422])

    # Scenario 5: Zero or Negative Payout Rejected
    def test_05_zero_or_negative_payout_rejected(self):
        requests.post(self.url("/accounts"), json={"id": 5, "owner": "Dave", "initial_balance_cents": 1000000})
        res_zero = requests.post(self.url("/payouts"), json={"source_account": 5, "target_iban": "DE89", "amount_cents": 0, "requester": "Dave"})
        self.assertIn(res_zero.status_code, [400, 422])
        res_neg = requests.post(self.url("/payouts"), json={"source_account": 5, "target_iban": "DE89", "amount_cents": -500, "requester": "Dave"})
        self.assertIn(res_neg.status_code, [400, 422])

    # Scenario 6: Tier 1 (<= 10k EUR) Single Approval & Disbursement
    def test_06_tier1_single_approval_disbursement(self):
        requests.post(self.url("/accounts"), json={"id": 6, "owner": "Eva", "initial_balance_cents": 1000000})
        p = requests.post(self.url("/payouts"), json={"source_account": 6, "target_iban": "DE89", "amount_cents": 500000, "requester": "Eva"}).json()["payout"]
        
        # Approve by Frank
        app_res = requests.post(self.url(f"/payouts/{p['id']}/approve"), json={"approver": "Frank"})
        self.assertEqual(app_res.status_code, 200)
        self.assertEqual(app_res.json()["payout"]["status"], "approved")

        # Disburse
        disb_res = requests.post(self.url(f"/payouts/{p['id']}/disburse"), json={})
        self.assertEqual(disb_res.status_code, 200)
        self.assertEqual(disb_res.json()["payout"]["status"], "disbursed")

    # Scenario 7: Tier 2 (> 10k EUR) Requires Two Distinct Approvals
    def test_07_tier2_requires_two_approvals(self):
        requests.post(self.url("/accounts"), json={"id": 7, "owner": "Grace", "initial_balance_cents": 5000000})
        p = requests.post(self.url("/payouts"), json={"source_account": 7, "target_iban": "DE89", "amount_cents": 2000000, "requester": "Grace"}).json()["payout"]
        
        # 1st approval by Heidi: must remain pending
        r1 = requests.post(self.url(f"/payouts/{p['id']}/approve"), json={"approver": "Heidi"})
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r1.json()["payout"]["status"], "pending")

        # 2nd approval by Ivan: now approved
        r2 = requests.post(self.url(f"/payouts/{p['id']}/approve"), json={"approver": "Ivan"})
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json()["payout"]["status"], "approved")

    # Scenario 8: Self-Approval Strictly Forbidden
    def test_08_self_approval_strictly_forbidden(self):
        requests.post(self.url("/accounts"), json={"id": 8, "owner": "Judy", "initial_balance_cents": 1000000})
        p = requests.post(self.url("/payouts"), json={"source_account": 8, "target_iban": "DE89", "amount_cents": 100000, "requester": "Judy"}).json()["payout"]
        
        # Judy attempts to approve her own payout
        res = requests.post(self.url(f"/payouts/{p['id']}/approve"), json={"approver": "Judy"})
        self.assertIn(res.status_code, [400, 403])

    # Scenario 9: Duplicate Approval Rejected
    def test_09_duplicate_approval_rejected(self):
        requests.post(self.url("/accounts"), json={"id": 9, "owner": "Kevin", "initial_balance_cents": 5000000})
        p = requests.post(self.url("/payouts"), json={"source_account": 9, "target_iban": "DE89", "amount_cents": 2000000, "requester": "Kevin"}).json()["payout"]
        
        # Leo approves once
        r1 = requests.post(self.url(f"/payouts/{p['id']}/approve"), json={"approver": "Leo"})
        self.assertEqual(r1.status_code, 200)
        
        # Leo tries to approve a second time
        r2 = requests.post(self.url(f"/payouts/{p['id']}/approve"), json={"approver": "Leo"})
        self.assertIn(r2.status_code, [400, 409])

    # Scenario 10: Cancel Pending Payout Refunds Source Account
    def test_10_cancel_pending_payout_refunds_source(self):
        requests.post(self.url("/accounts"), json={"id": 10, "owner": "Mia", "initial_balance_cents": 1000000})
        p = requests.post(self.url("/payouts"), json={"source_account": 10, "target_iban": "DE89", "amount_cents": 400000, "requester": "Mia"}).json()["payout"]
        
        # Cancel
        c_res = requests.post(self.url(f"/payouts/{p['id']}/cancel"), json={"reason": "Wrong amount"})
        self.assertEqual(c_res.status_code, 200)
        self.assertEqual(c_res.json()["payout"]["status"], "cancelled")

        # Verify full refund
        acc = requests.get(self.url("/accounts/10")).json()["account"]
        self.assertEqual(acc["balance_cents"], 1000000)

    # Scenario 11: Cancel Approved Payout Refunds Source Account
    def test_11_cancel_approved_payout_refunds_source(self):
        requests.post(self.url("/accounts"), json={"id": 11, "owner": "Noah", "initial_balance_cents": 1000000})
        p = requests.post(self.url("/payouts"), json={"source_account": 11, "target_iban": "DE89", "amount_cents": 400000, "requester": "Noah"}).json()["payout"]
        requests.post(self.url(f"/payouts/{p['id']}/approve"), json={"approver": "Olivia"})
        
        # Cancel approved payout
        c_res = requests.post(self.url(f"/payouts/{p['id']}/cancel"), json={"reason": "Cancelled by compliance"})
        self.assertEqual(c_res.status_code, 200)
        
        # Verify refund
        acc = requests.get(self.url("/accounts/11")).json()["account"]
        self.assertEqual(acc["balance_cents"], 1000000)

    # Scenario 12: Disburse Unapproved Payout Rejected
    def test_12_disburse_unapproved_rejected(self):
        requests.post(self.url("/accounts"), json={"id": 12, "owner": "Paul", "initial_balance_cents": 1000000})
        p = requests.post(self.url("/payouts"), json={"source_account": 12, "target_iban": "DE89", "amount_cents": 100000, "requester": "Paul"}).json()["payout"]
        
        # Try to disburse without approval
        res = requests.post(self.url(f"/payouts/{p['id']}/disburse"), json={})
        self.assertIn(res.status_code, [400, 422])

    # Scenario 13: Double Disbursement Rejected
    def test_13_double_disbursement_rejected(self):
        requests.post(self.url("/accounts"), json={"id": 13, "owner": "Quinn", "initial_balance_cents": 1000000})
        p = requests.post(self.url("/payouts"), json={"source_account": 13, "target_iban": "DE89", "amount_cents": 100000, "requester": "Quinn"}).json()["payout"]
        requests.post(self.url(f"/payouts/{p['id']}/approve"), json={"approver": "Rachel"})
        
        r1 = requests.post(self.url(f"/payouts/{p['id']}/disburse"), json={})
        self.assertEqual(r1.status_code, 200)
        
        # Second disbursement attempt
        r2 = requests.post(self.url(f"/payouts/{p['id']}/disburse"), json={})
        self.assertIn(r2.status_code, [400, 409])

    # Scenario 14: Cancel Already Disbursed Rejected
    def test_14_cancel_disbursed_rejected(self):
        requests.post(self.url("/accounts"), json={"id": 14, "owner": "Sam", "initial_balance_cents": 1000000})
        p = requests.post(self.url("/payouts"), json={"source_account": 14, "target_iban": "DE89", "amount_cents": 100000, "requester": "Sam"}).json()["payout"]
        requests.post(self.url(f"/payouts/{p['id']}/approve"), json={"approver": "Tina"})
        requests.post(self.url(f"/payouts/{p['id']}/disburse"), json={})
        
        res = requests.post(self.url(f"/payouts/{p['id']}/cancel"), json={"reason": "Should fail"})
        self.assertIn(res.status_code, [400, 409])

    # Scenario 15: Treasury Reconciliation (Zero-Sum Preservation)
    def test_15_reconciliation_zero_sum_preserved(self):
        res = requests.get(self.url("/reconciliation"))
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("balanced"), f"Reconciliation balance mismatch: {data}")

    # Scenario 16: Audit Log Integrity
    def test_16_audit_log_chronology(self):
        res = requests.get(self.url("/audit"))
        self.assertEqual(res.status_code, 200)
        entries = res.json()["entries"]
        self.assertGreater(len(entries), 10)
        # Check action types
        actions = {e["action"] for e in entries}
        self.assertTrue({"account_created", "payout_created", "payout_approved"}.issubset(actions))

    # Scenario 17: Non-existent Resource Returns 404
    def test_17_non_existent_account_or_payout_404(self):
        r1 = requests.get(self.url("/accounts/99999"))
        self.assertEqual(r1.status_code, 404)
        r2 = requests.get(self.url("/payouts/99999"))
        self.assertEqual(r2.status_code, 404)

    # Scenario 18: Parallel Competing Payouts (No Overdraft Race Condition)
    def test_18_parallel_competing_payouts_no_overdraft(self):
        requests.post(self.url("/accounts"), json={"id": 18, "owner": "Victor", "initial_balance_cents": 1000000})
        
        def create_payout():
            return requests.post(self.url("/payouts"), json={
                "source_account": 18,
                "target_iban": "DE89",
                "amount_cents": 800000,
                "requester": "Victor"
            }).status_code

        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(create_payout)
            f2 = ex.submit(create_payout)
            codes = [f1.result(), f2.result()]

        # Exactly one must succeed (201) and the other must be rejected (400 or 422)
        self.assertEqual(codes.count(201), 1, f"Concurrency race condition allowed duplicate deductions: {codes}")
        acc = requests.get(self.url("/accounts/18")).json()["account"]
        self.assertGreaterEqual(acc["balance_cents"], 0, "Account was overdrafted!")

    # Scenario 19: Boundary Value (Tier 1 vs Tier 2 Threshold)
    def test_19_boundary_value_exactly_tier_threshold(self):
        requests.post(self.url("/accounts"), json={"id": 19, "owner": "Wendy", "initial_balance_cents": 5000000})
        
        # Exactly 1,000,000 cents (10k EUR): Tier 1 -> 1 approval suffices
        p1 = requests.post(self.url("/payouts"), json={"source_account": 19, "target_iban": "DE89", "amount_cents": 1000000, "requester": "Wendy"}).json()["payout"]
        r1 = requests.post(self.url(f"/payouts/{p1['id']}/approve"), json={"approver": "Xavier"})
        self.assertEqual(r1.json()["payout"]["status"], "approved")

        # 1,000,001 cents (10k EUR + 1 cent): Tier 2 -> 1 approval must stay pending
        p2 = requests.post(self.url("/payouts"), json={"source_account": 19, "target_iban": "DE89", "amount_cents": 1000001, "requester": "Wendy"}).json()["payout"]
        r2 = requests.post(self.url(f"/payouts/{p2['id']}/approve"), json={"approver": "Xavier"})
        self.assertEqual(r2.json()["payout"]["status"], "pending")

    # Scenario 20: Complex Multi-Account Interleaved Flow
    def test_20_multi_account_interleaved_transfers(self):
        requests.post(self.url("/accounts"), json={"id": 201, "owner": "Yara", "initial_balance_cents": 3000000})
        requests.post(self.url("/accounts"), json={"id": 202, "owner": "Zane", "initial_balance_cents": 4000000})
        
        p1 = requests.post(self.url("/payouts"), json={"source_account": 201, "target_iban": "DE89", "amount_cents": 1500000, "requester": "Yara"}).json()["payout"]
        p2 = requests.post(self.url("/payouts"), json={"source_account": 202, "target_iban": "DE89", "amount_cents": 2500000, "requester": "Zane"}).json()["payout"]
        
        requests.post(self.url(f"/payouts/{p1['id']}/cancel"), json={"reason": "Cancel Yara"})
        requests.post(self.url(f"/payouts/{p2['id']}/approve"), json={"approver": "Auditor1"})
        requests.post(self.url(f"/payouts/{p2['id']}/approve"), json={"approver": "Auditor2"})
        requests.post(self.url(f"/payouts/{p2['id']}/disburse"), json={})

        rec = requests.get(self.url("/reconciliation")).json()
        self.assertTrue(rec.get("balanced"), f"Final multi-account reconciliation failed: {rec}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("http"):
        EscrowServiceTestSuite.BASE_URL = sys.argv.pop(1)
    unittest.main()
