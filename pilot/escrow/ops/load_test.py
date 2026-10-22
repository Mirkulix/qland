#!/usr/bin/env python3
"""
Concurrency, Stress & Race-Condition Load Test for QAL Escrow Engine.
Validates:
  - Parallel Competing Payouts (50 concurrent threads against 1 account)
  - Zero-overdraft guarantee (balance never goes negative)
  - Exactly N payouts succeed and remaining M are rejected with 422
  - Concurrent duplicate approvals & disbursals (state machine race protection)
  - Final Zero-Sum Reconciliation under high parallel load
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
import time
import requests


def run_load_test(base_url: str, api_key: str = "", concurrency: int = 50):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    def url(path: str) -> str:
        return f"{base_url.rstrip('/')}/{path.lstrip('/')}"

    print(f"\n{'='*70}")
    print(f"=== QAL Escrow High-Concurrency Load Test ({concurrency} Parallel Workers) ===")
    print(f"Target: {base_url}")
    print(f"{'='*70}\n")

    # 1. Create Target Account
    account_id = 990
    initial_balance = 10000000 # 100,000 EUR
    payout_amount = 3000000   # 30,000 EUR
    expected_success = initial_balance // payout_amount # Exactly 3 payouts can succeed!

    print(f"[1/4] Creating Test Account #{account_id} with {initial_balance/100:.2f} EUR...")
    r = requests.post(url("/accounts"), json={
        "id": account_id,
        "owner": "LoadTestCorp",
        "initial_balance_cents": initial_balance
    }, headers=headers)
    assert r.status_code == 201, f"Failed to create test account: {r.status_code} {r.text}"
    print(f"      [✓] Account created with balance: {initial_balance} cents.")

    # 2. Fire 50 Parallel Competing Payouts
    print(f"\n[2/4] Firing {concurrency} parallel competing payouts of {payout_amount/100:.2f} EUR each...")
    t0 = time.time()

    def attempt_payout(idx: int):
        try:
            resp = requests.post(url("/payouts"), json={
                "source_account": account_id,
                "target_iban": f"DE893704004405320130{idx:02d}",
                "amount_cents": payout_amount,
                "requester": f"Worker_{idx}"
            }, headers=headers, timeout=5.0)
            return resp.status_code, resp.json() if resp.status_code in [201, 400, 422] else resp.text
        except Exception as e:
            return 500, str(e)

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(attempt_payout, i) for i in range(concurrency)]
        for f in as_completed(futures):
            results.append(f.result())

    duration = time.time() - t0
    status_codes = [status for status, _ in results]
    from collections import Counter
    code_counts = Counter(status_codes)
    print(f"      Status Codes Breakdown: {dict(code_counts)}")
    success_count = code_counts.get(201, 0)
    # Both 422 (insufficient balance) and 429 (rate limited under burst) are valid rejections under load
    rejected_count = code_counts.get(422, 0) + code_counts.get(400, 0) + code_counts.get(429, 0)
    error_count = len(status_codes) - (success_count + rejected_count)

    # Assertions
    assert success_count == expected_success, f"RACE CONDITION DETECTED! Succeeded: {success_count}, Expected: {expected_success}"
    assert error_count == 0, f"Unexpected errors during concurrent payouts: {error_count}"

    # Allow token bucket to refill slightly after burst
    time.sleep(0.5)

    # Verify Account Balance
    acc_res = requests.get(url(f"/accounts/{account_id}"), headers=headers)
    assert acc_res.status_code == 200, f"Failed to get account {account_id}: {acc_res.status_code} {acc_res.text}"
    current_balance = acc_res.json()["account"]["balance_cents"]
    expected_final_balance = initial_balance - (expected_success * payout_amount) # 1,000,000 cents
    print(f"      Account #{account_id} balance after competition: {current_balance} cents (Expected: {expected_final_balance}).")
    assert current_balance == expected_final_balance, f"Balance mismatch! Expected {expected_final_balance}, got {current_balance}"
    assert current_balance >= 0, "CRITICAL: Account was overdrafted!"
    print(f"      [✓] Zero-overdraft invariant held perfectly under {concurrency} parallel requests.")

    # 3. Concurrent Duplicate Approvals and Disbursals
    print(f"\n[3/4] Testing concurrent duplicate approvals on successful payouts...")
    successful_payout_ids = [res[1]["payout"]["id"] for res in results if res[0] == 201]

    def concurrent_approve(payout_id: int):
        r1 = requests.post(url(f"/payouts/{payout_id}/approve"), json={"approver": "AuditorX"}, headers=headers)
        return r1.status_code

    for pid in successful_payout_ids:
        with ThreadPoolExecutor(max_workers=5) as ex:
            f_approvals = [ex.submit(concurrent_approve, pid) for _ in range(5)]
            appr_codes = [f.result() for f in f_approvals]
        # Exactly one must succeed (200), others must be rejected (409 Conflict already approved)
        assert appr_codes.count(200) == 1, f"Approval idempotency failure: {appr_codes}"
        assert appr_codes.count(409) == 4, f"Duplicate approvals not rejected: {appr_codes}"

    print(f"      [✓] Multi-Sig state machine idempotency verified under concurrent calls.")

    # 4. Final Reconciliation
    print(f"\n[4/4] Validating global treasury reconciliation post-load...")
    rec_res = requests.get(url("/reconciliation"), headers=headers)
    assert rec_res.status_code == 200, f"Reconciliation failed: {rec_res.status_code}"
    rec_data = rec_res.json()
    print(f"      Reconciliation Response: {json.dumps(rec_data)}")
    assert rec_data.get("balanced") is True, f"TREASURY UNBALANCED POST-LOAD: {rec_data}"
    print(f"      [✓] Treasury is 100% balanced! All mathematical conservation laws hold.")

    print(f"\n{'='*70}")
    print(f"=== LOAD TEST RESULT: PASSED (All {concurrency} concurrent requests handled cleanly) ===")
    print(f"{'='*70}\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Concurrency Load Test")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="Base URL of service/gateway")
    parser.add_argument("--key", default="escrow-prod-secret-key-2026", help="API Key")
    parser.add_argument("--concurrency", type=int, default=50, help="Concurrency worker threads")
    args = parser.parse_args()

    run_load_test(args.url, args.key, args.concurrency)
