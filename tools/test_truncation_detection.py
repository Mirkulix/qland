#!/usr/bin/env python3
"""
Test that qal verify with --head detects truncated stores.
1. Verifies that qal verify --head <head> succeeds on the untampered store.
2. Copies the store to a temp directory, deletes the latest record (truncation attack).
3. Verifies that qal verify --head <head> fails with E0922 chain_broken.
"""

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_release = os.path.join(REPO_ROOT, "qal", "target", "release", "qal")
QAL_BIN = os.environ.get("QAL_BIN") or (_release if os.path.exists(_release) else os.path.join(REPO_ROOT, "bin", "qal"))
PROJECT = os.path.join(REPO_ROOT, "pilot", "escrow", "qal")
TRUST_FILE = os.path.expanduser("~/.config/qal/sessions/escrow/trust.json")

def main():
    print("1. Verifying untampered store with pinned head...")
    res = subprocess.run([QAL_BIN, "verify", "--trust", TRUST_FILE], cwd=PROJECT, capture_output=True, text=True)
    assert res.returncode == 0, f"Verify failed: {res.stdout}"
    import json
    data = json.loads(res.stdout)
    pinned_head = data["head"]
    print(f"   Current head: {pinned_head}")

    res_pinned = subprocess.run([QAL_BIN, "verify", "--trust", TRUST_FILE, "--head", pinned_head], cwd=PROJECT, capture_output=True, text=True)
    assert res_pinned.returncode == 0, f"Pinned verify failed: {res_pinned.stdout}"
    pinned_data = json.loads(res_pinned.stdout)
    assert pinned_data["ok"] is True
    assert "warnings" not in pinned_data or len(pinned_data["warnings"]) == 0
    print("   ✓ Untampered store verified with 0 warnings.")

    print("\n2. Simulating truncation attack (cutting off newest record)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_project = os.path.join(tmpdir, "proj")
        shutil.copytree(PROJECT, tmp_project)
        db_path = os.path.join(tmp_project, ".qal", "store.db")
        
        # Truncate the newest record from the database
        conn = sqlite3.connect(db_path)
        max_seq = conn.execute("SELECT MAX(seq) FROM records").fetchone()[0]
        conn.execute("DELETE FROM records WHERE seq = ?", (max_seq,))
        conn.commit()
        conn.close()
        print(f"   Deleted record {max_seq} from temporary clone.")

        # Run verify with pinned_head
        res_attack = subprocess.run([QAL_BIN, "verify", "--trust", TRUST_FILE, "--head", pinned_head], cwd=tmp_project, capture_output=True, text=True)
        assert res_attack.returncode != 0, "Security failure: Truncated store was accepted!"
        attack_data = json.loads(res_attack.stdout)
        assert attack_data["ok"] is False
        assert attack_data["error"]["code"] == "E0922"
        assert "is not in the chain" in attack_data["error"]["message"]
        print(f"   ✓ Truncation attack correctly detected and rejected: {attack_data['error']['message']}")

    print("\nAll truncation tests passed!")

if __name__ == "__main__":
    main()
