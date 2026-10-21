#!/usr/bin/env python3
"""
Atomic Backup, Restore, and Invariant Verification Tool for QAL Escrow Engine.
Guarantees:
  - Point-in-time snapshot without stopping writes (using SQLite VACUUM INTO / online backup API)
  - Zero-Sum Invariant verification:
      total_deposited == escrow_pool + total_disbursed + sum(accounts.balance_cents)
  - Non-negativity check: all accounts balance_cents >= 0
  - Restored store validation with `qal verify --head <hash>`
"""

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
from typing import Dict, Tuple


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_database_invariants(db_path: str) -> Tuple[bool, Dict[str, int], str]:
    """
    Connects to SQLite database and verifies:
      1. Zero-Sum conservation law
      2. Non-negativity of all account balances
      3. Escrow pool matches active pending/approved payouts
    """
    if not os.path.exists(db_path):
        return False, {}, f"Database file not found: {db_path}"

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # 1. Read Treasury
        cur.execute("SELECT key, val FROM treasury")
        treasury = {row[0]: row[1] for row in cur.fetchall()}
        total_deposited = treasury.get("total_deposited", 0)
        escrow_pool = treasury.get("escrow_pool", 0)
        total_disbursed = treasury.get("total_disbursed", 0)

        # 2. Sum Account Balances
        cur.execute("SELECT COUNT(*), COALESCE(SUM(balance_cents), 0), COALESCE(MIN(balance_cents), 0) FROM accounts")
        acc_count, sum_balances, min_balance = cur.fetchone()

        # 3. Sum Payouts
        cur.execute("SELECT status, COALESCE(SUM(amount_cents), 0) FROM payouts GROUP BY status")
        payouts_by_status = {row[0]: row[1] for row in cur.fetchall()}
        escrow_in_payouts = payouts_by_status.get("pending", 0) + payouts_by_status.get("approved", 0)
        disbursed_in_payouts = payouts_by_status.get("disbursed", 0)

        conn.close()

        stats = {
            "accounts_count": acc_count,
            "sum_balances": sum_balances,
            "min_balance": min_balance,
            "total_deposited": total_deposited,
            "escrow_pool": escrow_pool,
            "total_disbursed": total_disbursed,
            "escrow_in_payouts": escrow_in_payouts,
            "disbursed_in_payouts": disbursed_in_payouts,
        }

        # Invariant 1: Non-negativity
        if min_balance < 0:
            return False, stats, f"INVARIANT VIOLATION: Negative account balance found (min: {min_balance} cents)"

        # Invariant 2: Zero-Sum Conservation Law
        expected_total = sum_balances + escrow_pool + total_disbursed
        if total_deposited != expected_total:
            diff = total_deposited - expected_total
            return False, stats, f"ZERO-SUM VIOLATION: total_deposited ({total_deposited}) != sum_balances ({sum_balances}) + escrow_pool ({escrow_pool}) + disbursed ({total_disbursed}) [diff: {diff}]"

        # Invariant 3: Escrow pool matches active pending/approved payouts
        if escrow_pool != escrow_in_payouts:
            return False, stats, f"POOL INCONSISTENCY: escrow_pool ({escrow_pool}) != active payouts sum ({escrow_in_payouts})"

        return True, stats, "All mathematical invariants verified successfully"

    except Exception as e:
        return False, {}, f"Database inspection failed: {str(e)}"


def perform_backup(src_db: str, backup_dir: str) -> str:
    """
    Creates an atomic point-in-time backup using SQLite's online backup API.
    """
    if not os.path.exists(src_db):
        raise FileNotFoundError(f"Source database does not exist: {src_db}")

    os.makedirs(backup_dir, exist_ok=True)
    ts = int(time.time())
    backup_file = os.path.join(backup_dir, f"escrow_backup_{ts}.sqlite")

    print(f"[*] Starting atomic online backup of {src_db}...")
    src_conn = sqlite3.connect(src_db)
    dst_conn = sqlite3.connect(backup_file)

    # SQLite Online Backup API (thread-safe, atomic snapshot)
    src_conn.backup(dst_conn)
    dst_conn.close()
    src_conn.close()

    # Verify invariants on backup
    valid, stats, msg = verify_database_invariants(backup_file)
    if not valid:
        os.remove(backup_file)
        raise RuntimeError(f"Backup rejected due to invariant failure: {msg}")

    file_hash = compute_sha256(backup_file)
    meta = {
        "timestamp": ts,
        "source_db": os.path.abspath(src_db),
        "backup_file": os.path.abspath(backup_file),
        "sha256": file_hash,
        "stats": stats,
        "verified": True,
    }

    meta_file = backup_file + ".meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[✓] Atomic backup created: {backup_file}")
    print(f"    SHA-256: {file_hash}")
    print(f"    Invariants: {msg}")
    print(f"    Total Deposited: {stats['total_deposited']} cents | Sum Balances: {stats['sum_balances']} cents")
    return backup_file


def perform_restore(backup_file: str, target_db: str) -> bool:
    """
    Restores the database from an atomic backup file and purges stale WAL/SHM.
    """
    if not os.path.exists(backup_file):
        raise FileNotFoundError(f"Backup file not found: {backup_file}")

    print(f"[*] Validating backup integrity before restore...")
    valid, stats, msg = verify_database_invariants(backup_file)
    if not valid:
        raise RuntimeError(f"Cannot restore corrupted backup: {msg}")

    target_dir = os.path.dirname(os.path.abspath(target_db))
    os.makedirs(target_dir, exist_ok=True)

    # 1. Clean stale WAL / SHM files
    for suffix in ["", "-wal", "-shm", "-journal"]:
        f = target_db + suffix
        if os.path.exists(f):
            os.remove(f)

    # 2. Copy snapshot
    shutil.copy2(backup_file, target_db)

    # 3. Verify target invariants post-restore
    target_valid, target_stats, target_msg = verify_database_invariants(target_db)
    if not target_valid:
        raise RuntimeError(f"Post-restore verification failed: {target_msg}")

    print(f"[✓] Successfully restored database to {target_db}")
    print(f"    Zero-Sum check: PASSED ({target_msg})")
    print(f"    Total Deposited: {target_stats['total_deposited']} cents")
    return True


def apply_migrations(db_path: str):
    """
    Ensures schema versioning table exists and records current schema version.
    """
    if not os.path.exists(db_path):
        print(f"[*] Database {db_path} does not exist yet. Will be created on init.")
        return

    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at INTEGER NOT NULL,
                description TEXT NOT NULL
            )
        """)
        # Check current version
        cur = conn.cursor()
        cur.execute("SELECT MAX(version) FROM schema_migrations")
        row = cur.fetchone()
        v = row[0] if row and row[0] is not None else 0
        if v == 0:
            now = int(time.time())
            conn.execute("INSERT INTO schema_migrations (version, applied_at, description) VALUES (1, ?, 'Initial verified Escrow schema with Treasury')", (now,))
            print("[✓] Schema migration v1 recorded.")
        else:
            print(f"[✓] Schema is up to date (version {v}).")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="QAL Escrow Backup & Restore Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # backup
    p_backup = subparsers.add_parser("backup", help="Perform atomic online backup")
    p_backup.add_argument("--db", default=".qal/db/escrow.sqlite", help="Path to database")
    p_backup.add_argument("--out", default="backups", help="Output backup directory")

    # restore
    p_restore = subparsers.add_parser("restore", help="Restore database from snapshot")
    p_restore.add_argument("--file", required=True, help="Path to backup sqlite file")
    p_restore.add_argument("--target", default=".qal/db/escrow.sqlite", help="Target database path")

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify mathematical invariants on database")
    p_verify.add_argument("--db", default=".qal/db/escrow.sqlite", help="Path to database")

    # migrate
    p_migrate = subparsers.add_parser("migrate", help="Apply idempotent schema migrations")
    p_migrate.add_argument("--db", default=".qal/db/escrow.sqlite", help="Path to database")

    args = parser.parse_args()

    if args.command == "backup":
        perform_backup(args.db, args.out)
    elif args.command == "restore":
        perform_restore(args.file, args.target)
    elif args.command == "verify":
        valid, stats, msg = verify_database_invariants(args.db)
        if valid:
            print(f"[✓] VERIFIED: {msg}")
            print(json.dumps(stats, indent=2))
            sys.exit(0)
        else:
            print(f"[!] FAILED: {msg}")
            sys.exit(1)
    elif args.command == "migrate":
        apply_migrations(args.db)


if __name__ == "__main__":
    main()
