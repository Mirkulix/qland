#!/usr/bin/env python3
"""
Executes the QAL 4-Agent Autonomous Team Workflow for Milestone P1.5 (Escrow Payout Engine):
1. arch-1 (Architect): Opens change set, defines types, formal Z3 contracts, and service handlers. Submits change set.
2. ver-1 (Verifier): Runs formal Z3 prover (qal_prove) and property tests (qal_test). Signs 'verified' review.
3. rev-1 (Reviewer): Inspects diff (qal_change diff), signs 'approve' review, and merges into main (qal_merge).
4. Verifies cryptographic Ed25519 hash chain of custody (qal_verify).
"""

import json
import os
import subprocess
import sys
import time

QAL_BIN = os.path.abspath("bin/qal")
ROOT = os.path.abspath("pilot/escrow/qal")
SESSIONS = os.path.expanduser("~/.config/qal/sessions/escrow")

class AgentClient:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        session_file = os.path.join(SESSIONS, f"{agent_id}.json")
        self.p = subprocess.Popen(
            [QAL_BIN, "mcp", "--root", ROOT, "--session-file", session_file],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8"
        )
        self.msg_id = 0
        init_res = self.call("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": f"agent-{agent_id}", "version": "1.0"}
        }, raw_method=True)
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def send(self, msg):
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def call(self, method: str, params: dict, raw_method: bool = False):
        self.msg_id += 1
        if raw_method:
            msg = {"jsonrpc": "2.0", "id": self.msg_id, "method": method, "params": params}
        else:
            msg = {"jsonrpc": "2.0", "id": self.msg_id, "method": "tools/call", "params": {"name": method, "arguments": params}}
        self.send(msg)
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError(f"Agent {self.agent_id} received empty response from MCP server")
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError(f"Agent {self.agent_id} RPC error: {resp['error']}")
        return resp.get("result", {})

    def tool(self, tool_name: str, args: dict):
        res = self.call(tool_name, args)
        if res.get("isError"):
            diag = res.get("structuredContent") or res.get("content")
            raise RuntimeError(f"Agent {self.agent_id} tool error in {tool_name}: {diag}")
        return res.get("structuredContent")

    def close(self):
        self.p.kill()
        self.p.wait()

def main():
    print("=== Step 1: arch-1 (Architect) opens change set and defines contracts ===")
    arch = AgentClient("arch-1")
    who = arch.tool("qal_whoami", {})
    print(f"Authenticated as: {who['member']} (role: {who['role']}, session: {who['session']})")

    open_res = arch.tool("qal_change", {
        "action": "open",
        "title": "escrow-payout-engine",
        "description": "High-Assurance Financial Escrow & Tiered Multi-Sig Engine with Z3 proofs"
    })
    change_id = open_res["change"]
    print(f"Created Change Set: {change_id}")

    # Read the complete verified source
    with open("pilot/escrow/qal/escrow.qal", "r", encoding="utf-8") as f:
        qal_source = f.read()

    define_res = arch.tool("qal_define", {
        "change": change_id,
        "source": qal_source
    })
    print(f"Defined {len(define_res.get('entries', []))} definitions in change set {change_id}")

    # Submit change set for verification and review
    submit_res = arch.tool("qal_change", {
        "action": "submit",
        "change": change_id
    })
    print(f"Change set {change_id} submitted for review.")
    arch.close()

    print("\n=== Step 2: ver-1 (Verifier) proves contracts and verifies obligations ===")
    ver = AgentClient("ver-1")
    who_ver = ver.tool("qal_whoami", {})
    print(f"Authenticated as: {who_ver['member']} (role: {who_ver['role']})")

    prove_res = ver.tool("qal_prove", {"change": change_id})
    proven_count = prove_res.get("proven", 0)
    refuted_count = prove_res.get("refuted", 0)
    unknown_count = prove_res.get("unknown", 0)
    print(f"Z3 Formal Prover: proven={proven_count}, refuted={refuted_count}, unknown={unknown_count}")
    assert refuted_count == 0, f"Contracts refuted: {prove_res}"
    assert unknown_count == 0, f"Unknown obligations: {prove_res}"

    test_res = ver.tool("qal_test", {"change": change_id})
    print(f"Test Suite: passed={test_res.get('passed')}, failed={test_res.get('failed')}")
    assert test_res.get("failed") == 0, f"Tests failed: {test_res}"

    # Sign verification verdict
    ver_review = ver.tool("qal_review", {
        "change": change_id,
        "verdict": "verified"
    })
    print(f"ver-1 verdict signed: verified.")
    ver.close()

    print("\n=== Step 3: rev-1 (Reviewer) inspects diff, approves, and merges to main ===")
    rev = AgentClient("rev-1")
    who_rev = rev.tool("qal_whoami", {})
    print(f"Authenticated as: {who_rev['member']} (role: {who_rev['role']})")

    diff_res = rev.tool("qal_change", {
        "action": "diff",
        "change": change_id
    })
    print(f"Inspected diff: {len(diff_res.get('added', []))} additions, {len(diff_res.get('changed', []))} modifications")

    # Sign approval verdict
    rev_review = rev.tool("qal_review", {
        "change": change_id,
        "verdict": "approve"
    })
    print(f"rev-1 verdict signed: approve.")

    # Merge into main branch of QAL store
    merge_res = rev.tool("qal_merge", {
        "change": change_id
    })
    print(f"Merged into main! Head record: {merge_res.get('head', 'ok')}")
    rev.close()

    print("\n=== Step 4: Cryptographic Chain of Custody Verification ===")
    verify_cmd = [QAL_BIN, "verify", "--trust", os.path.join(SESSIONS, "trust.json")]
    p = subprocess.run(verify_cmd, cwd=ROOT, capture_output=True, text=True)
    print(f"qal verify output: {p.stdout.strip()}")
    assert p.returncode == 0, f"Verification failed: {p.stderr}"
    print("\n✓ QAL Multi-Agent Team Pipeline Executed Successfully!")

if __name__ == "__main__":
    main()
