#!/usr/bin/env python3
"""
Autonomous AI Agent Self-Synthesis & Verification Loop (Gen-3)

Demonstrates how an AI Agent autonomously synthesizes, formally verifies,
optimizes, and JIT-executes a kernel at Silicon speeds without writing linear source files.
"""

import os
import sys
import subprocess
import json
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TGS_BIN = os.path.join(ROOT_DIR, "bin", "tgs")
STDLIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stdlib")

def print_header(title):
    print("\n" + "=" * 80)
    print(f"⚡  {title}")
    print("=" * 80)

def call_mcp_tool(tool_name, arguments):
    proc = subprocess.Popen([TGS_BIN, "mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }
    stdout, _ = proc.communicate(input=json.dumps(req) + "\n", timeout=5)
    for line in stdout.strip().split("\n"):
        if not line:
            continue
        resp = json.loads(line)
        if resp.get("id") == 1:
            content = resp.get("result", {}).get("content", [])
            if content:
                return content[0].get("text", "")
    return stdout

def main():
    if not os.path.exists(TGS_BIN):
        print(f"Error: {TGS_BIN} not found! Build tgs first.")
        sys.exit(1)

    print_header("AUTONOMOUS AGENT SELF-SYNTHESIS & FORMAL VERIFICATION LOOP")
    print("Scenario: AI Agent is tasked with generating a safe, bounded compound interest kernel.")
    print("Requirements: Capital >= 0, Interest rate clamped to [0, 25]%, No overflow, Return >= 0.")

    # 1. Agent loads or synthesizes the SSA representation
    interest_file = os.path.join(STDLIB_DIR, "interest_step.ssa")
    with open(interest_file, "r") as f:
        ssa_code = f.read()

    print("\n[Step 1: Agent Emits Topological SSA / Graph]")
    print(ssa_code.strip())

    # 2. Agent invokes tgs_verify via MCP
    print("\n[Step 2: Formal Verification via MCP (<1 µs)]")
    verify_output = call_mcp_tool("tgs_verify", {
        "graph": ssa_code,
        "input_bounds": [[0, 1000000], [-50, 200]]  # Untrusted inputs: capital in [0, 1M], rate in [-50, 200]
    })
    print(verify_output.strip())

    # 3. Agent invokes tgs_optimize via MCP
    print("\n[Step 3: Graph Optimization (Constant Folding & Redundancy Pruning)]")
    opt_output = call_mcp_tool("tgs_optimize", {
        "graph": ssa_code
    })
    print(opt_output.strip())

    # 4. Agent runs the kernel on the Zero-Alloc DVM
    print("\n[Step 4: Live Execution on Zero-Alloc DVM]")
    exec_output = call_mcp_tool("tgs_run", {
        "graph": ssa_code,
        "args": [10000, 5]  # 10,000 EUR at 5% interest
    })
    print(f"interest_step(capital=10000, rate=5%) -> {exec_output.strip()} (Expected: 10500)")

    # 5. Full Native Benchmark
    print_header("NATIVE HARDWARE (JIT) SILICON BENCHMARK")
    subprocess.run([TGS_BIN, "bench"])

if __name__ == "__main__":
    main()
