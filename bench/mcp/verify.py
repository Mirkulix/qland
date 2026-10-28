#!/usr/bin/env python3
"""MCP package roundtrip: qal_import -> qal_export -> qal_import."""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QAL = Path(os.environ.get("QAL_BIN", REPO / "qal" / "target" / "release" / "qal"))
PKG = REPO / "qal" / "packages" / "event_queue.qalpkg"


def rpc(proc, obj):
    proc.stdin.write(json.dumps(obj) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


def main():
    if not QAL.exists():
        sys.exit(f"missing qal: {QAL}")
    with tempfile.TemporaryDirectory(prefix="qal-mcp-verify-") as tmp:
        root = Path(tmp)
        subprocess.run([str(QAL), "init"], cwd=root, check=True, capture_output=True)
        shutil.copy2(PKG, root / "pkg.qalpkg")
        proc = subprocess.Popen([str(QAL), "mcp", "--root", str(root)],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, bufsize=1)
        try:
            rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                  "clientInfo": {"name": "verify", "version": "1"}}})
            proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
            proc.stdin.flush()
            imp = rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                             "params": {"name": "qal_import", "arguments": {"path": "pkg.qalpkg"}}})
            assert not imp["result"]["isError"], imp
            exp = rpc(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                             "params": {"name": "qal_export",
                                        "arguments": {"prefix": "enqueue", "out": "out.qalpkg",
                                                      "name": "qal_enqueue_only"}}})
            assert not exp["result"]["isError"], exp
            assert (root / "out.qalpkg").is_file()
            tests = rpc(proc, {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                               "params": {"name": "qal_test", "arguments": {"names": ["push_size"]}}})
            step = tests["result"]["content"][0]["text"]
            assert json.loads(step)["passed"] == 1, step
            no_trust = rpc(proc, {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                                  "params": {"name": "qal_install",
                                             "arguments": {"url": "http://127.0.0.1:9", "name": "x"}}})
            assert no_trust["result"]["isError"], no_trust
        finally:
            proc.stdin.close()
            proc.wait(timeout=10)
    print("mcp: import/export roundtrip, 1 test, install-no-trust rejected")


if __name__ == "__main__":
    main()
