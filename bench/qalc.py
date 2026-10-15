#!/usr/bin/env python3
"""Minimal MCP stdio client for the QAL server (`qal mcp`), used by the benchmark harness and, as a
copy in its work directory, by the agent of the QAL arm.

Usage:
  python3 qalc.py tools                     # server instructions and every tool with its description
  python3 qalc.py call <tool> '<json-args>' # call one tool; prints `isError: ...` and the result JSON
  python3 qalc.py call <tool> -             # the same, JSON arguments read from stdin
  python3 qalc.py define [<file>]           # qal_define with QAL source from <file> or stdin

Each invocation starts a fresh server (`qal mcp --root ROOT`). The project root ROOT (where
.qal/store.db lives) is $QAL_ROOT, else the current directory; the server binary is $QAL_BIN, else
the release build of this repository (`cargo build --release -p qal-cli` in qal/). `run.py prepare`
gives an agent a copy with both fixed.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get("QAL_BIN") or os.path.join(os.path.dirname(HERE), "qal", "target", "release", "qal")
ROOT = os.environ.get("QAL_ROOT") or os.getcwd()

class Client:
    """One `qal mcp` server process speaking MCP (initialize handshake, then tools/*)."""

    def __init__(self, root=ROOT, binary=BIN):
        self.p = subprocess.Popen([binary, "mcp", "--root", root], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, encoding="utf-8")
        self.next_id = 0
        init = self.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                           "clientInfo": {"name": "qal-bench", "version": "1"}})
        self.instructions = init.get("result", {}).get("instructions", "")
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def send(self, msg):
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def request(self, method, params):
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params})
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("the QAL server exited without answering")
        return json.loads(line)

    def tools(self):
        return self.request("tools/list", {})["result"]["tools"]

    def call(self, tool, args):
        """Returns (is_error, result text, structured result) of one tool call."""
        r = self.request("tools/call", {"name": tool, "arguments": args})
        if "error" in r:
            return True, "PROTOCOL ERROR: " + json.dumps(r["error"]), None
        res = r["result"]
        text = "\n".join(c.get("text", "") for c in res.get("content", []))
        return bool(res.get("isError", False)), text, res.get("structuredContent")

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=30)

def describe_tools(c):
    out = [f"Server instructions: {c.instructions}\n"]
    for t in c.tools():
        out.append(f"## {t['name']}\n{t.get('description', '')}\ninputSchema: {json.dumps(t.get('inputSchema'))}\n")
    return "\n".join(out)

def main(argv):
    if len(argv) < 2 or argv[1] not in ("tools", "call", "define"):
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    c = Client()
    try:
        if argv[1] == "tools":
            print(describe_tools(c))
            return 0
        if argv[1] == "define":
            src = open(argv[2], encoding="utf-8").read() if len(argv) > 2 else sys.stdin.read()
            tool, args = "qal_define", {"source": src}
        else:
            if len(argv) != 4:
                print("usage: qalc.py call <tool> '<json-args>' | -", file=sys.stderr)
                return 2
            tool, raw = argv[2], (sys.stdin.read() if argv[3] == "-" else argv[3])
            try:
                args = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"invalid JSON arguments: {e}", file=sys.stderr)
                return 2
        is_error, text, _ = c.call(tool, args)
        print("isError:", str(is_error).lower())
        print(text)
        return 0
    finally:
        c.close()

if __name__ == "__main__":
    sys.exit(main(sys.argv))
