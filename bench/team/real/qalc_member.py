#!/usr/bin/env python3
"""Minimal MCP stdio client for ONE QAL agent (a copy of this file lives in the agent's work directory).

Usage:
  python3 qalc.py tools                          # server instructions and the tools of YOUR role
  python3 qalc.py call <tool> '<json-args>'      # call one tool; prints `isError: ...` and the result JSON
  python3 qalc.py call <tool> -                  # the same, JSON arguments read from stdin
  python3 qalc.py define [<file>] [--change <id>]  # qal_define with QAL source from <file> or stdin

Each invocation starts a fresh `qal mcp` server bound to your session; the project, the server binary
and your session file are fixed below by `prep.py`. Do not change them.
"""

import json
import os
import select
import subprocess
import sys

BIN = "@BIN@"
ROOT = "@ROOT@"
SESSION_FILE = "@SESSION@"  # empty for a solo (non-team) project
KEY_DIR = "@KEYDIR@"        # empty for a solo (non-team) project

REQUEST_TIMEOUT = 300  # seconds to wait for one server answer (tests, proofs and merges can be slow)


class Client:
    def __init__(self):
        cmd = [BIN, "mcp", "--root", ROOT]
        env = os.environ.copy()
        if SESSION_FILE:
            cmd += ["--session-file", SESSION_FILE]
            env["QAL_KEY_DIR"] = KEY_DIR
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, encoding="utf-8", env=env)
        self.next_id = 0
        init = self.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                           "clientInfo": {"name": "qal-agent", "version": "1"}})
        self.instructions = init.get("result", {}).get("instructions", "")
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def send(self, msg):
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def request(self, method, params):
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params})
        if not select.select([self.p.stdout], [], [], REQUEST_TIMEOUT)[0]:
            self.p.kill()
            self.p.wait()
            raise RuntimeError(f"the QAL server did not answer within {REQUEST_TIMEOUT} s")
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("the QAL server exited without answering")
        return json.loads(line)

    def tools(self):
        return self.request("tools/list", {})["result"]["tools"]

    def call(self, tool, args):
        r = self.request("tools/call", {"name": tool, "arguments": args})
        if "error" in r:
            return True, "PROTOCOL ERROR: " + json.dumps(r["error"]), None
        res = r["result"]
        text = "\n".join(c.get("text", "") for c in res.get("content", []))
        structured = res.get("structuredContent")
        if structured is None:
            try:
                structured = json.loads(text)
            except json.JSONDecodeError:
                structured = None
        return bool(res.get("isError", False)), text, structured

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
            rest = argv[2:]
            change = None
            if "--change" in rest:
                i = rest.index("--change")
                change = rest[i + 1] if i + 1 < len(rest) else None
                rest = rest[:i] + rest[i + 2:]
            src = open(rest[0], encoding="utf-8").read() if rest else sys.stdin.read()
            tool, args = "qal_define", {"source": src}
            if change:
                args["change"] = change
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
