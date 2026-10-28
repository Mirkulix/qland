#!/usr/bin/env python3
"""Replicate two commands to three QAL node endpoints, then commit index 2."""
import json, os, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QAL = Path(os.environ.get("QAL_BIN", REPO / "qal" / "target" / "release" / "qal"))
NODE = REPO / "bench" / "sim" / "node.qal"


def http(url, body=None):
    data = body.encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode())


def main():
    with tempfile.TemporaryDirectory(prefix="qal-repl-") as tmp:
        procs, ports = [], []
        try:
            for i in range(3):
                d = Path(tmp) / f"n{i}"; d.mkdir()
                subprocess.run([str(QAL), "init"], cwd=d, check=True, capture_output=True)
                subprocess.run([str(QAL), "define", str(NODE)], cwd=d, check=True, capture_output=True)
                port = 8920 + i
                procs.append(subprocess.Popen([str(QAL), "serve", "handle", "--init", "setup",
                                               "--allow", "fs:.", "--listen", f"127.0.0.1:{port}"],
                                              cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
                ports.append(port)
            time.sleep(1.5)
            for cmd in ("alpha", "beta"):
                for port in ports:
                    http(f"http://127.0.0.1:{port}/append?term=1", cmd)
            logs = [http(f"http://127.0.0.1:{port}/log") for port in ports]
            assert logs[0] == logs[1] == logs[2] == ["1:1:alpha", "2:1:beta"], logs
            for port in ports:
                c = http(f"http://127.0.0.1:{port}/commit?index=2")
                assert c["commit"] == 2, c
        finally:
            for p in procs: p.terminate()
            for p in procs: p.wait(timeout=10)
    print("replicate: 3 nodes logs equal, commit index 2")


if __name__ == "__main__":
    main()
