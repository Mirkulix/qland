#!/usr/bin/env python3
"""Continuous leader loop: append, replicate, commit and heartbeat for 10 ticks."""
import json, os, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QAL = Path(os.environ.get("QAL_BIN", REPO / "qal" / "target" / "release" / "qal"))
NODE = REPO / "bench" / "sim" / "node.qal"
TICKS = 10


def http(url, body=None):
    data = body.encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode())


def main():
    with tempfile.TemporaryDirectory(prefix="qal-loop-") as tmp:
        procs, ports = [], []
        try:
            for i in range(3):
                d = Path(tmp) / f"n{i}"; d.mkdir()
                subprocess.run([str(QAL), "init"], cwd=d, check=True, capture_output=True)
                subprocess.run([str(QAL), "define", str(NODE)], cwd=d, check=True, capture_output=True)
                port = 8940 + i
                procs.append(subprocess.Popen([str(QAL), "serve", "handle", "--init", "setup",
                                               "--allow", "fs:.", "--listen", f"127.0.0.1:{port}"],
                                              cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
                ports.append(port)
            time.sleep(1.5)
            # Elect node 0 as leader term 1.
            for port in ports:
                http(f"http://127.0.0.1:{port}/heartbeat", "hb:0:1:100")
            for i in range(1, TICKS + 1):
                for port in ports:
                    http(f"http://127.0.0.1:{port}/append?term=1", f"cmd{i}")
                for port in ports:
                    http(f"http://127.0.0.1:{port}/commit?index={i}")
                    http(f"http://127.0.0.1:{port}/heartbeat", f"hb:0:1:{100 + i}")
                time.sleep(0.05)
            logs = [http(f"http://127.0.0.1:{port}/log") for port in ports]
            expected = [f"{i}:1:cmd{i}" for i in range(1, TICKS + 1)]
            assert logs[0] == logs[1] == logs[2] == expected, logs
            states = [http(f"http://127.0.0.1:{port}/state") for port in ports]
            assert all(s["leader"] == 0 and s["at"] == 110 for s in states), states
        finally:
            for p in procs: p.terminate()
            for p in procs: p.wait(timeout=10)
    print(f"loop: {TICKS} ticks replicated to 3 nodes, heartbeat at 110")


if __name__ == "__main__":
    main()
