#!/usr/bin/env python3
"""Transfer a leader snapshot prefix to a fresh follower via /snapshot + /restore."""
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


def start(d, port):
    return subprocess.Popen([str(QAL), "serve", "handle", "--init", "setup", "--allow", "fs:.",
                             "--listen", f"127.0.0.1:{port}"], cwd=d,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    with tempfile.TemporaryDirectory(prefix="qal-sync-") as tmp:
        procs, ports = [], []
        try:
            for i in range(2):
                d = Path(tmp) / f"n{i}"; d.mkdir()
                subprocess.run([str(QAL), "init"], cwd=d, check=True, capture_output=True)
                subprocess.run([str(QAL), "define", str(NODE)], cwd=d, check=True, capture_output=True)
                ports.append(8970 + i); procs.append(start(d, 8970 + i))
            time.sleep(1.5)
            for cmd in ("a", "b", "c"):
                http(f"http://127.0.0.1:{ports[0]}/append?term=1", cmd)
            http(f"http://127.0.0.1:{ports[0]}/commit?index=3")
            snap = http(f"http://127.0.0.1:{ports[0]}/snapshot?index=3")
            body = "\n".join(snap["entries"])
            restored = http(f"http://127.0.0.1:{ports[1]}/restore?index={snap['last_index']}", body)
            assert restored["last_index"] == 3, restored
            logs = [http(f"http://127.0.0.1:{p}/log") for p in ports]
            assert logs[0] == logs[1] == ["1:1:a", "2:1:b", "3:1:c"], logs
        finally:
            for p in procs: p.terminate()
            for p in procs: p.wait(timeout=10)
    print("snapshot: 3-entry prefix transferred to a fresh follower")


if __name__ == "__main__":
    main()
