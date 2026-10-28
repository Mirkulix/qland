#!/usr/bin/env python3
"""Serve a QAL cluster dashboard that aggregates three QAL node /metrics."""
import json, os, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QAL = Path(os.environ.get("QAL_BIN", REPO / "qal" / "target" / "release" / "qal"))
NODE = REPO / "bench" / "sim" / "node.qal"
DASH = REPO / "bench" / "sim" / "cluster_dashboard.qal"


def http(url, body=None):
    data = body.encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.read().decode(), r.headers.get("content-type", "")


def start(d, port, args):
    return subprocess.Popen([str(QAL), "serve", "handle", "--init", "setup"] + args +
                            ["--listen", f"127.0.0.1:{port}"], cwd=d,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    with tempfile.TemporaryDirectory(prefix="qal-cdash-") as tmp:
        procs = []
        try:
            for i in range(3):
                d = Path(tmp) / f"n{i}"; d.mkdir()
                subprocess.run([str(QAL), "init"], cwd=d, check=True, capture_output=True)
                subprocess.run([str(QAL), "define", str(NODE)], cwd=d, check=True, capture_output=True)
                procs.append(start(d, 8990 + i, ["--allow", "fs:."]))
            time.sleep(1.5)
            for i in range(3):
                base = f"http://127.0.0.1:{8990 + i}"
                http(base + "/heartbeat", "hb:0:1:100")
                http(base + "/append?term=1", "alpha")
                http(base + "/commit?index=1")
            d = Path(tmp) / "dash"; d.mkdir()
            subprocess.run([str(QAL), "init"], cwd=d, check=True, capture_output=True)
            subprocess.run([str(QAL), "define", str(DASH)], cwd=d, check=True, capture_output=True)
            allow = "net:127.0.0.1:8990,net:127.0.0.1:8991,net:127.0.0.1:8992"
            procs.append(subprocess.Popen([str(QAL), "serve", "handle", "--allow", allow,
                                           "--listen", "127.0.0.1:8999"], cwd=d,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            time.sleep(1.2)
            body, ctype = http("http://127.0.0.1:8999/api/metrics")
            data = json.loads(body)
            for k in ("n0", "n1", "n2"):
                assert json.loads(data[k])["log_len"] == 1, data
            page, ptype = http("http://127.0.0.1:8999/")
            assert "text/html" in ptype and "QAL cluster" in page, (ptype, page[:80])
            # kill node 2: health drops to 2/3
            procs[2].terminate(); procs[2].wait(timeout=10)
            time.sleep(0.3)
            health = json.loads(http("http://127.0.0.1:8999/api/health")[0])
            assert health == {"up": 2, "total": 3}, health
        finally:
            for p in procs: p.terminate()
            for p in procs: p.wait(timeout=10)
    print("dashboard: 3 nodes aggregated, health 2/3 after one dies")


if __name__ == "__main__":
    main()
