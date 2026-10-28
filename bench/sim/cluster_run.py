#!/usr/bin/env python3
"""Start N QAL node endpoints and send the same heartbeat to all of them."""
import json, os, shutil, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QAL = Path(os.environ.get("QAL_BIN", REPO / "qal" / "target" / "release" / "qal"))
NODE = REPO / "bench" / "sim" / "node.qal"


def http(url, body=None):
    data = body.encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode())


def main():
    with tempfile.TemporaryDirectory(prefix="qal-cluster-") as tmp:
        procs = []
        ports = []
        try:
            for i in range(3):
                d = Path(tmp) / f"n{i}"; d.mkdir()
                subprocess.run([str(QAL), "init"], cwd=d, check=True, capture_output=True)
                subprocess.run([str(QAL), "define", str(NODE)], cwd=d, check=True, capture_output=True)
                port = 8910 + i
                p = subprocess.Popen([str(QAL), "serve", "handle", "--init", "setup",
                                      "--allow", "fs:.", "--listen", f"127.0.0.1:{port}"],
                                     cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                procs.append(p); ports.append(port)
            time.sleep(1.5)
            for at in (100, 101, 102):
                for port in ports:
                    r = http(f"http://127.0.0.1:{port}/heartbeat", f"hb:1:2:{at}")
                    assert (r["leader"], r["term"], r["at"]) == (1, 2, at), r
            for port in ports:
                s = http(f"http://127.0.0.1:{port}/state")
                assert (s["leader"], s["term"], s["at"]) == (1, 2, 102), s
            # Leader failure: no heartbeat until now=200/timeout=50 -> candidates.
            for port in ports:
                t = http(f"http://127.0.0.1:{port}/timeout?now=200&timeout=50")
                assert t["role"] == "candidate" and t["term"] == 3, t
            # The host runs a vote: candidate 0, term 3; all three grant.
            granted = 0
            for port in ports:
                v = http(f"http://127.0.0.1:{port}/vote?term=3&candidate=0")
                granted += 1 if v.get("granted") else 0
            assert granted == 3, granted
            stale = http(f"http://127.0.0.1:{ports[0]}/vote?term=2&candidate=1")
            assert stale.get("granted") is False, stale
            # The leader sends its heartbeat; all become followers again.
            for port in ports:
                http(f"http://127.0.0.1:{port}/heartbeat", "hb:0:3:100")
            for port in ports:
                s = http(f"http://127.0.0.1:{port}/state")
                assert (s["leader"], s["term"], s["role"]) == (0, 3, "follower"), s
        finally:
            for p in procs:
                p.terminate()
            for p in procs:
                p.wait(timeout=10)
    print("cluster: 3/3 votes -> leader 0, stale vote rejected")


if __name__ == "__main__":
    main()
