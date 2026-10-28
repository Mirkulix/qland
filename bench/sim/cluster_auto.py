#!/usr/bin/env python3
"""Leader dies, two followers time out/vote, new leader continues, old leader catches up."""
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
    with tempfile.TemporaryDirectory(prefix="qal-auto-") as tmp:
        dirs, procs, ports = [], [], []
        try:
            for i in range(3):
                d = Path(tmp) / f"n{i}"; d.mkdir()
                subprocess.run([str(QAL), "init"], cwd=d, check=True, capture_output=True)
                subprocess.run([str(QAL), "define", str(NODE)], cwd=d, check=True, capture_output=True)
                dirs.append(d); ports.append(8950 + i); procs.append(start(d, 8950 + i))
            time.sleep(1.5)
            for p in ports:
                http(f"http://127.0.0.1:{p}/heartbeat", "hb:0:1:0")
                http(f"http://127.0.0.1:{p}/append?term=1", "cmd1")
                http(f"http://127.0.0.1:{p}/commit?index=1")
            # leader 0 dies.
            procs[0].terminate(); procs[0].wait(timeout=10)
            # followers time out and vote for node 1 at term 2.
            for p in ports[1:]:
                t = http(f"http://127.0.0.1:{p}/timeout?now=200&timeout=50")
                assert t["role"] == "candidate" and t["term"] == 2, t
                v = http(f"http://127.0.0.1:{p}/vote?term=2&candidate=1")
                assert v["granted"] is True, v
                http(f"http://127.0.0.1:{p}/heartbeat", "hb:1:2:200")
                http(f"http://127.0.0.1:{p}/append?term=2", "cmd2")
                http(f"http://127.0.0.1:{p}/commit?index=2")
            # old leader comes back and catches up.
            procs[0] = start(dirs[0], ports[0]); time.sleep(1.2)
            http(f"http://127.0.0.1:{ports[0]}/heartbeat", "hb:1:2:200")
            http(f"http://127.0.0.1:{ports[0]}/append?term=2", "cmd2")
            http(f"http://127.0.0.1:{ports[0]}/commit?index=2")
            logs = [http(f"http://127.0.0.1:{p}/log") for p in ports]
            expected = ["1:1:cmd1", "2:2:cmd2"]
            assert logs[0] == logs[1] == logs[2] == expected, logs
            states = [http(f"http://127.0.0.1:{p}/state") for p in ports]
            assert all(s["leader"] == 1 and s["term"] == 2 for s in states), states
        finally:
            for p in procs:
                if p.poll() is None: p.terminate()
            for p in procs:
                if p.poll() is None: p.wait(timeout=10)
    print("auto: leader 0 died -> node 1 term 2 -> old leader caught up")


if __name__ == "__main__":
    main()
