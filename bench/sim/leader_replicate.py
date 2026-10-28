#!/usr/bin/env python3
"""Leader replicates alpha, loses one follower, commits on 2/3, then catches it up."""
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


def start(dir, port):
    return subprocess.Popen([str(QAL), "serve", "handle", "--init", "setup", "--allow", "fs:.",
                             "--listen", f"127.0.0.1:{port}"], cwd=dir,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    with tempfile.TemporaryDirectory(prefix="qal-leader-") as tmp:
        dirs, procs, ports = [], [], []
        try:
            for i in range(3):
                d = Path(tmp) / f"n{i}"; d.mkdir()
                subprocess.run([str(QAL), "init"], cwd=d, check=True, capture_output=True)
                subprocess.run([str(QAL), "define", str(NODE)], cwd=d, check=True, capture_output=True)
                dirs.append(d); ports.append(8930 + i); procs.append(start(d, 8930 + i))
            time.sleep(1.5)
            # alpha reaches all three.
            for port in ports:
                http(f"http://127.0.0.1:{port}/append?term=1", "alpha")
            # follower 2 dies.
            procs[2].terminate(); procs[2].wait(timeout=10)
            # beta reaches the leader + one follower, commit 2 on 2/3.
            for port in ports[:2]:
                http(f"http://127.0.0.1:{port}/append?term=1", "beta")
                http(f"http://127.0.0.1:{port}/commit?index=2")
            # follower 2 comes back and is caught up.
            procs[2] = start(dirs[2], ports[2])
            time.sleep(1.2)
            http(f"http://127.0.0.1:{ports[2]}/append?term=1", "beta")
            http(f"http://127.0.0.1:{ports[2]}/commit?index=2")
            logs = [http(f"http://127.0.0.1:{port}/log") for port in ports]
            assert logs[0] == logs[1] == logs[2] == ["1:1:alpha", "2:1:beta"], logs
        finally:
            for p in procs:
                if p.poll() is None:
                    p.terminate()
            for p in procs:
                if p.poll() is None:
                    p.wait(timeout=10)
    print("leader: alpha on 3, beta on 2/3, follower restarted and caught up")


if __name__ == "__main__":
    main()
