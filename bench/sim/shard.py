#!/usr/bin/env python3
"""Run the QAL simulation in independent shards: each process has its own 256 MiB."""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QAL = Path(os.environ.get("QAL_BIN", REPO / "qal" / "target" / "release" / "qal"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=100)
    ap.add_argument("--agents", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()

    def one(i):
        r = subprocess.run([str(QAL), "run", str(REPO / "bench" / "sim" / "shard.qal"),
                            "--allow", "io,rand", "--", str(a.agents), str(a.steps)],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"shard {i} failed: {(r.stdout or r.stderr)[-300:]}")
        try:
            return int(json.loads(r.stdout)["stdout"].strip())
        except Exception as e:
            raise RuntimeError(f"shard {i} bad output: {r.stdout!r} ({e})")

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        total = sum(ex.map(one, range(a.shards)))
    expected = a.shards * a.agents * 1000
    if total != expected:
        sys.exit(f"total mismatch: {total} != {expected}")
    print(json.dumps({"shards": a.shards, "agents": a.agents, "steps": a.steps,
                      "total_agents": a.shards * a.agents, "jobs": a.jobs, "total": total,
                      "expected": expected, "seconds": round(time.time() - t0, 2)}))


if __name__ == "__main__":
    main()
