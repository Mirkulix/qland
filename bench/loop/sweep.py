#!/usr/bin/env python3
"""The 1-turn verification loop, measured: every function of a QAL store is defined again with a
change that keeps its meaning (`let zz = 0 in (...)`), and the qal_define reply is read.

Usage: python3 bench/loop/sweep.py <store-root>

<store-root> holds .qal/store.db and is changed (use a copy). Prints the wall time of each define
(one MCP call), the tests it ran, its property statuses and its property warnings.
"""

import collections
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qalc  # noqa: E402


def main(root):
    c = qalc.Client(root=root)
    try:
        _, _, q = c.call("qal_query", {"outline": True})
        fns = [e["name"] for e in q["results"] if e["kind"] == "fn" and e["name"] != "main"]
        times, tests, relinked = [], [], []
        props, warns, found = collections.Counter(), collections.Counter(), []
        for n in fns:
            _, _, v = c.call("qal_view", {"names": [n]})
            t = v["views"][0]["text"]
            i = t.index("\n= ")
            t0 = time.time()
            err, _, o = c.call("qal_define", {"source": t[:i] + "\n= let zz = 0 in (" + t[i + 3:] + ")"})
            times.append(time.time() - t0)
            if err:
                print("refused", n, [d["message"][:80] for d in o.get("diagnostics", [])][:1])
                continue
            tests.append(len(o.get("tests") or []))
            relinked.append(sum(1 for d in o.get("defined", []) if d["status"] == "relinked"))
            for p in o.get("properties") or []:
                if p["def"] == n:
                    props[p["status"]] += 1
            for w in o.get("warnings") or []:
                if w["code"] in ("E0803", "E0804", "E0805"):
                    warns[w["code"]] += 1
                    found.append((n, w["code"], w["message"][:110]))
        ts = sorted(times)
        med = lambda xs: sorted(xs)[len(xs) // 2]
        print(f"{len(fns)} functions; define wall time median {med(ts):.2f}s, p90 {ts[int(len(ts) * 0.9)]:.2f}s, max {ts[-1]:.2f}s")
        print(f"per define (median): {med(tests)} tests run, {med(relinked)} dependents relinked")
        print("property of the defined fn:", dict(props), "warnings:", dict(warns))
        for f in found:
            print("  ", *f)
    finally:
        c.close()


if __name__ == "__main__":
    main(sys.argv[1])
