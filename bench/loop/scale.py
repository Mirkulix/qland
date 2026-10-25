#!/usr/bin/env python3
"""Builds a codebase ten times the size of a store and measures one define in it.

Usage: python3 bench/loop/scale.py <store-root> <out-dir>

Every definition of <store-root>'s project is copied ten times with the suffix `_c0` .. `_c9` on
every name, variant and field (not inside string literals; records with the same fields would be
one type), and the copies are defined into a new store in <out-dir>. Then ten functions of copy 3
are defined again with a change that keeps their meaning, and the wall time of each define is
printed with the tests it ran and the dependents it relinked.
"""

import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qalc  # noqa: E402

QAL = os.environ.get("QAL_BIN", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "qal", "target", "release", "qal"))
BUILTIN = {"Int", "Str", "Bool", "List", "Map", "Option", "Some", "None", "Ok", "Err", "Result", "Unit", "Float", "Bytes"}


def main(root, out):
    o = json.loads(subprocess.run([QAL, "query", "--outline"], cwd=root, capture_output=True, text=True).stdout)["results"]
    names = [e["name"] for e in o]
    views = json.loads(subprocess.run([QAL, "view", *names], cwd=root, capture_output=True, text=True).stdout)["views"]
    src = "\n\n".join(v["text"] for v in views)
    words = set(names)
    for v in views:
        if v["kind"] == "type":
            words |= {w for w in re.findall(r"\b([A-Za-z_]\w*)\b", v["text"]) if w not in BUILTIN and w != "type"}
    pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, words), key=len, reverse=True)) + r")\b")

    def copy(k):
        parts = re.split(r'("(?:[^"\\]|\\.)*")', src)
        return "".join(p if i % 2 else pat.sub(lambda m: m.group(1) + f"_c{k}", p) for i, p in enumerate(parts))

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "big.qal"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(copy(k) for k in range(10)) + "\n")
    subprocess.run([QAL, "init"], cwd=out, capture_output=True, check=True)
    r = json.loads(subprocess.run([QAL, "define", "big.qal"], cwd=out, capture_output=True, text=True).stdout)
    print(f"{len(r.get('defined', []))} definitions stored, {len(r.get('tests') or [])} tests, failed {r.get('failed')}")
    c = qalc.Client(root=out)
    try:
        _, _, q = c.call("qal_query", {"outline": True})
        fns = [e["name"] for e in q["results"] if e["kind"] == "fn" and e["name"].endswith("_c3") and not e["name"].startswith("main")]
        for n in fns[::6][:10]:
            _, _, v = c.call("qal_view", {"names": [n]})
            t = v["views"][0]["text"]
            i = t.index("\n= ")
            t0 = time.time()
            _, _, o = c.call("qal_define", {"source": t[:i] + "\n= let zz = 0 in (" + t[i + 3:] + ")"})
            print(f"{n:<22} {time.time() - t0:5.2f}s tests={len(o.get('tests') or [])} relinked={sum(1 for d in o.get('defined', []) if d['status'] == 'relinked')}")
    finally:
        c.close()


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]))
