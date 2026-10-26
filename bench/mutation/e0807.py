#!/usr/bin/env python3
"""E0807 on the mutation study: how many of the 15 mutants does the behaviour memory report?

Usage: python3 bench/mutation/e0807.py <qal-root> <transcripts-glob> <out.json>

<qal-root> is the final store of a QAL line (pressure run 9, as in mutate.py); <transcripts-glob>
matches the agent transcripts of that line's 12 stages. Every io-only `qal_run` input the agents
made is replayed once on a copy of the store (that fills the behaviour memory, as it was filled
while they worked); then each mutant is defined and the reply's E0807 warning is recorded, with
the stored tests that fail and the hidden tests of stage 12. A mutant counts only where the hidden
tests fail on it.
"""

import glob
import json
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
os.environ.setdefault("BENCH_TASKS", os.path.join(BENCH, "living", "tasks-pressure"))
sys.path.insert(0, BENCH)
sys.path.insert(0, HERE)
import check  # noqa: E402
import qalc  # noqa: E402
from mutate import apply  # noqa: E402


def runs_of(pattern):
    """The `qal_run` arguments the agents passed, in transcript order."""
    cases = []
    for f in sorted(glob.glob(pattern)):
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            if d.get("type") != "assistant":
                continue
            for b in d["message"].get("content", []):
                if b.get("type") != "tool_use" or b["name"] != "Bash":
                    continue
                cmd = b["input"]["command"]
                m = re.search(r"call qal_run '(\{.*\})'", cmd, re.S) or re.search(r"call qal_run - <<'?(\w+)'?\n(.*?)\n\1", cmd, re.S)
                if not m:
                    continue
                try:
                    cases.append(json.loads(m.group(1) if m.lastindex == 1 else m.group(2)))
                except ValueError:
                    pass
    return cases


def main(argv):
    if len(argv) != 4:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    root, pattern, out = os.path.abspath(argv[1]), argv[2], os.path.abspath(argv[3])
    cases = runs_of(pattern)
    with open(os.path.join(HERE, "mutants.json"), encoding="utf-8") as f:
        mutants = json.load(f)
    check.ensure_qal_binary()
    results = []
    for m in mutants:
        with tempfile.TemporaryDirectory(prefix="e0807-") as tmp:
            shutil.copytree(os.path.join(root, ".qal"), os.path.join(tmp, ".qal"))
            for f in os.listdir(os.path.join(tmp, ".qal")):
                if f != "store.db":
                    os.remove(os.path.join(tmp, ".qal", f))
            c = qalc.Client(root=tmp)
            try:
                for a in cases:
                    c.call("qal_run", a)
                defs = sorted({e["def"] for e in m["qal"]})
                _, _, views = c.call("qal_view", {"names": defs})
                texts = {v["name"]: v["text"] for v in views["views"]}
                src = "\n\n".join(apply(texts[d], [e for e in m["qal"] if e["def"] == d], m["id"]) for d in defs)
                is_error, _, reply = c.call("qal_define", {"source": src})
            finally:
                c.close()
            e0807 = [w for w in reply.get("warnings") or [] if w["code"] == "E0807"]
            r = {
                "id": m["id"],
                "stored": not is_error,
                "e0807": [w["message"] for w in e0807],
                "changed_runs": sum(len(w.get("changed_runs") or []) for w in e0807) if e0807 else 0,
                "failed_tests": [t["name"] for t in reply.get("tests") or [] if not t.get("passed")],
                "hidden_failed": None,
            }
            if not is_error:
                hidden = check.check_qal("42-living-m", root=tmp)
                r["hidden_failed"] = len(hidden["failed"])
            results.append(r)
            print(m["id"], "| stored", r["stored"], "| E0807", bool(e0807), "| failed tests", len(r["failed_tests"]), "| hidden failed", r["hidden_failed"], flush=True)
    print("observed runs replayed:", len(cases))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"observed_runs": len(cases), "results": results}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
