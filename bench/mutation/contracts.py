#!/usr/bin/env python3
"""Contract-only mutation study: remove req/ens or weaken a contract comparison."""
import json, os, re, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bench"))
import check

SOLUTIONS = REPO / "bench" / "teacher-2026-10-27" / "solutions"
QAL = Path(os.environ.get("QAL_BIN", REPO / "qal" / "target" / "release" / "qal"))


def mutate(src, kind):
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*(req|ens)\b", line):
            if kind == "drop_req" and "req" in line:
                return "\n".join(lines[:i] + lines[i+1:]) + "\n"
            if kind == "drop_ens" and "ens" in line:
                return "\n".join(lines[:i] + lines[i+1:]) + "\n"
            if kind == "weaken_contract_ge" and ">=" in line:
                lines[i] = line.replace(">=", ">", 1)
                return "\n".join(lines) + "\n"
    return src


def validate(task, source):
    with tempfile.NamedTemporaryFile("w", suffix=".qal", delete=False, encoding="utf-8") as f:
        f.write(source)
        path = f.name
    try:
        v = check.check_qal(task, source=path)
        r = subprocess.run([str(QAL), "prove", path], capture_output=True, text=True, timeout=90)
        try:
            p = json.loads(r.stdout)
        except Exception:
            p = {"proven": 0, "refuted": 0, "unknown": 0}
        return v, p
    finally:
        os.unlink(path)


def main():
    results = []
    for sol in sorted(SOLUTIONS.glob("*.qal")):
        src = sol.read_text(encoding="utf-8")
        for kind in ("drop_req", "drop_ens", "weaken_contract_ge"):
            m = mutate(src, kind)
            if m == src:
                continue
            v, p = validate(sol.stem, m)
            total, passed = v.get("total", 0), v.get("passed", 0)
            status = ("invalid" if not total else
                      "caught_tests" if passed < total else
                      "caught_proof" if p.get("refuted", 0) else "escaped")
            results.append({"task": sol.stem, "mutator": kind, "status": status,
                            "passed": passed, "total": total,
                            "proven": p.get("proven", 0), "refuted": p.get("refuted", 0),
                            "unknown": p.get("unknown", 0)})
            print(f"{sol.stem:32} {kind:20} {status:14} tests {passed}/{total} ref {p.get('refuted',0)}", flush=True)
    out = REPO / "bench" / "mutation" / "results-contracts.json"
    out.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(json.dumps({"counts": counts, "total": len(results), "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
