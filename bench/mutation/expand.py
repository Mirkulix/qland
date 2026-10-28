#!/usr/bin/env python3
"""Expanded mutation benchmark: 138 mutants over the 23 teacher solutions.

Each mutant is one source-level change (operator/constant/boolean) applied to the
first occurrence. A mutant counts only if it compiles and the hidden tests fail;
compile errors are invalid, passing hidden tests are escaped/equivalent.
"""
import json, os, re, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bench"))
import check

SOLUTIONS = REPO / "bench" / "teacher-2026-10-27" / "solutions"
MUTATORS = [
    ("plus_to_minus", lambda s: s.replace(" + ", " - ", 1)),
    ("minus_to_plus", lambda s: s.replace(" - ", " + ", 1)),
    ("times_to_plus", lambda s: s.replace(" * ", " + ", 1)),
    ("eq_to_ne", lambda s: s.replace(" == ", " != ", 1)),
    ("true_to_false", lambda s: s.replace("true", "false", 1)),
    ("zero_to_one", lambda s: s.replace("0", "1", 1)),
]


def validate(task, source):
    with tempfile.NamedTemporaryFile("w", suffix=".qal", delete=False, encoding="utf-8") as f:
        f.write(source)
        path = f.name
    try:
        return check.check_qal(task, source=path)
    finally:
        os.unlink(path)


def main():
    results = []
    for sol in sorted(SOLUTIONS.glob("*.qal")):
        task = sol.stem
        src = sol.read_text(encoding="utf-8")
        for name, fn in MUTATORS:
            mutant = fn(src)
            if mutant == src:
                continue
            try:
                v = validate(task, mutant)
            except Exception as e:
                results.append({"task": task, "mutator": name, "status": "invalid", "error": str(e)[:120]})
                continue
            total = v.get("total", 0)
            passed = v.get("passed", 0)
            if not total:
                status = "invalid"
            elif passed < total:
                status = "caught"
            else:
                status = "escaped"
            results.append({"task": task, "mutator": name, "status": status,
                            "passed": passed, "total": total})
            print(f"{task:32} {name:16} {status:8} {passed}/{total}", flush=True)
    out = REPO / "bench" / "mutation" / "results-expanded.json"
    out.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(json.dumps({"counts": counts, "total": len(results), "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
