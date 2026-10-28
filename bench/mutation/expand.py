#!/usr/bin/env python3
"""Mutation benchmark over the 23 teacher solutions, code spans only.

Six source-level mutators (operators/booleans/constants) plus contract mutators
(remove req/ens, weaken >=). A mutant is `caught` only if it compiles and hidden
tests fail; `escaped` if it compiles and all hidden tests pass; `invalid` if it
does not compile.
"""
import json, os, re, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bench"))
import check

SOLUTIONS = REPO / "bench" / "teacher-2026-10-27" / "solutions"


def spans(src):
    """(start,end) spans outside QAL comments and string literals."""
    out, i, start, in_str = [], 0, 0, False
    while i < len(src):
        c = src[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
                start = i + 1
            i += 1
            continue
        if c == '"':
            if start < i:
                out.append((start, i))
            in_str = True
            i += 1
            continue
        if c == "#":
            if start < i:
                out.append((start, i))
            while i < len(src) and src[i] != "\n":
                i += 1
            start = i
            continue
        i += 1
    if start < len(src):
        out.append((start, len(src)))
    return out


def replace_code(src, old, new):
    for a, b in spans(src):
        idx = src.find(old, a, b)
        if idx >= 0:
            return src[:idx] + new + src[idx + len(old):]
    return src


def drop_first_line(src, pattern):
    for m in re.finditer(pattern, src, re.M):
        end = src.find("\n", m.start())
        if end < 0:
            return src[:m.start()]
        return src[:m.start()] + src[end + 1:]
    return src


MUTATORS = [
    ("plus_to_minus", lambda s: replace_code(s, " + ", " - ")),
    ("minus_to_plus", lambda s: replace_code(s, " - ", " + ")),
    ("times_to_plus", lambda s: replace_code(s, " * ", " + ")),
    ("eq_to_ne", lambda s: replace_code(s, " == ", " != ")),
    ("true_to_false", lambda s: replace_code(s, "true", "false")),
    ("zero_to_one", lambda s: replace_code(s, "0", "1")),
    ("drop_req", lambda s: drop_first_line(s, r"^\s*req\b.*$")),
    ("drop_ens", lambda s: drop_first_line(s, r"^\s*ens\b.*$")),
    ("weaken_ge", lambda s: replace_code(s, ">=", ">")),
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
