#!/usr/bin/env python3
"""100-task AST vs Python comparison (10 families x 10 variants)."""
import json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import env, api, strip_code, run_qal, run_py  # noqa: E402
from ast_run import lower  # noqa: E402

PRELUDE = (HERE / "prelude.qal").read_text(encoding="utf-8")

FAMILIES = {
    "sum": ("Int", "sum_all", "Print the sum of the whitespace-separated integers.", ["2 3", "10 -4 6", "1 2 3 4 5", "-1 1", "100 200", "0 0 0", "7", "8 9", "-5 -6", "42 8"]),
    "factorial": ("Int", "fact_first", "Print the factorial of the first integer.", ["5", "0", "1", "3", "6", "7", "4", "2", "8", "9"]),
    "word_count": ("Int", "count_words", "Print the number of whitespace-separated words.", ["hello world", "a  b\tc", "one", "  spaced  out  ", "a b c d", "x y", "", "many words here now", "tab\tsep", "last one"]),
    "max": ("Int", "max_all", "Print the largest whitespace-separated integer.", ["3 9 2", "-5 -2 -9", "1", "10 10", "0 -1", "100 99 101", "7 7 7", "-1 -2 -3 0", "5 4 3 2 1", "42"]),
    "reverse": ("Str", "reverse_line", "Print the input line reversed.", ["abc", "qal", "hello", "x", "12345", "a b c", "racecar", "Hello, World", "ab", "zz"]),
    "count_chars": ("Int", "count_chars", "Print the number of characters in the trimmed input.", ["abc", "hello world", "x", "", "  padded  ", "12345", "qal", "a b", "longer line here", "z"]),
    "min": ("Int", "min_all", "Print the smallest whitespace-separated integer.", ["3 9 2", "-5 -2 -9", "1", "10 10", "0 -1", "100 99 101", "7 7 7", "-1 -2 -3 0", "5 4 3 2 1", "42"]),
    "product": ("Int", "product_all", "Print the product of the whitespace-separated integers.", ["2 3", "1 2 3 4", "5", "0 7", "-2 3", "10 10", "1 1 1", "-1 -1", "2 2 2 2", "6 7"]),
    "abs_diff": ("Int", "abs_diff", "Print the absolute difference of the first two integers.", ["2 5", "10 3", "-4 4", "0 7", "9 1", "-2 -5", "100 99", "3 3", "8 2", "-1 1"]),
    "first_int": ("Int", "first_int", "Print the first whitespace-separated integer.", ["7 8", "42", "-3 4", "0 9", "10 20 30", "5", "-1 -2", "100 1", "9 9", "2 3 4"]),
}


def calls(e):
    out = []
    if isinstance(e, dict):
        if isinstance(e.get("call"), str):
            out.append(e["call"])
            for a in e.get("args", []):
                out += calls(a)
        for k in ("op", "list"):
            v = e.get(k)
            if isinstance(v, list):
                for a in v:
                    out += calls(a)
        for k in ("field", "if", "match", "in"):
            if k in e:
                out += calls(e[k])
        for k in ("then", "else"):
            if k in e:
                out += calls(e[k])
        for b in e.get("let", []):
            out += calls(b.get("value"))
        for c in e.get("cases", []):
            out += calls(c.get("body"))
        for v in e.get("fields", {}).values():
            out += calls(v)
    return out


def tasks():
    out = []
    for fam, (ret, helper, desc, inputs) in FAMILIES.items():
        for i, inp in enumerate(inputs):
            expected = None
            numeric = {"sum", "factorial", "max", "min", "product", "abs_diff", "first_int"}
            nums = [int(x) for x in inp.split()] if (fam in numeric and inp.split()) else []
            if fam == "sum": expected = sum(nums)
            elif fam == "factorial": expected = 1 if nums[0] <= 1 else __import__("math").factorial(nums[0])
            elif fam == "word_count": expected = len(inp.split())
            elif fam == "max": expected = max(nums)
            elif fam == "reverse": expected = inp[::-1]
            elif fam == "count_chars": expected = len(inp.strip())
            elif fam == "min": expected = min(nums)
            elif fam == "product":
                import math
                expected = math.prod(nums)
            elif fam == "abs_diff": expected = abs(nums[0] - nums[1])
            elif fam == "first_int": expected = nums[0]
            text = str(expected) + "\n"
            out.append({"id": f"{fam}-{i}", "prompt": desc, "helper": helper, "ret": ret,
                        "tests": [{"stdin": inp + "\n", "stdout": text}],
                        "python_tests": [{"stdin": inp + "\n", "stdout": text}]})
    return out


def main():
    e = env()
    if not e.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY missing")
    ts = tasks()
    results = []
    for n, t in enumerate(ts, 1):
        row = {"id": t["id"]}
        ok = False
        # AST arm
        system = ('Return only one JSON object {"expr": <expr>}. <expr> is a call tree '
                  '{"call": name, "args": [...]} or a literal. Use read_all() and the task helper.')
        user = f"{t['prompt']} Use {t['helper']}(read_all()) and return {t['ret']}."
        obj = None
        for attempt in range(1, 3):
            try:
                r = api(e, system, user, 256)
                obj = json.loads(strip_code(r["choices"][0]["message"]["content"]))
                allowed = {t["helper"], "read_all", "from_int", "print"}
                bad = [c for c in calls(obj["expr"]) if c not in allowed]
                if bad:
                    raise ValueError(f"call not allowed: {bad}")
                expr = lower(obj["expr"])
                body = f"print(from_int({expr}))" if t["ret"] == "Int" else f"print({expr})"
                src = PRELUDE + f"\nfn main(args: List[Str]) -> Int !io = {{ {body}; 0 }}\n"
                code, detail = run_qal(src, t["tests"][0]["stdin"])
                ok = code == 0 and detail == t["tests"][0]["stdout"]
                if ok: break
                user += "\nThe lowered QAL failed: " + str(detail)[:300] + "\nFix the AST."
            except Exception as ex:
                detail = str(ex)[:200]
                user += "\nInvalid AST: " + detail + "\nReturn only the allowed call tree."
        row["qal"] = {"ok": bool(ok), "detail": "" if ok else str(detail)[:200]}
        # Python arm
        user = t["prompt"] + "\nWrite a standalone Python 3 script that reads stdin."
        okp = False
        for attempt in range(1, 3):
            try:
                r = api(e, "Return only Python source.", user, 512)
                src = strip_code(r["choices"][0]["message"]["content"])
                codep, detailp = run_py(src, t["tests"][0]["stdin"])
                okp = codep == 0 and detailp == t["tests"][0]["stdout"]
                if okp: break
                user += "\nPrevious attempt failed: " + str(detailp)[:300]
            except Exception as ex:
                detailp = str(ex)[:200]
        row["python"] = {"ok": bool(okp)}
        results.append(row)
        if n % 10 == 0:
            print(f"{n}/100 qal={sum(r['qal']['ok'] for r in results)} python={sum(r['python']['ok'] for r in results)}", flush=True)
    out = HERE / "results-100.json"
    out.write_text(json.dumps({"results": results}, indent=2) + "\n")
    print(json.dumps({"qal": sum(r["qal"]["ok"] for r in results),
                      "python": sum(r["python"]["ok"] for r in results),
                      "tasks": len(results)}, indent=2))


if __name__ == "__main__":
    main()
