#!/usr/bin/env python3
"""Agent-native surface pilot: model emits a JSON expression AST, we lower to QAL."""
import json, os, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import env, api, run_qal, strip_code  # noqa: E402

TASKS = {
    "sum": ("Int", "sum_all"),
    "factorial": ("Int", "fact_first"),
    "word_count": ("Int", "count_words"),
    "max_list": ("Int", "max_all"),
    "reverse": ("Str", "reverse_line"),
}


def lower(e):
    if isinstance(e, int):
        return str(e)
    if isinstance(e, str):
        return json.dumps(e)
    if not isinstance(e, dict) or "call" not in e:
        raise ValueError(f"bad AST node: {e!r}")
    args = e.get("args", [])
    if not isinstance(args, list):
        raise ValueError("args must be a list")
    return f"{e['call']}({', '.join(lower(a) for a in args)})"


def main():
    e = env()
    if not e.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY missing")
    prelude = (HERE / "prelude.qal").read_text(encoding="utf-8")
    tasks = json.loads((HERE / "tasks.json").read_text())
    results = []
    for t in tasks:
        ret, fn = TASKS[t["id"]]
        system = ("You emit a JSON expression AST for QAL. Return only one JSON object "
                  '{"expr": <expr>} where <expr> is {"call": name, "args": [...]} or a literal. '
                  "Allowed calls: read_all, print, from_int, sum_all, fact_first, count_words, "
                  "max_all, reverse_line. No prose, no markdown.")
        user = (f"Task: {t['prompt']}\n"
                f"Use read_all() to get stdin. The expression must return {ret}. "
                f"Available task helper: {fn}(s: Str).")
        row = {"id": t["id"], "ok": False, "attempts": 0, "detail": ""}
        for attempt in range(1, 5):
            row["attempts"] = attempt
            try:
                r = api(e, system, user, 512)
            except Exception as ex:
                row["detail"] = f"{type(ex).__name__}: {ex}"
                break
            text = strip_code(r["choices"][0]["message"]["content"])
            try:
                obj = json.loads(text[text.index("{"):])
                expr = lower(obj["expr"])
                body = f"print(from_int({expr}))" if ret == "Int" else f"print({expr})"
                source = prelude + f"\nfn main(args: List[Str]) -> Int !io = {{ {body}; 0 }}\n"
            except Exception as ex:
                row["detail"] = f"AST/lower: {ex}"
                user += f"\nPrevious output was invalid: {ex}. Return only JSON."
                continue
            ok, detail = run_qal(source, t["tests"][0]["stdin"])
            # run all tests
            if ok:
                for case in t["tests"][1:]:
                    ok, detail = run_qal(source, case["stdin"])
                    if not ok:
                        break
            row["ok"] = ok
            row["detail"] = "" if ok else detail[-300:]
            print(t["id"], "OK" if ok else "FAIL", flush=True)
            if ok:
                break
            user += f"\nThe lowered QAL failed:\n{detail[-400:]}\nFix the AST."
        results.append(row)
    out = HERE / "results-ast.json"
    out.write_text(json.dumps({"results": results}, indent=2) + "\n")
    print(json.dumps({"qal_ast": sum(1 for r in results if r["ok"]), "tasks": len(results)}, indent=2))


if __name__ == "__main__":
    main()
