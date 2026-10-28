#!/usr/bin/env python3
"""100 tasks with stdlib only (no task-specific helper prelude)."""
import json, os, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import env, api, strip_code, run_qal, run_py  # noqa: E402
from scale import calls  # noqa: E402
from qal_lower import lower  # noqa: E402
from ast_fix import fix  # noqa: E402

ALLOWED = {"read_all", "print", "from_int", "len", "trim", "to_upper", "to_lower",
           "replace", "split", "contains", "starts_with", "slice"}

FAMILIES = {
    "len_trim": ("Int", "Print the length of the trimmed input.",
                 lambda s: len(s.strip()), ["abc", "  x  ", "hello world", "", "a b", "12345", "qal", " long ", "zz", "tab\t"]),
    "upper": ("Str", "Print the trimmed input in uppercase.", lambda s: s.strip().upper(),
              ["abc", "Qal", "hello", "x", "123", "a b", "ß", "text", "ZZ", "q"]),
    "lower": ("Str", "Print the trimmed input in lowercase.", lambda s: s.strip().lower(),
              ["ABC", "Qal", "HELLO", "X", "123", "A B", "TEXT", "zz", "Q", "a"]),
    "replace_space": ("Str", "Replace every space in the trimmed input with an underscore.",
                      lambda s: s.strip().replace(" ", "_"), ["a b", "one two three", "x", "a  b", "hello world", "qal", "a b c", "zz", "1 2", "last one"]),
    "slice02": ("Str", "Print the first two characters of the trimmed input.", lambda s: s.strip()[:2],
                ["abcd", "qal", "x", "hello", "12345", "ab c", "zz", "longer", "Q", "hi"]),
    "slice13": ("Str", "Print characters 1..2 (the second and third) of the trimmed input.", lambda s: s.strip()[1:3],
                ["abcd", "qal", "xyz", "hello", "12345", "abc", "zz", "longer", "Q", "hi"]),
    "contains_a": ("Str", "Print yes if the trimmed input contains 'a', else no.", lambda s: "yes" if "a" in s.strip() else "no",
                   ["abc", "xyz", "qal", "hello", "aaa", "b", "cat", "dog", "A", "a"]),
    "starts_a": ("Str", "Print yes if the trimmed input starts with 'a', else no.", lambda s: "yes" if s.strip().startswith("a") else "no",
                 ["abc", "xyz", "qal", "abc", "aaa", "b", "cat", "dog", "A", "a"]),
    "comma_count": ("Int", "Print the number of comma-separated fields in the trimmed input.",
                    lambda s: len(s.strip().split(",")), ["a,b", "a", "a,b,c", "", "x,y,z,w", "1,2", ",", "a,,b", "q", "last,one"]),
    "space_count": ("Int", "Print the number of space-separated fields in the trimmed input.",
                    lambda s: len(s.strip().split(" ")), ["a b", "one two three", "x", "a  b", "hello world", "qal", "a b c", "zz", "1 2", "last one"]),
}


def tasks():
    out = []
    for fam, (ret, desc, fn, inputs) in FAMILIES.items():
        for i, inp in enumerate(inputs):
            out.append({"id": f"{fam}-{i}", "prompt": desc, "ret": ret,
                        "tests": [{"stdin": inp + "\n", "stdout": str(fn(inp)) + "\n"}]})
    return out


def expr_for(fam):
    s = {"call": "trim", "args": [{"call": "read_all", "args": []}]}
    if fam == "len_trim": return {"call": "print", "args": [{"call": "from_int", "args": [{"call": "len", "args": [s]}]}]}
    if fam == "upper": return {"call": "print", "args": [{"call": "to_upper", "args": [s]}]}
    if fam == "lower": return {"call": "print", "args": [{"call": "to_lower", "args": [s]}]}
    if fam == "replace_space": return {"call": "print", "args": [{"call": "replace", "args": [s, {"str": " "}, {"str": "_"}]}]}
    if fam == "slice02": return {"call": "print", "args": [{"call": "slice", "args": [s, {"int": 0}, {"int": 2}]}]}
    if fam == "slice13": return {"call": "print", "args": [{"call": "slice", "args": [s, {"int": 1}, {"int": 3}]}]}
    if fam == "contains_a": return {"call": "print", "args": [{"if": {"call": "contains", "args": [s, {"str": "a"}]}, "then": {"str": "yes"}, "else": {"str": "no"}}]}
    if fam == "starts_a": return {"call": "print", "args": [{"if": {"call": "starts_with", "args": [s, {"str": "a"}]}, "then": {"str": "yes"}, "else": {"str": "no"}}]}
    if fam == "comma_count": return {"call": "print", "args": [{"call": "from_int", "args": [{"call": "len", "args": [{"call": "split", "args": [s, {"str": ","}]}]}]}]}
    if fam == "space_count": return {"call": "print", "args": [{"call": "from_int", "args": [{"call": "len", "args": [{"call": "split", "args": [s, {"str": " "}]}]}]}]}
    raise ValueError(fam)


AST_SCHEMA = {
    "type": "object",
    "properties": {"expr": {"$ref": "#/$defs/expr"}},
    "required": ["expr"], "additionalProperties": False,
    "$defs": {"expr": {"oneOf": [
        {"type": "object", "properties": {
            "call": {"type": "string", "enum": sorted(ALLOWED)},
            "args": {"type": "array", "items": {"$ref": "#/$defs/expr"}}},
         "required": ["call", "args"], "additionalProperties": False},
        {"type": "object", "properties": {
            "if": {"$ref": "#/$defs/expr"}, "then": {"$ref": "#/$defs/expr"},
            "else": {"$ref": "#/$defs/expr"}},
         "required": ["if", "then", "else"], "additionalProperties": False},
        {"type": "object", "properties": {"str": {"type": "string"}},
         "required": ["str"], "additionalProperties": False},
        {"type": "object", "properties": {"int": {"type": "integer"}},
         "required": ["int"], "additionalProperties": False},
        {"type": "object", "properties": {"list": {"type": "array", "items": {"$ref": "#/$defs/expr"}}},
         "required": ["list"], "additionalProperties": False},
    ]}},
}


def api_tool(e, user):
    base = e.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    schema = AST_SCHEMA
    body = json.dumps({"model": e.get("DEEPSEEK_MODEL", "deepseek-chat"),
                       "messages": [{"role": "user", "content": user}],
                       "tools": [{"type": "function", "function": {"name": "emit_ast",
                                  "description": "Emit the AST", "parameters": schema}}],
                       "tool_choice": {"type": "function", "function": {"name": "emit_ast"}},
                       "temperature": 0}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + e["DEEPSEEK_API_KEY"],
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def main():
    e = env()
    if not e.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY missing")
    ts = tasks()
    results = []
    for n, t in enumerate(ts, 1):
        fam = t["id"].rsplit("-", 1)[0]
        row = {"id": t["id"], "qal": {"ok": False}, "python": {"ok": False}}
        # AST arm
        user = (t["prompt"] + " Use stdlib calls only: read_all, print, from_int, len, trim, "
                "to_upper, to_lower, replace, split, contains, starts_with, slice. Return {expr}.")
        for attempt in range(1, 3):
            try:
                r = api_tool(e, user)
                obj = json.loads(r["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
                bad = [c for c in calls(obj["expr"]) if c not in ALLOWED]
                if bad:
                    raise ValueError(f"call not allowed: {bad}")
                expr = lower(fix(obj["expr"]))
                src = f"fn main(args: List[Str]) -> Int !io = {{ {expr}; 0 }}\n"
                code, out = run_qal(src, t["tests"][0]["stdin"])
                ok = code == 0 and out == t["tests"][0]["stdout"]
                row["qal"] = {"ok": ok, "detail": "" if ok else str(out)[:200]}
                if ok: break
                user += "\nPrevious attempt failed: " + str(out)[:300]
            except Exception as ex:
                row["qal"] = {"ok": False, "detail": str(ex)[:200]}
                user += "\nInvalid AST: " + str(ex)[:300]
        # Python arm
        user = t["prompt"] + "\nWrite a standalone Python 3 script that reads stdin."
        for attempt in range(1, 3):
            try:
                r = api(e, "Return only Python source.", user, 512)
                src = strip_code(r["choices"][0]["message"]["content"])
                code, out = run_py(src, t["tests"][0]["stdin"])
                okp = code == 0 and out == t["tests"][0]["stdout"]
                row["python"] = {"ok": okp}
                if okp: break
                user += "\nPrevious attempt failed: " + str(out)[:300]
            except Exception as ex:
                row["python"] = {"ok": False, "detail": str(ex)[:200]}
        results.append(row)
        if n % 10 == 0:
            print(f"{n}/100 qal={sum(r['qal']['ok'] for r in results)} python={sum(r['python']['ok'] for r in results)}", flush=True)
    out = HERE / "results-100-stdlib.json"
    out.write_text(json.dumps({"results": results}, indent=2) + "\n")
    print(json.dumps({"qal": sum(r["qal"]["ok"] for r in results),
                      "python": sum(r["python"]["ok"] for r in results),
                      "tasks": len(results)}, indent=2))


if __name__ == "__main__":
    main()
