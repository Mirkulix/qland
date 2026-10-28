#!/usr/bin/env python3
import json, os, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lower import lower

REPO = Path(__file__).resolve().parents[1]
QAL = Path(os.environ.get("QAL_BIN", REPO / "qal" / "target" / "release" / "qal"))


def num(expr):
    return {"call": "print", "args": [{"call": "from_int", "args": [expr]}]}


CASES = [
    ("let/op", "", {"let": [{"name": "x", "value": {"int": 2}}],
                    "in": num({"op": "+", "args": [{"var": "x"}, {"int": 3}]})}, "", "5\n"),
    ("if", "", num({"if": {"op": "<", "args": [{"int": 1}, {"int": 2}]},
                        "then": {"int": 7}, "else": {"int": 9}}), "", "7\n"),
    ("match", "", num({"match": {"call": "to_int", "args": [{"str": "42"}]},
                       "cases": [{"pattern": "None", "body": {"int": 0}},
                                 {"pattern": "Some(n)", "body": {"var": "n"}}]}), "", "42\n"),
    ("record", "type P = { x: Int }\n",
     num({"field": {"record": "P", "fields": {"x": {"int": 8}}}, "name": "x"}), "", "8\n"),
]


def run(src, stdin):
    with tempfile.NamedTemporaryFile("w", suffix=".qal", delete=False, encoding="utf-8") as f:
        f.write(src); path = f.name
    try:
        r = subprocess.run([str(QAL), "run", path, "--allow", "io"], input=stdin,
                           capture_output=True, text=True, timeout=30)
        out = r.stdout
        try:
            out = json.loads(out).get("stdout", out)
        except Exception:
            pass
        return r.returncode, out, r.stderr
    finally:
        Path(path).unlink()


def main():
    results = []
    for name, prelude, ast, stdin, want in CASES:
        src = prelude + f"fn main(args: List[Str]) -> Int !io = {{ {lower(ast)}; 0 }}\n"
        code, out, err = run(src, stdin)
        results.append({"case": name, "ok": code == 0 and out == want,
                        "out": out, "want": want, "err": (err or "")[-200:]})
    print(json.dumps({"ok": all(r["ok"] for r in results), "results": results}, indent=2))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
