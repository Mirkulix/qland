#!/usr/bin/env python3
"""Lower a JSON AST to QAL source (agent-native surface prototype).

Supported nodes:
  literal: {"int": n}, {"str": s}, {"bool": true}
  var, call, op, field
  let: {"let":[{"name":n,"value":e}], "in": e}
  if:  {"if": c, "then": a, "else": b}
  match: {"match": e, "cases":[{"pattern": p, "body": e}]}
  record: {"record":"Type", "fields": {name: e}}
"""
import argparse, json, subprocess, sys
from pathlib import Path

QAL = Path(__file__).resolve().parents[1] / "qal" / "target" / "release" / "qal"


def lower(e):
    if isinstance(e, bool):
        return "true" if e else "false"
    if isinstance(e, int):
        return str(e)
    if isinstance(e, str):
        return json.dumps(e)
    if not isinstance(e, dict):
        raise ValueError(f"bad node: {e!r}")
    if "int" in e:
        return str(e["int"])
    if "str" in e:
        return json.dumps(e["str"])
    if "bool" in e:
        return "true" if e["bool"] else "false"
    if "var" in e:
        return e["var"]
    if "call" in e:
        return f"{e['call']}({', '.join(lower(a) for a in e.get('args', []))})"
    if "op" in e:
        return "(" + f" {e['op']} ".join(lower(a) for a in e.get("args", [])) + ")"
    if "field" in e:
        return f"{lower(e['field'])}.{e['name']}"
    if "let" in e:
        binds = e["let"]
        out = lower(e["in"])
        for b in reversed(binds):
            out = f"let {b['name']} = {lower(b['value'])} in {out}"
        return out
    if "if" in e:
        return f"(if {lower(e['if'])} then {lower(e['then'])} else {lower(e['else'])})"
    if "match" in e:
        cases = ", ".join(f"{c['pattern']} => {lower(c['body'])}" for c in e["cases"])
        return f"(match {lower(e['match'])} {{ {cases} }})"
    if "record" in e:
        fields = ", ".join(f"{k}: {lower(v)}" for k, v in e["fields"].items())
        return f"{e['record']} {{ {fields} }}"
    raise ValueError(f"unknown node keys: {sorted(e)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ast")
    ap.add_argument("--prelude", default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    ast = json.loads(Path(args.ast).read_text(encoding="utf-8"))
    expr = lower(ast["main"])
    prelude = Path(args.prelude).read_text(encoding="utf-8") + "\n" if args.prelude else ""
    source = prelude + f"fn main(args: List[Str]) -> Int !io = {{ {expr}; 0 }}\n"
    print(source)
    if args.check:
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".qal", delete=False, encoding="utf-8") as f:
            f.write(source); path = f.name
        try:
            r = subprocess.run([str(QAL), "check", path], capture_output=True, text=True)
            print(r.stdout or r.stderr, file=sys.stderr)
            return r.returncode
        finally:
            Path(path).unlink()


if __name__ == "__main__":
    raise SystemExit(main())
