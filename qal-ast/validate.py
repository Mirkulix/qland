#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import jsonschema

SCHEMA = json.loads((Path(__file__).parent / "schema.json").read_text())
ap = argparse.ArgumentParser()
ap.add_argument("ast", nargs="+")
args = ap.parse_args()
ok = True
for f in args.ast:
    doc = json.loads(Path(f).read_text())
    errors = sorted(jsonschema.Draft202012Validator(SCHEMA).iter_errors(doc), key=lambda e: e.path)
    if errors:
        ok = False
        for e in errors:
            print(f"{f}: {list(e.path)}: {e.message}")
    else:
        print(f"{f}: ok")
sys.exit(0 if ok else 1)
