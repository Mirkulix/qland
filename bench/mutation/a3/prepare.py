#!/usr/bin/env python3
"""Builds the code of measurement A3 from the code of A1: thin tests plus the same invariants.

Usage: python3 bench/mutation/a3/prepare.py <qal-root> <rust-project> <out-dir>

QAL: copies the store, deletes every stored test, then defines `qal_contracts.qal` (invariant
helpers, `req`/`ens` on the functions that write a customer, order or product record, bodies
unchanged, and the two public examples of stage 12 as the only tests).
Rust: copies the project, applies `rust_edits.json` (the same invariants as `assert!`s; what the
unsigned types already enforce is left to them) and replaces the test module with the same two
examples. Writes <out-dir>/qal and <out-dir>/rust.
"""

import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
QAL = os.environ.get("QAL_BIN", os.path.join(REPO, "qal", "target", "release", "qal"))


def qal(cwd, *args):
    r = subprocess.run([QAL, *args], cwd=cwd, capture_output=True, text=True)
    return json.loads(r.stdout)


def main(argv):
    if len(argv) != 4:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    qal_root, rust_project, out = (os.path.abspath(a) for a in argv[1:])
    q, r = os.path.join(out, "qal"), os.path.join(out, "rust")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(q)
    shutil.copytree(os.path.join(qal_root, ".qal"), os.path.join(q, ".qal"))
    tests = [e["name"] for e in qal(q, "query", "--outline")["results"] if e["kind"] == "test"]
    deleted = qal(q, "delete", *tests)
    assert len(deleted.get("deleted", [])) == len(tests), deleted
    defined = qal(q, "define", os.path.join(HERE, "qal_contracts.qal"))
    assert "diagnostics" not in defined and defined.get("failed") == 0, defined
    shutil.copytree(rust_project, r, ignore=shutil.ignore_patterns("target", ".git"))
    with open(os.path.join(HERE, "rust_edits.json"), encoding="utf-8") as f:
        edits = json.load(f)
    path = os.path.join(r, "src", "main.rs")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for old, new in edits["replace"]:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    text = text[: text.index("#[cfg(test)]\nmod tests {")] + edits["tests"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"deleted {len(tests)} QAL tests; {out}/qal and {out}/rust ready")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
