#!/usr/bin/env python3
"""Rebuilds qal/selfhost/qalc.qal as the concatenation of the modules in selfhost/src, in bundle order.

Usage: python3 qal/selfhost/tools/bundle.py [--check]   (--check: exit 1 if qalc.qal is not up to date)
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODS = ["qc_ast", "qc_lex", "qc_parse", "qc_print", "qc_canon", "qc_tform", "qc_prelude", "qc_builtin", "qc_unify", "qc_exhaust", "qc_sql", "qc_mono", "qc_check", "qc_enc", "qc_rt", "qc_wasm", "qc_proto"]


def main(argv):
    out = "".join(open(os.path.join(HERE, "src", m + ".qal"), encoding="utf-8").read() for m in MODS)
    path = os.path.join(HERE, "qalc.qal")
    if "--check" in argv:
        same = open(path, encoding="utf-8").read() == out
        print("qalc.qal is up to date" if same else "qalc.qal differs from selfhost/src: run tools/bundle.py")
        return 0 if same else 1
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"qalc.qal: {len(out)} bytes, {out.count(chr(10))} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
