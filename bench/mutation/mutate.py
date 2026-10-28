#!/usr/bin/env python3
"""Mutation study (stage 3, docs/superpowers/specs/2026-10-25-qal-stufe3-mutationsstudie-design.md):
puts the same bug into the final code of a QAL line and a Rust line of the living program and
records whether each arm's own verification reports it, and whether the hidden tests fail.

Usage:
  python3 bench/mutation/mutate.py <qal-root> <rust-project> <out.json> [<mutant-id>...]

<qal-root> holds `.qal/store.db`, <rust-project> `Cargo.toml` and `src/main.rs`; both are copied,
never changed. Mutants are in `mutants.json` next to this script ("none" is the unmutated code).

Per mutant and arm:
  QAL  define:  the `qal_define` reply of the mutated definitions: refuted contract (E0801, nothing
                stored), failed stored tests, or a property warning (E0803/E0804)
       test:    a following `qal_test` of everything (stored tests and property tests)
  Rust test:    `cargo test` (debug build: arithmetic overflow panics, as do `assert!`s)
  both hidden:  the hidden tests of stage 12 (`bench/living/tasks-pressure/42-living-m`)
A mutant counts only where the hidden tests fail on it.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
os.environ.setdefault("BENCH_TASKS", os.path.join(BENCH, "living", "tasks-pressure"))
sys.path.insert(0, BENCH)
import check  # noqa: E402
import qalc  # noqa: E402

TASK = "42-living-m"
PROPERTY_WARNINGS = ("E0803", "E0804")


def apply(text, edits, where):
    for e in edits:
        if text.count(e["old"]) != 1:
            raise SystemExit(f"{where}: the text to replace occurs {text.count(e['old'])} times: {e['old'][:80]!r}")
        text = text.replace(e["old"], e["new"])
    return text


def qal_arm(root, mutant):
    with tempfile.TemporaryDirectory(prefix="mut-qal-") as tmp:
        shutil.copytree(os.path.join(root, ".qal"), os.path.join(tmp, ".qal"))
        c = qalc.Client(root=tmp)
        try:
            res = {}
            # What the unmutated code already reports (A3's contracts have unknowns, gave_ups and
            # traps on extreme inputs): only what is new after the mutation counts.
            _, _, t0 = c.call("qal_test", {})
            base_tests = {x["name"] for x in t0.get("tests") or [] if not x.get("passed")}
            base_props = {p["def"] for p in t0.get("properties") or [] if p["status"] == "failed"}
            base_warn = {(p["def"], "E0803") for p in t0.get("properties") or [] if p["status"] == "failed"} | {(p["def"], "E0804") for p in t0.get("properties") or [] if p.get("traps")}
            if mutant["qal"]:
                defs = sorted({e["def"] for e in mutant["qal"]})
                _, _, views = c.call("qal_view", {"names": defs})
                texts = {v["name"]: v["text"] for v in views["views"]}
                src = "\n\n".join(apply(texts[d], [e for e in mutant["qal"] if e["def"] == d], f"qal {mutant['id']} {d}") for d in defs)
                is_error, _, out = c.call("qal_define", {"source": src})
                diags = out.get("diagnostics") or []
                warnings = [w for w in out.get("warnings") or [] if w["code"] in PROPERTY_WARNINGS and (w["def"], w["code"]) not in base_warn]
                if is_error and not any(d["code"] == "E0801" for d in diags):
                    return {"invalid": [d["message"] for d in diags][:3]}
                res["define"] = {
                    "refuted": [d["message"] for d in diags if d["code"] == "E0801"],
                    "failed_tests": [t["name"] for t in out.get("tests") or [] if not t.get("passed") and t["name"] not in base_tests],
                    "ran_tests": len(out.get("tests") or []),
                    "property_warnings": [f"{w['code']} {w['def']}: {w['message']}" for w in warnings],
                }
                res["define"]["detected"] = bool(res["define"]["refuted"] or res["define"]["failed_tests"] or warnings)
                if res["define"]["refuted"]:
                    # Nothing was stored: the bug never reached the program.
                    res["test"] = {"detected": True, "note": "not stored (E0801)"}
                    res["hidden"] = {"failed": None, "note": "not stored"}
                    return res
            _, _, t = c.call("qal_test", {})
            failed = [x["name"] for x in t.get("tests") or [] if not x.get("passed") and x["name"] not in base_tests]
            props = [f"{p['def']} {p['status']}" for p in t.get("properties") or [] if p["status"] == "failed" and p["def"] not in base_props]
            res["test"] = {"failed_tests": failed, "failed_properties": props, "detected": bool(failed or props)}
        finally:
            c.close()
        hidden = check.check_qal(TASK, root=tmp)
        res["hidden"] = {"failed": len(hidden["failed"]), "total": hidden["total"], "error": hidden["error"]}
        return res


def rust_arm(project, mutant, target):
    with tempfile.TemporaryDirectory(prefix="mut-rust-") as tmp:
        src = os.path.join(tmp, "project")
        shutil.copytree(project, src, ignore=shutil.ignore_patterns("target", ".git"))
        path = os.path.join(src, "src", "main.rs")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        with open(path, "w", encoding="utf-8") as f:
            f.write(apply(text, mutant["rust"], f"rust {mutant['id']}"))
        env = dict(os.environ, CARGO_TARGET_DIR=target, CARGO_TERM_COLOR="never")
        r = subprocess.run(["cargo", "test", "--offline", "--quiet"], cwd=src, env=env, capture_output=True, text=True, timeout=600)
        if "could not compile" in r.stderr:
            return {"invalid": [r.stderr[-1500:]]}
        # `--quiet` lists the failed tests at the end, under the last `failures:`, indented.
        tail = r.stdout.rsplit("\nfailures:\n", 1)[-1] if "\nfailures:\n" in r.stdout else ""
        failures = [l.strip() for l in tail.splitlines() if l.startswith("    ")]
        res = {"test": {"detected": r.returncode != 0, "failed_tests": failures}}
        hidden = check.check_rust(TASK, src)
        res["hidden"] = {"failed": len(hidden["failed"]), "total": hidden["total"], "error": hidden["error"]}
        return res


def main(argv):
    if len(argv) < 4:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    qal_root, rust_project, out = (os.path.abspath(a) for a in argv[1:4])
    with open(os.path.join(HERE, "mutants.json"), encoding="utf-8") as f:
        mutants = [{"id": "none", "what": "the unmutated code", "qal": [], "rust": []}] + json.load(f)
    if argv[4:]:
        mutants = [m for m in mutants if m["id"] in argv[4:]]
    check.ensure_qal_binary()
    results = []
    with tempfile.TemporaryDirectory(prefix="mut-target-") as target:
        for m in mutants:
            r = {"id": m["id"], "what": m["what"], "qal": qal_arm(qal_root, m), "rust": rust_arm(rust_project, m, target)}
            results.append(r)
            q, s = r["qal"], r["rust"]
            print(m["id"], "| qal define", q.get("define", {}).get("detected"), "test", q.get("test", {}).get("detected"), "hidden", q.get("hidden", {}).get("failed"),
                  "| rust test", s.get("test", {}).get("detected"), "hidden", s.get("hidden", {}).get("failed"), "| invalid" if "invalid" in q or "invalid" in s else "", flush=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
