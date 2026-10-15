#!/usr/bin/env python3
"""Verifies a benchmark solution against the hidden tests of its task (bench/tasks/<id>/tests.json).

  python3 bench/check.py qal <task> --root <dir>     # the QAL store of an agent's project root
  python3 bench/check.py qal <task> --source <file>  # QAL source, defined into a fresh project root
  python3 bench/check.py rust <task> <cargo-dir>     # a cargo project (built with --release --offline)
  python3 bench/check.py refs [<task>...]            # every reference solution in both arms
Add --json for one JSON object per check on stdout.

Both arms run the same cases: arguments, standard input, and for file tasks the files created
below a fresh directory (the QAL project root, or the Rust program's working directory). A case
passes if the exit code and stdout are exactly equal (every printed line ends with a newline). Each
run has a 10 s wall-clock limit; the QAL arm runs `main` via the MCP tool `qal_run` with
allow ["io"] (plus "fs:data" for file tasks), exactly as an agent can.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

BENCH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BENCH)
sys.path.insert(0, BENCH)
import qalc  # noqa: E402

TIMEOUT = 10
MAX_OUTPUT = 64 << 10

def task_ids():
    return sorted(d for d in os.listdir(os.path.join(BENCH, "tasks")) if re.fullmatch(r"\d\d-[a-z-]+", d))

def load(task):
    path = os.path.join(BENCH, "tasks", task, "tests.json")
    if not os.path.exists(path):
        raise SystemExit(f"unknown task `{task}` (tasks: {', '.join(task_ids())})")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def write_files(base, files):
    for rel, content in files.items():
        path = os.path.join(base, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(content)

def ensure_qal_binary():
    """Builds the release server unless $QAL_BIN names another one (a no-op when up to date)."""
    if os.environ.get("QAL_BIN"):
        return
    env = {k: v for k, v in os.environ.items() if k != "CARGO_TARGET_DIR"}
    r = subprocess.run(["cargo", "build", "--release", "--quiet", "-p", "qal-cli"], cwd=os.path.join(REPO, "qal"), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("building the QAL server failed:\n" + r.stderr[-3000:])

def result(task, arm, cases, failed, error=None):
    return {"task": task, "arm": arm, "passed": len(cases) - len(failed) if error is None else 0, "total": len(cases),
            "failed": failed, "error": error}

def check_qal(task, root=None, source=None):
    spec = load(task)
    cases = spec["cases"]
    fs = "fs" in spec["effects"]
    with tempfile.TemporaryDirectory(prefix="qal-check-") as tmp:
        if root is not None:
            store = os.path.join(root, ".qal", "store.db")
            if not os.path.exists(store):
                return result(task, "qal", cases, [], f"no QAL store in {root}")
            os.makedirs(os.path.join(tmp, ".qal"))
            # The store runs in WAL mode: copy the log too, in case a server did not checkpoint it.
            for suffix in ("", "-wal", "-shm"):
                if os.path.exists(store + suffix):
                    shutil.copy(store + suffix, os.path.join(tmp, ".qal", "store.db" + suffix))
        c = qalc.Client(root=tmp)
        try:
            if source is not None:
                with open(source, encoding="utf-8") as f:
                    is_error, text, _ = c.call("qal_define", {"source": f.read()})
                if is_error:
                    return result(task, "qal", cases, [], "qal_define failed: " + text[:2000])
            failed = []
            for case in cases:
                data = os.path.join(tmp, "data")
                shutil.rmtree(data, ignore_errors=True)
                if fs:
                    os.makedirs(data)
                write_files(tmp, case["files"])
                allow = ["io", "fs:data"] if fs else ["io"]
                _, text, out = c.call("qal_run", {"entry": "main", "args": case["args"], "stdin": case["stdin"], "allow": allow})
                ok = (isinstance(out, dict) and "error" not in out and "diagnostics" not in out and "stdout_truncated" not in out
                      and out.get("exit_code") == case["exit_code"] and out.get("stdout") == case["stdout"])
                if not ok:
                    failed.append(case["name"])
            return result(task, "qal", cases, failed)
        finally:
            c.close()

def check_rust(task, project):
    spec = load(task)
    cases = spec["cases"]
    with tempfile.TemporaryDirectory(prefix="rust-check-") as tmp:
        src = os.path.join(tmp, "project")
        shutil.copytree(project, src, ignore=shutil.ignore_patterns("target", ".git"))
        env = dict(os.environ, CARGO_TARGET_DIR=os.path.join(tmp, "target"), CARGO_TERM_COLOR="never")
        r = subprocess.run(["cargo", "build", "--release", "--offline", "--quiet"], cwd=src, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            return result(task, "rust", cases, [], "cargo build failed: " + r.stderr[-2000:])
        exes = [p for p in os.listdir(os.path.join(tmp, "target", "release")) if os.access(os.path.join(tmp, "target", "release", p), os.X_OK)
                and os.path.isfile(os.path.join(tmp, "target", "release", p))]
        if len(exes) != 1:
            return result(task, "rust", cases, [], f"expected one binary, found {exes}")
        exe = os.path.join(tmp, "target", "release", exes[0])
        failed = []
        for case in cases:
            cwd = os.path.join(tmp, "run")
            shutil.rmtree(cwd, ignore_errors=True)
            os.makedirs(cwd)
            write_files(cwd, case["files"])
            try:
                p = subprocess.run([exe] + case["args"], cwd=cwd, input=case["stdin"].encode(), capture_output=True, timeout=TIMEOUT)
                stdout = p.stdout[:MAX_OUTPUT + 1].decode("utf-8", errors="replace")
                ok = len(p.stdout) <= MAX_OUTPUT and p.returncode == case["exit_code"] and stdout == case["stdout"]
            except subprocess.TimeoutExpired:
                ok = False
            if not ok:
                failed.append(case["name"])
        return result(task, "rust", cases, failed)

def show(res, as_json):
    if as_json:
        print(json.dumps(res))
        return
    status = "PASS" if res["error"] is None and not res["failed"] else "FAIL"
    detail = f" ({res['error']})" if res["error"] else (f": {', '.join(res['failed'])}" if res["failed"] else "")
    print(f"{status} {res['arm']} {res['task']} {res['passed']}/{res['total']}{detail}")

def main(argv):
    as_json = "--json" in argv
    argv = [a for a in argv if a != "--json"]
    if len(argv) >= 2 and argv[1] == "refs":
        ensure_qal_binary()
        ok = True
        for task in argv[2:] or task_ids():
            for res in (check_qal(task, source=os.path.join(BENCH, "solutions", "qal", f"{task}.qal")),
                        check_rust(task, os.path.join(BENCH, "solutions", "rust", task))):
                show(res, as_json)
                ok = ok and res["error"] is None and not res["failed"]
        return 0 if ok else 1
    if len(argv) == 5 and argv[1] == "qal" and argv[3] in ("--root", "--source"):
        ensure_qal_binary()
        res = check_qal(argv[2], **{argv[3][2:]: os.path.abspath(argv[4])})
    elif len(argv) == 4 and argv[1] == "rust":
        res = check_rust(argv[2], os.path.abspath(argv[3]))
    else:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    show(res, as_json)
    return 0 if res["error"] is None and not res["failed"] else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv))
