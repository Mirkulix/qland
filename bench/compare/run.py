#!/usr/bin/env python3
"""Pilot comparison: DeepSeek writes QAL vs Python for the same stdin tasks."""
import json, os, re, subprocess, sys, tempfile, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QAL = Path(os.environ.get("QAL_BIN", REPO / "qal" / "target" / "release" / "qal"))
SECRETS = Path.home() / ".config" / "qal" / "secrets.env"
REF = REPO / "qal" / "crates" / "qal-cli" / "src" / "reference.md"
QAL_EXAMPLE = (Path(__file__).parent / "example.qal").read_text(encoding="utf-8")


def env():
    e = dict(os.environ)
    if SECRETS.exists():
        for line in SECRETS.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                e.setdefault(k.strip(), v.strip().strip("'\""))
    return e


def strip_code(text):
    m = re.search(r"```(?:qal|python|py)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    return text.strip() + "\n"


def api(e, system, user, max_tokens=2048):
    base = e.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    body = json.dumps({"model": e.get("DEEPSEEK_MODEL", "deepseek-chat"),
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                       "temperature": 0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + e["DEEPSEEK_API_KEY"],
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read().decode())


def run_qal(source, stdin):
    with tempfile.TemporaryDirectory(prefix="qal-cmp-") as d:
        p = Path(d) / "main.qal"
        p.write_text(source, encoding="utf-8")
        c = subprocess.run([str(QAL), "check", str(p)], capture_output=True, text=True, timeout=30)
        if c.returncode != 0:
            return c.returncode, (c.stdout or "") + (c.stderr or "")
        r = subprocess.run([str(QAL), "run", str(p), "--allow", "io"], input=stdin,
                           capture_output=True, text=True, timeout=30)
        return r.returncode, (r.stdout or "") + (r.stderr or "")


def run_py(source, stdin):
    with tempfile.TemporaryDirectory(prefix="py-cmp-") as d:
        p = Path(d) / "main.py"
        p.write_text(source, encoding="utf-8")
        r = subprocess.run([sys.executable, str(p)], input=stdin,
                           capture_output=True, text=True, timeout=30)
        return r.returncode, (r.stdout or "") + (r.stderr or "")


def evaluate(runner, source, tests):
    for t in tests:
        code, out = runner(source, t["stdin"])
        if code != 0 or out != t["stdout"]:
            return False, out[-300:]
    return True, ""


def main():
    e = env()
    if not e.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY missing")
    tasks = json.loads((Path(__file__).parent / "tasks.json").read_text())
    ref = REF.read_text(encoding="utf-8")
    results = []
    for task in tasks:
        row = {"id": task["id"]}
        for arm in ("qal", "python"):
            if arm == "qal":
                system = ("You write QAL, not Python. Return only QAL source. "
                          "The first non-comment line must be `fn main(args: List[Str]) -> Int !io`. "
                          "Use read_all(), print(from_int(...)), match, get, to_int. "
                          "No Python syntax: no `def`, no `import`, no `&`, no single quotes.")
                user = ("Example of a complete QAL stdin program:\n" + QAL_EXAMPLE +
                        "\nTask:\n" + task["prompt"] +
                        "\nWrite the complete `fn main(args: List[Str]) -> Int !io`. Reference:\n" + ref)
                runner = run_qal
            else:
                system = "You write Python 3. Return only Python source."
                user = task["prompt"] + "\nWrite a standalone script."
                runner = run_py
            src, ok, detail, attempts = "", False, "", 0
            for attempt in range(1, 5):
                attempts = attempt
                try:
                    r = api(e, system, user, 2048)
                except Exception as ex:
                    detail = f"{type(ex).__name__}: {ex}"
                    break
                src = strip_code(r["choices"][0]["message"]["content"])
                ok, detail = evaluate(runner, src, task["tests"])
                if ok:
                    break
                user = task["prompt"] + "\nThe previous attempt failed. Fix it and return only source.\nFailure:\n" + detail
            row[arm] = {"ok": ok, "attempts": attempts, "detail": detail[:300]}
            print(task["id"], arm, "OK" if ok else "FAIL", flush=True)
        results.append(row)
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps({"results": results}, indent=2) + "\n")
    print(json.dumps({"qal": sum(1 for r in results if r["qal"]["ok"]),
                      "python": sum(1 for r in results if r["python"]["ok"]),
                      "tasks": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
