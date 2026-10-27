#!/usr/bin/env python3
"""DeepSeek tool-using teacher for QAL (Phase 8).

Unlike deepseek_generate.py (prompt -> code -> one error), this runs a real
tool loop: define, check, test, run. Only a solution that passes the task's
hidden cases is kept.

  python3 bench/teacher/deepseek_agent.py --limit 1 --attempts 12
  python3 bench/teacher/deepseek_agent.py --out bench/teacher-agent-2026-10-27

Key/config: DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
QAL_MODEL_BUDGET_EUR in the environment or ~/.config/qal/secrets.env.
"""
import argparse, datetime, json, os, re, shutil, subprocess, sys, tempfile, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SECRETS = Path.home() / ".config" / "qal" / "secrets.env"
QAL = REPO / "qal" / "target" / "release" / "qal"
DEFAULT_BASE = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


def load_env():
    env = dict(os.environ)
    if SECRETS.exists():
        for line in SECRETS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip("'\""))
    return env


def strip_code(text):
    m = re.search(r"```(?:qal)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*(fn|type|test)\s+[A-Za-z_]", line):
            return "\n".join(lines[i:]).strip() + "\n"
    return text.strip() + "\n"


def api(env, messages, tools, max_tokens=4096):
    base = env.get("DEEPSEEK_BASE_URL", DEFAULT_BASE).rstrip("/")
    body = json.dumps({"model": env.get("DEEPSEEK_MODEL", DEFAULT_MODEL), "messages": messages,
                       "tools": tools, "tool_choice": "auto", "temperature": 0,
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + env["DEEPSEEK_API_KEY"],
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


def qal_tools():
    src = {"type": "object", "properties": {"source": {"type": "string"}}, "required": ["source"]}
    return [
        {"type": "function", "function": {"name": "qal_define", "description": "Define QAL source in the project store; returns diagnostics/proofs/tests.",
         "parameters": src}},
        {"type": "function", "function": {"name": "qal_check", "description": "Type-check QAL source without storing it.",
         "parameters": src}},
        {"type": "function", "function": {"name": "qal_test", "description": "Run the tests of a stored definition or file.",
         "parameters": {"type": "object", "properties": {"name": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "qal_run", "description": "Run main with args and stdin (io is granted; fs:data for file tasks).",
         "parameters": {"type": "object", "properties": {"args": {"type": "array", "items": {"type": "string"}}, "stdin": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "submit", "description": "Submit the final QAL source. Call once, after the tests/examples pass.",
         "parameters": src}},
    ]


def init_work(task):
    wd = Path("/tmp/qal-agent") / task
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True)
    subprocess.run([str(QAL), "init"], cwd=wd, capture_output=True, text=True, timeout=60)
    return wd


def run_qal(wd, task, tool, args):
    source = args.get("source")
    if source is not None:
        (wd / "solution.qal").write_text(source, encoding="utf-8")
    allow = "io,fs:data"
    cmd = {"qal_define": [str(QAL), "define", "solution.qal"],
           "qal_check": [str(QAL), "check", "solution.qal"],
           "qal_test": [str(QAL), "test", "solution.qal" if source is not None else args.get("name", "")],
           "qal_run": [str(QAL), "run", "solution.qal", "--allow", allow, "--"] + list(args.get("args") or [])}[tool]
    r = subprocess.run(cmd, cwd=wd, input=args.get("stdin", ""), capture_output=True, text=True, timeout=60)
    return ((r.stdout or "") + (r.stderr or ""))[:5000] or f"exit {r.returncode}"


def validate(task, source):
    sys.path.insert(0, str(REPO / "bench"))
    import check
    with tempfile.NamedTemporaryFile("w", suffix=".qal", delete=False, encoding="utf-8") as f:
        f.write(source); path = f.name
    try:
        return check.check_qal(task, source=path)
    finally:
        os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--attempts", type=int, default=12, help="API rounds per task")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    env = load_env()
    if not env.get("DEEPSEEK_API_KEY"):
        print(json.dumps({"error": "DEEPSEEK_API_KEY missing", "where": str(SECRETS)}, indent=2))
        return 2
    budget = float(env.get("QAL_MODEL_BUDGET_EUR", "10"))
    in_rate = float(env.get("DEEPSEEK_INPUT_EUR_PER_MTOK", "0.30"))
    out_rate = float(env.get("DEEPSEEK_OUTPUT_EUR_PER_MTOK", "1.20"))
    tasks = args.tasks or sorted(p.name for p in (REPO / "bench" / "tasks").iterdir() if p.is_dir())
    if args.limit:
        tasks = tasks[:args.limit]
    out = Path(args.out) if args.out else REPO / "bench" / ("teacher-agent-" + datetime.date.today().isoformat())
    (out / "solutions").mkdir(parents=True, exist_ok=True)
    ref = (REPO / "qal" / "crates" / "qal-cli" / "src" / "reference.md").read_text(encoding="utf-8")
    tools = qal_tools()
    results, spent = [], 0.0
    for task in tasks:
        statement = (REPO / "bench" / "tasks" / task / "task.md").read_text(encoding="utf-8")
        wd = init_work(task)
        messages = [{"role": "system", "content": "You write QAL. Use the tools; check and test before submit. When it passes, call submit with the complete source. Do not end without submit."},
                    {"role": "user", "content": "Solve this task. QAL reference:\n" + ref + "\n\nTASK:\n" + statement}]
        final, calls, cost = None, 0, 0.0
        for rnd in range(1, max(1, args.attempts) + 1):
            try:
                r = api(env, messages, tools, args.max_tokens)
            except Exception as e:
                messages.append({"role": "user", "content": "API error: " + str(e)[:300]})
                break
            u = r.get("usage", {})
            cost += u.get("prompt_tokens", 0) / 1e6 * in_rate + u.get("completion_tokens", 0) / 1e6 * out_rate
            spent += u.get("prompt_tokens", 0) / 1e6 * in_rate + u.get("completion_tokens", 0) / 1e6 * out_rate
            msg = r["choices"][0]["message"]
            messages.append(msg)
            tcs = msg.get("tool_calls") or []
            print("round", rnd, "tools", [t["function"]["name"] for t in tcs], "content", (msg.get("content") or "")[:80].replace("\n"," "), file=sys.stderr)
            if not tcs:
                final = strip_code(msg.get("content") or "")
                break
            for tc in tcs:
                name = tc["function"]["name"]
                try:
                    a = json.loads(tc["function"].get("arguments") or "{}")
                except Exception:
                    a = {}
                calls += 1
                if name == "submit":
                    final = strip_code(a.get("source", ""))
                res = run_qal(wd, task, name, a) if name != "submit" else "submitted"
                # A passing hidden-case check is the real success; do not rely on the model
                # remembering to call submit.
                if name in ("qal_define", "qal_check") and a.get("source"):
                    try:
                        hv = validate(task, a["source"])
                        if hv.get("total") and hv.get("passed") == hv.get("total"):
                            final = strip_code(a["source"])
                            res += "\nHIDDEN: PASS"
                        else:
                            res += f"\nHIDDEN: {hv.get('passed',0)}/{hv.get('total',0)} {str(hv.get('error') or '')[:800]}"
                    except Exception as e:
                        res += "\nHIDDEN: error " + str(e)[:300]
                print("  result", res[:200].replace("\n"," "), file=sys.stderr)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": res})
            if final:
                break
            if spent >= budget:
                break
        ok, v, err = False, {"passed": 0, "total": 0}, None
        if final:
            try:
                v = validate(task, final)
            except Exception as e:
                err = str(e)
            ok = bool(v.get("total")) and v.get("passed") == v.get("total")
            if ok:
                (out / "solutions" / f"{task}.qal").write_text(final, encoding="utf-8")
        results.append({"task": task, "ok": ok, "passed": v.get("passed", 0), "total": v.get("total", 0),
                        "calls": calls, "cost_eur": round(cost, 6), "error": err or v.get("error")})
        print(json.dumps({"task": task, "ok": ok, "passed": v.get("passed", 0), "total": v.get("total", 0),
                          "calls": calls, "spent_eur": round(spent, 4)}))
        if spent >= budget:
            print("budget exhausted", file=sys.stderr)
            break
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump({"model": env.get("DEEPSEEK_MODEL", DEFAULT_MODEL), "budget_eur": budget,
                   "spent_eur": round(spent, 6), "results": results}, f, indent=2)
    print(json.dumps({"solved": sum(1 for r in results if r["ok"]), "tasks": len(results),
                      "spent_eur": round(spent, 4), "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
