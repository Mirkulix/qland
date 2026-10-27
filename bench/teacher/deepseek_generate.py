#!/usr/bin/env python3
"""Generate verified QAL solutions with the DeepSeek API (Phase 8 teacher).

The key is read from the environment or ~/.config/qal/secrets.env (never from the
repository):

  DEEPSEEK_API_KEY=sk-...
  DEEPSEEK_BASE_URL=https://api.deepseek.com
  DEEPSEEK_MODEL=deepseek-chat
  QAL_MODEL_BUDGET_EUR=10

  python3 bench/teacher/deepseek_generate.py --dry-run
  python3 bench/teacher/deepseek_generate.py --limit 5

Only solutions that pass `qal check` and `qal test` are written to
bench/teacher-api-<date>/solutions/; every response and its usage is logged.
"""
import argparse, datetime, json, os, re, sys, tempfile, time, urllib.error, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SECRETS = Path.home() / ".config" / "qal" / "secrets.env"
DEFAULT_BASE = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
QAL = REPO / "qal" / "target" / "release" / "qal"


def load_env():
    """Environment first, then the 600-mode secrets file; values are not logged."""
    env = dict(os.environ)
    if SECRETS.exists():
        for line in SECRETS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
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


def request(env, messages, max_tokens):
    base = env.get("DEEPSEEK_BASE_URL", DEFAULT_BASE).rstrip("/")
    body = json.dumps({"model": env.get("DEEPSEEK_MODEL", DEFAULT_MODEL), "messages": messages,
                       "temperature": 0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + env["DEEPSEEK_API_KEY"],
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read().decode())


def validate(task, source):
    """Run the task's hidden cases by defining the source in a temp store."""
    sys.path.insert(0, str(REPO / "bench"))
    import check
    with tempfile.NamedTemporaryFile("w", suffix=".qal", delete=False, encoding="utf-8") as f:
        f.write(source)
        path = f.name
    try:
        return check.check_qal(task, source=path)
    finally:
        os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N tasks (0: all)")
    ap.add_argument("--tasks", nargs="*", default=None, help="task ids; default all bench/tasks")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--attempts", type=int, default=4, help="generate/fix attempts per task")
    ap.add_argument("--out", default=None, help="output dir (default bench/teacher-api-<date>)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    env = load_env()
    if not env.get("DEEPSEEK_API_KEY"):
        print(json.dumps({"error": "DEEPSEEK_API_KEY missing",
                          "where": str(SECRETS),
                          "hint": "add DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, QAL_MODEL_BUDGET_EUR"},
                         indent=2))
        return 2
    budget = float(env.get("QAL_MODEL_BUDGET_EUR", "10"))
    in_rate = float(env.get("DEEPSEEK_INPUT_EUR_PER_MTOK", "0.30"))
    out_rate = float(env.get("DEEPSEEK_OUTPUT_EUR_PER_MTOK", "1.20"))
    tasks = args.tasks or sorted(p.name for p in (REPO / "bench" / "tasks").iterdir() if p.is_dir())
    if args.limit:
        tasks = tasks[:args.limit]
    out = Path(args.out) if args.out else REPO / "bench" / ("teacher-api-" + datetime.date.today().isoformat())
    ref = (REPO / "qal" / "crates" / "qal-cli" / "src" / "reference.md").read_text(encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"dry_run": True, "model": env.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
                          "base_url": env.get("DEEPSEEK_BASE_URL", DEFAULT_BASE),
                          "budget_eur": budget, "tasks": len(tasks), "out": str(out)}, indent=2))
        return 0

    (out / "solutions").mkdir(parents=True, exist_ok=True)
    results, spent = [], 0.0
    for task in tasks:
        statement = (REPO / "bench" / "tasks" / task / "task.md").read_text(encoding="utf-8")
        prompt = ("Write one complete QAL program for the task below. It must define "
                  "main(args: List[Str]) -> Int !io and use only this language reference. "
                  "Return only QAL source.\n\nTASK:\n" + statement + "\n\nREFERENCE:\n" + ref)
        messages = [{"role": "system", "content": "You write QAL programs."},
                    {"role": "user", "content": prompt}]
        text, v, attempts, cost = "", {"passed": 0, "total": 0, "error": "no attempt"}, 0, 0.0
        for attempt in range(1, max(1, args.attempts) + 1):
            attempts = attempt
            try:
                r = request(env, messages, args.max_tokens)
            except Exception as e:
                v = {"passed": 0, "total": 0, "error": f"{type(e).__name__}: {e}"}
                break
            usage = r.get("usage", {})
            cost += usage.get("prompt_tokens", 0) / 1e6 * in_rate + usage.get("completion_tokens", 0) / 1e6 * out_rate
            text = strip_code(r["choices"][0]["message"]["content"])
            try:
                v = validate(task, text)
            except Exception as e:
                v = {"passed": 0, "total": 0, "error": str(e)}
            if v.get("total") and v.get("passed") == v.get("total"):
                break
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": "The QAL define failed. Fix it and return only QAL source. Error:\n" + str(v.get("error"))[:3000]}]
        spent += cost
        ok = bool(v.get("total")) and v.get("passed") == v.get("total")
        if ok:
            (out / "solutions" / f"{task}.qal").write_text(text, encoding="utf-8")
        results.append({"task": task, "ok": ok, "passed": v.get("passed", 0), "total": v.get("total", 0),
                        "error": v.get("error"), "attempts": attempts, "cost_eur": round(cost, 6)})
        print(json.dumps({"task": task, "ok": ok, "passed": v.get("passed", 0), "total": v.get("total", 0),
                          "attempts": attempts, "spent_eur": round(spent, 4)}))
        if spent >= budget:
            print("budget exhausted", file=sys.stderr)
            break
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump({"model": env.get("DEEPSEEK_MODEL", DEFAULT_MODEL), "budget_eur": budget,
                   "spent_eur": round(spent, 6), "results": results}, f, indent=2)
    solved = sum(1 for r in results if r.get("ok"))
    print(json.dumps({"solved": solved, "tasks": len(results), "spent_eur": round(spent, 4), "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
