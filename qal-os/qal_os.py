#!/usr/bin/env python3
"""QAL-OS MVP: a multi-agent supervisor, not a kernel.

This is the first runnable step toward the agent-OS design: one tenant, N
isolated project stores, a work queue with per-agent quotas, timeouts and an
audit view. It deliberately does NOT claim OS-level security: all agents run as
the same OS user. See README.md.

  python3 qal-os/qal_os.py up --tenant demo --agents 4
  python3 qal-os/qal_os.py run --tenant demo --tasks tasks.json --workers 4
  python3 qal-os/qal_os.py ps --tenant demo
  python3 qal-os/qal_os.py audit --tenant demo
"""
import argparse, concurrent.futures, json, os, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QAL = Path(os.environ.get("QAL_BIN", REPO / "qal" / "target" / "release" / "qal"))
HOME = Path(os.environ.get("QAL_OS_HOME", Path.home() / ".qal-os"))
TIMEOUT = int(os.environ.get("QAL_OS_TIMEOUT", "30"))
SANDBOX = "/usr/bin/sandbox-exec"
QUOTA = int(os.environ.get("QAL_OS_QUOTA", "8"))


def tenant_dir(tenant):
    return HOME / "tenants" / tenant


def run(cmd, cwd, timeout=TIMEOUT):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def meta(tenant):
    p = tenant_dir(tenant) / "meta.json"
    if not p.exists():
        sys.exit(f"tenant {tenant!r} is not up (run: up --tenant {tenant})")
    return json.loads(p.read_text())


def write_profiles(tenant, agents):
    """macOS: deny every other agent's directory and the network; allow the rest."""
    if not Path(SANDBOX).exists():
        return False
    t = tenant_dir(tenant)
    (t / "profiles").mkdir(exist_ok=True)
    for a in agents:
        deny = "".join(f'(deny file-read* (subpath "{os.path.realpath(o["dir"])}"))\n(deny file-write* (subpath "{os.path.realpath(o["dir"])}"))\n'
                       for o in agents if o["id"] != a["id"])
        text = f'(version 1)\n(allow default)\n(deny network*)\n{deny}'
        (t / "profiles" / (a["id"] + ".sb")).write_text(text)
    return True


def sandboxed(args, agent_id):
    p = tenant_dir(args.tenant) / "profiles" / (agent_id + ".sb")
    return [SANDBOX, "-f", str(p)] if p.exists() else []


def cmd_up(args):
    t = tenant_dir(args.tenant)
    if t.exists() and not args.force:
        sys.exit(f"{t} exists; use --force to reinitialize")
    if t.exists():
        shutil.rmtree(t)
    (t / "agents").mkdir(parents=True)
    agents = []
    for i in range(args.agents):
        d = t / "agents" / f"a{i}"
        d.mkdir()
        code, out = run([str(QAL), "init"], d)
        if code != 0:
            sys.exit(f"agent a{i} init failed: {out[:500]}")
        agents.append({"id": f"a{i}", "dir": str(d), "tasks": 0, "passed": 0, "failed": 0})
    sandbox = write_profiles(args.tenant, agents) if args.sandbox else False
    (t / "meta.json").write_text(json.dumps({"tenant": args.tenant, "agents": agents, "sandbox": sandbox,
                                              "created": time.time(), "qal": str(QAL)}, indent=2) + "\n")
    print(json.dumps({"tenant": args.tenant, "agents": len(agents), "sandbox": sandbox, "root": str(t)}, indent=2))


def one_task(tenant, task):
    m = meta(tenant)
    aid = task["agent"]
    agent = next((a for a in m["agents"] if a["id"] == aid), None)
    if agent is None:
        return {"agent": aid, "ok": False, "error": "unknown agent"}
    if agent["tasks"] >= QUOTA:
        return {"agent": aid, "ok": False, "error": "quota"}
    d = Path(agent["dir"])
    src = d / "tasks" / f"task-{agent['tasks']:04d}.qal"
    src.parent.mkdir(exist_ok=True)
    src.write_text(task["source"], encoding="utf-8")
    pre = sandboxed(argparse.Namespace(tenant=tenant), aid)
    code, out = run(pre + [str(QAL), "define", src.name], src.parent)
    if code != 0:
        return {"agent": aid, "ok": False, "error": "define", "detail": out[:2000]}
    code, out = run(pre + [str(QAL), "test", src.name], src.parent)
    return {"agent": aid, "ok": code == 0, "error": None if code == 0 else "test", "detail": out[:2000]}


def cmd_run(args):
    m = meta(args.tenant)
    tasks = json.loads(Path(args.tasks).read_text())
    if not isinstance(tasks, list):
        sys.exit("tasks must be a JSON list of {agent, source}")
    t0 = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_task, args.tenant, t): t for t in tasks}
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())
    # update counters
    by = {a["id"]: a for a in m["agents"]}
    for r in results:
        a = by.get(r["agent"])
        if a:
            a["tasks"] += 1
            a["passed" if r["ok"] else "failed"] += 1
    (tenant_dir(args.tenant) / "meta.json").write_text(json.dumps(m, indent=2) + "\n")
    with open(tenant_dir(args.tenant) / "results.jsonl", "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    summary = {"tasks": len(results), "ok": sum(1 for r in results if r["ok"]),
               "failed": sum(1 for r in results if not r["ok"]),
               "seconds": round(time.time() - t0, 2)}
    print(json.dumps({"summary": summary, "results": results}, indent=2)[:12000])
    sys.exit(0 if summary["failed"] == 0 else 1)


def cmd_ps(args):
    m = meta(args.tenant)
    for a in m["agents"]:
        print(f"{a['id']:4} tasks={a['tasks']:2} passed={a['passed']:2} failed={a['failed']:2} {a['dir']}")


def cmd_audit(args):
    m = meta(args.tenant)
    out = []
    for a in m["agents"]:
        code, text = run([str(QAL), "query"], Path(a["dir"]))
        try:
            defs = len(json.loads(text).get("results", []))
        except Exception:
            defs = -1
        out.append({"agent": a["id"], "definitions": defs, "tasks": a["tasks"],
                    "passed": a["passed"], "failed": a["failed"], "query_ok": code == 0})
    print(json.dumps({"tenant": args.tenant, "agents": out}, indent=2))


def cmd_isolation(args):
    m = meta(args.tenant)
    if not m.get("sandbox"):
        sys.exit("tenant was not started with --sandbox")
    results = []
    for a in m["agents"]:
        others = [o for o in m["agents"] if o["id"] != a["id"]]
        if not others:
            continue
        target = Path(others[0]["dir"]) / "secret"
        target.write_text("other-agent-secret", encoding="utf-8")
        pre = sandboxed(args, a["id"])
        code, out = run(pre + ["/bin/cat", str(target)], Path(a["dir"]), timeout=10)
        own = Path(a["dir"]) / "secret"
        own.write_text("own", encoding="utf-8")
        code2, _ = run(pre + ["/bin/cat", str(own)], Path(a["dir"]), timeout=10)
        results.append({"agent": a["id"], "other_read_denied": code != 0, "own_read_ok": code2 == 0,
                        "detail": out[:200]})
    print(json.dumps({"tenant": args.tenant, "results": results}, indent=2))
    sys.exit(0 if all(r["other_read_denied"] and r["own_read_ok"] for r in results) else 1)


def cmd_down(args):
    t = tenant_dir(args.tenant)
    if not t.exists():
        sys.exit("no such tenant")
    if not args.force:
        sys.exit("refusing to remove without --force")
    shutil.rmtree(t)
    print(json.dumps({"removed": str(t)}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("up"); up.add_argument("--tenant", required=True); up.add_argument("--agents", type=int, default=4); up.add_argument("--force", action="store_true"); up.add_argument("--sandbox", action="store_true"); up.set_defaults(fn=cmd_up)
    r = sub.add_parser("run"); r.add_argument("--tenant", required=True); r.add_argument("--tasks", required=True); r.add_argument("--workers", type=int, default=4); r.set_defaults(fn=cmd_run)
    ps = sub.add_parser("ps"); ps.add_argument("--tenant", required=True); ps.set_defaults(fn=cmd_ps)
    au = sub.add_parser("audit"); au.add_argument("--tenant", required=True); au.set_defaults(fn=cmd_audit)
    iso = sub.add_parser("isolation-check"); iso.add_argument("--tenant", required=True); iso.set_defaults(fn=cmd_isolation)
    dn = sub.add_parser("down"); dn.add_argument("--tenant", required=True); dn.add_argument("--force", action="store_true"); dn.set_defaults(fn=cmd_down)
    args = ap.parse_args()
    if not Path(QAL).exists():
        sys.exit(f"qal binary missing: {QAL}")
    args.fn(args)


if __name__ == "__main__":
    main()
