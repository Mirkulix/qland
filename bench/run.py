#!/usr/bin/env python3
"""Benchmark driver for the controlling Claude Code session (Phase 1e, Task 3).

  python3 bench/run.py plan [--runs N] [--seed S]     # randomized schedule -> bench/schedule.json
  python3 bench/run.py prepare <task> <arm> <run>     # fresh work dir; prints the agent prompt
  python3 bench/run.py finish <task> <arm> <run> --attempt K --agent-id ID --subagent-tokens N --tool-uses N
                              --duration-ms N --reply done|gave_up|incomplete|error
                                                      # check + transcript audit -> bench/results.jsonl
  python3 bench/run.py status                         # schedule entries and their outcome so far
  python3 bench/run.py audit <agent-id> <workdir> <arm>   # transcript statistics only

Work directories live outside the repository, in $BENCH_WORK (default /tmp/qal-bench), each under a
fresh random name, so an agent neither works inside the tree that holds the hidden tests and
reference solutions nor can guess where other agents work; the QAL server binary is copied to
$BENCH_WORK/bin. $BENCH_STATE (default bench/) holds schedule.json, results.jsonl and workdirs.json.
"""

import glob
import json
import os
import random
import re
import secrets
import shutil
import subprocess
import sys

BENCH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BENCH)
sys.path.insert(0, BENCH)
import check  # noqa: E402
import qalc  # noqa: E402

WORK = os.environ.get("BENCH_WORK", "/tmp/qal-bench")
# Where schedule.json and results.jsonl live (default bench/; the pilot used a scratch directory).
STATE = os.environ.get("BENCH_STATE", BENCH)
SCHEDULE = os.path.join(STATE, "schedule.json")
RESULTS = os.path.join(STATE, "results.jsonl")
WORKDIRS = os.path.join(STATE, "workdirs.json")
ARMS = ("qal", "rust")
MAX_ATTEMPTS = 3
TOOL_BUDGET = 60
CARGO_TOML = '[package]\nname = "solution"\nversion = "0.1.0"\nedition = "2021"\n\n[workspace]\n'
# Tools an agent may use (SubagentHandback is how a subagent reports back; ToolSearch only loads tools).
ALLOWED_TOOLS = {"Bash", "Read", "Write", "Edit", "MultiEdit", "Glob", "Grep", "TodoWrite", "ToolSearch", "SubagentHandback"}
# Absolute paths outside the work dir that a command may name (toolchains, devices, the server copy).
SYSTEM_PREFIXES = ("/usr/", "/bin/", "/sbin/", "/opt/homebrew/", "/dev/", "/Library/Developer/")

def entry_key(task, arm, run):
    return f"{task}-{arm}-run{run}"

def workdirs():
    return json.load(open(WORKDIRS, encoding="utf-8")) if os.path.exists(WORKDIRS) else {}

def workdir(task, arm, run):
    wd = workdirs().get(entry_key(task, arm, run))
    if wd is None:
        raise SystemExit(f"{entry_key(task, arm, run)} was not prepared")
    return wd

def new_workdir(task, arm, run):
    """A fresh directory under a random name; replaces the entry's previous one (a repeated dispatch)."""
    dirs = workdirs()
    old = dirs.get(entry_key(task, arm, run))
    if old and os.path.realpath(old).startswith(os.path.realpath(WORK) + os.sep):
        shutil.rmtree(old, ignore_errors=True)
    wd = os.path.join(os.path.realpath(WORK) if os.path.exists(WORK) else WORK, secrets.token_hex(8))
    os.makedirs(wd)
    wd = os.path.realpath(wd)
    dirs[entry_key(task, arm, run)] = wd
    os.makedirs(STATE, exist_ok=True)
    with open(WORKDIRS, "w", encoding="utf-8") as f:
        json.dump(dirs, f, indent=1)
        f.write("\n")
    return wd

ARCHIVE = os.environ.get("BENCH_ARCHIVE", os.path.expanduser("~/.cache/qal-bench-archive"))

def archive_workdir(wd, key, archive_root=None):
    """Moves a finished work dir out of $BENCH_WORK (where later agents could stumble on it) into the
    archive; returns the new path."""
    root = archive_root or ARCHIVE
    os.makedirs(root, exist_ok=True)
    dest = os.path.join(root, f"{key}-{os.path.basename(wd)}")
    shutil.move(wd, dest)
    return dest

def rules(template):
    """The shared part of a prompt template: everything before `## How to work`."""
    return template.split("## How to work")[0].replace("in QAL.", "in <language>.").replace("in Rust.", "in <language>.")

def templates():
    t = {arm: open(os.path.join(BENCH, "prompts", f"{arm}.md"), encoding="utf-8").read() for arm in ARMS}
    if rules(t["qal"]) != rules(t["rust"]):
        raise SystemExit("bench/prompts/qal.md and rust.md must have identical rules (all text before `## How to work`)")
    return t

def plan(runs, seed):
    entries = [{"task": t, "arm": a, "run": r} for r in range(1, runs + 1) for t in check.task_ids() for a in ARMS]
    random.Random(seed).shuffle(entries)
    with open(SCHEDULE, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "runs": runs, "model": "sonnet", "entries": entries}, f, indent=1)
        f.write("\n")
    print(f"wrote {SCHEDULE}: {len(entries)} dispatches, seed {seed!r}")

def prepare(task, arm, run):
    spec = check.load(task)
    fs = "fs" in spec["effects"]
    wd = new_workdir(task, arm, run)
    statement = open(os.path.join(BENCH, "tasks", task, "task.md"), encoding="utf-8").read().strip()
    if arm == "qal":
        check.ensure_qal_binary()
        # The agent's client points at a copy of the server outside the repository.
        binary = os.path.join(os.path.realpath(WORK), "bin", "qal")
        os.makedirs(os.path.dirname(binary), exist_ok=True)
        shutil.copy2(qalc.BIN, binary)
        src = open(os.path.join(BENCH, "qalc.py"), encoding="utf-8").read()
        src = re.sub(r"^BIN = .*$", f"BIN = {json.dumps(binary)}", src, count=1, flags=re.M)
        src = re.sub(r"^ROOT = .*$", f"ROOT = {json.dumps(wd)}", src, count=1, flags=re.M)
        with open(os.path.join(wd, "qalc.py"), "w", encoding="utf-8") as f:
            f.write(src)
        c = qalc.Client(root=wd, binary=binary)
        try:
            tools = qalc.describe_tools(c).strip()
        finally:
            c.close()
        allow = ["io", "fs:data"] if fs else ["io"]
        note = (f"; the test's files are created below `{wd}/data`, and relative paths resolve against the project root (e.g. `data/notes.txt`)"
                if fs else "")
        prompt = templates()["qal"].format(workdir=wd, allow=json.dumps(allow), fs_note=note,
                                           effects="`!io` and `!fs`" if fs else "`!io`", tools=tools, task=statement)
    else:
        # No src/main.rs yet: the agent writes it (an existing file would first have to be read).
        os.makedirs(os.path.join(wd, "src"))
        with open(os.path.join(wd, "Cargo.toml"), "w") as f:
            f.write(CARGO_TOML)
        version = subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip()
        note = ", in a working directory that contains the test's files (paths in arguments are relative to it, e.g. `data/notes.txt`)" if fs else ""
        prompt = templates()["rust"].format(workdir=wd, rust_version=version, fs_note=note, task=statement)
    os.makedirs(os.path.join(WORK, "prompts"), exist_ok=True)
    with open(os.path.join(WORK, "prompts", entry_key(task, arm, run) + ".md"), "w", encoding="utf-8") as f:
        f.write(prompt)
    print(prompt)

def transcript_path(agent_id):
    hits = glob.glob(os.path.expanduser(f"~/.claude/projects/**/agent-{agent_id}.jsonl"), recursive=True)
    return max(hits, key=os.path.getmtime) if hits else None

def outside(path, wd):
    p = os.path.realpath(os.path.join(wd, os.path.expanduser(path)))
    return not (p == wd or p.startswith(wd + os.sep))

def command_paths(cmd):
    """Path-like words of a shell command, ignoring heredoc bodies and quoted text with whitespace
    (QAL source, JSON arguments and test data travel there); quoted words without whitespace count."""
    cmd = re.sub(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2[ \t]*(?=\n|$)", "<<HEREDOC", cmd, flags=re.S)
    cmd = re.sub(r"'([^']*)'|\"((?:\\.|[^\"\\])*)\"", lambda m: (m.group(1) if m.group(1) is not None else m.group(2)) if not re.search(r"\s", m.group(0)) else "''", cmd)
    words = re.split(r"[\s;|&<>()=`]+", cmd)
    return [w for w in words if w.startswith(("/", "~", "$HOME", "${HOME}")) or w == ".." or w.startswith("../") or "/../" in w or w.endswith("/..")]

# Substrings that must not occur anywhere in a command (quoted text and heredoc bodies included):
# they name the benchmark's tree, hidden tests, reference solutions or other agents' work dirs.
FORBIDDEN = ("bench/", "/tasks/", "tasks/", "solutions/", "tests.json", "qal-bench", "$HOME", "${HOME}")

def forbidden_in(text, wd, repo):
    """Reasons why `text` (a command, or a Glob pattern) names something off limits; the agent's own
    work dir is masked first."""
    t = text.replace(wd, "<wd>")
    why = [f"names `{f}`" for f in FORBIDDEN if f in t]
    if repo in t:
        why.append("names the repository")
    if re.search(r"(^|[\s/='\"(;|&<>`])\.\.($|[\s/'\";|&<>)`])", t):
        why.append("names `..`")
    if re.search(r"(^|[\s='\"(;|&<>`])~", t):
        why.append("names `~`")
    return why

def tool_violations(name, inp, wd, repo):
    """Violations of one Read/Write/Edit/Glob/Grep call: an explicit absolute path inside the work dir
    is required (the tools' default directory is the repository), and a Glob pattern must stay inside."""
    v = []
    keys = [k for k in ("file_path", "path", "notebook_path") if isinstance(inp.get(k), str)]
    if name in ("Read", "Write", "Edit", "MultiEdit", "NotebookEdit") and not keys:
        v.append(f"{name} without file_path")
    if name in ("Glob", "Grep") and not isinstance(inp.get("path"), str):
        v.append(f"{name} without an explicit path (it would search the repository)")
    for k in keys:
        p = inp[k]
        if not os.path.isabs(p) or outside(p, wd):
            v.append(f"{name} {k}={p} is not an absolute path inside the work dir")
    if name == "Glob" and isinstance(inp.get("pattern"), str):
        pat = inp["pattern"]
        if os.path.isabs(pat) and outside(pat.split("*")[0].split("?")[0].split("[")[0].split("{")[0], wd):
            v.append(f"Glob pattern outside the work dir: {pat}")
        v += [f"Glob pattern {w}: {pat}" for w in forbidden_in(pat, wd, repo)]
    if name == "Grep" and isinstance(inp.get("glob"), str):
        v += [f"Grep glob {w}: {inp['glob']}" for w in forbidden_in(inp["glob"], wd, repo)]
    return v

def bash_violations(cmd, wd, repo, allowed_paths):
    v = []
    if not re.match(rf"\s*cd\s+(['\"]?){re.escape(wd)}\1\s*(&&|;)", cmd):
        v.append(f"Bash command does not start with `cd {wd} &&`: {cmd[:120]}")
    v += [f"Bash command {w}: {cmd[:200]}" for w in forbidden_in(cmd, wd, repo)]
    for w in command_paths(cmd):
        w2 = w.replace("${HOME}", "~").replace("$HOME", "~")
        p = os.path.realpath(os.path.join(wd, os.path.expanduser(w2)))
        if outside(w2, wd) and not (p + os.sep).startswith(SYSTEM_PREFIXES + allowed_paths) and not p.startswith(allowed_paths):
            v.append(f"Bash command names a path outside the work dir: {w}")
    return v

def audit(agent_id, wd, arm):
    """Statistics of a subagent transcript: API calls and usage (deduplicated per message; the
    transcript's output_tokens are only those of the first streamed chunk, so the dispatch result's
    subagent_tokens stay the primary token measure), work tool calls (without the final
    SubagentHandback), failed tool calls (any error the toolchain reported: a tool error, `isError:
    true` from qalc.py, a Rust compile error, panic or failed `cargo test`), and rule violations (a
    tool outside ALLOWED_TOOLS; a Bash call not starting with `cd <workdir> &&`; any forbidden substring
    or path outside the work dir in a command, quoted text and heredocs included; a file tool or
    Glob/Grep without an explicit path inside the work dir)."""
    path = transcript_path(agent_id)
    if path is None:
        return {"found": False}
    wd = os.path.realpath(wd)
    usage, calls, results = {}, {}, {}
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = d.get("message") or {}
        if d.get("type") == "assistant":
            if msg.get("id") and msg.get("usage"):
                usage[msg["id"]] = msg["usage"]
            for b in msg.get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    calls[b["id"]] = (b.get("name"), b.get("input") or {})
        elif d.get("type") == "user" and isinstance(msg.get("content"), list):
            for b in msg["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    c = b.get("content")
                    text = c if isinstance(c, str) else "\n".join(x.get("text", "") for x in c or [] if isinstance(x, dict))
                    results[b.get("tool_use_id")] = (bool(b.get("is_error")), text)
    tokens = {k: sum(u.get(k, 0) or 0 for u in usage.values()) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
    by_tool, failed, violations = {}, 0, []
    repo = os.path.realpath(REPO)
    allowed_paths = (os.path.realpath(os.path.join(WORK, "bin")) + os.sep,)
    order = {cid: n for n, cid in enumerate(calls, 1)}
    for cid, (name, _) in calls.items():
        if name not in ALLOWED_TOOLS:
            violations.append(f"call {order[cid]}: used tool {name}")
    calls = {k: v for k, v in calls.items() if v[0] != "SubagentHandback"}
    for cid, (name, inp) in calls.items():
        by_tool[name] = by_tool.get(name, 0) + 1
        is_error, text = results.get(cid, (False, ""))
        cmd = str(inp.get("command", ""))
        if arm == "qal" and name == "Bash" and "qalc.py" in cmd:
            is_error = is_error or "isError: true" in text
        if arm == "rust" and name == "Bash" and "cargo" in cmd:
            is_error = is_error or re.search(r"^error\[E\d+\]|^error: could not compile|panicked at|^test result: FAILED", text, re.M) is not None
        failed += is_error
        if name == "Bash":
            found = bash_violations(cmd, wd, repo, allowed_paths)
        elif name in ("Read", "Write", "Edit", "MultiEdit", "NotebookEdit", "Glob", "Grep"):
            found = tool_violations(name, inp, wd, repo)
        else:
            found = []
        violations += [f"call {order[cid]}: {x}" for x in found]
    return {"found": True, "path": path, "api_calls": len(usage), "tokens": tokens, "total_tokens": sum(tokens.values()),
            "tool_calls": len(calls), "over_budget": len(calls) > TOOL_BUDGET, "by_tool": by_tool, "failed_tool_calls": failed,
            "violations": violations}

def retry_message(res):
    detail = f" ({res['error'][:1000]})" if res["error"] else ""
    failed = res["total"] if res["error"] else len(res["failed"])
    return (f"The checker ran your solution against the hidden tests: {failed} of {res['total']} cases failed{detail}. "
            "Find and fix the problem (re-read the task statement and test more inputs), then reply DONE again, or GAVE UP: <reason>.")

def finish(task, arm, run, opts):
    wd = workdir(task, arm, run)
    res = check.check_qal(task, root=wd) if arm == "qal" else check.check_rust(task, wd)
    passed = res["error"] is None and not res["failed"]
    a = audit(opts["agent-id"], wd, arm)
    rec = {"task": task, "arm": arm, "run": run, "attempt": int(opts["attempt"]), "agent_id": opts["agent-id"], "reply": opts["reply"],
           "dispatch": {"total_tokens": int(opts["subagent-tokens"]), "tool_uses": int(opts["tool-uses"]), "duration_ms": int(opts["duration-ms"])},
           "check": res, "passed": passed, "transcript": a, "valid": a.get("found", False) and not a.get("violations")}
    final = passed or not rec["valid"] or rec["attempt"] >= MAX_ATTEMPTS
    rec["archive"] = None
    if final and os.path.isdir(wd):
        rec["archive"] = archive_workdir(wd, entry_key(task, arm, run))
        dirs = workdirs()
        dirs[entry_key(task, arm, run)] = rec["archive"]
        with open(WORKDIRS, "w", encoding="utf-8") as f:
            json.dump(dirs, f, indent=1)
            f.write("\n")
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    check.show(res, False)
    if not rec["valid"]:
        why = "transcript not found" if not a.get("found") else "; ".join(a["violations"][:5])
        print(f"NEXT: invalid run ({why}); dispatch a fresh agent for this entry once (prepare again), else record it as invalid")
    elif passed:
        print("NEXT: solved")
    elif rec["attempt"] < MAX_ATTEMPTS:
        print("NEXT: retry with SendMessage to the same agent:\n" + retry_message(res))
    else:
        print("NEXT: final, not solved")

def records():
    if not os.path.exists(RESULTS):
        return []
    return [json.loads(l) for l in open(RESULTS, encoding="utf-8") if l.strip()]

def status():
    sched = json.load(open(SCHEDULE, encoding="utf-8"))
    recs = records()
    for i, e in enumerate(sched["entries"], 1):
        rs = [r for r in recs if (r["task"], r["arm"], r["run"]) == (e["task"], e["arm"], e["run"])]
        last = rs[-1] if rs else None
        state = "pending" if last is None else ("solved" if last["passed"] and last["valid"] else f"attempt {last['attempt']} {'passed' if last['passed'] else 'failed'}{'' if last['valid'] else ' INVALID'}")
        print(f"{i:3} run{e['run']} {e['task']:18} {e['arm']:4} {state}")

def options(argv):
    opts = {}
    for k, v in zip(argv[::2], argv[1::2]):
        if not k.startswith("--"):
            raise SystemExit(f"expected an option, got {k}")
        opts[k[2:]] = v
    return opts

def main(argv):
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "plan":
        o = options(argv[2:])
        plan(int(o.get("runs", 2)), o.get("seed", "qal-1e"))
    elif cmd == "prepare" and len(argv) == 5 and argv[3] in ARMS:
        prepare(argv[2], argv[3], int(argv[4]))
    elif cmd == "finish" and len(argv) >= 5 and argv[3] in ARMS:
        o = options(argv[5:])
        missing = {"attempt", "agent-id", "subagent-tokens", "tool-uses", "duration-ms", "reply"} - set(o)
        if missing:
            raise SystemExit(f"finish needs --{' --'.join(sorted(missing))}")
        finish(argv[2], argv[3], int(argv[4]), o)
    elif cmd == "status":
        status()
    elif cmd == "audit" and len(argv) == 5:
        print(json.dumps(audit(argv[2], argv[3], argv[4]), indent=1))
    else:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
