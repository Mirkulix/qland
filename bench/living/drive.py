#!/usr/bin/env python3
"""Driver of the living-program experiment: one line = (arm, run) where run 1 is Sonnet and run 2
is Haiku; each line goes through the stages 0..12, every stage a fresh agent on the code the
previous stage left (the QAL store, or src/main.rs).

  drive.py stage <k> <arm> <run>                      prepare the work dir, seed it from stage k-1's
                                                      archive, write TASK.md; prints the work dir
  drive.py finish <k> <arm> <run> <agent-prefix> <attempt> <reply>
                                                      check against the hidden tests, record
  drive.py status                                     table of the lines
State: bench/living/state (results.jsonl, workdirs.json, dispatch.json); work: /tmp/qal-living.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
REPO = os.path.dirname(BENCH)
# LIVING_MODE=pressure: change requests without full specification and test duty
# (tasks-pressure), stage 1 seeded from the normal run's stage 0 of the same line.
PRESSURE = os.environ.get("LIVING_MODE") == "pressure"
STATE = os.path.join(HERE, "state-pressure" if PRESSURE else "state")
NORMAL_STATE = os.path.join(HERE, "state")
WORK = "/tmp/qal-living-p" if PRESSURE else "/tmp/qal-living"
ENV = dict(os.environ, QAL_BIN=os.path.join(REPO, "qal", "target", "release", "qal"), BENCH_TASKS=os.path.join(HERE, "tasks-pressure" if PRESSURE else "tasks"), BENCH_STATE=STATE, BENCH_WORK=WORK)
MODELS = {1: "sonnet", 2: "haiku"}
STAGES = 13
sys.path.insert(0, BENCH)
import analyze  # noqa: E402


def task_id(k):
    return f"{k + 30:02d}-living-{'abcdefghijklm'[k]}"


def key(k, arm, run):
    return f"{task_id(k)}-{arm}-run{run}"


def records(state=None):
    p = os.path.join(state or STATE, "results.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def archive_of(k, arm, run):
    """The archived work dir of the last recorded attempt of a stage."""
    recs = [r for r in records(NORMAL_STATE if PRESSURE and k == 0 else None) if r["task"] == task_id(k) and r["arm"] == arm and r["run"] == run and r.get("archive")]
    if not recs:
        raise SystemExit(f"stage {k} of {arm}/{MODELS[run]} has no archived result yet")
    return recs[-1]["archive"]


def stage(k, arm, run):
    os.makedirs(STATE, exist_ok=True)
    sched = os.path.join(STATE, "schedule.json")
    if not os.path.exists(sched):
        entries = [{"task": task_id(s), "arm": a, "run": r} for s in range(STAGES) for a in ("qal", "rust") for r in (1, 2)]
        json.dump({"seed": "living-2026-10-23", "runs": 2, "model": MODELS, "entries": entries}, open(sched, "w"), indent=1)
    r = subprocess.run([sys.executable, "bench/run.py", "prepare", task_id(k), arm, str(run)], cwd=REPO, env=ENV, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr, file=sys.stderr)
        return r.returncode
    wd = json.load(open(os.path.join(STATE, "workdirs.json")))[key(k, arm, run)]
    if k > 0:
        prev = archive_of(k - 1, arm, run)
        if arm == "qal":
            os.makedirs(os.path.join(wd, ".qal"), exist_ok=True)
            shutil.copy2(os.path.join(prev, ".qal", "store.db"), os.path.join(wd, ".qal", "store.db"))
        else:
            shutil.copy2(os.path.join(prev, "src", "main.rs"), os.path.join(wd, "src", "main.rs"))
    shutil.copy(os.path.join(WORK, "prompts", key(k, arm, run) + ".md"), os.path.join(wd, "TASK.md"))
    p = os.path.join(STATE, "dispatch.json")
    d = json.load(open(p)) if os.path.exists(p) else {}
    d[key(k, arm, run)] = time.time()
    json.dump(d, open(p, "w"))
    print(f"{key(k, arm, run)} ({MODELS[run]}) -> {wd}")
    return 0


def finish(k, arm, run, prefix, attempt, reply):
    hits = sorted(glob.glob(os.path.expanduser(f"~/.claude/projects/**/agent-{prefix}-*.jsonl"), recursive=True), key=os.path.getmtime)
    if not hits:
        print("transcript not found for", prefix, file=sys.stderr)
        return 1
    path = hits[-1]
    agent = os.path.basename(path)[len("agent-"):-len(".jsonl")]
    wd = json.load(open(os.path.join(STATE, "workdirs.json")))[key(k, arm, run)]
    a = json.loads(subprocess.run([sys.executable, "bench/run.py", "audit", agent, wd, arm], cwd=REPO, env=ENV, capture_output=True, text=True, check=True).stdout)
    t = a["tokens"]
    tokens = t["input_tokens"] + t["cache_creation_input_tokens"] + t["output_tokens"]
    d = json.load(open(os.path.join(STATE, "dispatch.json")))
    duration_ms = int((time.time() - d[key(k, arm, run)]) * 1000)
    print(f"{key(k, arm, run)} ({MODELS[run]}): processed={a['total_tokens']} rounds={a['api_calls']} tool_calls={a['tool_calls']} failed_calls={a['failed_tool_calls']} violations={a['violations']}")
    # Pressure lines get no retry: the attempt is final (archived) and the next stage builds on it.
    if PRESSURE:
        attempt = 3
    r = subprocess.run([sys.executable, "bench/run.py", "finish", task_id(k), arm, str(run), "--attempt", str(attempt), "--agent-id", agent,
                        "--subagent-tokens", str(tokens), "--tool-uses", str(a["tool_calls"]), "--duration-ms", str(duration_ms), "--reply", reply], cwd=REPO, env=ENV)
    caught = caught_regressions(path, arm)
    print("caught by the program's own verification during the run:", caught)
    return r.returncode


def caught_regressions(path, arm):
    """Tool results in which the program's own verification reported a failure: QAL define/test
    replies with failed tests or refuted proofs; Rust `cargo test` runs with failures."""
    calls, results = analyze.read_transcript(path)
    tool_of = {tid: (n, i) for c in calls for tid, n, i in c["tools"]}
    hits = []
    for tid, err, text, seen in results:
        n, inp = tool_of.get(tid, ("?", {}))
        cmd = str(inp.get("command", ""))
        if n != "Bash":
            continue  # a Read of TASK.md carries the reference's own mention of E0801
        if arm == "qal":
            if ("define" in cmd or "qal_test" in cmd) and ('"failed":' in text and '"failed":0' not in text.replace(" ", "")):
                hits.append(f"call {seen}: tests failed")
            if "E0801" in text and "qalc" in cmd and "cat TASK.md" not in cmd:
                hits.append(f"call {seen}: contract refuted")
        else:
            if "cargo test" in cmd and ("test result: FAILED" in text or "panicked" in text):
                hits.append(f"call {seen}: cargo test failed")
    return hits


def status():
    recs = records()
    print("stage  " + "  ".join(f"{a}/{MODELS[r]:6}" for a in ("qal", "rust") for r in (1, 2)))
    for k in range(STAGES):
        row = []
        for a in ("qal", "rust"):
            for r in (1, 2):
                rs = [x for x in recs if x["task"] == task_id(k) and x["arm"] == a and x["run"] == r]
                if not rs:
                    row.append("      -    ")
                else:
                    last = rs[-1]
                    c = last["check"]
                    row.append(f"{'ok ' if last['passed'] else 'FAIL'} {c['passed']:2}/{c['total']:2} a{last['attempt']}")
        print(f"{k:5}  " + "  ".join(row))


def main(argv):
    if argv[0] == "stage":
        return stage(int(argv[1]), argv[2], int(argv[3]))
    if argv[0] == "finish":
        return finish(int(argv[1]), argv[2], int(argv[3]), argv[4], int(argv[5]), argv[6])
    if argv[0] == "status":
        return status()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) or 0)
