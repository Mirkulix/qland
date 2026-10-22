#!/usr/bin/env python3
"""Scores a finished REAL Phase 4d run (team or solo) and appends one JSON line to results.jsonl.

  python3 bench/team/real/score.py <project dir> [--agent-stats stats.json] [--label text] [--no-append]

<project dir> is what prep.py printed (it holds run.json). It never modifies the store's code or records;
it reads `.qal/store.db` and `.qal/team-log.db` read-only, runs the 20 shop scenarios (5 public p*, 15 hidden h*)
against the merged `main` through a real `qal serve` per scenario (fresh database each), runs
`qal verify --trust team.json` and checks provenance. Token / tool-call numbers come ONLY from --agent-stats
(measured by the controller from the subagent results); nothing is estimated. Stdlib only.
"""

import argparse
import http.client
import json
import os
import re
import sqlite3
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIOS = os.path.join(os.path.dirname(HERE), "shop", "scenarios")
RESULTS = os.path.join(HERE, "results.jsonl")
WINDOW_MS = 5 * 60 * 1000
WRITE_OPS = {"define", "rename", "delete"}


# ---------------------------------------------------------------- running qal

class Qal:
    def __init__(self, binary, project, keydir):
        self.bin, self.project = binary, project
        self.env = os.environ.copy()
        if keydir:
            self.env["QAL_KEY_DIR"] = keydir

    def run(self, args, timeout=600):
        return subprocess.run([self.bin] + args, cwd=self.project, env=self.env, capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL)

    def json(self, args, timeout=600):
        r = self.run(args, timeout)
        try:
            return r.returncode, json.loads(r.stdout)
        except json.JSONDecodeError:
            return r.returncode, {"raw": r.stdout[-500:], "stderr": r.stderr[-500:]}


# ---------------------------------------------------------------- scenarios

class Server:
    def __init__(self, qal, dbname):
        args = ["serve", "--listen", "127.0.0.1:0", "--allow", f"db:{dbname},time", "--init", "setup"]
        self.p = subprocess.Popen([qal.bin] + args, cwd=qal.project, env=qal.env, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        self.addr, self.log = None, []
        deadline = time.time() + 120
        while time.time() < deadline:
            line = self.p.stderr.readline()
            if not line:
                break
            self.log.append(line.strip())
            try:
                v = json.loads(line)
            except json.JSONDecodeError:
                continue
            if v.get("event") == "listening":
                self.addr = v["addr"]
                break
        if not self.addr:
            self.stop()
            raise RuntimeError("qal serve did not start: " + " | ".join(self.log[-5:]))
        host, port = self.addr.rsplit(":", 1)
        self.conn = http.client.HTTPConnection(host, int(port), timeout=30)

    def request(self, method, path, headers, body):
        self.conn.request(method, path, body=body, headers=headers)
        r = self.conn.getresponse()
        data = r.read().decode("utf-8", "replace")
        return r.status, {k.lower(): v for k, v in r.getheaders()}, data

    def stop(self):
        try:
            self.p.terminate()
            self.p.wait(timeout=20)
        except Exception:
            self.p.kill()


def subst(x, env):
    """Replaces "{{v}}" (whole JSON string -> typed value) and {{v}} inside strings."""
    if isinstance(x, str):
        m = re.fullmatch(r"\{\{(\w+)\}\}", x)
        if m and m.group(1) in env:
            return env[m.group(1)]
        return re.sub(r"\{\{(\w+)\}\}", lambda m: str(env.get(m.group(1), m.group(0))), x)
    if isinstance(x, list):
        return [subst(v, env) for v in x]
    if isinstance(x, dict):
        return {k: subst(v, env) for k, v in x.items()}
    return x


def match(exp, act, env, path="$"):
    """Returns an error string or None. Objects: every expected key must match (extra keys allowed)."""
    if isinstance(exp, dict) and set(exp) == {"$var"}:
        n = exp["$var"]
        if n in env:
            return None if env[n] == act else f"{path}: expected {env[n]!r} (${n}), got {act!r}"
        env[n] = act
        return None
    if isinstance(exp, dict) and set(exp) == {"$type"}:
        want = exp["$type"]
        ok = {"int": isinstance(act, int) and not isinstance(act, bool), "str": isinstance(act, str),
              "bool": isinstance(act, bool), "float": isinstance(act, float), "list": isinstance(act, list),
              "object": isinstance(act, dict)}.get(want)
        return None if ok else f"{path}: expected type {want}, got {act!r}"
    if isinstance(exp, dict):
        if not isinstance(act, dict):
            return f"{path}: expected object, got {act!r}"
        for k, v in exp.items():
            if k not in act:
                return f"{path}.{k}: missing"
            e = match(v, act[k], env, f"{path}.{k}")
            if e:
                return e
        return None
    if isinstance(exp, list):
        if not isinstance(act, list) or len(exp) != len(act):
            return f"{path}: expected list of {len(exp)}, got {act!r}"
        for i, (v, a) in enumerate(zip(exp, act)):
            e = match(v, a, env, f"{path}[{i}]")
            if e:
                return e
        return None
    exp_n = exp if not isinstance(exp, str) else subst(exp, env)
    return None if exp_n == act and type(exp_n) is type(act) else f"{path}: expected {exp_n!r}, got {act!r}"


def pointer(doc, ptr):
    for part in [p for p in ptr.split("/") if p]:
        doc = doc[int(part)] if isinstance(doc, list) else doc[part]
    return doc


def run_scenario(qal, scen, dbname):
    """Returns None on success, else a one-line failure."""
    env = {}
    try:
        srv = Server(qal, dbname)
    except RuntimeError as e:
        return str(e)
    try:
        for step in scen["steps"]:
            rq = step["request"]
            path = subst(rq["path"], env)
            if rq.get("query"):
                path += "?" + subst(rq["query"], env)
            headers = {h["name"]: h["value"] for h in rq.get("headers", [])}
            body = rq.get("body")
            if "json" in rq:
                headers.setdefault("Content-Type", "application/json")
                body = json.dumps(subst(rq["json"], env))
            if body is not None:
                body = body.encode("utf-8")
            try:
                status, hdrs, text = srv.request(rq.get("method", "GET"), path, headers, body)
            except Exception as e:  # noqa: BLE001 - any transport failure fails the scenario
                return f"{step['name']}: transport error {e}"
            exp = step.get("expect", {})
            if "status" in exp and status != exp["status"]:
                return f"{step['name']}: expected status {exp['status']}, got {status} ({text[:120]!r})"
            parsed = None
            try:
                parsed = json.loads(text) if text else None
            except json.JSONDecodeError:
                pass
            if "json" in exp:
                if parsed is None:
                    return f"{step['name']}: body is not JSON ({text[:120]!r})"
                err = match(exp["json"], parsed, env)
                if err:
                    return f"{step['name']}: {err}"
            if "body" in exp and text != exp["body"]:
                return f"{step['name']}: expected body {exp['body']!r}, got {text[:120]!r}"
            for k, v in exp.get("headers", {}).items():
                want = subst(v, env)
                if hdrs.get(k.lower()) != want:
                    return f"{step['name']}: header {k}: expected {want!r}, got {hdrs.get(k.lower())!r}"
            for var, ptr in step.get("capture", {}).items():
                try:
                    env[var] = pointer(parsed, ptr)
                except Exception:  # noqa: BLE001
                    return f"{step['name']}: cannot capture {ptr} from {text[:120]!r}"
        return None
    finally:
        srv.stop()


def run_scenarios(qal):
    out = {"passed": 0, "total": 0, "public_passed": 0, "public_total": 0, "hidden_passed": 0, "hidden_total": 0,
           "failed": []}
    for i, fn in enumerate(sorted(f for f in os.listdir(SCENARIOS) if f.endswith(".json"))):
        scen = json.load(open(os.path.join(SCENARIOS, fn), encoding="utf-8"))
        kind = "public" if fn.startswith("p") else "hidden"
        fail = run_scenario(qal, scen, f"score{os.getpid()}x{i}")
        out["total"] += 1
        out[kind + "_total"] += 1
        if fail is None:
            out["passed"] += 1
            out[kind + "_passed"] += 1
        else:
            out["failed"].append(f"{fn}: {fail[:240]}")
    dbdir = os.path.join(qal.project, ".qal", "db")
    if os.path.isdir(dbdir):
        for f in os.listdir(dbdir):
            if f.startswith(f"score{os.getpid()}x"):
                os.unlink(os.path.join(dbdir, f))
    return out


# ---------------------------------------------------------------- store analysis

def ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def load_records(project):
    path = os.path.join(project, ".qal", "store.db")
    recs = []
    try:
        c = ro(path)
        for seq, body in c.execute("SELECT seq, body FROM records ORDER BY seq"):
            j = json.loads(body.split(b"\n", 1)[1])
            j["seq"] = seq
            recs.append(j)
        c.close()
    except sqlite3.Error:
        pass
    return recs


def analyse_records(recs, owner):
    out = {"records": len(recs), "records_by_member": {}, "changes": {}, "merges_fast_forward": 0, "merges_replayed": 0,
           "rebases": 0, "needs_rebase_marks": 0, "reviews": {}, "review_rounds": {}, "merge_gate_ok": True,
           "merge_gate_problems": []}
    by_seq = {r["seq"]: r for r in recs}
    changes = {}
    for r in recs:
        m, op, a = r["member"], r["op"], r["args"]
        out["records_by_member"][m] = out["records_by_member"].get(m, 0) + 1
        if op == "change":
            ch = changes.setdefault(a["id"], {"owner": None, "title": None, "claimed_by": None, "status": "open",
                                              "submits": 0, "defines": 0, "reviews": []})
            act = a["action"]
            if act == "open":
                ch["owner"], ch["title"] = m, a.get("fields", {}).get("title")
                ch["assignee"] = a.get("fields", {}).get("assignee")
            elif act == "claim":
                ch["owner"] = ch["claimed_by"] = m
            elif act == "submit":
                ch["submits"] += 1
                ch["status"] = "submitted"
            elif act == "merged":
                ch["status"] = "merged"
            elif act == "abandon":
                ch["status"] = "abandoned"
            elif act == "needs_rebase":
                out["needs_rebase_marks"] += 1
        elif op == "define" and r["space"] != "main" and r["space"] in changes:
            changes[r["space"]]["defines"] += 1
        elif op == "review":
            ch = changes.setdefault(a["change"], {"owner": None, "reviews": []})
            ch["reviews"].append({"seq": r["seq"], "member": m, "verdict": a["verdict"]})
            out["reviews"][a["verdict"]] = out["reviews"].get(a["verdict"], 0) + 1
        elif op == "rebase":
            out["rebases"] += 1
        elif op == "merge":
            kind = a.get("kind")
            out["merges_fast_forward" if kind == "fast_forward" else "merges_replayed"] += 1
            owner_of = changes.get(a["change"], {}).get("owner")
            gate = [by_seq[s] for s in a.get("reviews", []) if s in by_seq]
            verdicts = {g["args"]["verdict"] for g in gate if g["member"] != owner_of}
            if not {"approve", "verified"} <= verdicts:
                out["merge_gate_ok"] = False
                out["merge_gate_problems"].append(f"{a['change']}: reviews by non-owners {sorted(verdicts)}")
    for cid, ch in changes.items():
        # review rounds: how often the author had to rework (`changes` / `failed` verdicts)
        out["review_rounds"][cid] = sum(1 for rv in ch.get("reviews", []) if rv["verdict"] in ("changes", "failed"))
    out["changes"] = {cid: {k: v for k, v in ch.items() if k != "reviews"} | {"reviews": len(ch.get("reviews", []))}
                      for cid, ch in changes.items()}
    out["changes_merged"] = sum(1 for c in changes.values() if c.get("status") == "merged")
    out["changes_total"] = len(changes)
    # parallelism: the most members (not the owner) with a record in one 5-minute window
    def max_writers(ops):
        ev = sorted((r["at"], r["member"]) for r in recs if r["member"] != owner and (ops is None or r["op"] in ops))
        best, lo = 0, 0
        for hi in range(len(ev)):
            while ev[hi][0] - ev[lo][0] > WINDOW_MS:
                lo += 1
            best = max(best, len({m for _, m in ev[lo:hi + 1]}))
        return best
    out["writers_5min_define"] = max_writers(WRITE_OPS)
    out["writers_5min_any"] = max_writers(None)
    merges = [r["at"] for r in recs if r["op"] == "merge"]
    first = [r["at"] for r in recs if r["member"] != owner]
    out["first_member_record_at"] = min(first) if first else None
    out["last_merge_at"] = max(merges) if merges else None
    out["wall_s_first_record_to_last_merge"] = (round((max(merges) - min(first)) / 1000, 1) if merges and first else None)
    return out


def analyse_calls(project):
    out = {"calls": 0, "failed": 0, "failed_by_code": {}, "calls_by_member": {}, "failed_by_member": {},
           "lease_conflicts": 0, "stale_base": 0, "merge_conflicts": 0, "merge_failures": {}, "first_call_at": None,
           "last_call_at": None, "wait_calls": 0}
    path = os.path.join(project, ".qal", "team-log.db")
    try:
        c = ro(path)
        for member, tool, ok, code, at in c.execute("SELECT member, tool, ok, code, at FROM calls ORDER BY id"):
            out["calls"] += 1
            out["calls_by_member"][member] = out["calls_by_member"].get(member, 0) + 1
            out["first_call_at"] = out["first_call_at"] or at
            out["last_call_at"] = at
            if not ok:
                out["failed"] += 1
                out["failed_by_member"][member] = out["failed_by_member"].get(member, 0) + 1
                k = code or "none"
                out["failed_by_code"][k] = out["failed_by_code"].get(k, 0) + 1
                if code == "E0901":
                    out["lease_conflicts"] += 1
                elif code == "E0904":
                    out["stale_base"] += 1
                if tool == "qal_merge":
                    out["merge_failures"][k] = out["merge_failures"].get(k, 0) + 1
                    if code == "E0910":
                        out["merge_conflicts"] += 1
        c.close()
    except sqlite3.Error as e:
        out["error"] = str(e)
    return out


def provenance(project, owner, recs):
    """Every non-std definition on main must come from a team member's session: no `cli`, no owner."""
    out = {"names": 0, "ok": True, "problems": [], "authors": {}}
    rec_member = {r["seq"]: (r["member"], r["op"]) for r in recs}
    try:
        c = ro(os.path.join(project, ".qal", "store.db"))
        members = {m for (m,) in c.execute("SELECT id FROM members")} - {owner}
        names = c.execute("SELECT name, hash FROM names WHERE name NOT LIKE 'std.%'").fetchall()
        for name, h in names:
            out["names"] += 1
            rows = c.execute("SELECT agent, record, space FROM history WHERE name = ? AND new_hash = ?", (name, h)).fetchall()
            authors = {a for a, rec, sp in rows if a in members}
            bad = {a for a, rec, sp in rows if a not in members}
            writers = {rec_member[rec][0] for _, rec, _ in rows if rec in rec_member and rec_member[rec][1] == "define"}
            if not rows:
                out["problems"].append(f"{name}: no history entry for its current hash")
            if bad:
                out["problems"].append(f"{name}: written by {sorted(bad)} (cli/owner/foreign)")
            if owner in writers or not (authors or writers):
                out["problems"].append(f"{name}: no member define record")
            for a in (writers or authors):
                out["authors"][a] = out["authors"].get(a, 0) + 1
        owner_writes = c.execute("SELECT COUNT(*) FROM history WHERE agent NOT LIKE 'qal-std@%' AND agent NOT IN (SELECT id FROM members WHERE id != ?)", (owner,)).fetchone()[0]
        if owner_writes:
            out["problems"].append(f"{owner_writes} history rows by the owner/cli/unknown agents")
        c.close()
    except sqlite3.Error as e:
        out["problems"].append(f"store unreadable: {e}")
    owner_ops = [r["op"] for r in recs if r["member"] == owner and r["op"] in WRITE_OPS | {"merge", "review"}]
    if owner_ops:
        out["problems"].append(f"owner wrote records {owner_ops}")
    out["ok"] = not out["problems"]
    return out


def solo_provenance(project):
    out = {"names": 0, "authors": {}}
    try:
        c = ro(os.path.join(project, ".qal", "store.db"))
        for name, h in c.execute("SELECT name, hash FROM names WHERE name NOT LIKE 'std.%'").fetchall():
            out["names"] += 1
            for (a,) in c.execute("SELECT agent FROM history WHERE name = ? AND new_hash = ?", (name, h)).fetchall():
                out["authors"][a] = out["authors"].get(a, 0) + 1
        c.close()
    except sqlite3.Error as e:
        out["error"] = str(e)
    return out


def main_state(qal):
    """Non-std names on main, how many are fn/test/type, and how many bodies still call todo()."""
    rc, q = qal.json(["query"])
    items = q.get("results", [])
    mine = [d for d in items if isinstance(d, dict) and not str(d.get("name", "")).startswith("std.")]
    kinds = {}
    for d in mine:
        kinds[d.get("kind")] = kinds.get(d.get("kind"), 0) + 1
    stubs = []
    fns = [d["name"] for d in mine if d.get("kind") == "fn"]
    for i in range(0, len(fns), 40):
        _, v = qal.json(["view"] + fns[i:i + 40])
        for view in v.get("views", []):
            if re.search(r"\btodo\(\)", view.get("text", "")):
                stubs.append(view["name"])
    return {"definitions": len(mine), "kinds": kinds, "stubs_left": sorted(stubs)}


def stored_tests(qal):
    r = qal.run(["test", "--allow", "db:mem,time"], timeout=900)
    try:
        j = json.loads(r.stdout)
        return {"passed": j.get("passed"), "failed": j.get("failed"), "ok": r.returncode == 0}
    except json.JSONDecodeError:
        return {"ok": False, "raw": (r.stdout or r.stderr)[-300:]}


def agent_stats(path):
    """Accepts {"agents": {id: {tokens|total_tokens, tool_calls|tool_uses, duration_s|duration_ms}}, "wall_s"?}
    exactly as measured by the controller; sums per run. Nothing is estimated."""
    d = json.load(open(path, encoding="utf-8"))
    agents = d.get("agents", d)
    per, tot = {}, {"tokens": 0, "tool_calls": 0}
    for aid, s in agents.items():
        if not isinstance(s, dict):
            continue
        tok = s.get("tokens", s.get("total_tokens"))
        tc = s.get("tool_calls", s.get("tool_uses"))
        dur = s.get("duration_s", (s["duration_ms"] / 1000 if "duration_ms" in s else None))
        per[aid] = {"tokens": tok, "tool_calls": tc, "duration_s": dur, "outcome": s.get("outcome")}
        tot["tokens"] += tok or 0
        tot["tool_calls"] += tc or 0
    out = {"per_agent": per, **tot}
    if "wall_s" in d:
        out["wall_s"] = d["wall_s"]
    return out


# ---------------------------------------------------------------- main

def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--agent-stats")
    ap.add_argument("--label", default="")
    ap.add_argument("--no-append", action="store_true")
    ap.add_argument("--skip-scenarios", action="store_true", help="debug: do not run the 20 scenarios")
    a = ap.parse_args(argv[1:])
    project = os.path.realpath(a.project)
    info = json.load(open(os.path.join(project, "run.json"), encoding="utf-8"))
    team = info["kind"] == "team"
    qal = Qal(info["bin"], project, info.get("keydir"))
    owner = info.get("owner", "owner-1")
    res = {"arm": info["kind"], "run": info["run"], "label": a.label, "scored_at": int(time.time()),
           "project": project, "binary": info["bin"]}

    if a.skip_scenarios:
        res["scenarios"] = {"skipped": True}
    else:
        res["scenarios"] = run_scenarios(qal)
    res["state"] = main_state(qal)
    res["stored_tests"] = stored_tests(qal)

    if team:
        trust = os.path.join(project, "team.json")
        qal.run(["team", "export-trust", "--out", trust])
        rc, v = qal.json(["verify", "--trust", trust])
        res["verify"] = {"ok": bool(v.get("ok")) and rc == 0, "records": v.get("records"), "members": v.get("members"),
                         "trust": v.get("trust"), "detail": None if v.get("ok") else v}
        recs = load_records(project)
        res["records"] = analyse_records(recs, owner)
        res["calls"] = analyse_calls(project)
        res["provenance"] = provenance(project, owner, recs)
        res["lease_conflicts"] = res["calls"]["lease_conflicts"]
        res["merges"] = {"fast_forward": res["records"]["merges_fast_forward"],
                         "replayed": res["records"]["merges_replayed"], "conflict": res["calls"]["merge_conflicts"]}
        r = res["records"]
        res["criterion6"] = {
            "scenarios_all_pass": bool(res["scenarios"].get("total")) and res["scenarios"].get("passed") == res["scenarios"].get("total"),
            "writers_5min_ge_3": r["writers_5min_define"] >= 3,
            "non_ff_merge": r["merges_replayed"] >= 1,
            "merge_gate_ok": r["merge_gate_ok"] and r["changes_merged"] > 0,
            "verify_ok": res["verify"]["ok"],
            "provenance_ok": res["provenance"]["ok"],
        }
    else:
        res["verify"] = {"ok": None, "note": "not applicable: a solo project is not a team store (no signed records)"}
        res["provenance"] = solo_provenance(project)

    if a.agent_stats:
        res["agent_stats"] = agent_stats(a.agent_stats)

    line = json.dumps(res, sort_keys=True)
    if not a.no_append:
        with open(RESULTS, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    s = res["scenarios"]
    print(f"{info['kind']} run {info['run']}: scenarios {s.get('passed')}/{s.get('total')} "
          f"(public {s.get('public_passed')}/{s.get('public_total')}, hidden {s.get('hidden_passed')}/{s.get('hidden_total')}), "
          f"verify ok={res['verify']['ok']}, stubs left={len(res['state']['stubs_left'])}")
    for f in s.get("failed", [])[:20]:
        print("  FAIL", f)
    if team:
        print("  criterion 6:", json.dumps(res["criterion6"]))
        for p in res["provenance"]["problems"][:10]:
            print("  PROVENANCE", p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
