#!/usr/bin/env python3
"""Summarizes the living-program experiment from state/results.jsonl and the transcripts:
per line (arm, model) and stage: hidden-test result, attempts, rounds, processed tokens, and
what the program's own verification caught during the run (failing tests / refuted proofs /
failed cargo test, with the names of the failing tests where the reply shows them).

  python3 bench/living/summarize.py            # table on stdout
  python3 bench/living/summarize.py --json     # one JSON object
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)
sys.path.insert(0, HERE)
import analyze  # noqa: E402
import drive  # noqa: E402


def caught_detail(path, arm):
    """Like drive.caught_regressions, with the names of the failing tests."""
    calls, results = analyze.read_transcript(path)
    tool_of = {tid: (n, i) for c in calls for tid, n, i in c["tools"]}
    events = []
    for tid, err, text, seen in results:
        n, inp = tool_of.get(tid, ("?", {}))
        if n != "Bash":
            continue
        cmd = str(inp.get("command", ""))
        if arm == "qal":
            if "E0801" in text and "qalc" in cmd:
                events.append({"call": seen, "kind": "contract_refuted"})
            if ("define" in cmd or "qal_test" in cmd) and "qalc" in cmd:
                names = re.findall(r'\{"name":"([^"]+)","passed":false', text)
                if names:
                    events.append({"call": seen, "kind": "tests_failed", "tests": names})
                if re.search(r'"status":"failed"', text):
                    props = re.findall(r'\{"def":"([^"]+)","status":"failed"', text)
                    if props:
                        events.append({"call": seen, "kind": "property_failed", "fns": props})
            if "E0401" in text and "qalc" in cmd:
                events.append({"call": seen, "kind": "signature_change_refused"})
        else:
            if "cargo test" in cmd and ("test result: FAILED" in text or "panicked" in text):
                names = re.findall(r"---- (\S+) stdout ----", text)
                events.append({"call": seen, "kind": "cargo_test_failed", "tests": names})
            if "cargo" in cmd and re.search(r"^error\[E\d+\]:", text, re.M):
                events.append({"call": seen, "kind": "compile_error"})
    return events


def summarize():
    recs = drive.records()
    lines = {}
    for r in recs:
        stage = int(r["task"][:2]) - 30
        line = (r["arm"], drive.MODELS[r["run"]])
        path = analyze.transcript_path(r["agent_id"])
        calls, _ = analyze.read_transcript(path)
        entry = lines.setdefault(line, {}).setdefault(stage, {"attempts": []})
        entry["attempts"].append({
            "attempt": r["attempt"], "passed": r["passed"], "hidden": f"{r['check']['passed']}/{r['check']['total']}",
            "failed_cases": r["check"]["failed"], "rounds": r["transcript"]["api_calls"], "processed": r["transcript"]["total_tokens"],
            "output": sum(analyze.measures(c["usage"])["output"] for c in calls), "duration_s": round(r["dispatch"]["duration_ms"] / 1000),
            "tool_calls": r["transcript"]["tool_calls"], "caught": caught_detail(path, r["arm"]), "violations": r["transcript"].get("violations", []),
        })
    return {f"{a}/{m}": {str(s): v for s, v in sorted(st.items())} for (a, m), st in sorted(lines.items())}


def table(summary):
    out = []
    for line, stages in summary.items():
        tot_rounds = sum(a["rounds"] for s in stages.values() for a in s["attempts"])
        tot_proc = sum(a["processed"] for s in stages.values() for a in s["attempts"])
        solved = sum(1 for s in stages.values() if s["attempts"][-1]["passed"])
        first = sum(1 for s in stages.values() if s["attempts"][0]["passed"])
        out.append(f"== {line}: {solved}/{len(stages)} stages solved ({first} at the first attempt), {tot_rounds} rounds, {tot_proc:,} processed tokens")
        for s, v in stages.items():
            for a in v["attempts"]:
                caught = "; ".join(f"{e['kind']}" + (f" {','.join(e.get('tests', e.get('fns', [])))}" if e.get("tests") or e.get("fns") else "") for e in a["caught"])
                out.append(f"   stage {s:>2} a{a['attempt']} {'ok  ' if a['passed'] else 'FAIL'} {a['hidden']:>6}  rounds {a['rounds']:2}  processed {a['processed']:>9,}  {a['duration_s']:4}s  caught: {caught or '-'}")
    return "\n".join(out)


if __name__ == "__main__":
    s = summarize()
    print(json.dumps(s, indent=1) if "--json" in sys.argv else table(s))
