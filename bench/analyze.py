#!/usr/bin/env python3
"""Token attribution of benchmark transcripts (Phase 6, Task 1 of
docs/superpowers/plans/2026-10-23-qal-phase6-der-kreis-wird-billig.md).

  python3 bench/analyze.py <results.jsonl>... --out <dir> [--llm-cache <cache.jsonl>] [--report]

Attribution rule: API call 1 of a transcript carries the prompt; `prompt.reference` gets
min(new tokens of call 1, reference_chars / 4) (about 4 characters per token), `prompt.task` the
rest, which includes the agent harness's own system prompt (the same in both arms); processed and
output tokens of call 1 are split in the same ratio. API call n >= 2 carries the category of the
tool result that immediately precedes it (the agent reacts to that result), the last round
included; a call that no tool result precedes (a text-only continuation) is `final`. Output tokens
go to the same category as the call's new tokens. `diag_codes` counts every diagnostic code of a
failed QAL call.

Measures per API call: processed = input + cache_creation + cache_read + output (the main figure
of the design spec), new = input + cache_creation, output.
"""

import collections
import glob
import json
import os
import re
import statistics
import sys

BENCH = os.path.dirname(os.path.abspath(__file__))

CATEGORIES = (
    "prompt.reference", "prompt.task",
    "env.guard", "env.tool_error", "env.read_task",
    "lang.parse", "lang.name", "lang.type", "lang.effect", "lang.contract", "lang.args", "lang.other_diag",
    "lang.test_failed", "lang.run_failed",
    "rust.compile", "rust.test_failed", "rust.panic",
    "ok.define", "ok.test", "ok.run", "ok.other",
    "final", "unclassified",
)
MEASURES = ("processed", "new", "output")
ROUND_COST_KEYS = ("1", "5", "10")
TEXT_LIMIT = 2000
CHARS_PER_TOKEN = 4

CODE = re.compile(r'"code":\s*"(E\d{4})"')
RUST_COMPILE = re.compile(r"^error\[E\d+\]|^error: could not compile", re.M)
RUST_TEST_FAILED = re.compile(r"^test result: FAILED", re.M)


def diag_category(code):
    n = int(code[1:])
    if n in (100, 102):
        return "lang.parse"
    if n in (201, 202, 209):
        return "lang.name"
    if 203 <= n <= 214 or n == 240:
        return "lang.type"
    if 300 <= n < 400:
        return "lang.effect"
    if 700 <= n < 800:
        return "lang.args"
    if 800 <= n < 900:
        return "lang.contract"
    return "lang.other_diag"


def classify_result(arm, tool, inp, text, is_error):
    """The category of one tool result (never `prompt.*` or `final`); see the module doc."""
    inp = inp or {}
    text = text or ""
    if "isolated in the worktree" in text:
        return "env.guard"
    if tool == "Read" and str(inp.get("file_path", "")).endswith("TASK.md"):
        return "env.read_task"
    cmd = str(inp.get("command", ""))
    if arm == "qal" and tool == "Bash" and "qalc.py" in cmd:
        if "isError: true" in text:
            m = CODE.search(text)
            return diag_category(m.group(1)) if m else "lang.other_diag"
        if "qalc.py define" in cmd or "qal_define" in cmd:
            return "ok.define"
        if "qal_test" in cmd:
            return "lang.test_failed" if re.search(r'"failed":\s*\[\s*[^\]\s]', text) else "ok.test"
        if "qal_run" in cmd or "qal_call" in cmd:
            return "lang.run_failed" if re.search(r'"error":\s*(\{|")', text) else "ok.run"
        return "ok.other"
    if arm == "rust" and tool == "Bash" and "cargo" in cmd:
        if RUST_COMPILE.search(text):
            return "rust.compile"
        if "panicked at" in text:
            return "rust.panic"
        if RUST_TEST_FAILED.search(text):
            return "rust.test_failed"
        return "ok.other"
    if is_error:
        return "env.tool_error"
    return "ok.other"


def read_transcript(path):
    """API calls in order (first line per message id wins for order, usage taken once), each with
    its tool uses; and the tool results keyed by tool_use id, in order of appearance."""
    calls, order, results = {}, [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = d.get("message") or {}
            if d.get("type") == "assistant" and msg.get("id"):
                mid = msg["id"]
                if mid not in calls:
                    calls[mid] = {"usage": msg.get("usage") or {}, "tools": []}
                    order.append(mid)
                elif not calls[mid]["usage"] and msg.get("usage"):
                    calls[mid]["usage"] = msg["usage"]
                for b in msg.get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        calls[mid]["tools"].append((b.get("id"), b.get("name"), b.get("input") or {}))
            elif d.get("type") == "user" and isinstance(msg.get("content"), list):
                for b in msg["content"]:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        c = b.get("content")
                        text = c if isinstance(c, str) else "\n".join(x.get("text", "") for x in c or [] if isinstance(x, dict))
                        results.append((b.get("tool_use_id"), bool(b.get("is_error")), text, len(order)))
    return [calls[m] for m in order], results


def measures(usage):
    i, cc, cr, o = (usage.get(k, 0) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"))
    return {"processed": i + cc + cr + o, "new": i + cc, "output": o}


def empty_category():
    return {"processed": 0, "new": 0, "output": 0, "calls": 0}


def attribute(path, arm, reference_chars, prompt_chars):
    calls, results = read_transcript(path)
    tool_of = {}
    for n, c in enumerate(calls, 1):
        for tid, name, inp in c["tools"]:
            tool_of[tid] = (n, name, inp)
    # category of the result that precedes API call n (results seen before call n was issued)
    preceding, unclassified, retries = {}, [], 0
    diag_codes = collections.Counter()
    last_failed = None
    for tid, is_error, text, seen_calls in results:
        if tid in tool_of:
            n, name, inp = tool_of[tid]
            cat = classify_result(arm, name, inp, text, is_error)
            if arm == "qal" and name == "Bash" and "qalc.py" in str(inp.get("command", "")) and "isError: true" in text:
                diag_codes.update(CODE.findall(text))
            key = json.dumps(inp, sort_keys=True)
            if last_failed is not None and key == last_failed:
                retries += 1
            failed = cat.startswith(("env.", "lang.", "rust.")) and cat != "env.read_task"
            last_failed = key if failed else None
        else:
            cat = "unclassified"
            unclassified.append({"call": seen_calls, "tool": "?", "text": text[:TEXT_LIMIT]})
        preceding[seen_calls + 1] = (cat, text)
    by = collections.defaultdict(empty_category)
    per_round = []
    for n, c in enumerate(calls, 1):
        m = measures(c["usage"])
        per_round.append(m["processed"])
        if n == 1:
            ref_tokens = min(m["new"], reference_chars / CHARS_PER_TOKEN)
            share = (ref_tokens / m["new"]) if m["new"] else 0.0
            for cat, f in (("prompt.reference", share), ("prompt.task", 1 - share)):
                for k in MEASURES:
                    by[cat][k] += m[k] * f
                by[cat]["calls"] += 1 if f > 0 else 0
            continue
        cat = preceding[n][0] if n in preceding else "final"
        for k in MEASURES:
            by[cat][k] += m[k]
        by[cat]["calls"] += 1
    return {"by_category": dict(by), "api_calls": len(calls), "processed_per_round": per_round, "retries": retries,
            "unclassified": unclassified, "diag_codes": dict(diag_codes)}


def group_of(cat):
    return cat.split(".")[0]


def summarize(rows):
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        rs = [r for r in rows if r["arm"] == arm]
        totals = collections.defaultdict(empty_category)
        for r in rs:
            for cat, m in r["by_category"].items():
                for k in m:
                    totals[cat][k] += m[k]
        n = len(rs)
        means = {cat: {k: v / n for k, v in m.items()} for cat, m in totals.items()}
        processed = sum(m["processed"] for m in totals.values()) or 1
        shares = collections.defaultdict(float)
        for cat, m in totals.items():
            shares[group_of(cat)] += m["processed"] / processed
        top3 = sorted(((cat, m["processed"]) for cat, m in totals.items()), key=lambda x: -x[1])[:3]
        rounds = [r["api_calls"] for r in rs]
        diag_codes = collections.Counter()
        for r in rs:
            diag_codes.update(r.get("diag_codes") or {})
        round_cost = {}
        for key in ROUND_COST_KEYS:
            k = int(key)
            costs = [r["processed_per_round"][k - 1] for r in rs if len(r["processed_per_round"]) >= k]
            round_cost[key] = statistics.mean(costs) if costs else None
        out[arm] = {
            "runs": n,
            "totals": dict(totals),
            "means": means,
            "shares": dict(shares),
            "top3": top3,
            "rounds": {"mean": statistics.mean(rounds) if rounds else 0, "median": statistics.median(rounds) if rounds else 0, "max": max(rounds) if rounds else 0},
            "round_cost": round_cost,
            "retries": sum(r.get("retries", 0) for r in rs),
            "dispatch_tokens": sum(r.get("dispatch_tokens", 0) for r in rs),
            "diag_codes": dict(diag_codes.most_common()),
        }
    return out


def transcript_path(agent_id):
    hits = glob.glob(os.path.expanduser(f"~/.claude/projects/**/agent-{agent_id}.jsonl"), recursive=True)
    return hits[0] if hits else None


def prompt_sizes(rec):
    """(reference_chars, prompt_chars) of a run: from the archived TASK.md when present, else from
    the prompt template plus the reference as `run.prepare` builds it."""
    arm = rec["arm"]
    task_md = os.path.join(rec.get("archive") or "", "TASK.md")
    if rec.get("archive") and os.path.exists(task_md):
        with open(task_md, encoding="utf-8") as f:
            prompt = f.read()
    else:
        with open(os.path.join(BENCH, "prompts", f"{arm}.md"), encoding="utf-8") as f:
            prompt = f.read()
        if arm == "qal":
            prompt = prompt.replace("{tools}", reference_text())
    return (len(reference_text()) if arm == "qal" else 0), max(len(prompt), 1)


_REFERENCE = None


def reference_text():
    """The `{tools}` block of the QAL prompt: the tool list of the MCP server as `run.prepare`
    inserts it (read from the first archived QAL TASK.md found, else the reference file)."""
    global _REFERENCE
    if _REFERENCE is None:
        ref = os.path.join(BENCH, "..", "qal", "crates", "qal-cli", "src", "reference.md")
        with open(ref, encoding="utf-8") as f:
            _REFERENCE = f.read()
    return _REFERENCE


def load_rows(result_files, llm_cache=None):
    cache = load_cache(llm_cache) if llm_cache else {}
    rows = []
    for rf in result_files:
        with open(rf, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                path = transcript_path(rec["agent_id"])
                if not path:
                    print(f"no transcript for {rec['agent_id']}", file=sys.stderr)
                    continue
                ref_chars, prompt_chars = prompt_sizes(rec)
                a = attribute(path, rec["arm"], ref_chars, prompt_chars)
                if cache:
                    apply_cache(a, path, rec["arm"], cache)
                rows.append({"key": f"{rec['task']}-{rec['arm']}-{rec['run']}", "task": rec["task"], "arm": rec["arm"], "run": rec["run"],
                             "source": rf, "dispatch_tokens": (rec.get("dispatch") or {}).get("total_tokens", 0), **a})
    return rows


def load_cache(path):
    cache = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    cache[d["sha256"]] = d["category"]
    return cache


RECLASSIFY = ("unclassified", "env.tool_error", "lang.other_diag")


def apply_cache(a, path, arm, cache):
    """Replaces the categories the rules could not settle by the cached LLM verdicts (Task 2)."""
    import hashlib
    calls, results = read_transcript(path)
    tool_of = {tid: (name, inp) for c in calls for tid, name, inp in c["tools"]}
    moves = []
    for tid, is_error, text, seen in results:
        name, inp = tool_of.get(tid, ("?", {}))
        cat = classify_result(arm, name, inp, text, is_error) if tid in tool_of else "unclassified"
        if cat in RECLASSIFY:
            new = cache.get(hashlib.sha256(text[:TEXT_LIMIT].encode("utf-8")).hexdigest())
            if new and new != cat:
                moves.append((seen + 1, cat, new))
    per_call = {}
    for n, c in enumerate(calls, 1):
        per_call[n] = measures(c["usage"])
    for n, old, new in moves:
        m = per_call.get(n)
        if not m or old not in a["by_category"] or a["by_category"][old]["calls"] == 0:
            continue
        for k in MEASURES:
            a["by_category"][old][k] -= m[k]
            a["by_category"].setdefault(new, empty_category())[k] += m[k]
        a["by_category"][old]["calls"] -= 1
        a["by_category"][new]["calls"] += 1


def report(summary):
    lines = ["# Token attribution", ""]
    for arm, s in summary.items():
        lines += [f"## {arm} ({s['runs']} runs)", "", "| category | processed | new | output | calls | processed/run |", "|---|---|---|---|---|---|"]
        for cat, m in sorted(s["totals"].items(), key=lambda x: -x[1]["processed"]):
            lines.append(f"| {cat} | {m['processed']:.0f} | {m['new']:.0f} | {m['output']:.0f} | {m['calls']:.0f} | {s['means'][cat]['processed']:.0f} |")
        lines += ["", "Shares of processed tokens: " + ", ".join(f"{g} {v * 100:.1f} %" for g, v in sorted(s["shares"].items(), key=lambda x: -x[1])),
                  "", "Top 3: " + ", ".join(f"{cat} ({p:.0f})" for cat, p in s["top3"]),
                  "", f"Umgebung (env) {s['shares'].get('env', 0) * 100:.1f} % / Sprache (lang) {s['shares'].get('lang', 0) * 100:.1f} %",
                  "", f"Rounds per run: mean {s['rounds']['mean']:.1f}, median {s['rounds']['median']}, max {s['rounds']['max']}; retries {s['retries']}",
                  "", "Processed tokens of round " + ", ".join(f"{k}: {v:.0f}" if v is not None else f"{k}: –" for k, v in s["round_cost"].items()),
                  "", "Diagnostic codes of failed calls: " + (", ".join(f"{c} {n}" for c, n in s["diag_codes"].items()) or "none"), ""]
    return "\n".join(lines)


def main(argv):
    files, out, llm_cache, want_report = [], None, None, False
    i = 0
    while i < len(argv):
        if argv[i] == "--out":
            out = argv[i + 1]; i += 2
        elif argv[i] == "--llm-cache":
            llm_cache = argv[i + 1]; i += 2
        elif argv[i] == "--report":
            want_report = True; i += 1
        else:
            files.append(argv[i]); i += 1
    if not files or not out:
        print(__doc__, file=sys.stderr)
        return 2
    rows = load_rows(files, llm_cache)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "rows.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    n_unclassified = sum(len(r["unclassified"]) + r["by_category"].get("unclassified", empty_category())["calls"] for r in rows)
    print(f"runs: {len(rows)}; unclassified: {n_unclassified} results", file=sys.stderr)
    if want_report:
        with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
            f.write(report(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
