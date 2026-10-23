#!/usr/bin/env python3
"""Seed corpus of corrections (Phase 6b, Task 3).

A correction is a failed `qal_define` diagnostic followed, later in the same transcript, by a
successful `qal_define` that defines the diagnostic's definition. The record carries the failing
and the fixed text of that definition only:

  python3 bench/corrections.py <transcript.jsonl>... [--results <results.jsonl>]... --out corrections.json

`--results` adds the transcripts of the agent ids in a results file (`analyze.transcript_path`).
The output is the JSON list that `qal corrections import` reads.
"""

import collections
import json
import os
import re
import sys

BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH)
import analyze  # noqa: E402

HEREDOC = re.compile(r"qalc\.py\s+define\b[^\n]*<<\s*'?EOF'?[^\n]*\n(.*?)\nEOF\b", re.S)
JSON_CALL = re.compile(r"qalc\.py\s+call\s+qal_define\s+'(.*)'\s*$", re.S)
DEF_START = re.compile(r"^(fn|type|test)\s+([A-Za-z_][A-Za-z0-9_]*)")


def define_source(name, inp):
    """The source of a `qal_define` call (a Bash command of the bench wrapper, or the tool itself), else None."""
    if not isinstance(inp, dict):
        return None
    if name and name.endswith("qal_define"):
        return inp.get("source") if isinstance(inp.get("source"), str) else None
    cmd = inp.get("command")
    if not isinstance(cmd, str) or "qalc.py" not in cmd:
        return None
    m = HEREDOC.search(cmd)
    if m:
        return m.group(1)
    m = JSON_CALL.search(cmd)
    if m:
        try:
            src = json.loads(m.group(1)).get("source")
        except (json.JSONDecodeError, AttributeError):
            return None
        return src if isinstance(src, str) else None
    return None


def cut(source, name):
    """The definition `name` of `source`: from its `fn`/`type`/`test` line to the next definition
    start, without the blank and comment lines that lead into that next definition. None if absent."""
    lines = source.split("\n")
    start = next((i for i, l in enumerate(lines) if (m := DEF_START.match(l)) and m.group(2) == name), None)
    if start is None:
        return None
    end = next((i for i in range(start + 1, len(lines)) if DEF_START.match(lines[i])), len(lines))
    body = lines[start:end]
    while body and (not body[-1].strip() or body[-1].lstrip().startswith("#")):
        body.pop()
    return "\n".join(body)


def outcome(text):
    """('failed', diagnostics) or ('ok', defined) from a define result text; None if it is neither."""
    first, _, rest = text.partition("\n")
    if first.strip() not in ("isError: true", "isError: false"):
        return None
    try:
        # The wrapper may append more text (a summary line) after the JSON document.
        data, _ = json.JSONDecoder().raw_decode(rest.lstrip())
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if first.strip() == "isError: true":
        diags = data.get("diagnostics")
        return ("failed", diags) if isinstance(diags, list) else None
    defined = data.get("defined")
    return ("ok", defined) if isinstance(defined, list) else None


def extract(path):
    """The correction records of one transcript, in the order they were fixed."""
    agent = re.sub(r"^agent-|\.jsonl$", "", os.path.basename(path))
    calls, results = analyze.read_transcript(path)
    result = {tid: text for tid, _, text, _ in results}
    pending, records = {}, []
    for c in calls:
        for tid, name, inp in c["tools"]:
            source = define_source(name, inp)
            if source is None or tid not in result:
                continue
            got = outcome(result[tid])
            if got is None:
                continue
            kind, items = got
            if kind == "failed":
                firsts = {}
                for d in items:
                    if isinstance(d, dict) and d.get("def"):
                        firsts.setdefault(d["def"], d)
                for df, d in firsts.items():
                    before = cut(source, df)
                    if before:
                        pending[df] = (str(d.get("code", "")), str(d.get("path", "")), str(d.get("message", "")), before)
            else:
                for item in items:
                    df = item.get("name") if isinstance(item, dict) else None
                    if df not in pending or item.get("status") not in ("added", "replaced"):
                        continue
                    code, p, message, before = pending.pop(df)
                    after = cut(source, df)
                    if code and after and after != before:
                        records.append({"code": code, "path": p, "message": message, "before": before, "after": after, "agent": agent})
    return records


def main(argv):
    paths, out, i = [], None, 0
    while i < len(argv):
        a = argv[i]
        if a == "--out" and i + 1 < len(argv):
            out, i = argv[i + 1], i + 2
        elif a == "--results" and i + 1 < len(argv):
            with open(argv[i + 1], encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    p = analyze.transcript_path(rec.get("agent_id") or "") if rec.get("agent_id") else None
                    if p:
                        paths.append(p)
            i += 2
        else:
            paths.append(a)
            i += 1
    if out is None:
        sys.exit("usage: corrections.py <transcript.jsonl>... [--results <results.jsonl>]... --out <corrections.json>")
    records = []
    for p in sorted(set(paths)):
        records.extend(extract(p))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=1, ensure_ascii=False)
        f.write("\n")
    per_code = collections.Counter(r["code"] for r in records)
    print(f"{len(records)} corrections from {len(set(paths))} transcripts")
    for code, n in sorted(per_code.items()):
        print(f"  {code}: {n}")


if __name__ == "__main__":
    main(sys.argv[1:])
