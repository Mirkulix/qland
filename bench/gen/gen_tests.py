#!/usr/bin/env python3
"""Writes bench/tasks/<id>/{task.md,tests.json} from statements.py, cases.py and oracles.py.

Deterministic (fixed seeds): running it again reproduces the files byte for byte. The hidden tests'
expected outputs come from the oracles, a third implementation besides the reference solutions.
Agents must never see this directory or tests.json (see bench/prompts/ and run.py audit)."""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cases  # noqa: E402
import oracles  # noqa: E402
from statements import TASKS  # noqa: E402

BENCH = os.path.dirname(HERE)

def fence(s):
    body = s if s.endswith("\n") or s == "" else s + "\n"
    return "```\n" + body + "```\n"

def example(c):
    md = [f"### Example `{c['name']}`\n"]
    md.append("Arguments: " + (" ".join(f"`{a}`" for a in c["args"]) if c["args"] else "none") + "\n")
    for path, content in c["files"].items():
        md.append(f"File `{path}`:\n\n" + fence(content))
    if c["stdin"] == "":
        md.append("Standard input: empty\n")
    else:
        md.append("Standard input:\n\n" + fence(c["stdin"]))
    md.append(f"Standard output (exit code {c['exit_code']}):\n\n" + (fence(c["stdout"]) if c["stdout"] else "empty\n"))
    return "\n".join(md)

# The stated limits (statements.py), checked on every hidden input: total bytes, lines, characters
# per line (input and output).
LIMITS = {
    "01-word-count": (100_000, None, None), "02-roman": (None, 1000, None), "03-csv-sum": (None, 1001, None),
    "04-binary-search": (None, 501, None), "05-brackets": (60_000, None, 3000), "06-bank": (None, 1000, None),
    "07-tokenizer": (None, 50, 200), "08-expr-eval": (None, 200, 300), "09-rle": (None, 250, 200), "10-file-stats": (50_000, None, None),
    "11-prime-factors": (None, 200, None), "12-sort-words": (100_000, None, None), "13-word-freq": (100_000, None, None),
    "14-bst": (None, None, None), "15-caesar": (None, 500, 100), "16-anagrams": (None, 2000, None), "17-word-wrap": (50_000, None, None),
    "18-date-diff": (None, 500, None), "19-base-convert": (None, 500, None), "20-ini-query": (None, 1000, None),
}

def within_limits(tid, c):
    max_bytes, max_lines, max_chars = LIMITS[tid]
    for text in [c["stdin"]] + list(c["files"].values()):
        lines = oracles.lines_of(text)
        assert max_bytes is None or len(text.encode()) <= max_bytes, (tid, c["name"], "bytes")
        assert max_lines is None or len(lines) <= max_lines, (tid, c["name"], "lines")
        assert max_chars is None or all(len(l) <= max_chars for l in lines), (tid, c["name"], "line length")
    if tid == "09-rle":
        assert all(len(l) <= 200 for l in oracles.lines_of(c["stdout"])), (tid, c["name"], "output line length")
    if tid == "14-bst":
        nums = [int(x) for x in c["stdin"].split()]
        assert len(nums) <= 2000 and all(abs(v) <= 10 ** 6 for v in nums), (tid, c["name"])
    if tid == "11-prime-factors":
        assert sum(1 for l in oracles.lines_of(c["stdin"]) if l.strip().isdigit() and int(l.strip()) > 10 ** 9) <= 10, (tid, c["name"])
    if tid in ("12-sort-words", "13-word-freq"):
        assert len(set(oracles.letter_words(c["stdin"]))) <= 3000, (tid, c["name"])

def main():
    for tid, title, fs, statement, limits in TASKS:
        fn = tid[3:].replace("-", "_")
        cs = getattr(cases, fn)() + getattr(cases, "limit_" + fn, list)()
        names = [c["name"] for c in cs]
        assert len(set(names)) == len(names), tid
        for c in cs:
            c["stdout"], c["exit_code"] = getattr(oracles, fn)(c["args"], c["stdin"], c["files"])
            assert len(c["stdout"].encode()) < 60_000, (tid, c["name"], "output must stay below the 64 KiB limit")
            within_limits(tid, c)
            if not fs:
                assert not c["files"], (tid, c["name"])
        d = os.path.join(BENCH, "tasks", tid)
        os.makedirs(d, exist_ok=True)
        md = [f"# {title}\n", statement.strip() + "\n", f"**Limits:** {limits}\n", "## Examples\n"]
        md += [example(c) for c in cs if c["public"]]
        with open(os.path.join(d, "task.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(md))
        meta = {"id": tid, "title": title, "effects": ["io", "fs"] if fs else ["io"], "cases": cs}
        with open(os.path.join(d, "tests.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(f"{tid}: {len(cs)} cases ({sum(c['public'] for c in cs)} public)")

main()
