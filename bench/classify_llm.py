#!/usr/bin/env python3
"""Haiku pass over the tool results no rule classifies (Phase 6, Task 2 of
docs/superpowers/plans/2026-10-23-qal-phase6-der-kreis-wird-billig.md).

  python3 bench/classify_llm.py <rows.jsonl> [--max-eur 2] [--cache bench/analysis/llm-cache.jsonl]

Every text is sent once (cache by SHA-256 of its first 2000 characters); the model is
claude-haiku-5-5 at low effort; the cost is charged against QAL_MODEL_BUDGET_EUR minus what the
cache already spent, and against --max-eur for this pass. The key comes from ANTHROPIC_API_KEY or
~/.config/qal/secrets.env; nothing of it is written anywhere.
"""

import hashlib
import json
import os
import sys

BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH)
import analyze  # noqa: E402

MODEL = "claude-haiku-5-5"
EUR_PER_INPUT_TOKEN = 0.10 / 1_000_000   # Haiku 5.5, 1 EUR = 1 USD (conservative)
EUR_PER_OUTPUT_TOKEN = 0.50 / 1_000_000
DEFAULT_CACHE = os.path.join(BENCH, "analysis", "llm-cache.jsonl")
SECRETS = os.path.expanduser("~/.config/qal/secrets.env")

DESCRIPTIONS = {
    "env.guard": "a shell guard refused the command (not the agent's fault)",
    "env.tool_error": "the tool itself failed: shell error, missing file, bad path, timeout",
    "env.read_task": "the agent read the task file",
    "lang.parse": "QAL syntax error (E0100, E0102)",
    "lang.name": "QAL unknown or duplicate name (E0201, E0202)",
    "lang.type": "QAL type error (E0203..E0214)",
    "lang.effect": "QAL effect violation (E03xx)",
    "lang.contract": "QAL contract refuted or unproven (E08xx)",
    "lang.args": "bad arguments to a QAL tool (E07xx)",
    "lang.other_diag": "another QAL diagnostic",
    "lang.test_failed": "a QAL test failed",
    "lang.run_failed": "a QAL program trapped or returned an error at run time",
    "rust.compile": "Rust compile error",
    "rust.test_failed": "a Rust test failed",
    "rust.panic": "a Rust program panicked",
    "ok.define": "a successful QAL define",
    "ok.test": "a successful QAL test run",
    "ok.run": "a successful QAL program run",
    "ok.other": "other successful tool use",
    "unclassified": "none of the above fits",
}
SYSTEM = ("You classify the result of one tool call made by a coding agent in a benchmark. "
          "Answer with exactly one line: `<category> | <reason in at most 20 words>`. Categories:\n"
          + "\n".join(f"- {k}: {v}" for k, v in DESCRIPTIONS.items()))


class Budget:
    def __init__(self, limit_eur, spent_eur=0.0):
        self.limit_eur, self.spent_eur = limit_eur, spent_eur

    def charge(self, usage):
        cost = usage.get("input_tokens", 0) * EUR_PER_INPUT_TOKEN + usage.get("output_tokens", 0) * EUR_PER_OUTPUT_TOKEN
        self.spent_eur += cost
        return cost

    def exhausted(self):
        return self.spent_eur >= self.limit_eur


def classify(client, text, tool, arm):
    """(category, reason, usage) of one tool result; a reply outside the category list is
    `unclassified` but still costs."""
    user = f"Arm: {arm}. Tool: {tool}. Result (truncated):\n{text[:analyze.TEXT_LIMIT]}"
    # `output_config` and `thinking` through extra_body: the installed SDK (0.50) predates them.
    # Thinking is off (allowed at effort low): with it on, Haiku spent the whole answer budget in
    # the thinking block and returned no text.
    r = client.messages.create(model=MODEL, max_tokens=150, extra_body={"output_config": {"effort": "low"}, "thinking": {"type": "disabled"}},
                               system=SYSTEM, messages=[{"role": "user", "content": user}])
    usage = {"input_tokens": getattr(r.usage, "input_tokens", 0), "output_tokens": getattr(r.usage, "output_tokens", 0)}
    answer = "".join(getattr(b, "text", "") for b in r.content if getattr(b, "type", "") == "text").strip()
    cat, _, reason = answer.partition("|")
    cat, reason = cat.strip().strip("<>`* "), reason.strip()
    if cat not in analyze.CATEGORIES or cat.startswith("prompt.") or cat == "final":
        return "unclassified", answer[:80], usage
    return cat, reason, usage


def sha(text):
    return hashlib.sha256(text[:analyze.TEXT_LIMIT].encode("utf-8")).hexdigest()


def load_cache(path):
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    cache[d["sha256"]] = d
    return cache


def run(rows, client, budget, cache_path):
    """Classifies every distinct unclassified text of `rows` not yet in the cache; returns the
    number of texts left open (budget exhausted)."""
    # a cached `unclassified` verdict is not final: it is asked again (the last line wins when read)
    cache = {k: v for k, v in load_cache(cache_path).items() if v.get("category") != "unclassified"}
    todo = []
    seen = set()
    for row in rows:
        for u in row.get("unclassified", []) + row.get("reclassify", []):
            key = sha(u["text"])
            if key in cache or key in seen:
                continue
            seen.add(key)
            todo.append((key, u, row.get("arm", "?")))
    open_rest = 0
    with open(cache_path, "a", encoding="utf-8") as f:
        for key, u, arm in todo:
            if budget.exhausted():
                open_rest += 1
                continue
            cat, reason, usage = classify(client, u["text"], u.get("tool", "?"), arm)
            budget.charge(usage)
            f.write(json.dumps({"sha256": key, "category": cat, "reason": reason, "usage": usage}, ensure_ascii=False) + "\n")
    return open_rest


def load_secrets():
    if "ANTHROPIC_API_KEY" not in os.environ and os.path.exists(SECRETS):
        with open(SECRETS, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k, v)


def main(argv):
    rows_path, max_eur, cache_path = None, 2.0, DEFAULT_CACHE
    i = 0
    while i < len(argv):
        if argv[i] == "--max-eur":
            max_eur = float(argv[i + 1]); i += 2
        elif argv[i] == "--cache":
            cache_path = argv[i + 1]; i += 2
        else:
            rows_path = argv[i]; i += 1
    if not rows_path:
        print(__doc__, file=sys.stderr)
        return 2
    load_secrets()
    import anthropic
    client = anthropic.Anthropic()
    rows = [json.loads(l) for l in open(rows_path, encoding="utf-8") if l.strip()]
    spent_before = sum(d["usage"]["input_tokens"] * EUR_PER_INPUT_TOKEN + d["usage"]["output_tokens"] * EUR_PER_OUTPUT_TOKEN for d in load_cache(cache_path).values())
    total = float(os.environ.get("QAL_MODEL_BUDGET_EUR", "0"))
    budget = Budget(limit_eur=min(max_eur, max(total - spent_before, 0.0)))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    open_rest = run(rows, client, budget, cache_path)
    print(f"spent: {budget.spent_eur:.4f} EUR (total so far {spent_before + budget.spent_eur:.4f} of {total:.2f}), open: {open_rest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
