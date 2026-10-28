#!/usr/bin/env python3
"""Export the seed corpus as instruction/response examples for a later LoRA run.

Two kinds of examples are produced, both from data that already exists and is
verified by `qal check`:

- repair: a real failed definition, its diagnostic and the definition that fixed
  it (`corrections.jsonl`);
- test-first synthesis: a valid program's function signatures and its tests as
  the prompt, the verified program as the answer (`programs.jsonl`, only records
  with at least one `test` line).

This is a seed, not the Phase 8 corpus: ten repairs and the repository programs
are far below 5,000-20,000 model-written solutions.

  python3 bench/corpus/sft.py [--out bench/corpus/sft.jsonl]
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS = os.path.join(REPO, "bench", "corpus")


def read_jsonl(name):
    path = os.path.join(CORPUS, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def signatures_and_tests(source):
    sigs, tests = [], []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("fn ") and "=" in stripped:
            sigs.append(stripped.split("=", 1)[0].rstrip())
        elif stripped.startswith("test "):
            tests.append(stripped)
    return sigs, tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(CORPUS, "sft.jsonl"))
    args = ap.parse_args()

    examples, seen = [], set()

    for c in read_jsonl("corrections.jsonl"):
        prompt = f"Diagnostic {c['code']}: {c['message']}\n\nCode:\n{c['before']}"
        messages = [
            {"role": "system", "content": "You repair QAL code. Return only the corrected definition, without prose."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": c["after"]},
        ]
        key = json.dumps(messages, sort_keys=True)
        if key not in seen:
            seen.add(key)
            examples.append({"kind": "repair", "messages": messages})

    for p in read_jsonl("programs.jsonl"):
        if p["status"] != "ok":
            continue
        sigs, tests = signatures_and_tests(p["source"])
        if not tests or not sigs:
            continue
        prompt = ("Write QAL definitions with these signatures so that the tests pass. "
                  "Return only QAL source.\n\nSignatures:\n" + "\n".join(sigs) +
                  "\n\nTests:\n" + "\n".join(tests))
        messages = [
            {"role": "system", "content": "You write QAL, a small typed language. Return only source."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": p["source"]},
        ]
        key = json.dumps(messages, sort_keys=True)
        if key not in seen:
            seen.add(key)
            examples.append({"kind": "synthesis", "messages": messages})

    with open(args.out, "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    stats = {"examples": len(examples),
             "repair": sum(1 for e in examples if e["kind"] == "repair"),
             "synthesis": sum(1 for e in examples if e["kind"] == "synthesis")}
    print(json.dumps(stats, indent=2))
    if stats["repair"] == 0 or stats["synthesis"] == 0:
        sys.exit("sft: one of the two example kinds is empty; build the corpus first")


if __name__ == "__main__":
    main()
