#!/usr/bin/env python3
"""Validate the SFT seed before a LoRA run (Phase 8, item 2).

This does not train. It checks the chat format, roles, empty texts, duplicates and
prints approximate token counts (4 characters per token) so a training run and a
context limit can be planned. The real tokenizer is applied by train.py.

  python3 bench/lora/check_sft.py [--data bench/corpus/sft.jsonl] [--max-chars N]
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT = os.path.join(REPO, "bench", "corpus", "sft.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT)
    ap.add_argument("--max-chars", type=int, default=32000)
    args = ap.parse_args()
    if not os.path.exists(args.data):
        sys.exit(f"check_sft: {args.data} is missing; run python3 bench/corpus/sft.py")

    rows, seen, bad = [], set(), []
    with open(args.data, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                bad.append(f"line {n}: {e}")
                continue
            msgs = row.get("messages")
            if not isinstance(msgs, list) or [m.get("role") for m in msgs] != ["system", "user", "assistant"]:
                bad.append(f"line {n}: roles are not system/user/assistant")
                continue
            if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in msgs):
                bad.append(f"line {n}: empty content")
                continue
            text = "\n".join(m["content"] for m in msgs)
            if len(text) > args.max_chars:
                bad.append(f"line {n}: {len(text)} characters, above --max-chars {args.max_chars}")
                continue
            key = json.dumps(msgs, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            rows.append((row.get("kind", "?"), len(text)))

    kinds = {}
    for kind, _ in rows:
        kinds[kind] = kinds.get(kind, 0) + 1
    total_chars = sum(n for _, n in rows)
    print(json.dumps({
        "examples": len(rows),
        "kinds": kinds,
        "chars": total_chars,
        "approx_tokens": total_chars // 4,
        "problems": bad,
    }, indent=2))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
