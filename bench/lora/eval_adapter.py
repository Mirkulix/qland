#!/usr/bin/env python3
"""Evaluate a LoRA adapter on the SFT prompts (a proxy, not the Phase 8 benchmark).

The real Phase 8 acceptance is the agent benchmark (trained model ≤0.8x, untrained top
model ≤1.2x). This script only asks: can the adapter generate QAL that `qal check` and
`qal test` accept for a sample of held-out prompts? It is a smoke-level signal.

  python3 bench/lora/eval_adapter.py --model TinyLlama/... --adapter /tmp/qal-lora-4465 --limit 10
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA = os.path.join(REPO, "bench", "corpus", "sft.jsonl")
DEFAULT_QAL = os.path.join(REPO, "qal", "target", "release", "qal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--qal", default=DEFAULT_QAL)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--compare-base", action="store_true", help="also evaluate the unadapted base model")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
    random.Random(1).shuffle(rows)
    rows = rows[:args.limit]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, trust_remote_code=True)
    base.eval()

    def evaluate(model):
        passed = 0
        results = []
        for row in rows:
            messages = row["messages"][:2]  # system + user; the assistant is the target
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            ok, why = check_source(text, args.qal)
            passed += ok
            results.append({"prompt": messages[-1]["content"][:200], "ok": ok, "why": why, "generated": text[:1000]})
        return {"examples": len(rows), "passed": passed, "rate": passed / len(rows) if rows else 0}, results

    out = {}
    if args.compare_base:
        out["base"], out["base_results"] = evaluate(base)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    out["adapter"], out["adapter_results"] = evaluate(model)
    out["note"] = "proxy eval; not the Phase 8 benchmark"
    print(json.dumps({k: out[k] for k in ("base", "adapter") if k in out} | {"note": out["note"]}, indent=2))
    with open(os.path.join(args.adapter, "eval_proxy.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    sys.exit(0 if out["adapter"]["passed"] == len(rows) else 1)


def check_source(text, qal):
    with tempfile.TemporaryDirectory(prefix="qal-eval-") as d:
        path = os.path.join(d, "out.qal")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        c = subprocess.run([qal, "check", path], capture_output=True, text=True, timeout=20)
        if c.returncode != 0:
            return False, "check"
        t = subprocess.run([qal, "test", path], capture_output=True, text=True, timeout=30)
        return (t.returncode == 0), ("ok" if t.returncode == 0 else "test")


if __name__ == "__main__":
    main()
