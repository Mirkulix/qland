#!/usr/bin/env python3
"""LoRA smoke run for the Phase 8 SFT seed.

This is a harness, not the Phase 8 result: the seed has 105 examples, far below
the 5,000-20,000 target, and no three-arm evaluation. It trains a LoRA adapter on
a local/HF causal model and writes the adapter plus a small log.

  python3 bench/lora/train.py --dry-run
  python3 bench/lora/train.py --model ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/<hash> \
      --out bench/lora/out --epochs 1 --max-length 1024
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA = os.path.join(REPO, "bench", "corpus", "sft.jsonl")


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def render(messages, tokenizer):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=os.path.join(REPO, "bench", "lora", "out"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--cpu", action="store_true", help="force CPU (MPS model loading hung on this machine)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.data)
    if not rows:
        sys.exit("train: no examples")
    print(json.dumps({"examples": len(rows), "model": args.model, "dry_run": args.dry_run}, indent=2))
    if args.dry_run:
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                              Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = [render(r["messages"], tok) for r in rows]

    def encode(batch):
        out = tok(batch["text"], truncation=True, max_length=args.max_length)
        out["labels"] = [ids[:] for ids in out["input_ids"]]
        return out

    ds = Dataset.from_dict({"text": texts}).map(encode, batched=True, remove_columns=["text"])
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, trust_remote_code=True)
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                        target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, config)
    if args.cpu:
        model.to("cpu")
    model.print_trainable_parameters()

    os.makedirs(args.out, exist_ok=True)
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        use_cpu=args.cpu,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False))
    result = trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "train_result.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": result.metrics, "examples": len(rows), "model": args.model}, f, indent=2)
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
