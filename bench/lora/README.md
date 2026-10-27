# LoRA scaffold (Phase 8, item 2)

This directory holds the Phase 8 training scaffold. **A one-step smoke run and one full seed epoch
were executed on 2026-10-26/27; no evaluation on the benchmark has happened.** The data is
`bench/corpus/sft.jsonl`: **4,465 examples** (10 repairs, 4,455 test-first syntheses), about 760k
approximate tokens. The 105-example epoch below was the earlier smoke test; the full-corpus run
and the three-arm evaluation are still missing.

## Runs (2026-10-26/27, 105-example seed)

- **One-step smoke:** `--cpu --max-steps 1 --max-length 256`, loss **1.9108**, 11.8 s.
- **Full seed epoch:** `--cpu --max-length 512`, 105 examples, 1 epoch, **train_loss 1.5674**,
  360.9 s, adapter and tokenizer in `/tmp/qal-lora-full` (not committed). Rank-8 LoRA,
  1,126,400 trainable params (0.1023 %).
- The MPS path hung during model loading on this machine; `--cpu` is explicit. Training loss is
  **not** the Phase 8 acceptance (benchmark ≤0.8× trained / ≤1.2× untrained top model); that
  evaluation, and the 5,000–20,000-program corpus, are still missing.

## Check the data

```sh
python3 bench/corpus/sft.py
python3 bench/lora/check_sft.py
```

The checker validates the chat format, roles, empty content, duplicates and the per-example
length; it prints the counts and an approximate token total.

## Planned run

- Base model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (cached locally) for the smoke test; an open
  model with a stronger code prior for the real run.
- LoRA: rank 8–16, alpha 16–32, target the attention projections, sequence length 1024,
  one to three epochs, learning rate 1e-4 to 2e-4.
- Environment: `transformers`, `peft`, `accelerate` (not installed in the current Python yet) and
  Apple MPS or CUDA.
- Output: `bench/lora/out/` (adapter, tokenizer config, training log, exact data hash).

The run must be evaluated with the Phase 6 benchmark, not by training loss:

| Arm | Requirement |
|---|---|
| trained open model | solves the tasks at ≤ 0.8× the tokens of the untrained baseline |
| untrained top model | ≤ 1.2× the Rust tokens |

Those numbers are the Phase 8 acceptance criteria. Until they are measured, Phase 8 stays in
progress, regardless of how good the adapter's training loss looks.

## Proxy evaluation

After training, `eval_adapter.py --adapter <dir> --limit 10` generates QAL from held-out SFT
prompts and keeps a sample only if `qal check` and `qal test` accept it. This is a **proxy**, not
the Phase 8 benchmark (the three-arm agent evaluation is still missing).

## What is missing

1. 5,000–20,000 model-written, `qal check`-verified programs (the seed has 933, mostly from the
   repository).
2. More real error/correction traces (10 today).
3. The actual LoRA run and the three-arm comparison harness.
