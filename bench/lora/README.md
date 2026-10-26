# LoRA scaffold (Phase 8, item 2)

This directory holds the Phase 8 training scaffold. **A one-step smoke run was executed on
2026-10-26; a full training/evaluation run has not.** The seed data is `bench/corpus/sft.jsonl`:
105 examples (10 repairs, 95 test-first syntheses), about 27k approximate tokens. A model trained
on that alone is a smoke test, not the Phase 8 result.

## Smoke run (2026-10-26)

`train.py --cpu --max-steps 1 --max-length 256` loaded TinyLlama-1.1B-Chat, attached a rank-8 LoRA
(1,126,400 trainable params, 0.1023 %) and ran one step on all 105 examples: loss **1.9108**,
11.8 s, adapter written to `/tmp/qal-lora-cpu` (not committed). The MPS path hung during model
loading on this machine, hence `--cpu`. This proves the harness, not the model quality.

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

## What is missing

1. 5,000–20,000 model-written, `qal check`-verified programs (the seed has 933, mostly from the
   repository).
2. More real error/correction traces (10 today).
3. The actual LoRA run and the three-arm comparison harness.
