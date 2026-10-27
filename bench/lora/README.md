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

## MLX run (faster on Apple Silicon)

The PyTorch CPU run is slow (hours). MLX uses the Apple GPU:

```sh
# convert the SFT chat file once
python3 - <<'EOF'
import json, random
rows=[json.loads(l) for l in open("bench/corpus/sft.jsonl")]
random.Random(1).shuffle(rows); n=max(1,int(len(rows)*0.05))
for name,data in (("train",rows[n:]),("valid",rows[:n])):
    with open(f"/tmp/qal-mlx-data/{name}.jsonl","w") as f:
        for r in data: f.write(json.dumps({"messages":r["messages"]})+"\n")
EOF

python3 -m mlx_lm lora --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train --data /tmp/qal-mlx-data --iters 530 --batch-size 8 --max-seq-length 256 \
  --adapter-path /tmp/qal-mlx-adapter --mask-prompt --steps-per-report 25 --steps-per-eval 100
```

Measured on this machine: ~0.21 iterations/s; the run finished in **41 minutes**: final **val loss
0.171**, train loss 0.164, 530 iters, batch 8, max-length 256 (longer sequences truncated). The
adapter is in `/tmp/qal-mlx-adapter` (not committed). Proxy evaluation on 10 held-out prompts:
**base model 0/10, adapter 3/10** pass `qal check` + `qal test`. On 50 held-out prompts: **base
0/50, adapter 7/50**. The adapter and proxy results are stored in
`bench/lora/mlx-adapter-2026-10-27/`. A single-shot run over the first 5 benchmark tasks (hidden
cases, no tools/iteration) solved **0/5 for both** base and adapter (0/46 cases); the full 23-task
hidden-case run was stopped because bad programs spend up to 9×10 s per case. A fast check-only
run over all 23 tasks: **base 7/23 valid, adapter 2/23 valid** — the adapter fits the SFT prompts
but generalizes worse to benchmark tasks. This is a proxy, not the Phase 8 agent benchmark.

## Proxy evaluation

After training, `eval_adapter.py --adapter <dir> --limit 10` generates QAL from held-out SFT
prompts and keeps a sample only if `qal check` and `qal test` accept it. This is a **proxy**, not
the Phase 8 benchmark (the three-arm agent evaluation is still missing). `eval_mlx.py` is the MLX variant (loads the adapter and generates on the Apple GPU). Add `--compare-base` to run the same prompts through the unadapted base model.

## What is missing

1. 5,000–20,000 model-written, `qal check`-verified programs (the seed has 933, mostly from the
   repository).
2. More real error/correction traces (10 today).
3. The actual LoRA run and the three-arm comparison harness.
