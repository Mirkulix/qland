# MLX LoRA adapter, 2026-10-27

TinyLlama-1.1B-Chat + rank-8 LoRA, 4,242 train / 223 validation examples from
`bench/corpus/sft.jsonl`, 530 iterations (one epoch), batch 8, max length 256.

- final train loss **0.164**, final validation loss **0.171** (41 minutes, Apple GPU),
- proxy evaluation on 50 held-out SFT prompts: **base 0/50, adapter 7/50** pass `qal check` + `qal test`.
- check-only on all 23 benchmark tasks: **base 7/23, adapter 2/23** valid — specialization hurts task generalization.

This is a seed adapter and a **proxy**, not the Phase 8 three-arm benchmark. The acceptance
(trained ≤0.8x, untrained top model ≤1.2x on the agent benchmark) is still open.
