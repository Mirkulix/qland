# QAL seed corpus (Phase 8, item 1)

This directory holds the Phase 8 corpus: the repository programs plus DeepSeek-written programs,
each with the exact result of `qal check`. The 5,000-program lower bound is met; the LoRA run and
the three-arm evaluation are still missing.

## Build

From the repository root, with a release binary:

```sh
./install.sh --no-build          # or cargo build --release -p qal-cli
python3 bench/corpus/build.py
```

The builder reads `qal/selfhost/corpus/*.qal`, `bench/solutions/qal/*.qal` and
`qal/examples/*.qal`, deduplicates by SHA-256 and runs `qal check` on every file. It writes:

- `programs.jsonl` — one record per program:
  `{id, origin, status: "ok"|"error", diagnostics, source}`.
  `status: "error"` is intentional: the extracted corpus contains programs that must fail,
  and those diagnostics are the error side of a training pair.
- `summary.json` — counts and the Phase 8 target.

The 2026-10-27 corpus has **5,323 programs: 4,986 `ok`, 337 with diagnostics**, plus **10 real
correction traces** in `corrections.jsonl`. **4,390** are model-written by DeepSeek (4,367 corpus + 23 teacher solutions) and were kept
only after both `qal check` and `qal test` passed; the rest came from the repository. This meets
the 5,000-program lower bound of the Phase 8 specification. The builder takes about 3.5 minutes
for the repository part; the model records are preserved across a rebuild.

## Verify

```sh
python3 bench/corpus/verify.py            # 50 evenly spaced programs
python3 bench/corpus/verify.py --all      # all programs, about 3.5 minutes
```

The checker compares only the status (`ok` vs. at least one diagnostic), not the diagnostic
text, so a better checker does not invalidate the corpus.

## Generation provenance

`generation-2026-10-27.json` records the 20 DeepSeek batches: 4,400 raw programs, 4,367 kept
after `qal check` + `qal test`, 33 rejected, SHA-256 dedup, budget 10 EUR (no metering in this
environment).

## SFT seed

```sh
python3 bench/corpus/sft.py
```

Writes `sft.jsonl` with **4,465 instruction/response examples**; `sft_tasks.jsonl` additionally holds 23 task->solution teacher examples: 10 repairs (diagnostic + failed code
-> the fix) and 4,455 test-first synthesis examples (signatures + tests -> the verified source).
A full LoRA run and the three-arm evaluation are still missing.

## What is still missing for Phase 8

1. **5,000–20,000 verified programs**, written by strong models from the task types, the
   standard library and the Phase 7 use case; the checker filters them.
2. **Error/correction traces**: a failed attempt, its diagnostics, and the version that
   passed. `qal_recall` already stores such pairs in a project store; export them into a
   `corrections.jsonl` next to `programs.jsonl`.
3. **The LoRA run and the three-arm comparison** (untrained/trained open model vs. the top
   model), with its own harness. None of these artifacts are in this repository yet.

## Format for a model-written program

Add one JSON object per line to `programs.jsonl` (or emit it from your generator):

```json
{"id":"<sha256 of source, first 16 hex>","origin":"model:<model>/<run>","status":"ok",
 "diagnostics":[],"source":"fn ..."}
```

Only store a program as `ok` after `qal check` returned `{"ok":true}` on that exact source.
For a failed attempt, keep `status:"error"` and the diagnostics; the pair with the corrected
source is what a correction trace needs.
