# QAL seed corpus (Phase 8, item 1)

This directory is the repository-side seed for the Phase 8 corpus. It is **not** the
5,000–20,000-program corpus of the specification: it indexes the QAL programs that already
exist in the repository, with the exact result of `qal check`. Model-written solutions and
the traces of a LoRA run are added to the same format later.

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

The last run (2026-10-26) produced **933 programs: 596 ok, 337 with diagnostics**. That is a
seed, not the target. The builder takes about 3.5 minutes because it starts the CLI per file.

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
