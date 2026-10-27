# DeepSeek teacher run, 2026-10-27

23 DeepSeek subagents each solved one QAL benchmark task end-to-end with the QAL CLI
(`init`, `check`, `define`, `test`, `run`). Independent validation with `bench/check.py`:

- **23/23 tasks solved**, **232/232 hidden cases** passed,
- every solution is in `solutions/<task>.qal`,
- raw results in `results.json`.

This is the **teacher** baseline: a strong model can write QAL that passes the benchmark.
It is not the Phase 8 student result (TinyLlama base 7/23 check-valid, adapter 2/23; hidden
cases 0/5). It also provides task-oriented data for a later distillation run: each task has a
verified solution, unlike the signature/test SFT corpus.

Reproduce validation:

```sh
python3 - <<'PY'
import json, sys; sys.path.insert(0, "bench")
import check
for r in json.load(open("bench/teacher-2026-10-27/results.json")):
    print(r["task"], r["passed"], "/", r["total"])
PY
```
