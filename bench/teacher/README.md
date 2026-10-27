# DeepSeek teacher (Phase 8, API path)

`deepseek_generate.py` uses the **DeepSeek API** (OpenAI-compatible
`/chat/completions`) to generate QAL solutions for the benchmark tasks and keeps a
solution only after `qal check` **and** the task's hidden cases pass
(`bench/check.py`).

## Key

Create or append to `~/.config/qal/secrets.env` (mode `600`, never in the repo):

```sh
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-coder
QAL_MODEL_BUDGET_EUR=10
# optional price overrides, EUR per million tokens:
# DEEPSEEK_INPUT_EUR_PER_MTOK=0.30
# DEEPSEEK_OUTPUT_EUR_PER_MTOK=1.20
```

`DEEPSEEK_MODEL` is whatever the account exposes (`deepseek-coder`, `deepseek-chat`, ...).
The budget is checked after every task; the run stops when it is exhausted.

## Run

```sh
python3 bench/teacher/deepseek_generate.py --dry-run     # no request, prints model/budget
python3 bench/teacher/deepseek_generate.py --limit 5    # first five tasks
python3 bench/teacher/deepseek_generate.py              # all 23 tasks
```

Output: `bench/teacher-api-<date>/solutions/<task>.qal` plus `results.json` with
pass counts, token usage and cost. A failed validation is not copied.

## Verified without a real key

- `--dry-run` reports the missing key instead of sending a request.
- A local fake `/chat/completions` endpoint (returning the known `01-word-count`
  teacher solution) was used to exercise the full path: request → strip code →
  hidden cases → `9/9` → only then stored. The fake server and output were not
  committed.

## Difference to the local LoRA

This is the **teacher** path: DeepSeek generates; the repo validates. It needs a key and
network. The local LoRA path (`bench/lora/`) trains a student without an API key; its
current results are in `bench/lora/README.md`.
