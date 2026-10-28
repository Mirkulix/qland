# DeepSeek tool-loop teacher, 2026-10-27

Real API run with `deepseek-chat`, up to 10 tool rounds per task, budget 10 EUR.

- **6/23 tasks solved** (`01-word-count`, `04-binary-search`, `09-rle`,
  `12-sort-words`, `15-caesar`, `19-base-convert`),
- **0.601 EUR** spent,
- every kept solution passed the hidden cases; 0/0 means the agent never produced
  a final source that passed.

This is better than the plain prompt/repair generator (3/23), but still below the
agentic harness run (23/23). The remaining gap is tool-loop quality: the API model
often repeats `qal_check` instead of converging. No key is stored here.
