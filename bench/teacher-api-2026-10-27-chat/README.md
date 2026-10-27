# DeepSeek API teacher run, 2026-10-27

Real API run with `deepseek-chat`, budget 10 EUR, up to 4 repair attempts per task:

- **3/23 tasks solved** (`02-roman`, `03-csv-sum`, `19-base-convert`),
- **0.265 EUR** spent (well inside the budget),
- 3 solutions in `solutions/`, full usage/attempts in `results.json`.

This is the **raw API** path (prompt -> code -> hidden cases -> error feedback). It is much
weaker than the agentic DeepSeek run (`bench/teacher-2026-10-27/`: 23/23), because the agent
could use the QAL CLI and iterate with tools, not just receive an error string.

No API key is stored in this directory or anywhere in the repository.
