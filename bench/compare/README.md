# QAL vs Python pilot (5 tasks)

Same five stdin tasks, same DeepSeek `deepseek-chat` model, max three attempts with
the failure text fed back. QAL arm is prompted with the QAL reference; Python arm
gets a standalone-script instruction.

## Result

| Arm | Solved |
|---|---|
| **QAL** | **0 / 5** |
| **Python** | **5 / 5** |

Raw: `results.json`.

## Failure modes (QAL)

- Python syntax leaked into QAL (`&`, single quotes),
- code was emitted without a QAL `fn main` (top-level statements),
- parse errors at line 1/3, not semantic QAL errors.

This is the honest baseline: **a general model does not write QAL zero-shot**, even
for trivial stdin tasks. Python wins because the model has seen it millions of
times and the language accepts the idioms it reaches for.

## What it means

- The language is not yet agent-native enough for a plain prompt;
- the tool-loop/reference/examples must carry the model, not its Python habits;
- this is a 5-task pilot, not the 100-task comparison;
- the next step is a grammar-first prompt + structured repair loop, then the
  same tasks again.

## Reproduce

```sh
python3 bench/compare/run.py
```
