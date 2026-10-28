# QAL vs Python pilot (5 tasks)

Same five stdin tasks, same DeepSeek model. Three QAL surfaces compared:

| Arm | Solved |
|---|---|
| QAL text, zero-shot | **0 / 5** |
| QAL text, grammar-first + example | **0 / 5** |
| QAL text, `deepseek-reasoner` | **0 / 5** |
| **QAL JSON AST** | **5 / 5** |
| Python | **5 / 5** |

Raw: `results.json`, `results-ast.json`.

## The finding

The model cannot write QAL **text** zero-shot; it falls back to Python syntax
(`&`, `'...'`, `def`), misses `fn main`, or writes `let` without `in`.

But it can emit a JSON expression AST, which a deterministic lowering turns into
QAL:

```json
{"expr": {"call": "sum_all", "args": [{"call": "read_all", "args": []}]}}
```

This is lowered to:

```qal
fn main(args: List[Str]) -> Int !io = { print(from_int(sum_all(read_all()))); 0 }
```

The AST arm solves **5/5**, equal to Python, while QAL text is **0/5**.

## Caveat

The pilot provides a small task helper library (`bench/compare/prelude.qal`):
`sum_all`, `fact_first`, `count_words`, `max_all`, `reverse_line`. The AST model
chooses and composes calls; it does not write the helper bodies. This is a
surface test, not a general QAL synthesis result. It is five tasks, not 100.

## What it means

- The bottleneck was the **surface syntax**, not the semantics.
- An agent-native AST is a viable second surface over the same compiler.
- Next: generalize the AST schema and freeze the helper library as verified
  packages, then rerun on 100 tasks.

## Reproduce

```sh
python3 bench/compare/run.py       # QAL text vs Python
python3 bench/compare/ast_run.py   # QAL AST vs Python tasks
```
