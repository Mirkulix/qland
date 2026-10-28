# QAL vs Python pilot (5 tasks)

Same five stdin tasks, same DeepSeek model, max attempts with the failure text fed
back. The QAL arm gets the reference plus a complete `fn main` example; Python gets
a standalone-script instruction.

## Results

| Run | QAL | Python |
|---|---|---|
| first (zero-shot, 3 attempts) | **0/5** | **5/5** |
| grammar-first prompt + example (4 attempts) | **0/5** | **5/5** |
| `deepseek-reasoner` | **0/5** | **5/5** |

Raw: `results.json` (last run).

## Failure modes

- Python syntax in QAL (`&`, single quotes, `def`),
- missing `fn main` or a top-level `{` instead of `=`,
- `let` without `in`,
- reasoner often returns no `main` at all (`E0510`).

The parser errors are shallow, but the model does not repair them even with
structured diagnostics. This is **not** a semantics problem; it is a **surface
syntax** problem.

## Conclusion

A general LLM cannot write QAL zero-shot for trivial stdin tasks. Python wins
because it is the model's strongest prior, not because Python is more correct.

The next step is an **agent-native surface**: let the model emit a JSON AST or
tool calls that the compiler lowers to QAL. Human QAL syntax remains; the AST is
the machine surface. See `docs/QAL-AGENT-SURFACE.md`.

This is a 5-task pilot, not the 100-task comparison.
