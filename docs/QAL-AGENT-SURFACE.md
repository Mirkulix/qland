# QAL agent surface: JSON AST before source

## Finding

The 5-task QAL vs Python pilot (`bench/compare/`) is 0/5 for QAL and 5/5 for
Python. The failures are syntax, not semantics:

- Python idioms (`&`, `'...'`, `def`),
- missing `fn main`,
- `let` without `in`,
- a top-level `{` instead of `=`.

Adding a complete example and a grammar-first prompt did not fix it; the model
still reaches for Python. The bottleneck is the **text surface**, not the
language.

## Proposal

Give QAL two surfaces over one semantics:

1. **Human surface:** the existing QAL text.
2. **Agent surface:** a JSON AST (or structured tool calls) that lowers to the
   same canonical definitions.

```
agent -> JSON AST -> compiler -> canonical QAL -> hash/proofs/tests
human -> QAL text  -> compiler -> canonical QAL -> hash/proofs/tests
```

The AST must be:

- typed and closed (no free-form expressions),
- versioned,
- canonicalized to the same hash as the text form,
- validated by the same checker/prover.

## Minimal AST

```json
{
  "fn": "main",
  "params": [{"name": "args", "type": "List[Str]"}],
  "effects": ["io"],
  "returns": "Int",
  "body": {
    "let": {"name": "text", "value": {"call": "read_all", "args": []}},
    "in": {"call": "print", "args": [{"call": "from_int", "args": [{"op": "+", "args": [1, 2]}]}]}
  }
}
```

This is still a sketch; the important part is that the model never has to invent
QAL punctuation.

## Why it matters

- removes the largest measured failure class (surface syntax),
- gives structured repair (missing field/type, not "expected `in`"),
- token-efficient,
- keeps proofs/tests/hashes identical,
- still one language, two presentations.

## Next

1. define the AST schema for the 5 pilot tasks,
2. implement a small `qal-ast` lowering in Python for the pilot,
3. compare agent success on JSON AST vs QAL text,
4. if it works, move the lowering into the compiler.

## Measured result (5 tasks)

| Surface | Solved |
|---|---|
| QAL text, zero-shot | 0/5 |
| QAL text, grammar-first | 0/5 |
| QAL text, reasoner | 0/5 |
| **JSON AST → QAL** | **5/5** |
| Python | 5/5 |

The AST arm uses `bench/compare/prelude.qal` helpers and only chooses/composes
calls; it is not a general synthesis result. But it isolates the surface
variable: same model, same tasks, same tests, **only the output format changes**.
That is the evidence that an agent-native surface is the right next layer.

## Implemented surface (2026-10-28)

`qal ast <file.json>` and `qal define <file.json> --ast` now support literals,
list literals, vars, calls, ops, fields, let, if, match, records, `ast_version`,
`types`, `effects`, `returns` and `tests`. `qal define --ast` uses the normal store path, so the AST gets the same
type check, proofs, tests and hash as text. Verified: `sum_ok` test and `main`
run -> `5` from a single JSON file.

## 100-task pilot

| Surface | Solved |
|---|---|
| QAL JSON AST | 80/100 |
| Python | 100/100 |

The 20 AST failures are wrong call choices (`Int` as a function, wrong helper),
not syntax errors. The surface works; the model's composition is the remaining
gap. Ten families × ten variants; helper prelude still task-specific.
