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
