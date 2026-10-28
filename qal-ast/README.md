# qal-ast – agent-native JSON surface (prototype)

`lower.py` turns a JSON expression AST into QAL source. It is not a second
language: it targets the same QAL compiler, hashes, contracts and tests.

Supported nodes: literals (`int`, `str`, `bool`), `var`, `call`, `op`, `field`,
`let`, `if`, `match`, `record`.

```sh
python3 qal-ast/lower.py qal-ast/example.json \
  --prelude bench/compare/prelude.qal --check > /tmp/main.qal
```

## Tests

`test_lower.py` covers let/op, if, match (Option) and a record field. All four
lower to QAL that compiles and produces the expected stdout.

The 5-task pilot (`bench/compare/ast_run.py`) uses this surface: the model emits
only JSON; the lowering, compiler and hidden tests stay deterministic.

## Why

The measured failure of QAL text was surface syntax (`let` without `in`, Python
idioms, missing `fn main`). The AST removes that class while keeping the same
semantics. Next: generalise the schema (lists, polymorphism, effects) and move
the lowering into the compiler as `qal define --ast`.
