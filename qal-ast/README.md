# qal-ast – agent-native JSON surface (prototype)

`lower.py` turns a JSON expression AST into QAL source. It is not a second
language: it targets the same QAL compiler, hashes, contracts and tests.

Supported nodes: literals (`int`, `str`, `bool`), `var`, `call`, `op`, `field`,
`let`, `if`, `match`, `record`. The root also accepts `effects` (default `["io"]`),
`returns` (default `Int`) and `tests` (`[{name, body}]`).

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
semantics. The lowering is now also available in the compiler:

```sh
qal ast qal-ast/example.json > main.qal
cat bench/compare/prelude.qal main.qal > full.qal
qal check full.qal

# or direct into the store, no intermediate file:
qal define qal-ast/simple.json --ast
qal define qal-ast/tests.json --ast   # main + test sum_ok in one call
```

`qal ast` is the first native agent surface; Python `lower.py` remains the
reference for the pilot. Next: lists, polymorphism, effects and `qal define --ast`.
