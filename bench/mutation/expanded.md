# Expanded mutation benchmark (108 mutants)

`expand.py` applies six source-level changes to the 23 teacher solutions:

| Mutator | Count |
|---|---|
| `+` → `-`, `-` → `+`, `*` → `+`, `==` → `!=` | 4 per solution |
| `true` → `false`, `0` → `1` (first occurrence) | 2 per solution |

A mutant counts only if it **compiles** and the task's hidden tests fail on it.
Compile errors are invalid; all hidden tests passing is `escaped` (possibly an
equivalent mutant or a comment/string hit).

## Result

```
caught  80 / 108
escaped 28 / 108
invalid  0
```

80 of 108 compiled mutants are caught by the hidden tests. The 28 escapes are
concentrated in `zero_to_one` and `true_to_false`, which often hit comments,
string literals or equivalent constants; they are not all semantic bugs. The
operator mutants are the meaningful subset.

## What this measures

- QAL's hidden-test suite catches **80/108** mechanical mutations of the teacher
  solutions;
- it does **not** yet prove QAL beats Python/TypeScript;
- it identifies where tests/contracts are weak (e.g. `11-prime-factors`,
  `19-base-convert`, `22-expense-approval-contracts`).

## Next

- restrict mutators to code tokens (skip comments/strings) and classify
  equivalent mutants;
- add contract-specific mutants (remove a `req`/`ens`, weaken an invariant);
- run the same 100-task suite against Python/TypeScript agents for the real
  language comparison.
