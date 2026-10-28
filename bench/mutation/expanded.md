# Expanded mutation benchmark (134 mutants)

`expand.py` applies nine source-level changes to the 23 teacher solutions,
only outside comments and string literals:

| Class | Mutators |
|---|---|
| operators | `+`→`-`, `-`→`+`, `*`→`+`, `==`→`!=` |
| constants | `true`→`false`, `0`→`1` |
| contracts | drop first `req`, drop first `ens`, weaken `>=`→`>` |

A mutant counts only if it **compiles**; `caught` means the hidden tests fail on
it, `escaped` means every hidden test still passes.

## Result

```
caught  112 / 134
escaped  22 / 134
invalid   0
```

**112/134 (83.6%)** compiled mutants are caught by the hidden tests.

The 22 escapes split into two groups:

1. **contract mutants** (`drop_req`, `drop_ens`, `weaken_ge`) in
   `09-rle`, `22-expense-approval-contracts`, `23-expense-approval-change`,
   `01-word-count`, `20-ini-query`, `02-roman`. The hidden tests do not fail
   because removing a contract removes an obligation, not a behavior. Tests
   alone cannot catch this; the diff/policy must require that contracts are not
   silently deleted.
2. **constant/operator mutants** where the changed token is on an equivalent
   path or a value that the hidden cases do not exercise (`zero_to_one`,
   `true_to_false`, a `+`/`-` in an unused branch).

## What this measures

- QAL's hidden-test suite catches **83.6%** of compiled mechanical mutations;
- contract deletion is a real blind spot for test-only validation;
- it is **not** yet a Python/TypeScript comparison; that arm is still missing.

## Next

- run `qal prove` on every mutant and count contract mutants caught by proof
  obligations;
- add a "contract diff" rule: deleting/weakening `req`/`ens` requires explicit
  review and a new test;
- add Python/TypeScript agents on a 100-task suite for the language comparison.
