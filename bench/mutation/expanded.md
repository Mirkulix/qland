# Mutation benchmark (134 + 13 contract mutants)

`expand.py` applies nine source-level changes to the 23 teacher solutions,
only outside comments and string literals:

| Class | Mutators |
|---|---|
| operators | `+`→`-`, `-`→`+`, `*`→`+`, `==`→`!=` |
| constants | `true`→`false`, `0`→`1` |
| contracts (loose) | drop first `req`, drop first `ens`, weaken first `>=` |

A mutant counts only if it **compiles**; `caught_tests` means hidden tests fail,
`caught_proof` means `qal prove` refutes an obligation, `escaped` means neither.

## 134-mutant result

```
caught_tests  112 / 134
caught_proof    0 / 134
escaped        22 / 134
```

**112/134 (83.6%)** are caught by hidden tests. The prove arm adds no catch on
this broad set. The 22 escapes are equivalent/uncovered paths and contract
deletions.

## Contract-only study (`contracts.py`)

Targets only `req`/`ens` lines: drop `req`, drop `ens`, weaken `>=`→`>`.

```
caught_tests   6 / 13
caught_proof   0 / 13
escaped        7 / 13
```

- **`drop_req`/`drop_ens` escape**: removing a contract removes an obligation,
  not a behavior. Tests stay green; `qal prove` has nothing to refute.
- **`weaken_contract_ge` is caught by tests** in 6/6 cases (and produces
  refutations/unknowns in some), because it changes a checked obligation.
- `02-roman`, `09-rle`, `22-expense-approval-contracts`, `23-expense-approval-change`
  have contract deletions that escape both tests and proof.

## Consequence

Test-only or proof-only validation cannot catch **contract deletion**. The
policy in `docs/QAL-FIRST.md` requires:

1. a failing test,
2. a `qal prove` result,
3. an explicit review note

for any removal or weakening of `req`/`ens`. A diff rule must flag deleted
contract lines; otherwise a real safety obligation can disappear silently.

## Not yet

This is not a Python/TypeScript comparison. It measures QAL's own verification
strength and finds a concrete blind spot.
