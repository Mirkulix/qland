# QAL conformance smoke suite

Five independent cases pin the minimum observable behavior of the compiler:

| Case | Verb | Expected |
|---|---|---|
| `valid.qal` | `check`, `test` | exit 0 |
| `type_error.qal` | `check` | exit 1, `E0203` type mismatch |
| `test_failure.qal` | `test` | exit 1, `passed: 0` |
| `contract_ok.qal` | `prove` | exit 0, `proven: 1` |
| `contract_bad.qal` | `prove` | exit 1, `refuted: 1` |
| `effect_denied.qal` | `run` | exit 1, `E0508` permission denied |
| `unknown_name.qal` | `check` | exit 1, `E0201` |
| `parse_error.qal` | `check` | exit 1, `E0100` |
| `contract_diff/after_same.qal` | gate | exit 0 |
| `contract_diff/after_removed.qal` | gate | exit 1 |
| `contract_diff/after_weakened.qal` | gate | exit 1 |

Run:

```sh
QAL_BIN=$PWD/qal/target/release/qal python3 bench/conformance/run.py
```

Verified result: all twelve checks pass (`exit 0` from the runner), including the
refuted-contract counterexample `x = 1`.

This is the seed of the normative conformance suite: every QAL implementation
must satisfy these cases before it can call itself QAL.
