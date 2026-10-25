# Mutation study of 2026-10-25: does QAL's verification catch bugs Rust's lets through?

Design: `docs/superpowers/specs/2026-10-25-qal-stufe3-mutationsstudie-design.md`. Script:
`mutate.py`, mutants: `mutants.json`, raw results: `results-a1.json` (before the fix below),
`results-a1-transitive.json` (after it).

Nine pressure runs of the living program never showed a bug that QAL's guarantees caught and
the Rust arm let through, but bugs were too rare (about 1 in 100 stages) to tell. So this study
puts bugs into the code instead of waiting for them. It takes the final code of one line of each
arm (stage 12, both passing the hidden tests), puts the same 15 bugs into both, and records
whether the verification each arm carries reports them.

- **QAL:** the store of pressure run 9: 68 functions, 104 tests, 5 functions with contracts.
- **Rust:** `src/main.rs` of Rust pressure run 2: 1463 lines, 74 tests.
- **Bugs:** the bug classes of the runs, the same change at the source level in both languages:
  - cancelling a pending order deletes it (the bug of run 5);
  - cancel after a return;
  - `outstanding` falls below 0;
  - no 50 % discount cap;
  - the quantity discount is taken per line;
  - tax is not reversed on cancel or return;
  - the refund is computed without tax;
  - off-by-one at `bad_qty`, `overpay` and the credit limit;
  - `top` ties reversed;
  - backorders filled newest first;
  - the stock check per line;
  - promotions up to 100 %.
- **Detected:**
  - QAL, at write time: the `qal_define` reply that stores the bug reports a refuted contract,
    a failed test or a property warning.
  - QAL, full run: `qal_test` of everything.
  - Rust: `cargo test` fails (a test, an `assert!`, or a debug-build overflow panic).
- **Valid:** a mutant counts only if the hidden tests of stage 12 fail on it. All but one do:
  `fits_per_line` passes the hidden tests, yet both arms' own tests still catch it.

## Result

| | caught (of 15) | valid ones caught (of 14) |
|---|---|---|
| Rust `cargo test` | 15 | 14 |
| QAL `qal_test` (full run) | 15 | 14 |
| QAL `qal_define` reply, before the fix | **1** | 1 |
| QAL `qal_define` reply, after the fix | 15 | 14 |

1. **No advantage for QAL.** Both suites catch every bug, and in both the catch comes from a test
   the agents wrote, e.g. `cancel_backorder_twice` and
   `cancelled_backorder_never_fills_and_stays_cancelled` for the bug of run 5. Only one bug
   (`no_discount_cap`) was caught by a contract: `discount`'s `ens r <= 50` is refuted when it is
   defined, and nothing is stored. Rust's tests catch the same bug three times over. With test
   suites this thorough there is a ceiling: no mechanism can catch more than all of the bugs. The
   planned measurement A2, the same invariants added to both arms, therefore cannot show a
   difference on this code and was not run.
2. **A real weakness of QAL, now fixed.** `qal_define` ran the stored tests that use a defined
   function *directly*. The agents' tests reach `cancel`, `take_back`, `fill` and the rest through
   `step`/`out_of`, so the define that stored a bug ran **no test at all** in 13 of 14 cases. A QAL
   agent learns of the bug only from a separate `qal_test`, and the agents of runs 8–9 called that
   20 times in 45 defines. Now `qal_define` runs every test that reaches a defined function,
   through any number of calls: every bug is reported in the reply that stores it. Cost: 93 tests
   in 1.5 s on this store.

## What it means

On a program whose agents write thorough tests, QAL's guarantees add nothing measurable: the tests
do the work in both languages. The fix does not change that comparison. What it changes is that
QAL's write path now runs the tests on every change without being asked, which Rust's has no
equivalent of: there, running `cargo test` is the agent's habit.

A difference can only show where tests are thin. Two candidates remain:

- **A3:** the same mutants on code that keeps only the public examples as tests, plus local
  contracts on every function that changes a record, in both arms: QAL `req`/`ens`, proven and
  checked by generated inputs; Rust `assert!`s, which run only on the paths the examples take.
- **Phase B:** agent runs with trap tickets of these bug classes.

A3 is cheap and decides whether Phase B is worth its cost.
