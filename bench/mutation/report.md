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

## A3: thin tests, the same invariants

Script: `a3/prepare.py` (contracts `a3/qal_contracts.qal`, Rust edits `a3/rust_edits.json`), raw
results: `results-a3.json`. Both arms keep only the two public examples of stage 12 as tests.
Both get the invariants of the design:

- stock and sold ≥ 0, outstanding ≥ 0, tax ≤ spent, returned ≤ ordered, discount ≤ 50;
- QAL: as `req`/`ens` on the five functions that write a customer, order or product record
  (`put_back`, `take_back`, `fill`, `pay`, `sell`), with the bodies unchanged;
- Rust: as `assert!`s at the same places. The unsigned types already enforce ≥ 0 (a debug-build
  underflow panics).

The invariants were written after the mutants were chosen. They are the list fixed in the design,
not contracts aimed at single mutants, but the bias is there. One of my first QAL contracts was
wrong: `take_back` required `o.total <= c.spent`, which no longer holds after a partial return,
and the hidden tests' run stopped on it (`contract_violation`). It was corrected before the
measurement.

| | valid mutants caught (of 14) |
|---|---|
| Rust `cargo test` (2 example tests, `assert!`s) | 1 (`qty_discount_per_line`, by an example) |
| QAL `qal_define` / `qal_test` (2 example tests, contracts) | 2 (the same one, and `no_discount_cap` by `discount`'s proven `ens`) |

QAL is one ahead; the criterion of the design (at least 3) is not met. The invariants that would
catch four more mutants (outstanding below 0 on cancel, tax kept on cancel or return, one item
too many returned) are written down in QAL, but nothing checks them before the code runs:

1. **The prover** treats a Bool helper without `ens` (`cust_ok`, `cust_at`) as unknown, and
   approximates record fields: every obligation over the state is `unknown`.
2. **The property tests give up.** Random records almost never satisfy a `req` with several
   field conditions: `put_back` 3 runs of 1000 inputs, `take_back` 0, `fill` 37. So the contract
   is checked only on the inputs the two examples happen to reach, which is what Rust's `assert!`
   does too.

## Conclusion of the study

QAL's guarantees beat Rust in this study only where a contract is about integers the prover models
exactly (the discount cap). For invariants over records and maps they are, today, run-time
checks, as in Rust. Agent runs (Phase B) would not show more than that. What would make the
difference is one of:

- inputs generated so that they satisfy the `req` (e.g. field by field from its conditions, or
  from a Z3 model of it) instead of random ones filtered by it;
- a prover that sees through small Bool helpers (inline their bodies).

Either way, A3 is the measurement that would show it, with the same mutants.
