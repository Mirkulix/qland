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

## A3 again, with req-directed generation (2026-10-25, later)

`qal-prove/src/repair.rs` now reads a `req` as atoms on paths into the parameters, through Bool
helpers, `map_get` lookups and `fold(..., true, fn(a, x) => a && p(x))`, and repairs every other
generated input towards them before it runs. Every input still has to pass the real `req`, so
a run means what it meant before; only the chance of a run changed. Raw results:
`results-a3-repair.json`.

The first run of the property tests found three gaps in my own A3 contracts, on the
**unmutated** code:

- `fill` promised something about the stored records under `oid` and `c.id` without requiring
  them to be valid: a frame condition was missing (`order_at(st, oid)`, `cust_at(st, c.id)`).
- `take_back` can keep `taxes <= spent` only if the customer's net covers the refund's net; the
  exact precondition is `c.spent - c.taxes >= o.total * qty / o.qty - o.tax * qty / o.qty`.

That is the job of contracts: they force the precision a random test never asks for. After these
corrections, the five contracted functions pass 100 of 100 runs each. Before, they gave up at 37,
3 and 0 runs (`fill`, `put_back`, `take_back`).

| | valid mutants caught (of 14) |
|---|---|
| Rust `cargo test` (2 example tests, `assert!`s) | 1 |
| QAL before (random inputs) | 2 |
| **QAL with req-directed inputs** | **5** |

The three new catches are property tests of the mutated function failing its `ens` in the
`qal_define` reply that stores the bug. The same tests pass 100 of 100 runs on the unmutated code:

- `cancel_outstanding_negative`: `put_back`;
- `cancel_keeps_tax`: `put_back`;
- `return_keeps_tax`: `take_back`.

Rust's `assert!`s of the same invariants sit on the same paths, but no test runs them.

**The design's criterion is met: QAL catches 4 more than Rust.** Two caveats:

- One code base, one set of mutants. The contracts were written by me after the mutants were
  chosen, from the design's fixed list of invariants.
- The remaining 9 mutants are rules (ordering, off-by-one at limits, what an id means after a
  cancel) that no invariant of the list states. Only a test or a contract that states the rule
  catches them.

**Next:** Phase B, agent runs with trap tickets. These answer whether agents write such
contracts themselves, and whether the property warnings change what they deliver.

## A4: type invariants instead of function contracts

Source: `a4/qal_invariants.qal`, raw results: `results-a4.json`. `qal-prove/src/inv.rs` makes a
pure `fn inv_T(x: T) -> Bool` the invariant of the record `T`. The property tests then check it
for every pure function that takes or returns a `T`, also inside lists, maps and records:

- inputs are repaired towards it and must keep it;
- a result that holds a `T` breaking it fails the property.

The study's three invariants are three lines, one per record type. The first run reported six
functions of the **unmutated** program that do not keep them for every input:

- `add_product` takes any `stock`;
- `restore` and `back_units` can drive `sold` below 0;
- `put_back` and `take_back` lack the relation between an order and its customer's totals;
- `settle` accepts negative quantities.

Each warning was answered with the precondition it pointed to, as an agent would. That took six
`req`s, two helper predicates and one helper record. Two of my answers were wrong at first:

- `take_back` and `back_units` required `sold` to cover the whole order, which is false after a
  partial return. The hidden tests' run stopped on the first (`contract_violation`), and the
  property test of `take_back` pointed at the second.
- The right condition covers only the part that comes back (`sold_covers_part`).

After that, all 52 property tests pass on the unmutated code, and so do the hidden tests.

| | valid mutants caught (of 14) | what was written |
|---|---|---|
| Rust (2 example tests, `assert!`s of the invariants) | 1 | 7 `assert!`s |
| QAL A3 (contracts on 5 functions) | 5 | 6 helpers, 5 `req`/`ens` pairs |
| QAL A4 (type invariants) | 5 | 3 invariants; 6 `req`s and 3 helpers the warnings asked for |

The same three bugs are caught as in A3, now by the invariant: "the result of `put_back` holds a
`CustRec` that breaks `inv_CustRec`". The difference to A3 is where the work comes from:

- **A3:** I had to know which functions to give contracts.
- **A4:** the rules are written once per type, and the tool names every function that does not
  keep them. The invariant is checked in 52 functions, not 5.

Not caught: `return_one_too_many`. `give_back` would call `take_back` outside its `req`, but the
generated `st` almost never holds an order under the generated `oid`, so that path is not reached.
A generator that draws keys from the maps it generates would reach it.

## A5: reachable states (an experiment, not kept)

`a5/reach-experiment.patch` (raw results: `results-a5.json`; source: `a5/qal_invariants.qal`).

**Idea:** build the inputs of functions over the state from the program's own functions:

- a seed (`initial()`), then random steps (`add_customer`, `place_order`, `pay`, `cancel` ...);
- strings drawn mostly from the keys already in the state;
- remembered successful calls varied, as a fuzzer keeps the inputs that got somewhere;
- the function under test tried on every state.

On reachable states the relations between records hold by construction, so A5 needs only the
three invariant lines and two domain `req`s (`stock >= 0`, `qty > 0`), not A4's relational ones.

| | valid mutants caught (of 14) |
|---|---|
| A4, without reachable states | 5 |
| A4, with reachable states | 5 |
| A5, invariants only, with reachable states | 2 |

**What went wrong on the way, all fixed in the experiment:**

- the walks first learnt to place orders without lines, which always succeed;
- a step that only set an error flag counted as progress;
- a walk per input cost so much that 26 of 52 properties ran out of time;
- a pool shared across functions made the result of a function depend on which others were
  tested, so a define reported a failure the full run never had.

**Why it caught nothing more:** of 300 reached states only 3 held the situation the cancel bugs
need, an open order of a customer who has paid part of it.

**Verdict:** no measurable gain for about 400 lines and several seconds per test run, so it was
taken out. The idea is sound, but the walks would need guidance this generic generator does not
have, e.g. a grammar of the commands or the inputs of the program's tests.

**Kept from it:** the repair puts a path that is bounded on both sides by literals at its lowest
or highest value a third of the time each. Without this, repaired small inputs crowded out the
edges where `price * qty * 95` overflows, and an existing test caught that.

## E0807 on the mutants (2026-10-26)

**Setup** (`e0807.py`, raw results `results-e0807.json`):

- the final store of QAL pressure run 9;
- its behaviour memory, filled with the io-only `qal_run` inputs of its agents that the transcripts
  give back:
  - 10 of their 20 calls, holding **29 runs**;
  - the other 10 calls built their input in Python scripts and cannot be recovered;
  - run 9 predates the memory, so its store holds no `observed` table;
- each of the 15 mutants defined once.

**Counted:** 13 mutants.

- `no_discount_cap` is refuted by Z3 and never stored.
- `fits_per_line` fails no hidden test.

**Results:**

| | mutants reported (of 13) |
|---|---|
| E0807 alone (29 observed runs) | **5** (cancel_keeps_tax, top_ties_reversed, pay_exact_is_overpay, restock_newest_first, limit_reached_rejects) |
| the tests the define runs | 13 |

**Reading:**

- **Behaviour memory:** it catches regressions without any test, only from the runs the agent
  made anyway. With half of the runs the agents made that is 5 of 13 bugs, a lower bound.
- **Tests:** on this program the agents' own tests already catch every mutant, so here E0807 is a
  second net.
- **Live trial 12:** the agent's tests missed the regression, and E0807 alone reported it
  (`bench/loop/report.md`).
- **Coverage:** it grows with the inputs observed, so more runs in the memory mean more coverage.

**With the tickets' public examples in the memory as well** (`E0807_EXAMPLES=1`, raw results
`results-e0807-examples.json`; 36 calls):

- **Which inputs:** the examples of stages 1–11 that every agent runs, not the hidden tests.
- **Result:** E0807 alone reports **11 of 13** mutants. It misses `cancel_pending_deletes` and
  `cancel_outstanding_negative`, which no example exercises.
- **Counter-check:** four defines that change the form but not the behaviour (`cancel`, `fill`,
  `restock`, `pay`) leave E0807 silent.
