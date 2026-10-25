# The 1-turn verification loop, measured (2026-10-25)

**Claim:** one `qal_define` call tells an agent, in about a second and in one round, whether a
change is sound. The call performs:

1. type and effect checking;
2. the Z3 proof of the `req`/`ens` contracts;
3. every stored test that reaches the changed function, also through other functions;
4. a property test of each defined pure function at the edges of its domain;
5. the relinking of every dependent in the project.

**Measured on** the final store of pressure run 10: the living program after 12 change requests,
67 functions and 76 tests. `sweep.py` defines each of the 66 non-entry functions again with a
change that keeps its meaning, one MCP call each.

| | before (run 10's binary) | after |
|---|---|---|
| wall time of one define, median | 1.24 s | **0.64 s** |
| p90 / max | — / 1.70 s | 0.68 s / **0.90 s** |
| tests run per define (median) | 72 | 72 |
| dependents relinked per define (median) | 81 | 81 |
| defined functions with a property test | 1 of 5 sampled | **64 of 66** |
| property warnings over all 66 defines | — | 4 |

## What changed

- **No second check of the whole codebase.** With the prover on, a define checked and hashed the
  combined program twice: once without the lock for the proof, once under it to write. When the
  store did not change in between, which is the usual case, the first result is now written. This
  saved 0.28 s.
- **No reload for the tests.** The tests and property tests ran on the store loaded and checked
  again. They now run on the program the define just checked. Tests took 0.11 instead of 0.28 s;
  the property step went from 0.17 s to 0 when nothing has to run.
- **Point 4 for every function.** Property tests used to run only for functions with an `ens` or
  calling one with `req`, which an agent rarely writes. Now every pure function the define wrote
  gets one:
  - a function without contracts is tested at 0, ±1, empty values and the bounds of its `req`;
  - the ends of the 64-bit range are drawn only for functions with contracts.

  With the ends drawn for every function, the sweep reported 14 warnings, 12 of them overflows such
  as `4 + 9223372036854775806`, which no program with 9-digit amounts reaches. Without them it
  reports 4, all real partial functions:
  - `list_at` with index -1 in `fill_pending` and `give_back`;
  - `char_code("")` in `is_digit` and `is_ident_char`.

  The static warning E0805 still names unbounded products.

## What is left

- **Size limit of the second.** The remaining time is the whole-program check (≈ 0.16 s) and hash
  (≈ 0.11 s), linear in the codebase. A program ten times this size would need an incremental
  checker to stay near one second.
- **Agent rounds.** The loop gives the answer in one round, but whether agents then need fewer
  rounds is measured only by agent runs. Pressure run 10, with point 3 but not yet point 4: 101
  rounds.

## Point 6: the behaviour memory (E0807)

**What happened.** Pressure run 11 (this binary minus E0807) failed stage 12 (45/50): the first bug
to reach the hidden tests since run 5.

- **The bug.** The ticket said "tier discount and tax apply to the whole order", and the agent
  applied the tier discount after the line discounts. That rounds twice: 8835 instead of 8800.
- **Why the loop missed it.** In the same define the agent rewrote `total_tier_discount`, the test
  that pinned the old rule, so all 67 tests passed. The old versions of the three changed tests no
  longer even type-check: `total` gained a parameter. No test, proof or property could see the
  regression.

**What did see it** were the agent's own earlier runs. Of the 25 `qal_run` inputs of stages 1–11,
3 print differently on the stage-12 code. All three are stage-2 runs of tier discounts. The other
22 print the same, so the signal carries little noise.

**What was built.** Schema 8 adds a table `observed`:

- every run of an entry point that needs no effect but `io` is kept with its output (the latest
  output per input; 200 runs at most);
- a define runs the newest 64 again on the code it wrote, within a 2 s budget, and reports
  E0807 `behaviour_changed` with the first changed line and the input (`changed_runs`);
- an intended change is recorded by running the input again.

**Replayed on run 11:** the 25 runs, recorded on the stage-11 code, were put into a copy of the
stage-12 store, and one define was made. The reply, in 0.78 s:

```
E0807 `main` behaves differently on 3 of 25 earlier runs: `OK o1 8800` is now `OK o1 8835`
changed_runs: product p1 a 1000 100 / customer c1 ann gold / order o1 c1 p1 10 ...
```

That is the hidden test's failure, shown to the agent in the round in which it made the mistake.
Whether an agent then fixes it is a question for an agent run.
