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

## Scaling (2026-10-25, later)

`scale.py` makes the run-10 program ten times as large: 1520 definitions and 760 tests, all
passing. A define in it took **4.5–5.0 s**, so the one second held only for small programs. Where
the time went:

| step | time |
|---|---|
| whole-program check | 1.41 s |
| hash | 0.60 s |
| tests | 0.97 s, mostly compiling all 760 tests to run 72 |
| properties | 1.25 s, compiling every function to test one |

Fixed:

- **Tests** compile only the tests a define runs, and what they reach.
- **Properties** compile only the tested functions, their `req` checks and the invariants as entry
  points.
- **Check and hash** run at the same time; both only read the combined program.

| codebase | before | after |
|---|---|---|
| 67 functions (run 10), median | 0.64 s | **0.46 s** (max 0.70) |
| 1520 definitions, typical define with 72 tests | 4.5–5.0 s | **1.9–2.1 s** |

**Left:** the self-hosted compiler checks and hashes the whole codebase on every define, ≈ 1.6 s
at this size. An incremental check and hash would recompute only the defined names and their
dependents. That changes the protocol of the self-hosted compiler and so the bootstrap fixpoint,
which is the next step for programs of this size.

## Live trial of E0807 (2026-10-25, later)

The setup of run 11, stage 12, repeated three times in parallel:

- the stage-11 store of run 11;
- its behaviour memory, filled with the 25 real `qal_run` inputs of stages 1–11;
- the current binary;
- three fresh Haiku agents on the stage-12 ticket.

**Result:** all three passed the hidden tests (50/50), none made run 11's tier-discount mistake,
and E0807 never fired. None of the 25 kept runs changed output, so there was no false alarm.

**Eight more trials** (same setup, the binary with both selfhost changes): 8/8 at 50/50, no
E0807. Over 11 repeats of the stage the mistake did not recur, and the memory stayed silent every
time.

**What this shows and what it does not:**

- **Shown:** the memory stays quiet when nothing breaks. In run 13 it fired only for the three
  intended changes.
- **Not shown:** an agent correcting a live regression after an E0807. That mistake came up once
  in 18 attempts at stage 12 (runs 7–13 and these 11), so showing it needs many more attempts, or a
  ticket built to provoke it.
- **What is evidenced** is the replay on run 11: the warning names the hidden test's failure, with
  its input, in the round that made the mistake.

## Why the rest needs the self-hosted compiler (2026-10-25, later)

Two ways around it, both measured at 1520 definitions:

1. **`DEFINE` with the real store state** (protocol §4.5) and only the new text. Its hash equals
   the store's, but it takes 1.58 s, as long as check and hash together: it type-checks the whole
   program as well.
2. **A host-side cache of the fork/join work items.** The check splits into a main call of
   780 ms (parsing 400 KB and splitting it, on one core), 14 + 27 work items of 725 + 258 ms on
   8 cores, and joins of 230 ms. Each item carries about 100 definitions with their context, so a
   change alters several of them, and the single-core main call cannot be cached at all.

So an incremental define needs the self-hosted checker to take the stored signatures as given and
check only the bodies of the defined names and their dependents. That changes `qc_check` (QAL) and
the protocol, and needs the bootstrap fixpoint re-established. It is the open engineering step for
codebases above a few hundred functions. Until then: 0.46 s at 67 functions, about 2 s at 1520.

## Inside the self-hosted compiler (2026-10-25, later)

**Measured with fuel** (counted instructions) per compiler call, checking 1520 definitions:

- the parse alone costs 212 M;
- each of the 14 `CHECK_RANGE` work items costs ~206 M, the whole-text parse again, because
  checking its 128 definitions is cheap;
- the 27 validation items cost ~73 M each, decoding the whole typed form;
- the two joins cost 204 and 270 M.

**Two changes, each with the bootstrap fixpoint re-established and all suites green:**

- `cfbc6222`: the checker's globals look functions and variants up in maps, not by linear search.
  About 10 %, so the lookups were not the main cost.
- `9bbb6895`: `CHECK_FILE` splits into at most 8 work items (the host runs 8 at a time), so the
  whole-text parse is repeated 8 times instead of 14. `qal check` of the 1520 definitions takes
  1.17 s instead of 1.44 s.
- **Tried and kept out:** larger validation items. They ran out of the 1 GiB limit while
  bootstrapping the compiler itself; the small items exist for memory.

**Result:**

| codebase | at the start of the day | now |
|---|---|---|
| 67 functions, median define | 1.24 s | **0.46 s** |
| 1520 definitions, typical define | 4.5–5.0 s | **1.65–1.75 s** |

The rest is spread over single-core steps of the self-hosted pipeline: the main parse, the joins,
and the host's parse and print. Each of them needs the architectural change of the incremental
design (`docs/superpowers/specs/2026-10-25-qal-incremental-define-design.md`).
