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
