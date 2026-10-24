# Living program, pressure run of 2026-10-24: result

Protocol: `bench/living/README.md` plus pressure mode (commit 8df97e87). This is the same order system and the same 12 change requests with the same hidden tests as the run of 2026-10-23. The difference: each change request carries only the change and two examples, as a ticket would. There is no full specification and no word about tests or contracts. Each line starts from the normal run's stage 0 of the same line, which is code with contracts and tests. Both arms run on Haiku. There is no retry: every stage builds on what the previous one left.

## Result

| | QAL / Haiku | Rust / Haiku |
|---|---|---|
| stages with every hidden test green | **0 / 12** | **12 / 12** |
| bugs that reached the hidden tests | 1, introduced at stage 1 and never noticed | 0 |
| rounds / processed tokens | 108 / 8.81 M | 91 / 6.15 M |
| verification the agents ran unasked | `qal_test` 4×, `qal_prove` 0×; `qal_define` ran the stored tests every time | `cargo test` 14× |
| own verification that went red during a run | 1 (stage 7: stored tests broken by the backorder change, fixed) | 3 (stages 4, 5: `cargo test`, fixed) |

## The bug

Stage 1 (quantity discount) in QAL:

```
fn total(price: Int, qty: Int) -> Int
  ens qty >= 10 && r == price * qty * 95 / 100 || qty < 10 && r == price * qty
= if qty >= 10 then price * qty * 95 / 100 else price * qty
```

With price and quantity at their maximum of 999 999 999 each, `price * qty * 95` exceeds 64 bits. QAL traps on overflow, so the program stops with a runtime error instead of printing. The hidden test `s0_big_numbers` failed from stage 1 to stage 12. The Rust line has computed in `u128` since stage 0, so the same formula is correct there.

**Why QAL's guarantees did not catch it:**
- The **stored tests** that `qal_define` runs cover no large amounts.
- The **prover** proves the `ens`. With trapping overflow the equation holds whenever the function returns, so the proof is correct and still says nothing about the trap.
- The **property test** of `total` would have hit the overflow on random inputs, but only if someone runs `qal_test`. Even then, the reference says traps are counted (`traps`), not failures. The agent here ran `qal_test` 4 times in 12 stages, and the trap would not have turned it red.
- No contract bounds the inputs, such as `req price <= 999999999 && qty <= 999999999`. With such a `req`, a trap in the property test would be a real finding.

## What this means

- **This is the first experiment in which a bug got through, and it got through in QAL, not in Rust.** The cause is a language property (64-bit Int with traps, against `u128` in the Rust code) together with guarantees that exist but have to be invoked. Without being asked, the agents did not invoke them. The Rust agents ran `cargo test` on their own 14 times; the QAL agents trusted the tests that `define` runs, and those had no case for the bug.
- QAL's thesis was that machine-checked guarantees catch what other workflows let through. This experiment refutes it in this form. **A guarantee that has to be called does not protect an agent that does not call it.**

## Consequences for the language

1. **`qal_define` reports traps.** For every defined function with an `ens` or a `req`, `define` runs the generated property test too, as it already runs the stored tests. An overflow trap on inputs that satisfy `req` becomes a warning in the reply. The guarantee then works without being asked for.
2. **Bounding inputs pays off and should be suggested.** The trap warning should suggest a `req` that bounds the inputs. Without one, every overflow counts as possible, and nothing distinguishes a real finding.
3. **Integer width:** a 64-bit Int that traps is safer than one that wraps. It is still a pitfall for business arithmetic with products of large values. Options: a 128-bit intermediate result for `*` followed by `/`, or a `muldiv` built-in. To be decided.

Files: `state-pressure` (git-ignored) is copied here as `results.jsonl`, `workdirs.json` and `schedule.json`.

## Follow-up: `define` runs the property tests (commit b814522d)

The QAL/Haiku line was rerun at stage 1 from the same stage 0 with the new binary. The agent wrote the same formula. Its first `define` came back with the warning E0804: `total` trapped on generated inputs, with the hint to bound the inputs with `req`. The agent's own words: "the only warnings are the existing overflow traps for huge negative inputs, which the old code also had". It did not change the code, and the hidden test failed again (19/20). The line was stopped after stage 1, because the outcome of the remaining stages was clear.

**The agent was right about what it saw.** Without a `req`, the old `price * qty` also traps on extreme generated inputs. The warning reports the first trapping input the generator found, which has values far beyond any real amount. Nothing in it shows that the new formula already overflows at 311 588 000 × 311 588 000, inside the 9-digit domain of the task, where the old one did not. The language cannot know that domain. It is in the stage-0 specification, but not in the code (no `req`) and not in the ticket.

**Consequence:** this cannot be closed by a better warning, only by the arithmetic or by required bounds:
1. Wider integers for intermediate results (`a * b / c` in 128 bits, or a 128-bit `Int`), as the Rust code had with `u128`. This is a language decision with consequences for the compiler, the self-hosting bootstrap and the prover.
2. Or a `req` that bounds every `Int` parameter of an arithmetic function becomes a rule (a warning on `define` without bounds). This makes the domain visible, but costs every agent a line per function.

## Third run: warning on unbounded products, property tests at the bounds (commit 29ae7deb)

The QAL/Haiku line was run again through all 12 stages, from the same stage 0, with the same tickets and the binary that warns on products of unbounded `Int` parameters (E0805) and draws the `req` literals as property-test inputs.

| | QAL/Haiku run 1 | QAL/Haiku run 3 | Rust/Haiku |
|---|---|---|---|
| stages with every hidden test green | 0 / 12 | **12 / 12** | 12 / 12 |
| bugs that reached the hidden tests | 1 (overflow, stages 1–12) | 0 | 0 |
| rounds / processed tokens | 108 / 8.81 M | 113 / 9.07 M | 91 / 6.15 M |

In stage 1 the agent wrote the discount as `x - (x + 19) / 20` and gave the reason before its first `define`: "this also avoids the intermediate × 95 overflow". The warnings came afterwards and were confirmed. So the effect is most likely the reference line about traps and bounds, which the agent reads at the start, rather than the warning in the reply. This is one run per arm, a single sample. It shows that the gap of run 1 can be closed, not that it always will be.

Cost: QAL is at 1.47× Rust's processed tokens on this line, with the same result.

## Fourth run (repeat of run 3, same binary)

Stage 1 again: the agent wrote `price * qty * 95 / 100`, received both E0804 and E0805 in the `define` reply, did not mention them, and kept the formula. The hidden test failed (19/20); the line was stopped after stage 1. **With the warnings, stage 1 succeeded in 1 of 2 runs.** The warnings improve the odds, but they do not make the result reliable: an agent can always ignore a warning. Removing the bug class needs arithmetic that does not overflow in this range (wider intermediate results), not more advice.

## Fifth run: exact multiply-divide (merge 1d0e4a3a)

The QAL/Haiku line was run again through all 12 stages from the same stage 0 and with the same tickets. The binary now computes `(a * b * …) / c` exactly through a 128-bit product (`price * qty * 95 / 100` cannot overflow for 9-digit amounts any more).

| | run 1 | run 3 | run 4 | **run 5** | Rust/Haiku |
|---|---|---|---|---|---|
| stage 1 (the overflow) | red | green | red | **green** | green |
| stages with every hidden test green | 0 / 12 | 12 / 12 | stopped | **6 / 12** | 12 / 12 |
| rounds / processed tokens | 108 / 8.81 M | 113 / 9.07 M | | 109 / 8.72 M | 91 / 6.15 M |

**Stage 1:** the agent wrote `price * qty * 95 / 100` again, the formula of runs 1 and 4. With the new arithmetic it is correct (20/20). The overflow class is closed without the agent having to notice anything.

**Stages 7 to 12:** a different bug, introduced at stage 7 (backorders) and never noticed. Cancelling a pending backorder removed the order instead of marking it cancelled. A second `cancel` or a `return` then answered `unknown_order` instead of `not_open`, and the id could be used again (`duplicate` expected). The stage-7 ticket's example covers a cancel of a filled backorder, not of a pending one. The agent's stored tests went red once during stage 7 and were fixed; none covered this case. Stages 8 to 12 inherited the bug (1 or 2 hidden tests red each).

**What this means:** the arithmetic fix does what it should. The remaining gap to Rust in this setting is ordinary logic bugs under thin tickets, which no QAL mechanism covers today: no contract states "an order id, once used, stays known". A single sample per arm; the Rust line had no such bug in its one run.

**Rust/Haiku, second run (same tickets, same stage 0):** 12 / 12 again, 86 rounds / 5.34 M processed tokens (QAL run 5: 109 / 8.72 M, 1.63×). At stage 7 the Rust agent kept a pending backorder in the order map with an explicit `Status::Pending`, set `Status::Cancelled` on cancel, and wrote its own test `cancelled_id_stays_taken`. The QAL agent removed the order from its map. Across two Rust runs the Rust line had no bug reach the hidden tests; across QAL runs 1, 3, 4 and 5, three of four let one through (overflow twice, this logic bug once).

The difference at stage 7 is not a language guarantee but agent habit: the Rust agents write a test for each new rule unasked (`cargo test` is part of the routine), the QAL agents rely on the tests `define` runs and add few of their own.

## Seventh run: a change without a test is named (merges 0b1eff64, c082135f)

New: a `define` that replaces a stored function and defines no `test` gets the warning E0806 `untested_change`, and the reference says so in one line ("pin each new rule with a test in the same define"). Run 6 is void: its binary let the proof warnings of the same define replace E0806 (fixed in 68dcb653, with a test for that case); it is kept aside as `state-pressure-run6-invalid`, not counted.

| | QAL run 5 | **QAL run 7** | Rust run 1 | Rust run 2 |
|---|---|---|---|---|
| stages with every hidden test green | 6 / 12 | **12 / 12** | 12 / 12 | 12 / 12 |
| bugs that reached the hidden tests | 1 (stage 7) | **0** | 0 | 0 |
| rounds / processed tokens | 109 / 8.72 M | 132 / 10.58 M | 91 / 6.15 M | 86 / 5.34 M |
| `test` lines the agents wrote in defines | stage 7: 3 | stage 7: 28; all stages: 280 | | |

Stage 7 (backorders) passed: the agent's own tests went red once during the run (call 8) and were fixed before delivery. The warning itself appeared once in 12 stages (stage 12); in every other stage the agent had already put tests into the define. **What changed the behaviour is the reference line the agents read at the start, not the warning in the reply**, as in run 3 with the overflow line.

**Cost:** QAL at 1.72× to 1.98× Rust's processed tokens on this line (run 5: 1.63×). The tests cost rounds and context. The result is now the same as Rust's; the premium is larger.

**State of the pressure line after seven runs:** the two bug classes found (64-bit overflow, untested rule changes) are closed, the first by the language (exact multiply-divide), the second by the instructions. One run each; the second fix rests on agent compliance and should be repeated before it counts as reliable.

**Protocol note on run 7:** stage 10 is marked invalid by the driver (`valid: false`). The agent redirected a `qal_view` dump of the program to `/tmp/x`, a path outside its work dir (30 KB, no secrets; it reported the file as empty, which was wrong; removed afterwards). The stage's code and result (46/46) are unaffected; strictly by the protocol, run 7 counts 11 valid stages of 11 green plus one green stage with a path violation.

## Run 8: the tool friction of run 7 removed (commit 0627f2b1)

Run 7 spent its extra rounds on the tools, not on the language: E0401 `signature_change` 17 times (a changed type or signature refused although the stored dependents still checked), E0508 `permission_denied` 9 times (`io` not granted to `qal_run`), and 3 parse errors without a hint (a bracket left open, reported at the next definition). Built before run 8: dependents that still check are relinked, E0401 only lists the ones that break, followed by their errors; `io` (the call's own stdin and stdout) needs no `allow`; a parse error at the next definition names the open bracket; the outline leaves out empty lists and a test's own `uses` (26 → ~16 KB on the run-7 program); MCP define replies leave out the hashes (no agent used one in 295 view/history/query calls). Same tickets, same Haiku protocol, `results-run8.jsonl`.

| | QAL run 7 | **QAL run 8** | Rust run 1 | Rust run 2 |
|---|---|---|---|---|
| stages with every hidden test green | 12 / 12 | **12 / 12** | 12 / 12 | 12 / 12 |
| bugs that reached the hidden tests | 0 | **0** | 0 | 0 |
| rounds / processed tokens | 132 / 10.58 M | **100 / 6.95 M** | 91 / 6.15 M | 86 / 5.34 M |
| E0401 / E0508 in replies | 17 / 9 | **0 / 0** | | |
| `define` / `view` rounds | 33 / 40 | 17 / 25 | | |
| `test` lines the agents wrote in defines | 280 | 86 | | |

**Cost:** QAL at **1.13× to 1.30×** Rust's processed tokens (run 7: 1.72–1.98×), 1.10–1.16× its rounds. The remaining per-round premium is mostly the prompt: TASK.md (reference plus tool descriptions, 26 KB against Rust's 4 KB) is about half of the carried tool context.

**Caveat:** one run. The agents wrote fewer tests than in run 7 (86 lines, E0806 once); stage 7 (backorders, the logic bug of run 5) passed with them. Whether the result holds with fewer tests needs a repeat.

**Protocol note on run 8:** stage 3 is marked invalid: the agent ran `cat > /tmp/x` with no input (an empty file outside its work dir, removed afterwards) before a `qal_view`. Its result (26/26) is unaffected. As in run 7, strictly 11 valid green stages plus one green stage with a path violation; both violations are the same Haiku habit.
