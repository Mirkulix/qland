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
