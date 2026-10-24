# QAL-only rerun after `cases` on `qal_define` (2026-10-24)

Question: does `qal_define` with `cases` save the separate `qal_run` round (commit 1a243da9)? The QAL arm's 40 entries of `bench/rerun-2026-10-23` were rerun with the same seed, protocol and seed corpus. The Rust arm is unchanged, so its numbers from that rerun are the baseline. Token analysis: `bench/analysis/2026-10-24-qal/`. Rows of a retried entry appear once per attempt there; the figures below take each entry once, at its last attempt, which includes the whole transcript.

## Result

| | rerun 2026-10-23 | this rerun |
|---|---|---|
| solved | 40 / 40, all at the first attempt | 40 / 40, one at the third attempt (05-brackets run 2) |
| processed tokens per run | 212 441 | 227 394 |
| rounds per run | 3.95 | 4.13 |
| factor against Rust (149 650) | 1.42 | 1.52 |

Paired by entry (39 pairs): the mean difference is +18 375 tokens, driven by the retried entry; without it, −8 480 (−4 %); median +1 788. 12 entries are cheaper by more than 20k tokens, 18 within ±20k, 9 dearer. **The change has no measurable effect on cost.**

## Why

- Agents use it: 32 of 40 runs defined with `cases`.
- They then test further inputs of their own anyway: 87 separate `qal_run` calls in 40 runs. The round the feature was meant to save is spent on extra edge cases the agent invents after seeing the first results. Rust agents do the same, but inside one shell command.
- The failure (05-brackets run 2, hidden case `limit`) was an agent's quadratic stack that ran out of the instruction budget on a 57 KB input; fixed by the agent at the third attempt. It is unrelated to the change.

## Decision

`cases` on `qal_define` stays: it is used, it saves a round for MCP clients without a shell, and it costs nothing when unused. It does not close the gap. The remaining factor (about 1.4 to 1.5) is not a tooling round any more. It is how agents test: QAL agents run more separate checks than Rust agents. Add to that the larger prompt (26 KB against 5.5 KB), re-read every round. The next lever is not another tool. It is the prompt: the reference and tool descriptions could be cut by about half while measuring that the language errors stay down. That is a separate experiment.
