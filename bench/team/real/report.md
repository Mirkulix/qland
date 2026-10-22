# QAL Phase 4d — real agent-team run (replaces the scripted bench/team results)

Date 2026-10-17. Binary built from branch `p4d-real` (Phase 4a–4c with review fixes). Agents: Claude Code subagents, model Sonnet, each using QAL only through its own MCP session (`qal mcp --session-file`). No QAL code was given to any agent; the controller wrote none. Token and tool-call numbers are measured from the subagent results, not estimated.

| Run | Arm | Agents | Wall time | Tokens (sum) | Tool calls | Scenarios | Hidden | `qal verify` | Criterion 6 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Team | 6 (arch-1, impl-1..3, ver-1, rev-1) | ≈ 765 s | 632 k | 154 | 20/20 | 15/15 | ok | all true |
| 1 | Solo | 1 | ≈ 123 s | 101 k | 13 | 20/20 | 15/15 | — | — |

Criterion 6 for the team run (from `score.py`): all scenarios pass, ≥ 3 members wrote within 5 minutes, a non-fast-forward (replayed) merge happened, every merge passed the gate (verifier `verified` + reviewer `approve`), `qal verify` ok, provenance ok (every definition on `main` written by a member session). Per-member numbers: `results.jsonl`.

## Discussion

- **The team mechanics work end to end with real agents:** the architect cut the card into a skeleton with `todo()` stubs and three change sets; three implementers worked in parallel; the verifier added tests (51 on `main`) and verified; the reviewer approved and merged (1 fast-forward, 2 replayed); no conflicts, no lease errors, `qal verify` accepts the signed history.
- **For a service of this size the team is not more efficient:** the solo agent solved the same card with all hidden scenarios in about 1/6 of the tokens and 1/6 of the wall time. Coordination (waiting, notices, reviews) dominates when the work fits into one agent's context. The team's added value here is assurance (independent tests, review gate, signed provenance), not speed. Larger codebases (e.g. the Phase 5 compiler) are where splitting should pay off; this run does not show that.
- One run per arm only (budget); numbers are indicative.
- Found and fixed before the run: review comments were not readable by authors over MCP (`qal_change show` now lists them).
- The earlier `bench/team/report.md` / `results.jsonl` (commit ed91f378) came from a script with hardcoded QAL source and estimated tokens; they are superseded by this report.
