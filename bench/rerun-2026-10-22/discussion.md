## Discussion (rerun of 2026-10-22)

This is the Phase 1e benchmark run again, on the code base after Phases 2–5 (worktree branch `worktree-p6-gaps` at `31ed1442`, i.e. `13cf4246` plus the write-time signature check and the `sync-std` seed fix; neither touches the compiler, checker or MCP server). The question was whether the language changes made after the first run (closures, generics, Map/Bytes, the standard library, better diagnostics) changed the picture of the first run, where Rust won 12 of 20 tasks and QAL none. Same 20 tasks, same hidden tests, same prompts, same harness (`bench/run.py`), 2 runs per task and arm in a fresh randomized order (seed `qal-1e-rerun`), 80 dispatches.

### What is different from the first run

- **The QAL server ran the self-hosted compiler.** The benchmark binary was built with the default features, which no longer include the Rust reference compiler: every `qal_define`, `qal_test` and `qal_run` went through the QAL compiler written in QAL (the seed module `selfhost/qalc.wasm`). The first run used the Rust compiler.
- **Model.** Both runs used Claude Code subagents with the `sonnet` model alias; the model behind the alias is the one of the day, not necessarily the one of 2026-10-15.
- **Token measure.** The dispatch results of the Agent tool no longer report a token total, so `dispatch.total_tokens` of this run is `input + cache_creation + output` tokens summed over the API calls of the subagent's transcript (cache reads excluded, the same for both arms). The first run's figure came from the tool and is not the same quantity, so the absolute per-run numbers of the two runs cannot be compared. The *processed* tokens (all four categories) are computed the same way in both runs and can.
- **Prompt delivery changed after 36 recorded runs.** For the first 36 runs the controller pasted the prepared prompt into the subagent call; two dispatches had to be discarded before they were recorded because the pasted text differed from the prepared prompt (an extra `open x` line in the bank example; exit code 0 instead of 1 in a CSV example). From then on the exact prompt file was copied into the work directory as `TASK.md` and the agent was told to read it; both arms alike. Every later run therefore has one extra Read tool call, and the prompt tokens enter the context as a tool result. Details in `notes.md`.
- **A confound that favours Rust.** The controlling session ran inside a git worktree, and its subagents inherited a guard that refuses shell commands it "cannot show not to be git". It refused 47 commands in 35 of the 40 QAL runs and 17 commands in 16 of the 41 Rust dispatches, mostly heredocs (`qalc.py define <<'EOF'`) and redirections, and the agents had to rephrase and retry. These refusals are counted as failed tool calls by the audit. Their cost on the primary token measure is small: QAL runs without a refusal averaged 35.8k tokens against 36.8k with (0.7k per refusal), Rust 21.0k against 24.5k (3.3k per refusal). On processed tokens and tool-call counts the effect is larger (each retry re-reads the whole context). The rows below give the raw numbers; the adjusted per-run means are about 35.8k (QAL) and 21.0k (Rust).
- **One Rust run was dispatched twice**: the first transcript of `09-rle` run 1 named `src/../t1.txt` in a command (an audit violation) and was discarded as the protocol prescribes; the second dispatch counts.

### Results

| | QAL | Rust | first run (2026-10-15) |
|---|---|---|---|
| Solved | **40 / 40** | 40 / 40 | 39 / 40 vs 40 / 40 |
| Solved at the first attempt | 40 | 40 | 39 vs 39 |
| Tokens per verified solution (this run's measure) | 36.7k | 22.3k | not comparable |
| Processed tokens per solution (comparable) | 308.8k | 161.2k | 246.1k vs 126.5k |
| Ratio QAL / Rust, processed | 1.92 | | 1.95 |
| Tool calls per run | 6.8 | 4.3 | 3.4 vs 2.7 |
| Genuine language/runtime errors seen by the agent | 9 (E0601 ×3, E0100 ×2, E0510 ×2, E0201, E0502) | 0 compile errors | 38 vs 0 (failed tool calls) |
| Tasks won | 0 | 20 | 0 vs 12 (8 ties) |

### Reading

1. **Reliability went up.** QAL now solves every task in every run, at the first attempt, with the compiler written in QAL. The one task it failed in the first run (`08-expr-eval`) passed both runs. The agents hit 9 genuine diagnostics in 40 runs where the first run counted 38 failed calls; the diagnostics are store/runtime errors and one unknown name, not parse errors. The language reference in the tool descriptions is now enough for an agent that has never seen QAL.
2. **Cost did not go down.** On the comparable measure the QAL arm processes 1.9× the tokens of the Rust arm, as in the first run (1.95×). On this run's primary measure it is 1.65× raw, about 1.7× after the refusal adjustment. Rust wins every task on tokens, mostly by a margin well outside the 10 % tie band. The fixed part of the gap is the language reference the QAL prompt carries (about 4k tokens per run); the larger part is behaviour: QAL agents make 1.6× the tool calls and write about 2× the output tokens (more verbose recursive code, more probing of built-ins, test files written through `qalc.py` instead of a shell redirect).
3. **The thesis of the design spec is still not supported** for tasks of this size: an agent is not cheaper in QAL than in Rust, it is as correct and about 1.7–1.9× as expensive. What QAL buys in these runs is not speed but checked, stored, hash-addressed solutions with contracts available; none of the 20 tasks needs that.
4. **What would move the numbers.** (a) Native MCP tools instead of the `qalc.py` shell client: every QAL call costs a Bash invocation, a heredoc and the guard's scrutiny; (b) tasks larger than one context window, where `qal_view`/`qal_query` replace re-reading files; (c) a shorter language reference, or one the model has seen in training. None of these was measured here.

### Files

`schedule.json` (order), `results.jsonl` (one record per dispatch, with the transcript audit), `report.md` (generated by `bench/report.py` with `BENCH_STATE=bench/rerun-2026-10-22`), `notes.md` (discarded dispatches and the protocol change). Work directories and transcripts stayed outside the repository.
