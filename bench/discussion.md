### What the numbers show

QAL lost the first measurement, and it lost on the primary metric. Over 40 valid runs per arm, QAL solved 39 and Rust 40. Both arms solved 39 runs at the first attempt. QAL spent **69.5k tokens per verified solution** and Rust **60.1k**, so QAL used about 16% more. Rust won 12 of the 20 tasks and the other 8 were ties. QAL won none.

The secondary metrics agree:

- QAL processed about twice as many transcript tokens per solution (246k against 127k, cache reads included).
- QAL used 143 tool calls against 112.
- QAL had 38 failed tool calls against 0.
- QAL needed 26 minutes of agent wall time against 15.

The thesis that an AI-native language makes agents more effective is **not supported** by this measurement, in this form of QAL.

Most of the token count is fixed overhead. The cheapest Rust run used 59.3k tokens, and the median Rust run used 60.0k, so the harness and system prompt account for almost all of each dispatch. The rest is what each arm adds:

- **Rust:** about 1k tokens. The usual run is three tool calls: write `main.rs`, build and run it, finish.
- **QAL:** a median of about 7k tokens, with a maximum of 89k. About 3k of that is the language reference in the prompt (≈11 KB). The rest goes to define calls the checker rejects, each followed by a corrected resubmission.

Measured as marginal cost above the fixed overhead, QAL is several times more expensive than Rust, not 16%.

### Where QAL lost, from the transcripts

All 38 failed QAL calls were classified from the transcripts. They group into a few causes, most of them missing language features rather than bugs:

- **No closures (about half of the `unknown_name` errors).** Agents repeatedly wrote calls like `map(range(0, m), fn_get_dummy)`. They needed a function over indices that reads a local list (`keys`), could not express it, typed a placeholder name and were rejected. The reference says plainly that there are no closures. The agents still reached for them, then rebuilt the logic as extra top-level functions with more parameters.
- **Sequences require `Unit` items (11 `type_mismatch` errors, mostly `expected Unit, found Int`).** Agents put a value-producing call such as `fold(...)` or a helper returning `Int` into `{ a; b; c }`. In QAL a sequence item that is not `Unit` is an error. Most languages agents know would discard the value.
- **Built-ins are not values (`trim`, `to_lower`, `len`).** `map(xs, trim)` is rejected, and the agent has to write a wrapper. The rule is in the reference, but agents broke it anyway.
- **Type name `String` instead of `Str` (2 times).** This is prior knowledge leaking in.
- **Bad local test files (4 shell errors, plus 1 `fs:data` run against a missing directory).** These are not language problems: the agents made mistakes with their own test files.
- **Runtime limits (2 `int_overflow` while testing extreme values, 1 `out_of_memory` in word-wrap on a 25,000-word paragraph).** Word-wrap hit the quadratic `++`/`push` limit that the reference warns about. The agent restructured the program and passed.

The one unsolved run, 08-expr-eval QAL run 2, was not a language problem. The program removed all spaces before parsing, so `1 2` printed `12` instead of `error: syntax`. The statement says spaces may not appear inside a number. The agent missed that rule in three attempts and finally gave up. Its run 1 got it right. The one Rust retry (19-base-convert run 2) was similar: blank lines were skipped instead of printing `error`.

Where QAL did well:

- **Its diagnostics worked.** Every rejected definition was fixed within one or two calls, and no QAL run went over the tool budget.
- **Contracts, effects and the store caused no failure.** QAL agents more often wrote their own test inputs, and several checked against a Python reference or at the size limits. That is a plus for correctness, but it costs tokens.

### What Phase 2 should change because of this

1. **Closures first.** Lambdas with captured variables would remove the largest error class and the extra top-level helper functions. Phase 2a already plans them. The benchmark makes them the top priority within 2a.
2. **Built-ins as values.** Let `map(xs, trim)` work by generating wrappers automatically. This is cheap and removes an error class.
3. **Make the sequence rule easier on agents.** Either let `{ a; b; c }` discard non-`Unit` values, or add a single hint to the `type_mismatch` diagnostic for this case: "sequence items must be Unit; bind it with let or use a for-each fold". The second keeps the language strict, and it is the cheaper of the two.
4. **A shorter reference.** The ≈11 KB reference is a fixed cost on every task, and agents still miss rules in it. Moving examples into diagnostics, where errors actually happen, should cut both the prompt and the rule violations.
5. **Native MCP tools instead of the shell client.** This would remove one Bash wrapper per call. Doing it needs the native `/mcp` connection in Claude Code.
6. **Repeat the benchmark after Phase 2.** Use the same tasks, schedule format and metrics, so the effect of each change can be measured rather than assumed.

### Honest limits of this measurement

- **Sample size and model:** 20 small CLI tasks, 2 runs per arm, one model (`sonnet`). Differences of one or two tasks are noise. The 1-run gap in solved runs is not significant; the token and failed-call gaps are consistent across tasks.
- **Prior knowledge:** the model has seen a great deal of Rust and no QAL. Here that advantage counts against the language being new, not against its design. Even so, it is a real cost of any new language.
- **Prompt asymmetry:** the reference in the QAL prompt is counted as cost, on purpose, because an MCP client pays it too. The Rust arm relies on knowledge the model brings with it, at no visible token cost.
- **Shell transport:** QAL is reached through `qalc.py` over Bash, not as native tools.
- **Task shape:** small stateless programs suit a language the model already knows. The things QAL is designed for are not exercised: many agents working on one codebase, refactoring through content addressing, proofs, and capability-checked effects. Those claims are still untested.
- **Infrastructure:** a usage-limit pause stopped one controller session in the middle of the run. Every interrupted entry was completed afterwards, and none was redispatched or counted twice. All 80 entries have valid, checked results, and the transcript audit found no violations in any recorded result.
