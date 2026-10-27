# Live LLM team run, 2026-10-27

Three DeepSeek subagents worked on one QAL team store with real role sessions and keys.
Unlike `bench/team/run.py` (scripted roles), every step here was decided and executed
by an LLM agent through the MCP tools.

| Role | Session | Action | Result |
|---|---|---|---|
| architect (`arch`) | `arch.session` | `qal_change open`, `qal_define`, `qal_test`, `qal_change submit` | change `ca58d33355cbc`, `greet` + `greet_ok` stored, test passed |
| verifier (`ver`) | `ver.session` | `qal_test`, `qal_view`, `qal_review verified` | current `verified` review, record 8 |
| reviewer (`rev`) | `rev.session` | `qal_review approve`, `qal_merge`, `qal_test main`, `qal_verify` | **fast_forward**, `greet_ok` passes on main |

Evidence:

- change `ca58d33355cbc`, owner `arch`, 0 conflicts, no rebase,
- merge `kind=fast_forward`, merge-time tests passed,
- `qal_test` on `main`: `greet_ok` passed,
- `qal_verify --trust team.json --head ee51d95ac089d9bca182082899535255b402fb4d53f2aa54ba3e384ba8f41195`:
  **ok=true, 11 records, 4 members**.

The source:

```qal
fn greet(name: Str) -> Str = "Hello, " ++ name
test greet_ok = greet("QAL") == "Hello, QAL"
```

## What this proves

The **live role protocol** works: separate LLM agents, separate sessions, signed
records, verifier gate, reviewer merge, pinned-head verification.

## What it does not prove

- No concurrency: the three agents ran sequentially; the scripted 6-agent run
  (`bench/team/report.md`) is the concurrency evidence (3 writers, replayed merges).
- No emergence: no scheduler, no reputation, no selection.
- Keys and sessions stay in `/tmp/qal-swarm-keys`, not in the repository.

Next: run six live agents concurrently on separate change sets, inject a bad
change, and require replay/merge under the same gate.
