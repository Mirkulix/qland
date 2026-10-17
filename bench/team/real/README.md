# Phase 4d: the REAL agent-team run (shop)

Replaces the scripted `bench/team/run.py` (hardcoded QAL source, estimated tokens). Here every team member is a
Claude Code subagent that uses QAL only through its own MCP session (`qalc.py`), exactly like the 1e benchmark.
The controller writes no QAL body and does not measure by estimate.

## Steps for the controller (per run; 2 team runs and 2 solo runs, spec section 0 criterion 6)

1. Build once: `cd qal && cargo build --release --offline -p qal-cli` (prep copies the binary to `/tmp/qal-team/bin/qal`).
2. `python3 bench/team/real/prep.py team <run>` (for example `1`). Creates `/tmp/qal-team/team-<run>/` (the project, owner `owner-1`,
   members `arch-1`, `impl-1..3`, `ver-1`, `rev-1`, one 24 h session each), keys in `/tmp/qal-team/keys-<run>/` (outside the project),
   one work directory per member `/tmp/qal-team/team-<run>/<member>/` with `qalc.py` (bound to that member's session),
   `prompt.md` (the rendered role prompt with the card), `card.md` and `public_scenarios/p*.json` (hidden scenarios are never copied).
   It prints one line per member: role, workdir, prompt file. Record the start time.
3. Dispatch all SIX subagents in ONE message, in the background (they must run concurrently; they coordinate only through the store).
   Each subagent's prompt is the content of its `prompt.md` (or: "Read and follow `<workdir>/prompt.md`"). Do not add hints.
   Roles map: `arch-1` architect, `impl-1/2/3` implementer, `ver-1` verifier, `rev-1` reviewer. Give them Bash access only
   (python3 in their work directory is all they need).
4. Wait for all six to finish (each ends with `DONE: ...` or `GAVE UP: ...`). Hard stop suggestion: 90 minutes wall time;
   a stalled run is a result too (score it as it stands and note `GAVE UP`). For each agent save, from the task notification,
   `total_tokens`, `tool_uses`, `duration_ms` and its final line.
5. Write the stats file (measured, never estimated):
   `{"agents": {"arch-1": {"total_tokens": N, "tool_uses": N, "duration_ms": N, "outcome": "DONE"}, "impl-1": {...}, ...}, "wall_s": <seconds from dispatch to the last agent's end>}`
   (`tokens`/`tool_calls`/`duration_s` are accepted as aliases).
6. `python3 bench/team/real/score.py /tmp/qal-team/team-<run> --agent-stats stats.json --label "<free text>"`
   Prints a summary, appends one JSON line to `bench/team/real/results.jsonl`. Takes about two minutes (20 `qal serve` starts).
7. Solo baseline: `python3 bench/team/real/prep.py solo <run>`, dispatch ONE subagent with `/tmp/qal-team/solo-<run>/solo/prompt.md`,
   same stats file format (`{"agents": {"solo": {...}}}`), then `score.py /tmp/qal-team/solo-<run> --agent-stats ...`.
8. Do not let any agent see this directory, the hidden scenarios (`bench/team/shop/scenarios/h*.json`) or the repository.
   Never run `qal define`/`rename` as the owner on a team project (the owner writes no definition; the provenance check fails otherwise).

## What score.py measures (all from the finished project)

- Scenarios: the 20 scenarios (5 public `p*`, 15 hidden `h*`) are replayed against the merged `main` through a real `qal serve`
  (`--allow db:<fresh>,time --init setup`, one fresh database per scenario) and compared on status, JSON body (`$var`/`$type`),
  headers and exact body. Also: stored tests on `main` (`qal test`), definitions on `main`, `todo()` stubs left.
- `qal verify --trust team.json` (team.json exported from the project): ok, records, members.
- Records (decoded from `.qal/store.db`): records per member, change sets (owner, defines, submits, reviews), merges by kind
  (`fast_forward` / `replayed`), merge conflicts (failed `qal_merge` with E0910 from the call log), rebases, `needs_rebase` marks,
  review rounds per change set (verdicts `changes`/`failed`), the merge gate (each merge cites a non-owner `approve` and `verified`),
  parallelism = the maximum number of distinct members (not the owner) with a record in any 5-minute window (`writers_5min_define`
  for define/rename/delete records, `writers_5min_any` for all), wall time first member record to last merge.
- Call log (`.qal/team-log.db`, server side, not an estimate): calls and failures per member and per error code, lease conflicts
  (E0901), stale-base (E0904), merge refusals by code.
- Provenance: every non-`std.` definition on `main` has a history entry for its current hash written by a team member session, never
  `cli`/the owner/unknown; no owner write records (define/rename/delete/merge/review).
- `criterion6` block: all scenarios pass, >= 3 members with define records in 5 minutes, at least one replayed (non-fast-forward)
  merge, merge gate satisfied, verify ok, provenance ok. (The HTTP harness `qal/crates/qal-serve/tests/shop_http.rs` is a separate
  step: export `qal export "" --out shop.qalpkg --name shop` in the project, copy it to `qal/examples/shop/`, run
  `cargo test -p qal-serve --test shop_http` as in the spec; it is not run by score.py.)
- Tokens and tool calls per agent and in total come only from `--agent-stats`.

## Known limits of the setup (be honest in the report)

- Roles are guard rails: all agents run as the same OS user and could read each other's directories and the key files; the prompts forbid it.
- Review comment text is readable through `qal_change show` (notices carry only the verdict and a comment count); the prompts say so.
- `qal_whoami {wait_s}` waits are not counted against the 150-call cap (separate cap of 120 waits per agent).
