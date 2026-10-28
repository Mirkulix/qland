# The living program

The question the deployment tests of 2026-10-23 (`bench/einsatz-2026-10-23/report.md`) left open:
do QAL's guarantees (contracts proven at define time, tests that run with every definition, a
store that says what changed) catch mistakes that a Rust workflow lets through, once a program is
changed many times by agents that have never seen it before? Six runs on small tasks made no
mistake to catch. This experiment makes a program live through twelve change requests.

## Protocol

- **Program:** an order system (products, customers, orders, report), specified in `spec.py`,
  with a reference implementation for every stage in `oracle.py`.
- **Stages:** stage 0 builds it with contracts and tests; stages 1–12 are change requests
  (quantity discount, customer tiers, cancellation, tax, credit, restocking, backorders, price
  changes, top products, returns, promotions, multi-line orders). Each stage's statement is the
  change plus the full specification after it, with two public examples.
- **Hidden tests** per stage (`tasks/<NN-living-x>/tests.json`): hand-written scripts for the
  stage's rules and traps, every earlier stage's scripts re-evaluated under the new rules, and six
  seeded random command streams; 16 cases at stage 0 up to 50 at stage 12.
- **Lines:** QAL and Rust, each with Sonnet (run 1) and Haiku (run 2). Every stage is a fresh
  agent with only the task text and the code the previous stage of its line left (the QAL store,
  or `src/main.rs`), seeded by `drive.py stage` from the previous stage's archive.
- **Attempts:** a stage that fails the hidden tests gets one retry with the harness's retry
  message (the same agent, `bench/run.py`'s rule); after that the line continues on the failing
  code. Bugs may persist, as they do in real codebases.
- **Measured:** hidden-test results per stage and attempt; rounds, processed tokens and
  duration from the transcripts (`bench/analyze.py`); and what the program's own verification
  caught during a run (`drive.py finish` lists define/test replies with failed tests, refuted
  proofs and failed `cargo test` runs).

The work dirs live under `/tmp/qal-living`, the state (schedule, results, work dirs, dispatch
times) under `state/` (git-ignored; the analysis copies what it needs into the report).

## Running it

```
python3 bench/living/build.py                 # (re)generate the task set
python3 bench/living/drive.py stage 0 qal 1   # prepare a stage of a line; dispatch an agent on its TASK.md
python3 bench/living/drive.py finish 0 qal 1 <agent-name-prefix> 1 done
python3 bench/living/drive.py status
```
