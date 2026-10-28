# Rerun 2026-10-23 (Phase 6 + 6b), protocol notes

Same protocol as `bench/rerun-2026-10-22` from entry 37 on: 20 tasks x 2 arms x 2 runs = 80 entries, seed `qal-p6-rerun`, Sonnet, exact prompt copied into the work dir as `TASK.md`, controller session never in a worktree, at most 3 attempts per entry (retry message of `bench/run.py`), transcripts audited by `bench/run.py audit`.

Binary: `qal/target/release/qal` built from `Mac` at the merge of Phase 6b (commit c58ba42b, test fix fbd6add0): Phase 6 Tasks 5–6 (`qal_run` cases, define runs tests, E0201 sequence-let hint, reference sharpened), the deployment-test fixes (`qal_run` isError only on runtime errors, `qal_query` outline without std/hashes), and Phase 6b (schema 7, `qal_recall`, hints with memory).

Difference from 2026-10-22: every QAL project starts with the seed corpus of corrections imported (`qal init` + `qal corrections import bench/analysis/corrections-2026-10-23.json`, 10 records), so a diagnostic of a known shape carries an `earlier fix` hint from the first call. The first two QAL entries (05-brackets run1, 02-roman run2) got the import a minute after dispatch instead of before (the driver was fixed then); both before their first define reached an error, as far as the transcripts show.

Prompt size QAL: 26 111 bytes (188 lines) with the reference's recall line; Rust unchanged.
