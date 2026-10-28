# Benchmark rerun after Phase 6 and 6b (run 2026-10-23/24)

Protocol: `notes.md` in this directory. 20 tasks × 2 arms × 2 runs, Sonnet, seed `qal-p6-rerun`, exact prompt as `TASK.md`, up to 3 attempts. Binary from `Mac` at the Phase 6b merge. Every QAL project started with the seed corpus of corrections imported. Token analysis: `bench/analysis/2026-10-24/summary.json` (`bench/analyze.py`, 0 unclassified results). Comparison basis: `bench/analysis/2026-10-23/summary.json`, the analysis behind the Phase 6 gate decision.

## Result

**80 of 80 entries solved at the first attempt.** No retry, no invalid run.

| | before (analysis 2026-10-23) | now | change |
|---|---|---|---|
| QAL processed tokens per run | 301 207 | 212 441 | −29 % |
| Rust processed tokens per run | 145 929 | 149 650 | +3 % |
| **Factor QAL / Rust** | **2.06** | **1.42** | −31 % |
| QAL rounds per run | 5.00 | 3.95 | −21 % |
| Rust rounds per run | 2.95 | 3.30 | (TASK.md protocol adds a Read round) |
| QAL language-error rounds (`lang.*`) per run | 0.52 | **0.10** | −81 % |
| QAL tokens in language-error rounds per run | 35 424 | 5 919 | −83 % |
| diagnostic codes of failed QAL calls | E0201 34, E0510 25, E0203 19, E0502 7, … | E0100 3, E0203 2 | |

The Phase 6 design expected about 1.36 after Tasks 5–6. The measured 1.42 is close, and inside the design's middle band (1.2 to 1.5). The goal of below 1.2 is not reached.

## What caused it

- **Language errors almost vanished.** E0201 (an unknown name after `;`, a built-in used as a value) was the largest class with 34 occurrences and is gone. E0510 (25) and E0502 (7) are gone too. This is the effect of Phase 6 Tasks 5–6: the E0201 hint names the `let` that ended at `;`, the reference was sharpened at the four error classes, and `qal_run` gained `cases`.
- **The memory (Phase 6b) contributed almost nothing in this benchmark.** Across all 40 QAL runs only 4 tool results carried diagnostics at all, and only 1 of them got an `earlier fix` hint. No agent called `qal_recall` or the outline. The memory's hints have nothing left to fix when Phase 6 already prevents the errors. The 6b acceptance criterion, that `lang.*` rounds fall measurably, is met in the numbers. **It cannot be attributed to 6b**, and the honest reading is that 6b is unproven by this benchmark. Per the design's stop criterion it stays as built, as a corpus source for Phase 8 and for longer, error-prone work, such as the living program where an unproven obligation was inherited over 12 stages. It is not extended further for now.
- **The remaining gap is mostly the prompt.** QAL's prompt is 26 KB against Rust's 5.5 KB. It is read again in every round. With 3.95 against 3.30 rounds, the rounds explain about 1.20 of the factor and the larger context per round about 1.19.

## Next

1. Shorten the QAL prompt: tool descriptions and reference. This is the largest remaining lever. Target: factor below 1.2.
2. `define` relink option. It is irrelevant for this benchmark, but cost Haiku a round in 11 of 12 changes in the living program.
3. A pressure test of the guarantees: change requests without a full specification and without a test requirement.
