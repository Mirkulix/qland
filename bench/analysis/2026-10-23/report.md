# Token attribution

## QAL against Rust (per run)

| measure | QAL | Rust | factor |
|---|---|---|---|
| processed per run | 299376 | 145224 | 2.06 |
| rounds per run | 5.00 | 2.95 | 1.69 |
| processed per round | 59875 | 49208 | |
| dispatch tokens per run | 52670 | 41208 | 1.28 |

| rounds `env.*` per run | 0.83 | 0.50 | |
| rounds `lang.*` per run | 0.52 | 0.00 | |
| rounds `ok.*` per run | 2.63 | 1.43 | |
| rounds `prompt.*` per run | 1.00 | 1.00 | |

## qal (82 runs)

| category | processed | new | output | calls | processed/run |
|---|---|---|---|---|---|
| ok.run | 5769576 | 59301 | 4673 | 88 | 70361 |
| ok.define | 5605733 | 225340 | 3045 | 90 | 68363 |
| prompt.task | 3602377 | 1490327 | 1615 | 81 | 43931 |
| env.guard | 2379125 | 133666 | 604 | 44 | 29014 |
| ok.other | 2271839 | 33821 | 2300 | 38 | 27705 |
| lang.name | 1296090 | 63528 | 334 | 20 | 15806 |
| env.read_task | 1044641 | 255076 | 104 | 20 | 12740 |
| lang.type | 718475 | 49375 | 142 | 11 | 8762 |
| prompt.reference | 690255 | 246465 | 5341 | 82 | 8418 |
| lang.run_failed | 645625 | 19232 | 56 | 8 | 7873 |
| env.tool_error | 224793 | 2265 | 28 | 4 | 2741 |
| lang.parse | 173859 | 2639 | 114 | 3 | 2120 |
| lang.args | 65996 | 5382 | 3 | 1 | 805 |
| final | 60486 | 21991 | 6 | 1 | 738 |
| lang.other_diag | 0 | 0 | 0 | 0 | 0 |

Shares of processed tokens: ok 55.6 %, prompt 17.5 %, env 14.9 %, lang 11.8 %, final 0.2 %

Top 3: ok.run (5769576), ok.define (5605733), prompt.task (3602377)

Umgebung (env) 14.9 % / Sprache (lang) 11.8 %

Rounds per run: mean 5.0, median 5.0, max 18; retries 0

Processed tokens of round 1: 52349, 5: 59320, 10: 76703

Diagnostic codes of failed calls: E0201 34, E0510 25, E0203 19, E0502 7, E0701 4, E0100 3, E0601 3, E0504 2

## rust (82 runs)

| category | processed | new | output | calls | processed/run |
|---|---|---|---|---|---|
| ok.other | 6049362 | 258135 | 6291 | 117 | 73773 |
| prompt.task | 3927051 | 1376700 | 7782 | 82 | 47891 |
| env.read_task | 1057756 | 111012 | 1080 | 24 | 12899 |
| env.guard | 768486 | 36670 | 265 | 17 | 9372 |
| final | 59858 | 3906 | 4 | 1 | 730 |
| rust.panic | 45865 | 1617 | 64 | 1 | 559 |
| prompt.reference | 0 | 0 | 0 | 0 | 0 |

Shares of processed tokens: ok 50.8 %, prompt 33.0 %, env 15.3 %, final 0.5 %, rust 0.4 %

Top 3: ok.other (6049362), prompt.task (3927051), env.read_task (1057756)

Umgebung (env) 15.3 % / Sprache (lang) 0.0 %

Rounds per run: mean 3.0, median 2.0, max 6; retries 0

Processed tokens of round 1: 47891, 5: 47274, 10: –

Diagnostic codes of failed calls: none

## Per task (means per run)

| task | QAL processed | QAL rounds | QAL biggest | Rust processed | Rust rounds | Rust biggest | factor |
|---|---|---|---|---|---|---|---|
| 01-word-count | 257775 | 4.8 | ok.run | 125332 | 2.5 | ok.other | 2.06 |
| 02-roman | 218580 | 3.8 | ok.define | 146064 | 3.0 | ok.other | 1.50 |
| 03-csv-sum | 257496 | 4.5 | ok.define | 169564 | 3.5 | ok.other | 1.52 |
| 04-binary-search | 235538 | 4.2 | ok.define | 144358 | 3.0 | ok.other | 1.63 |
| 05-brackets | 209756 | 3.8 | ok.define | 99636 | 2.0 | ok.other | 2.11 |
| 06-bank | 265719 | 4.5 | ok.define | 184661 | 3.8 | ok.other | 1.44 |
| 07-tokenizer | 155210 | 2.8 | ok.define | 100196 | 2.0 | ok.other | 1.55 |
| 08-expr-eval | 843351 | 11.7 | ok.run | 178458 | 3.5 | ok.other | 4.73 |
| 09-rle | 258797 | 4.5 | ok.define | 180267 | 3.8 | ok.other | 1.44 |
| 10-file-stats | 244207 | 4.2 | prompt.task | 111907 | 2.2 | ok.other | 2.18 |
| 11-prime-factors | 241249 | 4.2 | ok.define | 122435 | 2.5 | ok.other | 1.97 |
| 12-sort-words | 295603 | 5.2 | ok.run | 121367 | 2.5 | ok.other | 2.44 |
| 13-word-freq | 368671 | 6.2 | ok.run | 156518 | 3.2 | ok.other | 2.36 |
| 14-bst | 365604 | 6.2 | ok.run | 122712 | 2.5 | prompt.task | 2.98 |
| 15-caesar | 268307 | 4.8 | ok.define | 174608 | 3.5 | ok.other | 1.54 |
| 16-anagrams | 140424 | 2.5 | ok.define | 121651 | 2.5 | ok.other | 1.15 |
| 17-word-wrap | 348318 | 5.8 | ok.define | 169814 | 3.5 | ok.other | 2.05 |
| 18-date-diff | 245641 | 4.2 | ok.define | 133868 | 2.8 | ok.other | 1.83 |
| 19-base-convert | 237089 | 4.2 | ok.define | 169353 | 3.2 | ok.other | 1.40 |
| 20-ini-query | 258209 | 4.5 | ok.define | 156918 | 3.2 | ok.other | 1.65 |
