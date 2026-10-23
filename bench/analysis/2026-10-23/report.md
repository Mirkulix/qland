# Token attribution

## QAL against Rust (per run)

| measure | QAL | Rust | factor |
|---|---|---|---|
| processed per run | 301207 | 145929 | 2.06 |
| rounds per run | 5.00 | 2.95 | 1.69 |
| processed per round | 60241 | 49447 | |
| dispatch tokens per run | 52670 | 41208 | 1.28 |

| rounds `env.*` per run | 0.89 | 0.50 | |
| rounds `lang.*` per run | 0.52 | 0.00 | |
| rounds `ok.*` per run | 2.57 | 1.43 | |
| rounds `prompt.*` per run | 0.99 | 1.00 | |

## qal (82 runs)

| category | processed | new | output | calls | processed/run |
|---|---|---|---|---|---|
| ok.define | 5617408 | 225340 | 14720 | 90 | 68505 |
| ok.run | 5450701 | 56663 | 9845 | 83 | 66472 |
| prompt.task | 3804191 | 1551182 | 37811 | 81 | 46393 |
| env.guard | 2419118 | 133666 | 40597 | 44 | 29501 |
| ok.other | 2279386 | 33821 | 9847 | 38 | 27797 |
| lang.name | 1298777 | 63528 | 3021 | 20 | 15839 |
| env.read_task | 1081632 | 255076 | 37095 | 20 | 13191 |
| lang.type | 718756 | 49375 | 423 | 11 | 8765 |
| lang.run_failed | 645854 | 19232 | 285 | 8 | 7876 |
| env.tool_error | 551956 | 4903 | 3144 | 9 | 6731 |
| prompt.reference | 529386 | 185610 | 10090 | 62 | 6456 |
| lang.parse | 175361 | 2639 | 1616 | 3 | 2139 |
| lang.args | 65996 | 5382 | 3 | 1 | 805 |
| final | 60486 | 21991 | 6 | 1 | 738 |
| lang.other_diag | 0 | 0 | 0 | 0 | 0 |

Shares of processed tokens: ok 54.0 %, prompt 17.5 %, env 16.4 %, lang 11.8 %, final 0.2 %

Top 3: ok.define (5617408), ok.run (5450701), prompt.task (3804191)

Umgebung (env) 16.4 % / Sprache (lang) 11.8 %

Rounds per run: mean 5.0, median 5.0, max 18; retries 0

Processed tokens of round 1: 52848, 5: 59553, 10: 76703

Diagnostic codes of failed calls: E0201 34, E0510 25, E0203 19, E0502 7, E0701 4, E0100 3, E0601 3, E0504 2

## rust (82 runs)

| category | processed | new | output | calls | processed/run |
|---|---|---|---|---|---|
| ok.other | 6054348 | 258135 | 11277 | 117 | 73834 |
| prompt.task | 3937727 | 1376700 | 18458 | 82 | 48021 |
| env.read_task | 1085421 | 111012 | 28745 | 24 | 13237 |
| env.guard | 782958 | 36670 | 14737 | 17 | 9548 |
| final | 59858 | 3906 | 4 | 1 | 730 |
| rust.panic | 45865 | 1617 | 64 | 1 | 559 |
| prompt.reference | 0 | 0 | 0 | 0 | 0 |

Shares of processed tokens: ok 50.6 %, prompt 32.9 %, env 15.6 %, final 0.5 %, rust 0.4 %

Top 3: ok.other (6054348), prompt.task (3937727), env.read_task (1085421)

Umgebung (env) 15.6 % / Sprache (lang) 0.0 %

Rounds per run: mean 3.0, median 2.0, max 6; retries 0

Processed tokens of round 1: 48021, 5: 47326, 10: –

Diagnostic codes of failed calls: none

## Per task (means per run)

| task | QAL processed | QAL rounds | QAL biggest | Rust processed | Rust rounds | Rust biggest | factor |
|---|---|---|---|---|---|---|---|
| 01-word-count | 258795 | 4.8 | ok.run | 125613 | 2.5 | ok.other | 2.06 |
| 02-roman | 220811 | 3.8 | ok.define | 146573 | 3.0 | ok.other | 1.51 |
| 03-csv-sum | 259559 | 4.5 | ok.define | 170920 | 3.5 | ok.other | 1.52 |
| 04-binary-search | 237373 | 4.2 | ok.define | 144754 | 3.0 | ok.other | 1.64 |
| 05-brackets | 210850 | 3.8 | ok.define | 100015 | 2.0 | ok.other | 2.11 |
| 06-bank | 268906 | 4.5 | ok.define | 186418 | 3.8 | ok.other | 1.44 |
| 07-tokenizer | 156767 | 2.8 | ok.define | 100685 | 2.0 | ok.other | 1.56 |
| 08-expr-eval | 846569 | 11.7 | ok.run | 179518 | 3.5 | ok.other | 4.72 |
| 09-rle | 260727 | 4.5 | ok.define | 181670 | 3.8 | ok.other | 1.44 |
| 10-file-stats | 246042 | 4.2 | prompt.task | 112432 | 2.2 | ok.other | 2.19 |
| 11-prime-factors | 242658 | 4.2 | ok.define | 122746 | 2.5 | ok.other | 1.98 |
| 12-sort-words | 296519 | 5.2 | ok.run | 121634 | 2.5 | ok.other | 2.44 |
| 13-word-freq | 370744 | 6.2 | ok.run | 157415 | 3.2 | ok.other | 2.36 |
| 14-bst | 367342 | 6.2 | ok.run | 123378 | 2.5 | prompt.task | 2.98 |
| 15-caesar | 269830 | 4.8 | ok.define | 175296 | 3.5 | ok.other | 1.54 |
| 16-anagrams | 140927 | 2.5 | ok.define | 121953 | 2.5 | ok.other | 1.16 |
| 17-word-wrap | 351122 | 5.8 | ok.define | 171126 | 3.5 | ok.other | 2.05 |
| 18-date-diff | 247270 | 4.2 | ok.define | 134506 | 2.8 | ok.other | 1.84 |
| 19-base-convert | 238800 | 4.2 | ok.define | 169555 | 3.2 | ok.other | 1.41 |
| 20-ini-query | 259858 | 4.5 | ok.define | 157532 | 3.2 | ok.other | 1.65 |
