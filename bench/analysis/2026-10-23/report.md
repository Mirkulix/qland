# Token attribution

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
