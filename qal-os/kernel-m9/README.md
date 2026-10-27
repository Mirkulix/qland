# QAL-OS M9 – QAL-1 inside two preemptive ring-3 tasks

M9 combines M6 (QAL-1, copy-in) with M8 (preemptive ring-3 timer). Both user
tasks run a QAL-1 program before entering their putc loop.

## What runs

- task A: QAL-1 `(1+2)*3` → **9**, then loops `putc('A')`;
- task B: QAL-1 `2+3` → **5**, then loops `putc('B')`;
- syscall 2 validates and copies 64 bytes into `qbuf` before interpreting;
- timer ISR preempts both ring-3 tasks at 100 Hz with per-task kernel stacks.

## Verified in QEMU 11.0.3

```
QAL-OS M9: QAL-1 in two preemptive ring3 tasks
[qal-1 = 5]
BBBB...AAAA...
```

A 2-second run contains **both** `[qal-1 = 9]` and `[qal-1 = 5]` (one each), and
**361,948 A** / **361,038 B** characters, so both QAL-1 agents run and are
preempted by the timer.

## Honest limits

- QAL-1 is a toy stack language, not QAL;
- no content-addressed object store yet;
- no page-fault handling, no SMP/UEFI, no persistence;
- the putc loop is noise; the evidence is the two QAL lines plus A/B counts.

Next M10: content-addressed object store, QAL-1 programs stored/loaded by hash,
and a timer-driven agent quantum.
