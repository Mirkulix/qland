# QAL-OS M8 – preemptive ring-3 timer

M8 makes the M7 cooperative scheduler **preemptive**: the PIT fires at 100 Hz,
the timer ISR saves the ring-3 frame, switches the kernel frame and TSS `esp0`,
and `iret`s into the other user task.

## What runs

- two ring-3 tasks A/B, each loops `putc('A'|'B')`;
- timer ISR handles CPL3 → CPL0 stack switch through TSS `esp0`;
- syscall handlers run with `cli`, so the timer cannot nest inside a syscall;
- per-task kernel stacks; TSS `esp0` updated on every switch.

## Verified in QEMU 11.0.3

```
QAL-OS M8: preemptive ring3 timer, two user tasks
BBBB...AAAA...
```

Over a 2-second run the serial log contains **30,685 A** and **29,340 B**
characters, i.e. both tasks are preempted and resumed by the timer, not by a
cooperative yield.

## Honest limits

- the tasks only call `putc`; QAL-1 runs separately in M6;
- no object store, no page-fault handling, no SMP/UEFI;
- no priority/fairness policy beyond strict A/B alternation.

Next M9: QAL-1 inside both ring-3 tasks, content-addressed object store, and a
timer-driven agent quantum.
