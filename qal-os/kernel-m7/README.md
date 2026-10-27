# QAL-OS M7 – cooperative ring-3 scheduler

M7 runs **two user tasks at CPL 3** with separate kernel stacks, TSS `esp0`
switching, and a cooperative `yield` syscall.

## What runs

- two user blobs at `0x200000` (A) and `0x201000` (B), stacks `0x300000`/`0x301000`;
- each task: putc char (syscall 6) → yield (syscall 7) → repeat;
- on yield the kernel saves the ring-3 frame, switches `ESP` and TSS `esp0`,
  restores the other frame and `iret`s back to CPL 3.

## Verified in QEMU 11.0.3

```
QAL-OS M7: cooperative ring3 scheduler, two user tasks
ABABABABABABABABABABABABABABABABABAB...
```

The alternating output proves two ring-3 tasks switch with separate kernel stacks.

## Honest limits

- **cooperative**, not preemptive: the timer is masked off; a task must yield;
- no QAL interpreter in this kernel (M6 has QAL-1);
- no general handle table / object store / page-fault handling / SMP.

Next M8: preemptive ring-3 timer scheduling (handle CPL0 nesting correctly), QAL-1
inside both tasks, and a content-addressed object store.
