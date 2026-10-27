# QAL-OS M6 – receive/copy-out + QAL-1 interpreter

M6 adds the return path: an endpoint receive copies a value back to a validated
user pointer. The QAL-0 bytecode grows into QAL-1 (`add`, `mul`, `sub`, `print`).

## What runs

- QAL-1 program `push 1; push 2; add; push 3; mul; print; halt` → `9`;
- program is copied into `qbuf` after a range check (copy-in);
- cap1 send → ep1, revoke ep1, next send denied;
- cap2 send → ep2, receive cap2 → copies `0x52` to the user pointer
  `0x300100` and prints `[ep2: recv 0x52]` (copy-out).

## Verified in QEMU 11.0.3

```
QAL-OS M6: receive/copy-out, QAL-1 interpreter, endpoints
agent: qal-1 = 9
[syscall: bad user ptr]
[ep1: send ok]
[ep1: revoked]
[ep1: denied]
[ep2: send ok]
[ep2: recv 0x52]
```

## Honest limits

- QAL-1 is still a tiny stack language, not QAL;
- no ring-3 scheduler (timer masked off), one user agent;
- endpoints are static kernel objects, no general handle table;
- no page-fault handling, no object store, no SMP/UEFI.

Next M7: ring-3 timer scheduling, a real QAL subset (Int/Bool/fn/test), and a
content-addressed object store.
