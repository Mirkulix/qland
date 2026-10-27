# QAL-OS M3 – ring 3 agent + syscall + QAL-0 interpreter

M3 leaves ring 0. A user task runs at CPL 3, calls `int 0x80`, and the kernel
executes a tiny QAL-0 bytecode program and prints the result.

## What runs

- 6-entry GDT (kernel/user code+data, TSS) and a 104-byte TSS with `esp0`;
- user pages: first 2 MiB supervisor-only, 2–4 MiB user-accessible (code at
  `0x200000`, stack at `0x300000`);
- syscall gate vector `0x80`, DPL 3, `iret` into ring 3;
- user blob calls `syscall 2` with a QAL-0 program pointer;
- kernel QAL-0 interpreter: `0x01 imm8` push, `0x10` add, `0xFF` halt, then prints.

## Verified in QEMU 11.0.3

```
QAL-OS M3: ring3 agent, syscall, QAL-0 interpreter
agent: qal-0 = 3
```

The user program is `push 1; push 2; add; halt`; the kernel prints `3`.

## Honest limits

- **QAL-0 is a toy**, not the QAL language/compiler.
- The timer/scheduler is masked off in this demo; one user agent runs.
- No capability broker/revocation yet, no object store, no page-fault handling.
- Syscall validation is minimal (`eax == 2`); no copy-in/copy-out checks.
- Identity mapping only; no dynamic page allocator.

Next M4: capability endpoints with revocation, syscall validation/copy-in,
scheduler with ring-3 tasks, and a real QAL subset interpreter.
