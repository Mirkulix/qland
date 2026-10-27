# QAL-OS M1 – timer, GDT/IDT, two tasks

M1 makes the bare-metal kernel alive. It still has no Linux, no POSIX, no libc.

## What runs

- GDT with ring-0 code/data, IDT with vector 32 (timer),
- 8259 PIC remapped to 0x20/0x28, only IRQ0 unmasked,
- PIT at 100 Hz,
- two pre-built iret frames on separate 4 KiB stacks,
- timer ISR saves `pusha`, round-robins `ESP`, sends EOI, `popa; iret`,
- task A and task B print one character each and `hlt`.

## Verified

```sh
cd qal-os/kernel-m1
make
```

QEMU 11.0.3, serial capture:

```
QAL-OS M1: timer, GDT/IDT, two tasks
BABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABAB...
```

The alternating `B`/`A` proves the timer interrupt and context switch work.

## Not yet

- no paging/page allocator, no heap,
- no capability IPC, no object store,
- no QAL interpreter/agent runtime,
- no SMP, no UEFI/real hardware.

Next: M2 = page allocator + page tables, capability IPC and a QAL interpreter.
