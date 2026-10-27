# QAL-OS M0 – bare-metal boot

The first real boot of the QAL-OS substrate: a 32-bit x86 kernel with a
multiboot header, its own stack, COM1 serial driver and a halt loop. **No Linux,
no POSIX, no libc, no bootloader beyond QEMU's multiboot loader.**

## Build and run

```sh
cd qal-os/kernel-asm
make
make run          # QEMU, serial on stdio; Ctrl-C to stop
```

Measured serial output (QEMU 11.0.3, macOS):

```
QAL-OS M0: bare-metal kernel booted
No Linux. No POSIX. Agent substrate alive.
Next: memory, interrupts, scheduler, capabilities.
```

## What M0 is

- multiboot entry, 32-bit protected mode (QEMU starts it there),
- 4 KiB stack in `.bss`,
- COM1 init at 0x3F8 (115200 8N1),
- serial string output,
- `hlt` loop.

## What M0 is not

- no paging, no heap, no interrupts, no scheduler,
- no capabilities, no IPC, no object store,
- no QAL runtime, no agent execution,
- only QEMU/multiboot; real hardware/UEFI is M1+.

M1 is the next step: page allocator, page tables, GDT/IDT, timer interrupt,
two tasks and a context switch. M2 adds capability IPC and a QAL interpreter.
