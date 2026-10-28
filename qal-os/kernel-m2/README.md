# QAL-OS M2 – paging + capability-gated IPC

M2 adds the first memory and capability primitives to the bare-metal kernel.
Still no Linux, no POSIX, no libc.

## What runs

- **Paging:** static page directory + one page table identity-map the first 4 MiB;
  `CR0.PG` is enabled before the tasks start.
- **Scheduler:** the M1 timer/ESP round-robin is unchanged (100 Hz).
- **Capability gate:** each task has a capability word; a send requires bit 0.
  - task A (`caps=1`) sends `0x51` to the mailbox and prints `[A:cap ok -> B]`,
  - task B (`caps=0`) tries to send, is denied and prints `[B:cap denied]`,
  - task B receives the mailbox and prints `[B:recv]`.

## Verified in QEMU 11.0.3

```
QAL-OS M2: paging + capability-gated IPC + scheduler
[B:cap denied]B[A:cap ok -> B]A[B:recv]BABABABABAB...
```

Both the denial and the granted send are visible; the alternating `B`/`A` proves
the scheduler still runs with paging enabled.

## Not yet

- no dynamic page allocator, no user mode, no page-fault handling,
- no kernel capability broker / revocation, only a task-word check,
- no QAL interpreter or agent runtime,
- no object store, virtio, UEFI, SMP.

Next M3: user mode + trap gate, capability endpoints with revocation, a small QAL
interpreter and the first agent task.
