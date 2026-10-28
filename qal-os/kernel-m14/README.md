# QAL-OS M14 – two-sector ATA SHA-256 store + GC

M14 splits the persistent store across **two ATA sectors**:

- **sector 1:** 128 hash bytes + 16 length bytes,
- **sector 2:** 64 data bytes.

## What runs

- `ata_write`/`ata_read` take the LBA in `%ebx`;
- syscall 10 saves both sectors, syscall 11 loads both;
- syscall 12 frees a hash (GC), and a later get misses;
- two ring-3 QAL-1 agents run under the preemptive timer.

## Verified in QEMU 11.0.3

```
QAL-OS M14: two-sector ATA store + slot GC
[store: put]
[disk: save]
[disk: load]
[store: get]
[qal-1 = 5]
...
[store: free]
[store: miss]
```

2-second run: 2× save, 1× load, free 1, miss 1, QAL 9 and 5,
**569,283 A** / **561,054 B**.

Disk image:

```text
sector 1: 2e5ff0f672b8946e…   (SHA-256 hash of task B)
sector 2: 010201031020ff00…   (task B's QAL-1 program)
```

## Honest limits

- still four slots, no virtio-blk, no partition/filesystem;
- no write-back cache or crash consistency;
- QAL-1 is a toy stack language;
- no SMP/UEFI.

Next M15: virtio-blk, more slots, SMP.
