# QAL-OS M15 – eight-slot two-sector ATA SHA-256 store

M15 doubles the persistent store from four to **eight slots**:

- sector 1: **256** hash bytes + **32** length bytes,
- sector 2: **128** data bytes,
- GC scans all eight slots.

## Verified in QEMU 11.0.3

```
QAL-OS M15: eight-slot two-sector ATA SHA-256 store
[store: put]
[disk: save]
[disk: load]
[store: get]
[qal-1 = 5]
...
[store: free]
[store: miss]
```

2-second run: 2× save, 1× load, free 1, miss 1, QAL 9/5,
**550,076 A** / **559,532 B**. Disk: sector 1 SHA-256, sector 2 QAL program.

## Honest limits

- eight slots still fit in two ATA sectors; no virtio-blk yet;
- no partition/filesystem, no crash consistency;
- QAL-1 is a toy stack language;
- no SMP/UEFI.

A virtio-blk attempt was started but did not complete a request; ATA remains the
working backend. Next: virtio-blk or SMP.
