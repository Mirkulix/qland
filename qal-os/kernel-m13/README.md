# QAL-OS M13 – ATA store + slot GC

M13 adds **slot garbage collection** to the persistent SHA-256 store: a user
can free a hash, and a subsequent get is guaranteed to miss.

## What runs

- syscall 12 `store_free`: validates a 32-byte hash pointer, finds the slot,
  zeroes the 32-byte hash and the length, and prints `[store: free]`;
- task A: put → save → get → QAL-1 → **free A** → get A → **miss**;
- task B: put → save → load → get → QAL-1;
- both then loop `putc` under the preemptive timer.

## Verified in QEMU 11.0.3

```
QAL-OS M13: ATA store + slot GC
[store: put]
[disk: save]
[disk: load]
[store: get]
[qal-1 = 5]
...
[store: free]
[store: miss]
```

2-second run: 2× put, 2× get, **1× free**, **1× miss**, 2× save, 1× load,
QAL 9 and 5, **580,216 A** / **578,839 B**.

## Honest limits

- four slots in one ATA sector, no partition/filesystem;
- free simply zeroes the hash/len, no compaction;
- no virtio-blk, no multiple sectors, no write-back cache;
- QAL-1 is a toy stack language.

Next M14: virtio-blk + multi-sector store, then SMP.
