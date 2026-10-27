# QAL-OS M10 – content-addressed store + QAL-1 agents

M10 adds a small in-kernel object store: programs are stored under a **32-bit
FNV-1a hash** and loaded back by hash before execution.

## What runs

- syscall 8 `store_put`: validates `[ptr, ptr+len)`, computes FNV-1a, stores
  16 bytes in one of four slots;
- syscall 9 `store_get`: finds the slot by hash and copies the bytes to a
  validated user pointer;
- task A stores/runs its QAL-1 program → **9**; task B → **5**;
- both then loop `putc` under the preemptive 100 Hz timer.

## Verified in QEMU 11.0.3

```
QAL-OS M10: content-addressed store + QAL-1 agents
[store: put]
[store: get]
[qal-1 = 5]
BBBB...AAAA...
```

A 2-second run contains:

- `[store: put]` × 2, `[store: get]` × 2,
- `[qal-1 = 9]` × 1, `[qal-1 = 5]` × 1,
- **380,492 A** / **390,325 B** characters.

## Honest limits

- FNV-1a is a **non-cryptographic** 32-bit hash, not SHA-256;
- four fixed 16-byte slots, no disk/persistence, no collision handling;
- QAL-1 is still a toy stack language;
- no page-fault handling, SMP or UEFI.

Next M11: SHA-256 store, persistent block backing (virtio), collision handling,
and a larger QAL subset.
