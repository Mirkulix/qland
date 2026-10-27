# QAL-OS M11 – SHA-256 content store + QAL-1 agents

M11 replaces the FNV-1a placeholder with a real **SHA-256** implementation
(freestanding C, no SSE, no libc). The store compares all 32 hash bytes on get.

## What runs

- syscall 8 hashes a user program with SHA-256 and stores it in one of four
  16-byte slots (32-byte hash per slot);
- syscall 9 compares all 32 bytes of a user-supplied hash and copies the stored
  program to a validated user pointer;
- task A runs its QAL-1 program → **9**; task B → **5**;
- both then loop `putc` under the preemptive 100 Hz timer.

## Verified in QEMU 11.0.3

```
QAL-OS M11: SHA-256 store + QAL-1 agents
[store: put]
[store: get]
[qal-1 = 5]
BBBB...AAAA...
```

2-second run: 2× put, 2× get, one `[qal-1 = 9]`, one `[qal-1 = 5]`,
**384,556 A** / **380,227 B** characters.

The expected SHA-256 vectors for the two programs are embedded as the user
hash buffers; the kernel hashes the program itself, so a match proves the
kernel SHA and the store lookup agree.

## Honest limits

- four fixed 16-byte slots, no persistence/virtio, no collision handling;
- QAL-1 is a toy stack language;
- no page-fault handling, SMP or UEFI;
- SHA-256 is implemented but not formally verified.

Next M12: persistent virtio-block backing, slot GC, larger QAL subset, and a
timer-driven agent quantum.
