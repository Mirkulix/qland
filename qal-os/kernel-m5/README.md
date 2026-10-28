# QAL-OS M5 – capability endpoints + copy-in

M5 turns the two capability words into **kernel endpoint objects** and copies the
user program into a kernel buffer after validating the whole range.

## What runs

- endpoints `ep1`/`ep2` each have `live` and `mail` state;
- caps 1/2 route to ep1/ep2;
- syscall 3 writes the endpoint mailbox; syscall 4 revokes the endpoint;
- syscall 2 validates `[ptr, ptr+64)` inside `[0x200000,0x400000)` and copies
  64 bytes into `qbuf` before the QAL-0 interpreter runs.

## Verified in QEMU 11.0.3

```
QAL-OS M5: endpoints, copy-in, revocable capabilities
agent: qal-0 = 3
[syscall: bad user ptr]
[ep1: send ok]
[ep1: revoked]
[ep1: denied]
```

The result `3` is computed from the **copied** kernel buffer; the kernel pointer
is rejected; ep1 accepts, is revoked, and the next send is denied.

## Honest limits

- no receive/copy-out syscall yet,
- QAL-0 is a toy bytecode,
- no ring-3 scheduler (timer masked off),
- endpoints are static kernel structures, not a general object table,
- no transitive revocation, no object store, no page-fault handling.

Next M6: receive/copy-out, ring-3 scheduling, and a real QAL subset interpreter.
