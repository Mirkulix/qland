# QAL-OS M4 – capability revocation + validated syscalls

M4 makes the ring-3 agent's capabilities revocable and validates user pointers
before the kernel reads them.

## What runs

- ring-3 user blob at `0x200000`, 2–4 MiB user pages,
- `syscall 2`: QAL-0 interpreter, **pointer validated** to `[0x200000, 0x400000)`,
- `syscall 3`: capability send; cap 1/2 must be live,
- `syscall 4`: revoke cap 1 or 2 (zeroes the capability word),
- a bad kernel pointer (`0x100000`) is rejected before interpretation.

## Verified in QEMU 11.0.3

```
QAL-OS M4: ring3 agent, capability revocation, validated syscalls
agent: qal-0 = 3
[syscall: bad user ptr]
[cap1: send ok]
[cap1: revoked]
[cap: denied]
```

That is the full cycle: valid agent code runs, an invalid pointer is blocked, a
capability works, is revoked, and the next send is denied.

## Honest limits

- QAL-0 is still a toy bytecode, not QAL.
- Timer/scheduler is masked off; one user agent.
- Capabilities are a static two-word table, not kernel objects with endpoints,
  copy-in/copy-out, or transitive revocation.
- No object store, page-fault handling, UEFI, SMP.

Next M5: capability endpoints (objects), copy-in/copy-out, ring-3 scheduling,
and a real QAL subset (Int/Bool/fn/test).
