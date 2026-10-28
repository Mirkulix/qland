# QAL-OS M12 – persistent ATA SHA-256 store

M12 writes the SHA-256 store to a real IDE/ATA sector and reads it back.

## What runs

- ATA PIO: sector 1, master LBA, `0x30` write / `0x20` read, `0xE7` flush;
- syscall 10 `store_save`: packs 128 hash bytes + 16 length bytes + 64 data
  bytes into `sector_buf`, writes sector 1;
- syscall 11 `store_load`: reads sector 1 and restores the in-memory store;
- task B does put → save → load → get → QAL-1; task A does put → save → get →
  QAL-1; both then loop `putc` under the preemptive timer.

## Verified in QEMU 11.0.3 (macOS, x86_64)

```sh
truncate -s 1M disk.img
qemu-system-x86_64 -kernel kernel.elf \
  -drive file=disk.img,format=raw,if=ide,index=0 \
  -serial file:serial.log -display none
```

```
QAL-OS M12: persistent ATA SHA-256 store
[store: put]
[disk: save]
[disk: load]
[store: get]
[qal-1 = 5]
BBBB...AAAA...
```

- 2× `[disk: save]`, 1× `[disk: load]`,
- `[qal-1 = 9]` and `[qal-1 = 5]`,
- 560,236 A / 554,973 B characters,
- disk sector 512 begins with the SHA-256 hash `2e5ff0f6…` of task B's program.

## Honest limits

- one fixed sector, no partition table, no filesystem, no virtio;
- no slot GC/collision handling, no durability guarantees beyond QEMU IDE;
- QAL-1 is still a toy stack language;
- no SMP/UEFI/page-fault handling.

Next M13: virtio-blk, multiple sectors, slot GC, and a larger QAL subset.
