# QAL on ARM64 (Ubuntu 26.04 VM)

Date: 2026-10-28

## Environment

| | |
|---|---|
| Host | macOS / Parallels |
| Guest | Ubuntu 26.04, kernel 7.0.0-29-generic, **aarch64** |
| CPU/RAM | 2 vCPU, 3.3 GiB RAM, 3.8 GiB swap |
| Virtualization | `systemd-detect-virt=parallels`; **no `/dev/kvm`**, no EL2 flags |
| QEMU | 10.2.1 (`qemu-system-x86_64`, `qemu-system-arm`) |
| Rust | 1.99.1 |

KVM is **not available** in this VM; nested virtualization is not exposed by the
host. QAL does not need it. QAL-OS runs under QEMU **TCG**.

## QAL build and tests

```sh
git clone https://github.com/Mirkulix/qland.git && cd qland
git fetch origin Mac && git checkout -B Mac FETCH_HEAD
cargo build --release --manifest-path qal/Cargo.toml
cargo test  --release --manifest-path qal/Cargo.toml
```

- build: **18m 17s**, exit 0, binary 54,782,504 bytes;
- `cargo test --release`: **656 passed, 0 failed**, 46 `test result: ok`,
  exit 0;
- `qal check/test qal/examples/math.qal`: 4/4 passed; the `fact` property test
  reports 6 stack-overflow traps per 100 runs (existing E0506 behavior).

## QAL-OS under TCG (no KVM)

M11 and M12 kernels were copied to the VM and booted with
`qemu-system-x86_64 -kernel ... -serial ...`:

- M11: `[store: put]` ×2, `[store: get]` ×2, `[qal-1 = 9]`, `[qal-1 = 5]`;
- M12: ATA persistence, `[disk: save]` ×2, `[disk: load]` ×1, QAL 9/5, and the
  SHA-256 hash is present in sector 512 of the disk image.

TCG is slower than KVM but sufficient for correctness tests.

## Not available here

- KVM/Firecracker/gVisor acceleration (no `/dev/kvm`);
- x86_64 hardware performance;
- SMP/UEFI tests (still M13/M14 work).
