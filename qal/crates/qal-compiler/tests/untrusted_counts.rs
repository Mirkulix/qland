//! Final review I2: decoding a malformed compiler answer never reserves memory from the counts it
//! claims. A counting allocator records the largest single allocation while 64 MiB inputs with
//! absurd counts (and then garbage, so decoding fails at the first element) are decoded; it must
//! stay near the input size, not 50-250 times it. (Valid but tiny elements still decode into
//! their in-memory size, up to about 24 bytes per input byte; responses are bounded by
//! `Limits::max_round_bytes`.)

use qal_compiler::proto::{self, Response};
use qal_core::tform;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static LARGEST: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        LARGEST.fetch_max(l.size(), Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }

    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        LARGEST.fetch_max(new, Ordering::Relaxed);
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static A: Counting = Counting;

const SIZE: usize = 64 << 20;

/// `prefix`, then the LEB128 count `n`, then filler up to SIZE bytes.
fn crafted(prefix: &[u8], n: u64, fill: u8) -> Vec<u8> {
    let mut b = prefix.to_vec();
    tform::put_uint(&mut b, n);
    b.resize(SIZE, fill);
    b
}

fn largest_during(f: impl FnOnce()) -> usize {
    LARGEST.store(0, Ordering::Relaxed);
    f();
    LARGEST.load(Ordering::Relaxed)
}

#[test]
fn absurd_counts_do_not_reserve_memory() {
    let n = (SIZE - 16) as u64;
    let inputs: Vec<(&str, Vec<u8>)> = vec![
        ("texts", crafted(b"", n, 0xff)),
        ("module funcs", {
            // wasm "" imports 0 table 0 types 0, then a huge funcs count
            crafted(&[0, 0, 0, 0], n, 0xff)
        }),
        ("module types", crafted(&[0, 0, 0], n, 0xff)),
        ("program", crafted(b"", n, 0xff)),
        ("prepared", crafted(&[0], n, 0xff)),
        ("QALT types", crafted(b"QALT\x01", n, 0xff)),
        ("QALT fns", crafted(b"QALT\x01\x00", n, 0xff)),
        ("QALR diagnostics", crafted(b"QALR\x01\x01", n, 0xff)),
        ("QALR fork items", crafted(b"QALR\x01\x02\x00", n, 0xff)),
    ];
    for (what, b) in &inputs {
        let largest = largest_during(|| {
            let r = match *what {
                "texts" => proto::decode_texts(b).err(),
                "module funcs" | "module types" => proto::decode_module(b).err(),
                "program" => proto::decode_program(b).err(),
                "prepared" => proto::decode_prepared(b).err(),
                "QALT types" | "QALT fns" => tform::decode(b).err(),
                _ => Response::decode(b).err(),
            };
            assert!(r.is_some(), "{what}: a malformed input decoded");
        });
        assert!(largest < 2 * SIZE, "{what}: a single allocation of {} MiB", largest >> 20);
    }
}
