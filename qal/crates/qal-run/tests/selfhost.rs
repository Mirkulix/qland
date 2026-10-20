//! The host side of the compiler protocol (self-hosting spec §4.3, §6.3): `selfhost::drive` with
//! fork/join, its limits, `--jobs` independence, and a real compiler module written in QAL
//! (`qal/selfhost/proto/forkjoin.qal`) run with `Limits::compiler()` (1 GiB, i32 audit).

use qal_compiler::proto::{self, Request, Response};
use qal_compiler::{Backend, CompileOptions, Entry};
use qal_run::selfhost::{self, Compiler, DriveError, Limits};
use std::sync::atomic::{AtomicUsize, Ordering};

fn parse_req(text: &str) -> Vec<u8> {
    Request::Parse { text: text.into() }.encode()
}

fn module(opts: CompileOptions) -> qal_compiler::CompiledModule {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../selfhost/proto/forkjoin.qal");
    let src = std::fs::read_to_string(path).unwrap();
    let tp = qal_compiler::check(&qal_compiler::parse_program(&src).unwrap()).unwrap_or_else(|d| panic!("{d:#?}"));
    qal_compiler::compile_with(&tp, &Entry::Fns(vec!["qc_main".into()]), &opts).unwrap()
}

fn done(r: Response) -> Vec<u8> {
    match r {
        Response::Done(p) => p,
        other => panic!("{other:?}"),
    }
}

/// A compiler in Rust that forks a PARSE of n characters into n items, each forked again once
/// (two rounds), and joins by concatenation.
fn fake(req: &[u8]) -> Result<Vec<u8>, String> {
    Ok(match Request::decode(req)? {
        Request::Parse { text } if text.chars().count() <= 1 => Response::Done(text.to_uppercase().into_bytes()),
        Request::Parse { text } => Response::Fork { continuation: b"cat".to_vec(), items: text.chars().map(|c| parse_req(&c.to_string())).collect() },
        Request::Join { continuation, results } => {
            let mut out = Vec::new();
            for r in &results {
                match Response::decode(r)? {
                    Response::Done(p) => out.extend(p),
                    other => return Err(format!("{other:?}")),
                }
            }
            if continuation == b"cat" {
                // A second round: reverse what the first one produced.
                Response::Fork { continuation: b"rev".to_vec(), items: String::from_utf8_lossy(&out).chars().rev().map(|c| parse_req(&c.to_string())).collect() }
            } else {
                Response::Done(out)
            }
        }
        _ => Response::Internal("unsupported".into()),
    }
    .encode())
}

#[test]
fn fork_join_results_are_in_item_order_whatever_the_jobs() {
    let limits = Limits::compiler();
    let text: String = (0..200).map(|i| char::from(b'a' + (i % 26) as u8)).collect();
    let mut outs = Vec::new();
    for jobs in [1, 2, 8] {
        let (r, stats) = selfhost::drive(parse_req(&text), jobs, &limits, &fake).unwrap();
        assert_eq!((stats.rounds, stats.work_items), (2, 400), "{stats:?}");
        outs.push(done(r));
    }
    let expected: String = text.to_uppercase().chars().rev().collect();
    assert!(outs.iter().all(|o| o == &outs[0]), "--jobs changes the output");
    assert_eq!(outs[0], expected.into_bytes());
}

#[test]
fn protocol_limits_and_failures_are_diagnostics() {
    let limits = Limits { max_rounds: 1, ..Limits::compiler() };
    let e = selfhost::drive(parse_req("ab"), 4, &limits, &fake).unwrap_err();
    assert!(matches!(&e, DriveError::Limit(m) if m.contains("rounds")), "{e:?}");
    assert_eq!(e.diagnostic("stage0", "PARSE").code, "E0954");
    let limits = Limits { max_items: 3, ..Limits::compiler() };
    assert!(matches!(selfhost::drive(parse_req("abcd"), 4, &limits, &fake).unwrap_err(), DriveError::Limit(m) if m.contains("work items")));
    let limits = Limits { max_round_bytes: 10, ..Limits::compiler() };
    assert!(matches!(selfhost::drive(parse_req("abcd"), 4, &limits, &fake).unwrap_err(), DriveError::Limit(m) if m.contains("bytes")));
    let limits = Limits::compiler();
    // INTERNAL, a trap and a malformed answer, also from inside a round.
    let e = selfhost::drive(Request::Signatures { state: Default::default(), names: vec![] }.encode(), 1, &limits, &fake).unwrap_err();
    assert_eq!(e.diagnostic("stage0", "SIGNATURES").code, "E0951");
    let trap = |_: &[u8]| -> Result<Vec<u8>, String> { Err("out_of_memory".into()) };
    assert!(matches!(selfhost::drive(parse_req("a"), 1, &limits, &trap).unwrap_err(), DriveError::Failed { error, .. } if error == "out_of_memory"));
    let garbage = |_: &[u8]| -> Result<Vec<u8>, String> { Ok(b"QALR\x01\x09".to_vec()) };
    let e = selfhost::drive(parse_req("a"), 1, &limits, &garbage).unwrap_err();
    assert_eq!(e.diagnostic("stage0", "PARSE").code, "E0952");
    let calls = AtomicUsize::new(0);
    let bad_item = |req: &[u8]| -> Result<Vec<u8>, String> {
        if calls.fetch_add(1, Ordering::SeqCst) == 0 {
            fake(req)
        } else {
            Ok(b"nope".to_vec())
        }
    };
    assert!(matches!(selfhost::drive(parse_req("ab"), 2, &limits, &bad_item).unwrap_err(), DriveError::Malformed(_)));
    // Diagnostics are a final answer, like DONE.
    let diags = |_: &[u8]| -> Result<Vec<u8>, String> { Ok(Response::Diagnostics(vec![qal_core::diag::Diagnostic::parse(1, 1, "x")]).encode()) };
    assert!(matches!(selfhost::drive(parse_req("a"), 1, &limits, &diags).unwrap().0, Response::Diagnostics(d) if d.len() == 1));
    assert_eq!(selfhost::default_jobs().clamp(1, 8), selfhost::default_jobs());
}

#[test]
fn a_compiler_module_in_qal_forks_and_joins() {
    let limits = Limits::compiler();
    let c = Compiler::new(module(CompileOptions::compiler()), &limits, 8).unwrap();
    let text = "self-hosting, 5a";
    let expected: Vec<u8> = text.bytes().flat_map(|b| qal_core::canon::Hash::of(&[b]).0).collect();
    let mut outs = Vec::new();
    for jobs in [1, 8] {
        let (r, stats) = selfhost::drive(parse_req(text), jobs, &limits, &c).unwrap();
        assert_eq!((stats.rounds, stats.work_items), (1, text.len()));
        outs.push(done(r));
    }
    assert_eq!(outs[0], expected);
    assert_eq!(outs[0], outs[1], "--jobs 1 == --jobs 8");
    // One byte: no fork. Unknown operations, traps: diagnostics, and the host keeps running.
    assert_eq!(done(selfhost::drive(parse_req("x"), 1, &limits, &c).unwrap().0), qal_core::canon::Hash::of(b"x").0.to_vec());
    let e = selfhost::drive(b"QALQ\x01\x65".to_vec(), 1, &limits, &c).unwrap_err();
    assert!(matches!(&e, DriveError::Failed { error, .. } if error.starts_with("index")), "{e:?}");
    assert_eq!(e.diagnostic("stage1", "op 101").code, "E0951");
    assert!(matches!(selfhost::drive(b"QALQ\x01\x03".to_vec(), 1, &limits, &c).unwrap_err(), DriveError::Failed { error, .. } if error.contains("unknown operation")));
    // The entry and the memory maximum must match what the host runs it with (review M4).
    let e = Compiler::new(module(CompileOptions::default()), &limits, 1).err().unwrap();
    assert!(e.contains("memory_max_pages"), "{e}");
    let small = Limits { memory_bytes: 256 << 20, ..limits };
    assert!(Compiler::new(module(CompileOptions::default()), &small, 1).is_ok());
    let tp = qal_compiler::check(&qal_compiler::parse_program("fn qc_main(b: Str) -> Bytes = to_bytes(b)").unwrap()).unwrap();
    let wrong = qal_compiler::compile_with(&tp, &Entry::All, &CompileOptions::compiler()).unwrap();
    assert!(Compiler::new(wrong, &limits, 1).err().unwrap().contains("fn(Bytes) -> Bytes"));
    // More jobs than instance slots: drive uses only the slots, with the same output (review M1).
    let one = Compiler::new(module(CompileOptions::compiler()), &limits, 1).unwrap();
    assert_eq!(done(selfhost::drive(parse_req(text), 8, &limits, &one).unwrap().0), expected);
}

fn mem_req(mib: usize) -> Vec<u8> {
    let mut b = b"QALQ\x01\x64".to_vec();
    let t = mib.to_string();
    b.push(t.len() as u8);
    b.extend(t.bytes());
    b
}

#[test]
fn compiler_runs_get_1_gib_and_user_programs_256_mib() {
    // i32 audit (spec §6.3): with 1 GiB, heap addresses pass 2^29; string functions with -1
    // sentinels (index_of) and Bytes/List lengths still work above 512 MiB.
    let limits = Limits::compiler();
    let c = Compiler::new(module(CompileOptions::compiler()), &limits, 1).unwrap();
    let (r, fuel, mem) = c.call_here(&mem_req(700), limits.item_timeout);
    let payload = done(Response::decode(&r.unwrap()).unwrap());
    assert_eq!(String::from_utf8(payload).unwrap(), (700usize << 20).to_string());
    assert!(mem > 700 << 20 && mem <= 1 << 30 && fuel > 0, "{mem} {fuel}");
    // Beyond 1 GiB: out_of_memory, as a diagnostic.
    let e = selfhost::drive(mem_req(1100), 1, &limits, &c).unwrap_err();
    assert!(matches!(&e, DriveError::Failed { error, .. } if error.starts_with("out_of_memory")), "{e:?}");
    // A module compiled for user programs cannot grow beyond 256 MiB.
    let small_limits = Limits { memory_bytes: 256 << 20, ..limits };
    let small = Compiler::new(module(CompileOptions::default()), &small_limits, 1).unwrap();
    assert!(matches!(selfhost::drive(mem_req(300), 1, &small_limits, &small).unwrap_err(), DriveError::Failed { error, .. } if error.starts_with("out_of_memory")));
    assert!(proto::Response::decode(&small.call_here(&mem_req(100), limits.item_timeout).0.unwrap()).is_ok());
}

#[test]
fn the_total_time_limit_holds_inside_a_round() {
    // Review I3: 64 items of 100 ms on 2 threads would take 3.2 s; the limit is 0.5 s.
    let slow = |req: &[u8]| -> Result<Vec<u8>, String> {
        match Request::decode(req)? {
            Request::Parse { text } if text.len() > 1 => Ok(Response::Fork { continuation: vec![], items: (0..64).map(|_| parse_req("x")).collect() }.encode()),
            _ => {
                std::thread::sleep(std::time::Duration::from_millis(100));
                Ok(Response::Done(vec![]).encode())
            }
        }
    };
    let limits = Limits { total_timeout: std::time::Duration::from_millis(500), ..Limits::compiler() };
    let t = std::time::Instant::now();
    let e = selfhost::drive(parse_req("many"), 2, &limits, &slow).unwrap_err();
    assert!(matches!(&e, DriveError::Limit(m) if m.contains("longer than")), "{e:?}");
    assert!(t.elapsed() < std::time::Duration::from_millis(1500), "{:?}", t.elapsed());
    assert_eq!(e.diagnostic("stage0", "PARSE").code, "E0954");
}

#[test]
fn a_failing_item_stops_the_later_ones() {
    // Review I3: item 3 fails; items after it are not started (one thread: exactly 4 calls + main).
    let calls = AtomicUsize::new(0);
    let fails = |req: &[u8]| -> Result<Vec<u8>, String> {
        calls.fetch_add(1, Ordering::SeqCst);
        match Request::decode(req)? {
            Request::Parse { text } if text.len() > 1 => Ok(Response::Fork { continuation: vec![], items: (0..100).map(|i| parse_req(&i.to_string()[..1])).collect() }.encode()),
            Request::Parse { text } if text == "3" => Err("trap".into()),
            _ => Ok(Response::Done(vec![]).encode()),
        }
    };
    let limits = Limits::compiler();
    // items "0".."9" then "1".."9" again (first digit): the first "3" is item 3
    let e = selfhost::drive(parse_req("go"), 1, &limits, &fails).unwrap_err();
    assert!(matches!(&e, DriveError::Failed { item, .. } if item == "round 1 item 3"), "{e:?}");
    assert_eq!(calls.load(Ordering::SeqCst), 5);
    // With several threads the reported failure is still the first in item order.
    calls.store(0, Ordering::SeqCst);
    let e = selfhost::drive(parse_req("go"), 8, &limits, &fails).unwrap_err();
    assert!(matches!(&e, DriveError::Failed { item, .. } if item == "round 1 item 3"), "{e:?}");
    assert!(calls.load(Ordering::SeqCst) < 101);
}

#[test]
fn seed_backend_initializes_and_parses_program() {
    selfhost::init_seed_backend(4).expect("init seed backend");
    assert!(qal_compiler::has_qal_driver());
    let p = qal_compiler::Qal.parse("fn add(x: Int, y: Int) -> Int = x + y").expect("parse via seed");
    assert_eq!(p.defs.len(), 1);
}
