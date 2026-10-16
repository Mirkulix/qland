//! The measurement gate of Phase 5a (self-hosting spec §6.2). The prototype is
//! `qal/selfhost/proto/gate.qal`; its tests run in every test run, the measurement itself with
//! `cargo test --release -p qal-gen --test gate -- --ignored --nocapture` (numbers in the 5a report).

use qal_compiler::{CompileOptions, Entry};
use qal_run::selfhost::Limits;
use qal_run::{commit_if_ok, ExecIn, Permissions, Runner, Value, GUEST_THREAD_STACK};
use std::time::Instant;

fn module() -> qal_compiler::CompiledModule {
    let src = std::fs::read_to_string(qal_gen::qal_root().join("selfhost/proto/gate.qal")).unwrap();
    let tp = qal_compiler::check(&qal_compiler::parse_program(&src).unwrap()).unwrap_or_else(|d| panic!("{d:#?}"));
    qal_compiler::compile_with(&tp, &Entry::All, &CompileOptions::compiler()).unwrap()
}

struct Run {
    value: Value,
    ms: f64,
    fuel: u64,
    mib: f64,
}

/// One call in a fresh instance with `Limits::compiler()`.
fn run(export: &str, args: Vec<Value>) -> Run {
    let limits = Limits::compiler();
    let runner = Runner::prepared(module(), Permissions::default(), limits.memory_bytes, 1).unwrap().with_fuel(limits.fuel).with_timeout(limits.item_timeout);
    let export = export.to_string();
    std::thread::Builder::new()
        .stack_size(GUEST_THREAD_STACK)
        .spawn(move || {
            let t = Instant::now();
            let out = runner.execute(&export, &args, ExecIn::default(), &commit_if_ok);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let value = out.outcome.result.unwrap_or_else(|e| panic!("{export}: {e:?}"));
            Run { value, ms, fuel: out.fuel, mib: out.mem_bytes as f64 / (1 << 20) as f64 }
        })
        .unwrap()
        .join()
        .unwrap()
}

#[test]
fn the_gate_prototype_is_correct() {
    let r = Runner::new(module()).unwrap();
    let results = r.run_tests();
    assert!(results.iter().all(|t| t.passed), "{results:#?}");
}

fn corpus_text() -> String {
    qal_gen::corpus().iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join("\n")
}

#[test]
#[ignore = "measurement (release build): cargo test --release -p qal-gen --test gate -- --ignored --nocapture"]
fn measurement_gate() {
    let text = corpus_text();
    // Baseline: the Rust pipeline on the same corpus.
    let t = Instant::now();
    let mut parsed = 0;
    for s in qal_gen::corpus() {
        if let Ok(p) = qal_compiler::parse_program(&s.text) {
            parsed += 1;
            let _ = qal_compiler::check(&p);
        }
    }
    let rust_ms = t.elapsed().as_secs_f64() * 1000.0;
    eprintln!("GATE rust parse+check of the corpus ({} bytes, {parsed} parsed): {rust_ms:.1} ms", text.len());
    // (1) Lexing the corpus: 1x, and 4x concatenated (to see the per-byte slope).
    for k in [1, 4] {
        let src = text.repeat(k);
        let r = run("gate_lex", vec![Value::Bytes(src.clone().into_bytes(), 0)]);
        let Value::Int(tokens) = r.value else { panic!() };
        eprintln!(
            "GATE lex {} KB: {tokens} tokens, {:.1} ms, fuel {} ({:.0}/byte), memory {:.1} MiB ({:.0} B/token incl. input)",
            src.len() / 1000,
            r.ms,
            r.fuel,
            r.fuel as f64 / src.len() as f64,
            r.mib,
            r.mib * (1 << 20) as f64 / tokens as f64
        );
    }
    // The same lexing loop in Rust (byte classes, tokens kept in a Vec): the QAL/Rust factor of
    // one workload that both sides really do.
    let bytes = text.as_bytes();
    let t = Instant::now();
    let mut rust_tokens = 0usize;
    for _ in 0..20 {
        rust_tokens = rust_lex(bytes).len();
    }
    let rust_lex_ms = t.elapsed().as_secs_f64() * 1000.0 / 20.0;
    eprintln!("GATE rust lex of the corpus: {rust_tokens} tokens, {rust_lex_ms:.2} ms");
    // (2) 10^6 map_put on Str keys: 10^4 distinct keys (a symbol table) and 10^5.
    for keys in [10_000i64, 100_000] {
        let r = run("gate_puts", vec![Value::Int(1_000_000), Value::Int(keys)]);
        eprintln!("GATE 1e6 map_put, {keys} keys -> size {:?}: {:.1} ms, fuel {} ({:.0}/put), memory {:.1} MiB ({:.0} B/put)", r.value, r.ms, r.fuel, r.fuel as f64 / 1e6, r.mib, r.mib * (1 << 20) as f64 / 1e6);
    }
    // Per-definition work item: the global signature table of 2 000 definitions.
    let r = run("gate_puts", vec![Value::Int(2_000), Value::Int(2_000)]);
    eprintln!("GATE signature table, 2000 map_put -> {:?}: {:.2} ms, fuel {}, memory {:.2} MiB", r.value, r.ms, r.fuel, r.mib);
    // (3) A 1 MiB module from 10^5 pieces (Accum + bytes_join).
    let r = run("gate_module", vec![Value::Int(100_000)]);
    eprintln!("GATE module of 1e5 pieces -> {:?} bytes: {:.1} ms, fuel {}, memory {:.1} MiB", r.value, r.ms, r.fuel, r.mib);
    let r = run("gate_module", vec![Value::Int(400_000)]);
    eprintln!("GATE module of 4e5 pieces -> {:?} bytes: {:.1} ms, fuel {}, memory {:.1} MiB", r.value, r.ms, r.fuel, r.mib);
}

/// The gate lexer's loop in Rust: (kind, start, end) per token.
fn rust_lex(b: &[u8]) -> Vec<(u8, usize, usize)> {
    let alpha = |c: u8| c.is_ascii_alphabetic() || c == b'_';
    let digit = |c: u8| c.is_ascii_digit();
    let mut out = Vec::new();
    let (mut i, n) = (0, b.len());
    while i < n {
        let c = b[i];
        if matches!(c, b' ' | b'\n' | b'\t' | b'\r') {
            i += 1;
        } else if c == b'#' {
            while i < n && b[i] != b'\n' {
                i += 1;
            }
        } else if alpha(c) || digit(c) {
            let j0 = i;
            i += 1;
            while i < n && (alpha(b[i]) || digit(b[i]) || b[i] == b'.') {
                i += 1;
            }
            out.push((if alpha(c) { 1 } else { 2 }, j0, i));
        } else if c == b'"' {
            let j0 = i;
            i += 1;
            while i < n && b[i] != b'"' {
                i += if b[i] == b'\\' { 2 } else { 1 };
            }
            i = (i + 1).min(n);
            out.push((3, j0, i));
        } else {
            let two = i + 1 < n && matches!((c, b[i + 1]), (b'=' | b'-', b'>') | (b'=' | b'!' | b'<' | b'>', b'=') | (b'&', b'&') | (b'|', b'|') | (b'+', b'+'));
            let k = if two { 2 } else { 1 };
            out.push((4, i, i + k));
            i += k;
        }
    }
    out
}
