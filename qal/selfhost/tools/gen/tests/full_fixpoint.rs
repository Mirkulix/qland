//! Full bootstrap fixpoint: the self-hosted compiler parses, checks and compiles its own source.
use qal_compiler::{Backend, CompileOptions, Entry, Qal, Rust};

// Open gap (2026-10-20): `Qal.parse` of qalc.qal succeeds and equals the Rust parse, but
// `Qal.check` runs CHECK_FILE as one execution and hits out_of_memory (E0504, 1 GiB). The
// checker must split the work per definition group via the existing FORK/JOIN protocol
// (phase 5 spec, measurement gate of 5a). Run with `--ignored` once that lands.
#[test]
#[ignore = "QAL checker runs out of memory on qalc.qal; needs FORK/JOIN split"]
fn full_selfhosted_fixpoint() {
    let src = std::fs::read_to_string(qal_gen::qal_root().join("selfhost/qalc.qal")).expect("read qalc.qal");
    // Stage 0: the Rust pipeline builds the compiler.
    let p0 = Rust.parse(&src).expect("rust parse");
    let t0 = Rust.check(&p0, &|_| true).expect("rust check");
    let stage1 = Rust.compile(&t0, &Entry::All, &CompileOptions::compiler()).expect("rust compile");
    qal_run::selfhost::init_qal_backend(stage1.clone(), 4).expect("install stage 1");

    // Stage 1 runs the whole pipeline on its own source -> Stage 2.
    let t = std::time::Instant::now();
    let p1 = Qal.parse(&src).unwrap_or_else(|d| panic!("qal parse: {d:?}"));
    eprintln!("qal parse {:?}", t.elapsed());
    assert_eq!(p1, p0, "QAL parse differs from Rust parse");
    let t1 = Qal.check(&p1, &|_| true).unwrap_or_else(|d| panic!("qal check: {:?}", d.first()));
    eprintln!("qal check {:?}", t.elapsed());
    let stage2 = Qal.compile(&t1, &Entry::All, &CompileOptions::compiler()).unwrap_or_else(|d| panic!("qal compile: {d:?}"));
    eprintln!("qal compile {:?}", t.elapsed());
    assert_eq!(stage1.wasm, stage2.wasm, "stage 1 vs stage 2");

    // Stage 2 builds Stage 3.
    qal_run::selfhost::init_qal_backend(stage2.clone(), 4).expect("install stage 2");
    let p2 = Qal.parse(&src).unwrap_or_else(|d| panic!("stage2 parse: {d:?}"));
    let t2 = Qal.check(&p2, &|_| true).unwrap_or_else(|d| panic!("stage2 check: {:?}", d.first()));
    let stage3 = Qal.compile(&t2, &Entry::All, &CompileOptions::compiler()).unwrap_or_else(|d| panic!("stage2 compile: {d:?}"));
    assert_eq!(stage2.wasm, stage3.wasm, "stage 2 vs stage 3");
}
