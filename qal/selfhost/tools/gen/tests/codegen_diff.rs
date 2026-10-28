//! Differential testing of the self-hosted QAL WASM codegen against the Rust reference.
//! (Phase 5c, spec §10.2, §15).

use qal_compiler::{Backend, Both, CompileOptions, Entry, Qal, Rust};
use std::sync::Once;

static INIT: Once = Once::new();

fn init_qalc() {
    INIT.call_once(|| {
        let src = std::fs::read_to_string(qal_gen::qal_root().join("selfhost/qalc.qal"))
            .expect("cannot read selfhost/qalc.qal");
        let prog = qal_compiler::parse_program(&src)
            .expect("qalc.qal does not parse with Rust parser");
        let tp = qal_compiler::check(&prog)
            .expect("qalc.qal does not check with Rust checker");
        let module = qal_compiler::compile_with(&tp, &Entry::All, &CompileOptions::compiler())
            .expect("failed to compile qalc.qal to wasm");
        qal_run::selfhost::init_qal_backend(module, 4)
            .expect("failed to initialize Qal backend with wasm module");
    });
}

#[test]
fn differential_wasm_corpus() {
    init_qalc();
    let mut compiled_count = 0;
    for s in qal_gen::corpus() {
        let Ok(p) = Rust.parse(&s.text) else { continue };
        let Ok(tp) = Rust.check(&p, &|_| true) else { continue };
        compiled_count += 1;

        let r_mod = Rust
            .compile(&tp, &Entry::All, &CompileOptions::default())
            .unwrap_or_else(|d| panic!("{}: Rust compile failed: {d:?}", s.name));
        let q_mod = Qal
            .compile(&tp, &Entry::All, &CompileOptions::default())
            .unwrap_or_else(|d| panic!("{}: Qal compile failed: {d:?}", s.name));

        assert_eq!(r_mod.imports, q_mod.imports, "{}: imports mismatch", s.name);
        assert_eq!(r_mod.table_len, q_mod.table_len, "{}: table_len mismatch", s.name);
        assert_eq!(
            r_mod.memory_max_pages, q_mod.memory_max_pages,
            "{}: memory_max_pages mismatch",
            s.name
        );
        assert_eq!(r_mod.types, q_mod.types, "{}: types mismatch", s.name);
        assert_eq!(r_mod.funcs, q_mod.funcs, "{}: funcs metadata mismatch", s.name);
        if r_mod.wasm != q_mod.wasm {
            let first_diff = r_mod
                .wasm
                .iter()
                .zip(q_mod.wasm.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(r_mod.wasm.len().min(q_mod.wasm.len()));
            panic!(
                "{}: wasm bytes mismatch at byte {} (rust len {}, qal len {})",
                s.name,
                first_diff,
                r_mod.wasm.len(),
                q_mod.wasm.len()
            );
        }
    }
    assert!(
        compiled_count > 400,
        "expected > 400 compiled corpus programs, got {compiled_count}"
    );
}

#[test]
fn differential_wasm_generated() {
    init_qalc();
    let count = std::env::var("QAL_GEN_COUNT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let mut rng = qal_gen::Rng(qal_gen::seed());
    for i in 0..count {
        let text = qal_gen::well_typed(&mut rng);
        let p = Rust
            .parse(&text)
            .unwrap_or_else(|d| panic!("gen #{i}: parse failed: {d:?}\n{text}"));
        let tp = Rust
            .check(&p, &|_| true)
            .unwrap_or_else(|ds| panic!("gen #{i}: check failed: {ds:?}\n{text}"));
        Both.compile(&tp, &Entry::All, &CompileOptions::default())
            .unwrap_or_else(|d| panic!("gen #{i}: Both.compile failed: {d:?}\n{text}"));
    }
}

#[test]
fn mixed_bootstrap_fixpoint() {
    init_qalc();
    let src = std::fs::read_to_string(qal_gen::qal_root().join("selfhost/qalc.qal"))
        .expect("cannot read selfhost/qalc.qal");
    let prog = Rust.parse(&src).expect("qalc.qal must parse");
    let tp = Rust.check(&prog, &|_| true).expect("qalc.qal must check");

    // Stage 0: Rust codegen
    let stage0 = Rust
        .compile(&tp, &Entry::All, &CompileOptions::compiler())
        .expect("Stage 0 Rust.compile failed");

    // Stage 1: QAL codegen (running on Stage 0)
    let stage1 = Qal
        .compile(&tp, &Entry::All, &CompileOptions::compiler())
        .expect("Stage 1 Qal.compile failed");

    assert_eq!(stage0.funcs, stage1.funcs, "Stage 0 vs Stage 1 funcs mismatch");
    assert_eq!(stage0.wasm, stage1.wasm, "Stage 0 vs Stage 1 wasm bytes mismatch");

    // Install Stage 1 as the active QAL backend and compile qalc.qal again -> Stage 2
    qal_run::selfhost::init_qal_backend(stage1.clone(), 4)
        .expect("failed to install Stage 1 wasm module");
    let stage2 = Qal
        .compile(&tp, &Entry::All, &CompileOptions::compiler())
        .expect("Stage 2 Qal.compile failed");

    assert_eq!(stage1.funcs, stage2.funcs, "Stage 1 vs Stage 2 funcs mismatch");
    assert_eq!(stage1.imports, stage2.imports, "Stage 1 vs Stage 2 imports mismatch");
    assert_eq!(stage1.wasm, stage2.wasm, "Stage 1 vs Stage 2 mixed fixpoint wasm mismatch");
}
