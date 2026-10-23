//! Differential testing of the self-hosted QAL checker against the Rust reference.
//! (Phase 5d, spec §10.2, §15).

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
fn differential_check_simple() {
    init_qalc();
    let src = "fn add(x: Int, y: Int) -> Int = x + y\ntest t = add(1, 2) == 3";
    let p = Rust.parse(src).expect("parse failed");
    let r_tp = Rust.check(&p, &|_| true).expect("Rust check failed");
    let q_tp = Qal.check(&p, &|_| true).expect("Qal check failed");
    let r_bytes = qal_core::tform::encode(&r_tp);
    let q_bytes = qal_core::tform::encode(&q_tp);
    assert_eq!(r_bytes, q_bytes, "QALT bytes mismatch on add");
}

#[test]
fn differential_check_corpus() {
    init_qalc();
    let mut ok_count = 0;
    let mut err_count = 0;
    let mut seq_let_hints = 0;
    for s in qal_gen::corpus() {
        let Ok(p) = Rust.parse(&s.text) else { continue };
        let r_res = Rust.check(&p, &|_| true);
        let q_res = Qal.check(&p, &|_| true);

        match (r_res, q_res) {
            (Ok(r_tp), Ok(q_tp)) => {
                ok_count += 1;
                let r_bytes = qal_core::tform::encode(&r_tp);
                let q_bytes = qal_core::tform::encode(&q_tp);
                if r_bytes != q_bytes {
                    let first = r_bytes.iter().zip(q_bytes.iter()).position(|(a, b)| a != b).unwrap_or(r_bytes.len().min(q_bytes.len()));
                    let r_context = &r_bytes[first.saturating_sub(10)..first.saturating_add(20).min(r_bytes.len())];
                    let q_context = &q_bytes[first.saturating_sub(10)..first.saturating_add(20).min(q_bytes.len())];
                    panic!("{}: QALT mismatch at byte {} (rust len {}, qal len {}):\nrust: {:?}\nqal:  {:?}", s.name, first, r_bytes.len(), q_bytes.len(), r_context, q_context);
                }
            }
            (Err(r_ds), Err(q_ds)) => {
                err_count += 1;
                let r_codes: Vec<(&str, &str)> = r_ds.iter().map(|d| (d.code, d.kind)).collect();
                let q_codes: Vec<(&str, &str)> = q_ds.iter().map(|d| (d.code, d.kind)).collect();
                assert_eq!(
                    r_codes, q_codes,
                    "{}: diagnostic codes/kinds divergence: Rust={r_codes:?}, Qal={q_codes:?}",
                    s.name
                );
                // The production binary runs the QAL checker: the text an agent reads must be the
                // tested one, so messages and hints match too. Only for E0201 so far: E0203 still
                // diverges (the Rust checker hints `unpack the Option` where the QAL one gives none,
                // and the QAL one adds `unpack the Result` to a discarded Result); widen this to
                // every code once those agree.
                let text = |ds: &[qal_core::diag::Diagnostic]| -> Vec<(String, Vec<String>)> {
                    ds.iter().filter(|d| d.code == "E0201").map(|d| (d.message.clone(), d.hints.clone())).collect()
                };
                let (r_text, q_text) = (text(&r_ds), text(&q_ds));
                assert_eq!(r_text, q_text, "{}: E0201 message/hints divergence", s.name);
                seq_let_hints += r_text.iter().filter(|(_, hs)| hs.iter().any(|h| h.contains("ends at its `;`"))).count();
            }
            (Ok(_), Err(q_ds)) => {
                panic!("{}: Rust accepted but Qal rejected with {q_ds:?}", s.name);
            }
            (Err(r_ds), Ok(_)) => {
                panic!("{}: Rust rejected with {r_ds:?} but Qal accepted", s.name);
            }
        }
    }
    assert!(ok_count > 400, "expected > 400 checked corpus programs, got {ok_count}");
    assert!(err_count > 100, "expected > 100 rejected corpus programs, got {err_count}");
    assert!(seq_let_hints > 0, "no corpus program exercises the E0201 hint of a sequence `let`");
}

#[test]
fn differential_check_generated() {
    init_qalc();
    let count = std::env::var("QAL_GEN_COUNT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let mut rng = qal_gen::Rng(qal_gen::seed());
    for i in 0..count {
        let text = qal_gen::well_typed(&mut rng);
        let Ok(p) = Rust.parse(&text) else { continue };
        Both.check(&p, &|_| true)
            .unwrap_or_else(|ds| panic!("gen #{i}: Both.check failed: {ds:?}\n{text}"));
    }
}

#[test]
fn differential_typed_op_corpus() {
    init_qalc();
    let mut tested = 0;
    for s in qal_gen::corpus() {
        let Ok(p) = Rust.parse(&s.text) else { continue };
        if Rust.check(&p, &|_| true).is_err() {
            continue;
        }
        let Ok(st) = qal_compiler::proto::state_of(&p) else { continue };
        if st.entries.is_empty() {
            continue;
        }
        tested += 1;
        let roots: Vec<String> = st.entries.iter().map(|e| e.name.clone()).collect();
        let req = qal_compiler::proto::Request::Typed {
            state: st,
            roots,
            options: Default::default(),
        }
        .encode();
        let r_res = qal_compiler::proto::serve(&req);
        let q_res = qal_compiler::call_qal(&req).expect("Qal failed on TYPED");
        if r_res != q_res {
            let first = r_res.iter().zip(q_res.iter()).position(|(a, b)| a != b).unwrap_or(r_res.len().min(q_res.len()));
            let r_context = &r_res[first.saturating_sub(10)..first.saturating_add(20).min(r_res.len())];
            let q_context = &q_res[first.saturating_sub(10)..first.saturating_add(20).min(q_res.len())];
            panic!("{}: TYPED mismatch at byte {} (rust len {}, qal len {}):\nrust: {:?}\nqal:  {:?}", s.name, first, r_res.len(), q_res.len(), r_context, q_context);
        }
    }
    assert!(tested > 400, "expected > 400 TYPED tested programs, got {tested}");
}

#[test]
fn differential_define_op_corpus() {
    init_qalc();
    let mut tested = 0;
    for s in qal_gen::corpus() {
        let Ok(p) = Rust.parse(&s.text) else { continue };
        if Rust.check(&p, &|_| true).is_err() || Rust.hash(&p).is_err() {
            continue;
        }
        tested += 1;
        let req = qal_compiler::proto::Request::Define {
            state: Default::default(),
            text: s.text.clone(),
            options: Default::default(),
            new_names: None,
        }
        .encode();
        let r_res = qal_compiler::proto::serve(&req);
        let q_res = match qal_compiler::call_qal(&req) {
            Ok(res) => res,
            Err(e) => panic!("{}: Qal failed on DEFINE: {:?}", s.name, e),
        };
        assert_eq!(r_res, q_res, "{}: DEFINE operation divergence", s.name);
    }
    assert!(tested > 400, "expected > 400 DEFINE tested programs, got {tested}");
}
