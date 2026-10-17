//! Differential testing of the self-hosted QAL compiler frontend against the Rust reference.
//! (Phase 5b, spec §10.1, §10.2).

use qal_compiler::{Backend, CompileOptions, Entry, Qal, Rust};
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
fn qalc_wasm_initializes() {
    init_qalc();
    assert!(qal_compiler::has_qal_driver());
}

#[test]
fn differential_parse_and_print_corpus() {
    init_qalc();
    let mut parsed_count = 0;
    for s in qal_gen::corpus() {
        let r_parse = Rust.parse(&s.text);
        let q_parse = Qal.parse(&s.text);

        match (&r_parse, &q_parse) {
            (Ok(p_r), Ok(p_q)) => {
                parsed_count += 1;
                assert_eq!(p_r, p_q, "{}: AST divergence", s.name);

                // Test canonical print
                let r_print = Rust.print(p_r);
                let q_print = Qal.print(p_q);
                assert_eq!(r_print, q_print, "{}: Print divergence", s.name);
            }
            (Err(d_r), Err(d_q)) => {
                assert_eq!(d_r.code, d_q.code, "{}: Error code divergence (Rust: {}, Qal: {})", s.name, d_r.code, d_q.code);
            }
            (Ok(_), Err(d_q)) => {
                panic!("{}: Rust parsed ok, but Qal failed with: {d_q:?}", s.name);
            }
            (Err(d_r), Ok(_)) => {
                panic!("{}: Rust failed with {d_r:?}, but Qal parsed ok", s.name);
            }
        }
    }
    assert!(parsed_count > 500, "expected > 500 parsed programs, got {parsed_count}");
}

#[test]
fn differential_hash_corpus() {
    init_qalc();
    let mut hashed_count = 0;
    for s in qal_gen::corpus() {
        let Ok(p) = Rust.parse(&s.text) else { continue };
        let r_hash = Rust.hash(&p);
        let q_hash = Qal.hash(&p);

        match (&r_hash, &q_hash) {
            (Ok(h_r), Ok(h_q)) => {
                hashed_count += 1;
                assert_eq!(h_r.defs.len(), h_q.defs.len(), "{}: defs count divergence", s.name);
                for (dr, dq) in h_r.defs.iter().zip(h_q.defs.iter()) {
                    assert_eq!(dr.name, dq.name, "{}: def name divergence", s.name);
                    assert_eq!(dr.kind, dq.kind, "{}: def kind divergence for {}", s.name, dr.name);
                    assert_eq!(dr.hash, dq.hash, "{}: def hash divergence for {}", s.name, dr.name);
                    assert_eq!(dr.canonical, dq.canonical, "{}: def canonical divergence for {}", s.name, dr.name);
                    assert_eq!(dr.signature, dq.signature, "{}: signature divergence for {}", s.name, dr.name);
                    assert_eq!(dr.effects, dq.effects, "{}: effects divergence for {}", s.name, dr.name);
                    assert_eq!(dr.local_names, dq.local_names, "{}: local_names divergence for {}", s.name, dr.name);
                    assert_eq!(dr.refs, dq.refs, "{}: refs divergence for {}", s.name, dr.name);
                    assert_eq!(dr.deps, dq.deps, "{}: deps divergence for {}", s.name, dr.name);
                    assert_eq!(dr.group, dq.group, "{}: group divergence for {}", s.name, dr.name);
                }
                assert_eq!(h_r.groups, h_q.groups, "{}: groups divergence", s.name);
            }
            (Err(_), Err(_)) => {}
            (Ok(_), Err(e)) => panic!("{}: Rust hashed ok, but Qal failed: {e:?}", s.name),
            (Err(e), Ok(_)) => panic!("{}: Rust failed with {e:?}, but Qal hashed ok", s.name),
        }
    }
    assert!(hashed_count > 500, "expected > 500 hashed programs, got {hashed_count}");
}

#[test]
fn differential_store_decode_corpus() {
    init_qalc();
    let mut decoded_count = 0;
    for s in qal_gen::corpus() {
        let Ok(p) = Rust.parse(&s.text) else { continue };
        if Rust.check(&p, &|_| true).is_err() {
            continue;
        }
        let Ok(st) = qal_compiler::proto::state_of(&p) else { continue };
        if st.entries.is_empty() {
            continue;
        }
        let mut names: Vec<String> = st.entries.iter().map(|e| e.name.clone()).collect();
        names.dedup();
        if names.len() != st.entries.len() {
            // StoreState invariant requires unique sorted entry names
            continue;
        }
        decoded_count += 1;

        let print_req = qal_compiler::proto::Request::Print {
            state: st.clone(),
            names: names.clone(),
        }
        .encode();
        let r_print_res = qal_compiler::proto::serve(&print_req);
        let q_print_res = qal_compiler::call_qal(&print_req).expect("Qal driver failed on PRINT");
        assert_eq!(r_print_res, q_print_res, "{}: StoreState PRINT divergence", s.name);

        let sig_req = qal_compiler::proto::Request::Signatures { state: st, names }.encode();
        let r_sig_res = qal_compiler::proto::serve(&sig_req);
        let q_sig_res = qal_compiler::call_qal(&sig_req).expect("Qal driver failed on SIGNATURES");
        assert_eq!(r_sig_res, q_sig_res, "{}: StoreState SIGNATURES divergence", s.name);
    }
    assert!(decoded_count > 500, "expected > 500 store-decoded programs, got {decoded_count}");
}

#[test]
fn differential_generated_programs() {
    init_qalc();
    let count = std::env::var("QAL_GEN_COUNT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let mut rng = qal_gen::Rng(qal_gen::seed());
    for i in 0..count {
        let text = qal_gen::well_typed(&mut rng);
        let p_r = Rust.parse(&text).unwrap_or_else(|d| panic!("gen #{i}: Rust parse failed: {d:?}\n{text}"));
        let p_q = Qal.parse(&text).unwrap_or_else(|d| panic!("gen #{i}: Qal parse failed: {d:?}\n{text}"));
        assert_eq!(p_r, p_q, "gen #{i}: AST mismatch");

        let r_print = Rust.print(&p_r);
        let q_print = Qal.print(&p_q);
        assert_eq!(r_print, q_print, "gen #{i}: print mismatch");

        let r_hash = Rust.hash(&p_r).unwrap();
        let q_hash = Qal.hash(&p_q).unwrap();
        assert_eq!(r_hash.defs.len(), q_hash.defs.len());
        for (dr, dq) in r_hash.defs.iter().zip(q_hash.defs.iter()) {
            assert_eq!(dr.hash, dq.hash, "gen #{i}: hash mismatch for {}", dr.name);
            assert_eq!(dr.canonical, dq.canonical, "gen #{i}: canonical mismatch for {}", dr.name);
            assert_eq!(dr.signature, dq.signature, "gen #{i}: signature mismatch for {}", dr.name);
        }
    }
}

fn has_non_tail_self_call(e: &qal_core::ast::Expr, fn_name: &str, in_tail: bool) -> bool {
    use qal_core::ast::Expr;
    match e {
        Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::Str(_) | Expr::Unit | Expr::Var(_) => false,
        Expr::Call(func, args) => {
            if args.iter().any(|a| has_non_tail_self_call(a, fn_name, false)) {
                return true;
            }
            if func == fn_name && !in_tail {
                return true;
            }
            false
        }
        Expr::Bin(_, left, right) => {
            has_non_tail_self_call(left, fn_name, false) || has_non_tail_self_call(right, fn_name, false)
        }
        Expr::Un(_, expr) | Expr::Field(expr, _) => has_non_tail_self_call(expr, fn_name, false),
        Expr::If(cond, then_branch, else_branch) => {
            has_non_tail_self_call(cond, fn_name, false)
                || has_non_tail_self_call(then_branch, fn_name, in_tail)
                || has_non_tail_self_call(else_branch, fn_name, in_tail)
        }
        Expr::Let(_, _, value, body) => {
            has_non_tail_self_call(value, fn_name, false) || has_non_tail_self_call(body, fn_name, in_tail)
        }
        Expr::List(items) => items.iter().any(|x| has_non_tail_self_call(x, fn_name, false)),
        Expr::Record(_, fields) => fields.iter().any(|(_, x)| has_non_tail_self_call(x, fn_name, false)),
        Expr::Match(scrutinee, arms) => {
            has_non_tail_self_call(scrutinee, fn_name, false)
                || arms.iter().any(|a| {
                    a.guard.as_ref().is_some_and(|g| has_non_tail_self_call(g, fn_name, false))
                        || has_non_tail_self_call(&a.body, fn_name, in_tail)
                })
        }
        Expr::Seq(items) => {
            let n = items.len();
            items.iter().enumerate().any(|(i, x)| has_non_tail_self_call(x, fn_name, in_tail && i + 1 == n))
        }
        Expr::Lambda(_, body) => has_non_tail_self_call(body, fn_name, false),
    }
}

#[test]
fn qalc_sequence_loops_are_tail_recursive() {
    let src = std::fs::read_to_string(qal_gen::qal_root().join("selfhost/qalc.qal"))
        .expect("cannot read selfhost/qalc.qal");
    let prog = qal_compiler::parse_program(&src).expect("qalc.qal must parse");
    for d in &prog.defs {
        if let qal_core::ast::Def::Fn(f) = d {
            if f.name.ends_with("_loop") || f.name.ends_with("_rec") {
                assert!(
                    !has_non_tail_self_call(&f.body, &f.name, true),
                    "sequence helper `{}` has a non-tail recursive self-call",
                    f.name
                );
            }
        }
    }
}


