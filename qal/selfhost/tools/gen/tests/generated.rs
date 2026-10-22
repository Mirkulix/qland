//! Generated programs (spec §10.1 (3)): `QAL_GEN_COUNT` (default 10 000) per run, the seed in the
//! log (`QAL_GEN_SEED` reproduces a run). Half are well-typed by construction and must check; half
//! are corpus programs with token mutations and exercise the diagnostic paths. Every program goes
//! through the whole Rust pipeline without a panic; whatever parses prints and re-parses to itself;
//! whatever checks round-trips through `QALT`, hashes deterministically and compiles.

use qal_compiler::Entry;
use qal_core::tform;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Default, Debug)]
struct Stats {
    well_typed: usize,
    mutated: usize,
    mutated_parsed: usize,
    mutated_checked: usize,
}

/// Runs the pipeline; Err describes the first broken property.
fn pipeline(src: &str, must_check: bool) -> Result<(bool, bool), String> {
    let p = match qal_compiler::parse_program(src) {
        Ok(p) => p,
        Err(d) if must_check => return Err(format!("does not parse: {d:?}")),
        Err(_) => return Ok((false, false)),
    };
    let printed = qal_compiler::print_program(&p);
    match qal_compiler::parse_program(&printed) {
        Ok(again) if again == p => {}
        other => return Err(format!("print/parse round trip broken: {other:?}\n--- printed:\n{printed}")),
    }
    let tp = match qal_compiler::check(&p) {
        Ok(tp) => tp,
        Err(d) if must_check => return Err(format!("does not check: {d:#?}")),
        Err(_) => return Ok((true, false)),
    };
    let bytes = tform::encode(&tp);
    match tform::decode(&bytes) {
        Ok(back) if tform::encode(&back) == bytes => {}
        other => return Err(format!("QALT round trip broken: {:?}", other.map(|_| "different bytes"))),
    }
    let h1 = qal_compiler::hash_program(&p).map_err(|e| e.to_string());
    let h2 = qal_compiler::hash_program(&p).map_err(|e| e.to_string());
    if h1 != h2 {
        return Err("hashing is not deterministic".into());
    }
    qal_compiler::compile(&tp, &Entry::All).map_err(|e| format!("does not compile: {e}"))?;
    Ok((true, true))
}

#[test]
fn generated_programs_pass_the_pipeline() {
    let seed = qal_gen::seed();
    let n = qal_gen::count();
    eprintln!("qal-gen: seed {seed}, {n} programs (QAL_GEN_SEED={seed} reproduces this run)");
    let mut rng = qal_gen::Rng(seed);
    // Small corpus programs are mutated (the large packages would dominate the time).
    let sources: Vec<String> = qal_gen::corpus().into_iter().map(|s| s.text).filter(|t| t.len() < 4000).collect();
    let mut st = Stats::default();
    let mut failures = Vec::new();
    for i in 0..n {
        let well = i % 2 == 0;
        let src = if well { qal_gen::well_typed(&mut rng) } else { qal_gen::mutate(rng.pick(&sources).clone().as_str(), &mut rng) };
        let r = catch_unwind(AssertUnwindSafe(|| pipeline(&src, well))).unwrap_or_else(|p| Err(format!("panic: {:?}", p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())))));
        match r {
            Ok((parsed, checked)) => {
                if well {
                    st.well_typed += 1;
                } else {
                    st.mutated += 1;
                    st.mutated_parsed += parsed as usize;
                    st.mutated_checked += checked as usize;
                }
            }
            Err(e) if failures.len() < 3 => failures.push(format!("program {i} (seed {seed}): {e}\n--- source:\n{src}")),
            Err(_) => {}
        }
    }
    eprintln!("qal-gen: {st:?}");
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn generation_is_a_function_of_the_seed() {
    let a: Vec<String> = (0..20).map(|k| qal_gen::well_typed(&mut qal_gen::Rng(k))).collect();
    let b: Vec<String> = (0..20).map(|k| qal_gen::well_typed(&mut qal_gen::Rng(k))).collect();
    assert_eq!(a, b);
    let src = "fn f(x: Int) -> Int = x + 1\ntest t = f(1) == 2";
    assert_eq!(qal_gen::mutate(src, &mut qal_gen::Rng(7)), qal_gen::mutate(src, &mut qal_gen::Rng(7)));
    assert!((0..50).any(|k| qal_gen::mutate(src, &mut qal_gen::Rng(k)) != src));
}
