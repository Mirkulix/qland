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
fn debug_json_body_hash() {
    init_qalc();
    let s = qal_gen::corpus().into_iter().find(|s| s.name.contains("todo_isolation")).unwrap();
    let p = Rust.parse(&s.text).unwrap();
    let r_hash = Rust.hash(&p).unwrap();
    let q_hash = Qal.hash(&p).unwrap();
    let r_def = r_hash.defs.iter().find(|d| d.name == "std.json.parse_value").unwrap();
    let q_def = q_hash.defs.iter().find(|d| d.name == "std.json.parse_value").unwrap();
    let g_r = r_def.group.unwrap();
    let g_q = q_def.group.unwrap();
    let r_members: Vec<_> = r_hash.defs.iter().filter(|d| d.group == Some(g_r)).map(|d| d.name.as_str()).collect();
    let q_members: Vec<_> = q_hash.defs.iter().filter(|d| d.group == Some(g_q)).map(|d| d.name.as_str()).collect();
    for name in &r_members {
        let rd = r_hash.defs.iter().find(|d| d.name == *name).unwrap();
        let qd = q_hash.defs.iter().find(|d| d.name == *name).unwrap();
        // Byte 5 is the varint index in member canonical form
        eprintln!("Member {name}: Rust index={}, Qal index={}", rd.canonical[6], qd.canonical[6]);
    }
    eprintln!("Rust hash: {}", r_def.hash);
    eprintln!("Qal hash:  {}", q_def.hash);
    eprintln!("Rust canon len: {}", r_def.canonical.len());
    eprintln!("Qal canon len:  {}", q_def.canonical.len());
    for (i, (rb, qb)) in r_def.canonical.iter().zip(q_def.canonical.iter()).enumerate() {
        if rb != qb {
            eprintln!("First diff at byte {i}: Rust {rb} (0x{rb:02x}), Qal {qb} (0x{qb:02x})");
            let h_r: [u8; 32] = r_def.canonical[i..i+32].try_into().unwrap();
            let h_q: [u8; 32] = q_def.canonical[i..i+32].try_into().unwrap();
            let name_r = r_hash.defs.iter().find(|d| d.hash == qal_core::canon::Hash(h_r)).map(|d| d.name.as_str());
            let name_q = q_hash.defs.iter().find(|d| d.hash == qal_core::canon::Hash(h_q)).map(|d| d.name.as_str());
            eprintln!("Ref hash in Rust: {name_r:?}");
            eprintln!("Ref hash in Qal:  {name_q:?}");
            break;
        }
    }
}

#[test]
fn check_corpus_groups() {
    let mut total_groups = 0;
    for s in qal_gen::corpus() {
        if let Ok(p) = qal_compiler::parse_program(&s.text) {
            if let Ok(h) = qal_compiler::hash_program(&p) {
                if !h.groups.is_empty() {
                    total_groups += h.groups.len();
                    let group_defs: Vec<_> = h.defs.iter().filter(|d| d.group.is_some()).map(|d| d.name.as_str()).collect();
                    eprintln!("Corpus program with groups: {} ({} groups, defs: {:?})", s.name, h.groups.len(), group_defs);
                }
            }
        }
    }
    eprintln!("Total groups in corpus: {}", total_groups);
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
                if r_print != q_print {
                    let r_lines: Vec<_> = r_print.lines().collect();
                    let q_lines: Vec<_> = q_print.lines().collect();
                    for (i, (rl, ql)) in r_lines.iter().zip(q_lines.iter()).enumerate() {
                        if rl != ql {
                            eprintln!("Line {i} mismatch:\n  Rust: {rl}\n  Qal:  {ql}");
                            break;
                        }
                    }
                    if r_lines.len() != q_lines.len() {
                        eprintln!("Line count mismatch: Rust {} lines, Qal {} lines", r_lines.len(), q_lines.len());
                    }
                }
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
                    assert_eq!(dr.hash, dq.hash, "{}: def hash divergence for {}", s.name, dr.name);
                    assert_eq!(dr.canonical, dq.canonical, "{}: def canonical divergence for {}", s.name, dr.name);
                }
            }
            (Err(_), Err(_)) => {}
            (Ok(_), Err(e)) => panic!("{}: Rust hashed ok, but Qal failed: {e:?}", s.name),
            (Err(e), Ok(_)) => panic!("{}: Rust failed with {e:?}, but Qal hashed ok", s.name),
        }
    }
    assert!(hashed_count > 500, "expected > 500 hashed programs, got {hashed_count}");
}

#[test]
fn differential_generated_programs() {
    init_qalc();
    let count = std::env::var("QAL_GEN_COUNT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
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
        }
    }
}
