//! The corpus (spec §10.1) through the Rust pipeline: its size is fixed here, every program that
//! parses prints and re-parses to itself, and every program that checks has a typed form that
//! round-trips through `QALT` (spec §4.4, the 5a criterion), a deterministic hash and a module.

use qal_compiler::Entry;
use qal_core::tform;

/// Programs of part (1): examples, fixtures, 20 benchmark solutions, the std and todo packages.
const FILES: usize = 27;
/// Programs of part (2), extracted by `selfhost/tools/extract_corpus.py`.
const EXTRACTED: usize = 640;

#[test]
fn the_corpus_size_is_fixed() {
    let files = qal_gen::file_corpus();
    let names: Vec<&str> = files.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(files.len(), FILES, "{names:#?}");
    assert_eq!(names.iter().filter(|n| n.starts_with("bench/solutions/qal/")).count(), 20);
    assert!(names.contains(&"std/std.qalpkg") && names.contains(&"examples/todo/todo.qalpkg"));
    assert!(files.iter().all(|s| !s.text.is_empty()), "unreadable corpus file");
    assert_eq!(qal_gen::extracted_corpus().len(), EXTRACTED, "re-run qal/selfhost/tools/extract_corpus.py and update EXTRACTED");
}

#[derive(Default, Debug)]
struct Stats {
    programs: usize,
    parsed: usize,
    checked: usize,
    typed_bytes: usize,
}

fn run(src: &str, name: &str, st: &mut Stats) {
    st.programs += 1;
    let Ok(p) = qal_compiler::parse_program(src) else { return };
    st.parsed += 1;
    let printed = qal_compiler::print_program(&p);
    let again = qal_compiler::parse_program(&printed).unwrap_or_else(|d| panic!("{name}: printed text does not parse: {d:?}\n{printed}"));
    assert_eq!(again, p, "{name}: print/parse round trip");
    let Ok(tp) = qal_compiler::check(&p) else { return };
    st.checked += 1;
    let bytes = tform::encode(&tp);
    st.typed_bytes += bytes.len();
    let back = tform::decode(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(tform::encode(&back), bytes, "{name}: QALT round trip");
    assert_eq!(format!("{back:?}"), format!("{tp:?}"), "{name}: QALT round trip");
    if let Ok(h) = qal_compiler::hash_program(&p) {
        assert_eq!(qal_compiler::hash_program(&p).ok(), Some(h), "{name}: hashing is deterministic");
    }
    let m = qal_compiler::compile(&tp, &Entry::All).unwrap_or_else(|e| panic!("{name}: {e}"));
    // HashMap audit (spec §14.1): every HashMap gets a fresh random seed, so a second run would
    // expose any output that depends on hash iteration order.
    let again = qal_compiler::check(&p).unwrap_or_else(|d| panic!("{name}: {d:?}"));
    assert_eq!(tform::encode(&again), bytes, "{name}: checking is not deterministic");
    assert_eq!(qal_compiler::compile(&again, &Entry::All).map(|m| m.wasm).ok(), Some(m.wasm), "{name}: codegen is not deterministic");
}

#[test]
fn every_checked_corpus_program_round_trips_through_qalt() {
    let mut st = Stats::default();
    for s in qal_gen::corpus() {
        run(&s.text, &s.name, &mut st);
    }
    eprintln!("corpus: {st:?}");
    assert_eq!(st.programs, FILES + EXTRACTED);
    // Every file of part (1) checks; the extracted part also holds programs with errors on purpose.
    let mut files = Stats::default();
    for s in qal_gen::file_corpus() {
        run(&s.text, &s.name, &mut files);
    }
    assert_eq!((files.parsed, files.checked), (FILES, FILES), "{files:?}");
    // Fixed with the corpus: a change means the corpus or the pipeline changed.
    assert_eq!((st.parsed, st.checked), (617, 429), "{st:?}");
}

