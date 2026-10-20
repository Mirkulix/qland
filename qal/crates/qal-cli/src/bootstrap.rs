//! `qal selfhost bootstrap` (phase 5 §0 criterion 3): Stage 1 is `qalc` compiled by the Rust
//! pipeline, Stage 2 is `qalc` compiled by Stage 1, Stage 3 is `qalc` compiled by Stage 2. Each
//! QAL stage parses, checks and compiles the whole source itself. The fixpoint holds when
//! Stage 2 == Stage 3 byte for byte; the seed module `selfhost/qalc.wasm` is that fixpoint.

use qal_compiler::{Backend, CompileOptions, CompiledModule, Entry, Qal, Rust};
use qal_core::canon::Hash;
use qal_core::diag::Diagnostic;
use serde_json::{json, Value};
use std::time::Instant;

fn failed(stage: &str, error: &str) -> Diagnostic {
    Diagnostic::compiler_failed(stage, "BOOTSTRAP", "main", error)
}

fn first(ds: Vec<Diagnostic>, stage: &str) -> Diagnostic {
    ds.into_iter().next().unwrap_or_else(|| failed(stage, "check failed without a diagnostic"))
}

/// `qalc` compiled by the current QAL backend (parse, check and compile all in QAL).
fn qal_stage(src: &str, stage: &str) -> Result<CompiledModule, Diagnostic> {
    let p = Qal.parse(src)?;
    let tp = Qal.check(&p, &|_| true).map_err(|ds| first(ds, stage))?;
    Qal.compile(&tp, &Entry::All, &CompileOptions::compiler()).map_err(|e| failed(stage, &e))
}

/// Runs the bootstrap on the compiler source `src` and returns the report and the Stage 3 module.
pub fn bootstrap(src: &str, jobs: usize) -> Result<(Value, CompiledModule), Diagnostic> {
    let t0 = Instant::now();
    let p = Rust.parse(src)?;
    let tp = Rust.check(&p, &|_| true).map_err(|ds| first(ds, "rust"))?;
    let stage1 = Rust.compile(&tp, &Entry::All, &CompileOptions::compiler()).map_err(|e| failed("rust", &e))?;
    let t1 = t0.elapsed();

    qal_run::selfhost::init_qal_backend(stage1.clone(), jobs).map_err(|e| failed("stage 1", &e))?;
    let stage2 = qal_stage(src, "stage 1")?;
    let t2 = t0.elapsed();

    qal_run::selfhost::init_qal_backend(stage2.clone(), jobs).map_err(|e| failed("stage 2", &e))?;
    let stage3 = qal_stage(src, "stage 2")?;
    let t3 = t0.elapsed();

    let report = json!({
        "ok": stage2.wasm == stage3.wasm,
        "stage1_equals_stage2": stage1.wasm == stage2.wasm,
        "stage2_equals_stage3": stage2.wasm == stage3.wasm,
        "wasm_bytes": stage3.wasm.len(),
        "sha256": Hash::of(&stage3.wasm).to_hex(),
        "seconds": { "stage1": t1.as_secs_f64(), "stage2": (t2 - t1).as_secs_f64(), "stage3": (t3 - t2).as_secs_f64() },
    });
    Ok((report, stage3))
}
