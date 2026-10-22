//! Generated property tests: every pure function with an `ens`, or that calls a function with a
//! `req`, is run on random inputs that satisfy its own `req`; the property is "no contract is
//! violated". A failure is shrunk to a smallest input that still violates the same contract.
//!
//! - Inputs satisfying `req` are found by rejection: a generated helper `req?<f>` evaluates f's `req`
//!   on the input (an input for which it is false or traps is discarded).
//! - A run that violates any contract (f's `ens`, a callee's `req` or `ens`, also f's own `req` at a
//!   recursive call, a built-in's precondition) fails the property. Other traps (Int overflow,
//!   division by zero, index, stack, memory, time) are counted as `traps`, not failures: like the
//!   proofs, properties are about contracts (partial correctness).
//! - Deterministic: the inputs follow from the seed and the function's name; sizes grow from 0 to
//!   `max_size` over the runs.

use crate::gen::{fnv, generatable, random, shrink, Rng};
use qal_core::ast::{BinOp, FnDef, Type};
use qal_core::tast::{TExpr, TFn, TKind, TProgram};
use qal_run::{RunError, Runner, Value, Values};
use qal_compiler::Entry;
use serde::Serialize;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq)]
pub struct PropOptions {
    /// Runs (inputs that satisfy `req`) per function.
    pub runs: u32,
    pub seed: u64,
    /// Size of the last inputs (Int magnitude, collection length).
    pub max_size: u64,
    /// Candidate inputs tried while shrinking a failure.
    pub shrink_steps: u32,
    /// Instruction budget per run.
    pub fuel: u64,
    /// Wall-clock limit per run.
    pub timeout: Duration,
    /// Wall-clock budget of all properties; functions after it are `skipped`.
    pub budget: Duration,
}

impl Default for PropOptions {
    fn default() -> PropOptions {
        PropOptions { runs: 100, seed: 0, max_size: 30, shrink_steps: 500, fuel: 20_000_000, timeout: Duration::from_secs(2), budget: Duration::from_secs(20) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PropStatus {
    Passed,
    Failed,
    /// Too few generated inputs satisfied `req`.
    GaveUp,
    /// Not run, or stopped before its runs were done, because the time budget was used up (`runs`
    /// says how many were run).
    Skipped,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PropReport {
    /// The function.
    pub def: String,
    pub status: PropStatus,
    /// Inputs that satisfied `req` and were run.
    pub runs: u32,
    /// Generated inputs that did not satisfy `req`.
    pub discarded: u32,
    /// Runs that ended in a trap other than a contract violation.
    pub traps: u32,
    pub seed: u64,
    /// For `failed`: the shrunk input (parameter name -> value).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub counterexample: Option<Values>,
    /// For `failed`: the contract violation on the counterexample.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RunError>,
    /// The first trap that is not a contract violation (counted in `traps`, not a failure).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trap: Option<RunError>,
}

/// Name of the generated function that evaluates the `req` of `f`.
fn req_name(f: &str) -> String {
    format!("req?{f}")
}

/// True if `f` can violate a contract on its own: it has an `ens`, or calls a function with `req`.
fn has_obligations(tp: &TProgram, f: &TFn) -> bool {
    fn calls_req(tp: &TProgram, e: &TExpr) -> bool {
        if let TKind::Call(i, _) = &e.kind {
            if tp.fns.get(*i).is_some_and(|g| !g.req.is_empty()) {
                return true;
            }
        }
        children(e).into_iter().any(|c| calls_req(tp, c))
    }
    !f.ens.is_empty() || calls_req(tp, &f.body)
}

/// The functions with properties, by index into `tp.fns`: pure, generatable parameters, obligations.
pub fn candidates(tp: &TProgram) -> Vec<usize> {
    (0..tp.fns.len())
        .filter(|&i| {
            let f = &tp.fns[i];
            // A stub (`todo()`) has no behaviour to test yet.
            f.def.effects.is_empty() && f.def.params.iter().all(|p| generatable(tp, &p.ty)) && has_obligations(tp, f) && !f.is_stub()
        })
        .collect()
}

/// A copy of `tp` with a function `req?f` (the conjunction of f's `req`, true if none) for every
/// candidate `f`, compiled with all functions and tests.
fn with_req_checks(tp: &TProgram, fns: &[usize]) -> TProgram {
    let mut out = tp.clone();
    for &i in fns {
        let f = &tp.fns[i];
        let truth = TExpr { ty: Type::Bool, kind: TKind::Bool(true) };
        let body = f.req.iter().cloned().reduce(|a, b| TExpr { ty: Type::Bool, kind: TKind::Bin(BinOp::And, Box::new(a), Box::new(b), "req".into()) }).unwrap_or(truth);
        let def = FnDef { name: req_name(&f.def.name), tparams: vec![], eparams: vec![], params: f.def.params.clone(), ret: Type::Bool, effects: vec![], req: vec![], ens: vec![], body: f.def.body.clone() };
        out.fns.push(TFn { def, locals: f.locals.clone(), req: vec![], ens: vec![], result_local: None, body, instance: None });
    }
    out
}

/// What one run on an input showed.
#[derive(Debug, Clone, PartialEq)]
enum Run {
    /// The input does not satisfy `req`.
    Discard,
    Pass,
    Trap(RunError),
    /// A contract violation.
    Fail(RunError),
}

fn run_one(runner: &Runner, f: &str, args: &[Value]) -> Run {
    match runner.call(&req_name(f), args) {
        Ok(Value::Bool(true)) => {}
        _ => return Run::Discard,
    }
    match runner.call(f, args) {
        Ok(_) => Run::Pass,
        Err(e) if e.kind == "contract_violation" => Run::Fail(e),
        Err(e) => Run::Trap(e),
    }
}

/// The same contract failed (definition and contract, not the values).
fn same_failure(a: &RunError, b: &RunError) -> bool {
    a.def == b.def && a.contract == b.contract && a.path == b.path
}

/// Runs the properties of the candidate functions named `only` (all if None). `runner` must have
/// been compiled from `with_req_checks` — use [`check_properties`].
fn run_properties(tp: &TProgram, runner: &Runner, fns: &[usize], opts: &PropOptions) -> Vec<PropReport> {
    let start = Instant::now();
    let mut out = Vec::new();
    for &i in fns {
        let f = &tp.fns[i];
        let name = &f.def.name;
        let seed = opts.seed ^ fnv(name);
        let mut report = PropReport { def: name.clone(), status: PropStatus::Passed, runs: 0, discarded: 0, traps: 0, seed: opts.seed, counterexample: None, error: None, trap: None };
        if start.elapsed() > opts.budget {
            report.status = PropStatus::Skipped;
            out.push(report);
            continue;
        }
        let mut rng = Rng::new(seed);
        let types: Vec<Type> = f.def.params.iter().map(|p| p.ty.clone()).collect();
        let max_tries = opts.runs.saturating_mul(10).max(1);
        let mut tries = 0;
        let mut out_of_time = false;
        while report.runs < opts.runs && tries < max_tries {
            if start.elapsed() > opts.budget {
                out_of_time = true;
                break;
            }
            // Sizes grow with every attempt (not only with successful runs), so a `req` that small
            // inputs never satisfy does not stall generation at size 0.
            let size = (opts.max_size * tries as u64 / opts.runs.max(1) as u64).min(opts.max_size);
            tries += 1;
            let Some(args) = types.iter().map(|t| random(tp, t, size, &mut rng)).collect::<Option<Vec<Value>>>() else { break };
            match run_one(runner, name, &args) {
                Run::Discard => report.discarded += 1,
                Run::Pass => report.runs += 1,
                Run::Trap(e) => {
                    report.runs += 1;
                    report.traps += 1;
                    report.trap.get_or_insert(e);
                }
                Run::Fail(e) => {
                    report.runs += 1;
                    let (args, e) = shrink_failure(tp, runner, name, &types, args, e, opts, start);
                    report.status = PropStatus::Failed;
                    report.counterexample = Some(Values(f.def.params.iter().map(|p| p.name.clone()).zip(args).collect()));
                    report.error = Some(e);
                    break;
                }
            }
        }
        // Running out of time is not giving up: `gave_up` means too few inputs satisfied `req`.
        if report.status == PropStatus::Passed && out_of_time {
            report.status = PropStatus::Skipped;
        } else if report.status == PropStatus::Passed && report.runs < opts.runs / 2 {
            report.status = PropStatus::GaveUp;
        }
        out.push(report);
    }
    out
}

/// Greedily replaces one argument at a time by a simpler one that still fails the same way, until
/// `opts.shrink_steps` candidates were tried or the time budget (counted from `start`) is used up.
#[allow(clippy::too_many_arguments)]
fn shrink_failure(tp: &TProgram, runner: &Runner, f: &str, types: &[Type], mut args: Vec<Value>, mut err: RunError, opts: &PropOptions, start: Instant) -> (Vec<Value>, RunError) {
    let mut steps = 0;
    'outer: loop {
        for k in 0..args.len() {
            for c in shrink(tp, &types[k], &args[k]) {
                if steps >= opts.shrink_steps || start.elapsed() > opts.budget {
                    break 'outer;
                }
                steps += 1;
                let mut cand = args.clone();
                cand[k] = c;
                if let Run::Fail(e) = run_one(runner, f, &cand) {
                    if same_failure(&e, &err) {
                        args = cand;
                        err = e;
                        continue 'outer;
                    }
                }
            }
        }
        break;
    }
    (args, err)
}

/// Runs the properties of the functions named `only` (every candidate if None) of a checked
/// program; names that are not candidates are left out.
pub fn check_properties(tp: &TProgram, only: Option<&[String]>, opts: &PropOptions) -> Result<Vec<PropReport>, RunError> {
    let fns: Vec<usize> = candidates(tp).into_iter().filter(|&i| only.is_none_or(|ns| ns.iter().any(|n| *n == tp.fns[i].def.name))).collect();
    if fns.is_empty() {
        return Ok(Vec::new());
    }
    let program = with_req_checks(tp, &fns);
    let module = qal_compiler::compile(&program, &Entry::All).map_err(|m| RunError::new("compile_error", "", format!("the program cannot be compiled to WebAssembly: {m}")))?;
    let runner = Runner::new(module).map_err(|m| RunError::new("compile_error", "", m))?.with_fuel(opts.fuel).with_timeout(opts.timeout);
    Ok(run_properties(&program, &runner, &fns, opts))
}

/// The direct subexpressions of `e` (closure bodies included).
pub(crate) fn children(e: &TExpr) -> Vec<&TExpr> {
    match &e.kind {
        TKind::Int(_) | TKind::Float(_) | TKind::Bool(_) | TKind::Str(_) | TKind::Unit | TKind::Local(_) | TKind::FnRef(_) | TKind::FnRefG(..) => vec![],
        TKind::Lambda(l) => vec![&l.body],
        TKind::Un(_, x, _) | TKind::Field(x, _) => vec![x],
        TKind::Bin(_, l, r, _) | TKind::Let(_, l, r) => vec![l, r],
        TKind::If(c, t, f) => vec![c, t, f],
        TKind::Call(_, xs) | TKind::CallG(_, _, xs) | TKind::List(xs) | TKind::Variant(_, xs) | TKind::Seq(xs) | TKind::Builtin(_, xs, _) => xs.iter().collect(),
        TKind::CallIndirect(f, xs) => std::iter::once(&**f).chain(xs.iter()).collect(),
        TKind::Record(fs) => fs.iter().map(|(_, x)| x).collect(),
        TKind::Match(s, arms) => std::iter::once(&**s).chain(arms.iter().flat_map(|a| a.guard.iter().chain([&a.body]))).collect(),
    }
}
