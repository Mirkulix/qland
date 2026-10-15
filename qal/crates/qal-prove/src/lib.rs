//! QAL static contract proofs and generated property tests (spec Phase 2).
//!
//! **Proofs** (`prove`): for every function (generic ones once, not their instances) and test, Z3
//! checks each obligation:
//! - `ens[k]`: if the `req` hold and the body returns, `ens[k]` evaluates to true without trapping;
//! - at each call of a function with `req` (in a body, also inside closures, not in contracts):
//!   the callee's `req[k]` holds whenever the call is reached.
//!
//! The result is `proven`, `refuted` (the solver found a counterexample and running the function on
//! it really violates the contract) or `unknown` (the solver gave up, or its counterexample could not
//! be confirmed: the encoding approximates what it does not model, see [`vc`]). Every contract stays
//! checked at runtime, proven or not.
//!
//! **Determinism:** each obligation is solved in a fresh Z3 context with a fixed random seed and a
//! resource limit (`rlimit`, deterministic for a given Z3 version); the wall-clock timeout is only a
//! safety net. Results are a function of the definition, its callees' contracts, the prover version,
//! the Z3 version and the resource limit — [`Options::key`] names all of them.
//!
//! **Property tests** (`props`): generated inputs that satisfy `req`, checked for contract violations.

mod gen;
pub mod props;
mod vc;

pub use gen::{fnv, generatable, random, shrink, smallest, Rng};
pub use props::{check_properties, PropOptions, PropReport, PropStatus};

use qal_core::ast::Type;
use qal_core::diag::Diagnostic;
use qal_core::tast::{TFn, TProgram, TTest};
use qal_run::{RunError, Runner, Value, Values};
use qal_wasm::Entry;
use serde::{Deserialize, Serialize};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::{Duration, Instant};
use vc::{Goal, Term, Vc};
use z3::{Config, Params, SatResult, Solver};

/// Changes whenever the encoding changes (stored results of another version are not reused).
pub const PROVER_VERSION: &str = "qal-prove/1";

/// Reasons of results that depend on the machine's speed or load, not only on the definition.
const NOT_ATTEMPTED: &str = "not attempted:";
const TIMED_OUT: &str = "the solver timed out after";
const INTERNAL: &str = "internal prover error";

#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// Z3 resource limit per obligation (deterministic effort bound).
    pub rlimit: u32,
    /// Wall-clock limit per obligation in milliseconds (safety net).
    pub timeout_ms: u32,
    /// Wall-clock budget of one `prove` call; obligations after it are `unknown`.
    pub budget: Duration,
    /// Instruction budget for running a counterexample.
    pub confirm_fuel: u64,
}

impl Default for Options {
    fn default() -> Options {
        Options { rlimit: 10_000_000, timeout_ms: 5_000, budget: Duration::from_secs(30), confirm_fuel: 50_000_000 }
    }
}

impl Options {
    /// Identifies everything a result depends on besides the definition (and through its hash, the
    /// contracts of its callees).
    pub fn key(&self) -> String {
        format!("{PROVER_VERSION} z3-{} rlimit-{}", z3::full_version(), self.rlimit)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Proven,
    Refuted,
    Unknown,
}

/// The result of one obligation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Report {
    /// The function or test the obligation belongs to.
    pub def: String,
    /// `ens[0]`, or `req[1] of g` for a call site.
    pub obligation: String,
    /// AST path: the contract (`ens[0]`) or the call (`body.then`).
    pub path: String,
    /// The contract as text.
    pub contract: String,
    pub status: Status,
    /// Parameter values: the confirmed counterexample (`refuted`) or an unconfirmed candidate (`unknown`).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub counterexample: Option<serde_json::Value>,
    /// Why the result is `unknown`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reason: Option<String>,
}

impl Report {
    /// False for results that another run could improve without any change (time budget used up,
    /// wall-clock timeout, internal error): those are not stored.
    pub fn is_stable(&self) -> bool {
        self.reason.as_deref().is_none_or(|r| !(r.starts_with(NOT_ATTEMPTED) || r.starts_with(TIMED_OUT) || r.starts_with(INTERNAL)))
    }

    /// `refuted` is the error `contract_refuted` (E0801); `unknown` the warning `contract_unproven` (E0802).
    pub fn diagnostic(&self) -> Option<Diagnostic> {
        let mut d = match self.status {
            Status::Proven => return None,
            Status::Refuted => Diagnostic::new("E0801", "contract_refuted", &self.def, &self.path, format!("`{}` ({}) does not hold for the counterexample", self.contract, self.obligation))
                .hint("running the function on the counterexample violates the contract: fix the body, the contract, or add a `req`"),
            Status::Unknown => Diagnostic::new("E0802", "contract_unproven", &self.def, &self.path, format!("`{}` ({}) could not be proven: {}", self.contract, self.obligation, self.reason.as_deref().unwrap_or("unknown")))
                .hint("the contract stays checked at runtime; a simpler or stronger `req`/`ens` (Int and Bool facts) may make it provable"),
        };
        d.counterexample = self.counterexample.clone();
        Some(d)
    }
}

/// What a proof is about: a function by index (monomorphic or generic), or a test.
#[derive(Debug, Clone, Copy)]
enum Which {
    Fn(usize),
    Generic(usize),
    Test(usize),
}

/// The definitions whose obligations are proven, in order: monomorphic functions (not instances),
/// generic functions, tests. `only` restricts them by name.
fn targets(tp: &TProgram, only: Option<&[String]>) -> Vec<(String, Which)> {
    let keep = |n: &str| only.is_none_or(|ns| ns.iter().any(|m| m == n));
    let mut out = Vec::new();
    for (i, f) in tp.fns.iter().enumerate().filter(|(_, f)| f.instance.is_none() && keep(&f.def.name)) {
        out.push((f.def.name.clone(), Which::Fn(i)));
    }
    for (i, f) in tp.generics.iter().enumerate().filter(|(_, f)| keep(&f.def.name)) {
        out.push((f.def.name.clone(), Which::Generic(i)));
    }
    for (i, t) in tp.tests.iter().enumerate().filter(|(_, t)| keep(&t.name)) {
        out.push((t.name.clone(), Which::Test(i)));
    }
    out
}

fn encode(tp: &TProgram, which: Which) -> Option<Vc> {
    Some(match which {
        Which::Fn(i) => vc::function(tp, tp.fns.get(i)?),
        Which::Generic(i) => vc::function(tp, tp.generics.get(i)?),
        Which::Test(i) => vc::test(tp, tp.tests.get(i)?),
    })
}

/// What the solver said about one obligation (plain data: Z3 terms do not leave their context).
enum Solved {
    Proven,
    /// A model: parameter values (None: not modelled by the solver), and for a callee's `req` the
    /// callee's arguments at this call.
    /// The arguments at the other calls with the same goal follow (in the same model).
    Model(Vec<(String, Type, Option<Value>)>, Vec<(String, Option<Value>)>, Vec<Vec<(String, Option<Value>)>>),
    Unknown(String),
}

/// Metadata of an obligation, outside any Z3 context.
struct Meta {
    goal: Goal,
    label: String,
    path: String,
    contract: String,
}

/// Proves the obligations of the definitions named `only` (all if None) of a checked program.
pub fn prove(tp: &TProgram, only: Option<&[String]>, opts: &Options) -> Vec<Report> {
    let start = Instant::now();
    let mut runner: Option<Option<Runner>> = None;
    let mut out = Vec::new();
    for (name, which) in targets(tp, only) {
        let Some((metas, approx, truncated)) = in_context(opts, || encode(tp, which).map(|v| (v.obligations.iter().map(|o| Meta { goal: o.goal.clone(), label: o.label.clone(), path: o.path.clone(), contract: o.contract.clone() }).collect::<Vec<_>>(), v.approx, v.truncated))).flatten() else {
            continue;
        };
        if truncated {
            let reason = format!("{name} and the contracts it uses are too large to encode (more than {} expressions); its contracts stay checked at run time", vc::MAX_VC_NODES);
            out.push(Report { def: name.clone(), obligation: "all".into(), path: String::new(), contract: String::new(), status: Status::Unknown, counterexample: None, reason: Some(reason) });
            continue;
        }
        for (k, m) in metas.into_iter().enumerate() {
            let mut r = Report { def: name.clone(), obligation: m.label, path: m.path, contract: m.contract, status: Status::Unknown, counterexample: None, reason: None };
            if start.elapsed() > opts.budget {
                r.reason = Some(format!("{NOT_ATTEMPTED} the prover's time budget of {} s was used up", opts.budget.as_secs()));
                out.push(r);
                continue;
            }
            // The wall-clock limit of one obligation never reaches past the budget.
            let left = opts.budget.saturating_sub(start.elapsed()).as_millis().max(1) as u32;
            let local = Options { timeout_ms: opts.timeout_ms.min(left), ..opts.clone() };
            match in_context(&local, || solve(tp, which, k, &local)).unwrap_or_else(|| Solved::Unknown(INTERNAL.into())) {
                Solved::Proven => r.status = Status::Proven,
                Solved::Unknown(why) => r.reason = Some(why),
                Solved::Model(params, call, others) => {
                    let shown: Vec<(String, Value)> = params.iter().filter_map(|(n, _, v)| v.clone().map(|v| (n.clone(), v))).collect();
                    r.counterexample = (!shown.is_empty()).then(|| serde_json::to_value(Values(shown)).ok()).flatten();
                    let runner = runner.get_or_insert_with(|| qal_wasm::compile(tp, &Entry::All).ok().and_then(|m| Runner::new(m).ok()).map(|r| r.with_fuel(opts.confirm_fuel).with_timeout(Duration::from_secs(5))));
                    match confirm(tp, which, &m.goal, &params, &call, &others, runner.as_ref()) {
                        Ok(values) => {
                            r.status = Status::Refuted;
                            r.counterexample = (!values.is_empty()).then(|| serde_json::to_value(Values(values)).ok()).flatten();
                        }
                        Err(why) => {
                            let approx: Vec<&str> = approx.iter().map(String::as_str).collect();
                            let ignores = if approx.is_empty() { String::new() } else { format!(" (the prover approximates {})", approx.join(", ")) };
                            r.reason = Some(format!("{why}{ignores}"));
                        }
                    }
                }
            }
            out.push(r);
        }
    }
    out
}

/// Stack of the thread each obligation is encoded and solved on: the encoding and Z3 recurse along
/// the nesting of expressions and inlined contracts, whatever thread `prove` is called from.
const PROVER_STACK: usize = 256 << 20;

/// Runs `f` with a fresh Z3 context on its own large-stack thread; None if Z3 (or the encoding)
/// panicked or the thread could not be started.
fn in_context<R: Send + Sync>(opts: &Options, f: impl FnOnce() -> R + Send + Sync) -> Option<R> {
    let timeout = opts.timeout_ms as u64;
    std::thread::scope(|scope| {
        let run = move || {
            let mut cfg = Config::new();
            cfg.set_timeout_msec(timeout);
            catch_unwind(AssertUnwindSafe(|| z3::with_z3_config(&cfg, f))).ok()
        };
        let worker = std::thread::Builder::new().name("qal-prove".into()).stack_size(PROVER_STACK).spawn_scoped(scope, run);
        worker.ok()?.join().ok().flatten()
    })
}

/// Solves obligation `k` of `which` (inside a fresh context).
fn solve(tp: &TProgram, which: Which, k: usize, opts: &Options) -> Solved {
    let Some(vc) = encode(tp, which) else { return Solved::Unknown(INTERNAL.into()) };
    let Some(o) = vc.obligations.get(k) else { return Solved::Unknown(INTERNAL.into()) };
    let s = Solver::new();
    let mut p = Params::new();
    p.set_u32("rlimit", opts.rlimit);
    p.set_u32("timeout", opts.timeout_ms);
    p.set_u32("random_seed", 0);
    s.set_params(&p);
    s.assert(&vc.pre);
    s.assert(&o.hyp);
    s.assert(o.holds.not());
    match s.check() {
        SatResult::Unsat => Solved::Proven,
        SatResult::Unknown => match s.get_reason_unknown() {
            Some(r) if r.contains("timeout") => Solved::Unknown(format!("{TIMED_OUT} {} ms", opts.timeout_ms)),
            r => Solved::Unknown(format!("the solver gave up within its resource limit (rlimit {}): {}", opts.rlimit, r.unwrap_or_else(|| "unknown".into()))),
        },
        SatResult::Sat => {
            let Some(model) = s.get_model() else { return Solved::Unknown("the solver found no model".into()) };
            let eval = |t: &Option<Term>| match t {
                Some(Term::I(x)) => model.eval(x, true).and_then(|v| v.as_i64()).map(Value::Int),
                Some(Term::B(b)) => model.eval(b, true).and_then(|v| v.as_bool()).map(Value::Bool),
                None => None,
            };
            let call = o.args.iter().map(|(n, t)| (n.clone(), eval(t))).collect();
            let others = vc.obligations.iter().enumerate().filter(|(j, p)| *j != k && p.goal == o.goal).map(|(_, p)| p.args.iter().map(|(n, t)| (n.clone(), eval(t))).collect()).collect();
            let params = vc
                .params
                .iter()
                .map(|(n, ty, t)| {
                    let v = match t {
                        Some(Term::I(x)) => model.eval(x, true).and_then(|v| v.as_i64()).map(Value::Int),
                        Some(Term::B(b)) => model.eval(b, true).and_then(|v| v.as_bool()).map(Value::Bool),
                        None => None,
                    };
                    (n.clone(), ty.clone(), v)
                })
                .collect();
            Solved::Model(params, call, others)
        }
    }
}

/// Runs the definition on the counterexample (unmodelled parameters get their smallest value):
/// Ok(arguments) if the expected contract really fails.
#[allow(clippy::too_many_arguments)]
fn confirm(tp: &TProgram, which: Which, goal: &Goal, params: &[(String, Type, Option<Value>)], call: &[(String, Option<Value>)], others: &[Vec<(String, Option<Value>)>], runner: Option<&Runner>) -> Result<Vec<(String, Value)>, String> {
    let (export, f): (String, Option<&TFn>) = match which {
        Which::Fn(i) => {
            let f = tp.fns.get(i).ok_or("internal prover error")?;
            (f.def.name.clone(), Some(f))
        }
        Which::Generic(_) => return Err("a counterexample candidate was found, but generic functions are not executed to confirm it".into()),
        Which::Test(i) => {
            let t: &TTest = tp.tests.get(i).ok_or("internal prover error")?;
            (format!("test:{}", t.name), None)
        }
    };
    if f.is_some_and(|f| !f.def.effects.is_empty()) {
        return Err("a counterexample candidate was found, but the function has effects, so it was not executed".into());
    }
    let mut args = Vec::new();
    for (n, ty, v) in params {
        match v.clone().or_else(|| smallest(tp, ty)) {
            Some(v) => args.push((n.clone(), v)),
            None => return Err(format!("a counterexample candidate was found, but parameter `{n}` has no value of at most {} nodes to run it with", gen::MAX_VALUE_NODES)),
        }
    }
    let runner = runner.ok_or("a counterexample candidate was found, but the program could not be compiled to run it")?;
    let values: Vec<Value> = args.iter().map(|(_, v)| v.clone()).collect();
    let (def, label) = match goal {
        Goal::Ens(k) => (f.map_or(export.as_str(), |f| f.def.name.as_str()), format!("ens[{k}]")),
        Goal::Req(g, k) => (g.as_str(), format!("req[{k}]")),
    };
    match runner.call(&export, &values) {
        // A callee's `req` must fail with this call's arguments (as far as the solver modelled them):
        // the same contract failing at another call of the callee is not this obligation.
        Err(e) if violates(&e, def, &label) && matches(&e, call) && !others.iter().any(|o| matches(&e, o)) => Ok(args),
        Err(e) if violates(&e, def, &label) => Err("a counterexample candidate was not confirmed: running it violated the same contract, maybe at another call".into()),
        Err(e) => Err(format!("a counterexample candidate was not confirmed: running it ended with `{}` instead", e.kind)),
        Ok(_) => Err("a counterexample candidate was not confirmed: running it satisfied the contract".into()),
    }
}

/// True if the values of a contract violation agree with a call's modelled arguments.
fn matches(e: &RunError, call: &[(String, Option<Value>)]) -> bool {
    call.iter().all(|(n, v)| v.as_ref().is_none_or(|v| e.values.0.iter().any(|(m, w)| m == n && w == v)))
}

/// True if `e` is the failure of contract `label` (`req[0]`, `ens[1]`) of `def`: the contract was
/// false, or evaluating it trapped.
fn violates(e: &RunError, def: &str, label: &str) -> bool {
    e.def == def && (e.contract.as_deref().is_some_and(|c| c.starts_with(&format!("{label}:"))) || e.path == label || e.path.starts_with(&format!("{label}.")))
}

#[cfg(test)]
mod tests;
