//! QAL static contract proofs and generated property tests (spec Phase 2).
//!
//! **Proofs** (`prove`): for every function (generic ones once, not their instances) and test, Z3
//! checks each obligation:
//! - `ens[k]`: if the `req` hold and the body returns, `ens[k]` evaluates to true without trapping;
//! - at each call of a function with `req` (in a body, also inside closures, not in contracts):
//!   the callee's `req[k]` holds whenever the call is reached.
//!
//! The result is `proven`, `refuted` (the solver found a counterexample and running the function on
//! it really violates the contract at this call) or `unknown` (the solver gave up, or its
//! counterexample could not be confirmed: the encoding approximates what it does not model, see
//! [`vc`]). Every contract stays checked at runtime, proven or not.
//!
//! **Determinism:** a definition is encoded once into a fresh Z3 context and one solver, which
//! holds every obligation behind its own guard; the obligations are checked in order, each under
//! the assumption of its guard alone, with a fixed random seed and a resource limit (`rlimit`,
//! counted per check and deterministic for a given Z3 version and order); the wall-clock timeout
//! is only a safety net. Results are a function of the definition, its callees' contracts,
//! the prover version, the Z3 version and the resource limit — [`Options::key`] names all of them.
//!
//! **Names:** a definition's hash does not depend on its name or the names of its parameters and
//! callees, so what is stored per hash is a name-free [`Record`]; [`render`] turns it into
//! [`Report`]s with the names of the current program.
//!
//! **Property tests** (`props`): generated inputs that satisfy `req`, checked for contract violations.

mod gen;
pub mod props;
mod vc;

pub use gen::{fnv, generatable, random, shrink, smallest, Rng};
pub use props::{check_properties, PropOptions, PropReport, PropStatus};

use qal_core::ast::Type;
use qal_core::diag::Diagnostic;
use qal_core::tast::{TExpr, TKind, TProgram};
use qal_run::{RunError, Runner, Value};
use qal_compiler::Entry;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::{Duration, Instant};
use vc::{Goal, Term, Vc};
use z3::ast::Bool;
use z3::{Config, Params, SatResult, Solver};

/// Changes whenever the encoding or the stored form changes (stored results of another version are
/// not reused).
pub const PROVER_VERSION: &str = "qal-prove/2";

/// Reasons of results that depend on the machine's speed or load, not only on the definition.
const NOT_ATTEMPTED: &str = "not attempted:";
const TIMED_OUT: &str = "the solver timed out after";
const INTERNAL: &str = "internal prover error";
const TIMED_OUT_RUN: &str = "a counterexample candidate was not confirmed: running it hit the wall-clock limit";

/// Longest time a counterexample candidate is run to confirm it.
const CONFIRM_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// Z3 resource limit per obligation (deterministic effort bound).
    pub rlimit: u32,
    /// Wall-clock limit per obligation in milliseconds (safety net).
    pub timeout_ms: u32,
    /// Wall-clock budget of one `prove` call (encoding, solving and confirming); obligations after
    /// it are `unknown`.
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
        self.reason.as_deref().is_none_or(|r| !(r.starts_with(NOT_ATTEMPTED) || r.starts_with(TIMED_OUT) || r.starts_with(INTERNAL) || r.starts_with(TIMED_OUT_RUN)))
    }

    /// `refuted` is the error `contract_refuted` (E0801); `unknown` the warning `contract_unproven` (E0802).
    pub fn diagnostic(&self) -> Option<Diagnostic> {
        let mut d = match self.status {
            Status::Proven => return None,
            Status::Refuted if self.counterexample.is_some() => Diagnostic::new("E0801", "contract_refuted", &self.def, &self.path, format!("`{}` ({}) does not hold for the counterexample", self.contract, self.obligation))
                .hint("running the function on the counterexample violates the contract: fix the body, the contract, or add a `req`"),
            // A test, or a function without parameters: nothing to show but the run itself.
            Status::Refuted => Diagnostic::new("E0801", "contract_refuted", &self.def, &self.path, format!("`{}` ({}) does not hold: running `{}` violates it", self.contract, self.obligation, self.def))
                .hint("running it violates the contract: fix the code or the contract"),
            Status::Unknown => Diagnostic::new("E0802", "contract_unproven", &self.def, &self.path, format!("`{}` ({}) could not be proven: {}", self.contract, self.obligation, self.reason.as_deref().unwrap_or("unknown")))
                .hint("the contract stays checked at runtime; a simpler or stronger `req`/`ens` (Int and Bool facts) may make it provable"),
        };
        d.counterexample = self.counterexample.clone();
        Some(d)
    }
}

/// Why an obligation is `unknown`, without names (they are rendered from the current program).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Why {
    /// The solver's own reason.
    Solver(String),
    /// The wall-clock limit (ms) was hit: not stored.
    TimedOut(u32),
    /// Why a candidate was not confirmed; the approximations of the encoding follow.
    Unconfirmed(String),
    /// Parameter `k` has no value small enough to run the candidate with; approximations follow.
    NoValue(usize),
    /// Not stored.
    Internal,
    /// Running the candidate hit its wall-clock limit (depends on the machine): not stored.
    RunTimedOut,
}

impl Why {
    fn is_stable(&self) -> bool {
        !matches!(self, Why::TimedOut(_) | Why::Internal | Why::RunTimedOut)
    }
}

/// The name-free result of one obligation (the stored form of a [`Report`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Outcome {
    status: Status,
    /// Counterexample values by parameter position.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cex: Vec<(usize, serde_json::Value)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    why: Option<Why>,
}

impl Outcome {
    fn new(status: Status) -> Outcome {
        Outcome { status, cex: Vec::new(), why: None }
    }
}

/// The proof results of one definition as stored per hash: one outcome per obligation, in the
/// order the encoding produces them, and no names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// The definition was too large to encode (no outcomes).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    too_large: bool,
    results: Vec<Outcome>,
}

/// The proof of one definition.
#[derive(Debug, Clone, PartialEq)]
pub struct Proof {
    pub def: String,
    pub reports: Vec<Report>,
    /// What to store for the definition's hash; None if a result depends on the machine (time
    /// budget, wall-clock timeout, internal error).
    pub record: Option<Record>,
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

fn encode(tp: &TProgram, which: Which, deadline: Instant) -> Option<Vc> {
    Some(match which {
        Which::Fn(i) => vc::function(tp, tp.fns.get(i)?, Some(deadline)),
        Which::Generic(i) => vc::function(tp, tp.generics.get(i)?, Some(deadline)),
        Which::Test(i) => vc::test(tp, tp.tests.get(i)?, Some(deadline)),
    })
}

/// What the solver said about one obligation (plain data: Z3 terms do not leave their context).
enum Solved {
    Proven,
    /// A model: parameter values (None: not modelled by the solver), and for a callee's `req` the
    /// callee's arguments at this call.
    /// The arguments at the other calls with the same goal follow (in the same model).
    Model(Vec<(String, Type, Option<Value>)>, Vec<(String, Option<Value>)>, Vec<Vec<(String, Option<Value>)>>),
    Unknown(Why),
}

/// Metadata of an obligation, outside any Z3 context.
struct Meta {
    goal: Goal,
    label: String,
    path: String,
    contract: String,
}

/// Everything about a definition's encoding that is shown with its results, outside any Z3 context.
struct Shape {
    params: Vec<String>,
    metas: Vec<Meta>,
    approx: BTreeSet<String>,
    too_large: bool,
    out_of_time: bool,
}

impl Shape {
    fn of(vc: &Vc) -> Shape {
        Shape {
            params: vc.params.iter().map(|(n, _, _)| n.clone()).collect(),
            metas: vc.obligations.iter().map(|o| Meta { goal: o.goal.clone(), label: o.label.clone(), path: o.path.clone(), contract: o.contract.clone() }).collect(),
            approx: vc.approx.clone(),
            too_large: vc.truncated,
            out_of_time: vc.out_of_time,
        }
    }

    fn reason(&self, why: &Why) -> String {
        let ignores = || {
            let approx: Vec<&str> = self.approx.iter().map(String::as_str).collect();
            if approx.is_empty() {
                String::new()
            } else {
                format!(" (the prover approximates {})", approx.join(", "))
            }
        };
        match why {
            Why::Solver(t) => t.clone(),
            Why::TimedOut(ms) => format!("{TIMED_OUT} {ms} ms"),
            Why::Unconfirmed(t) => format!("{t}{}", ignores()),
            Why::NoValue(k) => {
                let n = self.params.get(*k).map(String::as_str).unwrap_or("?");
                format!("a counterexample candidate was found, but parameter `{n}` has no value of at most {} nodes to run it with{}", gen::MAX_VALUE_NODES, ignores())
            }
            Why::Internal => INTERNAL.to_string(),
            Why::RunTimedOut => format!("{TIMED_OUT_RUN}{}", ignores()),
        }
    }

    fn counterexample(&self, cex: &[(usize, serde_json::Value)]) -> Option<serde_json::Value> {
        let mut m = serde_json::Map::new();
        for (k, v) in cex {
            if let Some(n) = self.params.get(*k) {
                m.insert(n.clone(), v.clone());
            }
        }
        (!m.is_empty()).then_some(serde_json::Value::Object(m))
    }

    /// The reports of a record, with the names of this encoding.
    fn reports(&self, name: &str, rec: &Record) -> Vec<Report> {
        if rec.too_large {
            let reason = format!("{name} and the contracts it uses are too large to encode (more than {} expressions); its contracts stay checked at run time", vc::MAX_VC_NODES);
            return vec![summary(name, "all", reason)];
        }
        self.metas
            .iter()
            .zip(&rec.results)
            .map(|(m, o)| Report {
                def: name.to_string(),
                obligation: m.label.clone(),
                path: m.path.clone(),
                contract: m.contract.clone(),
                status: o.status,
                counterexample: self.counterexample(&o.cex),
                reason: o.why.as_ref().map(|w| self.reason(w)),
            })
            .collect()
    }
}

/// One `unknown` report standing for several obligations of `name`.
fn summary(name: &str, obligation: &str, reason: String) -> Report {
    Report { def: name.to_string(), obligation: obligation.to_string(), path: String::new(), contract: String::new(), status: Status::Unknown, counterexample: None, reason: Some(reason) }
}

fn not_attempted(opts: &Options) -> String {
    format!("{NOT_ATTEMPTED} the prover's time budget of {} s was used up", opts.budget.as_secs())
}

/// Proves the obligations of the definitions named `only` (all if None) of a checked program.
pub fn prove(tp: &TProgram, only: Option<&[String]>, opts: &Options) -> Vec<Report> {
    prove_defs(tp, only, opts).into_iter().flat_map(|p| p.reports).collect()
}

/// Like [`prove`], per definition, with the records to store. Everything (encoding, solving and
/// confirming) stays within `opts.budget` plus at most one obligation's limits; once it is used up,
/// each remaining definition gets one summary report instead of one per obligation.
pub fn prove_defs(tp: &TProgram, only: Option<&[String]>, opts: &Options) -> Vec<Proof> {
    let start = Instant::now();
    let mut runner: Option<Option<Runner>> = None;
    targets(tp, only).into_iter().map(|(name, which)| prove_one(tp, &name, which, start, opts, &mut runner)).collect()
}

fn prove_one(tp: &TProgram, name: &str, which: Which, start: Instant, opts: &Options, runner: &mut Option<Option<Runner>>) -> Proof {
    let unstable = |reports: Vec<Report>| Proof { def: name.to_string(), reports, record: None };
    if start.elapsed() > opts.budget {
        return unstable(vec![summary(name, "all", not_attempted(opts))]);
    }
    let deadline = start + opts.budget;
    // One context per definition: encoded once, each obligation checked (and a counterexample
    // candidate confirmed) right away, so an early refutation is never lost to a later timeout.
    let solved = in_context(opts, || {
        let vc = encode(tp, which, deadline)?;
        let shape = Shape::of(&vc);
        let mut out = Vec::new();
        if !(shape.too_large || shape.out_of_time) {
            // Every obligation is asserted once, behind its own guard; each check assumes exactly
            // one guard (so the formula is internalized once, not once per obligation).
            let s = Solver::new();
            s.assert(&vc.pre);
            for d in &vc.defs {
                s.assert(d);
            }
            let guards: Vec<Bool> = vc.obligations.iter().map(|_| Bool::fresh_const("goal")).collect();
            for (g, o) in guards.iter().zip(&vc.obligations) {
                s.assert(g.implies(vc::and(&[o.hyp.clone(), o.holds.not()])));
            }
            for (k, m) in shape.metas.iter().enumerate() {
                if start.elapsed() > opts.budget {
                    break;
                }
                // The wall-clock limit of one obligation never reaches past the budget.
                let left = opts.budget.saturating_sub(start.elapsed()).as_millis().max(1) as u32;
                let only: Vec<Bool> = guards.iter().enumerate().map(|(j, g)| if j == k { g.clone() } else { g.not() }).collect();
                let o = match solve(&s, &only, &vc, k, opts.rlimit, opts.timeout_ms.min(left)) {
                    Solved::Proven => Outcome::new(Status::Proven),
                    Solved::Unknown(why) => Outcome { why: Some(why), ..Outcome::new(Status::Unknown) },
                    Solved::Model(params, call, others) => {
                        let left = opts.budget.saturating_sub(start.elapsed());
                        if left.is_zero() {
                            break;
                        }
                        let candidate: Vec<(usize, serde_json::Value)> = params.iter().enumerate().filter_map(|(k, (_, _, v))| v.as_ref().and_then(|v| serde_json::to_value(v).ok()).map(|j| (k, j))).collect();
                        let slot = runner.get_or_insert_with(|| qal_compiler::compile(tp, &Entry::All).ok().and_then(|m| Runner::new(m).ok()).map(|r| r.with_fuel(opts.confirm_fuel)));
                        // Running the candidate never reaches past the budget either.
                        *slot = slot.take().map(|r| r.with_timeout(CONFIRM_TIMEOUT.min(left)));
                        match confirm(tp, which, &m.goal, &params, &call, &others, slot.as_ref()) {
                            Ok(values) => Outcome { cex: values.iter().enumerate().filter_map(|(k, v)| serde_json::to_value(v).ok().map(|j| (k, j))).collect(), ..Outcome::new(Status::Refuted) },
                            Err(why) => Outcome { status: Status::Unknown, cex: candidate, why: Some(why) },
                        }
                    }
                };
                out.push(o);
            }
        }
        Some((shape, out))
    })
    .flatten();
    let Some((shape, results)) = solved else {
        return unstable(vec![summary(name, "all", INTERNAL.to_string())]);
    };
    if shape.out_of_time {
        return unstable(vec![summary(name, "all", not_attempted(opts))]);
    }
    if shape.too_large {
        let rec = Record { too_large: true, results: Vec::new() };
        return Proof { def: name.to_string(), reports: shape.reports(name, &rec), record: Some(rec) };
    }
    let left_out = shape.metas.len() - results.len().min(shape.metas.len());
    let stable = left_out == 0 && results.iter().all(|o| o.why.as_ref().is_none_or(Why::is_stable));
    let rec = Record { too_large: false, results };
    let mut reports = shape.reports(name, &rec);
    if left_out > 0 {
        let obligation = if left_out == shape.metas.len() { "all" } else { "rest" };
        reports.push(summary(name, obligation, format!("{} ({left_out} of {} obligations of {name})", not_attempted(opts), shape.metas.len())));
    }
    Proof { def: name.to_string(), reports, record: stable.then_some(rec) }
}

/// The reports of a stored record of definition `name` with the names of the current program;
/// None if the record does not fit the definition's encoding (prove it again then) or the
/// encoding did not finish before `deadline`.
pub fn render(tp: &TProgram, name: &str, rec: &Record, deadline: Instant) -> Option<Vec<Report>> {
    let (_, which) = targets(tp, Some(&[name.to_string()])).into_iter().next()?;
    let shape = in_context(&Options::default(), || encode(tp, which, deadline).map(|vc| Shape::of(&vc))).flatten()?;
    let fits = shape.too_large == rec.too_large && (rec.too_large || shape.metas.len() == rec.results.len());
    (fits && !shape.out_of_time).then(|| shape.reports(name, rec))
}

/// Stack of the thread each definition is encoded and solved on: the encoding and Z3 recurse along
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

/// Solves obligation `k` of `vc` with solver `s` (which holds every obligation behind a guard)
/// under the assumptions `only` (obligation k's guard; the others' negated).
fn solve(s: &Solver, only: &[Bool], vc: &Vc, k: usize, rlimit: u32, timeout_ms: u32) -> Solved {
    let Some(o) = vc.obligations.get(k) else { return Solved::Unknown(Why::Internal) };
    let mut p = Params::new();
    p.set_u32("rlimit", rlimit);
    p.set_u32("timeout", timeout_ms);
    p.set_u32("random_seed", 0);
    s.set_params(&p);
    let started = Instant::now();
    let result = s.check_assumptions(only);
    // Z3 names a cancelled check `canceled` whether the resource limit or the wall clock stopped
    // it; only the clock depends on the machine.
    let clock = started.elapsed() >= Duration::from_millis(timeout_ms as u64);
    match result {
        SatResult::Unsat => Solved::Proven,
        SatResult::Unknown => match s.get_reason_unknown() {
            Some(r) if r.contains("timeout") || clock => Solved::Unknown(Why::TimedOut(timeout_ms)),
            r => Solved::Unknown(Why::Solver(format!("the solver gave up within its resource limit (rlimit {rlimit}): {}", r.unwrap_or_else(|| "unknown".into())))),
        },
        SatResult::Sat => {
            let Some(model) = s.get_model() else { return Solved::Unknown(Why::Solver("the solver found no model".into())) };
            let eval = |t: &Option<Term>| match t {
                Some(Term::I(x)) => model.eval(x, true).and_then(|v| v.as_i64()).map(Value::Int),
                Some(Term::B(b)) => model.eval(b, true).and_then(|v| v.as_bool()).map(Value::Bool),
                None => None,
            };
            let call = o.args.iter().map(|(n, t)| (n.clone(), eval(t))).collect();
            let others = vc.obligations.iter().enumerate().filter(|(j, p)| *j != k && p.goal == o.goal).map(|(_, p)| p.args.iter().map(|(n, t)| (n.clone(), eval(t))).collect()).collect();
            let params = vc.params.iter().map(|(n, ty, t)| (n.clone(), ty.clone(), eval(t))).collect();
            Solved::Model(params, call, others)
        }
    }
}

fn unconfirmed(why: &str) -> Why {
    Why::Unconfirmed(why.to_string())
}

/// Runs the definition on the counterexample (unmodelled parameters get their smallest value):
/// Ok(arguments) if the expected contract really fails, at this call for a callee's `req`.
#[allow(clippy::too_many_arguments)]
fn confirm(tp: &TProgram, which: Which, goal: &Goal, params: &[(String, Type, Option<Value>)], call: &[(String, Option<Value>)], others: &[Vec<(String, Option<Value>)>], runner: Option<&Runner>) -> Result<Vec<Value>, Why> {
    let (export, f) = match which {
        Which::Fn(i) => {
            let f = tp.fns.get(i).ok_or(Why::Internal)?;
            (f.def.name.clone(), Some(f))
        }
        Which::Generic(_) => return Err(unconfirmed("a counterexample candidate was found, but generic functions are not executed to confirm it")),
        Which::Test(i) => (format!("test:{}", tp.tests.get(i).ok_or(Why::Internal)?.name), None),
    };
    if f.is_some_and(|f| !f.def.effects.is_empty()) {
        return Err(unconfirmed("a counterexample candidate was found, but the function has effects, so it was not executed"));
    }
    let mut values = Vec::new();
    for (k, (_, ty, v)) in params.iter().enumerate() {
        match v.clone().or_else(|| smallest(tp, ty)) {
            Some(v) => values.push(v),
            None => return Err(Why::NoValue(k)),
        }
    }
    let runner = runner.ok_or_else(|| unconfirmed("a counterexample candidate was found, but the program could not be compiled to run it"))?;
    let (def, label) = match goal {
        Goal::Ens(k) => (f.map_or(export.as_str(), |f| f.def.name.as_str()), format!("ens[{k}]")),
        Goal::Req(g, k) => (g.as_str(), format!("req[{k}]")),
    };
    // A callee's `req` failing is this obligation only if nothing but this definition's own calls
    // of the callee can raise it (review finding: a violation raised in another function was
    // blamed on this call).
    let elsewhere = match goal {
        Goal::Req(g, _) => tp.fn_index(g).is_none_or(|c| reaches_elsewhere(tp, which, c)),
        Goal::Ens(_) => false,
    };
    match runner.call(&export, &values) {
        Err(e) if violates(&e, def, &label) && elsewhere => {
            Err(unconfirmed("a counterexample candidate was not confirmed: running it violated the same contract, but the callee is also called by other functions this one runs, so the violation may come from another call"))
        }
        // A callee's `req` must fail with this call's arguments (as far as the solver modelled them):
        // the same contract failing at another call of the callee is not this obligation.
        Err(e) if violates(&e, def, &label) && matches(&e, call) && !others.iter().any(|o| matches(&e, o)) => Ok(values),
        Err(e) if violates(&e, def, &label) => Err(unconfirmed("a counterexample candidate was not confirmed: running it violated the same contract, maybe at another call")),
        // The instruction budget is deterministic, the wall clock is not.
        Err(e) if e.kind == "timeout" && e.message.starts_with("wall-clock") => Err(Why::RunTimedOut),
        Err(e) => Err(Why::Unconfirmed(format!("a counterexample candidate was not confirmed: running it ended with `{}` instead", e.kind))),
        Ok(_) => Err(unconfirmed("a counterexample candidate was not confirmed: running it satisfied the contract")),
    }
}

/// True if running `which` can call `fns[callee]` other than through a direct call in its own
/// body (the calls that have obligations): its own contracts or some other function it can run
/// (through calls, function values or contracts, the callee included) call or refer to the callee,
/// or its own body uses the callee as a value.
fn reaches_elsewhere(tp: &TProgram, which: Which, callee: usize) -> bool {
    let (own, roots): (Option<usize>, Vec<&TExpr>) = match which {
        Which::Fn(i) => match tp.fns.get(i) {
            // Calls in its own `req`/`ens` have no obligation (review finding N1): like other code.
            Some(f) if f.req.iter().chain(&f.ens).any(|c| mentions(c, callee)) => return true,
            Some(f) => (Some(i), f.req.iter().chain(&f.ens).chain([&f.body]).collect()),
            None => return true,
        },
        Which::Test(i) => match tp.tests.get(i) {
            Some(t) => (None, vec![&t.body]),
            None => return true,
        },
        Which::Generic(_) => return true,
    };
    let mut seen: HashSet<usize> = own.into_iter().collect();
    let mut queue = Vec::new();
    // Its own code: direct calls of the callee are this obligation's calls; everything it calls runs.
    let mut stack = roots;
    while let Some(e) = stack.pop() {
        match &e.kind {
            TKind::FnRef(c) if *c == callee => return true,
            TKind::Call(c, _) | TKind::FnRef(c) if seen.insert(*c) => queue.push(*c),
            _ => {}
        }
        stack.extend(props::children(e));
    }
    while let Some(j) = queue.pop() {
        let Some(f) = tp.fns.get(j) else { return true };
        let mut stack: Vec<&TExpr> = f.req.iter().chain(&f.ens).chain([&f.body]).collect();
        while let Some(e) = stack.pop() {
            match &e.kind {
                TKind::Call(c, _) | TKind::FnRef(c) if *c == callee => return true,
                TKind::Call(c, _) | TKind::FnRef(c) if seen.insert(*c) => queue.push(*c),
                _ => {}
            }
            stack.extend(props::children(e));
        }
    }
    false
}

/// True if `e` calls or refers to `fns[callee]` directly (closure bodies included).
fn mentions(e: &TExpr, callee: usize) -> bool {
    let mut stack = vec![e];
    while let Some(e) = stack.pop() {
        if matches!(&e.kind, TKind::Call(c, _) | TKind::FnRef(c) if *c == callee) {
            return true;
        }
        stack.extend(props::children(e));
    }
    false
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
