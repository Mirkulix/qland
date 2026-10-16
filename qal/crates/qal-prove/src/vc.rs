//! Verification conditions: typed QAL expressions -> Z3 formulas.
//!
//! Every expression is encoded as a value (a Z3 `Int` for `Int`, a `Bool` for `Bool`, nothing for
//! other types) and a condition `ok` that every evaluation of the expression which does not trap
//! satisfies. Values are exact mathematical integers: QAL traps on every Int overflow, so a value
//! that exists at runtime equals its mathematical value and lies in the 64-bit range. An arithmetic
//! result therefore contributes `in_range(result)` to `ok` (never a wrapped value); division and
//! remainder truncate toward zero like `i64.div_s` / `i64.rem_s`, `x / 0` and `MIN / -1` trap,
//! `MIN % -1` is 0.
//!
//! **Soundness rule:** a construct the encoding does not model (Float, Str, lists, records, enums,
//! maps, closures and calls through function values, field access, most built-ins, payloads bound
//! by non-literal patterns) becomes a fresh, unconstrained value, and its `ok` only states facts that
//! hold for every real evaluation (e.g. `0 <= len(s) <= 2^32 - 1`). Fresh values are universally
//! quantified in every check, so the solver considers at least every real execution: a proof stays
//! a proof, and only the precision suffers (such contracts are `unknown`).
//!
//! A call `g(args)` is abstracted by g's contracts: the caller must establish g's `req` (an
//! obligation at the call site, under the path condition), and may assume g's `req` and `ens` for
//! the result after the call returns. This is sound because every contract stays checked at runtime
//! (Phase 2 removes no check): a call that returns has passed g's `req` and `ens` checks. For a
//! recursive call it is the induction hypothesis on the depth of the call tree (partial correctness:
//! termination is not proven).

use qal_core::ast::{BinOp, Type, UnOp};
use qal_core::tast::{Builtin, TArm, TExpr, TFn, TKind, TPat, TProgram, TTest};
use std::collections::BTreeSet;
use std::time::Instant;
use z3::ast::{Bool, Int};

/// Callee contracts are encoded this many calls deep (contracts calling functions with contracts);
/// deeper calls are fresh values.
const MAX_INLINE: usize = 3;

/// Expressions encoded per function or test, inlined contracts included. Beyond it the encoding
/// stops (`Vc::truncated`) and the function's obligations are `unknown`: contracts that call
/// functions with large contracts multiply (k calls at 3 levels are k³ encodings).
pub const MAX_VC_NODES: usize = 50_000;

#[derive(Clone, Debug)]
pub enum Term {
    I(Int),
    B(Bool),
}

impl Term {
    fn int(&self) -> Option<&Int> {
        match self {
            Term::I(i) => Some(i),
            Term::B(_) => None,
        }
    }

    fn bool(&self) -> Option<&Bool> {
        match self {
            Term::B(b) => Some(b),
            Term::I(_) => None,
        }
    }
}

/// What must hold: an `ens` of the function, or a callee's `req` at a call site.
#[derive(Debug, Clone, PartialEq)]
pub enum Goal {
    Ens(usize),
    /// (callee as the runtime names it, index of its `req`)
    Req(String, usize),
}

/// One proof obligation: whenever `pre` and `hyp` hold, `goal` must hold.
pub struct Obligation {
    pub goal: Goal,
    /// `ens[0]`, or `req[1] of g`.
    pub label: String,
    /// AST path of the contract (`ens[0]`) or of the call (`body.then`).
    pub path: String,
    /// The contract as text (`ens r >= x`).
    pub contract: String,
    pub hyp: Bool,
    pub holds: Bool,
    /// For a callee's `req`: the callee's parameter names and the encoded arguments of this call
    /// (to tell a confirmed violation at this call from one at another call of the same callee).
    pub args: Vec<(String, Option<Term>)>,
}

/// The verification conditions of one function or test.
pub struct Vc {
    /// Parameters in range and the function's `req` (true for a test).
    pub pre: Bool,
    /// Parameter names, types and their Z3 constants (None: not modelled).
    pub params: Vec<(String, Type, Option<Term>)>,
    pub obligations: Vec<Obligation>,
    /// Constructs that were approximated (become part of the reason for `unknown`).
    pub approx: BTreeSet<String>,
    /// The encoding hit `MAX_VC_NODES`; the obligations are incomplete and must not be used.
    pub truncated: bool,
    /// The encoding passed its deadline; the obligations are incomplete and must not be used.
    pub out_of_time: bool,
    /// Definitions `p == c` of fresh constants that name shared conditions (true in every check).
    pub defs: Vec<Bool>,
}

/// The conjunction of `xs`, leaving out literal `true`s (most `ok` conditions are `true`, and a
/// conjunction that grows by one condition per item must stay linear in the number of items).
pub fn and(xs: &[Bool]) -> Bool {
    let kept: Vec<&Bool> = xs.iter().filter(|x| x.as_bool() != Some(true)).collect();
    match kept.as_slice() {
        [] => Bool::from_bool(true),
        [x] => (*x).clone(),
        _ => Bool::and(&kept),
    }
}

/// `acc && x` as one new node (a running conjunction: linear, where collecting a growing prefix
/// into one `and` per item would be quadratic).
fn conj(acc: &Bool, x: &Bool) -> Bool {
    and(&[acc.clone(), x.clone()])
}

pub fn in_range(x: &Int) -> Bool {
    Bool::and(&[x.ge(Int::from_i64(i64::MIN)), x.le(Int::from_i64(i64::MAX))])
}

fn truth() -> Bool {
    Bool::from_bool(true)
}

/// `a / b` truncated toward zero (Z3's `div` is Euclidean; for `a >= 0` both agree for either sign of `b`).
fn tdiv(a: &Int, b: &Int) -> Int {
    let zero = Int::from_i64(0);
    a.ge(&zero).ite(&a.div(b), &a.unary_minus().div(b).unary_minus())
}

/// Facts about the Int result `x` of a Phase 5a built-in that hold for every evaluation that
/// returns (two's complement on 64 bits; `x` itself is fresh and in range).
pub(crate) fn bit_facts(b: Builtin, x: &Int, vals: &[Option<Term>]) -> Vec<Bool> {
    let int = |k: usize| match vals.get(k) {
        Some(Some(Term::I(i))) => Some(i.clone()),
        _ => None,
    };
    let zero = Int::from_i64(0);
    let nonneg = |t: &Int| t.ge(&zero);
    match b {
        Builtin::BytesAt => vec![x.ge(&zero), x.le(Int::from_i64(255))],
        // ~a = -a - 1 exactly.
        Builtin::BitNot => int(0).map(|a| vec![x.eq(Int::sub(&[a.unary_minus(), Int::from_i64(1)]))]).unwrap_or_default(),
        Builtin::BitAnd | Builtin::BitOr | Builtin::BitXor => {
            let (Some(a), Some(c)) = (int(0), int(1)) else { return Vec::new() };
            match b {
                // a & c has a sign bit only if both do; it keeps no bit that a (or c) lacks.
                Builtin::BitAnd => vec![
                    Bool::or(&[nonneg(&a), nonneg(&c)]).implies(nonneg(x)),
                    nonneg(&a).implies(x.le(&a)),
                    nonneg(&c).implies(x.le(&c)),
                    Bool::and(&[a.lt(&zero), c.lt(&zero)]).implies(x.lt(&zero)),
                ],
                // a | c has every bit of a and of c.
                Builtin::BitOr => vec![
                    Bool::and(&[nonneg(&a), nonneg(&c)]).implies(Bool::and(&[x.ge(&a), x.ge(&c)])),
                    Bool::or(&[a.lt(&zero), c.lt(&zero)]).implies(x.lt(&zero)),
                ],
                _ => vec![nonneg(x).eq(nonneg(&a).eq(nonneg(&c)))],
            }
        }
        // Arithmetic shift keeps the sign and never grows the magnitude; n = 0 is the identity.
        Builtin::Shr => {
            let (Some(a), Some(n)) = (int(0), int(1)) else { return Vec::new() };
            vec![
                nonneg(&a).implies(Bool::and(&[nonneg(x), x.le(&a)])),
                a.lt(&zero).implies(Bool::and(&[x.ge(&a), x.lt(&zero)])),
                n.eq(&zero).implies(x.eq(&a)),
            ]
        }
        // A logical shift by at least one clears the sign bit.
        Builtin::Ushr => {
            let (Some(a), Some(n)) = (int(0), int(1)) else { return Vec::new() };
            vec![n.gt(&zero).implies(nonneg(x)), n.eq(&zero).implies(x.eq(&a)), nonneg(&a).implies(x.le(&a))]
        }
        Builtin::Shl => {
            let (Some(a), Some(n)) = (int(0), int(1)) else { return Vec::new() };
            vec![n.eq(&zero).implies(x.eq(&a))]
        }
        _ => Vec::new(),
    }
}

/// A fresh value of a modelled type and the facts every value of the type satisfies.
fn havoc(ty: &Type) -> (Option<Term>, Bool) {
    match ty {
        Type::Int => {
            let x = Int::fresh_const("v");
            let r = in_range(&x);
            (Some(Term::I(x)), r)
        }
        Type::Bool => (Some(Term::B(Bool::fresh_const("b"))), truth()),
        _ => (None, truth()),
    }
}

struct Enc<'p> {
    tp: &'p TProgram,
    /// Collect call-site obligations (only while encoding the body being proven, not contracts).
    record: bool,
    depth: usize,
    /// Local types of the definition whose expressions are being encoded (an inlined callee last).
    locals: Vec<&'p [Type]>,
    obligations: Vec<Obligation>,
    approx: BTreeSet<String>,
    nodes: usize,
    /// The encoding stops (`Vc::out_of_time`) once this has passed.
    deadline: Option<Instant>,
    out_of_time: bool,
    defs: Vec<Bool>,
}

/// The clock is read once per this many encoded expressions.
const CLOCK_EVERY: usize = 256;

/// The verification conditions of a function (monomorphic or generic).
pub fn function<'p>(tp: &'p TProgram, f: &'p TFn, deadline: Option<Instant>) -> Vc {
    let mut enc = Enc { tp, record: false, depth: 0, locals: vec![&f.locals], obligations: Vec::new(), approx: BTreeSet::new(), nodes: 0, deadline, out_of_time: false, defs: Vec::new() };
    let mut env: Vec<Option<Term>> = vec![None; f.locals.len()];
    let mut facts = Vec::new();
    let mut params = Vec::new();
    for (k, p) in f.def.params.iter().enumerate() {
        let ty = f.locals.get(k).cloned().unwrap_or(Type::Unit);
        let t = match ty {
            Type::Int => {
                let x = Int::new_const(p.name.as_str());
                facts.push(in_range(&x));
                Some(Term::I(x))
            }
            Type::Bool => Some(Term::B(Bool::new_const(p.name.as_str()))),
            _ => None,
        };
        params.push((p.name.clone(), ty, t.clone()));
        if let Some(slot) = env.get_mut(k) {
            *slot = t;
        }
    }
    // The `req` are checked in order: req[k] is evaluated only if the earlier ones held.
    let mut pre = and(&facts);
    for (k, c) in f.req.iter().enumerate() {
        let (v, ok) = enc.enc(c, &mut env, &pre, &format!("req[{k}]"));
        let v = enc.as_bool(v, "a contract over values the prover does not model");
        pre = and(&[pre, ok, v]);
    }
    enc.record = true;
    let (body, body_ok) = enc.enc(&f.body, &mut env, &truth(), "body");
    enc.record = false;
    if let Some(r) = f.result_local {
        if let Some(slot) = env.get_mut(r as usize) {
            *slot = body;
        }
        // Checked in order after the body returned: ens[k] may assume the earlier ones.
        let mut hyp = body_ok;
        for (k, c) in f.ens.iter().enumerate() {
            let (v, ok) = enc.enc(c, &mut env, &hyp, &format!("ens[{k}]"));
            let holds = and(&[ok, enc.as_bool(v, "a contract over values the prover does not model")]);
            let contract = f.def.ens.get(k).map(|e| format!("ens {}", qal_compiler::print_expr(e))).unwrap_or_default();
            enc.obligations.push(Obligation { goal: Goal::Ens(k), label: format!("ens[{k}]"), path: format!("ens[{k}]"), contract, hyp: hyp.clone(), holds: holds.clone(), args: Vec::new() });
            hyp = conj(&hyp, &holds);
        }
    }
    let truncated = enc.nodes > MAX_VC_NODES;
    Vc { pre, params, obligations: enc.obligations, approx: enc.approx, truncated, out_of_time: enc.out_of_time, defs: enc.defs }
}

/// The verification conditions of a test: the `req` of every function it calls.
pub fn test<'p>(tp: &'p TProgram, t: &'p TTest, deadline: Option<Instant>) -> Vc {
    let mut enc = Enc { tp, record: true, depth: 0, locals: vec![&t.locals], obligations: Vec::new(), approx: BTreeSet::new(), nodes: 0, deadline, out_of_time: false, defs: Vec::new() };
    let mut env: Vec<Option<Term>> = vec![None; t.locals.len()];
    enc.enc(&t.body, &mut env, &truth(), "body");
    let truncated = enc.nodes > MAX_VC_NODES;
    Vc { pre: truth(), params: Vec::new(), obligations: enc.obligations, approx: enc.approx, truncated, out_of_time: enc.out_of_time, defs: enc.defs }
}

impl<'p> Enc<'p> {
    /// A fresh constant defined as `c` (a literal stays itself): a condition that many obligations
    /// share, like the path condition after each item of a long list, then stays one node in each
    /// of them, where the solver would otherwise flatten it into every obligation (quadratic).
    fn name(&mut self, c: Bool) -> Bool {
        if c.as_bool().is_some() {
            return c;
        }
        let p = Bool::fresh_const("pc");
        self.defs.push(p.eq(&c));
        p
    }

    /// The Bool of a value (a fresh one, noted as `why`, if it is not modelled).
    fn as_bool(&mut self, v: Option<Term>, why: &str) -> Bool {
        match v {
            Some(Term::B(b)) => b,
            _ => {
                self.approx.insert(why.to_string());
                Bool::fresh_const("u")
            }
        }
    }

    /// Encodes `e` under path condition `pc`: (value, ok). A modelled type always has a value.
    fn enc(&mut self, e: &TExpr, env: &mut Vec<Option<Term>>, pc: &Bool, path: &str) -> (Option<Term>, Bool) {
        self.nodes += 1;
        if self.nodes % CLOCK_EVERY == 0 && self.deadline.is_some_and(|d| Instant::now() > d) {
            self.out_of_time = true;
        }
        if self.nodes > MAX_VC_NODES || self.out_of_time {
            // Too large or too slow: the result is discarded (`Vc::truncated`, `Vc::out_of_time`);
            // stop encoding.
            return havoc(&e.ty);
        }
        let (v, ok) = self.enc_inner(e, env, pc, path);
        match v {
            Some(v) => (Some(v), ok),
            None => {
                let (h, facts) = havoc(&e.ty);
                (h, and(&[ok, facts]))
            }
        }
    }

    /// A fresh value of type `ty` standing for `what` (noted when the type is modelled).
    fn unknown(&mut self, ty: &Type, what: &str) -> (Option<Term>, Bool) {
        if matches!(ty, Type::Int | Type::Bool) {
            self.approx.insert(what.to_string());
        }
        havoc(ty)
    }

    /// Encodes `xs` left to right; each is evaluated only if the previous ones did not trap.
    fn seq(&mut self, xs: &[TExpr], env: &mut Vec<Option<Term>>, pc: &Bool, path: &str, item: &str) -> (Vec<Option<Term>>, Bool) {
        let mut oks = Vec::new();
        let mut vals = Vec::new();
        // Item i is reached under pc and the oks of the items before it (a running conjunction).
        let mut here = pc.clone();
        for (i, x) in xs.iter().enumerate() {
            let (v, ok) = self.enc(x, env, &here, &format!("{path}.{item}[{i}]"));
            here = self.name(conj(&here, &ok));
            vals.push(v);
            oks.push(ok);
        }
        (vals, and(&oks))
    }

    fn enc_inner(&mut self, e: &TExpr, env: &mut Vec<Option<Term>>, pc: &Bool, path: &str) -> (Option<Term>, Bool) {
        match &e.kind {
            TKind::Int(n) => (Some(Term::I(Int::from_i64(*n))), truth()),
            TKind::Bool(b) => (Some(Term::B(Bool::from_bool(*b))), truth()),
            TKind::Float(_) | TKind::Str(_) | TKind::Unit | TKind::FnRef(_) | TKind::FnRefG(..) => (None, truth()),
            TKind::Local(id) => (env.get(*id as usize).cloned().flatten(), truth()),
            TKind::Lambda(l) => {
                // The body runs later, any number of times, with any arguments; everything it can see
                // besides its parameters is immutable and was bound when the closure was created, so
                // obligations inside are checked here under the creation's path condition.
                let mut facts = vec![pc.clone()];
                for p in &l.params {
                    let ty = self.locals.last().and_then(|ls| ls.get(*p as usize)).cloned().unwrap_or(Type::Unit);
                    let (h, f) = havoc(&ty);
                    if let Some(slot) = env.get_mut(*p as usize) {
                        *slot = h;
                    }
                    facts.push(f);
                }
                self.enc(&l.body, env, &and(&facts), &format!("{path}.body"));
                (None, truth())
            }
            TKind::Un(UnOp::Not, x, _) => {
                let (v, ok) = self.enc(x, env, pc, &format!("{path}.operand"));
                (v.as_ref().and_then(Term::bool).map(|b| Term::B(b.not())), ok)
            }
            TKind::Un(UnOp::Neg, x, _) => {
                let (v, ok) = self.enc(x, env, pc, &format!("{path}.operand"));
                match v.as_ref().and_then(Term::int) {
                    Some(i) => {
                        let r = i.unary_minus();
                        let ok = and(&[ok, in_range(&r)]);
                        (Some(Term::I(r)), ok)
                    }
                    None => {
                        let (h, f) = self.unknown(&e.ty, "Float arithmetic");
                        (h, and(&[ok, f]))
                    }
                }
            }
            TKind::Bin(op, l, r, _) => self.binary(*op, l, r, &e.ty, env, pc, path),
            TKind::If(c, a, b) => {
                let (cv, cok) = self.enc(c, env, pc, &format!("{path}.cond"));
                let cv = self.as_bool(cv, "a condition the prover does not model");
                let (av, aok) = self.enc(a, env, &and(&[pc.clone(), cok.clone(), cv.clone()]), &format!("{path}.then"));
                let (bv, bok) = self.enc(b, env, &and(&[pc.clone(), cok.clone(), cv.not()]), &format!("{path}.else"));
                let ok = and(&[cok, cv.ite(&aok, &bok)]);
                let v = match (av, bv) {
                    (Some(Term::I(x)), Some(Term::I(y))) => Some(Term::I(cv.ite(&x, &y))),
                    (Some(Term::B(x)), Some(Term::B(y))) => Some(Term::B(cv.ite(&x, &y))),
                    _ => None,
                };
                (v, ok)
            }
            TKind::Let(id, v, b) => {
                let (vv, vok) = self.enc(v, env, pc, &format!("{path}.value"));
                if let Some(slot) = env.get_mut(*id as usize) {
                    *slot = vv;
                }
                let (bv, bok) = self.enc(b, env, &and(&[pc.clone(), vok.clone()]), &format!("{path}.body"));
                (bv, and(&[vok, bok]))
            }
            TKind::Seq(items) => {
                let (mut vals, ok) = self.seq(items, env, pc, path, "items");
                (vals.pop().flatten(), ok)
            }
            TKind::List(items) => (None, self.seq(items, env, pc, path, "items").1),
            TKind::Variant(_, args) => (None, self.seq(args, env, pc, path, "args").1),
            TKind::Record(fs) => {
                let names = self.tp.record_fields(&e.ty);
                let mut oks = Vec::new();
                let mut here = pc.clone();
                for (i, x) in fs {
                    let name = names.get(*i as usize).map(|(n, _)| n.clone()).unwrap_or_default();
                    let ok = self.enc(x, env, &here, &format!("{path}.fields.{name}")).1;
                    here = self.name(conj(&here, &ok));
                    oks.push(ok);
                }
                (None, and(&oks))
            }
            TKind::Field(x, _) => {
                let ok = self.enc(x, env, pc, &format!("{path}.target")).1;
                let (h, f) = self.unknown(&e.ty, "record fields");
                (h, and(&[ok, f]))
            }
            TKind::CallIndirect(f, args) => {
                let fok = self.enc(f, env, pc, path).1;
                let (_, aok) = self.seq(args, env, &and(&[pc.clone(), fok.clone()]), path, "args");
                let (h, facts) = self.unknown(&e.ty, "calls of function values");
                (h, and(&[fok, aok, facts]))
            }
            TKind::Builtin(b, args, _) => {
                let (vals, ok) = self.seq(args, env, pc, path, "args");
                let (v, range) = self.unknown(&e.ty, &format!("built-in `{}`", b.name()));
                let mut facts = vec![ok, range];
                if let Some(Term::I(x)) = &v {
                    match b {
                        Builtin::StrLen | Builtin::ListLen | Builtin::BytesLen | Builtin::MapSize => {
                            facts.push(x.ge(Int::from_i64(0)));
                            facts.push(x.le(Int::from_i64(u32::MAX as i64)));
                        }
                        Builtin::Compare => {
                            facts.push(x.ge(Int::from_i64(-1)));
                            facts.push(x.le(Int::from_i64(1)));
                            // Int and Bool order numerically: exact.
                            match (vals.first(), vals.get(1)) {
                                (Some(Some(Term::I(a))), Some(Some(Term::I(c)))) => facts.push(x.eq(a.lt(c).ite(&Int::from_i64(-1), &a.eq(c).ite(&Int::from_i64(0), &Int::from_i64(1))))),
                                (Some(Some(Term::B(a))), Some(Some(Term::B(c)))) => facts.push(x.eq(a.eq(c).ite(&Int::from_i64(0), &a.ite(&Int::from_i64(1), &Int::from_i64(-1))))),
                                _ => {}
                            }
                        }
                        Builtin::RandInt => {
                            // It returns only if lo <= hi (else a contract violation).
                            if let (Some(Some(Term::I(lo))), Some(Some(Term::I(hi)))) = (vals.first(), vals.get(1)) {
                                facts.push(lo.le(x));
                                facts.push(x.le(hi));
                            }
                        }
                        _ => facts.extend(bit_facts(*b, x, &vals)),
                    }
                }
                facts.extend(self.builtin_traps(*b, &vals));
                (v, and(&facts))
            }
            TKind::Call(i, args) => {
                let (vals, ok) = self.seq(args, env, pc, path, "args");
                self.call(*i, false, vals, ok, &e.ty, pc, path)
            }
            TKind::CallG(g, _, args) => {
                let (vals, ok) = self.seq(args, env, pc, path, "args");
                self.call(*g, true, vals, ok, &e.ty, pc, path)
            }
            TKind::Match(s, arms) => self.match_expr(s, arms, env, pc, path),
        }
    }

    /// The conditions under which a built-in call returns instead of trapping. They are part of
    /// `ok`, so a contract whose evaluation may trap in a built-in is never proven (in bodies they
    /// only add assumptions). Where the encoding models the arguments, the condition is exact or a
    /// necessary part of it; what it cannot state (a length, a separator, the body of a function
    /// argument) is a fresh condition, so `ok` never claims more than every real evaluation
    /// satisfies. Resource limits (memory, fuel, stack) are not contract failures and are not
    /// modelled for any expression. The match is exhaustive: a new built-in must be classified.
    fn builtin_traps(&mut self, b: Builtin, vals: &[Option<Term>]) -> Vec<Bool> {
        use Builtin as B;
        let int = |k: usize| match vals.get(k) {
            Some(Some(Term::I(i))) => Some(i.clone()),
            _ => None,
        };
        let zero = Int::from_i64(0);
        match b {
            // Never trap (besides resource limits).
            B::StrLen | B::ListLen | B::Join | B::Trim | B::Contains | B::StartsWith | B::ToInt | B::ToFloat | B::FromInt | B::FromFloat | B::Chars | B::Get | B::Push | B::Concat | B::Range
            | B::Reverse | B::SortInt | B::SortStr | B::IntToFloat | B::FloatToInt | B::ToUpper | B::ToLower | B::IndexOf | B::MapNew | B::MapGet | B::MapPut | B::MapDel | B::MapHas | B::MapSize
            | B::MapKeys | B::MapValues | B::ToBytes | B::BytesLen | B::BytesGet | B::BytesToList | B::BytesToStr | B::Compare | B::BitAnd | B::BitOr | B::BitXor | B::BitNot | B::FloatBits
            | B::FloatOfBits | B::AccList | B::BytesJoin | B::Sha256 => Vec::new(),
            // Exact where the distance is modelled.
            B::Shl | B::Shr | B::Ushr => match int(1) {
                Some(n) => vec![n.ge(&zero), n.lt(Int::from_i64(64))],
                None => vec![self.untracked(b)],
            },
            // from_char_code: a Unicode scalar value.
            B::FromCharCode => match int(0) {
                Some(n) => vec![n.ge(&zero), n.le(Int::from_i64(0x10FFFF)), Bool::or(&[n.lt(Int::from_i64(0xD800)), n.gt(Int::from_i64(0xDFFF))])],
                None => vec![self.untracked(b)],
            },
            // rand_int: lo <= hi.
            B::RandInt => match (int(0), int(1)) {
                (Some(lo), Some(hi)) => vec![lo.le(&hi)],
                _ => vec![self.untracked(b)],
            },
            // An index below a length the encoding does not model: the index is at least 0.
            B::ListAt | B::BytesAt => {
                let mut c = vec![self.untracked(b)];
                if let Some(i) = int(1) {
                    c.push(i.ge(&zero));
                }
                c
            }
            // slice/bytes_slice(x, from, to): 0 <= from <= to <= length (the length is not modelled).
            B::Slice | B::BytesSlice => {
                let mut c = vec![self.untracked(b)];
                if let Some(from) = int(1) {
                    c.push(from.ge(&zero));
                    if let Some(to) = int(2) {
                        c.push(from.le(&to));
                    }
                }
                c
            }
            // Conditions on values the encoding does not model (a non-empty separator or string,
            // bytes in 0..=255), and calls of a function argument, which may trap themselves.
            B::Split | B::Replace | B::CharCode | B::BytesFrom | B::Map | B::Filter | B::Fold => vec![self.untracked(b)],
            // Effects: never in a contract (contracts are pure); in a body they may fail in many ways.
            B::Print | B::ReadLine | B::ReadFile | B::WriteFile | B::NowMs | B::ReadAll | B::Http | B::DbExec | B::DbQuery => vec![self.untracked(b)],
        }
    }

    /// A fresh condition standing for a precondition of built-in `b` the encoding does not model.
    fn untracked(&mut self, b: Builtin) -> Bool {
        self.approx.insert(format!("the precondition of built-in `{}`", b.name()));
        Bool::fresh_const("pre")
    }

    #[allow(clippy::too_many_arguments)]
    fn binary(&mut self, op: BinOp, l: &TExpr, r: &TExpr, ty: &Type, env: &mut Vec<Option<Term>>, pc: &Bool, path: &str) -> (Option<Term>, Bool) {
        let (lp, rp) = (format!("{path}.left"), format!("{path}.right"));
        let (lv, lok) = self.enc(l, env, pc, &lp);
        if matches!(op, BinOp::And | BinOp::Or) {
            let lb = self.as_bool(lv, "a condition the prover does not model");
            // The right operand is evaluated only if the left one does not decide.
            let when = if op == BinOp::And { lb.clone() } else { lb.not() };
            let (rv, rok) = self.enc(r, env, &and(&[pc.clone(), lok.clone(), when.clone()]), &rp);
            let rb = self.as_bool(rv, "a condition the prover does not model");
            let v = if op == BinOp::And { Bool::and(&[lb, rb]) } else { Bool::or(&[lb, rb]) };
            return (Some(Term::B(v)), and(&[lok, when.implies(&rok)]));
        }
        let (rv, rok) = self.enc(r, env, &and(&[pc.clone(), lok.clone()]), &rp);
        let ok = and(&[lok, rok]);
        let ints = match (&lv, &rv) {
            (Some(Term::I(a)), Some(Term::I(b))) => Some((a.clone(), b.clone())),
            _ => None,
        };
        match (op, ints) {
            (BinOp::Add | BinOp::Sub | BinOp::Mul, Some((a, b))) => {
                let v = match op {
                    BinOp::Add => Int::add(&[a, b]),
                    BinOp::Sub => Int::sub(&[a, b]),
                    _ => Int::mul(&[a, b]),
                };
                let ok = and(&[ok, in_range(&v)]);
                (Some(Term::I(v)), ok)
            }
            (BinOp::Div, Some((a, b))) => {
                let q = tdiv(&a, &b);
                let ok = and(&[ok, b.eq(Int::from_i64(0)).not(), in_range(&q)]);
                (Some(Term::I(q)), ok)
            }
            (BinOp::Rem, Some((a, b))) => {
                // |a % b| < |b|, so the remainder is always in range (MIN % -1 = 0).
                let q = tdiv(&a, &b);
                let v = Int::sub(&[a, Int::mul(&[b.clone(), q])]);
                let ok = and(&[ok, b.eq(Int::from_i64(0)).not()]);
                (Some(Term::I(v)), ok)
            }
            (BinOp::Lt, Some((a, b))) => (Some(Term::B(a.lt(&b))), ok),
            (BinOp::Le, Some((a, b))) => (Some(Term::B(a.le(&b))), ok),
            (BinOp::Gt, Some((a, b))) => (Some(Term::B(a.gt(&b))), ok),
            (BinOp::Ge, Some((a, b))) => (Some(Term::B(a.ge(&b))), ok),
            (BinOp::Eq | BinOp::Ne, _) => {
                let eq = match (&lv, &rv) {
                    (Some(Term::I(a)), Some(Term::I(b))) => Some(a.eq(b)),
                    (Some(Term::B(a)), Some(Term::B(b))) => Some(a.eq(b)),
                    _ => None,
                };
                match eq {
                    Some(q) => (Some(Term::B(if op == BinOp::Ne { q.not() } else { q })), ok),
                    None => {
                        let (h, f) = self.unknown(ty, &format!("`{}` on {}", op.symbol(), l.ty));
                        (h, and(&[ok, f]))
                    }
                }
            }
            _ => {
                let what = if l.ty == Type::Float { "Float arithmetic".to_string() } else { format!("`{}` on {}", op.symbol(), l.ty) };
                let (h, f) = self.unknown(ty, &what);
                (h, and(&[ok, f]))
            }
        }
    }

    /// A call of `fns[i]` (or `generics[i]`): obligations for its `req`, assumptions from its contracts.
    #[allow(clippy::too_many_arguments)]
    fn call(&mut self, i: usize, generic: bool, args: Vec<Option<Term>>, args_ok: Bool, ty: &Type, pc: &Bool, path: &str) -> (Option<Term>, Bool) {
        let tp = self.tp;
        let callee = if generic { tp.generics.get(i) } else { tp.fns.get(i) };
        let Some(callee) = callee else {
            let (h, f) = self.unknown(ty, "a call the prover cannot resolve");
            return (h, and(&[args_ok, f]));
        };
        let (v, range) = havoc(ty);
        let mut facts = vec![args_ok.clone(), range];
        if self.depth >= MAX_INLINE {
            if callee.req.is_empty() && callee.ens.is_empty() {
                return (v, and(&facts));
            }
            self.approx.insert("contracts nested more than 3 calls deep".into());
            return (v, and(&facts));
        }
        if callee.ens.is_empty() && matches!(ty, Type::Int | Type::Bool) {
            self.approx.insert(format!("the result of `{}` (it has no `ens`)", callee.def.name));
        }
        let (saved, record) = (self.depth, self.record);
        self.depth += 1;
        self.record = false;
        self.locals.push(&callee.locals);
        let mut cenv: Vec<Option<Term>> = vec![None; callee.locals.len()];
        let named: Vec<(String, Option<Term>)> = callee.def.params.iter().map(|p| p.name.clone()).zip(args.iter().cloned()).collect();
        for (k, a) in args.into_iter().enumerate() {
            if let Some(slot) = cenv.get_mut(k) {
                *slot = a;
            }
        }
        // The callee checks its `req` in order: req[k] is evaluated only if the earlier ones held.
        let mut held = truth();
        let reached = and(&[pc.clone(), args_ok.clone()]);
        for (k, c) in callee.req.iter().enumerate() {
            let (cv, cok) = self.enc(c, &mut cenv, &held, &format!("req[{k}]"));
            let holds = and(&[cok, self.as_bool(cv, "a contract over values the prover does not model")]);
            if record {
                let contract = callee.def.req.get(k).map(|e| format!("req {}", qal_compiler::print_expr(e))).unwrap_or_default();
                self.obligations.push(Obligation {
                    goal: Goal::Req(callee.def.name.clone(), k),
                    label: format!("req[{k}] of {}", callee.def.name),
                    path: path.to_string(),
                    contract,
                    hyp: conj(&reached, &held),
                    holds: holds.clone(),
                    args: named.clone(),
                });
            }
            held = conj(&held, &holds);
        }
        facts.push(held);
        let mut facts = and(&facts);
        if let Some(r) = callee.result_local {
            if let Some(slot) = cenv.get_mut(r as usize) {
                *slot = v.clone();
            }
            for (k, c) in callee.ens.iter().enumerate() {
                let (cv, cok) = self.enc(c, &mut cenv, &facts, &format!("ens[{k}]"));
                let cv = self.as_bool(cv, "a contract over values the prover does not model");
                facts = and(&[facts, cok, cv]);
            }
        }
        self.locals.pop();
        self.depth = saved;
        self.record = record;
        (v, facts)
    }

    fn match_expr(&mut self, s: &TExpr, arms: &[TArm], env: &mut Vec<Option<Term>>, pc: &Bool, path: &str) -> (Option<Term>, Bool) {
        let (sv, sok) = self.enc(s, env, pc, &format!("{path}.scrutinee"));
        let pcs = and(&[pc.clone(), sok.clone()]);
        let mut prev = Bool::from_bool(false);
        let mut oks = vec![sok];
        let mut sels = Vec::new();
        let mut vals = Vec::new();
        for (i, arm) in arms.iter().enumerate() {
            let mut facts = Vec::new();
            let pm = self.pattern(&arm.pat, sv.as_ref(), &s.ty, env, &mut facts);
            let reach = and(&[prev.not(), pm, and(&facts)]);
            let sel = match &arm.guard {
                Some(g) => {
                    let (gv, gok) = self.enc(g, env, &and(&[pcs.clone(), reach.clone()]), &format!("{path}.arms[{i}].guard"));
                    oks.push(reach.implies(&gok));
                    and(&[reach.clone(), self.as_bool(gv, "a guard the prover does not model")])
                }
                None => reach,
            };
            let (bv, bok) = self.enc(&arm.body, env, &and(&[pcs.clone(), sel.clone()]), &format!("{path}.arms[{i}].body"));
            oks.push(sel.implies(&bok));
            prev = Bool::or(&[prev, sel.clone()]);
            sels.push(sel);
            vals.push(bv);
        }
        // Exactly one arm is taken by every evaluation that does not trap (matches are exhaustive;
        // falling off the end is a trap), and its value is the match's value.
        oks.push(Bool::or(&sels));
        let mut v = vals.pop().flatten();
        for (sel, x) in sels.iter().zip(vals).rev() {
            v = match (x, v) {
                (Some(Term::I(a)), Some(Term::I(b))) => Some(Term::I(sel.ite(&a, &b))),
                (Some(Term::B(a)), Some(Term::B(b))) => Some(Term::B(sel.ite(&a, &b))),
                _ => None,
            };
        }
        (v, and(&oks))
    }

    /// The condition that `p` matches the scrutinee (exact for literal, binding and wildcard patterns
    /// on an Int or Bool, a fresh condition otherwise); binds the pattern's variables in `env` and
    /// adds the facts about fresh bindings to `facts`.
    fn pattern(&mut self, p: &TPat, s: Option<&Term>, ty: &Type, env: &mut [Option<Term>], facts: &mut Vec<Bool>) -> Bool {
        match (p, s) {
            (TPat::Wild, _) => truth(),
            (TPat::Bind(id), Some(t)) => {
                if let Some(slot) = env.get_mut(*id as usize) {
                    *slot = Some(t.clone());
                }
                truth()
            }
            (TPat::Int(n), Some(Term::I(x))) => x.eq(Int::from_i64(*n)),
            (TPat::Bool(b), Some(Term::B(x))) => x.eq(Bool::from_bool(*b)),
            _ => {
                self.bind_unknown(p, ty, env, facts);
                if matches!(ty, Type::Int | Type::Bool) || self.binds_modelled(p, ty) {
                    self.approx.insert(format!("patterns on {ty}"));
                }
                Bool::fresh_const("m")
            }
        }
    }

    /// True if `p` binds a variable of type Int or Bool inside a value of type `ty`.
    fn binds_modelled(&self, p: &TPat, ty: &Type) -> bool {
        match p {
            TPat::Bind(_) => matches!(ty, Type::Int | Type::Bool),
            TPat::Variant(tag, ps) => {
                let fields = self.tp.variant_fields(ty, *tag);
                ps.iter().zip(&fields).any(|(q, t)| self.binds_modelled(q, t))
            }
            TPat::Record(fs) => {
                let decl = self.tp.record_fields(ty);
                fs.iter().any(|(i, q)| decl.get(*i as usize).is_some_and(|(_, t)| self.binds_modelled(q, t)))
            }
            _ => false,
        }
    }

    /// Binds every variable of `p` (a pattern on a value of type `ty`) to a fresh value.
    fn bind_unknown(&mut self, p: &TPat, ty: &Type, env: &mut [Option<Term>], facts: &mut Vec<Bool>) {
        match p {
            TPat::Bind(id) => {
                let (v, f) = havoc(ty);
                facts.push(f);
                if let Some(slot) = env.get_mut(*id as usize) {
                    *slot = v;
                }
            }
            TPat::Variant(tag, ps) => {
                let fields = self.tp.variant_fields(ty, *tag);
                for (q, t) in ps.iter().zip(&fields) {
                    self.bind_unknown(q, t, env, facts);
                }
            }
            TPat::Record(fs) => {
                let decl = self.tp.record_fields(ty);
                for (i, q) in fs {
                    if let Some((_, t)) = decl.get(*i as usize) {
                        self.bind_unknown(q, t, env, facts);
                    }
                }
            }
            _ => {}
        }
    }
}
