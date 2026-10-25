//! The incremental check of a define: only the definitions the define changes are checked again.
//!
//! A define used to check the whole codebase in the self-hosted compiler: about 1.2 s at 1520
//! definitions (`bench/loop/report.md`). The definitions it changes are the input and its
//! transitive dependents (exactly the ones that get new hashes); every other definition checks as
//! before, and its typed form is the one the last define produced. So the store keeps the checked
//! program of the last define next to the database, with the fingerprint of the codebase it
//! belongs to. A define of the same codebase:
//!
//! 1. checks a reduced program: in full the changed definitions, the generic functions and the
//!    definitions that call an instance of one (instances come from the bodies that need them);
//!    every other function as its signature with the body `todo()` (and no contracts), every other
//!    test as `true`.
//! 2. takes the typed functions and tests of the unchanged definitions from the kept program, with
//!    their calls renumbered by name, and by generic name and type arguments for instances
//!    (positions differ from one define to the next).
//!
//! Hashing works the same way: a definition's hash depends only on the definitions it refers to,
//! so only the changed definitions and what they refer to (transitively) are hashed again; the
//! canons of the others are the kept ones. What an input refers to is not known before it is
//! hashed, so every identifier in its text that names a stored definition counts (plus the type of
//! every variant it names): more definitions only cost time, a missing one fails the hash.
//!
//! The results equal the full ones (`QAL_INCREMENTAL=verify` compares them and reports on
//! stderr). Wherever that is not certain it falls back to the full check: no kept
//! program for this codebase, a diagnostic, any name or instance missing on either side. `QAL_INCREMENTAL=off` always checks in full.

use qal_core::ast::{Def, Expr, Program};
use qal_core::ast::TypeBody;
use qal_core::canon::{Canon, Hash, Hashed, Kind};
use qal_core::diag::Diagnostic;
use qal_core::tast::{TExpr, TFn, TKind, TProgram, TTest};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// The checked and hashed program of the last define and the key of the codebase it belongs to.
#[derive(Debug)]
pub struct Prior {
    pub fingerprint: Hash,
    pub typed: TProgram,
    pub hashed: Hashed,
}

const MAGIC: &[u8; 8] = b"QALTYPE2";

/// The key of a kept program: the codebase's fingerprint and the compiler that checked it (a new
/// compiler may type a program differently, so its checked programs are not reused).
pub fn key(codebase: &Hash) -> Hash {
    let mut bytes = codebase.0.to_vec();
    bytes.extend_from_slice(qal_compiler::compiler_identity().as_bytes());
    Hash::of(&bytes)
}

/// Where the kept program of the store at `db` lives (next to `store.db`; deleting it is safe).
pub fn cache_path(db: &Path) -> PathBuf {
    db.with_file_name("typed.cache")
}

/// The kept program, if the file is there and readable.
pub fn load(path: &Path) -> Option<Prior> {
    if mode() == Mode::Off {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < MAGIC.len() + 32 || &bytes[..MAGIC.len()] != MAGIC {
        return None;
    }
    let mut fp = [0u8; 32];
    fp.copy_from_slice(&bytes[MAGIC.len()..MAGIC.len() + 32]);
    let p = qal_compiler::proto::decode_prepared(&bytes[MAGIC.len() + 32..]).ok()?;
    let typed = qal_core::tform::decode(&p.typed).ok()?;
    let defs = p
        .canons
        .into_iter()
        .map(|c| Canon { name: c.name, kind: c.kind, hash: c.hash, canonical: c.canonical, group: c.group, local_names: c.local_names, deps: c.deps, refs: c.refs, signature: c.signature, effects: c.effects })
        .collect();
    Some(Prior { fingerprint: Hash(fp), typed, hashed: Hashed { defs, groups: p.groups.into_iter().collect() } })
}

/// Keeps `typed` and `hashed` as the program of the codebase with `fingerprint` (written to a
/// temporary file and renamed, so a reader never sees half of it). A failure only loses the cache.
pub fn save(path: &Path, fingerprint: &Hash, typed: &TProgram, hashed: &Hashed) {
    let canons = hashed
        .defs
        .iter()
        .map(|c| qal_compiler::proto::PreparedCanon {
            name: c.name.clone(),
            kind: c.kind,
            hash: c.hash,
            canonical: c.canonical.clone(),
            signature: c.signature.clone(),
            effects: c.effects.clone(),
            local_names: c.local_names.clone(),
            refs: c.refs.clone(),
            deps: c.deps.clone(),
            group: c.group,
        })
        .collect();
    let groups = hashed.groups.iter().map(|(h, f)| (*h, f.clone())).collect();
    let prepared = qal_compiler::proto::Prepared { typed: qal_core::tform::encode(typed), canons, groups, sig_old: Vec::new() };
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&fingerprint.0);
    bytes.extend_from_slice(&qal_compiler::proto::encode_prepared(&prepared));
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::write(&tmp, &bytes).is_ok() && std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    On,
    Off,
    Verify,
}

fn mode() -> Mode {
    match std::env::var("QAL_INCREMENTAL").as_deref() {
        Ok("off") => Mode::Off,
        Ok("verify") => Mode::Verify,
        _ => Mode::On,
    }
}

/// Checks `combined` (the codebase as the define would write it), reusing `prior` for the
/// definitions outside `changed`. `full` is the full check.
pub fn check(combined: &Program, prior: Option<&TProgram>, changed: &HashSet<String>, full: &dyn Fn(&Program) -> Result<TProgram, Vec<Diagnostic>>) -> Result<TProgram, Vec<Diagnostic>> {
    let verify = mode() == Mode::Verify;
    let fallback = |why: &str| {
        if verify {
            eprintln!("QAL_INCREMENTAL=verify: full check ({why})");
        }
        full(combined)
    };
    let Some(prior) = prior.filter(|_| mode() != Mode::Off) else { return fallback("no checked program of this codebase") };
    // Generic functions, and functions with a parameter `todo` (it would shadow the stub's `todo()`).
    let generic: HashSet<&str> = combined.defs.iter().filter_map(|d| match d {
        Def::Fn(f) if !f.tparams.is_empty() || !f.eparams.is_empty() || f.params.iter().any(|p| p.name == "todo") => Some(f.name.as_str()),
        _ => None,
    }).collect();
    let calls_instance = |e: &TExpr| calls(e, &|i| prior.fns.get(i).is_some_and(|f| f.instance.is_some()));
    let instance_users: HashSet<&str> = prior
        .fns
        .iter()
        .filter(|f| f.instance.is_none() && (calls_instance(&f.body) || f.req.iter().chain(&f.ens).any(calls_instance)))
        .map(|f| f.def.name.as_str())
        .chain(prior.tests.iter().filter(|t| calls_instance(&t.body)).map(|t| t.name.as_str()))
        .collect();
    let kept = |name: &str| !changed.contains(name) && !generic.contains(name) && !instance_users.contains(name);
    let reduced = Program { defs: combined.defs.iter().map(|d| if kept(d.name()) { stub(d) } else { d.clone() }).collect() };
    let new = match full(&reduced) {
        Ok(new) => new,
        Err(d) => return fallback(&format!("a diagnostic: {}", d.first().map_or(String::new(), |d| format!("{} {} in `{}`", d.code, d.message, d.def)))),
    };
    let Some(spliced) = splice(prior, new, &kept) else { return fallback("the kept program does not fit") };
    if verify {
        let whole = full(combined);
        let same = whole.as_ref().ok() == Some(&spliced);
        eprintln!("QAL_INCREMENTAL=verify: incremental check of {} changed definitions is {}", changed.len(), if same { "identical to the full one" } else { "DIFFERENT from the full one; the full result is used" });
        return whole;
    }
    Ok(spliced)
}

/// Hashes `combined`, reusing `prior` for the definitions outside `changed` that the changed ones
/// do not refer to. `refs_of` gives the names a stored definition refers to; `full` is the full
/// hash.
pub fn hash(
    combined: &Program,
    input: &Program,
    prior: Option<&Hashed>,
    changed: &HashSet<String>,
    refs_of: &dyn Fn(&str) -> Option<Vec<String>>,
    full: &dyn Fn(&Program) -> Result<Hashed, Vec<Diagnostic>>,
) -> Result<Hashed, Vec<Diagnostic>> {
    let verify = mode() == Mode::Verify;
    let fallback = |why: &str| {
        if verify {
            eprintln!("QAL_INCREMENTAL=verify: full hash ({why})");
        }
        full(combined)
    };
    let Some(prior) = prior.filter(|_| mode() != Mode::Off) else { return fallback("no hashed program of this codebase") };
    let names: HashSet<&str> = combined.defs.iter().map(|d| d.name()).collect();
    let mut type_of_variant: HashMap<&str, &str> = HashMap::new();
    for d in &combined.defs {
        if let Def::Type(t) = d {
            if let TypeBody::Enum(vs) = &t.body {
                for v in vs {
                    type_of_variant.insert(v.name.as_str(), t.name.as_str());
                }
            }
        }
    }
    // The closure: the changed definitions, what the input's text names, and what those refer to.
    let input_names: HashSet<&str> = input.defs.iter().map(|d| d.name()).collect();
    let mut closure: HashSet<String> = HashSet::new();
    let mut todo: Vec<String> = changed.iter().cloned().collect();
    let text = qal_compiler::print_program(input);
    for word in text.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.')).filter(|w| !w.is_empty()) {
        let parts: Vec<&str> = word.split('.').collect();
        for i in 0..parts.len() {
            for j in i + 1..=parts.len() {
                let w = parts[i..j].join(".");
                if let Some(t) = type_of_variant.get(w.as_str()) {
                    todo.push(t.to_string());
                }
                if names.contains(w.as_str()) {
                    todo.push(w);
                }
            }
        }
    }
    while let Some(n) = todo.pop() {
        if !closure.insert(n.clone()) || input_names.contains(n.as_str()) {
            continue;
        }
        match refs_of(&n) {
            Some(refs) => todo.extend(refs),
            None => return fallback("a name without stored references"),
        }
    }
    let sub = Program { defs: combined.defs.iter().filter(|d| closure.contains(d.name())).cloned().collect() };
    let Ok(new) = full(&sub) else { return fallback("the hash of the changed definitions failed") };
    let mut by_name: HashMap<&str, &Canon> = prior.defs.iter().map(|c| (c.name.as_str(), c)).collect();
    for c in &new.defs {
        by_name.insert(c.name.as_str(), c);
    }
    let mut defs = Vec::with_capacity(combined.defs.len());
    for d in &combined.defs {
        match by_name.get(d.name()) {
            Some(c) if c.kind == Kind::of(d) => defs.push((*c).clone()),
            _ => return fallback("the kept canons do not fit"),
        }
    }
    let mut groups = std::collections::BTreeMap::new();
    for g in defs.iter().filter_map(|c| c.group) {
        match new.groups.get(&g).or_else(|| prior.groups.get(&g)) {
            Some(f) => groups.insert(g, f.clone()),
            None => return fallback("a group form is missing"),
        };
    }
    let spliced = Hashed { defs, groups };
    if verify {
        let whole = full(combined);
        let same = whole.as_ref().ok() == Some(&spliced);
        eprintln!("QAL_INCREMENTAL=verify: incremental hash of {} of {} definitions is {}", sub.defs.len(), combined.defs.len(), if same { "identical to the full one" } else { "DIFFERENT from the full one; the full result is used" });
        return whole;
    }
    Ok(spliced)
}

/// A definition reduced to what the global environment needs: its signature.
fn stub(d: &Def) -> Def {
    match d {
        Def::Fn(f) => {
            let mut f = f.clone();
            f.req.clear();
            f.ens.clear();
            f.body = Expr::Call("todo".into(), vec![]);
            Def::Fn(f)
        }
        Def::Test(t) => {
            let mut t = t.clone();
            t.body = Expr::Bool(true);
            Def::Test(t)
        }
        Def::Type(_) => d.clone(),
    }
}

/// `new` (the reduced program, checked) with the typed functions and tests of the kept names taken
/// from `prior`, their calls renumbered to `new`'s positions. None if that is not exact.
fn splice(prior: &TProgram, mut new: TProgram, kept: &dyn Fn(&str) -> bool) -> Option<TProgram> {
    // Program functions are the ones without an instance; instances follow them.
    let index_of = |p: &TProgram| -> HashMap<String, usize> { p.fns.iter().enumerate().filter(|(_, f)| f.instance.is_none()).map(|(i, f)| (f.def.name.clone(), i)).collect() };
    let new_index = index_of(&new);
    // Instances by generic name and type arguments.
    let instance_of = |p: &TProgram, f: &TFn| f.instance.as_ref().and_then(|(g, args)| p.generics.get(*g).map(|g| (g.def.name.clone(), args.clone())));
    let new_instances: Vec<((String, Vec<qal_core::ast::Type>), usize)> = new.fns.iter().enumerate().filter_map(|(i, f)| instance_of(&new, f).map(|k| (k, i))).collect();
    let image = |i: usize| -> Option<usize> {
        let f = prior.fns.get(i)?;
        match instance_of(prior, f) {
            None if f.instance.is_none() => new_index.get(&f.def.name).copied(),
            None => None,
            Some(k) => new_instances.iter().find(|(k2, _)| *k2 == k).map(|(_, j)| *j),
        }
    };
    let prior_fns: HashMap<&str, &TFn> = prior.fns.iter().filter(|f| f.instance.is_none()).map(|f| (f.def.name.as_str(), f)).collect();
    let prior_tests: HashMap<&str, &TTest> = prior.tests.iter().map(|t| (t.name.as_str(), t)).collect();
    let renumber = |e: &TExpr| -> Option<TExpr> { remap(e, &image) };
    for slot in new.fns.iter_mut().filter(|f| f.instance.is_none() && kept(&f.def.name)) {
        let old = prior_fns.get(slot.def.name.as_str())?;
        if old.def.params != slot.def.params || old.def.ret != slot.def.ret || old.def.effects != slot.def.effects {
            return None;
        }
        let mut f = (*old).clone();
        f.body = renumber(&old.body)?;
        f.req = old.req.iter().map(&renumber).collect::<Option<_>>()?;
        f.ens = old.ens.iter().map(&renumber).collect::<Option<_>>()?;
        *slot = f;
    }
    for slot in new.tests.iter_mut().filter(|t| kept(&t.name)) {
        let old = prior_tests.get(slot.name.as_str())?;
        let mut t = (*old).clone();
        t.body = renumber(&old.body)?;
        *slot = t;
    }
    Some(new)
}

/// True if `e` refers to a function index `i` with `pred(i)`.
fn calls(e: &TExpr, pred: &dyn Fn(usize) -> bool) -> bool {
    remap(e, &|i| if pred(i) { None } else { Some(i) }).is_none()
}

/// `e` with every function index `i` replaced by `map(i)`; None if one has no image.
fn remap(e: &TExpr, map: &dyn Fn(usize) -> Option<usize>) -> Option<TExpr> {
    let r = |x: &TExpr| remap(x, map);
    let b = |x: &TExpr| r(x).map(Box::new);
    let all = |xs: &[TExpr]| xs.iter().map(r).collect::<Option<Vec<_>>>();
    let kind = match &e.kind {
        TKind::FnRef(i) => TKind::FnRef(map(*i)?),
        TKind::Call(i, args) => TKind::Call(map(*i)?, all(args)?),
        // Typed programs of a define are monomorphic: generic calls do not occur.
        TKind::CallG(..) | TKind::FnRefG(..) => return None,
        TKind::Int(_) | TKind::Float(_) | TKind::Bool(_) | TKind::Str(_) | TKind::Unit | TKind::Local(_) => e.kind.clone(),
        TKind::Bin(op, l, rr, path) => TKind::Bin(*op, b(l)?, b(rr)?, path.clone()),
        TKind::Un(op, x, path) => TKind::Un(*op, b(x)?, path.clone()),
        TKind::If(c, t, f) => TKind::If(b(c)?, b(t)?, b(f)?),
        TKind::Let(x, v, body) => TKind::Let(*x, b(v)?, b(body)?),
        TKind::Lambda(l) => {
            let mut l2 = (**l).clone();
            l2.body = r(&l.body)?;
            TKind::Lambda(Box::new(l2))
        }
        TKind::CallIndirect(f, args) => TKind::CallIndirect(b(f)?, all(args)?),
        TKind::Builtin(bi, args, path) => TKind::Builtin(*bi, all(args)?, path.clone()),
        TKind::List(xs) => TKind::List(all(xs)?),
        TKind::Variant(tag, xs) => TKind::Variant(*tag, all(xs)?),
        TKind::Record(fs) => TKind::Record(fs.iter().map(|(i, x)| r(x).map(|x| (*i, x))).collect::<Option<_>>()?),
        TKind::Field(x, i) => TKind::Field(b(x)?, *i),
        TKind::Match(s, arms) => {
            let mut out = Vec::with_capacity(arms.len());
            for a in arms {
                let mut a2 = a.clone();
                a2.guard = match &a.guard {
                    Some(g) => Some(r(g)?),
                    None => None,
                };
                a2.body = r(&a.body)?;
                out.push(a2);
            }
            TKind::Match(b(s)?, out)
        }
        TKind::Seq(xs) => TKind::Seq(all(xs)?),
    };
    Some(TExpr { ty: e.ty.clone(), kind })
}
