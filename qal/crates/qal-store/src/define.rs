//! `define`: parse, assemble with the stored codebase, enforce the replacement rule, check, hash,
//! and write everything in one transaction (spec §6).

use crate::snapshot::Snapshot;
use crate::space::Space;
use crate::{now_ms, sql, Defined, Verify};
use qal_core::ast::{Def, Program, TypeBody};
use qal_core::canon::{self, Canon, Hash, HashError, Hashed, Kind};
use qal_core::diag::Diagnostic;
use qal_core::tast::TProgram;
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use qal_compiler::lexer::{is_std_name, std_parts};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Who may write `std.` names in a `define` (Phase 3 §4.1). In the std project (`meta.role = 'std'`)
/// every definition must have a `std.` name instead, whatever the writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdNames {
    /// A project's own `define`: `std.` names are reserved (E0240).
    Refuse,
    /// The import of a package other than std: a `std.` name only where it is identical to the
    /// stored definition (a package exports the std definitions it depends on).
    Unchanged,
    /// The import of the std package: `std.` names are added or replaced, signatures kept (E0244).
    Package,
}

/// The released standard library: the std package embedded in the `qal` binary (its hash is fixed
/// at build time). Only it may add or replace `std.` names in a project, and in the std store its
/// names keep their signatures and cannot be deleted unless the change is declared breaking.
#[derive(Debug, Clone, PartialEq)]
pub struct Release {
    pub hash: String,
    /// name -> (kind, signature text) of every released definition.
    pub signatures: BTreeMap<String, (Kind, String)>,
}

impl Release {
    /// The release a std package file describes (checksum verified).
    pub fn of(file: &str) -> Result<Release, Vec<Diagnostic>> {
        let pkg = crate::package::parse(file)?;
        if pkg.name != crate::package::STD_PACKAGE_NAME {
            return Err(vec![Diagnostic::new("E0243", "package_mismatch", "", "", format!("the standard library package is named `{}`, not `std`", pkg.name))]);
        }
        let program = qal_compiler::parse_program(&pkg.text).map_err(|d| vec![d])?;
        Ok(Release { hash: pkg.hash, signatures: program.defs.iter().map(|d| (d.name().to_string(), signature(d))).collect() })
    }
}

/// What a write may do with `std.` names.
#[derive(Debug, Clone, Copy)]
pub struct Rules<'a> {
    pub std: StdNames,
    pub release: Option<&'a Release>,
    /// The std store only: allow changing or deleting released names (`--breaking`).
    pub breaking: bool,
}

/// In the std store: released names keep their signatures unless the change is breaking (E0244).
fn released_signatures(rules: &Rules, input: &Program) -> Result<(), Vec<Diagnostic>> {
    let (Some(release), false) = (rules.release, rules.breaking) else { return Ok(()) };
    let diags: Vec<Diagnostic> = input
        .defs
        .iter()
        .filter_map(|d| {
            let before = release.signatures.get(d.name())?;
            let after = signature(d);
            (*before != after).then(|| {
                Diagnostic::std_signature_change(d.name(), &show(before), &show(&after))
                    .hint(format!("`{}` is in the released standard library (std {}); a deliberate break needs `qal define --breaking` on the command line (and again on every later define of it until qal is rebuilt with the new package)", d.name(), &release.hash[..release.hash.len().min(12)]))
            })
        })
        .collect();
    if diags.is_empty() {
        Ok(())
    } else {
        Err(diags)
    }
}

/// The value of `meta.role` that marks the std project's store.
pub const ROLE_STD: &str = "std";

/// True if the store is the std project's (`meta.role = 'std'`).
pub(crate) fn is_std_store(conn: &Connection) -> Result<bool, Vec<Diagnostic>> {
    let role: Option<String> = conn.query_row("SELECT value FROM meta WHERE key = 'role'", [], |r| r.get(0)).optional().map_err(sql)?;
    Ok(role.as_deref() == Some(ROLE_STD))
}

/// The namespace rule for the names `input` defines (before checking): see [`StdNames`].
fn namespace(std_store: bool, policy: StdNames, input: &Program) -> Result<(), Vec<Diagnostic>> {
    let mut diags = Vec::new();
    for d in &input.defs {
        let name = d.name();
        let variants = match d {
            Def::Type(t) => match &t.body {
                TypeBody::Enum(vs) => vs.iter().map(|v| v.name.as_str()).collect(),
                TypeBody::Record(_) => Vec::new(),
            },
            _ => Vec::new(),
        };
        // The std store, and (defence in depth) the std package anywhere: only `std.` names,
        // variants in their type's module.
        if std_store || policy == StdNames::Package {
            let Some((module, _)) = std_parts(name) else {
                diags.push(Diagnostic::std_only(name, "", name, None));
                continue;
            };
            for (i, v) in variants.iter().enumerate() {
                if std_parts(v).is_none_or(|(m, _)| m != module) {
                    diags.push(Diagnostic::std_only(name, &format!("variants[{i}]"), v, Some(module)));
                }
            }
            continue;
        }
        if is_std_name(name) {
            if policy == StdNames::Refuse {
                diags.push(Diagnostic::std_reserved(name, "", name));
            }
            continue;
        }
        for (i, v) in variants.iter().enumerate().filter(|(_, v)| is_std_name(v)) {
            diags.push(Diagnostic::std_reserved(name, &format!("variants[{i}]"), v));
        }
    }
    if diags.is_empty() {
        Ok(())
    } else {
        Err(diags)
    }
}

/// Imports keep the signature of every stored `std.` definition (E0244).
fn std_signatures(tx: &Transaction, snap: &Snapshot, input: &Program) -> Result<(), Vec<Diagnostic>> {
    let mut diags = Vec::new();
    for d in input.defs.iter().filter(|d| is_std_name(d.name()) && snap.names.contains_key(d.name())) {
        let (before, after) = (signature(&snap.decode_name(tx, d.name())?), signature(d));
        if before != after {
            diags.push(Diagnostic::std_signature_change(d.name(), &show(&before), &show(&after)));
        }
    }
    if diags.is_empty() {
        Ok(())
    } else {
        Err(diags)
    }
}

/// [`StdNames::Unchanged`]: the `std.` names of the input must hash as stored.
fn std_unchanged(snap: &Snapshot, input: &Program, canons: &[Canon]) -> Result<(), Vec<Diagnostic>> {
    let hash_of: HashMap<&str, Hash> = canons.iter().map(|c| (c.name.as_str(), c.hash)).collect();
    let diags: Vec<Diagnostic> = input
        .defs
        .iter()
        .map(|d| d.name())
        .filter(|n| is_std_name(n) && snap.names.get(*n).map(|e| e.hash) != hash_of.get(n).copied())
        .map(|n| Diagnostic::std_reserved(n, "", n).hint("the package was exported with another version of the standard library: export it again from a project with this project's std (if this write reported `std_warnings`, this project could not import the std of this `qal`: fix that first)"))
        .collect();
    if diags.is_empty() {
        Ok(())
    } else {
        Err(diags)
    }
}

/// A definition's signature as the replacement rule compares it: kind plus the signature text
/// with current names (so a replaced type keeps its name in the signatures that use it).
pub(crate) fn signature(d: &Def) -> (Kind, String) {
    (Kind::of(d), canon::signature_text(d, &|n| n.to_string()))
}

pub(crate) fn show(sig: &(Kind, String)) -> String {
    match sig.0 {
        Kind::Fn => sig.1.clone(),
        Kind::Type => format!("type = {}", sig.1),
        Kind::Test => "test".to_string(),
    }
}

/// Hashes a checked program; interchangeable recursive definitions are a diagnostic.
pub(crate) fn hash(p: &Program) -> Result<Hashed, Vec<Diagnostic>> {
    qal_compiler::hash_program(p).map_err(|e| match e {
        HashError::Symmetric(names) => vec![Diagnostic::symmetric_group(&names)],
        HashError::TooLarge(names) => vec![Diagnostic::group_too_large(&names)],
        HashError::Internal(m) => vec![Diagnostic::store_error(format!("cannot hash a checked program: {m}"))],
    })
}

/// How often `define` verifies on a snapshot, outside the write lock, before it verifies inside it.
pub(crate) const ATTEMPTS: usize = 3;

/// A check of a `define` before it is committed, inside the write transaction that would commit it
/// (it can also write more in the same transaction; used by package imports and the signed records
/// of a team store). It gets the result and the names as they were before the write.
pub type Check<'c> = dyn Fn(&[Defined], &Transaction, &crate::team::Rows) -> Result<(), Vec<Diagnostic>> + 'c;

/// A check of a `define` against the space before anything is written (the team rules: rights,
/// scope, interface, leases, stale base). `dry`: on a snapshot outside the write lock, it must not
/// write; otherwise inside the write transaction (it may write, e.g. leases).
pub type Guard<'c> = dyn Fn(&Transaction, &Snapshot, &Program, &TProgram, bool) -> Result<(), Vec<Diagnostic>> + 'c;

/// No guard (outside team mode).
pub fn no_guard(_: &Transaction, _: &Snapshot, _: &Program, _: &TProgram, _: bool) -> Result<(), Vec<Diagnostic>> {
    Ok(())
}

thread_local! {
    /// How long the last `define` of this thread held the write lock (Phase 4 spec §20, risk 3).
    static LOCK_HELD: std::cell::Cell<std::time::Duration> = const { std::cell::Cell::new(std::time::Duration::ZERO) };
}

/// How long the last `define` (or import) of this thread held SQLite's write lock.
pub fn last_lock_held() -> std::time::Duration {
    LOCK_HELD.with(|c| c.get())
}

/// Commits `tx` and records how long the write lock was held since `start`.
fn commit(tx: Transaction, start: std::time::Instant) -> Result<(), Vec<Diagnostic>> {
    tx.commit().map_err(sql)?;
    LOCK_HELD.with(|c| c.set(start.elapsed()));
    Ok(())
}

/// `define` in `space`, committed only if `guard` and `check` accept it (after the verification);
/// `rules` say who may write `std.` names.
#[allow(clippy::too_many_arguments)]
/// The codebase a `define` wrote, checked: what the tests and property tests that follow run on
/// (loading and checking it again cost a third of a define's time).
pub struct Written {
    pub program: Program,
    pub typed: TProgram,
}

#[allow(clippy::too_many_arguments)]
pub fn define_checked(conn: &mut Connection, src: &str, agent: &str, rules: &Rules, space: &Space, verify: Option<&mut Verify>, guard: &Guard, check: &Check, written: &mut Option<Written>) -> Result<Vec<Defined>, Vec<Diagnostic>> {
    let input = qal_compiler::parse_program(src).map_err(|d| vec![d])?;
    let Some(v) = verify else {
        let start = std::time::Instant::now();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
        let state = prepare(&tx, &input, rules, space)?;
        guard(&tx, &state.snap, &input, &state.typed, false)?;
        let out = state.write(&tx, &input, agent, space, false)?;
        check(&out, &tx, &crate::team::rows_of(&state.snap))?;
        commit(tx, start)?;
        *written = Some(state.into_written());
        return Ok(out);
    };
    // The verifier (the prover) may take its whole time budget, longer than another writer waits
    // for the lock (review finding): it runs on a snapshot without the lock, and the results are
    // written by a short IMMEDIATE transaction if the definitions to verify are still the same
    // (results depend only on a definition's hash). A rejection rejects the `define` as it would
    // have been on that snapshot.
    for _ in 0..ATTEMPTS {
        let (first, todo) = {
            // A deferred transaction that only reads: a consistent snapshot, no write lock.
            let tx = conn.transaction().map_err(sql)?;
            let state = prepare(&tx, &input, rules, space)?;
            guard(&tx, &state.snap, &input, &state.typed, true)?;
            let out = state.write(&tx, &input, agent, space, true)?;
            let todo = to_verify(&tx, &out, &v.key)?;
            (state, todo)
        };
        let rows = if todo.is_empty() { Vec::new() } else { (v.run)(&first.typed, &todo)? };
        // IMMEDIATE takes the write lock before reading, so two concurrent `define`s serialize
        // instead of both deciding on the same old state.
        let start = std::time::Instant::now();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
        // Unchanged since the snapshot (the usual case: one agent per store): the checked and
        // hashed program of the first pass is the one to write. Checking and hashing the whole
        // codebase again cost half of a define's time (0.28 of 0.74 s on the living program).
        let now_snap = Snapshot::load(&tx, space)?;
        let state = if now_snap.same_codebase(&first.snap) { first } else { prepare(&tx, &input, rules, space)? };
        guard(&tx, &state.snap, &input, &state.typed, false)?;
        let out = state.write(&tx, &input, agent, space, false)?;
        let now = to_verify(&tx, &out, &v.key)?;
        if now.iter().all(|d| todo.contains(d)) {
            let hashes: HashSet<&str> = now.iter().map(|(_, h)| h.as_str()).collect();
            store_results(&tx, &v.key, rows.into_iter().filter(|(h, _)| hashes.contains(h.as_str())))?;
            check(&out, &tx, &crate::team::rows_of(&state.snap))?;
            commit(tx, start)?;
            *written = Some(state.into_written());
            return Ok(out);
        }
        // Another writer changed what has to be verified: drop this transaction and start over.
    }
    let start = std::time::Instant::now();
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
    let state = prepare(&tx, &input, rules, space)?;
    guard(&tx, &state.snap, &input, &state.typed, false)?;
    let out = state.write(&tx, &input, agent, space, false)?;
    let todo = to_verify(&tx, &out, &v.key)?;
    if !todo.is_empty() {
        store_results(&tx, &v.key, (v.run)(&state.typed, &todo)?.into_iter())?;
    }
    check(&out, &tx, &crate::team::rows_of(&state.snap))?;
    commit(tx, start)?;
    *written = Some(state.into_written());
    Ok(out)
}

/// The checked and hashed codebase that `define` would write.
struct Prepared {
    snap: Snapshot,
    /// The stored definitions with the input, as checked.
    program: Program,
    typed: TProgram,
    hashed: Hashed,
}

impl Prepared {
    fn into_written(self) -> Written {
        Written { program: self.program, typed: self.typed }
    }

    /// Writes the definitions (`dry`: only computes the results).
    fn write(&self, tx: &Transaction, input: &Program, agent: &str, space: &Space, dry: bool) -> Result<Vec<Defined>, Vec<Diagnostic>> {
        write(tx, &self.snap, input, &self.hashed, agent, space, dry)
    }
}

#[cfg(test)]
thread_local! {
    /// Tests: write like a store from before Phase 3, which did not reserve the prelude names.
    pub(crate) static LEGACY_WRITER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn legacy_writer() -> bool {
    LEGACY_WRITER.with(|c| c.get())
}

#[cfg(not(test))]
fn legacy_writer() -> bool {
    false
}

/// Assembles the input with the stored codebase, enforces the replacement rule, checks and hashes.
fn prepare(tx: &Transaction, input: &Program, rules: &Rules, space: &Space) -> Result<Prepared, Vec<Diagnostic>> {
    let policy = rules.std;
    let snap = Snapshot::load(tx, space)?;
    let std_store = is_std_store(tx)?;
    namespace(std_store, policy, input)?;
    if std_store && policy == StdNames::Refuse {
        released_signatures(rules, input)?;
    }
    if policy != StdNames::Refuse {
        std_signatures(tx, &snap, input)?;
    }

    let input_names: HashSet<&str> = input.defs.iter().map(|d| d.name()).collect();
    let changed = replacement_rule(tx, &snap, input, &input_names, &Default::default())?;

    // Stored definitions keep the names they were written against; the ones that refer to a
    // redefined name get new hashes when the combined program is hashed.
    let stored = snap.program(tx, &|n| input_names.contains(n))?;
    let mut combined = stored;
    combined.defs.extend(input.defs.iter().cloned());
    // Only the definitions written now may not take prelude names (older stored ones keep working).
    let typed = qal_compiler::check_stored(&combined, &|n| input_names.contains(n) && !legacy_writer()).map_err(|d| broken_dependents(&changed, d))?;

    let hashed = hash(&combined)?;
    if policy == StdNames::Unchanged && !std_store {
        std_unchanged(&snap, input, &hashed.defs)?;
    }
    identical_types(&hashed.defs)?;
    group_aliases(&hashed.defs)?;
    alias_references(&hashed.defs, &input_names)?;
    canonical_check(&snap, &input_names, &hashed.defs)?;
    Ok(Prepared { snap, program: combined, typed, hashed })
}

/// The (name, hash) of the definitions to verify: the functions and tests that are new, got a new
/// hash, or have no stored result for `key`.
pub(crate) fn to_verify(tx: &Transaction, out: &[Defined], key: &str) -> Result<Vec<(String, String)>, Vec<Diagnostic>> {
    let mut todo = Vec::new();
    for d in out.iter().filter(|d| d.kind != "type") {
        let stored: Option<i64> = tx.query_row("SELECT 1 FROM proofs WHERE hash = ?1 AND prover = ?2", params![d.hash, key], |r| r.get(0)).optional().map_err(sql)?;
        if d.status != "unchanged" || stored.is_none() {
            todo.push((d.name.clone(), d.hash.clone()));
        }
    }
    Ok(todo)
}

/// Stores verification results (hash, JSON) for `key`.
pub(crate) fn store_results(tx: &Transaction, key: &str, rows: impl Iterator<Item = (String, String)>) -> Result<(), Vec<Diagnostic>> {
    let now = now_ms();
    for (hash, results) in rows {
        tx.execute(
            "INSERT INTO proofs (hash, prover, results, at) VALUES (?1, ?2, ?3, ?4) \
             ON CONFLICT(hash, prover) DO UPDATE SET results = excluded.results, at = excluded.at",
            params![hash, key, results, now],
        )
        .map_err(sql)?;
    }
    Ok(())
}

/// The result of replaying definitions on top of a space (the merge and the rebase of a change set,
/// Phase 4 spec §7.2, §7.3): the checked and hashed codebase, nothing written yet.
pub(crate) struct Replayed {
    pub program: Program,
    pub typed: TProgram,
    pub hashed: Hashed,
    /// The full row of every name after the replay.
    pub rows: crate::team::Rows,
    /// Every name whose row differs from the snapshot, with its status like `define`'s result
    /// (`added`/`replaced` for inputs, `relinked` for dependents; `unchanged` for an input whose
    /// hash stays), inputs first in input order, then the rest by name.
    pub defined: Vec<Defined>,
}

/// Replays `input` (definitions decoded from a change set, written against its names) on `snap`
/// with the names in `removed` gone, as one `define` would: the replacement rule (a changed
/// signature needs its other dependents to still check), the full check, hashing and the
/// store invariants. Nothing is written (see [`store_defs`]).
pub(crate) fn replay(tx: &Transaction, snap: &Snapshot, input: &Program, removed: &std::collections::BTreeSet<String>) -> Result<Replayed, Vec<Diagnostic>> {
    let input_names: HashSet<&str> = input.defs.iter().map(|d| d.name()).collect();
    let changed = replacement_rule(tx, snap, input, &input_names, removed)?;
    let stored = snap.program(tx, &|n| input_names.contains(n) || removed.contains(n))?;
    let mut combined = stored;
    combined.defs.extend(input.defs.iter().cloned());
    let typed = qal_compiler::check_stored(&combined, &|n| input_names.contains(n)).map_err(|d| broken_dependents(&changed, d))?;
    let hashed = hash(&combined)?;
    identical_types(&hashed.defs)?;
    group_aliases(&hashed.defs)?;
    alias_references(&hashed.defs, &input_names)?;
    canonical_check(snap, &input_names, &hashed.defs)?;
    let hash_of: HashMap<&str, Hash> = hashed.defs.iter().map(|c| (c.name.as_str(), c.hash)).collect();
    let mut rows = crate::team::Rows::new();
    let mut by_name: HashMap<&str, Defined> = HashMap::new();
    for c in &hashed.defs {
        let refs = ref_names(c, &hash_of);
        let row = crate::team::NameRow { hash: c.hash.to_hex(), refs, locals: c.local_names.clone() };
        let old = snap.names.get(&c.name);
        let is_input = input_names.contains(c.name.as_str());
        let differs = old.is_none_or(|e| e.hash != c.hash || e.refs != row.refs || e.locals != row.locals);
        let status = match old {
            None => "added",
            Some(e) if e.hash == c.hash => "unchanged",
            Some(_) if is_input => "replaced",
            Some(_) => "relinked",
        };
        if is_input || differs {
            let old_hash = old.map(|e| e.hash.to_hex()).filter(|_| matches!(status, "replaced" | "relinked"));
            by_name.insert(&c.name, Defined { name: c.name.clone(), kind: c.kind.as_str(), hash: c.hash.to_hex(), status, old_hash, resolved: false });
        }
        rows.insert(c.name.clone(), row);
    }
    let mut defined = Vec::new();
    for d in &input.defs {
        if let Some(r) = by_name.remove(d.name()) {
            defined.push(r);
        }
    }
    let mut rest: Vec<Defined> = by_name.into_values().collect();
    rest.sort_by(|a, b| a.name.cmp(&b.name));
    defined.extend(rest);
    Ok(Replayed { program: combined, typed, hashed, rows, defined })
}

/// A stored definition records one name per referenced definition, so a definition that refers to
/// two names of the same definition cannot be stored.
fn alias_references(canons: &[Canon], input_names: &HashSet<&str>) -> Result<(), Vec<Diagnostic>> {
    let hash_of: HashMap<&str, Hash> = canons.iter().map(|c| (c.name.as_str(), c.hash)).collect();
    let mut diags = Vec::new();
    for c in canons {
        let mut by_target: BTreeMap<Hash, Vec<&str>> = BTreeMap::new();
        for n in &c.refs {
            if let Some(h) = hash_of.get(n.as_str()) {
                by_target.entry(*h).or_default().push(n);
            }
        }
        for (h, names) in by_target.iter().filter(|(_, ns)| ns.len() > 1) {
            let names: Vec<String> = names.iter().map(|n| n.to_string()).collect();
            let mut d = Diagnostic::alias_reference(&c.name, "", &names, &h.short(), input_names.contains(c.name.as_str()));
            if let (Some(std), Some(own)) = (names.iter().find(|n| is_std_name(n)), names.iter().find(|n| !is_std_name(n))) {
                d = d.hint(format!("`{own}` has the same structure as the standard library's `{std}`: use `{std}` here (or replace `{own}` with it everywhere and delete `{own}`)"));
            }
            diags.push(d);
        }
    }
    if diags.is_empty() {
        Ok(())
    } else {
        Err(diags)
    }
}

/// A member of a recursive group can have only one name: a second name would decode its sibling
/// references through another name and leave the group.
/// As for types, a std group and a project group of the same structure may coexist (review N3):
/// their members' references are names in their own namespace (`names.refs`).
fn group_aliases(canons: &[Canon]) -> Result<(), Vec<Diagnostic>> {
    let mut by_hash: BTreeMap<(Hash, bool), Vec<&str>> = BTreeMap::new();
    for c in canons.iter().filter(|c| c.group.is_some()) {
        by_hash.entry((c.hash, is_std_name(&c.name))).or_default().push(&c.name);
    }
    let mut diags = Vec::new();
    for names in by_hash.values_mut().filter(|ns| ns.len() > 1) {
        names.sort_unstable();
        for dup in &names[1..] {
            diags.push(
                Diagnostic::new("E0202", "duplicate_name", dup, "", format!("`{dup}` is identical to the recursive definition `{}`", names[0]))
                    .hint(format!("a member of a mutually recursive group has one name: use `{}`, or rename it", names[0])),
            );
        }
    }
    if diags.is_empty() {
        Ok(())
    } else {
        Err(diags)
    }
}

/// A stored definition may only get a new hash if it (transitively) refers to a redefined name
/// whose hash changed; anything else means its stored bytes were not in canonical form.
fn canonical_check(snap: &Snapshot, input_names: &HashSet<&str>, canons: &[Canon]) -> Result<(), Vec<Diagnostic>> {
    let mut affected: HashSet<&str> =
        canons.iter().filter(|c| input_names.contains(c.name.as_str()) && snap.names.get(&c.name).is_none_or(|e| e.hash != c.hash)).map(|c| c.name.as_str()).collect();
    let mut queue: Vec<&str> = affected.iter().copied().collect();
    while let Some(n) = queue.pop() {
        for u in snap.users.get(n).into_iter().flatten() {
            if affected.insert(u.as_str()) {
                queue.push(u.as_str());
            }
        }
    }
    for c in canons {
        if input_names.contains(c.name.as_str()) {
            continue;
        }
        if let Some(e) = snap.names.get(&c.name) {
            if e.hash != c.hash && !affected.contains(c.name.as_str()) {
                return Err(vec![Diagnostic::store_error(format!(
                    "the store is corrupt: the stored definition `{}` (#{}) is not in canonical form",
                    c.name, e.hash
                ))]);
            }
        }
    }
    Ok(())
}

/// A definition replaced with a different signature, and the stored definitions written against it
/// that the call does not redefine.
struct Changed {
    name: String,
    before: String,
    after: String,
    dependents: Vec<String>,
}

/// A name keeps its kind (E0405). A replacement with a different signature (of the same kind) is
/// allowed when every stored dependent the call does not redefine still checks against it: those
/// are relinked like the dependents of a same-signature replacement (pressure run 7: refusing them
/// cost a round 17 times in 12 stages). Returns the changed signatures with such dependents, for
/// [`broken_dependents`] if the check fails.
fn replacement_rule(tx: &Transaction, snap: &Snapshot, input: &Program, input_names: &HashSet<&str>, removed: &std::collections::BTreeSet<String>) -> Result<Vec<Changed>, Vec<Diagnostic>> {
    let mut diags = Vec::new();
    let mut changed = Vec::new();
    let mut seen = HashSet::new();
    for d in &input.defs {
        if !snap.names.contains_key(d.name()) || !seen.insert(d.name()) {
            continue;
        }
        let old = snap.decode_name(tx, d.name())?;
        let (before, after) = (signature(&old), signature(d));
        if before == after {
            continue;
        }
        if before.0 != after.0 {
            diags.push(Diagnostic::kind_change(d.name(), before.0.as_str(), after.0.as_str()));
            continue;
        }
        let dependents: Vec<String> = snap.dependents(d.name()).into_iter().filter(|n| !input_names.contains(n.as_str()) && !removed.contains(n)).collect();
        if !dependents.is_empty() {
            changed.push(Changed { name: d.name().to_string(), before: show(&before), after: show(&after), dependents });
        }
    }
    if diags.is_empty() {
        Ok(changed)
    } else {
        Err(diags)
    }
}

/// The check failed: E0401 `signature_change` for each changed signature with the stored dependents
/// that do not check (their diagnostics follow), then the check's diagnostics.
fn broken_dependents(changed: &[Changed], check: Vec<Diagnostic>) -> Vec<Diagnostic> {
    let failing: HashSet<&str> = check.iter().map(|d| d.def.as_str()).collect();
    let mut out: Vec<Diagnostic> = changed
        .iter()
        .filter_map(|c| {
            let broken: Vec<String> = c.dependents.iter().filter(|n| failing.contains(n.as_str())).cloned().collect();
            (!broken.is_empty()).then(|| Diagnostic::signature_change(&c.name, &c.before, &c.after, &broken))
        })
        .collect();
    out.extend(check);
    out
}

/// Types are identified by their structure: a second name for the same type would need type
/// aliases, which Phase 1 does not have. A standard library type and a project type with the same
/// structure are allowed (review finding C1): std is written independently of every project, and
/// each is a distinct nominal type to the checker; stored definitions keep the name they were
/// written against (`names.refs`), so both decode as written.
fn identical_types(canons: &[Canon]) -> Result<(), Vec<Diagnostic>> {
    let mut by_hash: BTreeMap<(Hash, bool), Vec<&str>> = BTreeMap::new();
    for c in canons.iter().filter(|c| c.kind == Kind::Type) {
        by_hash.entry((c.hash, is_std_name(&c.name))).or_default().push(&c.name);
    }
    let mut diags = Vec::new();
    for names in by_hash.values_mut().filter(|ns| ns.len() > 1) {
        names.sort_unstable();
        for dup in &names[1..] {
            diags.push(
                Diagnostic::new("E0202", "duplicate_name", dup, "", format!("type `{dup}` is identical to type `{}`", names[0]))
                    .hint(format!("types with the same fields or variants are the same type: use `{}`, or rename it with `qal rename {} {dup}`", names[0], names[0])),
            );
        }
    }
    if diags.is_empty() {
        Ok(())
    } else {
        Err(diags)
    }
}

fn json(list: &[String]) -> String {
    serde_json::Value::from(list.to_vec()).to_string()
}

/// The names a definition is stored as written against: one per dependency hash (names are unique
/// per hash, which E0404 `alias_reference` guarantees), sorted.
fn ref_names(c: &Canon, hash_of: &HashMap<&str, Hash>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out: Vec<String> = c.refs.iter().filter(|n| hash_of.get(n.as_str()).is_some_and(|h| c.deps.contains(h))).inspect(|n| debug_assert!(seen.insert(hash_of[n.as_str()]), "`{}` names one definition twice", c.name)).cloned().collect();
    out.sort();
    out
}

/// Writes new definitions, dependencies, names and history; returns the per-name results:
/// the input definitions in input order, then relinked dependents by name. `dry`: writes nothing.
fn write(tx: &Transaction, snap: &Snapshot, input: &Program, hashed: &Hashed, agent: &str, space: &Space, dry: bool) -> Result<Vec<Defined>, Vec<Diagnostic>> {
    let now = now_ms();
    let canons = &hashed.defs;
    let input_names: HashSet<&str> = input.defs.iter().map(|d| d.name()).collect();
    if !dry {
        store_defs(tx, hashed)?;
    }
    let hash_of: HashMap<&str, Hash> = canons.iter().map(|c| (c.name.as_str(), c.hash)).collect();
    let mut results: HashMap<&str, Defined> = HashMap::new();
    for c in canons {
        let old = snap.names.get(&c.name);
        let refs = ref_names(c, &hash_of);
        let is_input = input_names.contains(c.name.as_str());
        let status = match old {
            None => "added",
            Some(e) if e.hash == c.hash => "unchanged",
            Some(_) if is_input => "replaced",
            Some(_) => "relinked",
        };
        if status == "unchanged" && !is_input {
            continue;
        }
        let old_hash = old.map(|e| e.hash.to_hex());
        // An input that restates a conflict-marked name of its change set resolves the mark even
        // when nothing else changes (the owner keeps its own version).
        let resolved = is_input && status == "unchanged" && marked(tx, space, &c.name)?;
        if !dry {
            if resolved {
                crate::space::put(tx, space, &c.name, &c.hash.to_hex(), &json(&c.local_names), &json(&refs), true, now)?;
            } else if status != "unchanged" || old.is_some_and(|e| e.locals != c.local_names || e.refs != refs) {
                // An input whose hash did not change (only its local names or refs) counts like a
                // relink in a change set: verify replays it from an `updated` entry.
                crate::space::put(tx, space, &c.name, &c.hash.to_hex(), &json(&c.local_names), &json(&refs), is_input && status != "unchanged", now)?;
            }
            if status != "unchanged" {
                tx.execute(
                    "INSERT INTO history (name, action, old_hash, new_hash, agent, at, space) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    params![c.name, status, old_hash, c.hash.to_hex(), agent, now, space.id()],
                )
                .map_err(sql)?;
            }
        }
        results.insert(&c.name, Defined { name: c.name.clone(), kind: c.kind.as_str(), hash: c.hash.to_hex(), status, old_hash: if matches!(status, "replaced" | "relinked") { old_hash } else { None }, resolved });
    }
    let mut out = Vec::new();
    for d in &input.defs {
        if let Some(r) = results.remove(d.name()) {
            out.push(r);
        }
    }
    let mut relinked: Vec<Defined> = results.into_values().collect();
    relinked.sort_by(|a, b| a.name.cmp(&b.name));
    out.extend(relinked);
    Ok(out)
}

/// True if `name` has a conflict mark in the change set `space` (never on `main`).
fn marked(tx: &Transaction, space: &Space, name: &str) -> Result<bool, Vec<Diagnostic>> {
    let Space::Change(c) = space else { return Ok(false) };
    let n: i64 = tx.query_row("SELECT count(*) FROM change_names WHERE change = ?1 AND name = ?2 AND conflict IS NOT NULL", params![c, name], |r| r.get(0)).map_err(sql)?;
    Ok(n > 0)
}

/// Stores the groups, definitions and dependencies of a hashed program (those already stored stay).
pub(crate) fn store_defs(tx: &Transaction, hashed: &Hashed) -> Result<(), Vec<Diagnostic>> {
    let now = now_ms();
    let canons = &hashed.defs;
    for (h, form) in &hashed.groups {
        tx.execute("INSERT OR IGNORE INTO groups (hash, form) VALUES (?1, ?2)", params![h.to_hex(), form]).map_err(sql)?;
    }
    for c in canons {
        tx.execute(
            "INSERT OR IGNORE INTO defs (hash, kind, canonical, signature, effects, local_names, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![c.hash.to_hex(), c.kind.as_str(), c.canonical, c.signature, json(&c.effects), json(&c.local_names), now],
        )
        .map_err(sql)?;
    }
    for c in canons {
        for d in &c.deps {
            tx.execute("INSERT OR IGNORE INTO deps (hash, dep_hash) VALUES (?1, ?2)", params![c.hash.to_hex(), d.to_hex()]).map_err(sql)?;
        }
    }
    Ok(())
}
