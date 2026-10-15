//! `define`: parse, assemble with the stored codebase, enforce the replacement rule, check, hash,
//! and write everything in one transaction (spec §6).

use crate::snapshot::Snapshot;
use crate::{now_ms, sql, Defined, Verify};
use qal_core::ast::{Def, Program};
use qal_core::canon::{self, Canon, Hash, HashError, Hashed, Kind};
use qal_core::diag::Diagnostic;
use qal_core::tast::TProgram;
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use std::collections::{BTreeMap, HashMap, HashSet};

/// A definition's signature as the replacement rule compares it: kind plus the signature text
/// with current names (so a replaced type keeps its name in the signatures that use it).
fn signature(d: &Def) -> (Kind, String) {
    (Kind::of(d), canon::signature_text(d, &|n| n.to_string()))
}

fn show(sig: &(Kind, String)) -> String {
    match sig.0 {
        Kind::Fn => sig.1.clone(),
        Kind::Type => format!("type = {}", sig.1),
        Kind::Test => "test".to_string(),
    }
}

/// Hashes a checked program; interchangeable recursive definitions are a diagnostic.
pub(crate) fn hash(p: &Program) -> Result<Hashed, Vec<Diagnostic>> {
    canon::hash_program(p).map_err(|e| match e {
        HashError::Symmetric(names) => vec![Diagnostic::symmetric_group(&names)],
        HashError::TooLarge(names) => vec![Diagnostic::group_too_large(&names)],
        HashError::Internal(m) => vec![Diagnostic::store_error(format!("cannot hash a checked program: {m}"))],
    })
}

/// How often `define` verifies on a snapshot, outside the write lock, before it verifies inside it.
const ATTEMPTS: usize = 3;

pub fn define(conn: &mut Connection, src: &str, agent: &str, verify: Option<&mut Verify>) -> Result<Vec<Defined>, Vec<Diagnostic>> {
    define_checked(conn, src, agent, verify, &|_, _| Ok(()))
}

/// A check of a `define` before it is committed, inside the write transaction that would commit it
/// (it can also write more in the same transaction; used by package imports).
pub type Check<'c> = dyn Fn(&[Defined], &Transaction) -> Result<(), Vec<Diagnostic>> + 'c;

/// `define`, committed only if `check` accepts the written result (after the verification).
pub fn define_checked(conn: &mut Connection, src: &str, agent: &str, verify: Option<&mut Verify>, check: &Check) -> Result<Vec<Defined>, Vec<Diagnostic>> {
    let input = qal_text::parse_program(src).map_err(|d| vec![d])?;
    let Some(v) = verify else {
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
        let out = prepare(&tx, &input)?.write(&tx, &input, agent, false)?;
        check(&out, &tx)?;
        tx.commit().map_err(sql)?;
        return Ok(out);
    };
    // The verifier (the prover) may take its whole time budget, longer than another writer waits
    // for the lock (review finding): it runs on a snapshot without the lock, and the results are
    // written by a short IMMEDIATE transaction if the definitions to verify are still the same
    // (results depend only on a definition's hash). A rejection rejects the `define` as it would
    // have been on that snapshot.
    for _ in 0..ATTEMPTS {
        let (typed, todo) = {
            // A deferred transaction that only reads: a consistent snapshot, no write lock.
            let tx = conn.transaction().map_err(sql)?;
            let state = prepare(&tx, &input)?;
            let out = state.write(&tx, &input, agent, true)?;
            let todo = to_verify(&tx, &out, &v.key)?;
            (state.typed, todo)
        };
        let rows = if todo.is_empty() { Vec::new() } else { (v.run)(&typed, &todo)? };
        // IMMEDIATE takes the write lock before reading, so two concurrent `define`s serialize
        // instead of both deciding on the same old state.
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
        let out = prepare(&tx, &input)?.write(&tx, &input, agent, false)?;
        let now = to_verify(&tx, &out, &v.key)?;
        if now.iter().all(|d| todo.contains(d)) {
            let hashes: HashSet<&str> = now.iter().map(|(_, h)| h.as_str()).collect();
            store_results(&tx, &v.key, rows.into_iter().filter(|(h, _)| hashes.contains(h.as_str())))?;
            check(&out, &tx)?;
            tx.commit().map_err(sql)?;
            return Ok(out);
        }
        // Another writer changed what has to be verified: drop this transaction and start over.
    }
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
    let state = prepare(&tx, &input)?;
    let out = state.write(&tx, &input, agent, false)?;
    let todo = to_verify(&tx, &out, &v.key)?;
    if !todo.is_empty() {
        store_results(&tx, &v.key, (v.run)(&state.typed, &todo)?.into_iter())?;
    }
    check(&out, &tx)?;
    tx.commit().map_err(sql)?;
    Ok(out)
}

/// The checked and hashed codebase that `define` would write.
struct Prepared {
    snap: Snapshot,
    typed: TProgram,
    hashed: Hashed,
}

impl Prepared {
    /// Writes the definitions (`dry`: only computes the results).
    fn write(&self, tx: &Transaction, input: &Program, agent: &str, dry: bool) -> Result<Vec<Defined>, Vec<Diagnostic>> {
        write(tx, &self.snap, input, &self.hashed, agent, dry)
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
fn prepare(tx: &Transaction, input: &Program) -> Result<Prepared, Vec<Diagnostic>> {
    let snap = Snapshot::load(tx)?;

    let input_names: HashSet<&str> = input.defs.iter().map(|d| d.name()).collect();
    replacement_rule(tx, &snap, input, &input_names)?;

    // Stored definitions keep the names they were written against; the ones that refer to a
    // redefined name get new hashes when the combined program is hashed.
    let stored = snap.program(tx, &|n| input_names.contains(n))?;
    let mut combined = stored;
    combined.defs.extend(input.defs.iter().cloned());
    // Only the definitions written now may not take prelude names (older stored ones keep working).
    let typed = qal_check::check_stored(&combined, &|n| input_names.contains(n) && !legacy_writer())?;

    let hashed = hash(&combined)?;
    identical_types(&hashed.defs)?;
    group_aliases(&hashed.defs)?;
    alias_references(&hashed.defs, &input_names)?;
    canonical_check(&snap, &input_names, &hashed.defs)?;
    Ok(Prepared { snap, typed, hashed })
}

/// The (name, hash) of the definitions to verify: the functions and tests that are new, got a new
/// hash, or have no stored result for `key`.
fn to_verify(tx: &Transaction, out: &[Defined], key: &str) -> Result<Vec<(String, String)>, Vec<Diagnostic>> {
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
fn store_results(tx: &Transaction, key: &str, rows: impl Iterator<Item = (String, String)>) -> Result<(), Vec<Diagnostic>> {
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
            diags.push(Diagnostic::alias_reference(&c.name, "", &names, &h.short(), input_names.contains(c.name.as_str())));
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
fn group_aliases(canons: &[Canon]) -> Result<(), Vec<Diagnostic>> {
    let mut by_hash: BTreeMap<Hash, Vec<&str>> = BTreeMap::new();
    for c in canons.iter().filter(|c| c.group.is_some()) {
        by_hash.entry(c.hash).or_default().push(&c.name);
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

/// Replacing a definition with a different signature fails unless the call also redefines every
/// stored definition written against it.
fn replacement_rule(tx: &Transaction, snap: &Snapshot, input: &Program, input_names: &HashSet<&str>) -> Result<(), Vec<Diagnostic>> {
    let mut diags = Vec::new();
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
        let dependents: Vec<String> = snap.dependents(d.name()).into_iter().filter(|n| !input_names.contains(n.as_str())).collect();
        if !dependents.is_empty() {
            diags.push(Diagnostic::signature_change(d.name(), &show(&before), &show(&after), &dependents));
        }
    }
    if diags.is_empty() {
        Ok(())
    } else {
        Err(diags)
    }
}

/// Types are identified by their structure: a second name for the same type would need type
/// aliases, which Phase 1 does not have.
fn identical_types(canons: &[Canon]) -> Result<(), Vec<Diagnostic>> {
    let mut by_hash: BTreeMap<Hash, Vec<&str>> = BTreeMap::new();
    for c in canons.iter().filter(|c| c.kind == Kind::Type) {
        by_hash.entry(c.hash).or_default().push(&c.name);
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
fn write(tx: &Transaction, snap: &Snapshot, input: &Program, hashed: &Hashed, agent: &str, dry: bool) -> Result<Vec<Defined>, Vec<Diagnostic>> {
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
        if !dry {
            if status != "unchanged" || old.is_some_and(|e| e.locals != c.local_names || e.refs != refs) {
                tx.execute(
                    "INSERT INTO names (name, hash, local_names, refs, updated_at) VALUES (?1, ?2, ?3, ?4, ?5) \
                     ON CONFLICT(name) DO UPDATE SET hash = excluded.hash, local_names = excluded.local_names, refs = excluded.refs, updated_at = excluded.updated_at",
                    params![c.name, c.hash.to_hex(), json(&c.local_names), json(&refs), now],
                )
                .map_err(sql)?;
            }
            if status != "unchanged" {
                tx.execute(
                    "INSERT INTO history (name, action, old_hash, new_hash, agent, at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![c.name, status, old_hash, c.hash.to_hex(), agent, now],
                )
                .map_err(sql)?;
            }
        }
        results.insert(&c.name, Defined { name: c.name.clone(), kind: c.kind.as_str(), hash: c.hash.to_hex(), status, old_hash: if matches!(status, "replaced" | "relinked") { old_hash } else { None } });
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

/// Stores the groups, definitions and dependencies of a hashed program (those already stored stay).
fn store_defs(tx: &Transaction, hashed: &Hashed) -> Result<(), Vec<Diagnostic>> {
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
