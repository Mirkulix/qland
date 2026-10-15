//! `define`: parse, assemble with the stored codebase, enforce the replacement rule, check, hash,
//! and write everything in one transaction (spec §6).

use crate::snapshot::Snapshot;
use crate::{now_ms, sql, Defined};
use qal_core::ast::{Def, Program};
use qal_core::canon::{self, Canon, Hash, HashError, Hashed, Kind};
use qal_core::diag::Diagnostic;
use rusqlite::{params, Connection, Transaction, TransactionBehavior};
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

pub fn define(conn: &mut Connection, src: &str, agent: &str) -> Result<Vec<Defined>, Vec<Diagnostic>> {
    let input = qal_text::parse_program(src).map_err(|d| vec![d])?;
    // IMMEDIATE takes the write lock before reading, so two concurrent `define`s serialize instead of
    // both deciding on the same old state.
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
    let snap = Snapshot::load(&tx)?;

    let input_names: HashSet<&str> = input.defs.iter().map(|d| d.name()).collect();
    // Hashes with several names, all redefined, where one name is restated unchanged: that name
    // keeps the hash (and its users). Found by a trial hash of the assembled program.
    let mut keep: HashMap<Hash, String> = HashMap::new();
    let (mut overrides, mut replaced) = plan_overrides(&snap, &input_names, &keep);
    let shared: Vec<Hash> = replaced.iter().filter(|h| snap.by_hash.get(h).is_some_and(|ns| ns.len() > 1)).copied().collect();
    if !shared.is_empty() {
        let mut trial = snap.program(&tx, &overrides, &|n| input_names.contains(n))?;
        trial.defs.extend(input.defs.iter().cloned());
        if let Ok(t) = canon::hash_program(&trial) {
            let new_of: HashMap<&str, Hash> = t.defs.iter().map(|c| (c.name.as_str(), c.hash)).collect();
            let mut diags = Vec::new();
            for h in &shared {
                let names = &snap.by_hash[h];
                if let Some(same) = names.iter().find(|n| new_of.get(n.as_str()) == Some(h)) {
                    keep.insert(*h, same.clone());
                    continue;
                }
                let news: HashSet<Hash> = names.iter().filter_map(|n| new_of.get(n.as_str()).copied()).collect();
                let users: Vec<String> = snap.dependents(&tx, h)?.into_iter().filter(|n| !input_names.contains(n.as_str())).collect();
                if news.len() > 1 && !users.is_empty() {
                    let mut d = Diagnostic::new(
                        "E0403",
                        "ambiguous_replacement",
                        &names[0],
                        "",
                        format!("{} were the same definition and now differ; its stored users cannot follow both", names.iter().map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", ")),
                    )
                    .hint(format!("redefine {} in the same call, or keep one of the names unchanged", users.join(", ")));
                    d.dependents = users;
                    diags.push(d);
                }
            }
            if !diags.is_empty() {
                return Err(diags);
            }
        }
        (overrides, replaced) = plan_overrides(&snap, &input_names, &keep);
    }

    replacement_rule(&tx, &snap, &input, &input_names, &overrides, &replaced)?;

    let stored = snap.program(&tx, &overrides, &|n| input_names.contains(n))?;
    let mut combined = stored;
    combined.defs.extend(input.defs.iter().cloned());
    qal_check::check(&combined)?;

    let hashed = hash(&combined)?;
    identical_types(&hashed.defs)?;
    group_aliases(&hashed.defs)?;
    canonical_check(&tx, &snap, &input_names, &replaced, &hashed.defs)?;

    let out = write(&tx, &snap, &input, &hashed, agent)?;
    tx.commit().map_err(sql)?;
    Ok(out)
}

/// Which stored hashes the input replaces, and the names their users are decoded with. A hash is
/// replaced only if all its names are redefined (and none is kept unchanged, `keep`); otherwise its
/// users keep a name that still has it.
fn plan_overrides(snap: &Snapshot, input_names: &HashSet<&str>, keep: &HashMap<Hash, String>) -> (HashMap<Hash, String>, HashSet<Hash>) {
    let mut overrides = HashMap::new();
    let mut replaced = HashSet::new();
    for (h, names) in &snap.by_hash {
        let (redefined, kept): (Vec<&String>, Vec<&String>) = names.iter().partition(|n| input_names.contains(n.as_str()));
        let Some(first) = redefined.first() else { continue };
        if let Some(k) = keep.get(h).or(kept.first().copied()) {
            overrides.insert(*h, k.clone());
        } else {
            overrides.insert(*h, (*first).clone());
            replaced.insert(*h);
        }
    }
    (overrides, replaced)
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

/// A stored definition may only get a new hash if it (transitively) uses a replaced one; anything
/// else means its stored bytes were not in canonical form.
fn canonical_check(tx: &Transaction, snap: &Snapshot, input_names: &HashSet<&str>, replaced: &HashSet<Hash>, canons: &[Canon]) -> Result<(), Vec<Diagnostic>> {
    let mut affected = replaced.clone();
    let mut refs: HashMap<Hash, Vec<Hash>> = HashMap::new();
    for (name, (h, _)) in &snap.names {
        if !input_names.contains(name.as_str()) && !refs.contains_key(h) {
            let row = snap.row(tx, h)?.ok_or_else(|| vec![Diagnostic::store_error(format!("the store is corrupt: definition #{h} is missing"))])?;
            refs.insert(*h, canon::references_of(row.form()).map_err(|e| vec![Diagnostic::store_error(e)])?);
        }
    }
    loop {
        let before = affected.len();
        for (h, rs) in &refs {
            if !affected.contains(h) && rs.iter().any(|r| affected.contains(r)) {
                affected.insert(*h);
            }
        }
        if affected.len() == before {
            break;
        }
    }
    for c in canons {
        if input_names.contains(c.name.as_str()) {
            continue;
        }
        if let Some((old, _)) = snap.names.get(&c.name) {
            if *old != c.hash && !affected.contains(old) {
                return Err(vec![Diagnostic::store_error(format!(
                    "the store is corrupt: the stored definition `{}` (#{old}) is not in canonical form",
                    c.name
                ))]);
            }
        }
    }
    Ok(())
}

/// Replacing a definition with a different signature fails unless the call also redefines every
/// stored definition that refers to it.
fn replacement_rule(
    tx: &Transaction,
    snap: &Snapshot,
    input: &Program,
    input_names: &HashSet<&str>,
    overrides: &HashMap<Hash, String>,
    replaced: &HashSet<Hash>,
) -> Result<(), Vec<Diagnostic>> {
    let mut diags = Vec::new();
    let mut seen = HashSet::new();
    for d in &input.defs {
        let Some((old_hash, _)) = snap.names.get(d.name()) else { continue };
        // A hash that keeps another name is not replaced: nothing refers to the new version yet.
        if !replaced.contains(old_hash) || !seen.insert(d.name()) {
            continue;
        }
        let old = snap.decode_name(tx, d.name(), overrides)?;
        let (before, after) = (signature(&old), signature(d));
        if before == after {
            continue;
        }
        let dependents: Vec<String> = snap.dependents(tx, old_hash)?.into_iter().filter(|n| !input_names.contains(n.as_str())).collect();
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

/// Writes new definitions, dependencies, names and history; returns the per-name results:
/// the input definitions in input order, then relinked dependents by name.
fn write(tx: &Transaction, snap: &Snapshot, input: &Program, hashed: &Hashed, agent: &str) -> Result<Vec<Defined>, Vec<Diagnostic>> {
    let now = now_ms();
    let canons = &hashed.defs;
    for (h, form) in &hashed.groups {
        tx.execute("INSERT OR IGNORE INTO groups (hash, form) VALUES (?1, ?2)", params![h.to_hex(), form]).map_err(sql)?;
    }
    let input_names: HashSet<&str> = input.defs.iter().map(|d| d.name()).collect();
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
    let mut results: HashMap<&str, Defined> = HashMap::new();
    for c in canons {
        let old = snap.names.get(&c.name);
        let is_input = input_names.contains(c.name.as_str());
        let status = match old {
            None => "added",
            Some((h, _)) if *h == c.hash => "unchanged",
            Some(_) if is_input => "replaced",
            Some(_) => "relinked",
        };
        if status == "unchanged" && !is_input {
            continue;
        }
        if status != "unchanged" || old.is_some_and(|(_, l)| *l != c.local_names) {
            tx.execute(
                "INSERT INTO names (name, hash, local_names, updated_at) VALUES (?1, ?2, ?3, ?4) \
                 ON CONFLICT(name) DO UPDATE SET hash = excluded.hash, local_names = excluded.local_names, updated_at = excluded.updated_at",
                params![c.name, c.hash.to_hex(), json(&c.local_names), now],
            )
            .map_err(sql)?;
        }
        let old_hash = old.map(|(h, _)| h.to_hex());
        if status != "unchanged" {
            tx.execute(
                "INSERT INTO history (name, action, old_hash, new_hash, agent, at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![c.name, status, old_hash, c.hash.to_hex(), agent, now],
            )
            .map_err(sql)?;
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
