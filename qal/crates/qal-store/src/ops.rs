//! `rename`, `view`, `query` and `history` (spec §9: the operations `qal_*` tools are built on).

use crate::pattern;
use crate::snapshot::Snapshot;
use crate::space::{self, Space};
use crate::team::Entry as Change;
use crate::{now_ms, sql};
use qal_core::ast::{Def, Program};
use qal_core::canon::{self, Hash, Kind};
use qal_core::diag::Diagnostic;
use qal_compiler::lexer::{is_std_name, std_parts, KEYWORDS, MAX_NAME_LEN};
use rusqlite::{params, Connection, TransactionBehavior};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Renamed {
    pub name: String,
    pub kind: &'static str,
    pub hash: String,
}

/// A name removed by `delete` (its definition stays stored by hash).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Deleted {
    pub name: String,
    pub kind: &'static str,
    pub hash: String,
}

/// A definition's text form plus the signatures of its direct dependencies.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct View {
    pub name: String,
    pub kind: &'static str,
    pub hash: String,
    pub signature: String,
    pub effects: Vec<String>,
    pub text: String,
    pub deps: Vec<DepView>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DepView {
    pub name: String,
    pub kind: &'static str,
    pub hash: String,
    pub signature: String,
}

/// Filters of `query`; all given filters must match.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Query {
    /// Name prefix (plain text, no wildcards).
    pub name: Option<String>,
    /// Signature pattern, e.g. `fn(Str) -> _` (see `signature_matches`).
    pub type_pattern: Option<String>,
    /// A name or `#<hash>`: only definitions that refer to it directly.
    pub uses: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Entry {
    pub name: String,
    pub kind: &'static str,
    pub hash: String,
    pub signature: String,
    pub effects: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HistoryEntry {
    /// `added`, `replaced`, `relinked`, `renamed` or `deleted`.
    pub action: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub old_hash: Option<String>,
    pub new_hash: String,
    pub agent: String,
    /// Unix time in milliseconds.
    pub at: i64,
}

/// The signature with current names: `fn(Str) -> Int !io`, a type's body, or `Bool` for tests.
pub(crate) fn signature(d: &Def) -> String {
    canon::signature_text(d, &|n| n.to_string())
}

fn effects(d: &Def) -> Vec<String> {
    match d {
        Def::Fn(f) => f.effects.clone(),
        _ => Vec::new(),
    }
}

fn text(d: &Def) -> String {
    qal_compiler::print_program(&Program { defs: vec![d.clone()] }).trim_end().to_string()
}

fn missing_row(h: &Hash) -> Vec<Diagnostic> {
    vec![Diagnostic::store_error(format!("the store is corrupt: definition #{h} is missing"))]
}

/// The current name of a definition that was once called `old` and renamed (following chains of
/// renames), if it still has that name.
pub(crate) fn renamed_to(conn: &Connection, space: &Space, names: &dyn Fn(&str) -> bool, old: &str) -> Result<Option<String>, Vec<Diagnostic>> {
    // Follow the latest rename of each name until the chain ends or comes back to a name already
    // seen (renames back and forth); the answer is the last name on the way that still exists.
    let mut seen = std::collections::HashSet::from([old.to_string()]);
    let mut current = old.to_string();
    let mut found = None;
    for _ in 0..1024 {
        let next: Option<String> = conn
            .query_row(&format!("SELECT to_name FROM history WHERE action = 'renamed' AND from_name = ?1 AND {} ORDER BY id DESC LIMIT 1", space::history_filter(space, 2)), params![current, space.id()], |r| r.get(0))
            .map(Some)
            .or_else(|e| if e == rusqlite::Error::QueryReturnedNoRows { Ok(None) } else { Err(e) })
            .map_err(sql)?;
        let Some(n) = next else { break };
        if names(&n) {
            found = Some(n.clone());
        }
        if !seen.insert(n.clone()) {
            break;
        }
        current = n;
    }
    Ok(found)
}

/// `unknown_name` for `name`, with a hint if the definition was renamed.
pub(crate) fn unknown(conn: &Connection, space: &Space, names: &dyn Fn(&str) -> bool, name: &str) -> Diagnostic {
    let d = Diagnostic::unknown_name(name, "", name);
    match renamed_to(conn, space, names, name) {
        Ok(Some(new)) => d.hint(format!("`{name}` was renamed to `{new}`")),
        _ => {
            let last: Option<String> = conn.query_row(&format!("SELECT action FROM history WHERE name = ?1 AND {} ORDER BY id DESC LIMIT 1", space::history_filter(space, 2)), params![name, space.id()], |r| r.get(0)).ok();
            if last.as_deref() == Some("deleted") {
                d.hint(format!("`{name}` was deleted; its history and old versions remain (history of `{name}`, view `#<hash>`)"))
            } else {
                d
            }
        }
    }
}

/// A name or `#<64 hex digits>` -> the hash it stands for.
fn resolve(snap: &Snapshot, conn: &Connection, space: &Space, s: &str) -> Result<Hash, Vec<Diagnostic>> {
    if let Some(hex) = s.strip_prefix('#') {
        if let Some(h) = Hash::from_hex(hex) {
            if snap.row(conn, &h)?.is_some() {
                return Ok(h);
            }
        }
        return Err(vec![Diagnostic::unknown_name("", "", s).hint("give a stored name or `#` followed by a full 64-digit hash")]);
    }
    snap.names.get(s).map(|e| e.hash).ok_or_else(|| vec![unknown(conn, space, &|n| snap.names.contains_key(n), s)])
}

/// Names follow the text form's rules: types start uppercase, functions and tests lowercase.
fn valid_name(name: &str, kind: Kind) -> Result<(), Vec<Diagnostic>> {
    let what = match kind {
        Kind::Type => "type",
        Kind::Fn => "function",
        Kind::Test => "test",
    };
    // A qualified `std.<module>.<name>` follows the rules in its last segment.
    let last = std_parts(name).map_or(name, |(_, last)| last);
    let chars_ok = last.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !last.starts_with(|c: char| c.is_ascii_digit()) && last != "_" && name.len() <= MAX_NAME_LEN;
    let case_ok = !last.is_empty() && last.starts_with(|c: char| c.is_ascii_uppercase()) == (kind == Kind::Type);
    if chars_ok && case_ok && !KEYWORDS.contains(&last) {
        return Ok(());
    }
    let rule = if kind == Kind::Type { "start with an uppercase letter" } else { "start with a lowercase letter or `_`" };
    Err(vec![Diagnostic::new("E0100", "parse", name, "", format!("`{name}` is not a valid {what} name"))
        .hint(format!("{what} names {rule} and contain only ASCII letters, digits and `_`, at most {MAX_NAME_LEN} (no keywords); standard library names are `std.<module>.<name>`"))])
}

/// Removes the names `names` (one transaction): only if no other stored definition refers to one
/// of them (E0406 `has_dependents` otherwise), and only names the store's namespace rule allows
/// (no `std.` names in a project; in the std project every name is one). Each gets a `deleted`
/// history entry; the definitions stay stored by hash but have no name any more.
/// `post` runs last inside the transaction with the changes (the signed record of a team store).
pub fn delete(conn: &mut Connection, space: &Space, names: &[String], agent: &str, rules: &crate::define::Rules, post: &Post) -> Result<Vec<Deleted>, Vec<Diagnostic>> {
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
    let snap = Snapshot::load(&tx, space)?;
    let std_store = crate::define::is_std_store(&tx)?;
    let wanted: std::collections::BTreeSet<&str> = names.iter().map(String::as_str).collect();
    let mut diags = Vec::new();
    let mut out = Vec::new();
    for &name in &wanted {
        // The namespace rule first: a `std.` name is read-only in a project, stored or not.
        if std_store && !is_std_name(name) {
            diags.push(Diagnostic::std_only(name, "", name, None));
            continue;
        }
        if !std_store && is_std_name(name) {
            diags.push(Diagnostic::std_reserved(name, "", name));
            continue;
        }
        let Some(entry) = snap.names.get(name) else {
            diags.push(unknown(&tx, space, &|n| snap.names.contains_key(n), name));
            continue;
        };
        // A released std name stays (std is additive), unless the break is deliberate.
        if let (true, Some(release), false) = (std_store, rules.release, rules.breaking) {
            if release.signatures.contains_key(name) {
                diags.push(Diagnostic::std_released_delete(name, &release.hash));
                continue;
            }
        }
        let users: Vec<String> = snap.dependents(name).into_iter().filter(|u| !wanted.contains(u.as_str())).collect();
        if !users.is_empty() {
            diags.push(Diagnostic::has_dependents(name, &users));
            continue;
        }
        let kind = snap.row(&tx, &entry.hash)?.map(|r| r.kind).ok_or_else(|| vec![Diagnostic::store_error(format!("`{name}` has no definition"))])?;
        out.push(Deleted { name: name.to_string(), kind: kind.as_str(), hash: entry.hash.to_hex() });
    }
    if !diags.is_empty() {
        return Err(diags);
    }
    let now = now_ms();
    for d in &out {
        space::remove(&tx, space, &d.name, "edited", now)?;
        tx.execute(
            "INSERT INTO history (name, action, old_hash, new_hash, agent, at, space) VALUES (?1, 'deleted', ?2, ?2, ?3, ?4, ?5)",
            params![d.name, d.hash, agent, now, space.id()],
        )
        .map_err(sql)?;
    }
    let entries: Vec<Change> = out.iter().map(|d| Change::label(&d.name, Some(&d.hash), None, "deleted")).collect();
    post(&tx, &entries, &crate::team::rows_of(&snap))?;
    tx.commit().map_err(sql)?;
    Ok(out)
}

/// A step run last inside the write transaction of `rename`/`delete`, with the name changes.
pub type Post<'c> = dyn Fn(&rusqlite::Transaction, &[Change], &crate::team::Rows) -> Result<(), Vec<Diagnostic>> + 'c;

/// `post` runs last inside the transaction (the signed record of a team store): the old name goes
/// (`renamed`), the new one comes (`added`), with the same hash.
pub fn rename(conn: &mut Connection, space: &Space, old: &str, new: &str, agent: &str, rules: &crate::define::Rules, post: &Post) -> Result<Renamed, Vec<Diagnostic>> {
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
    let snap = Snapshot::load(&tx, space)?;
    let Some(hash) = snap.names.get(old).map(|e| e.hash) else {
        return Err(vec![unknown(&tx, space, &|n| snap.names.contains_key(n), old)]);
    };
    let kind = snap.row(&tx, &hash)?.map(|r| r.kind).ok_or_else(|| vec![Diagnostic::store_error(format!("`{old}` has no definition"))])?;
    if snap.names.contains_key(new) {
        return Err(vec![Diagnostic::duplicate_name(new, "", new).hint(format!("`{new}` is already taken; rename or replace it first"))]);
    }
    valid_name(new, kind)?;
    // `std.` names are read-only in a project; the std project has only `std.` names.
    if crate::define::is_std_store(&tx)? {
        if !is_std_name(new) {
            return Err(vec![Diagnostic::std_only(old, "", new, None)]);
        }
        // Released names stay (std is additive); a name renamed onto a released one must keep
        // its signature (review N1).
        if let (Some(release), false) = (rules.release, rules.breaking) {
            if release.signatures.contains_key(old) {
                return Err(vec![Diagnostic::std_released_rename(old, new, &release.hash)]);
            }
            if let Some(before) = release.signatures.get(new) {
                let after = crate::define::signature(&snap.decode_name(&tx, old)?);
                if *before != after {
                    return Err(vec![Diagnostic::std_signature_change(new, &crate::define::show(before), &crate::define::show(&after))
                        .hint("a deliberate break needs `qal rename --breaking` on the command line")]);
                }
            }
        }
    } else if let Some(n) = [old, new].into_iter().find(|n| is_std_name(n)) {
        return Err(vec![Diagnostic::std_reserved(old, "", n)]);
    }
    // The renamed codebase must still check (built-in and reserved names, `main`'s signature) and
    // hash exactly as before.
    let renamed = snap.renamed(old, new);
    let program = renamed.program(&tx, &|_| false)?;
    if let Err(diags) = qal_compiler::check_stored(&program, &|n| n == new) {
        // Other definitions named like built-ins (stored before those built-ins existed) do not
        // block the rename: they are renamed one at a time.
        let rest: Vec<Diagnostic> = diags.into_iter().filter(|d| !(d.is_builtin_name() && d.def != new)).collect();
        if !rest.is_empty() {
            return Err(rest);
        }
    }
    let canons = crate::define::hash(&program)?.defs;
    for c in &canons {
        if renamed.names.get(&c.name).map(|e| e.hash) != Some(c.hash) {
            return Err(vec![Diagnostic::store_error(format!("renaming would change the hash of `{}`; nothing was changed", c.name))]);
        }
    }
    let now = now_ms();
    match space {
        Space::Main => {
            tx.execute("UPDATE names SET name = ?1, updated_at = ?2 WHERE name = ?3", params![new, now, old]).map_err(sql)?;
        }
        Space::Change(_) => {
            let e = &snap.names[old];
            let json = |l: &[String]| serde_json::Value::from(l.to_vec()).to_string();
            space::remove(&tx, space, old, "renamed", now)?;
            space::put(&tx, space, new, &hash.to_hex(), &json(&e.locals), &json(&e.refs), true, now)?;
        }
    }
    // The users written against `old` now read `new`.
    for user in snap.dependents(old) {
        let refs = &renamed.names[&user].refs;
        space::set_refs(&tx, space, &user, &serde_json::Value::from(refs.clone()).to_string(), now)?;
    }
    tx.execute("UPDATE history SET name = ?1 WHERE name = ?2 AND space = ?3", params![new, old, space.id()]).map_err(sql)?;
    tx.execute(
        "INSERT INTO history (name, action, old_hash, new_hash, from_name, to_name, agent, at, space) VALUES (?1, 'renamed', ?2, ?2, ?3, ?1, ?4, ?5, ?6)",
        params![new, hash.to_hex(), old, agent, now, space.id()],
    )
    .map_err(sql)?;
    let h = hash.to_hex();
    post(&tx, &[Change::label(old, Some(&h), None, "renamed"), Change::label(new, None, Some(&h), "added")], &crate::team::rows_of(&snap))?;
    tx.commit().map_err(sql)?;
    Ok(Renamed { name: new.to_string(), kind: kind.as_str(), hash: hash.to_hex() })
}

pub fn view(conn: &mut Connection, space: &Space, names: &[String]) -> Result<Vec<View>, Vec<Diagnostic>> {
    let tx = conn.transaction().map_err(sql)?;
    let snap = Snapshot::load(&tx, space)?;
    let mut out = Vec::new();
    let mut diags = Vec::new();
    for n in names {
        let hash = match resolve(&snap, &tx, space, n) {
            Ok(h) => h,
            Err(d) => {
                diags.extend(d);
                continue;
            }
        };
        let mut deps = Vec::new();
        let def = if n.starts_with('#') {
            let row = snap.row(&tx, &hash)?.ok_or_else(|| missing_row(&hash))?;
            let def = snap.decode_hash(&tx, &hash)?.ok_or_else(|| missing_row(&hash))?;
            for d in &row.refs {
                let dep = snap.decode_hash(&tx, d)?.ok_or_else(|| vec![Diagnostic::missing_dep(def.name(), &d.to_hex())])?;
                deps.push(DepView { name: dep.name().to_string(), kind: Kind::of(&dep).as_str(), hash: d.to_hex(), signature: signature(&dep) });
            }
            def
        } else {
            for r in &snap.names[n].refs {
                let dep = snap.decode_name(&tx, r)?;
                deps.push(DepView { name: r.clone(), kind: Kind::of(&dep).as_str(), hash: snap.names[r].hash.to_hex(), signature: signature(&dep) });
            }
            snap.decode_name(&tx, n)?
        };
        deps.sort_by(|a, b| a.name.cmp(&b.name));
        out.push(View {
            name: def.name().to_string(),
            kind: Kind::of(&def).as_str(),
            hash: hash.to_hex(),
            signature: signature(&def),
            effects: effects(&def),
            text: text(&def),
            deps,
        });
    }
    if diags.is_empty() {
        Ok(out)
    } else {
        Err(diags)
    }
}

pub fn query(conn: &mut Connection, space: &Space, q: &Query) -> Result<Vec<Entry>, Vec<Diagnostic>> {
    if let Some(p) = &q.type_pattern {
        pattern::validate(p).map_err(|reason| vec![Diagnostic::bad_pattern(p, &reason)])?;
    }
    let tx = conn.transaction().map_err(sql)?;
    let snap = Snapshot::load(&tx, space)?;
    // `uses` is a name (the definitions written against it) or `#<hash>` (those that refer to the hash).
    let uses = q.uses.as_deref().map(|u| resolve(&snap, &tx, space, u).map(|h| (u, h))).transpose()?;
    let mut out = Vec::new();
    for (name, entry) in &snap.names {
        let hash = &entry.hash;
        if q.name.as_deref().is_some_and(|p| !name.starts_with(p)) {
            continue;
        }
        if let Some((text, target)) = &uses {
            let hit = if text.starts_with('#') {
                snap.row(&tx, hash)?.ok_or_else(|| missing_row(hash))?.refs.contains(target)
            } else {
                entry.refs.iter().any(|r| r == text)
            };
            if !hit {
                continue;
            }
        }
        let def = snap.decode_name(&tx, name)?;
        let sig = signature(&def);
        if q.type_pattern.as_deref().is_some_and(|p| !pattern::matches(p, &sig)) {
            continue;
        }
        out.push(Entry { name: name.clone(), kind: Kind::of(&def).as_str(), hash: hash.to_hex(), signature: sig, effects: effects(&def) });
    }
    Ok(out)
}

pub fn history(conn: &mut Connection, space: &Space, name: &str) -> Result<Vec<HistoryEntry>, Vec<Diagnostic>> {
    let tx = conn.transaction().map_err(sql)?;
    space::require(&tx, space)?;
    let mut stmt = tx.prepare(&format!("SELECT action, name, from_name, to_name, old_hash, new_hash, agent, at FROM history WHERE name = ?1 AND {} ORDER BY id", space::history_filter(space, 2))).map_err(sql)?;
    let rows = stmt
        .query_map(params![name, space.id()], |r| {
            Ok(HistoryEntry { action: r.get(0)?, name: r.get(1)?, from_name: r.get(2)?, to_name: r.get(3)?, old_hash: r.get(4)?, new_hash: r.get(5)?, agent: r.get(6)?, at: r.get(7)? })
        })
        .map_err(sql)?;
    let out = rows.collect::<Result<Vec<_>, _>>().map_err(sql)?;
    if out.is_empty() {
        let current = |n: &str| space::hash_of(&tx, space, n).ok().flatten().is_some();
        return Err(vec![unknown(&tx, space, &current, name)]);
    }
    Ok(out)
}
