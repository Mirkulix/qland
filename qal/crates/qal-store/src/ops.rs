//! `rename`, `view`, `query` and `history` (spec §9: the operations `qal_*` tools are built on).

use crate::pattern;
use crate::snapshot::Snapshot;
use crate::{now_ms, sql};
use qal_core::ast::{Def, Program};
use qal_core::canon::{self, Hash, Kind};
use qal_core::diag::Diagnostic;
use qal_text::lexer::KEYWORDS;
use rusqlite::{params, Connection, TransactionBehavior};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Renamed {
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
    /// `added`, `replaced`, `relinked` or `renamed`.
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
    qal_text::print_program(&Program { defs: vec![d.clone()] }).trim_end().to_string()
}

fn missing_row(h: &Hash) -> Vec<Diagnostic> {
    vec![Diagnostic::store_error(format!("the store is corrupt: definition #{h} is missing"))]
}

/// A name or `#<64 hex digits>` -> the hash it stands for.
fn resolve(snap: &Snapshot, conn: &Connection, s: &str) -> Result<Hash, Vec<Diagnostic>> {
    if let Some(hex) = s.strip_prefix('#') {
        if let Some(h) = Hash::from_hex(hex) {
            if snap.row(conn, &h)?.is_some() {
                return Ok(h);
            }
        }
        return Err(vec![Diagnostic::unknown_name("", "", s).hint("give a stored name or `#` followed by a full 64-digit hash")]);
    }
    snap.names.get(s).map(|e| e.hash).ok_or_else(|| vec![Diagnostic::unknown_name(s, "", s)])
}

/// Names follow the text form's rules: types start uppercase, functions and tests lowercase.
fn valid_name(name: &str, kind: Kind) -> Result<(), Vec<Diagnostic>> {
    let what = match kind {
        Kind::Type => "type",
        Kind::Fn => "function",
        Kind::Test => "test",
    };
    let chars_ok = name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !name.starts_with(|c: char| c.is_ascii_digit()) && name != "_";
    let case_ok = !name.is_empty() && name.starts_with(|c: char| c.is_ascii_uppercase()) == (kind == Kind::Type);
    if chars_ok && case_ok && !KEYWORDS.contains(&name) {
        return Ok(());
    }
    let rule = if kind == Kind::Type { "start with an uppercase letter" } else { "start with a lowercase letter or `_`" };
    Err(vec![Diagnostic::new("E0100", "parse", name, "", format!("`{name}` is not a valid {what} name"))
        .hint(format!("{what} names {rule} and contain only ASCII letters, digits and `_` (no keywords)"))])
}

pub fn rename(conn: &mut Connection, old: &str, new: &str, agent: &str) -> Result<Renamed, Vec<Diagnostic>> {
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
    let snap = Snapshot::load(&tx)?;
    let Some(hash) = snap.names.get(old).map(|e| e.hash) else {
        return Err(vec![Diagnostic::unknown_name(old, "", old)]);
    };
    let kind = snap.row(&tx, &hash)?.map(|r| r.kind).ok_or_else(|| vec![Diagnostic::store_error(format!("`{old}` has no definition"))])?;
    if snap.names.contains_key(new) {
        return Err(vec![Diagnostic::duplicate_name(new, "", new).hint(format!("`{new}` is already taken; rename or replace it first"))]);
    }
    valid_name(new, kind)?;
    // The renamed codebase must still check (built-in and reserved names, `main`'s signature) and
    // hash exactly as before.
    let renamed = snap.renamed(old, new);
    let program = renamed.program(&tx, &|_| false)?;
    qal_check::check(&program)?;
    let canons = crate::define::hash(&program)?.defs;
    for c in &canons {
        if renamed.names.get(&c.name).map(|e| e.hash) != Some(c.hash) {
            return Err(vec![Diagnostic::store_error(format!("renaming would change the hash of `{}`; nothing was changed", c.name))]);
        }
    }
    let now = now_ms();
    tx.execute("UPDATE names SET name = ?1, updated_at = ?2 WHERE name = ?3", params![new, now, old]).map_err(sql)?;
    // The users written against `old` now read `new`.
    for user in snap.dependents(old) {
        let refs = &renamed.names[&user].refs;
        tx.execute("UPDATE names SET refs = ?1 WHERE name = ?2", params![serde_json::Value::from(refs.clone()).to_string(), user]).map_err(sql)?;
    }
    tx.execute("UPDATE history SET name = ?1 WHERE name = ?2", params![new, old]).map_err(sql)?;
    tx.execute(
        "INSERT INTO history (name, action, old_hash, new_hash, from_name, to_name, agent, at) VALUES (?1, 'renamed', ?2, ?2, ?3, ?1, ?4, ?5)",
        params![new, hash.to_hex(), old, agent, now],
    )
    .map_err(sql)?;
    tx.commit().map_err(sql)?;
    Ok(Renamed { name: new.to_string(), kind: kind.as_str(), hash: hash.to_hex() })
}

pub fn view(conn: &mut Connection, names: &[String]) -> Result<Vec<View>, Vec<Diagnostic>> {
    let tx = conn.transaction().map_err(sql)?;
    let snap = Snapshot::load(&tx)?;
    let mut out = Vec::new();
    let mut diags = Vec::new();
    for n in names {
        let hash = match resolve(&snap, &tx, n) {
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

pub fn query(conn: &mut Connection, q: &Query) -> Result<Vec<Entry>, Vec<Diagnostic>> {
    let tx = conn.transaction().map_err(sql)?;
    let snap = Snapshot::load(&tx)?;
    // `uses` is a name (the definitions written against it) or `#<hash>` (those that refer to the hash).
    let uses = q.uses.as_deref().map(|u| resolve(&snap, &tx, u).map(|h| (u, h))).transpose()?;
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

pub fn history(conn: &mut Connection, name: &str) -> Result<Vec<HistoryEntry>, Vec<Diagnostic>> {
    let tx = conn.transaction().map_err(sql)?;
    let mut stmt = tx.prepare("SELECT action, name, from_name, to_name, old_hash, new_hash, agent, at FROM history WHERE name = ?1 ORDER BY id").map_err(sql)?;
    let rows = stmt
        .query_map(params![name], |r| {
            Ok(HistoryEntry { action: r.get(0)?, name: r.get(1)?, from_name: r.get(2)?, to_name: r.get(3)?, old_hash: r.get(4)?, new_hash: r.get(5)?, agent: r.get(6)?, at: r.get(7)? })
        })
        .map_err(sql)?;
    let out = rows.collect::<Result<Vec<_>, _>>().map_err(sql)?;
    if out.is_empty() {
        return Err(vec![Diagnostic::unknown_name(name, "", name)]);
    }
    Ok(out)
}
