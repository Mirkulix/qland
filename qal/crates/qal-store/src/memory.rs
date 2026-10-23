//! The memory of corrections (Phase 6b): a failed `define` leaves an *attempt* per diagnostic; the
//! next successful `define` of that name by the same agent turns the newest attempt into a
//! *correction* (code, path, failed text, fixed text). Later diagnostics of the same code carry the
//! most frequent correction as a hint, and `recall` lists them.
//!
//! Everything here is bounded: at most [`MAX_ATTEMPTS`] attempts per (name, agent), none older than
//! [`RETENTION_MS`], excerpts of at most [`EXCERPT_BYTES`], replies of at most [`MAX_CORRECTIONS`]
//! corrections. Callers treat a failed memory write as no memory (it never changes a `define`).

use crate::ops::RecallQuery;
use crate::{now_ms, sql, Defined};
use qal_core::ast::{Program, Def};
use qal_core::diag::Diagnostic;
use rusqlite::{params, Transaction};
use serde::{Deserialize, Serialize};

/// Attempts kept per name and agent.
pub const MAX_ATTEMPTS: i64 = 20;
/// An attempt older than this (2 hours) is dropped and never matched.
pub const RETENTION_MS: i64 = 7_200_000;
/// Longest `before`/`after` excerpt in bytes.
pub const EXCERPT_BYTES: usize = 600;
/// Most corrections a `recall` returns.
pub const MAX_CORRECTIONS: usize = 5;
/// Longest side of a hint, in characters.
const HINT_SIDE: usize = 120;
/// How many of the newest corrections of a code are looked at (bounds the work of `recall` and hints).
const SCAN: i64 = 500;

/// A failed definition: one row per diagnostic.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Attempt {
    pub name: String,
    pub code: String,
    pub path: String,
    pub message: String,
    /// The failed definition as printed (the whole source when the input has no such definition).
    pub text: String,
    pub agent: String,
    pub at: i64,
}

/// A failed definition and its fix.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Correction {
    pub id: i64,
    pub code: String,
    pub path: String,
    pub message: String,
    pub before: String,
    pub after: String,
    pub after_hash: String,
    pub agent: String,
    pub at: i64,
    /// How many corrections with this (before, after) pair there are among the newest 500 of the code (a `recall` collapses them).
    pub count: i64,
}

/// A correction from outside (the seed corpus): what a failed text looked like and what fixed it.
/// Extra fields of a record (the extractor's `agent`) are ignored.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CorrectionInput {
    pub code: String,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub message: String,
    pub before: String,
    pub after: String,
}

/// Inserts the records as corrections (hash empty, agent `corpus`, texts cut like recorded ones), all
/// or nothing. A record without a code, a `before` or an `after` is refused (E0701) and nothing is stored.
pub fn import(tx: &Transaction, records: &[CorrectionInput]) -> Result<usize, Vec<Diagnostic>> {
    for (i, r) in records.iter().enumerate() {
        if r.code.trim().is_empty() || r.before.trim().is_empty() || r.after.trim().is_empty() {
            return Err(vec![Diagnostic::new("E0701", "bad_arguments", "", "", format!("correction record {i} needs a `code`, a `before` and an `after`"))]);
        }
    }
    let now = now_ms();
    for r in records {
        tx.execute(
            "INSERT INTO corrections (code, path, message, before, after, after_hash, agent, at) VALUES (?1, ?2, ?3, ?4, ?5, '', 'corpus', ?6)",
            params![r.code, r.path, r.message, excerpt(&r.before), excerpt(&r.after), now],
        )
        .map_err(sql)?;
    }
    Ok(records.len())
}

/// The text cut to [`EXCERPT_BYTES`] (the `…` included) at a line boundary, `…` appended when something was cut.
pub fn excerpt(text: &str) -> String {
    let text = text.trim_end();
    if text.len() <= EXCERPT_BYTES {
        return text.to_string();
    }
    let limit = EXCERPT_BYTES - '…'.len_utf8();
    let mut end = 0;
    for (i, c) in text.char_indices() {
        if c == '\n' && i <= limit {
            end = i;
        }
    }
    if end == 0 {
        // One long line: cut at a character boundary.
        end = (0..=limit).rev().find(|i| text.is_char_boundary(*i)).unwrap_or(0);
    }
    format!("{}…", &text[..end])
}

/// Records one attempt per diagnostic that names a definition (`def` non-empty; parse and store
/// errors have none and record nothing; neither do diagnostics about a definition not in `input`).
/// The text is that definition of `input` printed alone. Drops attempts past the retention window and keeps
/// the last [`MAX_ATTEMPTS`] per (name, agent).
pub fn record_attempts(tx: &Transaction, diags: &[Diagnostic], input: &Program, agent: &str) -> Result<(), Vec<Diagnostic>> {
    let now = now_ms();
    let mut names: Vec<&str> = Vec::new();
    for d in diags.iter().filter(|d| !d.def.is_empty()) {
        // A diagnostic about a definition the input does not contain (a dependent, a lease) is no mistake of this text.
        let Some(def) = input.defs.iter().find(|x| x.name() == d.def) else { continue };
        let text = print_one(def);
        tx.execute(
            "INSERT INTO attempts (name, agent, code, path, message, text, at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![d.def, agent, d.code, d.path, d.message, excerpt(&text), now],
        )
        .map_err(sql)?;
        if !names.contains(&d.def.as_str()) {
            names.push(&d.def);
        }
    }
    if names.is_empty() {
        return Ok(());
    }
    tx.execute("DELETE FROM attempts WHERE at < ?1", params![now - RETENTION_MS]).map_err(sql)?;
    for name in names {
        tx.execute(
            "DELETE FROM attempts WHERE name = ?1 AND agent = ?2 AND id NOT IN \
             (SELECT id FROM attempts WHERE name = ?1 AND agent = ?2 ORDER BY id DESC LIMIT ?3)",
            params![name, agent, MAX_ATTEMPTS],
        )
        .map_err(sql)?;
    }
    Ok(())
}

fn print_one(def: &Def) -> String {
    qal_compiler::print_program(&Program { defs: vec![def.clone()] }).trim_end().to_string()
}

/// True if `agent` has a live attempt for one of `names` (a cheap read before a `define` looks for
/// corrections; uses the (name, agent) index).
pub fn has_attempts_for(conn: &rusqlite::Connection, names: &[&str], agent: &str) -> Result<bool, Vec<Diagnostic>> {
    let since = now_ms() - RETENTION_MS;
    let mut stmt = conn.prepare_cached("SELECT 1 FROM attempts WHERE name = ?1 AND agent = ?2 AND at >= ?3 LIMIT 1").map_err(sql)?;
    for name in names {
        if stmt.exists(params![name, agent, since]).map_err(sql)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// For each added or replaced name with an attempt by `agent`: one correction from the first
/// diagnostic of its newest failed `define` (`after` = `after_text(name)`), then that agent's attempts
/// of the name are deleted. A fix that reads the same as the attempt records nothing. Returns the
/// number of corrections recorded.
pub fn record_corrections(tx: &Transaction, defined: &[Defined], after_text: &dyn Fn(&str) -> Option<String>, agent: &str) -> Result<usize, Vec<Diagnostic>> {
    let now = now_ms();
    let mut recorded = 0;
    for d in defined.iter().filter(|d| d.status == "added" || d.status == "replaced") {
        let newest: Option<(String, String, String, String)> = {
            let mut stmt = tx
                .prepare("SELECT code, path, message, text FROM attempts WHERE name = ?1 AND agent = ?2 AND at >= ?3 ORDER BY at DESC, id ASC LIMIT 1")
                .map_err(sql)?;
            let mut rows = stmt.query(params![d.name, agent, now - RETENTION_MS]).map_err(sql)?;
            match rows.next().map_err(sql)? {
                Some(r) => Some((r.get(0).map_err(sql)?, r.get(1).map_err(sql)?, r.get(2).map_err(sql)?, r.get(3).map_err(sql)?)),
                None => None,
            }
        };
        if let Some((code, path, message, before)) = newest {
            if let Some(after) = after_text(&d.name) {
                let after = excerpt(&after);
                if after != before {
                    tx.execute(
                        "INSERT INTO corrections (code, path, message, before, after, after_hash, agent, at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                        params![code, path, message, before, after, d.hash, agent, now],
                    )
                    .map_err(sql)?;
                    recorded += 1;
                }
            }
        }
        tx.execute("DELETE FROM attempts WHERE name = ?1 AND agent = ?2", params![d.name, agent]).map_err(sql)?;
    }
    Ok(recorded)
}

/// The corrections of `code`, newest first, identical (before, after) pairs collapsed into one with a
/// `count` (the newest represents the pair); the pairs are ordered by their newest member.
fn collapsed(tx: &Transaction, code: &str) -> Result<Vec<Correction>, Vec<Diagnostic>> {
    let mut stmt = tx
        .prepare("SELECT id, code, path, message, before, after, after_hash, agent, at FROM corrections WHERE code = ?1 ORDER BY id DESC LIMIT ?2")
        .map_err(sql)?;
    let rows = stmt
        .query_map(params![code, SCAN], |r| {
            Ok(Correction { id: r.get(0)?, code: r.get(1)?, path: r.get(2)?, message: r.get(3)?, before: r.get(4)?, after: r.get(5)?, after_hash: r.get(6)?, agent: r.get(7)?, at: r.get(8)?, count: 1 })
        })
        .map_err(sql)?;
    let mut out: Vec<Correction> = Vec::new();
    for row in rows {
        let c = row.map_err(sql)?;
        match out.iter_mut().find(|o| o.before == c.before && o.after == c.after) {
            Some(o) => o.count += 1,
            None => out.push(c),
        }
    }
    Ok(out)
}

/// The corrections for `q.code` (at most [`MAX_CORRECTIONS`], newest first, collapsed); empty without
/// a code or for a code nothing was corrected for.
pub fn recall(tx: &Transaction, q: &RecallQuery) -> Result<Vec<Correction>, Vec<Diagnostic>> {
    let Some(code) = q.code.as_deref() else { return Ok(Vec::new()) };
    let mut out = collapsed(tx, code)?;
    out.truncate(MAX_CORRECTIONS);
    Ok(out)
}

/// The hint for a diagnostic of `code`: its most frequent correction (the newest of equals), as the
/// first lines that differ.
pub fn hint_for(tx: &Transaction, code: &str) -> Result<Option<String>, Vec<Diagnostic>> {
    let mut best: Option<Correction> = None;
    for c in collapsed(tx, code)? {
        if best.as_ref().is_none_or(|b| c.count > b.count) {
            best = Some(c);
        }
    }
    Ok(best.map(|c| {
        let (b, a) = first_difference(&c.before, &c.after);
        format!("earlier fix of {} ({}x): {} -> {}", c.code, c.count, cut(b), cut(a))
    }))
}

/// The first line of each text where the two differ (an empty line where one text ended).
fn first_difference<'a>(before: &'a str, after: &'a str) -> (&'a str, &'a str) {
    let mut b = before.lines();
    let mut a = after.lines();
    loop {
        match (b.next(), a.next()) {
            (Some(x), Some(y)) if x == y => continue,
            (x, y) => return (x.unwrap_or("").trim(), y.unwrap_or("").trim()),
        }
    }
}

fn cut(line: &str) -> String {
    if line.chars().count() <= HINT_SIDE {
        line.to_string()
    } else {
        let mut s: String = line.chars().take(HINT_SIDE - 1).collect();
        s.push('…');
        s
    }
}
