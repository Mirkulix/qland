//! The memory of corrections (Phase 6b): a failed `define` leaves an *attempt* per diagnostic; the
//! next successful `define` of that name by the same agent turns the newest attempt into a
//! *correction* (code, path, failed text, fixed text). Later diagnostics of the same code and message
//! shape carry the most frequent correction of that shape as a hint, and `recall` lists them.
//!
//! Everything here is bounded: at most [`MAX_ATTEMPTS`] attempts per (name, agent), none older than
//! [`RETENTION_MS`], excerpts of at most [`EXCERPT_BYTES`], replies of at most [`MAX_CORRECTIONS`]
//! corrections. Callers treat a failed memory write as no memory (it never changes a `define`).

use crate::ops::RecallQuery;
use crate::{now_ms, sql, Defined};
use qal_core::ast::{Program, Def};
use qal_core::canon::Hash;
use qal_core::diag::Diagnostic;
use rusqlite::{params, Transaction};
use serde::{Deserialize, Serialize};

/// Attempts kept per name and agent.
pub const MAX_ATTEMPTS: i64 = 20;
/// An attempt older than this (2 hours) is dropped and never matched.
pub const RETENTION_MS: i64 = 7_200_000;
/// Longest `before`/`after` excerpt in bytes.
pub const EXCERPT_BYTES: usize = 600;
/// Longest diagnostic code of an imported record in bytes.
pub const CODE_BYTES: usize = 16;
/// Most corrections a `recall` returns.
pub const MAX_CORRECTIONS: usize = 5;
/// Characters a hint shows before the first difference of a fix.
const HINT_BEFORE: usize = 40;
/// Characters a hint shows from the first difference on.
const HINT_AFTER: usize = 80;
/// How many of the newest corrections of a code are looked at (bounds the work of `recall` and hints).
const SCAN: i64 = 500;

/// A failed definition: one row per failed `define` and name, from its first diagnostic of that name.
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
    /// The message with its backticked names replaced by `_` (see [`shape`]).
    pub shape: String,
    pub before: String,
    pub after: String,
    pub after_hash: String,
    pub agent: String,
    pub at: i64,
    /// How many corrections with this shape and the same first difference (the hint's two windows)
    /// there are among the newest 500 looked at; the newest of them represents them.
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
/// `code` is cut to [`CODE_BYTES`], `path` and `message` like the texts, so a recall stays bounded.
/// A record already stored with the same content (code, message, before, after) is skipped, so
/// importing a file twice stores it once. Returns how many records were stored.
pub fn import(tx: &Transaction, records: &[CorrectionInput]) -> Result<usize, Vec<Diagnostic>> {
    for (i, r) in records.iter().enumerate() {
        if r.code.trim().is_empty() || r.before.trim().is_empty() || r.after.trim().is_empty() {
            return Err(vec![Diagnostic::new("E0701", "bad_arguments", "", "", format!("correction record {i} needs a `code`, a `before` and an `after`"))]);
        }
    }
    let now = now_ms();
    let mut stored = 0;
    for r in records {
        let (code, message, before, after) = (cut_bytes(r.code.trim(), CODE_BYTES), excerpt(&r.message), excerpt(&r.before), excerpt(&r.after));
        stored += tx
            .execute(
                "INSERT OR IGNORE INTO corrections (code, path, message, shape, before, after, after_hash, agent, at, content) VALUES (?1, ?2, ?3, ?4, ?5, ?6, '', 'corpus', ?7, ?8)",
                params![code, excerpt(&r.path), message, shape(&message), before, after, now, content(code, &message, &before, &after, "corpus")],
            )
            .map_err(sql)?;
    }
    Ok(stored)
}

/// The message with every backticked span replaced by `_`: `unknown name `x`` becomes
/// `unknown name `_``. Diagnostics of one kind of mistake share a shape whatever names they mention.
pub fn shape(message: &str) -> String {
    let mut out = String::new();
    let mut rest = message;
    while let Some(i) = rest.find('`') {
        let Some(j) = rest[i + 1..].find('`') else { break };
        out.push_str(&rest[..i]);
        // A capitalised name (type, variant) is another class of mistake than a lowercase one
        // (local, function): `String` for `Str` must not share a shape with an unknown local.
        let name = &rest[i + 1..i + 1 + j];
        out.push_str(if name.starts_with(|c: char| c.is_ascii_uppercase()) { "`_T`" } else { "`_`" });
        rest = &rest[i + 1 + j + 1..];
    }
    out.push_str(rest);
    out
}

/// The content address of a correction: the hash of its fields, each prefixed by its length.
fn content(code: &str, message: &str, before: &str, after: &str, agent: &str) -> String {
    let mut bytes = Vec::new();
    for field in [code, message, before, after, agent] {
        bytes.extend_from_slice(&(field.len() as u64).to_le_bytes());
        bytes.extend_from_slice(field.as_bytes());
    }
    Hash::of(&bytes).to_hex()
}

/// The text cut to at most `max` bytes at a character boundary (no marker).
fn cut_bytes(text: &str, max: usize) -> &str {
    let end = (0..=max.min(text.len())).rev().find(|i| text.is_char_boundary(*i)).unwrap_or(0);
    &text[..end]
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

/// Records one attempt per name with a diagnostic, from its first diagnostic (`def` non-empty; parse
/// and store errors have none and record nothing; neither do diagnostics about a definition not in
/// `input`). The text is that definition of `input` printed alone; path and message are cut like it.
/// Drops attempts past the retention window and keeps the last [`MAX_ATTEMPTS`] per (name, agent).
pub fn record_attempts(tx: &Transaction, diags: &[Diagnostic], input: &Program, agent: &str) -> Result<(), Vec<Diagnostic>> {
    let now = now_ms();
    let mut names: Vec<&str> = Vec::new();
    for d in diags.iter().filter(|d| !d.def.is_empty()) {
        if names.contains(&d.def.as_str()) {
            continue;
        }
        // A diagnostic about a definition the input does not contain (a dependent, a lease) is no mistake of this text.
        let Some(def) = input.defs.iter().find(|x| x.name() == d.def) else { continue };
        let text = print_one(def);
        tx.execute(
            "INSERT INTO attempts (name, agent, code, path, message, text, at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![d.def, agent, d.code, excerpt(&d.path), excerpt(&d.message), excerpt(&text), now],
        )
        .map_err(sql)?;
        names.push(&d.def);
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
                        "INSERT INTO corrections (code, path, message, shape, before, after, after_hash, agent, at, content) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                        params![code, path, message, shape(&message), before, after, d.hash, agent, now, content(&code, &message, &before, &after, agent)],
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

/// A correction with the first difference of its texts and whether its message is the one asked for.
struct Seen {
    c: Correction,
    window: Option<(String, String)>,
    exact: bool,
}

/// The newest [`SCAN`] corrections matching the filters, newest first, grouped by (shape, first
/// difference): one entry per group, the newest member with the group's `count`, in the order of
/// their newest members. A correction whose texts do not differ beyond spacing is its own group.
fn grouped(tx: &Transaction, code: Option<&str>, shape_is: Option<&str>, message: Option<&str>) -> Result<Vec<Seen>, Vec<Diagnostic>> {
    let mut conds = Vec::new();
    let mut args: Vec<String> = Vec::new();
    if let Some(c) = code {
        args.push(c.to_string());
        conds.push(format!("code = ?{}", args.len()));
    }
    if let Some(s) = shape_is {
        args.push(s.to_string());
        conds.push(format!("shape = ?{}", args.len()));
    }
    if let Some(m) = message {
        args.push(m.to_string());
        args.push(shape(m));
        conds.push(format!("(instr(message, ?{}) > 0 OR instr(shape, ?{}) > 0)", args.len() - 1, args.len()));
    }
    let filter = if conds.is_empty() { String::new() } else { format!("WHERE {}", conds.join(" AND ")) };
    let mut stmt = tx
        .prepare(&format!("SELECT id, code, path, message, shape, before, after, after_hash, agent, at FROM corrections {filter} ORDER BY id DESC LIMIT {SCAN}"))
        .map_err(sql)?;
    let rows = stmt
        .query_map(rusqlite::params_from_iter(args.iter()), |r| {
            Ok(Correction { id: r.get(0)?, code: r.get(1)?, path: r.get(2)?, message: r.get(3)?, shape: r.get(4)?, before: r.get(5)?, after: r.get(6)?, after_hash: r.get(7)?, agent: r.get(8)?, at: r.get(9)?, count: 1 })
        })
        .map_err(sql)?;
    let mut out: Vec<Seen> = Vec::new();
    for row in rows {
        let c = row.map_err(sql)?;
        let window = windows(&c.before, &c.after);
        let exact = message.is_some_and(|m| m == c.message);
        let same = |o: &Seen| o.c.shape == c.shape && if window.is_some() { o.window == window } else { o.c.before == c.before && o.c.after == c.after };
        match out.iter_mut().find(|o| same(o)) {
            Some(o) => {
                o.c.count += 1;
                o.exact |= exact;
            }
            None => out.push(Seen { c, window, exact }),
        }
    }
    Ok(out)
}

/// Orders groups best first: more often seen, then with the message asked for, then newer.
fn by_rank(a: &Seen, b: &Seen) -> std::cmp::Ordering {
    (b.c.count, b.exact, b.c.id).cmp(&(a.c.count, a.exact, a.c.id))
}

/// The corrections for `q.code` and/or `q.message` (at most [`MAX_CORRECTIONS`]): the best fix of each
/// message shape first (shapes with more corrections first), then the second best of each, and so on,
/// so that one frequent shape does not hide the others of a code. Empty without a code or a message,
/// or for one nothing was corrected for.
pub fn recall(tx: &Transaction, q: &RecallQuery) -> Result<Vec<Correction>, Vec<Diagnostic>> {
    if q.code.is_none() && q.message.is_none() {
        return Ok(Vec::new());
    }
    let mut shapes: Vec<(String, i64, i64, Vec<Seen>)> = Vec::new(); // (shape, corrections, newest id, groups)
    for g in grouped(tx, q.code.as_deref(), None, q.message.as_deref())? {
        match shapes.iter_mut().find(|s| s.0 == g.c.shape) {
            Some(s) => {
                s.1 += g.c.count;
                s.3.push(g);
            }
            None => shapes.push((g.c.shape.clone(), g.c.count, g.c.id, vec![g])),
        }
    }
    shapes.sort_by(|a, b| (b.1, b.2).cmp(&(a.1, a.2)));
    let mut queues: Vec<std::vec::IntoIter<Seen>> = shapes
        .into_iter()
        .map(|mut s| {
            s.3.sort_by(by_rank);
            s.3.into_iter()
        })
        .collect();
    let mut out = Vec::new();
    while out.len() < MAX_CORRECTIONS {
        let before = out.len();
        for q in queues.iter_mut() {
            if out.len() < MAX_CORRECTIONS {
                out.extend(q.next().map(|s| s.c));
            }
        }
        if out.len() == before {
            break;
        }
    }
    Ok(out)
}

/// The hint for a diagnostic of `code` with `message`: the most frequent correction of the same code
/// and message shape (on a tie the one with this very message, then the newest), as a window around
/// the first difference of its texts. None when no correction has that shape or none shows a difference.
pub fn hint_for(tx: &Transaction, code: &str, message: &str) -> Result<Option<String>, Vec<Diagnostic>> {
    let mut groups: Vec<Seen> = grouped(tx, Some(code), Some(&shape(message)), None)?.into_iter().filter(|g| g.window.is_some()).collect();
    for g in groups.iter_mut() {
        g.exact = g.c.message == message;
    }
    groups.sort_by(by_rank);
    Ok(groups.into_iter().next().and_then(|g| {
        let (b, a) = g.window?;
        Some(format!("earlier fix of {} ({}x): {} -> {}", g.c.code, g.c.count, b, a))
    }))
}

/// The two texts, spacing collapsed, each cut to a window around their first differing character
/// ([`HINT_BEFORE`] characters before it, [`HINT_AFTER`] from it on, `…` at an elided end); None if
/// the windows read the same.
fn windows(before: &str, after: &str) -> Option<(String, String)> {
    let collapse = |t: &str| t.split_whitespace().collect::<Vec<_>>().join(" ").chars().collect::<Vec<char>>();
    let (b, a) = (collapse(before), collapse(after));
    let i = b.iter().zip(&a).take_while(|(x, y)| x == y).count();
    let window = |t: &[char]| {
        let start = i.saturating_sub(HINT_BEFORE);
        let end = (i + HINT_AFTER).min(t.len());
        let mut s = String::new();
        if start > 0 {
            s.push('…');
        }
        s.extend(&t[start.min(end)..end]);
        if end < t.len() {
            s.push('…');
        }
        s
    };
    let (wb, wa) = (window(&b), window(&a));
    (wb != wa).then_some((wb, wa))
}
