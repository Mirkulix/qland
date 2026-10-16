//! Change sets and leases (Phase 4 spec §5.2-§5.4, §6, §8): the spaces agents write code in, the
//! rules a write in one must follow (rights, scope, interface, leases, stale base), and the
//! operations of `qal_change` and `qal_lease`.
//!
//! Every operation is one write transaction that checks, writes and (for change sets) appends the
//! signed record. Leases never wait: acquiring is all-or-nothing in one transaction, so no call
//! holds a lease while it waits for another, and there is no deadlock (§5.2).

use super::canonical::hex;
use super::{root_of, space_rows, Rows};
use super::policy::{self, Role};
use super::{append_in, bad, members_of, notify, one, state, team_limit, Signer};
use crate::snapshot::Snapshot;
use crate::space::{self, Space};
use crate::{now_ms, sql, Store};
use qal_core::ast::{Def, Program};
use qal_core::diag::Diagnostic;
use qal_compiler::lexer::MAX_NAME_LEN;
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeSet;

/// Limits of spec §14.
pub const MAX_OPEN_CHANGES: i64 = 64;
pub const MAX_OPEN_PER_MEMBER: i64 = 8;
pub const MAX_DELTA: i64 = 2000;
pub const MAX_SCOPE: usize = 256;
pub const MAX_LEASES_PER_CHANGE: i64 = 256;
pub const MAX_LEASE_NAMES: usize = 64;
pub const DEFAULT_LEASE_TTL_MS: i64 = 15 * 60 * 1000;
pub const MIN_LEASE_TTL_MS: i64 = 60 * 1000;
pub const MAX_LEASE_TTL_MS: i64 = 2 * 3600 * 1000;
pub const MAX_TITLE_CHARS: usize = 200;
pub const MAX_DESCRIPTION_BYTES: usize = 8 << 10;

/// A lease on a name (§5.2): held by a change set for a member until `expires_at` (Unix ms).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Lease {
    pub name: String,
    pub change: String,
    pub member: String,
    pub expires_at: i64,
}

/// A row of `changes`.
#[derive(Debug, Clone, PartialEq)]
pub struct ChangeRow {
    pub id: String,
    pub title: String,
    pub description: String,
    pub scope: Vec<String>,
    pub owner: String,
    pub assignee: Option<String>,
    pub status: String,
    pub claimed: bool,
    pub base_seq: i64,
    pub base_root: String,
    pub needs_rebase: bool,
    pub created_at: i64,
    pub updated_at: i64,
}

impl ChangeRow {
    /// Open or submitted: it can still be written.
    pub fn is_live(&self) -> bool {
        self.status == "open" || self.status == "submitted"
    }

    /// An architect's work package nobody has claimed yet.
    pub fn is_package(&self) -> bool {
        !self.claimed
    }

    /// What this change set is to `member` (for the rights matrix).
    pub fn kind_for(&self, member: &str) -> policy::Space {
        if self.owner == member {
            policy::Space::Own
        } else if self.status == "submitted" {
            policy::Space::ForeignSubmitted
        } else {
            policy::Space::ForeignOther
        }
    }
}

/// E0901 `lease_held`.
pub fn lease_held(l: &Lease) -> Diagnostic {
    let mut d = Diagnostic::new("E0901", "lease_held", &l.name, "", format!("`{}` is leased by change set {} (member `{}`) until {} (Unix ms)", l.name, l.change, l.member, l.expires_at))
        .hint(format!("work on another name or change set, or ask the architect (qal_lease break); the lease expires at {} (Unix ms)", l.expires_at));
    d.detail = Some(json!({"name": l.name, "change": l.change, "member": l.member, "expires_at": l.expires_at}));
    d
}

/// E0909 `not_owner`.
pub fn not_owner(c: &ChangeRow, what: &str) -> Diagnostic {
    Diagnostic::new("E0909", "not_owner", "", "change", format!("{what} is only for the owner of change set {} (`{}`)", c.id, c.owner)).hint("work in your own change set (qal_whoami shows your current one; qal_change list with \"mine\": true)")
}

fn not_live(c: &ChangeRow) -> Diagnostic {
    state(format!("change set {} is {}: it cannot change any more", c.id, c.status)).hint("open a new change set (qal_change {\"action\": \"open\"})")
}

fn json_list(s: &str) -> Vec<String> {
    serde_json::from_str(s).unwrap_or_default()
}

fn list_json(xs: &[String]) -> String {
    Value::from(xs.to_vec()).to_string()
}

/// The change set `id`, if it exists.
pub fn load(conn: &Connection, id: &str) -> Result<Option<ChangeRow>, Vec<Diagnostic>> {
    conn.query_row(
        "SELECT id, title, description, scope, owner, assignee, status, claimed, base_seq, base_root, needs_rebase, created_at, updated_at FROM changes WHERE id = ?1",
        params![id],
        |r| {
            Ok(ChangeRow {
                id: r.get(0)?,
                title: r.get(1)?,
                description: r.get(2)?,
                scope: json_list(&r.get::<_, String>(3)?),
                owner: r.get(4)?,
                assignee: r.get(5)?,
                status: r.get(6)?,
                claimed: r.get::<_, i64>(7)? != 0,
                base_seq: r.get(8)?,
                base_root: r.get(9)?,
                needs_rebase: r.get::<_, i64>(10)? != 0,
                created_at: r.get(11)?,
                updated_at: r.get(12)?,
            })
        },
    )
    .optional()
    .map_err(sql)
}

/// The change set `id` (E0201 if there is none).
pub fn require(conn: &Connection, id: &str) -> Result<ChangeRow, Vec<Diagnostic>> {
    load(conn, id)?.ok_or_else(|| one(space::unknown_change(id)))
}

/// What `space` is to `member` (for the rights matrix): `main`, an own or a foreign change set.
pub(crate) fn kind_of(conn: &Connection, space: &Space, member: &str) -> Result<policy::Space, Vec<Diagnostic>> {
    match space {
        Space::Main => Ok(policy::Space::Main),
        Space::Change(c) => Ok(load(conn, c)?.map_or(policy::Space::ForeignOther, |row| row.kind_for(member))),
    }
}

/// The agent role of `member` (E0903 for the owner, who writes no code).
pub(crate) fn role_of(conn: &Connection, member: &str) -> Result<Role, Vec<Diagnostic>> {
    let m = super::member(conn, member)?.ok_or_else(|| one(super::no_session(format!("`{member}` is not a member of this team"))))?;
    Role::parse(&m.role).ok_or_else(|| {
        let mut d = Diagnostic::new("E0903", "role_forbidden", "", "", "the owner manages the team on the command line and writes no code");
        d.expected = Some("architect, implementer, verifier".into());
        d.found = Some(m.role.clone());
        one(d)
    })
}

/// A name as `scope` and `qal_lease` take it.
fn valid_name(n: &str) -> bool {
    !n.is_empty() && n.len() <= MAX_NAME_LEN && n.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'.')
}

fn check_names(names: &[String], what: &str, max: usize) -> Result<(), Vec<Diagnostic>> {
    if let Some(n) = names.iter().find(|n| !valid_name(n)) {
        return Err(one(bad(format!("`{n}` in `{what}` is not a definition name"))));
    }
    if names.len() > max {
        return Err(one(team_limit(format!("`{what}` lists {} names; at most {max}", names.len()))));
    }
    Ok(())
}

// --- Leases -------------------------------------------------------------------------------

/// The lease on `name`, expired or not.
fn lease_of(conn: &Connection, name: &str) -> Result<Option<Lease>, Vec<Diagnostic>> {
    conn.query_row("SELECT name, change, member, expires_at FROM leases WHERE name = ?1", params![name], |r| Ok(Lease { name: r.get(0)?, change: r.get(1)?, member: r.get(2)?, expires_at: r.get(3)? }))
        .optional()
        .map_err(sql)
}

/// The live leases of change set `change`, by name.
pub fn leases_of(conn: &Connection, change: &str) -> Result<Vec<Lease>, Vec<Diagnostic>> {
    let mut stmt = conn.prepare("SELECT name, change, member, expires_at FROM leases WHERE change = ?1 AND expires_at > ?2 ORDER BY name").map_err(sql)?;
    let rows = stmt.query_map(params![change, now_ms()], |r| Ok(Lease { name: r.get(0)?, change: r.get(1)?, member: r.get(2)?, expires_at: r.get(3)? })).map_err(sql)?;
    rows.collect::<Result<Vec<_>, _>>().map_err(sql)
}

/// E0901 for every name of `names` that another change set holds (an expired lease counts as free).
fn conflicts(conn: &Connection, change: &str, names: &[&str], now: i64) -> Result<Vec<Diagnostic>, Vec<Diagnostic>> {
    let mut out = Vec::new();
    for n in names {
        if let Some(l) = lease_of(conn, n)? {
            if l.change != change && l.expires_at > now {
                out.push(lease_held(&l));
            }
        }
    }
    Ok(out)
}

/// Takes (or extends) the leases on `names` for `change` (all or nothing): E0901 for those another
/// change set holds, nothing taken then. Returns the leases that were new to `change`.
/// `reserve`: the leases count toward the 256 reservations of names the change set has not
/// changed (explicit acquires and claims); a write's own leases count toward its delta instead.
pub(crate) fn acquire(tx: &Transaction, change: &str, member: &str, names: &[&str], ttl_ms: i64, now: i64, reserve: bool) -> Result<Vec<Lease>, Vec<Diagnostic>> {
    let held = conflicts(tx, change, names, now)?;
    if !held.is_empty() {
        return Err(held);
    }
    let mut fresh = Vec::new();
    let mut seen = BTreeSet::new();
    for n in names.iter().filter(|n| seen.insert(**n)) {
        let mine = lease_of(tx, n)?.filter(|l| l.change == change && l.expires_at > now);
        let expires_at = mine.as_ref().map_or(now + ttl_ms, |l| l.expires_at.max(now + ttl_ms));
        tx.execute(
            "INSERT INTO leases (name, change, member, acquired_at, expires_at) VALUES (?1, ?2, ?3, ?4, ?5) \
             ON CONFLICT(name) DO UPDATE SET change = excluded.change, member = excluded.member, \
             acquired_at = CASE WHEN leases.change = excluded.change AND leases.expires_at > ?4 THEN leases.acquired_at ELSE excluded.acquired_at END, expires_at = excluded.expires_at",
            params![n, change, member, now, expires_at],
        )
        .map_err(sql)?;
        if mine.is_none() {
            fresh.push(Lease { name: n.to_string(), change: change.into(), member: member.into(), expires_at });
        }
    }
    let count = if reserve { reserved(tx, change, now)? } else { 0 };
    if count > MAX_LEASES_PER_CHANGE {
        return Err(one(team_limit(format!("a change set reserves at most {MAX_LEASES_PER_CHANGE} names it has not changed ({change} would reserve {count})")).hint("release the leases you no longer need (qal_lease release); the names a change set changes count against its delta (2 000) instead")));
    }
    Ok(fresh)
}

/// The live leases of `change` on names it has not changed (§14: 256 per change set). Leases on
/// the names it changed are bounded by its delta (2 000) instead, so a large define is never
/// refused for its leases (review I2).
fn reserved(conn: &Connection, change: &str, now: i64) -> Result<i64, Vec<Diagnostic>> {
    conn.query_row(
        "SELECT count(*) FROM leases l WHERE l.change = ?1 AND l.expires_at > ?2 AND NOT EXISTS \
         (SELECT 1 FROM change_names c WHERE c.change = l.change AND c.name = l.name AND c.origin IN ('edited', 'added', 'renamed'))",
        params![change, now],
        |r| r.get(0),
    )
    .map_err(sql)
}

/// The delta of `change` (§14): the names it edited, added or renamed.
fn delta(conn: &Connection, change: &str) -> Result<i64, Vec<Diagnostic>> {
    conn.query_row("SELECT count(*) FROM change_names WHERE change = ?1 AND origin IN ('edited', 'added', 'renamed')", params![change], |r| r.get(0)).map_err(sql)
}

fn delta_limit(c: &str, n: i64) -> Vec<Diagnostic> {
    one(team_limit(format!("a change set changes at most {MAX_DELTA} names; {c} would change {n}")).hint("split the work into several change sets"))
}

/// The delta `change` would have after writing `names` (an upper bound: relinks do not count).
/// Checked before the prover runs.
fn delta_check(conn: &Connection, change: &str, names: &[&str]) -> Result<(), Vec<Diagnostic>> {
    let mut stmt = conn.prepare_cached("SELECT 1 FROM change_names WHERE change = ?1 AND name = ?2 AND origin IN ('edited', 'added', 'renamed')").map_err(sql)?;
    let mut new = 0i64;
    for n in names {
        if stmt.query_row(params![change, n], |_| Ok(())).optional().map_err(sql)?.is_none() {
            new += 1;
        }
    }
    let total = delta(conn, change)? + new;
    if total > MAX_DELTA {
        return Err(delta_limit(change, total));
    }
    Ok(())
}

// --- The rules of a write in a change set ---------------------------------------------------

/// What an implementer changed about the interface of a stored definition (None: nothing).
fn interface_change(old: &Def, new: &Def) -> Option<&'static str> {
    let (before, after) = (crate::define::signature(old), crate::define::signature(new));
    if before.0 != after.0 {
        return Some("kind");
    }
    match (old, new) {
        (Def::Type(_), Def::Type(_)) if before != after => Some("type"),
        (Def::Fn(_), Def::Fn(_)) if before != after => Some("signature"),
        (Def::Fn(a), Def::Fn(b)) => {
            let text = |xs: &[qal_core::ast::Expr]| xs.iter().map(qal_compiler::print_expr).collect::<Vec<_>>();
            (text(&a.req) != text(&b.req) || text(&a.ens) != text(&b.ens)).then_some("contracts")
        }
        _ => None,
    }
}

/// True if `name` exists for an implementer in change set `c`: it was in the base (main when the
/// change set was opened) or an architect added it here (e.g. the stubs of a work package). Names
/// the implementer or a verifier added in this change set are not interface.
fn existing(tx: &Transaction, c: &str, member: &str, snap: &Snapshot, name: &str) -> Result<bool, Vec<Diagnostic>> {
    if !snap.names.contains_key(name) {
        return Ok(false);
    }
    let base: Option<Option<String>> = tx.query_row("SELECT base_hash FROM change_names WHERE change = ?1 AND name = ?2", params![c, name], |r| r.get(0)).optional().map_err(sql)?;
    if base.flatten().is_some() {
        return Ok(true);
    }
    let added_by: Option<String> = tx
        .query_row("SELECT agent FROM history WHERE space = ?1 AND name = ?2 AND action = 'added' ORDER BY id LIMIT 1", params![c, name], |r| r.get(0))
        .optional()
        .map_err(sql)?;
    let Some(by) = added_by.filter(|b| b != member) else { return Ok(false) };
    Ok(super::member(tx, &by)?.is_some_and(|m| m.role == "architect"))
}

/// The session of a record (from its body).
fn record_session(body: &[u8]) -> Option<String> {
    let rest = body.strip_prefix(super::RECORD_PREFIX.as_bytes())?;
    let v: Value = serde_json::from_slice(rest).ok()?;
    v.get("session").and_then(Value::as_str).map(str::to_string)
}

/// The hash `session` last saw (viewed or wrote) of `name` in `space` (§5.3).
fn seen_hash(conn: &Connection, session: &str, space: &str, name: &str) -> Result<Option<String>, Vec<Diagnostic>> {
    conn.query_row("SELECT hash FROM seen WHERE session = ?1 AND space = ?2 AND name = ?3", params![session, space, name], |r| r.get(0)).optional().map_err(sql)
}

/// E0904 `stale_base` (§5.3): replacing a name whose last writer in this space was another session
/// that this session has not seen the current version of.
fn stale(tx: &Transaction, log: Option<&super::TeamLog>, signer: &Signer, c: &str, snap: &Snapshot, names: &[&str]) -> Result<(), Vec<Diagnostic>> {
    let mut diags = Vec::new();
    for n in names {
        let Some(current) = snap.names.get(*n).map(|e| e.hash.to_hex()) else { continue };
        let last: Option<(i64, String, String, Vec<u8>)> = tx
            .query_row(
                "SELECT h.id, h.new_hash, h.agent, r.body FROM history h JOIN records r ON r.seq = h.record WHERE h.space = ?1 AND h.name = ?2 AND h.action IN ('added', 'replaced') ORDER BY h.id DESC LIMIT 1",
                params![c, n],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .optional()
            .map_err(sql)?;
        let Some((id, written, by, body)) = last else { continue };
        if record_session(&body).as_deref() == Some(signer.session.as_str()) {
            continue;
        }
        // What it wrote (store.db) or viewed (the team log; unreadable counts as not seen).
        let wrote = seen_hash(tx, &signer.session, c, n)?;
        let viewed = log.and_then(|l| l.seen(&signer.session, c, n).ok().flatten());
        // The versions of that edit: its hash and the hashes relinks gave it since (same text).
        let mut versions: BTreeSet<String> = [written.clone(), current.clone()].into();
        {
            let mut stmt = tx.prepare_cached("SELECT new_hash FROM history WHERE space = ?1 AND name = ?2 AND action = 'relinked' AND id > ?3").map_err(sql)?;
            let rows = stmt.query_map(params![c, n, id], |r| r.get::<_, String>(0)).map_err(sql)?;
            for h in rows {
                versions.insert(h.map_err(sql)?);
            }
        }
        let ok = |h: &Option<String>| h.as_ref().is_some_and(|h| versions.contains(h));
        if !ok(&wrote) && !ok(&viewed) {
            let mut d = Diagnostic::new("E0904", "stale_base", n, "", format!("`{n}` was changed in change set {c} by `{by}` (another session) to #{current}, which this session has not seen"))
                .hint(format!("look at the current version first (qal_view {{\"names\": [\"{n}\"]}}), then write it again"));
            d.found = Some(current);
            diags.push(d);
        }
    }
    if diags.is_empty() {
        Ok(())
    } else {
        Err(diags)
    }
}

/// The change set of a write in `space`, checked to be live and writable for the signer with `tool`
/// (E0903 on main or by the matrix, E0908 if merged or abandoned).
fn writable(tx: &Transaction, signer: &Signer, space: &Space, tool: &str) -> Result<(Role, ChangeRow), Vec<Diagnostic>> {
    let role = role_of(tx, &signer.member)?;
    let Space::Change(c) = space else {
        return Err(one(policy::forbidden(tool, None, role, policy::Space::Main)));
    };
    let row = require(tx, c)?;
    if !row.is_live() {
        return Err(one(not_live(&row)));
    }
    let kind = row.kind_for(&signer.member);
    if !policy::allows(tool, None, role, kind) {
        return Err(one(policy::forbidden(tool, None, role, kind)));
    }
    Ok((role, row))
}

/// Takes the leases of a write and, when the owner writes a submitted change set, reopens it
/// (§6.2). `dry`: only checks that the leases are free.
fn lease_write(tx: &Transaction, signer: &Signer, row: &ChangeRow, names: &[&str], dry: bool) -> Result<Vec<Lease>, Vec<Diagnostic>> {
    let now = now_ms();
    if dry {
        let held = conflicts(tx, &row.id, names, now)?;
        return if held.is_empty() { Ok(Vec::new()) } else { Err(held) };
    }
    let fresh = acquire(tx, &row.id, &signer.member, names, DEFAULT_LEASE_TTL_MS, now, false)?;
    if row.owner == signer.member && row.status == "submitted" {
        tx.execute("UPDATE changes SET status = 'open', updated_at = ?2 WHERE id = ?1", params![row.id, now]).map_err(sql)?;
    } else {
        tx.execute("UPDATE changes SET updated_at = ?2 WHERE id = ?1", params![row.id, now]).map_err(sql)?;
    }
    Ok(fresh)
}

/// The team rules of a `define` in `space` (§5.2-§5.4, §8): the matrix, verifiers write only tests,
/// implementers stay inside the scope (E0905) and keep interfaces (E0906), no stale base (E0904),
/// and the leases of the defined names (E0901; taken unless `dry`). Returns the new leases.
pub(crate) fn guard_define(tx: &Transaction, log: Option<&super::TeamLog>, signer: Option<&Signer>, space: &Space, snap: &Snapshot, input: &Program, dry: bool) -> Result<Vec<Lease>, Vec<Diagnostic>> {
    let Some(signer) = signer else { return Ok(Vec::new()) };
    let (role, row) = writable(tx, signer, space, "qal_define")?;
    let mut names: Vec<&str> = Vec::new();
    for d in &input.defs {
        if !names.contains(&d.name()) {
            names.push(d.name());
        }
    }
    let mut diags = Vec::new();
    match role {
        Role::Verifier => {
            for d in input.defs.iter().filter(|d| !matches!(d, Def::Test(_))) {
                let mut x = Diagnostic::new("E0903", "role_forbidden", d.name(), "", format!("a verifier writes only `test` definitions, and `{}` is not one", d.name()));
                x.expected = Some("architect, implementer".into());
                x.found = Some(role.as_str().into());
                diags.push(x.hint("ask the owner of the change set to change it; add tests that show the problem"));
            }
        }
        Role::Implementer => {
            for d in &input.defs {
                let n = d.name();
                if !existing(tx, &row.id, &signer.member, snap, n)? {
                    continue;
                }
                if !row.scope.iter().any(|s| s == n) {
                    diags.push(
                        Diagnostic::new("E0905", "out_of_scope", n, "", format!("`{n}` exists and is not in the scope of change set {}", row.id))
                            .hint(format!("the scope is: {}; new names are always allowed; ask the architect to widen it (qal_change scope)", if row.scope.is_empty() { "empty".to_string() } else { row.scope.join(", ") })),
                    );
                    continue;
                }
                let old = snap.decode_name(tx, n)?;
                if let Some(what) = interface_change(&old, d) {
                    let mut x = Diagnostic::new("E0906", "interface_change", n, "", format!("an implementer keeps the {what} of `{n}`: it is part of its interface"))
                        .hint("replace only the body (the same signature and contracts relink its users automatically); ask the architect for an interface change");
                    let (b, a) = (crate::define::signature(&old), crate::define::signature(d));
                    x.expected = Some(crate::define::show(&b));
                    x.found = Some(crate::define::show(&a));
                    diags.push(x);
                }
            }
        }
        Role::Architect | Role::Reviewer => {}
    }
    if !diags.is_empty() {
        return Err(diags);
    }
    stale(tx, log, signer, &row.id, snap, &names)?;
    delta_check(tx, &row.id, &names)?;
    lease_write(tx, signer, &row, &names, dry)
}

/// The team rules of `rename`/`delete` (`tool`) in `space`, run after the change inside its
/// transaction: the matrix, the change set's state and the leases of the names it changes.
pub(crate) fn guard_names(tx: &Transaction, signer: Option<&Signer>, space: &Space, tool: &str, entries: &[super::Entry]) -> Result<(), Vec<Diagnostic>> {
    let Some(signer) = signer else { return Ok(()) };
    if !super::is_team(tx)? {
        return Ok(());
    }
    let (_, row) = writable(tx, signer, space, tool)?;
    let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    lease_write(tx, signer, &row, &names, false).map(|_| ())
}

/// After a code write in `space` (inside its transaction): the signer's session has seen what it
/// wrote; a change set keeps within its delta; on `main`, holders of leases on relinked names hear
/// of it (`relinked`).
pub(crate) fn after_write(tx: &Transaction, signer: &Signer, space: &Space, entries: &[super::Entry]) -> Result<(), Vec<Diagnostic>> {
    // The writer has seen what it wrote. A relinked dependent (or a user whose refs follow a
    // rename) it did not look at: its seen-mark moves to the new hash only if it had seen the old
    // one, so a relink never counts as a view (review I1).
    for e in entries {
        let carried = matches!(e.action.as_str(), "relinked" | "updated");
        match (&e.new, carried) {
            (Some(h), false) => tx.execute(
                "INSERT INTO seen (session, space, name, hash) VALUES (?1, ?2, ?3, ?4) ON CONFLICT(session, space, name) DO UPDATE SET hash = excluded.hash",
                params![signer.session, space.id(), e.name, h],
            ),
            (Some(h), true) => tx.execute("UPDATE seen SET hash = ?4 WHERE session = ?1 AND space = ?2 AND name = ?3 AND hash = ?5", params![signer.session, space.id(), e.name, h, e.old]),
            (None, _) => tx.execute("DELETE FROM seen WHERE session = ?1 AND space = ?2 AND name = ?3", params![signer.session, space.id(), e.name]),
        }
        .map_err(sql)?;
    }
    match space {
        Space::Change(c) => {
            let n = delta(tx, c)?;
            if n > MAX_DELTA {
                return Err(delta_limit(c, n));
            }
        }
        Space::Main => {
            let now = now_ms();
            for e in entries.iter().filter(|e| e.action == "relinked") {
                if let Some(l) = lease_of(tx, &e.name)?.filter(|l| l.expires_at > now) {
                    notify(tx, &l.member, "relinked", &json!({"name": e.name, "change": l.change, "old": e.old, "new": e.new}))?;
                }
            }
        }
    }
    Ok(())
}

// --- The signed metadata of a change set (review I4) -----------------------------------------

/// SHA-256 over everything a change set's decisions rest on, as stored: its `changes` row (title,
/// description, scope, owner, assignee, status, claimed, base_seq, base_root, created_at) and every
/// `change_names` row with its raw columns (name, hash, local_names, refs, origin, base_hash,
/// conflict), rows without a hash included. Every record about a change set signs this digest
/// (`args.meta_after`), and `qal verify` recomputes it, so any byte change of these columns is
/// found. Not included: times of the last change, `needs_rebase` (a hint others set), leases and
/// seen-marks (unsigned coordination, spec §9.1).
pub fn meta_digest(conn: &Connection, change: &str) -> Result<String, Vec<Diagnostic>> {
    let head: Option<[String; 10]> = conn
        .query_row(
            "SELECT id, title, description, scope, owner, coalesce(assignee, ''), status, CAST(claimed AS TEXT), CAST(base_seq AS TEXT), base_root || char(0) || CAST(created_at AS TEXT) FROM changes WHERE id = ?1",
            params![change],
            |r| Ok([r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?, r.get(7)?, r.get(8)?, r.get(9)?]),
        )
        .optional()
        .map_err(sql)?;
    let mut bytes = Vec::new();
    match head {
        Some(cols) => bytes.extend_from_slice(cols.join("\0").as_bytes()),
        None => bytes.extend_from_slice(b"no change set"),
    }
    bytes.push(b'\n');
    let mut stmt = conn
        .prepare_cached("SELECT name, coalesce(hash, ''), local_names, refs, origin, coalesce(base_hash, ''), coalesce(conflict, '') FROM change_names WHERE change = ?1 ORDER BY CAST(name AS BLOB)")
        .map_err(sql)?;
    let rows = stmt.query_map(params![change], |r| Ok([r.get::<_, String>(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?])).map_err(sql)?;
    for row in rows {
        bytes.extend_from_slice(row.map_err(sql)?.join("\0").as_bytes());
        bytes.push(b'\n');
    }
    Ok(qal_core::canon::Hash::of(&bytes).to_hex())
}

// --- Change set operations --------------------------------------------------------------------

/// The arguments of `qal_change open`.
#[derive(Debug, Clone, Default)]
pub struct OpenArgs {
    pub title: String,
    pub description: String,
    pub scope: Vec<String>,
    pub assignee: Option<String>,
    /// An architect's work package (claimable); default: when an assignee is given.
    pub package: Option<bool>,
}

/// Notices `kind` with `body` for every active member with one of `roles`, except `except`.
fn notify_roles(tx: &Transaction, roles: &[&str], except: &str, kind: &str, body: &Value) -> Result<(), Vec<Diagnostic>> {
    for m in members_of(tx)?.into_iter().filter(|m| m.revoked_at.is_none() && roles.contains(&m.role.as_str()) && m.id != except) {
        notify(tx, &m.id, kind, body)?;
    }
    Ok(())
}

/// Appends the `change` record of `action` on `row` (its space's names do not change).
fn record_change(tx: &Transaction, signer: &Signer, id: &str, action: &str, fields: Value) -> Result<i64, Vec<Diagnostic>> {
    let space = Space::Change(id.to_string());
    let names = space_rows(tx, &space)?;
    append_in(tx, signer, "change", json!({"id": id, "action": action, "fields": fields}), &space, &names, &names)
}

fn set_current(tx: &Transaction, session: &str, change: Option<&str>) -> Result<(), Vec<Diagnostic>> {
    tx.execute("UPDATE sessions SET current_change = ?1 WHERE id = ?2", params![change, session]).map(|_| ()).map_err(sql)
}

/// The JSON of a change set in lists (`root` = its current root).
fn summary(conn: &Connection, row: &ChangeRow) -> Result<Value, Vec<Diagnostic>> {
    let count = |origins: &str| -> Result<i64, Vec<Diagnostic>> {
        conn.query_row(&format!("SELECT count(*) FROM change_names WHERE change = ?1 AND origin IN ({origins})"), params![row.id], |r| r.get(0)).map_err(sql)
    };
    let conflicts: i64 = conn.query_row("SELECT count(*) FROM change_names WHERE change = ?1 AND conflict IS NOT NULL", params![row.id], |r| r.get(0)).map_err(sql)?;
    // Roots only of live change sets (finished ones keep their rows, but nobody reviews them).
    let current = if row.is_live() { root_of(&space_rows(conn, &Space::Change(row.id.clone()))?) } else { String::new() };
    let mut reviews = Vec::new();
    {
        let mut stmt = conn.prepare("SELECT member, verdict, root FROM reviews WHERE change = ?1 ORDER BY id").map_err(sql)?;
        let rows = stmt.query_map(params![row.id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))).map_err(sql)?;
        for r in rows {
            let (member, verdict, at) = r.map_err(sql)?;
            reviews.push(json!({"member": member, "verdict": verdict, "current": at == current}));
        }
    }
    Ok(json!({
        "change": row.id, "title": row.title, "status": row.status, "owner": row.owner, "assignee": row.assignee, "package": row.is_package(),
        "scope": row.scope, "edited": count("'edited', 'added', 'renamed'")?, "relinked": count("'relinked'")?, "needs_rebase": row.needs_rebase,
        "conflicts": conflicts, "root": if row.is_live() { Value::from(current) } else { Value::Null }, "reviews": reviews,
    }))
}

impl Store {
    fn team_tx(&mut self, what: &str) -> Result<Transaction<'_>, Vec<Diagnostic>> {
        if !super::is_team(&self.conn)? {
            return Err(one(super::not_a_team(what)));
        }
        self.conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)
    }

    /// `qal_change open` (§6.2): a change set whose space starts as a copy of `main`. An architect's
    /// work package is claimable (owner = the architect until `claim`); any other change set is the
    /// caller's own and becomes the session's current one.
    pub fn change_open(&mut self, signer: &Signer, a: &OpenArgs) -> Result<Value, Vec<Diagnostic>> {
        let title = a.title.trim();
        if title.is_empty() || title.chars().count() > MAX_TITLE_CHARS {
            return Err(one(bad(format!("`title` must be 1 to {MAX_TITLE_CHARS} characters"))));
        }
        if a.description.len() > MAX_DESCRIPTION_BYTES {
            return Err(one(bad(format!("`description` is longer than {MAX_DESCRIPTION_BYTES} bytes"))));
        }
        check_names(&a.scope, "scope", MAX_SCOPE)?;
        let scope: Vec<String> = a.scope.iter().cloned().collect::<BTreeSet<_>>().into_iter().collect();
        let tx = self.team_tx("`qal_change`")?;
        let role = role_of(&tx, &signer.member)?;
        if !policy::allows("qal_change", Some("open"), role, policy::Space::Own) {
            return Err(one(policy::forbidden("qal_change", Some("open"), role, policy::Space::Own)));
        }
        let package = a.package.unwrap_or(a.assignee.is_some());
        if (package || a.assignee.is_some()) && role != Role::Architect {
            let mut d = Diagnostic::new("E0903", "role_forbidden", "", "assignee", "only architects open work packages (with `assignee` or `package`)");
            d.expected = Some("architect".into());
            d.found = Some(role.as_str().into());
            return Err(one(d.hint("leave out `assignee` and `package`: the change set is then your own")));
        }
        if let Some(who) = &a.assignee {
            let ok = super::member(&tx, who)?.is_some_and(|m| m.revoked_at.is_none() && (m.role == "implementer" || m.role == "verifier"));
            if !ok {
                return Err(one(bad(format!("`assignee` must be an active implementer or verifier, not `{who}`")).hint("`qal_whoami` shows your id; the owner lists members with `qal team status`")));
            }
        }
        let open: i64 = tx.query_row("SELECT count(*) FROM changes WHERE status IN ('open', 'submitted')", [], |r| r.get(0)).map_err(sql)?;
        let mine: i64 = tx.query_row("SELECT count(*) FROM changes WHERE status IN ('open', 'submitted') AND owner = ?1", params![signer.member], |r| r.get(0)).map_err(sql)?;
        if open >= MAX_OPEN_CHANGES {
            return Err(one(team_limit(format!("at most {MAX_OPEN_CHANGES} change sets can be open at once")).hint("submit or abandon finished ones")));
        }
        if mine >= MAX_OPEN_PER_MEMBER {
            return Err(one(team_limit(format!("a member owns at most {MAX_OPEN_PER_MEMBER} open change sets")).hint("submit or abandon one of yours first (qal_change list with \"mine\": true)")));
        }
        let id = format!("c{}", &hex(super::keys::random32().map_err(one)?.as_slice())[..12]);
        let now = now_ms();
        let main = space_rows(&tx, &Space::Main)?;
        let base_root = root_of(&main);
        let base_seq: i64 = tx.query_row("SELECT coalesce(max(seq), 0) FROM records", [], |r| r.get(0)).map_err(sql)?;
        tx.execute(
            "INSERT INTO changes (id, title, description, scope, owner, assignee, status, claimed, base_seq, base_root, needs_rebase, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'open', ?7, ?8, ?9, 0, ?10, ?10)",
            params![id, title, a.description, list_json(&scope), signer.member, a.assignee, !package, base_seq, base_root, now],
        )
        .map_err(sql)?;
        tx.execute(
            "INSERT INTO change_names (change, name, hash, local_names, refs, origin, base_hash, conflict, updated_at) SELECT ?1, name, hash, local_names, refs, 'base', hash, NULL, ?2 FROM names",
            params![id, now],
        )
        .map_err(sql)?;
        let space = Space::Change(id.clone());
        let fields = json!({"title": title, "description": a.description, "scope": scope, "owner": signer.member, "assignee": a.assignee, "package": package, "base_root": base_root});
        let record = append_in(&tx, signer, "change", json!({"id": id, "action": "open", "fields": fields}), &space, &Rows::new(), &main)?;
        if package {
            let body = json!({"change": id, "title": title, "by": signer.member, "scope": scope});
            match &a.assignee {
                Some(who) => notify(&tx, who, "change_opened", &body)?,
                None => notify_roles(&tx, &["implementer", "verifier"], &signer.member, "change_opened", &body)?,
            }
        } else {
            set_current(&tx, &signer.session, Some(&id))?;
        }
        tx.commit().map_err(sql)?;
        Ok(json!({"change": id, "status": "open", "package": package, "base_root": base_root, "record": record, "current": !package}))
    }

    /// `qal_change claim`: an implementer or verifier takes an unclaimed work package (its
    /// assignee or anyone if it has none), with the leases of its scope (all or nothing); it becomes
    /// the session's current change set.
    pub fn change_claim(&mut self, signer: &Signer, id: &str) -> Result<Value, Vec<Diagnostic>> {
        let tx = self.team_tx("`qal_change`")?;
        let role = role_of(&tx, &signer.member)?;
        let row = require(&tx, id)?;
        if !policy::allows("qal_change", Some("claim"), role, policy::Space::Own) {
            return Err(one(policy::forbidden("qal_change", Some("claim"), role, row.kind_for(&signer.member))));
        }
        if row.status != "open" || row.claimed {
            let why = if row.claimed { format!("change set {id} is not a work package to claim (it belongs to `{}`)", row.owner) } else { format!("change set {id} is {}", row.status) };
            return Err(one(state(why).hint("qal_change list shows the open work packages (\"package\": true)")));
        }
        if row.assignee.as_deref().is_some_and(|a| a != signer.member) {
            return Err(one(Diagnostic::new("E0909", "not_owner", "", "change", format!("work package {id} is assigned to `{}`", row.assignee.as_deref().unwrap_or(""))).hint("claim a package assigned to you or to nobody")));
        }
        let mine: i64 = tx.query_row("SELECT count(*) FROM changes WHERE status IN ('open', 'submitted') AND owner = ?1", params![signer.member], |r| r.get(0)).map_err(sql)?;
        if mine >= MAX_OPEN_PER_MEMBER {
            return Err(one(team_limit(format!("a member owns at most {MAX_OPEN_PER_MEMBER} open change sets")).hint("submit or abandon one of yours first (qal_change list with \"mine\": true)")));
        }
        let now = now_ms();
        let scope: Vec<&str> = row.scope.iter().map(String::as_str).collect();
        let leases = acquire(&tx, id, &signer.member, &scope, DEFAULT_LEASE_TTL_MS, now, true)?;
        tx.execute("UPDATE changes SET owner = ?2, claimed = 1, updated_at = ?3 WHERE id = ?1", params![id, signer.member, now]).map_err(sql)?;
        // The claimer takes the package as it is: it has seen what the architect wrote into it (§5.3).
        tx.execute(
            "INSERT INTO seen (session, space, name, hash) SELECT ?2, change, name, hash FROM change_names WHERE change = ?1 AND origin != 'base' AND hash IS NOT NULL \
             ON CONFLICT(session, space, name) DO UPDATE SET hash = excluded.hash",
            params![id, signer.session],
        )
        .map_err(sql)?;
        let record = record_change(&tx, signer, id, "claim", json!({"owner": signer.member}))?;
        set_current(&tx, &signer.session, Some(id))?;
        notify(&tx, &row.owner, "change_claimed", &json!({"change": id, "title": row.title, "by": signer.member}))?;
        tx.commit().map_err(sql)?;
        Ok(json!({"change": id, "status": "open", "owner": signer.member, "leases": leases, "record": record, "current": true}))
    }

    /// `qal_change submit` (owner): ready for review; verifiers and reviewers hear of it.
    pub fn change_submit(&mut self, signer: &Signer, id: &str) -> Result<Value, Vec<Diagnostic>> {
        let tx = self.team_tx("`qal_change`")?;
        let role = role_of(&tx, &signer.member)?;
        let row = require(&tx, id)?;
        let kind = row.kind_for(&signer.member);
        if !policy::allows("qal_change", Some("submit"), role, kind) {
            return Err(one(if kind == policy::Space::Own { policy::forbidden("qal_change", Some("submit"), role, kind) } else { not_owner(&row, "submit") }));
        }
        if row.status != "open" {
            return Err(one(state(format!("change set {id} is {}; only an open change set is submitted", row.status))));
        }
        tx.execute("UPDATE changes SET status = 'submitted', updated_at = ?2 WHERE id = ?1", params![id, now_ms()]).map_err(sql)?;
        let record = record_change(&tx, signer, id, "submit", json!({}))?;
        let current = root_of(&space_rows(&tx, &Space::Change(id.to_string()))?);
        notify_roles(&tx, &["verifier", "reviewer"], &signer.member, "change_submitted", &json!({"change": id, "title": row.title, "by": signer.member, "root": current}))?;
        tx.commit().map_err(sql)?;
        Ok(json!({"change": id, "status": "submitted", "root": current, "record": record}))
    }

    /// `qal_change abandon` (owner, or any architect): the change set ends; its leases are free.
    pub fn change_abandon(&mut self, signer: &Signer, id: &str) -> Result<Value, Vec<Diagnostic>> {
        let tx = self.team_tx("`qal_change`")?;
        let role = role_of(&tx, &signer.member)?;
        let row = require(&tx, id)?;
        let kind = row.kind_for(&signer.member);
        if !policy::allows("qal_change", Some("abandon"), role, kind) {
            return Err(one(if role == Role::Reviewer { policy::forbidden("qal_change", Some("abandon"), role, kind) } else { not_owner(&row, "abandon") }));
        }
        if !row.is_live() {
            return Err(one(not_live(&row)));
        }
        tx.execute("UPDATE changes SET status = 'abandoned', updated_at = ?2 WHERE id = ?1", params![id, now_ms()]).map_err(sql)?;
        tx.execute("DELETE FROM leases WHERE change = ?1", params![id]).map_err(sql)?;
        tx.execute("DELETE FROM seen WHERE space = ?1", params![id]).map_err(sql)?;
        tx.execute("UPDATE sessions SET current_change = NULL WHERE current_change = ?1", params![id]).map_err(sql)?;
        let record = record_change(&tx, signer, id, "abandon", json!({}))?;
        if row.owner != signer.member {
            notify(&tx, &row.owner, "change_abandoned", &json!({"change": id, "title": row.title, "by": signer.member}))?;
        }
        tx.commit().map_err(sql)?;
        Ok(json!({"change": id, "status": "abandoned", "record": record}))
    }

    /// `qal_change scope` (architect): adds and removes names of a live change set's scope.
    pub fn change_scope(&mut self, signer: &Signer, id: &str, add: &[String], remove: &[String]) -> Result<Value, Vec<Diagnostic>> {
        check_names(add, "add", MAX_SCOPE)?;
        check_names(remove, "remove", MAX_SCOPE)?;
        let tx = self.team_tx("`qal_change`")?;
        let role = role_of(&tx, &signer.member)?;
        let row = require(&tx, id)?;
        let kind = row.kind_for(&signer.member);
        if !policy::allows("qal_change", Some("scope"), role, kind) {
            return Err(one(policy::forbidden("qal_change", Some("scope"), role, kind)));
        }
        if !row.is_live() {
            return Err(one(not_live(&row)));
        }
        let mut scope: BTreeSet<String> = row.scope.iter().cloned().collect();
        scope.extend(add.iter().cloned());
        for r in remove {
            scope.remove(r);
        }
        let scope: Vec<String> = scope.into_iter().collect();
        if scope.len() > MAX_SCOPE {
            return Err(one(team_limit(format!("a scope has at most {MAX_SCOPE} names"))));
        }
        tx.execute("UPDATE changes SET scope = ?2, updated_at = ?3 WHERE id = ?1", params![id, list_json(&scope), now_ms()]).map_err(sql)?;
        let record = record_change(&tx, signer, id, "scope", json!({"add": add, "remove": remove}))?;
        tx.commit().map_err(sql)?;
        Ok(json!({"change": id, "scope": scope, "record": record}))
    }

    /// `qal_change list` (the board): change sets with status in `statuses` (default: open and
    /// submitted), only `member`'s (owned or assigned) if `mine`, newest first, at most `limit`
    /// (default 50, at most 200; `more` counts the rest).
    pub fn change_list(&mut self, member: &str, statuses: &[String], mine: bool, limit: Option<usize>) -> Result<Value, Vec<Diagnostic>> {
        let live = ["open".to_string(), "submitted".to_string()];
        let statuses: &[String] = if statuses.is_empty() { &live } else { statuses };
        let limit = limit.unwrap_or(50).clamp(1, 200);
        if !super::is_team(&self.conn)? {
            return Err(one(super::not_a_team("`qal_change`")));
        }
        let tx = self.conn.transaction().map_err(sql)?;
        let ids: Vec<String> = {
            let mut stmt = tx.prepare("SELECT id FROM changes ORDER BY created_at DESC, id").map_err(sql)?;
            let rows = stmt.query_map([], |r| r.get(0)).map_err(sql)?;
            rows.collect::<Result<Vec<_>, _>>().map_err(sql)?
        };
        let mut out = Vec::new();
        let mut more = 0usize;
        for id in ids {
            let row = require(&tx, &id)?;
            if !statuses.contains(&row.status) {
                continue;
            }
            if mine && row.owner != member && row.assignee.as_deref() != Some(member) {
                continue;
            }
            if out.len() >= limit {
                more += 1;
                continue;
            }
            out.push(summary(&tx, &row)?);
        }
        Ok(if more > 0 { json!({"changes": out, "more": more}) } else { json!({"changes": out}) })
    }

    /// `qal_change show`: a change set with its description, base, changed names and leases.
    pub fn change_show(&mut self, id: &str) -> Result<Value, Vec<Diagnostic>> {
        if !super::is_team(&self.conn)? {
            return Err(one(super::not_a_team("`qal_change`")));
        }
        let tx = self.conn.transaction().map_err(sql)?;
        let row = require(&tx, id)?;
        let mut v = summary(&tx, &row)?;
        let mut names = Vec::new();
        {
            let mut stmt = tx.prepare("SELECT name, origin FROM change_names WHERE change = ?1 AND origin != 'base' ORDER BY name").map_err(sql)?;
            let rows = stmt.query_map(params![id], |r| Ok(json!({"name": r.get::<_, String>(0)?, "origin": r.get::<_, String>(1)?}))).map_err(sql)?;
            for r in rows {
                names.push(r.map_err(sql)?);
            }
        }
        if let Value::Object(m) = &mut v {
            m.insert("description".into(), json!(row.description));
            m.insert("base_seq".into(), json!(row.base_seq));
            m.insert("base_root".into(), json!(row.base_root));
            m.insert("created_at".into(), json!(row.created_at));
            m.insert("updated_at".into(), json!(row.updated_at));
            m.insert("names".into(), Value::Array(names));
            m.insert("leases".into(), serde_json::to_value(leases_of(&tx, id)?).unwrap_or(Value::Null));
        }
        Ok(v)
    }

    /// `qal_change diff`: every name the change set touched, with its hash at the base, in the
    /// change set (`ours`) and in `main` now (null where it has none).
    pub fn change_diff(&mut self, id: &str) -> Result<Value, Vec<Diagnostic>> {
        if !super::is_team(&self.conn)? {
            return Err(one(super::not_a_team("`qal_change`")));
        }
        let tx = self.conn.transaction().map_err(sql)?;
        require(&tx, id)?;
        let mut diff = Vec::new();
        {
            let mut stmt = tx
                .prepare("SELECT c.name, c.origin, c.base_hash, c.hash, n.hash, c.conflict FROM change_names c LEFT JOIN names n ON n.name = c.name WHERE c.change = ?1 AND c.origin != 'base' ORDER BY c.name")
                .map_err(sql)?;
            let rows = stmt
                .query_map(params![id], |r| {
                    Ok(json!({"name": r.get::<_, String>(0)?, "origin": r.get::<_, String>(1)?, "base": r.get::<_, Option<String>>(2)?, "ours": r.get::<_, Option<String>>(3)?, "main": r.get::<_, Option<String>>(4)?, "conflict": r.get::<_, Option<String>>(5)?.and_then(|c| serde_json::from_str::<Value>(&c).ok())}))
                })
                .map_err(sql)?;
            for r in rows {
                diff.push(r.map_err(sql)?);
            }
        }
        Ok(json!({"change": id, "diff": diff}))
    }

    /// The live leases of a change set.
    pub fn leases(&self, change: &str) -> Result<Vec<Lease>, Vec<Diagnostic>> {
        leases_of(&self.conn, change)
    }

    /// `qal_lease acquire | renew | release` for change set `change` (the member's own):
    /// acquire is all or nothing (E0901), renew extends own leases, release frees them.
    pub fn lease(&mut self, member: &str, action: &str, change: &str, names: &[String], ttl_ms: Option<i64>) -> Result<Value, Vec<Diagnostic>> {
        check_names(names, "names", MAX_LEASE_NAMES)?;
        let ttl = ttl_ms.unwrap_or(DEFAULT_LEASE_TTL_MS);
        if !(MIN_LEASE_TTL_MS..=MAX_LEASE_TTL_MS).contains(&ttl) {
            return Err(one(team_limit(format!("a lease lasts {} s to {} s, not {} s", MIN_LEASE_TTL_MS / 1000, MAX_LEASE_TTL_MS / 1000, ttl / 1000))));
        }
        let tx = self.team_tx("`qal_lease`")?;
        let role = role_of(&tx, member)?;
        let row = require(&tx, change)?;
        let kind = row.kind_for(member);
        if !policy::allows("qal_lease", Some(action), role, policy::Space::Own) {
            return Err(one(policy::forbidden("qal_lease", Some(action), role, kind)));
        }
        if kind != policy::Space::Own {
            return Err(one(not_owner(&row, "a lease")));
        }
        if !row.is_live() {
            return Err(one(not_live(&row)));
        }
        let now = now_ms();
        let list: Vec<&str> = names.iter().map(String::as_str).collect();
        let out = match action {
            "acquire" => {
                acquire(&tx, change, member, &list, ttl, now, true)?;
                json!({"leases": leases_of(&tx, change)?.into_iter().filter(|l| names.contains(&l.name)).collect::<Vec<_>>()})
            }
            "renew" => {
                let mut diags = Vec::new();
                for n in &list {
                    match lease_of(&tx, n)? {
                        Some(l) if l.change == change => {}
                        Some(l) if l.expires_at > now => diags.push(lease_held(&l)),
                        _ => diags.push(state(format!("change set {change} holds no lease on `{n}`")).hint("acquire it first (qal_lease acquire)")),
                    }
                }
                if !diags.is_empty() {
                    return Err(diags);
                }
                for n in &list {
                    tx.execute("UPDATE leases SET expires_at = ?3, member = ?4 WHERE name = ?1 AND change = ?2", params![n, change, now + ttl, member]).map_err(sql)?;
                }
                json!({"leases": leases_of(&tx, change)?.into_iter().filter(|l| names.contains(&l.name)).collect::<Vec<_>>()})
            }
            "release" => {
                let mut released = Vec::new();
                for n in &list {
                    if tx.execute("DELETE FROM leases WHERE name = ?1 AND change = ?2", params![n, change]).map_err(sql)? > 0 {
                        released.push(n.to_string());
                    }
                }
                json!({"released": released, "leases": leases_of(&tx, change)?})
            }
            other => return Err(one(bad(format!("`{other}` is not a lease action (acquire, renew, release, break)")))),
        };
        tx.commit().map_err(sql)?;
        Ok(out)
    }

    /// `qal_lease break` (architect): removes other change sets' live leases on `names`, each with a
    /// signed `lease_break` record; the holder hears of it and its change set needs a rebase.
    pub fn lease_break(&mut self, signer: &Signer, names: &[String]) -> Result<Value, Vec<Diagnostic>> {
        check_names(names, "names", MAX_LEASE_NAMES)?;
        let tx = self.team_tx("`qal_lease`")?;
        let role = role_of(&tx, &signer.member)?;
        if !policy::allows("qal_lease", Some("break"), role, policy::Space::ForeignOther) {
            return Err(one(policy::forbidden("qal_lease", Some("break"), role, policy::Space::ForeignOther)));
        }
        let now = now_ms();
        let mut broken = Vec::new();
        let mut skipped = Vec::new();
        for n in names {
            let Some(l) = lease_of(&tx, n)? else { continue };
            // Breaking applies to other members' change sets: own leases are released instead.
            if load(&tx, &l.change)?.is_some_and(|c| c.owner == signer.member) {
                skipped.push(json!({"name": n, "change": l.change, "why": "own change set: release it (qal_lease release)"}));
                continue;
            }
            tx.execute("DELETE FROM leases WHERE name = ?1", params![n]).map_err(sql)?;
            if l.expires_at <= now {
                continue;
            }
            let space = Space::Change(l.change.clone());
            let held = space_rows(&tx, &space)?;
            let record = append_in(&tx, signer, "lease_break", json!({"name": n, "change": l.change, "holder": l.member}), &space, &held, &held)?;
            tx.execute("UPDATE changes SET needs_rebase = 1, updated_at = ?2 WHERE id = ?1", params![l.change, now]).map_err(sql)?;
            notify(&tx, &l.member, "lease_broken", &json!({"name": n, "change": l.change, "by": signer.member, "record": record}))?;
            broken.push(json!({"name": n, "change": l.change, "holder": l.member, "record": record}));
        }
        tx.commit().map_err(sql)?;
        Ok(json!({"broken": broken, "skipped": skipped}))
    }

    /// Notes that `session` saw these (name, hash) in `space` (`qal_view`, §5.3), in the team log:
    /// a read never writes `store.db`. Best effort: a failure only means a later write may be E0904.
    pub fn mark_seen(&self, session: &str, space: &Space, seen: &[(String, String)]) -> Result<(), Vec<Diagnostic>> {
        if seen.is_empty() || !super::is_team(&self.conn)? {
            return Ok(());
        }
        let log = self.team_log()?;
        for (n, h) in seen {
            log.mark_seen(session, space.id(), n, h)?;
        }
        Ok(())
    }

    /// The change set `id`, if it exists.
    pub fn change(&self, id: &str) -> Result<Option<ChangeRow>, Vec<Diagnostic>> {
        load(&self.conn, id)
    }
}
