//! Team mode (Phase 4 spec §4, §8, §9, §10): members with roles and Ed25519 keys, sessions,
//! signed records of every write, notices and the call log.
//!
//! A store is a team store once `qal team init` set `meta.team = 1`. From then on every write of
//! code (define, rename, delete, import) needs a [`Signer`] and appends, in the same transaction, a
//! signed record to one hash chain (`records`); a write without a signer is refused (E0903).
//! Code is written in change sets (Phase 4b, [`change`]); `main` changes only by merge (4c), by the
//! owner's `qal team sync-std`, and by nothing else.

pub mod canonical;
pub mod change;
pub mod keys;
pub mod log;
pub mod policy;
pub mod verify;

#[cfg(test)]
mod tests;

use crate::{now_ms, sql, Defined, Store};
use canonical::hex;
use keys::{KeyDir, SecretKey};
pub use log::{Call, TeamLog};
use policy::{Role, OWNER};
use qal_core::canon::Hash;
use qal_core::diag::Diagnostic;
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;
use zeroize::Zeroizing;

/// `meta` keys of a team store.
pub const META_TEAM: &str = "team";
pub const META_TEAM_ALLOW: &str = "team_allow";
pub const META_OWNER: &str = "owner";
/// The largest `history.id` when the team began: later rows each belong to a record.
pub const META_TEAM_HISTORY: &str = "team_history_from";
/// The effects agents may grant in a team store unless the owner changes `meta.team_allow`.
pub const DEFAULT_ALLOW: &str = "io,time,rand,db:mem";
/// The first line of every record body.
pub const RECORD_PREFIX: &str = "qal-record-v1\n";
/// The `prev` of the first record.
pub const ZERO_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";
/// The session of the owner's command-line records.
pub const CLI_SESSION: &str = "cli";

/// Limits of spec §14 that apply in 4a.
pub const MAX_MEMBERS: i64 = 64;
pub const MAX_ACTIVE_SESSIONS: i64 = 128;
pub const DEFAULT_SESSION_TTL_MS: i64 = 24 * 3600 * 1000;
pub const MAX_SESSION_TTL_MS: i64 = 7 * 24 * 3600 * 1000;
pub const MAX_RECORD_BYTES: usize = 1 << 20;
pub const MAX_NOTICES: i64 = 1000;
pub const NOTICES_PER_REPLY: usize = 10;

/// E0900 `not_a_team`.
pub fn not_a_team(what: &str) -> Diagnostic {
    Diagnostic::new("E0900", "not_a_team", "", "", format!("{what} needs a team store, and this store is not one")).hint("make it one with `qal team init` (the owner, on the command line)")
}

/// E0902 `no_session`.
pub fn no_session(message: impl Into<String>) -> Diagnostic {
    Diagnostic::new("E0902", "no_session", "", "", message).hint("ask the owner for a session file (`qal team session <member> --out <file>`) and start the server with `qal mcp --session-file <file>` or QAL_SESSION_FILE")
}

/// E0914 `team_limit`.
pub fn team_limit(message: impl Into<String>) -> Diagnostic {
    Diagnostic::new("E0914", "team_limit", "", "", message)
}

/// E0908 `change_state` (an action that does not fit the current state).
pub(crate) fn state(message: impl Into<String>) -> Diagnostic {
    Diagnostic::new("E0908", "change_state", "", "", message)
}

pub(crate) fn bad(message: impl Into<String>) -> Diagnostic {
    Diagnostic::new("E0701", "bad_arguments", "", "", message)
}

pub(crate) fn one(d: Diagnostic) -> Vec<Diagnostic> {
    vec![d]
}

/// A member id: `[a-z][a-z0-9-]{0,31}`.
pub fn valid_member_id(id: &str) -> bool {
    (1..=32).contains(&id.len()) && id.starts_with(|c: char| c.is_ascii_lowercase()) && id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// Who signs the records of writes: a member, its session (`cli` for the owner) and its key.
#[derive(Debug)]
pub struct Signer {
    pub member: String,
    pub session: String,
    pub key: SecretKey,
}

/// One change of a name in a record: `old`/`new` are hashes, `None` where the name did not exist
/// before / does not exist after (a deleted name, the old name of a rename). `refs` and `locals`
/// are the name's full row after the change (`names.refs`, `names.local_names`; `None` with `new`),
/// so replaying the records rebuilds every name exactly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub name: String,
    pub old: Option<String>,
    pub new: Option<String>,
    pub action: String,
    #[serde(default)]
    pub refs: Option<Vec<String>>,
    #[serde(default)]
    pub locals: Option<Vec<String>>,
}

impl Entry {
    /// A label of a write's result (the action per name; the rows come from the store).
    pub fn label(name: &str, old: Option<&str>, new: Option<&str>, action: &str) -> Entry {
        Entry { name: name.into(), old: old.map(str::to_string), new: new.map(str::to_string), action: action.into(), refs: None, locals: None }
    }

    /// The labels of a `define`/import result: every name that changed (`unchanged` ones are not).
    pub fn of_defined(defined: &[Defined]) -> Vec<Entry> {
        defined.iter().filter(|d| d.status != "unchanged").map(|d| Entry::label(&d.name, d.old_hash.as_deref(), Some(&d.hash), d.status)).collect()
    }
}

/// The full stored state of one name: what `names` holds besides its time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NameRow {
    pub hash: String,
    pub refs: Vec<String>,
    pub locals: Vec<String>,
}

/// All names of a space with their rows.
pub type Rows = BTreeMap<String, NameRow>;

/// True if the store is a team store (`meta.team = 1`).
pub fn is_team(conn: &Connection) -> Result<bool, Vec<Diagnostic>> {
    Ok(crate::package::meta(conn, META_TEAM)?.as_deref() == Some("1"))
}

/// The current `name -> hash` of `main`.
pub fn main_names(conn: &Connection) -> Result<BTreeMap<String, String>, Vec<Diagnostic>> {
    Ok(main_rows(conn)?.into_iter().map(|(n, r)| (n, r.hash)).collect())
}

fn list(text: &str, what: &str, name: &str) -> Result<Vec<String>, Vec<Diagnostic>> {
    serde_json::from_str(text).map_err(|e| one(Diagnostic::store_error(format!("the store is corrupt: {what} of `{name}` is not a JSON list of strings ({e})"))))
}

/// The current rows of `main`.
pub fn main_rows(conn: &Connection) -> Result<Rows, Vec<Diagnostic>> {
    let mut stmt = conn.prepare("SELECT name, hash, refs, local_names FROM names").map_err(sql)?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?, r.get::<_, String>(3)?))).map_err(sql)?;
    let mut out = Rows::new();
    for row in rows {
        let (name, hash, refs, locals) = row.map_err(sql)?;
        let row = NameRow { hash, refs: list(&refs, "names.refs", &name)?, locals: list(&locals, "names.local_names", &name)? };
        out.insert(name, row);
    }
    Ok(out)
}

/// The current rows of a space (`main`, or a change set's names that have a hash).
pub fn space_rows(conn: &Connection, space: &Space) -> Result<Rows, Vec<Diagnostic>> {
    let Space::Change(_) = space else { return main_rows(conn) };
    let mut out = Rows::new();
    for (name, hash, locals, refs) in crate::space::rows(conn, space)? {
        let row = NameRow { hash, refs: list(&refs, "change_names.refs", &name)?, locals: list(&locals, "change_names.local_names", &name)? };
        out.insert(name, row);
    }
    Ok(out)
}

/// The rows of a snapshot (the state a write started from, inside its transaction).
pub(crate) fn rows_of(snap: &crate::snapshot::Snapshot) -> Rows {
    snap.names.iter().map(|(n, e)| (n.clone(), NameRow { hash: e.hash.to_hex(), refs: e.refs.clone(), locals: e.locals.clone() })).collect()
}

/// The root (§3) of rows: their names and hashes.
pub fn root_of(rows: &Rows) -> String {
    canonical::root_iter(rows.iter().map(|(n, r)| (n.as_str(), r.hash.as_str())))
}

/// SHA-256 over the full state of every name: per name (byte order) the line
/// `name 0 hash 0 refs 0 local_names 0 kind 0 signature 0 effects 0 defs.local_names 10`, with the
/// definition's columns as stored in `defs` (JSON lists in compact form). Each record carries this
/// digest of the state after it (`names_after`), so `qal verify` detects any change of a name's
/// row or of its definition's columns, not only of its hash.
pub fn names_digest(conn: &Connection, rows: &Rows) -> Result<String, Vec<Diagnostic>> {
    let mut stmt = conn.prepare_cached("SELECT kind, signature, effects, local_names FROM defs WHERE hash = ?1").map_err(sql)?;
    let mut bytes = Vec::new();
    let mut cache: BTreeMap<&str, [String; 4]> = BTreeMap::new();
    for (name, row) in rows {
        if !cache.contains_key(row.hash.as_str()) {
            let cols: Option<[String; 4]> = stmt.query_row(params![row.hash], |r| Ok([r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?])).optional().map_err(sql)?;
            let cols = cols.ok_or_else(|| one(Diagnostic::store_error(format!("the store is corrupt: `{name}` points to the missing definition #{}", row.hash))))?;
            cache.insert(row.hash.as_str(), cols);
        }
        let as_json = |v: &Vec<String>| serde_json::Value::from(v.clone()).to_string();
        let mut fields = vec![name.clone(), row.hash.clone(), as_json(&row.refs), as_json(&row.locals)];
        fields.extend(cache[row.hash.as_str()].iter().cloned());
        bytes.extend_from_slice(fields.join("\0").as_bytes());
        bytes.push(b'\n');
    }
    Ok(Hash::of(&bytes).to_hex())
}

/// The entries of a write: every name whose row differs between `before` and `after` (by name),
/// with the action from `labels` (or `updated` where only `refs`/`local_names` changed, e.g. the
/// users of a renamed name).
fn diff(before: &Rows, after: &Rows, labels: &[Entry]) -> Vec<Entry> {
    let label: BTreeMap<&str, &str> = labels.iter().map(|e| (e.name.as_str(), e.action.as_str())).collect();
    let names: std::collections::BTreeSet<&String> = before.keys().chain(after.keys()).collect();
    let mut out = Vec::new();
    for n in names {
        let (b, a) = (before.get(n), after.get(n));
        if b == a {
            continue;
        }
        let action = label.get(n.as_str()).copied().unwrap_or(match (b, a) {
            (None, _) => "added",
            (_, None) => "deleted",
            (Some(x), Some(y)) if x.hash == y.hash => "updated",
            _ => "replaced",
        });
        out.push(Entry { name: n.clone(), old: b.map(|r| r.hash.clone()), new: a.map(|r| r.hash.clone()), action: action.into(), refs: a.map(|r| r.refs.clone()), locals: a.map(|r| r.locals.clone()) });
    }
    out
}

/// A member row.
#[derive(Debug, Clone)]
pub struct Member {
    pub id: String,
    pub role: String,
    pub key: String,
    pub key_id: String,
    pub revoked_at: Option<i64>,
}

pub(crate) fn member(conn: &Connection, id: &str) -> Result<Option<Member>, Vec<Diagnostic>> {
    conn.query_row("SELECT id, role, key, key_id, revoked_at FROM members WHERE id = ?1", params![id], |r| {
        Ok(Member { id: r.get(0)?, role: r.get(1)?, key: r.get(2)?, key_id: r.get(3)?, revoked_at: r.get(4)? })
    })
    .optional()
    .map_err(sql)
}

/// Refuses a code write in a team store without a signer (E0903): only signed writes change a
/// team store. Returns whether the store is a team store.
pub(crate) fn require_signer(conn: &Connection, signer: Option<&Signer>) -> Result<bool, Vec<Diagnostic>> {
    if !is_team(conn)? {
        return Ok(false);
    }
    if signer.is_none() {
        let mut d = Diagnostic::new("E0903", "role_forbidden", "", "", "this is a team store: code is written by agents with a session, not on the command line");
        d.expected = Some("architect".into());
        d.found = Some(OWNER.into());
        return Err(one(d.hint("agents write over MCP (`qal mcp --session-file <file>`); with Phase 4b in change sets, merged into main; the owner manages the team (`qal team ...`) and imports the standard library with `qal team sync-std`")));
    }
    Ok(true)
}

/// Appends the signed record of a write to the chain inside `tx` (spec §9.1) and links the history
/// rows of this transaction to it. Refuses (and so rolls back the write) if the signer may not
/// make this record. Returns its sequence number.
pub(crate) fn append(tx: &Transaction, signer: &Signer, op: &str, args: Value, before: &Rows, after: &Rows) -> Result<i64, Vec<Diagnostic>> {
    append_in(tx, signer, op, args, &Space::Main, before, after)
}

/// [`append`] for a record about `space` (its `space` field; the roots and `names_after` are of
/// that space).
pub(crate) fn append_in(tx: &Transaction, signer: &Signer, op: &str, args: Value, space: &Space, before: &Rows, after: &Rows) -> Result<i64, Vec<Diagnostic>> {
    let m = member(tx, &signer.member)?.ok_or_else(|| one(no_session(format!("`{}` is not a member of this team", signer.member))))?;
    if m.revoked_at.is_some() {
        return Err(one(no_session(format!("the member `{}` was revoked", m.id))));
    }
    if m.key_id != signer.key.key_id() {
        return Err(one(keys::unusable(format!("key {} is not the current key of `{}` (that is {})", signer.key.key_id(), m.id, m.key_id)).hint("the member was rekeyed: restart the server so it loads the new key")));
    }
    let kind = change::kind_of(tx, space, &m.id)?;
    // A record about a change set signs its metadata (review I4).
    let mut args = args;
    if let (Space::Change(c), Value::Object(map)) = (space, &mut args) {
        map.insert("meta_after".into(), Value::from(change::meta_digest(tx, c)?));
    }
    let action = args.get("action").and_then(Value::as_str);
    if !policy::record_allowed(op, action, &m.role, kind) {
        let mut d = Diagnostic::new("E0903", "role_forbidden", "", "", format!("the role {} may not make `{op}` records {}", m.role, if space.is_main() { "on main".to_string() } else { format!("in change set {space}") }));
        d.expected = Some(if op.starts_with("member") || op == "import" || op == "genesis" { OWNER.to_string() } else { "architect".to_string() });
        d.found = Some(m.role.clone());
        return Err(one(d));
    }
    let last: Option<(i64, String)> = tx.query_row("SELECT seq, hash FROM records ORDER BY seq DESC LIMIT 1", [], |r| Ok((r.get(0)?, r.get(1)?))).optional().map_err(sql)?;
    let (seq, prev) = match last {
        Some((s, h)) => (s + 1, h),
        None => (1, ZERO_HASH.to_string()),
    };
    let body = json!({
        "v": 1, "seq": seq, "prev": prev, "at": now_ms(), "member": m.id, "key": m.key_id, "session": signer.session,
        "op": op, "space": space.id(), "args": args, "root_before": root_of(before), "root_after": root_of(after),
        "names_after": names_digest(tx, after)?,
    });
    let bytes = record_bytes(&body).map_err(|e| one(Diagnostic::store_error(e)))?;
    if bytes.len() > MAX_RECORD_BYTES {
        return Err(one(team_limit(format!("the record of this {op} would be {} bytes, more than {MAX_RECORD_BYTES}", bytes.len())).hint("write the code in several smaller calls")));
    }
    let hash = Hash::of(&bytes).to_hex();
    let sig = signer.key.sign_hex(&bytes);
    tx.execute("INSERT INTO records (seq, hash, body, sig) VALUES (?1, ?2, ?3, ?4)", params![seq, hash, bytes, sig]).map_err(sql)?;
    let from: i64 = crate::package::meta(tx, META_TEAM_HISTORY)?.and_then(|v| v.parse().ok()).unwrap_or(0);
    tx.execute("UPDATE history SET record = ?1 WHERE record IS NULL AND id > ?2", params![seq, from]).map_err(sql)?;
    Ok(seq)
}

/// The bytes a record's signature covers: `qal-record-v1\n` and the canonical JSON of its body.
pub fn record_bytes(body: &Value) -> Result<Vec<u8>, String> {
    let text = canonical::to_string(body)?;
    Ok([RECORD_PREFIX.as_bytes(), text.as_bytes()].concat())
}

/// The record of a code write (`define`, `rename`, `delete`, `import`) in `space`, written after it
/// in `tx` (only in a team store; `signer` was required before). `before` is the state the write
/// started from (in the same transaction); `labels` name the action per name; `extra` are more
/// arguments. The signer's session has seen what it wrote (§5.3); a change set stays within its
/// delta limit; on `main`, the holders of leases on relinked names get the notice `relinked`.
pub(crate) fn record_write(tx: &Transaction, signer: Option<&Signer>, op: &str, space: &Space, before: &Rows, labels: &[Entry], extra: serde_json::Map<String, Value>) -> Result<(), Vec<Diagnostic>> {
    if !is_team(tx)? {
        return Ok(());
    }
    let Some(signer) = signer else { return require_signer(tx, None).map(|_| ()) };
    let after = space_rows(tx, space)?;
    let entries = diff(before, &after, labels);
    let mut args = extra;
    args.insert("entries".into(), serde_json::to_value(&entries).map_err(|e| one(Diagnostic::store_error(format!("cannot serialize a record: {e}"))))?);
    append_in(tx, signer, op, Value::Object(args), space, before, &after)?;
    change::after_write(tx, signer, space, &entries)
}

/// What a team write reports besides its result: the leases it took and the stubs it defined.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct WriteReport {
    pub leases: Vec<change::Lease>,
    pub stubs: Vec<String>,
}

pub(crate) use change::{guard_define, guard_names};
use crate::space::Space;

/// An agent's identity, resolved from its session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Identity {
    pub member: String,
    pub role: String,
    pub session: String,
    pub key_id: String,
    pub expires_at: i64,
    /// The session's current change set (open or submitted), if any.
    pub change: Option<String>,
}

/// SHA-256 of a session token (what `sessions.token_sha256` holds).
pub fn token_hash(token: &str) -> String {
    Hash::of(token.as_bytes()).to_hex()
}

/// The identity of the session whose token hashes to `token_sha256`: E0902 if there is none, it
/// expired or was revoked, or its member was revoked.
pub fn resolve(conn: &Connection, token_sha256: &str) -> Result<Identity, Diagnostic> {
    let row: Option<(String, String, i64, Option<i64>, Option<String>)> = conn
        .query_row(
            "SELECT s.id, s.member, s.expires_at, s.revoked_at, (SELECT c.id FROM changes c WHERE c.id = s.current_change AND c.status IN ('open', 'submitted')) FROM sessions s WHERE s.token_sha256 = ?1",
            params![token_sha256],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )
        .optional()
        .map_err(|e| Diagnostic::store_error(format!("store database error: {e}")))?;
    let Some((session, member_id, expires_at, revoked_at, change)) = row else {
        return Err(no_session("the session token is not a session of this team store"));
    };
    if revoked_at.is_some() {
        return Err(no_session(format!("the session {session} was revoked")));
    }
    if expires_at <= now_ms() {
        return Err(no_session(format!("the session {session} expired")));
    }
    let m = member(conn, &member_id).map_err(|d| d.into_iter().next().unwrap_or_else(|| Diagnostic::store_error("store error")))?.ok_or_else(|| no_session(format!("the session {session} belongs to no member")))?;
    if m.revoked_at.is_some() {
        return Err(no_session(format!("the member `{}` was revoked", m.id)));
    }
    Ok(Identity { member: m.id, role: m.role, session, key_id: m.key_id, expires_at, change })
}

/// A notice for an agent (spec §10.3).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Notice {
    pub id: i64,
    pub kind: String,
    pub at: i64,
    #[serde(flatten)]
    pub body: serde_json::Map<String, Value>,
}

/// Leaves a notice for `member` (keeping at most [`MAX_NOTICES`] per member: the oldest go).
pub fn notify(conn: &Connection, member: &str, kind: &str, body: &Value) -> Result<(), Vec<Diagnostic>> {
    conn.execute("INSERT INTO notices (member, kind, body, at) VALUES (?1, ?2, ?3, ?4)", params![member, kind, body.to_string(), now_ms()]).map_err(sql)?;
    conn.execute(
        "DELETE FROM notices WHERE member = ?1 AND id NOT IN (SELECT id FROM notices WHERE member = ?1 ORDER BY id DESC LIMIT ?2)",
        params![member, MAX_NOTICES],
    )
    .map_err(sql)?;
    Ok(())
}

/// [`notify`] unless `member` already has a notice of this kind and body (seen or not).
pub fn notify_once(conn: &Connection, member: &str, kind: &str, body: &Value) -> Result<(), Vec<Diagnostic>> {
    let known: Option<i64> = conn.query_row("SELECT id FROM notices WHERE member = ?1 AND kind = ?2 AND body = ?3", params![member, kind, body.to_string()], |r| r.get(0)).optional().map_err(sql)?;
    if known.is_none() {
        notify(conn, member, kind, body)?;
    }
    Ok(())
}

/// The oldest notices of `member` it was not given yet (at most `max`), and how many more there
/// are. Reads `store.db` only; the seen-mark is the cursor in the team log ([`TeamLog`]), moved
/// past what is returned (best effort: if the log cannot be written, they may come again).
pub fn take_notices(conn: &Connection, log: &TeamLog, member: &str, max: usize) -> Result<(Vec<Notice>, usize), Vec<Diagnostic>> {
    let cursor = log.cursor(member)?;
    let mut out = Vec::new();
    {
        let mut stmt = conn.prepare("SELECT id, kind, body, at FROM notices WHERE member = ?1 AND id > ?2 ORDER BY id LIMIT ?3").map_err(sql)?;
        let rows = stmt.query_map(params![member, cursor, max as i64], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?, r.get::<_, i64>(3)?))).map_err(sql)?;
        for row in rows {
            let (id, kind, body, at) = row.map_err(sql)?;
            let body = match serde_json::from_str::<Value>(&body) {
                Ok(Value::Object(m)) => m,
                _ => serde_json::Map::new(),
            };
            out.push(Notice { id, kind, at, body });
        }
    }
    let last = out.last().map_or(cursor, |n| n.id);
    if last > cursor {
        let _ = log.set_cursor(member, last);
    }
    let more: i64 = conn.query_row("SELECT count(*) FROM notices WHERE member = ?1 AND id > ?2", params![member, last], |r| r.get(0)).map_err(sql)?;
    Ok((out, usize::try_from(more).unwrap_or(0)))
}

/// True if `member` has notices it was not given yet.
pub fn has_notices(conn: &Connection, log: &TeamLog, member: &str) -> Result<bool, Vec<Diagnostic>> {
    let cursor = log.cursor(member)?;
    let n: i64 = conn.query_row("SELECT count(*) FROM notices WHERE member = ?1 AND id > ?2", params![member, cursor], |r| r.get(0)).map_err(sql)?;
    Ok(n > 0)
}

/// The effects agents may grant (`meta.team_allow`, comma-separated).
pub fn team_allow(conn: &Connection) -> Result<Vec<String>, Vec<Diagnostic>> {
    let v = crate::package::meta(conn, META_TEAM_ALLOW)?.unwrap_or_else(|| DEFAULT_ALLOW.to_string());
    Ok(v.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect())
}

/// What `team init` did.
#[derive(Debug, Clone, Serialize)]
pub struct Initialized {
    pub owner: String,
    pub key_id: String,
    pub record: i64,
    pub root: String,
    /// Where the owner's key file is (its directory; the file is `<key_id>.ed25519`).
    pub key_dir: String,
}

/// A member as `team add`/`rekey`/`status` report it.
#[derive(Debug, Clone, Serialize)]
pub struct MemberInfo {
    pub id: String,
    pub role: String,
    pub key_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub record: Option<i64>,
}

/// A new session: its public id and the secret token (to be written to a 0600 file, never shown).
pub struct NewSession {
    pub id: String,
    pub member: String,
    pub expires_at: i64,
    pub token: Zeroizing<String>,
}

impl std::fmt::Debug for NewSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NewSession").field("id", &self.id).field("member", &self.member).field("expires_at", &self.expires_at).finish_non_exhaustive()
    }
}

/// Parses a TTL like `90s`, `30m`, `24h`, `7d` (a plain number is seconds) into milliseconds.
pub fn parse_ttl(s: &str) -> Option<i64> {
    let (num, unit) = match s.char_indices().last() {
        Some((i, c)) if c.is_ascii_alphabetic() => (&s[..i], c),
        _ => (s, 's'),
    };
    let n: i64 = num.parse().ok().filter(|n| *n > 0)?;
    let ms = match unit {
        's' => 1000,
        'm' => 60_000,
        'h' => 3_600_000,
        'd' => 86_400_000,
        _ => return None,
    };
    n.checked_mul(ms)
}

impl Store {
    /// The project directory (`<project>/.qal/store.db`).
    pub fn project_dir(&self) -> Result<&Path, Vec<Diagnostic>> {
        self.path.parent().and_then(Path::parent).ok_or_else(|| one(Diagnostic::store_error("the store has no project directory")))
    }

    /// True if this is a team store.
    pub fn is_team(&self) -> Result<bool, Vec<Diagnostic>> {
        is_team(&self.conn)
    }

    /// Signs every following write with `signer` (team mode); `None` writes unsigned (refused in a
    /// team store).
    pub fn set_signer(&mut self, signer: Option<std::sync::Arc<Signer>>) {
        self.signer = signer;
    }

    fn require_team(&self, what: &str) -> Result<(), Vec<Diagnostic>> {
        if is_team(&self.conn)? {
            Ok(())
        } else {
            Err(one(not_a_team(what)))
        }
    }

    /// The owner's signer (the team's command line), loaded from the key directory.
    pub fn owner_signer(&self, keys: &KeyDir) -> Result<Signer, Vec<Diagnostic>> {
        self.require_team("`qal team`")?;
        let owner = crate::package::meta(&self.conn, META_OWNER)?.ok_or_else(|| one(Diagnostic::store_error("the team store has no owner")))?;
        let m = member(&self.conn, &owner)?.ok_or_else(|| one(Diagnostic::store_error(format!("the owner `{owner}` is not a member"))))?;
        let key = keys.load(self.project_dir()?, &m.key_id).map_err(one)?;
        Ok(Signer { member: m.id, session: CLI_SESSION.into(), key })
    }

    /// The signer of an agent's session (its member's current key).
    pub fn member_signer(&self, keys: &KeyDir, who: &Identity) -> Result<Signer, Vec<Diagnostic>> {
        let key = keys.load(self.project_dir()?, &who.key_id).map_err(|d| one(d.hint(format!("this session of `{}` can read but not write until its key is available; the owner can rekey it with `qal team rekey {}`", who.member, who.member))))?;
        Ok(Signer { member: who.member.clone(), session: who.session.clone(), key })
    }

    /// `qal team init`: makes this store a team store owned by `owner` (spec §4.1): creates the
    /// owner's key, sets `meta.team`, and writes the genesis record that binds the current names.
    pub fn team_init(&mut self, owner: &str, keys: &KeyDir) -> Result<Initialized, Vec<Diagnostic>> {
        if !valid_member_id(owner) {
            return Err(one(bad(format!("`{owner}` is not a member id")).hint("member ids are 1 to 32 characters `a-z`, `0-9`, `-`, starting with a letter")));
        }
        if self.is_team()? {
            return Err(one(state("this store is already a team store")));
        }
        let project = self.project_dir()?.to_path_buf();
        let key_dir = keys.ready(&project, true).map_err(one)?;
        let key = keys.generate(&project).map_err(one)?;
        let key_id = key.key_id().to_string();
        // A key whose record is not committed is removed again (no orphan key files).
        let result = (|| {
            let tx = self.conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
            if is_team(&tx)? {
                return Err(one(state("this store is already a team store")));
            }
            let now = now_ms();
            let public = keys::public_hex(&key.public());
            tx.execute("INSERT INTO members (id, role, key, key_id, added_at) VALUES (?1, ?2, ?3, ?4, ?5)", params![owner, OWNER, public, key.key_id(), now]).map_err(sql)?;
            let history: i64 = tx.query_row("SELECT coalesce(max(id), 0) FROM history", [], |r| r.get(0)).map_err(sql)?;
            crate::package::set_meta(&tx, META_TEAM, "1")?;
            crate::package::set_meta(&tx, META_TEAM_ALLOW, DEFAULT_ALLOW)?;
            crate::package::set_meta(&tx, META_OWNER, owner)?;
            crate::package::set_meta(&tx, META_TEAM_HISTORY, &history.to_string())?;
            let names = main_rows(&tx)?;
            let main: Vec<Value> = names.iter().map(|(n, r)| json!([n, r.hash, r.refs, r.locals])).collect();
            let signer = Signer { member: owner.to_string(), session: CLI_SESSION.into(), key };
            let record = append(&tx, &signer, "genesis", json!({"owner": owner, "owner_key": public, "team_allow": DEFAULT_ALLOW, "main": main}), &names, &names)?;
            tx.commit().map_err(sql)?;
            Ok(Initialized { owner: owner.into(), key_id: signer.key.key_id().into(), record, root: root_of(&names), key_dir: key_dir.display().to_string() })
        })();
        if result.is_err() {
            keys.discard(&key_id);
        }
        result
    }

    /// `qal team add`: a new member with a new key (record `member_add`, signed by the owner).
    pub fn team_add(&mut self, owner: &Signer, id: &str, role: Role, keys: &KeyDir) -> Result<MemberInfo, Vec<Diagnostic>> {
        self.require_team("`qal team add`")?;
        if !valid_member_id(id) {
            return Err(one(bad(format!("`{id}` is not a member id")).hint("member ids are 1 to 32 characters `a-z`, `0-9`, `-`, starting with a letter")));
        }
        if member(&self.conn, id)?.is_some() {
            return Err(one(state(format!("`{id}` is already a member")).hint(format!("give it a new key with `qal team rekey {id}`"))));
        }
        let count: i64 = self.conn.query_row("SELECT count(*) FROM members", [], |r| r.get(0)).map_err(sql)?;
        if count >= MAX_MEMBERS {
            return Err(one(team_limit(format!("a team has at most {MAX_MEMBERS} members (owner included)"))));
        }
        let key = keys.generate(self.project_dir()?).map_err(one)?;
        let key_id = key.key_id().to_string();
        // A key whose record is not committed is removed again (no orphan key files).
        let result = (|| {
            let public = keys::public_hex(&key.public());
            let tx = self.conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
            if member(&tx, id)?.is_some() {
                return Err(one(state(format!("`{id}` is already a member"))));
            }
            tx.execute("INSERT INTO members (id, role, key, key_id, added_at) VALUES (?1, ?2, ?3, ?4, ?5)", params![id, role.as_str(), public, key.key_id(), now_ms()]).map_err(sql)?;
            let names = main_rows(&tx)?;
            let record = append(&tx, owner, "member_add", json!({"id": id, "role": role.as_str(), "key": public}), &names, &names)?;
            tx.commit().map_err(sql)?;
            Ok(MemberInfo { id: id.into(), role: role.as_str().into(), key_id: key.key_id().into(), record: Some(record) })
        })();
        if result.is_err() {
            keys.discard(&key_id);
        }
        result
    }

    fn agent_member(&self, id: &str) -> Result<Member, Vec<Diagnostic>> {
        let m = member(&self.conn, id)?.ok_or_else(|| one(Diagnostic::unknown_name("", "", id).hint("`qal team status` lists the members")))?;
        if m.role == OWNER {
            return Err(one(state("the owner's key and membership cannot be changed in Phase 4a")));
        }
        if m.revoked_at.is_some() {
            return Err(one(state(format!("`{id}` was revoked"))));
        }
        Ok(m)
    }

    /// `qal team rekey`: a new key for a member (record `member_key`); later records of the member
    /// are signed with it, the old one no longer counts.
    pub fn team_rekey(&mut self, owner: &Signer, id: &str, keys: &KeyDir) -> Result<MemberInfo, Vec<Diagnostic>> {
        self.require_team("`qal team rekey`")?;
        let m = self.agent_member(id)?;
        let key = keys.generate(self.project_dir()?).map_err(one)?;
        let key_id = key.key_id().to_string();
        // A key whose record is not committed is removed again (no orphan key files).
        let result = (|| {
            let public = keys::public_hex(&key.public());
            let tx = self.conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
            tx.execute("UPDATE members SET key = ?1, key_id = ?2 WHERE id = ?3", params![public, key.key_id(), id]).map_err(sql)?;
            let names = main_rows(&tx)?;
            let record = append(&tx, owner, "member_key", json!({"id": id, "role": m.role, "key": public}), &names, &names)?;
            tx.commit().map_err(sql)?;
            Ok(MemberInfo { id: id.into(), role: m.role, key_id: key.key_id().into(), record: Some(record) })
        })();
        if result.is_err() {
            keys.discard(&key_id);
        }
        result
    }

    /// `qal team revoke`: ends the membership and all its sessions (record `member_revoke`).
    pub fn team_revoke(&mut self, owner: &Signer, id: &str) -> Result<MemberInfo, Vec<Diagnostic>> {
        self.require_team("`qal team revoke`")?;
        let m = self.agent_member(id)?;
        let tx = self.conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
        let now = now_ms();
        tx.execute("UPDATE members SET revoked_at = ?1 WHERE id = ?2", params![now, id]).map_err(sql)?;
        tx.execute("UPDATE sessions SET revoked_at = ?1 WHERE member = ?2 AND revoked_at IS NULL", params![now, id]).map_err(sql)?;
        let names = main_rows(&tx)?;
        let record = append(&tx, owner, "member_revoke", json!({"id": id}), &names, &names)?;
        tx.commit().map_err(sql)?;
        Ok(MemberInfo { id: id.into(), role: m.role, key_id: m.key_id, record: Some(record) })
    }

    /// `qal team session`: a session token for an agent member (only its SHA-256 is stored).
    pub fn team_session(&mut self, id: &str, ttl_ms: i64) -> Result<NewSession, Vec<Diagnostic>> {
        self.require_team("`qal team session`")?;
        if !(60_000..=MAX_SESSION_TTL_MS).contains(&ttl_ms) {
            return Err(one(team_limit(format!("a session lasts 1 minute to 7 days, not {} s", ttl_ms / 1000))));
        }
        self.agent_member(id)?;
        // Reserved up front: the token's buffer never reallocates.
        let mut token = Zeroizing::new(String::with_capacity(67));
        token.push_str("qs_");
        canonical::hex_into(keys::random32().map_err(one)?.as_slice(), &mut token);
        let sid = format!("s{}", &hex(keys::random32().map_err(one)?.as_slice())[..12]);
        let now = now_ms();
        let tx = self.conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
        let active: i64 = tx.query_row("SELECT count(*) FROM sessions WHERE revoked_at IS NULL AND expires_at > ?1", params![now], |r| r.get(0)).map_err(sql)?;
        if active >= MAX_ACTIVE_SESSIONS {
            return Err(one(team_limit(format!("at most {MAX_ACTIVE_SESSIONS} sessions can be active at once"))));
        }
        let expires_at = now + ttl_ms;
        tx.execute("INSERT INTO sessions (token_sha256, id, member, created_at, expires_at) VALUES (?1, ?2, ?3, ?4, ?5)", params![token_hash(&token), sid, id, now, expires_at]).map_err(sql)?;
        tx.commit().map_err(sql)?;
        Ok(NewSession { id: sid, member: id.into(), expires_at, token })
    }

    /// `qal team export-trust`: the owner's and members' public keys (no secrets).
    pub fn team_trust(&self) -> Result<Value, Vec<Diagnostic>> {
        self.require_team("`qal team export-trust`")?;
        let owner = crate::package::meta(&self.conn, META_OWNER)?.unwrap_or_default();
        let m = member(&self.conn, &owner)?.ok_or_else(|| one(Diagnostic::store_error("the team store has no owner")))?;
        let mut members = Vec::new();
        for x in members_of(&self.conn)? {
            members.push(json!({"id": x.id, "role": x.role, "key": x.key, "key_id": x.key_id, "revoked": x.revoked_at.is_some()}));
        }
        Ok(json!({"format": "qal-trust-v1", "owner": owner, "owner_key": m.key, "members": members}))
    }

    /// `qal team status`: members, sessions (never tokens), open change sets, leases, the chain head.
    pub fn team_status(&self) -> Result<Value, Vec<Diagnostic>> {
        self.require_team("`qal team status`")?;
        let now = now_ms();
        let members: Vec<Value> = members_of(&self.conn)?
            .into_iter()
            .map(|m| json!({"id": m.id, "role": m.role, "key_id": m.key_id, "revoked": m.revoked_at.is_some()}))
            .collect();
        let mut sessions = Vec::new();
        {
            let mut stmt = self.conn.prepare("SELECT id, member, created_at, expires_at, revoked_at FROM sessions ORDER BY created_at, id").map_err(sql)?;
            let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?, r.get::<_, i64>(3)?, r.get::<_, Option<i64>>(4)?))).map_err(sql)?;
            for row in rows {
                let (id, member, created_at, expires_at, revoked_at) = row.map_err(sql)?;
                sessions.push(json!({"id": id, "member": member, "created_at": created_at, "expires_at": expires_at, "active": revoked_at.is_none() && expires_at > now}));
            }
        }
        let mut changes = Vec::new();
        {
            let mut stmt = self.conn.prepare("SELECT id, title, status, owner FROM changes WHERE status IN ('open', 'submitted') ORDER BY created_at").map_err(sql)?;
            let rows = stmt.query_map([], |r| Ok(json!({"change": r.get::<_, String>(0)?, "title": r.get::<_, String>(1)?, "status": r.get::<_, String>(2)?, "owner": r.get::<_, String>(3)?}))).map_err(sql)?;
            for row in rows {
                changes.push(row.map_err(sql)?);
            }
        }
        let mut leases = Vec::new();
        {
            let mut stmt = self.conn.prepare("SELECT name, change, member, expires_at FROM leases WHERE expires_at > ?1 ORDER BY name").map_err(sql)?;
            let rows = stmt.query_map(params![now], |r| Ok(json!({"name": r.get::<_, String>(0)?, "change": r.get::<_, String>(1)?, "member": r.get::<_, String>(2)?, "expires_at": r.get::<_, i64>(3)?}))).map_err(sql)?;
            for row in rows {
                leases.push(row.map_err(sql)?);
            }
        }
        let (records, head): (i64, Option<String>) = self.conn.query_row("SELECT count(*), (SELECT hash FROM records ORDER BY seq DESC LIMIT 1) FROM records", [], |r| Ok((r.get(0)?, r.get(1)?))).map_err(sql)?;
        let owner = crate::package::meta(&self.conn, META_OWNER)?;
        Ok(json!({"owner": owner, "members": members, "sessions": sessions, "changes": changes, "leases": leases, "records": records, "head": head, "allow": team_allow(&self.conn)?}))
    }

    /// `qal team stats`: the call log summed up per member and tool (spec §10.4).
    pub fn team_stats(&self) -> Result<Value, Vec<Diagnostic>> {
        self.require_team("`qal team stats`")?;
        let rows = self.team_log()?.calls()?;
        #[derive(Default)]
        struct Sum {
            calls: i64,
            failed: i64,
            ms: i64,
            tools: BTreeMap<String, i64>,
            codes: BTreeMap<String, i64>,
        }
        let mut by: BTreeMap<String, Sum> = BTreeMap::new();
        let mut total = Sum::default();
        for (member, tool, ok, code, ms) in rows {
            for s in [by.entry(member).or_default(), &mut total] {
                s.calls += 1;
                s.ms += ms;
                *s.tools.entry(tool.clone()).or_default() += 1;
                if !ok {
                    s.failed += 1;
                    *s.codes.entry(code.clone().unwrap_or_else(|| "error".into())).or_default() += 1;
                }
            }
        }
        let show = |s: &Sum| json!({"calls": s.calls, "failed": s.failed, "ms": s.ms, "tools": s.tools, "failed_by_code": s.codes});
        let members: BTreeMap<&String, Value> = by.iter().map(|(m, s)| (m, show(s))).collect();
        Ok(json!({"members": members, "total": show(&total)}))
    }

    /// The identity of a session token's hash (E0902 if it is not a valid session).
    pub fn resolve_session(&self, token_sha256: &str) -> Result<Identity, Diagnostic> {
        resolve(&self.conn, token_sha256)
    }

    pub fn notify(&self, member: &str, kind: &str, body: &Value) -> Result<(), Vec<Diagnostic>> {
        notify(&self.conn, member, kind, body)
    }

    pub fn notify_once(&self, member: &str, kind: &str, body: &Value) -> Result<(), Vec<Diagnostic>> {
        notify_once(&self.conn, member, kind, body)
    }

    pub fn take_notices(&self, log: &TeamLog, member: &str, max: usize) -> Result<(Vec<Notice>, usize), Vec<Diagnostic>> {
        take_notices(&self.conn, log, member, max)
    }

    pub fn has_notices(&self, log: &TeamLog, member: &str) -> Result<bool, Vec<Diagnostic>> {
        has_notices(&self.conn, log, member)
    }

    /// The bookkeeping file of this team store (`.qal/team-log.db`, see [`log`]).
    pub fn team_log(&self) -> Result<TeamLog, Vec<Diagnostic>> {
        TeamLog::open(&self.path)
    }

    pub fn team_allow(&self) -> Result<Vec<String>, Vec<Diagnostic>> {
        team_allow(&self.conn)
    }

    /// `qal verify`: checks the record chain (spec §9.2) against `trust` (or the genesis record);
    /// signatures from record `from` on; `head`: a record hash the chain must contain (pinned by
    /// the owner earlier, so a cut-off tail is found).
    pub fn verify(&mut self, trust: Option<&verify::Trust>, from: i64, head: Option<&str>) -> Result<verify::Outcome, Vec<Diagnostic>> {
        if !is_team(&self.conn)? {
            // Records without `meta.team`: the mark was removed, not a store that never was a team.
            let records: i64 = self.conn.query_row("SELECT count(*) FROM records", [], |r| r.get(0)).map_err(sql)?;
            if records > 0 {
                return Ok(verify::Outcome::tampered(format!("the store has {records} records but is not marked as a team store (meta.team was removed)")));
            }
            return Err(one(not_a_team("`qal verify`")));
        }
        let tx = self.conn.transaction().map_err(sql)?;
        verify::verify(&tx, trust, from, head)
    }
}

/// All members, by id.
pub(crate) fn members_of(conn: &Connection) -> Result<Vec<Member>, Vec<Diagnostic>> {
    let mut stmt = conn.prepare("SELECT id, role, key, key_id, revoked_at FROM members ORDER BY id").map_err(sql)?;
    let rows = stmt.query_map([], |r| Ok(Member { id: r.get(0)?, role: r.get(1)?, key: r.get(2)?, key_id: r.get(3)?, revoked_at: r.get(4)? })).map_err(sql)?;
    rows.collect::<Result<Vec<_>, _>>().map_err(sql)
}
