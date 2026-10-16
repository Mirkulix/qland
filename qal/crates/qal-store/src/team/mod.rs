//! Team mode (Phase 4 spec §4, §8, §9, §10): members with roles and Ed25519 keys, sessions,
//! signed records of every write, notices and the call log.
//!
//! A store is a team store once `qal team init` set `meta.team = 1`. From then on every write of
//! code (define, rename, delete, import) needs a [`Signer`] and appends, in the same transaction, a
//! signed record to one hash chain (`records`); a write without a signer is refused (E0903).
//! Until Phase 4b only architects write, directly on `main` ([`policy::ARCHITECTS_WRITE_MAIN`]).

pub mod canonical;
pub mod keys;
pub mod policy;
pub mod verify;

#[cfg(test)]
mod tests;

use crate::{now_ms, sql, Defined, Store};
use canonical::{hex, root};
use keys::{KeyDir, SecretKey};
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
fn state(message: impl Into<String>) -> Diagnostic {
    Diagnostic::new("E0908", "change_state", "", "", message)
}

fn bad(message: impl Into<String>) -> Diagnostic {
    Diagnostic::new("E0701", "bad_arguments", "", "", message)
}

fn one(d: Diagnostic) -> Vec<Diagnostic> {
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
/// before / does not exist after (a deleted name, the old name of a rename).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub name: String,
    pub old: Option<String>,
    pub new: Option<String>,
    pub action: String,
}

impl Entry {
    /// The entries of a `define`/import result: every name that changed (`unchanged` ones are not).
    pub fn of_defined(defined: &[Defined]) -> Vec<Entry> {
        defined.iter().filter(|d| d.status != "unchanged").map(|d| Entry { name: d.name.clone(), old: d.old_hash.clone(), new: Some(d.hash.clone()), action: d.status.to_string() }).collect()
    }
}

/// True if the store is a team store (`meta.team = 1`).
pub fn is_team(conn: &Connection) -> Result<bool, Vec<Diagnostic>> {
    Ok(crate::package::meta(conn, META_TEAM)?.as_deref() == Some("1"))
}

/// The current `name -> hash` of `main`.
pub fn main_names(conn: &Connection) -> Result<BTreeMap<String, String>, Vec<Diagnostic>> {
    let mut stmt = conn.prepare("SELECT name, hash FROM names").map_err(sql)?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))).map_err(sql)?;
    rows.collect::<Result<BTreeMap<_, _>, _>>().map_err(sql)
}

/// The state before `entries` were applied to `after`.
fn undo(after: &BTreeMap<String, String>, entries: &[Entry]) -> BTreeMap<String, String> {
    let mut m = after.clone();
    for e in entries.iter().rev() {
        match &e.old {
            Some(h) => m.insert(e.name.clone(), h.clone()),
            None => m.remove(&e.name),
        };
    }
    m
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

fn member(conn: &Connection, id: &str) -> Result<Option<Member>, Vec<Diagnostic>> {
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
pub(crate) fn append(tx: &Transaction, signer: &Signer, op: &str, args: Value, before: &BTreeMap<String, String>, after: &BTreeMap<String, String>) -> Result<i64, Vec<Diagnostic>> {
    let m = member(tx, &signer.member)?.ok_or_else(|| one(no_session(format!("`{}` is not a member of this team", signer.member))))?;
    if m.revoked_at.is_some() {
        return Err(one(no_session(format!("the member `{}` was revoked", m.id))));
    }
    if m.key_id != signer.key.key_id() {
        return Err(one(keys::unusable(format!("key {} is not the current key of `{}` (that is {})", signer.key.key_id(), m.id, m.key_id)).hint("the member was rekeyed: restart the server so it loads the new key")));
    }
    if !policy::record_allowed(op, &m.role) {
        let mut d = Diagnostic::new("E0903", "role_forbidden", "", "", format!("the role {} may not make `{op}` records", m.role));
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
        "op": op, "space": "main", "args": args, "root_before": root(before), "root_after": root(after),
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

/// The record of a code write (`define`, `rename`, `delete`, `import`) whose changes are `entries`,
/// written after them in `tx` (only in a team store; `signer` was required before).
pub(crate) fn record_write(tx: &Transaction, signer: Option<&Signer>, op: &str, entries: &[Entry]) -> Result<(), Vec<Diagnostic>> {
    if !is_team(tx)? {
        return Ok(());
    }
    let Some(signer) = signer else { return require_signer(tx, None).map(|_| ()) };
    let after = main_names(tx)?;
    let before = undo(&after, entries);
    let entries = serde_json::to_value(entries).map_err(|e| one(Diagnostic::store_error(format!("cannot serialize a record: {e}"))))?;
    append(tx, signer, op, json!({ "entries": entries }), &before, &after).map(|_| ())
}

/// An agent's identity, resolved from its session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Identity {
    pub member: String,
    pub role: String,
    pub session: String,
    pub key_id: String,
    pub expires_at: i64,
}

/// SHA-256 of a session token (what `sessions.token_sha256` holds).
pub fn token_hash(token: &str) -> String {
    Hash::of(token.as_bytes()).to_hex()
}

/// The identity of the session whose token hashes to `token_sha256`: E0902 if there is none, it
/// expired or was revoked, or its member was revoked.
pub fn resolve(conn: &Connection, token_sha256: &str) -> Result<Identity, Diagnostic> {
    let row: Option<(String, String, i64, Option<i64>)> = conn
        .query_row("SELECT id, member, expires_at, revoked_at FROM sessions WHERE token_sha256 = ?1", params![token_sha256], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
        .optional()
        .map_err(|e| Diagnostic::store_error(format!("store database error: {e}")))?;
    let Some((session, member_id, expires_at, revoked_at)) = row else {
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
    Ok(Identity { member: m.id, role: m.role, session, key_id: m.key_id, expires_at })
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
    conn.execute("INSERT INTO notices (member, kind, body, at, seen) VALUES (?1, ?2, ?3, ?4, 0)", params![member, kind, body.to_string(), now_ms()]).map_err(sql)?;
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

/// The oldest unseen notices of `member` (at most `max`), marked seen, and how many unseen remain.
pub fn take_notices(conn: &Connection, member: &str, max: usize) -> Result<(Vec<Notice>, usize), Vec<Diagnostic>> {
    let mut stmt = conn.prepare("SELECT id, kind, body, at FROM notices WHERE member = ?1 AND seen = 0 ORDER BY id LIMIT ?2").map_err(sql)?;
    let rows = stmt.query_map(params![member, max as i64], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?, r.get::<_, i64>(3)?))).map_err(sql)?;
    let mut out = Vec::new();
    for row in rows {
        let (id, kind, body, at) = row.map_err(sql)?;
        let body = match serde_json::from_str::<Value>(&body) {
            Ok(Value::Object(m)) => m,
            _ => serde_json::Map::new(),
        };
        out.push(Notice { id, kind, at, body });
    }
    for n in &out {
        conn.execute("UPDATE notices SET seen = 1 WHERE id = ?1", params![n.id]).map_err(sql)?;
    }
    let more: i64 = conn.query_row("SELECT count(*) FROM notices WHERE member = ?1 AND seen = 0", params![member], |r| r.get(0)).map_err(sql)?;
    Ok((out, usize::try_from(more).unwrap_or(0)))
}

/// True if `member` has unseen notices.
pub fn has_notices(conn: &Connection, member: &str) -> Result<bool, Vec<Diagnostic>> {
    let n: i64 = conn.query_row("SELECT count(*) FROM notices WHERE member = ?1 AND seen = 0", params![member], |r| r.get(0)).map_err(sql)?;
    Ok(n > 0)
}

/// One tool call in team mode (spec §10.4; measurement only, not signed).
#[derive(Debug, Clone)]
pub struct Call<'a> {
    pub member: &'a str,
    pub session: &'a str,
    pub tool: &'a str,
    pub ok: bool,
    pub code: Option<&'a str>,
    pub ms: i64,
}

pub fn log_call(conn: &Connection, c: &Call) -> Result<(), Vec<Diagnostic>> {
    conn.execute(
        "INSERT INTO calls (member, session, tool, ok, code, ms, at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![c.member, c.session, c.tool, c.ok, c.code, c.ms, now_ms()],
    )
    .map(|_| ())
    .map_err(sql)
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
        let names = main_names(&tx)?;
        let main: Vec<Value> = names.iter().map(|(n, h)| json!([n, h])).collect();
        let signer = Signer { member: owner.to_string(), session: CLI_SESSION.into(), key };
        let record = append(&tx, &signer, "genesis", json!({"owner": owner, "owner_key": public, "main": main}), &names, &names)?;
        tx.commit().map_err(sql)?;
        Ok(Initialized { owner: owner.into(), key_id: signer.key.key_id().into(), record, root: root(&names), key_dir: key_dir.display().to_string() })
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
        let public = keys::public_hex(&key.public());
        let tx = self.conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
        if member(&tx, id)?.is_some() {
            return Err(one(state(format!("`{id}` is already a member"))));
        }
        tx.execute("INSERT INTO members (id, role, key, key_id, added_at) VALUES (?1, ?2, ?3, ?4, ?5)", params![id, role.as_str(), public, key.key_id(), now_ms()]).map_err(sql)?;
        let names = main_names(&tx)?;
        let record = append(&tx, owner, "member_add", json!({"id": id, "role": role.as_str(), "key": public}), &names, &names)?;
        tx.commit().map_err(sql)?;
        Ok(MemberInfo { id: id.into(), role: role.as_str().into(), key_id: key.key_id().into(), record: Some(record) })
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
        let public = keys::public_hex(&key.public());
        let tx = self.conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
        tx.execute("UPDATE members SET key = ?1, key_id = ?2 WHERE id = ?3", params![public, key.key_id(), id]).map_err(sql)?;
        let names = main_names(&tx)?;
        let record = append(&tx, owner, "member_key", json!({"id": id, "role": m.role, "key": public}), &names, &names)?;
        tx.commit().map_err(sql)?;
        Ok(MemberInfo { id: id.into(), role: m.role, key_id: key.key_id().into(), record: Some(record) })
    }

    /// `qal team revoke`: ends the membership and all its sessions (record `member_revoke`).
    pub fn team_revoke(&mut self, owner: &Signer, id: &str) -> Result<MemberInfo, Vec<Diagnostic>> {
        self.require_team("`qal team revoke`")?;
        let m = self.agent_member(id)?;
        let tx = self.conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
        let now = now_ms();
        tx.execute("UPDATE members SET revoked_at = ?1 WHERE id = ?2", params![now, id]).map_err(sql)?;
        tx.execute("UPDATE sessions SET revoked_at = ?1 WHERE member = ?2 AND revoked_at IS NULL", params![now, id]).map_err(sql)?;
        let names = main_names(&tx)?;
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
        let token = Zeroizing::new(format!("qs_{}", hex(keys::random32().map_err(one)?.as_slice())));
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
        let mut stmt = self.conn.prepare("SELECT member, tool, ok, code, ms FROM calls ORDER BY id").map_err(sql)?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, bool>(2)?, r.get::<_, Option<String>>(3)?, r.get::<_, i64>(4)?))).map_err(sql)?;
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
        for row in rows {
            let (member, tool, ok, code, ms) = row.map_err(sql)?;
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

    pub fn take_notices(&self, member: &str, max: usize) -> Result<(Vec<Notice>, usize), Vec<Diagnostic>> {
        take_notices(&self.conn, member, max)
    }

    pub fn has_notices(&self, member: &str) -> Result<bool, Vec<Diagnostic>> {
        has_notices(&self.conn, member)
    }

    pub fn log_call(&self, call: &Call) -> Result<(), Vec<Diagnostic>> {
        log_call(&self.conn, call)
    }

    pub fn team_allow(&self) -> Result<Vec<String>, Vec<Diagnostic>> {
        team_allow(&self.conn)
    }

    /// `qal verify`: checks the record chain (spec §9.2) against `trust` (or the genesis record).
    pub fn verify(&mut self, trust: Option<&verify::Trust>, from: i64) -> Result<verify::Outcome, Vec<Diagnostic>> {
        self.require_team("`qal verify`")?;
        let tx = self.conn.transaction().map_err(sql)?;
        verify::verify(&tx, trust, from)
    }
}

/// All members, by id.
fn members_of(conn: &Connection) -> Result<Vec<Member>, Vec<Diagnostic>> {
    let mut stmt = conn.prepare("SELECT id, role, key, key_id, revoked_at FROM members ORDER BY id").map_err(sql)?;
    let rows = stmt.query_map([], |r| Ok(Member { id: r.get(0)?, role: r.get(1)?, key: r.get(2)?, key_id: r.get(3)?, revoked_at: r.get(4)? })).map_err(sql)?;
    rows.collect::<Result<Vec<_>, _>>().map_err(sql)
}
