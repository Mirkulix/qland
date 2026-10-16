//! `qal verify` (Phase 4 spec §9.2): the record chain, every signature, the rights of each record,
//! and the replay of all name changes up to the current tables.
//!
//! Order of the checks, each failure naming the record (`seq`) where it was found:
//! 1. sequence without gaps, `prev` chain, `hash` = SHA-256(body), body canonical (E0922);
//! 2. every signature with the signer's key valid at that point; the owner's key from the trust
//!    file (`--trust team.json`), else from the genesis record (E0921, E0925);
//! 3. the rights of each record (§8, as far as records decide them) (E0924);
//! 4. replaying every entry from the genesis gives exactly the current `names` rows (hash, refs,
//!    local names), the full-state digest of the last record (`names_after`, which also binds each
//!    named definition's kind, signature, effects and local names in `defs`), and, in 4a, no
//!    `change_names`, the `members` table and the team's `meta` as recorded (E0923);
//! 5. every definition the names and records refer to is stored and self-verifying: its canonical
//!    form (and group form) match their hashes, its `deps` rows are exactly its references (E0923).
//! Merge gates and leases are not checked here (Phase 4c; leases are not signed).
//!
//! Signatures are checked in parallel, in batches, alongside the sequential replay; the first
//! failure by sequence number is reported either way.

use super::canonical::{self, unhex};
use super::keys::{key_id, parse_public};
use super::policy::{self, Role, OWNER};
use super::{main_rows, names_digest, valid_member_id, Entry, NameRow, Rows, META_OWNER, META_TEAM_ALLOW, RECORD_PREFIX, ZERO_HASH};
use crate::snapshot::Snapshot;
use ed25519_dalek::{Signature, VerifyingKey};
use qal_core::canon::Hash;
use qal_core::diag::Diagnostic;
use rusqlite::{params, Connection};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// The trust anchor: the owner and its public key, from `team.json` (`qal team export-trust`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trust {
    pub owner: String,
    pub owner_key: String,
}

impl Trust {
    /// Parses a trust file (`{"format": "qal-trust-v1", "owner": ..., "owner_key": ...}`).
    pub fn parse(text: &str) -> Result<Trust, Diagnostic> {
        let bad = |m: &str| Diagnostic::new("E0925", "untrusted_key", "", "", format!("not a QAL trust file: {m}")).hint("export one with `qal team export-trust --out team.json`");
        let v: Value = serde_json::from_str(text).map_err(|e| bad(&e.to_string()))?;
        if v.get("format").and_then(Value::as_str) != Some("qal-trust-v1") {
            return Err(bad("`format` must be \"qal-trust-v1\""));
        }
        let owner = v.get("owner").and_then(Value::as_str).ok_or_else(|| bad("`owner` is missing"))?;
        let key = v.get("owner_key").and_then(Value::as_str).filter(|k| parse_public(k).is_some()).ok_or_else(|| bad("`owner_key` is not a public key"))?;
        Ok(Trust { owner: owner.to_string(), owner_key: key.to_string() })
    }
}

/// Why verification failed.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Failure {
    pub code: &'static str,
    pub kind: &'static str,
    pub seq: i64,
    pub message: String,
}

/// The result of `qal verify`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Outcome {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub records: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub members: Option<usize>,
    /// `file` (checked against a trust file) or `store` (the genesis record's own owner key).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trust: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head: Option<String>,
    /// Signatures were checked only from this record on (`--from`); absent: all of them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signatures_from: Option<i64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<Failure>,
    /// Records checked completely before the failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checked: Option<i64>,
}

impl Outcome {
    fn failed(code: &'static str, kind: &'static str, seq: i64, message: String, checked: i64) -> Outcome {
        Outcome { ok: false, records: None, members: None, trust: None, head: None, signatures_from: None, warnings: Vec::new(), error: Some(Failure { code, kind, seq, message }), checked: Some(checked) }
    }

    /// A store whose team tables were changed so that the chain cannot even be read as a team's.
    pub fn tampered(message: String) -> Outcome {
        Outcome::failed("E0923", "replay_mismatch", 0, message, 0)
    }
}

struct Known {
    role: String,
    key: VerifyingKey,
    key_id: String,
    revoked: bool,
}

/// A failed check: (code, kind, message).
type Bad = (&'static str, &'static str, String);

fn chain(m: String) -> Bad {
    ("E0922", "chain_broken", m)
}
fn signature(m: String) -> Bad {
    ("E0921", "signature_invalid", m)
}
fn unauthorized(m: String) -> Bad {
    ("E0924", "unauthorized_record", m)
}
fn replay(m: String) -> Bad {
    ("E0923", "replay_mismatch", m)
}
fn untrusted(m: String) -> Bad {
    ("E0925", "untrusted_key", m)
}

fn text<'a>(body: &'a Value, key: &str) -> Result<&'a str, Bad> {
    body.get(key).and_then(Value::as_str).ok_or_else(|| chain(format!("the record has no string `{key}`")))
}

fn strings(v: Option<&Value>) -> Option<Vec<String>> {
    v.and_then(|v| serde_json::from_value(v.clone()).ok())
}

/// The replayed state.
struct State {
    members: BTreeMap<String, Known>,
    rows: Rows,
    /// The root of `rows`, kept up to date (records that change no name do not recompute it).
    root: String,
    /// Every hash a record names.
    hashes: BTreeSet<String>,
    /// From the genesis record.
    owner: String,
    team_allow: Option<String>,
    /// The last record's `names_after` and hash.
    names_after: String,
    head: String,
    /// Every record hash (for `--head`), only collected when asked for.
    seen_head: bool,
}

/// A signature to check: the record, its signer and key.
struct Job {
    seq: i64,
    member: String,
    key_id: String,
    key: VerifyingKey,
    body: Vec<u8>,
    sig: [u8; 64],
}

/// Records per batch of parallel signature checks.
const BATCH: usize = 4096;

/// The first record (by `seq`) in `jobs` whose signature does not verify.
fn first_bad_signature(jobs: &[Job]) -> Option<(i64, Bad)> {
    if jobs.is_empty() {
        return None;
    }
    let threads = std::thread::available_parallelism().map_or(1, usize::from).clamp(1, 16);
    let chunk = jobs.len().div_ceil(threads).max(64);
    let bad = |js: &[Job]| js.iter().find(|j| j.key.verify_strict(&j.body, &Signature::from_bytes(&j.sig)).is_err()).map(|j| j.seq);
    let found: Vec<i64> = if threads == 1 || jobs.len() <= chunk {
        bad(jobs).into_iter().collect()
    } else {
        std::thread::scope(|s| {
            let handles: Vec<_> = jobs.chunks(chunk).map(|c| s.spawn(move || bad(c))).collect();
            // A thread that panicked counts its chunk as failed at its first record.
            handles.into_iter().zip(jobs.chunks(chunk)).filter_map(|(h, c)| h.join().unwrap_or(Some(c[0].seq))).collect()
        })
    };
    let seq = found.into_iter().min()?;
    let j = jobs.iter().find(|j| j.seq == seq)?;
    Some((seq, signature(format!("the signature of record {seq} by `{}` (key {}) is not valid", j.member, j.key_id))))
}

pub fn verify(conn: &Connection, trust: Option<&Trust>, from: i64, head: Option<&str>) -> Result<Outcome, Vec<Diagnostic>> {
    let mut st = State {
        members: BTreeMap::new(),
        rows: Rows::new(),
        root: canonical::root(&BTreeMap::new()),
        hashes: BTreeSet::new(),
        owner: String::new(),
        team_allow: None,
        names_after: String::new(),
        head: ZERO_HASH.to_string(),
        seen_head: false,
    };
    let mut jobs: Vec<Job> = Vec::with_capacity(BATCH);
    let mut count = 0i64;
    let mut stmt = conn.prepare("SELECT seq, hash, body, sig FROM records ORDER BY seq").map_err(crate::sql)?;
    let mut rows = stmt.query([]).map_err(crate::sql)?;
    // On a failure at `seq`: an invalid signature of an earlier record (still in the batch) is
    // reported first; `own` adds the failing record's own signature job (failures after it).
    let fail = |jobs: &mut Vec<Job>, own: Option<Job>, (code, kind, m): Bad, seq: i64| -> Outcome {
        jobs.extend(own);
        match first_bad_signature(jobs) {
            Some((s, (c, k, msg))) => Outcome::failed(c, k, s, msg, s - 1),
            None => Outcome::failed(code, kind, seq, m, seq - 1),
        }
    };
    while let Some(r) = rows.next().map_err(crate::sql)? {
        let (seq, hash, body, sig): (i64, String, Vec<u8>, String) = (r.get(0).map_err(crate::sql)?, r.get(1).map_err(crate::sql)?, r.get(2).map_err(crate::sql)?, r.get(3).map_err(crate::sql)?);
        let expected = count + 1;
        let at = if seq == expected { seq } else { expected };
        let (parsed, job) = match before_signature(&mut st, expected, seq, &hash, body, &sig, trust) {
            Ok(x) => x,
            Err(b) => return Ok(fail(&mut jobs, None, b, at)),
        };
        if head == Some(hash.as_str()) {
            st.seen_head = true;
        }
        if let Err(b) = after_signature(&mut st, &parsed) {
            return Ok(fail(&mut jobs, (seq >= from).then_some(job), b, seq));
        }
        if seq >= from {
            jobs.push(job);
        }
        count += 1;
        if jobs.len() >= BATCH {
            if let Some((s, (c, k, m))) = first_bad_signature(&jobs) {
                return Ok(Outcome::failed(c, k, s, m, s - 1));
            }
            jobs.clear();
        }
    }
    if let Some((s, (c, k, m))) = first_bad_signature(&jobs) {
        return Ok(Outcome::failed(c, k, s, m, s - 1));
    }
    if count == 0 {
        return Ok(Outcome::failed("E0922", "chain_broken", 1, "the store has no records (the genesis record is missing)".into(), 0));
    }
    if let Some(h) = head {
        if !st.seen_head {
            return Ok(Outcome::failed("E0922", "chain_broken", count, format!("the pinned head {h} is not in the chain (records were removed or replaced)"), count));
        }
    }
    if let Err((code, kind, m)) = final_state(conn, &st) {
        return Ok(Outcome::failed(code, kind, count, m, count));
    }
    let mut warnings = Vec::new();
    if trust.is_none() {
        warnings.push("no trust file: the owner's key was taken from this store's genesis record, which whoever can rewrite the store can rewrite too; verify with --trust team.json".to_string());
    }
    if head.is_none() {
        warnings.push(format!("no pinned head: cutting off the newest records together with their changes cannot be found; keep the head ({}) and verify with --head", st.head));
    }
    if from > 1 {
        warnings.push(format!("signatures were checked only from record {from} on (--from); the records before it were replayed but not authenticated"));
    }
    Ok(Outcome {
        ok: true,
        records: Some(count),
        members: Some(st.members.len()),
        trust: Some(if trust.is_some() { "file" } else { "store" }),
        head: Some(st.head.clone()),
        signatures_from: (from > 1).then_some(from),
        warnings,
        error: None,
        checked: None,
    })
}

/// A record parsed and authenticated (up to its deferred signature).
struct Parsed {
    seq: i64,
    body: Value,
    role: String,
}

/// Steps 1 and 2 up to the signature itself: the chain, the body, the signer and its key.
fn before_signature(st: &mut State, expected: i64, seq: i64, hash: &str, bytes: Vec<u8>, sig: &str, trust: Option<&Trust>) -> Result<(Parsed, Job), Bad> {
    if seq != expected {
        return Err(chain(format!("record {expected} is missing (the next one is {seq})")));
    }
    if Hash::of(&bytes).to_hex() != hash {
        return Err(chain(format!("record {seq} does not match its hash")));
    }
    let rest = bytes.strip_prefix(RECORD_PREFIX.as_bytes()).ok_or_else(|| chain(format!("record {seq} does not start with `qal-record-v1`")))?;
    let body: Value = serde_json::from_slice(rest).map_err(|e| chain(format!("record {seq} is not JSON: {e}")))?;
    if canonical::to_string(&body).ok().as_deref().map(str::as_bytes) != Some(rest) {
        return Err(chain(format!("record {seq} is not in canonical form")));
    }
    if body.get("v").and_then(Value::as_i64) != Some(1) {
        return Err(chain(format!("record {seq} is not of version 1")));
    }
    if body.get("seq").and_then(Value::as_i64) != Some(seq) {
        return Err(chain(format!("record {seq} says it is record {}", body.get("seq").map_or("?".into(), Value::to_string))));
    }
    if text(&body, "prev")? != st.head.as_str() {
        return Err(chain(format!("record {seq} does not follow record {}", seq - 1)));
    }
    st.head = hash.to_string();
    let (op, member, key) = (text(&body, "op")?, text(&body, "member")?, text(&body, "key")?);
    for k in ["session", "space", "root_before", "root_after", "names_after"] {
        text(&body, k)?;
    }
    body.get("at").and_then(Value::as_i64).ok_or_else(|| chain(format!("record {seq} has no time")))?;
    let args = body.get("args").filter(|a| a.is_object()).ok_or_else(|| chain(format!("record {seq} has no arguments")))?;
    // Only the lowercase form: any other spelling would be an undetected byte change.
    let sig_bytes = unhex::<64>(sig).ok_or_else(|| signature(format!("the signature of record {seq} is not 128 lowercase hex digits")))?;

    let (role, public) = if seq == 1 {
        if op != "genesis" {
            return Err(unauthorized("the first record is not the genesis record".into()));
        }
        let owner = args.get("owner").and_then(Value::as_str).ok_or_else(|| chain("the genesis record names no owner".into()))?;
        let owner_key = args.get("owner_key").and_then(Value::as_str).unwrap_or("");
        let public = parse_public(owner_key).ok_or_else(|| chain("the genesis record has no valid owner key".into()))?;
        if let Some(t) = trust {
            if t.owner_key != owner_key || t.owner != owner {
                return Err(untrusted(format!("the store's owner `{owner}` (key {}) is not the trusted owner `{}` (key {})", key_id(&public), t.owner, parse_public(&t.owner_key).map_or_else(String::new, |k| key_id(&k)))));
            }
        }
        if member != owner || key != key_id(&public) {
            return Err(unauthorized("the genesis record is not signed by its owner".into()));
        }
        st.owner = owner.to_string();
        st.team_allow = args.get("team_allow").and_then(Value::as_str).map(str::to_string);
        st.members.insert(owner.to_string(), Known { role: OWNER.into(), key: public, key_id: key_id(&public), revoked: false });
        (OWNER.to_string(), public)
    } else {
        if op == "genesis" {
            return Err(unauthorized(format!("record {seq} is a second genesis record")));
        }
        let m = st.members.get(member).ok_or_else(|| unauthorized(format!("record {seq} is signed by `{member}`, who is not a member at that point")))?;
        if m.revoked {
            return Err(unauthorized(format!("record {seq} is signed by `{member}`, who was revoked before")));
        }
        if key != m.key_id {
            return Err(signature(format!("record {seq} names key {key}, but `{member}`'s key at that point is {}", m.key_id)));
        }
        (m.role.clone(), m.key)
    };
    let job = Job { seq, member: member.to_string(), key_id: key.to_string(), key: public, body: bytes, sig: sig_bytes };
    Ok((Parsed { seq, body, role }, job))
}

/// Steps 3 and 4 for one record: its rights and its replay.
fn after_signature(st: &mut State, p: &Parsed) -> Result<(), Bad> {
    let (seq, body, role) = (p.seq, &p.body, p.role.as_str());
    let (op, member) = (text(body, "op")?, text(body, "member")?);
    let args = body.get("args").filter(|a| a.is_object()).ok_or_else(|| chain(format!("record {seq} has no arguments")))?;
    let (before, after) = (text(body, "root_before")?, text(body, "root_after")?);
    if !policy::record_allowed(op, role) {
        return Err(unauthorized(format!("record {seq}: `{member}` (role {role}) may not make `{op}` records")));
    }
    if text(body, "space")? != "main" {
        return Err(unauthorized(format!("record {seq} writes to a change set, which this version does not have")));
    }
    if op != "genesis" && st.root != before {
        return Err(replay(format!("record {seq}: the names before it do not have the recorded root (changed without a record?)")));
    }
    let id = || args.get("id").and_then(Value::as_str).filter(|i| valid_member_id(i)).ok_or_else(|| unauthorized(format!("record {seq} names no valid member")));
    match op {
        "genesis" => {
            let main = args.get("main").and_then(Value::as_array).ok_or_else(|| chain("the genesis record has no `main` list".into()))?;
            for item in main {
                let (Some(n), Some(h), Some(refs), Some(locals)) = (item.get(0).and_then(Value::as_str), item.get(1).and_then(Value::as_str), strings(item.get(2)), strings(item.get(3))) else {
                    return Err(chain("the genesis record's `main` list is malformed".into()));
                };
                st.hashes.insert(h.to_string());
                st.rows.insert(n.to_string(), NameRow { hash: h.to_string(), refs, locals });
            }
            st.root = super::root_of(&st.rows);
            if st.root != before {
                return Err(replay("the genesis record's names do not have its root".into()));
            }
        }
        "member_add" => {
            let id = id()?;
            let role = args.get("role").and_then(Value::as_str).filter(|r| Role::parse(r).is_some()).ok_or_else(|| unauthorized(format!("record {seq} adds a member without an agent role")))?;
            let k = args.get("key").and_then(Value::as_str).and_then(parse_public).ok_or_else(|| chain(format!("record {seq} has no valid key")))?;
            if st.members.contains_key(id) {
                return Err(unauthorized(format!("record {seq} adds `{id}`, who is already a member")));
            }
            st.members.insert(id.to_string(), Known { role: role.to_string(), key: k, key_id: key_id(&k), revoked: false });
        }
        "member_key" | "member_revoke" => {
            let id = id()?;
            let k = if op == "member_key" { Some(args.get("key").and_then(Value::as_str).and_then(parse_public).ok_or_else(|| chain(format!("record {seq} has no valid key")))?) } else { None };
            let m = st.members.get_mut(id).filter(|m| m.role != OWNER && !m.revoked).ok_or_else(|| unauthorized(format!("record {seq} changes `{id}`, who is not an active agent member")))?;
            match k {
                Some(k) => {
                    if args.get("role").and_then(Value::as_str) != Some(m.role.as_str()) {
                        return Err(unauthorized(format!("record {seq} rekeys `{id}` with another role than it has")));
                    }
                    m.key_id = key_id(&k);
                    m.key = k;
                }
                None => m.revoked = true,
            }
        }
        "define" | "rename" | "delete" | "import" => {
            if op == "import" && args.get("package").and_then(Value::as_str) != Some(crate::package::STD_PACKAGE_NAME) {
                return Err(unauthorized(format!("record {seq} imports a package other than the standard library")));
            }
            let entries: Vec<Entry> = args.get("entries").cloned().map(serde_json::from_value).transpose().ok().flatten().ok_or_else(|| chain(format!("record {seq} has no valid `entries`")))?;
            for e in &entries {
                let had = st.rows.get(&e.name).map(|r| &r.hash);
                if had != e.old.as_ref() {
                    return Err(replay(format!("record {seq}: `{}` had {} before it, the record says {}", e.name, had.map_or("no definition", |s| s.as_str()), e.old.as_deref().unwrap_or("none"))));
                }
                st.hashes.extend(e.old.iter().chain(e.new.iter()).cloned());
                match (&e.new, &e.refs, &e.locals) {
                    (Some(h), Some(refs), Some(locals)) => {
                        st.rows.insert(e.name.clone(), NameRow { hash: h.clone(), refs: refs.clone(), locals: locals.clone() });
                    }
                    (None, _, _) => {
                        st.rows.remove(&e.name);
                    }
                    _ => return Err(chain(format!("record {seq}: the entry of `{}` has no refs or local names", e.name))),
                }
            }
            if !entries.is_empty() {
                st.root = super::root_of(&st.rows);
            }
        }
        other => return Err(unauthorized(format!("record {seq} has the unknown operation `{other}`"))),
    }
    if st.root != after {
        return Err(replay(format!("record {seq}: replaying it does not give its recorded root")));
    }
    st.names_after = text(body, "names_after")?.to_string();
    Ok(())
}

/// Steps 4 (the tables now) and 5 (the definitions).
fn final_state(conn: &Connection, st: &State) -> Result<(), Bad> {
    let db = |e: Vec<Diagnostic>| replay(e.into_iter().next().map_or_else(|| "store error".to_string(), |d| d.message));
    let sqlerr = |e: rusqlite::Error| db(crate::sql(e));
    let current = main_rows(conn).map_err(db)?;
    if current != st.rows {
        let (n, what) = current
            .keys()
            .chain(st.rows.keys())
            .find_map(|n| {
                let (c, r) = (current.get(n), st.rows.get(n));
                let what = match (c, r) {
                    (Some(c), Some(r)) if c.hash != r.hash => "hash",
                    (Some(c), Some(r)) if c.refs != r.refs => "refs",
                    (Some(c), Some(r)) if c.locals != r.locals => "local_names",
                    (Some(_), Some(_)) => return None,
                    (None, _) => "existence (missing now)",
                    (_, None) => "existence (no record made it)",
                };
                Some((n.clone(), what))
            })
            .unwrap_or_default();
        return Err(replay(format!("the current names differ from the replayed records (first at `{n}`: {what})")));
    }
    let digest = names_digest(conn, &st.rows).map_err(db)?;
    if digest != st.names_after {
        return Err(replay("the stored definitions of the names (kind, signature, effects or local names in `defs`) differ from what the last record signed".into()));
    }
    let change_names: i64 = conn.query_row("SELECT count(*) FROM change_names", [], |r| r.get(0)).map_err(sqlerr)?;
    if change_names > 0 {
        return Err(replay(format!("`change_names` has {change_names} rows that no record made")));
    }
    // The team's own state as recorded.
    let meta = |k: &str| crate::package::meta(conn, k).map_err(db);
    if meta(META_OWNER)?.as_deref() != Some(st.owner.as_str()) {
        return Err(replay("meta.owner is not the owner of the genesis record".into()));
    }
    if st.team_allow.is_some() && meta(META_TEAM_ALLOW)? != st.team_allow {
        return Err(replay("meta.team_allow (the effects agents may grant) differs from the genesis record".into()));
    }
    let mut stmt = conn.prepare("SELECT id, role, key, revoked_at FROM members ORDER BY id").map_err(sqlerr)?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?, r.get::<_, Option<i64>>(3)?))).map_err(sqlerr)?;
    let mut seen = 0;
    for row in rows {
        let (id, role, key, revoked) = row.map_err(sqlerr)?;
        let ok = st.members.get(&id).is_some_and(|m| m.role == role && super::keys::public_hex(&m.key) == key && m.revoked == revoked.is_some());
        if !ok {
            return Err(replay(format!("the member `{id}` in the members table differs from the records")));
        }
        seen += 1;
    }
    if seen != st.members.len() {
        return Err(replay("the members table lacks members the records added".into()));
    }
    // Step 5: every definition decodes from verified forms, and `deps` matches its references.
    let snap = Snapshot::load(conn).map_err(db)?;
    let mut deps = conn.prepare("SELECT dep_hash FROM deps WHERE hash = ?1 ORDER BY dep_hash").map_err(sqlerr)?;
    let all: BTreeSet<&String> = current.values().map(|r| &r.hash).chain(st.hashes.iter()).collect();
    for h in all {
        let hash = Hash::from_hex(h).ok_or_else(|| replay(format!("`{h}` is not a hash")))?;
        let row = snap.row(conn, &hash).map_err(db)?.ok_or_else(|| replay(format!("the definition #{h} is not stored")))?;
        let stored: Vec<String> = deps.query_map(params![h], |r| r.get::<_, String>(0)).map_err(sqlerr)?.collect::<Result<_, _>>().map_err(sqlerr)?;
        let mut expected: Vec<String> = row.refs.iter().map(Hash::to_hex).collect();
        expected.sort();
        if stored != expected {
            return Err(replay(format!("the dependencies stored for #{h} are not the ones its definition refers to")));
        }
    }
    Ok(())
}
