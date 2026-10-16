//! `qal verify` (Phase 4 spec §9.2): the record chain, every signature, the rights of each record,
//! and the replay of all name changes up to the current tables.
//!
//! Order of the checks, each failure naming the record (`seq`) where it was found:
//! 1. sequence without gaps, `prev` chain, `hash` = SHA-256(body), body canonical (E0922);
//! 2. every signature with the signer's key valid at that point; the owner's key from the trust
//!    file (`--trust team.json`), else from the genesis record (E0921, E0925);
//! 3. the rights of each record (§8, as far as records decide them) (E0924);
//! 4. replaying every entry from the genesis gives exactly the current `names` (and, in 4a, no
//!    `change_names` and the `members` table as recorded) (E0923);
//! 5. every hash the names and records refer to is stored in `defs` and matches its canonical
//!    bytes (E0923).
//! Merge gates and leases are not checked here (Phase 4c; leases are not signed).

use super::canonical::{self, root, unhex};
use super::keys::{key_id, parse_public};
use super::policy::{self, Role, OWNER};
use super::{main_names, valid_member_id, Entry, RECORD_PREFIX, ZERO_HASH};
use crate::sql;
use ed25519_dalek::{Signature, VerifyingKey};
use qal_core::canon::Hash;
use qal_core::diag::Diagnostic;
use rusqlite::{params, Connection, OptionalExtension};
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
        Outcome { ok: false, records: None, members: None, trust: None, head: None, warnings: Vec::new(), error: Some(Failure { code, kind, seq, message }), checked: Some(checked) }
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

/// The replayed state.
struct State {
    members: BTreeMap<String, Known>,
    names: BTreeMap<String, String>,
    /// `root(names)`, kept up to date (records that change no name do not recompute it).
    root: String,
    /// Every hash a record names.
    hashes: BTreeSet<String>,
}

pub fn verify(conn: &Connection, trust: Option<&Trust>, from: i64) -> Result<Outcome, Vec<Diagnostic>> {
    let mut rows = Vec::new();
    {
        let mut stmt = conn.prepare("SELECT seq, hash, body, sig FROM records ORDER BY seq").map_err(sql)?;
        let it = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, Vec<u8>>(2)?, r.get::<_, String>(3)?))).map_err(sql)?;
        for r in it {
            rows.push(r.map_err(sql)?);
        }
    }
    let mut st = State { members: BTreeMap::new(), names: BTreeMap::new(), root: root(&BTreeMap::new()), hashes: BTreeSet::new() };
    let mut prev = ZERO_HASH.to_string();
    let mut checked = 0i64;
    for (i, (seq, hash, body, sig)) in rows.iter().enumerate() {
        let expected = i as i64 + 1;
        let at = if *seq == expected { *seq } else { expected };
        if let Err((code, kind, m)) = check_record(&mut st, &mut prev, expected, *seq, hash, body, sig, trust, from) {
            return Ok(Outcome::failed(code, kind, at, m, checked));
        }
        checked += 1;
    }
    let last = rows.last().map_or(0, |r| r.0);
    if rows.is_empty() {
        return Ok(Outcome::failed("E0922", "chain_broken", 1, "the store has no records (the genesis record is missing)".into(), 0));
    }
    if let Err((code, kind, m)) = final_state(conn, &st) {
        return Ok(Outcome::failed(code, kind, last, m, checked));
    }
    let mut warnings = Vec::new();
    if trust.is_none() {
        warnings.push("no trust file: the owner's key was taken from this store's genesis record, which whoever can rewrite the store can rewrite too; verify with --trust team.json".to_string());
    }
    Ok(Outcome {
        ok: true,
        records: Some(rows.len() as i64),
        members: Some(st.members.len()),
        trust: Some(if trust.is_some() { "file" } else { "store" }),
        head: Some(prev),
        warnings,
        error: None,
        checked: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn check_record(st: &mut State, prev: &mut String, expected: i64, seq: i64, hash: &str, bytes: &[u8], sig: &str, trust: Option<&Trust>, from: i64) -> Result<(), Bad> {
    // 1. The chain.
    if seq != expected {
        return Err(chain(format!("record {expected} is missing (the next one is {seq})")));
    }
    if Hash::of(bytes).to_hex() != hash {
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
    if text(&body, "prev")? != prev.as_str() {
        return Err(chain(format!("record {seq} does not follow record {}", seq - 1)));
    }
    *prev = hash.to_string();
    let (op, member, key) = (text(&body, "op")?, text(&body, "member")?, text(&body, "key")?);
    text(&body, "session")?;
    body.get("at").and_then(Value::as_i64).ok_or_else(|| chain(format!("record {seq} has no time")))?;
    let args = body.get("args").filter(|a| a.is_object()).ok_or_else(|| chain(format!("record {seq} has no arguments")))?;
    let (before, after) = (text(&body, "root_before")?, text(&body, "root_after")?);

    // 2. The signature, with the key the signer had at this point.
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
    let sig_bytes = unhex::<64>(sig).ok_or_else(|| signature(format!("the signature of record {seq} is not 128 hex digits")))?;
    if seq >= from && public.verify_strict(bytes, &Signature::from_bytes(&sig_bytes)).is_err() {
        return Err(signature(format!("the signature of record {seq} by `{member}` (key {key}) is not valid")));
    }

    // 3. Rights.
    if !policy::record_allowed(op, &role) {
        return Err(unauthorized(format!("record {seq}: `{member}` (role {role}) may not make `{op}` records")));
    }
    if text(&body, "space")? != "main" {
        return Err(unauthorized(format!("record {seq} writes to a change set, which this version does not have")));
    }

    // 4. Replay.
    if op != "genesis" && st.root != before {
        return Err(replay(format!("record {seq}: the names before it do not have the recorded root (changed without a record?)")));
    }
    let id = || args.get("id").and_then(Value::as_str).filter(|i| valid_member_id(i)).ok_or_else(|| unauthorized(format!("record {seq} names no valid member")));
    match op {
        "genesis" => {
            let main = args.get("main").and_then(Value::as_array).ok_or_else(|| chain("the genesis record has no `main` list".into()))?;
            for pair in main {
                let (Some(n), Some(h)) = (pair.get(0).and_then(Value::as_str), pair.get(1).and_then(Value::as_str)) else {
                    return Err(chain("the genesis record's `main` list is malformed".into()));
                };
                st.hashes.insert(h.to_string());
                st.names.insert(n.to_string(), h.to_string());
            }
            st.root = root(&st.names);
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
                    m.key_id = key_id(&k);
                    m.key = k;
                }
                None => m.revoked = true,
            }
        }
        "define" | "rename" | "delete" | "import" => {
            let entries: Vec<Entry> = args.get("entries").cloned().map(serde_json::from_value).transpose().ok().flatten().ok_or_else(|| chain(format!("record {seq} has no valid `entries`")))?;
            for e in &entries {
                if st.names.get(&e.name) != e.old.as_ref() {
                    return Err(replay(format!("record {seq}: `{}` had {} before it, the record says {}", e.name, st.names.get(&e.name).map_or("no definition", |s| s.as_str()), e.old.as_deref().unwrap_or("none"))));
                }
                st.hashes.extend(e.old.iter().chain(e.new.iter()).cloned());
                match &e.new {
                    Some(h) => st.names.insert(e.name.clone(), h.clone()),
                    None => st.names.remove(&e.name),
                };
            }
            st.root = root(&st.names);
        }
        other => return Err(unauthorized(format!("record {seq} has the unknown operation `{other}`"))),
    }
    if st.root != after {
        return Err(replay(format!("record {seq}: replaying it does not give its recorded root")));
    }
    Ok(())
}

/// Steps 4 (the tables now) and 5 (the definitions).
fn final_state(conn: &Connection, st: &State) -> Result<(), Bad> {
    let db = |e: Vec<Diagnostic>| replay(e.into_iter().next().map_or_else(|| "store error".to_string(), |d| d.message));
    let current = main_names(conn).map_err(db)?;
    if current != st.names {
        let differs = current.keys().chain(st.names.keys()).find(|n| current.get(*n) != st.names.get(*n)).cloned().unwrap_or_default();
        return Err(replay(format!("the current names differ from the replayed records (first at `{differs}`)")));
    }
    let change_names: i64 = conn.query_row("SELECT count(*) FROM change_names", [], |r| r.get(0)).map_err(|e| db(sql(e)))?;
    if change_names > 0 {
        return Err(replay(format!("`change_names` has {change_names} rows that no record made")));
    }
    let mut stmt = conn.prepare("SELECT id, role, key, revoked_at FROM members ORDER BY id").map_err(|e| db(sql(e)))?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?, r.get::<_, Option<i64>>(3)?))).map_err(|e| db(sql(e)))?;
    let mut seen = 0;
    for row in rows {
        let (id, role, key, revoked) = row.map_err(|e| db(sql(e)))?;
        let ok = st.members.get(&id).is_some_and(|m| m.role == role && super::keys::public_hex(&m.key) == key && m.revoked == revoked.is_some());
        if !ok {
            return Err(replay(format!("the member `{id}` in the members table differs from the records")));
        }
        seen += 1;
    }
    if seen != st.members.len() {
        return Err(replay("the members table lacks members the records added".into()));
    }
    for h in current.values().chain(st.hashes.iter()) {
        let canonical: Option<Vec<u8>> = conn.query_row("SELECT canonical FROM defs WHERE hash = ?1", params![h], |r| r.get(0)).optional().map_err(|e| db(sql(e)))?;
        match canonical {
            None => return Err(replay(format!("the definition #{h} is not stored"))),
            Some(c) if Hash::of(&c).to_hex() != *h => return Err(replay(format!("the stored definition #{h} does not match its hash"))),
            Some(_) => {}
        }
    }
    Ok(())
}
