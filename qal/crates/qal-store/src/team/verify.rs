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
//!    named definition's kind, signature, effects and local names in `defs`), the same for every
//!    change set's space (`change_names`), the `changes` (owner, status, scope), the `members`
//!    table and the team's `meta` as recorded (E0923);
//! 5. every definition the names and records refer to is stored and self-verifying: its canonical
//!    form (and group form) match their hashes, its `deps` rows are exactly its references (E0923).
//! Rights in change sets are checked as far as the records decide them: role, owner and status of
//! the change set, an implementer's scope (E0905) and interface (E0906) rules (decided from the
//! replayed metadata and the definitions in `defs`), a verifier's tests-only rule, reviews (verdict
//! by role, for the change set's root at that point), and every merge's gate (§7.4) from the review
//! records it names, with the same function the merge used. Leases are not signed.
//!
//! Signatures are checked in parallel, in batches, alongside the sequential replay; the first
//! failure by sequence number is reported either way.

use super::canonical::{self, unhex};
use super::keys::{key_id, parse_public};
use super::policy::{self, Role, OWNER};
use super::{main_rows, names_digest, space_rows, valid_member_id, Entry, NameRow, Rows, META_OWNER, META_TEAM_ALLOW, RECORD_PREFIX, ZERO_HASH};
use crate::space::Space;
use crate::snapshot::Snapshot;
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

/// A replayed change set.
#[derive(Debug, Clone)]
struct Change {
    owner: String,
    assignee: Option<String>,
    status: String,
    claimed: bool,
    scope: BTreeSet<String>,
    needs_rebase: bool,
}

/// A review record as replayed: what its `reviews` row must say.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ReviewRow {
    seq: i64,
    change: String,
    member: String,
    verdict: String,
    root: String,
    comments_sha256: String,
}

/// An implementer's replacement of an existing name: its interface must stay (checked against the
/// definitions at the end). The name maps give the names each version was written against.
struct Interface {
    seq: i64,
    name: String,
    old: String,
    old_names: BTreeMap<String, String>,
    new: String,
    new_names: BTreeMap<String, String>,
}

/// The replayed state.
struct State {
    members: BTreeMap<String, Known>,
    /// The rows of every space (`main` and each change set).
    spaces: BTreeMap<String, Rows>,
    /// The root of each space, kept up to date (records that change no name do not recompute it).
    roots: BTreeMap<String, String>,
    changes: BTreeMap<String, Change>,
    /// The `names_after` of the last record about each space.
    digests: BTreeMap<String, String>,
    /// The `meta_after` of the last record about each change set (its signed metadata).
    metas: BTreeMap<String, String>,
    /// The replayed metadata of each change set (review N3).
    meta: BTreeMap<String, super::change::Meta>,
    /// The seq of the record that opened each change set.
    opened: BTreeMap<String, i64>,
    /// The reviews of each change set, as the merge gate reads them.
    reviews: BTreeMap<String, Vec<super::merge::ReviewRec>>,
    /// Every review record, for the `reviews` table.
    review_rows: Vec<ReviewRow>,
    /// A merge record whose change set's `merged` record has not come yet: (seq, change).
    merge_open: Option<(i64, String)>,
    /// The merge whose `merged`/`needs_rebase` records may follow now.
    merge_group: Option<i64>,
    /// Who first added a name in a change set (seq, member): an architect's names are interface.
    first_adder: BTreeMap<(String, String), (i64, String)>,
    /// Implementers' replacements of existing names, checked against `defs` at the end.
    interfaces: Vec<Interface>,
    /// (seq, hash) of what verifiers defined: must be tests (checked against `defs` at the end).
    verifier_defs: Vec<(i64, String)>,
    /// Every hash a record names.
    hashes: BTreeSet<String>,
    /// From the genesis record.
    owner: String,
    team_allow: Option<String>,
    /// The last record's hash.
    head: String,
    /// Every record hash (for `--head`), only collected when asked for.
    seen_head: bool,
}

impl State {
    fn rows(&self, space: &str) -> Rows {
        self.spaces.get(space).cloned().unwrap_or_default()
    }

    fn root(&self, space: &str) -> String {
        self.roots.get(space).cloned().unwrap_or_else(|| canonical::root(&BTreeMap::new()))
    }

    fn set(&mut self, space: &str, rows: Rows) {
        self.roots.insert(space.to_string(), super::root_of(&rows));
        self.spaces.insert(space.to_string(), rows);
    }
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
        spaces: BTreeMap::new(),
        roots: BTreeMap::new(),
        changes: BTreeMap::new(),
        digests: BTreeMap::new(),
        metas: BTreeMap::new(),
        meta: BTreeMap::new(),
        opened: BTreeMap::new(),
        reviews: BTreeMap::new(),
        review_rows: Vec::new(),
        merge_open: None,
        merge_group: None,
        first_adder: BTreeMap::new(),
        interfaces: Vec::new(),
        verifier_defs: Vec::new(),
        hashes: BTreeSet::new(),
        owner: String::new(),
        team_allow: None,
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
    // Rights: the matrix for the space as the record found it.
    let space = text(body, "space")?.to_string();
    // A review's `action` for the matrix is its verdict.
    let action = args.get("action").or_else(|| args.get("verdict")).and_then(Value::as_str);
    // A merge is one transaction: its record, then the merged change set's `merged` record, then
    // the `needs_rebase` records of the change sets it affects.
    let merge_step = op == "change" && matches!(action, Some("merged" | "needs_rebase"));
    if let Some((m, c)) = &st.merge_open {
        if !(op == "change" && action == Some("merged") && space == *c) {
            return Err(unauthorized(format!("record {seq} follows the merge record {m} of change set {c}, which its `merged` record must follow")));
        }
    }
    if !merge_step {
        st.merge_group = None;
    }
    let opening = op == "change" && action == Some("open");
    if space != "main" && !opening && !st.changes.contains_key(&space) {
        return Err(unauthorized(format!("record {seq} is about change set {space}, which no record opened")));
    }
    let kind = match st.changes.get(&space) {
        _ if opening => policy::Space::Own,
        None => policy::Space::Main,
        Some(c) if c.owner == member => policy::Space::Own,
        Some(c) if c.status == "submitted" => policy::Space::ForeignSubmitted,
        Some(_) => policy::Space::ForeignOther,
    };
    if !policy::record_allowed(op, action, role, kind) {
        return Err(unauthorized(format!("record {seq}: `{member}` (role {role}) may not make `{op}` records{} {}", action.map(|a| format!(" ({a})")).unwrap_or_default(), if space == "main" { "on main".to_string() } else { format!("in change set {space}") })));
    }
    let live = st.changes.get(&space).is_some_and(|c| c.status == "open" || c.status == "submitted");
    if space != "main" && !opening && op != "lease_break" && !live {
        return Err(unauthorized(format!("record {seq} changes change set {space}, which is no longer open")));
    }
    if op != "genesis" && st.root(&space) != before {
        return Err(replay(format!("record {seq}: the names of {space} before it do not have the recorded root (changed without a record?)")));
    }
    let id = || args.get("id").and_then(Value::as_str).filter(|i| valid_member_id(i)).ok_or_else(|| unauthorized(format!("record {seq} names no valid member")));
    match op {
        "genesis" => {
            let main = args.get("main").and_then(Value::as_array).ok_or_else(|| chain("the genesis record has no `main` list".into()))?;
            let mut rows = Rows::new();
            for item in main {
                let (Some(n), Some(h), Some(refs), Some(locals)) = (item.get(0).and_then(Value::as_str), item.get(1).and_then(Value::as_str), strings(item.get(2)), strings(item.get(3))) else {
                    return Err(chain("the genesis record's `main` list is malformed".into()));
                };
                st.hashes.insert(h.to_string());
                rows.insert(n.to_string(), NameRow { hash: h.to_string(), refs, locals });
            }
            st.set("main", rows);
            if st.root("main") != before {
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
            if op == "import" && space != "main" {
                return Err(unauthorized(format!("record {seq} imports into a change set")));
            }
            let entries: Vec<Entry> = args.get("entries").cloned().map(serde_json::from_value).transpose().ok().flatten().ok_or_else(|| chain(format!("record {seq} has no valid `entries`")))?;
            let mut rows = st.rows(&space);
            let before_rows = rows.clone();
            if role == "implementer" && op == "define" && space != "main" {
                implementer_rules(st, seq, member, &space, &entries, &before_rows)?;
            }
            if op == "define" && space != "main" {
                for e in entries.iter().filter(|e| e.action == "added") {
                    st.first_adder.entry((space.clone(), e.name.clone())).or_insert((seq, member.to_string()));
                }
            }
            if op == "rename" && space != "main" {
                // The history of the old name moves to the new one (ops::rename), so does its adder.
                if let (Some(from), Some(to)) = (entries.iter().find(|e| e.action == "renamed"), entries.iter().find(|e| e.action == "added")) {
                    if let Some(a) = st.first_adder.get(&(space.clone(), from.name.clone())).cloned() {
                        let slot = st.first_adder.entry((space.clone(), to.name.clone())).or_insert(a.clone());
                        if a.0 < slot.0 {
                            *slot = a;
                        }
                    }
                }
            }
            if role == "verifier" {
                st.verifier_defs.extend(entries.iter().filter(|e| matches!(e.action.as_str(), "added" | "replaced")).filter_map(|e| e.new.clone()).map(|h| (seq, h)));
            }
            for e in &entries {
                let had = rows.get(&e.name).map(|r| &r.hash);
                if had != e.old.as_ref() {
                    return Err(replay(format!("record {seq}: `{}` had {} before it, the record says {}", e.name, had.map_or("no definition", |s| s.as_str()), e.old.as_deref().unwrap_or("none"))));
                }
                st.hashes.extend(e.old.iter().chain(e.new.iter()).cloned());
                match (&e.new, &e.refs, &e.locals) {
                    (Some(h), Some(refs), Some(locals)) => {
                        rows.insert(e.name.clone(), NameRow { hash: h.clone(), refs: refs.clone(), locals: locals.clone() });
                    }
                    (None, _, _) => {
                        rows.remove(&e.name);
                    }
                    _ => return Err(chain(format!("record {seq}: the entry of `{}` has no refs or local names", e.name))),
                }
            }
            if !entries.is_empty() {
                st.set(&space, rows);
            }
            // The owner writing a submitted change set reopens it (§6.2).
            if let Some(c) = st.changes.get_mut(&space) {
                if c.owner == member && c.status == "submitted" {
                    c.status = "open".into();
                }
            }
            if let Some(m) = st.meta.get_mut(&space) {
                for e in &entries {
                    let list = |v: &Option<Vec<String>>| Value::from(v.clone().unwrap_or_default()).to_string();
                    match (&e.new, e.action.as_str()) {
                        (Some(h), "relinked" | "updated") => m.put(&e.name, h, list(&e.locals), list(&e.refs), false),
                        (Some(h), _) => m.put(&e.name, h, list(&e.locals), list(&e.refs), true),
                        (None, "renamed") => m.remove(&e.name, "renamed"),
                        (None, _) => m.remove(&e.name, "edited"),
                    }
                }
                // A define that kept a marked name's own version cleared its conflict mark (§7.3).
                if op == "define" {
                    if let Some(v) = args.get("resolved") {
                        for n in strings(Some(v)).ok_or_else(|| chain(format!("record {seq} has an invalid `resolved` list")))? {
                            let row = m.rows.get(&n).filter(|r| r.conflict.is_some()).cloned();
                            let Some((h, r)) = row.and_then(|r| Some((r.hash.clone()?, r))) else {
                                return Err(replay(format!("record {seq} resolves `{n}`, which has no conflict mark in change set {space}")));
                            };
                            m.put(&n, &h, r.locals, r.refs, true);
                        }
                    }
                }
                if let (Some(h), Some(c)) = (m.head.as_mut(), st.changes.get(&space)) {
                    h.status = c.status.clone();
                }
            }
        }
        "change" => change_op(st, seq, member, &space, action.unwrap_or(""), args)?,
        "lease_break" => {
            if args.get("name").and_then(Value::as_str).is_none() || args.get("change").and_then(Value::as_str) != Some(space.as_str()) {
                return Err(chain(format!("record {seq} is not a valid lease break")));
            }
            if let Some(c) = st.changes.get_mut(&space) {
                c.needs_rebase = true;
            }
        }
        "review" => review_op(st, seq, member, role, &space, args)?,
        "merge" => merge_op(st, seq, member, args)?,
        "rebase" => rebase_op(st, seq, &space, args)?,
        other => return Err(unauthorized(format!("record {seq} has the unknown operation `{other}`"))),
    }
    if st.root(&space) != after {
        return Err(replay(format!("record {seq}: replaying it does not give its recorded root")));
    }
    if space != "main" {
        let meta = args.get("meta_after").and_then(Value::as_str).ok_or_else(|| chain(format!("record {seq} about change set {space} does not sign its metadata (`meta_after`)")))?;
        // The metadata replayed from the records so far must be what this record signed.
        if let Some(m) = st.meta.get_mut(&space) {
            if let (Some(h), Some(c)) = (m.head.as_mut(), st.changes.get(&space)) {
                h.owner = c.owner.clone();
                h.status = c.status.clone();
                h.claimed = c.claimed;
                h.scope = Value::from(c.scope.iter().cloned().collect::<Vec<String>>()).to_string();
                h.needs_rebase = c.needs_rebase;
            }
            if m.digest() != meta {
                return Err(replay(format!("record {seq}: the metadata of change set {space} it signed differs from the metadata replayed from the records (changed without a record?)")));
            }
        }
        st.metas.insert(space.clone(), meta.to_string());
        if opening {
            st.opened.insert(space.clone(), seq);
        }
    }
    st.digests.insert(space, text(body, "names_after")?.to_string());
    Ok(())
}

/// Replays a `change` record (open, claim, submit, abandon, scope) on `space` (= its change set).
fn change_op(st: &mut State, seq: i64, member: &str, space: &str, action: &str, args: &Value) -> Result<(), Bad> {
    if args.get("id").and_then(Value::as_str) != Some(space) {
        return Err(chain(format!("record {seq} names another change set than its space")));
    }
    let fields = args.get("fields").filter(|f| f.is_object()).ok_or_else(|| chain(format!("record {seq} has no `fields`")))?;
    let names_of = |key: &str| -> Vec<String> { strings(fields.get(key)).unwrap_or_default() };
    if action == "open" {
        if st.changes.contains_key(space) || !crate::space::is_change_id(space) {
            return Err(unauthorized(format!("record {seq} opens change set {space} again (or with an invalid id)")));
        }
        if fields.get("owner").and_then(Value::as_str) != Some(member) {
            return Err(unauthorized(format!("record {seq} opens a change set for another member")));
        }
        let package = fields.get("package").and_then(Value::as_bool).unwrap_or(false);
        let assignee = fields.get("assignee").and_then(Value::as_str).map(str::to_string);
        if (package || assignee.is_some()) && st.members.get(member).map(|m| m.role.as_str()) != Some("architect") {
            return Err(unauthorized(format!("record {seq}: only architects open work packages")));
        }
        let main = st.rows("main");
        if fields.get("base_root").and_then(Value::as_str) != Some(super::root_of(&main).as_str()) {
            return Err(replay(format!("record {seq}: the base of change set {space} is not main at that point")));
        }
        let scope: BTreeSet<String> = names_of("scope").into_iter().collect();
        let text = |k: &str| fields.get(k).and_then(Value::as_str).unwrap_or_default().to_string();
        let head = super::change::MetaHead {
            id: space.to_string(),
            title: text("title"),
            description: text("description"),
            scope: Value::from(scope.iter().cloned().collect::<Vec<String>>()).to_string(),
            owner: member.to_string(),
            assignee: assignee.clone(),
            status: "open".into(),
            claimed: !package,
            base_seq: seq - 1,
            base_root: text("base_root"),
            needs_rebase: false,
        };
        let rows = main
            .iter()
            .map(|(n, r)| (n.clone(), super::change::MetaRow { hash: Some(r.hash.clone()), locals: Value::from(r.locals.clone()).to_string(), refs: Value::from(r.refs.clone()).to_string(), origin: "base".into(), base_hash: Some(r.hash.clone()), conflict: None }))
            .collect();
        st.meta.insert(space.to_string(), super::change::Meta { head: Some(head), rows });
        st.set(space, main);
        st.changes.insert(space.to_string(), Change { owner: member.to_string(), assignee, status: "open".into(), claimed: !package, scope, needs_rebase: false });
        return Ok(());
    }
    if action == "merged" {
        let ok = st.merge_open.as_ref().is_some_and(|(m, x)| x == space && fields.get("merge").and_then(Value::as_i64) == Some(*m));
        let c = st.changes.get_mut(space).ok_or_else(|| unauthorized(format!("record {seq}: no change set {space}")))?;
        if !ok || c.status != "submitted" {
            return Err(unauthorized(format!("record {seq} ends change set {space} as merged without its merge record before it")));
        }
        c.status = "merged".into();
        st.merge_open = None;
        return Ok(());
    }
    if action == "needs_rebase" {
        if st.merge_group.is_none() || fields.get("merge").and_then(Value::as_i64) != st.merge_group || st.merge_open.is_some() {
            return Err(unauthorized(format!("record {seq} flags change set {space} outside of a merge")));
        }
        let c = st.changes.get_mut(space).ok_or_else(|| unauthorized(format!("record {seq}: no change set {space}")))?;
        c.needs_rebase = true;
        return Ok(());
    }
    let c = st.changes.get_mut(space).ok_or_else(|| unauthorized(format!("record {seq}: no change set {space}")))?;
    match action {
        "claim" => {
            if c.claimed || c.status != "open" || c.assignee.as_deref().is_some_and(|a| a != member) {
                return Err(unauthorized(format!("record {seq} claims change set {space}, which is not a work package `{member}` may claim")));
            }
            c.owner = member.to_string();
            c.claimed = true;
        }
        "submit" if c.status == "open" => c.status = "submitted".into(),
        "abandon" => c.status = "abandoned".into(),
        "scope" => {
            c.scope.extend(names_of("add"));
            for r in names_of("remove") {
                c.scope.remove(&r);
            }
        }
        other => return Err(unauthorized(format!("record {seq}: `{other}` does not fit change set {space} ({})", c.status))),
    }
    Ok(())
}

/// The first name whose rows differ, and what differs.
fn first_difference(current: &Rows, replayed: &Rows) -> (String, &'static str) {
    current
        .keys()
        .chain(replayed.keys())
        .find_map(|n| {
            let (c, r) = (current.get(n), replayed.get(n));
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
        .unwrap_or_default()
}

/// Steps 4 (the tables now) and 5 (the definitions).
fn final_state(conn: &Connection, st: &State) -> Result<(), Bad> {
    let db = |e: Vec<Diagnostic>| replay(e.into_iter().next().map_or_else(|| "store error".to_string(), |d| d.message));
    let sqlerr = |e: rusqlite::Error| db(crate::sql(e));
    let current = main_rows(conn).map_err(db)?;
    let main = st.rows("main");
    if current != main {
        let (n, what) = first_difference(&current, &main);
        return Err(replay(format!("the current names differ from the replayed records (first at `{n}`: {what})")));
    }
    // The stored lists in their one spelling (compact JSON): no byte of a row goes unchecked.
    {
        let mut stmt = conn.prepare("SELECT name, refs, local_names FROM names").map_err(sqlerr)?;
        let raw = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))).map_err(sqlerr)?;
        for row in raw {
            let (name, refs, locals) = row.map_err(sqlerr)?;
            let Some(r) = current.get(&name) else { return Err(replay(format!("`{name}` changed while it was verified"))) };
            if refs != Value::from(r.refs.clone()).to_string() || locals != Value::from(r.locals.clone()).to_string() {
                return Err(replay(format!("the stored lists of `{name}` are not in their canonical spelling")));
            }
        }
    }
    if st.digests.get("main").is_some_and(|d| *d != names_digest(conn, &main).unwrap_or_default()) {
        return Err(replay("the stored definitions of the names (kind, signature, effects or local names in `defs`) differ from what the last record signed".into()));
    }
    // Every change set: its space, its digest, its row in `changes`.
    let tables: Vec<String> = {
        let mut stmt = conn.prepare("SELECT DISTINCT change FROM change_names").map_err(sqlerr)?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0)).map_err(sqlerr)?;
        rows.collect::<Result<_, _>>().map_err(sqlerr)?
    };
    if let Some(c) = tables.iter().find(|c| !st.changes.contains_key(*c)) {
        return Err(replay(format!("`change_names` has rows of change set {c}, which no record opened")));
    }
    for (c, ch) in &st.changes {
        let stored = space_rows(conn, &Space::Change(c.clone())).map_err(db)?;
        let replayed = st.rows(c);
        if stored != replayed {
            let (n, what) = first_difference(&stored, &replayed);
            return Err(replay(format!("the names of change set {c} differ from the replayed records (first at `{n}`: {what})")));
        }
        // Everything else the change set's decisions rest on (origins, base hashes, conflict marks,
        // rows without a hash, its row in `changes`), as its last record signed it.
        let meta = super::change::meta_digest(conn, c).map_err(db)?;
        if st.metas.get(c) != Some(&meta) || st.meta.get(c).map(super::change::Meta::digest) != Some(meta) {
            return Err(replay(format!("the metadata of change set {c} (change_names origin/base_hash/conflict, rows without a hash, or its changes row) differs from the records")));
        }
        let base_seq: i64 = conn.query_row("SELECT base_seq FROM changes WHERE id = ?1", params![c], |r| r.get(0)).map_err(sqlerr)?;
        let replayed_base = st.meta.get(c).and_then(|m| m.head.as_ref()).map(|h| h.base_seq);
        if !st.opened.contains_key(c) || replayed_base != Some(base_seq) {
            return Err(replay(format!("change set {c}'s base_seq is not the record before the one that opened (or last rebased) it")));
        }
        if st.digests.get(c).is_some_and(|d| *d != names_digest(conn, &replayed).unwrap_or_default()) {
            return Err(replay(format!("the stored definitions of change set {c}'s names differ from what its last record signed")));
        }
        let row: Option<(String, String, String)> = conn.query_row("SELECT owner, status, scope FROM changes WHERE id = ?1", params![c], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))).optional().map_err(sqlerr)?;
        let scope: Option<BTreeSet<String>> = row.as_ref().and_then(|(_, _, s)| serde_json::from_str::<Vec<String>>(s).ok()).map(|v| v.into_iter().collect());
        if row.as_ref().map(|(o, s, _)| (o.as_str(), s.as_str())) != Some((ch.owner.as_str(), ch.status.as_str())) || scope.as_ref() != Some(&ch.scope) {
            return Err(replay(format!("change set {c} in the changes table differs from the records (owner, status or scope)")));
        }
    }
    let changes: i64 = conn.query_row("SELECT count(*) FROM changes", [], |r| r.get(0)).map_err(sqlerr)?;
    if usize::try_from(changes).unwrap_or(usize::MAX) != st.changes.len() {
        return Err(replay("the changes table has change sets no record opened".into()));
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
    if let Some((m, c)) = &st.merge_open {
        return Err(replay(format!("the merge record {m} of change set {c} is the last record: its `merged` record is missing")));
    }
    // The reviews table says what the review records say (the merge gate reads both).
    {
        let mut stmt = conn.prepare("SELECT record, change, member, verdict, root, comments FROM reviews ORDER BY record").map_err(sqlerr)?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?, r.get::<_, String>(3)?, r.get::<_, String>(4)?, r.get::<_, String>(5)?))).map_err(sqlerr)?;
        let mut stored = Vec::new();
        for row in rows {
            let (seq, change, member, verdict, root, comments) = row.map_err(sqlerr)?;
            stored.push(ReviewRow { seq, change, member, verdict, root, comments_sha256: Hash::of(comments.as_bytes()).to_hex() });
        }
        if stored != st.review_rows {
            let at = stored.iter().zip(&st.review_rows).find(|(a, b)| a != b).map_or_else(|| "a row is missing or extra".to_string(), |(a, _)| format!("the review of record {}", a.seq));
            return Err(replay(format!("the reviews table differs from the review records ({at})")));
        }
    }
    // An implementer kept the interface of every existing name it replaced (§8, E0906).
    if !st.interfaces.is_empty() {
        let snap = Snapshot::load(conn, &Space::Main).map_err(db)?;
        for i in &st.interfaces {
            let old = decode_named(&snap, conn, &i.old, &i.name, &i.old_names)?;
            let new = decode_named(&snap, conn, &i.new, &i.name, &i.new_names)?;
            if let Some(what) = super::change::interface_change(&old, &new) {
                return Err(unauthorized(format!("record {}: an implementer changed the {what} of `{}`, which is interface", i.seq, i.name)));
            }
        }
    }
    // Verifiers write only tests (§8), decided by the kind of what their records defined.
    for (seq, h) in &st.verifier_defs {
        let kind: Option<String> = conn.query_row("SELECT kind FROM defs WHERE hash = ?1", params![h], |r| r.get(0)).optional().map_err(sqlerr)?;
        if kind.as_deref() != Some("test") {
            return Err(unauthorized(format!("record {seq}: a verifier defined #{h}, which is not a test")));
        }
    }
    // Step 5: every definition decodes from verified forms, and `deps` matches its references.
    let snap = Snapshot::load(conn, &Space::Main).map_err(db)?;
    let mut deps = conn.prepare("SELECT dep_hash FROM deps WHERE hash = ?1 ORDER BY dep_hash").map_err(sqlerr)?;
    let all: BTreeSet<&String> = st.spaces.values().flat_map(|r| r.values().map(|r| &r.hash)).chain(st.hashes.iter()).collect();
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

/// Applies the `entries` of a record to the rows of `space` (each `old` must be what the name had).
fn apply_entries(st: &mut State, seq: i64, space: &str, entries: &[Entry]) -> Result<(), Bad> {
    let mut rows = st.rows(space);
    for e in entries {
        let had = rows.get(&e.name).map(|r| &r.hash);
        if had != e.old.as_ref() {
            return Err(replay(format!("record {seq}: `{}` had {} before it, the record says {}", e.name, had.map_or("no definition", |s| s.as_str()), e.old.as_deref().unwrap_or("none"))));
        }
        st.hashes.extend(e.old.iter().chain(e.new.iter()).cloned());
        match (&e.new, &e.refs, &e.locals) {
            (Some(h), Some(refs), Some(locals)) => {
                rows.insert(e.name.clone(), NameRow { hash: h.clone(), refs: refs.clone(), locals: locals.clone() });
            }
            (None, _, _) => {
                rows.remove(&e.name);
            }
            _ => return Err(chain(format!("record {seq}: the entry of `{}` has no refs or local names", e.name))),
        }
    }
    if !entries.is_empty() {
        st.set(space, rows);
    }
    Ok(())
}

fn entries_of(args: &Value, seq: i64) -> Result<Vec<Entry>, Bad> {
    args.get("entries").cloned().map(serde_json::from_value).transpose().ok().flatten().ok_or_else(|| chain(format!("record {seq} has no valid `entries`")))
}

/// An implementer's define in a change set (§8): an existing name (in the base, or added there by
/// an architect) only inside the scope (E0905), and with its interface kept (E0906, checked at the end).
fn implementer_rules(st: &mut State, seq: i64, member: &str, space: &str, entries: &[Entry], before: &Rows) -> Result<(), Bad> {
    let Some(c) = st.changes.get(space) else { return Ok(()) };
    let meta = st.meta.get(space);
    let mut after = before.clone();
    for e in entries {
        match (&e.new, &e.refs, &e.locals) {
            (Some(h), Some(refs), Some(locals)) => {
                after.insert(e.name.clone(), NameRow { hash: h.clone(), refs: refs.clone(), locals: locals.clone() });
            }
            _ => {
                after.remove(&e.name);
            }
        }
    }
    let names_of = |rows: &Rows, refs: &[String]| -> BTreeMap<String, String> { refs.iter().filter_map(|r| rows.get(r).map(|x| (x.hash.clone(), r.clone()))).collect() };
    let mut found = Vec::new();
    for e in entries.iter().filter(|e| matches!(e.action.as_str(), "added" | "replaced")) {
        let Some(old) = before.get(&e.name) else { continue };
        let in_base = meta.and_then(|m| m.rows.get(&e.name)).is_some_and(|r| r.base_hash.is_some());
        let by_architect = st.first_adder.get(&(space.to_string(), e.name.clone())).is_some_and(|(_, who)| who != member && st.members.get(who).is_some_and(|m| m.role == "architect"));
        if !in_base && !by_architect {
            continue;
        }
        if !c.scope.contains(&e.name) {
            return Err(unauthorized(format!("record {seq}: the implementer `{member}` changed `{}`, which exists and is not in the scope of change set {space}", e.name)));
        }
        if let (Some(new), Some(refs)) = (&e.new, &e.refs) {
            found.push(Interface { seq, name: e.name.clone(), old: old.hash.clone(), old_names: names_of(before, &old.refs), new: new.clone(), new_names: names_of(&after, refs) });
        }
    }
    st.interfaces.extend(found);
    Ok(())
}

/// Decodes the definition `hash` as `name`, naming each reference by `names` (hash -> name).
fn decode_named(snap: &Snapshot, conn: &Connection, hash: &str, name: &str, names: &BTreeMap<String, String>) -> Result<qal_core::ast::Def, Bad> {
    let h = Hash::from_hex(hash).ok_or_else(|| replay(format!("`{hash}` is not a hash")))?;
    let row = snap.row(conn, &h).map_err(|d| replay(d.into_iter().next().map_or_else(String::new, |d| d.message)))?.ok_or_else(|| replay(format!("the definition #{hash} is not stored")))?;
    let locals = row.local_names.clone();
    row.decode(name, &locals, &mut |r| names.get(&r.to_hex()).cloned().unwrap_or_else(|| format!("#{}", r.short()))).map_err(|e| replay(format!("#{hash}: {e}")))
}

/// A `review` record (§9.1): a verdict the matrix allows the role, by a member for the root the
/// change set had then; every verdict but `comment` needs it submitted; `changes` reopens it.
fn review_op(st: &mut State, seq: i64, member: &str, role: &str, space: &str, args: &Value) -> Result<(), Bad> {
    let verdict = args.get("verdict").and_then(Value::as_str).filter(|v| super::merge::VERDICTS.contains(v)).ok_or_else(|| chain(format!("record {seq} has no valid verdict")))?;
    let root = args.get("root").and_then(Value::as_str).ok_or_else(|| chain(format!("record {seq} names no root")))?;
    let sha = args.get("comments_sha256").and_then(Value::as_str).filter(|h| canonical::is_hex64(h)).ok_or_else(|| chain(format!("record {seq} has no comments digest")))?;
    if args.get("change").and_then(Value::as_str) != Some(space) {
        return Err(chain(format!("record {seq} reviews another change set than its space")));
    }
    if root != st.root(space) {
        return Err(replay(format!("record {seq} reviews change set {space} at a root it did not have then")));
    }
    let c = st.changes.get_mut(space).ok_or_else(|| unauthorized(format!("record {seq}: no change set {space}")))?;
    if verdict != "comment" && c.status != "submitted" {
        return Err(unauthorized(format!("record {seq} gives the verdict `{verdict}` to change set {space}, which is {}", c.status)));
    }
    if verdict == "changes" {
        c.status = "open".into();
    }
    st.reviews.entry(space.to_string()).or_default().push(super::merge::ReviewRec { seq, member: member.into(), role: role.into(), verdict: verdict.into(), root: root.into() });
    st.review_rows.push(ReviewRow { seq, change: space.into(), member: member.into(), verdict: verdict.into(), root: root.into(), comments_sha256: sha.into() });
    Ok(())
}

/// A `merge` record (on main): the change set was submitted, had no conflict marks, its root is
/// the one recorded, the gate held with exactly the reviews named (the same rule as at merge time,
/// with the roles and revocations then), and a fast-forward is one exactly when main had not
/// moved; its entries replay on main (a fast-forward's give exactly the change set's names).
fn merge_op(st: &mut State, seq: i64, member: &str, args: &Value) -> Result<(), Bad> {
    let c = args.get("change").and_then(Value::as_str).ok_or_else(|| chain(format!("record {seq} names no change set")))?.to_string();
    let ch = st.changes.get(&c).cloned().ok_or_else(|| unauthorized(format!("record {seq} merges change set {c}, which no record opened")))?;
    if ch.status != "submitted" {
        return Err(unauthorized(format!("record {seq} merges change set {c}, which is {}", ch.status)));
    }
    let root = args.get("change_root").and_then(Value::as_str).ok_or_else(|| chain(format!("record {seq} has no change_root")))?;
    if root != st.root(&c) {
        return Err(replay(format!("record {seq} merges change set {c} at a root it did not have")));
    }
    let meta = st.meta.get(&c).cloned().unwrap_or_default();
    if meta.rows.values().any(|r| r.conflict.is_some()) {
        return Err(unauthorized(format!("record {seq} merges change set {c} with conflict marks")));
    }
    let reviews: Vec<super::merge::ReviewRec> = st.reviews.get(&c).cloned().unwrap_or_default().into_iter().filter(|r| st.members.get(&r.member).is_some_and(|m| !m.revoked)).collect();
    let (a, v) = super::merge::gate(&reviews, &ch.owner, root).map_err(|m| unauthorized(format!("record {seq}: `{member}` merged change set {c} without its gate: it needed {m}")))?;
    let named: Vec<i64> = args.get("reviews").and_then(Value::as_array).map(|xs| xs.iter().filter_map(Value::as_i64).collect()).unwrap_or_default();
    if named != [a, v] {
        return Err(unauthorized(format!("record {seq} names the reviews {named:?}, but the gate of change set {c} rests on records {a} and {v}")));
    }
    let base_root = meta.head.as_ref().map(|h| h.base_root.clone()).unwrap_or_default();
    let fast_forward = st.root("main") == base_root;
    let kind = args.get("kind").and_then(Value::as_str);
    if kind != Some(if fast_forward { "fast_forward" } else { "replayed" }) {
        return Err(replay(format!("record {seq} says it is a {} merge, but main {} since the base of change set {c}", kind.unwrap_or("?"), if fast_forward { "had not moved" } else { "had moved" })));
    }
    let entries = entries_of(args, seq)?;
    apply_entries(st, seq, "main", &entries)?;
    if fast_forward && st.rows("main") != st.rows(&c) {
        return Err(replay(format!("record {seq}: the fast-forward of change set {c} does not give exactly its names")));
    }
    st.merge_open = Some((seq, c));
    st.merge_group = Some(seq);
    Ok(())
}

/// A `rebase` record (§7.3, in the change set's space): its entries replay on the change set, its
/// base becomes main as it is now, and its metadata follows [`super::merge::rebase_meta`].
fn rebase_op(st: &mut State, seq: i64, space: &str, args: &Value) -> Result<(), Bad> {
    let base_root = args.get("base_root").and_then(Value::as_str).ok_or_else(|| chain(format!("record {seq} has no base_root")))?.to_string();
    if base_root != st.root("main") {
        return Err(replay(format!("record {seq} rebases change set {space} onto a root main did not have")));
    }
    let mut marks = BTreeMap::new();
    for c in args.get("conflicts").and_then(Value::as_array).ok_or_else(|| chain(format!("record {seq} has no conflict list")))? {
        let name = c.get("name").and_then(Value::as_str).ok_or_else(|| chain(format!("record {seq} has a conflict without a name")))?;
        marks.insert(name.to_string(), super::merge::conflict_mark(c.get("base").and_then(Value::as_str), c.get("theirs").and_then(Value::as_str)));
    }
    let root_before = st.root(space);
    let entries = entries_of(args, seq)?;
    apply_entries(st, seq, space, &entries)?;
    let changed_root = st.root(space) != root_before;
    let (main, after) = (st.rows("main"), st.rows(space));
    if let Some(m) = st.meta.get_mut(space) {
        m.rows = super::merge::rebase_meta(m, &main, &after, &marks);
        if let Some(h) = m.head.as_mut() {
            h.base_seq = seq - 1;
            h.base_root = base_root;
        }
    }
    if let Some(c) = st.changes.get_mut(space) {
        c.needs_rebase = false;
        if c.status == "submitted" && changed_root {
            c.status = "open".into();
        }
    }
    Ok(())
}
