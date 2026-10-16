//! Review, merge and rebase (Phase 4 spec §7, §8, §9.1): reviews bound to a change set's root, the
//! merge gate (§7.4), the classification of every changed name (§7.2), the fast-forward and the
//! replay of a change set onto `main` as one full `define` (check, fresh proofs, tests), and the
//! rebase of a change set onto `main` with conflict marks (§7.3).
//!
//! Every operation is one write transaction with its signed records. A merge writes the `merge`
//! record on `main`, then a `change`/`merged` record in the merged change set's space and a
//! `change`/`needs_rebase` record in the space of every other change set it flags (review N6:
//! whatever changes a change set's metadata signs it in that change set's space). The prover and
//! the tests run on a snapshot outside the write lock; the write then checks that neither `main`
//! nor the change set moved (roots compared), else it starts over (at most [`ATTEMPTS`] times, then
//! under the lock).

use super::canonical;
use super::change::{self, interface_change, load_meta, record_change, require, role_of, ChangeRow, Meta, MetaRow};
use super::policy::{self, Role};
use super::{append, append_in, diff, notify, one, root_of, rows_of, space_rows, state, team_limit, Entry, Rows, Signer};
use crate::define::{self, Replayed, ATTEMPTS};
use crate::snapshot::Snapshot;
use crate::space::{self, Space};
use crate::{now_ms, sql, Defined, Store, Verify};
use qal_core::ast::Program;
use qal_core::canon::Hash;
use qal_core::diag::Diagnostic;
use qal_core::tast::TProgram;
use rusqlite::{params, OptionalExtension, Transaction, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashSet};

/// Limits of spec §14.
pub const MAX_COMMENTS: usize = 32;
pub const MAX_COMMENT_BYTES: usize = 2 << 10;

/// The verdicts of `qal_review` (§8: who may give which is the matrix).
pub const VERDICTS: [&str; 5] = ["approve", "changes", "verified", "failed", "comment"];

/// One comment of a review.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Comment {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub text: String,
}

/// A review as the merge gate reads it: from its signed record, with the reviewer's role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewRec {
    pub seq: i64,
    pub member: String,
    pub role: String,
    pub verdict: String,
    pub root: String,
}

/// The merge gate (§7.4) on the reviews of a change set owned by `owner` whose current root is
/// `root`: per member (not the owner) its latest verdict for that root counts; none of them may be
/// `changes` or `failed`, a reviewer's must be `approve` and a verifier's `verified`. Returns the
/// records of the approval and the verification (the latest of each), or the unmet condition.
/// The same function decides at merge time and in `qal verify`.
pub fn gate(reviews: &[ReviewRec], owner: &str, root: &str) -> Result<(i64, i64), String> {
    let mut latest: BTreeMap<&str, &ReviewRec> = BTreeMap::new();
    for r in reviews.iter().filter(|r| r.root == root && r.member != owner && r.verdict != "comment") {
        if latest.get(r.member.as_str()).is_none_or(|x| x.seq < r.seq) {
            latest.insert(&r.member, r);
        }
    }
    if let Some(r) = latest.values().find(|r| r.verdict == "changes" || r.verdict == "failed") {
        return Err(format!("`{}` answered `{}` for the current root (record {}): its latest verdict must be `{}` again", r.member, r.verdict, r.seq, if r.verdict == "changes" { "approve" } else { "verified" }));
    }
    let best = |verdict: &str, role: &str| latest.values().filter(|r| r.verdict == verdict && r.role == role).map(|r| r.seq).max();
    match (best("approve", "reviewer"), best("verified", "verifier")) {
        (Some(a), Some(v)) => Ok((a, v)),
        (None, _) => Err("an `approve` by a reviewer who does not own the change set, for its current root".into()),
        (_, None) => Err("a `verified` by a verifier who does not own the change set, for its current root".into()),
    }
}

/// E0913 `gate_unmet`.
fn gate_unmet(change: &str, missing: impl std::fmt::Display) -> Diagnostic {
    Diagnostic::new("E0913", "gate_unmet", "", "change", format!("change set {change} cannot be merged yet: it needs {missing}"))
        .hint("reviews count only for the change set's current root (any write changes it): the owner submits, a verifier reviews with `verified`, a reviewer with `approve` (qal_review), then a reviewer or architect merges")
}

/// E0911 `merge_check_failed` (first) and the inner diagnostics (after it).
fn check_failed(what: &str, change: &str, inner: Vec<Diagnostic>) -> Vec<Diagnostic> {
    let names: BTreeSet<&str> = inner.iter().map(|d| d.def.as_str()).filter(|n| !n.is_empty()).collect();
    let mut d = Diagnostic::new("E0911", "merge_check_failed", "", "change", format!("{what} change set {change} onto main does not check: {} problem(s), in {}", inner.len(), if names.is_empty() { "the codebase".to_string() } else { names.iter().map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", ") }))
        .hint("main moved since the change set's base (e.g. a signature it uses changed): qal_rebase it, fix the names the diagnostics after this one name in the change set (qal_view `name@main` shows main's version), and submit again");
    d.detail = Some(json!({"names": names, "diagnostics": inner.len()}));
    let mut out = vec![d];
    out.extend(inner);
    out
}

/// E0912 `merge_tests_failed` (built by the merge's check: a refuted contract, a failed test, or
/// the test budget used up).
pub fn tests_failed(change: &str, message: impl Into<String>, detail: Value) -> Diagnostic {
    let mut d = Diagnostic::new("E0912", "merge_tests_failed", "", "change", format!("merging change set {change}: {}", message.into()))
        .hint("nothing was merged; reproduce it in the change set (qal_test, qal_prove with `change`), fix it there and submit again");
    d.detail = Some(detail);
    d
}

/// What a merge checks on the new main view besides the type check: it proves the contracts of
/// `changed` (name, hash: the functions and tests whose hash is new to main) afresh, never from the
/// `proofs` cache (4a review N4), and runs the tests. `Err` rejects the merge (E0912). The command
/// line crate implements it (the store has no prover and no runtime).
pub type MergeCheckFn<'a> = dyn FnMut(&Program, &TProgram, &[(String, String)]) -> Result<MergeChecked, Vec<Diagnostic>> + 'a;

/// The outcome of a [`MergeCheckFn`]: proof results to store (hash, JSON, as [`Verify`]) and what
/// the merge result shows (`tests`, `proofs`, `warnings`).
#[derive(Debug, Clone, Default)]
pub struct MergeChecked {
    pub proofs: Vec<(String, String)>,
    pub report: serde_json::Map<String, Value>,
}

/// The merge's check with the prover key its proof results are stored under.
pub struct MergeCheck<'a> {
    pub key: String,
    pub run: &'a mut MergeCheckFn<'a>,
}

/// The reviews of change set `change` whose records say exactly what their rows say (a row without
/// such a record does not count), by members not revoked.
fn reviews_of(tx: &Transaction, change: &str) -> Result<Vec<ReviewRec>, Vec<Diagnostic>> {
    let mut stmt = tx
        .prepare("SELECT r.member, r.verdict, r.root, r.record, rec.body, m.role, m.revoked_at FROM reviews r JOIN records rec ON rec.seq = r.record JOIN members m ON m.id = r.member WHERE r.change = ?1 ORDER BY r.record")
        .map_err(sql)?;
    let rows = stmt
        .query_map(params![change], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?, r.get::<_, i64>(3)?, r.get::<_, Vec<u8>>(4)?, r.get::<_, String>(5)?, r.get::<_, Option<i64>>(6)?)))
        .map_err(sql)?;
    let mut out = Vec::new();
    for row in rows {
        let (member, verdict, root, seq, body, role, revoked) = row.map_err(sql)?;
        let Some(v) = body.strip_prefix(super::RECORD_PREFIX.as_bytes()).and_then(|b| serde_json::from_slice::<Value>(b).ok()) else { continue };
        let args = &v["args"];
        let matches = v["op"] == "review" && v["member"] == member.as_str() && args["change"] == change && args["verdict"] == verdict.as_str() && args["root"] == root.as_str();
        if matches && revoked.is_none() {
            out.push(ReviewRec { seq, member, role, verdict, root });
        }
    }
    Ok(out)
}

/// The SHA-256 of a review's comments: of their canonical JSON (what `reviews.comments` holds).
pub fn comments_json(comments: &[Comment]) -> Result<String, Vec<Diagnostic>> {
    let v = serde_json::to_value(comments).map_err(|e| one(Diagnostic::store_error(format!("cannot serialize comments: {e}"))))?;
    canonical::to_string(&v).map_err(|e| one(Diagnostic::store_error(e)))
}

// --- Classification (§7.2) ---------------------------------------------------------------------

/// The class of one name of a change set against `main` now.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Class {
    /// Replay the change set's version (`main` did not touch the name, or only relinked it).
    Clean,
    /// Both came to the same definition: nothing to do.
    Converged,
    /// A real conflict: `edit_edit`, `add_add` or `rename`, with who changed it in `main` last.
    Conflict { kind: &'static str, by: Option<String>, record: Option<i64> },
}

/// One name of a change set's delta (or a user that follows its renames), with its hashes.
#[derive(Debug, Clone)]
struct Delta {
    name: String,
    origin: String,
    base: Option<String>,
    ours: Option<String>,
    theirs: Option<String>,
    /// A user of a renamed or deleted name: same hash on both sides, but its refs must follow.
    follower: bool,
    class: Class,
}

/// What `main` did to `name` since record `base_seq`: (action, agent, record) in order.
fn main_changes(tx: &Transaction, base_seq: i64, name: &str) -> Result<Vec<(String, String, Option<i64>)>, Vec<Diagnostic>> {
    let mut stmt = tx
        .prepare_cached("SELECT action, agent, record FROM history WHERE space = 'main' AND record > ?1 AND (name = ?2 OR from_name = ?2) ORDER BY id")
        .map_err(sql)?;
    let rows = stmt.query_map(params![base_seq, name], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))).map_err(sql)?;
    rows.collect::<Result<Vec<_>, _>>().map_err(sql)
}

/// §7.2 rows 2-7 for one name (row 1, the fast-forward, is decided for the whole change set).
fn classify(tx: &Transaction, base_seq: i64, d: &Delta) -> Result<Class, Vec<Diagnostic>> {
    if d.ours == d.theirs && !d.follower {
        return Ok(Class::Converged);
    }
    if d.theirs == d.base {
        return Ok(Class::Clean);
    }
    let changes = main_changes(tx, base_seq, &d.name)?;
    let (by, record) = changes.last().map_or((None, None), |(_, a, r)| (Some(a.clone()), *r));
    let has = |a: &[&str]| changes.iter().any(|(x, _, _)| a.contains(&x.as_str()));
    if !has(&["added", "replaced", "renamed", "deleted"]) {
        // Only relinked in main (row 4): the change set's text is replayed against the new dependencies.
        return Ok(Class::Clean);
    }
    let kind = if has(&["renamed"]) || d.origin == "renamed" {
        "rename"
    } else if d.base.is_none() && d.theirs.is_some() {
        "add_add"
    } else {
        "edit_edit"
    };
    Ok(Class::Conflict { kind, by, record })
}

/// The conflict mark of a rebase (`change_names.conflict`): the canonical JSON `{base, theirs}`.
pub fn conflict_mark(base: Option<&str>, theirs: Option<&str>) -> String {
    canonical::to_string(&json!({"base": base, "theirs": theirs})).unwrap_or_default()
}

fn conflict_diag(change: &str, d: &Delta) -> Diagnostic {
    let Class::Conflict { kind, by, record } = &d.class else { return Diagnostic::store_error("not a conflict") };
    let what = match *kind {
        "add_add" => "was added in main too, with another definition",
        "rename" => "was renamed (or its name reused) on one side and changed on the other",
        _ => "was changed in main too, differently",
    };
    let mut x = Diagnostic::new("E0910", "merge_conflict", &d.name, "", format!("`{}` {what} since change set {change}'s base ({kind})", d.name))
        .hint(format!("qal_rebase {{\"change\": \"{change}\"}} marks the conflict; compare with qal_view [\"{}\", \"{}@main\"], define `{}` again in the change set, and submit it again", d.name, d.name, d.name));
    x.detail = Some(json!({"conflict": kind, "name": d.name, "base": d.base, "ours": d.ours, "theirs": d.theirs, "theirs_by": by, "record": record}));
    x
}

/// The names of a change set's delta (`E`, §7.1), plus the users that must follow its renames and
/// deletes, each classified against `main` now.
fn deltas(tx: &Transaction, c: &ChangeRow, meta: &Meta, main: &Rows) -> Result<Vec<Delta>, Vec<Diagnostic>> {
    let removed: BTreeSet<&str> = meta.rows.iter().filter(|(_, r)| r.hash.is_none() && r.base_hash.is_some()).map(|(n, _)| n.as_str()).collect();
    let mut out = Vec::new();
    for (n, r) in &meta.rows {
        let edited = matches!(r.origin.as_str(), "edited" | "added" | "renamed");
        // A user of a renamed or deleted name follows the change set's version (its refs).
        let follower = r.origin == "relinked" && r.hash.is_some() && main.get(n).is_some_and(|m| m.refs.iter().any(|x| removed.contains(x.as_str())));
        if !edited && !follower {
            continue;
        }
        let mut d = Delta { name: n.clone(), origin: r.origin.clone(), base: r.base_hash.clone(), ours: r.hash.clone(), theirs: main.get(n).map(|m| m.hash.clone()), follower, class: Class::Clean };
        d.class = classify(tx, c.base_seq, &d)?;
        out.push(d);
    }
    Ok(out)
}

/// What a replay puts on top of `main`: the definitions to replay (decoded from the change set,
/// written against its names) and the names to remove (with the name a rename gave them).
struct Overlay {
    input: Program,
    removed: BTreeMap<String, Option<String>>,
}

/// The overlay of the deltas that are not converged and not in `keep_conflicts`' way: clean names
/// always; conflicting ones only for a rebase (they keep the change set's version).
fn overlay(tx: &Transaction, c: &ChangeRow, deltas: &[Delta], with_conflicts: bool) -> Result<Overlay, Vec<Diagnostic>> {
    let snap = Snapshot::load(tx, &Space::Change(c.id.clone()))?;
    let mut input = Program { defs: Vec::new() };
    let mut removed = BTreeMap::new();
    for d in deltas {
        let take = match d.class {
            Class::Clean => true,
            Class::Converged => false,
            Class::Conflict { .. } => with_conflicts,
        };
        if !take {
            continue;
        }
        match &d.ours {
            Some(_) => input.defs.push(snap.decode_name(tx, &d.name)?),
            None if d.theirs.is_some() => {
                removed.insert(d.name.clone(), renamed_to(tx, &c.id, &d.name, &snap)?);
            }
            None => {}
        }
    }
    Ok(Overlay { input, removed })
}

/// The name the change set renamed `old` to, if it still has it (else `old` was deleted).
fn renamed_to(tx: &Transaction, change: &str, old: &str, snap: &Snapshot) -> Result<Option<String>, Vec<Diagnostic>> {
    let to: Option<String> = tx
        .query_row("SELECT to_name FROM history WHERE space = ?1 AND action = 'renamed' AND from_name = ?2 ORDER BY id DESC LIMIT 1", params![change, old], |r| r.get(0))
        .optional()
        .map_err(sql)?;
    Ok(to.filter(|t| snap.names.contains_key(t)))
}

// --- Merge --------------------------------------------------------------------------------------

/// Everything a merge writes, computed from a snapshot (inside or outside the write lock).
struct Plan {
    row: ChangeRow,
    c_root: String,
    main_root: String,
    fast_forward: bool,
    reviews: (i64, i64),
    before: Rows,
    after: Rows,
    /// The per-name results (added/replaced/relinked; `define`'s statuses relative to main).
    defined: Vec<Defined>,
    removed: BTreeMap<String, Option<String>>,
    program: Program,
    typed: TProgram,
    /// The definitions to store (a replay); a fast-forward's are stored already.
    replayed: Option<Replayed>,
    /// The functions and tests whose hash is new to main (to prove and test).
    changed: Vec<(String, String)>,
    /// Delta names whose hash in main differs from the change set's (relinked by the replay).
    rehashed: Vec<String>,
    stubs: Vec<String>,
    /// Names whose interface (kind, signature, contracts) the merge changed or removed in main.
    interfaces: BTreeSet<String>,
}

/// The merge plan of change set `id` for `signer`: the rights, the gate (§7.4), the classification
/// (§7.2), and the fast-forward or the checked replay. Writes nothing.
fn plan(tx: &Transaction, signer: &Signer, id: &str) -> Result<Plan, Vec<Diagnostic>> {
    let role = role_of(tx, &signer.member)?;
    let row = require(tx, id)?;
    let kind = row.kind_for(&signer.member);
    if !policy::allows("qal_merge", None, role, kind) {
        return Err(one(policy::forbidden("qal_merge", None, role, kind)));
    }
    if !row.is_live() {
        return Err(one(change::not_live(&row)));
    }
    if row.status != "submitted" {
        return Err(one(gate_unmet(id, "the status `submitted` (it is open: its owner submits it with qal_change submit)")));
    }
    let marks: i64 = tx.query_row("SELECT count(*) FROM change_names WHERE change = ?1 AND conflict IS NOT NULL", params![id], |r| r.get(0)).map_err(sql)?;
    if marks > 0 {
        return Err(one(state(format!("change set {id} has {marks} conflict mark(s) from its rebase")).hint("resolve each marked name by defining it again in the change set (qal_change diff shows them), then submit again")));
    }
    let space = Space::Change(id.to_string());
    let c_rows = space_rows(tx, &space)?;
    let c_root = root_of(&c_rows);
    let reviews = gate(&reviews_of(tx, id)?, &row.owner, &c_root).map_err(|m| one(gate_unmet(id, m)))?;
    let main_snap = Snapshot::load(tx, &Space::Main)?;
    let before = rows_of(&main_snap);
    let main_root = root_of(&before);
    let meta = load_meta(tx, id)?;
    let fast_forward = main_root == row.base_root;
    let c_snap = Snapshot::load(tx, &space)?;
    let input_origin = |n: &str| meta.rows.get(n).is_some_and(|r| matches!(r.origin.as_str(), "edited" | "added"));
    let (after, defined, removed, program, typed, replayed) = if fast_forward {
        // Row 1: main did not move, so the change set's checked view becomes main as it is.
        let program = c_snap.program(tx, &|_| false)?;
        let typed = qal_compiler::check_stored(&program, &|_| false).map_err(|d| check_failed("fast-forwarding", id, d))?;
        let mut defined = Vec::new();
        for (n, r) in &c_rows {
            let old = before.get(n);
            if old == Some(r) {
                continue;
            }
            let kind = c_snap.row(tx, &hash_of(&r.hash)?)?.map_or("fn", |x| x.kind.as_str());
            let status = match old {
                None => "added",
                Some(o) if o.hash == r.hash => "unchanged",
                Some(_) if input_origin(n) => "replaced",
                Some(_) => "relinked",
            };
            defined.push(Defined { name: n.clone(), kind, hash: r.hash.clone(), status, old_hash: old.map(|o| o.hash.clone()).filter(|_| matches!(status, "replaced" | "relinked")) });
        }
        let mut removed = BTreeMap::new();
        for n in before.keys().filter(|n| !c_rows.contains_key(*n)) {
            removed.insert(n.clone(), renamed_to(tx, id, n, &c_snap)?);
        }
        (c_rows.clone(), defined, removed, program, typed, None)
    } else {
        let ds = deltas(tx, &row, &meta, &before)?;
        let conflicts: Vec<Diagnostic> = ds.iter().filter(|d| matches!(d.class, Class::Conflict { .. })).map(|d| conflict_diag(id, d)).collect();
        if !conflicts.is_empty() {
            return Err(conflicts);
        }
        let o = overlay(tx, &row, &ds, false)?;
        let removed_set: BTreeSet<String> = o.removed.keys().cloned().collect();
        let r = define::replay(tx, &main_snap, &o.input, &removed_set).map_err(|d| check_failed("replaying", id, d))?;
        let mut after = r.rows.clone();
        for n in &removed_set {
            after.remove(n);
        }
        (after, r.defined.clone(), o.removed, r.program.clone(), r.typed.clone(), Some(r))
    };
    let changed: Vec<(String, String)> = defined.iter().filter(|d| d.kind != "type" && d.status != "unchanged").map(|d| (d.name.clone(), d.hash.clone())).collect();
    let rehashed: Vec<String> = meta.rows.iter().filter(|(n, r)| matches!(r.origin.as_str(), "edited" | "added") && r.hash.is_some() && after.get(*n).map(|a| &a.hash) != r.hash.as_ref()).map(|(n, _)| n.clone()).collect();
    let delta: BTreeSet<&str> = meta.rows.iter().filter(|(_, r)| matches!(r.origin.as_str(), "edited" | "added" | "renamed")).map(|(n, _)| n.as_str()).collect();
    let stubs: Vec<String> = typed.stubs().into_iter().filter(|n| delta.contains(n.as_str())).collect();
    // The interfaces this merge changes in main (for `needs_rebase` of the other change sets).
    let mut interfaces: BTreeSet<String> = removed.keys().cloned().collect();
    for d in defined.iter().filter(|d| matches!(d.status, "replaced")) {
        let (Ok(old), Ok(new)) = (main_snap.decode_name(tx, &d.name), c_snap.decode_name(tx, &d.name)) else { continue };
        if interface_change(&old, &new).is_some() {
            interfaces.insert(d.name.clone());
        }
    }
    Ok(Plan { row, c_root, main_root, fast_forward, reviews, before, after, defined, removed, program, typed, replayed, changed, rehashed, stubs, interfaces })
}

fn hash_of(h: &str) -> Result<Hash, Vec<Diagnostic>> {
    Hash::from_hex(h).ok_or_else(|| one(Diagnostic::store_error(format!("the store is corrupt: `{h}` is not a hash"))))
}

/// Writes `rows` over the names of `main` (and history rows for every hash that changes), removing
/// `removed` (a `deleted` or `renamed` history row each).
fn apply_main(tx: &Transaction, before: &Rows, after: &Rows, defined: &[Defined], removed: &BTreeMap<String, Option<String>>, agent: &str) -> Result<(), Vec<Diagnostic>> {
    let now = now_ms();
    let list = |v: &[String]| Value::from(v.to_vec()).to_string();
    let status: BTreeMap<&str, &Defined> = defined.iter().map(|d| (d.name.as_str(), d)).collect();
    for (n, r) in after {
        let old = before.get(n);
        if old == Some(r) {
            continue;
        }
        space::put(tx, &Space::Main, n, &r.hash, &list(&r.locals), &list(&r.refs), true, now)?;
        if old.map(|o| &o.hash) != Some(&r.hash) {
            let action = status.get(n.as_str()).map_or("relinked", |d| d.status);
            tx.execute(
                "INSERT INTO history (name, action, old_hash, new_hash, agent, at, space) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'main')",
                params![n, action, old.map(|o| o.hash.clone()), r.hash, agent, now],
            )
            .map_err(sql)?;
        }
    }
    for (n, to) in removed {
        let Some(old) = before.get(n) else { continue };
        space::remove(tx, &Space::Main, n, "edited", now)?;
        match to {
            Some(t) => tx.execute(
                "INSERT INTO history (name, action, old_hash, new_hash, from_name, to_name, agent, at, space) VALUES (?1, 'renamed', ?2, ?2, ?3, ?1, ?4, ?5, 'main')",
                params![t, old.hash, n, agent, now],
            ),
            None => tx.execute("INSERT INTO history (name, action, old_hash, new_hash, agent, at, space) VALUES (?1, 'deleted', ?2, ?2, ?3, ?4, 'main')", params![n, old.hash, agent, now]),
        }
        .map_err(sql)?;
    }
    Ok(())
}

/// The names change set `d` builds on: its delta and everything those (transitively) refer to in
/// its space.
fn builds_on(tx: &Transaction, d: &str) -> Result<(BTreeSet<String>, BTreeSet<String>), Vec<Diagnostic>> {
    let snap = Snapshot::load(tx, &Space::Change(d.to_string()))?;
    let delta: BTreeSet<String> = {
        let mut stmt = tx.prepare_cached("SELECT name FROM change_names WHERE change = ?1 AND origin IN ('edited', 'added', 'renamed')").map_err(sql)?;
        let rows = stmt.query_map(params![d], |r| r.get::<_, String>(0)).map_err(sql)?;
        rows.collect::<Result<_, _>>().map_err(sql)?
    };
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut stack: Vec<String> = delta.iter().cloned().collect();
    while let Some(n) = stack.pop() {
        if let Some(e) = snap.names.get(&n) {
            for r in &e.refs {
                if seen.insert(r.clone()) {
                    stack.push(r.clone());
                }
            }
        }
    }
    Ok((delta, seen))
}

/// The merge's writes (inside the write transaction): main's names and history, the definitions
/// and proof results, the signed records, the change set's end, `needs_rebase` of the change sets
/// it affects, and the notices.
fn write_merge(tx: &Transaction, signer: &Signer, p: &Plan, checked: &MergeChecked, key: &str) -> Result<Value, Vec<Diagnostic>> {
    let id = p.row.id.as_str();
    if let Some(r) = &p.replayed {
        define::store_defs(tx, &r.hashed)?;
    }
    let hashes: HashSet<&str> = p.changed.iter().map(|(_, h)| h.as_str()).collect();
    define::store_results(tx, key, checked.proofs.iter().filter(|(h, _)| hashes.contains(h.as_str())).cloned())?;
    apply_main(tx, &p.before, &p.after, &p.defined, &p.removed, &signer.member)?;
    let now_rows = space_rows(tx, &Space::Main)?;
    if now_rows != p.after {
        return Err(one(Diagnostic::store_error("the merge did not give the planned names of main")));
    }
    let mut labels = Entry::of_defined(&p.defined);
    for (n, to) in &p.removed {
        labels.push(Entry::label(n, p.before.get(n).map(|r| r.hash.as_str()), None, if to.is_some() { "renamed" } else { "deleted" }));
    }
    let entries = diff(&p.before, &now_rows, &labels);
    let kind = if p.fast_forward { "fast_forward" } else { "replayed" };
    let args = json!({"change": id, "change_root": p.c_root, "kind": kind, "reviews": [p.reviews.0, p.reviews.1], "entries": serde_json::to_value(&entries).map_err(|e| one(Diagnostic::store_error(e.to_string())))?});
    // The change set ends: its leases are free (§5.2), nobody works in it any more.
    tx.execute("DELETE FROM leases WHERE change = ?1", params![id]).map_err(sql)?;
    let seq = append(tx, signer, "merge", args, &p.before, &now_rows)?;
    change::after_write(tx, signer, &Space::Main, &entries)?;
    let now = now_ms();
    tx.execute("UPDATE changes SET status = 'merged', updated_at = ?2 WHERE id = ?1", params![id, now]).map_err(sql)?;
    tx.execute("DELETE FROM seen WHERE space = ?1", params![id]).map_err(sql)?;
    tx.execute("UPDATE sessions SET current_change = NULL WHERE current_change = ?1", params![id]).map_err(sql)?;
    record_change(tx, signer, id, "merged", json!({"merge": seq}))?;
    // Other change sets that change a name this merge changed, or build on an interface it changed
    // (§5.4), need a rebase: flagged with a record in their own space (review N6).
    let changed_names: BTreeSet<&str> = entries.iter().filter(|e| e.action != "relinked" && e.action != "updated").map(|e| e.name.as_str()).collect();
    let others: Vec<(String, String, bool)> = {
        let mut stmt = tx.prepare("SELECT id, owner, needs_rebase FROM changes WHERE status IN ('open', 'submitted') AND id != ?1 ORDER BY id").map_err(sql)?;
        let rows = stmt.query_map(params![id], |r| Ok((r.get(0)?, r.get(1)?, r.get::<_, i64>(2)? != 0))).map_err(sql)?;
        rows.collect::<Result<_, _>>().map_err(sql)?
    };
    let mut flagged = Vec::new();
    for (d, owner, already) in others {
        let (delta, deps) = builds_on(tx, &d)?;
        let mut why: Vec<String> = delta.iter().filter(|n| changed_names.contains(n.as_str())).cloned().collect();
        why.extend(deps.iter().filter(|n| p.interfaces.contains(*n) && !delta.contains(*n)).cloned());
        if why.is_empty() {
            continue;
        }
        if !already {
            tx.execute("UPDATE changes SET needs_rebase = 1, updated_at = ?2 WHERE id = ?1", params![d, now]).map_err(sql)?;
            record_change(tx, signer, &d, "needs_rebase", json!({"merge": seq, "names": why}))?;
        }
        notify(tx, &owner, "needs_rebase", &json!({"change": d, "merged": id, "names": why, "record": seq}))?;
        flagged.push(json!({"change": d, "names": why}));
    }
    for m in super::members_of(tx)?.into_iter().filter(|m| m.revoked_at.is_none() && m.id != signer.member && m.role != policy::OWNER) {
        notify(tx, &m.id, "merged", &json!({"change": id, "title": p.row.title, "by": signer.member, "merge_kind": kind, "record": seq}))?;
    }
    let mut out = json!({"merged": true, "change": id, "kind": kind, "record": seq, "entries": entries, "rehashed": p.rehashed, "reviews": [p.reviews.0, p.reviews.1]});
    if let Value::Object(m) = &mut out {
        for (k, v) in &checked.report {
            m.insert(k.clone(), v.clone());
        }
        if !flagged.is_empty() {
            m.insert("needs_rebase".into(), Value::Array(flagged));
        }
        if !p.stubs.is_empty() {
            m.insert("stubs".into(), json!(p.stubs));
            let mut warnings: Vec<Value> = m.get("warnings").and_then(Value::as_array).cloned().unwrap_or_default();
            for n in &p.stubs {
                let d = Diagnostic::new("E0930", "stub_present", n, "body", format!("`{n}` is a stub: calling it reaches `todo()` (runtime error todo_reached)")).hint(format!("replace the `todo()` in `{n}` in a later change set (same signature: its users relink)"));
                warnings.push(serde_json::to_value(d).unwrap_or(Value::Null));
            }
            m.insert("warnings".into(), Value::Array(warnings));
        }
    }
    Ok(out)
}

impl Store {
    /// `qal_review` (§8, §9.1): a verdict on change set `id` for its current root, signed in its
    /// space. `changes` sends a submitted change set back to `open`; every verdict but `comment`
    /// needs it submitted. The owner hears of it (notice `review`).
    pub fn review(&mut self, signer: &Signer, id: &str, verdict: &str, comments: &[Comment]) -> Result<Value, Vec<Diagnostic>> {
        if !VERDICTS.contains(&verdict) {
            return Err(one(super::bad(format!("`verdict` must be approve, changes, verified, failed or comment, not `{verdict}`"))));
        }
        if comments.len() > MAX_COMMENTS || comments.iter().any(|c| c.text.len() > MAX_COMMENT_BYTES || c.name.as_ref().is_some_and(|n| n.len() > MAX_COMMENT_BYTES)) {
            return Err(one(team_limit(format!("a review has at most {MAX_COMMENTS} comments of at most {MAX_COMMENT_BYTES} bytes each"))));
        }
        let text = comments_json(comments)?;
        let tx = self.team_tx("`qal_review`")?;
        let role = role_of(&tx, &signer.member)?;
        let row = require(&tx, id)?;
        let kind = row.kind_for(&signer.member);
        if !policy::allows("qal_review", Some(verdict), role, kind) {
            return Err(one(policy::forbidden("qal_review", Some(verdict), role, kind)));
        }
        if !row.is_live() {
            return Err(one(change::not_live(&row)));
        }
        if verdict != "comment" && row.status != "submitted" {
            return Err(one(state(format!("change set {id} is {}: only a submitted change set gets the verdict `{verdict}`", row.status)).hint("wait for its owner to submit it (notice change_submitted); a `comment` is possible any time")));
        }
        let space = Space::Change(id.to_string());
        let rows = space_rows(&tx, &space)?;
        let root = root_of(&rows);
        let now = now_ms();
        if verdict == "changes" {
            tx.execute("UPDATE changes SET status = 'open', updated_at = ?2 WHERE id = ?1", params![id, now]).map_err(sql)?;
        }
        let digest = Hash::of(text.as_bytes()).to_hex();
        let record = append_in(&tx, signer, "review", json!({"change": id, "root": root, "verdict": verdict, "comments_sha256": digest}), &space, &rows, &rows)?;
        tx.execute("INSERT INTO reviews (change, member, verdict, root, comments, at, record) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)", params![id, signer.member, verdict, root, text, now, record]).map_err(sql)?;
        let review = tx.last_insert_rowid();
        if row.owner != signer.member {
            notify(&tx, &row.owner, "review", &json!({"change": id, "verdict": verdict, "by": signer.member, "root": root, "comments": comments.len(), "review": review, "record": record}))?;
        }
        let status = if verdict == "changes" { "open" } else { row.status.as_str() };
        let out = json!({"review": review, "change": id, "verdict": verdict, "root": root, "status": status, "record": record});
        tx.commit().map_err(sql)?;
        Ok(out)
    }

    /// The root (§3) of a space now.
    pub fn root_of_space(&self, space: &Space) -> Result<String, Vec<Diagnostic>> {
        Ok(root_of(&space_rows(&self.conn, space)?))
    }

    /// The reviews of change set `id`, oldest first, with whether each is for its current root.
    pub fn reviews(&mut self, id: &str) -> Result<Vec<Value>, Vec<Diagnostic>> {
        let tx = self.conn.transaction().map_err(sql)?;
        require(&tx, id)?;
        let root = root_of(&space_rows(&tx, &Space::Change(id.to_string()))?);
        let mut stmt = tx.prepare("SELECT id, member, verdict, root, comments, record FROM reviews WHERE change = ?1 ORDER BY id").map_err(sql)?;
        let rows = stmt
            .query_map(params![id], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?, r.get::<_, String>(3)?, r.get::<_, String>(4)?, r.get::<_, i64>(5)?)))
            .map_err(sql)?;
        let mut out = Vec::new();
        for row in rows {
            let (rid, member, verdict, at, comments, record) = row.map_err(sql)?;
            let comments: Value = serde_json::from_str(&comments).unwrap_or(Value::Null);
            out.push(json!({"review": rid, "member": member, "verdict": verdict, "current": at == root, "comments": comments, "record": record}));
        }
        Ok(out)
    }

    /// `qal_merge` (§7): merges submitted change set `id` into `main` if the gate holds (§7.4): a
    /// fast-forward if `main` did not move since its base, else every name is classified (§7.2;
    /// conflicts are E0910) and the change set's definitions are replayed onto `main` as one
    /// `define` (E0911 if that does not check), then `check` proves afresh and runs the tests
    /// (E0912). Nothing is written unless all of it holds.
    pub fn merge(&mut self, signer: &Signer, id: &str, check: &mut MergeCheck) -> Result<Value, Vec<Diagnostic>> {
        if !super::is_team(&self.conn)? {
            return Err(one(super::not_a_team("`qal_merge`")));
        }
        for _ in 0..ATTEMPTS {
            let (roots, checked) = {
                let tx = self.conn.transaction().map_err(sql)?;
                let p = plan(&tx, signer, id)?;
                drop(tx);
                let checked = (check.run)(&p.program, &p.typed, &p.changed)?;
                ((p.main_root, p.c_root), checked)
            };
            let tx = self.conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
            let p = plan(&tx, signer, id)?;
            if (p.main_root.as_str(), p.c_root.as_str()) == (roots.0.as_str(), roots.1.as_str()) {
                let out = write_merge(&tx, signer, &p, &checked, &check.key)?;
                tx.commit().map_err(sql)?;
                return Ok(out);
            }
            // Another writer moved main or the change set: start over.
        }
        let tx = self.conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
        let p = plan(&tx, signer, id)?;
        let checked = (check.run)(&p.program, &p.typed, &p.changed)?;
        let out = write_merge(&tx, signer, &p, &checked, &check.key)?;
        tx.commit().map_err(sql)?;
        Ok(out)
    }

    /// `qal_rebase` (§7.3, owner): the change set's base becomes `main` now; its names are replayed
    /// like in a merge, but in its own space. A conflicting name keeps the change set's version and
    /// gets a conflict mark `{base, theirs}` (a later define of it clears the mark). If the replay
    /// does not check, nothing changes (E0911 with the diagnostics). Proofs as in `define`.
    pub fn rebase(&mut self, signer: &Signer, id: &str, verify: Option<&mut Verify>) -> Result<Value, Vec<Diagnostic>> {
        if !super::is_team(&self.conn)? {
            return Err(one(super::not_a_team("`qal_rebase`")));
        }
        let Some(v) = verify else {
            let tx = self.conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
            let out = match plan_rebase(&tx, signer, id)? {
                Rebase::Nothing(v) => v,
                Rebase::Plan(p) => write_rebase(&tx, signer, &p, None)?,
            };
            tx.commit().map_err(sql)?;
            return Ok(out);
        };
        for _ in 0..ATTEMPTS {
            let (roots, rows) = {
                let tx = self.conn.transaction().map_err(sql)?;
                let p = match plan_rebase(&tx, signer, id)? {
                    Rebase::Nothing(v) => return Ok(v),
                    Rebase::Plan(p) => p,
                };
                let todo = define::to_verify(&tx, &p.verify, &v.key)?;
                drop(tx);
                let rows = if todo.is_empty() { Vec::new() } else { (v.run)(&p.replayed.typed, &todo)? };
                ((p.main_root, p.c_root), rows)
            };
            let tx = self.conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
            let p = match plan_rebase(&tx, signer, id)? {
                Rebase::Nothing(v) => return Ok(v),
                Rebase::Plan(p) => p,
            };
            if (p.main_root.as_str(), p.c_root.as_str()) == (roots.0.as_str(), roots.1.as_str()) {
                let out = write_rebase(&tx, signer, &p, Some((&v.key, rows)))?;
                tx.commit().map_err(sql)?;
                return Ok(out);
            }
        }
        let tx = self.conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql)?;
        let out = match plan_rebase(&tx, signer, id)? {
            Rebase::Nothing(v) => v,
            Rebase::Plan(p) => {
                let todo = define::to_verify(&tx, &p.verify, &v.key)?;
                let rows = if todo.is_empty() { Vec::new() } else { (v.run)(&p.replayed.typed, &todo)? };
                write_rebase(&tx, signer, &p, Some((&v.key, rows)))?
            }
        };
        tx.commit().map_err(sql)?;
        Ok(out)
    }
}

// --- Rebase -------------------------------------------------------------------------------------

/// The metadata rows of a change set after a rebase onto `main` (§7.3), from what it was (`prev`),
/// main's rows, the change set's rows after the replay and the conflict marks. Deterministic, so
/// `qal verify` replays the rebase record with the same function:
/// - a marked name: `edited` (or `added` if main has no such name) with its mark;
/// - a name whose row equals main's: `base`;
/// - a name the change set wrote (it was `edited`/`added`): `edited`, or `added` if main has none;
/// - any other name: `relinked` (it differs from main only through what it refers to);
/// - a name the change set removed that main still has: kept without a hash (its origin stays);
/// the base hash of every row is main's hash of the name.
pub fn rebase_meta(prev: &Meta, main: &Rows, after: &Rows, marks: &BTreeMap<String, String>) -> BTreeMap<String, MetaRow> {
    let list = |v: &[String]| Value::from(v.to_vec()).to_string();
    let mut out = BTreeMap::new();
    for (n, r) in after {
        let m = main.get(n);
        let base_hash = m.map(|m| m.hash.clone());
        let was_input = prev.rows.get(n).is_some_and(|p| matches!(p.origin.as_str(), "edited" | "added"));
        let (origin, conflict) = if let Some(mark) = marks.get(n) {
            (if base_hash.is_some() { "edited" } else { "added" }, Some(mark.clone()))
        } else if m == Some(r) {
            ("base", None)
        } else if was_input {
            (if base_hash.is_some() { "edited" } else { "added" }, None)
        } else if base_hash.is_some() {
            ("relinked", None)
        } else {
            ("added", None)
        };
        out.insert(n.clone(), MetaRow { hash: Some(r.hash.clone()), locals: list(&r.locals), refs: list(&r.refs), origin: origin.into(), base_hash, conflict });
    }
    for (n, p) in prev.rows.iter().filter(|(n, p)| p.hash.is_none() && !after.contains_key(*n)) {
        if let Some(m) = main.get(n) {
            out.insert(n.clone(), MetaRow { hash: None, locals: "[]".into(), refs: "[]".into(), origin: p.origin.clone(), base_hash: Some(m.hash.clone()), conflict: marks.get(n).cloned() });
        }
    }
    out
}

enum Rebase {
    /// Nothing to do (main did not move and nothing asks for a rebase): the answer.
    Nothing(Value),
    Plan(Box<RebasePlan>),
}

struct RebasePlan {
    row: ChangeRow,
    c_root: String,
    main_root: String,
    before: Rows,
    replayed: Replayed,
    after: Rows,
    meta: BTreeMap<String, MetaRow>,
    marks: BTreeMap<String, String>,
    conflicts: Vec<Value>,
    replayed_names: Vec<String>,
    /// The per-name results relative to the change set before (for the prover, as `define`).
    verify: Vec<Defined>,
}

fn plan_rebase(tx: &Transaction, signer: &Signer, id: &str) -> Result<Rebase, Vec<Diagnostic>> {
    let role = role_of(tx, &signer.member)?;
    let row = require(tx, id)?;
    let kind = row.kind_for(&signer.member);
    if !policy::allows("qal_rebase", None, role, kind) {
        return Err(one(if kind == policy::Space::Own || role == Role::Reviewer { policy::forbidden("qal_rebase", None, role, kind) } else { change::not_owner(&row, "a rebase") }));
    }
    if !row.is_live() {
        return Err(one(change::not_live(&row)));
    }
    let main_snap = Snapshot::load(tx, &Space::Main)?;
    let main = rows_of(&main_snap);
    let main_root = root_of(&main);
    let space = Space::Change(id.to_string());
    let before = space_rows(tx, &space)?;
    let c_root = root_of(&before);
    if main_root == row.base_root && !row.needs_rebase {
        return Ok(Rebase::Nothing(json!({"change": id, "base_root": main_root, "unchanged": true, "replayed": [], "conflicts": []})));
    }
    let meta = load_meta(tx, id)?;
    let ds = deltas(tx, &row, &meta, &main)?;
    let o = overlay(tx, &row, &ds, true)?;
    let removed_set: BTreeSet<String> = o.removed.keys().cloned().collect();
    let replayed = define::replay(tx, &main_snap, &o.input, &removed_set).map_err(|d| check_failed("rebasing", id, d))?;
    let mut after = replayed.rows.clone();
    for n in &removed_set {
        after.remove(n);
    }
    let mut marks = BTreeMap::new();
    let mut conflicts = Vec::new();
    for d in &ds {
        if let Class::Conflict { kind, by, record } = &d.class {
            marks.insert(d.name.clone(), conflict_mark(d.base.as_deref(), d.theirs.as_deref()));
            conflicts.push(json!({"name": d.name, "conflict": kind, "base": d.base, "ours": d.ours, "theirs": d.theirs, "theirs_by": by, "record": record}));
        }
    }
    // A mark nobody resolved yet stays (a later rebase does not decide it), unless both sides
    // now have the same definition.
    for (n, r) in &meta.rows {
        if marks.contains_key(n) {
            continue;
        }
        let converged = ds.iter().any(|d| d.name == *n && d.class == Class::Converged);
        let kept = r.hash.is_some() || main.contains_key(n);
        if let (Some(mark), false, true) = (&r.conflict, converged, kept) {
            marks.insert(n.clone(), mark.clone());
            let m: Value = serde_json::from_str(mark).unwrap_or(Value::Null);
            conflicts.push(json!({"name": n, "conflict": "unresolved", "base": m["base"], "ours": r.hash, "theirs": m["theirs"], "theirs_by": Value::Null, "record": Value::Null}));
        }
    }
    let meta_rows = rebase_meta(&meta, &main, &after, &marks);
    let replayed_names: Vec<String> = o.input.defs.iter().map(|d| d.name().to_string()).collect();
    // For the prover: what changed relative to the change set (as its `define` would report).
    let mut verify = Vec::new();
    for (n, r) in &after {
        let old = before.get(n);
        if old.map(|o| &o.hash) == Some(&r.hash) {
            continue;
        }
        let kind = replayed.defined.iter().find(|d| d.name == *n).map_or_else(|| main_snap_kind(&main_snap, tx, n), |d| d.kind);
        verify.push(Defined { name: n.clone(), kind, hash: r.hash.clone(), status: if old.is_some() { "relinked" } else { "added" }, old_hash: old.map(|o| o.hash.clone()) });
    }
    Ok(Rebase::Plan(Box::new(RebasePlan { row, c_root, main_root, before, replayed, after, meta: meta_rows, marks, conflicts, replayed_names, verify })))
}

fn main_snap_kind(snap: &Snapshot, tx: &Transaction, name: &str) -> &'static str {
    snap.names.get(name).and_then(|e| snap.row(tx, &e.hash).ok().flatten()).map_or("fn", |r| r.kind.as_str())
}

fn write_rebase(tx: &Transaction, signer: &Signer, p: &RebasePlan, proofs: Option<(&str, Vec<(String, String)>)>) -> Result<Value, Vec<Diagnostic>> {
    let id = p.row.id.as_str();
    let now = now_ms();
    define::store_defs(tx, &p.replayed.hashed)?;
    if let Some((key, rows)) = proofs {
        let hashes: HashSet<&str> = p.verify.iter().map(|d| d.hash.as_str()).collect();
        define::store_results(tx, key, rows.into_iter().filter(|(h, _)| hashes.contains(h.as_str())))?;
    }
    tx.execute("DELETE FROM change_names WHERE change = ?1", params![id]).map_err(sql)?;
    for (n, r) in &p.meta {
        tx.execute(
            "INSERT INTO change_names (change, name, hash, local_names, refs, origin, base_hash, conflict, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![id, n, r.hash, r.locals, r.refs, r.origin, r.base_hash, r.conflict, now],
        )
        .map_err(sql)?;
    }
    // The change set's own definitions that got a new hash were relinked (their text is the same);
    // a seen-mark of the old hash moves with them (a relink is not a view, review I1).
    let input: BTreeSet<&str> = p.replayed_names.iter().map(String::as_str).collect();
    for (n, r) in p.after.iter().filter(|(n, _)| input.contains(n.as_str())) {
        if let Some(old) = p.before.get(n).filter(|o| o.hash != r.hash) {
            tx.execute("INSERT INTO history (name, action, old_hash, new_hash, agent, at, space) VALUES (?1, 'relinked', ?2, ?3, ?4, ?5, ?6)", params![n, old.hash, r.hash, signer.member, now, id]).map_err(sql)?;
            tx.execute("UPDATE seen SET hash = ?3 WHERE space = ?1 AND name = ?2 AND hash = ?4", params![id, n, r.hash, old.hash]).map_err(sql)?;
        }
    }
    let after_rows = space_rows(tx, &Space::Change(id.to_string()))?;
    let root_after = root_of(&after_rows);
    let base_seq: i64 = tx.query_row("SELECT coalesce(max(seq), 0) FROM records", [], |r| r.get(0)).map_err(sql)?;
    // An owner's write reopens a submitted change set (§6.2) when it changes its root.
    let status = if p.row.status == "submitted" && root_after != p.c_root { "open" } else { p.row.status.as_str() };
    tx.execute(
        "UPDATE changes SET base_seq = ?2, base_root = ?3, needs_rebase = 0, status = ?4, updated_at = ?5 WHERE id = ?1",
        params![id, base_seq, p.main_root, status, now],
    )
    .map_err(sql)?;
    let labels: Vec<Entry> = p.after.iter().filter(|(n, _)| input.contains(n.as_str())).map(|(n, r)| Entry::label(n, p.before.get(n).map(|o| o.hash.as_str()), Some(&r.hash), "relinked")).collect();
    let entries = diff(&p.before, &after_rows, &labels);
    let conflicts: Vec<Value> = p.marks.keys().map(|n| {
        let mark: Value = serde_json::from_str(&p.marks[n]).unwrap_or(Value::Null);
        json!({"name": n, "base": mark["base"], "theirs": mark["theirs"]})
    }).collect();
    let args = json!({"entries": serde_json::to_value(&entries).map_err(|e| one(Diagnostic::store_error(e.to_string())))?, "base_root": p.main_root, "conflicts": conflicts});
    let record = append_in(tx, signer, "rebase", args, &Space::Change(id.to_string()), &p.before, &after_rows)?;
    Ok(json!({"change": id, "base_root": p.main_root, "root": root_after, "status": status, "replayed": p.replayed_names, "conflicts": p.conflicts, "record": record}))
}
