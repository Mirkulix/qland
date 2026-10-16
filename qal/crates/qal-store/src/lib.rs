//! QAL codebase store (spec §5, §6): content-addressed, checked definitions in `.qal/store.db`.
//!
//! The store keeps definitions by hash (`defs`), what they refer to (`deps`), the readable names
//! (`names`, name -> hash) and every change of a name (`history`). Invariants:
//! 1. A definition is stored only if the whole codebase including it parses and checks.
//! 2. Every dependency of a stored definition is stored; every dependency of a named definition is named.
//! 3. `define` and `rename` are atomic: all their changes are written in one transaction, or none.
//!
//! Every operation reports problems as diagnostics (store failures are `store_error`).

mod define;
mod ops;
pub mod package;
mod pattern;
mod schema;
mod snapshot;
pub mod space;
pub mod team;

pub use define::{last_lock_held, Release, StdNames};
pub use space::Space;
pub use ops::{Deleted, DepView, Entry, HistoryEntry, Query, Renamed, View};
pub use package::{Exported, Imported};
pub use pattern::matches as signature_matches;

use qal_core::ast::Program;
use qal_core::tast::TProgram;
use qal_core::diag::Diagnostic;
use rusqlite::Connection;
use serde::Serialize;
use snapshot::Snapshot;
use std::path::{Path, PathBuf};

/// The store directory and file inside a project.
pub const STORE_DIR: &str = ".qal";
pub const STORE_FILE: &str = "store.db";

/// The result of `define` for one name.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Defined {
    pub name: String,
    /// `type`, `fn` or `test`.
    pub kind: &'static str,
    pub hash: String,
    /// `added` (new name), `replaced` (new version), `unchanged` (same hash), or `relinked`
    /// (a stored dependent that now refers to the new versions).
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub old_hash: Option<String>,
}

/// The project directory around `start`: the nearest directory (`start` or an ancestor) that
/// contains `.qal/store.db`.
pub fn project_of(start: &Path) -> Option<PathBuf> {
    start.ancestors().find(|d| d.join(STORE_DIR).join(STORE_FILE).is_file()).map(Path::to_path_buf)
}

pub struct Store {
    conn: Connection,
    path: PathBuf,
    /// The released standard library (see [`Store::set_release`]).
    release: Option<define::Release>,
    /// The std store only: changing or deleting released names is allowed.
    breaking: bool,
    /// Team mode: who signs the records of writes (see [`team`]).
    signer: Option<std::sync::Arc<team::Signer>>,
    /// The space every operation works in (`main` unless a change set is chosen).
    space: Space,
    /// What the last team write reported besides its result (leases taken, stubs).
    last: team::WriteReport,
}

/// The checked codebase and the (name, hash) of the definitions to verify.
pub type VerifyFn<'a> = dyn FnMut(&TProgram, &[(String, String)]) -> Result<Vec<(String, String)>, Vec<Diagnostic>> + 'a;

/// A verification step of `define` (the contract prover): `run` gets the checked codebase including
/// the new definitions and the functions and tests that are new, got a new hash (given or relinked),
/// or have no result for `key` yet; it returns their results as (hash, JSON) for the `proofs` table,
/// or diagnostics that reject the whole `define`. It runs on a snapshot of the store without holding
/// the write lock, so it may be called again (rarely) when another writer changed what has to be
/// verified in the meantime.
pub struct Verify<'a> {
    /// Identifies the verifier and its settings; results are stored per key.
    pub key: String,
    pub run: &'a mut VerifyFn<'a>,
}

pub(crate) fn sql(e: rusqlite::Error) -> Vec<Diagnostic> {
    let busy = matches!(e.sqlite_error_code(), Some(rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked));
    let mut d = Diagnostic::store_error(format!("store database error: {e}"));
    if busy {
        // Another writer held the lock past the busy timeout (team mode reports it as E0907).
        d.detail = Some(serde_json::json!({"sqlite": "busy"}));
    }
    vec![d]
}

/// True for the store error of a write lock held past the busy timeout (see [`sql`]).
pub fn is_busy(d: &Diagnostic) -> bool {
    d.kind == "store_error" && d.detail.as_ref().and_then(|v| v.get("sqlite")).and_then(|v| v.as_str()) == Some("busy")
}

pub(crate) fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

impl Store {
    /// Creates `<project>/.qal/store.db` if it does not exist yet, and opens it.
    pub fn init(project: &Path) -> Result<Store, Vec<Diagnostic>> {
        let dir = project.join(STORE_DIR);
        std::fs::create_dir_all(&dir).map_err(|e| vec![Diagnostic::store_error(format!("cannot create {}: {e}", dir.display()))])?;
        let path = dir.join(STORE_FILE);
        let mut conn = Connection::open(&path).map_err(sql)?;
        schema::configure(&conn).map_err(sql)?;
        schema::create(&mut conn, &path)?;
        Store::checked(conn, path)
    }

    /// Opens an existing store file.
    pub fn open(path: &Path) -> Result<Store, Vec<Diagnostic>> {
        if !path.is_file() {
            return Err(vec![Diagnostic::store_error(format!("no QAL store at {}", path.display())).hint("create one with `qal init`")]);
        }
        let conn = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE).map_err(sql)?;
        schema::configure(&conn).map_err(sql)?;
        Store::checked(conn, path.to_path_buf())
    }

    /// Opens the store of the project containing `start`: the nearest `.qal/store.db` in `start` or
    /// one of its ancestors.
    pub fn discover(start: &Path) -> Result<Store, Vec<Diagnostic>> {
        for dir in start.ancestors() {
            let candidate = dir.join(STORE_DIR).join(STORE_FILE);
            if candidate.is_file() {
                return Store::open(&candidate);
            }
        }
        Err(vec![Diagnostic::store_error(format!("no QAL store in {} or its parents", start.display())).hint("create one with `qal init`")])
    }

    fn checked(mut conn: Connection, path: PathBuf) -> Result<Store, Vec<Diagnostic>> {
        schema::migrate(&mut conn)?;
        let version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0)).map_err(sql)?;
        if version != schema::SCHEMA_VERSION {
            return Err(vec![Diagnostic::store_error(format!(
                "{} is not a QAL store of version {} (found version {version})",
                path.display(),
                schema::SCHEMA_VERSION
            ))]);
        }
        Ok(Store { conn, path, release: None, breaking: false, signer: None, space: Space::Main, last: team::WriteReport::default() })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Every following operation works in `space` (a change set must exist; checked by each operation).
    pub fn set_space(&mut self, space: Space) {
        self.space = space;
    }

    pub fn space(&self) -> &Space {
        &self.space
    }

    /// The leases the last write took and the stubs it defined (team mode).
    pub fn last_write(&self) -> &team::WriteReport {
        &self.last
    }

    /// The whole named codebase as a program (checked when it was stored).
    pub fn program(&mut self) -> Result<Program, Vec<Diagnostic>> {
        let tx = self.conn.transaction().map_err(sql)?;
        let snap = Snapshot::load(&tx, &self.space)?;
        snap.program(&tx, &|_| false)
    }

    /// The whole named codebase and the hash of every name, read in one transaction: the hashes
    /// are those of exactly these definitions, whatever another writer does meanwhile.
    pub fn program_with_hashes(&mut self) -> Result<(Program, std::collections::HashMap<String, String>), Vec<Diagnostic>> {
        let tx = self.conn.transaction().map_err(sql)?;
        let snap = Snapshot::load(&tx, &self.space)?;
        let hashes = snap.names.iter().map(|(n, e)| (n.clone(), e.hash.to_hex())).collect();
        Ok((snap.program(&tx, &|_| false)?, hashes))
    }

    /// Adds or replaces the definitions in `src` (spec §6). Either everything is stored and the
    /// result lists every name that changed or was given, or nothing is stored and the diagnostics
    /// say why. `agent` is recorded in the history.
    /// In a team store, `agent` is ignored: the history names the signer's member.
    pub fn define(&mut self, src: &str, agent: &str) -> Result<Vec<Defined>, Vec<Diagnostic>> {
        self.define_with(src, agent, None)
    }

    /// Like `define`, with a verification step before anything is committed (see [`Verify`]).
    pub fn define_verified(&mut self, src: &str, agent: &str, verify: &mut Verify) -> Result<Vec<Defined>, Vec<Diagnostic>> {
        self.define_with(src, agent, Some(verify))
    }

    fn define_with(&mut self, src: &str, agent: &str, verify: Option<&mut Verify>) -> Result<Vec<Defined>, Vec<Diagnostic>> {
        let signer = self.signer.as_deref();
        let team = team::require_signer(&self.conn, signer)?;
        let agent = signer.map_or(agent, |s| s.member.as_str());
        let rules = define::Rules { std: StdNames::Refuse, release: self.release.as_ref(), breaking: self.breaking };
        let space = self.space.clone();
        let report = std::cell::RefCell::new(team::WriteReport::default());
        // What the session viewed is in the team log (reads never write store.db).
        let log = if team { self.team_log().ok() } else { None };
        let guard = |tx: &rusqlite::Transaction, snap: &Snapshot, input: &Program, typed: &TProgram, dry: bool| -> Result<(), Vec<Diagnostic>> {
            let names: std::collections::BTreeSet<&str> = input.defs.iter().map(|d| d.name()).collect();
            let stubs: Vec<String> = typed.stubs().into_iter().filter(|n| names.contains(n.as_str())).collect();
            let leases = if team { team::guard_define(tx, log.as_ref(), signer, &space, snap, input, dry)? } else { Vec::new() };
            *report.borrow_mut() = team::WriteReport { leases, stubs };
            Ok(())
        };
        let record = |out: &[Defined], tx: &rusqlite::Transaction, before: &team::Rows| team::record_write(tx, signer, "define", &space, before, &team::Entry::of_defined(out), serde_json::Map::new());
        let out = define::define_checked(&mut self.conn, src, agent, &rules, &self.space, verify, &guard, &record);
        self.last = report.into_inner();
        out
    }

    /// Renames a definition. No hash changes; the history of `old` moves to `new`.
    pub fn rename(&mut self, old: &str, new: &str, agent: &str) -> Result<Renamed, Vec<Diagnostic>> {
        let signer = self.signer.as_deref();
        team::require_signer(&self.conn, signer)?;
        let agent = signer.map_or(agent, |s| s.member.as_str());
        let rules = define::Rules { std: StdNames::Refuse, release: self.release.as_ref(), breaking: self.breaking };
        let space = self.space.clone();
        let log = if team::is_team(&self.conn)? { self.team_log().ok() } else { None };
        let record = |tx: &rusqlite::Transaction, labels: &[team::Entry], before: &team::Rows| {
            team::guard_names(tx, log.as_ref(), signer, &space, "qal_rename", labels)?;
            team::record_write(tx, signer, "rename", &space, before, labels, serde_json::Map::new())
        };
        ops::rename(&mut self.conn, &self.space, old, new, agent, &rules, &record)
    }

    /// Removes names nothing else refers to (see [`ops::delete`]); their history stays.
    pub fn delete(&mut self, names: &[String], agent: &str) -> Result<Vec<Deleted>, Vec<Diagnostic>> {
        let signer = self.signer.as_deref();
        team::require_signer(&self.conn, signer)?;
        let agent = signer.map_or(agent, |s| s.member.as_str());
        let rules = define::Rules { std: StdNames::Refuse, release: self.release.as_ref(), breaking: self.breaking };
        let space = self.space.clone();
        let log = if team::is_team(&self.conn)? { self.team_log().ok() } else { None };
        let record = |tx: &rusqlite::Transaction, labels: &[team::Entry], before: &team::Rows| {
            team::guard_names(tx, log.as_ref(), signer, &space, "qal_delete", labels)?;
            team::record_write(tx, signer, "delete", &space, before, labels, serde_json::Map::new())
        };
        ops::delete(&mut self.conn, &self.space, names, agent, &rules, &record)
    }

    /// Text, signature and direct dependencies of each name (or `#<hash>`).
    pub fn view(&mut self, names: &[String]) -> Result<Vec<View>, Vec<Diagnostic>> {
        ops::view(&mut self.conn, &self.space, names)
    }

    /// Named definitions matching every given filter, sorted by name.
    pub fn query(&mut self, q: &Query) -> Result<Vec<Entry>, Vec<Diagnostic>> {
        ops::query(&mut self.conn, &self.space, q)
    }

    /// The current name of a definition that was renamed from `old` (None if it was not, or if
    /// the definition no longer has a name). Based on the history: it still answers when `old`
    /// has since been reused for a different definition, so callers use it only for names that
    /// do not exist.
    pub fn renamed_to(&mut self, old: &str) -> Result<Option<String>, Vec<Diagnostic>> {
        let conn = &self.conn;
        let space = &self.space;
        let current = |n: &str| space::hash_of(conn, space, n).ok().flatten().is_some();
        ops::renamed_to(conn, space, &current, old)
    }

    /// Stored proof results (JSON) of definition `hash` for prover key `prover`.
    pub fn proofs(&self, hash: &str, prover: &str) -> Result<Option<String>, Vec<Diagnostic>> {
        use rusqlite::OptionalExtension;
        self.conn
            .query_row("SELECT results FROM proofs WHERE hash = ?1 AND prover = ?2", rusqlite::params![hash, prover], |r| r.get(0))
            .optional()
            .map_err(sql)
    }

    /// Records proof results (JSON) of the stored definition `hash`; replaces earlier ones for `prover`.
    pub fn save_proofs(&mut self, hash: &str, prover: &str, results: &str) -> Result<(), Vec<Diagnostic>> {
        self.conn
            .execute(
                "INSERT INTO proofs (hash, prover, results, at) VALUES (?1, ?2, ?3, ?4) \
                 ON CONFLICT(hash, prover) DO UPDATE SET results = excluded.results, at = excluded.at",
                rusqlite::params![hash, prover, results, now_ms()],
            )
            .map(|_| ())
            .map_err(sql)
    }

    /// [`Store::save_proofs`] as a cache fill that never waits: if another writer holds the lock,
    /// nothing is stored (the results are computed again next time). Returns whether it stored.
    pub fn try_save_proofs(&mut self, hash: &str, prover: &str, results: &str) -> bool {
        if self.conn.busy_timeout(std::time::Duration::ZERO).is_err() {
            return false;
        }
        let saved = self.save_proofs(hash, prover, results).is_ok();
        let _ = schema::configure(&self.conn);
        saved
    }

    /// Exports the names starting with `prefix` and their dependencies as package `package`
    /// (the file text is in [`Exported::file`]).
    pub fn export(&mut self, prefix: &str, package: &str) -> Result<Exported, Vec<Diagnostic>> {
        package::export(&mut self.conn, prefix, package)
    }

    /// Imports a package file atomically, verifying its checksum and every hash.
    pub fn import(&mut self, file: &str) -> Result<Imported, Vec<Diagnostic>> {
        self.import_with(file, None)
    }

    /// [`Store::import`] with the verification step of [`Store::define_verified`].
    pub fn import_verified(&mut self, file: &str, verify: &mut Verify) -> Result<Imported, Vec<Diagnostic>> {
        self.import_with(file, Some(verify))
    }

    fn import_with(&mut self, file: &str, verify: Option<&mut Verify>) -> Result<Imported, Vec<Diagnostic>> {
        let signer = self.signer.as_deref();
        let team = team::require_signer(&self.conn, signer)?;
        let pkg = package::parse(file)?;
        // A team store imports only the standard library (`qal team sync-std`), and its record
        // says which package it was.
        if team && pkg.name != package::STD_PACKAGE_NAME {
            let mut d = Diagnostic::new("E0903", "role_forbidden", "", "", format!("a team store imports only the standard library, not package `{}`", pkg.name));
            d.expected = Some("architect".into());
            return Err(vec![d.hint("agents write the code over MCP; the owner imports std with `qal team sync-std`")]);
        }
        let mut extra = serde_json::Map::new();
        extra.insert("package".into(), serde_json::Value::from(pkg.name.clone()));
        extra.insert("hash".into(), serde_json::Value::from(pkg.hash.clone()));
        let record = |out: &[Defined], tx: &rusqlite::Transaction, before: &team::Rows| team::record_write(tx, signer, "import", &Space::Main, before, &team::Entry::of_defined(out), extra.clone());
        package::import(&mut self.conn, file, verify, self.release.as_ref(), &record)
    }

    /// Sets the released standard library: the std package embedded in the binary (`file`). Only
    /// this package may be imported as `std` into a project; in the std store its names keep their
    /// signatures and cannot be deleted (E0244) unless [`Store::set_breaking`] allows it, and a
    /// rebuild from it keeps its recorded agents.
    pub fn set_release(&mut self, file: &str) -> Result<(), Vec<Diagnostic>> {
        self.release = Some(define::Release::of(file)?);
        Ok(())
    }

    /// Allows changing the signature of, or deleting, released std names in the std store.
    pub fn set_breaking(&mut self, breaking: bool) {
        self.breaking = breaking;
    }

    /// Makes the store at `<project>/.qal/store.db` (created if missing) the std project's store
    /// (`meta.role = 'std'`), where only `std.` names may be defined and no std package is imported
    /// automatically. Refused for a store that already holds definitions or imported a std package
    /// as an ordinary project.
    pub fn init_std(project: &Path) -> Result<Store, Vec<Diagnostic>> {
        let mut store = Store::init(project)?;
        let tx = store.conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate).map_err(sql)?;
        if !define::is_std_store(&tx)? {
            let names: i64 = tx.query_row("SELECT count(*) FROM names", [], |r| r.get(0)).map_err(sql)?;
            let imported = package::meta(&tx, package::META_STD_PACKAGE)?.is_some();
            if names > 0 || imported {
                return Err(vec![Diagnostic::store_error(format!("{} is the store of an ordinary project", store.path.display()))
                    .hint("the std project needs its own store: run `qal init --std` in an empty directory (qal/std)")]);
            }
            package::set_meta(&tx, package::META_ROLE, define::ROLE_STD)?;
        }
        tx.commit().map_err(sql)?;
        Ok(store)
    }

    /// True for the std project's store (`meta.role = 'std'`).
    pub fn is_std(&self) -> Result<bool, Vec<Diagnostic>> {
        define::is_std_store(&self.conn)
    }

    /// Imports the std package `file` (and makes it the release, see [`Store::set_release`]) unless the store already has it (`meta.std_package` is its
    /// hash) or is the std project's store; returns the import, if one happened. Called before
    /// every write of a project (Phase 3 §4.3), never by reads.
    pub fn ensure_std(&mut self, file: &str, verify: Option<&mut Verify>) -> Result<Option<Imported>, Vec<Diagnostic>> {
        // The package this is called with is the release (the one embedded in the binary).
        let release = define::Release::of(file)?;
        // A team store imports std only by a signed `qal team sync-std` (Phase 4 spec §4.1).
        if self.is_std()? || team::is_team(&self.conn)? {
            self.release = Some(release);
            return Ok(None);
        }
        if self.meta(package::META_STD_PACKAGE)?.as_deref() == Some(release.hash.as_str()) {
            self.release = Some(release);
            return Ok(None);
        }
        self.release = Some(release);
        package::import(&mut self.conn, file, verify, self.release.as_ref(), &|_, _, _| Ok(())).map(Some)
    }

    /// A value of the store's `meta` table.
    pub fn meta(&self, key: &str) -> Result<Option<String>, Vec<Diagnostic>> {
        package::meta(&self.conn, key)
    }

    /// Every recorded change of `name`, oldest first.
    pub fn history(&mut self, name: &str) -> Result<Vec<HistoryEntry>, Vec<Diagnostic>> {
        ops::history(&mut self.conn, &self.space, name)
    }
}

#[cfg(test)]
mod tests;
