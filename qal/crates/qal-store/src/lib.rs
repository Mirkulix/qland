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
mod pattern;
mod schema;
mod snapshot;

pub use ops::{DepView, Entry, HistoryEntry, Query, Renamed, View};
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

pub struct Store {
    conn: Connection,
    path: PathBuf,
}

/// The checked codebase and the (name, hash) of the definitions to verify.
pub type VerifyFn<'a> = dyn FnMut(&TProgram, &[(String, String)]) -> Result<Vec<(String, String)>, Vec<Diagnostic>> + 'a;

/// A verification step of `define` (the contract prover): `run` gets the checked codebase including
/// the new definitions and the functions and tests that are new, got a new hash (given or relinked),
/// or have no result for `key` yet; it returns their results as (hash, JSON) for the `proofs` table,
/// or diagnostics that reject the whole `define`.
pub struct Verify<'a> {
    /// Identifies the verifier and its settings; results are stored per key.
    pub key: String,
    pub run: &'a mut VerifyFn<'a>,
}

pub(crate) fn sql(e: rusqlite::Error) -> Vec<Diagnostic> {
    vec![Diagnostic::store_error(format!("store database error: {e}"))]
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
        Ok(Store { conn, path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The whole named codebase as a program (checked when it was stored).
    pub fn program(&mut self) -> Result<Program, Vec<Diagnostic>> {
        let tx = self.conn.transaction().map_err(sql)?;
        let snap = Snapshot::load(&tx)?;
        snap.program(&tx, &|_| false)
    }

    /// Adds or replaces the definitions in `src` (spec §6). Either everything is stored and the
    /// result lists every name that changed or was given, or nothing is stored and the diagnostics
    /// say why. `agent` is recorded in the history.
    pub fn define(&mut self, src: &str, agent: &str) -> Result<Vec<Defined>, Vec<Diagnostic>> {
        define::define(&mut self.conn, src, agent, None)
    }

    /// Like `define`, with a verification step before anything is committed (see [`Verify`]).
    pub fn define_verified(&mut self, src: &str, agent: &str, verify: &mut Verify) -> Result<Vec<Defined>, Vec<Diagnostic>> {
        define::define(&mut self.conn, src, agent, Some(verify))
    }

    /// Renames a definition. No hash changes; the history of `old` moves to `new`.
    pub fn rename(&mut self, old: &str, new: &str, agent: &str) -> Result<Renamed, Vec<Diagnostic>> {
        ops::rename(&mut self.conn, old, new, agent)
    }

    /// Text, signature and direct dependencies of each name (or `#<hash>`).
    pub fn view(&mut self, names: &[String]) -> Result<Vec<View>, Vec<Diagnostic>> {
        ops::view(&mut self.conn, names)
    }

    /// Named definitions matching every given filter, sorted by name.
    pub fn query(&mut self, q: &Query) -> Result<Vec<Entry>, Vec<Diagnostic>> {
        ops::query(&mut self.conn, q)
    }

    /// The current name of a definition that was renamed from `old` (None if it was not, or if
    /// the definition no longer has a name). Based on the history: it still answers when `old`
    /// has since been reused for a different definition, so callers use it only for names that
    /// do not exist.
    pub fn renamed_to(&mut self, old: &str) -> Result<Option<String>, Vec<Diagnostic>> {
        let conn = &self.conn;
        let current = |n: &str| conn.query_row("SELECT 1 FROM names WHERE name = ?1", [n], |_| Ok(())).is_ok();
        ops::renamed_to(conn, &current, old)
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

    /// Every recorded change of `name`, oldest first.
    pub fn history(&mut self, name: &str) -> Result<Vec<HistoryEntry>, Vec<Diagnostic>> {
        ops::history(&mut self.conn, name)
    }
}

#[cfg(test)]
mod tests;
