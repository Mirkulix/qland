//! SQLite schema of `.qal/store.db` (spec §6) and connection setup.

use crate::sql;
use qal_core::diag::Diagnostic;
use rusqlite::Connection;
use std::path::Path;
use std::time::Duration;

/// `PRAGMA user_version` of stores this code reads and writes. Version 2 is Phase 1; version 3
/// (Phase 2b) adds the `proofs` table; version 4 (Phase 3a) adds `meta`. Older stores are upgraded
/// in place when they are opened ([`migrate`]: 2 -> 3 -> 4, one transaction).
pub const SCHEMA_VERSION: i64 = 4;

/// Proof results by definition hash and prover key (prover version, Z3 version, resource limit):
/// a definition's hash fixes everything a proof depends on (its callees' contracts are referenced
/// by hash), so results stay valid until the prover changes.
pub const PROOFS: &str = "
CREATE TABLE proofs (
    hash    TEXT NOT NULL REFERENCES defs(hash),
    prover  TEXT NOT NULL,
    results TEXT NOT NULL,
    at      INTEGER NOT NULL,
    PRIMARY KEY (hash, prover)
);
";

/// Key-value facts about the store (`package:<name>` = hash of the last imported package; later
/// phases add `role` and `std_package`).
pub const META: &str = "CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);";

/// How long a writer waits for another process's `define`/`rename` to finish.
pub const BUSY_TIMEOUT: Duration = Duration::from_secs(10);

/// Hashes are lowercase hex (`TEXT`); times are Unix milliseconds; list columns are JSON arrays.
/// A member of a mutually recursive group stores a short canonical form that names its group;
/// the group form itself is stored once in `groups`.
/// `names.local_names` is the local-variable naming of that name's latest text (two names can share
/// a hash but name their locals differently); `defs.local_names` is the one it was first stored with.
/// `names.refs` is the JSON list of names that text refers to (one per referenced hash): users follow
/// the name they were written against, even when other names share its hash.
pub const SCHEMA: &str = "
CREATE TABLE groups (
    hash TEXT PRIMARY KEY,
    form BLOB NOT NULL
);
CREATE TABLE defs (
    hash        TEXT PRIMARY KEY,
    kind        TEXT NOT NULL CHECK (kind IN ('type', 'fn', 'test')),
    canonical   BLOB NOT NULL,
    signature   TEXT NOT NULL,
    effects     TEXT NOT NULL,
    local_names TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);
CREATE TABLE deps (
    hash     TEXT NOT NULL REFERENCES defs(hash),
    dep_hash TEXT NOT NULL REFERENCES defs(hash),
    PRIMARY KEY (hash, dep_hash)
);
CREATE INDEX deps_by_dep ON deps(dep_hash);
CREATE TABLE names (
    name        TEXT PRIMARY KEY,
    hash        TEXT NOT NULL REFERENCES defs(hash),
    local_names TEXT NOT NULL,
    refs        TEXT NOT NULL,
    updated_at  INTEGER NOT NULL
);
CREATE INDEX names_by_hash ON names(hash);
CREATE TABLE history (
    id        INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    action    TEXT NOT NULL CHECK (action IN ('added', 'replaced', 'relinked', 'renamed')),
    old_hash  TEXT,
    new_hash  TEXT NOT NULL,
    from_name TEXT,
    to_name   TEXT,
    agent     TEXT NOT NULL,
    at        INTEGER NOT NULL
);
CREATE INDEX history_by_name ON history(name);
";

/// Per-connection settings: wait for other writers, enforce references.
pub fn configure(conn: &Connection) -> rusqlite::Result<()> {
    conn.busy_timeout(BUSY_TIMEOUT)?;
    conn.pragma_update(None, "foreign_keys", true)?;
    Ok(())
}

/// Creates the tables in an empty database (one transaction) and switches it to WAL mode, so
/// readers (`view`, `query`, `run`) are not blocked by a running `define`. Nothing is changed in a
/// database that is not empty and not a QAL store of this version.
pub fn create(conn: &mut Connection, path: &Path) -> Result<(), Vec<Diagnostic>> {
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate).map_err(sql)?;
    let version: i64 = tx.query_row("PRAGMA user_version", [], |r| r.get(0)).map_err(sql)?;
    let objects: i64 = tx.query_row("SELECT count(*) FROM sqlite_master", [], |r| r.get(0)).map_err(sql)?;
    if version == 0 && objects == 0 {
        tx.execute_batch(SCHEMA).map_err(sql)?;
        tx.execute_batch(PROOFS).map_err(sql)?;
        tx.execute_batch(META).map_err(sql)?;
        tx.pragma_update(None, "user_version", SCHEMA_VERSION).map_err(sql)?;
    } else if (2..SCHEMA_VERSION).contains(&version) {
        upgrade(&tx, version)?;
    } else if version != SCHEMA_VERSION {
        let what = if version == 0 { "not a QAL store".to_string() } else { format!("not a QAL store of version {SCHEMA_VERSION} (found version {version})") };
        return Err(vec![Diagnostic::store_error(format!("{} is {what}", path.display()))]);
    }
    tx.commit().map_err(sql)?;
    conn.pragma_update(None, "journal_mode", "WAL").map_err(sql)?;
    Ok(())
}

/// Upgrades a store of version `from` (2 or 3) to [`SCHEMA_VERSION`] inside `tx`, step by step:
/// 2 -> 3 creates `proofs`, 3 -> 4 creates `meta`.
fn upgrade(tx: &rusqlite::Transaction, from: i64) -> Result<(), Vec<Diagnostic>> {
    if from <= 2 {
        tx.execute_batch(PROOFS).map_err(sql)?;
        tx.pragma_update(None, "user_version", 3).map_err(sql)?;
    }
    if from <= 3 {
        tx.execute_batch(META).map_err(sql)?;
        tx.pragma_update(None, "user_version", 4).map_err(sql)?;
    }
    Ok(())
}

/// Brings a store of an older version (2 or 3) to [`SCHEMA_VERSION`] in one transaction; it only
/// creates tables (2 -> 3 `proofs`, 3 -> 4 `meta`). Other versions are left alone (the caller
/// reports them).
pub fn migrate(conn: &mut Connection) -> Result<(), Vec<Diagnostic>> {
    let version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0)).map_err(sql)?;
    if !(2..SCHEMA_VERSION).contains(&version) {
        return Ok(());
    }
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate).map_err(sql)?;
    // Another process may have migrated it meanwhile.
    let version: i64 = tx.query_row("PRAGMA user_version", [], |r| r.get(0)).map_err(sql)?;
    if (2..SCHEMA_VERSION).contains(&version) {
        upgrade(&tx, version)?;
    }
    tx.commit().map_err(sql)
}
