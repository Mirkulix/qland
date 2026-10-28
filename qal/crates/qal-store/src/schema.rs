//! SQLite schema of `.qal/store.db` (spec §6) and connection setup.

use crate::sql;
use qal_core::diag::Diagnostic;
use rusqlite::Connection;
use std::path::Path;
use std::time::Duration;

/// `PRAGMA user_version` of stores this code reads and writes. Version 2 is Phase 1; version 3
/// (Phase 2b) adds the `proofs` table; version 4 (Phase 3a) adds `meta`; version 5 (Phase 3b) allows
/// the history action `deleted`; version 6 (Phase 4a) adds the team tables and `history.space` /
/// `history.record`; version 7 (Phase 6b) adds `attempts` and `corrections`, the memory of fixes;
/// version 8 adds `observed`, the runs whose output a define compares against.
/// Older stores are upgraded in place when they are opened
/// ([`migrate`]: 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8, one transaction).
pub const SCHEMA_VERSION: i64 = 8;

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

/// Key-value facts about the store: `package:<name>` = hash of the last imported package,
/// `std_package` = hash of the last imported std package (Phase 3b), `role` = `std` for the std
/// project's store (absent for every other project).
pub const META: &str = "CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);";

/// 4 -> 5: `history` again with the action `deleted` allowed (SQLite cannot change a CHECK
/// constraint in place); ids and rows are kept.
const HISTORY_V5: &str = "
CREATE TABLE history_v5 (
    id        INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    action    TEXT NOT NULL CHECK (action IN ('added', 'replaced', 'relinked', 'renamed', 'deleted')),
    old_hash  TEXT,
    new_hash  TEXT NOT NULL,
    from_name TEXT,
    to_name   TEXT,
    agent     TEXT NOT NULL,
    at        INTEGER NOT NULL
);
INSERT INTO history_v5 (id, name, action, old_hash, new_hash, from_name, to_name, agent, at)
    SELECT id, name, action, old_hash, new_hash, from_name, to_name, agent, at FROM history;
DROP TABLE history;
ALTER TABLE history_v5 RENAME TO history;
CREATE INDEX history_by_name ON history(name);
";

/// 5 -> 6 (Phase 4 spec §12): the team tables, created empty (a store becomes a team store only with
/// `qal team init`, `meta.team = 1`). Nothing else changes, so a store without team mode works
/// exactly as before. `IF NOT EXISTS`: a store downgraded by hand may still have them.
pub const TEAM: &str = "
CREATE TABLE IF NOT EXISTS members (
    id         TEXT PRIMARY KEY,
    role       TEXT NOT NULL CHECK (role IN ('owner', 'architect', 'implementer', 'verifier', 'reviewer')),
    key        TEXT NOT NULL,
    key_id     TEXT NOT NULL,
    added_at   INTEGER NOT NULL,
    revoked_at INTEGER
);
CREATE TABLE IF NOT EXISTS sessions (
    token_sha256   TEXT PRIMARY KEY,
    id             TEXT NOT NULL UNIQUE,
    member         TEXT NOT NULL REFERENCES members(id),
    created_at     INTEGER NOT NULL,
    expires_at     INTEGER NOT NULL,
    revoked_at     INTEGER,
    current_change TEXT
);
CREATE TABLE IF NOT EXISTS changes (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    description  TEXT NOT NULL,
    scope        TEXT NOT NULL,
    owner        TEXT NOT NULL,
    assignee     TEXT,
    status       TEXT NOT NULL CHECK (status IN ('open', 'submitted', 'merged', 'abandoned')),
    claimed      INTEGER NOT NULL DEFAULT 0,
    base_seq     INTEGER NOT NULL,
    base_root    TEXT NOT NULL,
    needs_rebase INTEGER NOT NULL DEFAULT 0,
    created_at   INTEGER NOT NULL,
    updated_at   INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS change_names (
    change      TEXT NOT NULL,
    name        TEXT NOT NULL,
    hash        TEXT,
    local_names TEXT NOT NULL,
    refs        TEXT NOT NULL,
    origin      TEXT NOT NULL CHECK (origin IN ('base', 'edited', 'added', 'relinked', 'renamed')),
    base_hash   TEXT,
    conflict    TEXT,
    updated_at  INTEGER NOT NULL,
    PRIMARY KEY (change, name)
);
CREATE TABLE IF NOT EXISTS reviews (
    id       INTEGER PRIMARY KEY,
    change   TEXT NOT NULL,
    member   TEXT NOT NULL,
    verdict  TEXT NOT NULL,
    root     TEXT NOT NULL,
    comments TEXT NOT NULL,
    at       INTEGER NOT NULL,
    record   INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS leases (
    name        TEXT PRIMARY KEY,
    change      TEXT NOT NULL,
    member      TEXT NOT NULL,
    acquired_at INTEGER NOT NULL,
    expires_at  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS seen (
    session TEXT NOT NULL,
    space   TEXT NOT NULL,
    name    TEXT NOT NULL,
    hash    TEXT NOT NULL,
    PRIMARY KEY (session, space, name)
);
CREATE TABLE IF NOT EXISTS notices (
    id     INTEGER PRIMARY KEY,
    member TEXT NOT NULL,
    kind   TEXT NOT NULL,
    body   TEXT NOT NULL,
    at     INTEGER NOT NULL
);
-- Which notices a member has seen, and the call log, are in `.qal/team-log.db` (team::log): reads
-- never write this file.
CREATE INDEX IF NOT EXISTS notices_by_member ON notices(member, id);
CREATE TABLE IF NOT EXISTS records (
    seq  INTEGER PRIMARY KEY,
    hash TEXT NOT NULL UNIQUE,
    body BLOB NOT NULL,
    sig  TEXT NOT NULL
);
";

/// The 5 -> 6 step: the team tables plus `history.space` and `history.record` (added only if missing).
fn upgrade_v6(tx: &rusqlite::Transaction) -> Result<(), Vec<Diagnostic>> {
    tx.execute_batch(TEAM).map_err(sql)?;
    let mut columns = Vec::new();
    {
        let mut stmt = tx.prepare("SELECT name FROM pragma_table_info('history')").map_err(sql)?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0)).map_err(sql)?;
        for r in rows {
            columns.push(r.map_err(sql)?);
        }
    }
    if !columns.iter().any(|c| c == "space") {
        tx.execute_batch("ALTER TABLE history ADD COLUMN space TEXT NOT NULL DEFAULT 'main';").map_err(sql)?;
    }
    if !columns.iter().any(|c| c == "record") {
        tx.execute_batch("ALTER TABLE history ADD COLUMN record INTEGER NULL;").map_err(sql)?;
    }
    // The rows a team write links to its record (`record IS NULL AND id > ...`) without a scan.
    tx.execute_batch("CREATE INDEX IF NOT EXISTS history_unrecorded ON history(id) WHERE record IS NULL;").map_err(sql)?;
    Ok(())
}

/// 6 -> 7 (Phase 6b): the memory of corrections, created empty. `attempts` holds the failed
/// definitions of a name until its next successful `define` (bounded, see `memory`); `corrections`
/// holds the pairs (failed text, fixed text) that were made of them, with the `shape` of the
/// diagnostic's message (backticked names replaced by `_`, what hints and `recall` match on) and the
/// `content` hash of (code, message, before, after, agent); the seed corpus is unique by content, so
/// importing it twice stores it once, while real corrections keep counting. `IF NOT EXISTS`: a store
/// downgraded by hand may still have them.
pub const MEMORY: &str = "
CREATE TABLE IF NOT EXISTS attempts (
    id      INTEGER PRIMARY KEY,
    name    TEXT NOT NULL,
    agent   TEXT NOT NULL,
    code    TEXT NOT NULL,
    path    TEXT NOT NULL,
    message TEXT NOT NULL,
    text    TEXT NOT NULL,
    at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS attempts_by_name ON attempts(name, agent);
CREATE TABLE IF NOT EXISTS corrections (
    id         INTEGER PRIMARY KEY,
    code       TEXT NOT NULL,
    path       TEXT NOT NULL,
    message    TEXT NOT NULL,
    shape      TEXT NOT NULL,
    before     TEXT NOT NULL,
    after      TEXT NOT NULL,
    after_hash TEXT NOT NULL,
    agent      TEXT NOT NULL,
    at         INTEGER NOT NULL,
    content    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS corrections_by_code ON corrections(code, shape);
CREATE UNIQUE INDEX IF NOT EXISTS corrections_corpus ON corrections(content) WHERE agent = 'corpus';
";

/// 7 -> 8: the behaviour memory. Every run of an entry point an agent makes (`qal_run`, `cases`)
/// that needs no effect but `io` is kept with its output, once per entry, arguments and input (the
/// latest output wins); a `define` runs them again on the new code and reports the ones whose
/// output changed (E0807). Pressure run 11 stage 12: a change request rewrote the tests that
/// pinned the tier discount along with the code, and only runs of stage 2 still showed the rule.
pub const OBSERVED: &str = "
CREATE TABLE IF NOT EXISTS observed (
    id        INTEGER PRIMARY KEY,
    entry     TEXT NOT NULL,
    args      TEXT NOT NULL,
    stdin     TEXT NOT NULL,
    stdout    TEXT NOT NULL,
    exit_code INTEGER NOT NULL,
    at        INTEGER NOT NULL,
    UNIQUE (entry, args, stdin)
);
";

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
    action    TEXT NOT NULL CHECK (action IN ('added', 'replaced', 'relinked', 'renamed', 'deleted')),
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
        upgrade_v6(&tx)?;
        tx.execute_batch(MEMORY).map_err(sql)?;
        tx.execute_batch(OBSERVED).map_err(sql)?;
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

/// Upgrades a store of version `from` (2 to 6) to [`SCHEMA_VERSION`] inside `tx`, step by step:
/// 2 -> 3 creates `proofs`, 3 -> 4 creates `meta`, 4 -> 5 rebuilds `history`, 5 -> 6 adds the team
/// tables and the history columns `space` and `record`, 6 -> 7 the memory tables, 7 -> 8 `observed`.
fn upgrade(tx: &rusqlite::Transaction, from: i64) -> Result<(), Vec<Diagnostic>> {
    if from <= 2 {
        tx.execute_batch(PROOFS).map_err(sql)?;
        tx.pragma_update(None, "user_version", 3).map_err(sql)?;
    }
    if from <= 3 {
        tx.execute_batch(META).map_err(sql)?;
        tx.pragma_update(None, "user_version", 4).map_err(sql)?;
    }
    if from <= 4 {
        tx.execute_batch(HISTORY_V5).map_err(sql)?;
        tx.pragma_update(None, "user_version", 5).map_err(sql)?;
    }
    if from <= 5 {
        upgrade_v6(tx)?;
        tx.pragma_update(None, "user_version", 6).map_err(sql)?;
    }
    if from <= 6 {
        tx.execute_batch(MEMORY).map_err(sql)?;
        tx.pragma_update(None, "user_version", 7).map_err(sql)?;
    }
    if from <= 7 {
        tx.execute_batch(OBSERVED).map_err(sql)?;
        tx.pragma_update(None, "user_version", 8).map_err(sql)?;
    }
    Ok(())
}

/// Brings a store of an older version (2 to 6) to [`SCHEMA_VERSION`] in one transaction (2 -> 3
/// creates `proofs`, 3 -> 4 `meta`, 4 -> 5 rebuilds `history` with its rows, 5 -> 6 the team tables,
/// 6 -> 7 the memory tables).
/// Other versions are left alone (the caller reports them).
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
