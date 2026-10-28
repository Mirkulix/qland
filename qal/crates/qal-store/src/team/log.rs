//! The team's bookkeeping of reads (Phase 4 spec §5.1, §10.3, §10.4): the call log, which notices
//! each member has seen, and one-time flags. It lives in its own SQLite file next to the store,
//! `.qal/team-log.db`, so that a read tool call never writes to (and never waits for the write
//! lock of) `store.db`: writers there hold their lock for a whole `define`.
//!
//! What a session viewed (`qal_view`, spec §5.3) is noted here too; what it wrote is noted in
//! `store.db` by the write itself, and a write's stale-base check (E0904) reads both.
//!
//! Notices themselves stay in `store.db` (written by the writes that cause them); a member's
//! seen-mark is a cursor here (the highest notice id it was given). Writes to this file are short
//! and wait at most [`BUSY_TIMEOUT`]; callers treat them as best effort.

use crate::sql;
use qal_core::diag::Diagnostic;
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;
use std::time::Duration;

/// The file next to `store.db`.
pub const FILE: &str = "team-log.db";
/// How long a bookkeeping write waits for another one.
pub const BUSY_TIMEOUT: Duration = Duration::from_secs(2);
/// The call log keeps the newest this many calls.
pub const MAX_CALLS: i64 = 100_000;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS calls (
    id      INTEGER PRIMARY KEY,
    member  TEXT NOT NULL,
    session TEXT NOT NULL,
    tool    TEXT NOT NULL,
    ok      INTEGER NOT NULL,
    code    TEXT,
    ms      INTEGER NOT NULL,
    at      INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS notice_cursor (
    member TEXT PRIMARY KEY,
    seen   INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS seen (
    session TEXT NOT NULL,
    space   TEXT NOT NULL,
    name    TEXT NOT NULL,
    hash    TEXT NOT NULL,
    PRIMARY KEY (session, space, name)
);
CREATE TABLE IF NOT EXISTS flags (
    member TEXT NOT NULL,
    flag   TEXT NOT NULL,
    at     INTEGER NOT NULL,
    PRIMARY KEY (member, flag)
);
";

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

/// The bookkeeping file of a team store.
pub struct TeamLog {
    conn: Connection,
}

impl TeamLog {
    /// Opens (creating if needed) the log next to the store file `store`.
    pub fn open(store: &Path) -> Result<TeamLog, Vec<Diagnostic>> {
        let path = store.with_file_name(FILE);
        let conn = Connection::open(&path).map_err(sql)?;
        conn.busy_timeout(BUSY_TIMEOUT).map_err(sql)?;
        conn.pragma_update(None, "journal_mode", "WAL").map_err(sql)?;
        conn.execute_batch(SCHEMA).map_err(sql)?;
        Ok(TeamLog { conn })
    }

    pub fn log_call(&self, c: &Call) -> Result<(), Vec<Diagnostic>> {
        self.conn
            .execute("INSERT INTO calls (member, session, tool, ok, code, ms, at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)", params![c.member, c.session, c.tool, c.ok, c.code, c.ms, crate::now_ms()])
            .map_err(sql)?;
        let id = self.conn.last_insert_rowid();
        if id % 1000 == 0 {
            self.conn.execute("DELETE FROM calls WHERE id <= ?1", params![id - MAX_CALLS]).map_err(sql)?;
        }
        Ok(())
    }

    /// The highest notice id `member` was given (0: none).
    pub fn cursor(&self, member: &str) -> Result<i64, Vec<Diagnostic>> {
        Ok(self.conn.query_row("SELECT seen FROM notice_cursor WHERE member = ?1", params![member], |r| r.get(0)).optional().map_err(sql)?.unwrap_or(0))
    }

    pub fn set_cursor(&self, member: &str, seen: i64) -> Result<(), Vec<Diagnostic>> {
        self.conn
            .execute("INSERT INTO notice_cursor (member, seen) VALUES (?1, ?2) ON CONFLICT(member) DO UPDATE SET seen = max(seen, excluded.seen)", params![member, seen])
            .map(|_| ())
            .map_err(sql)
    }

    /// Notes that `session` saw `hash` of `name` in `space` (a view).
    pub fn mark_seen(&self, session: &str, space: &str, name: &str, hash: &str) -> Result<(), Vec<Diagnostic>> {
        self.conn
            .execute("INSERT INTO seen (session, space, name, hash) VALUES (?1, ?2, ?3, ?4) ON CONFLICT(session, space, name) DO UPDATE SET hash = excluded.hash", params![session, space, name, hash])
            .map(|_| ())
            .map_err(sql)
    }

    /// The hash of `name` in `space` that `session` last viewed.
    pub fn seen(&self, session: &str, space: &str, name: &str) -> Result<Option<String>, Vec<Diagnostic>> {
        self.conn.query_row("SELECT hash FROM seen WHERE session = ?1 AND space = ?2 AND name = ?3", params![session, space, name], |r| r.get(0)).optional().map_err(sql)
    }

    /// Sets `flag` for `member`; true if it was not set before.
    pub fn flag_once(&self, member: &str, flag: &str) -> Result<bool, Vec<Diagnostic>> {
        let n = self.conn.execute("INSERT OR IGNORE INTO flags (member, flag, at) VALUES (?1, ?2, ?3)", params![member, flag, crate::now_ms()]).map_err(sql)?;
        Ok(n > 0)
    }

    /// `flag` is set for `member`.
    pub fn has_flag(&self, member: &str, flag: &str) -> Result<bool, Vec<Diagnostic>> {
        Ok(self.conn.query_row("SELECT 1 FROM flags WHERE member = ?1 AND flag = ?2", params![member, flag], |_| Ok(())).optional().map_err(sql)?.is_some())
    }

    /// All logged calls: (member, tool, ok, code, ms).
    pub fn calls(&self) -> Result<Vec<(String, String, bool, Option<String>, i64)>, Vec<Diagnostic>> {
        let mut stmt = self.conn.prepare("SELECT member, tool, ok, code, ms FROM calls ORDER BY id").map_err(sql)?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))).map_err(sql)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(sql)
    }
}
