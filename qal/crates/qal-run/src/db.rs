//! The `db` effect: SQLite through the host (Phase 3 spec §3.2, §5.2).
//!
//! The guest never chooses a file: the host opens `<project>/.qal/db/<name>.sqlite` (or an
//! in-memory database) and runs only the SQL literals of the program, with positional parameters,
//! one statement at a time, under an authorizer that refuses attaching, pragmas, transaction
//! control, extension functions and schema-table writes. The host owns the one transaction per
//! execution (`BEGIN IMMEDIATE` before the entry point, commit or rollback after it).

use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
use rusqlite::types::ValueRef;
use rusqlite::{Connection, ErrorCode, OpenFlags};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Where the database of an execution lives.
#[derive(Debug, Clone, PartialEq)]
pub enum DbGrant {
    /// `db`: `<project>/.qal/db/main.sqlite`
    Main,
    /// `db:<name>`: `<project>/.qal/db/<name>.sqlite`
    Named(String),
    /// `db:mem`: a fresh in-memory database
    Memory,
}

impl DbGrant {
    /// Parses the part after `db` in a grant: ``, `:mem` or `:<name>`.
    pub fn parse(item: &str) -> Result<DbGrant, String> {
        match item.strip_prefix("db") {
            Some("") => Ok(DbGrant::Main),
            Some(":mem") => Ok(DbGrant::Memory),
            Some(rest) => {
                let name = rest.strip_prefix(':').unwrap_or("");
                let valid = name.len() <= 32 && name.starts_with(|c: char| c.is_ascii_lowercase()) && name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
                if valid {
                    Ok(DbGrant::Named(name.to_string()))
                } else {
                    Err(format!("`{item}`: a database name is 1 to 32 characters `a-z`, `0-9`, `_`, starting with a letter (`db`, `db:<name>` or `db:mem`)"))
                }
            }
            None => Err(format!("`{item}` is not a database grant")),
        }
    }

    /// The grant as written.
    pub fn spec(&self) -> String {
        match self {
            DbGrant::Main => "db".into(),
            DbGrant::Named(n) => format!("db:{n}"),
            DbGrant::Memory => "db:mem".into(),
        }
    }

    /// The database file below `dir` (None for `db:mem`).
    pub fn file(&self, dir: &Path) -> Option<PathBuf> {
        match self {
            DbGrant::Main => Some(dir.join("main.sqlite")),
            DbGrant::Named(n) => Some(dir.join(format!("{n}.sqlite"))),
            DbGrant::Memory => None,
        }
    }
}

/// A value of a parameter or a result cell (`SqlValue` in QAL).
#[derive(Debug, Clone, PartialEq)]
pub enum SqlVal {
    Null,
    Int(i64),
    Float(f64),
    Text(String),
}

/// A failed statement (`DbError` in QAL), or the host budget exhausted while reading results.
#[derive(Debug, Clone, PartialEq)]
pub enum DbErr {
    Constraint(String),
    Sql(String),
    Busy,
    Full,
    TooLarge,
    /// More result bytes than the host budget has left (the execution ends with `out_of_memory`).
    HostOom(usize),
    /// The execution's wall-clock limit passed while the statement ran (the execution ends with `timeout`).
    Interrupted,
}

/// Rows of a query: column names and values.
#[derive(Debug, Clone, PartialEq)]
pub struct Rows {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<SqlVal>>,
    /// Host bytes the result holds (strings plus 8 per value).
    pub bytes: usize,
}

/// How long a statement waits for another connection's write lock.
pub const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// SQLite page size of new databases (`max_page_count` is computed from it).
const PAGE_SIZE: u64 = 4096;

/// Functions the guest's SQL may not call (most do not exist in the library build; listed so a
/// future build cannot quietly add them).
const FORBIDDEN_FUNCTIONS: [&str; 5] = ["load_extension", "readfile", "writefile", "fts3_tokenizer", "edit"];

/// The authorizer for guest statements. Writes to the schema table cannot be told apart from the
/// ones CREATE/ALTER/DROP make internally (same actions, no accessor), so they are left to SQLite:
/// without `writable_schema` (a pragma, refused) and in defensive mode it refuses direct writes.
fn authorize(ctx: AuthContext<'_>) -> Authorization {
    match ctx.action {
        AuthAction::Attach { .. } | AuthAction::Detach { .. } | AuthAction::Pragma { .. } | AuthAction::Transaction { .. } | AuthAction::Savepoint { .. } => Authorization::Deny,
        AuthAction::Function { function_name } if FORBIDDEN_FUNCTIONS.iter().any(|f| f.eq_ignore_ascii_case(function_name)) => Authorization::Deny,
        // TEMP objects would live on the connection beyond the execution (a server worker reuses
        // its connection) and are not bounded by `max_page_count`.
        AuthAction::CreateTempTable { .. } | AuthAction::CreateTempIndex { .. } | AuthAction::CreateTempTrigger { .. } | AuthAction::CreateTempView { .. } => Authorization::Deny,
        AuthAction::Insert { .. } | AuthAction::Update { .. } | AuthAction::Delete { .. } if ctx.database_name.is_some_and(|d| d.eq_ignore_ascii_case("temp")) => Authorization::Deny,
        AuthAction::Unknown { .. } => Authorization::Deny,
        _ => Authorization::Allow,
    }
}

fn sql_err(e: rusqlite::Error) -> DbErr {
    match &e {
        rusqlite::Error::SqliteFailure(f, msg) => {
            let m = msg.clone().unwrap_or_else(|| e.to_string());
            match f.code {
                ErrorCode::ConstraintViolation => DbErr::Constraint(m),
                ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked => DbErr::Busy,
                ErrorCode::DiskFull => DbErr::Full,
                ErrorCode::OperationInterrupted => DbErr::Interrupted,
                _ => DbErr::Sql(m),
            }
        }
        rusqlite::Error::MultipleStatement => DbErr::Sql("only one statement is allowed".into()),
        rusqlite::Error::ExecuteReturnedResults => DbErr::Sql("the statement returns rows: use db_query".into()),
        _ => DbErr::Sql(e.to_string()),
    }
}

/// A short text for a database error outside a guest call.
pub fn describe(e: &DbErr) -> String {
    match e {
        DbErr::Constraint(m) | DbErr::Sql(m) => m.clone(),
        DbErr::Busy => format!("the database is busy: another execution held its write lock for more than {} s", BUSY_TIMEOUT.as_secs()),
        DbErr::Full => "the database is full".into(),
        DbErr::TooLarge => "too many rows".into(),
        DbErr::HostOom(n) => format!("{n} bytes exceed the memory limit"),
        DbErr::Interrupted => "the wall-clock limit passed".into(),
    }
}

/// One execution's (or one worker's) database connection.
pub struct Db {
    conn: Connection,
    in_tx: bool,
    /// When the current execution's wall-clock limit ends; the progress handler interrupts a
    /// statement that runs past it.
    deadline: Arc<Mutex<Option<Instant>>>,
}

/// Virtual-machine steps between two deadline checks.
const PROGRESS_STEPS: i32 = 1000;

impl Db {
    /// Opens the database of `grant` below `dir` (created if needed; the file must not be a
    /// symbolic link) with the host's settings; `max_mib` bounds its size.
    pub fn open(grant: &DbGrant, dir: &Path, max_mib: usize) -> Result<Db, String> {
        let conn = match grant.file(dir) {
            None => Connection::open_in_memory().map_err(|e| format!("cannot open an in-memory database: {e}"))?,
            Some(file) => {
                std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
                if std::fs::symlink_metadata(dir).map(|m| m.file_type().is_symlink()).unwrap_or(false) {
                    return Err(format!("{} is a symbolic link", dir.display()));
                }
                // SQLITE_OPEN_NOFOLLOW refuses a link anywhere in the path: open by the real path,
                // so only a link at the file itself is refused.
                let real = dir.canonicalize().map_err(|e| format!("cannot open {}: {e}", dir.display()))?;
                let file = real.join(file.file_name().unwrap_or_default());
                if std::fs::symlink_metadata(&file).map(|m| m.file_type().is_symlink()).unwrap_or(false) {
                    return Err(format!("{} is a symbolic link", file.display()));
                }
                let flags = OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE | OpenFlags::SQLITE_OPEN_NOFOLLOW | OpenFlags::SQLITE_OPEN_NO_MUTEX;
                Connection::open_with_flags(&file, flags).map_err(|e| format!("cannot open {}: {e}", file.display()))?
            }
        };
        let setup = |c: &Connection| -> rusqlite::Result<()> {
            c.busy_timeout(BUSY_TIMEOUT)?;
            c.set_db_config(rusqlite::config::DbConfig::SQLITE_DBCONFIG_DEFENSIVE, true)?;
            c.set_db_config(rusqlite::config::DbConfig::SQLITE_DBCONFIG_TRUSTED_SCHEMA, false)?;
            c.pragma_update(None, "foreign_keys", true)?;
            c.pragma_update(None, "trusted_schema", false)?;
            c.pragma_update(None, "cell_size_check", true)?;
            c.pragma_update(None, "page_size", PAGE_SIZE as i64)?;
            let _: String = c.query_row("PRAGMA journal_mode = WAL", [], |r| r.get(0))?;
            let pages = (max_mib as u64).saturating_mul(1 << 20) / PAGE_SIZE;
            let _: i64 = c.query_row(&format!("PRAGMA max_page_count = {}", pages.max(1)), [], |r| r.get(0))?;
            // Fails early on a file that is not a database.
            let _: i64 = c.query_row("SELECT count(*) FROM sqlite_schema", [], |r| r.get(0))?;
            Ok(())
        };
        setup(&conn).map_err(|e| format!("cannot set up the database: {e}"))?;
        let deadline: Arc<Mutex<Option<Instant>>> = Arc::new(Mutex::new(None));
        let watched = deadline.clone();
        conn.progress_handler(
            PROGRESS_STEPS,
            Some(move || watched.lock().map(|d| d.is_some_and(|d| Instant::now() >= d)).unwrap_or(true)),
        )
        .map_err(|e| format!("cannot set up the database: {e}"))?;
        let _ = conn.set_limit(rusqlite::limits::Limit::SQLITE_LIMIT_SQL_LENGTH, qal_core::sql::MAX_SQL_BYTES as i32);
        Ok(Db { conn, in_tx: false, deadline })
    }

    /// Sets the limits of the next execution: statements are interrupted at `deadline`, and no
    /// string or blob SQLite builds may be larger than `max_value` bytes (SQLite allocates those
    /// outside the host's accounting).
    pub fn limit(&self, deadline: Option<Instant>, max_value: usize) {
        if let Ok(mut d) = self.deadline.lock() {
            *d = deadline;
        }
        let _ = self.conn.set_limit(rusqlite::limits::Limit::SQLITE_LIMIT_LENGTH, max_value.clamp(1024, 1 << 30) as i32);
    }

    /// Starts the execution's transaction (`BEGIN IMMEDIATE`: writers serialize).
    pub fn begin(&mut self) -> Result<(), DbErr> {
        self.conn.execute_batch("BEGIN IMMEDIATE").map_err(sql_err)?;
        self.in_tx = true;
        Ok(())
    }

    /// Ends the transaction: commits if `commit` (falling back to a rollback if that fails).
    /// Returns what happened: `commit` or `rollback`.
    pub fn finish(&mut self, commit: bool) -> Result<&'static str, String> {
        // The host's own COMMIT/ROLLBACK is never interrupted.
        self.limit(None, 1 << 30);
        if !self.in_tx {
            return Ok("none");
        }
        self.in_tx = false;
        // SQLite rolls a transaction back by itself after some errors (SQLITE_FULL, ...).
        if self.conn.is_autocommit() {
            return if commit { Err("the transaction was rolled back by the database after an error".into()) } else { Ok("rollback") };
        }
        if commit {
            match self.conn.execute_batch("COMMIT") {
                Ok(()) => return Ok("commit"),
                Err(e) => {
                    let _ = self.conn.execute_batch("ROLLBACK");
                    return Err(format!("the transaction could not be committed: {e}"));
                }
            }
        }
        self.conn.execute_batch("ROLLBACK").map_err(|e| format!("the transaction could not be rolled back: {e}"))?;
        Ok("rollback")
    }

    /// Prepares a guest statement under the authorizer and binds its parameters.
    fn prepare<'c>(&'c self, sql: &str, params: &[SqlVal]) -> Result<rusqlite::Statement<'c>, DbErr> {
        qal_core::sql::check_statement(sql).map_err(DbErr::Sql)?;
        if self.in_tx && self.conn.is_autocommit() {
            // Running on would write outside the execution's transaction.
            return Err(DbErr::Sql("the transaction of this execution was rolled back after an earlier error; no further statements run".into()));
        }
        self.conn.authorizer(Some(authorize)).map_err(sql_err)?;
        let prepared = self.conn.prepare(sql);
        // Statements can be re-prepared on schema changes while they run: keep the authorizer
        // until the statement is finished (`guarded` below clears it).
        let mut stmt = match prepared {
            Ok(s) => s,
            Err(e) => {
                let _ = self.conn.authorizer(None::<fn(AuthContext<'_>) -> Authorization>);
                return Err(sql_err(e));
            }
        };
        let expected = stmt.parameter_count();
        if expected != params.len() {
            let _ = self.conn.authorizer(None::<fn(AuthContext<'_>) -> Authorization>);
            return Err(DbErr::Sql(format!("expected {expected} parameters, got {}", params.len())));
        }
        for (i, p) in params.iter().enumerate() {
            let r = match p {
                SqlVal::Null => stmt.raw_bind_parameter(i + 1, rusqlite::types::Null),
                SqlVal::Int(n) => stmt.raw_bind_parameter(i + 1, n),
                SqlVal::Float(x) => stmt.raw_bind_parameter(i + 1, x),
                SqlVal::Text(s) => stmt.raw_bind_parameter(i + 1, s.as_str()),
            };
            if let Err(e) = r {
                let _ = self.conn.authorizer(None::<fn(AuthContext<'_>) -> Authorization>);
                return Err(sql_err(e));
            }
        }
        Ok(stmt)
    }

    fn unguard(&self) {
        let _ = self.conn.authorizer(None::<fn(AuthContext<'_>) -> Authorization>);
    }

    /// `db_exec`: runs a statement that returns no rows; the number of changed rows.
    pub fn exec(&self, sql: &str, params: &[SqlVal]) -> Result<i64, DbErr> {
        let before = self.conn.total_changes();
        let mut stmt = self.prepare(sql, params)?;
        let r = stmt.raw_execute();
        drop(stmt);
        self.unguard();
        r.map_err(sql_err)?;
        // `changes()` keeps the count of the last INSERT/UPDATE/DELETE; other statements change nothing.
        Ok(if self.conn.total_changes() == before { 0 } else { self.conn.changes() as i64 })
    }

    /// `db_query`: at most `max_rows` rows and `room` bytes of results.
    pub fn query(&self, sql: &str, params: &[SqlVal], max_rows: usize, room: usize) -> Result<Rows, DbErr> {
        let mut stmt = self.prepare(sql, params)?;
        let r = Self::read_rows(&mut stmt, max_rows, room);
        drop(stmt);
        self.unguard();
        r
    }

    fn read_rows(stmt: &mut rusqlite::Statement<'_>, max_rows: usize, room: usize) -> Result<Rows, DbErr> {
        let columns: Vec<String> = stmt.column_names().into_iter().map(str::to_string).collect();
        let mut bytes: usize = columns.iter().map(|c| c.len() + 8).sum();
        let n = columns.len();
        let mut rows = Vec::new();
        let mut it = stmt.raw_query();
        while let Some(row) = it.next().map_err(sql_err)? {
            if rows.len() == max_rows {
                return Err(DbErr::TooLarge);
            }
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let v = match row.get_ref(i).map_err(sql_err)? {
                    ValueRef::Null => SqlVal::Null,
                    ValueRef::Integer(k) => SqlVal::Int(k),
                    ValueRef::Real(x) => SqlVal::Float(x),
                    ValueRef::Text(t) => {
                        bytes = bytes.saturating_add(t.len());
                        if bytes > room {
                            return Err(DbErr::HostOom(bytes));
                        }
                        SqlVal::Text(String::from_utf8(t.to_vec()).map_err(|_| DbErr::Sql(format!("column `{}` holds text that is not UTF-8", columns[i])))?)
                    }
                    ValueRef::Blob(_) => return Err(DbErr::Sql("BLOB columns are not supported".into())),
                };
                out.push(v);
            }
            bytes = bytes.saturating_add(8 * n + 8);
            if bytes > room {
                return Err(DbErr::HostOom(bytes));
            }
            rows.push(out);
        }
        Ok(Rows { columns, rows, bytes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem() -> Db {
        Db::open(&DbGrant::Memory, Path::new("/nonexistent"), 1024).unwrap()
    }

    #[test]
    fn grants_parse_and_name_files() {
        assert_eq!(DbGrant::parse("db"), Ok(DbGrant::Main));
        assert_eq!(DbGrant::parse("db:mem"), Ok(DbGrant::Memory));
        assert_eq!(DbGrant::parse("db:todo_2"), Ok(DbGrant::Named("todo_2".into())));
        for bad in ["db:", "db:Todo", "db:2x", "db:../x", "db:a-b", "db:abcdefghijklmnopqrstuvwxyzabcdefg", "dbx"] {
            assert!(DbGrant::parse(bad).is_err(), "{bad}");
        }
        assert_eq!(DbGrant::Named("t".into()).file(Path::new("/p/.qal/db")), Some(PathBuf::from("/p/.qal/db/t.sqlite")));
        assert_eq!(DbGrant::Memory.file(Path::new("/p")), None);
        assert_eq!(DbGrant::Named("t".into()).spec(), "db:t");
    }

    #[test]
    fn statements_run_with_positional_parameters() {
        let db = mem();
        assert_eq!(db.exec("CREATE TABLE t (id INTEGER PRIMARY KEY, title TEXT NOT NULL, score REAL, note TEXT)", &[]), Ok(0));
        assert_eq!(db.exec("INSERT INTO t (title, score, note) VALUES (?, ?, ?)", &[SqlVal::Text("a'; DROP TABLE t; --".into()), SqlVal::Float(1.5), SqlVal::Null]), Ok(1));
        assert_eq!(db.exec("INSERT INTO t (title) VALUES (?)", &[SqlVal::Text("b".into())]), Ok(1));
        let r = db.query("SELECT id, title, score, note FROM t ORDER BY id", &[], 100, 1 << 20).unwrap();
        assert_eq!(r.columns, vec!["id", "title", "score", "note"]);
        assert_eq!(r.rows[0], vec![SqlVal::Int(1), SqlVal::Text("a'; DROP TABLE t; --".into()), SqlVal::Float(1.5), SqlVal::Null]);
        assert_eq!(r.rows.len(), 2);
        let r = db.query("INSERT INTO t (title) VALUES (?) RETURNING id", &[SqlVal::Text("c".into())], 100, 1 << 20).unwrap();
        assert_eq!(r.rows, vec![vec![SqlVal::Int(3)]]);
        assert_eq!(db.exec("SELECT 1", &[]), Err(DbErr::Sql("the statement returns rows: use db_query".into())));
        assert_eq!(db.exec("INSERT INTO t (title) VALUES (?)", &[]), Err(DbErr::Sql("expected 1 parameters, got 0".into())));
        assert!(matches!(db.exec("INSERT INTO t (id, title) VALUES (1, 'dup')", &[]), Err(DbErr::Constraint(_))));
        assert!(matches!(db.exec("INSERT INTO t (title) VALUES (NULL)", &[]), Err(DbErr::Constraint(_))));
        assert!(matches!(db.query("SELECT nosuch FROM t", &[], 10, 1 << 20), Err(DbErr::Sql(_))));
        assert_eq!(db.query("SELECT x'00'", &[], 10, 1 << 20), Err(DbErr::Sql("BLOB columns are not supported".into())));
        assert_eq!(db.query("SELECT CAST(x'ff' AS TEXT)", &[], 10, 1 << 20), Err(DbErr::Sql("column `CAST(x'ff' AS TEXT)` holds text that is not UTF-8".into())));
        assert_eq!(db.query("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 11) SELECT x FROM c", &[], 10, 1 << 20).err(), Some(DbErr::TooLarge));
        assert!(matches!(db.query("SELECT title FROM t", &[], 10, 10), Err(DbErr::HostOom(_))));
    }

    #[test]
    fn the_authorizer_refuses_control_statements_and_functions() {
        let db = mem();
        db.exec("CREATE TABLE t (a TEXT)", &[]).unwrap();
        db.exec("CREATE INDEX t_a ON t (a)", &[]).unwrap();
        db.exec("CREATE VIEW v AS SELECT a FROM t", &[]).unwrap();
        db.exec("CREATE TRIGGER tr AFTER INSERT ON t BEGIN UPDATE t SET a = upper(new.a) WHERE rowid = new.rowid; END", &[]).unwrap();
        db.exec("INSERT INTO t VALUES ('x')", &[]).unwrap();
        db.exec("DROP VIEW v", &[]).unwrap();
        db.exec("ALTER TABLE t ADD COLUMN b INTEGER", &[]).unwrap();
        let refused = |sql: &str| match db.exec(sql, &[]) {
            Err(DbErr::Sql(m)) => m,
            other => panic!("{sql}: {other:?}"),
        };
        // The static check refuses these first; the authorizer is the second line.
        for sql in ["ATTACH 'x.db' AS x", "PRAGMA writable_schema = 1", "BEGIN", "SAVEPOINT a", "VACUUM"] {
            assert!(refused(sql).contains("not allowed"), "{sql}");
        }
        assert!(refused("SELECT load_extension('x')").contains("not authorized") || refused("SELECT load_extension('x')").contains("no such function"));
        assert!(refused("UPDATE sqlite_master SET sql = 'x'").contains("may not be modified"));
        assert!(refused("DELETE FROM sqlite_schema").contains("may not be modified"));
        assert!(refused("INSERT INTO sqlite_master VALUES ('table', 'x', 'x', 0, 'CREATE TABLE x (a)')").contains("may not be modified"));
        assert!(refused("SELECT * FROM pragma_table_info('t')").contains("not authorized"), "table-valued pragmas are pragmas");
        assert!(refused("SELECT 1; SELECT 2").contains("only one statement"));
        // The host's own statements are not affected.
        let mut db = db;
        db.begin().unwrap();
        assert_eq!(db.finish(true), Ok("commit"));
    }

    #[test]
    fn transactions_commit_or_roll_back() {
        let dir = std::env::temp_dir().join(format!("qal-db-tx-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let grant = DbGrant::Named("t".into());
        let mut db = Db::open(&grant, &dir, 16).unwrap();
        db.begin().unwrap();
        db.exec("CREATE TABLE t (a INTEGER)", &[]).unwrap();
        assert_eq!(db.finish(true), Ok("commit"));
        db.begin().unwrap();
        db.exec("INSERT INTO t VALUES (1)", &[]).unwrap();
        assert_eq!(db.finish(false), Ok("rollback"));
        assert_eq!(db.finish(true), Ok("none"));
        let other = Db::open(&grant, &dir, 16).unwrap();
        assert_eq!(other.query("SELECT count(*) FROM t", &[], 10, 1 << 20).unwrap().rows, vec![vec![SqlVal::Int(0)]]);
        assert!(dir.join("t.sqlite").exists());
        // A writer blocks another writer's BEGIN IMMEDIATE (busy after the timeout).
        db.begin().unwrap();
        let mut second = Db::open(&grant, &dir, 16).unwrap();
        second.conn.busy_timeout(Duration::from_millis(50)).unwrap();
        assert_eq!(second.begin(), Err(DbErr::Busy));
        db.finish(false).unwrap();
        // The size limit: max_page_count makes the database full.
        let mut small = Db::open(&DbGrant::Named("small".into()), &dir, 1).unwrap();
        small.begin().unwrap();
        small.exec("CREATE TABLE b (s TEXT)", &[]).unwrap();
        let big = SqlVal::Text("x".repeat(300_000));
        let mut full = false;
        for _ in 0..10 {
            if small.exec("INSERT INTO b VALUES (?)", std::slice::from_ref(&big)) == Err(DbErr::Full) {
                full = true;
                break;
            }
        }
        assert!(full, "1 MiB limit reached");
        assert!(matches!(small.exec("INSERT INTO b VALUES ('y')", &[]), Err(DbErr::Sql(m)) if m.contains("rolled back")), "nothing runs outside the transaction");
        assert_eq!(small.finish(false), Ok("rollback"));
        small.begin().unwrap();
        assert!(small.finish(true).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn database_files_are_never_symlinks() {
        let dir = std::env::temp_dir().join(format!("qal-db-link-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("elsewhere.sqlite");
        std::os::unix::fs::symlink(&target, dir.join("main.sqlite")).unwrap();
        assert!(Db::open(&DbGrant::Main, &dir, 16).err().unwrap().contains("symbolic link"));
        assert!(!target.exists());
        std::fs::write(dir.join("junk.sqlite"), "this is not a database, just some text").unwrap();
        assert!(Db::open(&DbGrant::Named("junk".into()), &dir, 16).is_err());
    }
}
