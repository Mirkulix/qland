//! Host functions (`qal_host.*`) and the permission model: only granted effect groups are linked
//! (ungranted ones are stubs that fail with `permission_denied`), `fs` only reaches files inside
//! granted directories (through their descriptors, see [`crate::fs`]), and everything the host
//! buffers for the guest counts against the execution's memory limit.

use crate::db::{Db, DbErr, DbGrant, SqlVal};
use crate::fs::{self as gfs, Access, FsRoot};
use crate::guest::Guest;
use crate::net::{self, Call, Lookup, NetErr, NetGrant};
use crate::value::{decode_exact_counted, read_str, Value};
use qal_core::ast::Type;
use qal_core::tast::TProgram;
use qal_wasm::{host, FAIL_OOM, HOST_MODULE, OOM_SCRATCH};
use std::fs::File;
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use wasmtime::{Caller, Extern, FuncType, Linker, Memory, StoreLimits, ValType};

/// Failure kind recorded by the stub of an ungranted effect function; `func` indexes `host::EFFECT_FNS`.
pub const FAIL_DENIED: i32 = -1;
/// Failure kind recorded by the host when a per-execution budget is used up (see [`BudgetHit`]).
pub const FAIL_BUDGET: i32 = -2;
/// Failure kind recorded by the host when the wall-clock limit passed inside a host call.
pub const FAIL_TIMEOUT: i32 = -3;

/// Effect permissions for one execution, e.g. parsed from `--allow io,fs:./data`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Permissions {
    pub io: bool,
    pub time: bool,
    pub rand: bool,
    /// Directories `fs` may read and write below, held open. Empty = no `fs`.
    pub fs: Vec<FsRoot>,
    /// Directory that relative guest paths are resolved against (None: the working directory).
    pub base: Option<PathBuf>,
    /// If set, no path component below a granted directory may start with `.` (protects `.git`,
    /// `.qal`, `.claude`, `.env`, ... inside a granted project directory).
    pub no_hidden: bool,
    /// The database grant (`db`, `db:<name>`, `db:mem`); at most one per execution.
    pub db: Option<DbGrant>,
    /// Where database files live (`<project>/.qal/db`); None: `.qal/db` below the working directory.
    pub db_dir: Option<PathBuf>,
    /// Hosts outgoing HTTP may reach (`net:<host>[:port]`, repeatable). Empty = no `net`.
    pub net: Vec<NetGrant>,
}

/// Known grants, for error messages.
const KNOWN_GRANTS: &str = "io, time, rand, fs:<dir>, net:<host>[:port], db, db:<name>, db:mem";

/// Adds a database grant (only one per execution).
fn grant_db(p: &mut Permissions, item: &str) -> Result<(), String> {
    let g = DbGrant::parse(item)?;
    if let Some(old) = &p.db {
        return Err(format!("only one database can be granted per execution: `{}` and `{item}`", old.spec()));
    }
    p.db = Some(g);
    Ok(())
}

fn is_db_item(item: &str) -> bool {
    item == "db" || item.starts_with("db:")
}

impl Permissions {
    /// Parses a comma-separated list: `io`, `time`, `rand`, `fs:<dir>` (repeatable). Directories must exist.
    pub fn parse(spec: &str) -> Result<Permissions, String> {
        let mut p = Permissions::default();
        for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            match item {
                "io" => p.io = true,
                "time" => p.time = true,
                "rand" => p.rand = true,
                "fs" => return Err("`fs` needs a directory: `fs:<dir>`, e.g. `fs:./data`".into()),
                _ if is_db_item(item) => grant_db(&mut p, item)?,
                _ if item == "net" || item.starts_with("net:") => p.net.push(NetGrant::parse(item)?),
                _ => match item.strip_prefix("fs:") {
                    Some(dir) => p.fs.push(FsRoot::open(Path::new(dir))?),
                    None => return Err(format!("unknown effect `{item}` (known: {KNOWN_GRANTS})")),
                },
            }
        }
        Ok(p)
    }

    /// Parses grants for a program run on behalf of an agent (MCP): items as in [`Permissions::parse`],
    /// but `fs:<dir>` is relative to `root` and must stay inside it, hidden entries (`.git`, `.qal`,
    /// ...) can neither be granted nor reached, and guest paths are resolved against `root`.
    pub fn parse_in(items: &[String], root: &Path) -> Result<Permissions, String> {
        let given = root.to_path_buf();
        let root = root.canonicalize().map_err(|e| format!("project root {}: {e}", root.display()))?;
        let mut p = Permissions { base: Some(root.clone()), no_hidden: true, db_dir: Some(root.join(".qal").join("db")), ..Permissions::default() };
        for item in items.iter().map(|s| s.trim()) {
            match item {
                "io" => p.io = true,
                "time" => p.time = true,
                "rand" => p.rand = true,
                "fs" => return Err("`fs` needs a directory: `fs:<dir>` relative to the project root, e.g. `fs:data` or `fs:.`".into()),
                _ if is_db_item(item) => grant_db(&mut p, item)?,
                _ if item == "net" || item.starts_with("net:") => p.net.push(NetGrant::parse(item)?),
                _ => match item.strip_prefix("fs:") {
                    Some(dir) if dir.trim().is_empty() => return Err("`fs:` needs a directory: `fs:<dir>` relative to the project root, e.g. `fs:data` or `fs:.`".into()),
                    Some(dir) => {
                        let dir = dir.trim();
                        let canonical = root.join(dir).canonicalize().map_err(|_| format!("fs:{dir}: no such directory in the project"))?;
                        let rel = canonical.strip_prefix(&root).map_err(|_| format!("fs:{dir}: outside the project root"))?;
                        if hidden(rel) {
                            return Err(format!("fs:{dir}: hidden directories (starting with `.`) cannot be granted"));
                        }
                        // Opened by the path as given, so guest paths may use either spelling of the root.
                        let path = given.join(dir);
                        p.fs.push(FsRoot::open(&path).map_err(|e| e.replace(&format!("fs:{}", path.display()), &format!("fs:{dir}")))?);
                    }
                    None => return Err(format!("unknown effect `{item}` (known: {KNOWN_GRANTS})")),
                },
            }
        }
        Ok(p)
    }

    /// Sets where database files live (the CLI passes `<project>/.qal/db` of the store it found).
    pub fn with_db_dir(mut self, dir: PathBuf) -> Permissions {
        self.db_dir = Some(dir);
        self
    }

    /// The directory of the database files.
    pub fn db_dir(&self) -> PathBuf {
        match (&self.db_dir, &self.base) {
            (Some(d), _) => d.clone(),
            (None, Some(b)) => b.join(".qal").join("db"),
            (None, None) => PathBuf::from(".qal").join("db"),
        }
    }

    pub fn allows(&self, effect: &str) -> bool {
        match effect {
            "io" => self.io,
            "time" => self.time,
            "rand" => self.rand,
            "fs" => !self.fs.is_empty(),
            "db" => self.db.is_some(),
            "net" => !self.net.is_empty(),
            _ => false,
        }
    }

    /// Opens a guest path inside the granted directories (see [`crate::fs::open`]).
    fn open(&self, path: &str, access: Access) -> Result<File, String> {
        let base = match &self.base {
            Some(b) => b.clone(),
            None => std::env::current_dir().map_err(|e| format!("{path}: no working directory: {e}"))?,
        };
        gfs::open(&self.fs, &base, path, self.no_hidden, access)
    }
}

/// True if a path below a granted directory has a component starting with `.`.
fn hidden(rel: &Path) -> bool {
    rel.components().any(|c| c.as_os_str().to_string_lossy().starts_with('.'))
}

fn read_text(file: File, path: &str, limit: usize) -> Result<Result<String, String>, usize> {
    let len = file.metadata().map(|m| m.len()).unwrap_or(0) as usize;
    if len > limit {
        return Err(len);
    }
    let mut bytes = Vec::new();
    if let Err(e) = file.take(limit as u64 + 1).read_to_end(&mut bytes) {
        return Ok(Err(format!("{path}: {e}")));
    }
    if bytes.len() > limit {
        return Err(bytes.len());
    }
    Ok(String::from_utf8(bytes).map_err(|_| format!("{path}: not valid UTF-8")))
}

/// Per-execution budgets besides fuel, time and memory (Phase 3 spec §2.5, §5.4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limits {
    /// Bytes `write_file` may write in total.
    pub fs_write_bytes: usize,
    /// Number of `write_file` calls.
    pub fs_writes: usize,
    /// Rows one `db_query` may return (more is `Err(DbTooLarge)`).
    pub db_max_rows: usize,
    /// Size limit of a database file in MiB (`max_page_count`; reaching it is `Err(DbFull)`).
    pub db_max_mib: usize,
    /// Outgoing `http` calls per execution.
    pub net_calls: usize,
}

impl Default for Limits {
    fn default() -> Limits {
        Limits { fs_write_bytes: 64 << 20, fs_writes: 256, db_max_rows: 10_000, db_max_mib: 1024, net_calls: 16 }
    }
}

/// Which budget an execution used up: reported as `budget_exceeded` (E0522).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetHit {
    /// `fs_bytes`, `fs_writes` or `net_calls`.
    pub kind: &'static str,
    pub budget: usize,
    pub used: usize,
}

/// A failure reported by the guest through `qal_host.fail`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Failure {
    pub kind: i32,
    pub func: i32,
    pub site: i32,
    pub vals: i32,
}

pub struct HostState {
    pub failure: Option<Failure>,
    pub stdout: Vec<u8>,
    pub stdin: Box<dyn BufRead + Send>,
    pub perms: Permissions,
    pub rng: u64,
    pub limits: StoreLimits,
    /// Memory limit in bytes; also the budget of everything the host buffers (`host_used`).
    pub limit: usize,
    /// If set, printed output beyond this many bytes is counted in `stdout_bytes` but not kept.
    pub stdout_limit: Option<usize>,
    pub stdout_bytes: usize,
    /// Bytes the host holds for this execution (kept output, file contents, ...), at most `limit`.
    pub host_used: usize,
    /// Set when `host_used` would have exceeded `limit` (the error then says `host: true`).
    pub host_oom: bool,
    pub budgets: Limits,
    pub budget_hit: Option<BudgetHit>,
    pub fs_bytes: usize,
    pub fs_writes: usize,
    /// The execution's database connection (inside its transaction), if `db` is granted and used.
    pub db: Option<Db>,
    pub net_calls: usize,
    /// When the execution's wall-clock limit ends (outgoing requests never wait longer).
    pub deadline: Option<std::time::Instant>,
    /// Name resolution for `net` (the system resolver; replaced in tests).
    pub lookup: Lookup,
    /// Size of the guest memory when the call ended (for the service log).
    pub mem_bytes: usize,
}

impl HostState {
    /// A fresh state for one execution.
    pub fn new(perms: Permissions, stdin: Box<dyn BufRead + Send>, limit: usize, budgets: Limits) -> HostState {
        HostState {
            failure: None,
            stdout: Vec::new(),
            stdin,
            perms,
            rng: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1),
            limits: wasmtime::StoreLimitsBuilder::new().memory_size(limit).build(),
            limit,
            stdout_limit: None,
            stdout_bytes: 0,
            host_used: 0,
            host_oom: false,
            budgets,
            budget_hit: None,
            fs_bytes: 0,
            fs_writes: 0,
            db: None,
            net_calls: 0,
            deadline: None,
            lookup: net::system_lookup,
            mem_bytes: 0,
        }
    }
}

fn memory(c: &mut Caller<'_, HostState>) -> Result<Memory, wasmtime::Error> {
    match c.get_export("memory") {
        Some(Extern::Memory(m)) => Ok(m),
        _ => Err(wasmtime::Error::msg("module has no memory export")),
    }
}

fn guest_str(c: &mut Caller<'_, HostState>, ptr: i32) -> Result<String, wasmtime::Error> {
    let mem = memory(c)?;
    read_str(mem.data(&c), ptr as u32).ok_or_else(|| wasmtime::Error::msg("invalid string pointer"))
}

/// Reports a host-side allocation over the memory limit like a guest allocation failure, marked
/// as the host's.
pub fn host_oom(c: &mut Caller<'_, HostState>, requested: usize) -> wasmtime::Error {
    if let Ok(mem) = memory(c) {
        let _ = mem.write(&mut *c, OOM_SCRATCH as usize, &(requested as i64).to_le_bytes());
    }
    c.data_mut().host_oom = true;
    c.data_mut().failure = Some(Failure { kind: FAIL_OOM, func: 0, site: 0, vals: 0 });
    wasmtime::Error::msg("out of memory")
}

/// Counts `bytes` against the execution's host budget (= its memory limit).
pub fn charge(c: &mut Caller<'_, HostState>, bytes: usize) -> Result<(), wasmtime::Error> {
    let requested = c.data().host_used.saturating_add(bytes);
    if requested > c.data().limit {
        return Err(host_oom(c, requested));
    }
    c.data_mut().host_used = requested;
    Ok(())
}

/// Ends the execution with `budget_exceeded`.
fn budget_exceeded(c: &mut Caller<'_, HostState>, kind: &'static str, budget: usize, used: usize) -> wasmtime::Error {
    c.data_mut().budget_hit = Some(BudgetHit { kind, budget, used });
    c.data_mut().failure = Some(Failure { kind: FAIL_BUDGET, func: 0, site: 0, vals: 0 });
    wasmtime::Error::msg("budget exceeded")
}

/// The prelude types: everything host functions read and write is built from them.
pub fn prelude_program() -> &'static TProgram {
    static PRELUDE: std::sync::OnceLock<TProgram> = std::sync::OnceLock::new();
    PRELUDE.get_or_init(|| TProgram { types: qal_core::prelude::types(), ..TProgram::default() })
}

/// The guest's memory and allocator, from inside a host function.
fn guest(c: &mut Caller<'_, HostState>) -> Result<Guest, wasmtime::Error> {
    let mem = memory(c)?;
    let alloc = c.get_export("qal_alloc").and_then(Extern::into_func).ok_or_else(|| wasmtime::Error::msg("no qal_alloc"))?;
    Ok(Guest::new(mem, alloc.typed::<i64, i32>(&c)?))
}

/// Writes a host-built value of a prelude-based type into guest memory; returns the pointer.
fn write_value(c: &mut Caller<'_, HostState>, ty: &Type, v: &Value) -> Result<i32, wasmtime::Error> {
    let g = guest(c)?;
    match g.encode(c, prelude_program(), ty, v) {
        Ok(slot) => Ok(slot as i32),
        Err(crate::guest::EncodeError::Trap(e)) => Err(e),
        Err(crate::guest::EncodeError::Mismatch(m)) => Err(wasmtime::Error::msg(format!("internal error: a host value does not match its type: {m}"))),
    }
}

/// Reads a value of a prelude-based type from guest memory, exactly; the host copy counts against
/// the host budget.
fn read_value(c: &mut Caller<'_, HostState>, ty: &Type, ptr: i32) -> Result<Value, wasmtime::Error> {
    let mem = memory(c)?;
    let room = c.data().limit.saturating_sub(c.data().host_used);
    match decode_exact_counted(prelude_program(), mem.data(&c), ty, ptr as u32 as i64, room) {
        Ok((v, bytes)) => {
            charge(c, bytes)?;
            Ok(v)
        }
        Err(e) if e.starts_with("the value is larger") => {
            let requested = c.data().host_used.saturating_add(room).saturating_add(1);
            Err(host_oom(c, requested))
        }
        Err(e) => Err(wasmtime::Error::msg(format!("cannot read the argument: {e}"))),
    }
}

fn named(n: &str) -> Type {
    Type::Named(n.to_string())
}

fn ok(v: Value) -> Value {
    Value::Variant("Ok".into(), vec![v])
}

fn err(v: Value) -> Value {
    Value::Variant("Err".into(), vec![v])
}

fn variant(name: &str, payload: Vec<Value>) -> Value {
    Value::Variant(name.into(), payload)
}

/// SQL parameters from a guest `List[SqlValue]`.
fn sql_params(v: Value) -> Result<Vec<SqlVal>, wasmtime::Error> {
    let Value::List(xs) = v else { return Err(wasmtime::Error::msg("internal error: parameters are not a list")) };
    xs.into_iter()
        .map(|x| match x {
            Value::Variant(n, p) => Ok(match (n.as_str(), p.as_slice()) {
                ("SqlNull", []) => SqlVal::Null,
                ("SqlInt", [Value::Int(k)]) => SqlVal::Int(*k),
                ("SqlFloat", [Value::Float(f)]) => SqlVal::Float(*f),
                ("SqlText", [Value::Str(s)]) => SqlVal::Text(s.clone()),
                _ => return Err(wasmtime::Error::msg("internal error: malformed SqlValue")),
            }),
            _ => Err(wasmtime::Error::msg("internal error: malformed SqlValue")),
        })
        .collect()
}

fn sql_value(v: SqlVal) -> Value {
    match v {
        SqlVal::Null => variant("SqlNull", vec![]),
        SqlVal::Int(n) => variant("SqlInt", vec![Value::Int(n)]),
        SqlVal::Float(x) => variant("SqlFloat", vec![Value::Float(x)]),
        SqlVal::Text(s) => variant("SqlText", vec![Value::Str(s)]),
    }
}

/// A `DbError` value, or the host OOM trap for results larger than the budget.
fn db_error(c: &mut Caller<'_, HostState>, e: DbErr) -> Result<Value, wasmtime::Error> {
    Ok(match e {
        DbErr::Constraint(m) => variant("DbConstraint", vec![Value::Str(m)]),
        DbErr::Sql(m) => variant("DbSql", vec![Value::Str(m)]),
        DbErr::Busy => variant("DbBusy", vec![]),
        DbErr::Full => variant("DbFull", vec![]),
        DbErr::TooLarge => variant("DbTooLarge", vec![]),
        DbErr::HostOom(bytes) => {
            let requested = c.data().host_used.saturating_add(bytes);
            return Err(host_oom(c, requested));
        }
        DbErr::Interrupted => {
            c.data_mut().failure = Some(Failure { kind: FAIL_TIMEOUT, func: 0, site: 0, vals: 0 });
            return Err(wasmtime::Error::msg("wall-clock limit exceeded"));
        }
    })
}

/// Runs `db_exec` / `db_query` for the guest: reads the SQL and parameters, runs the statement on
/// the execution's connection and writes the `Result` into guest memory.
fn db_call(c: &mut Caller<'_, HostState>, sql: i32, params: i32, query: bool) -> Result<i32, wasmtime::Error> {
    let sql = guest_str(c, sql)?;
    let params = sql_params(read_value(c, &Type::list(named("SqlValue")), params)?)?;
    let (max_rows, room) = (c.data().budgets.db_max_rows, c.data().limit.saturating_sub(c.data().host_used));
    let no_db = || DbErr::Sql("no database is open for this execution".into());
    let (ty, result) = if query {
        let r = c.data().db.as_ref().ok_or_else(no_db).and_then(|db| db.query(&sql, &params, max_rows, room));
        let v = match r {
            Ok(rows) => {
                charge(c, rows.bytes)?;
                let cols = Value::List(rows.columns.into_iter().map(Value::Str).collect());
                let cells = Value::List(rows.rows.into_iter().map(|r| Value::List(r.into_iter().map(sql_value).collect())).collect());
                ok(Value::Record("DbRows".into(), vec![("columns".into(), cols), ("rows".into(), cells)]))
            }
            Err(e) => err(db_error(c, e)?),
        };
        (Type::result(named("DbRows"), named("DbError")), v)
    } else {
        let r = c.data().db.as_ref().ok_or_else(no_db).and_then(|db| db.exec(&sql, &params));
        let v = match r {
            Ok(n) => ok(Value::Int(n)),
            Err(e) => err(db_error(c, e)?),
        };
        (Type::result(Type::Int, named("DbError")), v)
    };
    write_value(c, &ty, &result)
}

fn header_list(headers: Vec<(String, String)>) -> Value {
    Value::List(headers.into_iter().map(|(n, v)| Value::Record("HttpHeader".into(), vec![("name".into(), Value::Str(n)), ("value".into(), Value::Str(v))])).collect())
}

/// The fields of a decoded `HttpHeader` list.
pub fn headers_of(v: &Value) -> Option<Vec<(String, String)>> {
    let Value::List(xs) = v else { return None };
    xs.iter()
        .map(|h| match h {
            Value::Record(_, f) => match f.as_slice() {
                [(_, Value::Str(n)), (_, Value::Str(v))] => Some((n.clone(), v.clone())),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

fn field<'v>(fields: &'v [(String, Value)], name: &str) -> Option<&'v Value> {
    fields.iter().find(|(n, _)| n == name).map(|(_, v)| v)
}

fn call_of(v: &Value) -> Option<Call> {
    let Value::Record(_, f) = v else { return None };
    let s = |n: &str| match field(f, n) {
        Some(Value::Str(s)) => Some(s.clone()),
        _ => None,
    };
    Some(Call { method: s("method")?, url: s("url")?, headers: headers_of(field(f, "headers")?)?, body: s("body")? })
}

/// Runs `http` for the guest: budget, grants and SSRF checks, the request, and the `Result`.
fn net_call(c: &mut Caller<'_, HostState>, ptr: i32) -> Result<i32, wasmtime::Error> {
    let budget = c.data().budgets.net_calls;
    let calls = c.data().net_calls + 1;
    if calls > budget {
        return Err(budget_exceeded(c, "net_calls", budget, calls));
    }
    c.data_mut().net_calls = calls;
    let v = read_value(c, &named("HttpCall"), ptr)?;
    let call = call_of(&v).ok_or_else(|| wasmtime::Error::msg("internal error: malformed HttpCall"))?;
    let deadline = c.data().deadline;
    let left = || deadline.map_or(net::TOTAL_TIMEOUT, |d| d.saturating_duration_since(std::time::Instant::now()));
    let lookup = c.data().lookup;
    // The time left is taken again after the lookup, so the whole call stays within the limit.
    let result = net::plan(&c.data().perms.net, &call, lookup, left().min(net::CONNECT_TIMEOUT)).and_then(|(url, addrs)| net::send(&call, url, addrs, left()));
    let v = match result {
        Ok(reply) => {
            charge(c, reply.body.len() + reply.headers.iter().map(|(n, v)| n.len() + v.len() + 16).sum::<usize>())?;
            ok(Value::Record("HttpResponse".into(), vec![("status".into(), Value::Int(reply.status)), ("headers".into(), header_list(reply.headers)), ("body".into(), Value::Str(reply.body))]))
        }
        Err(e) => err(match e {
            NetErr::Denied(m) => variant("NetDenied", vec![Value::Str(m)]),
            NetErr::Invalid(m) => variant("NetInvalid", vec![Value::Str(m)]),
            NetErr::Connect(m) => variant("NetConnect", vec![Value::Str(m)]),
            NetErr::Timeout => variant("NetTimeout", vec![]),
            NetErr::TooLarge => variant("NetTooLarge", vec![]),
        }),
    };
    write_value(c, &Type::result(named("HttpResponse"), named("NetError")), &v)
}

/// Allocates a Str in guest memory via the exported allocator (which may fail with out_of_memory).
pub fn new_str(c: &mut Caller<'_, HostState>, s: &str) -> Result<i32, wasmtime::Error> {
    let alloc = c.get_export("qal_alloc").and_then(Extern::into_func).ok_or_else(|| wasmtime::Error::msg("no qal_alloc"))?;
    let p = alloc.typed::<i64, i32>(&c)?.call(&mut *c, 4 + s.len() as i64)?;
    let mem = memory(c)?;
    mem.write(&mut *c, p as usize, &(s.len() as u32).to_le_bytes())?;
    mem.write(&mut *c, p as usize + 4, s.as_bytes())?;
    Ok(p)
}

fn val_type(t: &host::ValType) -> Result<ValType, wasmtime::Error> {
    match t {
        host::ValType::I32 => Ok(ValType::I32),
        host::ValType::I64 => Ok(ValType::I64),
        host::ValType::F64 => Ok(ValType::F64),
        other => Err(wasmtime::Error::msg(format!("host signatures use no {other:?}"))),
    }
}

/// splitmix64
fn next_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Keeps `line` of printed output: up to `stdout_limit` (the rest is only counted), or up to the
/// host budget (then `out_of_memory`).
fn print_line(c: &mut Caller<'_, HostState>, line: &[u8]) -> Result<(), wasmtime::Error> {
    if let Some(keep) = c.data().stdout_limit {
        let data = c.data_mut();
        data.stdout_bytes = data.stdout_bytes.saturating_add(line.len());
        let room = keep.saturating_sub(data.stdout.len()).min(line.len());
        charge(c, room)?;
        c.data_mut().stdout.extend_from_slice(&line[..room]);
        return Ok(());
    }
    let room = c.data().limit.saturating_sub(c.data().host_used);
    if line.len() > room {
        c.data_mut().stdout.extend_from_slice(&line[..room]);
        let requested = c.data().host_used + line.len();
        return Err(host_oom(c, requested));
    }
    charge(c, line.len())?;
    c.data_mut().stdout.extend_from_slice(line);
    Ok(())
}

/// Defines `fail`, the pure helpers, the functions of every granted effect, and stubs for the
/// ungranted ones: a module may import them (a fn value of an effectful fn makes it reachable)
/// without ever calling them, so it must still instantiate.
pub fn link(linker: &mut Linker<HostState>, perms: &Permissions) -> Result<(), wasmtime::Error> {
    for (k, name) in host::EFFECT_FNS.iter().enumerate() {
        if perms.allows(host::effect(name).unwrap_or("")) {
            continue;
        }
        let (params, results) = host::signature(name).ok_or_else(|| wasmtime::Error::msg(format!("unknown host function `{name}`")))?;
        let params = params.iter().map(val_type).collect::<Result<Vec<_>, _>>()?;
        let results = results.iter().map(val_type).collect::<Result<Vec<_>, _>>()?;
        let ty = FuncType::new(linker.engine(), params, results);
        linker.func_new(HOST_MODULE, name, ty, move |mut c, _, _| {
            c.data_mut().failure = Some(Failure { kind: FAIL_DENIED, func: k as i32, site: 0, vals: 0 });
            Err(wasmtime::Error::msg("permission denied"))
        })?;
    }
    linker.func_wrap(HOST_MODULE, host::FAIL, |mut c: Caller<'_, HostState>, kind: i32, func: i32, site: i32, vals: i32| {
        c.data_mut().failure = Some(Failure { kind, func, site, vals });
        Err::<(), _>(wasmtime::Error::msg("qal failure"))
    })?;
    linker.func_wrap(HOST_MODULE, host::FLOAT_TO_STR, |mut c: Caller<'_, HostState>, x: f64| new_str(&mut c, &format!("{x:?}")))?;
    linker.func_wrap(HOST_MODULE, host::UTF8_VALID, |mut c: Caller<'_, HostState>, p: i32| -> Result<i32, wasmtime::Error> {
        // Bytes have the layout of Str: [len: i32][bytes].
        let mem = memory(&mut c)?;
        let data = mem.data(&c);
        let at = p as u32 as usize;
        let bytes = data
            .get(at..at + 4)
            .and_then(|l| <[u8; 4]>::try_from(l).ok())
            .map(|l| u32::from_le_bytes(l) as usize)
            .and_then(|len| data.get(at + 4..at + 4 + len))
            .ok_or_else(|| wasmtime::Error::msg("invalid bytes pointer"))?;
        Ok(std::str::from_utf8(bytes).is_ok() as i32)
    })?;
    linker.func_wrap(HOST_MODULE, host::STR_TO_FLOAT, |mut c: Caller<'_, HostState>, p: i32| {
        Ok(match guest_str(&mut c, p)?.parse::<f64>() {
            Ok(x) => (1, x),
            Err(_) => (0, 0.0),
        })
    })?;
    if perms.io {
        linker.func_wrap(HOST_MODULE, host::PRINT, |mut c: Caller<'_, HostState>, p: i32| {
            let mut line = guest_str(&mut c, p)?.into_bytes();
            line.push(b'\n');
            print_line(&mut c, &line)
        })?;
        linker.func_wrap(HOST_MODULE, host::READ_LINE, |mut c: Caller<'_, HostState>| {
            let mut buf = Vec::new();
            let room = c.data().limit.saturating_sub(c.data().host_used);
            let n = c.data_mut().stdin.by_ref().take(room as u64 + 1).read_until(b'\n', &mut buf).map_err(wasmtime::Error::msg)?;
            if n == 0 {
                return Ok(-1);
            }
            charge(&mut c, n)?;
            if buf.ends_with(b"\n") {
                buf.pop();
                if buf.ends_with(b"\r") {
                    buf.pop();
                }
            }
            new_str(&mut c, &String::from_utf8_lossy(&buf))
        })?;
        linker.func_wrap(HOST_MODULE, host::READ_ALL, |mut c: Caller<'_, HostState>| {
            let mut buf = Vec::new();
            let room = c.data().limit.saturating_sub(c.data().host_used);
            let n = c.data_mut().stdin.by_ref().take(room as u64 + 1).read_to_end(&mut buf).map_err(wasmtime::Error::msg)?;
            charge(&mut c, n)?;
            new_str(&mut c, &String::from_utf8_lossy(&buf))
        })?;
    }
    if !perms.fs.is_empty() {
        linker.func_wrap(HOST_MODULE, host::READ_FILE, |mut c: Caller<'_, HostState>, p: i32| {
            let path = guest_str(&mut c, p)?;
            let room = c.data().limit.saturating_sub(c.data().host_used);
            let result = match c.data().perms.open(&path, Access::Read) {
                Ok(file) => match read_text(file, &path, room) {
                    Ok(r) => r,
                    Err(size) => {
                        let requested = c.data().host_used.saturating_add(size);
                        return Err(host_oom(&mut c, requested));
                    }
                },
                Err(e) => Err(e),
            };
            let (ok, text) = match result {
                Ok(s) => (1, s),
                Err(e) => (0, e),
            };
            charge(&mut c, text.len())?;
            Ok((ok, new_str(&mut c, &text)?))
        })?;
        linker.func_wrap(HOST_MODULE, host::WRITE_FILE, |mut c: Caller<'_, HostState>, p: i32, content: i32| {
            let path = guest_str(&mut c, p)?;
            let text = guest_str(&mut c, content)?;
            let b = c.data().budgets;
            let (writes, bytes) = (c.data().fs_writes + 1, c.data().fs_bytes.saturating_add(text.len()));
            if writes > b.fs_writes {
                return Err(budget_exceeded(&mut c, "fs_writes", b.fs_writes, writes));
            }
            if bytes > b.fs_write_bytes {
                return Err(budget_exceeded(&mut c, "fs_bytes", b.fs_write_bytes, bytes));
            }
            c.data_mut().fs_writes = writes;
            let result = c.data().perms.open(&path, Access::Write).and_then(|mut file| file.write_all(text.as_bytes()).map_err(|e| format!("{path}: {e}")));
            match result {
                Ok(()) => {
                    c.data_mut().fs_bytes = bytes;
                    Ok((1, 0))
                }
                Err(e) => Ok((0, new_str(&mut c, &e)?)),
            }
        })?;
    }
    if !perms.net.is_empty() {
        linker.func_wrap(HOST_MODULE, host::NET_HTTP, |mut c: Caller<'_, HostState>, call: i32| net_call(&mut c, call))?;
    }
    if perms.db.is_some() {
        linker.func_wrap(HOST_MODULE, host::DB_EXEC, |mut c: Caller<'_, HostState>, sql: i32, params: i32| db_call(&mut c, sql, params, false))?;
        linker.func_wrap(HOST_MODULE, host::DB_QUERY, |mut c: Caller<'_, HostState>, sql: i32, params: i32| db_call(&mut c, sql, params, true))?;
    }
    if perms.time {
        linker.func_wrap(HOST_MODULE, host::NOW_MS, || {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
        })?;
    }
    if perms.rand {
        linker.func_wrap(HOST_MODULE, host::RAND_INT, |mut c: Caller<'_, HostState>, lo: i64, hi: i64| {
            let span = (hi as u64).wrapping_sub(lo as u64).wrapping_add(1);
            let r = next_random(&mut c.data_mut().rng);
            let offset = if span == 0 { r } else { r % span };
            lo.wrapping_add(offset as i64)
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_permissions() {
        let p = Permissions::parse("io, rand").unwrap();
        assert!(p.allows("io") && p.allows("rand") && !p.allows("fs") && !p.allows("time"));
        let dir = std::env::temp_dir();
        let p = Permissions::parse(&format!("fs:{}", dir.display())).unwrap();
        assert!(p.allows("fs"));
        assert_eq!(p.fs[0].path, dir.canonicalize().unwrap());
        assert!(Permissions::parse("fs").unwrap_err().contains("needs a directory"));
        assert!(Permissions::parse("web").unwrap_err().contains("unknown effect `web`"));
        assert!(Permissions::parse("fs:/definitely/not/here").is_err());
        assert_eq!(Permissions::parse("").unwrap(), Permissions::default());
    }

    #[test]
    fn random_numbers_stay_in_range() {
        let mut s = 1;
        for _ in 0..1000 {
            let span = 6u64;
            let v = 1 + (next_random(&mut s) % span) as i64;
            assert!((1..=6).contains(&v));
        }
    }
}
