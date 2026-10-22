//! QAL runtime: executes compiled modules in wasmtime with capability-gated host functions and
//! reports every failure as data in the diagnostic format (code, kind, def, path, message, values, hints).

mod db;
mod fs;
mod guest;
mod host;
mod net;
pub mod selfhost;
mod value;

pub use db::{Db, DbGrant};
pub use fs::FsRoot;
pub use host::{Limits, Permissions};
pub use net::NetGrant;
pub use value::{compare as compare_values, from_json, Value, Values};

use guest::{EncodeError, Guest};
use host::{Failure, HostState, FAIL_BUDGET, FAIL_DENIED, FAIL_TIMEOUT};
use qal_core::ast::Type;
use qal_core::tast::TProgram;
use qal_compiler::print_expr;
use qal_compiler::{CompiledModule, FuncMeta, FAIL_BUILTIN, FAIL_DIV_ZERO, FAIL_ENS, FAIL_INDEX, FAIL_OOM, FAIL_OVERFLOW, FAIL_REQ, FAIL_TODO, OOM_SCRATCH};
use serde::Serialize;
use std::io::Cursor;
use std::sync::mpsc;
use std::time::Duration;
use std::sync::Arc;
use wasmtime::{Config, Engine, Instance, InstanceAllocationStrategy, InstancePre, Linker, Memory, Module, PoolingAllocationConfig, Store, Trap, Val};

/// Instruction budget per call; exhausting it reports `timeout` instead of hanging.
pub const DEFAULT_FUEL: u64 = 2_000_000_000;
/// Memory limit per execution (spec §8).
pub const DEFAULT_MEMORY_LIMIT: usize = 256 << 20;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RunError {
    pub code: &'static str,
    /// contract_violation | int_overflow | div_by_zero | out_of_memory | index | stack_overflow | timeout
    /// | permission_denied | bad_args | unknown_function | compile_error | bad_entry | bad_response
    /// | budget_exceeded | db_open | init_failed | todo_reached | trap; services also log request_rejected | overloaded
    /// | listen_failed | internal
    pub kind: &'static str,
    /// Name of the definition where the failure happened (never an export name like `test:x`).
    pub def: String,
    /// AST path inside `def` (e.g. `req[0]`, `body.left`); empty if unknown.
    pub path: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contract: Option<String>,
    /// Values involved: parameters (and `r`) for contracts, operands for arithmetic, arguments for built-ins.
    #[serde(skip_serializing_if = "Values::is_empty")]
    pub values: Values,
    pub hints: Vec<String>,
}

impl RunError {
    pub fn new(kind: &'static str, def: &str, message: impl Into<String>) -> RunError {
        let code = match kind {
            "contract_violation" => "E0501",
            "int_overflow" => "E0502",
            "div_by_zero" => "E0503",
            "out_of_memory" => "E0504",
            "index" => "E0505",
            "stack_overflow" => "E0506",
            "timeout" => "E0507",
            "permission_denied" => "E0508",
            "bad_args" => "E0509",
            "unknown_function" => "E0510",
            "compile_error" => "E0511",
            "bad_entry" => "E0520",
            "bad_response" => "E0521",
            "budget_exceeded" => "E0522",
            "db_open" => "E0523",
            "init_failed" => "E0524",
            "todo_reached" => "E0540",
            "request_rejected" => "E0800",
            "overloaded" => "E0801",
            "listen_failed" => "E0802",
            "internal" => "E0819",
            _ => "E0599",
        };
        RunError {
            code,
            kind,
            def: def.strip_prefix("test:").unwrap_or(def).to_string(),
            path: String::new(),
            message: message.into(),
            contract: None,
            values: Values::default(),
            hints: Vec::new(),
        }
    }

    fn at(mut self, path: &str) -> RunError {
        self.path = path.to_string();
        self
    }

    pub fn hint(mut self, h: impl Into<String>) -> RunError {
        self.hints.push(h.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TestResult {
    pub name: String,
    pub passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RunError>,
}

/// Result of one execution plus everything the program printed.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub result: Result<Value, RunError>,
    pub stdout: String,
    /// Bytes the program printed; more than `stdout.len()` if output beyond `with_stdout_limit` was dropped.
    pub stdout_bytes: usize,
    /// What happened to the execution's database transaction: `commit`, `rollback` or `none`.
    pub db: &'static str,
}

impl Outcome {
    fn failed(export: &str, message: String) -> Outcome {
        Outcome { result: Err(RunError::new("trap", export, message)), stdout: String::new(), stdout_bytes: 0, db: "none" }
    }
}

/// When an execution's database transaction is committed (spec Phase 3 §3.2): a function call
/// without a runtime error ...
pub fn commit_if_ok(r: &Result<Value, RunError>) -> bool {
    r.is_ok()
}

/// ... and an entry point `fn(List[Str]) -> Int` only with exit code 0.
pub fn commit_if_exit_zero(r: &Result<Value, RunError>) -> bool {
    matches!(r, Ok(Value::Int(0)))
}

/// Wasm call stack limit; every invocation runs on a thread with a larger native stack.
pub const WASM_STACK_BYTES: usize = 64 << 20;
/// Native stack of a thread that runs guest code (the wasm stack plus room for the host).
pub const GUEST_THREAD_STACK: usize = WASM_STACK_BYTES + (64 << 20);
/// Interval of the epoch ticker of a prepared runner; deadlines are whole ticks.
pub const EPOCH_TICK: Duration = Duration::from_millis(10);

/// What a caller hands one execution of a prepared runner ([`Runner::execute`]).
#[derive(Default)]
pub struct ExecIn {
    /// The connection to run on (a server worker's); the execution's transaction is begun and
    /// ended on it if the function uses `db`. None: opened per execution if needed.
    pub db: Option<Db>,
    /// Host bytes already held for this execution (a request's head and body).
    pub host_used: usize,
}

/// The result of [`Runner::execute`] plus what the caller needs back.
pub struct ExecOut {
    pub outcome: Outcome,
    /// The connection given in [`ExecIn::db`] (or opened), after the transaction ended.
    pub db: Option<Db>,
    /// Fuel the execution used.
    pub fuel: u64,
    /// Size of the guest's linear memory at the end.
    pub mem_bytes: usize,
}

/// Increments an engine's epoch every [`EPOCH_TICK`] until dropped: concurrent executions on one
/// engine each have their own deadline in ticks (a one-shot watchdog would stop them all).
struct Ticker {
    stop: Option<mpsc::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Ticker {
    fn start(engine: Engine) -> Result<Ticker, String> {
        let (stop, wait) = mpsc::channel::<()>();
        let thread = std::thread::Builder::new()
            .name("qal-epoch".into())
            .spawn(move || {
                while let Err(mpsc::RecvTimeoutError::Timeout) = wait.recv_timeout(EPOCH_TICK) {
                    engine.increment_epoch();
                }
            })
            .map_err(|e| format!("cannot start the epoch ticker: {e}"))?;
        Ok(Ticker { stop: Some(stop), thread: Some(thread) })
    }
}

impl Drop for Ticker {
    fn drop(&mut self) {
        drop(self.stop.take());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

pub struct Runner {
    engine: Engine,
    module: Module,
    meta: CompiledModule,
    types: TProgram,
    fuel: u64,
    memory_limit: usize,
    perms: Permissions,
    stdin: Option<Vec<u8>>,
    timeout: Option<Duration>,
    stdout_limit: Option<usize>,
    budgets: Limits,
    /// Prepared runners: the module linked once (per-execution stores instantiate it).
    pre: Option<InstancePre<HostState>>,
    ticker: Option<Arc<Ticker>>,
    pooled: bool,
    /// Results are decoded exactly up to this many bytes (else with display limits).
    exact_results: Option<usize>,
}

/// The engine settings every runner uses.
fn base_config() -> Config {
    let mut config = Config::new();
    config.wasm_tail_call(true);
    config.consume_fuel(true);
    // Interrupts a call that runs longer than its wall-clock timeout (see `with_timeout`).
    config.epoch_interruption(true);
    // Compile time must stay linear in the module size (a 150 000-element list literal took
    // minutes with the optimizing defaults): no optimization passes, single-pass register
    // allocation. Programs are bounded by fuel, not by speed.
    config.cranelift_opt_level(wasmtime::OptLevel::None);
    config.cranelift_regalloc_algorithm(wasmtime::RegallocAlgorithm::SinglePass);
    // The unoptimized frames are large: allow ~20 000+ nested non-tail calls; deeper recursion
    // reports `stack_overflow` (guest code runs on a thread with a larger native stack).
    config.max_wasm_stack(WASM_STACK_BYTES);
    config.async_stack_size(WASM_STACK_BYTES); // only validated against the limit; calls are synchronous
    config
}

/// Loads a WebAssembly module from precompiled cache if available and valid, or compiles and caches it.
fn load_module(engine: &Engine, wasm: &[u8]) -> Result<Module, String> {
    let hash = qal_core::canon::Hash::of(wasm).to_hex();
    let cache_dir = std::env::var_os("QAL_CACHE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("qal-cache"));
    let cache_file = cache_dir.join(format!("mod-{hash}.cwasm"));
    if let Ok(bytes) = std::fs::read(&cache_file) {
        if let Ok(m) = unsafe { Module::deserialize(engine, &bytes) } {
            return Ok(m);
        }
    }
    let m = Module::new(engine, wasm).map_err(|e| e.to_string())?;
    if let Ok(bytes) = m.serialize() {
        let _ = std::fs::create_dir_all(&cache_dir);
        let _ = std::fs::write(&cache_file, bytes);
    }
    Ok(m)
}

impl Runner {
    pub fn new(meta: CompiledModule) -> Result<Runner, String> {
        let engine = Engine::new(&base_config()).map_err(|e| e.to_string())?;
        let module = load_module(&engine, &meta.wasm)?;
        let types = TProgram { types: meta.types.clone(), ..TProgram::default() };
        Ok(Runner { engine, module, meta, types, fuel: DEFAULT_FUEL, memory_limit: DEFAULT_MEMORY_LIMIT, perms: Permissions::default(), stdin: None, timeout: None, stdout_limit: None, budgets: Limits::default(), pre: None, ticker: None, pooled: false, exact_results: None })
    }

    /// A runner for a server (spec Phase 3 §2.2-§2.4): the module is compiled and linked once with
    /// `perms`; every [`Runner::execute`] instantiates it in a fresh store (from wasmtime's pooling
    /// allocator with `slots` instance slots of at most `memory_limit` bytes, falling back to the
    /// on-demand allocator if the pool cannot be set up) and discards it afterwards, so nothing of
    /// one execution survives into the next. Wall-clock limits use an epoch ticker, so concurrent
    /// executions do not affect each other's deadlines.
    pub fn prepared(meta: CompiledModule, perms: Permissions, memory_limit: usize, slots: usize) -> Result<Runner, String> {
        // Every table slot counts: named functions used as values and each closure (Phase 2a).
        let table = meta.table_len.max(1);
        let mut pool = PoolingAllocationConfig::default();
        let slots = slots.clamp(1, 1024) as u32;
        pool.total_memories(slots).total_core_instances(slots).total_tables(slots);
        pool.max_memories_per_module(1).max_tables_per_module(1).table_elements(table).max_memory_size(memory_limit);
        let mut config = base_config();
        config.allocation_strategy(InstanceAllocationStrategy::Pooling(pool));
        let (engine, pooled) = match Engine::new(&config).ok().and_then(|e| load_module(&e, &meta.wasm).ok().map(|m| (e, m))) {
            Some((e, m)) => ((e, m), true),
            None => {
                let e = Engine::new(&base_config()).map_err(|e| e.to_string())?;
                let m = load_module(&e, &meta.wasm)?;
                ((e, m), false)
            }
        };
        let (engine, module) = engine;
        let mut linker = Linker::new(&engine);
        host::link(&mut linker, &perms).map_err(|e| e.to_string())?;
        let pre = linker.instantiate_pre(&module).map_err(|e| e.to_string())?;
        let ticker = Ticker::start(engine.clone())?;
        let types = TProgram { types: meta.types.clone(), ..TProgram::default() };
        Ok(Runner {
            engine,
            module,
            meta,
            types,
            fuel: DEFAULT_FUEL,
            memory_limit,
            perms,
            stdin: Some(Vec::new()),
            timeout: None,
            stdout_limit: None,
            budgets: Limits::default(),
            pre: Some(pre),
            ticker: Some(Arc::new(ticker)),
            pooled,
            exact_results: None,
        })
    }

    /// True if executions use wasmtime's pooling allocator (see [`Runner::prepared`]).
    pub fn pooled(&self) -> bool {
        self.pooled
    }

    /// Decodes results exactly (no shortened strings) up to `bytes`; larger results are `bad_response`.
    pub fn with_exact_results(mut self, bytes: usize) -> Runner {
        self.exact_results = Some(bytes);
        self
    }

    pub fn permissions(&self) -> &Permissions {
        &self.perms
    }

    pub fn memory_limit(&self) -> usize {
        self.memory_limit
    }

    pub fn limits(&self) -> Limits {
        self.budgets
    }

    /// Instantiates the module in a fresh store and discards it (the per-request overhead of a
    /// prepared runner, measured by the performance gate).
    pub fn instantiate_only(&self) -> Result<(), String> {
        let mut store = Store::new(&self.engine, HostState::new(self.perms.clone(), Box::new(Cursor::new(Vec::new())), self.memory_limit, self.budgets));
        store.limiter(|s| &mut s.limits);
        store.set_fuel(self.fuel).map_err(|e| e.to_string())?;
        store.set_epoch_deadline(u64::MAX / 2);
        match &self.pre {
            Some(pre) => pre.instantiate(&mut store).map(|_| ()).map_err(|e| e.to_string()),
            None => Err("not a prepared runner".into()),
        }
    }

    pub fn with_fuel(mut self, fuel: u64) -> Runner {
        self.fuel = fuel;
        self
    }

    pub fn with_memory_limit(mut self, bytes: usize) -> Runner {
        self.memory_limit = bytes;
        self
    }

    /// Per-execution budgets besides fuel, time and memory (`fs` writes, ...).
    pub fn with_limits(mut self, budgets: Limits) -> Runner {
        self.budgets = budgets;
        self
    }

    pub fn with_permissions(mut self, perms: Permissions) -> Runner {
        self.perms = perms;
        self
    }

    /// Wall-clock limit per call; exceeding it reports `timeout` (fuel bounds instructions, this
    /// also bounds time spent in host functions and large memory copies). The epoch is engine-wide:
    /// concurrent calls on one `Runner` interfere with each other's deadline, so use one call at a time.
    pub fn with_timeout(mut self, timeout: Duration) -> Runner {
        self.timeout = Some(timeout);
        self
    }

    /// Keeps only the first `bytes` of printed output (the rest is counted, not stored), instead of
    /// failing with `out_of_memory` once the output reaches the memory limit.
    pub fn with_stdout_limit(mut self, bytes: usize) -> Runner {
        self.stdout_limit = Some(bytes);
        self
    }

    /// Feeds `read_line` from these bytes instead of the process's stdin.
    pub fn with_stdin(mut self, input: impl Into<Vec<u8>>) -> Runner {
        self.stdin = Some(input.into());
        self
    }

    pub fn meta(&self) -> &CompiledModule {
        &self.meta
    }

    /// Calls the function exported as `export` (a fn name, or `test:<name>`).
    pub fn call(&self, export: &str, args: &[Value]) -> Result<Value, RunError> {
        self.invoke(export, args).result
    }

    /// Runs the entry point `main(args: List[Str]) -> Int`; the result is the exit code.
    pub fn run_main(&self, args: &[String]) -> Outcome {
        self.run_entry("main", args)
    }

    /// Runs an entry point `fn(List[Str]) -> Int`; its database work is committed only on exit code 0.
    pub fn run_entry(&self, export: &str, args: &[String]) -> Outcome {
        let list = Value::List(args.iter().map(|a| Value::Str(a.clone())).collect());
        self.invoke_with(export, &[list], &commit_if_exit_zero)
    }

    /// Runs every test definition in program order.
    pub fn run_tests(&self) -> Vec<TestResult> {
        self.meta
            .funcs
            .iter()
            .filter(|f| f.is_test)
            .map(|f| self.run_test(&f.export))
            .collect()
    }

    /// Runs one test by its export name (`test:<name>`).
    pub fn run_test(&self, export: &str) -> TestResult {
        let name = export.strip_prefix("test:").unwrap_or(export).to_string();
        match self.call(export, &[]) {
            Ok(Value::Bool(passed)) => TestResult { name, passed, error: None },
            Ok(other) => {
                let e = RunError::new("trap", &name, format!("test returned {other:?} instead of Bool"));
                TestResult { name, passed: false, error: Some(e) }
            }
            Err(e) => TestResult { name, passed: false, error: Some(e) },
        }
    }

    /// Runs one invocation on a dedicated thread whose native stack exceeds the wasm stack limit,
    /// so deep guest recursion reports `stack_overflow` whatever thread the caller is on.
    pub fn invoke(&self, export: &str, args: &[Value]) -> Outcome {
        self.invoke_with(export, args, &commit_if_ok)
    }

    /// Like [`Runner::invoke`]; `commit` decides from the result whether the database transaction
    /// (one per execution, only if the function uses `db`) is committed.
    pub fn invoke_with(&self, export: &str, args: &[Value], commit: &(dyn Fn(&Result<Value, RunError>) -> bool + Sync)) -> Outcome {
        std::thread::scope(|scope| {
            let spawned = std::thread::Builder::new().name("qal-guest".into()).stack_size(GUEST_THREAD_STACK).spawn_scoped(scope, || self.invoke_here(export, args, commit, ExecIn::default(), self.timeout).outcome);
            match spawned {
                Ok(h) => h.join().unwrap_or_else(|_| Outcome::failed(export, "the runtime panicked while running the program".into())),
                Err(e) => Outcome::failed(export, format!("cannot start the execution thread: {e}")),
            }
        })
    }

    /// Runs one execution of a prepared runner on the calling thread, which must have a stack of
    /// at least [`GUEST_THREAD_STACK`] bytes (server workers do).
    pub fn execute(&self, export: &str, args: &[Value], input: ExecIn, commit: &dyn Fn(&Result<Value, RunError>) -> bool) -> ExecOut {
        self.invoke_here(export, args, commit, input, self.timeout)
    }

    /// [`Runner::execute`] with the wall-clock limit lowered to `timeout` for this execution (a
    /// compiler work item gets at most what is left of its whole compilation).
    pub fn execute_within(&self, export: &str, args: &[Value], input: ExecIn, commit: &dyn Fn(&Result<Value, RunError>) -> bool, timeout: Duration) -> ExecOut {
        let t = Some(self.timeout.map_or(timeout, |own| own.min(timeout)));
        self.invoke_here(export, args, commit, input, t)
    }

    fn invoke_here(&self, export: &str, args: &[Value], commit: &dyn Fn(&Result<Value, RunError>) -> bool, input: ExecIn, timeout: Option<Duration>) -> ExecOut {
        let stdin: Box<dyn std::io::BufRead + Send> = match &self.stdin {
            Some(bytes) => Box::new(Cursor::new(bytes.clone())),
            None => Box::new(std::io::BufReader::new(std::io::stdin())),
        };
        let mut state = HostState::new(self.perms.clone(), stdin, self.memory_limit, self.budgets);
        state.stdout_limit = self.stdout_limit;
        state.deadline = timeout.map(|t| std::time::Instant::now() + t);
        state.timeout = timeout;
        state.host_used = input.host_used;
        let early = |outcome: Outcome, db: Option<Db>| ExecOut { outcome, db, fuel: 0, mem_bytes: 0 };
        let uses_db = self.meta.func(export).is_some_and(|m| m.effects.iter().any(|e| e == "db"));
        let mut given = input.db;
        if let (true, Some(grant)) = (uses_db, &self.perms.db) {
            let opened = match given.take() {
                Some(mut db) => match db.begin() {
                    Ok(()) => Ok(db),
                    Err(e) => {
                        let m = format!("cannot start the transaction: {}", db::describe(&e));
                        return early(Outcome { result: Err(RunError::new("timeout", export, m)), stdout: String::new(), stdout_bytes: 0, db: "none" }, Some(db));
                    }
                },
                None => open_db(grant, &self.perms, self.budgets, export),
            };
            match opened {
                Ok(db) => {
                    db.limit(state.deadline, self.memory_limit);
                    state.db = Some(db);
                }
                Err(e) => return early(Outcome { result: Err(e), stdout: String::new(), stdout_bytes: 0, db: "none" }, None),
            }
        }
        let mut store = Store::new(&self.engine, state);
        store.limiter(|s| &mut s.limits);
        let watchdog = match (&self.ticker, timeout) {
            (Some(_), Some(t)) => {
                // Whole ticks, rounded up, plus one: the current tick may be about to end.
                let ticks = t.as_nanos().div_ceil(EPOCH_TICK.as_nanos()) as u64 + 1;
                store.set_epoch_deadline(ticks);
                None
            }
            (Some(_), None) => {
                store.epoch_deadline_trap();
                store.set_epoch_deadline(u64::MAX / 2);
                None
            }
            (None, t) => {
                // Relative to the engine's current epoch, so ticks of earlier watchdogs do not count.
                store.set_epoch_deadline(1);
                match t.map(|t| Watchdog::start(self.engine.clone(), t)).transpose() {
                    Ok(w) => w,
                    Err(m) => return early(Outcome::failed(export, m), store.data_mut().db.take()),
                }
            }
        };
        let mut result = self.invoke_in(&mut store, export, args);
        drop(watchdog);
        let fuel = self.fuel.saturating_sub(store.get_fuel().unwrap_or(0));
        let mut db = store.data_mut().db.take();
        let label = finish_db_ref(db.as_mut(), &mut result, commit, export);
        let stdout_bytes = store.data().stdout_bytes.max(store.data().stdout.len());
        let stdout = String::from_utf8_lossy(&store.data().stdout).into_owned();
        let mem_bytes = store.data().mem_bytes;
        // A connection the caller gave but this function did not need goes back unchanged.
        ExecOut { outcome: Outcome { result, stdout, stdout_bytes, db: label }, db: db.or(given), fuel, mem_bytes }
    }

    fn invoke_in(&self, store: &mut Store<HostState>, export: &str, args: &[Value]) -> Result<Value, RunError> {
        let meta = self.meta.func(export).ok_or_else(|| {
            RunError::new("unknown_function", export, format!("no function `{export}` in this module"))
                .hint("only functions reachable from the compiled entry point can be called")
        })?;
        let missing: Vec<&String> = meta.effects.iter().filter(|e| !self.perms.allows(e)).collect();
        if !missing.is_empty() {
            let list: Vec<&str> = missing.iter().map(|s| s.as_str()).collect();
            let flag = list.iter().map(|e| grant_flag(e)).collect::<Vec<_>>().join(",");
            return Err(RunError::new("permission_denied", &meta.name, format!("`{}` needs effect(s) {} which were not granted", meta.name, list.join(", ")))
                .hint(format!("grant them with `--allow {flag}`")));
        }
        if meta.params.len() != args.len() {
            return Err(RunError::new("bad_args", &meta.name, format!("`{}` takes {} argument(s), {} given", meta.name, meta.params.len(), args.len())));
        }
        store.set_fuel(self.fuel).map_err(|e| RunError::new("trap", &meta.name, e.to_string()))?;
        let instance = match &self.pre {
            Some(pre) => pre.instantiate(&mut *store),
            None => {
                let mut linker = Linker::new(&self.engine);
                host::link(&mut linker, &self.perms).map_err(|e| RunError::new("trap", &meta.name, e.to_string()))?;
                linker.instantiate(&mut *store, &self.module)
            }
        }
        .map_err(|e| RunError::new("trap", &meta.name, format!("cannot instantiate the module: {e}")))?;
        let mem = memory_of(store, &instance).map_err(|m| RunError::new("trap", &meta.name, m))?;
        let mut params = Vec::new();
        for (i, (a, (pname, t))) in args.iter().zip(&meta.params).enumerate() {
            let slot = match self.encode(store, &instance, t, a) {
                Ok(slot) => slot,
                Err(e) => {
                    return Err(match (store.data().failure, e) {
                        (Some(f), _) => self.describe(store, mem, &meta.name, f),
                        (None, EncodeError::Mismatch(m)) => RunError::new("bad_args", &meta.name, format!("argument {i} (`{pname}`): {m}")),
                        (None, EncodeError::Trap(e)) => RunError::new("trap", &meta.name, e.to_string()),
                    })
                }
            };
            params.push(match t {
                Type::Int => Val::I64(slot),
                Type::Float => Val::F64(slot as u64),
                _ => Val::I32(slot as i32),
            });
        }
        let func = instance.get_func(&mut *store, export).ok_or_else(|| RunError::new("unknown_function", export, format!("no export `{export}`")))?;
        let mut results = [Val::I32(0)];
        match func.call(&mut *store, &params, &mut results) {
            Ok(()) => {
                let slot = match results[0] {
                    Val::I64(v) => v,
                    Val::F64(bits) => bits as i64,
                    Val::I32(v) => v as u32 as i64,
                    ref other => return Err(RunError::new("trap", &meta.name, format!("unexpected result {other:?}"))),
                };
                store.data_mut().mem_bytes = mem.data_size(&*store);
                match self.exact_results {
                    Some(max) => value::decode_exact(&self.types, mem.data(&*store), &meta.ret, slot, max).map_err(|m| RunError::new("bad_response", &meta.name, format!("the result cannot be used: {m}"))),
                    None => Ok(value::decode(&self.types, mem.data(&*store), &meta.ret, slot)),
                }
            }
            Err(e) => Err({
                store.data_mut().mem_bytes = mem.data_size(&*store);
                match store.data().failure {
                Some(f) => self.describe(store, mem, &meta.name, f),
                None => match e.downcast_ref::<Trap>() {
                    Some(Trap::StackOverflow) => RunError::new("stack_overflow", &meta.name, "call stack exhausted")
                        .hint("make the recursion tail-recursive: the recursive call must be the last thing the function does (functions with `ens` never use tail calls)"),
                    Some(Trap::OutOfFuel) => RunError::new("timeout", &meta.name, format!("instruction budget of {} exhausted", self.fuel))
                        .hint("look for unbounded recursion"),
                    Some(Trap::Interrupt) => {
                        let ms = store.data().timeout.map_or(0, |t| t.as_millis());
                        RunError::new("timeout", &meta.name, format!("wall-clock limit of {ms} ms exceeded")).hint("look for unbounded recursion or very large intermediate values")
                    }
                    _ => RunError::new("trap", &meta.name, e.to_string()),
                },
            }
            }),
        }
    }

    /// Writes an argument into guest memory (any type; heap values are allocated) and returns its slot.
    fn encode(&self, store: &mut Store<HostState>, instance: &Instance, t: &Type, v: &Value) -> Result<i64, EncodeError> {
        let alloc = instance.get_typed_func::<i64, i32>(&mut *store, "qal_alloc").map_err(EncodeError::Trap)?;
        let mem = memory_of(store, instance).map_err(|m| EncodeError::Trap(wasmtime::Error::msg(m)))?;
        Guest::new(mem, alloc).encode(store, &self.types, t, v)
    }

    fn values(&self, store: &Store<HostState>, mem: Memory, names: &[(String, Type)], ptr: i32) -> Values {
        let data = mem.data(store);
        Values(
            names
                .iter()
                .enumerate()
                .map(|(k, (n, t))| {
                    let slot = value::slot_at(data, ptr as usize + 8 * k).unwrap_or(0);
                    (n.clone(), value::decode(&self.types, data, t, slot))
                })
                .collect(),
        )
    }

    /// Turns a failure reported through `qal_host.fail` into a RunError with values read from guest memory.
    fn describe(&self, store: &Store<HostState>, mem: Memory, entry: &str, f: Failure) -> RunError {
        if f.kind == FAIL_OOM {
            let requested = value::slot_at(mem.data(store), OOM_SCRATCH as usize);
            let host = store.data().host_oom;
            let mut e = RunError::new("out_of_memory", entry, format!("out of memory: the limit is {} bytes per execution", self.memory_limit))
                .hint("memory is never freed during one execution; avoid building large intermediate lists or strings in loops (`push` and `++` copy)");
            if host {
                e = e.hint("the host's buffers for this execution (output, file contents, request and response bodies, database results) count against the same limit");
            }
            if let Some(r) = requested {
                e.values = Values(vec![("requested_bytes".into(), Value::Int(r)), ("limit_bytes".into(), Value::Int(self.memory_limit as i64))]);
                if host {
                    e.values.0.push(("host".into(), Value::Bool(true)));
                }
            }
            return e;
        }
        if f.kind == FAIL_TIMEOUT {
            let ms = store.data().timeout.map_or(0, |t| t.as_millis());
            return RunError::new("timeout", entry, format!("wall-clock limit of {ms} ms exceeded")).hint("a database statement ran past the limit; look for unbounded recursive queries or missing indexes");
        }
        if f.kind == FAIL_BUDGET {
            let Some(b) = store.data().budget_hit else {
                return RunError::new("trap", entry, "a budget was exceeded, but none is recorded");
            };
            let what = match b.kind {
                "fs_bytes" => format!("`write_file` may write at most {} bytes per execution", b.budget),
                "fs_writes" => format!("`write_file` may be called at most {} times per execution", b.budget),
                other => format!("the `{other}` budget of {} is used up", b.budget),
            };
            let mut e = RunError::new("budget_exceeded", entry, format!("budget exceeded: {what}"));
            e.values = Values(vec![("budget".into(), Value::Int(b.budget as i64)), ("used".into(), Value::Int(b.used as i64)), ("kind".into(), Value::Str(b.kind.into()))]);
            return e;
        }
        if f.kind == FAIL_DENIED {
            let name = qal_compiler::host::EFFECT_FNS.get(f.func as usize).copied().unwrap_or("?");
            let effect = qal_compiler::host::effect(name).unwrap_or("?");
            let flag = grant_flag(effect);
            return RunError::new("permission_denied", entry, format!("`{name}` needs effect {effect} which was not granted"))
                .hint(format!("grant it with `--allow {flag}`"));
        }
        let site = usize::try_from(f.func).ok().and_then(|i| self.meta.funcs.get(i)).and_then(|m| Some((m, m.sites.get(usize::try_from(f.site).ok()?)?)));
        let Some((meta, site)) = site else {
            return RunError::new("trap", entry, format!("failure kind {} reported for an unknown site ({}, {})", f.kind, f.func, f.site));
        };
        let meta: &FuncMeta = meta;
        let values = self.values(store, mem, &site.values, f.vals);
        let def = meta.name.as_str();
        let mut e = match f.kind {
            FAIL_REQ | FAIL_ENS => {
                let (label, list) = if f.kind == FAIL_REQ { ("req", &meta.req) } else { ("ens", &meta.ens) };
                let index = site.contract.unwrap_or(0);
                let text = list.get(index).map(print_expr).unwrap_or_default();
                let mut e = RunError::new("contract_violation", def, format!("`{label} {text}` of `{def}` is false"));
                e.contract = Some(format!("{label}[{index}]: {text}"));
                if f.kind == FAIL_REQ {
                    e.hint(format!("the caller must ensure `{text}` before calling `{def}`"))
                } else {
                    e.hint(format!("`{def}` returned a result that violates its postcondition; fix the body or the contract"))
                }
            }
            FAIL_OVERFLOW => RunError::new("int_overflow", def, format!("Int overflow in `{}` (64-bit)", site.label))
                .hint("check the operand range before the operation"),
            FAIL_DIV_ZERO => RunError::new("div_by_zero", def, format!("division by zero in `{}`", site.label))
                .hint("check the divisor first, e.g. `if b == 0 then ... else a / b`"),
            FAIL_INDEX => RunError::new("index", def, format!("`{}` index out of range", site.label)).hint(if site.label == "bytes_slice" {
                "valid byte indices satisfy 0 <= start <= end <= bytes_len(b)"
            } else {
                "valid character indices satisfy 0 <= start <= end <= len(s)"
            }),
            FAIL_BUILTIN => {
                let mut e = RunError::new("contract_violation", def, format!("precondition of a built-in was violated: `{}`", site.label));
                e.contract = Some(site.label.clone());
                e
            }
            FAIL_TODO => RunError::new("todo_reached", def, format!("`todo()` reached in `{def}` (at `{}`): `{def}` is a stub", if site.path.is_empty() { "body" } else { site.path.as_str() }))
                .hint(format!("`{def}` is not implemented yet: replace its `todo()` with the real code (same signature, so its users relink automatically)")),
            other => RunError::new("trap", def, format!("unknown failure kind {other}")),
        };
        e.values = values;
        e.at(&site.path)
    }
}

/// Opens the database of an execution and starts its transaction.
pub fn open_db(grant: &DbGrant, perms: &Permissions, budgets: Limits, def: &str) -> Result<Db, RunError> {
    let mut db = Db::open(grant, &perms.db_dir(), budgets.db_max_mib).map_err(|m| RunError::new("db_open", def, m).hint("the database file is chosen by the host: `db` is .qal/db/main.sqlite, `db:<name>` is .qal/db/<name>.sqlite, `db:mem` is in memory"))?;
    db.begin().map_err(|e| RunError::new("timeout", def, format!("cannot start the transaction: {}", db::describe(&e))))?;
    Ok(db)
}

/// Ends the transaction of an execution by `commit(result)`; a failed commit turns the result into
/// an error. Returns `commit`, `rollback` or `none`.
pub fn finish_db_ref(db: Option<&mut Db>, result: &mut Result<Value, RunError>, commit: &dyn Fn(&Result<Value, RunError>) -> bool, def: &str) -> &'static str {
    let Some(db) = db else { return "none" };
    match db.finish(commit(result)) {
        Ok(what) => what,
        Err(m) => {
            if result.is_ok() {
                *result = Err(RunError::new("trap", def, m));
            }
            "rollback"
        }
    }
}

/// How an effect is granted on the command line (`--allow <flag>`).
fn grant_flag(effect: &str) -> &str {
    match effect {
        "fs" => "fs:<dir>",
        "net" => "net:<host>",
        other => other,
    }
}

/// The exported linear memory of an instance.
fn memory_of(store: &mut Store<HostState>, instance: &Instance) -> Result<Memory, String> {
    instance.get_memory(&mut *store, "memory").ok_or_else(|| "module has no memory export".to_string())
}

/// Increments the engine's epoch once `timeout` has passed, which interrupts the running call;
/// dropping the watchdog before that stops it.
struct Watchdog {
    stop: Option<mpsc::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Watchdog {
    fn start(engine: Engine, timeout: Duration) -> Result<Watchdog, String> {
        let (stop, wait) = mpsc::channel::<()>();
        let thread = std::thread::Builder::new()
            .name("qal-watchdog".into())
            .spawn(move || {
                if let Err(mpsc::RecvTimeoutError::Timeout) = wait.recv_timeout(timeout) {
                    engine.increment_epoch();
                }
            })
            .map_err(|e| format!("cannot start the wall-clock watchdog: {e}"))?;
        Ok(Watchdog { stop: Some(stop), thread: Some(thread) })
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        drop(self.stop.take());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests;
