//! Running a compiler module (self-hosting spec §4.1, §4.3, §6.3).
//!
//! The compiler is a pure function `qc_main(req: Bytes) -> Bytes` speaking `QALQ`/`QALR`
//! ([`qal_compiler::proto`]). [`drive`] is the host side of the fork/join protocol: it calls the
//! compiler, and while the answer is `FORK continuation items`, it runs every item in a fresh
//! instance (in parallel on up to `jobs` threads), then sends `JOIN continuation results` with the
//! results in item order. The host understands nothing of the content, and the output never
//! depends on `jobs`. Limits are [`Limits::compiler`]; breaking one is E0954 `protocol_limit`, a
//! failing compiler E0951 `compiler_failed`, a malformed answer E0952 `bad_compiler_response` —
//! always a diagnostic, never a crash of the host.

use crate::{commit_if_ok, ExecIn, Permissions, RunError, Runner, Value, GUEST_THREAD_STACK};
use qal_compiler::proto::{self, Response};
use qal_compiler::CompiledModule;
use qal_core::diag::Diagnostic;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// The export every compiler module has.
pub const ENTRY: &str = "qc_main";

/// Limits of compiler runs (spec §6.3) and of the fork/join protocol (spec §4.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limits {
    /// Memory of one work item (the module must be compiled with the matching `memory_max_pages`).
    pub memory_bytes: usize,
    /// Instructions of one work item.
    pub fuel: u64,
    /// Wall-clock time of one work item.
    pub item_timeout: Duration,
    /// Wall-clock time of a whole [`drive`].
    pub total_timeout: Duration,
    /// Fork rounds of one drive.
    pub max_rounds: usize,
    /// Items of one fork.
    pub max_items: usize,
    /// Bytes of the items, and of their results, of one round.
    pub max_round_bytes: usize,
}

impl Limits {
    /// `Limits::compiler()`: 1 GiB, 2·10^11 fuel and 300 s per work item, 600 s in total, at most
    /// 8 rounds of at most 100 000 items and 256 MiB. (The wasm stack is the runtime's 64 MiB.)
    pub fn compiler() -> Limits {
        Limits {
            memory_bytes: (qal_compiler::COMPILER_MAX_PAGES as usize) << 16,
            fuel: 200_000_000_000,
            item_timeout: Duration::from_secs(300),
            total_timeout: Duration::from_secs(600),
            max_rounds: 8,
            max_items: 100_000,
            max_round_bytes: 256 << 20,
        }
    }
}

/// Parallel work items by default: the cores, at most 8 (spec §4.3).
pub fn default_jobs() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).clamp(1, 8)
}

/// One call of the compiler in a fresh instance.
pub trait Invoke: Sync {
    /// The compiler's answer to `req` within `timeout`, or the runtime error that ended the call.
    fn invoke(&self, req: &[u8], timeout: Duration) -> Result<Vec<u8>, String>;

    /// How many calls may run at once (a prepared runner has a fixed number of instance slots).
    fn max_parallel(&self) -> usize {
        usize::MAX
    }
}

/// A plain function as a compiler (tests, the Rust reference server); it ignores the timeout.
impl<F: Fn(&[u8]) -> Result<Vec<u8>, String> + Sync> Invoke for F {
    fn invoke(&self, req: &[u8], _timeout: Duration) -> Result<Vec<u8>, String> {
        self(req)
    }
}

/// What ended a drive without an answer.
#[derive(Debug, Clone, PartialEq)]
pub enum DriveError {
    /// The compiler trapped, hit a limit or answered `INTERNAL` (E0951); `item` says where.
    Failed { item: String, error: String },
    /// The answer is not valid `QALR` (E0952).
    Malformed(String),
    /// A protocol limit (E0954).
    Limit(String),
}

impl DriveError {
    pub fn diagnostic(&self, stage: &str, op: &str) -> Diagnostic {
        match self {
            DriveError::Failed { item, error } => Diagnostic::compiler_failed(stage, op, item, error),
            DriveError::Malformed(e) => Diagnostic::bad_compiler_response(e),
            DriveError::Limit(e) => Diagnostic::protocol_limit(e),
        }
    }
}

/// Statistics of a drive (for the measurement ledgers).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DriveStats {
    pub rounds: usize,
    pub work_items: usize,
    pub calls: usize,
}

fn decode(bytes: &[u8]) -> Result<Response, DriveError> {
    Response::decode(bytes).map_err(DriveError::Malformed)
}

/// Runs `req` to its final answer (`DONE` payload or `DIAGNOSTICS`), following forks. The whole
/// drive, forks included, ends within `limits.total_timeout`: every call gets at most the time left.
pub fn drive(req: Vec<u8>, jobs: usize, limits: &Limits, compiler: &dyn Invoke) -> Result<(Response, DriveStats), DriveError> {
    let deadline = Instant::now() + limits.total_timeout;
    let jobs = jobs.clamp(1, 64).min(compiler.max_parallel().max(1));
    let mut stats = DriveStats::default();
    let mut req = req;
    loop {
        let item = if stats.rounds == 0 { "main".to_string() } else { format!("join {}", stats.rounds) };
        let left = time_left(deadline, limits)?;
        stats.calls += 1;
        let bytes = compiler.invoke(&req, left.min(limits.item_timeout)).map_err(|error| DriveError::Failed { item: item.clone(), error })?;
        time_left(deadline, limits)?;
        match decode(&bytes)? {
            Response::Fork { continuation, items } => {
                stats.rounds += 1;
                if stats.rounds > limits.max_rounds {
                    return Err(DriveError::Limit(format!("more than {} fork rounds", limits.max_rounds)));
                }
                if items.len() > limits.max_items {
                    return Err(DriveError::Limit(format!("{} work items in one round (at most {})", items.len(), limits.max_items)));
                }
                let size: usize = items.iter().map(Vec::len).sum();
                if size > limits.max_round_bytes {
                    return Err(DriveError::Limit(format!("{size} bytes of work items in one round (at most {})", limits.max_round_bytes)));
                }
                let results = run_items(&items, jobs, compiler, stats.rounds, deadline, limits)?;
                let size: usize = results.iter().map(Vec::len).sum();
                if size > limits.max_round_bytes {
                    return Err(DriveError::Limit(format!("{size} bytes of results in one round (at most {})", limits.max_round_bytes)));
                }
                stats.work_items += items.len();
                stats.calls += items.len();
                req = proto::join(&continuation, &results);
            }
            Response::Internal(m) => return Err(DriveError::Failed { item, error: format!("internal compiler error: {m}") }),
            done => return Ok((done, stats)),
        }
    }
}

/// The time left until `deadline`, or the total-time limit error.
fn time_left(deadline: Instant, limits: &Limits) -> Result<Duration, DriveError> {
    match deadline.checked_duration_since(Instant::now()) {
        Some(d) if !d.is_zero() => Ok(d),
        _ => Err(DriveError::Limit(format!("the compilation took longer than {} s", limits.total_timeout.as_secs()))),
    }
}

/// Runs the items on up to `jobs` threads; results in item order. An item's answer must be valid
/// `QALR` (it is passed on unread otherwise). After a failure no later item is started (the first
/// failure in item order is reported, so earlier items still run); after the deadline none is.
fn run_items(items: &[Vec<u8>], jobs: usize, compiler: &dyn Invoke, round: usize, deadline: Instant, limits: &Limits) -> Result<Vec<Vec<u8>>, DriveError> {
    let slots: Vec<Mutex<Option<Result<Vec<u8>, DriveError>>>> = items.iter().map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let first_failure = AtomicUsize::new(usize::MAX);
    let work = || loop {
        let i = next.fetch_add(1, Ordering::Relaxed);
        let Some(item) = items.get(i) else { break };
        if i > first_failure.load(Ordering::Relaxed) {
            break;
        }
        let r = match time_left(deadline, limits) {
            Err(e) => Err(e),
            Ok(left) => compiler.invoke(item, left.min(limits.item_timeout)).map_err(|error| DriveError::Failed { item: format!("round {round} item {i}"), error }).and_then(|b| decode(&b).map(|_| b)),
        };
        if r.is_err() {
            first_failure.fetch_min(i, Ordering::Relaxed);
        }
        if let Ok(mut s) = slots[i].lock() {
            *s = Some(r);
        }
    };
    let threads = jobs.min(items.len()).max(1);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for k in 0..threads {
            match std::thread::Builder::new().name(format!("qal-compile-{k}")).stack_size(GUEST_THREAD_STACK).spawn_scoped(scope, work) {
                Ok(h) => handles.push(h),
                Err(_) if k > 0 => break,
                Err(e) => return Err(DriveError::Failed { item: format!("round {round}"), error: format!("cannot start a worker thread: {e}") }),
            }
        }
        for h in handles {
            if h.join().is_err() {
                return Err(DriveError::Failed { item: format!("round {round}"), error: "a worker thread panicked".into() });
            }
        }
        Ok(())
    })?;
    time_left(deadline, limits)?;
    slots
        .into_iter()
        .enumerate()
        .map(|(i, s)| match s.into_inner().ok().flatten() {
            Some(r) => r,
            None => Err(DriveError::Failed { item: format!("round {round} item {i}"), error: "the item did not run".into() }),
        })
        .collect()
}

/// A compiler module, linked once; every call runs in a fresh instance with no system access.
pub struct Compiler {
    runner: Runner,
    slots: usize,
}

impl Compiler {
    /// `module` must export a pure `qc_main(Bytes) -> Bytes` and declare a memory maximum equal to
    /// `limits.memory_bytes`; `slots` instances can run at once (the most `drive` will use).
    pub fn new(module: CompiledModule, limits: &Limits, slots: usize) -> Result<Compiler, String> {
        let f = module.func(ENTRY).ok_or_else(|| format!("the compiler module does not export `{ENTRY}`"))?;
        let bytes = qal_core::ast::Type::Bytes;
        if f.params.len() != 1 || f.params[0].1 != bytes || f.ret != bytes || !f.effects.is_empty() {
            return Err(format!("`{ENTRY}` must be a pure fn(Bytes) -> Bytes"));
        }
        let declared = (module.memory_max_pages as usize) << 16;
        if declared != limits.memory_bytes {
            return Err(format!("the compiler module allows {} MiB of memory, but its limits {} MiB (compile it with the matching memory_max_pages)", declared >> 20, limits.memory_bytes >> 20));
        }
        let slots = slots.max(1);
        let runner = Runner::prepared(module, Permissions::default(), limits.memory_bytes, slots)?
            .with_fuel(limits.fuel)
            .with_timeout(limits.item_timeout)
            .with_exact_results(limits.max_round_bytes);
        Ok(Compiler { runner, slots })
    }

    /// One call in a fresh instance on the calling thread (which needs [`GUEST_THREAD_STACK`]),
    /// within `timeout` (at most the item timeout), with the fuel it used and the final memory size.
    pub fn call_here(&self, req: &[u8], timeout: Duration) -> (Result<Vec<u8>, RunError>, u64, usize) {
        let out = self.runner.execute_within(ENTRY, &[Value::Bytes(req.to_vec(), 0)], ExecIn::default(), &commit_if_ok, timeout);
        let r = match out.outcome.result {
            Ok(Value::Bytes(b, 0)) => Ok(b),
            Ok(other) => Err(RunError::new("bad_response", ENTRY, format!("`{ENTRY}` returned {other:?}"))),
            Err(e) => Err(e),
        };
        (r, out.fuel, out.mem_bytes)
    }
}

impl Invoke for Compiler {
    fn invoke(&self, req: &[u8], timeout: Duration) -> Result<Vec<u8>, String> {
        let run = || self.call_here(req, timeout).0;
        let r = std::thread::scope(|scope| match std::thread::Builder::new().name("qal-compiler".into()).stack_size(GUEST_THREAD_STACK).spawn_scoped(scope, run) {
            Ok(h) => h.join().unwrap_or_else(|_| Err(RunError::new("trap", ENTRY, "the runtime panicked"))),
            Err(e) => Err(RunError::new("trap", ENTRY, format!("cannot start the execution thread: {e}"))),
        });
        r.map_err(|e| format!("{} ({}): {}", e.kind, e.code, e.message))
    }

    fn max_parallel(&self) -> usize {
        self.slots
    }
}

/// Initializes the `Qal` backend in `qal-compiler` to run against `module`.
pub fn init_qal_backend(module: CompiledModule, jobs: usize) -> Result<(), String> {
    let limits = Limits::compiler();
    let compiler = std::sync::Arc::new(Compiler::new(module, &limits, jobs)?);
    qal_compiler::set_qal_driver(Box::new(move |req: &[u8]| {
        let (res, _stats) = drive(req.to_vec(), jobs, &limits, compiler.as_ref())
            .map_err(|e| e.diagnostic("qal", "qc_main"))?;
        Ok(res.encode())
    }));
    Ok(())
}

