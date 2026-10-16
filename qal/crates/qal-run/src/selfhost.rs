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
    /// The compiler's answer to `req`, or the runtime error that ended the call.
    fn invoke(&self, req: &[u8]) -> Result<Vec<u8>, String>;
}

impl<F: Fn(&[u8]) -> Result<Vec<u8>, String> + Sync> Invoke for F {
    fn invoke(&self, req: &[u8]) -> Result<Vec<u8>, String> {
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

/// Runs `req` to its final answer (`DONE` payload or `DIAGNOSTICS`), following forks.
pub fn drive(req: Vec<u8>, jobs: usize, limits: &Limits, compiler: &dyn Invoke) -> Result<(Response, DriveStats), DriveError> {
    let start = Instant::now();
    let jobs = jobs.clamp(1, 64);
    let mut stats = DriveStats::default();
    let mut req = req;
    loop {
        if start.elapsed() > limits.total_timeout {
            return Err(DriveError::Limit(format!("the compilation took longer than {} s", limits.total_timeout.as_secs())));
        }
        stats.calls += 1;
        let item = if stats.rounds == 0 { "main".to_string() } else { format!("join {}", stats.rounds) };
        let bytes = compiler.invoke(&req).map_err(|error| DriveError::Failed { item: item.clone(), error })?;
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
                let results = run_items(&items, jobs, compiler, stats.rounds)?;
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

/// Runs the items on up to `jobs` threads; results in item order. An item's answer must be valid
/// `QALR` (it is passed on unread otherwise); the first failure in item order is reported.
fn run_items(items: &[Vec<u8>], jobs: usize, compiler: &dyn Invoke, round: usize) -> Result<Vec<Vec<u8>>, DriveError> {
    let slots: Vec<Mutex<Option<Result<Vec<u8>, DriveError>>>> = items.iter().map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let work = || loop {
        let i = next.fetch_add(1, Ordering::Relaxed);
        let Some(item) = items.get(i) else { break };
        let r = compiler.invoke(item).map_err(|error| DriveError::Failed { item: format!("round {round} item {i}"), error }).and_then(|b| decode(&b).map(|_| b));
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
}

impl Compiler {
    /// `module` must export `qc_main(Bytes) -> Bytes`, be pure and be compiled with
    /// `memory_max_pages` matching `limits.memory_bytes`.
    pub fn new(module: CompiledModule, limits: &Limits, jobs: usize) -> Result<Compiler, String> {
        let f = module.func(ENTRY).ok_or_else(|| format!("the compiler module does not export `{ENTRY}`"))?;
        if f.params.len() != 1 || !f.effects.is_empty() {
            return Err(format!("`{ENTRY}` must be a pure fn(Bytes) -> Bytes"));
        }
        let runner = Runner::prepared(module, Permissions::default(), limits.memory_bytes, jobs.max(1))?
            .with_fuel(limits.fuel)
            .with_timeout(limits.item_timeout)
            .with_exact_results(limits.max_round_bytes);
        Ok(Compiler { runner })
    }

    /// One call in a fresh instance on the calling thread (which needs [`GUEST_THREAD_STACK`]),
    /// with the fuel it used and the instance's final memory size.
    pub fn call_here(&self, req: &[u8]) -> (Result<Vec<u8>, RunError>, u64, usize) {
        let out = self.runner.execute(ENTRY, &[Value::Bytes(req.to_vec(), 0)], ExecIn::default(), &commit_if_ok);
        let r = match out.outcome.result {
            Ok(Value::Bytes(b, 0)) => Ok(b),
            Ok(other) => Err(RunError::new("bad_response", ENTRY, format!("`{ENTRY}` returned {other:?}"))),
            Err(e) => Err(e),
        };
        (r, out.fuel, out.mem_bytes)
    }
}

impl Invoke for Compiler {
    fn invoke(&self, req: &[u8]) -> Result<Vec<u8>, String> {
        let run = || self.call_here(req).0;
        let r = std::thread::scope(|scope| match std::thread::Builder::new().name("qal-compiler".into()).stack_size(GUEST_THREAD_STACK).spawn_scoped(scope, run) {
            Ok(h) => h.join().unwrap_or_else(|_| Err(RunError::new("trap", ENTRY, "the runtime panicked"))),
            Err(e) => Err(RunError::new("trap", ENTRY, format!("cannot start the execution thread: {e}"))),
        });
        r.map_err(|e| format!("{} ({}): {}", e.kind, e.code, e.message))
    }
}
