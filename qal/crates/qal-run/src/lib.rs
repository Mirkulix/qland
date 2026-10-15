//! QAL runtime: executes compiled modules in wasmtime with capability-gated host functions and
//! reports every failure as data in the diagnostic format (code, kind, def, path, message, values, hints).

mod host;
mod value;

pub use host::Permissions;
pub use value::{compare as compare_values, from_json, Value, Values};

use host::{Failure, HostState, FAIL_DENIED};
use qal_core::ast::Type;
use qal_core::tast::TProgram;
use qal_text::print_expr;
use qal_wasm::{CompiledModule, FuncMeta, FAIL_BUILTIN, FAIL_DIV_ZERO, FAIL_ENS, FAIL_INDEX, FAIL_OOM, FAIL_OVERFLOW, FAIL_REQ, OOM_SCRATCH};
use serde::Serialize;
use std::io::Cursor;
use std::sync::mpsc;
use std::time::Duration;
use wasmtime::{Config, Engine, Instance, Linker, Memory, Module, Store, StoreLimitsBuilder, Trap, Val};

/// Instruction budget per call; exhausting it reports `timeout` instead of hanging.
pub const DEFAULT_FUEL: u64 = 2_000_000_000;
/// Memory limit per execution (spec §8).
pub const DEFAULT_MEMORY_LIMIT: usize = 256 << 20;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RunError {
    pub code: &'static str,
    /// contract_violation | int_overflow | div_by_zero | out_of_memory | index | stack_overflow | timeout
    /// | permission_denied | bad_args | unknown_function | compile_error | trap
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

    fn hint(mut self, h: impl Into<String>) -> RunError {
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
}

/// Wasm call stack limit; every invocation runs on a thread with a larger native stack.
pub const WASM_STACK_BYTES: usize = 64 << 20;

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
}

impl Runner {
    pub fn new(meta: CompiledModule) -> Result<Runner, String> {
        let mut config = Config::new();
        config.wasm_tail_call(true);
        config.consume_fuel(true);
        // Interrupts a call that runs longer than its wall-clock timeout (see `with_timeout`).
        config.epoch_interruption(true);
        // Compile time must stay linear in the module size (a 150 000-element list literal took
        // minutes with the optimizing defaults): no optimization passes, single-pass register
        // allocation. Phase 1 programs are bounded by fuel, not by speed.
        config.cranelift_opt_level(wasmtime::OptLevel::None);
        config.cranelift_regalloc_algorithm(wasmtime::RegallocAlgorithm::SinglePass);
        // The unoptimized frames are large: allow ~20 000+ nested non-tail calls; deeper recursion
        // reports `stack_overflow` (`invoke` runs guest code on a thread with a larger native stack).
        config.max_wasm_stack(WASM_STACK_BYTES);
        config.async_stack_size(WASM_STACK_BYTES); // only validated against the limit; calls are synchronous
        let engine = Engine::new(&config).map_err(|e| e.to_string())?;
        let module = Module::new(&engine, &meta.wasm).map_err(|e| e.to_string())?;
        let types = TProgram { types: meta.types.clone(), ..TProgram::default() };
        Ok(Runner { engine, module, meta, types, fuel: DEFAULT_FUEL, memory_limit: DEFAULT_MEMORY_LIMIT, perms: Permissions::default(), stdin: None, timeout: None, stdout_limit: None })
    }

    pub fn with_fuel(mut self, fuel: u64) -> Runner {
        self.fuel = fuel;
        self
    }

    pub fn with_memory_limit(mut self, bytes: usize) -> Runner {
        self.memory_limit = bytes;
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
        let list = Value::List(args.iter().map(|a| Value::Str(a.clone())).collect());
        self.invoke("main", &[list])
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
        let failed = |m: String| Outcome { result: Err(RunError::new("trap", export, m)), stdout: String::new(), stdout_bytes: 0 };
        std::thread::scope(|scope| {
            let spawned = std::thread::Builder::new().name("qal-guest".into()).stack_size(WASM_STACK_BYTES + (64 << 20)).spawn_scoped(scope, || self.invoke_here(export, args));
            match spawned {
                Ok(h) => h.join().unwrap_or_else(|_| failed("the runtime panicked while running the program".into())),
                Err(e) => failed(format!("cannot start the execution thread: {e}")),
            }
        })
    }

    fn invoke_here(&self, export: &str, args: &[Value]) -> Outcome {
        let stdin: Box<dyn std::io::BufRead + Send> = match &self.stdin {
            Some(bytes) => Box::new(Cursor::new(bytes.clone())),
            None => Box::new(std::io::BufReader::new(std::io::stdin())),
        };
        let state = HostState {
            failure: None,
            stdout: Vec::new(),
            stdin,
            perms: self.perms.clone(),
            rng: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1),
            limits: StoreLimitsBuilder::new().memory_size(self.memory_limit).build(),
            limit: self.memory_limit,
            stdout_limit: self.stdout_limit,
            stdout_bytes: 0,
        };
        let mut store = Store::new(&self.engine, state);
        store.limiter(|s| &mut s.limits);
        // Relative to the engine's current epoch, so ticks of earlier watchdogs do not count.
        store.set_epoch_deadline(1);
        let watchdog = match self.timeout.map(|t| Watchdog::start(self.engine.clone(), t)).transpose() {
            Ok(w) => w,
            Err(m) => return Outcome { result: Err(RunError::new("trap", export, m)), stdout: String::new(), stdout_bytes: 0 },
        };
        let result = self.invoke_in(&mut store, export, args);
        drop(watchdog);
        let stdout_bytes = store.data().stdout_bytes.max(store.data().stdout.len());
        let stdout = String::from_utf8_lossy(&store.data().stdout).into_owned();
        Outcome { result, stdout, stdout_bytes }
    }

    fn invoke_in(&self, store: &mut Store<HostState>, export: &str, args: &[Value]) -> Result<Value, RunError> {
        let meta = self.meta.func(export).ok_or_else(|| {
            RunError::new("unknown_function", export, format!("no function `{export}` in this module"))
                .hint("only functions reachable from the compiled entry point can be called")
        })?;
        let missing: Vec<&String> = meta.effects.iter().filter(|e| !self.perms.allows(e)).collect();
        if !missing.is_empty() {
            let list: Vec<&str> = missing.iter().map(|s| s.as_str()).collect();
            let flag = list.iter().map(|e| if *e == "fs" { "fs:<dir>" } else { e }).collect::<Vec<_>>().join(",");
            return Err(RunError::new("permission_denied", &meta.name, format!("`{}` needs effect(s) {} which were not granted", meta.name, list.join(", ")))
                .hint(format!("grant them with `--allow {flag}`")));
        }
        if meta.params.len() != args.len() {
            return Err(RunError::new("bad_args", &meta.name, format!("`{}` takes {} argument(s), {} given", meta.name, meta.params.len(), args.len())));
        }
        store.set_fuel(self.fuel).map_err(|e| RunError::new("trap", &meta.name, e.to_string()))?;
        let mut linker = Linker::new(&self.engine);
        host::link(&mut linker, &self.perms).map_err(|e| RunError::new("trap", &meta.name, e.to_string()))?;
        let instance = linker.instantiate(&mut *store, &self.module).map_err(|e| RunError::new("trap", &meta.name, e.to_string()))?;
        let mem = memory_of(store, &instance).map_err(|m| RunError::new("trap", &meta.name, m))?;
        let mut params = Vec::new();
        for (i, (a, (pname, t))) in args.iter().zip(&meta.params).enumerate() {
            let slot = match self.encode(store, &instance, t, a) {
                Ok(slot) => slot,
                Err(m) => {
                    return Err(match store.data().failure {
                        Some(f) => self.describe(store, mem, &meta.name, f),
                        None => RunError::new("bad_args", &meta.name, format!("argument {i} (`{pname}`): {m}")),
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
                Ok(value::decode(&self.types, mem.data(&*store), &meta.ret, slot))
            }
            Err(e) => Err(match store.data().failure {
                Some(f) => self.describe(store, mem, &meta.name, f),
                None => match e.downcast_ref::<Trap>() {
                    Some(Trap::StackOverflow) => RunError::new("stack_overflow", &meta.name, "call stack exhausted")
                        .hint("make the recursion tail-recursive: the recursive call must be the last thing the function does (functions with `ens` never use tail calls)"),
                    Some(Trap::OutOfFuel) => RunError::new("timeout", &meta.name, format!("instruction budget of {} exhausted", self.fuel))
                        .hint("look for unbounded recursion"),
                    Some(Trap::Interrupt) => {
                        let ms = self.timeout.map_or(0, |t| t.as_millis());
                        RunError::new("timeout", &meta.name, format!("wall-clock limit of {ms} ms exceeded")).hint("look for unbounded recursion or very large intermediate values")
                    }
                    _ => RunError::new("trap", &meta.name, e.to_string()),
                },
            }),
        }
    }

    /// Writes an argument into guest memory (heap values are allocated) and returns its slot.
    fn encode(&self, store: &mut Store<HostState>, instance: &Instance, t: &Type, v: &Value) -> Result<i64, String> {
        Ok(match (t, v) {
            (Type::Int, Value::Int(n)) => *n,
            (Type::Float, Value::Float(x)) => x.to_bits() as i64,
            (Type::Bool, Value::Bool(b)) => *b as i64,
            (Type::Unit, Value::Unit) => 0,
            (Type::Str, Value::Str(s)) => self.bytes_block(store, instance, s.as_bytes())?,
            (Type::Bytes, Value::Bytes(b, 0)) => self.bytes_block(store, instance, b)?,
            (Type::List(elem), Value::List(xs)) => {
                let mut slots = vec![xs.len() as i64];
                for x in xs {
                    slots.push(self.encode(store, instance, elem, x)?);
                }
                self.slots_block(store, instance, &slots)?
            }
            (Type::Map(kt, vt), Value::Map(entries)) => {
                let mut sorted: Vec<&(Value, Value)> = entries.iter().collect();
                sorted.sort_by(|a, b| value::compare(&self.types, kt, &a.0, &b.0));
                if sorted.windows(2).any(|w| value::compare(&self.types, kt, &w[0].0, &w[1].0).is_eq()) {
                    return Err(format!("a {t} has a key twice"));
                }
                let mut slots = Vec::new();
                for (k, v) in sorted {
                    slots.push((self.encode(store, instance, kt, k)?, self.encode(store, instance, vt, v)?));
                }
                self.map_tree(store, instance, &slots)?.0 as i64
            }
            (_, Value::Record(_, fields)) if self.types.is_record(t) => {
                let decl = self.types.record_fields(t);
                if fields.len() != decl.len() {
                    return Err(format!("a {t} has {} fields, got {}", decl.len(), fields.len()));
                }
                let mut slots = Vec::new();
                for (f, ft) in &decl {
                    let x = fields.iter().find(|(g, _)| g == f).map(|(_, x)| x).ok_or_else(|| format!("{t} without field `{f}`"))?;
                    slots.push(self.encode(store, instance, ft, x)?);
                }
                self.slots_block(store, instance, &slots)?
            }
            (_, Value::Variant(name, payload)) if !self.types.variant_names(t).is_empty() => {
                let tag = self.types.variant_names(t).iter().position(|n| n == name).ok_or_else(|| format!("{t} has no variant `{name}`"))? as u32;
                let fields = self.types.variant_fields(t, tag);
                if fields.len() != payload.len() {
                    return Err(format!("`{name}` takes {} value(s), got {}", fields.len(), payload.len()));
                }
                let mut slots = vec![tag as i64];
                for (ft, x) in fields.iter().zip(payload) {
                    slots.push(self.encode(store, instance, ft, x)?);
                }
                self.slots_block(store, instance, &slots)?
            }
            (t, v) => return Err(format!("expected {t}, got {}", serde_json::to_string(v).unwrap_or_default())),
        })
    }

    /// Allocates `[len: i32][bytes]` (Str, Bytes).
    fn bytes_block(&self, store: &mut Store<HostState>, instance: &Instance, b: &[u8]) -> Result<i64, String> {
        let len = u32::try_from(b.len()).map_err(|_| "value too large".to_string())?;
        let p = self.alloc(store, instance, 4 + b.len() as i64)?;
        let mem = memory_of(store, instance)?;
        mem.write(&mut *store, p as u32 as usize, &len.to_le_bytes()).map_err(|e| e.to_string())?;
        mem.write(&mut *store, p as u32 as usize + 4, b).map_err(|e| e.to_string())?;
        Ok(p as u32 as i64)
    }

    /// Allocates a block of 8-byte slots (a list `[len][pad][slots]` passes its length as the first
    /// slot, a variant its tag: both are the i32 at offset 0).
    fn slots_block(&self, store: &mut Store<HostState>, instance: &Instance, slots: &[i64]) -> Result<i64, String> {
        let p = self.alloc(store, instance, 8 * slots.len().max(1) as i64)? as u32 as usize;
        let mem = memory_of(store, instance)?;
        for (k, s) in slots.iter().enumerate() {
            mem.write(&mut *store, p + 8 * k, &s.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        Ok(p as i64)
    }

    /// Builds a perfectly balanced map tree (a valid AVL tree) over sorted entry slots; returns
    /// (node pointer or 0, height).
    fn map_tree(&self, store: &mut Store<HostState>, instance: &Instance, entries: &[(i64, i64)]) -> Result<(u32, u32), String> {
        if entries.is_empty() {
            return Ok((0, 0));
        }
        let mid = entries.len() / 2;
        let (l, hl) = self.map_tree(store, instance, &entries[..mid])?;
        let (r, hr) = self.map_tree(store, instance, &entries[mid + 1..])?;
        let height = hl.max(hr) + 1;
        let p = self.alloc(store, instance, 32)? as u32;
        let mut node = Vec::with_capacity(32);
        node.extend((entries.len() as u32).to_le_bytes());
        node.extend(height.to_le_bytes());
        node.extend(entries[mid].0.to_le_bytes());
        node.extend(entries[mid].1.to_le_bytes());
        node.extend(l.to_le_bytes());
        node.extend(r.to_le_bytes());
        let mem = memory_of(store, instance)?;
        mem.write(&mut *store, p as usize, &node).map_err(|e| e.to_string())?;
        Ok((p, height))
    }

    fn alloc(&self, store: &mut Store<HostState>, instance: &Instance, size: i64) -> Result<i32, String> {
        let f = instance.get_typed_func::<i64, i32>(&mut *store, "qal_alloc").map_err(|e| e.to_string())?;
        f.call(&mut *store, size).map_err(|e| e.to_string())
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
            let mut e = RunError::new("out_of_memory", entry, format!("out of memory: the limit is {} bytes per execution", self.memory_limit))
                .hint("memory is never freed in Phase 1; avoid building large intermediate lists or strings in loops (`push` and `++` copy)");
            if let Some(r) = requested {
                e.values = Values(vec![("requested_bytes".into(), Value::Int(r)), ("limit_bytes".into(), Value::Int(self.memory_limit as i64))]);
            }
            return e;
        }
        if f.kind == FAIL_DENIED {
            let name = qal_wasm::host::EFFECT_FNS.get(f.func as usize).copied().unwrap_or("?");
            let effect = qal_wasm::host::effect(name).unwrap_or("?");
            let flag = if effect == "fs" { "fs:<dir>" } else { effect };
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
            FAIL_INDEX => RunError::new("index", def, format!("`{}` index out of range", site.label))
                .hint("valid character indices satisfy 0 <= start <= end <= len(s)"),
            FAIL_BUILTIN => {
                let mut e = RunError::new("contract_violation", def, format!("precondition of a built-in was violated: `{}`", site.label));
                e.contract = Some(site.label.clone());
                e
            }
            other => RunError::new("trap", def, format!("unknown failure kind {other}")),
        };
        e.values = values;
        e.at(&site.path)
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
