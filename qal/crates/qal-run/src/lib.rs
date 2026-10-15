//! QAL runtime: executes compiled modules in wasmtime with capability-gated host functions and
//! reports every failure as data in the diagnostic format (code, kind, def, path, message, values, hints).

mod host;
mod value;

pub use host::Permissions;
pub use value::{Value, Values};

use host::{Failure, HostState};
use qal_core::ast::Type;
use qal_core::tast::TProgram;
use qal_text::print_expr;
use qal_wasm::{CompiledModule, FuncMeta, FAIL_BUILTIN, FAIL_DIV_ZERO, FAIL_ENS, FAIL_INDEX, FAIL_OOM, FAIL_OVERFLOW, FAIL_REQ, OOM_SCRATCH};
use serde::Serialize;
use std::io::Cursor;
use wasmtime::{Config, Engine, Linker, Memory, Module, Store, StoreLimitsBuilder, Trap, Val};

/// Instruction budget per call; exhausting it reports `timeout` instead of hanging.
pub const DEFAULT_FUEL: u64 = 2_000_000_000;
/// Memory limit per execution (spec §8).
pub const DEFAULT_MEMORY_LIMIT: usize = 256 << 20;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RunError {
    pub code: &'static str,
    /// contract_violation | int_overflow | div_by_zero | out_of_memory | index | stack_overflow | timeout
    /// | permission_denied | bad_args | unknown_function | trap
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
}

impl Runner {
    pub fn new(meta: CompiledModule) -> Result<Runner, String> {
        let mut config = Config::new();
        config.wasm_tail_call(true);
        config.consume_fuel(true);
        let engine = Engine::new(&config).map_err(|e| e.to_string())?;
        let module = Module::new(&engine, &meta.wasm).map_err(|e| e.to_string())?;
        let types = TProgram { types: meta.types.clone(), ..TProgram::default() };
        Ok(Runner { engine, module, meta, types, fuel: DEFAULT_FUEL, memory_limit: DEFAULT_MEMORY_LIMIT, perms: Permissions::default(), stdin: None })
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
            .map(|f| match self.call(&f.export, &[]) {
                Ok(Value::Bool(passed)) => TestResult { name: f.name.clone(), passed, error: None },
                Ok(_) => unreachable!("tests are checked to return Bool"),
                Err(e) => TestResult { name: f.name.clone(), passed: false, error: Some(e) },
            })
            .collect()
    }

    pub fn invoke(&self, export: &str, args: &[Value]) -> Outcome {
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
        };
        let mut store = Store::new(&self.engine, state);
        store.limiter(|s| &mut s.limits);
        let result = self.invoke_in(&mut store, export, args);
        let stdout = String::from_utf8_lossy(&store.data().stdout).into_owned();
        Outcome { result, stdout }
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
        let mem = instance.get_memory(&mut *store, "memory").expect("memory export");
        let mut params = Vec::new();
        for (i, (a, (pname, t))) in args.iter().zip(&meta.params).enumerate() {
            let slot = match self.encode(store, &instance, t, a) {
                Ok(slot) => slot,
                Err(_) if store.data().failure.is_some() => {
                    let f = store.data().failure.expect("checked");
                    return Err(self.describe(store, mem, &meta.name, f));
                }
                Err(m) => return Err(RunError::new("bad_args", &meta.name, format!("argument {i} (`{pname}`): {m}"))),
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
                    _ => RunError::new("trap", &meta.name, e.to_string()),
                },
            }),
        }
    }

    /// Writes an argument into guest memory (Str and List[...] are allocated) and returns its slot.
    fn encode(&self, store: &mut Store<HostState>, instance: &wasmtime::Instance, t: &Type, v: &Value) -> Result<i64, String> {
        Ok(match (t, v) {
            (Type::Int, Value::Int(n)) => *n,
            (Type::Float, Value::Float(x)) => x.to_bits() as i64,
            (Type::Bool, Value::Bool(b)) => *b as i64,
            (Type::Unit, Value::Unit) => 0,
            (Type::Str, Value::Str(s)) => {
                let p = self.alloc(store, instance, 4 + s.len() as i64)?;
                let mem = instance.get_memory(&mut *store, "memory").expect("memory export");
                mem.write(&mut *store, p as usize, &(s.len() as u32).to_le_bytes()).map_err(|e| e.to_string())?;
                mem.write(&mut *store, p as usize + 4, s.as_bytes()).map_err(|e| e.to_string())?;
                p as i64
            }
            (Type::List(elem), Value::List(xs)) => {
                let mut slots = Vec::new();
                for x in xs {
                    slots.push(self.encode(store, instance, elem, x)?);
                }
                let p = self.alloc(store, instance, 8 + 8 * xs.len() as i64)?;
                let mem = instance.get_memory(&mut *store, "memory").expect("memory export");
                mem.write(&mut *store, p as usize, &(xs.len() as u32).to_le_bytes()).map_err(|e| e.to_string())?;
                for (k, s) in slots.iter().enumerate() {
                    mem.write(&mut *store, p as usize + 8 + 8 * k, &s.to_le_bytes()).map_err(|e| e.to_string())?;
                }
                p as i64
            }
            (t, v) => return Err(format!("expected {t}, got {}", serde_json::to_string(v).unwrap_or_default())),
        })
    }

    fn alloc(&self, store: &mut Store<HostState>, instance: &wasmtime::Instance, size: i64) -> Result<i32, String> {
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
        let meta: &FuncMeta = &self.meta.funcs[f.func as usize];
        let site = &meta.sites[f.site as usize];
        let values = self.values(store, mem, &site.values, f.vals);
        let def = meta.name.as_str();
        let mut e = match f.kind {
            FAIL_REQ | FAIL_ENS => {
                let (label, list) = if f.kind == FAIL_REQ { ("req", &meta.req) } else { ("ens", &meta.ens) };
                let index = site.contract.expect("contract site");
                let text = print_expr(&list[index]);
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

#[cfg(test)]
mod tests;
