//! QAL runtime: executes compiled modules in wasmtime and reports failures as data.

use qal_core::ast::Type;
use qal_text::print_expr;
use qal_wasm::{CompiledModule, FAIL_DIV_ZERO, FAIL_ENS, FAIL_OVERFLOW, FAIL_REQ};
use serde::Serialize;
use wasmtime::{Caller, Config, Engine, Linker, Module, Store, Trap, Val};

/// Instruction budget per call; exhausting it reports `timeout` instead of hanging.
pub const DEFAULT_FUEL: u64 = 2_000_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum Value {
    Int(i64),
    Bool(bool),
    /// Serialized as JSON `null`.
    Unit(()),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RunError {
    /// contract_violation | int_overflow | div_by_zero | stack_overflow | timeout | bad_args | unknown_function | trap
    pub kind: &'static str,
    pub def: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contract: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TestResult {
    pub name: String,
    pub passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RunError>,
}

pub struct Runner {
    engine: Engine,
    module: Module,
    meta: CompiledModule,
    fuel: u64,
}

impl Runner {
    pub fn new(meta: CompiledModule) -> Result<Runner, String> {
        let mut config = Config::new();
        config.wasm_tail_call(true);
        config.consume_fuel(true);
        let engine = Engine::new(&config).map_err(|e| e.to_string())?;
        let module = Module::new(&engine, &meta.wasm).map_err(|e| e.to_string())?;
        Ok(Runner { engine, module, meta, fuel: DEFAULT_FUEL })
    }

    pub fn with_fuel(mut self, fuel: u64) -> Runner {
        self.fuel = fuel;
        self
    }

    /// Calls the function exported as `export` (a fn name, or `test:<name>`).
    pub fn call(&self, export: &str, args: &[Value]) -> Result<Value, RunError> {
        let err = |kind, message: String| RunError { kind, def: export.to_string(), message, contract: None };
        let meta = self.meta.func(export).ok_or_else(|| err("unknown_function", format!("no function `{export}`")))?;
        if meta.params.len() != args.len() {
            return Err(err("bad_args", format!("`{export}` takes {} argument(s), {} given", meta.params.len(), args.len())));
        }
        let mut params = Vec::new();
        for (i, (a, t)) in args.iter().zip(&meta.params).enumerate() {
            params.push(match (a, t) {
                (Value::Int(v), Type::Int) => Val::I64(*v),
                (Value::Bool(b), Type::Bool) => Val::I32(*b as i32),
                (Value::Unit(()), Type::Unit) => Val::I32(0),
                _ => return Err(err("bad_args", format!("argument {i} must be {}", t.name()))),
            });
        }

        let mut store: Store<Option<(i32, i32, i32)>> = Store::new(&self.engine, None);
        store.set_fuel(self.fuel).map_err(|e| err("trap", e.to_string()))?;
        let mut linker = Linker::new(&self.engine);
        linker
            .func_wrap("qal_host", "fail", |mut c: Caller<'_, Option<(i32, i32, i32)>>, kind: i32, func: i32, index: i32| {
                *c.data_mut() = Some((kind, func, index));
                Err::<(), _>(wasmtime::Error::msg("qal failure"))
            })
            .map_err(|e| err("trap", e.to_string()))?;
        let instance = linker.instantiate(&mut store, &self.module).map_err(|e| err("trap", e.to_string()))?;
        let func = instance.get_func(&mut store, export).ok_or_else(|| err("unknown_function", format!("no export `{export}`")))?;
        let mut results = [Val::I32(0)];
        match func.call(&mut store, &params, &mut results) {
            Ok(()) => Ok(match (meta.ret, &results[0]) {
                (Type::Int, Val::I64(v)) => Value::Int(*v),
                (Type::Bool, Val::I32(v)) => Value::Bool(*v != 0),
                (Type::Unit, _) => Value::Unit(()),
                (t, v) => return Err(err("trap", format!("result {v:?} does not match {}", t.name()))),
            }),
            Err(e) => Err(match *store.data() {
                Some(failure) => self.describe_failure(failure),
                None => match e.downcast_ref::<Trap>() {
                    Some(Trap::StackOverflow) => err("stack_overflow", "call stack exhausted (non-tail recursion too deep?)".into()),
                    Some(Trap::OutOfFuel) => err("timeout", format!("instruction budget of {} exhausted", self.fuel)),
                    _ => err("trap", e.to_string()),
                },
            }),
        }
    }

    fn describe_failure(&self, (kind, func, index): (i32, i32, i32)) -> RunError {
        let meta = &self.meta.funcs[func as usize];
        let def = meta.name.clone();
        match kind {
            FAIL_REQ | FAIL_ENS => {
                let (label, list) = if kind == FAIL_REQ { ("req", &meta.req) } else { ("ens", &meta.ens) };
                let text = print_expr(&list[index as usize]);
                RunError {
                    kind: "contract_violation",
                    message: format!("`{label} {text}` of `{def}` is false"),
                    contract: Some(format!("{label}[{index}]: {text}")),
                    def,
                }
            }
            FAIL_OVERFLOW => RunError { kind: "int_overflow", message: format!("Int overflow in `{def}`"), def, contract: None },
            FAIL_DIV_ZERO => RunError { kind: "div_by_zero", message: format!("division by zero in `{def}`"), def, contract: None },
            other => RunError { kind: "trap", message: format!("unknown failure kind {other}"), def, contract: None },
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runner(src: &str) -> Runner {
        let p = qal_text::parse_program(src).unwrap();
        assert!(qal_check::check_program(&p).is_empty());
        Runner::new(qal_wasm::compile(&p)).unwrap()
    }

    #[test]
    fn calls_with_typed_values() {
        let r = runner("fn pick(c: Bool, a: Int, b: Int) -> Int = if c then a else b\nfn u() -> Unit = ()\nfn neg(b: Bool) -> Bool = !b");
        assert_eq!(r.call("pick", &[Value::Bool(false), Value::Int(1), Value::Int(2)]), Ok(Value::Int(2)));
        assert_eq!(r.call("u", &[]), Ok(Value::Unit(())));
        assert_eq!(r.call("neg", &[Value::Bool(true)]), Ok(Value::Bool(false)));
    }

    #[test]
    fn rejects_bad_arguments_and_unknown_functions() {
        let r = runner("fn f(a: Int) -> Int = a");
        assert_eq!(r.call("f", &[]).unwrap_err().kind, "bad_args");
        assert_eq!(r.call("f", &[Value::Bool(true)]).unwrap_err().kind, "bad_args");
        assert_eq!(r.call("g", &[]).unwrap_err().kind, "unknown_function");
    }

    #[test]
    fn contract_violation_names_def_and_contract() {
        let r = runner("fn half(n: Int) -> Int req n % 2 == 0 = n / 2");
        let e = r.call("half", &[Value::Int(3)]).unwrap_err();
        assert_eq!(e.kind, "contract_violation");
        assert_eq!(e.def, "half");
        assert_eq!(e.contract.as_deref(), Some("req[0]: n % 2 == 0"));
        assert_eq!(
            serde_json::to_value(&e).unwrap(),
            serde_json::json!({"kind": "contract_violation", "def": "half",
                "message": "`req n % 2 == 0` of `half` is false", "contract": "req[0]: n % 2 == 0"})
        );
    }

    #[test]
    fn arithmetic_failures_name_the_def() {
        let r = runner("fn inner(a: Int) -> Int = a / 0\nfn outer() -> Int = inner(1) + 1");
        let e = r.call("outer", &[]).unwrap_err();
        assert_eq!((e.kind, e.def.as_str()), ("div_by_zero", "inner"));
    }

    #[test]
    fn infinite_loop_times_out() {
        let r = runner("fn spin(n: Int) -> Int = spin(n)").with_fuel(10_000_000);
        assert_eq!(r.call("spin", &[Value::Int(0)]).unwrap_err().kind, "timeout");
    }

    #[test]
    fn deep_non_tail_recursion_reports_stack_overflow() {
        let r = runner("fn down(n: Int) -> Int = if n == 0 then 0 else 1 + down(n - 1)");
        assert_eq!(r.call("down", &[Value::Int(100_000_000)]).unwrap_err().kind, "stack_overflow");
    }

    #[test]
    fn runs_tests_in_order() {
        let r = runner("test a = 1 == 1\ntest b = 1 == 2\ntest c = 1 / 0 == 0");
        let res = r.run_tests();
        assert_eq!(res.iter().map(|t| (t.name.as_str(), t.passed)).collect::<Vec<_>>(), vec![("a", true), ("b", false), ("c", false)]);
        assert_eq!(res[2].error.as_ref().unwrap().kind, "div_by_zero");
    }

    #[test]
    fn values_serialize_as_plain_json() {
        assert_eq!(serde_json::to_string(&[Value::Int(-3), Value::Bool(true), Value::Unit(())]).unwrap(), "[-3,true,null]");
    }
}
