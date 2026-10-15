//! End-to-end codegen tests: text -> check -> wasm -> execute in wasmtime.

use qal_wasm::{compile, CompiledModule, FAIL_DIV_ZERO, FAIL_ENS, FAIL_OVERFLOW, FAIL_REQ};
use wasmtime::{Caller, Config, Engine, Instance, Linker, Module, Store, Val};

fn build(src: &str) -> CompiledModule {
    let p = qal_text::parse_program(src).expect("parse");
    let diags = qal_check::check_program(&p);
    assert!(diags.is_empty(), "check failed: {diags:?}");
    let m = compile(&p);
    wasmparser::validate(&m.wasm).expect("valid wasm");
    m
}

/// Calls `export` with i64 args. Ok(i64 result, or i32 widened) or Err((kind, func, index)) from `fail`.
fn call(m: &CompiledModule, export: &str, args: &[i64]) -> Result<i64, (i32, i32, i32)> {
    let mut config = Config::new();
    config.wasm_tail_call(true);
    let engine = Engine::new(&config).unwrap();
    let module = Module::new(&engine, &m.wasm).unwrap();
    let mut store: Store<Option<(i32, i32, i32)>> = Store::new(&engine, None);
    let mut linker = Linker::new(&engine);
    linker
        .func_wrap("qal_host", "fail", |mut c: Caller<'_, Option<(i32, i32, i32)>>, k: i32, f: i32, i: i32| {
            *c.data_mut() = Some((k, f, i));
            Err::<(), _>(wasmtime::Error::msg("qal failure"))
        })
        .unwrap();
    let instance: Instance = linker.instantiate(&mut store, &module).unwrap();
    let func = instance.get_func(&mut store, export).expect("export");
    let params: Vec<Val> = args.iter().map(|a| Val::I64(*a)).collect();
    let mut results = vec![Val::I64(0)];
    match func.call(&mut store, &params, &mut results) {
        Ok(()) => Ok(match results[0] {
            Val::I64(v) => v,
            Val::I32(v) => v as i64,
            _ => panic!("unexpected result"),
        }),
        Err(e) => Err(store.data().unwrap_or_else(|| panic!("unexpected trap: {e:?}"))),
    }
}

#[test]
fn arithmetic_and_precedence() {
    let m = build("fn f(a: Int, b: Int) -> Int = a + b * 2 - a / b % 3");
    assert_eq!(call(&m, "f", &[7, 2]), Ok(7 + 2 * 2 - (7 / 2) % 3));
    assert_eq!(call(&m, "f", &[-7, 2]), Ok(-7 + 4 - (-7 / 2) % 3));
}

#[test]
fn booleans_if_let_and_unit() {
    let m = build(
        "fn max(a: Int, b: Int) -> Int = if a > b then a else b\n\
         fn both(a: Bool, b: Bool) -> Bool = a && !b || false\n\
         fn twice(x: Int) -> Int = let y = x + x in let y = y + 1 in y\n\
         fn u() -> Unit = ()\n\
         test unit_eq = u() == ()",
    );
    assert_eq!(call(&m, "max", &[3, 9]), Ok(9));
    assert_eq!(call(&m, "twice", &[5]), Ok(11));
    assert_eq!(call(&m, "test:unit_eq", &[]), Ok(1));
}

#[test]
fn short_circuit_skips_right_side() {
    // If `&&` / `||` evaluated their right side, 1 / 0 would fail.
    let m = build("test and_sc = !(false && 1 / 0 == 0)\ntest or_sc = true || 1 / 0 == 0");
    assert_eq!(call(&m, "test:and_sc", &[]), Ok(1));
    assert_eq!(call(&m, "test:or_sc", &[]), Ok(1));
}

#[test]
fn recursion_and_tests() {
    let m = build("fn fact(n: Int) -> Int = if n <= 1 then 1 else n * fact(n - 1)\ntest t5 = fact(5) == 120\ntest bad = fact(3) == 7");
    assert_eq!(call(&m, "fact", &[20]), Ok(2432902008176640000));
    assert_eq!(call(&m, "test:t5", &[]), Ok(1));
    assert_eq!(call(&m, "test:bad", &[]), Ok(0));
}

#[test]
fn integer_overflow_is_an_error_not_a_wrap() {
    let m = build(
        "fn add(a: Int, b: Int) -> Int = a + b\nfn sub(a: Int, b: Int) -> Int = a - b\n\
         fn mul(a: Int, b: Int) -> Int = a * b\nfn div(a: Int, b: Int) -> Int = a / b\n\
         fn rem(a: Int, b: Int) -> Int = a % b\nfn neg(a: Int) -> Int = -a",
    );
    let (max, min) = (i64::MAX, i64::MIN);
    assert_eq!(call(&m, "add", &[max, 1]), Err((FAIL_OVERFLOW, 0, 0)));
    assert_eq!(call(&m, "add", &[max, -1]), Ok(max - 1));
    assert_eq!(call(&m, "sub", &[min, 1]), Err((FAIL_OVERFLOW, 1, 0)));
    assert_eq!(call(&m, "mul", &[min, -1]), Err((FAIL_OVERFLOW, 2, 0)));
    assert_eq!(call(&m, "mul", &[-1, min]), Err((FAIL_OVERFLOW, 2, 0)));
    assert_eq!(call(&m, "mul", &[1 << 32, 1 << 31]), Err((FAIL_OVERFLOW, 2, 0)));
    assert_eq!(call(&m, "mul", &[-3, 4]), Ok(-12));
    assert_eq!(call(&m, "div", &[min, -1]), Err((FAIL_OVERFLOW, 3, 0)));
    assert_eq!(call(&m, "div", &[1, 0]), Err((FAIL_DIV_ZERO, 3, 0)));
    assert_eq!(call(&m, "rem", &[1, 0]), Err((FAIL_DIV_ZERO, 4, 0)));
    assert_eq!(call(&m, "rem", &[min, -1]), Ok(0));
    assert_eq!(call(&m, "neg", &[min]), Err((FAIL_OVERFLOW, 5, 0)));
}

#[test]
fn contracts_are_checked_at_runtime() {
    let m = build(
        "fn sqrt_floor(n: Int) -> Int\n  req n >= 0\n  ens r * r <= n\n  ens (r + 1) * (r + 1) > n\n= go(n, 0)\n\
         fn go(n: Int, i: Int) -> Int = if (i + 1) * (i + 1) > n then i else go(n, i + 1)\n\
         fn liar(n: Int) -> Int ens r > n = n",
    );
    assert_eq!(call(&m, "sqrt_floor", &[17]), Ok(4));
    assert_eq!(call(&m, "sqrt_floor", &[-1]), Err((FAIL_REQ, 0, 0)));
    assert_eq!(call(&m, "liar", &[3]), Err((FAIL_ENS, 2, 0)));
}

#[test]
fn ens_in_recursive_function_checks_each_call() {
    // Inner call fact(1) returns 1 which violates `r > n` (1 > 1 is false): reported at index 0.
    let m = build("fn fact(n: Int) -> Int ens r > n = if n <= 1 then 1 else n * fact(n - 1)");
    assert_eq!(call(&m, "fact", &[4]), Err((FAIL_ENS, 0, 0)));
}

#[test]
fn ens_is_checked_even_when_body_is_a_call_in_tail_position() {
    // `id(n)` would be a tail call; under `ens` it must not be, or the contract is skipped.
    let m = build("fn id(n: Int) -> Int = n\nfn bad(n: Int) -> Int ens r > n = id(n)");
    assert_eq!(call(&m, "bad", &[3]), Err((FAIL_ENS, 1, 0)));
}

#[test]
fn deep_tail_recursion_does_not_overflow_the_stack() {
    let m = build("fn count(n: Int, acc: Int) -> Int = if n == 0 then acc else count(n - 1, acc + 1)");
    assert_eq!(call(&m, "count", &[1_000_000, 0]), Ok(1_000_000));
}
