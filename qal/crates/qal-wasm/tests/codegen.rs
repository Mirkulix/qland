//! End-to-end codegen tests: text -> check -> wasm -> execute in wasmtime with a minimal host.

use qal_wasm::{compile, host, CompiledModule, Entry, FAIL_BUILTIN, FAIL_DIV_ZERO, FAIL_ENS, FAIL_INDEX, FAIL_OOM, FAIL_OVERFLOW, FAIL_REQ, HOST_MODULE};
use wasmtime::{Caller, Config, Engine, Extern, Instance, Linker, Memory, Module, Store, Val};

/// (kind, func, site, vals_ptr) recorded by `fail`.
type Failure = (i32, i32, i32, i32);

fn build(src: &str, entry: Entry) -> CompiledModule {
    let p = qal_text::parse_program(src).expect("parse");
    let tp = qal_check::check(&p).unwrap_or_else(|d| panic!("check failed: {d:#?}"));
    let m = compile(&tp, &entry);
    wasmparser::validate(&m.wasm).expect("valid wasm");
    m
}

fn memory(c: &mut Caller<'_, Option<Failure>>) -> Memory {
    match c.get_export("memory") {
        Some(Extern::Memory(m)) => m,
        _ => panic!("no memory export"),
    }
}

fn read_str(c: &mut Caller<'_, Option<Failure>>, ptr: i32) -> String {
    let mem = memory(c);
    let data = mem.data(&c);
    let len = u32::from_le_bytes(data[ptr as usize..ptr as usize + 4].try_into().unwrap()) as usize;
    String::from_utf8(data[ptr as usize + 4..ptr as usize + 4 + len].to_vec()).unwrap()
}

fn new_str(c: &mut Caller<'_, Option<Failure>>, s: &str) -> i32 {
    let alloc = c.get_export("qal_alloc").and_then(Extern::into_func).unwrap().typed::<i64, i32>(&c).unwrap();
    let p = alloc.call(&mut *c, 4 + s.len() as i64).unwrap();
    let mem = memory(c);
    mem.write(&mut *c, p as usize, &(s.len() as u32).to_le_bytes()).unwrap();
    mem.write(&mut *c, p as usize + 4, s.as_bytes()).unwrap();
    p
}

struct Run {
    store: Store<Option<Failure>>,
    instance: Instance,
}

fn instantiate(m: &CompiledModule) -> Run {
    let mut config = Config::new();
    config.wasm_tail_call(true);
    let engine = Engine::new(&config).unwrap();
    let module = Module::new(&engine, &m.wasm).unwrap();
    let mut store: Store<Option<Failure>> = Store::new(&engine, None);
    let mut linker = Linker::new(&engine);
    linker
        .func_wrap(HOST_MODULE, host::FAIL, |mut c: Caller<'_, Option<Failure>>, k: i32, f: i32, s: i32, v: i32| {
            *c.data_mut() = Some((k, f, s, v));
            Err::<(), _>(wasmtime::Error::msg("qal failure"))
        })
        .unwrap();
    linker
        .func_wrap(HOST_MODULE, host::FLOAT_TO_STR, |mut c: Caller<'_, Option<Failure>>, x: f64| new_str(&mut c, &format!("{x:?}")))
        .unwrap();
    linker
        .func_wrap(HOST_MODULE, host::STR_TO_FLOAT, |mut c: Caller<'_, Option<Failure>>, p: i32| match read_str(&mut c, p).parse::<f64>() {
            Ok(x) => (1, x),
            Err(_) => (0, 0.0),
        })
        .unwrap();
    let instance = linker.instantiate(&mut store, &module).unwrap();
    Run { store, instance }
}

impl Run {
    fn call(&mut self, export: &str, args: &[Val]) -> Result<Val, Failure> {
        let func = self.instance.get_func(&mut self.store, export).unwrap_or_else(|| panic!("no export {export}"));
        let mut results = vec![Val::I32(0)];
        match func.call(&mut self.store, args, &mut results) {
            Ok(()) => Ok(results[0]),
            Err(e) => Err(self.store.data().unwrap_or_else(|| panic!("unexpected trap: {e:?}"))),
        }
    }

    fn slot(&mut self, ptr: i32, k: usize) -> i64 {
        let mem = self.instance.get_memory(&mut self.store, "memory").unwrap();
        let at = ptr as usize + 8 * k;
        i64::from_le_bytes(mem.data(&self.store)[at..at + 8].try_into().unwrap())
    }
}

fn call_i64(m: &CompiledModule, export: &str, args: &[i64]) -> Result<i64, Failure> {
    let mut r = instantiate(m);
    let args: Vec<Val> = args.iter().map(|a| Val::I64(*a)).collect();
    r.call(export, &args).map(|v| match v {
        Val::I64(v) => v,
        Val::I32(v) => v as i64,
        other => panic!("unexpected {other:?}"),
    })
}

/// Runs every test definition and asserts each returns true.
fn all_tests_pass(src: &str) {
    let m = build(src, Entry::Tests);
    let mut r = instantiate(&m);
    for f in m.funcs.iter().filter(|f| f.is_test) {
        let result = r.call(&f.export, &[]).map(|v| v.unwrap_i32());
        assert_eq!(result, Ok(1), "test `{}` failed", f.name);
    }
}

#[test]
fn arithmetic_and_precedence() {
    let m = build("fn f(a: Int, b: Int) -> Int = a + b * 2 - a / b % 3", Entry::All);
    assert_eq!(call_i64(&m, "f", &[7, 2]), Ok(7 + 2 * 2 - (7 / 2) % 3));
    assert_eq!(call_i64(&m, "f", &[-7, 2]), Ok(-7 + 4 - (-7 / 2) % 3));
}

#[test]
fn booleans_if_let_and_unit() {
    all_tests_pass(
        "fn max(a: Int, b: Int) -> Int = if a > b then a else b\n\
         fn both(a: Bool, b: Bool) -> Bool = a && !b || false\n\
         fn twice(x: Int) -> Int = let y = x + x in let y = y + 1 in y\n\
         fn u() -> Unit = ()\n\
         test t1 = max(3, 9) == 9 && max(-1, -5) == -1\n\
         test t2 = both(true, false) && !both(true, true)\n\
         test t3 = twice(4) == 9 && u() == ()",
    );
}

#[test]
fn integer_overflow_is_an_error_not_a_wrap() {
    let m = build(
        "fn add(a: Int, b: Int) -> Int = a + b\nfn sub(a: Int, b: Int) -> Int = a - b\n\
         fn mul(a: Int, b: Int) -> Int = a * b\nfn div(a: Int, b: Int) -> Int = a / b\n\
         fn rem(a: Int, b: Int) -> Int = a % b\nfn neg(a: Int) -> Int = -a",
        Entry::All,
    );
    let (max, min) = (i64::MAX, i64::MIN);
    let kind = |r: Result<i64, Failure>| r.map_err(|f| f.0);
    assert_eq!(kind(call_i64(&m, "add", &[max, 1])), Err(FAIL_OVERFLOW));
    assert_eq!(kind(call_i64(&m, "sub", &[min, 1])), Err(FAIL_OVERFLOW));
    assert_eq!(kind(call_i64(&m, "mul", &[min, -1])), Err(FAIL_OVERFLOW));
    assert_eq!(kind(call_i64(&m, "mul", &[-1, min])), Err(FAIL_OVERFLOW));
    assert_eq!(kind(call_i64(&m, "mul", &[1 << 32, 1 << 32])), Err(FAIL_OVERFLOW));
    assert_eq!(kind(call_i64(&m, "div", &[min, -1])), Err(FAIL_OVERFLOW));
    assert_eq!(kind(call_i64(&m, "div", &[1, 0])), Err(FAIL_DIV_ZERO));
    assert_eq!(kind(call_i64(&m, "rem", &[1, 0])), Err(FAIL_DIV_ZERO));
    assert_eq!(kind(call_i64(&m, "neg", &[min])), Err(FAIL_OVERFLOW));
    assert_eq!(call_i64(&m, "rem", &[min, -1]), Ok(0));
    assert_eq!(call_i64(&m, "add", &[max - 1, 1]), Ok(max));
}

#[test]
fn failures_carry_site_and_operand_values() {
    let m = build("fn f(a: Int, b: Int) -> Int = a + (a * b)", Entry::All);
    let mut r = instantiate(&m);
    let (kind, func, site, vals) = r.call("f", &[Val::I64(2), Val::I64(i64::MAX)]).unwrap_err();
    assert_eq!((kind, func), (FAIL_OVERFLOW, 0));
    let s = &m.funcs[0].sites[site as usize];
    assert_eq!((s.path.as_str(), s.label.as_str()), ("body.right", "*"));
    assert_eq!((r.slot(vals, 0), r.slot(vals, 1)), (2, i64::MAX));
}

#[test]
fn contracts_are_checked_at_runtime_with_values() {
    let m = build(
        "fn sqrt_floor(n: Int) -> Int\n  req n >= 0\n  ens r * r <= n\n  ens (r + 1) * (r + 1) > n\n= go(n, 0)\n\
         fn go(n: Int, i: Int) -> Int = if (i + 1) * (i + 1) > n then i else go(n, i + 1)\n\
         fn liar(n: Int) -> Int ens r > n = n",
        Entry::All,
    );
    assert_eq!(call_i64(&m, "sqrt_floor", &[17]), Ok(4));
    let mut r = instantiate(&m);
    let (kind, _, site, vals) = r.call("sqrt_floor", &[Val::I64(-4)]).unwrap_err();
    assert_eq!(kind, FAIL_REQ);
    assert_eq!(m.funcs[0].sites[site as usize].contract, Some(0));
    assert_eq!(r.slot(vals, 0), -4);
    let (kind, func, site, vals) = r.call("liar", &[Val::I64(5)]).unwrap_err();
    assert_eq!(kind, FAIL_ENS);
    let s = &m.funcs[func as usize].sites[site as usize];
    assert_eq!(s.values.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(), vec!["n", "r"]);
    assert_eq!((r.slot(vals, 0), r.slot(vals, 1)), (5, 5));
}

#[test]
fn ens_in_recursive_function_checks_each_call() {
    // fact(1) = 1 violates `r > n` only for n = 1; the inner call must be checked, not just the outer.
    let m = build("fn fact(n: Int) -> Int ens r > n = if n <= 1 then 1 else n * fact(n - 1)", Entry::All);
    assert_eq!(call_i64(&m, "fact", &[3]).map_err(|f| f.0), Err(FAIL_ENS));
}

#[test]
fn deep_tail_recursion_does_not_overflow_the_stack() {
    let m = build(
        "fn count(n: Int, acc: Int) -> Int = if n == 0 then acc else count(n - 1, acc + 1)\n\
         fn walk(n: Int) -> Int = match n { 0 => 0, _ => { walk(n - 1) } }",
        Entry::All,
    );
    assert_eq!(call_i64(&m, "count", &[1_000_000, 0]), Ok(1_000_000));
    assert_eq!(call_i64(&m, "walk", &[1_000_000]), Ok(0));
}

#[test]
fn floats() {
    all_tests_pass(
        "fn avg(a: Float, b: Float) -> Float = (a + b) / 2.0\n\
         test arith = avg(1.0, 2.0) == 1.5 && -avg(1.0, 1.0) < 0.0 && 1.0 / 0.0 > 1e308\n\
         test convert = int_to_float(3) == 3.0 && float_to_int(2.9) == Some(2) && float_to_int(-2.9) == Some(-2)\n\
         test convert_range = float_to_int(1e19) == None && float_to_int(0.0 / 0.0) == None && float_to_int(-9.2e18) != None\n\
         test text = from_float(1.5) == \"1.5\" && to_float(\"2.25\") == Some(2.25) && to_float(\"x\") == None\n\
         test nan = 0.0 / 0.0 != 0.0 / 0.0\n\
         test nan_in_values = (let l = [0.0 / 0.0] in l != l) && (let o = Some(0.0 / 0.0) in o != o)",
    );
}

#[test]
fn strings() {
    all_tests_pass(
        "test lit = \"héllo\" == \"héllo\" && \"a\" != \"b\" && \"\" == \"\"\n\
         test cat = \"ab\" ++ \"\" ++ \"cd\" == \"abcd\"\n\
         test length = len(\"héllo\") == 5 && len(\"\") == 0\n\
         test slicing = slice(\"héllo\", 1, 3) == \"él\" && slice(\"abc\", 0, 3) == \"abc\" && slice(\"abc\", 2, 2) == \"\"\n\
         test splitting = split(\"a,b,,c\", \",\") == [\"a\", \"b\", \"\", \"c\"] && split(\"abc\", \"--\") == [\"abc\"] && split(\"a--b\", \"--\") == [\"a\", \"b\"]\n\
         test joining = join([\"a\", \"b\", \"c\"], \", \") == \"a, b, c\" && join([], \"x\") == \"\" && join([\"z\"], \"-\") == \"z\"\n\
         test trimming = trim(\" \\t a b \\n\") == \"a b\" && trim(\"   \") == \"\" && trim(\"x\") == \"x\"\n\
         test search = contains(\"hello\", \"ll\") && !contains(\"hello\", \"lo!\") && contains(\"a\", \"\") && starts_with(\"hello\", \"he\") && !starts_with(\"h\", \"he\")\n\
         test ints = to_int(\"42\") == Some(42) && to_int(\"-7\") == Some(-7) && to_int(\"+3\") == Some(3) && to_int(\"\") == None && to_int(\"-\") == None && to_int(\"1x\") == None && to_int(\" 1\") == None\n\
         test int_limits = to_int(\"9223372036854775807\") == Some(9223372036854775807) && to_int(\"-9223372036854775808\") == Some(-9223372036854775807 - 1) && to_int(\"9223372036854775808\") == None\n\
         test from_ints = from_int(0) == \"0\" && from_int(-120) == \"-120\" && from_int(-9223372036854775807 - 1) == \"-9223372036854775808\"\n\
         test char_list = chars(\"aé😀\") == [\"a\", \"é\", \"😀\"] && chars(\"\") == []",
    );
}

#[test]
fn lists_and_function_values() {
    all_tests_pass(
        "fn double(x: Int) -> Int = x * 2\n\
         fn positive(x: Int) -> Bool = x > 0\n\
         fn add(a: Int, b: Int) -> Int = a + b\n\
         fn shout(s: Str) -> Str = s ++ \"!\"\n\
         fn half(x: Int) -> Float = int_to_float(x) / 2.0\n\
         fn apply(f: fn(Int) -> Int, x: Int) -> Int = f(x)\n\
         fn pick(b: Bool) -> fn(Int) -> Int = if b then double else negate\n\
         fn negate(x: Int) -> Int = -x\n\
         test basics = len([1, 2, 3]) == 3 && get([1, 2], 1) == Some(2) && get([1], 1) == None && get([1], -1) == None\n\
         test building = push([1], 2) == [1, 2] && concat([1], [2, 3]) == [1, 2, 3] && [1] ++ [] == [1] && reverse([1, 2, 3]) == [3, 2, 1]\n\
         test ranges = range(0, 4) == [0, 1, 2, 3] && range(3, 3) == [] && range(5, 1) == [] && len(range(-2, 2)) == 4\n\
         test sorting = sort_int([3, -1, 2, 2, 0]) == [-1, 0, 2, 2, 3] && sort_int([]) == [] && sort_int(reverse(range(0, 1000))) == range(0, 1000)\n\
         test higher_order = map([1, 2], double) == [2, 4] && filter(range(-3, 4), positive) == [1, 2, 3] && fold([1, 2, 3], 10, add) == 16\n\
         test mixed_reprs = map([\"a\"], shout) == [\"a!\"] && map([1, 3], half) == [0.5, 1.5]\n\
         test fn_values = apply(double, 5) == 10 && (let f = pick(false) in f(3)) == -3 && apply(pick(true), 1) == 2\n\
         test nested = [[1], []] == [[1], []] && [[1]] != [[2]] && get([[\"x\"]], 0) == Some([\"x\"])",
    );
}

#[test]
fn records_enums_and_match() {
    all_tests_pass(
        "type User = { id: Int, name: Str, tags: List[Str] }\n\
         type Shape = Circle(Float) | Rect(Float, Float) | Empty\n\
         type Tree = Leaf | Node(Tree, Int, Tree)\n\
         fn area(s: Shape) -> Float = match s { Circle(r) => 3.0 * r * r, Rect(w, h) => w * h, Empty => 0.0 }\n\
         fn sum(t: Tree) -> Int = match t { Leaf => 0, Node(l, v, r) => sum(l) + v + sum(r) }\n\
         fn classify(n: Int) -> Str = match n { 0 => \"zero\", -1 => \"minus one\", x if x > 0 => \"pos\", _ => \"neg\" }\n\
         fn greet(u: User) -> Str = match u { User { name: \"root\" } => \"admin\", User { id, name } => name ++ from_int(id) }\n\
         fn first(xs: List[Int]) -> Int = match get(xs, 0) { Some(n) => n, None => -1 }\n\
         fn word(s: Str) -> Int = match s { \"one\" => 1, \"two\" => 2, _ => 0 }\n\
         fn both(a: Bool, b: Bool) -> Int = match a { true => match b { true => 3, false => 2 }, false => 1 }\n\
         test records = User { id: 1, name: \"a\", tags: [] }.name == \"a\" && User { id: 1, name: \"a\", tags: [\"x\"] } == User { tags: [\"x\"], name: \"a\", id: 1 }\n\
         test record_ne = User { id: 1, name: \"a\", tags: [] } != User { id: 2, name: \"a\", tags: [] }\n\
         test enums = area(Rect(2.0, 3.0)) == 6.0 && area(Empty) == 0.0 && Circle(1.0) == Circle(1.0) && Circle(1.0) != Rect(1.0, 1.0) && Empty == Empty\n\
         test trees = sum(Node(Node(Leaf, 1, Leaf), 2, Node(Leaf, 3, Leaf))) == 6 && Node(Leaf, 1, Leaf) == Node(Leaf, 1, Leaf)\n\
         test literals = classify(0) == \"zero\" && classify(-1) == \"minus one\" && classify(5) == \"pos\" && classify(-5) == \"neg\"\n\
         test record_patterns = greet(User { id: 7, name: \"root\", tags: [] }) == \"admin\" && greet(User { id: 7, name: \"bob\", tags: [] }) == \"bob7\"\n\
         test options = first([4, 5]) == 4 && first([]) == -1 && Some(Some(1)) == Some(Some(1)) && Ok(1) != Err(\"x\")\n\
         test strings_and_bools = word(\"two\") == 2 && word(\"three\") == 0 && both(true, false) == 2 && both(false, true) == 1",
    );
}

#[test]
fn sequences_evaluate_left_to_right() {
    all_tests_pass("fn f(x: Int) -> Int = { (); (); x + 1 }\ntest t = f(1) == 2 && { (); true }");
}

#[test]
fn builtin_preconditions_fail_with_values() {
    let m = build("fn cut() -> Str = slice(\"héllo\", 2, 9)\nfn parts() -> List[Str] = split(\"abc\", \"\")", Entry::All);
    let mut r = instantiate(&m);
    let (kind, func, site, vals) = r.call("cut", &[]).unwrap_err();
    assert_eq!(kind, FAIL_INDEX);
    assert_eq!(m.funcs[func as usize].sites[site as usize].label, "slice");
    assert_eq!((r.slot(vals, 0), r.slot(vals, 1), r.slot(vals, 2)), (2, 9, 5));
    let (kind, _, _, _) = r.call("parts", &[]).unwrap_err();
    assert_eq!(kind, FAIL_BUILTIN);
}

#[test]
fn exhausting_memory_is_out_of_memory() {
    let m = build("fn big(n: Int) -> Int = len(range(0, n))", Entry::All);
    let mut r = instantiate(&m);
    let (kind, func, site, ptr) = r.call("big", &[Val::I64(40_000_000)]).unwrap_err();
    assert_eq!((kind, func, site), (FAIL_OOM, -1, -1));
    assert_eq!(r.slot(ptr, 0), 8 + 8 * 40_000_000);
    assert_eq!(call_i64(&m, "big", &[1_000_000]), Ok(1_000_000));
}

#[test]
fn huge_join_is_out_of_memory_not_a_wrapped_length() {
    // 4 Mi empty strings joined by a 1 KiB separator need 4 GiB: the length must not wrap in i32.
    let src = "fn empty(i: Int) -> Str = \"\"\n\
               fn rep(s: Str, n: Int) -> Str = if n == 0 then s else rep(s ++ s, n - 1)\n\
               fn big() -> Int = len(join(map(range(0, 4194305), empty), rep(\"a\", 10)))";
    let m = build(src, Entry::All);
    assert_eq!(call_i64(&m, "big", &[]).map_err(|f| f.0), Err(FAIL_OOM));
}

#[test]
fn only_needed_host_functions_are_imported() {
    let pure = build("fn f(x: Int) -> Int = x\nfn g(x: Float) -> Str = from_float(x)", Entry::Fn("f".into()));
    assert_eq!(pure.imports, vec![host::FAIL]);
    assert_eq!(pure.funcs.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), vec!["f"]);
    let io = build(
        "fn main(args: List[Str]) -> Int !io !fs = { print(\"hi\"); helper() }\nfn helper() -> Int !io = { print(\"x\"); 0 }\n\
         fn unused(p: Str) -> Result[Str, Str] !fs = read_file(p)",
        Entry::Fn("main".into()),
    );
    assert_eq!(io.imports, vec![host::FAIL, host::PRINT]);
    let imports: Vec<String> = wasmparser::Parser::new(0)
        .parse_all(&io.wasm)
        .filter_map(|p| match p.unwrap() {
            wasmparser::Payload::ImportSection(s) => Some(s.into_imports().map(|i| i.unwrap().name.to_string()).collect::<Vec<_>>()),
            _ => None,
        })
        .flatten()
        .collect();
    assert_eq!(imports, vec!["fail", "io.print"]);
}
