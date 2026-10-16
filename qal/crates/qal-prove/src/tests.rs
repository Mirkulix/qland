use super::*;
use serde_json::json;

fn typed(src: &str) -> TProgram {
    qal_compiler::check(&qal_compiler::parse_program(src).unwrap()).unwrap_or_else(|d| panic!("{d:#?}"))
}

fn reports(src: &str) -> Vec<Report> {
    prove(&typed(src), None, &Options::default())
}

/// (def, obligation, status) of every report.
fn statuses(src: &str) -> Vec<(String, String, Status)> {
    reports(src).into_iter().map(|r| (r.def, r.obligation, r.status)).collect()
}

fn one(src: &str, def: &str) -> Report {
    let rs: Vec<Report> = reports(src).into_iter().filter(|r| r.def == def).collect();
    assert_eq!(rs.len(), 1, "{rs:#?}");
    rs.into_iter().next().unwrap()
}

fn s(def: &str, obligation: &str, st: Status) -> (String, String, Status) {
    (def.into(), obligation.into(), st)
}

#[test]
fn overflow_traps_so_abs_is_non_negative() {
    // With wrapping arithmetic abs(MIN) = MIN < 0 would refute it; QAL traps on `-MIN`.
    let r = one("fn abs(x: Int) -> Int ens r >= 0 = if x < 0 then -x else x", "abs");
    assert_eq!((r.status, r.obligation.as_str(), r.path.as_str(), r.contract.as_str()), (Status::Proven, "ens[0]", "ens[0]", "ens r >= 0"));
    assert_eq!(one("fn inc(x: Int) -> Int ens r > x = x + 1", "inc").status, Status::Proven);
    assert_eq!(one("fn sq(x: Int) -> Int ens r >= 0 = x * x", "sq").status, Status::Proven);
}

#[test]
fn refutations_are_confirmed_by_running() {
    let r = one("fn bad_abs(x: Int) -> Int ens r > 0 = if x < 0 then -x else x", "bad_abs");
    assert_eq!((r.status, r.counterexample.clone()), (Status::Refuted, Some(json!({"x": 0}))));
    let d = r.diagnostic().unwrap();
    assert_eq!((d.code, d.kind, d.def.as_str(), d.path.as_str()), ("E0801", "contract_refuted", "bad_abs", "ens[0]"));
    assert_eq!(d.message, "`ens r > 0` (ens[0]) does not hold for the counterexample");
    assert_eq!(d.counterexample, Some(json!({"x": 0})));
}

#[test]
fn division_truncates_toward_zero() {
    // With floor division (Z3's `div` on negative numbers) r * b <= a would hold for every a.
    let r = one("fn q(a: Int, b: Int) -> Int req b > 0 ens r * b <= a = a / b", "q");
    assert_eq!(r.status, Status::Refuted);
    let a = r.counterexample.as_ref().unwrap()["a"].as_i64().unwrap();
    let b = r.counterexample.as_ref().unwrap()["b"].as_i64().unwrap();
    assert!(a < 0 && a % b != 0, "{a} {b}");
    assert_eq!(one("fn q(a: Int, b: Int) -> Int req b > 0 req a >= 0 ens r * b <= a = a / b", "q").status, Status::Proven);
    assert_eq!(one("fn m(a: Int) -> Int ens r >= 0 = a % 2", "m").status, Status::Refuted);
    assert_eq!(one("fn m(a: Int) -> Int req a >= 0 ens r >= 0 && r < 2 = a % 2", "m").status, Status::Proven);
    // `MIN / -1` traps (overflow), so the result is always the negation.
    assert_eq!(one("fn n(a: Int) -> Int ens r == 0 - a = a / -1", "n").status, Status::Proven);
    assert_eq!(one("fn z(a: Int, b: Int) -> Int ens b != 0 = a / b", "z").status, Status::Proven);
}

#[test]
fn call_sites_must_establish_the_callee_req() {
    let src = "fn pos(x: Int) -> Int req x > 0 = x\n\
               fn caller(y: Int) -> Int = pos(y)\n\
               fn guarded(y: Int) -> Int req y > 5 = pos(y - 1)\n\
               fn branch(y: Int) -> Int = if y > 0 then pos(y) else 0\n\
               test t = pos(0) == 0";
    let rs = reports(src);
    let st: Vec<_> = rs.iter().map(|r| (r.def.clone(), r.obligation.clone(), r.status)).collect();
    assert_eq!(st, vec![s("caller", "req[0] of pos", Status::Refuted), s("guarded", "req[0] of pos", Status::Proven), s("branch", "req[0] of pos", Status::Proven), s("t", "req[0] of pos", Status::Refuted)]);
    assert_eq!((rs[0].path.as_str(), rs[0].contract.as_str()), ("body", "req x > 0"));
    assert_eq!(rs[2].path, "body.then");
    assert_eq!(rs[0].counterexample.as_ref().map(|c| c["y"].as_i64().unwrap() <= 0), Some(true));
    assert_eq!(rs[3].counterexample, None, "a test has no parameters");
}

#[test]
fn callee_contracts_are_assumed_after_the_call() {
    let src = "fn succ(x: Int) -> Int req x < 100 ens r == x + 1 = x + 1\n\
               fn two(x: Int) -> Int req x < 50 ens r == x + 2 = succ(succ(x))\n\
               fn fact(n: Int) -> Int req n >= 0 ens r >= 1 = if n <= 1 then 1 else n * fact(n - 1)\n\
               fn opaque(x: Int) -> Int = x\n\
               fn uses_opaque(x: Int) -> Int ens r == x = opaque(x)";
    assert_eq!(
        statuses(src),
        vec![
            s("succ", "ens[0]", Status::Proven),
            s("two", "req[0] of succ", Status::Proven),
            s("two", "req[0] of succ", Status::Proven),
            s("two", "ens[0]", Status::Proven),
            s("fact", "req[0] of fact", Status::Proven),
            s("fact", "ens[0]", Status::Proven),
            s("uses_opaque", "ens[0]", Status::Unknown),
        ]
    );
    // A callee without `ens` is opaque: its result is unknown, the candidate is not confirmed.
    let r = one(src, "uses_opaque");
    assert!(r.reason.as_deref().unwrap().contains("the result of `opaque` (it has no `ens`)"), "{r:#?}");
}

#[test]
fn recursion_over_enums_uses_the_recursive_contract() {
    // One arm is always taken: the value of a match is one of its arms' values.
    let src = "type Tree[T] = Leaf | Node(Tree[T], T, Tree[T])\n\
               fn size[T](t: Tree[T]) -> Int ens r >= 0 = match t { Leaf => 0, Node(l, _, r) => size(l) + 1 + size(r) }\n\
               type IntTree = Nil | Branch(IntTree, Int, IntTree)\n\
               fn total(t: IntTree) -> Int ens r >= 0 = match t { Nil => 0, Branch(l, v, r) => if v < 0 then total(l) + total(r) else total(l) + v + total(r) }\n\
               fn wrong(t: IntTree) -> Int ens r > 0 = match t { Nil => 0, Branch(l, _, _) => wrong(l) }";
    let st = statuses(src);
    assert_eq!(st, vec![s("total", "ens[0]", Status::Proven), s("wrong", "ens[0]", Status::Refuted), s("size", "ens[0]", Status::Proven)]);
}

#[test]
fn compare_orders_ints_exactly() {
    assert_eq!(one("fn sign(a: Int, b: Int) -> Int ens r >= -1 && r <= 1 && (r == 0) == (a == b) = compare(a, b)", "sign").status, Status::Proven);
    assert_eq!(one("fn s(a: Str, b: Str) -> Int ens r >= -1 && r <= 1 = compare(a, b)", "s").status, Status::Proven);
}

#[test]
fn match_arms_are_tried_in_order() {
    // The second arm only sees y != 3.
    assert_eq!(one("fn m(x: Int) -> Int ens r != 3 = match x { 3 => 4, y => if y == 3 then 3 else y }", "m").status, Status::Proven);
    assert_eq!(one("fn b(c: Bool) -> Int ens r > 0 = match c { true => 1, false => 2 }", "b").status, Status::Proven);
    assert_eq!(one("fn g(x: Int) -> Int ens r > 0 = match x { y if y > 0 => y, _ => 1 }", "g").status, Status::Proven);
}

#[test]
fn unmodelled_values_make_contracts_unknown_not_wrong() {
    let src = "fn sum(xs: List[Int]) -> Int ens r >= 0 = fold(xs, 0, fn(a, x) => a + x)\n\
               fn size(xs: List[Int]) -> Int ens r >= 0 = len(xs)\n\
               fn name_len(s: Str) -> Int ens r > 0 = len(s)\n\
               fn f(x: Float) -> Bool ens r == true = x > 0.0 || x <= 0.0";
    let rs = reports(src);
    assert_eq!(rs.iter().map(|r| (r.def.as_str(), r.status)).collect::<Vec<_>>(), vec![("sum", Status::Unknown), ("size", Status::Proven), ("name_len", Status::Refuted), ("f", Status::Unknown)]);
    // The candidate for `sum` runs with xs = [] (smallest value), which satisfies the contract.
    assert!(rs[0].reason.as_deref().unwrap().starts_with("a counterexample candidate was not confirmed: running it satisfied the contract"), "{:?}", rs[0].reason);
    assert!(rs[0].reason.as_deref().unwrap().contains("built-in `fold`"));
    // `name_len("")` = 0 really violates it.
    assert_eq!(rs[2].counterexample, Some(json!({"s": ""})));
    // NaN makes both comparisons false: the prover does not model Float, so it cannot prove it — correctly.
    assert!(rs[3].reason.as_deref().unwrap().contains("Float"), "{:?}", rs[3].reason);
}

#[test]
fn obligations_inside_closures_are_checked_where_the_closure_is_created() {
    let src = "fn pos(x: Int) -> Int req x > 0 = x\n\
               fn ok(xs: List[Int]) -> List[Int] = map(xs, fn(x) => pos(x * x + 1))\n\
               fn maybe(xs: List[Int]) -> List[Int] = map(xs, fn(x) => pos(x))\n\
               fn captured(k: Int, xs: List[Int]) -> List[Int] req k > 0 = map(xs, fn(x) => pos(k))";
    let rs = reports(src);
    assert_eq!(rs.iter().map(|r| (r.def.as_str(), r.status, r.path.as_str())).collect::<Vec<_>>(), vec![("ok", Status::Proven, "body.args[1].body"), ("maybe", Status::Unknown, "body.args[1].body"), ("captured", Status::Proven, "body.args[1].body")]);
    // Running `maybe([])` never calls the closure: the candidate is not confirmed.
    assert!(rs[1].reason.as_deref().unwrap().contains("not confirmed"), "{:?}", rs[1].reason);
}

#[test]
fn generic_functions_are_proven_once() {
    let src = "fn count[T](xs: List[T]) -> Int ens r >= 0 = len(xs)\n\
               fn pick[T](c: Bool, a: T, b: T) -> T = if c then a else b\n\
               fn pos(x: Int) -> Int req x > 0 = x\n\
               fn use_both(n: Int) -> Int req n > 0 = pos(pick(true, n, 1)) + count([n]) + count([\"a\"])";
    let st = statuses(src);
    // `pick`'s result is a T: unknown at the call site; instances `count[Int]`, `count[Str]` are not reported.
    assert_eq!(st, vec![s("use_both", "req[0] of pos", Status::Unknown), s("count", "ens[0]", Status::Proven)]);
}

#[test]
fn effects_and_generics_are_not_executed() {
    let rs = reports("fn pos(x: Int) -> Int req x > 0 = x\nfn e(x: Int) -> Int !io = { print(\"a\"); pos(x) }\nfn g[T](x: T, n: Int) -> T = let k = pos(n) in x");
    assert_eq!(rs.iter().map(|r| r.status).collect::<Vec<_>>(), vec![Status::Unknown, Status::Unknown]);
    assert!(rs[0].reason.as_deref().unwrap().contains("has effects"), "{:?}", rs[0].reason);
    assert!(rs[1].reason.as_deref().unwrap().contains("generic"), "{:?}", rs[1].reason);
    assert_eq!(rs[0].counterexample.as_ref().map(|c| c["x"].as_i64().unwrap() <= 0), Some(true));
    let d = rs[0].diagnostic().unwrap();
    assert_eq!((d.code, d.kind), ("E0802", "contract_unproven"));
}

#[test]
fn results_are_deterministic_and_limited() {
    let src = "fn fact(n: Int) -> Int req n >= 0 ens r >= 1 = if n <= 1 then 1 else n * fact(n - 1)\nfn bad(x: Int) -> Int ens r > x = x - 1";
    assert_eq!(reports(src), reports(src));
    let none = prove(&typed(src), None, &Options { budget: Duration::ZERO, ..Options::default() });
    assert!(none.iter().all(|r| r.status == Status::Unknown && r.reason.as_deref().unwrap().starts_with("not attempted") && !r.is_stable()), "{none:#?}");
    let tight = prove(&typed("fn h(a: Int, b: Int, c: Int) -> Int req a > 1 req b > 1 req c > 1 ens r != 1000003 = a * b * c"), None, &Options { rlimit: 1, ..Options::default() });
    assert_eq!(tight[0].status, Status::Unknown);
    assert!(tight[0].reason.as_deref().unwrap().starts_with("the solver gave up within its resource limit (rlimit 1)"), "{:?}", tight[0].reason);
    assert!(tight[0].is_stable(), "a resource limit is deterministic");
    let slow = prove(&typed("fn h(a: Int, b: Int, c: Int) -> Int req a > 1 req b > 1 req c > 1 ens r != 1000003 = a * b * c"), None, &Options { rlimit: 0, timeout_ms: 1, ..Options::default() });
    assert_eq!(slow[0].reason.as_deref(), Some("the solver timed out after 1 ms"));
    assert!(!slow[0].is_stable());
    let only = prove(&typed(src), Some(&["bad".to_string()]), &Options::default());
    assert_eq!(only.iter().map(|r| r.def.as_str()).collect::<Vec<_>>(), vec!["bad"]);
    assert!(Options::default().key().starts_with("qal-prove/3 z3-"));
}

#[test]
fn reports_round_trip_as_json() {
    let rs = reports("fn bad(x: Int) -> Int ens r > x = x - 1");
    let text = serde_json::to_string(&rs).unwrap();
    assert_eq!(serde_json::from_str::<Vec<Report>>(&text).unwrap(), rs);
    assert!(text.contains("\"status\":\"refuted\""), "{text}");
}

#[test]
fn large_contract_inlining_is_bounded() {
    // 80 calls per level, 3 levels of contracts calling functions with contracts (review finding:
    // 111 s and 3 GB before the encoding had a size limit).
    let calls = |g: &str| (0..80).map(|_| format!("{g}(x)")).collect::<Vec<_>>().join(" + ");
    let src = format!(
        "fn g3(x: Int) -> Int ens r >= 0 = 0\nfn g2(x: Int) -> Int ens r >= {} + 0 = 0\nfn g1(x: Int) -> Int ens r >= {} + 0 = 0\nfn f(x: Int) -> Int ens r >= 0 = {} + 0",
        calls("g3"),
        calls("g2"),
        calls("g1")
    );
    let start = std::time::Instant::now();
    let rs = reports(&src);
    assert!(start.elapsed() < Duration::from_secs(30), "{:?}", start.elapsed());
    let f = rs.iter().find(|r| r.def == "f").unwrap();
    assert_eq!((f.status, f.obligation.as_str()), (Status::Unknown, "all"));
    assert!(f.reason.as_deref().unwrap().contains("too large to encode"), "{f:#?}");
}

#[test]
fn huge_smallest_values_are_not_built() {
    // Review finding: 14 s and 4 GB at define time.
    let src = huge_record_source();
    let start = std::time::Instant::now();
    let r = one(&src, "f");
    assert!(start.elapsed() < Duration::from_secs(5), "{:?}", start.elapsed());
    assert!(r.reason.as_deref().unwrap().contains("has no value of at most 10000 nodes"), "{r:#?}");
}

/// Records nested 14 deep, 3 fields each: 3^14 Int fields.
fn huge_record_source() -> String {
    let mut src = String::from("type R0 = { a: Int }\n");
    for i in 1..=14 {
        src += &format!("type R{i} = {{ a: R{p}, b: R{p}, c: R{p} }}\n", p = i - 1);
    }
    src + "fn f(x: R14) -> Int ens r == 1 = 0"
}

#[test]
fn a_refutation_names_the_call_that_fails() {
    // Both calls of `g` have the obligation `req[0] of g`; running f(0) violates it at `g(x)` only.
    let src = "type P = { a: Int }\nfn g(x: Int) -> Int req x > 0 = x\nfn f(x: Int) -> Int ens r >= 0 = let p = P { a: 1 } in g(p.a) + 0 * g(x)";
    let rs: Vec<(String, Status)> = reports(src).into_iter().filter(|r| r.def == "f").map(|r| (r.path, r.status)).collect();
    assert_eq!(rs, vec![("body.body.left".to_string(), Status::Unknown), ("body.body.right.right".to_string(), Status::Refuted), ("ens[0]".to_string(), Status::Unknown)]);
}

#[test]
fn simple_recursive_counterexamples_are_found() {
    // With too small a resource limit this was `unknown` (review finding).
    let r = one("fn f(x: Int, n: Int) -> Int req x > 0 = if n == 0 then 0 else f(x - 1, n - 1)", "f");
    assert_eq!(r.status, Status::Refuted, "{r:#?}");
}

// ---------------------------------------------------------------------------------------------
// Property tests

fn props(src: &str) -> Vec<PropReport> {
    check_properties(&typed(src), None, &PropOptions::default()).unwrap()
}

#[test]
fn failing_properties_are_shrunk() {
    let rs = props("fn head(xs: List[Int]) -> Int ens r >= 0 = match get(sort_int(xs), 0) { Some(x) => x, None => 0 }");
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0].status, PropStatus::Failed);
    assert_eq!(serde_json::to_value(rs[0].counterexample.as_ref().unwrap()).unwrap(), json!({"xs": [-1]}));
    assert_eq!(rs[0].error.as_ref().map(|e| (e.kind, e.contract.as_deref())), Some(("contract_violation", Some("ens[0]: r >= 0"))));
}

#[test]
fn passing_properties_count_runs_discards_and_traps() {
    let rs = props(
        "fn clamp(x: Int) -> Int ens r >= 0 && r <= 10 = if x < 0 then 0 else if x > 10 then 10 else x\n\
         fn inv(x: Int) -> Int req x != 0 ens r == 100 / x = 100 / x\n\
         fn sq(x: Int) -> Int ens r >= 0 = x * x\n\
         fn rare(x: Int) -> Int req x == 123456 ens r == x = x",
    );
    let st: Vec<(&str, PropStatus, u32)> = rs.iter().map(|r| (r.def.as_str(), r.status, r.runs)).collect();
    assert_eq!(st, vec![("clamp", PropStatus::Passed, 100), ("inv", PropStatus::Passed, 100), ("sq", PropStatus::Passed, 100), ("rare", PropStatus::GaveUp, 0)]);
    assert!(rs[1].discarded > 0, "x = 0 is generated");
    assert!(rs[2].traps > 0 && rs[2].trap.as_ref().unwrap().kind == "int_overflow" && rs[2].error.is_none(), "edge values overflow: {:?}", rs[2]);
    assert_eq!(rs[3].discarded, 1000);
}

#[test]
fn recursive_calls_that_break_their_own_req_fail() {
    let rs = props("fn down(n: Int) -> Int req n >= 0 ens r == 0 = if n == 0 then 0 else down(n - 2)");
    assert_eq!(rs[0].status, PropStatus::Failed);
    assert_eq!(serde_json::to_value(rs[0].counterexample.as_ref().unwrap()).unwrap(), json!({"n": 1}));
    assert_eq!(rs[0].error.as_ref().unwrap().contract.as_deref(), Some("req[0]: n >= 0"));
}

#[test]
fn structured_inputs_are_generated_and_shrunk() {
    let src = "type P = { name: Str, tags: Map[Str, Int] }\ntype Shape = Dot | Line(Int) | Poly(List[Shape])\n\
               fn name_len(p: P) -> Int ens r < 5 = len(p.name)\n\
               fn depth(s: Shape) -> Int ens r < 2 = match s { Dot => 0, Line(_) => 0, Poly(xs) => 1 + fold(xs, 0, fn(m, x) => if depth(x) > m then depth(x) else m) }\n\
               fn bytes_ok(b: Bytes, o: Option[Result[Bool, Float]]) -> Int ens r >= 0 = bytes_len(b)";
    let rs = props(src);
    let by = |n: &str| rs.iter().find(|r| r.def == n).unwrap();
    assert_eq!(by("name_len").status, PropStatus::Failed);
    assert_eq!(serde_json::to_value(by("name_len").counterexample.as_ref().unwrap()).unwrap(), json!({"p": {"name": "aaaaa", "tags": []}}));
    assert_eq!(by("depth").status, PropStatus::Failed);
    assert_eq!(serde_json::to_value(by("depth").counterexample.as_ref().unwrap()).unwrap(), json!({"s": {"Poly": [[{"Poly": [[]]}]]}}));
    assert_eq!(by("bytes_ok").status, PropStatus::Passed);
}

#[test]
fn only_pure_functions_with_contracts_and_generatable_inputs_are_tested() {
    let src = "fn pos(x: Int) -> Int req x > 0 = x\n\
               fn plain(x: Int) -> Int = x\n\
               fn io(x: Int) -> Int !io ens r == x = { print(\"x\"); x }\n\
               fn higher(f: fn(Int) -> Int) -> Int ens r == r = f(1)\n\
               fn calls(x: Int) -> Int req x > 10 = pos(x)\n\
               fn id[T](x: T) -> T ens r == x = x\n\
               test t = id(1) == 1";
    let rs = props(src);
    // `pos` only has a `req` (nothing to check beyond it); `id[Int]` is the instance in use.
    assert_eq!(rs.iter().map(|r| (r.def.as_str(), r.status)).collect::<Vec<_>>(), vec![("calls", PropStatus::Passed), ("id[Int]", PropStatus::Passed)]);
    let only = check_properties(&typed(src), Some(&["calls".to_string()]), &PropOptions::default()).unwrap();
    assert_eq!(only.len(), 1);
}

#[test]
fn properties_are_deterministic_per_seed() {
    let src = "fn f(xs: List[Int], s: Str) -> Int ens r < 40 = len(xs) + len(s)";
    let a = props(src);
    assert_eq!(a, props(src));
    assert_eq!(a[0].status, PropStatus::Failed);
    let other = check_properties(&typed(src), None, &PropOptions { seed: 99, ..PropOptions::default() }).unwrap();
    assert_eq!((other[0].seed, other[0].status), (99, PropStatus::Failed));
    let skipped = check_properties(&typed(src), None, &PropOptions { budget: Duration::ZERO, ..PropOptions::default() }).unwrap();
    assert_eq!(skipped[0].status, PropStatus::Skipped);
}

#[test]
fn generation_and_shrinking_basics() {
    let tp = typed("type T = Leaf | Node(T, Int, T)");
    let t = Type::Named("T".into());
    assert_eq!(smallest(&tp, &t), Some(Value::Variant("Leaf".into(), vec![])));
    assert!(smallest(&tp, &Type::Fn(vec![], Box::new(Type::Int), vec![])).is_none());
    let mut rng = Rng::new(1);
    for size in 0..40 {
        assert!(random(&tp, &t, size, &mut rng).is_some());
    }
    assert_eq!(shrink(&tp, &Type::Int, &Value::Int(10)), vec![Value::Int(0), Value::Int(5), Value::Int(9)]);
    assert_eq!(shrink(&tp, &Type::Int, &Value::Int(i64::MIN))[0], Value::Int(0));
    let node = Value::Variant("Node".into(), vec![Value::Variant("Leaf".into(), vec![]), Value::Int(3), Value::Variant("Leaf".into(), vec![])]);
    assert_eq!(shrink(&tp, &t, &node)[0], Value::Variant("Leaf".into(), vec![]));
    assert!(!generatable(&tp, &Type::list(Type::Fn(vec![], Box::new(Type::Int), vec![]))));
    assert_eq!(Rng::new(5).next(), Rng::new(5).next());
}

#[test]
fn huge_inputs_are_not_generated() {
    let start = std::time::Instant::now();
    let ps = check_properties(&typed(&huge_record_source()), None, &PropOptions::default()).unwrap();
    assert!(start.elapsed() < Duration::from_secs(10), "{:?}", start.elapsed());
    assert_eq!(ps[0].status, PropStatus::GaveUp);
}

// ---------------------------------------------------------------------------------------------
// Review findings of Phase 2b (final review)

#[test]
fn long_lists_are_encoded_in_linear_time() {
    // C1: each item's path condition was a fresh `and` of all earlier items, quadratic in the
    // length (40 000 items: 178 s); the encoding was not under any time limit either.
    let items = |n: usize| vec!["1"; n].join(",");
    let start = std::time::Instant::now();
    let r = one(&format!("fn big(x: Int) -> Int ens r >= 0 = {{ let xs = [{}] in len(xs) }}", items(40_000)), "big");
    assert_eq!(r.status, Status::Proven, "{r:#?}");
    let rec = format!("type P = {{ {} }}\nfn mk(x: Int) -> P = P {{ {} }}", (0..2000).map(|i| format!("f{i}: Int")).collect::<Vec<_>>().join(", "), (0..2000).map(|i| format!("f{i}: x + {i}")).collect::<Vec<_>>().join(", "));
    assert!(reports(&rec).is_empty());
    let r = one(&format!("fn big(x: Int) -> Int ens r >= 0 = {{ let xs = [{}] in len(xs) }}", items(60_000)), "big");
    assert_eq!((r.status, r.obligation.as_str()), (Status::Unknown, "all"));
    assert!(start.elapsed() < Duration::from_secs(10), "{:?}", start.elapsed());
}

#[test]
fn the_encoding_stops_at_the_budget() {
    // C1: a deadline is checked while encoding, not only between obligations.
    let src = format!("fn big(x: Int) -> Int ens r >= 0 = {{ let xs = [{}] in len(xs) }}", vec!["x + 1"; 40_000].join(","));
    let start = std::time::Instant::now();
    let rs = prove(&typed(&src), None, &Options { budget: Duration::from_millis(1), ..Options::default() });
    assert!(start.elapsed() < Duration::from_secs(2), "{:?}", start.elapsed());
    assert_eq!(rs.len(), 1, "{rs:#?}");
    assert!(rs[0].reason.as_deref().unwrap().starts_with("not attempted"), "{rs:#?}");
}

#[test]
fn a_violation_in_another_function_is_not_blamed_on_this_call() {
    // I1: `g("x")` inside `h` violates g's `req`; f's own call `g(t)` is correct for t = "".
    let src = "fn g(t: Str) -> Int req len(t) == 0 = 0\n\
               fn h[T](x: T) -> Int = if 1 == 1 then g(\"x\") else 0\n\
               fn f(t: Str) -> Int req len(t) == 0 = h(1) + g(t)\n\
               fn k(t: Str) -> Int = h(1) + g(t)";
    for def in ["f", "k"] {
        let rs: Vec<Report> = reports(src).into_iter().filter(|r| r.def == def && r.obligation == "req[0] of g").collect();
        assert_eq!(rs.len(), 1, "{rs:#?}");
        assert_eq!(rs[0].status, Status::Unknown, "{rs:#?}");
        assert!(rs[0].reason.as_deref().unwrap().contains("also called by other functions"), "{rs:#?}");
    }
    // Modelled arguments are no proof of the call site either: h traps before f's call is reached.
    let src = "fn g(x: Int) -> Int req x > 0 = x\nfn h() -> Int = g(0)\nfn f(x: Int) -> Int = h() + g(x)";
    let r = reports(src).into_iter().find(|r| r.def == "f" && r.path == "body.right").unwrap();
    assert_eq!(r.status, Status::Unknown, "{r:#?}");
    // A callee that calls itself can raise its own `req` too.
    let src = "fn g(t: Str, n: Int) -> Int req len(t) == 0 = if n > 0 then g(\"x\", n - 1) else 0\nfn f(t: Str) -> Int = g(t, 1)";
    let r = reports(src).into_iter().find(|r| r.def == "f").unwrap();
    assert_eq!(r.status, Status::Unknown, "{r:#?}");
    // N1 (re-review): a call in the definition's own `ens` has no obligation, so a violation it
    // raises is no proof against the guarded body call.
    let src = "fn g(t: Str) -> Int req len(t) == 0 = 0\nfn f(t: Str) -> Int ens g(\"x\") == 0 = if len(t) == 0 then g(t) else 0";
    let r = reports(src).into_iter().find(|r| r.def == "f" && r.path == "body.then").unwrap();
    assert_eq!(r.status, Status::Unknown, "{r:#?}");
    // Still refuted: the only call of the callee is this one.
    let r = reports("fn g(t: Str) -> Int req len(t) > 0 = 0\nfn f(t: Str) -> Int = g(t)").into_iter().find(|r| r.def == "f").unwrap();
    assert_eq!(r.status, Status::Refuted, "{r:#?}");
}

#[test]
fn many_call_sites_are_solved_in_one_encoding() {
    // M4: the definition was encoded again for every obligation: of 3 000 trivial call sites only
    // 50 were attempted within the budget.
    let calls = vec!["g(x)"; 3000].join(", ");
    let src = format!("fn g(a: Int) -> Int req a > 0 = a\nfn f(x: Int) -> Int req x > 0 = len([{calls}])");
    let start = std::time::Instant::now();
    let rs: Vec<Report> = reports(&src).into_iter().filter(|r| r.def == "f").collect();
    assert_eq!(rs.len(), 3000);
    // About 1.5 s; all 3 000 must fit the 30 s budget (generous for loaded machines).
    assert!(rs.iter().all(|r| r.status == Status::Proven), "{:#?}", rs.iter().find(|r| r.status != Status::Proven));
    assert!(start.elapsed() < Duration::from_secs(40), "{:?}", start.elapsed());
}

#[test]
fn shrinking_stops_at_the_budget() {
    // M1: shrinking ran up to 500 candidates (two runs each) whatever the budget. Here every run
    // fails, and shrinking from a large `x` halves it about 60 times (several seconds).
    let src = "fn spin(n: Int, acc: Int) -> Int = if n == 0 then acc else spin(n - 1, acc + 1)\n\
               fn f(x: Int) -> Int req x > 1000 ens r < 0 = spin(3000000, 0) + 0 * x";
    let opts = PropOptions { budget: Duration::from_millis(500), fuel: 500_000_000, ..PropOptions::default() };
    let start = std::time::Instant::now();
    let rs = check_properties(&typed(src), None, &opts).unwrap();
    assert_eq!(rs[0].status, PropStatus::Failed, "{:?}", rs[0]);
    assert!(start.elapsed() < Duration::from_millis(3000), "{:?}", start.elapsed());
}

#[test]
fn running_out_of_time_is_not_giving_up() {
    // M2: a function stopped by the budget after a few runs was reported `gave_up`.
    let src = "fn spin(n: Int, acc: Int) -> Int = if n == 0 then acc else spin(n - 1, acc + 1)\n\
               fn f(x: Int) -> Int ens r >= 0 = spin(300000, 0) + 0 * x";
    let rs = check_properties(&typed(src), None, &PropOptions { budget: Duration::from_millis(100), ..PropOptions::default() }).unwrap();
    assert_eq!(rs[0].status, PropStatus::Skipped, "{:?}", rs[0]);
    assert!(rs[0].runs > 0 && rs[0].runs < 50, "{:?}", rs[0]);
}

#[test]
fn map_keys_are_shrunk() {
    // M5: only the values and the entry list of a map were shrunk.
    let tp = typed("type T = Leaf");
    let m = Value::Map(vec![(Value::Str("9".into()), Value::Int(1)), (Value::Str("a".into()), Value::Int(2))]);
    let cands = shrink(&tp, &Type::Map(Box::new(Type::Str), Box::new(Type::Int)), &m);
    assert!(cands.contains(&Value::Map(vec![(Value::Str("".into()), Value::Int(1)), (Value::Str("a".into()), Value::Int(2))])), "{cands:?}");
    // "9" -> "a" would duplicate the other key.
    assert!(!cands.contains(&Value::Map(vec![(Value::Str("a".into()), Value::Int(1)), (Value::Str("a".into()), Value::Int(2))])), "{cands:?}");
}

#[test]
fn a_refuted_test_does_not_promise_a_counterexample() {
    // M3: E0801 of a test said "for the counterexample", but a test has none.
    let r = reports("fn sq(x: Int) -> Int req x > 0 = x * x\ntest t1 = sq(0) == 0").into_iter().find(|r| r.def == "t1").unwrap();
    assert_eq!((r.status, &r.counterexample), (Status::Refuted, &None));
    let d = r.diagnostic().unwrap();
    assert_eq!(d.message, "`req x > 0` (req[0] of sq) does not hold: running `t1` violates it");
}

#[test]
fn an_early_refutation_is_not_lost_to_the_budget() {
    // N2 (re-review): all obligations were solved before any candidate was confirmed, so a budget
    // used up by later satisfiable obligations dropped the refutation of the first one.
    let calls = (0..3000).map(|i| format!("g(x - {i})")).collect::<Vec<_>>().join(", ");
    let src = format!("fn g(a: Int) -> Int req a > 0 = a\nfn f(x: Int) -> Int = len([{calls}])");
    let start = std::time::Instant::now();
    let rs: Vec<Report> = prove(&typed(&src), None, &Options { budget: Duration::from_secs(2), ..Options::default() }).into_iter().filter(|r| r.def == "f").collect();
    // The budget plus compiling the program once to run the candidate (generous for loaded machines).
    assert!(start.elapsed() < Duration::from_secs(20), "{:?}", start.elapsed());
    assert_eq!((rs[0].path.as_str(), rs[0].status), ("body.args[0].items[0]", Status::Refuted), "{:#?}", &rs[..2.min(rs.len())]);
}

#[test]
fn a_candidate_run_stopped_by_the_clock_is_not_stored() {
    // N3 (re-review): a confirmation run cut by the wall clock was a stable `unknown`.
    let src = "fn spin(n: Int, acc: Int) -> Int = if n == 0 then acc else spin(n - 1, acc + 1)\n\
               fn f(x: Int) -> Int ens r == 0 = spin(1000000000, x)";
    let ps = prove_defs(&typed(src), Some(&["f".to_string()]), &Options { budget: Duration::from_millis(500), confirm_fuel: u64::MAX / 4, ..Options::default() });
    assert_eq!(ps.len(), 1);
    assert!(ps[0].reports[0].reason.as_deref().unwrap_or("").contains("wall-clock"), "{:#?}", ps[0].reports);
    assert_eq!(ps[0].record, None);
}

#[test]
fn self_hosting_builtin_facts_prove_what_holds() {
    // Phase 5a: facts about bytes_at and the bit operations make simple bounds provable.
    for src in [
        "fn lo(b: Bytes, i: Int) -> Int ens r >= 0 && r <= 255 = bytes_at(b, i)",
        "fn m(a: Int) -> Int ens r >= 0 && r <= 255 = bit_and(a, 255)",
        "fn n(a: Int) -> Int ens r + a == -1 = bit_not(a)",
        "fn u(a: Int) -> Int ens r >= 0 = ushr(a, 1)",
        "fn s(a: Int) -> Int req a >= 0 ens r >= 0 && r <= a = shr(a, 3)",
        "fn o(a: Int) -> Int req a >= 0 ens r >= a = bit_or(a, 1)",
        "fn x(a: Int) -> Int req a >= 0 ens r >= 0 = bit_xor(a, 12345)",
        "fn z(a: Int) -> Int ens r == a = shl(a, 0)",
        // A shift by a constant distance below 64 cannot trap: callers' obligations hold.
        "fn k(a: Int) -> Int ens r == a = shr(shl(a, 0), 0)",
    ] {
        let rs = reports(src);
        assert!(!rs.is_empty() && rs.iter().all(|r| r.status == Status::Proven), "{src}: {rs:#?}");
    }
}

#[test]
fn self_hosting_builtins_never_prove_what_can_fail() {
    // Soundness: false bounds are refuted (or unknown), and a contract whose evaluation can trap in
    // a new built-in (a shift distance outside 0..64, an index outside the list) is never proven.
    for src in [
        "fn f(a: Int) -> Int ens r >= 0 = bit_or(a, 1)",
        "fn f(a: Int) -> Int ens r > 0 = ushr(a, 0)",
        "fn f(a: Int, b: Int) -> Int ens r >= 0 = bit_xor(a, b)",
        "fn f(a: Int) -> Int req a >= 0 ens r >= 0 = shl(a, 1)",
        "fn f(a: Int) -> Int req a >= 0 ens r >= a = shl(a, 1)",
        "fn f(a: Int) -> Int ens r < 0 = shr(a, 63)",
        "fn f(b: Bytes) -> Int ens r < 255 = bytes_at(b, 0)",
        "fn f(n: Int) -> Int ens shl(1, n) != 0 || true = 0",
        "fn f(n: Int) -> Int ens ushr(1, n) >= 0 = 0",
        "fn f(xs: List[Int]) -> Int ens list_at(xs, 0) == list_at(xs, 0) = 0",
        "fn f(b: Bytes) -> Int ens bytes_at(b, 5) <= 255 = 0",
        "fn f(b: Bytes, i: Int) -> Int req i >= 0 ens bytes_at(b, i) >= 0 = 0",
        "fn f(x: Float) -> Int ens r >= 0 = float_bits(x)",
    ] {
        let rs = reports(src);
        assert!(!rs.is_empty() && rs.iter().all(|r| r.status != Status::Proven), "{src}: {rs:#?}");
    }
    // A call's `req` is not established by an unmodelled index either.
    let rs = reports("fn g(n: Int) -> Int req n >= 0 = n\nfn f(xs: List[Int]) -> Int = g(list_at(xs, 0))");
    assert!(rs.iter().any(|r| r.def == "f" && r.status != Status::Proven), "{rs:#?}");
}

#[test]
fn bit_facts_hold_for_the_real_results() {
    use qal_core::tast::Builtin as B;
    use z3::ast::{Ast, Int as ZInt};
    let samples = [0i64, 1, -1, 2, -2, 255, -256, 0x5555, i64::MAX, i64::MIN, i64::MIN + 1, 1 << 62, -(1 << 62)];
    let real = |b: B, a: i64, c: i64| -> Option<i64> {
        Some(match b {
            B::BitAnd => a & c,
            B::BitOr => a | c,
            B::BitXor => a ^ c,
            B::BitNot => !a,
            B::Shl if (0..64).contains(&c) => a.wrapping_shl(c as u32),
            B::Shr if (0..64).contains(&c) => a >> c,
            B::Ushr if (0..64).contains(&c) => ((a as u64) >> c) as i64,
            B::BytesAt => a & 255,
            _ => return None,
        })
    };
    let mut checked = 0;
    for b in [B::BitAnd, B::BitOr, B::BitXor, B::BitNot, B::Shl, B::Shr, B::Ushr, B::BytesAt] {
        for &a in &samples {
            for &c in samples.iter().chain(&[3, 31, 63, 64]) {
                let Some(x) = real(b, a, c) else { continue };
                let vals = [Some(vc::Term::I(ZInt::from_i64(a))), Some(vc::Term::I(ZInt::from_i64(c)))];
                for f in vc::bit_facts(b, &ZInt::from_i64(x), &vals) {
                    assert_eq!(f.simplify().as_bool(), Some(true), "{b:?}({a}, {c}) = {x}: {f}");
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 1000, "{checked}");
}

#[test]
fn contracts_that_can_trap_in_any_builtin_are_never_proven() {
    // Final review I1 (predates Phase 5a): a built-in that can trap contributes its trap
    // condition to `ok`, in `ens`, in `req` at call sites, and through function arguments.
    for src in [
        "fn g(s: Str) -> Int ens let t = slice(s, 5, 3) in true = 0",
        "fn h(s: Str) -> Int ens slice(s, 5, 3) == slice(s, 5, 3) || true = 0",
        "fn h(s: Str) -> Int ens slice(s, 0, 1) == slice(s, 0, 1) || true = 0",
        "fn k(n: Int) -> Int ens from_char_code(n) == from_char_code(n) || true = 0",
        "fn k(n: Int) -> Int req n >= 0 ens let c = from_char_code(n) in true = 0",
        "fn m(b: Bytes) -> Int ens bytes_slice(b, 3, 1) == b || true = 0",
        "fn m(b: Bytes) -> Int ens bytes_slice(b, 0, 1) == b || true = 0",
        "fn c(s: Str) -> Int ens let x = char_code(s) in true = 0",
        "fn s(t: Str, sep: Str) -> Int ens let xs = split(t, sep) in true = 0",
        "fn rp(t: Str, a: Str) -> Int ens let x = replace(t, a, \"b\") in true = 0",
        "fn b(xs: List[Int]) -> Int ens let x = bytes_from(xs) in true = 0",
        "fn f(xs: List[Int]) -> Int ens let ys = map(xs, fn(x) => 10 / x) in true = 0",
        "fn f(xs: List[Int]) -> Int ens let ys = filter(xs, fn(x) => 10 / x > 0) in true = 0",
        "fn f(xs: List[Int]) -> Int ens let y = fold(xs, 0, fn(a, x) => a / x) in true = 0",
        "fn f(xs: List[Int], i: Int) -> Int ens let y = list_at(xs, i) in true = 0",
        "fn f(b: Bytes, i: Int) -> Int ens let y = bytes_at(b, i) in true = 0",
        "fn f(a: Int, n: Int) -> Int ens let y = shl(a, n) in true = 0",
    ] {
        let rs = reports(src);
        assert!(!rs.is_empty() && rs.iter().all(|r| r.status != Status::Proven), "{src}: {rs:#?}");
    }
    // A callee's `req` that may trap is not established at the call either.
    let rs = reports("fn g2(s: Str) -> Int req slice(s, 5, 3) == \"\" || true = 0\nfn f(s: Str) -> Int = g2(s)");
    assert!(rs.iter().filter(|r| r.def == "f").all(|r| r.status != Status::Proven) && rs.iter().any(|r| r.def == "f"), "{rs:#?}");
    // Exact conditions where the arguments are modelled: these cannot trap and stay proven.
    for src in [
        "fn k(n: Int) -> Int ens let c = from_char_code(65) in true = 0",
        "fn k(n: Int) -> Int req n >= 0 && n < 55296 ens let c = from_char_code(n) in true = 0",
        "fn t(s: Str) -> Int ens let x = trim(s) ++ to_upper(s) in true = 0",
        "fn t(xs: List[Int]) -> Int ens let x = get(xs, 3) in true = 0",
    ] {
        let rs = reports(src);
        assert!(!rs.is_empty() && rs.iter().all(|r| r.status == Status::Proven), "{src}: {rs:#?}");
    }
}
