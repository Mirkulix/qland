use super::*;
use serde_json::json;

fn typed(src: &str) -> TProgram {
    qal_check::check(&qal_text::parse_program(src).unwrap()).unwrap_or_else(|d| panic!("{d:#?}"))
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
    assert!(Options::default().key().starts_with("qal-prove/1 z3-"));
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
