use super::*;
use qal_core::tast::{TExpr, TKind, TPat};
use qal_text::parse_program;

/// Checks on a 64 MiB stack: test threads have 2 MiB, the CLI's main thread 8 MiB and the MCP worker
/// 256 MiB, and the checker recurses along the expression nesting (up to the parser's limit).
fn check_src(src: &str) -> Vec<Diagnostic> {
    let src = src.to_string();
    std::thread::Builder::new().stack_size(64 << 20).spawn(move || check_program(&parse_program(&src).unwrap())).unwrap().join().unwrap()
}

fn typed(src: &str) -> TProgram {
    check(&parse_program(src).unwrap()).unwrap_or_else(|d| panic!("unexpected diagnostics: {d:#?}"))
}

fn kinds(src: &str) -> Vec<(&'static str, String, String)> {
    check_src(src).into_iter().map(|d| (d.kind, d.def, d.path)).collect()
}

fn one(src: &str) -> Diagnostic {
    let d = check_src(src);
    assert_eq!(d.len(), 1, "{d:#?}");
    d.into_iter().next().unwrap()
}

#[test]
fn accepts_well_typed_program() {
    let src = "fn fact(n: Int) -> Int req n >= 0 ens r >= 1 = if n <= 1 then 1 else n * fact(n - 1)\n\
               test fact_5 = fact(5) == 120\n\
               fn noop() -> Unit = ()\n\
               test unit_eq = noop() == ()";
    assert_eq!(check_src(src), vec![]);
}

#[test]
fn unknown_function_hints_at_builtins_and_definitions() {
    let d = one("fn f() -> Int = nosuch(1)");
    assert_eq!(d.kind, "unknown_name");
    assert!(d.hints[0].contains("not a built-in or stored definition"), "{:?}", d.hints);
}

#[test]
fn reports_type_mismatch_with_path() {
    let d = one("fn f(x: Int) -> Int = if x then 1 else 2");
    assert_eq!((d.kind, d.path.as_str()), ("type_mismatch", "body.cond"));
    assert_eq!((d.expected.as_deref(), d.found.as_deref()), (Some("Bool"), Some("Int")));
    assert_eq!(d.hints, vec!["compare explicitly, e.g. `x != 0`".to_string()]);
}

#[test]
fn reports_unknown_names_arity_and_duplicates() {
    assert_eq!(kinds("fn f() -> Int = y"), vec![("unknown_name", "f".into(), "body".into())]);
    assert_eq!(kinds("fn f() -> Int = g()"), vec![("unknown_name", "f".into(), "body".into())]);
    let d = check_src("fn g(a: Int) -> Int = a\nfn f() -> Int = g(1, 2)");
    assert_eq!((d[0].code, d[0].kind), ("E0204", "type_mismatch"));
    assert_eq!(kinds("fn f() -> Int = 1\nfn f() -> Int = 2")[0].0, "duplicate_name");
    assert_eq!(kinds("fn f(a: Int, a: Int) -> Int = a")[0], ("duplicate_name", "f".into(), "params[1]".into()));
}

#[test]
fn arguments_are_checked_even_when_arity_is_wrong() {
    let k = kinds("fn g(a: Int) -> Int = a\nfn f() -> Int = g(1, true && 1)");
    assert_eq!(k, vec![("type_mismatch", "f".into(), "body".into()), ("type_mismatch", "f".into(), "body.args[1].right".into())]);
    let k = kinds("fn f() -> Int = len(1 + true, 2)");
    assert_eq!(k, vec![("type_mismatch", "f".into(), "body".into()), ("type_mismatch", "f".into(), "body.args[0].right".into())]);
}

#[test]
fn let_scoping_shadowing_and_annotations() {
    assert_eq!(check_src("fn f() -> Bool = let x = 1 in let x = true in x"), vec![]);
    assert_eq!(kinds("fn f() -> Int = (let x = 1 in x) + x"), vec![("unknown_name", "f".into(), "body.right".into())]);
    assert_eq!(check_src("fn f() -> Int = let x: Option[Int] = None in match x { Some(n) => n, None => 0 }"), vec![]);
    assert_eq!(kinds("fn f() -> Int = let x: Int = true in x")[0], ("type_mismatch", "f".into(), "body.value".into()));
}

#[test]
fn let_value_errors_still_check_the_body() {
    let k = kinds("fn f() -> Int = let x = nope in x + true");
    assert_eq!(k, vec![("unknown_name", "f".into(), "body.value".into()), ("type_mismatch", "f".into(), "body.body.right".into())]);
}

#[test]
fn contracts_must_be_bool_and_see_result() {
    assert_eq!(kinds("fn f(x: Int) -> Int req x = x"), vec![("type_mismatch", "f".into(), "req[0]".into())]);
    assert_eq!(kinds("fn f(x: Int) -> Int req r > 0 = x"), vec![("unknown_name", "f".into(), "req[0].left".into())]);
    assert_eq!(check_src("fn f(x: Int) -> Bool ens r == (x > 0) = x > 0"), vec![]);
}

#[test]
fn parameter_named_r_is_rejected_when_ens_exists() {
    let d = one("fn f(r: Int) -> Int ens r > 0 = r");
    assert_eq!((d.kind, d.path.as_str()), ("duplicate_name", "params[0]"));
    assert_eq!(check_src("fn f(r: Int) -> Int req r > 0 = r"), vec![]);
}

#[test]
fn effects_must_be_declared_and_known() {
    let src = "fn say() -> Unit !io = ()\nfn pure_caller() -> Unit = say()\nfn ok() -> Unit !io = say()";
    let d = one(src);
    assert_eq!((d.kind, d.def.as_str(), d.code), ("effect_violation", "pure_caller", "E0301"));
    assert_eq!(d.hints, vec!["add `!io` to the signature of `pure_caller`".to_string()]);
    assert_eq!(kinds("fn f() -> Unit !net = ()")[0], ("unknown_name", "f".into(), "effects[0]".into()));
    assert_eq!(kinds("fn say() -> Unit !io = ()\ntest t = say() == ()")[0].0, "effect_violation");
    assert_eq!(kinds("fn say() -> Unit !io = ()\nfn f() -> Int !io req say() == () = 1")[0].0, "effect_violation");
}

#[test]
fn builtin_effects_and_function_values() {
    assert_eq!(kinds("fn f() -> Unit = print(\"x\")"), vec![("effect_violation", "f".into(), "body".into())]);
    assert_eq!(check_src("fn f(p: Str) -> Result[Str, Str] !fs = read_file(p)"), vec![]);
    let src = "fn shout(s: Str) -> Str !io = { print(s); s }\n\
               fn all(xs: List[Str]) -> List[Str] = map(xs, shout)";
    let d = one(src);
    assert_eq!((d.kind, d.path.as_str()), ("effect_violation", "body.args[1]"));
    let src = "fn twice(f: fn(Int) -> Int, x: Int) -> Int = f(f(x))\nfn inc(x: Int) -> Int = x + 1\ntest t = twice(inc, 1) == 3";
    assert_eq!(check_src(src), vec![]);
    let d = one("fn run(f: fn() -> Unit !io) -> Unit = f()");
    assert_eq!(d.kind, "effect_violation");
    let d = one("fn hi() -> Unit !io = print(\"hi\")\nfn take(f: fn() -> Unit) -> Unit = f()\nfn g() -> Unit = take(hi)");
    assert_eq!((d.kind, d.path.as_str()), ("type_mismatch", "body.args[0]"));
    assert_eq!(kinds("fn f() -> Int = len")[0].0, "unknown_name");
    assert_eq!(kinds("fn len(s: Str) -> Int = 1")[0].0, "duplicate_name");
}

#[test]
fn tests_must_be_bool() {
    assert_eq!(kinds("test t = 1"), vec![("type_mismatch", "t".into(), "body".into())]);
}

#[test]
fn infers_generic_builtins_and_literals() {
    let src = "fn double(x: Int) -> Int = x * 2\n\
               fn add(a: Int, b: Int) -> Int = a + b\n\
               fn positive(x: Int) -> Bool = x > 0\n\
               test t = fold(map(filter(range(-2, 3), positive), double), 0, add) == 6\n\
               test u = len(\"héllo\") == 5 && len([1, 2]) == 2 && get([1.5], 0) == Some(1.5)\n\
               test v = [] == concat([], []) && reverse([\"a\"]) ++ [\"b\"] == [\"a\", \"b\"]\n\
               test w = to_int(\"12\") == Some(12) && from_float(int_to_float(2)) == \"2.0\"";
    let tp = typed(src);
    let TKind::Bin(_, l, _, _) = &tp.tests[1].body.kind else { panic!() };
    let TKind::Bin(_, ll, _, _) = &l.kind else { panic!() };
    let TKind::Bin(_, len_call, _, _) = &ll.kind else { panic!() };
    assert!(matches!(len_call.kind, TKind::Builtin(Builtin::StrLen, ..)));
    assert_eq!(check_src("fn f() -> Int = len(true)")[0].expected.as_deref(), Some("Str or List[_]"));
}

#[test]
fn float_arithmetic_and_no_mixing() {
    assert_eq!(check_src("fn avg(a: Float, b: Float) -> Float = (a + b) / 2.0\ntest t = avg(1.0, 2.0) > 1.4"), vec![]);
    let d = one("fn f(a: Float) -> Float = a + 1");
    assert_eq!((d.path.as_str(), d.expected.as_deref(), d.found.as_deref()), ("body.right", Some("Float"), Some("Int")));
    assert_eq!(d.hints, vec!["convert with `int_to_float(n)`".to_string()]);
    assert_eq!(kinds("fn f(a: Float) -> Float = a % 2.0")[0].2, "body.left");
    assert_eq!(one("fn f(a: Str) -> Str = -a").expected.as_deref(), Some("Int or Float"));
}

#[test]
fn concat_works_on_str_and_list_only() {
    assert_eq!(check_src("test t = \"a\" ++ \"b\" == \"ab\" && [1] ++ [2] == [1, 2]"), vec![]);
    assert_eq!(one("test t = 1 ++ 2 == 3").expected.as_deref(), Some("Str, Bytes or List[_]"));
    assert_eq!(one("test t = \"a\" ++ 1 == \"a\"").path, "body.left.right");
}

#[test]
fn records_and_fields() {
    let src = "type User = { id: Int, name: Str }\n\
               fn mk() -> User = User { name: \"a\", id: 1 }\n\
               test t = mk().name == \"a\" && mk() == User { id: 1, name: \"a\" }";
    let tp = typed(src);
    let TKind::Record(fields) = &tp.fns[0].body.kind else { panic!() };
    let order: Vec<(u32, &TKind)> = fields.iter().map(|(i, x)| (*i, &x.kind)).collect();
    assert_eq!(order, vec![(1, &TKind::Str("a".into())), (0, &TKind::Int(1))], "source order with declaration indices");
    let ty = "type User = { id: Int, name: Str }\n";
    let d = one(&format!("{ty}fn f() -> User = User {{ id: 1 }}"));
    assert_eq!((d.code, d.message.as_str()), ("E0206", "`User` needs field `name`"));
    assert_eq!(kinds(&format!("{ty}fn f() -> User = User {{ id: 1, name: \"\", age: 3 }}"))[0], ("unknown_name", "f".into(), "body.fields.age".into()));
    assert_eq!(kinds(&format!("{ty}fn f(u: User) -> Int = u.age"))[0], ("unknown_name", "f".into(), "body".into()));
    assert_eq!(one(&format!("{ty}fn f(u: Int) -> Int = u.id")).expected.as_deref(), Some("a record"));
    assert_eq!(kinds("fn f(u: Usr) -> Int = 1")[0], ("unknown_name", "f".into(), "params[0]".into()));
}

#[test]
fn enums_variants_and_option_result() {
    let src = "type Err = Empty | NotNumber(Str) | Negative(Int)\n\
               fn parse_age(s: Str) -> Result[Int, Err]\n\
                 ens match r { Ok(n) => n >= 0, _ => true }\n\
               = if len(trim(s)) == 0 then Err(Empty)\n\
                 else match to_int(trim(s)) { None => Err(NotNumber(s)), Some(n) if n < 0 => Err(Negative(n)), Some(n) => Ok(n) }\n\
               test parse_age_ok = parse_age(\" 42 \") == Ok(42)\n\
               test parse_age_neg = parse_age(\"-1\") == Err(Negative(-1))";
    assert_eq!(check_src(src), vec![]);
    assert_eq!(kinds("type E = A | B(Int)\nfn f() -> E = B")[0], ("type_mismatch", "f".into(), "body".into()));
    assert_eq!(kinds("type E = A | B(Int)\nfn f() -> E = B(1, 2)")[0].0, "type_mismatch");
    assert_eq!(kinds("type E = A | Some")[0].0, "duplicate_name");
    assert_eq!(kinds("type E = A\ntype F = A")[0].0, "duplicate_name");
    assert_eq!(kinds("type Int = A")[0].0, "duplicate_name");
    let d = one("fn f() -> Int = Some(1)");
    assert_eq!(d.hints, vec!["unpack the Option with `match v { Some(x) => ..., None => ... }`".to_string()]);
}

#[test]
fn match_patterns_bind_and_type_check() {
    let src = "type P = { x: Int, y: Int }\n\
               fn f(p: P) -> Int = match p { P { x: 0, y } => y, P { x, y: 1 } => x, _ => 0 }";
    let tp = typed(src);
    let TKind::Match(_, arms) = &tp.fns[0].body.kind else { panic!() };
    assert_eq!(arms[0].pat, TPat::Record(vec![(0, TPat::Int(0)), (1, TPat::Bind(1))]));
    assert_eq!(tp.fns[0].locals, vec![Type::Named("P".into()), Type::Int, Type::Int]);
    assert_eq!(kinds("fn f(x: Int) -> Int = match x { \"a\" => 1, _ => 2 }")[0], ("type_mismatch", "f".into(), "body.arms[0].pattern".into()));
    assert_eq!(kinds("fn f(x: Int) -> Int = match x { 1 => true, _ => 2 }")[0], ("type_mismatch", "f".into(), "body.arms[1].body".into()));
    assert_eq!(kinds("fn f(x: Option[Int]) -> Int = match x { Some(a, b) => 1, _ => 2 }")[0].0, "type_mismatch");
    assert_eq!(kinds("fn f(x: Option[Int]) -> Int = match x { Some(a) if a => 1, _ => 2 }")[0].2, "body.arms[0].guard");
    assert_eq!(kinds("type T = A(Int, Int)\nfn f(x: T) -> Int = match x { A(a, a) => a }")[0].0, "duplicate_name");
}

#[test]
fn non_exhaustive_match_is_reported_with_a_witness() {
    let d = one("fn f(x: Option[Int]) -> Int = match x { Some(n) => n }");
    assert_eq!((d.kind, d.code, d.path.as_str(), d.message.as_str()), ("non_exhaustive_match", "E0205", "body", "match does not cover `None`"));
    let d = one("type E = A | B(Bool)\nfn f(x: E) -> Int = match x { A => 1, B(true) => 2 }");
    assert_eq!(d.message, "match does not cover `B(false)`");
    let d = one("fn f(x: Option[Int]) -> Int = match x { None => 0, Some(n) if n > 0 => n }");
    assert_eq!(d.message, "match does not cover `Some(_)`");
    let d = one("fn f(x: Int) -> Int = match x { 0 => 1, 1 => 2 }");
    assert_eq!(d.message, "match does not cover `_`");
    assert_eq!(check_src("fn f(x: Bool, y: Int) -> Int = match x { true => y, false => 0 }"), vec![]);
}

#[test]
fn sequences_discard_values_but_not_results() {
    assert_eq!(check_src("fn f() -> Int !io = { print(\"a\"); print(\"b\"); 3 }"), vec![]);
    // Since the 1e benchmark: any value but a Result may be discarded.
    assert_eq!(check_src("fn f() -> Int = { 1; [1, 2]; Some(\"x\"); 3 }"), vec![]);
    let d = one("fn f() -> Int !fs = { write_file(\"a\", \"b\"); 2 }");
    assert_eq!((d.kind, d.path.as_str(), d.expected.as_deref()), ("type_mismatch", "body.items[0]", Some("Unit")));
    assert!(d.hints[0].starts_with("a Result must be handled with `match` (or bound with `let`)"), "{:?}", d.hints);
    // A value of a type not known yet becomes Unit, as before.
    let tp = typed("fn f() -> Int = { []; 1 }");
    let TKind::Seq(items) = &tp.fns[0].body.kind else { panic!() };
    assert_eq!(items[0].ty, Type::list(Type::Unit));
}

#[test]
fn main_signature_is_checked() {
    assert_eq!(check_src("fn main(args: List[Str]) -> Int !io = { print(\"hi\"); 0 }"), vec![]);
    let k = kinds("fn main() -> Unit = ()");
    assert_eq!(k, vec![("type_mismatch", "main".into(), "params".into()), ("type_mismatch", "main".into(), "ret".into())]);
}

#[test]
fn functions_cannot_be_compared() {
    let d = one("fn a() -> Int = 1\ntest t = a == a");
    assert_eq!((d.path.as_str(), d.hints[0].as_str()), ("body", "functions cannot be compared"));
    let d = one("type R = { f: fn(Int) -> Int, n: Int }\ntype L = Nil | Cons(R, L)\nfn eq(a: L, b: L) -> Bool = a == b");
    assert_eq!(d.hints, vec!["functions cannot be compared".to_string()]);
    assert_eq!(check_src("type L = Nil | Cons(Int, L)\nfn eq(a: L, b: L) -> Bool = a == b"), vec![]);
}

#[test]
fn runtime_export_names_are_reserved() {
    assert_eq!(kinds("fn memory() -> Int = 1")[0].0, "duplicate_name");
    assert_eq!(one("fn qal_alloc() -> Int = 1").hints, vec!["`qal_alloc` is reserved by the runtime".to_string()]);
}

#[test]
fn effects_are_contravariant_in_parameter_position() {
    let src = "fn hi() -> Unit !io = print(\"hi\")\n\
               fn h(cb: fn() -> Unit) -> Unit = cb()\n\
               fn g(k: fn(fn() -> Unit !io) -> Unit) -> Unit = k(hi)\n\
               fn run() -> Unit = g(h)";
    assert!(!check_src(src).is_empty());
    let ok = "fn h(cb: fn() -> Unit !io) -> Unit !io = cb()\n\
              fn g(k: fn(fn() -> Unit) -> Unit !io) -> Unit !io = k(h2)\n\
              fn h2() -> Unit = ()\n\
              fn run() -> Unit !io = g(h)";
    assert_eq!(check_src(ok), vec![]);
}

#[test]
fn duplicate_fields_in_record_patterns_are_rejected() {
    let src = "type R = { x: Bool }\nfn f(p: R) -> Int = match p { R { x: true, x: false } => 1, R { x: false } => 2 }";
    let k = kinds(src);
    assert_eq!(k[0], ("duplicate_name", "f".into(), "body.arms[0].pattern.fields.x".into()));
}

/// `let a1 = [[…a0…]] in let a2 = [[…a1…]] …`: small ASTs, but the inferred type nests `lets * 100` deep.
fn nested_lets(lets: usize) -> String {
    let mut s = "fn main(args: List[Str]) -> Int = let a0 = 1 in ".to_string();
    for i in 1..=lets {
        s += &format!("let a{i} = {}a{}{} in ", "[".repeat(100), i - 1, "]".repeat(100));
    }
    s + &format!("if a{lets} == a{lets} then 0 else 1")
}

#[test]
fn deeply_nested_inferred_types_are_rejected_quickly() {
    assert_eq!(check_src(&nested_lets(2)), vec![], "depth 201 is fine");
    let start = std::time::Instant::now();
    let d = check_src(&nested_lets(16));
    assert!(start.elapsed() < std::time::Duration::from_secs(2), "took {:?}", start.elapsed());
    assert!(!d.is_empty() && d.iter().all(|d| (d.code, d.kind) == ("E0207", "type_too_deep")), "{d:#?}");
    assert_eq!(d[0].message, "inferred type is nested too deeply (max 256 levels)");
    assert_eq!(d[0].path, format!("body.body.body.body.value{}", ".items[0]".repeat(45)), "a3 = List^100(a2: depth 201) crosses 256");
}

#[test]
fn type_depth_is_tracked_when_inner_variables_are_bound_later() {
    // `a` is List^100 of a still unknown element; binding it to List^201 afterwards makes `a` too deep.
    let src = format!(
        "test t = let x = [] in let a = {o}x{c} in let b1 = {o}1{c} in let b = {o}b1{c} in x == [b] && len(a) == 0",
        o = "[".repeat(100),
        c = "]".repeat(100)
    );
    let d = check_src(&src);
    assert!(!d.is_empty() && d.iter().all(|d| d.kind == "type_too_deep"), "{d:#?}");
}

#[test]
fn effect_hint_parenthesizes_a_function_result_type() {
    // `-> fn(Int) -> Int !io` would give the effect to the returned function, not to `mk`.
    let inc = "fn inc(x: Int) -> Int = x + 1\n";
    let d = one(&format!("{inc}fn mk() -> fn(Int) -> Int = {{ print(\"m\"); inc }}"));
    assert_eq!(d.kind, "effect_violation");
    assert_eq!(d.hints, vec!["add `!io` to the signature of `mk`: `-> (fn(Int) -> Int) !io`".to_string()]);
    let d = one(&format!("{inc}fn mk() -> (fn(Int) -> Int !rand) !time = {{ print(\"m\"); inc }}"));
    assert_eq!(d.hints, vec!["add `!io` to the signature of `mk`: `-> (fn(Int) -> Int !rand) !time !io`".to_string()]);
}

/// `let a1 = if true then Ok(a0) else Err(a0) in ...`: each step doubles the inferred type.
fn doubling_lets(lets: usize) -> String {
    let mut s = "test t = let a0 = 1 in ".to_string();
    for i in 1..=lets {
        s += &format!("let a{i} = if true then Ok(a{p}) else Err(a{p}) in ", p = i - 1);
    }
    s + "true"
}

#[test]
fn exponentially_growing_inferred_types_are_rejected_quickly() {
    assert_eq!(check_src(&doubling_lets(10)), vec![], "2^11 - 1 nodes are fine");
    let start = std::time::Instant::now();
    let d = check_src(&doubling_lets(60));
    assert!(start.elapsed() < std::time::Duration::from_secs(2), "took {:?}", start.elapsed());
    assert!(!d.is_empty() && d.iter().all(|d| (d.code, d.kind) == ("E0208", "type_too_large")), "{d:#?}");
    assert_eq!(d[0].message, "inferred type is too large (max 4096 nodes)");
}

#[test]
fn enums_without_variants_and_empty_sequences() {
    assert_eq!(check_src("type Never = |\nfn absurd(n: Never) -> Int = match n { }"), vec![]);
    assert_eq!(kinds("fn f(x: Option[Int]) -> Int = match x { }"), vec![("non_exhaustive_match", "f".into(), "body".into())]);
    // `{}` cannot be parsed, but a hand-built empty sequence must not crash the checker.
    let p = Program { defs: vec![Def::Test(TestDef { name: "t".into(), body: Expr::Bin(BinOp::Eq, Box::new(Expr::Seq(vec![])), Box::new(Expr::Unit)) })] };
    assert_eq!(check_program(&p), vec![]);
}

#[test]
fn a_function_cannot_be_named_r() {
    let d = one("fn r() -> Int = 1");
    assert_eq!((d.kind, d.hints[0].as_str()), ("duplicate_name", "`r` names the result inside `ens`; choose another function name"));
    assert_eq!(check_src("test r = true\nfn f(r: Int) -> Int = r"), vec![]);
}

#[test]
fn builtins_used_as_values_say_so() {
    let d = one("fn f(xs: List[Str]) -> List[Int] = map(xs, len)");
    assert_eq!((d.kind, d.message.as_str()), ("unknown_name", "built-in `len` is not a value"));
    assert_eq!(d.hints, ["built-in `len` cannot be used as a value; wrap it in a closure: `fn(x) => len(x)`"]);
    let d = one("fn f(xs: List[Int]) -> Int = fold(xs, 0, compare)");
    assert_eq!(d.hints, ["built-in `compare` cannot be used as a value; wrap it in a closure: `fn(x, y) => compare(x, y)`"]);
    assert!(one("fn f() -> fn() -> Int !time = now_ms").hints[0].ends_with("`fn() => now_ms()`"));
}

#[test]
fn foreign_type_names_get_a_hint() {
    for (src, want) in [
        ("fn f(s: String) -> Int = 0", "QAL has no type `String`; use `Str`"),
        ("fn f(m: HashMap[Str, Int]) -> Int = 0", "QAL has no type `HashMap`; use `Map[K, V]`"),
        ("fn f(xs: Vec[Int]) -> Int = 0", "QAL has no type `Vec`; use `List[T]`"),
        ("type R = { o: Optional[Int] }", "QAL has no type `Optional`; use `Option[T]`"),
        ("fn f(n: Integer) -> Int = 0", "QAL has no type `Integer`; use `Int` (64-bit)"),
    ] {
        let d = one(src);
        assert_eq!((d.kind, d.hints.as_slice()), ("unknown_name", [want.to_string()].as_slice()), "{src}");
    }
    // Lower-case names are a parse error; the hint is the same.
    for (name, want) in [("i64", "`Int`"), ("bool", "`Bool`"), ("void", "`Unit`"), ("char", "a one-character `Str`"), ("u8", "`Int`"), ("f64", "`Float`"), ("dict", "`Map[K, V]`")] {
        let d = parse_program(&format!("fn f(x: {name}) -> Int = 0")).unwrap_err();
        assert!(d.hints.iter().any(|h| h.starts_with(&format!("QAL has no type `{name}`; use {want}"))), "{name}: {:?}", d.hints);
    }
    // A declared type of that name is fine.
    assert_eq!(check_src("type String = { s: Str }\nfn f(s: String) -> Str = s.s"), vec![]);
}

#[test]
fn strings_are_ordered_and_text_builtins_type_check() {
    let src = "fn before(a: Str, b: Str) -> Bool = a < b || a <= b || a > b || a >= b\n\
               fn words(s: Str) -> List[Str] = sort_str(split(to_lower(replace(s, \",\", \" \")), \" \"))\n\
               fn at(s: Str, sub: Str) -> Int = match index_of(s, sub) { Some(i) => i, None => -1 }\n\
               fn shift(c: Str) -> Str = from_char_code(char_code(c) + 1) ++ to_upper(c)\n\
               fn main(args: List[Str]) -> Int !io = len(read_all())";
    assert_eq!(check_src(src), vec![]);
    let d = one("fn f(a: Bool, b: Bool) -> Bool = a < b");
    assert_eq!((d.kind, d.path.as_str(), d.expected.as_deref()), ("type_mismatch", "body.left", Some("Int, Float or Str")));
    let d = one("fn f(a: Str) -> Bool = a < 1");
    assert_eq!((d.kind, d.path.as_str()), ("type_mismatch", "body.right"));
    assert_eq!(kinds("fn f() -> Str = read_all()"), vec![("effect_violation", "f".into(), "body".into())]);
    assert_eq!(kinds("fn f(xs: List[Int]) -> List[Int] = sort_str(xs)")[0].0, "type_mismatch");
    assert_eq!(kinds("fn char_code(s: Str) -> Int = 1")[0].0, "duplicate_name");
    // The operand types are found from either side.
    assert_eq!(check_src("fn f(xs: List[Str]) -> Bool = match get([], 0) { Some(x) => x < \"a\", None => false }"), vec![]);
    assert_eq!(check_src("fn g() -> Bool = match get([], 0) { Some(x) => x <= 1.5, None => false }"), vec![]);
}

// ---------------------------------------------------------------------------------------------
// Phase 2a: generics, closures, maps, bytes

const GENERIC_LIB: &str = "type Pair[A, B] = { fst: A, snd: B }\n\
    type Tree[T] = Leaf | Node(Tree[T], T, Tree[T])\n\
    fn id[T](x: T) -> T = x\n\
    fn swap[A, B](p: Pair[A, B]) -> Pair[B, A] = Pair { fst: p.snd, snd: p.fst }\n\
    fn size[T](t: Tree[T]) -> Int = match t { Leaf => 0, Node(l, _, r) => size(l) + 1 + size(r) }\n\
    fn insert(t: Tree[Int], x: Int) -> Tree[Int] = match t { Leaf => Node(Leaf, x, Leaf), Node(l, v, r) => if x < v then Node(insert(l, x), v, r) else Node(l, v, insert(r, x)) }\n";

#[test]
fn generic_functions_are_monomorphised_per_type_argument_list() {
    let tp = typed(&format!(
        "{GENERIC_LIB}test t1 = id(1) == 1 && id(\"a\") == \"a\" && id(2) == 2\n\
         test t2 = swap(Pair {{ fst: 1, snd: \"x\" }}).fst == \"x\"\n\
         test t3 = size(insert(insert(Leaf, 2), 1)) == 2"
    ));
    let names: Vec<&str> = tp.fns.iter().map(|f| f.def.name.as_str()).collect();
    assert_eq!(names, ["insert", "id[Int]", "id[Str]", "swap[Int, Str]", "size[Int]"]);
    assert_eq!(tp.generics.len(), 3);
    let swap = &tp.fns[3];
    assert_eq!(swap.def.ret, Type::App("Pair".into(), vec![Type::Str, Type::Int]));
    assert_eq!(swap.instance.as_ref().map(|(g, a)| (*g, a.clone())), Some((1, vec![Type::Int, Type::Str])));
    // No generic node survives monomorphisation.
    fn generic_free(e: &TExpr) -> bool {
        !matches!(e.kind, TKind::CallG(..) | TKind::FnRefG(..)) && crate::infer::children(e).into_iter().all(generic_free)
    }
    assert!(tp.fns.iter().all(|f| generic_free(&f.body)) && tp.tests.iter().all(|t| generic_free(&t.body)));
}

#[test]
fn type_parameters_are_rigid_and_must_be_used() {
    assert_eq!(kinds("fn f[T](x: T) -> Int = x")[0], ("type_mismatch", "f".into(), "body".into()));
    assert_eq!(kinds("fn f[T](x: T) -> T = x + 1")[0].0, "type_mismatch");
    assert_eq!(kinds("fn f[T](x: Int) -> Int = x"), vec![("unused_type_parameter", "f".into(), "tparams[0]".into())]);
    assert_eq!(kinds("fn f[T, T](x: T) -> T = x")[0].0, "duplicate_name");
    assert_eq!(kinds("type T = { a: Int }\nfn f[T](x: T) -> T = x")[0], ("duplicate_name", "f".into(), "tparams[0]".into()));
    assert_eq!(kinds("type Box[T] = { v: T }\nfn f(b: Box) -> Int = 1")[0], ("type_mismatch", "f".into(), "params[0]".into()));
    assert_eq!(kinds("type Box[T] = { v: T }\nfn f(b: Box[Int, Int]) -> Int = 1")[0].0, "type_mismatch");
    assert!(kinds("fn main[T](args: List[Str]) -> Int = 0").contains(&("type_mismatch", "main".into(), "tparams".into())));
    // A type parameter is not comparable with Int, but `==` on it is allowed.
    assert_eq!(check_src("fn same[T](a: T, b: T) -> Bool = a == b\ntest t = same(1, 1) && !same(\"a\", \"b\")"), vec![]);
}

#[test]
fn recursion_must_be_regular() {
    let d = one("type Nest[T] = Nil | Cons(T, Nest[List[T]])");
    assert_eq!((d.code, d.kind), ("E0212", "irregular_recursion"));
    assert_eq!(check_src("type Tree[T] = Leaf | Node(Forest[T], T)\ntype Forest[T] = Empty | More(Tree[T], Forest[T])"), vec![]);
    let d = one("fn f[T](x: T, n: Int) -> Int = if n == 0 then 0 else f([x], n - 1)\ntest t = f(1, 3) == 0");
    assert_eq!((d.code, d.kind, d.def.as_str()), ("E0213", "polymorphic_recursion", "f"));
    // Also when the generic function is never instantiated.
    assert_eq!(one("fn f[T](x: T) -> Int = g(Some(x))\nfn g[U](y: U) -> Int = f(y)").code, "E0213");
    assert_eq!(check_src("fn f[T](x: T, n: Int) -> Int = if n == 0 then 0 else f(x, n - 1)\ntest t = f(true, 3) == 0"), vec![]);
}

#[test]
fn nested_instantiation_is_bounded() {
    // d0[T] calls d1[Pair[T, T]] ...: each level doubles the type argument.
    let mut src = String::from("type Pair[A, B] = { fst: A, snd: B }\n");
    for i in 0..14 {
        src += &format!("fn d{i}[T](x: T) -> Int = d{}(Pair {{ fst: x, snd: x }})\n", i + 1);
    }
    src += "fn d14[T](x: T) -> Int = 0\ntest t = d0(1) == 0";
    let d = check_src(&src);
    assert_eq!(d.len(), 1, "{d:#?}");
    assert_eq!((d[0].code, d[0].kind), ("E0214", "instantiation_limit"));
}

#[test]
fn instances_of_large_bodies_are_bounded() {
    // 450 distinct type arguments of a generic function with a 3 000-node body would copy 1.35
    // million nodes; the limit (500 000) stops monomorphisation early and quickly.
    let mut src = format!("fn big[T](x: T) -> Int = len([{}])\n", vec!["x"; 3000].join(", "));
    for i in 0..450 {
        let (a, b) = (i / 30, i % 30);
        src += &format!("test t{i} = big({}{}1{}{}) == 3000\n", "[".repeat(a), "Some(".repeat(b), ")".repeat(b), "]".repeat(a));
    }
    let start = std::time::Instant::now();
    let d = check_src(&src);
    assert!(start.elapsed() < std::time::Duration::from_secs(10), "{:?}", start.elapsed());
    assert_eq!(d.len(), 1, "{d:#?}");
    assert_eq!((d[0].code, d[0].kind), ("E0214", "instantiation_limit"));
    // 10 instances at small types fit.
    let small: String = src.lines().take(11).collect::<Vec<_>>().join("\n");
    assert_eq!(check_src(&small), vec![]);
}

#[test]
fn closures_capture_locals_and_carry_their_effects() {
    let tp = typed("fn adder(n: Int) -> fn(Int) -> Int = fn(x) => x + n\n\
                    fn logger() -> fn(Str) -> Unit !io = fn(s) => print(s)\n\
                    fn use_it(xs: List[Int], k: Int) -> List[Int] = map(xs, fn(x) => x * k)\n\
                    test t = let f = adder(2) in f(3) == 5");
    let TKind::Lambda(l) = &tp.fns[0].body.kind else { panic!("{:?}", tp.fns[0].body) };
    assert_eq!((l.params.clone(), l.captures.clone()), (vec![1], vec![0]));
    assert_eq!(tp.fns[1].body.ty.to_string(), "fn(Str) -> Unit !io");
    // A pure function may create an io closure, but not call it.
    let d = one("fn bad() -> Unit = let f = fn() => print(\"x\") in f()");
    assert_eq!((d.kind, d.path.as_str()), ("effect_violation", "body.body"));
    assert_eq!(one("fn bad(xs: List[Str]) -> List[Unit] = map(xs, fn(s) => print(s))").kind, "effect_violation");
    assert_eq!(check_src("fn ok(xs: List[Str]) -> List[Unit] !io = map(xs, fn(s) => print(s))"), vec![]);
    // A pure function-typed parameter does not accept an io closure.
    assert_eq!(one("fn app[T](f: fn(T) -> T, x: T) -> T = f(x)\nfn bad() -> Unit !io = app(fn(u) => print(\"x\"), ())").kind, "type_mismatch");
    // Nested closures: the outer one captures what the inner one needs.
    let tp = typed("fn k(a: Int) -> fn() -> fn() -> Int = fn() => fn() => a");
    let TKind::Lambda(outer) = &tp.fns[0].body.kind else { panic!() };
    assert_eq!(outer.captures, vec![0]);
    // Calling a parameter whose type is not known yet asks for an annotation.
    assert_eq!(one("test t = let g = fn(f) => f(1) in true").kind, "type_mismatch");
}

#[test]
fn effects_cannot_be_laundered_through_generic_types() {
    // `Sink[T]` uses T as a parameter type: a sink for pure callbacks must not accept io ones.
    let src = "type Sink[T] = { put: fn(T) -> Unit }\n\
               fn run(cb: fn() -> Unit) -> Unit = cb()\n\
               fn pure_sink() -> Sink[fn() -> Unit] = Sink { put: run }\n\
               fn feed(s: Sink[fn() -> Unit !io]) -> Unit !io = let p = s.put in p(fn() => print(\"x\"))\n\
               fn main(args: List[Str]) -> Int !io = { feed(pure_sink()); 0 }";
    let d = check_src(src);
    assert!(d.iter().any(|d| d.kind == "type_mismatch" && d.def == "main"), "{d:#?}");
}

#[test]
fn comparisons_and_map_keys_need_comparable_types() {
    let d = one("fn same[T](a: T, b: T) -> Bool = a == b\nfn f(g: fn() -> Int) -> Bool = same(g, g)");
    assert_eq!((d.kind, d.def.as_str(), d.path.as_str()), ("type_mismatch", "same[fn() -> Int]", "body"));
    assert_eq!(one("fn f(m: Map[fn() -> Int, Int]) -> Int = 0").expected.as_deref(), Some("a comparable key type"));
    assert_eq!(one("test t = map_size(map_put(map_new(), fn() => 1, 2)) == 1").expected.as_deref(), Some("a comparable key type"));
    let d = one("fn keys[K, V](m: Map[K, V]) -> List[K] = map_keys(m)\nfn f(m: Map[fn() -> Int, Int]) -> Int = 0");
    assert_eq!(d.def, "f");
}

#[test]
fn maps_and_bytes_type_check() {
    let src = "fn count(words: List[Str]) -> Map[Str, Int] = fold(words, map_new(), fn(m, w) => map_put(m, w, match map_get(m, w) { Some(n) => n + 1, None => 1 }))\n\
               test t = map_size(count([\"a\", \"b\", \"a\"])) == 2 && map_keys(count([\"b\", \"a\"])) == [\"a\", \"b\"]\n\
               test b = bytes_len(to_bytes(\"hé\")) == 3 && bytes_get(bytes_from([1, 2]), 1) == Some(2) && bytes_to_str(to_bytes(\"x\") ++ to_bytes(\"y\")) == Some(\"xy\")\n\
               test c = bytes_to_list(bytes_slice(to_bytes(\"abc\"), 1, 3)) == [98, 99] && map_has(map_del(map_put(map_new(), 1, true), 1), 1) == false && map_values(map_put(map_new(), 1, 2)) == [2]";
    assert_eq!(check_src(src), vec![]);
    assert_eq!(kinds("test t = bytes_len(\"s\") == 1")[0].0, "type_mismatch");
}

#[test]
fn compare_needs_a_comparable_type() {
    assert_eq!(check_src("test t = compare(\"a\", \"b\") == -1 && compare([1], [2]) < 0"), vec![]);
    assert_eq!(kinds("fn f(g: fn() -> Int) -> Int = compare(g, g)")[0].0, "type_mismatch");
    assert_eq!(kinds("test t = compare(1, \"a\") == 0")[0].0, "type_mismatch");
    // Inside a generic function it is checked per instance.
    let d = check_src("fn c[T](x: T) -> Int = compare(x, x)\ntest t = c(fn() => 1) == 0");
    assert_eq!(d[0].kind, "type_mismatch", "{d:#?}");
}

#[test]
fn irregular_types_are_rejected_without_expanding_them() {
    // `==` and map keys walk the type; an irregular type would expand forever (review finding).
    for src in [
        "type T[A] = L | N(T[List[A]])\ntest t = L == L",
        "type T[A] = L | N(T[List[A]])\nfn f(x: T[Int]) -> Bool = x == x",
        "type T[A] = { n: Option[T[List[A]]] }\ntest t = T { n: None } == T { n: None }",
        "type T[A] = L | N(T[List[A]])\ntype M = { m: Map[T[Int], Int] }",
    ] {
        let start = std::time::Instant::now();
        let d = check_src(src);
        assert!(start.elapsed() < std::time::Duration::from_secs(2), "{src}: {:?}", start.elapsed());
        assert!(d.iter().any(|d| d.kind == "irregular_recursion"), "{src}: {d:#?}");
    }
}

#[test]
fn instances_at_large_types_are_bounded() {
    // 32 instances of a 150-let body at type arguments of about 2000 nodes each (review finding:
    // 48 s and 2 GB before the copied nodes were weighed by the size of the type arguments).
    let mut src = String::from("type Pair[A, B] = { fst: A, snd: B }\nfn d[T](x: T) -> Pair[T, T] = Pair { fst: x, snd: x }\nfn o[T](x: T) -> Option[T] = Some(x)\nfn l[T](x: T) -> List[T] = [x]\n");
    let chain: String = (0..150).map(|i| format!("let a{i} = {} in ", if i == 0 { "x".to_string() } else { format!("a{}", i - 1) })).collect();
    src += &format!("fn big[T](x: T) -> T = {chain}a149\n");
    let b = (0..10).fold("1".to_string(), |e, _| format!("d({e})"));
    let calls: String = (0..32).map(|k| format!("let r{k} = big({}) in ", (0..5).fold("b0".to_string(), |e, bit| if (k >> bit) & 1 == 1 { format!("o({e})") } else { format!("l({e})") }))).collect();
    src += &format!("fn main2(n: Int) -> Int = let b0 = {b} in {calls}0");
    let start = std::time::Instant::now();
    let d = check_src(&src);
    assert!(start.elapsed() < std::time::Duration::from_secs(10), "{:?}", start.elapsed());
    assert_eq!((d.len(), d[0].kind), (1, "instantiation_limit"), "{d:#?}");
}

/// `B[B[...B[Int]...]]` with `depth` applications.
fn nested_boxes(depth: usize) -> String {
    format!("{}Int{}", "B[".repeat(depth), "]".repeat(depth))
}

#[test]
fn nested_type_applications_unify_in_linear_time() {
    // Review finding C1: invariant arguments were unified once in each direction, 2^depth in all.
    let t = nested_boxes(200);
    let src = format!("type B[T] = {{ v: T }}\nfn f(x: {t}) -> {t} = x\nfn g(x: {t}) -> {t} = let y = f(x) in if true then y else f(y)");
    let start = std::time::Instant::now();
    assert_eq!(check_src(&src), vec![]);
    assert!(start.elapsed() < std::time::Duration::from_secs(1), "took {:?}", start.elapsed());
}

#[test]
fn deep_instance_type_arguments_are_rejected() {
    // Review finding I1: each call wraps the type argument in 241 more levels; inference stays
    // within 256 levels per definition, but the instances nest about 3 900 levels deep.
    let mut src = format!("fn w[T](x: T) -> Option[{}T{}] = None\n", "List[".repeat(240), "]".repeat(240));
    for i in 0..16 {
        src += &format!("fn f{i}[T](x: T) -> Bool = f{}(w(x))\n", i + 1);
    }
    src += "fn f16[T](x: T) -> Bool = map_has(map_put(map_new(), x, 1), x) && x == x && compare(x, x) == 0\n";
    src += "fn main(args: List[Str]) -> Int = if f0(1) && f0(\"s\") then 0 else 1";
    let start = std::time::Instant::now();
    let d = check_src(&src);
    assert!(start.elapsed() < std::time::Duration::from_secs(2), "took {:?}", start.elapsed());
    assert_eq!(d.len(), 1, "{d:#?}");
    assert_eq!((d[0].code, d[0].kind), ("E0214", "instantiation_limit"));
    // One wrapping level fits.
    assert_eq!(check_src(&src.replace("fn f1[T](x: T) -> Bool = f2(w(x))", "fn f1[T](x: T) -> Bool = f16(x)")), vec![]);
}

#[test]
fn long_chains_of_types_and_generic_functions_check_in_linear_time() {
    // Review finding I2: n x n reachability matrices (690 MB for 24 000 types without generics).
    let n = 24_000;
    let mut types: String = (0..n).map(|i| format!("type U{i} = M{i}(U{})\n", i + 1)).collect();
    types += &format!("type U{n} = Z(Int)\n");
    let start = std::time::Instant::now();
    assert_eq!(check_src(&types), vec![]);
    let plain = start.elapsed();
    // The same chain with a generic type at the end (so the irregularity check runs).
    let generic = format!("{types}type G[T] = {{ v: T }}\n").replace(&format!("type U{n} = Z(Int)"), &format!("type U{n} = Z(G[Int])"));
    let start = std::time::Instant::now();
    assert_eq!(check_src(&generic), vec![]);
    let with_generic = start.elapsed();
    let m = 24_000;
    let mut fns: String = (0..m).map(|i| format!("fn g{i}[T](x: T) -> T = g{}(x)\n", i + 1)).collect();
    fns += &format!("fn g{m}[T](x: T) -> T = x\n");
    let start = std::time::Instant::now();
    assert_eq!(check_src(&fns), vec![]);
    let generic_fns = start.elapsed();
    let limit = std::time::Duration::from_secs(5);
    assert!(plain + with_generic + generic_fns < limit, "{plain:?} {with_generic:?} {generic_fns:?}");
}

#[test]
fn strongly_connected_components() {
    // 0 -> 1 -> 2 -> 0, 2 -> 3, 3 -> 4 -> 3, 5 alone.
    let c = crate::scc(&[vec![1], vec![2], vec![0, 3], vec![4], vec![3], vec![]]);
    assert!(c[0] == c[1] && c[1] == c[2] && c[3] == c[4]);
    assert!(c[0] != c[3] && c[5] != c[0] && c[5] != c[3]);
    // A long chain and a long cycle: no recursion, so no stack overflow on a test thread.
    let chain: Vec<Vec<usize>> = (0..200_000).map(|i| if i + 1 < 200_000 { vec![i + 1] } else { vec![] }).collect();
    let c = crate::scc(&chain);
    assert_eq!(c.iter().collect::<std::collections::HashSet<_>>().len(), 200_000);
    let ring: Vec<Vec<usize>> = (0..200_000).map(|i| vec![(i + 1) % 200_000]).collect();
    assert!(crate::scc(&ring).iter().all(|&x| x == 0));
}

#[test]
fn branches_and_list_items_join_function_effects() {
    // Review finding M2: the type of the first branch or item used to be the expected one.
    let p = "fn p() -> Unit !io = print(\"x\")\n";
    for body in [
        "if b then fn() => () else p",
        "if b then p else fn() => ()",
        "match b { true => fn() => (), false => p }",
        "match b { true => p, false => fn() => () }",
    ] {
        assert_eq!(check_src(&format!("{p}fn f(b: Bool) -> fn() -> Unit !io = {body}")), vec![], "{body}");
        // The joined type keeps the effect: calling it needs `!io`.
        let d = one(&format!("{p}fn f(b: Bool) -> Unit = let g = {body} in g()"));
        assert_eq!(d.kind, "effect_violation", "{body}");
    }
    for items in ["[fn() => (), fn() => print(\"x\")]", "[fn() => print(\"x\"), fn() => ()]", "[fn() => (), p, fn() => ()]"] {
        assert_eq!(check_src(&format!("{p}fn f() -> List[fn() -> Unit !io] = {items}")), vec![], "{items}");
        let d = one(&format!("{p}fn f() -> Unit = match get({items}, 0) {{ Some(g) => g(), None => () }}"));
        assert_eq!(d.kind, "effect_violation", "{items}");
    }
    // Nested in containers too.
    assert_eq!(check_src(&format!("{p}fn f(b: Bool) -> Option[fn() -> Unit !io] = if b then Some(fn() => ()) else Some(p)")), vec![]);
    // Parameters meet: a function that may get an io callback and one that only takes pure ones
    // join to a function that takes only pure callbacks.
    let src = format!("{p}fn f(b: Bool) -> Unit = let h = if b then fn(g: fn() -> Unit !io) => () else fn(g: fn() -> Unit) => () in h(p)");
    assert_eq!(one(&src).kind, "type_mismatch");
    assert_eq!(check_src(&src.replace("h(p)", "h(fn() => ())")), vec![]);
    // Generic type arguments stay invariant (a type parameter may be used as a parameter type).
    let boxed = format!("type Box[T] = {{ v: T }}\n{p}fn f(b: Bool) -> Box[fn() -> Unit !io] = if b then Box {{ v: fn() => () }} else Box {{ v: p }}");
    let d = check_src(&boxed);
    assert!(!d.is_empty() && d.iter().all(|d| d.kind == "type_mismatch"), "{d:#?}");
    // Other mismatches are still reported at the later branch.
    let d = one("fn f(b: Bool) -> Int = if b then 1 else \"s\"");
    assert_eq!((d.kind, d.path.as_str()), ("type_mismatch", "body.else"));
}
