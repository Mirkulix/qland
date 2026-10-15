use super::*;
use qal_core::tast::{TKind, TPat};
use qal_text::parse_program;

fn check_src(src: &str) -> Vec<Diagnostic> {
    check_program(&parse_program(src).unwrap())
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
    assert_eq!(one("test t = 1 ++ 2 == 3").expected.as_deref(), Some("Str or List[_]"));
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
fn sequences_discard_only_unit() {
    assert_eq!(check_src("fn f() -> Int !io = { print(\"a\"); print(\"b\"); 3 }"), vec![]);
    let d = one("fn f() -> Int = { 1; 2 }");
    assert_eq!((d.path.as_str(), d.expected.as_deref()), ("body.items[0]", Some("Unit")));
    assert!(d.hints[0].starts_with("only Unit values can be discarded"));
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
    assert_eq!(d.hints, ["built-in `len` cannot be used as a value; wrap it in a fn, e.g. `fn my_len(...) = len(...)`, and pass `my_len`"]);
}
