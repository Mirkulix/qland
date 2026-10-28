//! The self-hosting built-ins of Phase 5a (spec §5.2): bit operations, float bits, allocation-free
//! indexing, the `Accum` prelude type with `acc_list`, `bytes_join` and the host helper `sha256`.

use qal_run::{Runner, Value};
use qal_compiler::Entry;

fn runner(src: &str) -> Runner {
    let p = qal_compiler::parse_program(src).unwrap_or_else(|d| panic!("{d:?}"));
    let tp = qal_compiler::check(&p).unwrap_or_else(|d| panic!("{d:#?}"));
    Runner::new(qal_compiler::compile(&tp, &Entry::All).unwrap()).unwrap()
}

fn diags(src: &str) -> Vec<qal_core::diag::Diagnostic> {
    qal_compiler::check(&qal_compiler::parse_program(src).unwrap()).err().unwrap_or_default()
}

fn int(n: i64) -> Value {
    Value::Int(n)
}

fn bytes(b: &[u8]) -> Value {
    Value::Bytes(b.to_vec(), 0)
}

#[test]
fn bit_operations_are_twos_complement_on_64_bits() {
    let r = runner(
        "fn and(a: Int, b: Int) -> Int = bit_and(a, b)\nfn or(a: Int, b: Int) -> Int = bit_or(a, b)\n\
         fn xor(a: Int, b: Int) -> Int = bit_xor(a, b)\nfn not(a: Int) -> Int = bit_not(a)",
    );
    let samples = [0i64, 1, -1, 2, 7, 0x55, -128, 1 << 40, i64::MAX, i64::MIN, -0x1234_5678_9abc];
    for &a in &samples {
        assert_eq!(r.call("not", &[int(a)]), Ok(int(!a)), "not {a}");
        for &b in &samples {
            assert_eq!(r.call("and", &[int(a), int(b)]), Ok(int(a & b)), "{a} & {b}");
            assert_eq!(r.call("or", &[int(a), int(b)]), Ok(int(a | b)), "{a} | {b}");
            assert_eq!(r.call("xor", &[int(a), int(b)]), Ok(int(a ^ b)), "{a} ^ {b}");
        }
    }
}

#[test]
fn shifts_need_a_distance_below_64() {
    let r = runner("fn l(a: Int, n: Int) -> Int = shl(a, n)\nfn sr(a: Int, n: Int) -> Int = shr(a, n)\nfn u(a: Int, n: Int) -> Int = ushr(a, n)");
    for &a in &[0i64, 1, -1, 300, -300, i64::MAX, i64::MIN, 0x0123_4567_89ab_cdef] {
        for n in [0i64, 1, 7, 31, 32, 63] {
            assert_eq!(r.call("l", &[int(a), int(n)]), Ok(int(a.wrapping_shl(n as u32))), "{a} << {n}");
            assert_eq!(r.call("sr", &[int(a), int(n)]), Ok(int(a >> n)), "{a} >> {n}");
            assert_eq!(r.call("u", &[int(a), int(n)]), Ok(int(((a as u64) >> n) as i64)), "{a} >>> {n}");
        }
    }
    for (f, name) in [("l", "shl"), ("sr", "shr"), ("u", "ushr")] {
        for n in [64i64, 65, -1, i64::MIN, i64::MAX] {
            let e = r.call(f, &[int(5), int(n)]).unwrap_err();
            let label = format!("{name}: req 0 <= n && n < 64");
            assert_eq!((e.kind, e.def.as_str(), e.contract.as_deref()), ("contract_violation", f, Some(label.as_str())), "{name}(5, {n})");
            let json = serde_json::to_value(&e).unwrap();
            assert_eq!((json["values"]["a"].clone(), json["values"]["n"].clone()), (serde_json::json!(5), serde_json::json!(n)), "{json}");
        }
    }
}

#[test]
fn float_bits_round_trip_every_pattern() {
    let r = runner("fn b(x: Float) -> Int = float_bits(x)\nfn f(i: Int) -> Float = float_of_bits(i)\nfn rt(i: Int) -> Int = float_bits(float_of_bits(i))");
    for x in [0.0f64, -0.0, 1.5, -2.25, f64::MAX, f64::MIN_POSITIVE, 5e-324, f64::INFINITY, f64::NEG_INFINITY, 0.1] {
        assert_eq!(r.call("b", &[Value::Float(x)]), Ok(int(x.to_bits() as i64)), "{x}");
        match r.call("f", &[int(x.to_bits() as i64)]) {
            Ok(Value::Float(y)) => assert_eq!(y.to_bits(), x.to_bits(), "{x}"),
            other => panic!("{other:?}"),
        }
    }
    // NaN payloads and every sign survive the round trip (reinterpretation never canonicalizes).
    for i in [0x7ff8_0000_0000_0000u64 as i64, 0x7ff0_0000_0000_0001u64 as i64, -1, i64::MIN, 0, 42] {
        assert_eq!(r.call("rt", &[int(i)]), Ok(int(i)), "{i:#x}");
    }
}

#[test]
fn list_at_and_bytes_at_index_without_allocating() {
    let r = runner(
        "fn i(xs: List[Int], k: Int) -> Int = list_at(xs, k)\nfn s(xs: List[Str], k: Int) -> Str = list_at(xs, k)\n\
         fn f(xs: List[Float], k: Int) -> Float = list_at(xs, k)\nfn b(xs: List[Bool], k: Int) -> Bool = list_at(xs, k)\n\
         fn o(xs: List[Option[Int]], k: Int) -> Option[Int] = list_at(xs, k)\nfn by(b: Bytes, k: Int) -> Int = bytes_at(b, k)\n\
         fn sum(b: Bytes, k: Int, acc: Int) -> Int = if k == bytes_len(b) then acc else sum(b, k + 1, acc + bytes_at(b, k))",
    );
    let ints = Value::List(vec![int(10), int(-20), int(i64::MIN)]);
    assert_eq!(r.call("i", &[ints.clone(), int(2)]), Ok(int(i64::MIN)));
    assert_eq!(r.call("s", &[Value::List(vec![Value::Str("a".into()), Value::Str("ü".into())]), int(1)]), Ok(Value::Str("ü".into())));
    assert_eq!(r.call("f", &[Value::List(vec![Value::Float(0.5), Value::Float(-3.25)]), int(1)]), Ok(Value::Float(-3.25)));
    assert_eq!(r.call("b", &[Value::List(vec![Value::Bool(false), Value::Bool(true)]), int(1)]), Ok(Value::Bool(true)));
    assert_eq!(r.call("o", &[Value::List(vec![Value::Variant("Some".into(), vec![int(4)])]), int(0)]), Ok(Value::Variant("Some".into(), vec![int(4)])));
    assert_eq!(r.call("by", &[bytes(&[0, 128, 255]), int(2)]), Ok(int(255)));
    assert_eq!(r.call("sum", &[bytes(&(0..=255).collect::<Vec<u8>>()), int(0), int(0)]), Ok(int(255 * 128)));
    for k in [3i64, -1, i64::MIN, i64::MAX, 1 << 32] {
        let e = r.call("i", &[ints.clone(), int(k)]).unwrap_err();
        assert_eq!((e.kind, e.code), ("index", "E0505"), "list_at({k})");
        let json = serde_json::to_value(&e).unwrap();
        assert_eq!((json["values"]["i"].clone(), json["values"]["len"].clone()), (serde_json::json!(k), serde_json::json!(3)), "{json}");
        let e = r.call("by", &[bytes(&[1, 2, 3]), int(k)]).unwrap_err();
        assert_eq!(e.kind, "index", "bytes_at({k})");
    }
    assert_eq!(r.call("i", &[Value::List(vec![]), int(0)]).unwrap_err().kind, "index");
    assert_eq!(r.call("by", &[bytes(&[]), int(0)]).unwrap_err().kind, "index");
}

#[test]
fn accum_builds_lists_in_append_order() {
    let r = runner(
        "fn build(n: Int, a: Accum[Int]) -> Accum[Int] = if n == 0 then a else build(n - 1, AccPush(a, n))\n\
         fn list(n: Int) -> List[Int] = acc_list(build(n, AccEnd))\n\
         fn strs() -> List[Str] = acc_list(AccPush(AccPush(AccPush(AccEnd, \"a\"), \"b\"), \"c\"))\n\
         fn floats() -> List[Float] = acc_list(AccPush(AccPush(AccEnd, 1.5), -0.0))\n\
         fn nested() -> List[List[Int]] = acc_list(AccPush(AccPush(AccEnd, [1]), [2, 3]))\n\
         fn count(a: Accum[Str]) -> Int = match a { AccEnd => 0, AccPush(rest, _) => 1 + count(rest) }\n\
         fn last(a: Accum[Str]) -> Option[Str] = match a { AccEnd => None, AccPush(_, x) => Some(x) }\n\
         type Builder = { items: Accum[Str], n: Int }\n\
         fn add(b: Builder, s: Str) -> Builder = Builder { items: AccPush(b.items, s), n: b.n + 1 }\n\
         fn built() -> List[Str] = acc_list(add(add(Builder { items: AccEnd, n: 0 }, \"x\"), \"y\").items)\n\
         fn same() -> Bool = AccPush(AccEnd, 1) == AccPush(AccEnd, 1) && compare(AccPush(AccEnd, 1), AccPush(AccEnd, 2)) == -1\n\
         fn empty() -> List[Int] = acc_list(AccEnd)",
    );
    assert_eq!(r.call("list", &[int(4)]), Ok(Value::List(vec![int(4), int(3), int(2), int(1)])));
    assert_eq!(r.call("empty", &[]), Ok(Value::List(vec![])));
    let strs = ["a", "b", "c"].iter().map(|s| Value::Str(s.to_string())).collect();
    assert_eq!(r.call("strs", &[]), Ok(Value::List(strs)));
    match r.call("floats", &[]) {
        Ok(Value::List(xs)) => assert!(matches!(xs.as_slice(), [Value::Float(a), Value::Float(b)] if *a == 1.5 && b.to_bits() == (-0.0f64).to_bits())),
        other => panic!("{other:?}"),
    }
    assert_eq!(r.call("nested", &[]), Ok(Value::List(vec![Value::List(vec![int(1)]), Value::List(vec![int(2), int(3)])])));
    assert_eq!(r.call("built", &[]), Ok(Value::List(vec![Value::Str("x".into()), Value::Str("y".into())])));
    assert_eq!(r.call("same", &[]), Ok(Value::Bool(true)));
    // A long accumulator converts in linear time and memory (one list allocation).
    let long = runner(
        "fn build(n: Int, a: Accum[Int]) -> Accum[Int] = if n == 0 then a else build(n - 1, AccPush(a, n))\n\
         fn ends(n: Int) -> List[Int] = let xs = acc_list(build(n, AccEnd)) in [len(xs), list_at(xs, 0), list_at(xs, n - 1)]",
    );
    assert_eq!(long.call("ends", &[int(200_000)]), Ok(Value::List(vec![int(200_000), int(200_000), int(1)])));
    // Accum values cross the host boundary like any generic enum.
    let acc = Value::Variant("AccPush".into(), vec![Value::Variant("AccEnd".into(), vec![]), Value::Str("q".into())]);
    assert_eq!(r.call("count", &[acc.clone()]), Ok(int(1)));
    assert_eq!(r.call("last", &[acc]), Ok(Value::Variant("Some".into(), vec![Value::Str("q".into())])));
}

#[test]
fn bytes_join_concatenates_with_one_copy() {
    let r = runner(
        "fn j(bs: List[Bytes]) -> Bytes = bytes_join(bs)\n\
         fn pieces(n: Int, a: Accum[Bytes]) -> Accum[Bytes] = if n == 0 then a else pieces(n - 1, AccPush(a, bytes_from([n % 256, 0, 1, 2, 3, 4, 5, 6, 7, 8])))\n\
         fn big(n: Int) -> Int = bytes_len(bytes_join(acc_list(pieces(n, AccEnd))))\n\
         fn str_too(s: Str) -> Option[Str] = bytes_to_str(bytes_join([to_bytes(s), to_bytes(\"!\"), to_bytes(s)]))",
    );
    assert_eq!(r.call("j", &[Value::List(vec![])]), Ok(bytes(&[])));
    assert_eq!(r.call("j", &[Value::List(vec![bytes(&[1]), bytes(&[]), bytes(&[2, 3]), bytes(&[255])])]), Ok(bytes(&[1, 2, 3, 255])));
    assert_eq!(r.call("str_too", &[Value::Str("é".into())]), Ok(Value::Variant("Some".into(), vec![Value::Str("é!é".into())])));
    // 10^5 pieces of 10 bytes: a 1 MB result in linear time.
    assert_eq!(r.call("big", &[int(100_000)]), Ok(int(1_000_000)));
}

#[test]
fn sha256_is_the_standard_digest() {
    let r = runner("fn h(b: Bytes) -> Bytes = sha256(b)\nfn hs(s: Str) -> Int = bytes_len(sha256(to_bytes(s)))\ntest twice = sha256(to_bytes(\"a\")) == sha256(to_bytes(\"a\"))");
    let hex = |v: Value| match v {
        Value::Bytes(b, 0) => b.iter().map(|x| format!("{x:02x}")).collect::<String>(),
        other => panic!("{other:?}"),
    };
    assert_eq!(hex(r.call("h", &[bytes(b"")]).unwrap()), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert_eq!(hex(r.call("h", &[bytes(b"abc")]).unwrap()), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    let long: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
    assert_eq!(hex(r.call("h", &[bytes(&long)]).unwrap()), qal_core::canon::Hash::of(&long).to_hex());
    assert_eq!(r.call("hs", &[Value::Str("x".repeat(1000))]), Ok(int(32)));
    assert!(r.run_tests().iter().all(|t| t.passed));
    // Hashing costs fuel (review M5): hashing 1 MiB 2000 times exceeds a budget of 10^7.
    let looping = runner("fn big(b: Bytes, k: Int) -> Bytes = if k == 0 then b else big(b ++ b, k - 1)\n\
                          fn spin(b: Bytes, n: Int) -> Int = if n == 0 then 0 else bytes_len(sha256(b)) + spin(b, n - 1) - 32\n\
                          fn go(n: Int) -> Int = spin(big(to_bytes(\"x\"), 20), n)").with_fuel(10_000_000);
    assert_eq!(looping.call("go", &[int(1)]), Ok(int(0)));
    assert_eq!(looping.call("go", &[int(2000)]).unwrap_err().kind, "timeout");
    // Only a module that uses it imports the helper.
    let p = qal_compiler::check(&qal_compiler::parse_program("fn f(b: Bytes) -> Bytes = sha256(b)\nfn g(x: Int) -> Int = x").unwrap()).unwrap();
    assert!(qal_compiler::compile(&p, &Entry::Fn("f".into())).unwrap().imports.contains(&"sha256"));
    assert!(!qal_compiler::compile(&p, &Entry::Fn("g".into())).unwrap().imports.contains(&"sha256"));
}

#[test]
fn new_builtins_are_typed_and_their_names_reserved() {
    let mismatch = |src: &str| diags(src).iter().any(|d| d.kind == "type_mismatch");
    assert!(mismatch("fn f(x: Float) -> Int = bit_and(x, 1)"));
    assert!(mismatch("fn f(x: Int) -> Int = float_bits(x)"));
    assert!(mismatch("fn f(xs: List[Int]) -> Str = list_at(xs, 0)"));
    assert!(mismatch("fn f(b: Str) -> Int = bytes_at(b, 0)"));
    assert!(mismatch("fn f(xs: List[Str]) -> Bytes = bytes_join(xs)"));
    assert!(mismatch("fn f(a: Accum[Int]) -> List[Str] = acc_list(a)"));
    assert!(diags("fn f(x: Int) -> Int = shl(x)").iter().any(|d| d.code == "E0204"));
    for name in ["bit_and", "shl", "ushr", "float_bits", "list_at", "bytes_at", "acc_list", "bytes_join", "sha256"] {
        let d = diags(&format!("fn {name}(x: Int) -> Int = x"));
        assert!(d.iter().any(|d| d.is_builtin_name()), "{name}: {d:?}");
    }
    // Accum and its variants are prelude names; it takes exactly one type argument.
    for src in ["type Accum = A | B", "type T = AccEnd | Other", "type T = { x: Int }\nfn f(a: Accum) -> Int = 0", "fn f(a: Accum[Int, Str]) -> Int = 0"] {
        assert!(!diags(src).is_empty(), "{src}");
    }
    assert!(diags("fn f(a: Accum[Accum[Str]]) -> List[Accum[Str]] = acc_list(a)").is_empty());
    // The prelude's type parameter does not clash with a user type named `T`.
    assert!(diags("type T = A | B\nfn f(a: Accum[T]) -> List[T] = acc_list(AccPush(a, A))").is_empty());
    // A codebase stored before Phase 5a with its own `Accum` keeps working without the prelude,
    // and cannot use acc_list until it renames the type (like http/db_* in Phase 3).
    let legacy = qal_compiler::parse_program("type Accum = Box(Int)\nfn f(a: Accum) -> Int = match a { Box(n) => n }").unwrap();
    assert!(qal_compiler::check_stored(&legacy, &|_| false).is_ok());
    let legacy = qal_compiler::parse_program("type Accum = Box(Int)\nfn g(xs: List[Int]) -> List[Int] = acc_list(xs)").unwrap();
    let d = qal_compiler::check_stored(&legacy, &|n| n == "g").unwrap_err();
    assert!(d.iter().any(|d| d.code == "E0240" && d.def == "g"), "{d:?}");
    // Built-ins are pure: no effect is needed.
    assert!(diags("fn f(b: Bytes) -> Int = bytes_at(sha256(b), 0) + shr(bit_not(0), 1)\ntest t = f(to_bytes(\"\")) > 0").is_empty());
}
