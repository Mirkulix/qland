use super::*;
use qal_compiler::Entry;
use serde_json::json;

fn compile(src: &str, entry: Entry) -> CompiledModule {
    let p = qal_compiler::parse_program(src).unwrap();
    let tp = qal_compiler::check(&p).unwrap_or_else(|d| panic!("{d:#?}"));
    qal_compiler::compile(&tp, &entry).unwrap()
}

fn runner(src: &str) -> Runner {
    Runner::new(compile(src, Entry::All)).unwrap()
}

fn main_runner(src: &str, allow: &str) -> Runner {
    Runner::new(compile(src, Entry::Fn("main".into()))).unwrap().with_permissions(Permissions::parse(allow).unwrap())
}

fn str_list(xs: &[&str]) -> Value {
    Value::List(xs.iter().map(|s| Value::Str(s.to_string())).collect())
}

#[test]
fn calls_with_typed_values() {
    let r = runner(
        "fn pick(c: Bool, a: Int, b: Int) -> Int = if c then a else b\nfn u() -> Unit = ()\nfn neg(b: Bool) -> Bool = !b\n\
         fn greet(name: Str, xs: List[Str]) -> Str = name ++ \":\" ++ join(xs, \",\")\nfn half(x: Float) -> Float = x / 2.0",
    );
    assert_eq!(r.call("pick", &[Value::Bool(false), Value::Int(1), Value::Int(2)]), Ok(Value::Int(2)));
    assert_eq!(r.call("u", &[]), Ok(Value::Unit));
    assert_eq!(r.call("neg", &[Value::Bool(true)]), Ok(Value::Bool(false)));
    assert_eq!(r.call("greet", &[Value::Str("é".into()), str_list(&["a", "b"])]), Ok(Value::Str("é:a,b".into())));
    assert_eq!(r.call("half", &[Value::Float(3.0)]), Ok(Value::Float(1.5)));
}

#[test]
fn returns_structured_values() {
    let r = runner(
        "type P = { x: Int, tags: List[Str] }\ntype E = A | B(P, Option[Int])\n\
         fn mk() -> E = B(P { x: 1, tags: [\"t\"] }, None)\nfn parse(s: Str) -> Result[Int, Str] = match to_int(s) { Some(n) => Ok(n), None => Err(s) }",
    );
    let v = r.call("mk", &[]).unwrap();
    assert_eq!(serde_json::to_value(&v).unwrap(), json!({"B": [{"x": 1, "tags": ["t"]}, "None"]}));
    assert_eq!(serde_json::to_value(r.call("parse", &[Value::Str("x".into())]).unwrap()).unwrap(), json!({"Err": ["x"]}));
}

#[test]
fn rejects_bad_arguments_and_unknown_functions() {
    let r = runner("fn f(a: Int) -> Int = a\ntest t = true");
    assert_eq!(r.call("f", &[]).unwrap_err().kind, "bad_args");
    let e = r.call("f", &[Value::Bool(true)]).unwrap_err();
    assert_eq!((e.kind, e.code, e.message.as_str()), ("bad_args", "E0509", "argument 0 (`a`): expected Int, got true"));
    let e = r.call("g", &[]).unwrap_err();
    assert_eq!((e.kind, e.def.as_str()), ("unknown_function", "g"));
    assert_eq!(r.call("test:nope", &[]).unwrap_err().def, "nope", "def is the definition name, not the export");
}

#[test]
fn contract_violation_is_a_diagnostic_with_values() {
    let r = runner("fn half(n: Int, label: Str) -> Int req n % 2 == 0 = n / 2");
    let e = r.call("half", &[Value::Int(3), Value::Str("x".into())]).unwrap_err();
    assert_eq!(
        serde_json::to_value(&e).unwrap(),
        json!({"code": "E0501", "kind": "contract_violation", "def": "half", "path": "req[0]",
            "message": "`req n % 2 == 0` of `half` is false", "contract": "req[0]: n % 2 == 0",
            "values": {"n": 3, "label": "x"},
            "hints": ["the caller must ensure `n % 2 == 0` before calling `half`"]})
    );
}

#[test]
fn ens_violation_reports_parameters_and_result() {
    let r = runner("fn inc(n: Int) -> Int ens r > n = if n > 5 then n else n + 1");
    assert_eq!(r.call("inc", &[Value::Int(1)]), Ok(Value::Int(2)));
    let e = r.call("inc", &[Value::Int(9)]).unwrap_err();
    assert_eq!((e.kind, e.path.as_str(), e.contract.as_deref()), ("contract_violation", "ens[0]", Some("ens[0]: r > n")));
    assert_eq!(serde_json::to_value(&e.values).unwrap(), json!({"n": 9, "r": 9}));
}

#[test]
fn arithmetic_failures_name_def_path_and_operands() {
    let r = runner("fn inner(a: Int) -> Int = a / 0\nfn outer() -> Int = inner(1) + 1\nfn big(x: Int) -> Int = 2 * (x * x)");
    let e = r.call("outer", &[]).unwrap_err();
    assert_eq!((e.kind, e.code, e.def.as_str(), e.path.as_str()), ("div_by_zero", "E0503", "inner", "body"));
    assert_eq!(serde_json::to_value(&e.values).unwrap(), json!({"left": 1, "right": 0}));
    let e = r.call("big", &[Value::Int(4_000_000_000)]).unwrap_err();
    assert_eq!((e.kind, e.path.as_str()), ("int_overflow", "body.right"));
    assert_eq!(serde_json::to_value(&e.values).unwrap(), json!({"left": 4_000_000_000i64, "right": 4_000_000_000i64}));
}

#[test]
fn builtin_failures_are_reported() {
    let r = runner("fn cut(s: Str) -> Str = slice(s, 1, 10)\nfn parts(s: Str) -> List[Str] = split(s, \"\")");
    let e = r.call("cut", &[Value::Str("abc".into())]).unwrap_err();
    assert_eq!((e.kind, e.code, e.def.as_str(), e.path.as_str()), ("index", "E0505", "cut", "body"));
    assert_eq!(serde_json::to_value(&e.values).unwrap(), json!({"start": 1, "end": 10, "len": 3}));
    let e = r.call("parts", &[Value::Str("abc".into())]).unwrap_err();
    assert_eq!((e.kind, e.contract.as_deref()), ("contract_violation", Some("split: req sep != \"\"")));
    let r = main_runner("fn main(args: List[Str]) -> Int !rand = rand_int(5, 1)", "rand");
    let e = r.run_main(&[]).result.unwrap_err();
    assert_eq!(serde_json::to_value(&e.values).unwrap(), json!({"lo": 5, "hi": 1}));
}

#[test]
fn memory_limit_reports_out_of_memory() {
    let r = runner("fn big(n: Int) -> Int = len(range(0, n))").with_memory_limit(1 << 20);
    assert_eq!(r.call("big", &[Value::Int(1000)]), Ok(Value::Int(1000)));
    let e = r.call("big", &[Value::Int(1_000_000)]).unwrap_err();
    assert_eq!((e.kind, e.code, e.def.as_str()), ("out_of_memory", "E0504", "big"));
    assert_eq!(serde_json::to_value(&e.values).unwrap(), json!({"requested_bytes": 8_000_008, "limit_bytes": 1 << 20}));
}

#[test]
fn infinite_loop_times_out() {
    let r = runner("fn spin(n: Int) -> Int = spin(n)").with_fuel(10_000_000);
    assert_eq!(r.call("spin", &[Value::Int(0)]).unwrap_err().kind, "timeout");
}

#[test]
fn deep_non_tail_recursion_reports_stack_overflow() {
    let r = runner("fn down(n: Int) -> Int = if n == 0 then 0 else 1 + down(n - 1)");
    let e = r.call("down", &[Value::Int(100_000_000)]).unwrap_err();
    assert_eq!(e.kind, "stack_overflow");
    assert!(e.hints[0].contains("tail-recursive"));
}

#[test]
fn recursion_depth_of_twenty_thousand_works() {
    let r = runner("fn deep(n: Int) -> Int = if n == 0 then 0 else 1 + deep(n - 1)");
    assert_eq!(r.call("deep", &[Value::Int(20_000)]), Ok(Value::Int(20_000)));
}

#[test]
fn deep_recursion_is_safe_on_a_small_caller_stack() {
    let r = runner("fn deep(n: Int) -> Int = if n == 0 then 0 else 1 + deep(n - 1)");
    let kind = std::thread::Builder::new()
        .stack_size(256 << 10)
        .spawn(move || {
            (r.call("deep", &[Value::Int(20_000)]), r.call("deep", &[Value::Int(100_000_000)]).unwrap_err().kind)
        })
        .unwrap()
        .join()
        .unwrap();
    assert_eq!(kind, (Ok(Value::Int(20_000)), "stack_overflow"));
}

#[test]
fn runs_tests_in_order() {
    let r = Runner::new(compile("test a = 1 == 1\ntest b = 1 == 2\ntest c = 1 / 0 == 0", Entry::Tests)).unwrap();
    let res = r.run_tests();
    assert_eq!(res.iter().map(|t| (t.name.as_str(), t.passed)).collect::<Vec<_>>(), vec![("a", true), ("b", false), ("c", false)]);
    let e = res[2].error.as_ref().unwrap();
    assert_eq!((e.kind, e.def.as_str()), ("div_by_zero", "c"));
}

#[test]
fn main_gets_args_and_prints() {
    let src = "fn main(args: List[Str]) -> Int !io = match get(args, 0) {\n\
                 None => { print(\"usage: hello <name>\"); 2 },\n\
                 Some(n) => { print(\"hello \" ++ n); print(from_int(len(args))); 0 },\n\
               }";
    let r = main_runner(src, "io");
    assert_eq!(r.run_main(&["ada".into(), "x".into()]), Outcome { result: Ok(Value::Int(0)), stdout: "hello ada\n2\n".into(), stdout_bytes: 12, db: "none" });
    assert_eq!(r.run_main(&[]), Outcome { result: Ok(Value::Int(2)), stdout: "usage: hello <name>\n".into(), stdout_bytes: 20, db: "none" });
}

#[test]
fn read_line_reads_stdin_until_end() {
    let src = "fn main(args: List[Str]) -> Int !io = loop(0)\n\
               fn loop(total: Int) -> Int !io = match read_line() {\n\
                 None => { print(from_int(total)); 0 },\n\
                 Some(line) => loop(total + (match to_int(trim(line)) { Some(n) => n, None => 0 })),\n\
               }";
    let r = main_runner(src, "io").with_stdin("1\n2\r\nx\n39");
    assert_eq!(r.run_main(&[]).stdout, "42\n");
}

#[test]
fn read_all_reads_the_rest_of_stdin() {
    let src = "fn main(args: List[Str]) -> Int !io = match read_line() {\n\
                 None => 1,\n\
                 Some(first) => let rest = read_all() in { print(first ++ \"|\" ++ rest ++ \"|\" ++ read_all()); len(split(rest, \"\\n\")) },\n\
               }";
    let r = main_runner(src, "io").with_stdin("head\r\nb\r\nc é");
    let out = r.run_main(&[]);
    assert_eq!((out.result, out.stdout.as_str()), (Ok(Value::Int(2)), "head|b\r\nc é|\n"));
    let r = main_runner("fn main(args: List[Str]) -> Int !io = len(read_all())", "io").with_stdin(vec![0xFF, b'a']);
    assert_eq!(r.run_main(&[]).result, Ok(Value::Int(2)), "invalid UTF-8 is replaced by U+FFFD");
    let r = main_runner("fn main(args: List[Str]) -> Int !io = len(read_all())", "io").with_memory_limit(1 << 20).with_stdin(vec![b'x'; 2 << 20]);
    assert_eq!(r.run_main(&[]).result.unwrap_err().kind, "out_of_memory");
    let r = main_runner("fn main(args: List[Str]) -> Int !io = len(read_all())", "");
    assert_eq!(r.run_main(&[]).result.unwrap_err().kind, "permission_denied");
}

#[test]
fn text_builtin_failures_are_contract_violations_with_values() {
    let r = runner("fn code(s: Str) -> Int = char_code(s)\nfn text(n: Int) -> Str = from_char_code(n)\nfn rep(s: Str, f: Str) -> Str = replace(s, f, \"-\")");
    let e = r.call("code", &[Value::Str("".into())]).unwrap_err();
    assert_eq!((e.kind, e.contract.as_deref()), ("contract_violation", Some("char_code: req s != \"\"")));
    assert_eq!(serde_json::to_value(&e.values).unwrap(), json!({"s": ""}));
    let e = r.call("text", &[Value::Int(55296)]).unwrap_err();
    assert_eq!(e.message, "precondition of a built-in was violated: `from_char_code: req 0 <= n <= 1114111 && !(55296 <= n <= 57343)`");
    assert_eq!(serde_json::to_value(&e.values).unwrap(), json!({"n": 55296}));
    assert_eq!(r.call("text", &[Value::Int(97)]), Ok(Value::Str("a".into())));
    let e = r.call("rep", &[Value::Str("abc".into()), Value::Str("".into())]).unwrap_err();
    assert_eq!(serde_json::to_value(&e.values).unwrap(), json!({"from": ""}));
    assert_eq!(r.call("rep", &[Value::Str("a b".into()), Value::Str(" ".into())]), Ok(Value::Str("a-b".into())));
}

#[test]
fn effects_must_be_granted() {
    let r = main_runner("fn main(args: List[Str]) -> Int !io = { print(\"hi\"); 0 }", "");
    let out = r.run_main(&[]);
    let e = out.result.unwrap_err();
    assert_eq!((e.kind, e.code, e.def.as_str()), ("permission_denied", "E0508", "main"));
    assert_eq!(e.hints, vec!["grant them with `--allow io`".to_string()]);
    assert_eq!(out.stdout, "");
    let r = main_runner("fn main(args: List[Str]) -> Int !fs !time = 0", "time");
    assert_eq!(r.run_main(&[]).result.unwrap_err().hints, vec!["grant them with `--allow fs:<dir>`".to_string()]);
}

#[test]
fn record_fields_are_evaluated_in_source_order() {
    let src = "type P = { x: Unit, y: Unit }\nfn main(args: List[Str]) -> Int !io = let p = P { y: print(\"y\"), x: print(\"x\") } in 0";
    assert_eq!(main_runner(src, "io").run_main(&[]).stdout, "y\nx\n");
    let r = runner("type Q = { a: Int, b: Int }\nfn mk(n: Int) -> Q = Q { b: 1 / n, a: 9223372036854775807 + 1 }");
    let e = r.call("mk", &[Value::Int(0)]).unwrap_err();
    assert_eq!((e.kind, e.path.as_str()), ("div_by_zero", "body.fields.b"));
    assert_eq!(r.call("mk", &[Value::Int(1)]).unwrap_err().kind, "int_overflow");
}

#[test]
fn referencing_an_effectful_fn_needs_no_permission() {
    let src = "fn pr(x: Int) -> Int !io = { print(\"p\"); x }\nfn main(args: List[Str]) -> Int = let f = pr in 0";
    assert_eq!(main_runner(src, "").run_main(&[]).result, Ok(Value::Int(0)));
    let r = Runner::new(compile("fn pr(x: Int) -> Int !io = { print(\"p\"); x }\ntest t = let f = pr in true\ntest u = 1 == 1", Entry::Tests)).unwrap();
    let results = r.run_tests();
    assert_eq!(results.iter().map(|t| (t.name.as_str(), t.passed, t.error.is_none())).collect::<Vec<_>>(), vec![("t", true, true), ("u", true, true)]);
}

#[test]
fn calling_an_ungranted_effect_reports_permission_denied() {
    // Checked code cannot reach an ungranted host function; bypass the entry check to exercise the stub.
    let mut m = compile("fn main(args: List[Str]) -> Int !io = { print(\"hi\"); 0 }", Entry::Fn("main".into()));
    m.funcs.iter_mut().for_each(|f| f.effects.clear());
    let out = Runner::new(m).unwrap().run_main(&[]);
    let e = out.result.unwrap_err();
    assert_eq!((e.kind, e.code, e.def.as_str()), ("permission_denied", "E0508", "main"));
    assert_eq!(e.message, "`io.print` needs effect io which was not granted");
    assert_eq!(e.hints, vec!["grant it with `--allow io`".to_string()]);
    assert_eq!(out.stdout, "");
}

#[test]
fn pure_programs_import_no_effects() {
    let m = compile("fn main(args: List[Str]) -> Int = len(args)\nfn other() -> Unit !io = print(\"x\")", Entry::Fn("main".into()));
    assert_eq!(m.imports, vec![qal_compiler::host::FAIL]);
    let r = Runner::new(m).unwrap();
    assert_eq!(r.run_main(&["a".into()]).result, Ok(Value::Int(1)));
}

#[test]
fn fs_is_confined_to_granted_directories() {
    let root = std::env::temp_dir().join(format!("qal-run-fs-{}", std::process::id()));
    let data = root.join("data");
    std::fs::create_dir_all(&data).unwrap();
    std::fs::write(root.join("secret.txt"), "top secret").unwrap();
    let src = "fn main(args: List[Str]) -> Int !fs !io = {\n\
                 print(show(write_file(dir(args) ++ \"/out.txt\", \"written\")));\n\
                 print(show_read(read_file(dir(args) ++ \"/out.txt\")));\n\
                 print(show_read(read_file(dir(args) ++ \"/../secret.txt\")));\n\
                 print(show(write_file(dir(args) ++ \"/../evil.txt\", \"x\")));\n\
                 0 }\n\
               fn dir(args: List[Str]) -> Str = match get(args, 0) { Some(d) => d, None => \".\" }\n\
               fn show(r: Result[Unit, Str]) -> Str = match r { Ok(_) => \"ok\", Err(e) => e }\n\
               fn show_read(r: Result[Str, Str]) -> Str = match r { Ok(s) => s, Err(e) => e }";
    let r = main_runner(src, &format!("io,fs:{}", data.display()));
    let out = r.run_main(&[data.to_string_lossy().into_owned()]);
    assert_eq!(out.result, Ok(Value::Int(0)));
    let lines: Vec<&str> = out.stdout.lines().collect();
    assert_eq!(lines[0], "ok");
    assert_eq!(lines[1], "written");
    assert!(lines[2].starts_with("permission denied:"), "{}", lines[2]);
    assert!(lines[3].starts_with("permission denied:"), "{}", lines[3]);
    assert!(!root.join("evil.txt").exists());
}

#[test]
fn time_and_rand() {
    let src = "fn main(args: List[Str]) -> Int !time !rand = if now_ms() > 1700000000000 && in_range(100) then 0 else 1\n\
               fn in_range(n: Int) -> Bool !rand = if n == 0 then true else (let x = rand_int(1, 6) in x >= 1 && x <= 6) && in_range(n - 1)";
    assert_eq!(main_runner(src, "time,rand").run_main(&[]).result, Ok(Value::Int(0)));
    let full = main_runner("fn main(args: List[Str]) -> Int !rand = rand_int(-9223372036854775807 - 1, 9223372036854775807)", "rand");
    assert!(full.run_main(&[]).result.is_ok());
}

#[test]
fn quadratic_push_hits_the_memory_limit_cleanly() {
    // push copies the list, so building n elements this way allocates ~4n² bytes.
    let src = "fn collect(i: Int, n: Int, acc: List[Int]) -> List[Int] = if i == n then acc else collect(i + 1, n, push(acc, i))\n\
               fn build(n: Int) -> Int = len(collect(0, n, []))";
    let r = runner(src).with_memory_limit(1 << 20);
    assert_eq!(r.call("build", &[Value::Int(100)]), Ok(Value::Int(100)));
    let e = r.call("build", &[Value::Int(5_000)]).unwrap_err();
    assert_eq!((e.kind, e.def.as_str()), ("out_of_memory", "build"));
    assert!(e.hints[0].contains("`push`"), "{:?}", e.hints);
}

#[test]
fn huge_input_is_out_of_memory_not_a_crash() {
    let src = "fn main(args: List[Str]) -> Int !io = match read_line() { Some(l) => len(l), None => 0 }";
    let r = main_runner(src, "io").with_memory_limit(1 << 20).with_stdin(vec![b'x'; 2 << 20]);
    let e = r.run_main(&[]).result.unwrap_err();
    assert_eq!((e.kind, e.def.as_str()), ("out_of_memory", "main"));
}

#[test]
fn stdout_survives_runtime_errors() {
    let out = main_runner("fn main(args: List[Str]) -> Int !io = { print(\"before\"); 1 / (len(args) - len(args)) }", "io").run_main(&[]);
    assert_eq!(out.stdout, "before\n");
    assert_eq!(out.result.unwrap_err().kind, "div_by_zero");
}

#[test]
fn deep_values_are_truncated_in_json() {
    let src = "type Chain = End | Link(Int, Chain)\n\
               fn chain(n: Int, acc: Chain) -> Chain = if n == 0 then acc else chain(n - 1, Link(n, acc))\n\
               fn long() -> Chain = chain(100000, End)";
    let v = runner(src).call("long", &[]).unwrap();
    let json = serde_json::to_string(&v).unwrap();
    assert!(json.starts_with(r#"{"Link":[1,{"Link":[2,"#), "{}", &json[..40]);
    assert!(json.contains(r#""...""#));
}

#[test]
fn fs_errors_do_not_reveal_whether_outside_paths_exist() {
    let root = std::env::temp_dir().join(format!("qal-run-oracle-{}", std::process::id()));
    let data = root.join("data");
    std::fs::create_dir_all(&data).unwrap();
    std::fs::write(root.join("exists.txt"), "x").unwrap();
    let src = "fn main(args: List[Str]) -> Int !fs !io = {\n\
                 print(show(read_file(\"/nonexistent-dir-qal/missing\")));\n\
                 print(show(read_file(\"/etc/hosts\")));\n\
                 print(show(read_file(\"/nonexistent-dir-qal/../etc/hosts\")));\n\
                 print(show(read_file(\"/etc/hosts/x\")));\n\
                 0 }\n\
               fn show(r: Result[Str, Str]) -> Str = match r { Ok(_) => \"ok\", Err(e) => e }";
    let out = main_runner(src, &format!("io,fs:{}", data.display())).run_main(&[]);
    let lines: Vec<String> = out.stdout.lines().map(|l| l.split('`').next().unwrap().to_string() + l.rsplit('`').next().unwrap()).collect();
    assert_eq!(lines.len(), 4, "{out:?}");
    assert!(lines[0].starts_with("permission denied:"), "{}", lines[0]);
    assert!(lines.iter().all(|l| *l == lines[0]), "{lines:?}");
}

#[test]
fn stdout_is_capped_at_the_memory_limit() {
    let src = "fn main(args: List[Str]) -> Int !io = emit(0)\n\
               fn emit(i: Int) -> Int !io = { print(\"0123456789012345678901234567890123456789\"); emit(i + 1) }";
    let out = main_runner(src, "io").with_memory_limit(1 << 20).run_main(&[]);
    let e = out.result.unwrap_err();
    assert_eq!((e.kind, e.def.as_str()), ("out_of_memory", "main"));
    assert_eq!(out.stdout.len(), 1 << 20);
}

#[test]
fn shared_values_decode_within_a_node_budget() {
    let src = "type T = Leaf | Node(T, T)\n\
               fn build(n: Int) -> T = if n == 0 then Leaf else (let t = build(n - 1) in Node(t, t))\n\
               fn f(n: Int) -> T = build(n)";
    let r = runner(src);
    let t = std::time::Instant::now();
    let v = r.call("f", &[Value::Int(40)]).unwrap();
    let json = serde_json::to_string(&v).unwrap();
    assert!(t.elapsed() < std::time::Duration::from_secs(5));
    assert!(json.contains(r#""...""#));
    assert!(json.len() < 10_000_000, "{}", json.len());
}

#[test]
fn oversized_file_is_out_of_memory() {
    let root = std::env::temp_dir().join(format!("qal-run-big-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("big.txt"), vec![b'x'; 2 << 20]).unwrap();
    let src = "fn main(args: List[Str]) -> Int !fs = match read_file(match get(args, 0) { Some(p) => p, None => \"\" }) { Ok(s) => len(s), Err(_) => 0 - 1 }";
    let r = main_runner(src, &format!("fs:{}", root.display())).with_memory_limit(1 << 20);
    let e = r.run_main(&[root.join("big.txt").to_string_lossy().into_owned()]).result.unwrap_err();
    assert_eq!((e.kind, e.def.as_str()), ("out_of_memory", "main"));
}

#[test]
fn wall_clock_timeout_interrupts_and_the_runner_stays_usable() {
    let r = runner("fn spin(n: Int) -> Int = spin(n + 1)\nfn one() -> Int = 1").with_fuel(u64::MAX).with_timeout(std::time::Duration::from_millis(200));
    let t = std::time::Instant::now();
    let e = r.call("spin", &[Value::Int(0)]).unwrap_err();
    assert!(t.elapsed() < std::time::Duration::from_secs(5), "{:?}", t.elapsed());
    assert_eq!((e.kind, e.code, e.message.as_str()), ("timeout", "E0507", "wall-clock limit of 200 ms exceeded"));
    // The epoch tick of the first call does not interrupt the next one.
    assert_eq!(r.call("one", &[]), Ok(Value::Int(1)));
    assert_eq!(r.call("one", &[]), Ok(Value::Int(1)));
}

#[test]
fn run_test_reports_one_test() {
    let r = Runner::new(compile("test a = 1 == 1\ntest b = 1 / 0 == 0", Entry::Tests)).unwrap();
    assert_eq!(r.run_test("test:a"), TestResult { name: "a".into(), passed: true, error: None });
    let b = r.run_test("test:b");
    assert_eq!((b.passed, b.error.map(|e| e.kind)), (false, Some("div_by_zero")));
    assert_eq!(r.run_test("test:nope").error.map(|e| e.kind), Some("unknown_function"));
}

/// A main that reads (or, with a second argument, writes) the file named by its first argument.
const FILE_PROGRAM: &str = "fn main(args: List[Str]) -> Int !fs !io = match get(args, 0) {\n\
      None => 2,\n\
      Some(p) => match get(args, 1) {\n\
        None => match read_file(p) { Ok(s) => { print(s); 0 }, Err(e) => { print(e); 1 } },\n\
        Some(c) => match write_file(p, c) { Ok(_) => 0, Err(e) => { print(e); 1 } },\n\
      },\n\
    }";

#[cfg(unix)]
#[test]
fn fifos_in_granted_directories_do_not_block() {
    let root = std::env::temp_dir().join(format!("qal-run-fifo-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let fifo = root.join("pipe");
    let _ = std::fs::remove_file(&fifo);
    assert!(std::process::Command::new("mkfifo").arg(&fifo).status().unwrap().success());
    let r = main_runner(FILE_PROGRAM, &format!("io,fs:{}", root.display()));
    let path = fifo.to_string_lossy().into_owned();
    let t = std::time::Instant::now();
    let out = r.run_main(&[path.clone()]);
    assert_eq!(out.result, Ok(Value::Int(1)));
    assert_eq!(out.stdout, format!("{path}: not a regular file\n"));
    // Writing: no reader, so the non-blocking open fails instead of waiting for one.
    let out = r.run_main(&[path.clone(), "x".into()]);
    assert_eq!(out.result, Ok(Value::Int(1)));
    assert!(out.stdout.starts_with(&format!("{path}: ")), "{}", out.stdout);
    assert!(t.elapsed() < std::time::Duration::from_secs(5));
}

#[test]
fn confined_permissions_resolve_against_the_root_and_hide_dot_entries() {
    let root = std::env::temp_dir().join(format!("qal-run-confined-{}", std::process::id()));
    std::fs::create_dir_all(root.join("data")).unwrap();
    std::fs::create_dir_all(root.join(".git")).unwrap();
    std::fs::write(root.join("data/a.txt"), "A").unwrap();
    std::fs::write(root.join(".git/config"), "secret").unwrap();
    std::fs::write(root.join("data/.env"), "secret").unwrap();
    let grant = |items: &[&str]| Permissions::parse_in(&items.iter().map(|s| s.to_string()).collect::<Vec<_>>(), &root);
    // Directories are relative to the root and must stay inside it; hidden ones cannot be granted.
    assert!(grant(&["fs:../"]).unwrap_err().contains("outside the project root"));
    assert!(grant(&["fs:/"]).unwrap_err().contains("outside the project root"));
    assert!(grant(&["fs:.git"]).unwrap_err().contains("hidden directories"));
    assert!(grant(&["fs:nope"]).unwrap_err().contains("no such directory"));
    assert!(grant(&["fs"]).unwrap_err().contains("needs a directory"));
    assert!(grant(&["fs:"]).unwrap_err().contains("needs a directory"));
    assert!(grant(&["fs: "]).unwrap_err().contains("needs a directory"));
    assert!(grant(&["web"]).unwrap_err().contains("unknown effect"));
    #[cfg(unix)]
    {
        let outside = std::env::temp_dir().join(format!("qal-run-outside-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&outside);
        let symlink_path = root.join("symlink_dir");
        let _ = std::os::unix::fs::symlink(&outside, &symlink_path);
        if symlink_path.is_symlink() {
            assert!(grant(&["fs:symlink_dir"]).is_err(), "symlink directory grant must be refused");
        }
        let sub_target = root.join("data");
        let intermediate_symlink = root.join("link_to_data");
        let _ = std::os::unix::fs::symlink(&sub_target, &intermediate_symlink);
        if intermediate_symlink.is_symlink() {
            assert!(grant(&["fs:link_to_data"]).is_err(), "intermediate symlink grant must be refused");
        }
    }
    let whole = grant(&["io", "fs:."]).unwrap();
    assert!(whole.io && whole.no_hidden && whole.base.is_some());
    let r = main_runner(FILE_PROGRAM, "").with_permissions(whole);
    let run = |args: &[&str]| {
        let out = r.run_main(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        (out.result.unwrap(), out.stdout)
    };
    // Guest paths are relative to the root, not the working directory.
    assert_eq!(run(&["data/a.txt"]), (Value::Int(0), "A\n".into()));
    let denied = |p: &str| (Value::Int(1), format!("permission denied: `{p}` is outside the allowed directories\n"));
    assert_eq!(run(&[".git/config"]), denied(".git/config"));
    assert_eq!(run(&["data/.env"]), denied("data/.env"));
    assert_eq!(run(&["data/../.git/config"]), denied("data/../.git/config"));
    assert_eq!(run(&[".git/hooks-pre-commit", "x"]), denied(".git/hooks-pre-commit"));
    assert_eq!(run(&["../outside.txt", "x"]), denied("../outside.txt"));
    assert!(!root.join("../outside.txt").exists());
    // Absolute paths inside the root are fine.
    let abs = root.join("data/a.txt").to_string_lossy().into_owned();
    assert_eq!(run(&[&abs]), (Value::Int(0), "A\n".into()));
    let narrow = grant(&["io", "fs:data"]).unwrap();
    let r = main_runner(FILE_PROGRAM, "").with_permissions(narrow);
    let out = r.run_main(&["data/b.txt".into(), "B".into()]);
    assert_eq!(out.result, Ok(Value::Int(0)));
    assert_eq!(std::fs::read_to_string(root.join("data/b.txt")).unwrap(), "B");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn output_beyond_the_stdout_limit_is_counted_not_kept() {
    let src = "fn main(args: List[Str]) -> Int !io = loop(0)\nfn loop(i: Int) -> Int !io = if i == 1000 then 7 else { print(\"0123456789\"); loop(i + 1) }";
    let out = main_runner(src, "io").with_stdout_limit(25).run_main(&[]);
    assert_eq!((out.result, out.stdout.as_str(), out.stdout_bytes), (Ok(Value::Int(7)), "0123456789\n0123456789\n012", 11_000));
    // Without a limit the output is kept up to the memory limit, as before.
    let out = main_runner(src, "io").run_main(&[]);
    assert_eq!((out.stdout.len(), out.stdout_bytes), (11_000, 11_000));
}

#[test]
fn large_functions_compile_in_linear_time() {
    // Optimizing compilation took minutes for this (quadratic register allocation). Compares
    // scaling instead of an absolute time, which is flaky under parallel load: 4x the size must
    // cost well under 10x the time (linear is about 4x, quadratic at least 16x).
    let time = |n: usize| {
        let items: Vec<String> = (0..n).map(|i| i.to_string()).collect();
        let src = format!("fn main(args: List[Str]) -> Int = len([{}])", items.join(","));
        let t = std::time::Instant::now();
        let out = main_runner(&src, "").run_main(&[]);
        let elapsed = t.elapsed();
        assert_eq!(out.result, Ok(Value::Int(n as i64)));
        assert!(elapsed < std::time::Duration::from_secs(60), "{elapsed:?}");
        elapsed
    };
    let small = time(5_000).min(time(5_000)).max(std::time::Duration::from_millis(200));
    let large = time(20_000);
    assert!(large < small * 30, "5000 elements: {small:?}, 20000 elements: {large:?}");
}

#[test]
fn confined_fs_path_is_trimmed() {
    let root = std::env::temp_dir().join(format!("qal-run-trim-{}", std::process::id()));
    std::fs::create_dir_all(root.join("data")).unwrap();
    let p = Permissions::parse_in(&["fs: data".to_string()], &root).unwrap();
    assert_eq!(p.fs.iter().map(|r| r.path.clone()).collect::<Vec<_>>(), vec![root.join("data").canonicalize().unwrap()]);
}

#[test]
fn records_and_enums_are_passed_as_arguments() {
    let r = runner(
        "type P = { x: Int, tags: List[Str] }\ntype E = A | B(P, Option[Int])\n\
         fn echo(e: E) -> E = e\nfn sum(ps: List[P]) -> Int = fold(ps, 0, add)\nfn add(acc: Int, p: P) -> Int = acc + p.x\n\
         fn path(rq: HttpRequest) -> Str = rq.method ++ \" \" ++ rq.path ++ \" \" ++ join(map(rq.headers, hname), \",\")\nfn hname(h: HttpHeader) -> Str = h.name\n\
         fn same(a: Result[Str, Int], b: Result[Str, Int]) -> Bool = a == b",
    );
    let p = |x: i64| Value::Record("P".into(), vec![("x".into(), Value::Int(x)), ("tags".into(), Value::List(vec![Value::Str("t".into())]))]);
    let b = Value::Variant("B".into(), vec![p(1), Value::Variant("Some".into(), vec![Value::Int(7)])]);
    assert_eq!(r.call("echo", std::slice::from_ref(&b)), Ok(b));
    assert_eq!(r.call("echo", &[Value::Variant("A".into(), vec![])]), Ok(Value::Variant("A".into(), vec![])));
    assert_eq!(r.call("sum", &[Value::List(vec![p(2), p(3)])]), Ok(Value::Int(5)));
    let h = |n: &str| Value::Record("HttpHeader".into(), vec![("name".into(), Value::Str(n.into())), ("value".into(), Value::Str("v".into()))]);
    let rq = Value::Record("HttpRequest".into(), vec![
        ("method".into(), Value::Str("GET".into())),
        ("path".into(), Value::Str("/todos".into())),
        ("query".into(), Value::Str(String::new())),
        ("headers".into(), Value::List(vec![h("host"), h("accept")])),
        ("body".into(), Value::Str(String::new())),
    ]);
    assert_eq!(r.call("path", &[rq]), Ok(Value::Str("GET /todos host,accept".into())));
    let ok = Value::Variant("Ok".into(), vec![Value::Str("a".into())]);
    assert_eq!(r.call("same", &[ok.clone(), ok]), Ok(Value::Bool(true)));
    let e = r.call("echo", &[Value::Variant("C".into(), vec![])]).unwrap_err();
    assert_eq!((e.kind, e.message.as_str()), ("bad_args", "argument 0 (`e`): E has no variant `C`"));
    let e = r.call("sum", &[Value::List(vec![Value::Record("P".into(), vec![("x".into(), Value::Int(1))])])]).unwrap_err();
    assert_eq!(e.kind, "bad_args");
}

#[test]
fn host_buffers_count_against_the_memory_limit() {
    let root = std::env::temp_dir().join(format!("qal-run-hostmem-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("big.txt"), vec![b'x'; 600 << 10]).unwrap();
    // 600 KiB of kept output plus a 600 KiB file exceed a 1 MiB limit together, not alone.
    let src = "fn main(args: List[Str]) -> Int !fs !io = { emit(15, slice(join(map(range(0, 4096), ten), \"\"), 0, 40959)); match read_file(match get(args, 0) { Some(p) => p, None => \"\" }) { Ok(s) => len(s), Err(_) => 0 - 1 } }\n\
               fn emit(n: Int, line: Str) -> Unit !io = if n == 0 then () else { print(line); emit(n - 1, line) }\n\
               fn ten(i: Int) -> Str = \"0123456789\"";
    let path = root.join("big.txt").to_string_lossy().into_owned();
    let r = main_runner(src, &format!("io,fs:{}", root.display())).with_memory_limit(4 << 20);
    assert_eq!(r.run_main(std::slice::from_ref(&path)).result, Ok(Value::Int(600 << 10)));
    let r = main_runner(src, &format!("io,fs:{}", root.display())).with_memory_limit(1 << 20).with_stdout_limit(usize::MAX);
    let e = r.run_main(&[path]).result.unwrap_err();
    assert_eq!(e.kind, "out_of_memory");
    assert_eq!(serde_json::to_value(&e.values).unwrap()["host"], json!(true), "{e:?}");
    assert!(e.hints.iter().any(|h| h.contains("host's buffers")), "{:?}", e.hints);
}

#[test]
fn fs_writes_have_a_budget() {
    let root = std::env::temp_dir().join(format!("qal-run-wbudget-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let src = "fn main(args: List[Str]) -> Int !fs = go(0, match get(args, 0) { Some(d) => d, None => \".\" })\n\
               fn go(i: Int, d: Str) -> Int !fs = if i == 3 then 0 else match write_file(d ++ \"/f\" ++ from_int(i), \"0123456789\") { Ok(_) => go(i + 1, d), Err(_) => 9 }";
    let dir = root.to_string_lossy().into_owned();
    let run = |limits: Limits| main_runner(src, &format!("fs:{dir}")).with_limits(limits).run_main(std::slice::from_ref(&dir)).result;
    assert_eq!(run(Limits::default()), Ok(Value::Int(0)));
    let e = run(Limits { fs_writes: 2, ..Limits::default() }).unwrap_err();
    assert_eq!((e.kind, e.code), ("budget_exceeded", "E0522"));
    assert_eq!(serde_json::to_value(&e.values).unwrap(), json!({"budget": 2, "used": 3, "kind": "fs_writes"}));
    let e = run(Limits { fs_write_bytes: 25, ..Limits::default() }).unwrap_err();
    assert_eq!(serde_json::to_value(&e.values).unwrap(), json!({"budget": 25, "used": 30, "kind": "fs_bytes"}));
    assert!(e.message.contains("at most 25 bytes"), "{}", e.message);
}

/// A fresh project directory for database tests.
fn db_project(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("qal-run-db-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const TASKS_DB: &str = "fn setup() -> Result[Int, DbError] !db = db_exec(\"CREATE TABLE IF NOT EXISTS todos (id INTEGER PRIMARY KEY, title TEXT NOT NULL UNIQUE, done INTEGER NOT NULL DEFAULT 0)\", [])\n\
    fn add(t: Str) -> Int !db = match db_query(\"INSERT INTO todos (title) VALUES (?) RETURNING id\", [SqlText(t)]) { Ok(r) => match get(r.rows, 0) { Some(row) => match get(row, 0) { Some(SqlInt(id)) => id, _ => 0 - 2 }, None => 0 - 3 }, Err(DbConstraint(_)) => 0 - 1, Err(_) => 0 - 4 }\n\
    fn count() -> Int !db = match db_query(\"SELECT count(*) AS n, ? AS echo FROM todos\", [SqlText(\"x'); DROP TABLE todos; --\")]) { Ok(r) => match get(r.rows, 0) { Some(row) => match get(row, 0) { Some(SqlInt(n)) => n, _ => 0 - 2 }, None => 0 - 3 }, Err(_) => 0 - 4 }\n\
    fn cols() -> List[Str] !db = match db_query(\"SELECT id, title AS name FROM todos\", []) { Ok(r) => r.columns, Err(_) => [] }\n\
    fn main(args: List[Str]) -> Int !db = { let s = setup() in match get(args, 0) { Some(\"add\") => match get(args, 1) { Some(t) => if add(t) > 0 then 0 else 1, None => 2 }, Some(\"fail\") => { let x = add(\"temp\") in 3 }, Some(\"trap\") => { let x = add(\"temp\") in 1 / (x - x) }, _ => count() } }";

#[test]
fn db_runs_parameterized_statements_with_one_transaction_per_execution() {
    let project = db_project("tx");
    let run = |args: &[&str]| {
        let p = Permissions::parse("db:todo").unwrap().with_db_dir(project.join(".qal/db"));
        let out = main_runner(TASKS_DB, "").with_permissions(p).run_main(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        (out.result.map_err(|e| e.kind), out.db)
    };
    assert_eq!(run(&["add", "a"]), (Ok(Value::Int(0)), "commit"));
    assert_eq!(run(&["add", "b"]), (Ok(Value::Int(0)), "commit"));
    assert_eq!(run(&["add", "a"]), (Ok(Value::Int(1)), "rollback"), "unique constraint: Err(DbConstraint), exit 1");
    // Exit code != 0 and traps roll the execution's writes back.
    assert_eq!(run(&["fail"]), (Ok(Value::Int(3)), "rollback"));
    assert_eq!(run(&["trap"]), (Err("div_by_zero"), "rollback"));
    assert_eq!(run(&[]), (Ok(Value::Int(2)), "rollback"), "two rows; exit code 2 also rolls back");
    assert!(project.join(".qal/db/todo.sqlite").is_file());
    // `call` commits unless there is a runtime error; values written as parameters stay data.
    let p = Permissions::parse("db:todo").unwrap().with_db_dir(project.join(".qal/db"));
    let r = Runner::new(compile(TASKS_DB, Entry::All)).unwrap().with_permissions(p);
    assert_eq!(r.call("count", &[]), Ok(Value::Int(2)));
    let out = r.invoke("add", &[Value::Str("c".into())]);
    assert_eq!((out.result, out.db), (Ok(Value::Int(3)), "commit"));
    assert_eq!(r.call("cols", &[]), Ok(Value::List(vec![Value::Str("id".into()), Value::Str("name".into())])));
    let out = r.invoke("main", &[Value::List(vec![])]);
    assert_eq!((out.result, out.db), (Ok(Value::Int(3)), "commit"), "invoke commits on success; only run_entry needs exit code 0");
    // A function without `db` opens no database.
    assert_eq!(main_runner("fn main(args: List[Str]) -> Int = 0", "").with_permissions(Permissions::parse("db").unwrap()).run_main(&[]).db, "none");
}

#[test]
fn db_mem_is_fresh_per_execution_and_db_needs_a_grant() {
    let r = main_runner(TASKS_DB, "db:mem");
    assert_eq!(r.run_main(&["add".into(), "a".into()]).result, Ok(Value::Int(0)));
    assert_eq!(r.run_main(&[]).result, Ok(Value::Int(0)), "a fresh in-memory database: no rows");
    let r = Runner::new(compile(TASKS_DB, Entry::All)).unwrap().with_permissions(Permissions::parse("db:mem").unwrap());
    assert_eq!(r.call("add", &[Value::Str("a".into())]), Ok(Value::Int(-4)), "no table: Err(DbSql(..))");
    let e = main_runner(TASKS_DB, "io").run_main(&[]).result.unwrap_err();
    assert_eq!((e.kind, e.hints.clone()), ("permission_denied", vec!["grant them with `--allow db`".to_string()]));
    assert!(Permissions::parse("db,db:mem").unwrap_err().contains("only one database"));
    assert!(Permissions::parse("db:Bad").unwrap_err().contains("a database name"));
    let root = db_project("confined");
    let p = Permissions::parse_in(&["db".to_string()], &root).unwrap();
    assert_eq!(p.db_dir(), root.canonicalize().unwrap().join(".qal/db"));
}

#[test]
fn db_files_that_are_links_or_not_databases_are_db_open_errors() {
    let project = db_project("open");
    std::fs::create_dir_all(project.join(".qal/db")).unwrap();
    std::fs::write(project.join(".qal/db/main.sqlite"), "not a database file at all, just text").unwrap();
    let p = Permissions::parse("db").unwrap().with_db_dir(project.join(".qal/db"));
    let e = main_runner(TASKS_DB, "").with_permissions(p).run_main(&[]).result.unwrap_err();
    assert_eq!((e.kind, e.code), ("db_open", "E0523"));
}

#[test]
fn db_results_count_against_the_memory_limit_and_rows_are_bounded() {
    let src = "fn big(n: Int) -> Int !db = match db_query(\"WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < ?) SELECT x, printf('%.1000c', 'a') FROM c\", [SqlInt(n)]) { Ok(r) => len(r.rows), Err(DbTooLarge) => 0 - 1, Err(_) => 0 - 2 }";
    let r = Runner::new(compile(src, Entry::All)).unwrap().with_permissions(Permissions::parse("db:mem").unwrap()).with_memory_limit(4 << 20);
    assert_eq!(r.call("big", &[Value::Int(100)]), Ok(Value::Int(100)));
    let e = r.call("big", &[Value::Int(9000)]).unwrap_err();
    assert_eq!(e.kind, "out_of_memory", "9 MB of results exceed the 4 MiB host budget");
    assert_eq!(serde_json::to_value(&e.values).unwrap()["host"], json!(true));
    let r = Runner::new(compile(src, Entry::All)).unwrap().with_permissions(Permissions::parse("db:mem").unwrap()).with_limits(Limits { db_max_rows: 50, ..Limits::default() });
    assert_eq!(r.call("big", &[Value::Int(51)]), Ok(Value::Int(-1)));
}

/// A local HTTP server answering `n` connections: `respond(request text)` gives the raw response.
fn serve_http(n: usize, respond: fn(&str) -> Vec<u8>) -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        use std::io::{Read, Write};
        for s in l.incoming().take(n) {
            let mut s = s.unwrap();
            s.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            // Read the head, then as much body as `content-length` says.
            loop {
                let k = s.read(&mut chunk).unwrap_or(0);
                buf.extend_from_slice(&chunk[..k]);
                let text = String::from_utf8_lossy(&buf).to_string();
                if let Some(end) = text.find("\r\n\r\n") {
                    let len = text.lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0))).unwrap_or(0);
                    if buf.len() >= end + 4 + len || k == 0 {
                        break;
                    }
                }
                if k == 0 {
                    break;
                }
            }
            let _ = s.write_all(&respond(&String::from_utf8_lossy(&buf)));
        }
    });
    port
}

const FETCH: &str = "fn fetch(m: Str, u: Str, b: Str) -> Str !net = match http(HttpCall { method: m, url: u, headers: [HttpHeader { name: \"x-test\", value: \"1\" }], body: b }) {\n\
      Ok(r) => from_int(r.status) ++ \"|\" ++ join(map(r.headers, show), \";\") ++ \"|\" ++ r.body,\n\
      Err(NetDenied(m)) => \"denied: \" ++ m, Err(NetInvalid(m)) => \"invalid: \" ++ m, Err(NetConnect(m)) => \"connect: \" ++ m, Err(NetTimeout) => \"timeout\", Err(NetTooLarge) => \"too large\" }\n\
    fn show(h: HttpHeader) -> Str = h.name ++ \"=\" ++ h.value\n\
    fn twice(u: Str) -> Str !net = fetch(\"GET\", u, \"\") ++ fetch(\"GET\", u, \"\")";

fn net_runner(allow: &str) -> Runner {
    Runner::new(compile(FETCH, Entry::All)).unwrap().with_permissions(Permissions::parse(allow).unwrap())
}

fn fetch(r: &Runner, m: &str, u: &str, b: &str) -> String {
    match r.call("fetch", &[Value::Str(m.into()), Value::Str(u.into()), Value::Str(b.into())]) {
        Ok(Value::Str(s)) => s,
        other => panic!("{other:?}"),
    }
}

#[test]
fn http_calls_reach_granted_hosts_only() {
    let port = serve_http(2, |req| {
        let first = req.lines().next().unwrap_or("").to_string();
        let ua = req.lines().find(|l| l.to_ascii_lowercase().starts_with("user-agent:")).unwrap_or("").to_string();
        let body = format!("{first}\n{}\n{}", ua.to_ascii_lowercase(), req.split("\r\n\r\n").nth(1).unwrap_or(""));
        format!("HTTP/1.1 201 Created\r\nX-Reply: Yes\r\nContent-Length: {}\r\n\r\n{body}", body.len()).into_bytes()
    });
    let r = net_runner(&format!("net:127.0.0.1:{port}"));
    let out = fetch(&r, "POST", &format!("http://127.0.0.1:{port}/todos?x=1"), "{\"a\":1}");
    let (status, rest) = out.split_once('|').unwrap();
    assert_eq!(status, "201");
    assert!(rest.contains("x-reply=Yes"), "header names are lowercase: {rest}");
    assert!(rest.contains("POST /todos?x=1 HTTP/1.1"), "{rest}");
    assert!(rest.contains(&format!("user-agent: qal/{}", env!("CARGO_PKG_VERSION"))), "{rest}");
    assert!(rest.ends_with("{\"a\":1}"), "{rest}");
    // Another port, another host, an IP spelled as a name: not granted, and no connection is made.
    assert!(fetch(&r, "GET", &format!("http://127.0.0.1:{}/", port + 1), "").starts_with(&format!("denied: `127.0.0.1:{}` is not granted", port + 1)));
    assert!(fetch(&r, "GET", "https://example.com/", "").contains("it needs `net:example.com`"));
    assert!(fetch(&r, "GET", &format!("ftp://127.0.0.1:{port}/"), "").starts_with("invalid: unsupported URL scheme"));
    // Without the `net` grant the function cannot run at all.
    let e = net_runner("io").call("fetch", &[Value::Str("GET".into()), Value::Str("http://x/".into()), Value::Str(String::new())]).unwrap_err();
    assert_eq!((e.kind, e.hints.clone()), ("permission_denied", vec!["grant them with `--allow net:<host>`".to_string()]));
    assert!(Permissions::parse("net").unwrap_err().contains("needs a host"));
}

#[test]
fn redirects_are_returned_not_followed() {
    let target = serve_http(1, |_| b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\nSECRET".to_vec());
    let port = serve_http(1, |_| b"HTTP/1.1 302 Found\r\nLocation: http://169.254.169.254/latest/meta-data/\r\nContent-Length: 0\r\n\r\n".to_vec());
    let r = net_runner(&format!("net:127.0.0.1:{port},net:127.0.0.1:{target}"));
    let out = fetch(&r, "GET", &format!("http://127.0.0.1:{port}/"), "");
    assert!(out.starts_with("302|") && out.contains("location=http://169.254.169.254/latest/meta-data/"), "{out}");
    assert!(!out.contains("SECRET"));
}

#[test]
fn responses_must_be_utf8_and_bounded() {
    let big = serve_http(1, |_| {
        let n = 5 << 20;
        let mut v = format!("HTTP/1.1 200 OK\r\nContent-Length: {n}\r\n\r\n").into_bytes();
        v.extend(std::iter::repeat_n(b'x', n));
        v
    });
    assert_eq!(fetch(&net_runner(&format!("net:127.0.0.1:{big}")), "GET", &format!("http://127.0.0.1:{big}/"), ""), "too large");
    let bad = serve_http(1, |_| b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n\xff\xfe".to_vec());
    assert_eq!(fetch(&net_runner(&format!("net:127.0.0.1:{bad}")), "GET", &format!("http://127.0.0.1:{bad}/"), ""), "invalid: response body is not UTF-8");
    let closed = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    assert!(fetch(&net_runner(&format!("net:127.0.0.1:{closed}")), "GET", &format!("http://127.0.0.1:{closed}/"), "").starts_with("connect: "));
    // Response bodies count against the memory limit.
    let mid = serve_http(1, |_| {
        let n = 3 << 20;
        let mut v = format!("HTTP/1.1 200 OK\r\nContent-Length: {n}\r\n\r\n").into_bytes();
        v.extend(std::iter::repeat_n(b'y', n));
        v
    });
    let r = net_runner(&format!("net:127.0.0.1:{mid}")).with_memory_limit(2 << 20);
    let e = r.call("fetch", &[Value::Str("GET".into()), Value::Str(format!("http://127.0.0.1:{mid}/")), Value::Str(String::new())]).unwrap_err();
    assert_eq!((e.kind, serde_json::to_value(&e.values).unwrap()["host"].clone()), ("out_of_memory", json!(true)));
}

#[test]
fn outgoing_calls_have_a_budget() {
    let port = serve_http(1, |_| b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok".to_vec());
    let r = net_runner(&format!("net:127.0.0.1:{port}")).with_limits(Limits { net_calls: 1, ..Limits::default() });
    let e = r.call("twice", &[Value::Str(format!("http://127.0.0.1:{port}/"))]).unwrap_err();
    assert_eq!((e.kind, serde_json::to_value(&e.values).unwrap()), ("budget_exceeded", json!({"budget": 1, "used": 2, "kind": "net_calls"})));
}

/// Runs `f` on a thread with a guest-sized stack (like a server worker).
fn on_guest_thread<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    std::thread::scope(|s| std::thread::Builder::new().stack_size(GUEST_THREAD_STACK).spawn_scoped(s, f).unwrap().join().unwrap())
}

const PREPARED: &str = "fn hog(mib: Int) -> Int = len(grow(\"0123456789abcdef\", mib * 1048576))\n\
    fn grow(s: Str, n: Int) -> Str = if len(s) >= n then s else grow(s ++ s, n)\n\
    fn spin(n: Int) -> Int = spin(n + 1)\n\
    fn quick(n: Int) -> Int = n + 1\n\
    fn text(n: Int) -> Str = slice(grow(\"ab\", n), 0, n)\n\
    fn rows() -> Int !db = match db_query(\"SELECT 1\", []) { Ok(r) => len(r.rows), Err(_) => 0 }";

fn prepared(perms: Permissions, memory: usize) -> Runner {
    Runner::prepared(compile(PREPARED, Entry::All), perms, memory, 4).unwrap()
}

#[test]
fn prepared_runners_instantiate_per_execution_and_free_memory() {
    let r = prepared(Permissions::default(), 24 << 20);
    assert!(r.pooled(), "the pooling allocator is available on the development platforms");
    on_guest_thread(|| {
        for _ in 0..20 {
            // ~16 MiB per execution (the doubling string's copies): only possible 20 times within
            // 24 MiB if every execution starts with fresh memory.
            let out = r.execute("hog", &[Value::Int(8)], ExecIn::default(), &commit_if_ok);
            assert_eq!(out.outcome.result, Ok(Value::Int(8 << 20)));
            assert!(out.mem_bytes >= 16 << 20 && out.fuel > 0, "{} {}", out.mem_bytes, out.fuel);
        }
        let e = r.execute("hog", &[Value::Int(16)], ExecIn::default(), &commit_if_ok).outcome.result.unwrap_err();
        assert_eq!(e.kind, "out_of_memory");
        assert_eq!(r.execute("quick", &[Value::Int(1)], ExecIn::default(), &commit_if_ok).outcome.result, Ok(Value::Int(2)), "usable after a trap");
        r.instantiate_only().unwrap();
    });
}

#[test]
fn concurrent_executions_keep_their_own_deadlines() {
    let r = prepared(Permissions::default(), 16 << 20).with_fuel(u64::MAX / 4).with_timeout(Duration::from_millis(400));
    std::thread::scope(|s| {
        let slow = std::thread::Builder::new().stack_size(GUEST_THREAD_STACK).spawn_scoped(s, || {
            let t = std::time::Instant::now();
            let e = r.execute("spin", &[Value::Int(0)], ExecIn::default(), &commit_if_ok).outcome.result.unwrap_err();
            (e.kind, t.elapsed())
        });
        let fast = std::thread::Builder::new().stack_size(GUEST_THREAD_STACK).spawn_scoped(s, || {
            let t = std::time::Instant::now();
            let mut n = 0;
            while t.elapsed() < Duration::from_millis(900) {
                assert_eq!(r.execute("quick", &[Value::Int(n)], ExecIn::default(), &commit_if_ok).outcome.result, Ok(Value::Int(n + 1)));
                n += 1;
            }
            n
        });
        let (kind, took) = slow.unwrap().join().unwrap();
        assert_eq!(kind, "timeout");
        assert!(took >= Duration::from_millis(400) && took < Duration::from_secs(3), "{took:?}");
        assert!(fast.unwrap().join().unwrap() > 10, "short executions are not interrupted by the long one's deadline");
    });
}

#[test]
fn prepared_executions_use_the_callers_database_and_exact_results() {
    let r = prepared(Permissions::parse("db:mem").unwrap(), 16 << 20).with_exact_results(1000);
    on_guest_thread(|| {
        let db = Db::open(&DbGrant::Memory, std::path::Path::new("."), 16).unwrap();
        let out = r.execute("rows", &[], ExecIn { db: Some(db), host_used: 0 }, &commit_if_ok);
        assert_eq!((out.outcome.result, out.outcome.db, out.db.is_some()), (Ok(Value::Int(1)), "commit", true));
        let out = r.execute("quick", &[Value::Int(1)], ExecIn { db: out.db, host_used: 0 }, &commit_if_ok);
        assert!(out.db.is_some(), "an unused connection is handed back");
        assert_eq!(r.execute("text", &[Value::Int(900)], ExecIn::default(), &commit_if_ok).outcome.result, Ok(Value::Str("ab".repeat(450))));
        let e = r.execute("text", &[Value::Int(5000)], ExecIn::default(), &commit_if_ok).outcome.result.unwrap_err();
        assert_eq!((e.kind, e.code), ("bad_response", "E0521"));
        // Bytes the caller already holds count against the memory limit.
        let e = r.execute("text", &[Value::Int(10)], ExecIn { db: None, host_used: 17 << 20 }, &commit_if_ok);
        assert!(e.outcome.result.is_ok(), "host_used is only checked when the host buffers more");
    });
}

// Findings of the independent review (Phase 3a), pinned.

#[test]
fn sql_statements_stop_at_the_wall_clock_limit() {
    let src = "fn slow(n: Int) -> Int !db = match db_query(\"WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < ?) SELECT count(*) FROM c\", [SqlInt(n)]) { Ok(_) => 1, Err(_) => 2 }";
    let r = Runner::prepared(compile(src, Entry::All), Permissions::parse("db:mem").unwrap(), 16 << 20, 2).unwrap().with_timeout(Duration::from_millis(300));
    let (res, took) = on_guest_thread(|| {
        let t = std::time::Instant::now();
        let out = r.execute("slow", &[Value::Int(30_000_000)], ExecIn::default(), &commit_if_ok);
        (out.outcome.result.map_err(|e| e.kind), t.elapsed())
    });
    assert_eq!(res, Err("timeout"));
    assert!(took < Duration::from_millis(1500), "{took:?}");
    // The same database keeps working afterwards (the host's own COMMIT is never interrupted).
    let out = r.invoke("slow", &[Value::Int(10)]);
    assert_eq!(out.result, Ok(Value::Int(1)));
}

#[test]
fn values_sqlite_builds_are_bounded_by_the_memory_limit() {
    let src = "fn big(n: Int) -> Int !db = match db_query(\"SELECT length(randomblob(?))\", [SqlInt(n)]) { Ok(r) => match get(r.rows, 0) { Some(row) => match get(row, 0) { Some(SqlInt(k)) => k, _ => 0 - 2 }, None => 0 - 3 }, Err(DbSql(_)) => 0 - 1, Err(_) => 0 - 4 }";
    let r = Runner::new(compile(src, Entry::All)).unwrap().with_permissions(Permissions::parse("db:mem").unwrap()).with_memory_limit(16 << 20);
    assert_eq!(r.call("big", &[Value::Int(1000)]), Ok(Value::Int(1000)));
    assert_eq!(r.call("big", &[Value::Int(600_000_000)]), Ok(Value::Int(-1)), "string or blob too big");
}

#[test]
fn temp_objects_are_refused() {
    let src = "fn put() -> Int !db = match db_exec(\"CREATE TEMP TABLE IF NOT EXISTS leak AS SELECT 'secret' AS s\", []) { Ok(_) => 1, Err(DbSql(_)) => 0, Err(_) => 2 }";
    let dir = db_project("temp");
    let perms = Permissions::parse("db:t").unwrap().with_db_dir(dir.join(".qal/db"));
    let r = Runner::prepared(compile(src, Entry::All), perms, 16 << 20, 2).unwrap();
    on_guest_thread(|| {
        let db = Db::open(&DbGrant::Named("t".into()), &dir.join(".qal/db"), 16).unwrap();
        let out = r.execute("put", &[], ExecIn { db: Some(db), host_used: 0 }, &commit_if_ok);
        assert_eq!(out.outcome.result, Ok(Value::Int(0)), "TEMP state would outlive the execution on a worker's connection");
    });
}

#[test]
fn decoded_arguments_count_against_the_host_budget() {
    let src = "fn mk(i: Int) -> SqlValue = SqlNull\n\
               fn p(n: Int) -> Int !db = match db_exec(\"SELECT 1\", map(range(0, n), mk)) { Ok(_) => 0, Err(_) => 1 }";
    let r = Runner::new(compile(src, Entry::All)).unwrap().with_permissions(Permissions::parse("db:mem").unwrap()).with_memory_limit(32 << 20);
    assert_eq!(r.call("p", &[Value::Int(10)]), Ok(Value::Int(1)), "10 parameters for none: DbSql");
    let e = r.call("p", &[Value::Int(1_000_000)]).unwrap_err();
    assert_eq!((e.kind, serde_json::to_value(&e.values).unwrap()["host"].clone()), ("out_of_memory", json!(true)));
}

// ---------------------------------------------------------------------------------------------
// Phase 2a: closures, generics, maps, bytes (end to end)

fn all_pass(src: &str) {
    let r = runner(src);
    let results = r.run_tests();
    assert!(!results.is_empty());
    for t in results {
        assert!(t.passed, "test {} failed: {:?}", t.name, t.error);
    }
}

#[test]
fn closures_capture_values_and_compose() {
    all_pass(
        "fn adder(n: Int) -> fn(Int) -> Int = fn(x) => x + n\n\
         fn compose(f: fn(Int) -> Int, g: fn(Int) -> Int) -> fn(Int) -> Int = fn(x) => g(f(x))\n\
         fn twice(f: fn(Int) -> Int, x: Int) -> Int = f(f(x))\n\
         fn inc(x: Int) -> Int = x + 1\n\
         type Op = { name: Str, run: fn(Float) -> Float }\n\
         fn scale(k: Float) -> Op = Op { name: \"scale\", run: fn(x) => x * k }\n\
         test adds = let a = adder(10) in a(5) == 15 && twice(a, 1) == 21\n\
         test composes = let h = compose(adder(1), fn(x) => x * 3) in h(2) == 9 && twice(inc, 0) == 2\n\
         test nested = let k = 7 in let outer = fn(a: Int) => fn(b: Int) => a + b + k in let f = outer(1) in f(2) == 10\n\
         test in_records = let op = scale(2.5) in let f = op.run in f(2.0) == 5.0 && op.name == \"scale\"\n\
         test with_builtins = let k = 3 in map([1, 2], fn(x) => x * k) == [3, 6] && filter([1, 2, 3, 4], fn(x) => x % 2 == 0) == [2, 4] && fold([\"a\", \"b\"], \"\", fn(acc, s) => acc ++ s ++ from_int(k)) == \"a3b3\"\n\
         test fn_refs_still_work = map([1, 2], inc) == [2, 3]\n\
         test captures_strings_and_lists = let xs = [\"x\", \"y\"] in let s = \"!\" in let f = fn(i: Int) => match get(xs, i) { Some(v) => v ++ s, None => s } in f(1) == \"y!\" && f(5) == \"!\"",
    );
}

#[test]
fn effectful_closures_run_with_their_effects() {
    let r = main_runner(
        "fn logger(prefix: Str) -> fn(Str) -> Unit !io = fn(s) => print(prefix ++ s)\n\
         fn main(args: List[Str]) -> Int !io = { let log = logger(\"> \") in { log(\"a\"); log(\"b\") }; fold(args, (), fn(u, a) => print(a)); 0 }",
        "io",
    );
    let out = r.run_main(&["x".into()]);
    assert_eq!(out.result, Ok(Value::Int(0)));
    assert_eq!(out.stdout, "> a\n> b\nx\n");
}

#[test]
fn generic_functions_and_types_run() {
    all_pass(
        "type Pair[A, B] = { fst: A, snd: B }\n\
         type Tree[T] = Leaf | Node(Tree[T], T, Tree[T])\n\
         fn swap[A, B](p: Pair[A, B]) -> Pair[B, A] = Pair { fst: p.snd, snd: p.fst }\n\
         fn insert[T](t: Tree[T], x: T, less: fn(T, T) -> Bool) -> Tree[T] = match t { Leaf => Node(Leaf, x, Leaf), Node(l, v, r) => if less(x, v) then Node(insert(l, x, less), v, r) else Node(l, v, insert(r, x, less)) }\n\
         fn to_list[T](t: Tree[T]) -> List[T] = match t { Leaf => [], Node(l, v, r) => to_list(l) ++ [v] ++ to_list(r) }\n\
         fn sort[T](xs: List[T], less: fn(T, T) -> Bool) -> List[T] = to_list(fold(xs, Leaf, fn(t, x) => insert(t, x, less)))\n\
         fn my_map[A, B](xs: List[A], f: fn(A) -> B) -> List[B] = fold(xs, [], fn(acc, x) => push(acc, f(x)))\n\
         fn position[T](xs: List[T], x: T) -> Option[Int] = fold(range(0, len(xs)), None, fn(found, i) => match found { Some(_) => found, None => if get(xs, i) == Some(x) then Some(i) else None })\n\
         test swaps = swap(Pair { fst: 1, snd: \"a\" }) == Pair { fst: \"a\", snd: 1 }\n\
         test sorts_ints = sort([3, 1, 2], fn(a, b) => a < b) == [1, 2, 3]\n\
         test sorts_strs = sort([\"b\", \"c\", \"a\"], fn(a, b) => len(a) < len(b) || starts_with(a, \"a\")) == [\"a\", \"b\", \"c\"]\n\
         test maps = my_map([1, 2], fn(x) => from_int(x)) == [\"1\", \"2\"] && my_map([\"a\"], fn(s) => Pair { fst: s, snd: 0.5 }) == [Pair { fst: \"a\", snd: 0.5 }]\n\
         test finds = position([\"x\", \"y\"], \"y\") == Some(1) && position([1.5], 2.5) == None\n\
         fn id[T](x: T) -> T = x\n\
         test generic_values = map([1, 2], id) == [1, 2] && map([\"a\"], id) == [\"a\"] && my_map([Some(1)], id) == [Some(1)]",
    );
}

#[test]
fn maps_are_ordered_and_persistent() {
    all_pass(
        "fn count(words: List[Str]) -> Map[Str, Int] = fold(words, map_new(), fn(m, w) => map_put(m, w, match map_get(m, w) { Some(n) => n + 1, None => 1 }))\n\
         type K = { a: Int, b: Str }\n\
         test counts = let m = count([\"b\", \"a\", \"b\", \"c\"]) in map_keys(m) == [\"a\", \"b\", \"c\"] && map_values(m) == [1, 2, 1] && map_get(m, \"b\") == Some(2) && map_get(m, \"z\") == None\n\
         test persistent = let m1 = map_put(map_new(), 1, \"one\") in let m2 = map_put(m1, 2, \"two\") in map_size(m1) == 1 && map_size(m2) == 2 && map_has(m2, 2) && !map_has(m1, 2)\n\
         test delete = let m = map_put(map_put(map_put(map_new(), 3, 0), 1, 0), 2, 0) in map_keys(map_del(m, 2)) == [1, 3] && map_del(m, 9) == m && map_size(map_del(map_del(map_del(m, 1), 2), 3)) == 0\n\
         test replaces = let m = map_put(map_put(map_new(), \"k\", 1), \"k\", 2) in map_size(m) == 1 && map_get(m, \"k\") == Some(2)\n\
         test equality_is_structural = map_put(map_put(map_new(), 1, true), 2, false) == map_put(map_put(map_new(), 2, false), 1, true)\n\
         test negative_and_large_keys = map_keys(map_put(map_put(map_put(map_new(), 5, 0), -9223372036854775807 - 1, 0), -1, 0)) == [-9223372036854775807 - 1, -1, 5]\n\
         test record_and_list_keys = map_keys(map_put(map_put(map_new(), K { a: 1, b: \"z\" }, 0), K { a: 1, b: \"a\" }, 0)) == [K { a: 1, b: \"a\" }, K { a: 1, b: \"z\" }] && map_keys(map_put(map_put(map_new(), [2], 0), [1, 5], 0)) == [[1, 5], [2]]\n\
         test float_and_option_keys = map_keys(map_put(map_put(map_put(map_new(), 1.5, 0), -0.5, 0), 0.0, 0)) == [-0.5, 0.0, 1.5] && map_keys(map_put(map_put(map_new(), Some(1), 0), None, 0)) == [None, Some(1)]\n\
         test str_order_is_bytewise = map_keys(map_put(map_put(map_put(map_new(), \"ab\", 0), \"a\", 0), \"B\", 0)) == [\"B\", \"a\", \"ab\"]",
    );
    let r = runner("fn big(n: Int) -> Int = map_size(fold(range(0, n), map_new(), fn(m, i) => map_put(m, (i * 7919) % n, i)))");
    assert_eq!(r.call("big", &[Value::Int(2000)]), Ok(Value::Int(2000)));
    let r = runner("fn m() -> Map[Str, List[Int]] = map_put(map_new(), \"k\", [1])");
    assert_eq!(serde_json::to_value(r.call("m", &[]).unwrap()).unwrap(), json!([["k", [1]]]));
}

#[test]
fn bytes_hold_any_bytes() {
    all_pass(
        "test conv = bytes_len(to_bytes(\"hé\")) == 3 && bytes_to_list(to_bytes(\"hé\")) == [104, 195, 169] && bytes_to_str(to_bytes(\"hé\")) == Some(\"hé\")\n\
         test invalid_utf8 = bytes_to_str(bytes_from([255, 0])) == None && bytes_to_str(bytes_from([])) == Some(\"\")\n\
         test get_slice_concat = let b = bytes_from([1, 2, 3]) in bytes_get(b, 0) == Some(1) && bytes_get(b, 3) == None && bytes_get(b, -1) == None && bytes_to_list(bytes_slice(b, 1, 3) ++ b) == [2, 3, 1, 2, 3]\n\
         test equality = bytes_from([104, 105]) == to_bytes(\"hi\") && bytes_from([1]) != bytes_from([1, 0])",
    );
    let r = runner("fn bad(x: Int) -> Bytes = bytes_from([1, x])\nfn cut(n: Int) -> Bytes = bytes_slice(to_bytes(\"ab\"), 0, n)\nfn raw() -> Bytes = bytes_from([0, 255])");
    let e = r.call("bad", &[Value::Int(256)]).unwrap_err();
    assert_eq!((e.kind, e.contract.as_deref()), ("contract_violation", Some("bytes_from: req 0 <= x <= 255")));
    assert_eq!(serde_json::to_value(&e.values).unwrap(), json!({"x": 256}));
    let e = r.call("cut", &[Value::Int(3)]).unwrap_err();
    assert_eq!(e.kind, "index");
    // Review finding M5: the hint talks about bytes, not characters.
    assert_eq!(e.hints, ["valid byte indices satisfy 0 <= start <= end <= bytes_len(b)"]);
    assert_eq!(r.call("raw", &[]), Ok(Value::Bytes(vec![0, 255], 0)));
}

#[test]
fn sequences_discard_values_of_any_type_but_result() {
    // Since the 1e benchmark: `{ a; b; c }` drops the values of a and b (Int, List, Float, closure ...).
    let r = runner("fn f(n: Int) -> Int = { n + 1; [n, n]; 2.5; \"s\"; fn(x) => x + n; Some(n); n * 2 }");
    assert_eq!(r.call("f", &[Value::Int(21)]), Ok(Value::Int(42)));
}

#[test]
fn maps_stay_balanced_under_ordered_inserts_and_deletes() {
    // An unbalanced tree would recurse 50 000 levels deep in `map_put` (stack overflow) and take
    // quadratic time; the AVL tree stays below 24 levels.
    let r = runner(
        "fn up(n: Int) -> Map[Int, Int] = fold(range(0, n), map_new(), fn(m, i) => map_put(m, i, i * 2))\n\
         fn down(m: Map[Int, Int], n: Int) -> Map[Int, Int] = fold(range(0, n), m, fn(acc, i) => map_del(acc, i * 2))\n\
         fn check(n: Int) -> Bool = let m = up(n) in let d = down(m, n / 2) in map_size(m) == n && map_get(m, n - 1) == Some(2 * n - 2) && map_size(d) == n / 2 && map_keys(d) == filter(range(0, n), fn(k) => k % 2 == 1)",
    );
    let t = std::time::Instant::now();
    assert_eq!(r.call("check", &[Value::Int(50_000)]), Ok(Value::Bool(true)));
    assert!(t.elapsed() < std::time::Duration::from_secs(20), "{:?}", t.elapsed());
}

#[test]
fn maps_agree_with_a_reference_map() {
    use std::collections::BTreeMap;
    let r = runner(
        "fn put(m: Map[Int, Str], k: Int, v: Str) -> Map[Int, Str] = map_put(m, k, v)\n\
         fn del(m: Map[Int, Str], k: Int) -> Map[Int, Str] = map_del(m, k)\n\
         fn lookup(m: Map[Int, Str], k: Int) -> Option[Str] = map_get(m, k)\n\
         fn same(a: Map[Int, Str], b: Map[Int, Str]) -> Bool = a == b",
    );
    let to_value = |m: &BTreeMap<i64, String>| Value::Map(m.iter().map(|(k, v)| (Value::Int(*k), Value::Str(v.clone()))).collect());
    let mut reference = BTreeMap::new();
    let mut m = to_value(&reference);
    let mut seed: u64 = 7;
    for step in 0..400 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let k = ((seed >> 33) % 40) as i64 - 20;
        if (seed >> 20) % 3 == 0 {
            reference.remove(&k);
            m = r.call("del", &[m, Value::Int(k)]).unwrap();
        } else {
            reference.insert(k, format!("v{step}"));
            m = r.call("put", &[m, Value::Int(k), Value::Str(format!("v{step}"))]).unwrap();
        }
        assert_eq!(m, to_value(&reference), "after step {step}");
        let want = reference.get(&(k + 1)).map_or(Value::Variant("None".into(), vec![]), |v| Value::Variant("Some".into(), vec![Value::Str(v.clone())]));
        assert_eq!(r.call("lookup", &[m.clone(), Value::Int(k + 1)]), Ok(want));
    }
    // A map built by the host (balanced) equals the same entries built by the guest (other shape),
    // and entries given out of order are sorted.
    let unsorted = Value::Map(vec![(Value::Int(3), Value::Str("c".into())), (Value::Int(1), Value::Str("a".into()))]);
    let built = r.call("put", &[r.call("put", &[Value::Map(vec![]), Value::Int(1), Value::Str("a".into())]).unwrap(), Value::Int(3), Value::Str("c".into())]).unwrap();
    assert_eq!(r.call("same", &[unsorted.clone(), built]), Ok(Value::Bool(true)));
    let dup = Value::Map(vec![(Value::Int(1), Value::Str("a".into())), (Value::Int(1), Value::Str("b".into()))]);
    assert_eq!(r.call("same", &[dup, unsorted]).unwrap_err().kind, "bad_args");
}

#[test]
fn maps_of_maps_and_structured_arguments() {
    let r = runner(
        "type P = { x: Int, name: Str }\ntype Shape = Dot | Box(P, Option[Bytes])\n\
         fn nest() -> Map[Map[Int, Int], Str] = map_put(map_put(map_new(), map_put(map_new(), 2, 0), \"b\"), map_put(map_new(), 1, 9), \"a\")\n\
         fn first(m: Map[Map[Int, Int], Str]) -> List[Str] = map_values(m)\n\
         fn echo(s: Shape, p: P, r: Result[Int, Str]) -> List[Shape] = [s, Box(p, None), match r { Ok(_) => Dot, Err(e) => Box(P { x: 0, name: e }, Some(to_bytes(e))) }]",
    );
    let nested = r.call("nest", &[]).unwrap();
    assert_eq!(serde_json::to_value(&nested).unwrap(), json!([[[[1, 9]], "a"], [[[2, 0]], "b"]]));
    assert_eq!(r.call("first", &[nested]), Ok(Value::List(vec![Value::Str("a".into()), Value::Str("b".into())])));
    let p = Value::Record("P".into(), vec![("name".into(), Value::Str("n".into())), ("x".into(), Value::Int(4))]);
    let s = Value::Variant("Box".into(), vec![p.clone(), Value::Variant("Some".into(), vec![Value::Bytes(vec![1, 2], 0)])]);
    let out = r.call("echo", &[s, p, Value::Variant("Err".into(), vec![Value::Str("e".into())])]).unwrap();
    assert_eq!(
        serde_json::to_value(&out).unwrap(),
        json!([{"Box": [{"x": 4, "name": "n"}, {"Some": [[1, 2]]}]}, {"Box": [{"x": 4, "name": "n"}, "None"]}, {"Box": [{"x": 0, "name": "e"}, {"Some": [[101]]}]}])
    );
    let bad = r.call("echo", &[Value::Variant("Circle".into(), vec![]), Value::Unit, Value::Unit]).unwrap_err();
    assert_eq!((bad.kind, bad.message.as_str()), ("bad_args", "argument 0 (`s`): Shape has no variant `Circle`"));
}

#[test]
fn compare_orders_every_comparable_type() {
    all_pass(
        "type P = { a: Int, b: Str }\n\
         fn insert_sorted[T](xs: List[T], x: T) -> List[T] = concat(push(filter(xs, fn(y) => compare(y, x) <= 0), x), filter(xs, fn(y) => compare(y, x) > 0))\n\
         fn sort[T](xs: List[T]) -> List[T] = fold(xs, [], fn(acc, x) => insert_sorted(acc, x))\n\
         test ints = compare(1, 2) == -1 && compare(2, 2) == 0 && compare(-9223372036854775807 - 1, 0) == -1\n\
         test strs = sort([\"b\", \"ab\", \"a\", \"B\"]) == [\"B\", \"a\", \"ab\", \"b\"]\n\
         test structured = compare(P { a: 1, b: \"z\" }, P { a: 1, b: \"a\" }) == 1 && compare(None, Some(0)) == -1 && compare([1, 2], [1]) == 1 && compare(true, false) == 1\n\
         test floats = compare(-0.0, 0.0) == -1 && compare(1.5, 1.5) == 0\n\
         test maps = compare(map_put(map_new(), 1, \"a\"), map_put(map_new(), 1, \"b\")) == -1",
    );
}

#[test]
fn todo_traps_with_todo_reached() {
    // Phase 4 §11: a stub has any type and traps when it is reached; code around it runs.
    let r = runner("type Order = { id: Int }\nfn make(id: Int) -> Order = todo()\nfn pick(x: Int) -> Int = if x > 0 then x else todo()\ntest t = make(1).id == 1");
    let e = r.call("make", &[Value::Int(1)]).unwrap_err();
    assert_eq!((e.code, e.kind, e.def.as_str(), e.path.as_str()), ("E0540", "todo_reached", "make", "body"));
    assert!(e.hints[0].contains("replace its `todo()`"), "{e:?}");
    assert_eq!(r.call("pick", &[Value::Int(5)]), Ok(Value::Int(5)));
    assert_eq!(r.call("pick", &[Value::Int(0)]).unwrap_err().path, "body.else");
    let t = r.run_test("test:t");
    assert_eq!((t.passed, t.error.map(|e| e.kind)), (false, Some("todo_reached")));
}

#[test]
fn seed_module_checksum_verifies() {
    let m = crate::selfhost::seed_module();
    assert_eq!(qal_core::canon::Hash::of(&m.wasm).to_hex(), crate::selfhost::SEED_SHA256.trim());
    assert!(crate::selfhost::verify_seed(&m.wasm).is_ok());
}

#[test]
fn tampered_seed_module_fails_with_seed_mismatch() {
    let mut wasm = crate::selfhost::seed_module().wasm;
    if let Some(b) = wasm.last_mut() {
        *b = b.wrapping_add(1);
    }
    let err = crate::selfhost::verify_seed(&wasm).unwrap_err();
    assert_eq!((err.code, err.kind), ("E0953", "seed_mismatch"));
}
