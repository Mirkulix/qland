use super::*;
use qal_wasm::Entry;
use serde_json::json;

fn compile(src: &str, entry: Entry) -> CompiledModule {
    let p = qal_text::parse_program(src).unwrap();
    let tp = qal_check::check(&p).unwrap_or_else(|d| panic!("{d:#?}"));
    qal_wasm::compile(&tp, &entry).unwrap()
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
    assert_eq!(r.run_main(&["ada".into(), "x".into()]), Outcome { result: Ok(Value::Int(0)), stdout: "hello ada\n2\n".into(), stdout_bytes: 12 });
    assert_eq!(r.run_main(&[]), Outcome { result: Ok(Value::Int(2)), stdout: "usage: hello <name>\n".into(), stdout_bytes: 20 });
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
    assert_eq!(m.imports, vec![qal_wasm::host::FAIL]);
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
    assert!(grant(&["net"]).unwrap_err().contains("unknown effect"));
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
    // Optimizing compilation took minutes for this (quadratic register allocation).
    let items: Vec<String> = (0..30_000).map(|i| i.to_string()).collect();
    let src = format!("fn main(args: List[Str]) -> Int = len([{}])", items.join(","));
    let t = std::time::Instant::now();
    let out = main_runner(&src, "").run_main(&[]);
    assert_eq!(out.result, Ok(Value::Int(30_000)));
    assert!(t.elapsed() < std::time::Duration::from_secs(20), "{:?}", t.elapsed());
}
