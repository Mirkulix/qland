//! Drives the real `qal` binary and checks its JSON output.

use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn example(name: &str) -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples").join(name).to_string_lossy().into_owned()
}

fn temp_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qal-cli-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn temp_file(name: &str, content: &str) -> String {
    let path = temp_dir().join(name);
    std::fs::write(&path, content).unwrap();
    path.to_string_lossy().into_owned()
}

fn qal(args: &[&str]) -> (i32, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_qal")).args(args).stdin(Stdio::null()).output().unwrap();
    (out.status.code().unwrap(), String::from_utf8(out.stdout).unwrap())
}

fn qal_json(args: &[&str]) -> (i32, Value) {
    let (code, out) = qal(args);
    (code, serde_json::from_str(&out).unwrap_or_else(|e| panic!("not JSON ({e}): {out}")))
}

#[test]
fn check_ok_and_call() {
    let math = example("math.qal");
    assert_eq!(qal_json(&["check", &math]), (0, json!({"ok": true})));
    assert_eq!(qal_json(&["call", &math, "gcd", "48", "18"]), (0, json!({"ok": 6, "stdout": ""})));
    assert_eq!(qal_json(&["call", &math, "is_prime", "97"]), (0, json!({"ok": true, "stdout": ""})));
}

#[test]
fn call_reports_contract_violation_and_bad_args() {
    let math = example("math.qal");
    let (code, v) = qal_json(&["call", &math, "fact", "-1"]);
    assert_eq!(code, 1);
    assert_eq!(v["error"]["kind"], "contract_violation");
    assert_eq!(v["error"]["contract"], "req[0]: n >= 0");
    assert_eq!(v["error"]["values"], json!({"n": -1}));
    let (code, v) = qal_json(&["call", &math, "gcd", "x", "1"]);
    assert_eq!((code, v["error"]["kind"].as_str()), (1, Some("bad_args")));
    let (code, v) = qal_json(&["call", &math, "nope"]);
    assert_eq!((code, v["error"]["kind"].as_str()), (1, Some("unknown_function")));
}

#[test]
fn call_with_strings_and_effects() {
    let src = temp_file("say.qal", "fn say(s: Str) -> Int !io = { print(s ++ \"!\"); len(s) }");
    assert_eq!(qal_json(&["call", &src, "--allow", "io", "say", "héllo"]), (0, json!({"ok": 5, "stdout": "héllo!\n"})));
    let (code, v) = qal_json(&["call", &src, "say", "x"]);
    assert_eq!((code, v["error"]["kind"].as_str()), (1, Some("permission_denied")));
}

#[test]
fn output_keeps_declaration_order() {
    let src = temp_file("order.qal", "type P = { z: Int, a: Str }\nfn mk(s: Str) -> P = P { z: 1, a: s }\nfn chk(z: Int, a: Int) -> Int req z < a = 0");
    assert_eq!(qal(&["call", &src, "mk", "x"]), (0, "{\"ok\":{\"z\":1,\"a\":\"x\"},\"stdout\":\"\"}\n".to_string()));
    let (_, out) = qal(&["call", &src, "chk", "2", "1"]);
    assert!(out.contains(r#""values":{"z":2,"a":1}"#), "{out}");
}

#[test]
fn non_utf8_arguments_do_not_crash() {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let src = temp_file("argc.qal", "fn main(args: List[Str]) -> Int = len(args)");
        let out = Command::new(env!("CARGO_BIN_EXE_qal"))
            .args([std::ffi::OsStr::new("run"), std::ffi::OsStr::new(&src), std::ffi::OsStr::from_bytes(b"\xff")])
            .output()
            .unwrap();
        assert_eq!(String::from_utf8(out.stdout).unwrap(), "{\"exit_code\":1,\"stdout\":\"\"}\n");
    }
}

#[test]
fn test_command_summarizes() {
    let (code, v) = qal_json(&["test", &example("math.qal")]);
    assert_eq!(code, 0);
    assert_eq!((v["passed"].as_u64(), v["failed"].as_u64()), (Some(4), Some(0)));
    let failing = temp_file("failing.qal", "test good = true\ntest bad = 1 == 2");
    let (code, v) = qal_json(&["test", &failing]);
    assert_eq!((code, v["failed"].as_u64()), (1, Some(1)));
}

#[test]
fn run_passes_args_and_returns_exit_code() {
    let age = example("age.qal");
    assert_eq!(qal_json(&["run", &age, "--allow", "io", "--", " 42"]), (0, json!({"exit_code": 0, "stdout": "ok 42\n"})));
    assert_eq!(qal_json(&["run", &age, "--allow", "io", "abc"]), (1, json!({"exit_code": 1, "stdout": "invalid\n"})));
    assert_eq!(qal_json(&["run", &age, "--allow", "io"]), (2, json!({"exit_code": 2, "stdout": "usage: age <n>\n"})));
    let (code, v) = qal_json(&["test", &age]);
    assert_eq!((code, v["passed"].as_u64()), (0, Some(2)));
}

#[test]
fn run_refuses_ungranted_effects() {
    let (code, v) = qal_json(&["run", &example("age.qal"), "42"]);
    assert_eq!(code, 1);
    assert_eq!(v["error"]["kind"], "permission_denied");
    assert_eq!(v["error"]["hints"], json!(["grant them with `--allow io`"]));
    assert_eq!(v["stdout"], "");
}

#[test]
fn run_reads_files_only_in_granted_directories() {
    let dir = temp_dir().join("wc");
    std::fs::create_dir_all(&dir).unwrap();
    let text = dir.join("text.txt");
    std::fs::write(&text, "one two\nthree\n").unwrap();
    let allow = format!("io,fs:{}", dir.display());
    let wc = example("wordcount.qal");
    assert_eq!(qal_json(&["run", &wc, "--allow", &allow, &text.to_string_lossy()]), (0, json!({"exit_code": 0, "stdout": "2 3 14\n"})));
    let (code, v) = qal_json(&["run", &wc, "--allow", &allow, &example("math.qal")]);
    assert_eq!(code, 1);
    assert!(v["stdout"].as_str().unwrap().starts_with("error: permission denied"), "{v}");
}

#[test]
fn exit_codes_outside_0_to_255_exit_1_but_keep_the_value() {
    let src = temp_file("exit.qal", "fn main(args: List[Str]) -> Int = match get(args, 0) { Some(a) => match to_int(a) { Some(n) => n, None => 0 }, None => 0 }");
    assert_eq!(qal_json(&["run", &src, "300"]), (1, json!({"exit_code": 300, "stdout": ""})));
    assert_eq!(qal_json(&["run", &src, "--", "-1"]), (1, json!({"exit_code": -1, "stdout": ""})));
    assert_eq!(qal_json(&["run", &src, "7"]), (7, json!({"exit_code": 7, "stdout": ""})));
}

#[test]
fn run_without_main_is_an_error() {
    let (code, v) = qal_json(&["run", &example("math.qal")]);
    assert_eq!((code, v["error"]["kind"].as_str()), (1, Some("unknown_function")));
}

#[test]
fn diagnostics_are_json() {
    let bad = temp_file("bad.qal", "fn f(x: Int) -> Bool = x + 1");
    let (code, v) = qal_json(&["check", &bad]);
    assert_eq!(code, 1);
    assert_eq!(v["diagnostics"][0]["kind"], "type_mismatch");
    assert_eq!(v["diagnostics"][0]["path"], "body");
    let broken = temp_file("broken.qal", "fn f( -> Int = 1");
    let (code, v) = qal_json(&["call", &broken, "f"]);
    assert_eq!((code, v["diagnostics"][0]["kind"].as_str()), (1, Some("parse")));
    let partial = temp_file("partial.qal", "fn f(x: Option[Int]) -> Int = match x { Some(n) => n }");
    let (_, v) = qal_json(&["check", &partial]);
    assert_eq!(v["diagnostics"][0]["kind"], "non_exhaustive_match");
}

#[test]
fn fmt_prints_canonical_text() {
    let messy = temp_file("messy.qal", "fn   f(a:Int)->Int=a*(2+3)");
    assert_eq!(qal(&["fmt", &messy]), (0, "fn f(a: Int) -> Int\n= a * (2 + 3)\n".to_string()));
}

#[test]
fn usage_and_io_errors_exit_2() {
    assert_eq!(qal(&[]).0, 2);
    assert_eq!(qal(&["check", "/nonexistent/x.qal"]).0, 2);
    assert_eq!(qal(&["frobnicate", &example("math.qal")]).0, 2);
    assert_eq!(qal(&["run", &example("age.qal"), "--allow", "net"]).0, 2);
    assert_eq!(qal(&["call", &example("math.qal")]).0, 2);
}

#[test]
fn check_rejects_deeply_nested_input_with_json_diagnostic() {
    let src = format!("test t = {}1{} == 1\n", "(".repeat(10_000), ")".repeat(10_000));
    let file = temp_file("deep.qal", &src);
    let (code, v) = qal_json(&["check", &file]);
    assert_eq!(code, 1);
    assert_eq!(v["diagnostics"][0]["kind"], "parse");
    assert!(v["diagnostics"][0]["message"].as_str().unwrap().contains("nested too deeply"), "{v}");
}

/// A fresh project directory (no store yet).
fn project(tag: &str) -> PathBuf {
    let dir = temp_dir().join(format!("project-{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn qal_in(dir: &PathBuf, args: &[&str]) -> (i32, Value) {
    let out = Command::new(env!("CARGO_BIN_EXE_qal")).args(args).current_dir(dir).stdin(Stdio::null()).output().unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    (out.status.code().unwrap(), serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("not JSON ({e}): {stdout}")))
}

fn write(dir: &PathBuf, name: &str, content: &str) -> String {
    let path = dir.join(name);
    std::fs::write(&path, content).unwrap();
    path.to_string_lossy().into_owned()
}

#[test]
fn store_workflow_define_view_query_rename_history() {
    let dir = project("workflow");
    let (code, v) = qal_in(&dir, &["init"]);
    assert_eq!((code, v["ok"].as_bool()), (0, Some(true)));
    assert!(dir.join(".qal/store.db").is_file());
    let (code, v) = qal_in(&dir, &["define", &example("math.qal"), "--agent", "agent-a"]);
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["defined"].as_array().unwrap().len(), 8);
    assert_eq!((v["defined"][1]["name"].as_str(), v["defined"][1]["status"].as_str()), (Some("gcd"), Some("added")));
    assert_eq!(v["defined"][1]["hash"].as_str().unwrap().len(), 64);
    let (code, v) = qal_in(&dir, &["view", "is_prime"]);
    assert_eq!(code, 0);
    assert_eq!(v["views"][0]["text"], "fn is_prime(n: Int) -> Bool\n= n >= 2 && no_divisor(n, 2)");
    assert_eq!(v["views"][0]["deps"][0]["signature"], "fn(Int, Int) -> Bool");
    let (_, v) = qal_in(&dir, &["query", "--type", "fn(Int, Int) -> _"]);
    let names: Vec<&str> = v["results"].as_array().unwrap().iter().map(|e| e["name"].as_str().unwrap()).collect();
    assert_eq!(names, vec!["gcd", "no_divisor"]);
    let hash = v["results"][1]["hash"].clone();
    let (code, v) = qal_in(&dir, &["rename", "no_divisor", "has_no_divisor", "--agent", "agent-b"]);
    assert_eq!((code, &v["renamed"]["hash"]), (0, &hash));
    let (_, v) = qal_in(&dir, &["view", "is_prime"]);
    assert_eq!(v["views"][0]["text"], "fn is_prime(n: Int) -> Bool\n= n >= 2 && has_no_divisor(n, 2)");
    let (_, v) = qal_in(&dir, &["history", "has_no_divisor"]);
    assert_eq!(v["history"][0]["agent"], "agent-a");
    assert_eq!((v["history"][1]["action"].as_str(), v["history"][1]["from_name"].as_str()), (Some("renamed"), Some("no_divisor")));
    let (code, v) = qal_in(&dir, &["view", "no_divisor"]);
    assert_eq!((code, v["diagnostics"][0]["kind"].as_str()), (1, Some("unknown_name")));
}

#[test]
fn store_code_can_be_called_tested_and_run() {
    let dir = project("run");
    qal_in(&dir, &["init"]);
    assert_eq!(qal_in(&dir, &["define", &example("age.qal")]).0, 0);
    assert_eq!(qal_in(&dir, &["run", "main", "--allow", "io", "--", " 42"]), (0, json!({"exit_code": 0, "stdout": "ok 42\n"})));
    assert_eq!(qal_in(&dir, &["run", "main", "--allow", "io", "abc"]), (1, json!({"exit_code": 1, "stdout": "invalid\n"})));
    let (code, v) = qal_in(&dir, &["run", "main", "42"]);
    assert_eq!((code, v["error"]["kind"].as_str()), (1, Some("permission_denied")));
    let (code, v) = qal_in(&dir, &["call", "parse_age", " 7 "]);
    assert_eq!((code, &v["ok"]), (0, &json!({"Ok": [7]})));
    let (code, v) = qal_in(&dir, &["test"]);
    assert_eq!((code, v["passed"].as_u64(), v["failed"].as_u64()), (0, Some(2), Some(0)));
    let (code, v) = qal_in(&dir, &["test", "parse_age_ok"]);
    assert_eq!((code, v["passed"].as_u64(), v["tests"].as_array().unwrap().len()), (0, Some(1), 1));
    let (code, v) = qal_in(&dir, &["test", "nope"]);
    assert_eq!((code, v["diagnostics"][0]["kind"].as_str()), (1, Some("unknown_name")));
    // Any `fn(List[Str]) -> Int` can be run; other functions cannot.
    let app = write(&dir, "app.qal", "fn app(args: List[Str]) -> Int = len(args)");
    qal_in(&dir, &["define", &app]);
    assert_eq!(qal_in(&dir, &["run", "app", "a", "b"]), (2, json!({"exit_code": 2, "stdout": ""})));
    let (code, v) = qal_in(&dir, &["run", "parse_age"]);
    assert_eq!((code, v["error"]["kind"].as_str()), (1, Some("bad_args")));
    let (code, v) = qal_in(&dir, &["run", "nope"]);
    assert_eq!((code, v["error"]["kind"].as_str()), (1, Some("unknown_function")));
}

#[test]
fn define_reports_diagnostics_and_signature_changes_as_json() {
    let dir = project("define-errors");
    qal_in(&dir, &["init"]);
    let base = write(&dir, "base.qal", "fn base() -> Int = 1\nfn user() -> Int = base() + 1");
    assert_eq!(qal_in(&dir, &["define", &base]).0, 0);
    let bad = write(&dir, "bad.qal", "fn base(x: Int) -> Int = x");
    let (code, v) = qal_in(&dir, &["define", &bad]);
    assert_eq!(code, 1);
    assert_eq!(v["diagnostics"][0]["kind"], "signature_change");
    assert_eq!(v["diagnostics"][0]["dependents"], json!(["user"]));
    let same = write(&dir, "same.qal", "fn base() -> Int = 2");
    let (_, v) = qal_in(&dir, &["define", &same]);
    let statuses: Vec<(&str, &str)> = v["defined"].as_array().unwrap().iter().map(|d| (d["name"].as_str().unwrap(), d["status"].as_str().unwrap())).collect();
    assert_eq!(statuses, vec![("base", "replaced"), ("user", "relinked")]);
    assert_eq!(qal_in(&dir, &["call", "user"]).1["ok"], 3);
    let broken = write(&dir, "broken.qal", "fn f( -> Int = 1");
    let (code, v) = qal_in(&dir, &["define", &broken]);
    assert_eq!((code, v["diagnostics"][0]["kind"].as_str()), (1, Some("parse")));
}

#[test]
fn missing_or_corrupt_store_is_exit_2_with_json() {
    let dir = project("no-store");
    for args in [&["view", "x"][..], &["query"], &["run", "main"], &["test"], &["history", "x"]] {
        let (code, v) = qal_in(&dir, args);
        assert_eq!((code, v["diagnostics"][0]["kind"].as_str()), (2, Some("store_error")), "{args:?}");
        assert_eq!(v["diagnostics"][0]["hints"][0], "create one with `qal init`");
    }
    std::fs::create_dir_all(dir.join(".qal")).unwrap();
    std::fs::write(dir.join(".qal/store.db"), "garbage that is not a database at all, really not").unwrap();
    let (code, v) = qal_in(&dir, &["query"]);
    assert_eq!((code, v["diagnostics"][0]["kind"].as_str()), (2, Some("store_error")));
    // Usage errors stay exit 2 without touching the store.
    assert_eq!(qal(&["define"]).0, 2);
    assert_eq!(qal(&["rename", "a"]).0, 2);
    assert_eq!(qal(&["query", "--bogus", "1"]).0, 2);
    assert_eq!(qal(&["test", &example("math.qal"), "extra"]).0, 2);
}

#[test]
fn concurrent_define_processes_both_succeed() {
    let dir = project("concurrent");
    qal_in(&dir, &["init"]);
    let files: Vec<String> = (0..4).map(|i| write(&dir, &format!("f{i}.qal"), &format!("fn f{i}() -> Int = {i}\nfn shared() -> Int = {i}"))).collect();
    let children: Vec<_> = files
        .iter()
        .map(|f| Command::new(env!("CARGO_BIN_EXE_qal")).args(["define", f]).current_dir(&dir).stdout(Stdio::piped()).spawn().unwrap())
        .collect();
    for c in children {
        let out = c.wait_with_output().unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stdout));
    }
    let (_, v) = qal_in(&dir, &["query"]);
    assert_eq!(v["results"].as_array().unwrap().len(), 5);
    let (_, v) = qal_in(&dir, &["history", "shared"]);
    assert_eq!(v["history"].as_array().unwrap().len(), 4);
}

#[test]
fn options_may_come_before_the_file_or_name() {
    let age = example("age.qal");
    assert_eq!(qal_json(&["run", "--allow", "io", &age, "--", " 42"]), (0, json!({"exit_code": 0, "stdout": "ok 42\n"})));
    let src = temp_file("say2.qal", "fn say(s: Str) -> Int !io = { print(s); len(s) }");
    assert_eq!(qal_json(&["call", "--allow", "io", &src, "say", "hey"]), (0, json!({"ok": 3, "stdout": "hey\n"})));
    assert_eq!(qal_json(&["test", "--allow", "io", &age]).0, 0);
    // After the file, an unknown `--x` is the first program argument (as before this change).
    let count = temp_file("argc2.qal", "fn main(args: List[Str]) -> Int = len(args)");
    assert_eq!(qal_json(&["run", &count, "--verbose", "x"]), (2, json!({"exit_code": 2, "stdout": ""})));
    let dir = project("options-first");
    qal_in(&dir, &["init"]);
    assert_eq!(qal_in(&dir, &["define", "--agent", "first", &age]).0, 0);
    assert_eq!(qal_in(&dir, &["run", "--allow", "io", "main", "--", " 42"]), (0, json!({"exit_code": 0, "stdout": "ok 42\n"})));
    assert_eq!(qal_in(&dir, &["call", "--allow", "io", "parse_age", "5"]).1["ok"], json!({"Ok": [5]}));
    assert_eq!(qal_in(&dir, &["history", "main"]).1["history"][0]["agent"], "first");
}

#[test]
fn extra_arguments_and_unknown_options_are_usage_errors() {
    let math = example("math.qal");
    assert_eq!(qal(&["check", &math, "extra"]).0, 2);
    assert_eq!(qal(&["fmt", &math, "extra"]).0, 2);
    assert_eq!(qal(&["check", "--allow", "io", &math]).0, 2);
    assert_eq!(qal(&["test", &math, "fact_5"]).0, 2);
    assert_eq!(qal(&["test", &math, "--", "x"]).0, 2);
    assert_eq!(qal(&["run", "--bogus", "x", &math]).0, 2);
    assert_eq!(qal(&["mcp", "extra"]).0, 2);
    assert_eq!(qal(&["build"]).0, 2);
}

#[test]
fn paths_with_a_separator_are_files_and_plain_words_are_names() {
    let dir = project("file-or-name");
    qal_in(&dir, &["init"]);
    // `./age` is a file (missing: exit 2), `age` a stored name (unknown: exit 1).
    let out = Command::new(env!("CARGO_BIN_EXE_qal")).args(["run", "./age"]).current_dir(&dir).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot read ./age"));
    let (code, v) = qal_in(&dir, &["run", "age"]);
    assert_eq!((code, v["error"]["kind"].as_str(), v["error"]["message"].as_str()), (1, Some("unknown_function"), Some("no entry point `age`")));
    // A file without the .qal extension works when it has a path.
    write(&dir, "prog", "fn main(args: List[Str]) -> Int = 3");
    assert_eq!(qal_in(&dir, &["run", "./prog"]), (3, json!({"exit_code": 3, "stdout": ""})));
}

#[test]
fn store_errors_name_tests_functions_and_renamed_names() {
    let dir = project("store-messages");
    qal_in(&dir, &["init"]);
    qal_in(&dir, &["define", &example("math.qal")]);
    // `gcd` calls a function with `req` (itself), so it has a property test; `is_prime` has none.
    let (code, v) = qal_in(&dir, &["test", "is_prime"]);
    assert_eq!((code, v["diagnostics"][0]["message"].as_str()), (1, Some("`is_prime` is a function, not a test")));
    assert!(v["diagnostics"][0]["hints"][0].as_str().unwrap().contains("qal call is_prime"));
    let (code, v) = qal_in(&dir, &["test", "gcd"]);
    assert_eq!((code, v["tests"].as_array().map(Vec::len), v["properties"][0]["def"].as_str(), v["properties"][0]["status"].as_str()), (0, Some(0), Some("gcd"), Some("passed")));
    let (code, v) = qal_in(&dir, &["call", "fact_5"]);
    assert_eq!((code, v["error"]["message"].as_str()), (1, Some("`fact_5` is a test, not a function")));
    qal_in(&dir, &["rename", "gcd", "greatest_common_divisor"]);
    let renamed = json!(["`gcd` was renamed to `greatest_common_divisor`"]);
    assert_eq!(qal_in(&dir, &["view", "gcd"]).1["diagnostics"][0]["hints"], renamed);
    assert_eq!(qal_in(&dir, &["history", "gcd"]).1["diagnostics"][0]["hints"], renamed);
    assert_eq!(qal_in(&dir, &["call", "gcd", "4", "6"]).1["error"]["hints"], renamed);
    assert_eq!(qal_in(&dir, &["call", "greatest_common_divisor", "4", "6"]).1["ok"], 2);
    let (code, v) = qal_in(&dir, &["query", "--type", "fn("]);
    assert_eq!((code, v["diagnostics"][0]["kind"].as_str()), (1, Some("bad_pattern")));
    let as_test = write(&dir, "kind.qal", "test fact = true");
    let (code, v) = qal_in(&dir, &["define", &as_test]);
    assert_eq!((code, v["diagnostics"][0]["kind"].as_str()), (1, Some("kind_change")));
}

#[test]
fn build_writes_a_module_and_its_manifest() {
    let dir = project("build");
    qal_in(&dir, &["init"]);
    qal_in(&dir, &["define", &example("wordcount.qal")]);
    let (code, v) = qal_in(&dir, &["build", "main"]);
    assert_eq!(code, 0, "{v}");
    let b = &v["built"];
    assert_eq!((b["wasm"].as_str(), b["manifest"].as_str()), (Some(".qal/build/main.wasm"), Some(".qal/build/main.manifest.json")));
    assert_eq!(b["effects"], json!(["fs", "io"]));
    let wasm = std::fs::read(dir.join(".qal/build/main.wasm")).unwrap();
    assert_eq!(&wasm[..4], b"\0asm");
    assert_eq!(b["bytes"].as_u64(), Some(wasm.len() as u64));
    let m: Value = serde_json::from_str(&std::fs::read_to_string(dir.join(".qal/build/main.manifest.json")).unwrap()).unwrap();
    assert_eq!((m["format"].as_str(), m["name"].as_str(), m["entry_point"].as_bool()), (Some("qal-build/1"), Some("main"), Some(true)));
    assert_eq!(m["hash"], b["hash"]);
    assert_eq!(m["signature"], "fn(List[Str]) -> Int !fs !io");
    assert_eq!(m["imports"], json!(["fail", "io.print", "fs.read_file"]));
    assert_eq!(m["bytes"].as_u64(), Some(wasm.len() as u64));
    assert_eq!(m["wasm_sha256"].as_str().map(str::len), Some(64));
    let (code, v) = qal_in(&dir, &["build", "nope"]);
    assert_eq!((code, v["error"]["kind"].as_str()), (1, Some("unknown_function")));
}

#[test]
fn a_closed_stdout_is_not_a_panic() {
    let (reader, writer) = std::io::pipe().unwrap();
    drop(reader);
    let status = Command::new(env!("CARGO_BIN_EXE_qal")).args(["check", &example("math.qal")]).stdout(writer).stderr(Stdio::piped()).output().unwrap();
    assert_eq!(status.status.code(), Some(0), "{}", String::from_utf8_lossy(&status.stderr));
    assert!(!String::from_utf8_lossy(&status.stderr).contains("panicked"));
    let (reader, writer) = std::io::pipe().unwrap();
    drop(reader);
    let status = Command::new(env!("CARGO_BIN_EXE_qal")).args(["fmt", &example("math.qal")]).stdout(writer).output().unwrap();
    assert_eq!(status.status.code(), Some(0));
}

#[test]
fn deep_recursion_works_and_overflow_is_reported() {
    let f = temp_file("recursion.qal", "fn deep(n: Int) -> Int = if n == 0 then 0 else 1 + deep(n - 1)\n");
    assert_eq!(qal_json(&["call", &f, "deep", "10000"]), (0, json!({"ok": 10000, "stdout": ""})));
    let (code, v) = qal_json(&["call", &f, "deep", "100000000"]);
    assert_eq!((code, v["error"]["kind"].as_str()), (1, Some("stack_overflow")));
}

#[test]
fn stored_proofs_are_shown_with_current_names() {
    // Review finding I2: results are stored per hash, but carried the names of whichever
    // definition stored them last.
    let dir = project("prove-names");
    qal_in(&dir, &["init"]);
    let a = write(&dir, "a.qal", "fn a(x: Int) -> Int ens r > x = x + 1\nfn pos(x: Int) -> Int req x > 0 = x\nfn user(y: Int) -> Int req y > 1 = pos(y)");
    assert_eq!(qal_in(&dir, &["define", &a]).0, 0);
    let b = write(&dir, "b.qal", "fn b(y: Int) -> Int ens r > y = y + 1");
    let (code, v) = qal_in(&dir, &["define", &b]);
    assert_eq!((code, v["defined"][0]["hash"].as_str()), (0, qal_in(&dir, &["query", "--name", "a"]).1["results"][0]["hash"].as_str()), "{v}");
    let (_, v) = qal_in(&dir, &["prove", "a"]);
    assert_eq!((v["proofs"][0]["def"].as_str(), v["proofs"][0]["contract"].as_str()), (Some("a"), Some("ens r > x")), "{v}");
    assert_eq!(qal_in(&dir, &["rename", "a", "c"]).0, 0);
    assert_eq!(qal_in(&dir, &["rename", "pos", "positive"]).0, 0);
    let (_, v) = qal_in(&dir, &["prove", "c", "user"]);
    let shown: Vec<(&str, &str, &str)> = v["proofs"].as_array().unwrap().iter().map(|r| (r["def"].as_str().unwrap(), r["obligation"].as_str().unwrap(), r["contract"].as_str().unwrap())).collect();
    assert_eq!(shown, [("c", "ens[0]", "ens r > x"), ("user", "req[0] of positive", "req x > 0")], "{v}");
}

#[test]
fn prove_and_property_tests_from_files_and_the_store() {
    let dir = project("prove");
    qal_in(&dir, &["init"]);
    let ok = write(&dir, "ok.qal", "fn abs(x: Int) -> Int ens r >= 0 = if x < 0 then -x else x\nfn pos(x: Int) -> Int req x > 0 = x\ntest t = pos(abs(-3) + 1) == 4");
    let (code, v) = qal_in(&dir, &["define", &ok]);
    assert_eq!((code, v["proofs"].as_array().map(Vec::len)), (0, Some(2)), "{v}");
    let (code, v) = qal_in(&dir, &["prove"]);
    assert_eq!((code, v["proven"].as_u64(), v["refuted"].as_u64(), v["unknown"].as_u64()), (0, Some(2), Some(0), Some(0)));
    let (code, v) = qal_in(&dir, &["prove", "abs"]);
    assert_eq!((code, v["proofs"][0]["contract"].as_str()), (0, Some("ens r >= 0")));
    let bad = write(&dir, "bad.qal", "fn dec(x: Int) -> Int ens r >= 0 = x - 1");
    let (code, v) = qal_in(&dir, &["define", &bad]);
    assert_eq!((code, v["diagnostics"][0]["kind"].as_str()), (1, Some("contract_refuted")));
    let (code, v) = qal_in(&dir, &["prove", &bad]);
    assert_eq!((code, v["refuted"].as_u64(), v["proofs"][0]["status"].as_str()), (1, Some(1), Some("refuted")));
    let props = write(&dir, "props.qal", "fn head(xs: List[Int]) -> Int ens r >= 0 = match get(sort_int(xs), 0) { Some(x) => x, None => 0 }\ntest t = head([1]) == 1");
    let (code, v) = qal_in(&dir, &["test", &props]);
    assert_eq!((code, v["passed"].as_u64(), v["failed_properties"].as_u64(), &v["properties"][0]["counterexample"]), (1, Some(1), Some(1), &json!({"xs": [-1]})));
    let (_, v) = qal_in(&dir, &["test", &props, "--seed", "3"]);
    assert_eq!(v["properties"][0]["seed"].as_u64(), Some(3));
    assert_eq!(qal_in(&dir, &["prove", "nope"]).1["diagnostics"][0]["kind"], "unknown_name");
}

#[test]
fn databases_live_in_the_project_of_the_store() {
    let project = temp_dir().join("dbproject");
    let _ = std::fs::remove_dir_all(&project);
    std::fs::create_dir_all(project.join("sub")).unwrap();
    let file = project.join("count.qal");
    std::fs::write(&file, "fn main(args: List[Str]) -> Int !db !io = match db_exec(\"CREATE TABLE IF NOT EXISTS hits (n INTEGER)\", []) { Err(_) => 9, Ok(_) => match db_exec(\"INSERT INTO hits VALUES (1)\", []) { Err(_) => 8, Ok(_) => match db_query(\"SELECT count(*) FROM hits\", []) { Ok(r) => { print(join(map(r.columns, same), \",\")); match get(r.rows, 0) { Some(row) => match get(row, 0) { Some(SqlInt(n)) => if n == 2 then 3 else 0, _ => 7 }, None => 6 } }, Err(_) => 5 } } }\nfn same(s: Str) -> Str = s").unwrap();
    let run = |dir: &std::path::Path| {
        let out = Command::new(env!("CARGO_BIN_EXE_qal")).current_dir(dir).args(["run", file.to_str().unwrap(), "--allow", "db,io"]).stdin(Stdio::null()).output().unwrap();
        (out.status.code().unwrap(), serde_json::from_slice::<Value>(&out.stdout).unwrap())
    };
    assert_eq!(qal_json_in(&project, &["init"]).0, 0);
    assert_eq!(run(&project.join("sub")), (0, json!({"exit_code": 0, "stdout": "count(*)\n", "db": "commit"})));
    assert!(project.join(".qal/db/main.sqlite").is_file(), "next to the store, found from a subdirectory");
    // The second run sees 2 rows and exits 3: its insert is rolled back.
    assert_eq!(run(&project), (3, json!({"exit_code": 3, "stdout": "count(*)\n", "db": "rollback"})));
    assert_eq!(run(&project).0, 3, "still 2 rows: the rolled back insert is gone");
}

fn qal_json_in(dir: &std::path::Path, args: &[&str]) -> (i32, Value) {
    let out = Command::new(env!("CARGO_BIN_EXE_qal")).current_dir(dir).args(args).stdin(Stdio::null()).output().unwrap();
    (out.status.code().unwrap(), serde_json::from_slice(&out.stdout).unwrap())
}

/// A project with the fixture service of qal-serve stored in it.
fn service_project(name: &str) -> PathBuf {
    let project = temp_dir().join(name);
    let _ = std::fs::remove_dir_all(&project);
    std::fs::create_dir_all(&project).unwrap();
    let app = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../qal-serve/tests/fixtures/app.qal");
    assert_eq!(qal_json_in(&project, &["init"]).0, 0);
    let (code, v) = qal_json_in(&project, &["define", app.to_str().unwrap()]);
    assert_eq!(code, 0, "{v}");
    project
}

#[test]
fn qal_request_runs_a_json_file_of_requests() {
    let project = service_project("request");
    std::fs::write(project.join("r.json"), r#"{"requests": [{"method": "POST", "path": "/todos", "body": "a"}, {"path": "/todos"}]}"#).unwrap();
    let (code, v) = qal_json_in(&project, &["request", "--allow", "db,io", "--init", "setup", "r.json"]);
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["responses"][1]["json"], json!([{"id": 1, "title": "a"}]));
    assert!(project.join(".qal/db/main.sqlite").is_file());
    let (code, v) = qal_json_in(&project, &["request", "handle", "--allow", "db,io", "r.json"]);
    assert_eq!((code, v["responses"][1]["json"].as_array().map(|a| a.len())), (0, Some(2)), "the project's database persists");
    std::fs::write(project.join("bad.json"), r#"[{"path": 5}]"#).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_qal")).current_dir(&project).args(["request", "--allow", "db", "bad.json"]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("request 0: request field `path` must be a string"));
}

#[test]
fn qal_serve_answers_http_and_stops_on_sigterm() {
    use std::io::{BufRead, BufReader, Read, Write};
    let project = service_project("serve");
    let mut child = Command::new(env!("CARGO_BIN_EXE_qal"))
        .current_dir(&project)
        .args(["serve", "--listen", "127.0.0.1:0", "--allow", "db,io", "--init", "setup", "--workers", "2", "--request-timeout", "800"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut log = BufReader::new(child.stderr.take().unwrap());
    let mut line = String::new();
    log.read_line(&mut line).unwrap();
    let listening: Value = serde_json::from_str(&line).unwrap_or_else(|e| panic!("{e}: {line}"));
    assert_eq!((listening["event"].as_str(), listening["entry"].as_str(), listening["workers"].as_u64()), (Some("listening"), Some("handle"), Some(2)));
    assert_eq!(listening["hash"].as_str().map(str::len), Some(64));
    let addr: std::net::SocketAddr = listening["addr"].as_str().unwrap().parse().unwrap();
    let send = |raw: &str| {
        let mut s = std::net::TcpStream::connect(addr).unwrap();
        s.write_all(raw.as_bytes()).unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        out
    };
    let created = send("POST /todos HTTP/1.1\r\nHost: x\r\nContent-Length: 3\r\nConnection: close\r\n\r\negg");
    assert!(created.starts_with("HTTP/1.1 201 Created\r\n") && created.ends_with(r#"{"id":1,"title":"egg"}"#), "{created}");
    let spin = send("GET /spin HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
    assert!(spin.starts_with("HTTP/1.1 503 "), "{spin}");
    let mut request = String::new();
    log.read_line(&mut request).unwrap();
    let request: Value = serde_json::from_str(&request).unwrap();
    assert_eq!((request["event"].as_str(), request["status"].as_u64(), request["db"].as_str()), (Some("request"), Some(201), Some("commit")));
    assert!(Command::new("kill").args(["-TERM", &child.id().to_string()]).status().unwrap().success());
    let status = child.wait().unwrap();
    assert_eq!(status.code(), Some(0));
    let rest: Vec<Value> = log.lines().map(|l| serde_json::from_str(&l.unwrap()).unwrap()).collect();
    assert_eq!(rest.last().unwrap(), &json!({"event": "stopped", "served": 2}));
    // Start failures: a bad handler is exit 1 with the error as JSON; `db:mem` is a usage error.
    let out = Command::new(env!("CARGO_BIN_EXE_qal")).current_dir(&project).args(["serve", "text", "--listen", "127.0.0.1:0", "--allow", "db,io"]).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    let failed: Value = serde_json::from_str(String::from_utf8_lossy(&out.stderr).lines().next().unwrap()).unwrap();
    assert_eq!((failed["event"].as_str(), failed["error"]["code"].as_str()), (Some("start_failed"), Some("E0520")));
    let out = Command::new(env!("CARGO_BIN_EXE_qal")).current_dir(&project).args(["serve", "--allow", "db:mem,io"]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    let out = Command::new(env!("CARGO_BIN_EXE_qal")).current_dir(&project).args(["serve", "--workers", "0", "--allow", "db,io"]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("`--workers` must be a number in 1..=64"));
}

#[test]
fn packages_round_trip_between_projects() {
    let a = temp_dir().join("pkg-a");
    let b = temp_dir().join("pkg-b");
    for d in [&a, &b] {
        let _ = std::fs::remove_dir_all(d);
        std::fs::create_dir_all(d).unwrap();
        assert_eq!(qal_json_in(d, &["init"]).0, 0);
    }
    assert_eq!(qal_json_in(&a, &["define", &example("math.qal")]).0, 0);
    let (code, v) = qal_json_in(&a, &["export", "", "--out", "out/math.qalpkg"]);
    // The empty prefix does not export the standard library imported into the project.
    assert_eq!((code, v["exported"]["package"].as_str(), v["exported"]["defs"].as_u64()), (0, Some("math"), Some(8)));
    let file = a.join("out/math.qalpkg");
    let (code, v) = qal_json_in(&b, &["import", file.to_str().unwrap()]);
    assert_eq!((code, v["imported"]["defined"].as_array().map(|d| d.len())), (0, Some(8)), "{v}");
    let (_, qa) = qal_json_in(&a, &["query"]);
    let (_, qb) = qal_json_in(&b, &["query"]);
    assert_eq!(qa, qb, "identical names and hashes");
    std::fs::write(&file, std::fs::read_to_string(&file).unwrap().replace("gcd", "gcf")).unwrap();
    let (code, v) = qal_json_in(&b, &["import", file.to_str().unwrap()]);
    assert_eq!((code, v["diagnostics"][0]["code"].as_str()), (1, Some("E0243")));
    let out = Command::new(env!("CARGO_BIN_EXE_qal")).current_dir(&a).args(["export", "x"]).output().unwrap();
    assert_eq!(out.status.code(), Some(2), "--out is required");
}

// ---------------------------------------------------------------------------------------------
// Phase 3b: the standard library namespace and package.

fn std_hash() -> String {
    qal_store::package::parse(qal_cli::STD_PACKAGE).unwrap().hash
}

fn meta(dir: &std::path::Path, key: &str) -> Option<String> {
    qal_store::Store::open(&dir.join(".qal/store.db")).unwrap().meta(key).unwrap()
}

#[test]
fn the_checked_in_std_package_has_agent_provenance() {
    // Spec Phase 3 §4.4 step 5: every std definition was written by an agent through MCP, never by
    // the CLI, and the package says who.
    let on_disk = std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../std/std.qalpkg")).unwrap();
    assert_eq!(on_disk, qal_cli::STD_PACKAGE, "the binary embeds the checked-in file");
    let pkg = qal_store::package::parse(qal_cli::STD_PACKAGE).unwrap();
    assert_eq!(pkg.name, "std");
    for d in &pkg.defs {
        assert!(!d.agent.trim().is_empty() && d.agent != "cli" && d.agent != "unknown", "`{}` has agent `{}`", d.name, d.agent);
        assert!(qal_text::lexer::is_std_name(&d.name), "`{}` is not a std. name", d.name);
    }
    // It imports into a fresh project with exactly the recorded hashes.
    let dir = project("std-fresh");
    let mut s = qal_store::Store::init(&dir).unwrap();
    let i = s.import(qal_cli::STD_PACKAGE).unwrap();
    assert_eq!(i.defined.len(), pkg.defs.len());
    let stored: std::collections::BTreeMap<String, String> = s.query(&qal_store::Query::default()).unwrap().into_iter().map(|e| (e.name, e.hash)).collect();
    for d in &pkg.defs {
        assert_eq!(stored.get(&d.name), Some(&d.hash), "{}", d.name);
    }
}

#[test]
fn init_and_every_write_import_the_std_package_but_reads_do_not() {
    let dir = project("std-auto");
    let (code, v) = qal_in(&dir, &["init"]);
    assert_eq!((code, v["std_imported"].as_str()), (0, Some(std_hash().as_str())), "{v}");
    assert_eq!(meta(&dir, "std_package"), Some(std_hash()));
    let (_, v) = qal_in(&dir, &["init"]);
    assert!(v.get("std_imported").is_none(), "already there: {v}");
    // A store made without `qal init` (e.g. by an older qal): reads leave it alone, writes import.
    for (tag, op) in [("define", &["define", "f.qal"][..]), ("rename", &["rename", "f", "g"][..]), ("import", &["import", "p.qalpkg"][..])] {
        let dir = project(&format!("std-auto-{tag}"));
        let mut s = qal_store::Store::init(&dir).unwrap();
        s.define("fn f() -> Int = 1", "test").unwrap();
        std::fs::write(dir.join("p.qalpkg"), s.export("f", "p").unwrap().file).unwrap();
        drop(s);
        write(&dir, "f.qal", "fn f() -> Int = 2");
        for read in [&["view", "f"][..], &["query"][..], &["history", "f"][..], &["test"][..]] {
            assert_eq!(qal_in(&dir, read).0, 0, "{read:?}");
        }
        assert_eq!(meta(&dir, "std_package"), None, "reads never write");
        let (code, v) = qal_in(&dir, op);
        assert_eq!((code, v["std_imported"].as_str()), (0, Some(std_hash().as_str())), "{tag}: {v}");
        assert_eq!(meta(&dir, "std_package"), Some(std_hash()), "{tag}");
    }
}

#[test]
fn std_names_are_read_only_in_projects() {
    let dir = project("std-readonly");
    qal_in(&dir, &["init"]);
    let f = write(&dir, "s.qal", "fn std.json.parse(s: Str) -> Int = 1");
    let (code, v) = qal_in(&dir, &["define", &f]);
    assert_eq!((code, v["diagnostics"][0]["code"].as_str(), v["diagnostics"][0]["kind"].as_str()), (1, Some("E0240"), Some("reserved_name")), "{v}");
    let f = write(&dir, "p.qal", "type HttpRequest = { a: Int }");
    assert_eq!(qal_in(&dir, &["define", &f]).1["diagnostics"][0]["code"], "E0240");
    let f = write(&dir, "g.qal", "fn g() -> Int = 1");
    assert_eq!(qal_in(&dir, &["define", &f]).0, 0);
    let (code, v) = qal_in(&dir, &["rename", "g", "std.json.g"]);
    assert_eq!((code, v["diagnostics"][0]["code"].as_str()), (1, Some("E0240")), "{v}");
}

#[test]
fn the_std_project_store_defines_only_std_names_and_exports_the_package() {
    let dir = project("std-project");
    // A new std store is filled from the embedded package (with its recorded agents).
    let (code, v) = qal_in(&dir, &["init", "--std"]);
    assert_eq!((code, v["std_imported"].as_str()), (0, Some(std_hash().as_str())), "{v}");
    assert_eq!(meta(&dir, "role").as_deref(), Some("std"));
    let (_, v) = qal_in(&dir, &["init", "--std"]);
    assert!(v.get("std_imported").is_none(), "only a new std store is filled: {v}");
    let f = write(&dir, "a.qal", "type std.a.T = std.a.One | std.a.Two(Int)\nfn std.a.f(t: std.a.T) -> Int = match t { std.a.One => 1, std.a.Two(n) => n }\ntest std.a.t = std.a.f(std.a.Two(2)) == 2\nfn std.a.inc(x: Int) -> Int = x + 1");
    let (code, v) = qal_in(&dir, &["define", &f, "--agent", "std-a-1"]);
    assert_eq!(code, 0, "{v}");
    assert!(v.get("std_imported").is_none(), "no automatic import in the std project: {v}");
    let f = write(&dir, "b.qal", "fn helper() -> Int = 1");
    let (code, v) = qal_in(&dir, &["define", &f]);
    assert_eq!((code, v["diagnostics"][0]["code"].as_str()), (1, Some("E0240")), "{v}");
    // Qualified names are store names, not files, on the command line.
    let (code, v) = qal_in(&dir, &["view", "std.a.f"]);
    assert_eq!((code, v["views"][0]["name"].as_str()), (0, Some("std.a.f")), "{v}");
    let (code, v) = qal_in(&dir, &["test", "std.a.t"]);
    assert_eq!((code, v["passed"].as_u64()), (0, Some(1)), "{v}");
    let (code, v) = qal_in(&dir, &["call", "std.a.inc", "2"]);
    assert_eq!((code, &v["ok"]), (0, &json!(3)), "{v}");
    let (code, v) = qal_in(&dir, &["export", "std.", "--out", "std.qalpkg"]);
    assert_eq!((code, v["exported"]["package"].as_str(), v["exported"]["defs"].as_u64()), (0, Some("std"), Some(4)), "{v}");
    let file = std::fs::read_to_string(dir.join("std.qalpkg")).unwrap();
    assert!(file.lines().filter(|l| l.starts_with("# def ")).all(|l| l.ends_with(" std-a-1")), "{file}");
    // A project importing that package gets the same hashes and cannot change them.
    let user = project("std-user");
    qal_in(&user, &["init"]);
    let (code, v) = qal_in(&user, &["import", dir.join("std.qalpkg").to_str().unwrap()]);
    assert_eq!(code, 0, "{v}");
    assert_eq!(qal_in(&user, &["view", "std.a.f"]).1["views"][0]["hash"], qal_in(&dir, &["view", "std.a.f"]).1["views"][0]["hash"]);
    // `--std` on an ordinary project with code is refused.
    let f = write(&user, "u.qal", "fn u() -> Int = std.a.f(std.a.One)");
    assert_eq!(qal_in(&user, &["define", &f]).0, 0);
    let (code, v) = qal_in(&user, &["init", "--std"]);
    assert_eq!((code, v["diagnostics"][0]["kind"].as_str()), (2, Some("store_error")), "{v}");
}

#[test]
fn the_std_launcher_and_registration_exist() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let script = std::fs::read_to_string(repo.join("qal/mcp.sh")).unwrap();
    assert!(script.contains("std)") && script.contains("init --std") && script.contains("$here/std"), "{script}");
    let mcp: Value = serde_json::from_str(&std::fs::read_to_string(repo.join(".mcp.json")).unwrap()).unwrap();
    assert_eq!(mcp["mcpServers"]["qal-std"]["args"], json!(["qal/mcp.sh", "std"]));
    assert_eq!(std::fs::read_to_string(repo.join("qal/std/.gitignore")).unwrap().lines().last(), Some(".qal/"));
    let r = qal_cli::mcp::REFERENCE;
    for needle in ["`std.<module>.<name>` is one name", "std.json.Str(\"a\")", "read-only in a project", "E0240", "imported automatically"] {
        assert!(r.contains(needle), "{needle}");
    }
}

#[test]
fn delete_removes_unused_names() {
    let dir = project("delete");
    qal_in(&dir, &["init"]);
    let f = write(&dir, "d.qal", "fn base() -> Int = 1\nfn top() -> Int = base() + 1\nfn lone() -> Int = 3");
    assert_eq!(qal_in(&dir, &["define", &f]).0, 0);
    let (code, v) = qal_in(&dir, &["delete", "base"]);
    assert_eq!((code, v["diagnostics"][0]["code"].as_str(), &v["diagnostics"][0]["dependents"]), (1, Some("E0406"), &json!(["top"])), "{v}");
    let (code, v) = qal_in(&dir, &["delete", "lone", "--agent", "cleaner"]);
    assert_eq!((code, v["deleted"][0]["name"].as_str(), v["deleted"][0]["kind"].as_str()), (0, Some("lone"), Some("fn")), "{v}");
    assert_eq!(qal_in(&dir, &["view", "lone"]).1["diagnostics"][0]["kind"], "unknown_name");
    let (_, v) = qal_in(&dir, &["history", "lone"]);
    assert_eq!((v["history"][1]["action"].as_str(), v["history"][1]["agent"].as_str()), (Some("deleted"), Some("cleaner")));
    let (code, v) = qal_in(&dir, &["delete", "top", "base"]);
    assert_eq!((code, v["deleted"].as_array().map(Vec::len)), (0, Some(2)), "{v}");
    assert_eq!(qal_in(&dir, &["query"]).1["results"], json!([]));
    let out = Command::new(env!("CARGO_BIN_EXE_qal")).current_dir(&dir).arg("delete").output().unwrap();
    assert_eq!(out.status.code(), Some(2), "a name is required");
}

#[test]
fn project_tests_skip_std_tests_unless_named() {
    // A std with a test and a property-tested function, imported into a project.
    let std_dir = project("std-tests-lib");
    let mut lib = qal_store::Store::init_std(&std_dir).unwrap();
    lib.define("fn std.a.abs(x: Int) -> Int ens r >= 0 = if x < 0 then 0 - x else x\ntest std.a.t = std.a.abs(-2) == 2", "std-a-1").unwrap();
    let pkg = lib.export("std.", "std").unwrap();
    drop(lib);
    let dir = project("std-tests-user");
    let mut s = qal_store::Store::init(&dir).unwrap();
    s.ensure_std(&pkg.file, None).unwrap();
    s.define("fn own(x: Int) -> Int ens r >= x = x + 0\ntest mine = own(1) == 1", "test").unwrap();
    drop(s);
    let names = |v: &Value, key: &str, field: &str| -> Vec<String> { v[key].as_array().map(|xs| xs.iter().map(|x| x[field].as_str().unwrap_or("").to_string()).collect()).unwrap_or_default() };
    let (code, v) = qal_in(&dir, &["test"]);
    assert_eq!(code, 0, "{v}");
    assert_eq!(names(&v, "tests", "name"), ["mine"], "{v}");
    assert_eq!(names(&v, "properties", "def"), ["own"], "{v}");
    let (_, v) = qal_in(&dir, &["test", "std.a.t", "std.a.abs"]);
    assert_eq!((names(&v, "tests", "name"), names(&v, "properties", "def")), (vec!["std.a.t".to_string()], vec!["std.a.abs".to_string()]), "{v}");
    // In the std store, all means all.
    let (_, v) = qal_in(&std_dir, &["test"]);
    assert_eq!((names(&v, "tests", "name"), names(&v, "properties", "def")), (vec!["std.a.t".to_string()], vec!["std.a.abs".to_string()]), "{v}");
}
