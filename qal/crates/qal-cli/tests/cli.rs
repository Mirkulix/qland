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
