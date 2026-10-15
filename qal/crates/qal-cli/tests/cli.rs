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
