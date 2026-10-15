//! Drives the real `qal` binary and checks its JSON output.

use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::Command;

fn example() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/math.qal").to_string_lossy().into_owned()
}

fn temp_file(name: &str, content: &str) -> String {
    let dir = std::env::temp_dir().join(format!("qal-cli-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, content).unwrap();
    path.to_string_lossy().into_owned()
}

fn qal(args: &[&str]) -> (i32, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_qal")).args(args).output().unwrap();
    (out.status.code().unwrap(), String::from_utf8(out.stdout).unwrap())
}

fn qal_json(args: &[&str]) -> (i32, Value) {
    let (code, out) = qal(args);
    (code, serde_json::from_str(&out).unwrap_or_else(|e| panic!("not JSON ({e}): {out}")))
}

#[test]
fn check_ok_and_call() {
    assert_eq!(qal_json(&["check", &example()]), (0, json!({"ok": true})));
    assert_eq!(qal_json(&["call", &example(), "gcd", "48", "18"]), (0, json!({"ok": 6})));
    assert_eq!(qal_json(&["call", &example(), "is_prime", "97"]), (0, json!({"ok": true})));
}

#[test]
fn call_reports_contract_violation_and_bad_args() {
    let (code, v) = qal_json(&["call", &example(), "fact", "-1"]);
    assert_eq!(code, 1);
    assert_eq!(v["error"]["kind"], "contract_violation");
    assert_eq!(v["error"]["contract"], "req[0]: n >= 0");
    let (code, v) = qal_json(&["call", &example(), "gcd", "x", "1"]);
    assert_eq!((code, v["error"]["kind"].as_str()), (1, Some("bad_args")));
    let (code, v) = qal_json(&["call", &example(), "nope"]);
    assert_eq!((code, v["error"]["kind"].as_str()), (1, Some("unknown_function")));
}

#[test]
fn test_command_summarizes() {
    let (code, v) = qal_json(&["test", &example()]);
    assert_eq!(code, 0);
    assert_eq!((v["passed"].as_u64(), v["failed"].as_u64()), (Some(4), Some(0)));
    let failing = temp_file("failing.qal", "test good = true\ntest bad = 1 == 2");
    let (code, v) = qal_json(&["test", &failing]);
    assert_eq!((code, v["failed"].as_u64()), (1, Some(1)));
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
    assert_eq!(qal(&["frobnicate", &example()]).0, 2);
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
