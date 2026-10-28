//! Phase 7 tests: The `model` effect with `model_ask`.

use qal_compiler::Entry;
use qal_run::{Permissions, Runner, Value};

fn runner_with_perms(src: &str, perms: Permissions) -> Runner {
    let p = qal_compiler::parse_program(src).unwrap_or_else(|d| panic!("{d:?}"));
    let tp = qal_compiler::check(&p).unwrap_or_else(|d| panic!("{d:#?}"));
    Runner::new(qal_compiler::compile(&tp, &Entry::All).unwrap())
        .unwrap()
        .with_permissions(perms)
}

#[test]
fn model_call_denied_without_permission() {
    let src = r#"
fn ask_something(q: Str) -> Result[ModelResponse, ModelError] !model
  = model_ask(q, "")
"#;
    let r = runner_with_perms(src, Permissions::default());
    let res = r.call("ask_something", &[Value::Str("hello".into())]);
    assert!(res.is_err());
    let err = res.unwrap_err();
    assert_eq!(err.kind, "permission_denied");
}

#[test]
fn model_call_with_permission_and_mock() {
    let tmp = std::env::temp_dir().join(format!("qal-test-model-effect-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let _ = std::fs::create_dir_all(&tmp);
    let db_file = tmp.join("test_store.db");

    std::env::set_var("QAL_MODEL_MOCK", "{\"summary\": \"ok\"}");

    let src = r#"
fn ask_summary(q: Str) -> Result[ModelResponse, ModelError] !model
  = model_ask(q, "{\"type\": \"object\"}")
"#;
    let mut perms = Permissions::default();
    perms.model = true;

    let r = runner_with_perms(src, perms.clone())
        .with_model_db(&db_file);

    let res = r.call("ask_summary", &[Value::Str("evaluate this".into())]).unwrap();
    if let Value::Variant(tag, payload) = &res {
        assert_eq!(tag, "Ok");
        if let Value::Record(rec_name, fields) = &payload[0] {
            assert_eq!(rec_name, "ModelResponse");
            let text_val = fields.iter().find(|(k, _)| k == "text").unwrap().1.clone();
            assert_eq!(text_val, Value::Str("{\"summary\": \"ok\"}".into()));
        } else {
            panic!("expected ModelResponse record, got {:?}", payload[0]);
        }
    } else {
        panic!("expected Ok result, got {:?}", res);
    }

    // Now disable mock mode: call must succeed from SQLite cache!
    std::env::remove_var("QAL_MODEL_MOCK");

    let r2 = runner_with_perms(src, perms.clone())
        .with_model_db(&db_file)
        .with_replay(true);

    let res2 = r2.call("ask_summary", &[Value::Str("evaluate this".into())]).unwrap();
    assert_eq!(res, res2);

    // An unrecorded call in replay mode must return Err(ModelFailed)
    let res3 = r2.call("ask_summary", &[Value::Str("new question".into())]).unwrap();
    if let Value::Variant(tag, payload) = &res3 {
        assert_eq!(tag, "Err");
        if let Value::Variant(err_tag, _) = &payload[0] {
            assert_eq!(err_tag, "ModelFailed");
        } else {
            panic!("expected ModelFailed variant");
        }
    } else {
        panic!("expected Err result, got {:?}", res3);
    }

    let _ = std::fs::remove_dir_all(&tmp);
}
