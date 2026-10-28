//! Lower a JSON expression AST to QAL source (agent-native surface).

use serde_json::{Map, Value};

fn lit(v: &Value) -> Result<String, String> {
    match v {
        Value::Bool(b) => Ok(if *b { "true".into() } else { "false".into() }),
        Value::Number(n) => Ok(n.to_string()),
        Value::String(s) => serde_json::to_string(s).map_err(|e| e.to_string()),
        _ => Err(format!("bad literal: {v}")),
    }
}

/// Lower one AST expression.
pub fn lower(e: &Value) -> Result<String, String> {
    if !e.is_object() {
        return lit(e);
    }
    let o: &Map<String, Value> = e.as_object().unwrap();
    if let Some(v) = o.get("int") { return lit(v); }
    if let Some(v) = o.get("str") { return lit(v); }
    if let Some(v) = o.get("bool") { return lit(v); }
    if let Some(v) = o.get("list") {
        let xs = v.as_array().ok_or("list must be an array")?;
        let mut out = Vec::new();
        for x in xs { out.push(lower(x)?); }
        return Ok(format!("[{}]", out.join(", ")));
    }
    if let Some(v) = o.get("var") { return v.as_str().map(str::to_string).ok_or_else(|| "var must be a string".into()); }
    if let Some(v) = o.get("call") {
        let name = v.as_str().ok_or("call must be a string")?;
        let args = o.get("args").and_then(Value::as_array).cloned().unwrap_or_default();
        let mut out = Vec::new();
        for a in &args { out.push(lower(a)?); }
        return Ok(format!("{name}({})", out.join(", ")));
    }
    if let Some(v) = o.get("op") {
        let op = v.as_str().ok_or("op must be a string")?;
        let args = o.get("args").and_then(Value::as_array).ok_or("op needs args")?;
        let mut out = Vec::new();
        for a in args { out.push(lower(a)?); }
        return Ok(format!("({})", out.join(&format!(" {op} "))));
    }
    if let Some(v) = o.get("field") {
        let name = o.get("name").and_then(Value::as_str).ok_or("field needs name")?;
        return Ok(format!("{}.{name}", lower(v)?));
    }
    if let Some(v) = o.get("let") {
        let binds = v.as_array().ok_or("let must be an array")?;
        let mut out = lower(o.get("in").ok_or("let needs in")?)?;
        for b in binds.iter().rev() {
            let name = b.get("name").and_then(Value::as_str).ok_or("let bind needs name")?;
            let value = lower(b.get("value").ok_or("let bind needs value")?)?;
            out = format!("let {name} = {value} in {out}");
        }
        return Ok(out);
    }
    if let Some(v) = o.get("if") {
        let c = lower(v)?;
        let t = lower(o.get("then").ok_or("if needs then")?)?;
        let f = lower(o.get("else").ok_or("if needs else")?)?;
        return Ok(format!("(if {c} then {t} else {f})"));
    }
    if let Some(v) = o.get("match") {
        let scrut = lower(v)?;
        let cases = o.get("cases").and_then(Value::as_array).ok_or("match needs cases")?;
        let mut out = Vec::new();
        for c in cases {
            let p = c.get("pattern").and_then(Value::as_str).ok_or("case needs pattern")?;
            let b = lower(c.get("body").ok_or("case needs body")?)?;
            out.push(format!("{p} => {b}"));
        }
        return Ok(format!("(match {scrut} {{ {} }})", out.join(", ")));
    }
    if let Some(v) = o.get("record") {
        let name = v.as_str().ok_or("record must be a string")?;
        let fields = o.get("fields").and_then(Value::as_object).ok_or("record needs fields")?;
        let mut out = Vec::new();
        for (k, val) in fields { out.push(format!("{k}: {}", lower(val)?)); }
        return Ok(format!("{name} {{ {} }}", out.join(", ")));
    }
    Err(format!("unknown AST keys: {:?}", o.keys().collect::<Vec<_>>()))
}

fn collect_calls(e: &Value, out: &mut Vec<String>) {
    let Some(o) = e.as_object() else { return; };
    if let Some(Value::String(name)) = o.get("call") {
        out.push(name.clone());
        if let Some(Value::Array(args)) = o.get("args") { for a in args { collect_calls(a, out); } }
    }
    if let Some(Value::Array(args)) = o.get("op") { for a in args { collect_calls(a, out); } }
    if let Some(v) = o.get("field") { collect_calls(v, out); }
    if let Some(Value::Array(binds)) = o.get("let") {
        for b in binds { if let Some(v) = b.get("value") { collect_calls(v, out); } }
        if let Some(v) = o.get("in") { collect_calls(v, out); }
    }
    if let Some(v) = o.get("if") {
        collect_calls(v, out);
        if let Some(x) = o.get("then") { collect_calls(x, out); }
        if let Some(x) = o.get("else") { collect_calls(x, out); }
    }
    if let Some(v) = o.get("match") {
        collect_calls(v, out);
        if let Some(Value::Array(cases)) = o.get("cases") { for c in cases { if let Some(b) = c.get("body") { collect_calls(b, out); } } }
    }
    if let Some(Value::Object(fields)) = o.get("fields") { for v in fields.values() { collect_calls(v, out); } }
    if let Some(Value::Array(xs)) = o.get("list") { for x in xs { collect_calls(x, out); } }
}

/// Lower `{"main": expr, "effects": [...]?, "returns": ...?, "tests": [...]?}`
/// to a complete QAL source (main plus tests).
pub fn lower_main(ast: &Value) -> Result<String, String> {
    match ast.get("ast_version") {
        None => {}
        Some(v) if v.as_u64() == Some(1) => {}
        Some(v) => return Err(format!("unsupported ast_version: {v}")),
    }
    if let Some(Value::Array(allowed)) = ast.get("allowed_calls") {
        let allowed: Vec<&str> = allowed.iter().filter_map(Value::as_str).collect();
        let mut calls = Vec::new();
        if let Some(m) = ast.get("main") { collect_calls(m, &mut calls); }
        if let Some(Value::Array(tests)) = ast.get("tests") {
            for t in tests { if let Some(b) = t.get("body") { collect_calls(b, &mut calls); } }
        }
        for c in calls {
            if !allowed.contains(&c.as_str()) {
                return Err(format!("call not allowed by allowed_calls: {c}"));
            }
        }
    }
    let mut prefix = String::new();
    if let Some(Value::Array(types)) = ast.get("types") {
        for t in types {
            let name = t.get("name").and_then(Value::as_str).ok_or("type needs name")?;
            if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                return Err(format!("bad type name: {name}"));
            }
            let fields = t.get("fields").and_then(Value::as_object).ok_or("type needs fields")?;
            let mut fs = Vec::new();
            for (k, v) in fields {
                let ty = v.as_str().ok_or("type field type must be a string")?;
                fs.push(format!("{k}: {ty}"));
            }
            prefix.push_str(&format!("type {name} = {{ {} }}\n", fs.join(", ")));
        }
    }
    let main = ast.get("main");
    if main.is_none() && !matches!(ast.get("tests"), Some(Value::Array(xs)) if !xs.is_empty()) {
        return Err("AST needs a main or a non-empty tests array".into());
    }
    let mut src = prefix;
    if let Some(main) = main {
        let ret = ast.get("returns").and_then(Value::as_str).unwrap_or("Int");
        let effects = match ast.get("effects") {
            None => vec!["io".to_string()],
            Some(Value::Array(xs)) => xs.iter().map(|e| e.as_str().map(str::to_string).ok_or("effects must be strings"))
                .collect::<Result<Vec<_>, _>>()?,
            Some(_) => return Err("effects must be an array".into()),
        };
        for e in &effects {
            if !e.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
                return Err(format!("bad effect: {e}"));
            }
        }
        src.push_str(&format!("fn main(args: List[Str]) -> {ret} !{} = {{ {}; 0 }}\n",
                              effects.join(" !"), lower(main)?));
    }
    if let Some(Value::Array(tests)) = ast.get("tests") {
        for t in tests {
            let name = t.get("name").and_then(Value::as_str).ok_or("test needs name")?;
            if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                return Err(format!("bad test name: {name}"));
            }
            let body = lower(t.get("body").ok_or("test needs body")?)?;
            src.push_str(&format!("test {name} = {body}\n"));
        }
    }
    Ok(src)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn let_op() {
        let v: Value = serde_json::json!({"main":{"let":[{"name":"x","value":{"int":2}}],
            "in":{"call":"print","args":[{"call":"from_int","args":[{"op":"+","args":[{"var":"x"},{"int":3}]}]}]}}});
        let s = lower_main(&v).unwrap();
        assert!(s.contains("let x = 2 in print(from_int((x + 3)))"), "{s}");
    }
    #[test]
    fn match_record() {
        let v: Value = serde_json::json!({"main":{"match":{"call":"to_int","args":[{"str":"42"}]},
            "cases":[{"pattern":"None","body":{"int":0}},{"pattern":"Some(n)","body":{"var":"n"}}]}});
        assert!(lower_main(&v).unwrap().contains("match to_int(\"42\") { None => 0, Some(n) => n }"));
    }
    #[test]
    fn list_and_type() {
        let v: Value = serde_json::json!({"ast_version":1,
            "types":[{"name":"P","fields":{"x":"Int"}}],
            "main":{"call":"print","args":[{"call":"from_int","args":[{"field":{"record":"P","fields":{"x":{"int":7}}},"name":"x"}]}]}});
        let s = lower_main(&v).unwrap();
        assert!(s.contains("type P = { x: Int }"), "{s}");
    }
    #[test]
    fn list_literal() {
        let v: Value = serde_json::json!({"main":{"call":"print","args":[{"call":"from_int","args":[{"call":"len","args":[{"list":[{"int":1},{"int":2}]}]}]}]}});
        assert!(lower_main(&v).unwrap().contains("len([1, 2])"));
    }
    #[test]
    fn allowed_calls() {
        let v: Value = serde_json::json!({"allowed_calls":["read_all","sum_all","print","from_int"],
            "main":{"call":"print","args":[{"call":"from_int","args":[{"call":"sum_all","args":[{"call":"read_all","args":[]}]}]}]}});
        assert!(lower_main(&v).is_ok());
        let bad: Value = serde_json::json!({"allowed_calls":["read_all"],
            "main":{"call":"Int","args":[]}});
        assert!(lower_main(&bad).is_err());
    }
    #[test]
    fn tests_without_main() {
        let v: Value = serde_json::json!({"tests":[{"name":"t","body":{"op":"==","args":[{"int":1},{"int":1}]}}]});
        let s = lower_main(&v).unwrap();
        assert!(s.contains("test t = (1 == 1)") && !s.contains("fn main"), "{s}");
    }
    #[test]
    fn effects_and_tests() {
        let v: Value = serde_json::json!({"main":{"call":"print","args":[{"str":"ok"}]},
            "effects":["io","time"],"tests":[{"name":"t","body":{"op":"==","args":[{"int":1},{"int":1}]}}]});
        let s = lower_main(&v).unwrap();
        assert!(s.contains("fn main(args: List[Str]) -> Int !io !time"), "{s}");
        assert!(s.contains("test t = (1 == 1)"), "{s}");
    }
}
