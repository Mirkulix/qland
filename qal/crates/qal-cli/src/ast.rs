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

/// Lower `{"main": expr}` to a complete main definition.
pub fn lower_main(ast: &Value) -> Result<String, String> {
    let main = ast.get("main").ok_or("AST needs a main field")?;
    Ok(format!("fn main(args: List[Str]) -> Int !io = {{ {}; 0 }}\n", lower(main)?))
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
}
