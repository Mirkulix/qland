//! Phase 3c, spec Phase 3 §8 test layer 2: the Todo backend (`examples/todo/todo.qalpkg`, written
//! by an agent over MCP) against every scenario in `examples/todo/scenarios/*.json`, through
//! `qal request` (format and runner semantics: `examples/todo/scenarios/README.md`; a port of the
//! acceptance runner `run_scenarios.py`).
//!
//! The package is imported into a fresh `qal init` project (every hash must match the package).
//! Each scenario gets its own empty database file (`db:s<n>`), so scenarios run in parallel.

use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const MAX_PER_CALL: usize = 50;

/// A JSON value with the key order (and duplicate keys) of its text, ints apart from floats.
#[derive(Clone, Debug, PartialEq)]
enum J {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl<'de> Deserialize<'de> for J {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<J, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = J;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("JSON")
            }
            fn visit_unit<E>(self) -> Result<J, E> {
                Ok(J::Null)
            }
            fn visit_bool<E>(self, b: bool) -> Result<J, E> {
                Ok(J::Bool(b))
            }
            fn visit_i64<E>(self, n: i64) -> Result<J, E> {
                Ok(J::Int(n))
            }
            fn visit_u64<E: de::Error>(self, n: u64) -> Result<J, E> {
                i64::try_from(n).map(J::Int).map_err(|_| E::custom("integer out of range"))
            }
            fn visit_f64<E>(self, x: f64) -> Result<J, E> {
                Ok(J::Float(x))
            }
            fn visit_str<E>(self, s: &str) -> Result<J, E> {
                Ok(J::Str(s.to_string()))
            }
            fn visit_string<E>(self, s: String) -> Result<J, E> {
                Ok(J::Str(s))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<J, A::Error> {
                let mut v = Vec::new();
                while let Some(x) = a.next_element()? {
                    v.push(x);
                }
                Ok(J::Arr(v))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<J, A::Error> {
                let mut v = Vec::new();
                while let Some((k, x)) = a.next_entry::<String, J>()? {
                    v.push((k, x));
                }
                Ok(J::Obj(v))
            }
        }
        d.deserialize_any(V)
    }
}

impl Serialize for J {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            J::Null => s.serialize_unit(),
            J::Bool(b) => s.serialize_bool(*b),
            J::Int(n) => s.serialize_i64(*n),
            J::Float(x) => s.serialize_f64(*x),
            J::Str(t) => s.serialize_str(t),
            J::Arr(xs) => {
                let mut q = s.serialize_seq(Some(xs.len()))?;
                for x in xs {
                    q.serialize_element(x)?;
                }
                q.end()
            }
            J::Obj(ps) => {
                let mut m = s.serialize_map(Some(ps.len()))?;
                for (k, x) in ps {
                    m.serialize_entry(k, x)?;
                }
                m.end()
            }
        }
    }
}

impl J {
    fn get(&self, key: &str) -> Option<&J> {
        match self {
            J::Obj(ps) => ps.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    fn kind(&self) -> &'static str {
        match self {
            J::Null => "null",
            J::Bool(_) => "bool",
            J::Int(_) => "int",
            J::Float(_) => "float",
            J::Str(_) => "str",
            J::Arr(_) => "array",
            J::Obj(_) => "object",
        }
    }
    fn plain(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

/// The scenario file itself is invalid.
#[derive(Debug)]
struct Invalid(String);

type Env = BTreeMap<String, J>;

fn text_of(v: &J) -> String {
    match v {
        J::Str(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// `{{name}}` references in `s`, in order.
fn var_refs(s: &str) -> Vec<(usize, usize, String)> {
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(a) = s[i..].find("{{").map(|a| a + i) {
        let Some(b) = s[a + 2..].find("}}").map(|b| b + a + 2) else { break };
        let name = &s[a + 2..b];
        let valid = name.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_') && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if valid {
            out.push((a, b + 2, name.to_string()));
            i = b + 2;
        } else {
            i = a + 2;
        }
    }
    out
}

fn subst_text(s: &str, env: &Env) -> Result<String, Invalid> {
    let mut out = String::new();
    let mut last = 0;
    for (a, b, name) in var_refs(s) {
        let v = env.get(&name).ok_or_else(|| Invalid(format!("variable `{name}` is used before it is captured")))?;
        out.push_str(&s[last..a]);
        out.push_str(&text_of(v));
        last = b;
    }
    out.push_str(&s[last..]);
    Ok(out)
}

/// For `json` request bodies: a string that is exactly `"{{x}}"` becomes the captured value
/// itself; other strings get textual substitution.
fn subst_value(v: &J, env: &Env) -> Result<J, Invalid> {
    Ok(match v {
        J::Str(s) => match var_refs(s).as_slice() {
            [(0, end, name)] if *end == s.len() => env.get(name).cloned().ok_or_else(|| Invalid(format!("variable `{name}` is used before it is captured")))?,
            _ => J::Str(subst_text(s, env)?),
        },
        J::Arr(xs) => J::Arr(xs.iter().map(|x| subst_value(x, env)).collect::<Result<_, _>>()?),
        J::Obj(ps) => J::Obj(ps.iter().map(|(k, x)| Ok((subst_text(k, env)?, subst_value(x, env)?))).collect::<Result<_, Invalid>>()?),
        other => other.clone(),
    })
}

fn vars_in(v: &J) -> Vec<String> {
    match v {
        J::Str(s) => var_refs(s).into_iter().map(|(_, _, n)| n).collect(),
        J::Arr(xs) => xs.iter().flat_map(vars_in).collect(),
        J::Obj(ps) => ps.iter().flat_map(|(k, x)| var_refs(k).into_iter().map(|(_, _, n)| n).chain(vars_in(x))).collect(),
        _ => Vec::new(),
    }
}

fn build_request(rq: &J, env: &Env) -> Result<J, Invalid> {
    let J::Obj(ps) = rq else { return Err(Invalid("`request` must be an object".into())) };
    let mut out = Vec::new();
    for (k, v) in ps {
        let built = match (k.as_str(), v) {
            ("method" | "path" | "query" | "body", J::Str(s)) => J::Str(subst_text(s, env)?),
            ("headers", J::Arr(hs)) => J::Arr(
                hs.iter()
                    .map(|h| match (h.get("name"), h.get("value"), h) {
                        (Some(J::Str(n)), Some(J::Str(val)), J::Obj(f)) if f.len() == 2 => Ok(J::Obj(vec![("name".into(), J::Str(n.clone())), ("value".into(), J::Str(subst_text(val, env)?))])),
                        _ => Err(Invalid("request `headers` must be a list of {name, value}".into())),
                    })
                    .collect::<Result<_, _>>()?,
            ),
            ("json", v) => subst_value(v, env)?,
            (k, _) => return Err(Invalid(format!("invalid request field `{k}`"))),
        };
        out.push((k.clone(), built));
    }
    Ok(J::Obj(out))
}

const TYPES: [&str; 9] = ["int", "float", "number", "str", "bool", "null", "array", "object", "any"];

fn type_ok(t: &str, v: &J) -> bool {
    match t {
        "any" => true,
        "number" => matches!(v, J::Int(_) | J::Float(_)),
        t => v.kind() == t,
    }
}

fn is_matcher(e: &J) -> bool {
    matches!(e, J::Obj(ps) if ps.iter().any(|(k, _)| k.starts_with('$')))
}

/// A mismatch: (path, expected, actual, reason).
type Mismatch = (String, Value, Value, String);

/// `None` if `act` matches `exp` (structurally, key order and types included).
fn match_json(exp: &J, act: &J, env: &Env, path: &str) -> Result<Option<Mismatch>, Invalid> {
    let miss = |reason: String| Ok(Some((path.to_string(), exp.plain(), act.plain(), reason)));
    if is_matcher(exp) {
        let J::Obj(ps) = exp else { return Err(Invalid("matcher".into())) };
        return match ps.as_slice() {
            [(k, J::Str(t))] if k == "$type" => {
                if !TYPES.contains(&t.as_str()) {
                    return Err(Invalid(format!("unknown $type `{t}`")));
                }
                if type_ok(t, act) {
                    Ok(None)
                } else {
                    miss(format!("expected a value of type {t}, got {}", act.kind()))
                }
            }
            [(k, J::Str(name))] if k == "$var" => {
                let v = env.get(name).ok_or_else(|| Invalid(format!("variable `{name}` is used before it is captured")))?;
                if match_json(v, act, &Env::new(), path)?.is_none() {
                    Ok(None)
                } else {
                    Ok(Some((path.to_string(), v.plain(), act.plain(), format!("expected the captured value of `{name}`"))))
                }
            }
            [(k, _)] if k == "$any" => Ok(None),
            _ => Err(Invalid(format!("invalid matcher {}", exp.plain()))),
        };
    }
    match (exp, act) {
        (J::Obj(e), J::Obj(a)) => {
            let (ek, ak): (Vec<&String>, Vec<&String>) = (e.iter().map(|p| &p.0).collect(), a.iter().map(|p| &p.0).collect());
            if ek != ak {
                return miss(format!("object keys differ (order matters): expected {ek:?}, got {ak:?}"));
            }
            for ((k, x), (_, y)) in e.iter().zip(a) {
                if let Some(m) = match_json(x, y, env, &format!("{path}/{k}"))? {
                    return Ok(Some(m));
                }
            }
            Ok(None)
        }
        (J::Obj(_), _) => miss(format!("expected an object, got {}", act.kind())),
        (J::Arr(e), J::Arr(a)) => {
            if e.len() != a.len() {
                return miss(format!("expected {} elements, got {}", e.len(), a.len()));
            }
            for (i, (x, y)) in e.iter().zip(a).enumerate() {
                if let Some(m) = match_json(x, y, env, &format!("{path}/{i}"))? {
                    return Ok(Some(m));
                }
            }
            Ok(None)
        }
        (J::Arr(_), _) => miss(format!("expected an array, got {}", act.kind())),
        (J::Str(s), _) => {
            let want = J::Str(subst_text(s, env)?);
            if &want == act {
                Ok(None)
            } else {
                Ok(Some((path.to_string(), want.plain(), act.plain(), if act.kind() == "str" { "different value".into() } else { format!("expected str, got {}", act.kind()) })))
            }
        }
        _ if exp.kind() != act.kind() => miss(format!("expected {}, got {}", exp.kind(), act.kind())),
        _ if exp != act => miss("different value".into()),
        _ => Ok(None),
    }
}

/// One response of `qal request`.
struct Resp {
    raw: Value,
}

impl Resp {
    fn status(&self) -> Value {
        self.raw["status"].clone()
    }
    fn body(&self) -> &str {
        self.raw["body"].as_str().unwrap_or("")
    }
    /// Header names lowercase; the first of a name wins.
    fn headers(&self) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        for h in self.raw["headers"].as_array().into_iter().flatten() {
            if let (Some(n), Some(v)) = (h["name"].as_str(), h["value"].as_str()) {
                out.entry(n.to_ascii_lowercase()).or_insert_with(|| v.to_string());
            }
        }
        out
    }
    fn short(&self) -> Value {
        let mut r = json!({"status": self.status(), "headers": self.headers(), "body": self.body()});
        if let Some(e) = self.raw.get("error").filter(|e| !e.is_null()) {
            r["error"] = e.clone();
        }
        r
    }
}

const EXPECT_KEYS: [&str; 5] = ["status", "headers", "json", "body", "db"];

/// The first mismatch of one response.
fn check(step: &J, resp: &Resp, env: &Env) -> Result<Option<Value>, Invalid> {
    let empty = J::Obj(Vec::new());
    let exp = step.get("expect").unwrap_or(&empty);
    let J::Obj(fields) = exp else { return Err(Invalid("`expect` must be an object".into())) };
    if let Some((k, _)) = fields.iter().find(|(k, _)| !EXPECT_KEYS.contains(&k.as_str())) {
        return Err(Invalid(format!("unknown expect field `{k}`")));
    }
    if exp.get("json").is_some() && exp.get("body").is_some() {
        return Err(Invalid("expect has both `json` and `body`".into()));
    }
    if let Some(s) = exp.get("status") {
        if s.plain() != resp.status() {
            return Ok(Some(json!({"what": "status", "expected": s.plain(), "actual": resp.status()})));
        }
    }
    let hs = resp.headers();
    if let Some(J::Obj(want)) = exp.get("headers") {
        for (name, val) in want {
            let name = name.to_ascii_lowercase();
            match val {
                J::Null => {
                    if let Some(a) = hs.get(&name) {
                        return Ok(Some(json!({"what": format!("header {name}"), "expected": "absent", "actual": a})));
                    }
                }
                J::Str(v) => {
                    let w = subst_text(v, env)?;
                    if hs.get(&name) != Some(&w) {
                        return Ok(Some(json!({"what": format!("header {name}"), "expected": w, "actual": hs.get(&name)})));
                    }
                }
                _ => return Err(Invalid(format!("header `{name}`: expected a string or null"))),
            }
        }
    }
    if let Some(J::Str(b)) = exp.get("body") {
        let w = subst_text(b, env)?;
        if resp.body() != w {
            return Ok(Some(json!({"what": "body", "expected": w, "actual": resp.body()})));
        }
    }
    if let Some(want) = exp.get("json") {
        let act: J = match serde_json::from_str(resp.body()) {
            Ok(a) => a,
            Err(e) => return Ok(Some(json!({"what": "json", "expected": want.plain(), "actual": resp.body(), "reason": format!("body is not JSON: {e}")}))),
        };
        if let Some((path, e, a, reason)) = match_json(want, &act, env, "")? {
            return Ok(Some(json!({"what": "json", "path": if path.is_empty() { "/".into() } else { path }, "expected": e, "actual": a, "reason": reason})));
        }
    }
    if let Some(db) = exp.get("db") {
        if db.plain() != resp.raw["db"] {
            return Ok(Some(json!({"what": "db", "expected": db.plain(), "actual": resp.raw["db"]})));
        }
    }
    Ok(None)
}

/// Applies the captures of `step` to `env`; a capture that finds nothing is a mismatch.
fn capture(step: &J, resp: &Resp, env: &mut Env) -> Result<Option<Value>, Invalid> {
    let Some(J::Obj(caps)) = step.get("capture") else { return Ok(None) };
    for (name, at) in caps {
        let J::Str(at) = at else { return Err(Invalid(format!("capture `{name}` must be a string"))) };
        if let Some(h) = at.strip_prefix("header:") {
            let h = h.to_ascii_lowercase();
            match resp.headers().get(&h) {
                Some(v) => env.insert(name.clone(), J::Str(v.clone())),
                None => return Ok(Some(json!({"what": format!("capture {name}"), "expected": format!("header {h}"), "actual": null}))),
            };
            continue;
        }
        if !at.starts_with('/') && !at.is_empty() {
            return Err(Invalid(format!("capture `{name}`: use a JSON pointer like /id or header:<name>")));
        }
        let Ok(mut v) = serde_json::from_str::<J>(resp.body()) else {
            return Ok(Some(json!({"what": format!("capture {name}"), "expected": "a JSON body", "actual": resp.body()})));
        };
        for part in at.split('/').skip(1).map(|p| p.replace("~1", "/").replace("~0", "~")) {
            let next = match &v {
                J::Obj(_) => v.get(&part).cloned(),
                J::Arr(xs) => part.parse::<usize>().ok().and_then(|i| xs.get(i).cloned()),
                _ => None,
            };
            match next {
                Some(n) => v = n,
                None => return Ok(Some(json!({"what": format!("capture {name}"), "expected": format!("a value at {at}"), "actual": serde_json::from_str::<Value>(resp.body()).unwrap_or(Value::Null)}))),
            }
        }
        env.insert(name.clone(), v);
    }
    Ok(None)
}

/// Splits the steps into calls: a call ends before a request that uses a variable captured
/// inside it (and after 50 requests).
fn segments(steps: &[J]) -> Vec<Vec<usize>> {
    let (mut segs, mut cur, mut pending) = (Vec::new(), Vec::new(), Vec::<String>::new());
    for (i, st) in steps.iter().enumerate() {
        let need = st.get("request").map(vars_in).unwrap_or_default();
        if !cur.is_empty() && (need.iter().any(|n| pending.contains(n)) || cur.len() >= MAX_PER_CALL) {
            segs.push(std::mem::take(&mut cur));
            pending.clear();
        }
        cur.push(i);
        if let Some(J::Obj(caps)) = st.get("capture") {
            pending.extend(caps.iter().map(|(k, _)| k.clone()));
        }
    }
    if !cur.is_empty() {
        segs.push(cur);
    }
    segs
}

static CALLS: AtomicUsize = AtomicUsize::new(0);

/// One `qal request` call; the responses, or the mismatch that describes the failed call.
fn run_call(project: &Path, db: &str, requests: &[J], first: bool) -> Result<Vec<Resp>, Value> {
    let n = CALLS.fetch_add(1, Ordering::SeqCst);
    let file = project.join(format!("requests-{n}.json"));
    std::fs::write(&file, serde_json::to_string(&J::Arr(requests.to_vec())).unwrap_or_default()).map_err(|e| json!({"what": "qal request", "reason": e.to_string()}))?;
    let allow = format!("db:{db},time");
    let mut args = vec!["request", "handle", "--allow", allow.as_str()];
    if first {
        args.extend(["--init", "setup"]);
    }
    let path = file.to_string_lossy().into_owned();
    args.push(&path);
    let out = Command::new(env!("CARGO_BIN_EXE_qal")).args(&args).current_dir(project).stdin(Stdio::null()).output();
    let _ = std::fs::remove_file(&file);
    let out = out.map_err(|e| json!({"what": "qal request", "reason": e.to_string()}))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: Value = serde_json::from_str(&stdout).map_err(|_| json!({"what": "qal request", "reason": format!("exit {:?}, output is not JSON", out.status.code()), "stdout": stdout, "stderr": String::from_utf8_lossy(&out.stderr)}))?;
    let Some(rs) = v.get("responses").and_then(Value::as_array) else {
        return Err(json!({"what": "qal request", "reason": format!("exit {:?}", out.status.code()), "error": v.get("error").unwrap_or(&v)}));
    };
    if rs.len() != requests.len() {
        return Err(json!({"what": "qal request", "reason": format!("{} responses for {} requests", rs.len(), requests.len())}));
    }
    Ok(rs.iter().map(|r| Resp { raw: r.clone() }).collect())
}

/// Runs one scenario on its own database `db`; `None` when it passes, else the failure.
fn run_scenario(project: &Path, db: &str, sc: &J) -> Result<Option<Value>, Invalid> {
    let Some(J::Arr(steps)) = sc.get("steps") else { return Err(Invalid("`steps` must be a non-empty list".into())) };
    if steps.is_empty() {
        return Err(Invalid("`steps` must be a non-empty list".into()));
    }
    let mut env = Env::new();
    for (si, seg) in segments(steps).iter().enumerate() {
        let empty = J::Obj(Vec::new());
        let reqs = seg.iter().map(|&i| build_request(steps[i].get("request").unwrap_or(&empty), &env)).collect::<Result<Vec<_>, _>>()?;
        let resps = match run_call(project, db, &reqs, si == 0) {
            Ok(r) => r,
            Err(m) => return Ok(Some(json!({"step": seg[0], "mismatch": m}))),
        };
        for ((&i, resp), rq) in seg.iter().zip(&resps).zip(&reqs) {
            let st = &steps[i];
            // Captures first, so a step's expectations may use its own captures; a failed
            // capture reports the check mismatch first.
            let mut new_env = env.clone();
            let m = match capture(st, resp, &mut new_env)? {
                Some(cm) => Some(check(st, resp, &env).ok().flatten().unwrap_or(cm)),
                None => check(st, resp, &new_env)?,
            };
            match m {
                None => env = new_env,
                Some(m) => {
                    let name = match st.get("name") {
                        Some(J::Str(n)) => n.clone(),
                        _ => String::new(),
                    };
                    return Ok(Some(json!({"step": i, "step_name": name, "request": rq.plain(), "mismatch": m, "response": resp.short()})));
                }
            }
        }
    }
    Ok(None)
}

fn qal(dir: &Path, args: &[&str]) -> Value {
    let out = Command::new(env!("CARGO_BIN_EXE_qal")).args(args).current_dir(dir).stdin(Stdio::null()).output().unwrap();
    assert!(out.status.success(), "qal {args:?}: {}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    serde_json::from_slice(&out.stdout).unwrap()
}

fn todo_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/todo")
}

/// A fresh project with the std and the todo package; checks that every hash of the package
/// is reproduced.
fn todo_project() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qal-todo-scenarios-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    qal(&dir, &["init"]);
    let pkg = todo_dir().join("todo.qalpkg");
    let text = std::fs::read_to_string(&pkg).unwrap();
    let recorded: BTreeMap<&str, (&str, &str)> = text
        .lines()
        .filter_map(|l| l.strip_prefix("# def "))
        .map(|l| {
            let f: Vec<&str> = l.split(' ').collect();
            (f[0], (f[1], f[2]))
        })
        .collect();
    assert!(recorded.values().all(|(_, agent)| *agent == "todo-1"), "every definition was written by the agent todo-1");
    assert!(recorded.contains_key("handle") && recorded.contains_key("setup"));
    let imported = qal(&dir, &["import", pkg.to_str().unwrap()]);
    let defined = imported["imported"]["defined"].as_array().unwrap();
    assert_eq!(defined.len(), recorded.len());
    for d in defined {
        let name = d["name"].as_str().unwrap();
        assert_eq!(Some(d["hash"].as_str().unwrap()), recorded.get(name).map(|r| r.0), "{name}");
        assert_eq!(d["status"], "added", "{name}");
    }
    dir
}

#[test]
fn every_todo_scenario_passes_through_qal_request() {
    let project = todo_project();
    let mut files: Vec<PathBuf> = std::fs::read_dir(todo_dir().join("scenarios")).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
    files.sort();
    assert_eq!(files.len(), 54, "7 public and 47 hidden scenarios");
    let next = AtomicUsize::new(0);
    let failures = Mutex::new(Vec::new());
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                let Some(f) = files.get(i) else { break };
                let stem = f.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
                let result = std::fs::read_to_string(f)
                    .map_err(|e| Invalid(e.to_string()))
                    .and_then(|t| serde_json::from_str::<J>(&t).map_err(|e| Invalid(e.to_string())))
                    .and_then(|sc| run_scenario(&project, &format!("s{i}"), &sc));
                let failure = match result {
                    Ok(None) => None,
                    Ok(Some(f)) => Some(f),
                    Err(Invalid(reason)) => Some(json!({"step": null, "mismatch": {"what": "invalid scenario", "reason": reason}})),
                };
                if let Some(mut fl) = failure {
                    fl["scenario"] = json!(stem);
                    failures.lock().unwrap().push(fl.to_string());
                }
            });
        }
    });
    let _ = std::fs::remove_dir_all(&project);
    let failures = failures.into_inner().unwrap();
    assert!(failures.is_empty(), "{} of {} scenarios failed:\n{}", failures.len(), files.len(), failures.join("\n"));
}

/// The runner itself: key order, int vs float, matchers, captures and segmentation.
#[test]
fn the_runner_compares_strictly() {
    let p = |s: &str| serde_json::from_str::<J>(s).unwrap();
    let mut env = Env::new();
    env.insert("a".into(), J::Int(7));
    let ok = |e: &str, a: &str, env: &Env| match_json(&p(e), &p(a), env, "").unwrap().is_none();
    assert!(ok(r#"{"id":{"$var":"a"},"t":{"$type":"int"}}"#, r#"{"id":7,"t":3}"#, &env));
    assert!(!ok(r#"{"a":1,"b":2}"#, r#"{"b":2,"a":1}"#, &env), "key order matters");
    assert!(!ok(r#"{"a":1}"#, r#"{"a":1,"a":1}"#, &env), "duplicate keys are seen");
    assert!(!ok("1", "1.0", &env) && !ok("1", "true", &env) && !ok(r#"{"$type":"int"}"#, "1.5", &env));
    assert!(ok(r#"{"$type":"number"}"#, "1.5", &env) && ok(r#"{"$any":true}"#, "[1]", &env));
    assert!(ok(r#""/todos/{{a}}""#, r#""/todos/7""#, &env) && !ok("[1,2]", "[1]", &env));
    assert!(match_json(&p(r#"{"$var":"zz"}"#), &J::Null, &env, "").is_err(), "an unknown variable is an invalid scenario");
    assert_eq!(subst_value(&p(r#"{"id":"{{a}}","t":"x{{a}}"}"#), &env).unwrap(), p(r#"{"id":7,"t":"x7"}"#));
    let steps = p(r#"[{"request":{},"capture":{"a":"/id"}},{"request":{"path":"/x"}},{"request":{"path":"/t/{{a}}"}},{"request":{}}]"#);
    let J::Arr(steps) = steps else { panic!() };
    assert_eq!(segments(&steps), vec![vec![0, 1], vec![2, 3]]);
    let resp = Resp { raw: json!({"status": 201, "headers": [{"name": "Location", "value": "/todos/9"}], "body": "[{\"id\":9}]"}) };
    let mut env = Env::new();
    assert!(capture(&p(r#"{"capture":{"a":"/0/id","l":"header:location"}}"#), &resp, &mut env).unwrap().is_none());
    assert_eq!((env.get("a"), env.get("l")), (Some(&J::Int(9)), Some(&J::Str("/todos/9".into()))));
    assert!(check(&p(r#"{"expect":{"status":201,"headers":{"location":"/todos/{{a}}","content-type":null}}}"#), &resp, &env).unwrap().is_none());
    assert!(check(&p(r#"{"expect":{"status":200}}"#), &resp, &env).unwrap().is_some());
}
