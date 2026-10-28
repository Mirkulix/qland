//! QAL values on the host side: decoding from and encoding into guest memory, and JSON form.

use qal_core::ast::Type;
use qal_core::tast::TProgram;
use serde::ser::{SerializeMap, SerializeSeq, Serializer};
use serde::Serialize;

/// Values deeper than this are shown as `"..."` (guards the host stack against huge recursive values).
const MAX_DEPTH: usize = 200;
/// Total nodes per decoded value; shared substructure can otherwise expand exponentially.
const MAX_NODES: usize = 100_000;
/// Total string bytes per decoded value; longer strings are cut and end in `...(+N bytes)`.
pub const MAX_STR_BYTES: usize = 256 << 10;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    /// JSON `null`.
    Unit,
    Str(String),
    List(Vec<Value>),
    /// Record type name and fields in declaration order; JSON object.
    Record(String, Vec<(String, Value)>),
    /// Variant name and payload; JSON `"Name"` without payload, else `{"Name": [payload...]}`.
    Variant(String, Vec<Value>),
    /// The kept bytes and the number of bytes cut off by the string-byte budget; JSON array of numbers
    /// 0..=255, followed by the string `"...(+N bytes)"` if bytes were cut off.
    Bytes(Vec<u8>, usize),
    /// Entries in key order; JSON array of `[key, value]` pairs.
    Map(Vec<(Value, Value)>),
}

impl Serialize for Value {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Value::Int(n) => s.serialize_i64(*n),
            Value::Float(x) if x.is_finite() => s.serialize_f64(*x),
            Value::Float(x) => s.serialize_str(&format!("{x:?}")),
            Value::Bool(b) => s.serialize_bool(*b),
            Value::Unit => s.serialize_unit(),
            Value::Str(t) => s.serialize_str(t),
            Value::List(xs) => xs.serialize(s),
            Value::Record(_, fields) => {
                let mut m = s.serialize_map(Some(fields.len()))?;
                for (k, v) in fields {
                    m.serialize_entry(k, v)?;
                }
                m.end()
            }
            Value::Variant(name, xs) if xs.is_empty() => s.serialize_str(name),
            Value::Variant(name, xs) => {
                let mut m = s.serialize_map(Some(1))?;
                m.serialize_entry(name, xs)?;
                m.end()
            }
            Value::Bytes(b, 0) => b.serialize(s),
            Value::Bytes(b, cut) => {
                let mut seq = s.serialize_seq(Some(b.len() + 1))?;
                for x in b {
                    seq.serialize_element(x)?;
                }
                seq.serialize_element(&format!("...(+{cut} bytes)"))?;
                seq.end()
            }
            Value::Map(es) => es.iter().map(|(k, v)| [k, v]).collect::<Vec<_>>().serialize(s),
        }
    }
}

/// Named values (contract parameters, operands); serialized as a JSON object in order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Values(pub Vec<(String, Value)>);

impl Values {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Serialize for Values {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            m.serialize_entry(k, v)?;
        }
        m.end()
    }
}

fn u32_at(mem: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(mem.get(at..at + 4)?.try_into().ok()?))
}

pub fn slot_at(mem: &[u8], at: usize) -> Option<i64> {
    Some(i64::from_le_bytes(mem.get(at..at + 8)?.try_into().ok()?))
}

pub fn read_str(mem: &[u8], ptr: u32) -> Option<String> {
    let len = u32_at(mem, ptr as usize)? as usize;
    let bytes = mem.get(ptr as usize + 4..ptr as usize + 4 + len)?;
    Some(String::from_utf8_lossy(bytes).into_owned())
}

/// Decodes a value of type `ty` from its slot form (see `qal_wasm` layout).
pub fn decode(tp: &TProgram, mem: &[u8], ty: &Type, slot: i64) -> Value {
    decode_at(tp, mem, ty, slot, 0, &mut Budget { nodes: 0, bytes: 0 })
}

/// What one decoded value has used of its node and string-byte limits.
struct Budget {
    nodes: usize,
    bytes: usize,
}

/// The string at `ptr`, cut (at a character boundary) to what is left of the string-byte budget.
/// Only the kept prefix is copied and validated, so repeated references to one huge string stay cheap;
/// the `...(+N bytes)` suffix comes from the declared length.
fn budget_str(mem: &[u8], ptr: u32, budget: &mut Budget) -> Option<Value> {
    let len = u32_at(mem, ptr as usize)? as usize;
    let start = (ptr as usize).checked_add(4)?;
    let bytes = mem.get(start..start.checked_add(len)?)?;
    let room = MAX_STR_BYTES.saturating_sub(budget.bytes);
    if len <= room {
        let s = String::from_utf8_lossy(bytes).into_owned();
        budget.bytes += s.len();
        return Some(Value::Str(s));
    }
    let mut end = room;
    while end > 0 && (bytes[end] & 0xC0) == 0x80 {
        end -= 1;
    }
    let kept = String::from_utf8_lossy(&bytes[..end]).into_owned();
    budget.bytes = MAX_STR_BYTES;
    Some(Value::Str(format!("{kept}...(+{} bytes)", len - end)))
}

/// The bytes at `ptr`, cut to what is left of the string-byte budget (bytes count like string bytes).
fn budget_bytes(mem: &[u8], ptr: u32, budget: &mut Budget) -> Option<Value> {
    let len = u32_at(mem, ptr as usize)? as usize;
    let start = (ptr as usize).checked_add(4)?;
    let bytes = mem.get(start..start.checked_add(len)?)?;
    let keep = len.min(MAX_STR_BYTES.saturating_sub(budget.bytes));
    budget.bytes += keep;
    Some(Value::Bytes(bytes[..keep].to_vec(), len - keep))
}

/// Height bound of a map tree in guest memory (an AVL tree of 2^32 entries is lower than 64).
const MAX_MAP_HEIGHT: usize = 64;

/// Collects the (key slot, value slot) pairs of the map tree at `ptr` in key order (at most
/// `MAX_NODES`); false if the memory does not hold a well-formed tree.
fn map_slots(mem: &[u8], ptr: u32, height: usize, out: &mut Vec<(i64, i64)>) -> bool {
    if ptr == 0 {
        return true;
    }
    if height > MAX_MAP_HEIGHT {
        return false;
    }
    if out.len() >= MAX_NODES {
        return true;
    }
    let at = ptr as usize;
    let (Some(k), Some(v), Some(l), Some(r)) = (slot_at(mem, at + 8), slot_at(mem, at + 16), u32_at(mem, at + 24), u32_at(mem, at + 28)) else { return false };
    if !map_slots(mem, l, height + 1, out) {
        return false;
    }
    out.push((k, v));
    map_slots(mem, r, height + 1, out)
}

fn decode_at(tp: &TProgram, mem: &[u8], ty: &Type, slot: i64, depth: usize, budget: &mut Budget) -> Value {
    budget.nodes += 1;
    if depth > MAX_DEPTH || budget.nodes > MAX_NODES {
        return Value::Str("...".into());
    }
    let ptr = slot as u32;
    let bad = || Value::Str(format!("<invalid {ty} at {ptr}>"));
    match ty {
        Type::Int => Value::Int(slot),
        Type::Float => Value::Float(f64::from_bits(slot as u64)),
        Type::Bool => Value::Bool(slot != 0),
        Type::Unit | Type::Var(_) | Type::Param(_) => Value::Unit,
        Type::Fn(..) => Value::Str(format!("<fn #{ptr}>")),
        Type::Str => budget_str(mem, ptr, budget).unwrap_or_else(bad),
        Type::Bytes => budget_bytes(mem, ptr, budget).unwrap_or_else(bad),
        Type::Map(kt, vt) => {
            let mut slots = Vec::new();
            if !map_slots(mem, ptr, 0, &mut slots) {
                return bad();
            }
            let mut out = Vec::new();
            for (ks, vs) in slots {
                let key = decode_at(tp, mem, kt, ks, depth + 1, budget);
                out.push((key, decode_at(tp, mem, vt, vs, depth + 1, budget)));
                if budget.nodes > MAX_NODES {
                    break;
                }
            }
            Value::Map(out)
        }
        Type::List(elem) => {
            let Some(len) = u32_at(mem, ptr as usize) else { return bad() };
            let mut out = Vec::new();
            for k in 0..len as usize {
                let Some(s) = slot_at(mem, ptr as usize + 8 + 8 * k) else { return bad() };
                out.push(decode_at(tp, mem, elem, s, depth + 1, budget));
            }
            Value::List(out)
        }
        _ if tp.is_record(ty) => {
            let name = ty.to_string();
            let mut fields = Vec::new();
            for (k, (f, ft)) in tp.record_fields(ty).iter().enumerate() {
                let Some(s) = slot_at(mem, ptr as usize + 8 * k) else { return bad() };
                fields.push((f.clone(), decode_at(tp, mem, ft, s, depth + 1, budget)));
            }
            Value::Record(name, fields)
        }
        _ => {
            let Some(tag) = u32_at(mem, ptr as usize) else { return bad() };
            let names = tp.variant_names(ty);
            let Some(name) = names.get(tag as usize) else { return bad() };
            let mut payload = Vec::new();
            for (k, ft) in tp.variant_fields(ty, tag).iter().enumerate() {
                let Some(s) = slot_at(mem, ptr as usize + 8 + 8 * k) else { return bad() };
                payload.push(decode_at(tp, mem, ft, s, depth + 1, budget));
            }
            Value::Variant(name.clone(), payload)
        }
    }
}

/// Converts a JSON value to a value of type `ty`, in the JSON form values are reported in: Int and
/// Float numbers, Bool, Unit `null`, Str strings, List and Bytes arrays (Bytes of numbers 0..=255),
/// Map an array of `[key, value]` pairs, a record an object with every field, a variant `"Name"`
/// without payload or `{"Name": [payload...]}`. Function values cannot be passed.
pub fn from_json(tp: &TProgram, ty: &Type, v: &serde_json::Value) -> Result<Value, String> {
    use serde_json::Value as J;
    let shown = |v: &J| -> String { v.to_string().chars().take(40).collect() };
    let mismatch = |v: &J| Err(format!("expected {ty}, got {}", shown(v)));
    Ok(match (ty, v) {
        (Type::Int, J::Number(n)) => match n.as_i64() {
            Some(i) => Value::Int(i),
            None => return mismatch(v),
        },
        (Type::Float, J::Number(n)) => match n.as_f64() {
            Some(x) => Value::Float(x),
            None => return mismatch(v),
        },
        // The forms non-finite results are reported in.
        (Type::Float, J::String(s)) if matches!(s.as_str(), "NaN" | "inf" | "-inf") => Value::Float(match s.as_str() {
            "NaN" => f64::NAN,
            "inf" => f64::INFINITY,
            _ => f64::NEG_INFINITY,
        }),
        (Type::Bool, J::Bool(b)) => Value::Bool(*b),
        (Type::Unit, J::Null) => Value::Unit,
        (Type::Str, J::String(s)) => Value::Str(s.clone()),
        (Type::List(elem), J::Array(xs)) => Value::List(xs.iter().map(|x| from_json(tp, elem, x)).collect::<Result<_, _>>()?),
        (Type::Bytes, J::Array(xs)) => {
            let bytes = xs.iter().map(|x| x.as_u64().and_then(|b| u8::try_from(b).ok()).ok_or_else(|| format!("expected a byte (0..=255), got {}", shown(x)))).collect::<Result<_, _>>()?;
            Value::Bytes(bytes, 0)
        }
        (Type::Map(kt, vt), J::Array(xs)) => {
            let mut entries = Vec::new();
            for x in xs {
                match x.as_array().map(Vec::as_slice) {
                    Some([k, v]) => entries.push((from_json(tp, kt, k)?, from_json(tp, vt, v)?)),
                    _ => return Err(format!("expected a [key, value] pair of {ty}, got {}", shown(x))),
                }
            }
            Value::Map(entries)
        }
        (Type::Fn(..), _) => return Err(format!("has type {ty}; functions cannot be passed as arguments")),
        (_, J::Object(fields)) if tp.is_record(ty) => {
            let decl = tp.record_fields(ty);
            if let Some(extra) = fields.keys().find(|k| !decl.iter().any(|(f, _)| f == *k)) {
                return Err(format!("{ty} has no field `{extra}`"));
            }
            let mut out = Vec::new();
            for (f, ft) in &decl {
                let x = fields.get(f).ok_or_else(|| format!("{ty} needs the field `{f}`"))?;
                out.push((f.clone(), from_json(tp, ft, x).map_err(|m| format!("field `{f}`: {m}"))?));
            }
            Value::Record(ty.to_string(), out)
        }
        (_, J::String(_) | J::Object(_)) if !tp.variant_names(ty).is_empty() => {
            let (name, payload): (&str, &[J]) = match v {
                J::String(n) => (n, &[]),
                J::Object(o) if o.len() == 1 => match o.iter().next() {
                    Some((n, J::Array(xs))) => (n, xs),
                    _ => return mismatch(v),
                },
                _ => return mismatch(v),
            };
            let names = tp.variant_names(ty);
            let Some(tag) = names.iter().position(|n| n == name) else {
                return Err(format!("{ty} has no variant `{name}` (variants: {})", names.join(", ")));
            };
            let fields = tp.variant_fields(ty, tag as u32);
            if fields.len() != payload.len() {
                return Err(format!("`{name}` takes {} value(s), got {}", fields.len(), payload.len()));
            }
            Value::Variant(name.to_string(), fields.iter().zip(payload).map(|(ft, x)| from_json(tp, ft, x)).collect::<Result<_, _>>()?)
        }
        _ => return mismatch(v),
    })
}

/// The total order of map keys, as the generated code's compare functions define it: Int and Bool
/// numerically, Float by IEEE total order, Str and Bytes bytewise (a prefix first), lists and maps
/// lexicographically (maps by their entries in key order), variants by tag then payload, records
/// by their fields in declaration order. Values that do not fit `ty` compare equal.
pub fn compare(tp: &TProgram, ty: &Type, a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let total = |x: f64| {
        let bits = x.to_bits() as i64;
        bits ^ (((bits >> 63) as u64) >> 1) as i64
    };
    match (ty, a, b) {
        (_, Value::Int(x), Value::Int(y)) => x.cmp(y),
        (_, Value::Bool(x), Value::Bool(y)) => x.cmp(y),
        (_, Value::Float(x), Value::Float(y)) => total(*x).cmp(&total(*y)),
        (_, Value::Str(x), Value::Str(y)) => x.as_bytes().cmp(y.as_bytes()),
        (_, Value::Bytes(x, _), Value::Bytes(y, _)) => x.cmp(y),
        (Type::List(t), Value::List(xs), Value::List(ys)) => {
            for (x, y) in xs.iter().zip(ys) {
                let c = compare(tp, t, x, y);
                if c != Ordering::Equal {
                    return c;
                }
            }
            xs.len().cmp(&ys.len())
        }
        (Type::Map(kt, vt), Value::Map(xs), Value::Map(ys)) => {
            let sorted = |es: &[(Value, Value)]| {
                let mut es: Vec<(Value, Value)> = es.to_vec();
                es.sort_by(|a, b| compare(tp, kt, &a.0, &b.0));
                es
            };
            let (xs, ys) = (sorted(xs), sorted(ys));
            for ((xk, xv), (yk, yv)) in xs.iter().zip(&ys) {
                let c = compare(tp, kt, xk, yk).then_with(|| compare(tp, vt, xv, yv));
                if c != Ordering::Equal {
                    return c;
                }
            }
            xs.len().cmp(&ys.len())
        }
        (_, Value::Record(_, xs), Value::Record(_, ys)) => {
            for (f, ft) in tp.record_fields(ty) {
                let field = |fs: &[(String, Value)]| fs.iter().find(|(g, _)| *g == f).map(|(_, v)| v.clone());
                let c = match (field(xs), field(ys)) {
                    (Some(x), Some(y)) => compare(tp, &ft, &x, &y),
                    _ => Ordering::Equal,
                };
                if c != Ordering::Equal {
                    return c;
                }
            }
            Ordering::Equal
        }
        (_, Value::Variant(x, xs), Value::Variant(y, ys)) => {
            let names = tp.variant_names(ty);
            let tag = |n: &str| names.iter().position(|m| m == n).unwrap_or(usize::MAX);
            let (tx, ty_) = (tag(x), tag(y));
            if tx != ty_ {
                return tx.cmp(&ty_);
            }
            for ((x, y), ft) in xs.iter().zip(ys).zip(tp.variant_fields(ty, tx as u32)) {
                let c = compare(tp, &ft, x, y);
                if c != Ordering::Equal {
                    return c;
                }
            }
            Ordering::Equal
        }
        _ => Ordering::Equal,
    }
}

/// Host bytes one decoded node is counted as (the `Value` plus its share of names and vectors).
pub const NODE_BYTES: usize = 64;

/// Decodes a value the host acts on (an `HttpCall`, SQL parameters, a handler's `HttpResponse`):
/// exactly, without cutting strings, and failing on anything malformed or on more than `max_bytes`
/// (string bytes plus [`NODE_BYTES`] per node).
pub fn decode_exact(tp: &TProgram, mem: &[u8], ty: &Type, slot: i64, max_bytes: usize) -> Result<Value, String> {
    decode_exact_counted(tp, mem, ty, slot, max_bytes).map(|(v, _)| v)
}

/// [`decode_exact`], also returning the host bytes the value is counted as.
pub fn decode_exact_counted(tp: &TProgram, mem: &[u8], ty: &Type, slot: i64, max_bytes: usize) -> Result<(Value, usize), String> {
    let mut used = 0usize;
    let v = exact_at(tp, mem, ty, slot, 0, max_bytes, &mut used)?;
    Ok((v, used))
}

fn exact_at(tp: &TProgram, mem: &[u8], ty: &Type, slot: i64, depth: usize, max: usize, used: &mut usize) -> Result<Value, String> {
    *used = used.saturating_add(NODE_BYTES);
    if *used > max {
        return Err(format!("the value is larger than {max} bytes"));
    }
    if depth > MAX_DEPTH {
        return Err("the value is nested too deeply".into());
    }
    let ptr = slot as u32 as usize;
    let bad = || format!("invalid {ty} at {ptr}");
    Ok(match ty {
        Type::Int => Value::Int(slot),
        Type::Float => Value::Float(f64::from_bits(slot as u64)),
        Type::Bool => Value::Bool(slot != 0),
        Type::Unit => Value::Unit,
        Type::Str => {
            let len = u32_at(mem, ptr).ok_or_else(bad)? as usize;
            *used = used.saturating_add(len);
            if *used > max {
                return Err(format!("the value is larger than {max} bytes"));
            }
            let bytes = mem.get(ptr + 4..ptr + 4 + len).ok_or_else(bad)?;
            Value::Str(String::from_utf8(bytes.to_vec()).map_err(|_| format!("a string is not valid UTF-8 at {ptr}"))?)
        }
        Type::List(elem) => {
            let len = u32_at(mem, ptr).ok_or_else(bad)? as usize;
            let mut out = Vec::with_capacity(len.min(1024));
            for k in 0..len {
                let s = slot_at(mem, ptr + 8 + 8 * k).ok_or_else(bad)?;
                out.push(exact_at(tp, mem, elem, s, depth + 1, max, used)?);
            }
            Value::List(out)
        }
        Type::Bytes => {
            let len = u32_at(mem, ptr).ok_or_else(bad)? as usize;
            *used = used.saturating_add(len);
            if *used > max {
                return Err(format!("the value is larger than {max} bytes"));
            }
            Value::Bytes(mem.get(ptr + 4..ptr + 4 + len).ok_or_else(bad)?.to_vec(), 0)
        }
        Type::Map(kt, vt) => {
            let mut out = Vec::new();
            exact_map(tp, mem, (kt, vt), ptr as u32, 0, depth, max, used, &mut out)?;
            Value::Map(out)
        }
        Type::Fn(..) | Type::Var(_) | Type::Param(_) => return Err(format!("{ty} cannot be read by the host")),
        // Records and enums, also generic instances (`Type::App`), and Option/Result.
        _ if tp.is_record(ty) => {
            let mut fields = Vec::new();
            for (k, (f, ft)) in tp.record_fields(ty).iter().enumerate() {
                let s = slot_at(mem, ptr + 8 * k).ok_or_else(bad)?;
                fields.push((f.clone(), exact_at(tp, mem, ft, s, depth + 1, max, used)?));
            }
            Value::Record(ty.to_string(), fields)
        }
        _ => {
            let tag = u32_at(mem, ptr).ok_or_else(bad)?;
            let names = tp.variant_names(ty);
            let name = names.get(tag as usize).ok_or_else(bad)?;
            let mut payload = Vec::new();
            for (k, ft) in tp.variant_fields(ty, tag).iter().enumerate() {
                let s = slot_at(mem, ptr + 8 + 8 * k).ok_or_else(bad)?;
                payload.push(exact_at(tp, mem, ft, s, depth + 1, max, used)?);
            }
            Value::Variant(name.clone(), payload)
        }
    })
}

/// The entries of the map tree at `ptr` in key order, exactly (a malformed tree is an error).
#[allow(clippy::too_many_arguments)]
fn exact_map(tp: &TProgram, mem: &[u8], (kt, vt): (&Type, &Type), ptr: u32, height: usize, depth: usize, max: usize, used: &mut usize, out: &mut Vec<(Value, Value)>) -> Result<(), String> {
    if ptr == 0 {
        return Ok(());
    }
    let at = ptr as usize;
    let (Some(k), Some(v), Some(l), Some(r)) = (slot_at(mem, at + 8), slot_at(mem, at + 16), u32_at(mem, at + 24), u32_at(mem, at + 28)) else {
        return Err(format!("invalid map at {ptr}"));
    };
    if height > MAX_MAP_HEIGHT {
        return Err(format!("invalid map at {ptr}"));
    }
    exact_map(tp, mem, (kt, vt), l, height + 1, depth, max, used, out)?;
    let key = exact_at(tp, mem, kt, k, depth + 1, max, used)?;
    out.push((key, exact_at(tp, mem, vt, v, depth + 1, max, used)?));
    exact_map(tp, mem, (kt, vt), r, height + 1, depth, max, used, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_as_plain_json() {
        let v = vec![
            Value::Int(-3),
            Value::Bool(true),
            Value::Unit,
            Value::Float(1.5),
            Value::Float(f64::NAN),
            Value::Str("a".into()),
            Value::List(vec![Value::Int(1)]),
            Value::Variant("None".into(), vec![]),
            Value::Variant("Some".into(), vec![Value::Int(2)]),
            Value::Record("P".into(), vec![("y".into(), Value::Int(1)), ("x".into(), Value::Int(2))]),
        ];
        assert_eq!(serde_json::to_string(&v).unwrap(), r#"[-3,true,null,1.5,"NaN","a",[1],"None",{"Some":[2]},{"y":1,"x":2}]"#);
        let m = Value::Map(vec![(Value::Str("a".into()), Value::Bytes(vec![1, 255], 0))]);
        assert_eq!(serde_json::to_string(&m).unwrap(), r#"[["a",[1,255]]]"#);
        assert_eq!(serde_json::to_string(&Value::Bytes(vec![7], 3)).unwrap(), r#"[7,"...(+3 bytes)"]"#);
        let vals = Values(vec![("n".into(), Value::Int(3))]);
        assert_eq!(serde_json::to_string(&vals).unwrap(), r#"{"n":3}"#);
    }

    #[test]
    fn non_finite_floats_round_trip_through_json() {
        let tp = TProgram::default();
        for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.5] {
            let json = serde_json::to_value(Value::Float(x)).unwrap();
            let Value::Float(y) = from_json(&tp, &Type::Float, &json).unwrap() else { panic!("not a Float") };
            assert_eq!(x.to_bits(), y.to_bits(), "{json}");
        }
        assert!(from_json(&tp, &Type::Float, &serde_json::json!("nan")).is_err());
    }

    #[test]
    fn decodes_from_memory_layout() {
        let tp = TProgram::default();
        // Str at 0: "hi"; List[Int] at 8: [7]; Some(5) at 24
        let mut mem = vec![0u8; 48];
        mem[0..4].copy_from_slice(&2u32.to_le_bytes());
        mem[4..6].copy_from_slice(b"hi");
        mem[8..12].copy_from_slice(&1u32.to_le_bytes());
        mem[16..24].copy_from_slice(&7i64.to_le_bytes());
        mem[24..28].copy_from_slice(&1u32.to_le_bytes());
        mem[32..40].copy_from_slice(&5i64.to_le_bytes());
        assert_eq!(decode(&tp, &mem, &Type::Str, 0), Value::Str("hi".into()));
        assert_eq!(decode(&tp, &mem, &Type::list(Type::Int), 8), Value::List(vec![Value::Int(7)]));
        assert_eq!(decode(&tp, &mem, &Type::option(Type::Int), 24), Value::Variant("Some".into(), vec![Value::Int(5)]));
        assert_eq!(decode(&tp, &mem, &Type::Str, 1000), Value::Str("<invalid Str at 1000>".into()));
    }

    #[test]
    fn strings_share_one_byte_budget_per_value() {
        let tp = TProgram::default();
        // List of two strings of MAX_STR_BYTES - 1 and 10 bytes.
        let big = MAX_STR_BYTES - 1;
        let mut mem = vec![b'a'; 32 + big + 4 + 10];
        mem[0..4].copy_from_slice(&2u32.to_le_bytes());
        mem[8..16].copy_from_slice(&24i64.to_le_bytes());
        let second = 24 + 4 + big;
        mem[16..24].copy_from_slice(&(second as i64).to_le_bytes());
        mem[24..28].copy_from_slice(&(big as u32).to_le_bytes());
        mem[second..second + 4].copy_from_slice(&10u32.to_le_bytes());
        let Value::List(xs) = decode(&tp, &mem, &Type::list(Type::Str), 0) else { panic!("not a list") };
        assert_eq!(xs[0], Value::Str("a".repeat(big)));
        assert_eq!(xs[1], Value::Str("a...(+9 bytes)".into()));
    }

    #[test]
    fn many_references_to_one_large_string_decode_quickly() {
        let tp = TProgram::default();
        // One 16 MiB string after the slots, referenced by 50 000 list slots.
        let len = 16usize << 20;
        let n = 50_000usize;
        let mut mem = vec![b'a'; 8 + 8 * n + 4 + len];
        let at = 8 + 8 * n;
        mem[0..4].copy_from_slice(&(n as u32).to_le_bytes());
        for k in 0..n {
            mem[8 + 8 * k..16 + 8 * k].copy_from_slice(&(at as i64).to_le_bytes());
        }
        mem[at..at + 4].copy_from_slice(&(len as u32).to_le_bytes());
        let t = std::time::Instant::now();
        let Value::List(xs) = decode(&tp, &mem, &Type::list(Type::Str), 0) else { panic!("not a list") };
        assert!(t.elapsed() < std::time::Duration::from_secs(2), "{:?}", t.elapsed());
        let Value::Str(first) = &xs[0] else { panic!("not a string") };
        assert_eq!(first.len(), MAX_STR_BYTES + format!("...(+{} bytes)", len - MAX_STR_BYTES).len());
        assert_eq!(xs[1], Value::Str(format!("...(+{len} bytes)")));
    }

    #[test]
    fn exact_decoding_never_cuts_and_bounds_the_size() {
        let tp = TProgram::default();
        let mut mem = vec![0u8; 64];
        mem[0..4].copy_from_slice(&5u32.to_le_bytes());
        mem[4..9].copy_from_slice(b"hello");
        mem[16..20].copy_from_slice(&2u32.to_le_bytes());
        mem[24..32].copy_from_slice(&0i64.to_le_bytes());
        mem[32..40].copy_from_slice(&0i64.to_le_bytes());
        let list = Type::list(Type::Str);
        assert_eq!(decode_exact_counted(&tp, &mem, &list, 16, 1000), Ok((Value::List(vec![Value::Str("hello".into()), Value::Str("hello".into())]), 3 * NODE_BYTES + 10)));
        assert_eq!(decode_exact(&tp, &mem, &list, 16, 100).unwrap_err(), "the value is larger than 100 bytes");
        mem[4] = 0xff;
        assert!(decode_exact(&tp, &mem, &Type::Str, 0, 1000).unwrap_err().contains("not valid UTF-8"));
        assert!(decode_exact(&tp, &mem, &Type::Str, 1 << 20, 1000).unwrap_err().starts_with("invalid Str"));
        assert!(decode_exact(&tp, &mem, &Type::option(Type::Int), 4, 1000).is_err(), "tag out of range");
    }
}
