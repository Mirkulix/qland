//! QAL values on the host side: decoding from and encoding into guest memory, and JSON form.

use qal_core::ast::Type;
use qal_core::tast::TProgram;
use serde::ser::{SerializeMap, Serializer};
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
        Type::Unit | Type::Var(_) => Value::Unit,
        Type::Fn(..) => Value::Str(format!("<fn #{ptr}>")),
        Type::Str => budget_str(mem, ptr, budget).unwrap_or_else(bad),
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
            let Type::Named(name) = ty else { return bad() };
            let mut fields = Vec::new();
            for (k, (f, ft)) in tp.record_fields(ty).iter().enumerate() {
                let Some(s) = slot_at(mem, ptr as usize + 8 * k) else { return bad() };
                fields.push((f.clone(), decode_at(tp, mem, ft, s, depth + 1, budget)));
            }
            Value::Record(name.clone(), fields)
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
        let vals = Values(vec![("n".into(), Value::Int(3))]);
        assert_eq!(serde_json::to_string(&vals).unwrap(), r#"{"n":3}"#);
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
}
