//! Writing host values into guest memory: any QAL value except functions (Int, Float, Bool, Unit,
//! Str, Bytes, List, Map, records, enums, generic instances, Option, Result) in the layout of `qal_wasm`, allocated with the module's own
//! `qal_alloc` (so the memory limit and `out_of_memory` apply exactly as for guest allocations).

use crate::host::HostState;
use crate::value::{self, Value};
use qal_core::ast::Type;
use qal_core::tast::TProgram;
use wasmtime::{AsContextMut, Memory, TypedFunc};

/// Values nested deeper than this are refused (host values are shallow; this guards the stack).
const MAX_DEPTH: usize = 200;

/// The guest's memory and allocator.
#[derive(Clone)]
pub struct Guest {
    pub mem: Memory,
    alloc: TypedFunc<i64, i32>,
}

/// Why a value could not be written: the guest trapped (out of memory: the failure is recorded in
/// the host state) or the value does not have the type.
#[derive(Debug)]
pub enum EncodeError {
    Trap(wasmtime::Error),
    Mismatch(String),
}

impl From<wasmtime::Error> for EncodeError {
    fn from(e: wasmtime::Error) -> EncodeError {
        EncodeError::Trap(e)
    }
}

impl Guest {
    pub fn new(mem: Memory, alloc: TypedFunc<i64, i32>) -> Guest {
        Guest { mem, alloc }
    }

    fn block(&self, cx: &mut impl AsContextMut<Data = HostState>, bytes: &[u8]) -> Result<i32, EncodeError> {
        let p = self.alloc.call(&mut *cx, bytes.len() as i64)?;
        self.mem.write(&mut *cx, p as usize, bytes).map_err(|e| EncodeError::Trap(wasmtime::Error::msg(e.to_string())))?;
        Ok(p)
    }

    /// Allocates `[len: u32][bytes]` (Str, Bytes).
    fn bytes(&self, cx: &mut impl AsContextMut<Data = HostState>, bytes: &[u8]) -> Result<i64, EncodeError> {
        let len = u32::try_from(bytes.len()).map_err(|_| EncodeError::Mismatch("value too large".into()))?;
        let mut b = Vec::with_capacity(4 + bytes.len());
        b.extend(len.to_le_bytes());
        b.extend(bytes);
        Ok(self.block(cx, &b)? as i64)
    }

    /// Builds a perfectly balanced map tree (a valid AVL tree) over sorted entry slots; returns
    /// (node pointer or 0, height). Node: `[size: u32][height: u32][key][value][left: u32][right: u32]`.
    fn map_tree(&self, cx: &mut impl AsContextMut<Data = HostState>, entries: &[(i64, i64)]) -> Result<(u32, u32), EncodeError> {
        if entries.is_empty() {
            return Ok((0, 0));
        }
        let mid = entries.len() / 2;
        let (l, hl) = self.map_tree(cx, &entries[..mid])?;
        let (r, hr) = self.map_tree(cx, &entries[mid + 1..])?;
        let height = hl.max(hr) + 1;
        let mut node = Vec::with_capacity(32);
        node.extend((entries.len() as u32).to_le_bytes());
        node.extend(height.to_le_bytes());
        node.extend(entries[mid].0.to_le_bytes());
        node.extend(entries[mid].1.to_le_bytes());
        node.extend(l.to_le_bytes());
        node.extend(r.to_le_bytes());
        Ok((self.block(cx, &node)? as u32, height))
    }

    /// Writes `v` (of type `ty`) and returns its slot: the scalar itself or the pointer.
    pub fn encode(&self, cx: &mut impl AsContextMut<Data = HostState>, tp: &TProgram, ty: &Type, v: &Value) -> Result<i64, EncodeError> {
        self.encode_at(cx, tp, ty, v, 0)
    }

    fn encode_at(&self, cx: &mut impl AsContextMut<Data = HostState>, tp: &TProgram, ty: &Type, v: &Value, depth: usize) -> Result<i64, EncodeError> {
        let too_large = || EncodeError::Mismatch("value too large".into());
        let mismatch = || EncodeError::Mismatch(format!("expected {ty}, got {}", serde_json::to_string(v).unwrap_or_default().chars().take(60).collect::<String>()));
        if depth > MAX_DEPTH {
            return Err(EncodeError::Mismatch("value nested too deeply".into()));
        }
        Ok(match (ty, v) {
            (Type::Int, Value::Int(n)) => *n,
            (Type::Float, Value::Float(x)) => x.to_bits() as i64,
            (Type::Bool, Value::Bool(b)) => *b as i64,
            (Type::Unit, Value::Unit) => 0,
            (Type::Str, Value::Str(s)) => self.bytes(cx, s.as_bytes())?,
            (Type::Bytes, Value::Bytes(b, 0)) => self.bytes(cx, b)?,
            (Type::List(elem), Value::List(xs)) => {
                let mut b = Vec::with_capacity(8 + 8 * xs.len());
                b.extend((u32::try_from(xs.len()).map_err(|_| too_large())?).to_le_bytes());
                b.extend([0; 4]);
                for x in xs {
                    b.extend(self.encode_at(cx, tp, elem, x, depth + 1)?.to_le_bytes());
                }
                self.block(cx, &b)? as i64
            }
            (Type::Map(kt, vt), Value::Map(entries)) => {
                let mut sorted: Vec<&(Value, Value)> = entries.iter().collect();
                sorted.sort_by(|a, b| value::compare(tp, kt, &a.0, &b.0));
                if sorted.windows(2).any(|w| value::compare(tp, kt, &w[0].0, &w[1].0).is_eq()) {
                    return Err(EncodeError::Mismatch(format!("a {ty} has a key twice")));
                }
                let mut slots = Vec::with_capacity(sorted.len());
                for (k, v) in sorted {
                    slots.push((self.encode_at(cx, tp, kt, k, depth + 1)?, self.encode_at(cx, tp, vt, v, depth + 1)?));
                }
                self.map_tree(cx, &slots)?.0 as i64
            }
            // Records and enums, also generic instances (`Type::App`).
            (_, Value::Record(_, fields)) if tp.is_record(ty) => {
                let decl = tp.record_fields(ty);
                if fields.len() != decl.len() {
                    return Err(EncodeError::Mismatch(format!("a {ty} has {} fields, got {}", decl.len(), fields.len())));
                }
                let mut b = Vec::with_capacity(8 * decl.len());
                for (name, ft) in &decl {
                    let x = fields.iter().find(|(n, _)| n == name).map(|(_, x)| x).ok_or_else(|| EncodeError::Mismatch(format!("{ty} without field `{name}`")))?;
                    b.extend(self.encode_at(cx, tp, ft, x, depth + 1)?.to_le_bytes());
                }
                // A record without fields still needs a distinct, valid address.
                if b.is_empty() {
                    b.extend([0; 8]);
                }
                self.block(cx, &b)? as i64
            }
            (_, Value::Variant(name, payload)) if !tp.variant_names(ty).is_empty() => {
                let tag = tp.variant_names(ty).iter().position(|n| n == name).ok_or_else(|| EncodeError::Mismatch(format!("{ty} has no variant `{name}`")))? as u32;
                let fields = tp.variant_fields(ty, tag);
                if fields.len() != payload.len() {
                    return Err(EncodeError::Mismatch(format!("`{name}` takes {} value(s), got {}", fields.len(), payload.len())));
                }
                let mut b = Vec::with_capacity(8 + 8 * fields.len());
                b.extend(tag.to_le_bytes());
                b.extend([0; 4]);
                for (ft, x) in fields.iter().zip(payload) {
                    b.extend(self.encode_at(cx, tp, ft, x, depth + 1)?.to_le_bytes());
                }
                self.block(cx, &b)? as i64
            }
            _ => return Err(mismatch()),
        })
    }
}
