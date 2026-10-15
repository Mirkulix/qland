//! Values for executing functions: the smallest value of a type, to run a counterexample whose
//! unmodelled parameters the solver left open.

use qal_core::tast::TProgram;
use qal_core::ast::Type;
use qal_run::Value;

/// Deepest nesting of generated values (recursive types stop growing below their size anyway).
const MAX_DEPTH: usize = 24;
/// Most nodes of one generated value: nested records multiply (`R1 = { a: R0, b: R0, c: R0 }` …),
/// so even the smallest value of a small type can be exponentially large.
pub const MAX_VALUE_NODES: usize = 10_000;

/// The variant tags of `ty` ordered by how small their values are (fewest fields first).
fn variants_by_size(tp: &TProgram, ty: &Type) -> Vec<u32> {
    let n = tp.variant_names(ty).len() as u32;
    let mut tags: Vec<u32> = (0..n).collect();
    tags.sort_by_key(|t| tp.variant_fields(ty, *t).len());
    tags
}

/// The smallest value of `ty`: 0, false, "", empty collections, None, the variant with the fewest
/// fields. None if there is no finite one (functions, enums without a terminating variant).
pub fn smallest(tp: &TProgram, ty: &Type) -> Option<Value> {
    smallest_at(tp, ty, 0, &mut { MAX_VALUE_NODES })
}

fn smallest_at(tp: &TProgram, ty: &Type, depth: usize, left: &mut usize) -> Option<Value> {
    if depth > MAX_DEPTH || *left == 0 {
        return None;
    }
    *left -= 1;
    Some(match ty {
        Type::Int => Value::Int(0),
        Type::Float => Value::Float(0.0),
        Type::Bool => Value::Bool(false),
        Type::Unit => Value::Unit,
        Type::Str => Value::Str(String::new()),
        Type::Bytes => Value::Bytes(Vec::new(), 0),
        Type::List(_) => Value::List(Vec::new()),
        Type::Map(..) => Value::Map(Vec::new()),
        _ if tp.is_record(ty) => {
            let mut fields = Vec::new();
            for (f, ft) in tp.record_fields(ty) {
                fields.push((f, smallest_at(tp, &ft, depth + 1, left)?));
            }
            Value::Record(ty.to_string(), fields)
        }
        _ => {
            let names = tp.variant_names(ty);
            variants_by_size(tp, ty).into_iter().find_map(|tag| {
                let payload: Option<Vec<Value>> = tp.variant_fields(ty, tag).iter().map(|ft| smallest_at(tp, ft, depth + 1, left)).collect();
                Some(Value::Variant(names.get(tag as usize)?.clone(), payload?))
            })?
        }
    })
}
