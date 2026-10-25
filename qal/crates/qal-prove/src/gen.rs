//! Values for executing functions: the smallest value of a type (to run a counterexample whose
//! unmodelled parameters the solver left open), random values of a size, and shrinking.
//!
//! Generation is deterministic: a SplitMix64 generator seeded per function.

use qal_core::ast::{Type, TypeBody};
use qal_core::tast::TProgram;
use qal_run::{compare_values, Value};

/// Deepest nesting of generated values (recursive types stop growing below their size anyway).
const MAX_DEPTH: usize = 24;
/// Most nodes of one generated value: nested records multiply (`R1 = { a: R0, b: R0, c: R0 }` …),
/// so even the smallest value of a small type can be exponentially large.
pub const MAX_VALUE_NODES: usize = 10_000;

/// SplitMix64 (deterministic, platform-independent).
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (n > 0).
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }

    /// Uniform in `lo..=hi`.
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        let span = (hi as i128 - lo as i128 + 1) as u128;
        (lo as i128 + (self.next() as u128 % span) as i128) as i64
    }

    fn chance(&mut self, one_in: u64) -> bool {
        self.below(one_in) == 0
    }
}

/// FNV-1a of `s` (seeds per function name).
pub fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// True if values of `ty` can be generated (no function inside, no unresolved type).
pub fn generatable(tp: &TProgram, ty: &Type) -> bool {
    fn walk(tp: &TProgram, ty: &Type, seen: &mut Vec<String>) -> bool {
        if seen.len() >= 256 {
            return false;
        }
        match ty {
            Type::Int | Type::Float | Type::Bool | Type::Unit | Type::Str | Type::Bytes => true,
            Type::List(x) | Type::Option(x) => walk(tp, x, seen),
            Type::Result(a, b) | Type::Map(a, b) => walk(tp, a, seen) && walk(tp, b, seen),
            Type::Named(_) | Type::App(..) => {
                let key = ty.to_string();
                if seen.contains(&key) {
                    return true;
                }
                seen.push(key);
                match tp.decl(ty) {
                    Some(TypeBody::Record(fs)) => fs.iter().all(|(_, t)| walk(tp, t, seen)),
                    // An enum without variants has no values.
                    Some(TypeBody::Enum(vs)) => !vs.is_empty() && vs.iter().all(|v| v.fields.iter().all(|t| walk(tp, t, seen))),
                    None => false,
                }
            }
            Type::Fn(..) | Type::Param(_) | Type::Var(_) => false,
        }
    }
    walk(tp, ty, &mut Vec::new())
}

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

const EDGE_INTS: [i64; 7] = [0, 1, -1, i64::MAX, i64::MIN, i64::MAX - 1, i64::MIN + 1];
const EDGE_FLOATS: [f64; 9] = [0.0, -0.0, 1.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MIN_POSITIVE, f64::MAX];
const CHARS: &[char] = &['a', 'b', 'c', 'x', 'y', 'z', 'A', 'Z', '0', '1', '9', ' ', ',', '-', '.', '\n', 'é', 'ß', '中', '🙂'];

/// A random value of `ty` of about `size` (collection lengths and Int magnitudes up to `size`).
/// None if `ty` has no value within `MAX_VALUE_NODES`.
pub fn random(tp: &TProgram, ty: &Type, size: u64, rng: &mut Rng) -> Option<Value> {
    random_with(tp, ty, size, rng, &[])
}

/// `random` that also draws the Int values `edges` (the bounds a function's `req` names, and their
/// neighbours), as often as the fixed edge values: a domain is tested at its limits.
pub fn random_with(tp: &TProgram, ty: &Type, size: u64, rng: &mut Rng, edges: &[i64]) -> Option<Value> {
    random_drawn(tp, ty, size, rng, edges, true)
}

/// `random_with` that draws the Int edges 0, 1 and -1 but not the ends of the 64-bit range: the
/// edges of a function that declares no domain. Every sum of two such ends overflows, and a
/// warning about it on `a + b` taught agents to ignore the warnings (pressure run 4).
pub fn random_ordinary(tp: &TProgram, ty: &Type, size: u64, rng: &mut Rng, edges: &[i64]) -> Option<Value> {
    random_drawn(tp, ty, size, rng, edges, false)
}

fn random_drawn(tp: &TProgram, ty: &Type, size: u64, rng: &mut Rng, edges: &[i64], extremes: bool) -> Option<Value> {
    random_at(tp, ty, size, rng, 0, &mut { MAX_VALUE_NODES }, edges, extremes)
}

#[allow(clippy::too_many_arguments)]
fn random_at(tp: &TProgram, ty: &Type, size: u64, rng: &mut Rng, depth: usize, left: &mut usize, edges: &[i64], ext: bool) -> Option<Value> {
    if depth > MAX_DEPTH {
        return smallest_at(tp, ty, 0, left);
    }
    if *left == 0 {
        return None;
    }
    *left -= 1;
    let s = size as i64;
    Some(match ty {
        Type::Int if !edges.is_empty() && rng.chance(5) => Value::Int(edges[rng.below(edges.len() as u64) as usize]),
        // Without `ext`, only 0, 1 and -1 (the first three).
        Type::Int if rng.chance(10) => Value::Int(EDGE_INTS[rng.below(if ext { EDGE_INTS.len() } else { 3 } as u64) as usize]),
        Type::Int => Value::Int(rng.range(-s, s)),
        Type::Float if rng.chance(10) => Value::Float(EDGE_FLOATS[rng.below(EDGE_FLOATS.len() as u64) as usize]),
        Type::Float => Value::Float(rng.range(-s * 100, s * 100) as f64 / 100.0),
        Type::Bool => Value::Bool(rng.chance(2)),
        Type::Unit => Value::Unit,
        Type::Str => {
            let n = rng.below(size + 1);
            Value::Str((0..n).map(|_| CHARS[rng.below(CHARS.len() as u64) as usize]).collect())
        }
        Type::Bytes => {
            let n = rng.below(size + 1);
            Value::Bytes((0..n).map(|_| rng.below(256) as u8).collect(), 0)
        }
        Type::List(x) => {
            let n = rng.below(size + 1);
            Value::List((0..n).map(|_| random_at(tp, x, size / 2, rng, depth + 1, left, edges, ext)).collect::<Option<_>>()?)
        }
        Type::Map(k, v) => {
            let n = rng.below(size / 2 + 1);
            let mut entries: Vec<(Value, Value)> = Vec::new();
            for _ in 0..n {
                let key = random_at(tp, k, size / 2, rng, depth + 1, left, edges, ext)?;
                let val = random_at(tp, v, size / 2, rng, depth + 1, left, edges, ext)?;
                if !entries.iter().any(|(e, _)| compare_values(tp, k, e, &key).is_eq()) {
                    entries.push((key, val));
                }
            }
            Value::Map(entries)
        }
        _ if tp.is_record(ty) => {
            let mut fields = Vec::new();
            for (f, ft) in tp.record_fields(ty) {
                fields.push((f, random_at(tp, &ft, size, rng, depth + 1, left, edges, ext)?));
            }
            Value::Record(ty.to_string(), fields)
        }
        _ => {
            let names = tp.variant_names(ty);
            if names.is_empty() {
                return None;
            }
            // At size 0 the smallest variants only, so recursive types terminate.
            let tag = if size == 0 {
                return smallest_at(tp, ty, 0, left);
            } else {
                rng.below(names.len() as u64) as u32
            };
            let fields = tp.variant_fields(ty, tag);
            let sub = size / (fields.len() as u64 + 1);
            let payload = fields.iter().map(|ft| random_at(tp, ft, sub, rng, depth + 1, left, edges, ext)).collect::<Option<Vec<_>>>()?;
            Value::Variant(names.get(tag as usize)?.clone(), payload)
        }
    })
}

/// Candidates tried per nested value (list element, field, payload) in one shrinking step.
const NESTED: usize = 16;

/// Simpler values than `v` (of type `ty`), most promising first.
pub fn shrink(tp: &TProgram, ty: &Type, v: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    match (ty, v) {
        (_, Value::Int(x)) => {
            for c in [0, x / 2, x - x.signum()] {
                if c != *x && !out.contains(&Value::Int(c)) {
                    out.push(Value::Int(c));
                }
            }
            if *x < 0 && *x != i64::MIN {
                out.push(Value::Int(-x));
            }
        }
        (_, Value::Bool(true)) => out.push(Value::Bool(false)),
        (_, Value::Float(x)) => {
            for c in [0.0, x.trunc(), x / 2.0] {
                if c.to_bits() != x.to_bits() && c.is_finite() {
                    out.push(Value::Float(c));
                }
            }
        }
        (_, Value::Str(s)) => {
            let chars: Vec<char> = s.chars().collect();
            for part in shrink_seq(&chars) {
                out.push(Value::Str(part.into_iter().collect()));
            }
            for (i, c) in chars.iter().enumerate().take(8) {
                if *c != 'a' {
                    let mut t = chars.clone();
                    t[i] = 'a';
                    out.push(Value::Str(t.into_iter().collect()));
                }
            }
        }
        (_, Value::Bytes(b, _)) => {
            for part in shrink_seq(b) {
                out.push(Value::Bytes(part, 0));
            }
        }
        (Type::List(t), Value::List(xs)) => {
            for part in shrink_seq(xs) {
                out.push(Value::List(part));
            }
            for (i, x) in xs.iter().enumerate().take(8) {
                for c in shrink(tp, t, x).into_iter().take(NESTED) {
                    let mut ys = xs.clone();
                    ys[i] = c;
                    out.push(Value::List(ys));
                }
            }
        }
        (Type::Map(kt, vt), Value::Map(es)) => {
            for part in shrink_seq(es) {
                out.push(Value::Map(part));
            }
            // Simpler keys, unless one equals another key of the map.
            for (i, (k, _)) in es.iter().enumerate().take(8) {
                for c in shrink(tp, kt, k).into_iter().take(NESTED) {
                    if es.iter().enumerate().any(|(j, (e, _))| j != i && compare_values(tp, kt, e, &c).is_eq()) {
                        continue;
                    }
                    let mut ys = es.clone();
                    ys[i].0 = c;
                    out.push(Value::Map(ys));
                }
            }
            for (i, (_, x)) in es.iter().enumerate().take(8) {
                for c in shrink(tp, vt, x).into_iter().take(NESTED) {
                    let mut ys = es.clone();
                    ys[i].1 = c;
                    out.push(Value::Map(ys));
                }
            }
        }
        (_, Value::Record(name, fields)) => {
            let decl = tp.record_fields(ty);
            for (i, (f, x)) in fields.iter().enumerate() {
                let Some((_, ft)) = decl.iter().find(|(g, _)| g == f) else { continue };
                for c in shrink(tp, ft, x).into_iter().take(NESTED) {
                    let mut fs = fields.clone();
                    fs[i].1 = c;
                    out.push(Value::Record(name.clone(), fs));
                }
            }
        }
        (_, Value::Variant(name, payload)) => {
            let names = tp.variant_names(ty);
            let tag = names.iter().position(|n| n == name).unwrap_or(0) as u32;
            let fields = tp.variant_fields(ty, tag);
            // A smaller variant first, then a payload value of the same type (a subtree), then
            // smaller payloads.
            if let Some(Value::Variant(small, sp)) = smallest(tp, ty) {
                if (small != *name || sp != *payload) && !payload.is_empty() {
                    out.push(Value::Variant(small, sp));
                }
            }
            for (x, ft) in payload.iter().zip(&fields) {
                if ft == ty {
                    out.push(x.clone());
                }
            }
            for (i, (x, ft)) in payload.iter().zip(&fields).enumerate() {
                for c in shrink(tp, ft, x).into_iter().take(NESTED) {
                    let mut ps = payload.clone();
                    ps[i] = c;
                    out.push(Value::Variant(name.clone(), ps));
                }
            }
        }
        _ => {}
    }
    out
}

/// Shorter versions of a sequence: empty, each half, without one element (at most 8 positions).
fn shrink_seq<T: Clone>(xs: &[T]) -> Vec<Vec<T>> {
    let n = xs.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out = vec![Vec::new()];
    if n > 1 {
        out.push(xs[..n / 2].to_vec());
        out.push(xs[n / 2..].to_vec());
    }
    for i in (0..n).take(8) {
        let mut ys = xs.to_vec();
        ys.remove(i);
        if !ys.is_empty() && n >= 2 {
            out.push(ys);
        }
    }
    out
}
