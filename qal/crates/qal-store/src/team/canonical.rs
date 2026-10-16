//! Canonical JSON of signed records (Phase 4 spec §9.1) and the root of a name space (§3).

use qal_core::canon::Hash;
use serde_json::Value;
use std::collections::BTreeMap;

/// Largest integer magnitude a record may contain (exactly representable in every JSON reader).
pub const MAX_SAFE_INT: u64 = (1 << 53) - 1;

/// The canonical text of `v`: keys sorted by bytes, no whitespace, integers only (|x| < 2^53),
/// strings escaped as `serde_json` does. `serde_json::Map` is a `BTreeMap` (the feature
/// `preserve_order` is not enabled in the workspace; a test pins this), so its compact output has
/// sorted keys; numbers are checked here.
pub fn to_string(v: &Value) -> Result<String, String> {
    check(v)?;
    serde_json::to_string(v).map_err(|e| format!("cannot serialize a record: {e}"))
}

fn check(v: &Value) -> Result<(), String> {
    match v {
        Value::Null | Value::Bool(_) | Value::String(_) => Ok(()),
        Value::Number(n) => {
            let ok = n.as_u64().is_some_and(|x| x <= MAX_SAFE_INT) || n.as_i64().is_some_and(|x| x.unsigned_abs() <= MAX_SAFE_INT);
            if ok {
                Ok(())
            } else {
                Err(format!("a record may only contain integers below 2^53, not `{n}`"))
            }
        }
        Value::Array(xs) => xs.iter().try_for_each(check),
        Value::Object(m) => m.values().try_for_each(check),
    }
}

/// The root of a name space: SHA-256 over the byte-sorted lines `name 0x00 hash-hex 0x0a`.
/// Equal roots mean equal name states.
pub fn root(names: &BTreeMap<String, String>) -> String {
    // `String`'s order is byte order, so the map iterates sorted.
    root_iter(names.iter().map(|(n, h)| (n.as_str(), h.as_str())))
}

/// [`root`] of (name, hash) pairs given in byte order of the names.
pub fn root_iter<'a>(names: impl Iterator<Item = (&'a str, &'a str)>) -> String {
    let mut bytes = Vec::new();
    for (n, h) in names {
        bytes.extend_from_slice(n.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(h.as_bytes());
        bytes.push(b'\n');
    }
    Hash::of(&bytes).to_hex()
}

/// Lowercase hex of `bytes`.
pub fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(2 * bytes.len());
    hex_into(bytes, &mut s);
    s
}

/// Appends the lowercase hex of `bytes` to `out` (for secrets: into a buffer the caller zeroizes,
/// reserved beforehand so it never reallocates).
pub fn hex_into(bytes: &[u8], out: &mut String) {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    for b in bytes {
        out.push(DIGITS[usize::from(b >> 4)] as char);
        out.push(DIGITS[usize::from(b & 15)] as char);
    }
}

/// Decodes exactly `2 * N` lowercase hex digits into `out`; false (and `out` unspecified) otherwise.
pub fn unhex_into<const N: usize>(s: &str, out: &mut [u8; N]) -> bool {
    let b = s.as_bytes();
    if b.len() != 2 * N {
        return false;
    }
    let digit = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    };
    for (i, o) in out.iter_mut().enumerate() {
        match (digit(b[2 * i]), digit(b[2 * i + 1])) {
            (Some(h), Some(l)) => *o = h * 16 + l,
            _ => return false,
        }
    }
    true
}

/// The bytes of exactly `2 * N` lowercase hex digits (only lowercase: the stored form is unique).
pub fn unhex<const N: usize>(s: &str) -> Option<[u8; N]> {
    let mut out = [0u8; N];
    unhex_into(s, &mut out).then_some(out)
}

/// True for 64 lowercase hex digits (a hash, a key or a seed).
pub fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
