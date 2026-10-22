//! Name rules and standard prefix helpers (shared between host and compiler).

/// Longest identifier, in characters (names become file names in `qal build`).
pub const MAX_NAME_LEN: usize = 128;

pub const KEYWORDS: &[&str] =
    &["fn", "type", "test", "let", "in", "if", "then", "else", "match", "req", "ens", "true", "false", "std"];

/// The prefix of the standard library's names (Phase 3 §4.1): `std.<module>.<name>` is one name
/// token, not a field access. The module segment is lowercase (`a-z`, `0-9`, `_`, starting with a
/// letter); the last segment is a function, test, type or variant name.
pub const STD_PREFIX: &str = "std.";

/// True if `name` is a qualified standard library name (`std.<module>.<name>`), checked as the
/// lexer would accept it as one token.
pub fn is_std_name(name: &str) -> bool {
    std_parts(name).is_some()
}

/// (module, last segment) of a qualified standard library name.
pub fn std_parts(name: &str) -> Option<(&str, &str)> {
    let rest = name.strip_prefix(STD_PREFIX)?;
    let (module, last) = rest.split_once('.')?;
    let module_ok = module.starts_with(|c: char| c.is_ascii_lowercase())
        && module.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        && !KEYWORDS.contains(&module);
    let last_ok = last.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        && last.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && last != "_"
        && !KEYWORDS.contains(&last);
    (module_ok && last_ok && name.len() <= MAX_NAME_LEN).then_some((module, last))
}

/// The segment of a name that decides its case: the last one (`std.json.Str` is uppercase).
pub fn last_segment(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}
