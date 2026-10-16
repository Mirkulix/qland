//! Structured diagnostics: every compiler error is data, not prose.

use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Diagnostic {
    pub code: &'static str,
    pub kind: &'static str,
    pub def: String,
    pub path: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub found: Option<String>,
    pub hints: Vec<String>,
    /// For `signature_change`: the stored definitions that would break (to include in the call).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub dependents: Vec<String>,
    /// For refuted or unproven contracts and failed properties: the input (parameter name -> value).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub counterexample: Option<serde_json::Value>,
    /// Structured details of team diagnostics (e.g. E0901: the lease `{name, change, member, expires_at}`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<serde_json::Value>,
}

impl Diagnostic {
    pub fn new(code: &'static str, kind: &'static str, def: &str, path: &str, message: impl Into<String>) -> Self {
        Diagnostic {
            code,
            kind,
            def: def.to_string(),
            path: path.to_string(),
            message: message.into(),
            expected: None,
            found: None,
            hints: Vec::new(),
            dependents: Vec::new(),
            counterexample: None,
            detail: None,
        }
    }

    pub fn expected_found(mut self, expected: impl Into<String>, found: impl Into<String>) -> Self {
        self.expected = Some(expected.into());
        self.found = Some(found.into());
        self
    }

    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        self.hints.push(hint.into());
        self
    }

    pub fn parse(line: usize, col: usize, message: impl Into<String>) -> Self {
        Diagnostic::new("E0100", "parse", "", &format!("{line}:{col}"), message)
    }

    pub fn unknown_name(def: &str, path: &str, name: &str) -> Self {
        Diagnostic::new("E0201", "unknown_name", def, path, format!("unknown name `{name}`"))
    }

    pub fn duplicate_name(def: &str, path: &str, name: &str) -> Self {
        Diagnostic::new("E0202", "duplicate_name", def, path, format!("`{name}` is defined more than once"))
    }

    /// A definition named like a built-in function. A store written before that built-in existed can
    /// hold one: it must be renamed (`qal_rename` accepts the store while others remain).
    pub fn builtin_name(name: &str) -> Self {
        Diagnostic::new("E0202", "duplicate_name", name, "", format!("`{name}` is the name of a built-in function")).hint(format!(
            "choose another name; a `{name}` stored before this built-in existed must be renamed first: qal_rename it, e.g. to `my_{name}`"
        ))
    }

    /// Whether this is a [`Diagnostic::builtin_name`] diagnostic.
    pub fn is_builtin_name(&self) -> bool {
        self.code == "E0202" && self.message.ends_with("` is the name of a built-in function")
    }

    pub fn type_mismatch(def: &str, path: &str, expected: &str, found: &str) -> Self {
        Diagnostic::new("E0203", "type_mismatch", def, path, format!("expected {expected}, found {found}"))
            .expected_found(expected, found)
    }

    pub fn arity(def: &str, path: &str, callee: &str, expected: usize, found: usize) -> Self {
        Diagnostic::new(
            "E0204",
            "type_mismatch",
            def,
            path,
            format!("`{callee}` takes {expected} argument(s), {found} given"),
        )
        .expected_found(expected.to_string(), found.to_string())
    }

    pub fn effect_violation(def: &str, path: &str, callee: &str, missing: &[String]) -> Self {
        Diagnostic::new(
            "E0301",
            "effect_violation",
            def,
            path,
            format!("call to `{callee}` needs effect(s) {} not declared here", missing.join(", ")),
        )
        .hint(format!("add `!{}` to the signature of `{def}`", missing.join(" !")))
    }

    pub fn non_exhaustive(def: &str, path: &str, missing: &str) -> Self {
        Diagnostic::new("E0205", "non_exhaustive_match", def, path, format!("match does not cover `{missing}`"))
            .hint(format!("add an arm `{missing} => ...` or a final `_ => ...` arm"))
    }

    pub fn type_too_deep(def: &str, path: &str, max: u32) -> Self {
        Diagnostic::new("E0207", "type_too_deep", def, path, format!("inferred type is nested too deeply (max {max} levels)"))
            .hint("declare a record or enum type for the nested data instead of nesting List/Option/Result this deeply")
    }

    pub fn type_too_large(def: &str, path: &str, max: u32) -> Self {
        Diagnostic::new("E0208", "type_too_large", def, path, format!("inferred type is too large (max {max} nodes)"))
            .hint("annotate the values with `let x: T = ...` or declare a record or enum type; shared `Ok`/`Err`/`None` values can make inferred types grow exponentially")
    }

    /// Replacing `def` changes its signature while stored definitions still depend on it.
    pub fn signature_change(def: &str, old: &str, new: &str, dependents: &[String]) -> Self {
        let mut d = Diagnostic::new(
            "E0401",
            "signature_change",
            def,
            "",
            format!("changing the signature of `{def}` breaks {} dependent definition(s): {}", dependents.len(), dependents.join(", ")),
        )
        .expected_found(old, new)
        .hint(format!("define updated versions of {} in the same call, or keep the signature `{old}`", dependents.join(", ")));
        d.dependents = dependents.to_vec();
        d
    }

    /// `def` refers to several names of one definition (`names`, all hashing to `short_hash`); a
    /// stored definition can name each referenced definition only once.
    pub fn alias_reference(def: &str, path: &str, names: &[String], short_hash: &str, referrer_in_input: bool) -> Self {
        let quoted: Vec<String> = names.iter().map(|n| format!("`{n}`")).collect();
        let list = match quoted.split_last() {
            Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
            _ => quoted.join(""),
        };
        let hint = if referrer_in_input {
            format!("refer to only one of {} (they are identical)", quoted.join(", "))
        } else {
            format!("refer to one of them, or make them differ (redefine `{def}` in the same call)")
        };
        Diagnostic::new(
            "E0404",
            "alias_reference",
            def,
            path,
            format!("`{def}` refers to {list}, which are the same definition (#{short_hash}); a stored definition can name each referenced definition only once."),
        )
        .hint(hint)
    }

    /// Mutually recursive definitions that are interchangeable (the same definition under several
    /// names) cannot get name-independent hashes.
    pub fn symmetric_group(names: &[String]) -> Self {
        let list = names.iter().map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", ");
        Diagnostic::new("E0209", "duplicate_name", names.first().map_or("", |n| n.as_str()), "", format!("the recursive definitions {list} are interchangeable: they are the same definition under different names"))
            .hint("define the logic once (one function that calls itself), or make the definitions differ")
    }

    /// A recursive group too large and uniform for the canonical labelling search.
    pub fn group_too_large(names: &[String]) -> Self {
        Diagnostic::new(
            "E0210",
            "group_too_large",
            names.first().map_or("", |n| n.as_str()),
            "",
            format!("the {} mutually recursive definitions starting with `{}` are too many and too alike to hash", names.len(), names.first().map_or("", |n| n.as_str())),
        )
        .hint("split the recursion: let most of the definitions call helpers outside the cycle, or give them distinguishing bodies")
    }

    /// A stored definition refers to a hash that has no name (the store is inconsistent).
    pub fn missing_dep(def: &str, hash: &str) -> Self {
        Diagnostic::new("E0402", "missing_dep", def, "", format!("`{def}` refers to definition #{hash}, which has no name in the store"))
    }

    /// A stored name would change what kind of definition it is (e.g. a function redefined as a test).
    pub fn kind_change(def: &str, old: &str, new: &str) -> Self {
        Diagnostic::new("E0405", "kind_change", def, "", format!("`{def}` is a {old} in the store and cannot be redefined as a {new}"))
            .expected_found(old, new)
            .hint(format!("give the {new} another name, or rename the stored {old} first"))
    }

    /// Deleting `def` while stored definitions still refer to it.
    pub fn has_dependents(def: &str, dependents: &[String]) -> Self {
        let mut d = Diagnostic::new("E0406", "has_dependents", def, "", format!("`{def}` cannot be deleted: {} stored definition(s) refer to it: {}", dependents.len(), dependents.join(", ")))
            .hint(format!("delete or redefine {} first (without `{def}`), or delete them in the same call", dependents.join(", ")));
        d.dependents = dependents.to_vec();
        d
    }

    /// A `query` signature pattern that cannot match anything (unbalanced brackets, empty, too long).
    pub fn bad_pattern(pattern: &str, reason: &str) -> Self {
        Diagnostic::new("E0102", "bad_pattern", "", "", format!("invalid signature pattern `{pattern}`: {reason}"))
            .hint("write a signature with `_` for any one type, e.g. `fn(Str, _) -> Int` or `fn(_) -> Option[_]`")
    }

    /// The store itself failed: SQLite error, unreadable or corrupt file, hash mismatch.
    pub fn store_error(message: impl Into<String>) -> Self {
        Diagnostic::new("E0601", "store_error", "", "", message)
    }

    /// A type parameter of a function that its parameter and result types do not mention: its type
    /// argument could never be inferred.
    pub fn unused_type_param(def: &str, path: &str, param: &str) -> Self {
        Diagnostic::new("E0211", "unused_type_parameter", def, path, format!("type parameter `{param}` does not occur in the parameter or result types"))
            .hint(format!("remove `{param}` or use it in a parameter type"))
    }

    /// A generic type refers to itself (or its recursive group) with arguments other than plain
    /// type parameters, e.g. `type Nest[T] = Nil | Cons(T, Nest[List[T]])`: it would need infinitely many instances.
    pub fn irregular_recursion(def: &str, path: &str, used: &str) -> Self {
        Diagnostic::new("E0212", "irregular_recursion", def, path, format!("recursive use `{used}` must pass plain type parameters"))
            .hint("inside its own definition, a recursive generic type can only be applied to type parameters, e.g. `Tree[T]`")
    }

    /// A generic function calls itself (or its recursive group) with type arguments other than
    /// plain type parameters: monomorphisation would not terminate.
    pub fn polymorphic_recursion(def: &str, path: &str, callee: &str, targs: &str) -> Self {
        Diagnostic::new("E0213", "polymorphic_recursion", def, path, format!("recursive call of `{callee}` at type [{targs}]"))
            .hint("recursive calls of a generic function must use its own type parameters (e.g. `f[T]` calls `f[T]`)")
    }

    /// Monomorphisation produced too many instances or too large type arguments.
    pub fn instantiation_limit(def: &str, path: &str, instance: &str, max: usize) -> Self {
        Diagnostic::new("E0214", "instantiation_limit", def, path, format!("instantiating `{instance}` exceeds the limit of {max} generic instances, the type size or depth limit or the size limit of all instances"))
            .hint("deeply nested generic calls multiply instances; use fewer distinct type arguments")
    }

    /// A user definition takes a reserved name: a prelude type or variant (`HttpRequest`, `SqlInt`, ...).
    pub fn reserved_name(def: &str, path: &str, name: &str) -> Self {
        Diagnostic::new("E0240", "reserved_name", def, path, format!("`{name}` is a built-in prelude name"))
            .hint("prelude types (HttpHeader, HttpRequest, HttpResponse, HttpCall, NetError, SqlValue, DbRows, DbError, Accum) and their variants exist in every program; choose another name")
    }

    /// `http`/`db_exec`/`db_query` in a legacy codebase whose own definitions take prelude names
    /// (stored before Phase 3): the prelude types those built-ins use are not available.
    pub fn prelude_shadowed(def: &str, path: &str, callee: &str) -> Self {
        Diagnostic::new("E0240", "reserved_name", def, path, format!("`{callee}` needs the prelude types, but this codebase defines prelude names itself"))
            .hint("rename the stored types (qal_rename) or redefine the enums whose names or variants are prelude names (HttpHeader, HttpRequest, HttpResponse, HttpCall, NetError, SqlValue, DbRows, DbError, Accum and their variants)")
    }

    /// A project's own definition (or rename) takes a `std.` name: the standard library namespace
    /// is read-only outside the std project; its names come only from the std package.
    pub fn std_reserved(def: &str, path: &str, name: &str) -> Self {
        Diagnostic::new("E0240", "reserved_name", def, path, format!("`{name}` is in the read-only standard library namespace `std.`"))
            .hint("`std.` names come from the standard library package (imported automatically); choose a name without the `std.` prefix")
    }

    /// A definition (or rename) in the std project (`meta.role = 'std'`) without a `std.` name, or a
    /// variant outside its type's module.
    pub fn std_only(def: &str, path: &str, name: &str, module: Option<&str>) -> Self {
        let (message, hint) = match module {
            Some(m) => (format!("variant `{name}` is not in its type's module `std.{m}.`"), format!("name it `std.{m}.<Variant>`")),
            None => (format!("`{name}` is not a `std.` name; the std project defines only standard library names"), format!("name it `std.<module>.{name}` (e.g. `std.json.{name}`)")),
        };
        Diagnostic::new("E0240", "reserved_name", def, path, message).hint(hint)
    }

    /// An import would change the signature of an existing `std.` definition: the standard
    /// library is additive in its signatures (spec Phase 3 §4.3).
    pub fn std_signature_change(def: &str, old: &str, new: &str) -> Self {
        Diagnostic::new("E0244", "std_signature_change", def, "", format!("the package changes the signature of the standard library definition `{def}`"))
            .expected_found(old, new)
            .hint(format!("std signatures never change: give the new version a new name (e.g. `{def}2`) and keep `{def}` as it is"))
    }

    /// Deleting a name of the released standard library in the std project (std is additive).
    pub fn std_released_delete(def: &str, release: &str) -> Self {
        Diagnostic::new("E0244", "std_signature_change", def, "", format!("`{def}` is in the released standard library (std {}); deleting it breaks the projects that use it", release.chars().take(12).collect::<String>()))
            .hint("std is additive: keep it (it may be replaced with the same signature); a deliberate break needs `qal delete --breaking` on the command line")
    }

    /// Renaming a name of the released standard library in the std project (std is additive).
    pub fn std_released_rename(def: &str, new: &str, release: &str) -> Self {
        Diagnostic::new("E0244", "std_signature_change", def, "", format!("`{def}` is in the released standard library (std {}); renaming it to `{new}` removes it from the projects that use it", release.chars().take(12).collect::<String>()))
            .hint("std is additive: keep the name; a deliberate break needs `qal rename --breaking` on the command line")
    }

    /// The `sql` argument of `db_exec` / `db_query` is not a string literal.
    pub fn sql_not_literal(def: &str, path: &str, callee: &str) -> Self {
        Diagnostic::new("E0241", "sql_not_literal", def, path, format!("the SQL text of `{callee}` must be a string literal"))
            .hint("values belong in `params` as `?` placeholders, e.g. db_query(\"SELECT * FROM t WHERE id = ?\", [SqlInt(id)]); for variants (e.g. ORDER BY) write one literal per variant and choose with `match`")
    }

    /// The SQL literal of `db_exec` / `db_query` is refused by the static check (`qal_core::sql`).
    pub fn sql_invalid(def: &str, path: &str, reason: &str) -> Self {
        Diagnostic::new("E0242", "sql_invalid", def, path, format!("invalid SQL literal: {reason}"))
            .hint("one statement, positional `?` parameters only; ATTACH, DETACH, PRAGMA, VACUUM and transaction statements are not allowed (one transaction per execution is automatic)")
    }

    pub fn missing_field(def: &str, path: &str, record: &str, field: &str) -> Self {
        Diagnostic::new("E0206", "type_mismatch", def, path, format!("`{record}` needs field `{field}`"))
            .hint(format!("add `{field}: ...` to the `{record} {{ ... }}` literal"))
    }

    /// `--compiler both`: the backends' outputs differ (self-hosting spec §11); `level` is text,
    /// hash, diagnostics, typed form or WASM, `detail` the first difference.
    pub fn compiler_divergence(level: &str, detail: &str) -> Self {
        Diagnostic::new("E0950", "compiler_divergence", "", "", format!("the Rust and QAL compilers disagree on the {level}: {detail}"))
            .hint("this is a compiler bug, not a problem of the program; report it with the program")
    }

    /// The QAL compiler trapped, exceeded a limit or reported a broken invariant (`INTERNAL`).
    pub fn compiler_failed(stage: &str, op: &str, item: &str, error: &str) -> Self {
        Diagnostic::new("E0951", "compiler_failed", "", "", format!("the compiler ({stage}) failed on {op}, work item {item}: {error}"))
            .hint("this is a compiler bug or limit, not a problem of the program; the host keeps running")
    }

    /// The compiler's response violates `QALR` (strict host-side decoding).
    pub fn bad_compiler_response(error: &str) -> Self {
        Diagnostic::new("E0952", "bad_compiler_response", "", "", format!("the compiler's response is malformed: {error}"))
    }

    /// The embedded seed does not match `qalc.wasm.sha256`, or is not a fixpoint.
    pub fn seed_mismatch(detail: &str) -> Self {
        Diagnostic::new("E0953", "seed_mismatch", "", "", format!("the compiler seed does not verify: {detail}"))
    }

    /// The fork/join protocol exceeded its limits (rounds, items, bytes or time; spec §4.3).
    pub fn protocol_limit(detail: &str) -> Self {
        Diagnostic::new("E0954", "protocol_limit", "", "", format!("the compiler exceeded a protocol limit: {detail}"))
    }
}

/// Every `(code, kind)` pair a diagnostic can carry. Diagnostics that cross the compiler boundary
/// (`QALR`, self-hosting spec §4.2) are decoded against this list: `code` and `kind` stay
/// `&'static str` (no `Cow` needed, and the ~100 places that compare them keep working), and a
/// compiler cannot invent codes — an unknown pair is a malformed response (E0952).
pub const KNOWN: &[(&str, &str)] = &[
    ("E0100", "parse"),
    ("E0102", "bad_pattern"),
    ("E0201", "unknown_name"),
    ("E0202", "duplicate_name"),
    ("E0203", "type_mismatch"),
    ("E0204", "type_mismatch"),
    ("E0205", "non_exhaustive_match"),
    ("E0206", "type_mismatch"),
    ("E0207", "type_too_deep"),
    ("E0208", "type_too_large"),
    ("E0209", "duplicate_name"),
    ("E0210", "group_too_large"),
    ("E0211", "unused_type_parameter"),
    ("E0212", "irregular_recursion"),
    ("E0213", "polymorphic_recursion"),
    ("E0214", "instantiation_limit"),
    ("E0240", "reserved_name"),
    ("E0241", "sql_not_literal"),
    ("E0242", "sql_invalid"),
    ("E0243", "package_mismatch"),
    ("E0244", "std_signature_change"),
    ("E0301", "effect_violation"),
    ("E0401", "signature_change"),
    ("E0402", "missing_dep"),
    ("E0404", "alias_reference"),
    ("E0405", "kind_change"),
    ("E0406", "has_dependents"),
    ("E0601", "store_error"),
    ("E0701", "bad_arguments"),
    ("E0702", "result_too_large"),
    ("E0801", "contract_refuted"),
    ("E0802", "contract_unproven"),
    // Phase 4 (team procedure, signed records).
    ("E0900", "not_a_team"),
    ("E0902", "no_session"),
    ("E0903", "role_forbidden"),
    ("E0908", "change_state"),
    ("E0914", "team_limit"),
    ("E0915", "grant_forbidden"),
    ("E0920", "key_unusable"),
    ("E0921", "signature_invalid"),
    ("E0922", "chain_broken"),
    ("E0923", "replay_mismatch"),
    ("E0924", "unauthorized_record"),
    ("E0925", "untrusted_key"),
    ("E0926", "key_dir_in_project"),
    // Phase 5 (self-hosting).
    ("E0950", "compiler_divergence"),
    ("E0951", "compiler_failed"),
    ("E0952", "bad_compiler_response"),
    ("E0953", "seed_mismatch"),
    ("E0954", "protocol_limit"),
];

/// The static `(code, kind)` of a known pair (see [`KNOWN`]).
pub fn known(code: &str, kind: &str) -> Option<(&'static str, &'static str)> {
    KNOWN.iter().copied().find(|(c, k)| *c == code && *k == kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_diagnostic_constructed_anywhere_is_known() {
        // Scans the workspace sources for calls of Diagnostic::new with a literal code and kind, so a new code cannot
        // be forgotten in KNOWN (and so become undecodable from the self-hosted compiler).
        fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for e in std::fs::read_dir(dir).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() && p.file_name().is_some_and(|n| n != "target") {
                    walk(&p, out);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    out.push(p);
                }
            }
        }
        let mut files = Vec::new();
        walk(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".."), &mut files);
        let mut found = 0;
        for f in files {
            let src = std::fs::read_to_string(&f).unwrap();
            for (at, _) in src.match_indices("Diagnostic::new(") {
                let rest = &src[at + 16..];
                let lit = |s: &str| -> Option<(String, usize)> {
                    let s0 = s.trim_start();
                    let skipped = s.len() - s0.len();
                    let body = s0.strip_prefix('"')?;
                    let end = body.find('"')?;
                    Some((body[..end].to_string(), skipped + end + 2))
                };
                let Some((code, n)) = lit(rest) else { continue };
                let Some(after) = rest[n..].trim_start().strip_prefix(',') else { continue };
                let Some((kind, _)) = lit(after) else { continue };
                assert!(known(&code, &kind).is_some(), "{}: ({code}, {kind}) is not in diag::KNOWN", f.display());
                found += 1;
            }
        }
        assert!(found >= 40, "{found}");
        let mut sorted = KNOWN.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), KNOWN.len(), "duplicates in KNOWN");
        assert_eq!(known("E0203", "type_mismatch"), Some(("E0203", "type_mismatch")));
        assert_eq!((known("E0203", "parse"), known("E9999", "x")), (None, None));
        let d = Diagnostic::protocol_limit("9 rounds");
        assert_eq!((d.code, d.kind), ("E0954", "protocol_limit"));
    }

    #[test]
    fn alias_reference_hints_depend_on_where_the_referrer_is() {
        let names = ["mk".to_string(), "mk2".to_string()];
        let d = Diagnostic::alias_reference("u", "", &names, "abc123", true);
        assert_eq!(d.hints, ["refer to only one of `mk`, `mk2` (they are identical)"]);
        assert_eq!(d.message, "`u` refers to `mk` and `mk2`, which are the same definition (#abc123); a stored definition can name each referenced definition only once.");
        let d = Diagnostic::alias_reference("u", "", &names, "abc123", false);
        assert_eq!(d.hints, ["refer to one of them, or make them differ (redefine `u` in the same call)"]);
    }

    #[test]
    fn serializes_without_empty_optionals() {
        let d = Diagnostic::type_mismatch("f", "body.then", "Int", "Bool").hint("use if");
        let json = serde_json::to_value(&d).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "code": "E0203", "kind": "type_mismatch", "def": "f", "path": "body.then",
                "message": "expected Int, found Bool", "expected": "Int", "found": "Bool",
                "hints": ["use if"]
            })
        );
        let p = serde_json::to_value(Diagnostic::parse(3, 7, "unexpected `)`")).unwrap();
        assert!(p.get("expected").is_none());
        assert_eq!(p["path"], "3:7");
    }

    #[test]
    fn match_and_record_diagnostics() {
        let d = Diagnostic::non_exhaustive("f", "body", "Some(_)");
        assert_eq!((d.code, d.kind, d.message.as_str()), ("E0205", "non_exhaustive_match", "match does not cover `Some(_)`"));
        assert_eq!(d.hints, vec!["add an arm `Some(_) => ...` or a final `_ => ...` arm".to_string()]);
        let d = Diagnostic::missing_field("f", "body", "User", "id");
        assert_eq!((d.code, d.kind), ("E0206", "type_mismatch"));
    }

    #[test]
    fn store_diagnostics() {
        let d = Diagnostic::signature_change("f", "fn(Int) -> Int", "fn(Str) -> Int", &["g".into(), "t".into()]);
        let json = serde_json::to_value(&d).unwrap();
        assert_eq!(json["code"], "E0401");
        assert_eq!(json["kind"], "signature_change");
        assert_eq!(json["dependents"], serde_json::json!(["g", "t"]));
        assert_eq!((json["expected"].as_str(), json["found"].as_str()), (Some("fn(Int) -> Int"), Some("fn(Str) -> Int")));
        assert!(serde_json::to_value(Diagnostic::store_error("x")).unwrap().get("dependents").is_none());
        assert_eq!(Diagnostic::type_too_large("f", "body", 4096).code, "E0208");
        assert_eq!(Diagnostic::missing_dep("f", "ab").kind, "missing_dep");
        let d = Diagnostic::symmetric_group(&["f".into(), "g".into()]);
        assert_eq!((d.code, d.kind, d.def.as_str()), ("E0209", "duplicate_name", "f"));
        assert_eq!(Diagnostic::group_too_large(&["a".into()]).code, "E0210");
        let d = Diagnostic::kind_change("t", "fn", "test");
        assert_eq!((d.code, d.kind, d.message.as_str()), ("E0405", "kind_change", "`t` is a fn in the store and cannot be redefined as a test"));
        let d = Diagnostic::bad_pattern("fn(", "unclosed `(`");
        assert_eq!((d.code, d.kind, d.message.as_str()), ("E0102", "bad_pattern", "invalid signature pattern `fn(`: unclosed `(`"));
    }
}
