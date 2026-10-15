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

    pub fn missing_field(def: &str, path: &str, record: &str, field: &str) -> Self {
        Diagnostic::new("E0206", "type_mismatch", def, path, format!("`{record}` needs field `{field}`"))
            .hint(format!("add `{field}: ...` to the `{record} {{ ... }}` literal"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
