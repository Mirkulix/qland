//! Signature patterns for `query --type`: the signature text with `_` standing for any one type.
//!
//! Spaces do not matter. `_` matches a non-empty, bracket-balanced run of tokens without a
//! top-level `,` or `|`, e.g. `fn(_, Int) -> _` matches `fn(List[Str], Int) -> Option[Int] !io`.

/// Longest pattern accepted, in tokens; longer patterns match nothing.
const MAX_PATTERN_TOKENS: usize = 256;

fn tokens(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_alphanumeric() || c == '_' || c == '#' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '#') {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
        } else if c == '-' && chars.get(i + 1) == Some(&'>') {
            out.push("->".into());
            i += 2;
        } else {
            out.push(c.to_string());
            i += 1;
        }
    }
    out
}

/// True if `signature` matches `pattern`.
pub fn matches(pattern: &str, signature: &str) -> bool {
    let p = tokens(pattern);
    if p.is_empty() || p.len() > MAX_PATTERN_TOKENS {
        return false;
    }
    let s = tokens(signature);
    let mut memo = std::collections::HashMap::new();
    go(&p, &s, 0, 0, &mut memo)
}

fn go(p: &[String], s: &[String], mut pi: usize, mut si: usize, memo: &mut std::collections::HashMap<(usize, usize), bool>) -> bool {
    // Plain tokens are compared in a loop; only `_` recurses (at most MAX_PATTERN_TOKENS deep).
    while pi < p.len() && p[pi] != "_" {
        if si >= s.len() || p[pi] != s[si] {
            return false;
        }
        pi += 1;
        si += 1;
    }
    if pi == p.len() {
        return si == s.len();
    }
    if let Some(&r) = memo.get(&(pi, si)) {
        return r;
    }
    let mut depth = 0usize;
    let mut result = false;
    for e in si..s.len() {
        match s[e].as_str() {
            "(" | "[" | "{" => depth += 1,
            ")" | "]" | "}" => {
                if depth == 0 {
                    break;
                }
                depth -= 1;
            }
            "," | "|" if depth == 0 => break,
            _ => {}
        }
        if depth == 0 && go(p, s, pi + 1, e + 1, memo) {
            result = true;
            break;
        }
    }
    memo.insert((pi, si), result);
    result
}

#[cfg(test)]
mod tests {
    use super::matches;

    #[test]
    fn wildcards_match_one_type() {
        assert!(matches("fn(Int) -> Int", "fn(Int) -> Int"));
        assert!(matches("fn( Int )->Int", "fn(Int) -> Int"));
        assert!(!matches("fn(Int) -> Int", "fn(Int) -> Int !io"));
        assert!(matches("fn(_) -> Int", "fn(List[Option[Str]]) -> Int"));
        assert!(!matches("fn(_) -> Int", "fn(Int, Int) -> Int"));
        assert!(matches("fn(_, _) -> _", "fn(Int, fn(Int) -> Int !io) -> Result[Int, Str] !fs"));
        assert!(matches("fn(fn(_) -> _) -> Int", "fn(fn(Str) -> Bool) -> Int"));
        assert!(matches("_ | B(_)", "A | B(Int)"));
        assert!(!matches("_", "A | B(Int)"));
        assert!(!matches("", "Bool"));
        assert!(!matches(&"_ ".repeat(300), "Bool"));
    }
}
