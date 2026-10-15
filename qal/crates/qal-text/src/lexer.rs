//! Lexer: source text -> tokens with line/column positions.

use qal_core::diag::Diagnostic;

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Ident(String),
    /// Integer literal magnitude; the parser checks that it fits (`-9223372036854775808` is a valid pattern).
    Int(u64),
    Float(f64),
    /// String literal with escapes already resolved.
    Str(String),
    /// Punctuation and operators, e.g. "(", "->", "==", "&&".
    Sym(&'static str),
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub tok: Tok,
    pub line: usize,
    pub col: usize,
}

pub const KEYWORDS: &[&str] =
    &["fn", "type", "test", "let", "in", "if", "then", "else", "match", "req", "ens", "true", "false"];

/// Longest symbols first so that "->" wins over "-" and "++" over "+".
const SYMBOLS: &[&str] = &[
    "->", "=>", "==", "!=", "<=", ">=", "&&", "||", "++", "(", ")", "[", "]", "{", "}", ",", ";", ":", ".", "|", "=", "<",
    ">", "+", "-", "*", "/", "%", "!",
];

pub fn lex(src: &str) -> Result<Vec<Token>, Diagnostic> {
    let chars: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let (mut i, mut line, mut col) = (0usize, 1usize, 1usize);
    while i < chars.len() {
        let c = chars[i];
        if c == '\n' {
            i += 1;
            line += 1;
            col = 1;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            col += 1;
            continue;
        }
        if c == '#' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
                col += 1;
            }
            continue;
        }
        let (start_line, start_col) = (line, col);
        if c.is_ascii_digit() {
            let start = i;
            let digits = |i: &mut usize| {
                while *i < chars.len() && chars[*i].is_ascii_digit() {
                    *i += 1;
                }
            };
            digits(&mut i);
            let mut is_float = false;
            if i + 1 < chars.len() && chars[i] == '.' && chars[i + 1].is_ascii_digit() {
                is_float = true;
                i += 1;
                digits(&mut i);
            }
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                let sign = usize::from(i + 1 < chars.len() && (chars[i + 1] == '+' || chars[i + 1] == '-'));
                if i + 1 + sign < chars.len() && chars[i + 1 + sign].is_ascii_digit() {
                    is_float = true;
                    i += 1 + sign;
                    digits(&mut i);
                }
            }
            // A literal must end at a separator: `10abc` is one bad token, not `10` followed by `abc`.
            if i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let text: String = chars[start..i].iter().collect();
                return Err(Diagnostic::parse(start_line, start_col, format!("invalid number literal `{text}`")));
            }
            let text: String = chars[start..i].iter().collect();
            col += i - start;
            let tok = if is_float {
                match text.parse::<f64>() {
                    Ok(v) if v.is_finite() => Tok::Float(v),
                    _ => return Err(Diagnostic::parse(start_line, start_col, format!("float literal `{text}` is out of range"))),
                }
            } else {
                Tok::Int(text.parse::<u64>().map_err(|_| {
                    Diagnostic::parse(start_line, start_col, format!("integer literal `{text}` does not fit in Int (64-bit)"))
                })?)
            };
            out.push(Token { tok, line: start_line, col: start_col });
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            col += i - start;
            let text: String = chars[start..i].iter().collect();
            out.push(Token { tok: Tok::Ident(text), line: start_line, col: start_col });
            continue;
        }
        if c == '"' {
            let s = string(&chars, &mut i, &mut line, &mut col)?;
            out.push(Token { tok: Tok::Str(s), line: start_line, col: start_col });
            continue;
        }
        let rest: String = chars[i..chars.len().min(i + 2)].iter().collect();
        match SYMBOLS.iter().find(|s| rest.starts_with(**s)) {
            Some(sym) => {
                i += sym.len();
                col += sym.len();
                out.push(Token { tok: Tok::Sym(sym), line: start_line, col: start_col });
            }
            None => return Err(Diagnostic::parse(start_line, start_col, format!("unexpected character `{c}`"))),
        }
    }
    out.push(Token { tok: Tok::Eof, line, col });
    Ok(out)
}

/// Lexes a string literal starting at the opening quote. Escapes: `\n \t \r \0 \\ \" \u{HEX}`.
fn string(chars: &[char], i: &mut usize, line: &mut usize, col: &mut usize) -> Result<String, Diagnostic> {
    let (start_line, start_col) = (*line, *col);
    let mut s = String::new();
    *i += 1;
    *col += 1;
    loop {
        let Some(&c) = chars.get(*i) else {
            return Err(Diagnostic::parse(start_line, start_col, "unterminated string literal"));
        };
        let (esc_line, esc_col) = (*line, *col);
        *i += 1;
        *col += 1;
        match c {
            '"' => return Ok(s),
            '\n' => {
                *line += 1;
                *col = 1;
                s.push('\n');
            }
            '\\' => {
                let bad = |what: &str| Diagnostic::parse(esc_line, esc_col, format!("invalid escape `\\{what}` in string literal"));
                let Some(&e) = chars.get(*i) else {
                    return Err(Diagnostic::parse(start_line, start_col, "unterminated string literal"));
                };
                *i += 1;
                *col += 1;
                match e {
                    'n' => s.push('\n'),
                    't' => s.push('\t'),
                    'r' => s.push('\r'),
                    '0' => s.push('\0'),
                    '\\' => s.push('\\'),
                    '"' => s.push('"'),
                    'u' => {
                        if chars.get(*i) != Some(&'{') {
                            return Err(bad("u"));
                        }
                        let start = *i + 1;
                        let mut end = start;
                        while end < chars.len() && chars[end].is_ascii_hexdigit() && end - start < 6 {
                            end += 1;
                        }
                        let hex: String = chars[start..end].iter().collect();
                        if chars.get(end) != Some(&'}') || hex.is_empty() {
                            return Err(bad(&format!("u{{{hex}")));
                        }
                        let ch = u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32).ok_or_else(|| bad(&format!("u{{{hex}}}")))?;
                        s.push(ch);
                        *col += end + 1 - *i;
                        *i = end + 1;
                    }
                    other => return Err(bad(&other.to_string())),
                }
            }
            c => s.push(c),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(src: &str) -> Vec<Tok> {
        lex(src).unwrap().into_iter().map(|t| t.tok).collect()
    }

    #[test]
    fn lexes_function_header() {
        assert_eq!(
            toks("fn f(x: Int) -> Int"),
            vec![
                Tok::Ident("fn".into()),
                Tok::Ident("f".into()),
                Tok::Sym("("),
                Tok::Ident("x".into()),
                Tok::Sym(":"),
                Tok::Ident("Int".into()),
                Tok::Sym(")"),
                Tok::Sym("->"),
                Tok::Ident("Int".into()),
                Tok::Eof,
            ]
        );
    }

    #[test]
    fn lexes_two_char_operators_and_skips_comments() {
        assert_eq!(
            toks("a <= b # comment\n&& c != 10"),
            vec![
                Tok::Ident("a".into()),
                Tok::Sym("<="),
                Tok::Ident("b".into()),
                Tok::Sym("&&"),
                Tok::Ident("c".into()),
                Tok::Sym("!="),
                Tok::Int(10),
                Tok::Eof,
            ]
        );
    }

    #[test]
    fn tracks_positions() {
        let t = lex("fn\n  x").unwrap();
        assert_eq!((t[1].line, t[1].col), (2, 3));
    }

    #[test]
    fn eof_column_accounts_for_trailing_comment() {
        let t = lex("x # c").unwrap();
        let eof = t.last().unwrap();
        assert_eq!((eof.line, eof.col), (1, 6));
    }

    #[test]
    fn rejects_oversized_integer_and_unknown_characters() {
        let e = lex("99999999999999999999").unwrap_err();
        assert_eq!(e.kind, "parse");
        assert_eq!(e.path, "1:1");
        assert_eq!(toks("9223372036854775808"), vec![Tok::Int(1 << 63), Tok::Eof]);
        let e = lex("x @ y").unwrap_err();
        assert_eq!(e.path, "1:3");
        let e = lex("größe").unwrap_err();
        assert_eq!(e.kind, "parse");
    }

    #[test]
    fn lexes_floats_and_new_symbols() {
        assert_eq!(
            toks("1.5 2e3 7.25E-2 x.y ++ => [ ] { } ; |"),
            vec![
                Tok::Float(1.5),
                Tok::Float(2000.0),
                Tok::Float(0.0725),
                Tok::Ident("x".into()),
                Tok::Sym("."),
                Tok::Ident("y".into()),
                Tok::Sym("++"),
                Tok::Sym("=>"),
                Tok::Sym("["),
                Tok::Sym("]"),
                Tok::Sym("{"),
                Tok::Sym("}"),
                Tok::Sym(";"),
                Tok::Sym("|"),
                Tok::Eof,
            ]
        );
        // `1.` is not a float: the dot is a separate token.
        assert_eq!(toks("1.x"), vec![Tok::Int(1), Tok::Sym("."), Tok::Ident("x".into()), Tok::Eof]);
    }

    #[test]
    fn number_literals_need_a_separator() {
        let e = lex("x = 10abc").unwrap_err();
        assert_eq!(e.path, "1:5");
        assert!(e.message.contains("invalid number literal `10abc`"), "{}", e.message);
        assert!(lex("1.5e").unwrap_err().message.contains("`1.5e`"));
        assert!(lex("1e999").unwrap_err().message.contains("out of range"));
    }

    #[test]
    fn lexes_strings_with_escapes_and_utf8() {
        assert_eq!(toks(r#""a\"b\\c\n\u{1F600}" "größe""#), vec![Tok::Str("a\"b\\c\n😀".into()), Tok::Str("größe".into()), Tok::Eof]);
        let t = lex("\"a\nb\" x").unwrap();
        assert_eq!((t[1].line, t[1].col), (2, 4));
        let e = lex("x \"abc").unwrap_err();
        assert_eq!((e.path.as_str(), e.message.as_str()), ("1:3", "unterminated string literal"));
        assert!(lex(r#""\q""#).unwrap_err().message.contains("invalid escape `\\q`"));
        assert!(lex(r#""\u{110000}""#).unwrap_err().message.contains("invalid escape"));
    }
}
