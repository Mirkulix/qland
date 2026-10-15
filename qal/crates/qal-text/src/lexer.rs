//! Lexer: source text -> tokens with line/column positions.

use qal_core::diag::Diagnostic;

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Ident(String),
    Int(i64),
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

pub const KEYWORDS: &[&str] = &["fn", "test", "let", "in", "if", "then", "else", "req", "ens", "true", "false"];

/// Longest symbols first so that "->" wins over "-".
const SYMBOLS: &[&str] = &[
    "->", "==", "!=", "<=", ">=", "&&", "||", "(", ")", ",", ":", "=", "<", ">", "+", "-", "*", "/", "%", "!",
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
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            let text: String = chars[start..i].iter().collect();
            col += i - start;
            let value = text
                .parse::<i64>()
                .map_err(|_| Diagnostic::parse(start_line, start_col, format!("integer literal `{text}` does not fit in Int (64-bit)")))?;
            out.push(Token { tok: Tok::Int(value), line: start_line, col: start_col });
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
        let e = lex("x @ y").unwrap_err();
        assert_eq!(e.path, "1:3");
        let e = lex("größe").unwrap_err();
        assert_eq!(e.kind, "parse");
    }
}
