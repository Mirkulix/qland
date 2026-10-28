//! Static checks of the SQL literals of `db_exec` / `db_query` (Phase 3 spec §3.2).
//!
//! A small SQLite-compatible tokenizer (strings, quoted identifiers, comments, parameters) decides
//! without any schema: the text is exactly one complete statement (one trailing `;` allowed), uses
//! only positional `?` parameters, and does not start with a keyword that controls the database or
//! the transaction. The checker runs this on every literal (`sql_invalid`, E0242), and the host
//! runs it again before preparing a statement.

/// Statement keywords the guest may not use: attaching files, pragmas, vacuuming (`VACUUM INTO`
/// writes files) and transaction control (the host owns the one transaction per execution).
pub const FORBIDDEN: [&str; 10] = ["ATTACH", "DETACH", "PRAGMA", "VACUUM", "BEGIN", "COMMIT", "END", "ROLLBACK", "SAVEPOINT", "RELEASE"];

/// Longest accepted SQL text in bytes.
pub const MAX_SQL_BYTES: usize = 64 << 10;

#[derive(Debug, PartialEq)]
enum Tok {
    Word(String),
    Param,
    Semicolon,
    Other,
}

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '$' || !c.is_ascii()
}

/// Tokens of `sql`, without whitespace and comments.
fn tokens(sql: &str) -> Result<Vec<Tok>, String> {
    let cs: Vec<char> = sql.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    let close = |cs: &[char], mut i: usize, q: char, what: &str| -> Result<usize, String> {
        // Doubling the quote escapes it (`'it''s'`).
        loop {
            match cs.get(i) {
                None => return Err(format!("unterminated {what}")),
                Some(c) if *c == q && cs.get(i + 1) == Some(&q) => i += 2,
                Some(c) if *c == q => return Ok(i + 1),
                Some(_) => i += 1,
            }
        }
    };
    while i < cs.len() {
        let c = cs[i];
        match c {
            _ if c.is_whitespace() => i += 1,
            '-' if cs.get(i + 1) == Some(&'-') => {
                while i < cs.len() && cs[i] != '\n' {
                    i += 1;
                }
            }
            '/' if cs.get(i + 1) == Some(&'*') => {
                let mut j = i + 2;
                loop {
                    match cs.get(j) {
                        None => return Err("unterminated comment".into()),
                        Some('*') if cs.get(j + 1) == Some(&'/') => break,
                        Some(_) => j += 1,
                    }
                }
                i = j + 2;
            }
            '\'' => {
                i = close(&cs, i + 1, '\'', "string")?;
                out.push(Tok::Other);
            }
            '"' | '`' => {
                i = close(&cs, i + 1, c, "quoted identifier")?;
                out.push(Tok::Other);
            }
            '[' => {
                let j = cs[i..].iter().position(|c| *c == ']').ok_or("unterminated quoted identifier")?;
                i += j + 1;
                out.push(Tok::Other);
            }
            ';' => {
                out.push(Tok::Semicolon);
                i += 1;
            }
            '?' => {
                if cs.get(i + 1).is_some_and(char::is_ascii_digit) {
                    return Err("numbered parameters (`?NNN`) are not allowed: use `?` and pass the values in order".into());
                }
                out.push(Tok::Param);
                i += 1;
            }
            ':' | '@' | '$' | '#' if cs.get(i + 1).is_some_and(|c| is_ident(*c)) => {
                return Err(format!("named parameters (`{c}name`) are not allowed: use `?` and pass the values in order"));
            }
            _ if is_ident(c) => {
                let start = i;
                while i < cs.len() && is_ident(cs[i]) {
                    i += 1;
                }
                out.push(Tok::Word(cs[start..i].iter().collect()));
            }
            _ => {
                out.push(Tok::Other);
                i += 1;
            }
        }
    }
    Ok(out)
}

fn word(t: Option<&Tok>) -> Option<String> {
    match t {
        Some(Tok::Word(w)) => Some(w.to_ascii_uppercase()),
        _ => None,
    }
}

/// Index of the `;` that ends the first statement (or the number of tokens). As in
/// `sqlite3_complete`, a `CREATE [TEMP] TRIGGER` statement holds `;`s in its body and ends only
/// at `; END ;` (or at the end of the text).
fn statement_end(toks: &[Tok]) -> usize {
    let mut i = 0;
    if word(toks.get(i)).as_deref() == Some("CREATE") {
        i += 1;
        if matches!(word(toks.get(i)).as_deref(), Some("TEMP" | "TEMPORARY")) {
            i += 1;
        }
    }
    let trigger = i > 0 && word(toks.get(i)).as_deref() == Some("TRIGGER");
    for (k, t) in toks.iter().enumerate() {
        if *t != Tok::Semicolon {
            continue;
        }
        if !trigger || (k >= 2 && word(toks.get(k - 1)).as_deref() == Some("END") && toks[k - 2] == Tok::Semicolon) {
            return k;
        }
    }
    toks.len()
}

/// Checks one SQL literal; returns the number of `?` parameters, or why the text is refused.
pub fn check_statement(sql: &str) -> Result<usize, String> {
    if sql.len() > MAX_SQL_BYTES {
        return Err(format!("the SQL text is longer than {MAX_SQL_BYTES} bytes"));
    }
    if sql.contains('\0') {
        return Err("the SQL text contains a NUL character".into());
    }
    let toks = tokens(sql)?;
    let end = statement_end(&toks);
    if toks.len() > end + 1 {
        return Err("only one statement is allowed (a single trailing `;` is fine)".into());
    }
    let stmt = &toks[..end];
    let mut words = stmt.iter().map(|t| match t {
        Tok::Word(w) => Some(w.to_ascii_uppercase()),
        _ => None,
    });
    let mut first = words.next().flatten().ok_or("the SQL text has no statement")?;
    // `EXPLAIN [QUERY PLAN] <stmt>` is judged by the explained statement.
    if first == "EXPLAIN" {
        let mut next = words.next().flatten().unwrap_or_default();
        if next == "QUERY" {
            words.next();
            next = words.next().flatten().unwrap_or_default();
        }
        first = next;
    }
    if FORBIDDEN.contains(&first.as_str()) {
        return Err(format!("`{first}` statements are not allowed: the host controls the database file, its settings and the transaction"));
    }
    Ok(stmt.iter().filter(|t| **t == Tok::Param).count())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_one_statement_with_positional_parameters() {
        assert_eq!(check_statement("SELECT id, title FROM todos WHERE id = ?"), Ok(1));
        assert_eq!(check_statement("insert into t (a, b) values (?, ?);"), Ok(2));
        assert_eq!(check_statement("SELECT 'a;b ? :x', \"col;\" FROM [we;ird] -- ; ?\n"), Ok(0));
        assert_eq!(check_statement("/* ; */ UPDATE t SET a = ? WHERE b = 'it''s'"), Ok(1));
        assert_eq!(check_statement("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, a$b TEXT)"), Ok(0));
        assert_eq!(check_statement("WITH x AS (SELECT ?) SELECT * FROM x"), Ok(1));
        let trigger = "CREATE TRIGGER tr AFTER INSERT ON t BEGIN UPDATE t SET a = CASE WHEN 1 THEN 2 END; DELETE FROM u; END;";
        assert_eq!(check_statement(trigger), Ok(0));
        assert_eq!(check_statement("create temp trigger tr after insert on t begin select 1; end"), Ok(0));
    }

    #[test]
    fn refuses_several_statements_named_parameters_and_control_keywords() {
        let e = |s: &str| check_statement(s).unwrap_err();
        assert!(e("SELECT 1; SELECT 2").contains("only one statement"));
        assert!(e("SELECT 1;;").contains("only one statement"));
        assert!(e("SELECT 1; -- x\n DROP TABLE t").contains("only one statement"));
        assert!(e("CREATE TRIGGER tr AFTER INSERT ON t BEGIN SELECT 1; END; DROP TABLE t").contains("only one statement"));
        assert!(e("SELECT 1; END; DROP TABLE t").contains("only one statement"), "only triggers span `;`");
        assert!(e("SELECT :id").contains("named parameters"));
        assert!(e("SELECT @id").contains("named parameters"));
        assert!(e("SELECT $id").contains("named parameters"));
        assert!(e("SELECT ?1").contains("numbered parameters"));
        for kw in ["ATTACH 'x' AS y", "detach y", "PRAGMA writable_schema = 1", "VACUUM INTO '/tmp/x'", "BEGIN", "COMMIT", "end", "ROLLBACK", "SAVEPOINT a", "RELEASE a", "EXPLAIN PRAGMA x", "explain query plan attach 'f' as g"] {
            assert!(e(kw).contains("not allowed"), "{kw}");
        }
        assert!(e("").contains("no statement"));
        assert!(e("  -- only a comment").contains("no statement"));
        assert!(e("SELECT 'open").contains("unterminated string"));
        assert!(e("SELECT /* open").contains("unterminated comment"));
        assert!(e("SELECT \"open").contains("unterminated"));
        assert!(e("SELECT 1\0").contains("NUL"));
        assert!(e(&format!("SELECT '{}'", "a".repeat(MAX_SQL_BYTES))).contains("longer than"));
    }
}
