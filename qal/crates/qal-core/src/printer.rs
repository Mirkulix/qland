//! Canonical printer: Program -> text. `parse_program(print_program(p)) == p` for every program the
//! parser produces. The printer is total: ASTs the parser cannot produce (negative or non-finite
//! literals, an empty sequence) print as parseable text with the same meaning.

use crate::ast::*;

pub fn print_program(p: &Program) -> String {
    let parts: Vec<String> = p.defs.iter().map(print_def).collect();
    let mut out = parts.join("\n\n");
    if !out.is_empty() {
        out.push('\n');
    }
    out
}

fn print_def(d: &Def) -> String {
    match d {
        Def::Test(t) => format!("test {} = {}", t.name, print_expr(&t.body)),
        Def::Type(t) => {
            let head = format!("type {}{}", t.name, tparams(&t.params));
            type_body(&head, &t.body)
        }
        Def::Fn(f) => {
            let params: Vec<String> = f.params.iter().map(|p| format!("{}: {}", p.name, p.ty)).collect();
            let mut heads = f.tparams.clone();
            heads.extend(f.eparams.iter().map(|e| format!("!{e}")));
            let name = format!("{}{}", f.name, tparams(&heads));
            let mut s = match f.ret {
                // `fn f() -> fn() -> Int !io` would give `!io` to the result type; parenthesize it.
                Type::Fn(..) => format!("fn {name}({}) -> ({})", params.join(", "), f.ret),
                _ => format!("fn {name}({}) -> {}", params.join(", "), f.ret),
            };
            for e in &f.effects {
                s.push_str(&format!(" !{e}"));
            }
            for c in &f.req {
                s.push_str(&format!("\n  req {}", print_expr(c)));
            }
            for c in &f.ens {
                s.push_str(&format!("\n  ens {}", print_expr(c)));
            }
            s.push_str(&format!("\n= {}", print_expr(&f.body)));
            s
        }
    }
}

/// `[A, B]`, or nothing for a monomorphic definition.
fn tparams(ps: &[String]) -> String {
    if ps.is_empty() {
        String::new()
    } else {
        format!("[{}]", ps.join(", "))
    }
}

fn type_body(head: &str, body: &TypeBody) -> String {
    match body {
        TypeBody::Record(fields) => {
            let fs: Vec<String> = fields.iter().map(|(n, ty)| format!("{n}: {ty}")).collect();
            format!("{head} = {}", braces(&fs))
        }
        TypeBody::Enum(variants) if variants.is_empty() => format!("{head} = |"),
        TypeBody::Enum(variants) => {
            let vs: Vec<String> = variants
                .iter()
                .map(|v| {
                    if v.fields.is_empty() {
                        v.name.clone()
                    } else {
                        let fs: Vec<String> = v.fields.iter().map(|f| f.to_string()).collect();
                        format!("{}({})", v.name, fs.join(", "))
                    }
                })
                .collect();
            format!("{head} = {}", vs.join(" | "))
        }
    }
}

/// Precedence of an expression's outermost construct (higher binds tighter).
fn prec(e: &Expr) -> u8 {
    match e {
        Expr::Let(..) | Expr::If(..) | Expr::Match(..) | Expr::Lambda(..) => 0,
        Expr::Bin(op, ..) => op.precedence(),
        Expr::Un(..) => 6,
        Expr::Field(..) => 8,
        _ => 9,
    }
}

fn wrap(e: &Expr, needs_parens: bool) -> String {
    let s = print_expr(e);
    if needs_parens {
        format!("({s})")
    } else {
        s
    }
}

/// True if `e` contains a record literal outside of any delimiter, which a match scrutinee must not.
fn has_bare_record(e: &Expr) -> bool {
    match e {
        Expr::Record(..) => true,
        Expr::Bin(_, l, r) => has_bare_record(l) || has_bare_record(r),
        Expr::Un(_, x) | Expr::Field(x, _) => has_bare_record(x),
        Expr::If(c, t, f) => has_bare_record(c) || has_bare_record(t) || has_bare_record(f),
        Expr::Let(_, _, v, b) => has_bare_record(v) || has_bare_record(b),
        Expr::Lambda(_, b) => has_bare_record(b),
        Expr::Match(s, arms) => {
            has_bare_record(s) || arms.iter().any(|a| a.guard.as_ref().is_some_and(has_bare_record) || has_bare_record(&a.body))
        }
        _ => false,
    }
}

/// `{ a, b }`, or `{}` when empty.
fn braces(items: &[String]) -> String {
    if items.is_empty() {
        "{}".to_string()
    } else {
        format!("{{ {} }}", items.join(", "))
    }
}

/// An Int literal; negative values (which the parser builds as `-n`) are parenthesized negations.
fn print_int(n: i64) -> String {
    match n {
        i64::MIN => "(-9223372036854775807 - 1)".to_string(),
        n if n < 0 => format!("(-{})", n.unsigned_abs()),
        n => n.to_string(),
    }
}

/// A Float literal; negative and non-finite values print as expressions with that value.
fn print_float(x: f64) -> String {
    if x.is_nan() {
        "(0.0 / 0.0)".to_string()
    } else if x.is_infinite() {
        if x > 0.0 { "(1.0 / 0.0)" } else { "(-1.0 / 0.0)" }.to_string()
    } else if x.is_sign_negative() {
        format!("(-{:?})", -x)
    } else {
        format!("{x:?}")
    }
}

pub fn print_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            '\0' => out.push_str("\\0"),
            c if c.is_control() => out.push_str(&format!("\\u{{{:x}}}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

pub fn print_pat(p: &Pat) -> String {
    match p {
        Pat::Wild => "_".into(),
        Pat::Bind(n) => n.clone(),
        Pat::Int(n) => n.to_string(),
        Pat::Str(s) => print_str(s),
        Pat::Bool(b) => b.to_string(),
        Pat::Unit => "()".into(),
        Pat::Variant(n, ps) if ps.is_empty() => n.clone(),
        Pat::Variant(n, ps) => {
            let ps: Vec<String> = ps.iter().map(print_pat).collect();
            format!("{n}({})", ps.join(", "))
        }
        Pat::Record(n, fs) => {
            let fs: Vec<String> = fs
                .iter()
                .map(|(f, p)| match p {
                    Pat::Bind(b) if b == f => f.clone(),
                    _ => format!("{f}: {}", print_pat(p)),
                })
                .collect();
            format!("{n} {}", braces(&fs))
        }
    }
}

fn comma(items: &[Expr]) -> String {
    items.iter().map(print_expr).collect::<Vec<_>>().join(", ")
}

pub fn print_expr(e: &Expr) -> String {
    match e {
        Expr::Int(n) => print_int(*n),
        Expr::Float(x) => print_float(*x),
        Expr::Bool(b) => b.to_string(),
        Expr::Str(s) => print_str(s),
        Expr::Unit => "()".to_string(),
        Expr::Var(v) => v.clone(),
        Expr::Call(f, args) => format!("{f}({})", comma(args)),
        Expr::List(items) => format!("[{}]", comma(items)),
        // An empty sequence (never parsed) evaluates to `()`.
        Expr::Seq(items) if items.is_empty() => "()".to_string(),
        Expr::Seq(items) => format!("{{ {} }}", items.iter().map(print_expr).collect::<Vec<_>>().join("; ")),
        Expr::Record(n, fs) => {
            let fs: Vec<String> = fs.iter().map(|(f, v)| format!("{f}: {}", print_expr(v))).collect();
            format!("{n} {}", braces(&fs))
        }
        Expr::Field(t, f) => format!("{}.{f}", wrap(t, prec(t) < 8)),
        Expr::Un(op, x) => {
            let sym = match op {
                UnOp::Neg => "-",
                UnOp::Not => "!",
            };
            format!("{sym}{}", wrap(x, prec(x) < 6))
        }
        Expr::Bin(op, l, r) => {
            let p = op.precedence();
            // Left-associative: the left side may share our precedence, the right may not.
            // Comparisons are non-associative: neither side may share it.
            let left_parens = prec(l) < p || (op.is_comparison() && prec(l) == p);
            let right_parens = prec(r) <= p;
            format!("{} {} {}", wrap(l, left_parens), op.symbol(), wrap(r, right_parens))
        }
        Expr::If(c, t, f) => format!("if {} then {} else {}", print_expr(c), print_expr(t), print_expr(f)),
        Expr::Let(n, ann, v, b) => match ann {
            Some(t) => format!("let {n}: {t} = {} in {}", print_expr(v), print_expr(b)),
            None => format!("let {n} = {} in {}", print_expr(v), print_expr(b)),
        },
        Expr::Match(s, arms) => {
            let arms: Vec<String> = arms
                .iter()
                .map(|a| match &a.guard {
                    Some(g) => format!("{} if {} => {}", print_pat(&a.pat), print_expr(g), print_expr(&a.body)),
                    None => format!("{} => {}", print_pat(&a.pat), print_expr(&a.body)),
                })
                .collect();
            format!("match {} {}", wrap(s, has_bare_record(s)), braces(&arms))
        }
        Expr::Lambda(params, body) => {
            let ps: Vec<String> = params
                .iter()
                .map(|(n, ann)| match ann {
                    Some(t) => format!("{n}: {t}"),
                    None => n.clone(),
                })
                .collect();
            format!("fn({}) => {}", ps.join(", "), print_expr(body))
        }
    }
}
