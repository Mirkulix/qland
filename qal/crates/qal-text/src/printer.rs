//! Canonical printer: Program -> text. `parse_program(print_program(p)) == p`.

use qal_core::ast::*;

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
        Def::Fn(f) => {
            let params: Vec<String> = f.params.iter().map(|p| format!("{}: {}", p.name, p.ty.name())).collect();
            let mut s = format!("fn {}({}) -> {}", f.name, params.join(", "), f.ret.name());
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

/// Precedence of an expression's outermost construct (higher binds tighter).
fn prec(e: &Expr) -> u8 {
    match e {
        Expr::Let(..) | Expr::If(..) => 0,
        Expr::Bin(op, ..) => op.precedence(),
        Expr::Un(..) => 6,
        _ => 7,
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

pub fn print_expr(e: &Expr) -> String {
    match e {
        Expr::Int(n) => n.to_string(),
        Expr::Bool(b) => b.to_string(),
        Expr::Unit => "()".to_string(),
        Expr::Var(v) => v.clone(),
        Expr::Call(f, args) => {
            let a: Vec<String> = args.iter().map(print_expr).collect();
            format!("{f}({})", a.join(", "))
        }
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
            // Comparisons (3) are non-associative: neither side may share it.
            let left_parens = prec(l) < p || (p == 3 && prec(l) == 3);
            let right_parens = prec(r) <= p;
            format!("{} {} {}", wrap(l, left_parens), op.symbol(), wrap(r, right_parens))
        }
        Expr::If(c, t, f) => format!("if {} then {} else {}", print_expr(c), print_expr(t), print_expr(f)),
        Expr::Let(n, v, b) => format!("let {n} = {} in {}", print_expr(v), print_expr(b)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_program;

    fn roundtrip(src: &str) {
        let p = parse_program(src).unwrap();
        let printed = print_program(&p);
        let reparsed = parse_program(&printed).unwrap_or_else(|e| panic!("reparse failed: {e:?}\n{printed}"));
        assert_eq!(reparsed, p, "printed:\n{printed}");
    }

    #[test]
    fn prints_canonical_form() {
        let p = parse_program("fn  add(a:Int,b:Int)->Int !io req a>=0 = a+b*2").unwrap();
        assert_eq!(print_program(&p), "fn add(a: Int, b: Int) -> Int !io\n  req a >= 0\n= a + b * 2\n");
    }

    #[test]
    fn keeps_needed_parentheses_only() {
        let p = parse_program("test t = (1 + 2) * 3 - (4 - 5) == ((6))").unwrap();
        assert_eq!(print_program(&p), "test t = (1 + 2) * 3 - (4 - 5) == 6\n");
    }

    #[test]
    fn roundtrips_all_constructs() {
        roundtrip("fn f(x: Int) -> Int req x > 0 ens r >= 1 = if x == 1 then 1 else x * f(x - 1)");
        roundtrip("test t = let a = (if true then 1 else 2) in a + (let b = 3 in b)");
        roundtrip("test t = -(-1) == 1 && !(1 < 2) || (1 == 1) == true");
        roundtrip("fn u() -> Unit !io !fs = ()");
        roundtrip("test t = 1 - (2 - 3) - 4 * (5 % 2) / 1");
    }
}
