//! Canonical printer: Program -> text. `parse_program(print_program(p)) == p` for every program the
//! parser produces. The printer is total: ASTs the parser cannot produce (negative or non-finite
//! literals, an empty sequence) print as parseable text with the same meaning.

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
        Def::Type(t) => {
            let head = format!("type {}{}", t.name, tparams(&t.params));
            type_body(&head, &t.body)
        }
        Def::Fn(f) => {
            let params: Vec<String> = f.params.iter().map(|p| format!("{}: {}", p.name, p.ty)).collect();
            let name = format!("{}{}", f.name, tparams(&f.tparams));
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
    fn prints_data_definitions_and_match() {
        let src = "type   E =|A|B( Int,Str )\ntype R={x:Float,}\n\
                   fn f(e: E) -> Str = match e {A=>\"a\\n\",B(n,s) if n>0=>s,_=>\"\"}";
        let p = parse_program(src).unwrap();
        assert_eq!(
            print_program(&p),
            "type E = A | B(Int, Str)\n\ntype R = { x: Float }\n\n\
             fn f(e: E) -> Str\n= match e { A => \"a\\n\", B(n, s) if n > 0 => s, _ => \"\" }\n"
        );
    }

    #[test]
    fn roundtrips_all_constructs() {
        roundtrip("fn f(x: Int) -> Int req x > 0 ens r >= 1 = if x == 1 then 1 else x * f(x - 1)");
        roundtrip("test t = let a = (if true then 1 else 2) in a + (let b = 3 in b)");
        roundtrip("test t = -(-1) == 1 && !(1 < 2) || (1 == 1) == true");
        roundtrip("fn u() -> Unit !io !fs = ()");
        roundtrip("test t = 1 - (2 - 3) - 4 * (5 % 2) / 1");
        roundtrip("type User = { id: Int, tags: List[Str] }\ntype T = Leaf | Node(T, Int, T)");
        roundtrip("fn g(k: fn(Int, Str) -> Option[Int] !io, r2: Result[Unit, Str]) -> List[Float] = [1.5, 0.1, 1e300, 2.5e-7]");
        roundtrip("fn f() -> (fn(Int) -> Int) !io = g\nfn h() -> fn(Int) -> Int !io = g\nfn k(x: fn() -> (fn() -> Int !io)) -> Int = 1");
        roundtrip(r#"test t = "q\"\\\n\t\r\0\u{7}é" ++ "x" == "y""#);
        roundtrip("test t = { print(\"a\"); { 1 }; 2 } == (User { id: 1, n: [] }).id");
        roundtrip("test t = (-a).b.c == -a.b.c && (a + b).c == 1");
        roundtrip("test t = let x: Option[List[Int]] = None in match x { Some(_) => 0, None => 1 }");
        roundtrip("test t = match (P { x: 1 }) { P { x: 1, y } => y, P { x: n } if n < -1 => -1, _ => match y { -3 => 1, () => 2 } }");
        roundtrip("test t = match (P { x: 1 }.x + 1) { _ => (match a { _ => 1 }) + 1 }");
        roundtrip("test t = match f(P { x: 1 }) { \"s\" => true, _ => false }");
        roundtrip("type Never = |\ntype R = {}\nfn f(n: Never) -> Int = match n {}\ntest t = (match (R {}) { R {} => 1 }) == 1");
        roundtrip("test t = match x { -9223372036854775808 => 1, -1 => 2, _ => 3 }");
    }

    #[test]
    fn roundtrips_generics_closures_maps_and_bytes() {
        roundtrip("type Pair[A, B] = { fst: A, snd: B }
type Tree[T] = Leaf | Node(Tree[T], T, Tree[T])");
        roundtrip("fn swap[A, B](p: Pair[A, B]) -> Pair[B, A] = Pair { fst: p.snd, snd: p.fst }");
        roundtrip("fn m(x: Map[Str, List[Int]], b: Bytes) -> Option[Bytes] = None");
        roundtrip("fn adder(n: Int) -> fn(Int) -> Int = fn(x) => x + n");
        roundtrip("test t = map([1], fn(x: Int, y) => { print(\"a\"); x }) == [] && f(fn(g: fn(Int) -> Int !io) => g)");
        roundtrip("test t = let f = (fn(x) => x) in 1 + f(2) == (fn(x) => x + 1 == 2) && g(fn() => fn() => 1)");
        let p = parse_program("fn  id[T](x:T)->T=x\ntest t = apply(fn(a)=>a+1, 2)").unwrap();
        assert_eq!(print_program(&p), "fn id[T](x: T) -> T\n= x\n\ntest t = apply(fn(a) => a + 1, 2)\n");
    }

    fn body(src: &str) -> Expr {
        let Def::Test(t) = &parse_program(&format!("test t = {src}")).unwrap().defs[0] else { panic!() };
        t.body.clone()
    }

    #[test]
    fn printer_is_total_for_asts_the_parser_does_not_build() {
        let cases: Vec<(Expr, &str, Expr)> = vec![
            (Expr::Int(-5), "(-5)", body("-5")),
            (Expr::Int(i64::MIN), "(-9223372036854775807 - 1)", body("-9223372036854775807 - 1")),
            (Expr::Float(-2.5), "(-2.5)", body("-2.5")),
            (Expr::Float(-0.0), "(-0.0)", body("-0.0")),
            (Expr::Float(f64::NAN), "(0.0 / 0.0)", body("0.0 / 0.0")),
            (Expr::Float(f64::INFINITY), "(1.0 / 0.0)", body("1.0 / 0.0")),
            (Expr::Float(f64::NEG_INFINITY), "(-1.0 / 0.0)", body("-1.0 / 0.0")),
            (Expr::Seq(vec![]), "()", Expr::Unit),
            (Expr::Bin(BinOp::Sub, Box::new(Expr::Int(1)), Box::new(Expr::Int(-2))), "1 - (-2)", body("1 - -2")),
            (Expr::Field(Box::new(Expr::Int(-1)), "x".into()), "(-1).x", body("(-1).x")),
        ];
        for (e, text, reparsed) in cases {
            assert_eq!(print_expr(&e), text);
            assert_eq!(body(text), reparsed, "{text}");
        }
        let empty = Program { defs: vec![Def::Type(TypeDef { name: "E".into(), params: vec![], body: TypeBody::Enum(vec![]) })] };
        assert_eq!(print_program(&empty), "type E = |\n");
        assert_eq!(parse_program(&print_program(&empty)).unwrap(), empty);
        assert_eq!(print_pat(&Pat::Int(i64::MIN)), "-9223372036854775808");
        assert_eq!(print_pat(&Pat::Record("P".into(), vec![])), "P {}");
    }
}
