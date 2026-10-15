//! Recursive-descent parser: tokens -> Program.
//!
//! Grammar (Phase 1a):
//!   program  := def*
//!   def      := "fn" IDENT "(" params ")" "->" type effect* contract* "=" expr
//!             | "test" IDENT "=" expr
//!   params   := (IDENT ":" type ("," IDENT ":" type)* ","?)?
//!   type     := "Int" | "Bool" | "Unit"
//!   effect   := "!" IDENT
//!   contract := ("req" | "ens") expr
//!   expr     := "let" IDENT "=" expr "in" expr
//!             | "if" expr "then" expr "else" expr
//!             | binary
//!   binary   := precedence climbing over BinOp::precedence (comparisons non-associative)
//!   unary    := ("-" | "!") unary | atom
//!   atom     := INT | "true" | "false" | "(" ")" | "(" expr ")" | IDENT | IDENT "(" args ")"

use crate::lexer::{lex, Tok, Token, KEYWORDS};
use qal_core::ast::*;
use qal_core::diag::Diagnostic;

/// Maximum AST height (and parser nesting). Every later stage only sees parser-built ASTs and
/// recurses over them, so bounding this here bounds recursion everywhere.
const MAX_DEPTH: usize = 256;

pub fn parse_program(src: &str) -> Result<Program, Diagnostic> {
    let tokens = lex(src)?;
    let mut p = Parser { tokens, pos: 0, depth: 0, height: 0 };
    let mut defs = Vec::new();
    while p.peek() != &Tok::Eof {
        defs.push(p.def()?);
    }
    Ok(Program { defs })
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    /// Current recursion depth of expr()/unary().
    depth: usize,
    /// Height of the AST most recently built.
    height: usize,
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.tokens[self.pos].tok
    }

    fn next(&mut self) -> Token {
        let t = self.tokens[self.pos].clone();
        if t.tok != Tok::Eof {
            self.pos += 1;
        }
        t
    }

    fn err_here(&self, message: impl Into<String>) -> Diagnostic {
        let t = &self.tokens[self.pos];
        Diagnostic::parse(t.line, t.col, message)
    }

    #[cold]
    fn too_deep(&self) -> Diagnostic {
        self.err_here(format!("expression nested too deeply (max {MAX_DEPTH} levels)"))
    }

    /// Records a new AST node whose tallest child has height `child`.
    fn node(&mut self, child: usize) -> Result<(), Diagnostic> {
        self.height = child + 1;
        if self.height > MAX_DEPTH {
            return Err(self.too_deep());
        }
        Ok(())
    }

    fn describe(tok: &Tok) -> String {
        match tok {
            Tok::Ident(s) => format!("`{s}`"),
            Tok::Int(n) => format!("`{n}`"),
            Tok::Sym(s) => format!("`{s}`"),
            Tok::Eof => "end of input".to_string(),
        }
    }

    fn is_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Tok::Ident(s) if s == kw)
    }

    fn is_sym(&self, sym: &str) -> bool {
        matches!(self.peek(), Tok::Sym(s) if *s == sym)
    }

    fn expect_kw(&mut self, kw: &str) -> Result<(), Diagnostic> {
        if self.is_kw(kw) {
            self.next();
            Ok(())
        } else {
            Err(self.err_here(format!("expected `{kw}`, found {}", Self::describe(self.peek()))))
        }
    }

    fn expect_sym(&mut self, sym: &str) -> Result<(), Diagnostic> {
        if self.is_sym(sym) {
            self.next();
            Ok(())
        } else {
            Err(self.err_here(format!("expected `{sym}`, found {}", Self::describe(self.peek()))))
        }
    }

    fn ident(&mut self) -> Result<String, Diagnostic> {
        match self.peek().clone() {
            Tok::Ident(s) if !KEYWORDS.contains(&s.as_str()) => {
                self.next();
                Ok(s)
            }
            other => Err(self.err_here(format!("expected a name, found {}", Self::describe(&other)))),
        }
    }

    fn ty(&mut self) -> Result<Type, Diagnostic> {
        let t = match self.peek() {
            Tok::Ident(s) if s == "Int" => Type::Int,
            Tok::Ident(s) if s == "Bool" => Type::Bool,
            Tok::Ident(s) if s == "Unit" => Type::Unit,
            other => return Err(self.err_here(format!("expected a type (Int, Bool, Unit), found {}", Self::describe(other)))),
        };
        self.next();
        Ok(t)
    }

    fn def(&mut self) -> Result<Def, Diagnostic> {
        if self.is_kw("fn") {
            self.next();
            self.fn_def().map(Def::Fn)
        } else if self.is_kw("test") {
            self.next();
            let name = self.ident()?;
            self.expect_sym("=")?;
            let body = self.expr()?;
            Ok(Def::Test(TestDef { name, body }))
        } else {
            Err(self.err_here(format!("expected `fn` or `test`, found {}", Self::describe(self.peek()))))
        }
    }

    fn fn_def(&mut self) -> Result<FnDef, Diagnostic> {
        let name = self.ident()?;
        self.expect_sym("(")?;
        let mut params = Vec::new();
        while !self.is_sym(")") {
            let pname = self.ident()?;
            self.expect_sym(":")?;
            let ty = self.ty()?;
            params.push(Param { name: pname, ty });
            if !self.is_sym(")") {
                self.expect_sym(",")?;
            }
        }
        self.expect_sym(")")?;
        self.expect_sym("->")?;
        let ret = self.ty()?;
        let mut effects = Vec::new();
        while self.is_sym("!") {
            self.next();
            effects.push(self.ident()?);
        }
        let (mut req, mut ens) = (Vec::new(), Vec::new());
        loop {
            if self.is_kw("req") {
                self.next();
                req.push(self.expr()?);
            } else if self.is_kw("ens") {
                self.next();
                ens.push(self.expr()?);
            } else {
                break;
            }
        }
        self.expect_sym("=")?;
        let body = self.expr()?;
        Ok(FnDef { name, params, ret, effects, req, ens, body })
    }

    fn expr(&mut self) -> Result<Expr, Diagnostic> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.too_deep());
        }
        let e = self.expr_inner();
        self.depth -= 1;
        e
    }

    fn expr_inner(&mut self) -> Result<Expr, Diagnostic> {
        if self.is_kw("let") {
            self.let_expr()
        } else if self.is_kw("if") {
            self.if_expr()
        } else {
            self.binary(1)
        }
    }

    // let/if/call/paren live in their own functions so the frames on the common recursion path stay small.
    fn let_expr(&mut self) -> Result<Expr, Diagnostic> {
        {
            self.next();
            let name = self.ident()?;
            self.expect_sym("=")?;
            let value = self.expr()?;
            self.expect_kw("in")?;
            let value_h = self.height;
            let body = self.expr()?;
            self.node(value_h.max(self.height))?;
            Ok(Expr::Let(name, Box::new(value), Box::new(body)))
        }
    }

    fn if_expr(&mut self) -> Result<Expr, Diagnostic> {
        {
            self.next();
            let c = self.expr()?;
            let mut h = self.height;
            self.expect_kw("then")?;
            let t = self.expr()?;
            h = h.max(self.height);
            self.expect_kw("else")?;
            let e = self.expr()?;
            self.node(h.max(self.height))?;
            Ok(Expr::If(Box::new(c), Box::new(t), Box::new(e)))
        }
    }

    fn peek_binop(&self) -> Option<BinOp> {
        let op = match self.peek() {
            Tok::Sym("+") => BinOp::Add,
            Tok::Sym("-") => BinOp::Sub,
            Tok::Sym("*") => BinOp::Mul,
            Tok::Sym("/") => BinOp::Div,
            Tok::Sym("%") => BinOp::Rem,
            Tok::Sym("==") => BinOp::Eq,
            Tok::Sym("!=") => BinOp::Ne,
            Tok::Sym("<") => BinOp::Lt,
            Tok::Sym("<=") => BinOp::Le,
            Tok::Sym(">") => BinOp::Gt,
            Tok::Sym(">=") => BinOp::Ge,
            Tok::Sym("&&") => BinOp::And,
            Tok::Sym("||") => BinOp::Or,
            _ => return None,
        };
        Some(op)
    }

    fn binary(&mut self, min_prec: u8) -> Result<Expr, Diagnostic> {
        let mut lhs = self.unary()?;
        while let Some(op) = self.peek_binop() {
            let prec = op.precedence();
            if prec < min_prec {
                break;
            }
            self.next();
            let lhs_h = self.height;
            let rhs = self.binary(prec + 1)?;
            self.node(lhs_h.max(self.height))?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
            if prec == 3 {
                if let Some(next) = self.peek_binop() {
                    if next.precedence() == 3 {
                        return Err(self.chained());
                    }
                }
            }
        }
        Ok(lhs)
    }

    #[cold]
    fn chained(&self) -> Diagnostic {
        self.err_here("comparisons cannot be chained; use `&&`")
    }

    fn unary(&mut self) -> Result<Expr, Diagnostic> {
        let op = if self.is_sym("-") {
            UnOp::Neg
        } else if self.is_sym("!") {
            UnOp::Not
        } else {
            return self.atom();
        };
        self.next();
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.too_deep());
        }
        let inner = self.unary()?;
        self.depth -= 1;
        self.node(self.height)?;
        Ok(Expr::Un(op, Box::new(inner)))
    }

    fn atom(&mut self) -> Result<Expr, Diagnostic> {
        match self.peek().clone() {
            Tok::Int(n) => {
                self.next();
                self.height = 1;
                Ok(Expr::Int(n))
            }
            Tok::Ident(s) if s == "true" => {
                self.next();
                self.height = 1;
                Ok(Expr::Bool(true))
            }
            Tok::Ident(s) if s == "false" => {
                self.next();
                self.height = 1;
                Ok(Expr::Bool(false))
            }
            Tok::Sym("(") => self.paren(),
            Tok::Ident(_) => self.name_or_call(),
            other => Err(self.expected_expr(&other)),
        }
    }

    #[cold]
    fn expected_expr(&self, found: &Tok) -> Diagnostic {
        self.err_here(format!("expected an expression, found {}", Self::describe(found)))
    }

    fn paren(&mut self) -> Result<Expr, Diagnostic> {
        self.next();
        if self.is_sym(")") {
            self.next();
            self.height = 1;
            return Ok(Expr::Unit);
        }
        let e = self.expr()?;
        self.expect_sym(")")?;
        Ok(e)
    }

    fn name_or_call(&mut self) -> Result<Expr, Diagnostic> {
        let name = self.ident()?;
        if !self.is_sym("(") {
            self.height = 1;
            return Ok(Expr::Var(name));
        }
        self.next();
        let mut args = Vec::new();
        let mut h = 0;
        while !self.is_sym(")") {
            args.push(self.expr()?);
            h = h.max(self.height);
            if !self.is_sym(")") {
                self.expect_sym(",")?;
            }
        }
        self.expect_sym(")")?;
        self.node(h)?;
        Ok(Expr::Call(name, args))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(n: i64) -> Box<Expr> {
        Box::new(Expr::Int(n))
    }
    fn var(s: &str) -> Box<Expr> {
        Box::new(Expr::Var(s.into()))
    }

    #[test]
    fn parses_function_with_contracts_and_effects() {
        let p = parse_program("fn f(x: Int, b: Bool) -> Int !io req x >= 0 ens r > x = x + 1").unwrap();
        let Def::Fn(f) = &p.defs[0] else { panic!("expected fn") };
        assert_eq!(f.name, "f");
        assert_eq!(f.params, vec![Param { name: "x".into(), ty: Type::Int }, Param { name: "b".into(), ty: Type::Bool }]);
        assert_eq!(f.ret, Type::Int);
        assert_eq!(f.effects, vec!["io".to_string()]);
        assert_eq!(f.req, vec![Expr::Bin(BinOp::Ge, var("x"), int(0))]);
        assert_eq!(f.ens, vec![Expr::Bin(BinOp::Gt, var("r"), var("x"))]);
        assert_eq!(f.body, Expr::Bin(BinOp::Add, var("x"), int(1)));
    }

    #[test]
    fn respects_precedence_and_left_associativity() {
        let p = parse_program("test t = 1 + 2 * 3 - 4 == 3 || false").unwrap();
        let Def::Test(t) = &p.defs[0] else { panic!() };
        let sum = Expr::Bin(BinOp::Sub, Box::new(Expr::Bin(BinOp::Add, int(1), Box::new(Expr::Bin(BinOp::Mul, int(2), int(3))))), int(4));
        let cmp = Expr::Bin(BinOp::Eq, Box::new(sum), int(3));
        assert_eq!(t.body, Expr::Bin(BinOp::Or, Box::new(cmp), Box::new(Expr::Bool(false))));
    }

    #[test]
    fn parses_let_if_calls_unary_and_unit() {
        let p = parse_program("fn g() -> Unit = ()\ntest t = let y = -f(1, 2) in if !true then y else y").unwrap();
        assert_eq!(p.defs.len(), 2);
        let Def::Test(t) = &p.defs[1] else { panic!() };
        let call = Expr::Call("f".into(), vec![Expr::Int(1), Expr::Int(2)]);
        let neg = Expr::Un(UnOp::Neg, Box::new(call));
        let cond = Expr::Un(UnOp::Not, Box::new(Expr::Bool(true)));
        let iff = Expr::If(Box::new(cond), var("y"), var("y"));
        assert_eq!(t.body, Expr::Let("y".into(), Box::new(neg), Box::new(iff)));
    }

    #[test]
    fn reports_errors_with_position() {
        let e = parse_program("fn f(x: Int) -> Int = (x + 1").unwrap_err();
        assert_eq!(e.kind, "parse");
        assert_eq!(e.path, "1:29");
        assert!(e.message.contains("expected `)`"), "{}", e.message);
        let e = parse_program("fn if() -> Int = 1").unwrap_err();
        assert!(e.message.contains("expected a name"), "{}", e.message);
        let e = parse_program("test t = 1 < 2 < 3").unwrap_err();
        assert!(e.message.contains("chained"), "{}", e.message);
        let e = parse_program("fn f() -> Float = 1").unwrap_err();
        assert!(e.message.contains("expected a type"), "{}", e.message);
    }

    /// Debug-build parser frames are large; give deep-but-legal inputs more room than a 2 MiB test thread.
    fn parse_big_stack(src: String) -> Result<Program, Diagnostic> {
        std::thread::Builder::new().stack_size(32 << 20).spawn(move || parse_program(&src)).unwrap().join().unwrap()
    }

    fn rejects_deep(src: String) {
        let e = parse_big_stack(src).unwrap_err();
        assert_eq!(e.kind, "parse");
        assert!(e.message.contains("nested too deeply (max 256 levels)"), "{}", e.message);
    }

    #[test]
    fn rejects_deeply_nested_parens() {
        let n = 10_000;
        rejects_deep(format!("test t = {}1{}", "(".repeat(n), ")".repeat(n)));
    }

    #[test]
    fn rejects_long_operator_chain() {
        let mut src = String::from("test t = 1");
        for _ in 0..10_000 {
            src.push_str(" + 1");
        }
        rejects_deep(src);
    }

    #[test]
    fn rejects_long_unary_chain() {
        rejects_deep(format!("test t = {}1", "-".repeat(10_000)));
    }

    #[test]
    fn rejects_deeply_nested_lets() {
        rejects_deep(format!("test t = {}1", "let x = 1 in ".repeat(10_000)));
    }

    #[test]
    fn accepts_moderate_chains_and_nesting() {
        let mut src = String::from("test t = 1");
        for _ in 0..200 {
            src.push_str(" + 1");
        }
        parse_big_stack(src).unwrap();
        parse_big_stack(format!("test t = {}1{}", "(".repeat(200), ")".repeat(200))).unwrap();
    }

    #[test]
    fn empty_source_is_empty_program() {
        assert_eq!(parse_program("  # nothing\n").unwrap(), Program::default());
    }
}
