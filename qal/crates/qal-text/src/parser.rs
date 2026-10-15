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

pub fn parse_program(src: &str) -> Result<Program, Diagnostic> {
    let tokens = lex(src)?;
    let mut p = Parser { tokens, pos: 0 };
    let mut defs = Vec::new();
    while p.peek() != &Tok::Eof {
        defs.push(p.def()?);
    }
    Ok(Program { defs })
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
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
        if self.is_kw("let") {
            self.next();
            let name = self.ident()?;
            self.expect_sym("=")?;
            let value = self.expr()?;
            self.expect_kw("in")?;
            let body = self.expr()?;
            return Ok(Expr::Let(name, Box::new(value), Box::new(body)));
        }
        if self.is_kw("if") {
            self.next();
            let c = self.expr()?;
            self.expect_kw("then")?;
            let t = self.expr()?;
            self.expect_kw("else")?;
            let e = self.expr()?;
            return Ok(Expr::If(Box::new(c), Box::new(t), Box::new(e)));
        }
        self.binary(1)
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
            let rhs = self.binary(prec + 1)?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
            if prec == 3 {
                if let Some(next) = self.peek_binop() {
                    if next.precedence() == 3 {
                        return Err(self.err_here("comparisons cannot be chained; use `&&`"));
                    }
                }
            }
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<Expr, Diagnostic> {
        if self.is_sym("-") {
            self.next();
            return Ok(Expr::Un(UnOp::Neg, Box::new(self.unary()?)));
        }
        if self.is_sym("!") {
            self.next();
            return Ok(Expr::Un(UnOp::Not, Box::new(self.unary()?)));
        }
        self.atom()
    }

    fn atom(&mut self) -> Result<Expr, Diagnostic> {
        match self.peek().clone() {
            Tok::Int(n) => {
                self.next();
                Ok(Expr::Int(n))
            }
            Tok::Ident(s) if s == "true" => {
                self.next();
                Ok(Expr::Bool(true))
            }
            Tok::Ident(s) if s == "false" => {
                self.next();
                Ok(Expr::Bool(false))
            }
            Tok::Sym("(") => {
                self.next();
                if self.is_sym(")") {
                    self.next();
                    return Ok(Expr::Unit);
                }
                let e = self.expr()?;
                self.expect_sym(")")?;
                Ok(e)
            }
            Tok::Ident(_) => {
                let name = self.ident()?;
                if self.is_sym("(") {
                    self.next();
                    let mut args = Vec::new();
                    while !self.is_sym(")") {
                        args.push(self.expr()?);
                        if !self.is_sym(")") {
                            self.expect_sym(",")?;
                        }
                    }
                    self.expect_sym(")")?;
                    Ok(Expr::Call(name, args))
                } else {
                    Ok(Expr::Var(name))
                }
            }
            other => Err(self.err_here(format!("expected an expression, found {}", Self::describe(&other)))),
        }
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

    #[test]
    fn empty_source_is_empty_program() {
        assert_eq!(parse_program("  # nothing\n").unwrap(), Program::default());
    }
}
