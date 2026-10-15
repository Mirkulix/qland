//! Recursive-descent parser: tokens -> Program.
//!
//! Grammar (Phase 2a). LOWER = name starting with a lowercase letter or `_`, UPPER = with an uppercase letter.
//!   program  := def*
//!   def      := "fn" LOWER tparams? "(" params ")" "->" type effect* contract* "=" expr
//!             | "test" LOWER "=" expr
//!             | "type" UPPER tparams? "=" ( "{" fields "}" | "|"? variant ("|" variant)* | "|" )   (a lone `|`: enum without variants)
//!   tparams  := "[" UPPER ("," UPPER)* ","? "]"                (type parameters)
//!   params   := (LOWER ":" type ("," LOWER ":" type)* ","?)?
//!   fields   := (LOWER ":" type ("," LOWER ":" type)* ","?)?
//!   variant  := UPPER ("(" type ("," type)* ","? ")")?
//!   type     := "Int" | "Float" | "Bool" | "Unit" | "Str" | "List" "[" type "]" | "Option" "[" type "]"
//!             | "Result" "[" type "," type "]" | "Map" "[" type "," type "]" | "Bytes"
//!             | UPPER | UPPER "[" type ("," type)* ","? "]" | "fn" "(" (type ("," type)*)? ")" "->" type effect*
//!             | "(" type ")"
//!   effect   := "!" LOWER                                    (effects are kept as a sorted set without duplicates)
//!   contract := ("req" | "ens") expr
//!   expr     := "let" LOWER (":" type)? "=" expr "in" expr
//!             | "if" expr "then" expr "else" expr
//!             | "match" expr "{" (arm ("," arm)* ","?)? "}"   (no `Name {` record literal directly in the scrutinee;
//!                                                             no arms only type-checks for an enum without variants)
//!             | "fn" "(" (LOWER (":" type)? ("," LOWER (":" type)?)* ","?)? ")" "=>" expr   (closure)
//!             | binary
//!   arm      := pattern ("if" expr)? "=>" expr
//!   pattern  := "_" | LOWER | "-"? INT | STR | "true" | "false" | "(" ")"     (`-9223372036854775808` is allowed here)
//!             | UPPER ("(" pattern ("," pattern)* ","? ")")? | UPPER "{" (LOWER (":" pattern)?),* "}"
//!   LOWER and UPPER of definitions, types, variants and references may be qualified standard
//!   library names `std.<module>.<name>` (one token; the last segment decides the case); the names
//!   of parameters, locals, fields, effects and type parameters may not.
//!   binary   := precedence climbing over BinOp::precedence (comparisons non-associative)
//!   unary    := ("-" | "!") unary | postfix
//!   postfix  := atom ("." LOWER)*
//!   atom     := INT | FLOAT | STR | "true" | "false" | "(" ")" | "(" expr ")" | "[" (expr ("," expr)* ","?)? "]"
//!             | "{" expr (";" expr)* "}" | UPPER "{" (LOWER ":" expr),* "}" | IDENT | IDENT "(" args ")"

use crate::lexer::{is_std_name, last_segment, lex, Tok, Token, KEYWORDS};
use qal_core::ast::*;
use qal_core::diag::Diagnostic;

/// Maximum AST height (and parser nesting). Every later stage only sees parser-built ASTs, so this
/// bounds all recursion over expressions, patterns and written types. Inferred types can still
/// nest deeper than the AST (each `let` may wrap the previous value); the checker caps those
/// separately (`type_too_deep`).
const MAX_DEPTH: usize = 256;

pub fn parse_program(src: &str) -> Result<Program, Diagnostic> {
    let tokens = lex(src)?;
    let mut p = Parser { tokens, pos: 0, depth: 0, height: 0, no_struct: false };
    let mut defs = Vec::new();
    while p.peek() != &Tok::Eof {
        defs.push(p.def()?);
    }
    Ok(Program { defs })
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    /// Current recursion depth of expr()/unary()/pattern()/ty().
    depth: usize,
    /// Height of the AST most recently built.
    height: usize,
    /// True while parsing a match scrutinee: `Name {` is not a record literal there.
    no_struct: bool,
}

/// The case of a name is the case of its last segment (`std.json.Str` is an uppercase name).
fn is_upper(s: &str) -> bool {
    last_segment(s).starts_with(|c: char| c.is_ascii_uppercase())
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.tokens[self.pos].tok
    }

    fn peek_at(&self, n: usize) -> &Tok {
        &self.tokens[(self.pos + n).min(self.tokens.len() - 1)].tok
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

    fn enter(&mut self) -> Result<(), Diagnostic> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.too_deep());
        }
        Ok(())
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
            Tok::Float(x) => format!("`{x:?}`"),
            Tok::Str(s) => format!("{s:?}"),
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

    /// A name for a function, test, parameter, local, field or effect.
    fn lower(&mut self, what: &str) -> Result<String, Diagnostic> {
        if matches!(self.peek(), Tok::Ident(s) if is_upper(s)) {
            return Err(self.err_here(format!("{what} names start with a lowercase letter, found {}", Self::describe(self.peek()))));
        }
        self.ident()
    }

    /// A name for a parameter, local, field or effect: never a qualified `std.` name.
    fn local(&mut self, what: &str) -> Result<String, Diagnostic> {
        self.unqualified(what)?;
        self.lower(what)
    }

    fn unqualified(&self, what: &str) -> Result<(), Diagnostic> {
        match self.peek() {
            Tok::Ident(s) if is_std_name(s) => Err(self.err_here(format!("{what} names cannot be qualified, found {}", Self::describe(self.peek())))
                .hint("only definitions (functions, tests, types and variants) have `std.<module>.<name>` names")),
            _ => Ok(()),
        }
    }

    /// A name for a type or variant.
    fn upper(&mut self, what: &str) -> Result<String, Diagnostic> {
        if matches!(self.peek(), Tok::Ident(s) if !is_upper(s)) {
            return Err(self.err_here(format!("{what} names start with an uppercase letter, found {}", Self::describe(self.peek()))));
        }
        self.ident()
    }

    /// Parses `open item (sep item)* sep? close`, calling `item` for each element.
    fn list_of<T>(&mut self, close: &str, mut item: impl FnMut(&mut Self) -> Result<T, Diagnostic>) -> Result<Vec<T>, Diagnostic> {
        let mut out = Vec::new();
        while !self.is_sym(close) {
            out.push(item(self)?);
            if !self.is_sym(close) {
                self.expect_sym(",")?;
            }
        }
        self.expect_sym(close)?;
        Ok(out)
    }

    fn ty(&mut self) -> Result<Type, Diagnostic> {
        self.enter()?;
        let t = self.ty_inner();
        self.depth -= 1;
        t
    }

    fn ty_inner(&mut self) -> Result<Type, Diagnostic> {
        let name = match self.peek().clone() {
            Tok::Sym("(") => {
                self.next();
                let t = self.ty()?;
                self.expect_sym(")")?;
                return Ok(t);
            }
            Tok::Ident(s) if s == "fn" => {
                self.next();
                self.expect_sym("(")?;
                let params = self.list_of(")", |p| p.ty())?;
                self.expect_sym("->")?;
                let ret = self.ty()?;
                let effects = self.effects()?;
                return Ok(Type::Fn(params, Box::new(ret), effects));
            }
            Tok::Ident(s) if is_upper(&s) => s,
            other => {
                let mut d = self.err_here(format!("expected a type (e.g. Int, Str, List[Int]), found {}", Self::describe(&other)));
                if let Some(h) = if let Tok::Ident(s) = &other { Type::foreign_name_hint(s) } else { None } {
                    d = d.hint(h);
                }
                return Err(d);
            }
        };
        self.next();
        let arg = |p: &mut Self| -> Result<Type, Diagnostic> {
            p.expect_sym("[")?;
            let t = p.ty()?;
            Ok(t)
        };
        Ok(match name.as_str() {
            "Int" => Type::Int,
            "Float" => Type::Float,
            "Bool" => Type::Bool,
            "Unit" => Type::Unit,
            "Str" => Type::Str,
            "List" | "Option" => {
                let t = arg(self)?;
                self.expect_sym("]")?;
                if name == "List" { Type::list(t) } else { Type::option(t) }
            }
            "Result" | "Map" => {
                let t = arg(self)?;
                self.expect_sym(",")?;
                let e = self.ty()?;
                self.expect_sym("]")?;
                if name == "Result" { Type::result(t, e) } else { Type::map(t, e) }
            }
            "Bytes" => Type::Bytes,
            _ if self.is_sym("[") => {
                self.next();
                let args = self.list_of("]", |p| p.ty())?;
                if args.is_empty() {
                    return Err(self.err_here(format!("`{name}[]` needs at least one type argument")));
                }
                Type::App(name, args)
            }
            _ => Type::Named(name),
        })
    }

    /// `!e` annotations as a sorted set: `!io !fs !io` is `["fs", "io"]`.
    fn effects(&mut self) -> Result<Vec<String>, Diagnostic> {
        let mut effects = Vec::new();
        while self.is_sym("!") {
            self.next();
            effects.push(self.local("effect")?);
        }
        effects.sort_unstable();
        effects.dedup();
        Ok(effects)
    }

    fn def(&mut self) -> Result<Def, Diagnostic> {
        if self.is_kw("fn") {
            self.next();
            self.fn_def().map(Def::Fn)
        } else if self.is_kw("test") {
            self.next();
            let name = self.lower("test")?;
            self.expect_sym("=")?;
            let body = self.expr()?;
            Ok(Def::Test(TestDef { name, body }))
        } else if self.is_kw("type") {
            self.next();
            self.type_def().map(Def::Type)
        } else {
            Err(self.err_here(format!("expected `fn`, `type` or `test`, found {}", Self::describe(self.peek()))))
        }
    }

    /// `[A, B]` after a definition name (none: empty).
    fn tparams(&mut self) -> Result<Vec<String>, Diagnostic> {
        if !self.is_sym("[") {
            return Ok(Vec::new());
        }
        self.next();
        let ps = self.list_of("]", |p| {
            p.unqualified("type parameter")?;
            p.upper("type parameter")
        })?;
        if ps.is_empty() {
            return Err(self.err_here("`[]` needs at least one type parameter"));
        }
        Ok(ps)
    }

    fn type_def(&mut self) -> Result<TypeDef, Diagnostic> {
        let name = self.upper("type")?;
        let params = self.tparams()?;
        self.expect_sym("=")?;
        if self.is_sym("{") {
            self.next();
            let fields = self.list_of("}", |p| {
                let f = p.local("field")?;
                p.expect_sym(":")?;
                Ok((f, p.ty()?))
            })?;
            return Ok(TypeDef { name, params, body: TypeBody::Record(fields) });
        }
        if self.is_sym("|") {
            self.next();
            // A lone `|` (followed by the next definition or the end) is an enum without variants.
            if self.peek() == &Tok::Eof || self.is_kw("fn") || self.is_kw("type") || self.is_kw("test") {
                return Ok(TypeDef { name, params, body: TypeBody::Enum(Vec::new()) });
            }
        }
        let mut variants = Vec::new();
        loop {
            let vname = self.upper("variant")?;
            let fields = if self.is_sym("(") {
                self.next();
                self.list_of(")", |p| p.ty())?
            } else {
                Vec::new()
            };
            variants.push(Variant { name: vname, fields });
            if !self.is_sym("|") {
                break;
            }
            self.next();
        }
        Ok(TypeDef { name, params, body: TypeBody::Enum(variants) })
    }

    fn fn_def(&mut self) -> Result<FnDef, Diagnostic> {
        let name = self.lower("function")?;
        let tparams = self.tparams()?;
        self.expect_sym("(")?;
        let params = self.list_of(")", |p| {
            let pname = p.local("parameter")?;
            p.expect_sym(":")?;
            Ok(Param { name: pname, ty: p.ty()? })
        })?;
        self.expect_sym("->")?;
        let ret = self.ty()?;
        let effects = self.effects()?;
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
        Ok(FnDef { name, tparams, params, ret, effects, req, ens, body })
    }

    fn expr(&mut self) -> Result<Expr, Diagnostic> {
        self.enter()?;
        let e = self.expr_inner();
        self.depth -= 1;
        e
    }

    fn expr_inner(&mut self) -> Result<Expr, Diagnostic> {
        if self.is_kw("let") {
            self.let_expr()
        } else if self.is_kw("if") {
            self.if_expr()
        } else if self.is_kw("match") {
            self.match_expr()
        } else {
            self.binary(1)
        }
    }

    // let/if/match/call/paren live in their own functions so the frames on the common recursion path stay small.
    fn let_expr(&mut self) -> Result<Expr, Diagnostic> {
        self.next();
        let name = self.local("variable")?;
        let ann = if self.is_sym(":") {
            self.next();
            Some(self.ty()?)
        } else {
            None
        };
        self.expect_sym("=")?;
        let value = self.expr()?;
        self.expect_kw("in")?;
        let value_h = self.height;
        let body = self.expr()?;
        self.node(value_h.max(self.height))?;
        Ok(Expr::Let(name, ann, Box::new(value), Box::new(body)))
    }

    fn if_expr(&mut self) -> Result<Expr, Diagnostic> {
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

    fn match_expr(&mut self) -> Result<Expr, Diagnostic> {
        self.next();
        let saved = std::mem::replace(&mut self.no_struct, true);
        let scrutinee = self.expr();
        self.no_struct = saved;
        let scrutinee = scrutinee?;
        let mut h = self.height;
        self.expect_sym("{")?;
        let arms = self.list_of("}", |p| {
            let pat = p.pattern()?;
            h = h.max(p.height);
            let guard = if p.is_kw("if") {
                p.next();
                let g = p.expr()?;
                h = h.max(p.height);
                Some(g)
            } else {
                None
            };
            p.expect_sym("=>")?;
            let body = p.expr()?;
            h = h.max(p.height);
            Ok(Arm { pat, guard, body })
        })?;
        self.node(h)?;
        Ok(Expr::Match(Box::new(scrutinee), arms))
    }

    fn pattern(&mut self) -> Result<Pat, Diagnostic> {
        self.enter()?;
        let p = self.pattern_inner();
        self.depth -= 1;
        p
    }

    fn pattern_inner(&mut self) -> Result<Pat, Diagnostic> {
        self.height = 1;
        match self.peek().clone() {
            Tok::Int(n) if n <= i64::MAX as u64 => {
                self.next();
                Ok(Pat::Int(n as i64))
            }
            Tok::Sym("-") if matches!(self.peek_at(1), Tok::Int(n) if *n <= 1 << 63) => {
                self.next();
                let n = match self.next().tok {
                    Tok::Int(n) => n,
                    _ => 0,
                };
                // `n <= 2^63` (checked by the guard), so the negation fits; `-2^63` is i64::MIN.
                Ok(Pat::Int(0i64.wrapping_sub(n as i64)))
            }
            Tok::Str(s) => {
                self.next();
                Ok(Pat::Str(s))
            }
            Tok::Sym("(") if self.peek_at(1) == &Tok::Sym(")") => {
                self.next();
                self.next();
                Ok(Pat::Unit)
            }
            Tok::Ident(s) if s == "true" || s == "false" => {
                self.next();
                Ok(Pat::Bool(s == "true"))
            }
            Tok::Ident(s) if s == "_" => {
                self.next();
                Ok(Pat::Wild)
            }
            Tok::Ident(s) if is_upper(&s) => {
                self.next();
                if self.is_sym("(") {
                    self.next();
                    let mut h = 0;
                    let ps = self.list_of(")", |p| {
                        let x = p.pattern()?;
                        h = h.max(p.height);
                        Ok(x)
                    })?;
                    self.node(h)?;
                    Ok(Pat::Variant(s, ps))
                } else if self.is_sym("{") {
                    self.next();
                    let mut h = 0;
                    let fs = self.list_of("}", |p| {
                        let f = p.local("field")?;
                        let x = if p.is_sym(":") {
                            p.next();
                            p.pattern()?
                        } else {
                            Pat::Bind(f.clone())
                        };
                        h = h.max(p.height);
                        Ok((f, x))
                    })?;
                    self.node(h)?;
                    Ok(Pat::Record(s, fs))
                } else {
                    Ok(Pat::Variant(s, vec![]))
                }
            }
            Tok::Ident(_) => Ok(Pat::Bind(self.local("variable")?)),
            other => Err(self.expected_pattern(&other)),
        }
    }

    fn peek_binop(&self) -> Option<BinOp> {
        let op = match self.peek() {
            Tok::Sym("+") => BinOp::Add,
            Tok::Sym("-") => BinOp::Sub,
            Tok::Sym("*") => BinOp::Mul,
            Tok::Sym("/") => BinOp::Div,
            Tok::Sym("%") => BinOp::Rem,
            Tok::Sym("++") => BinOp::Concat,
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
            if op.is_comparison() && self.peek_binop().is_some_and(BinOp::is_comparison) {
                return Err(self.chained());
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
            return self.postfix();
        };
        self.next();
        self.enter()?;
        let inner = self.unary()?;
        self.depth -= 1;
        self.node(self.height)?;
        Ok(Expr::Un(op, Box::new(inner)))
    }

    fn postfix(&mut self) -> Result<Expr, Diagnostic> {
        let mut e = self.atom()?;
        while self.is_sym(".") {
            self.next();
            let field = self.local("field")?;
            self.node(self.height)?;
            e = Expr::Field(Box::new(e), field);
        }
        Ok(e)
    }

    #[cold]
    fn expected_pattern(&self, found: &Tok) -> Diagnostic {
        match found {
            Tok::Int(n) => self.int_too_big(*n),
            other => self.err_here(format!("expected a pattern, found {}", Self::describe(other))),
        }
    }

    /// An integer literal token above `i64::MAX` (only `-9223372036854775808` as a pattern may use one).
    #[cold]
    fn int_too_big(&self, n: u64) -> Diagnostic {
        if n == 1 << 63 {
            self.err_here(format!("integer literal `{n}` does not fit in Int (64-bit); write `-9223372036854775807 - 1` for the minimum"))
        } else {
            self.err_here(format!("integer literal `{n}` does not fit in Int (64-bit)"))
        }
    }

    fn atom(&mut self) -> Result<Expr, Diagnostic> {
        self.height = 1;
        match self.peek().clone() {
            Tok::Int(n) if n <= i64::MAX as u64 => {
                self.next();
                Ok(Expr::Int(n as i64))
            }
            Tok::Float(x) => {
                self.next();
                Ok(Expr::Float(x))
            }
            Tok::Str(s) => {
                self.next();
                Ok(Expr::Str(s))
            }
            Tok::Ident(s) if s == "true" || s == "false" => {
                self.next();
                Ok(Expr::Bool(s == "true"))
            }
            Tok::Sym("(") => self.nested(Self::paren),
            Tok::Sym("[") => self.nested(Self::list_lit),
            Tok::Sym("{") => self.nested(Self::seq),
            Tok::Ident(s) if s == "fn" && self.peek_at(1) == &Tok::Sym("(") => self.nested(Self::lambda),
            Tok::Ident(s) if KEYWORDS.contains(&s.as_str()) => Err(self.expected_expr(&Tok::Ident(s))),
            Tok::Ident(s) if is_upper(&s) && !self.no_struct && self.peek_at(1) == &Tok::Sym("{") => self.nested(Self::record_lit),
            Tok::Ident(_) => self.nested(Self::name_or_call),
            other => Err(self.expected_expr(&other)),
        }
    }

    /// Runs `f` with record literals allowed again (inside delimiters the match-scrutinee restriction ends).
    fn nested(&mut self, f: fn(&mut Self) -> Result<Expr, Diagnostic>) -> Result<Expr, Diagnostic> {
        let saved = std::mem::replace(&mut self.no_struct, false);
        let e = f(self);
        self.no_struct = saved;
        e
    }

    #[cold]
    fn expected_expr(&self, found: &Tok) -> Diagnostic {
        if let Tok::Int(n) = found {
            return self.int_too_big(*n);
        }
        let mut msg = format!("expected an expression, found {}", Self::describe(found));
        if let Tok::Ident(k) = found {
            if k == "if" || k == "let" || k == "match" {
                msg.push_str(&format!(" (wrap the `{k}` expression in parentheses here)"));
            }
        }
        self.err_here(msg)
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

    fn list_lit(&mut self) -> Result<Expr, Diagnostic> {
        self.next();
        let mut h = 0;
        let items = self.list_of("]", |p| {
            let e = p.expr()?;
            h = h.max(p.height);
            Ok(e)
        })?;
        self.node(h)?;
        Ok(Expr::List(items))
    }

    fn seq(&mut self) -> Result<Expr, Diagnostic> {
        self.next();
        let mut items = vec![self.expr()?];
        let mut h = self.height;
        while self.is_sym(";") {
            self.next();
            items.push(self.expr()?);
            h = h.max(self.height);
        }
        self.expect_sym("}")?;
        self.node(h)?;
        Ok(Expr::Seq(items))
    }

    /// `fn(x, y: T) => body`; the body extends as far as possible (like `let`).
    fn lambda(&mut self) -> Result<Expr, Diagnostic> {
        self.next();
        self.expect_sym("(")?;
        let params = self.list_of(")", |p| {
            let n = p.local("parameter")?;
            let ann = if p.is_sym(":") {
                p.next();
                Some(p.ty()?)
            } else {
                None
            };
            Ok((n, ann))
        })?;
        self.expect_sym("=>")?;
        let body = self.expr()?;
        self.node(self.height)?;
        Ok(Expr::Lambda(params, Box::new(body)))
    }

    fn record_lit(&mut self) -> Result<Expr, Diagnostic> {
        let name = self.upper("type")?;
        self.expect_sym("{")?;
        let mut h = 0;
        let fields = self.list_of("}", |p| {
            let f = p.local("field")?;
            p.expect_sym(":")?;
            let e = p.expr()?;
            h = h.max(p.height);
            Ok((f, e))
        })?;
        self.node(h)?;
        Ok(Expr::Record(name, fields))
    }

    fn name_or_call(&mut self) -> Result<Expr, Diagnostic> {
        let name = self.ident()?;
        if !self.is_sym("(") {
            self.height = 1;
            return Ok(Expr::Var(name));
        }
        self.next();
        let mut h = 0;
        let args = self.list_of(")", |p| {
            let e = p.expr()?;
            h = h.max(p.height);
            Ok(e)
        })?;
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
    fn body(src: &str) -> Expr {
        let p = parse_program(&format!("test t = {src}")).unwrap();
        let Def::Test(t) = &p.defs[0] else { panic!() };
        t.body.clone()
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
        let sum = Expr::Bin(BinOp::Sub, Box::new(Expr::Bin(BinOp::Add, int(1), Box::new(Expr::Bin(BinOp::Mul, int(2), int(3))))), int(4));
        let cmp = Expr::Bin(BinOp::Eq, Box::new(sum), int(3));
        assert_eq!(body("1 + 2 * 3 - 4 == 3 || false"), Expr::Bin(BinOp::Or, Box::new(cmp), Box::new(Expr::Bool(false))));
        let s = |x: &str| Box::new(Expr::Str(x.into()));
        assert_eq!(body(r#""a" ++ "b" == "ab""#), Expr::Bin(BinOp::Eq, Box::new(Expr::Bin(BinOp::Concat, s("a"), s("b"))), s("ab")));
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
        assert_eq!(t.body, Expr::Let("y".into(), None, Box::new(neg), Box::new(iff)));
    }

    #[test]
    fn parses_type_definitions_and_type_syntax() {
        let p = parse_program(
            "type User = { id: Int, name: Str, }\n\
             type Err = Empty | NotNumber(Str) | Pair(Int, List[Float])\n\
             type One = | Only\n\
             fn f(u: User, k: fn(Int) -> Option[Int] !io, r2: Result[Unit, Str]) -> List[Err] = []",
        )
        .unwrap();
        assert_eq!(
            p.defs[0],
            Def::Type(TypeDef { name: "User".into(), params: vec![], body: TypeBody::Record(vec![("id".into(), Type::Int), ("name".into(), Type::Str)]) })
        );
        let Def::Type(TypeDef { body: TypeBody::Enum(vs), .. }) = &p.defs[1] else { panic!() };
        assert_eq!(vs.iter().map(|v| (v.name.as_str(), v.fields.len())).collect::<Vec<_>>(), vec![("Empty", 0), ("NotNumber", 1), ("Pair", 2)]);
        assert_eq!(vs[2].fields[1], Type::list(Type::Float));
        let Def::Type(TypeDef { body: TypeBody::Enum(one), .. }) = &p.defs[2] else { panic!() };
        assert_eq!(one.len(), 1);
        let Def::Fn(f) = &p.defs[3] else { panic!() };
        assert_eq!(f.params[1].ty, Type::Fn(vec![Type::Int], Box::new(Type::option(Type::Int)), vec!["io".into()]));
        assert_eq!(f.params[2].ty, Type::result(Type::Unit, Type::Str));
        assert_eq!(f.ret, Type::list(Type::Named("Err".into())));
    }

    #[test]
    fn parses_data_expressions() {
        assert_eq!(body("[1, 2,]"), Expr::List(vec![Expr::Int(1), Expr::Int(2)]));
        assert_eq!(body("[]"), Expr::List(vec![]));
        assert_eq!(body("{ print(\"a\"); 1 }"), Expr::Seq(vec![Expr::Call("print".into(), vec![Expr::Str("a".into())]), Expr::Int(1)]));
        assert_eq!(
            body("User { id: 1, name: \"a\" }.name"),
            Expr::Field(Box::new(Expr::Record("User".into(), vec![("id".into(), Expr::Int(1)), ("name".into(), Expr::Str("a".into()))])), "name".into())
        );
        assert_eq!(body("-a.b.c"), Expr::Un(UnOp::Neg, Box::new(Expr::Field(Box::new(Expr::Field(var("a"), "b".into())), "c".into()))));
        assert_eq!(body("let x: Option[Int] = None in x"), Expr::Let("x".into(), Some(Type::option(Type::Int)), var("None"), var("x")));
        assert_eq!(body("1.5"), Expr::Float(1.5));
    }

    #[test]
    fn parses_match_with_patterns_and_guards() {
        let e = body("match get(xs, 0) { None => 0, Some(n) if n < 0 => -1, Some(_) => 1, }");
        let Expr::Match(scrut, arms) = e else { panic!() };
        assert_eq!(*scrut, Expr::Call("get".into(), vec![Expr::Var("xs".into()), Expr::Int(0)]));
        assert_eq!(arms.len(), 3);
        assert_eq!(arms[0].pat, Pat::Variant("None".into(), vec![]));
        assert_eq!(arms[1].pat, Pat::Variant("Some".into(), vec![Pat::Bind("n".into())]));
        assert_eq!(arms[1].guard, Some(Expr::Bin(BinOp::Lt, var("n"), int(0))));
        assert_eq!(arms[2].pat, Pat::Variant("Some".into(), vec![Pat::Wild]));
        let Expr::Match(_, arms) = body("match u { User { id: 1, name } => name, _ => \"\" }") else { panic!() };
        assert_eq!(arms[0].pat, Pat::Record("User".into(), vec![("id".into(), Pat::Int(1)), ("name".into(), Pat::Bind("name".into()))]));
        let Expr::Match(_, arms) = body("match x { -1 => 0, \"s\" => 1, true => 2, () => 3 }") else { panic!() };
        let pats: Vec<Pat> = arms.into_iter().map(|a| a.pat).collect();
        assert_eq!(pats, vec![Pat::Int(-1), Pat::Str("s".into()), Pat::Bool(true), Pat::Unit]);
    }

    #[test]
    fn record_literal_is_not_parsed_in_match_scrutinee() {
        // `match Empty { ... }`: the braces open the arms, not a record literal.
        let Expr::Match(scrut, arms) = body("match Empty { Empty => 1, _ => 2 }") else { panic!() };
        assert_eq!(*scrut, Expr::Var("Empty".into()));
        assert_eq!(arms.len(), 2);
        // Inside parentheses record literals are allowed again.
        let Expr::Match(scrut, _) = body("match (P { x: 1 }) { _ => 1 }") else { panic!() };
        assert_eq!(*scrut, Expr::Record("P".into(), vec![("x".into(), Expr::Int(1))]));
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
        let e = parse_program("fn f() -> 3 = 1").unwrap_err();
        assert!(e.message.contains("expected a type"), "{}", e.message);
    }

    #[test]
    fn keyword_in_expression_position_is_expected_an_expression() {
        let e = parse_program("fn f() -> Int = then").unwrap_err();
        assert_eq!(e.message, "expected an expression, found `then`");
        let e = parse_program("test t = 1 + if true then 1 else 2 == 2").unwrap_err();
        assert!(e.message.starts_with("expected an expression, found `if` (wrap"), "{}", e.message);
    }

    #[test]
    fn enforces_name_case() {
        assert!(parse_program("fn F() -> Int = 1").unwrap_err().message.contains("function names start with a lowercase letter"));
        assert!(parse_program("type user = { a: Int }").unwrap_err().message.contains("type names start with an uppercase letter"));
        assert!(parse_program("type U = a | B").unwrap_err().message.contains("variant names start with an uppercase letter"));
        assert!(parse_program("test t = let X = 1 in X").unwrap_err().message.contains("variable names"));
        assert!(parse_program("test t = {}").unwrap_err().message.contains("expected an expression, found `}`"));
    }

    #[test]
    fn effects_are_a_sorted_set() {
        let p = parse_program("fn f(k: fn() -> Int !rand !io !rand) -> Unit !io !fs !io = ()").unwrap();
        let Def::Fn(f) = &p.defs[0] else { panic!() };
        assert_eq!(f.effects, vec!["fs".to_string(), "io".to_string()]);
        assert_eq!(f.params[0].ty, Type::Fn(vec![], Box::new(Type::Int), vec!["io".into(), "rand".into()]));
    }

    #[test]
    fn empty_enums_and_matches() {
        let p = parse_program("type Never = |\nfn absurd(n: Never) -> Int = match n { }\ntype Last = |").unwrap();
        assert_eq!(p.defs[0], Def::Type(TypeDef { name: "Never".into(), params: vec![], body: TypeBody::Enum(vec![]) }));
        let Def::Fn(f) = &p.defs[1] else { panic!() };
        assert_eq!(f.body, Expr::Match(var("n"), vec![]));
        assert_eq!(p.defs[2], Def::Type(TypeDef { name: "Last".into(), params: vec![], body: TypeBody::Enum(vec![]) }));
        // A leading `|` before variants still works, and a lowercase variant is still reported.
        assert!(parse_program("type U = | a").unwrap_err().message.contains("variant names start with an uppercase letter"));
    }

    #[test]
    fn integer_literal_bounds() {
        let Expr::Match(_, arms) = body("match x { -9223372036854775808 => 1, 9223372036854775807 => 2, _ => 3 }") else { panic!() };
        assert_eq!((arms[0].pat.clone(), arms[1].pat.clone()), (Pat::Int(i64::MIN), Pat::Int(i64::MAX)));
        let e = parse_program("test t = 9223372036854775808 == 1").unwrap_err();
        assert_eq!(e.path, "1:10");
        assert!(e.message.contains("does not fit in Int") && e.message.contains("-9223372036854775807 - 1"), "{}", e.message);
        assert!(parse_program("test t = match x { 9223372036854775808 => 1, _ => 2 }").unwrap_err().message.contains("does not fit"));
        assert!(parse_program("test t = match x { -9223372036854775809 => 1, _ => 2 }").is_err());
    }

    /// Debug-build parser frames are large; give deep-but-legal inputs more room than a 2 MiB test thread.
    fn parse_big_stack(src: String) -> Result<Program, Diagnostic> {
        std::thread::Builder::new().stack_size(64 << 20).spawn(move || parse_program(&src)).unwrap().join().unwrap()
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
    fn rejects_deeply_nested_data_and_patterns() {
        rejects_deep(format!("test t = {}1{}", "[".repeat(10_000), "]".repeat(10_000)));
        rejects_deep(format!("test t = {}1{}", "{".repeat(10_000), "}".repeat(10_000)));
        rejects_deep(format!("test t = x{}", ".f".repeat(10_000)));
        rejects_deep(format!("test t = match x {{ {}_{} => 1 }}", "Some(".repeat(10_000), ")".repeat(10_000)));
        rejects_deep(format!("fn f(x: {}Int{}) -> Int = 1", "List[".repeat(10_000), "]".repeat(10_000)));
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
    fn qualified_std_names_are_one_name() {
        let src = "type std.json.Json = std.json.Null | std.json.Str(Str) | std.json.Arr(List[std.json.Json])\n\
                   type std.route.Route[A] = { method: Str, tag: A }\n\
                   fn std.json.str(s: Str) -> std.json.Json = std.json.Str(s)\n\
                   fn std.route.tag[A](r: std.route.Route[A]) -> A = r.tag\n\
                   test std.json.t = match std.json.str(\"a\") { std.json.Str(x) => x == \"a\", std.json.Null => false, _ => false }";
        let p = parse_program(src).unwrap();
        let Def::Type(t) = &p.defs[0] else { panic!() };
        assert_eq!(t.name, "std.json.Json");
        let TypeBody::Enum(vs) = &t.body else { panic!() };
        assert_eq!(vs[0].name, "std.json.Null");
        assert_eq!(vs[2].fields, vec![Type::list(Type::Named("std.json.Json".into()))]);
        let Def::Fn(f) = &p.defs[2] else { panic!() };
        assert_eq!((f.name.as_str(), &f.ret), ("std.json.str", &Type::Named("std.json.Json".into())));
        assert_eq!(f.body, Expr::Call("std.json.Str".into(), vec![Expr::Var("s".into())]));
        let Def::Fn(g) = &p.defs[3] else { panic!() };
        assert_eq!(g.params[0].ty, Type::App("std.route.Route".into(), vec![Type::Named("A".into())]));
        let Def::Test(t) = &p.defs[4] else { panic!() };
        let Expr::Match(scrut, arms) = &t.body else { panic!() };
        assert_eq!(**scrut, Expr::Call("std.json.str".into(), vec![Expr::Str("a".into())]));
        assert_eq!(arms[0].pat, Pat::Variant("std.json.Str".into(), vec![Pat::Bind("x".into())]));
        assert_eq!(arms[1].pat, Pat::Variant("std.json.Null".into(), vec![]));
        // Round trip through the printer.
        let printed = crate::print_program(&p);
        assert!(printed.contains("std.json.Str(s)") && printed.contains("std.route.Route[A]"), "{printed}");
        assert_eq!(parse_program(&printed).unwrap(), p);
        // A field access after a qualified call, and a record literal of a qualified type.
        assert_eq!(body("std.a.f(1).x"), Expr::Field(Box::new(Expr::Call("std.a.f".into(), vec![Expr::Int(1)])), "x".into()));
        assert_eq!(body("std.a.P { x: 1 }"), Expr::Record("std.a.P".into(), vec![("x".into(), Expr::Int(1))]));
        let Expr::Match(_, arms) = body("match p { std.a.P { x } => x }") else { panic!() };
        assert_eq!(arms[0].pat, Pat::Record("std.a.P".into(), vec![("x".into(), Pat::Bind("x".into()))]));
    }

    #[test]
    fn std_is_a_keyword_and_locals_are_never_qualified() {
        for (src, msg) in [
            ("fn std() -> Int = 1", "expected a name, found `std`"),
            ("test t = let std = 1 in std", "expected a name, found `std`"),
            ("test t = r.std", "expected a name, found `std`"),
            ("test t = std . json . f(1)", "expected an expression, found `std`"),
            ("fn f(std.a.x: Int) -> Int = 1", "parameter names cannot be qualified"),
            ("test t = let std.a.x = 1 in 2", "variable names cannot be qualified"),
            ("type T = { std.a.x: Int }", "field names cannot be qualified"),
            ("fn f() -> Int !std.a.io = 1", "effect names cannot be qualified"),
            ("fn f[std.a.T](x: Int) -> Int = 1", "type parameter names cannot be qualified"),
            ("test t = match 1 { std.a.x => 1 }", "variable names cannot be qualified"),
            ("fn std.a.F() -> Int = 1", "function names start with a lowercase letter"),
            ("type std.a.t = { x: Int }", "type names start with an uppercase letter"),
        ] {
            let e = parse_program(src).unwrap_err();
            assert!(e.message.contains(msg), "{src}: {}", e.message);
        }
    }

    #[test]
    fn empty_source_is_empty_program() {
        assert_eq!(parse_program("  # nothing\n").unwrap(), Program::default());
    }
}
