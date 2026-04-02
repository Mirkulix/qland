//! QLANG Virtual Machine — Stack-based interpreter for general-purpose programming.
//!
//! This module turns QLANG from a graph-only ML language into a complete
//! programming language with variables, arithmetic, conditionals, loops,
//! functions, arrays, and more.

use std::collections::HashMap;
use std::fmt;

// ─── Value type ─────────────────────────────────────────────────────────────

/// Runtime value in the QLANG VM.
#[derive(Debug, Clone)]
pub enum Value {
    Number(f64),
    Bool(bool),
    String(String),
    Array(Vec<f64>),
    Tensor(Vec<f64>, Vec<usize>),
    Null,
}

impl Value {
    pub fn as_number(&self) -> Result<f64, VmError> {
        match self {
            Value::Number(n) => Ok(*n),
            Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
            other => Err(VmError::TypeError(format!("expected number, got {}", other.type_name()))),
        }
    }

    pub fn as_bool(&self) -> Result<bool, VmError> {
        match self {
            Value::Bool(b) => Ok(*b),
            Value::Number(n) => Ok(*n != 0.0),
            other => Err(VmError::TypeError(format!("expected bool, got {}", other.type_name()))),
        }
    }

    pub fn as_array(&self) -> Result<&Vec<f64>, VmError> {
        match self {
            Value::Array(a) => Ok(a),
            other => Err(VmError::TypeError(format!("expected array, got {}", other.type_name()))),
        }
    }

    fn type_name(&self) -> &'static str {
        match self {
            Value::Number(_) => "number",
            Value::Bool(_) => "bool",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Tensor(_, _) => "tensor",
            Value::Null => "null",
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Number(n) => {
                if *n == (*n as i64) as f64 && n.abs() < 1e15 {
                    write!(f, "{}", *n as i64)
                } else {
                    write!(f, "{n}")
                }
            }
            Value::Bool(b) => write!(f, "{b}"),
            Value::String(s) => write!(f, "{s}"),
            Value::Array(a) => {
                write!(f, "[")?;
                for (i, v) in a.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    if *v == (*v as i64) as f64 && v.abs() < 1e15 {
                        write!(f, "{}", *v as i64)?;
                    } else {
                        write!(f, "{v}")?;
                    }
                }
                write!(f, "]")
            }
            Value::Tensor(data, shape) => write!(f, "tensor({data:?}, shape={shape:?})"),
            Value::Null => write!(f, "null"),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Number(a), Value::Number(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Array(a), Value::Array(b)) => a == b,
            (Value::Null, Value::Null) => true,
            _ => false,
        }
    }
}

// ─── Errors ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum VmError {
    UndefinedVariable(String),
    UndefinedFunction(String),
    TypeError(String),
    DivisionByZero,
    IndexOutOfBounds { index: usize, len: usize },
    ArityMismatch { expected: usize, got: usize },
    ParseError(String),
    RuntimeError(String),
}

impl fmt::Display for VmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VmError::UndefinedVariable(name) => write!(f, "undefined variable: {name}"),
            VmError::UndefinedFunction(name) => write!(f, "undefined function: {name}"),
            VmError::TypeError(msg) => write!(f, "type error: {msg}"),
            VmError::DivisionByZero => write!(f, "division by zero"),
            VmError::IndexOutOfBounds { index, len } => {
                write!(f, "index {index} out of bounds for array of length {len}")
            }
            VmError::ArityMismatch { expected, got } => {
                write!(f, "expected {expected} arguments, got {got}")
            }
            VmError::ParseError(msg) => write!(f, "parse error: {msg}"),
            VmError::RuntimeError(msg) => write!(f, "runtime error: {msg}"),
        }
    }
}

impl std::error::Error for VmError {}

// ─── Tokens ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literals
    NumberLit(f64),
    StringLit(String),
    Ident(String),
    BoolLit(bool),

    // Keywords
    Let,
    Fn,
    If,
    Else,
    While,
    For,
    In,
    Return,
    Print,
    And,
    Or,
    Not,

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Eq,        // =
    EqEq,      // ==
    BangEq,    // !=
    Lt,        // <
    Gt,        // >
    LtEq,      // <=
    GtEq,      // >=

    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    DotDot,    // ..
    Semicolon,

    Eof,
}

// ─── Lexer ──────────────────────────────────────────────────────────────────

pub fn tokenize(source: &str) -> Result<Vec<Token>, VmError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = source.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];

        // Skip whitespace
        if ch.is_whitespace() {
            i += 1;
            continue;
        }

        // Skip line comments
        if ch == '/' && i + 1 < chars.len() && chars[i + 1] == '/' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }

        // Numbers
        if ch.is_ascii_digit() || (ch == '.' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit()) {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            // Consume decimal point only if followed by a digit (not '..' range)
            if i < chars.len() && chars[i] == '.'
                && i + 1 < chars.len() && chars[i + 1] != '.' && chars[i + 1].is_ascii_digit()
            {
                i += 1; // consume '.'
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let s: String = chars[start..i].iter().collect();
            let n: f64 = s.parse().map_err(|_| VmError::ParseError(format!("invalid number: {s}")))?;
            tokens.push(Token::NumberLit(n));
            continue;
        }

        // Strings
        if ch == '"' {
            i += 1;
            let start = i;
            while i < chars.len() && chars[i] != '"' {
                i += 1;
            }
            if i >= chars.len() {
                return Err(VmError::ParseError("unterminated string".into()));
            }
            let s: String = chars[start..i].iter().collect();
            tokens.push(Token::StringLit(s));
            i += 1; // skip closing "
            continue;
        }

        // Identifiers and keywords
        if ch.is_alphabetic() || ch == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let tok = match word.as_str() {
                "let" => Token::Let,
                "fn" => Token::Fn,
                "if" => Token::If,
                "else" => Token::Else,
                "while" => Token::While,
                "for" => Token::For,
                "in" => Token::In,
                "return" => Token::Return,
                "print" => Token::Print,
                "and" => Token::And,
                "or" => Token::Or,
                "not" => Token::Not,
                "true" => Token::BoolLit(true),
                "false" => Token::BoolLit(false),
                _ => Token::Ident(word),
            };
            tokens.push(tok);
            continue;
        }

        // Two-character operators
        if i + 1 < chars.len() {
            let two: String = chars[i..i + 2].iter().collect();
            match two.as_str() {
                "==" => { tokens.push(Token::EqEq); i += 2; continue; }
                "!=" => { tokens.push(Token::BangEq); i += 2; continue; }
                "<=" => { tokens.push(Token::LtEq); i += 2; continue; }
                ">=" => { tokens.push(Token::GtEq); i += 2; continue; }
                ".." => { tokens.push(Token::DotDot); i += 2; continue; }
                _ => {}
            }
        }

        // Single-character tokens
        let tok = match ch {
            '+' => Token::Plus,
            '-' => Token::Minus,
            '*' => Token::Star,
            '/' => Token::Slash,
            '=' => Token::Eq,
            '<' => Token::Lt,
            '>' => Token::Gt,
            '(' => Token::LParen,
            ')' => Token::RParen,
            '{' => Token::LBrace,
            '}' => Token::RBrace,
            '[' => Token::LBracket,
            ']' => Token::RBracket,
            ',' => Token::Comma,
            ';' => Token::Semicolon,
            _ => return Err(VmError::ParseError(format!("unexpected character: '{ch}'"))),
        };
        tokens.push(tok);
        i += 1;
    }

    tokens.push(Token::Eof);
    Ok(tokens)
}

// ─── AST ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Expr {
    NumberLit(f64),
    BoolLit(bool),
    StringLit(String),
    ArrayLit(Vec<Expr>),
    Var(String),
    BinOp { op: BinOp, left: Box<Expr>, right: Box<Expr> },
    UnaryOp { op: UnaryOp, operand: Box<Expr> },
    Call { name: String, args: Vec<Expr> },
    Index { array: Box<Expr>, index: Box<Expr> },
}

#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    Add, Sub, Mul, Div,
    Eq, Ne, Lt, Gt, Le, Ge,
    And, Or,
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Neg, Not,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let { name: String, value: Expr },
    Assign { name: String, value: Expr },
    If { cond: Expr, then_body: Vec<Stmt>, else_body: Vec<Stmt> },
    While { cond: Expr, body: Vec<Stmt> },
    For { var: String, start: Expr, end: Expr, body: Vec<Stmt> },
    FnDef { name: String, params: Vec<String>, body: Vec<Stmt> },
    Return(Expr),
    Print(Expr),
    ExprStmt(Expr),
}

// ─── Parser ─────────────────────────────────────────────────────────────────

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> Token {
        let tok = self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof);
        self.pos += 1;
        tok
    }

    fn expect(&mut self, expected: &Token) -> Result<(), VmError> {
        let tok = self.advance();
        if &tok == expected {
            Ok(())
        } else {
            Err(VmError::ParseError(format!("expected {expected:?}, got {tok:?}")))
        }
    }

    fn at(&self, expected: &Token) -> bool {
        self.peek() == expected
    }

    fn parse_program(&mut self) -> Result<Vec<Stmt>, VmError> {
        let mut stmts = Vec::new();
        while !self.at(&Token::Eof) {
            stmts.push(self.parse_stmt()?);
        }
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, VmError> {
        match self.peek().clone() {
            Token::Let => self.parse_let(),
            Token::If => self.parse_if(),
            Token::While => self.parse_while(),
            Token::For => self.parse_for(),
            Token::Fn => self.parse_fn_def(),
            Token::Return => self.parse_return(),
            Token::Print => self.parse_print(),
            Token::Ident(_) => {
                // Could be assignment (x = ...) or expression statement (foo(...))
                // Look ahead for '='
                if self.pos + 1 < self.tokens.len() && self.tokens[self.pos + 1] == Token::Eq {
                    // Check it's not == (comparison)
                    if self.pos + 2 < self.tokens.len() && self.tokens[self.pos + 2] != Token::Eq {
                        return self.parse_assign();
                    }
                }
                let expr = self.parse_expr()?;
                Ok(Stmt::ExprStmt(expr))
            }
            _ => {
                let expr = self.parse_expr()?;
                Ok(Stmt::ExprStmt(expr))
            }
        }
    }

    fn parse_let(&mut self) -> Result<Stmt, VmError> {
        self.advance(); // consume 'let'
        let name = match self.advance() {
            Token::Ident(n) => n,
            other => return Err(VmError::ParseError(format!("expected identifier after 'let', got {other:?}"))),
        };
        self.expect(&Token::Eq)?;
        let value = self.parse_expr()?;
        Ok(Stmt::Let { name, value })
    }

    fn parse_assign(&mut self) -> Result<Stmt, VmError> {
        let name = match self.advance() {
            Token::Ident(n) => n,
            other => return Err(VmError::ParseError(format!("expected identifier, got {other:?}"))),
        };
        self.expect(&Token::Eq)?;
        let value = self.parse_expr()?;
        Ok(Stmt::Assign { name, value })
    }

    fn parse_if(&mut self) -> Result<Stmt, VmError> {
        self.advance(); // consume 'if'
        let cond = self.parse_expr()?;
        let then_body = self.parse_block()?;
        let else_body = if self.at(&Token::Else) {
            self.advance();
            if self.at(&Token::If) {
                vec![self.parse_if()?]
            } else {
                self.parse_block()?
            }
        } else {
            vec![]
        };
        Ok(Stmt::If { cond, then_body, else_body })
    }

    fn parse_while(&mut self) -> Result<Stmt, VmError> {
        self.advance(); // consume 'while'
        let cond = self.parse_expr()?;
        let body = self.parse_block()?;
        Ok(Stmt::While { cond, body })
    }

    fn parse_for(&mut self) -> Result<Stmt, VmError> {
        self.advance(); // consume 'for'
        let var = match self.advance() {
            Token::Ident(n) => n,
            other => return Err(VmError::ParseError(format!("expected identifier after 'for', got {other:?}"))),
        };
        self.expect(&Token::In)?;
        let start = self.parse_unary()?;
        self.expect(&Token::DotDot)?;
        let end = self.parse_unary()?;
        let body = self.parse_block()?;
        Ok(Stmt::For { var, start, end, body })
    }

    fn parse_fn_def(&mut self) -> Result<Stmt, VmError> {
        self.advance(); // consume 'fn'
        let name = match self.advance() {
            Token::Ident(n) => n,
            other => return Err(VmError::ParseError(format!("expected function name, got {other:?}"))),
        };
        self.expect(&Token::LParen)?;
        let mut params = Vec::new();
        while !self.at(&Token::RParen) {
            if !params.is_empty() {
                self.expect(&Token::Comma)?;
            }
            match self.advance() {
                Token::Ident(p) => params.push(p),
                other => return Err(VmError::ParseError(format!("expected parameter name, got {other:?}"))),
            }
        }
        self.expect(&Token::RParen)?;
        let body = self.parse_block()?;
        Ok(Stmt::FnDef { name, params, body })
    }

    fn parse_return(&mut self) -> Result<Stmt, VmError> {
        self.advance(); // consume 'return'
        let value = self.parse_expr()?;
        Ok(Stmt::Return(value))
    }

    fn parse_print(&mut self) -> Result<Stmt, VmError> {
        self.advance(); // consume 'print'
        self.expect(&Token::LParen)?;
        let value = self.parse_expr()?;
        self.expect(&Token::RParen)?;
        Ok(Stmt::Print(value))
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, VmError> {
        self.expect(&Token::LBrace)?;
        let mut stmts = Vec::new();
        while !self.at(&Token::RBrace) && !self.at(&Token::Eof) {
            stmts.push(self.parse_stmt()?);
        }
        self.expect(&Token::RBrace)?;
        Ok(stmts)
    }

    // Expression parsing with precedence climbing

    fn parse_expr(&mut self) -> Result<Expr, VmError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, VmError> {
        let mut left = self.parse_and()?;
        while self.at(&Token::Or) {
            self.advance();
            let right = self.parse_and()?;
            left = Expr::BinOp { op: BinOp::Or, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, VmError> {
        let mut left = self.parse_equality()?;
        while self.at(&Token::And) {
            self.advance();
            let right = self.parse_equality()?;
            left = Expr::BinOp { op: BinOp::And, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_equality(&mut self) -> Result<Expr, VmError> {
        let mut left = self.parse_comparison()?;
        loop {
            let op = match self.peek() {
                Token::EqEq => BinOp::Eq,
                Token::BangEq => BinOp::Ne,
                _ => break,
            };
            self.advance();
            let right = self.parse_comparison()?;
            left = Expr::BinOp { op, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<Expr, VmError> {
        let mut left = self.parse_additive()?;
        loop {
            let op = match self.peek() {
                Token::Lt => BinOp::Lt,
                Token::Gt => BinOp::Gt,
                Token::LtEq => BinOp::Le,
                Token::GtEq => BinOp::Ge,
                _ => break,
            };
            self.advance();
            let right = self.parse_additive()?;
            left = Expr::BinOp { op, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_additive(&mut self) -> Result<Expr, VmError> {
        let mut left = self.parse_multiplicative()?;
        loop {
            let op = match self.peek() {
                Token::Plus => BinOp::Add,
                Token::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let right = self.parse_multiplicative()?;
            left = Expr::BinOp { op, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, VmError> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                _ => break,
            };
            self.advance();
            let right = self.parse_unary()?;
            left = Expr::BinOp { op, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, VmError> {
        match self.peek().clone() {
            Token::Minus => {
                self.advance();
                let operand = self.parse_unary()?;
                Ok(Expr::UnaryOp { op: UnaryOp::Neg, operand: Box::new(operand) })
            }
            Token::Not => {
                self.advance();
                let operand = self.parse_unary()?;
                Ok(Expr::UnaryOp { op: UnaryOp::Not, operand: Box::new(operand) })
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr, VmError> {
        let mut expr = self.parse_primary()?;
        // Handle indexing: expr[index]
        while self.at(&Token::LBracket) {
            self.advance();
            let index = self.parse_expr()?;
            self.expect(&Token::RBracket)?;
            expr = Expr::Index { array: Box::new(expr), index: Box::new(index) };
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, VmError> {
        match self.peek().clone() {
            Token::NumberLit(n) => {
                self.advance();
                Ok(Expr::NumberLit(n))
            }
            Token::BoolLit(b) => {
                self.advance();
                Ok(Expr::BoolLit(b))
            }
            Token::StringLit(s) => {
                self.advance();
                Ok(Expr::StringLit(s))
            }
            Token::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(expr)
            }
            Token::LBracket => {
                self.advance();
                let mut elems = Vec::new();
                while !self.at(&Token::RBracket) && !self.at(&Token::Eof) {
                    if !elems.is_empty() {
                        self.expect(&Token::Comma)?;
                    }
                    elems.push(self.parse_expr()?);
                }
                self.expect(&Token::RBracket)?;
                Ok(Expr::ArrayLit(elems))
            }
            Token::Ident(name) => {
                self.advance();
                // Function call?
                if self.at(&Token::LParen) {
                    self.advance();
                    let mut args = Vec::new();
                    while !self.at(&Token::RParen) && !self.at(&Token::Eof) {
                        if !args.is_empty() {
                            self.expect(&Token::Comma)?;
                        }
                        args.push(self.parse_expr()?);
                    }
                    self.expect(&Token::RParen)?;
                    Ok(Expr::Call { name, args })
                } else {
                    Ok(Expr::Var(name))
                }
            }
            other => Err(VmError::ParseError(format!("unexpected token: {other:?}"))),
        }
    }
}

pub fn parse_program(tokens: &[Token]) -> Result<Vec<Stmt>, VmError> {
    let mut parser = Parser::new(tokens.to_vec());
    parser.parse_program()
}

// ─── VM State ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct FnDef {
    params: Vec<String>,
    body: Vec<Stmt>,
}

/// Signal used to propagate return values up the call stack.
enum ExecSignal {
    None,
    Return(Value),
}

pub struct VmState {
    /// Stack of variable scopes (innermost last).
    scopes: Vec<HashMap<String, Value>>,
    /// User-defined functions.
    functions: HashMap<String, FnDef>,
    /// Captured print output (for testing).
    pub output: Vec<String>,
}

impl VmState {
    fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
            functions: HashMap::new(),
            output: Vec::new(),
        }
    }

    fn get_var(&self, name: &str) -> Result<Value, VmError> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.get(name) {
                return Ok(val.clone());
            }
        }
        Err(VmError::UndefinedVariable(name.to_string()))
    }

    fn set_var(&mut self, name: &str, value: Value) {
        // Set in the innermost scope that already has it, or the current scope
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), value);
                return;
            }
        }
        // New variable in current scope
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), value);
        }
    }

    fn declare_var(&mut self, name: &str, value: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), value);
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn exec_stmts(&mut self, stmts: &[Stmt]) -> Result<ExecSignal, VmError> {
        for stmt in stmts {
            match self.exec_stmt(stmt)? {
                ExecSignal::Return(v) => return Ok(ExecSignal::Return(v)),
                ExecSignal::None => {}
            }
        }
        Ok(ExecSignal::None)
    }

    fn exec_stmt(&mut self, stmt: &Stmt) -> Result<ExecSignal, VmError> {
        match stmt {
            Stmt::Let { name, value } => {
                let val = self.eval_expr(value)?;
                self.declare_var(name, val);
                Ok(ExecSignal::None)
            }
            Stmt::Assign { name, value } => {
                // Verify variable exists somewhere
                let _ = self.get_var(name)?;
                let val = self.eval_expr(value)?;
                self.set_var(name, val);
                Ok(ExecSignal::None)
            }
            Stmt::If { cond, then_body, else_body } => {
                let c = self.eval_expr(cond)?;
                if c.as_bool()? {
                    self.push_scope();
                    let sig = self.exec_stmts(then_body)?;
                    self.pop_scope();
                    Ok(sig)
                } else {
                    self.push_scope();
                    let sig = self.exec_stmts(else_body)?;
                    self.pop_scope();
                    Ok(sig)
                }
            }
            Stmt::While { cond, body } => {
                let mut iterations = 0u64;
                loop {
                    let c = self.eval_expr(cond)?;
                    if !c.as_bool()? {
                        break;
                    }
                    self.push_scope();
                    match self.exec_stmts(body)? {
                        ExecSignal::Return(v) => {
                            self.pop_scope();
                            return Ok(ExecSignal::Return(v));
                        }
                        ExecSignal::None => {}
                    }
                    self.pop_scope();
                    iterations += 1;
                    if iterations > 1_000_000 {
                        return Err(VmError::RuntimeError("loop exceeded 1000000 iterations".into()));
                    }
                }
                Ok(ExecSignal::None)
            }
            Stmt::For { var, start, end, body } => {
                let s = self.eval_expr(start)?.as_number()? as i64;
                let e = self.eval_expr(end)?.as_number()? as i64;
                for i in s..e {
                    self.push_scope();
                    self.declare_var(var, Value::Number(i as f64));
                    match self.exec_stmts(body)? {
                        ExecSignal::Return(v) => {
                            self.pop_scope();
                            return Ok(ExecSignal::Return(v));
                        }
                        ExecSignal::None => {}
                    }
                    self.pop_scope();
                }
                Ok(ExecSignal::None)
            }
            Stmt::FnDef { name, params, body } => {
                self.functions.insert(name.clone(), FnDef {
                    params: params.clone(),
                    body: body.clone(),
                });
                Ok(ExecSignal::None)
            }
            Stmt::Return(expr) => {
                let val = self.eval_expr(expr)?;
                Ok(ExecSignal::Return(val))
            }
            Stmt::Print(expr) => {
                let val = self.eval_expr(expr)?;
                let s = format!("{val}");
                self.output.push(s);
                Ok(ExecSignal::None)
            }
            Stmt::ExprStmt(expr) => {
                self.eval_expr(expr)?;
                Ok(ExecSignal::None)
            }
        }
    }

    fn eval_expr(&mut self, expr: &Expr) -> Result<Value, VmError> {
        match expr {
            Expr::NumberLit(n) => Ok(Value::Number(*n)),
            Expr::BoolLit(b) => Ok(Value::Bool(*b)),
            Expr::StringLit(s) => Ok(Value::String(s.clone())),
            Expr::ArrayLit(elems) => {
                let mut vals = Vec::with_capacity(elems.len());
                for e in elems {
                    vals.push(self.eval_expr(e)?.as_number()?);
                }
                Ok(Value::Array(vals))
            }
            Expr::Var(name) => self.get_var(name),
            Expr::BinOp { op, left, right } => {
                let lv = self.eval_expr(left)?;
                // Short-circuit for logical ops
                match op {
                    BinOp::And => {
                        if !lv.as_bool()? {
                            return Ok(Value::Bool(false));
                        }
                        let rv = self.eval_expr(right)?;
                        return Ok(Value::Bool(rv.as_bool()?));
                    }
                    BinOp::Or => {
                        if lv.as_bool()? {
                            return Ok(Value::Bool(true));
                        }
                        let rv = self.eval_expr(right)?;
                        return Ok(Value::Bool(rv.as_bool()?));
                    }
                    _ => {}
                }
                let rv = self.eval_expr(right)?;
                let a = lv.as_number()?;
                let b = rv.as_number()?;
                match op {
                    BinOp::Add => Ok(Value::Number(a + b)),
                    BinOp::Sub => Ok(Value::Number(a - b)),
                    BinOp::Mul => Ok(Value::Number(a * b)),
                    BinOp::Div => {
                        if b == 0.0 {
                            return Err(VmError::DivisionByZero);
                        }
                        Ok(Value::Number(a / b))
                    }
                    BinOp::Eq => Ok(Value::Bool(a == b)),
                    BinOp::Ne => Ok(Value::Bool(a != b)),
                    BinOp::Lt => Ok(Value::Bool(a < b)),
                    BinOp::Gt => Ok(Value::Bool(a > b)),
                    BinOp::Le => Ok(Value::Bool(a <= b)),
                    BinOp::Ge => Ok(Value::Bool(a >= b)),
                    BinOp::And | BinOp::Or => unreachable!(),
                }
            }
            Expr::UnaryOp { op, operand } => {
                let v = self.eval_expr(operand)?;
                match op {
                    UnaryOp::Neg => Ok(Value::Number(-v.as_number()?)),
                    UnaryOp::Not => Ok(Value::Bool(!v.as_bool()?)),
                }
            }
            Expr::Call { name, args } => {
                let evaluated_args: Vec<Value> = args
                    .iter()
                    .map(|a| self.eval_expr(a))
                    .collect::<Result<_, _>>()?;

                // Built-in functions
                match name.as_str() {
                    "len" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let arr = evaluated_args[0].as_array()?;
                        return Ok(Value::Number(arr.len() as f64));
                    }
                    "sqrt" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let n = evaluated_args[0].as_number()?;
                        return Ok(Value::Number(n.sqrt()));
                    }
                    "abs" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let n = evaluated_args[0].as_number()?;
                        return Ok(Value::Number(n.abs()));
                    }
                    "floor" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let n = evaluated_args[0].as_number()?;
                        return Ok(Value::Number(n.floor()));
                    }
                    "push" => {
                        if evaluated_args.len() != 2 {
                            return Err(VmError::ArityMismatch { expected: 2, got: evaluated_args.len() });
                        }
                        let mut arr = evaluated_args[0].as_array()?.clone();
                        let val = evaluated_args[1].as_number()?;
                        arr.push(val);
                        return Ok(Value::Array(arr));
                    }
                    _ => {}
                }

                // User-defined functions
                let fndef = self.functions.get(name)
                    .ok_or_else(|| VmError::UndefinedFunction(name.clone()))?
                    .clone();

                if evaluated_args.len() != fndef.params.len() {
                    return Err(VmError::ArityMismatch {
                        expected: fndef.params.len(),
                        got: evaluated_args.len(),
                    });
                }

                self.push_scope();
                for (param, arg) in fndef.params.iter().zip(evaluated_args.iter()) {
                    self.declare_var(param, arg.clone());
                }
                let signal = self.exec_stmts(&fndef.body)?;
                self.pop_scope();

                match signal {
                    ExecSignal::Return(v) => Ok(v),
                    ExecSignal::None => Ok(Value::Null),
                }
            }
            Expr::Index { array, index } => {
                let arr_val = self.eval_expr(array)?;
                let idx = self.eval_expr(index)?.as_number()? as usize;
                let arr = arr_val.as_array()?;
                if idx >= arr.len() {
                    return Err(VmError::IndexOutOfBounds { index: idx, len: arr.len() });
                }
                Ok(Value::Number(arr[idx]))
            }
        }
    }
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Execute a QLANG program and return the final value and captured output.
pub fn execute_program(stmts: &[Stmt]) -> Result<Value, VmError> {
    let mut vm = VmState::new();
    vm.exec_stmts(stmts)?;
    Ok(Value::Null)
}

/// Top-level entry point: lex, parse, and execute a QLANG script.
pub fn run_qlang_script(source: &str) -> Result<(Value, Vec<String>), VmError> {
    let tokens = tokenize(source)?;
    let stmts = parse_program(&tokens)?;
    let mut vm = VmState::new();
    let signal = vm.exec_stmts(&stmts)?;
    let result = match signal {
        ExecSignal::Return(v) => v,
        ExecSignal::None => Value::Null,
    };
    Ok((result, vm.output))
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn run(src: &str) -> (Value, Vec<String>) {
        run_qlang_script(src).expect("script should succeed")
    }

    fn run_err(src: &str) -> VmError {
        run_qlang_script(src).expect_err("script should fail")
    }

    #[test]
    fn test_variable_binding_and_arithmetic() {
        let (_, out) = run(r#"
            let x = 5.0
            let y = x + 3.0 * 2.0
            print(y)
        "#);
        assert_eq!(out, vec!["11"]);
    }

    #[test]
    fn test_arithmetic_precedence() {
        let (_, out) = run(r#"
            let a = 2.0 + 3.0 * 4.0
            let b = (2.0 + 3.0) * 4.0
            print(a)
            print(b)
        "#);
        assert_eq!(out, vec!["14", "20"]);
    }

    #[test]
    fn test_if_else_branches() {
        let (_, out) = run(r#"
            let x = 10.0
            if x > 5.0 {
                print(1.0)
            } else {
                print(0.0)
            }
            if x < 5.0 {
                print(1.0)
            } else {
                print(0.0)
            }
        "#);
        assert_eq!(out, vec!["1", "0"]);
    }

    #[test]
    fn test_for_loop_with_range() {
        let (_, out) = run(r#"
            let sum = 0.0
            for i in 0..5 {
                sum = sum + i
            }
            print(sum)
        "#);
        // 0+1+2+3+4 = 10
        assert_eq!(out, vec!["10"]);
    }

    #[test]
    fn test_while_loop() {
        let (_, out) = run(r#"
            let x = 1.0
            while x < 100.0 {
                x = x * 2.0
            }
            print(x)
        "#);
        assert_eq!(out, vec!["128"]);
    }

    #[test]
    fn test_function_definition_and_call() {
        let (_, out) = run(r#"
            fn add(a, b) {
                return a + b
            }
            let result = add(3.0, 4.0)
            print(result)
        "#);
        assert_eq!(out, vec!["7"]);
    }

    #[test]
    fn test_recursive_function_fibonacci() {
        let (_, out) = run(r#"
            fn fibonacci(n) {
                if n <= 1.0 {
                    return n
                }
                return fibonacci(n - 1.0) + fibonacci(n - 2.0)
            }
            let result = fibonacci(10.0)
            print(result)
        "#);
        assert_eq!(out, vec!["55"]);
    }

    #[test]
    fn test_array_creation_and_indexing() {
        let (_, out) = run(r#"
            let arr = [10.0, 20.0, 30.0]
            print(arr[0])
            print(arr[1])
            print(arr[2])
        "#);
        assert_eq!(out, vec!["10", "20", "30"]);
    }

    #[test]
    fn test_array_length() {
        let (_, out) = run(r#"
            let arr = [1.0, 2.0, 3.0, 4.0, 5.0]
            print(len(arr))
        "#);
        assert_eq!(out, vec!["5"]);
    }

    #[test]
    fn test_comparison_operators() {
        let (_, out) = run(r#"
            print(1.0 == 1.0)
            print(1.0 != 2.0)
            print(3.0 < 5.0)
            print(5.0 > 3.0)
            print(3.0 <= 3.0)
            print(3.0 >= 4.0)
        "#);
        assert_eq!(out, vec!["true", "true", "true", "true", "true", "false"]);
    }

    #[test]
    fn test_nested_function_calls() {
        let (_, out) = run(r#"
            fn double(x) { return x * 2.0 }
            fn triple(x) { return x * 3.0 }
            let result = double(triple(5.0))
            print(result)
        "#);
        assert_eq!(out, vec!["30"]);
    }

    #[test]
    fn test_print_output_capture() {
        let (_, out) = run(r#"
            print(42.0)
            print("hello")
            print(true)
        "#);
        assert_eq!(out, vec!["42", "hello", "true"]);
    }

    #[test]
    fn test_string_values() {
        let (_, out) = run(r#"
            let s = "hello world"
            print(s)
        "#);
        assert_eq!(out, vec!["hello world"]);
    }

    #[test]
    fn test_error_on_undefined_variable() {
        let err = run_err("print(undefined_var)");
        match err {
            VmError::UndefinedVariable(name) => assert_eq!(name, "undefined_var"),
            other => panic!("expected UndefinedVariable, got: {other}"),
        }
    }

    #[test]
    fn test_error_on_division_by_zero() {
        let err = run_err("let x = 1.0 / 0.0");
        match err {
            VmError::DivisionByZero => {}
            other => panic!("expected DivisionByZero, got: {other}"),
        }
    }

    #[test]
    fn test_complex_program_fibonacci_and_arrays() {
        let (_, out) = run(r#"
            fn fibonacci(n) {
                if n <= 1.0 {
                    return n
                }
                return fibonacci(n - 1.0) + fibonacci(n - 2.0)
            }

            let result = fibonacci(10.0)
            print(result)

            let data = [1.0, 2.0, 3.0, 4.0, 5.0]
            let sum = 0.0
            for i in 0..len(data) {
                sum = sum + data[i]
            }
            print(sum)
        "#);
        assert_eq!(out, vec!["55", "15"]);
    }

    #[test]
    fn test_logical_operators() {
        let (_, out) = run(r#"
            print(true and false)
            print(true or false)
            print(not true)
            print(true and true)
        "#);
        assert_eq!(out, vec!["false", "true", "false", "true"]);
    }

    #[test]
    fn test_unary_negation() {
        let (_, out) = run(r#"
            let x = 5.0
            let y = -x
            print(y)
            print(-3.0)
        "#);
        assert_eq!(out, vec!["-5", "-3"]);
    }

    #[test]
    fn test_error_undefined_function() {
        let err = run_err("let x = nope(1.0)");
        match err {
            VmError::UndefinedFunction(name) => assert_eq!(name, "nope"),
            other => panic!("expected UndefinedFunction, got: {other}"),
        }
    }

    #[test]
    fn test_index_out_of_bounds() {
        let err = run_err(r#"
            let arr = [1.0, 2.0]
            print(arr[5])
        "#);
        match err {
            VmError::IndexOutOfBounds { index: 5, len: 2 } => {}
            other => panic!("expected IndexOutOfBounds, got: {other}"),
        }
    }
}
