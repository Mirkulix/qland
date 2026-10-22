use std::io::BufRead;

#[derive(Clone, Copy, PartialEq)]
enum Tok {
    Num(i64),
    Sym(char),
}

enum E {
    Lit(i64),
    Neg(Box<E>),
    Bin(char, Box<E>, Box<E>),
}

fn lex(line: &str) -> Option<Vec<Tok>> {
    let cs: Vec<char> = line.chars().collect();
    let (mut out, mut i) = (Vec::new(), 0);
    while i < cs.len() {
        let c = cs[i];
        if c == ' ' || c == '\t' {
            i += 1;
        } else if c.is_ascii_digit() {
            let mut n = 0i64;
            while i < cs.len() && cs[i].is_ascii_digit() {
                n = n * 10 + (cs[i] as i64 - '0' as i64);
                i += 1;
            }
            out.push(Tok::Num(n));
        } else if "+-*/%()".contains(c) {
            out.push(Tok::Sym(c));
            i += 1;
        } else {
            return None;
        }
    }
    Some(out)
}

struct P<'a> {
    t: &'a [Tok],
    i: usize,
}

impl P<'_> {
    fn sym(&self) -> Option<char> {
        match self.t.get(self.i) {
            Some(Tok::Sym(c)) => Some(*c),
            _ => None,
        }
    }
    fn expr(&mut self) -> Option<E> {
        let mut v = self.term()?;
        while let Some(op @ ('+' | '-')) = self.sym() {
            self.i += 1;
            v = E::Bin(op, Box::new(v), Box::new(self.term()?));
        }
        Some(v)
    }
    fn term(&mut self) -> Option<E> {
        let mut v = self.unary()?;
        while let Some(op @ ('*' | '/' | '%')) = self.sym() {
            self.i += 1;
            v = E::Bin(op, Box::new(v), Box::new(self.unary()?));
        }
        Some(v)
    }
    fn unary(&mut self) -> Option<E> {
        if self.sym() == Some('-') {
            self.i += 1;
            return Some(E::Neg(Box::new(self.unary()?)));
        }
        self.atom()
    }
    fn atom(&mut self) -> Option<E> {
        match self.t.get(self.i).copied() {
            Some(Tok::Num(n)) => {
                self.i += 1;
                Some(E::Lit(n))
            }
            Some(Tok::Sym('(')) => {
                self.i += 1;
                let e = self.expr()?;
                if self.sym() != Some(')') {
                    return None;
                }
                self.i += 1;
                Some(e)
            }
            _ => None,
        }
    }
}

fn eval(e: &E) -> Result<i64, ()> {
    Ok(match e {
        E::Lit(n) => *n,
        E::Neg(x) => -eval(x)?,
        E::Bin(op, l, r) => {
            let (a, b) = (eval(l)?, eval(r)?);
            match op {
                '+' => a + b,
                '-' => a - b,
                '*' => a * b,
                _ if b == 0 => return Err(()),
                '/' => a / b,
                _ => a % b,
            }
        }
    })
}

fn calc(line: &str) -> String {
    let Some(t) = lex(line) else { return "error: syntax".into() };
    let mut p = P { t: &t, i: 0 };
    match p.expr() {
        Some(e) if p.i == t.len() => eval(&e).map_or("error: division by zero".into(), |v| v.to_string()),
        _ => "error: syntax".into(),
    }
}

fn main() {
    for line in std::io::stdin().lock().lines() {
        println!("{}", calc(&line.unwrap()));
    }
}
