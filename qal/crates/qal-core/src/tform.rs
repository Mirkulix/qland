//! The typed form `QALT` (self-hosting spec §4.4): a binary encoding of a checked, monomorphic
//! [`TProgram`] — types, functions with their instances, tests, generics, typed expressions,
//! closures with captures, patterns and built-in calls — with an exact decoder.
//!
//! It is the interface between a type checker and a code generator (Rust checker -> QAL codegen
//! in 5c), the way the host gets the `TProgram` it needs (prover, value encoding) from the
//! self-hosted compiler, and the byte-exact oracle for the QAL checker (5d). Encoding is a pure
//! function of the program: equal programs have equal bytes, and `decode(encode(p)) == p`.
//!
//! ```text
//! qalt     := "QALT" version:u8(1) n typedef{n} n tfn{n} n ttest{n} n tfn{n}     (types, fns, tests, generics)
//! typedef  := name:str n str{n} body                                          (name, type parameters)
//! body     := 0 n (str type){n} | 1 n (str n type{n}){n}                       (record | enum)
//! tfn      := fndef n type{n} n texpr{n} n texpr{n} opt(uint) texpr opt(uint n type{n})
//!             (def, locals, req, ens, result local, body, instance: generic index and type arguments)
//! ttest    := name:str n type{n} texpr
//! fndef    := name n str{n} n (str type){n} type n str{n} n expr{n} n expr{n} expr
//!             (name, tparams, params, ret, effects as written, req, ens, body: the source AST)
//! texpr    := type tkind                       expr, pat, tpat, type: a tag byte, then the fields
//! opt(x)   := 0 | 1 x          uint := minimal unsigned LEB128     int := minimal signed LEB128
//! str      := uint utf8        float := IEEE bits, 8 bytes little endian       bool := 0 | 1
//! ```
//!
//! Tags are the constants below; built-ins are their index in [`Builtin::ALL`] (append-only), binary
//! operators their index in [`BINOPS`]. The decoder is strict: minimal LEB128, known tags, valid
//! UTF-8, no trailing bytes, nesting at most [`MAX_DEPTH`], and every function, generic, local,
//! instance and built-in index in range — a malformed form is an error, never a panic.

use crate::ast::*;
use crate::tast::*;

pub const MAGIC: &[u8; 4] = b"QALT";
pub const VERSION: u8 = 1;

/// Nesting accepted while decoding (expressions, patterns and types together). Checked programs
/// stay far below it: the parser bounds the height of written code and of types at 256 each. A
/// debug build decodes this depth within a 2 MiB thread stack (tested).
pub const MAX_DEPTH: usize = 1024;

/// Binary operators in tag order.
pub const BINOPS: [BinOp; 14] = [
    BinOp::Add,
    BinOp::Sub,
    BinOp::Mul,
    BinOp::Div,
    BinOp::Rem,
    BinOp::Concat,
    BinOp::Eq,
    BinOp::Ne,
    BinOp::Lt,
    BinOp::Le,
    BinOp::Gt,
    BinOp::Ge,
    BinOp::And,
    BinOp::Or,
];

fn binop_tag(op: BinOp) -> u8 {
    match op {
        BinOp::Add => 0,
        BinOp::Sub => 1,
        BinOp::Mul => 2,
        BinOp::Div => 3,
        BinOp::Rem => 4,
        BinOp::Concat => 5,
        BinOp::Eq => 6,
        BinOp::Ne => 7,
        BinOp::Lt => 8,
        BinOp::Le => 9,
        BinOp::Gt => 10,
        BinOp::Ge => 11,
        BinOp::And => 12,
        BinOp::Or => 13,
    }
}

mod tag {
    pub const T_INT: u8 = 0;
    pub const T_FLOAT: u8 = 1;
    pub const T_BOOL: u8 = 2;
    pub const T_UNIT: u8 = 3;
    pub const T_STR: u8 = 4;
    pub const T_LIST: u8 = 5;
    pub const T_OPTION: u8 = 6;
    pub const T_RESULT: u8 = 7;
    pub const T_NAMED: u8 = 8;
    pub const T_FN: u8 = 9;
    pub const T_APP: u8 = 10;
    pub const T_PARAM: u8 = 11;
    pub const T_MAP: u8 = 12;
    pub const T_BYTES: u8 = 13;
    pub const T_VAR: u8 = 14;

    pub const E_INT: u8 = 0;
    pub const E_FLOAT: u8 = 1;
    pub const E_BOOL: u8 = 2;
    pub const E_STR: u8 = 3;
    pub const E_UNIT: u8 = 4;
    pub const E_VAR: u8 = 5;
    pub const E_BIN: u8 = 6;
    pub const E_UN: u8 = 7;
    pub const E_IF: u8 = 8;
    pub const E_LET: u8 = 9;
    pub const E_CALL: u8 = 10;
    pub const E_LIST: u8 = 11;
    pub const E_RECORD: u8 = 12;
    pub const E_FIELD: u8 = 13;
    pub const E_MATCH: u8 = 14;
    pub const E_SEQ: u8 = 15;
    pub const E_LAMBDA: u8 = 16;

    pub const P_WILD: u8 = 0;
    pub const P_BIND: u8 = 1;
    pub const P_INT: u8 = 2;
    pub const P_STR: u8 = 3;
    pub const P_BOOL: u8 = 4;
    pub const P_UNIT: u8 = 5;
    pub const P_VARIANT: u8 = 6;
    pub const P_RECORD: u8 = 7;

    pub const K_INT: u8 = 0;
    pub const K_FLOAT: u8 = 1;
    pub const K_BOOL: u8 = 2;
    pub const K_STR: u8 = 3;
    pub const K_UNIT: u8 = 4;
    pub const K_LOCAL: u8 = 5;
    pub const K_FN_REF: u8 = 6;
    pub const K_BIN: u8 = 7;
    pub const K_UN: u8 = 8;
    pub const K_IF: u8 = 9;
    pub const K_LET: u8 = 10;
    pub const K_CALL: u8 = 11;
    pub const K_CALL_G: u8 = 12;
    pub const K_FN_REF_G: u8 = 13;
    pub const K_LAMBDA: u8 = 14;
    pub const K_CALL_INDIRECT: u8 = 15;
    pub const K_BUILTIN: u8 = 16;
    pub const K_LIST: u8 = 17;
    pub const K_VARIANT: u8 = 18;
    pub const K_RECORD: u8 = 19;
    pub const K_FIELD: u8 = 20;
    pub const K_MATCH: u8 = 21;
    pub const K_SEQ: u8 = 22;
}
use tag::*;

// ---------------------------------------------------------------------------------------------
// Encoding

/// Minimal unsigned LEB128.
pub fn put_uint(out: &mut Vec<u8>, mut n: u64) {
    loop {
        let b = (n & 0x7f) as u8;
        n >>= 7;
        if n == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

/// Minimal signed LEB128.
pub fn put_int(out: &mut Vec<u8>, mut n: i64) {
    loop {
        let b = (n & 0x7f) as u8;
        n >>= 7;
        let done = (n == 0 && b & 0x40 == 0) || (n == -1 && b & 0x40 != 0);
        if done {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

/// The encoder; also used by the `QALQ`/`QALR` protocol for its AST, type and module fields.
#[derive(Default)]
pub struct Writer {
    pub out: Vec<u8>,
}

impl Writer {
    /// `len bytes`.
    pub fn bytes(&mut self, b: &[u8]) {
        self.len(b.len());
        self.out.extend_from_slice(b);
    }

    /// A definition: 0 typedef | 1 fndef | 2 name expr (test).
    pub fn def(&mut self, d: &Def) {
        match d {
            Def::Type(t) => {
                self.byte(0);
                self.typedef(t);
            }
            Def::Fn(f) => {
                self.byte(1);
                self.fndef(f);
            }
            Def::Test(t) => {
                self.byte(2);
                self.str(&t.name);
                self.expr(&t.body);
            }
        }
    }

    pub fn byte(&mut self, b: u8) {
        self.out.push(b);
    }

    pub fn uint(&mut self, n: u64) {
        put_uint(&mut self.out, n);
    }

    pub fn len(&mut self, n: usize) {
        self.uint(n as u64);
    }

    pub fn int(&mut self, n: i64) {
        put_int(&mut self.out, n);
    }

    pub fn float(&mut self, x: f64) {
        self.out.extend_from_slice(&x.to_bits().to_le_bytes());
    }

    pub fn bool(&mut self, b: bool) {
        self.byte(b as u8);
    }

    pub fn str(&mut self, s: &str) {
        self.len(s.len());
        self.out.extend_from_slice(s.as_bytes());
    }

    pub fn strs(&mut self, xs: &[String]) {
        self.len(xs.len());
        for x in xs {
            self.str(x);
        }
    }

    pub fn ty(&mut self, t: &Type) {
        match t {
            Type::Int => self.byte(T_INT),
            Type::Float => self.byte(T_FLOAT),
            Type::Bool => self.byte(T_BOOL),
            Type::Unit => self.byte(T_UNIT),
            Type::Str => self.byte(T_STR),
            Type::Bytes => self.byte(T_BYTES),
            Type::List(x) => {
                self.byte(T_LIST);
                self.ty(x);
            }
            Type::Option(x) => {
                self.byte(T_OPTION);
                self.ty(x);
            }
            Type::Result(a, b) => {
                self.byte(T_RESULT);
                self.ty(a);
                self.ty(b);
            }
            Type::Map(a, b) => {
                self.byte(T_MAP);
                self.ty(a);
                self.ty(b);
            }
            Type::Named(n) => {
                self.byte(T_NAMED);
                self.str(n);
            }
            Type::Param(n) => {
                self.byte(T_PARAM);
                self.str(n);
            }
            Type::App(n, args) => {
                self.byte(T_APP);
                self.str(n);
                self.types(args);
            }
            Type::Fn(ps, r, effs) => {
                self.byte(T_FN);
                self.types(ps);
                self.ty(r);
                self.strs(effs);
            }
            Type::Var(v) => {
                self.byte(T_VAR);
                self.uint(*v as u64);
            }
        }
    }

    pub fn types(&mut self, ts: &[Type]) {
        self.len(ts.len());
        for t in ts {
            self.ty(t);
        }
    }

    pub fn opt_ty(&mut self, t: &Option<Type>) {
        match t {
            Some(t) => {
                self.byte(1);
                self.ty(t);
            }
            None => self.byte(0),
        }
    }

    pub fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Int(n) => {
                self.byte(E_INT);
                self.int(*n);
            }
            Expr::Float(x) => {
                self.byte(E_FLOAT);
                self.float(*x);
            }
            Expr::Bool(b) => {
                self.byte(E_BOOL);
                self.bool(*b);
            }
            Expr::Str(s) => {
                self.byte(E_STR);
                self.str(s);
            }
            Expr::Unit => self.byte(E_UNIT),
            Expr::Var(v) => {
                self.byte(E_VAR);
                self.str(v);
            }
            Expr::Bin(op, l, r) => {
                self.byte(E_BIN);
                self.byte(binop_tag(*op));
                self.expr(l);
                self.expr(r);
            }
            Expr::Un(op, x) => {
                self.byte(E_UN);
                self.byte(unop_tag(*op));
                self.expr(x);
            }
            Expr::If(c, t, f) => {
                self.byte(E_IF);
                self.expr(c);
                self.expr(t);
                self.expr(f);
            }
            Expr::Let(n, ann, v, b) => {
                self.byte(E_LET);
                self.str(n);
                self.opt_ty(ann);
                self.expr(v);
                self.expr(b);
            }
            Expr::Call(n, args) => {
                self.byte(E_CALL);
                self.str(n);
                self.exprs(args);
            }
            Expr::List(xs) => {
                self.byte(E_LIST);
                self.exprs(xs);
            }
            Expr::Record(n, fs) => {
                self.byte(E_RECORD);
                self.str(n);
                self.len(fs.len());
                for (f, x) in fs {
                    self.str(f);
                    self.expr(x);
                }
            }
            Expr::Field(x, f) => {
                self.byte(E_FIELD);
                self.expr(x);
                self.str(f);
            }
            Expr::Match(s, arms) => {
                self.byte(E_MATCH);
                self.expr(s);
                self.len(arms.len());
                for a in arms {
                    self.pat(&a.pat);
                    match &a.guard {
                        Some(g) => {
                            self.byte(1);
                            self.expr(g);
                        }
                        None => self.byte(0),
                    }
                    self.expr(&a.body);
                }
            }
            Expr::Seq(xs) => {
                self.byte(E_SEQ);
                self.exprs(xs);
            }
            Expr::Lambda(ps, body) => {
                self.byte(E_LAMBDA);
                self.len(ps.len());
                for (n, t) in ps {
                    self.str(n);
                    self.opt_ty(t);
                }
                self.expr(body);
            }
        }
    }

    pub fn exprs(&mut self, xs: &[Expr]) {
        self.len(xs.len());
        for x in xs {
            self.expr(x);
        }
    }

    pub fn pat(&mut self, p: &Pat) {
        match p {
            Pat::Wild => self.byte(P_WILD),
            Pat::Bind(n) => {
                self.byte(P_BIND);
                self.str(n);
            }
            Pat::Int(n) => {
                self.byte(P_INT);
                self.int(*n);
            }
            Pat::Str(s) => {
                self.byte(P_STR);
                self.str(s);
            }
            Pat::Bool(b) => {
                self.byte(P_BOOL);
                self.bool(*b);
            }
            Pat::Unit => self.byte(P_UNIT),
            Pat::Variant(n, ps) => {
                self.byte(P_VARIANT);
                self.str(n);
                self.len(ps.len());
                for p in ps {
                    self.pat(p);
                }
            }
            Pat::Record(n, fs) => {
                self.byte(P_RECORD);
                self.str(n);
                self.len(fs.len());
                for (f, p) in fs {
                    self.str(f);
                    self.pat(p);
                }
            }
        }
    }

    pub fn fndef(&mut self, f: &FnDef) {
        self.str(&f.name);
        // Effect parameters travel in the type-parameter list as `!name` entries, so a function
        // without them keeps its version 1 bytes.
        let mut heads = f.tparams.clone();
        heads.extend(f.eparams.iter().map(|e| format!("!{e}")));
        self.strs(&heads);
        self.len(f.params.len());
        for p in &f.params {
            self.str(&p.name);
            self.ty(&p.ty);
        }
        self.ty(&f.ret);
        self.strs(&f.effects);
        self.exprs(&f.req);
        self.exprs(&f.ens);
        self.expr(&f.body);
    }

    pub fn texpr(&mut self, e: &TExpr) {
        self.ty(&e.ty);
        match &e.kind {
            TKind::Int(n) => {
                self.byte(K_INT);
                self.int(*n);
            }
            TKind::Float(x) => {
                self.byte(K_FLOAT);
                self.float(*x);
            }
            TKind::Bool(b) => {
                self.byte(K_BOOL);
                self.bool(*b);
            }
            TKind::Str(s) => {
                self.byte(K_STR);
                self.str(s);
            }
            TKind::Unit => self.byte(K_UNIT),
            TKind::Local(id) => {
                self.byte(K_LOCAL);
                self.uint(*id as u64);
            }
            TKind::FnRef(i) => {
                self.byte(K_FN_REF);
                self.len(*i);
            }
            TKind::Bin(op, l, r, path) => {
                self.byte(K_BIN);
                self.byte(binop_tag(*op));
                self.texpr(l);
                self.texpr(r);
                self.str(path);
            }
            TKind::Un(op, x, path) => {
                self.byte(K_UN);
                self.byte(unop_tag(*op));
                self.texpr(x);
                self.str(path);
            }
            TKind::If(c, t, f) => {
                self.byte(K_IF);
                self.texpr(c);
                self.texpr(t);
                self.texpr(f);
            }
            TKind::Let(id, v, b) => {
                self.byte(K_LET);
                self.uint(*id as u64);
                self.texpr(v);
                self.texpr(b);
            }
            TKind::Call(i, args) => {
                self.byte(K_CALL);
                self.len(*i);
                self.texprs(args);
            }
            TKind::CallG(i, tys, args) => {
                self.byte(K_CALL_G);
                self.len(*i);
                self.types(tys);
                self.texprs(args);
            }
            TKind::FnRefG(i, tys) => {
                self.byte(K_FN_REF_G);
                self.len(*i);
                self.types(tys);
            }
            TKind::Lambda(l) => {
                self.byte(K_LAMBDA);
                self.ids(&l.params);
                self.ids(&l.captures);
                self.texpr(&l.body);
            }
            TKind::CallIndirect(f, args) => {
                self.byte(K_CALL_INDIRECT);
                self.texpr(f);
                self.texprs(args);
            }
            TKind::Builtin(b, args, path) => {
                self.byte(K_BUILTIN);
                self.len(builtin_index(*b));
                self.texprs(args);
                self.str(path);
            }
            TKind::List(xs) => {
                self.byte(K_LIST);
                self.texprs(xs);
            }
            TKind::Variant(tag, xs) => {
                self.byte(K_VARIANT);
                self.uint(*tag as u64);
                self.texprs(xs);
            }
            TKind::Record(fs) => {
                self.byte(K_RECORD);
                self.len(fs.len());
                for (i, x) in fs {
                    self.uint(*i as u64);
                    self.texpr(x);
                }
            }
            TKind::Field(x, i) => {
                self.byte(K_FIELD);
                self.texpr(x);
                self.uint(*i as u64);
            }
            TKind::Match(s, arms) => {
                self.byte(K_MATCH);
                self.texpr(s);
                self.len(arms.len());
                for a in arms {
                    self.tpat(&a.pat);
                    match &a.guard {
                        Some(g) => {
                            self.byte(1);
                            self.texpr(g);
                        }
                        None => self.byte(0),
                    }
                    self.texpr(&a.body);
                }
            }
            TKind::Seq(xs) => {
                self.byte(K_SEQ);
                self.texprs(xs);
            }
        }
    }

    pub fn texprs(&mut self, xs: &[TExpr]) {
        self.len(xs.len());
        for x in xs {
            self.texpr(x);
        }
    }

    pub fn ids(&mut self, ids: &[u32]) {
        self.len(ids.len());
        for i in ids {
            self.uint(*i as u64);
        }
    }

    pub fn tpat(&mut self, p: &TPat) {
        match p {
            TPat::Wild => self.byte(P_WILD),
            TPat::Bind(id) => {
                self.byte(P_BIND);
                self.uint(*id as u64);
            }
            TPat::Int(n) => {
                self.byte(P_INT);
                self.int(*n);
            }
            TPat::Str(s) => {
                self.byte(P_STR);
                self.str(s);
            }
            TPat::Bool(b) => {
                self.byte(P_BOOL);
                self.bool(*b);
            }
            TPat::Unit => self.byte(P_UNIT),
            TPat::Variant(tag, ps) => {
                self.byte(P_VARIANT);
                self.uint(*tag as u64);
                self.len(ps.len());
                for p in ps {
                    self.tpat(p);
                }
            }
            TPat::Record(fs) => {
                self.byte(P_RECORD);
                self.len(fs.len());
                for (i, p) in fs {
                    self.uint(*i as u64);
                    self.tpat(p);
                }
            }
        }
    }

    pub fn typedef(&mut self, t: &TypeDef) {
        self.str(&t.name);
        self.strs(&t.params);
        match &t.body {
            TypeBody::Record(fs) => {
                self.byte(0);
                self.len(fs.len());
                for (f, ty) in fs {
                    self.str(f);
                    self.ty(ty);
                }
            }
            TypeBody::Enum(vs) => {
                self.byte(1);
                self.len(vs.len());
                for v in vs {
                    self.str(&v.name);
                    self.types(&v.fields);
                }
            }
        }
    }

    pub fn tfn(&mut self, f: &TFn) {
        self.fndef(&f.def);
        self.types(&f.locals);
        self.texprs(&f.req);
        self.texprs(&f.ens);
        match f.result_local {
            Some(r) => {
                self.byte(1);
                self.uint(r as u64);
            }
            None => self.byte(0),
        }
        self.texpr(&f.body);
        match &f.instance {
            Some((g, args)) => {
                self.byte(1);
                self.len(*g);
                self.types(args);
            }
            None => self.byte(0),
        }
    }
}

fn unop_tag(op: UnOp) -> u8 {
    match op {
        UnOp::Neg => 0,
        UnOp::Not => 1,
    }
}

/// The built-in's position in [`Builtin::ALL`] (append-only, so indices never change).
pub fn builtin_index(b: Builtin) -> usize {
    Builtin::ALL.iter().position(|x| *x == b).unwrap_or(usize::MAX)
}

/// The `QALT` bytes of a typed program.
pub fn encode(p: &TProgram) -> Vec<u8> {
    let mut e = Writer::default();
    e.out.extend_from_slice(MAGIC);
    e.byte(VERSION);
    e.len(p.types.len());
    for t in &p.types {
        e.typedef(t);
    }
    e.len(p.fns.len());
    for f in &p.fns {
        e.tfn(f);
    }
    e.len(p.tests.len());
    for t in &p.tests {
        e.str(&t.name);
        e.types(&t.locals);
        e.texpr(&t.body);
    }
    e.len(p.generics.len());
    for f in &p.generics {
        e.tfn(f);
    }
    e.out
}

// ---------------------------------------------------------------------------------------------
// Decoding

type R<T> = Result<T, String>;

/// A bounded reader of minimal LEB128 integers and strings (also used by the `QALQ`/`QALR`
/// protocol, which shares the integer and string encodings).
pub struct Reader<'b> {
    b: &'b [u8],
    pub pos: usize,
    depth: usize,
}

impl<'b> Reader<'b> {
    pub fn new(b: &'b [u8]) -> Reader<'b> {
        Reader { b, pos: 0, depth: 0 }
    }

    pub fn at_end(&self) -> bool {
        self.pos == self.b.len()
    }

    pub fn finish(&self) -> R<()> {
        if self.at_end() {
            Ok(())
        } else {
            Err(format!("{} trailing bytes at offset {}", self.b.len() - self.pos, self.pos))
        }
    }

    pub fn byte(&mut self) -> R<u8> {
        let b = *self.b.get(self.pos).ok_or_else(|| format!("truncated at offset {}", self.pos))?;
        self.pos += 1;
        Ok(b)
    }

    pub fn bytes(&mut self, n: usize) -> R<&'b [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.b.len()).ok_or_else(|| format!("truncated at offset {}", self.pos))?;
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    /// Minimal unsigned LEB128 of at most 64 bits.
    pub fn uint(&mut self) -> R<u64> {
        let start = self.pos;
        let (mut n, mut shift) = (0u64, 0u32);
        loop {
            let b = self.byte()?;
            let v = (b & 0x7f) as u64;
            if shift == 63 && v > 1 {
                return Err(format!("integer too large at offset {start}"));
            }
            n |= v << shift;
            if b & 0x80 == 0 {
                if b == 0 && shift > 0 {
                    return Err(format!("non-minimal integer at offset {start}"));
                }
                return Ok(n);
            }
            shift += 7;
            if shift > 63 {
                return Err(format!("integer too large at offset {start}"));
            }
        }
    }

    /// Minimal signed LEB128 of at most 64 bits.
    pub fn int(&mut self) -> R<i64> {
        let start = self.pos;
        let (mut n, mut shift) = (0i64, 0u32);
        let mut bytes = Vec::new();
        loop {
            let b = self.byte()?;
            bytes.push(b);
            if bytes.len() > 10 {
                return Err(format!("integer too large at offset {start}"));
            }
            n |= ((b & 0x7f) as i64).wrapping_shl(shift);
            shift += 7;
            if b & 0x80 == 0 {
                if shift < 64 && b & 0x40 != 0 {
                    n |= -1i64 << shift;
                }
                break;
            }
        }
        // Minimal and in range: re-encoding gives the same bytes.
        let mut again = Vec::new();
        put_int(&mut again, n);
        if again != bytes {
            return Err(format!("non-minimal or out-of-range integer at offset {start}"));
        }
        Ok(n)
    }

    /// A count or index that must fit the platform (bounded by the input length for counts).
    pub fn usize(&mut self) -> R<usize> {
        usize::try_from(self.uint()?).map_err(|_| format!("index too large at offset {}", self.pos))
    }

    /// A count of items each at least one byte long (so absurd counts fail before allocating).
    pub fn count(&mut self) -> R<usize> {
        let n = self.usize()?;
        if n > self.b.len() - self.pos {
            return Err(format!("count {n} exceeds the remaining input at offset {}", self.pos));
        }
        Ok(n)
    }

    pub fn u32(&mut self) -> R<u32> {
        u32::try_from(self.uint()?).map_err(|_| format!("value too large at offset {}", self.pos))
    }

    pub fn bool(&mut self) -> R<bool> {
        match self.byte()? {
            0 => Ok(false),
            1 => Ok(true),
            b => Err(format!("invalid flag {b} at offset {}", self.pos - 1)),
        }
    }

    pub fn f64(&mut self) -> R<f64> {
        let b = self.bytes(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(f64::from_bits(u64::from_le_bytes(a)))
    }

    pub fn str(&mut self) -> R<String> {
        let n = self.count()?;
        let at = self.pos;
        let b = self.bytes(n)?;
        String::from_utf8(b.to_vec()).map_err(|_| format!("invalid UTF-8 at offset {at}"))
    }

    pub fn strs(&mut self) -> R<Vec<String>> {
        let n = self.count()?;
        (0..n).map(|_| self.str()).collect()
    }

    fn enter(&mut self) -> R<()> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(format!("nested too deeply at offset {}", self.pos));
        }
        Ok(())
    }

    fn leave(&mut self) {
        self.depth -= 1;
    }

    pub fn ty(&mut self) -> R<Type> {
        self.enter()?;
        let t = self.ty_inner();
        self.leave();
        t
    }

    fn ty_inner(&mut self) -> R<Type> {
        Ok(match self.byte()? {
            T_INT => Type::Int,
            T_FLOAT => Type::Float,
            T_BOOL => Type::Bool,
            T_UNIT => Type::Unit,
            T_STR => Type::Str,
            T_BYTES => Type::Bytes,
            T_LIST => Type::list(self.ty()?),
            T_OPTION => Type::option(self.ty()?),
            T_RESULT => {
                let a = self.ty()?;
                Type::result(a, self.ty()?)
            }
            T_MAP => {
                let a = self.ty()?;
                Type::map(a, self.ty()?)
            }
            T_NAMED => Type::Named(self.str()?),
            T_PARAM => Type::Param(self.str()?),
            T_APP => {
                let n = self.str()?;
                Type::App(n, self.types()?)
            }
            T_FN => {
                let ps = self.types()?;
                let r = self.ty()?;
                Type::Fn(ps, Box::new(r), self.strs()?)
            }
            T_VAR => Type::Var(self.u32()?),
            t => return Err(bad_tag("type", t, self.pos)),
        })
    }

    pub fn types(&mut self) -> R<Vec<Type>> {
        let n = self.count()?;
        (0..n).map(|_| self.ty()).collect()
    }

    pub fn opt_ty(&mut self) -> R<Option<Type>> {
        Ok(if self.bool()? { Some(self.ty()?) } else { None })
    }

    fn binop(&mut self) -> R<BinOp> {
        let t = self.byte()?;
        BINOPS.get(t as usize).copied().ok_or_else(|| bad_tag("operator", t, self.pos))
    }

    fn unop(&mut self) -> R<UnOp> {
        match self.byte()? {
            0 => Ok(UnOp::Neg),
            1 => Ok(UnOp::Not),
            t => Err(bad_tag("unary operator", t, self.pos)),
        }
    }

    pub fn expr(&mut self) -> R<Expr> {
        self.enter()?;
        let e = self.expr_inner();
        self.leave();
        e
    }

    fn expr_inner(&mut self) -> R<Expr> {
        Ok(match self.byte()? {
            E_INT => Expr::Int(self.int()?),
            E_FLOAT => Expr::Float(self.f64()?),
            E_BOOL => Expr::Bool(self.bool()?),
            E_STR => Expr::Str(self.str()?),
            E_UNIT => Expr::Unit,
            E_VAR => Expr::Var(self.str()?),
            E_BIN => {
                let op = self.binop()?;
                let l = self.expr()?;
                Expr::Bin(op, Box::new(l), Box::new(self.expr()?))
            }
            E_UN => {
                let op = self.unop()?;
                Expr::Un(op, Box::new(self.expr()?))
            }
            E_IF => {
                let c = self.expr()?;
                let t = self.expr()?;
                Expr::If(Box::new(c), Box::new(t), Box::new(self.expr()?))
            }
            E_LET => {
                let n = self.str()?;
                let ann = self.opt_ty()?;
                let v = self.expr()?;
                Expr::Let(n, ann, Box::new(v), Box::new(self.expr()?))
            }
            E_CALL => {
                let n = self.str()?;
                Expr::Call(n, self.exprs()?)
            }
            E_LIST => Expr::List(self.exprs()?),
            E_RECORD => {
                let n = self.str()?;
                let k = self.count()?;
                let mut fs = Vec::new();
                for _ in 0..k {
                    let f = self.str()?;
                    fs.push((f, self.expr()?));
                }
                Expr::Record(n, fs)
            }
            E_FIELD => {
                let x = self.expr()?;
                Expr::Field(Box::new(x), self.str()?)
            }
            E_MATCH => {
                let s = self.expr()?;
                let k = self.count()?;
                let mut arms = Vec::new();
                for _ in 0..k {
                    let pat = self.pat()?;
                    let guard = if self.bool()? { Some(self.expr()?) } else { None };
                    arms.push(Arm { pat, guard, body: self.expr()? });
                }
                Expr::Match(Box::new(s), arms)
            }
            E_SEQ => Expr::Seq(self.exprs()?),
            E_LAMBDA => {
                let k = self.count()?;
                let mut ps = Vec::new();
                for _ in 0..k {
                    let n = self.str()?;
                    ps.push((n, self.opt_ty()?));
                }
                Expr::Lambda(ps, Box::new(self.expr()?))
            }
            t => return Err(bad_tag("expression", t, self.pos)),
        })
    }

    pub fn exprs(&mut self) -> R<Vec<Expr>> {
        let n = self.count()?;
        (0..n).map(|_| self.expr()).collect()
    }

    pub fn pat(&mut self) -> R<Pat> {
        self.enter()?;
        let p = self.pat_inner();
        self.leave();
        p
    }

    fn pat_inner(&mut self) -> R<Pat> {
        Ok(match self.byte()? {
            P_WILD => Pat::Wild,
            P_BIND => Pat::Bind(self.str()?),
            P_INT => Pat::Int(self.int()?),
            P_STR => Pat::Str(self.str()?),
            P_BOOL => Pat::Bool(self.bool()?),
            P_UNIT => Pat::Unit,
            P_VARIANT => {
                let n = self.str()?;
                let k = self.count()?;
                Pat::Variant(n, (0..k).map(|_| self.pat()).collect::<R<_>>()?)
            }
            P_RECORD => {
                let n = self.str()?;
                let k = self.count()?;
                let mut fs = Vec::new();
                for _ in 0..k {
                    let f = self.str()?;
                    fs.push((f, self.pat()?));
                }
                Pat::Record(n, fs)
            }
            t => return Err(bad_tag("pattern", t, self.pos)),
        })
    }

    pub fn fndef(&mut self) -> R<FnDef> {
        let name = self.str()?;
        let heads = self.strs()?;
        let (eparams, tparams): (Vec<String>, Vec<String>) = heads.into_iter().partition(|h| h.starts_with('!'));
        let eparams: Vec<String> = eparams.into_iter().map(|e| e[1..].to_string()).collect();
        let k = self.count()?;
        let mut params = Vec::new();
        for _ in 0..k {
            let n = self.str()?;
            params.push(Param { name: n, ty: self.ty()? });
        }
        let ret = self.ty()?;
        let effects = self.strs()?;
        let req = self.exprs()?;
        let ens = self.exprs()?;
        Ok(FnDef { name, tparams, eparams, params, ret, effects, req, ens, body: self.expr()? })
    }

    pub fn texpr(&mut self) -> R<TExpr> {
        self.enter()?;
        let e = self.texpr_inner();
        self.leave();
        e
    }

    fn texpr_inner(&mut self) -> R<TExpr> {
        let ty = self.ty()?;
        let kind = match self.byte()? {
            K_INT => TKind::Int(self.int()?),
            K_FLOAT => TKind::Float(self.f64()?),
            K_BOOL => TKind::Bool(self.bool()?),
            K_STR => TKind::Str(self.str()?),
            K_UNIT => TKind::Unit,
            K_LOCAL => TKind::Local(self.u32()?),
            K_FN_REF => TKind::FnRef(self.usize()?),
            K_BIN => {
                let op = self.binop()?;
                let l = self.texpr()?;
                let r = self.texpr()?;
                TKind::Bin(op, Box::new(l), Box::new(r), self.str()?)
            }
            K_UN => {
                let op = self.unop()?;
                let x = self.texpr()?;
                TKind::Un(op, Box::new(x), self.str()?)
            }
            K_IF => {
                let c = self.texpr()?;
                let t = self.texpr()?;
                TKind::If(Box::new(c), Box::new(t), Box::new(self.texpr()?))
            }
            K_LET => {
                let id = self.u32()?;
                let v = self.texpr()?;
                TKind::Let(id, Box::new(v), Box::new(self.texpr()?))
            }
            K_CALL => {
                let i = self.usize()?;
                TKind::Call(i, self.texprs()?)
            }
            K_CALL_G => {
                let i = self.usize()?;
                let tys = self.types()?;
                TKind::CallG(i, tys, self.texprs()?)
            }
            K_FN_REF_G => {
                let i = self.usize()?;
                TKind::FnRefG(i, self.types()?)
            }
            K_LAMBDA => {
                let params = self.ids()?;
                let captures = self.ids()?;
                TKind::Lambda(Box::new(TLambda { params, captures, body: self.texpr()? }))
            }
            K_CALL_INDIRECT => {
                let f = self.texpr()?;
                TKind::CallIndirect(Box::new(f), self.texprs()?)
            }
            K_BUILTIN => {
                let at = self.pos;
                let i = self.usize()?;
                let b = *Builtin::ALL.get(i).ok_or_else(|| format!("unknown built-in {i} at offset {at}"))?;
                let args = self.texprs()?;
                TKind::Builtin(b, args, self.str()?)
            }
            K_LIST => TKind::List(self.texprs()?),
            K_VARIANT => {
                let tag = self.u32()?;
                TKind::Variant(tag, self.texprs()?)
            }
            K_RECORD => {
                let k = self.count()?;
                let mut fs = Vec::new();
                for _ in 0..k {
                    let i = self.u32()?;
                    fs.push((i, self.texpr()?));
                }
                TKind::Record(fs)
            }
            K_FIELD => {
                let x = self.texpr()?;
                TKind::Field(Box::new(x), self.u32()?)
            }
            K_MATCH => {
                let s = self.texpr()?;
                let k = self.count()?;
                let mut arms = Vec::new();
                for _ in 0..k {
                    let pat = self.tpat()?;
                    let guard = if self.bool()? { Some(self.texpr()?) } else { None };
                    arms.push(TArm { pat, guard, body: self.texpr()? });
                }
                TKind::Match(Box::new(s), arms)
            }
            K_SEQ => TKind::Seq(self.texprs()?),
            t => return Err(bad_tag("typed expression", t, self.pos)),
        };
        Ok(TExpr { ty, kind })
    }

    pub fn texprs(&mut self) -> R<Vec<TExpr>> {
        let n = self.count()?;
        (0..n).map(|_| self.texpr()).collect()
    }

    fn ids(&mut self) -> R<Vec<u32>> {
        let n = self.count()?;
        (0..n).map(|_| self.u32()).collect()
    }

    pub fn tpat(&mut self) -> R<TPat> {
        self.enter()?;
        let p = self.tpat_inner();
        self.leave();
        p
    }

    fn tpat_inner(&mut self) -> R<TPat> {
        Ok(match self.byte()? {
            P_WILD => TPat::Wild,
            P_BIND => TPat::Bind(self.u32()?),
            P_INT => TPat::Int(self.int()?),
            P_STR => TPat::Str(self.str()?),
            P_BOOL => TPat::Bool(self.bool()?),
            P_UNIT => TPat::Unit,
            P_VARIANT => {
                let tag = self.u32()?;
                let k = self.count()?;
                TPat::Variant(tag, (0..k).map(|_| self.tpat()).collect::<R<_>>()?)
            }
            P_RECORD => {
                let k = self.count()?;
                let mut fs = Vec::new();
                for _ in 0..k {
                    let i = self.u32()?;
                    fs.push((i, self.tpat()?));
                }
                TPat::Record(fs)
            }
            t => return Err(bad_tag("typed pattern", t, self.pos)),
        })
    }

    /// A definition (see [`Writer::def`]).
    pub fn def(&mut self) -> R<Def> {
        Ok(match self.byte()? {
            0 => Def::Type(self.typedef()?),
            1 => Def::Fn(self.fndef()?),
            2 => {
                let name = self.str()?;
                Def::Test(TestDef { name, body: self.expr()? })
            }
            t => return Err(bad_tag("definition", t, self.pos)),
        })
    }

    pub fn typedef(&mut self) -> R<TypeDef> {
        let name = self.str()?;
        let params = self.strs()?;
        let body = match self.byte()? {
            0 => {
                let k = self.count()?;
                let mut fs = Vec::new();
                for _ in 0..k {
                    let f = self.str()?;
                    fs.push((f, self.ty()?));
                }
                TypeBody::Record(fs)
            }
            1 => {
                let k = self.count()?;
                let mut vs = Vec::new();
                for _ in 0..k {
                    let name = self.str()?;
                    vs.push(Variant { name, fields: self.types()? });
                }
                TypeBody::Enum(vs)
            }
            t => return Err(bad_tag("type body", t, self.pos)),
        };
        Ok(TypeDef { name, params, body })
    }

    pub fn tfn(&mut self) -> R<TFn> {
        let def = self.fndef()?;
        let locals = self.types()?;
        let req = self.texprs()?;
        let ens = self.texprs()?;
        let result_local = if self.bool()? { Some(self.u32()?) } else { None };
        let body = self.texpr()?;
        let instance = if self.bool()? {
            let g = self.usize()?;
            Some((g, self.types()?))
        } else {
            None
        };
        Ok(TFn { def, locals, req, ens, result_local, body, instance })
    }
}

fn bad_tag(what: &str, t: u8, pos: usize) -> String {
    format!("unknown {what} tag {t} at offset {}", pos.saturating_sub(1))
}

/// Native stack of the thread [`decode`] runs on: a debug build needs about 10 KiB per level of
/// nesting, so [`MAX_DEPTH`] levels fit with a wide margin wherever the caller runs.
pub const DECODE_STACK: usize = 64 << 20;

/// Decodes `QALT` bytes; every index is checked (see the module documentation). Runs on its own
/// thread with a [`DECODE_STACK`] stack, so no input can overflow the caller's stack.
pub fn decode(bytes: &[u8]) -> Result<TProgram, String> {
    std::thread::scope(|scope| {
        let t = std::thread::Builder::new().name("qal-tform".into()).stack_size(DECODE_STACK).spawn_scoped(scope, || decode_here(bytes));
        match t {
            Ok(h) => h.join().unwrap_or_else(|_| Err("internal error while decoding a typed form".into())),
            Err(e) => Err(format!("cannot start the decoding thread: {e}")),
        }
    })
}

/// [`decode`] on the calling thread (which needs a stack of about 10 KiB per nesting level).
pub fn decode_here(bytes: &[u8]) -> Result<TProgram, String> {
    let mut r = Reader::new(bytes);
    if r.bytes(4)? != MAGIC {
        return Err("not a QALT typed form".into());
    }
    let v = r.byte()?;
    if v != VERSION {
        return Err(format!("unsupported QALT version {v}"));
    }
    let n = r.count()?;
    let types = (0..n).map(|_| r.typedef()).collect::<R<Vec<_>>>()?;
    let n = r.count()?;
    let fns = (0..n).map(|_| r.tfn()).collect::<R<Vec<_>>>()?;
    let n = r.count()?;
    let mut tests = Vec::new();
    for _ in 0..n {
        let name = r.str()?;
        let locals = r.types()?;
        tests.push(TTest { name, locals, body: r.texpr()? });
    }
    let n = r.count()?;
    let generics = (0..n).map(|_| r.tfn()).collect::<R<Vec<_>>>()?;
    r.finish()?;
    let p = TProgram { types, fns, tests, generics };
    validate(&p)?;
    Ok(p)
}

/// Every reference of a decoded program is in range: functions, generics, instances, locals.
fn validate(p: &TProgram) -> Result<(), String> {
    struct V<'a> {
        p: &'a TProgram,
        locals: usize,
        what: String,
    }
    impl V<'_> {
        fn local(&self, id: u32) -> R<()> {
            if (id as usize) < self.locals {
                Ok(())
            } else {
                Err(format!("{}: local {id} out of range ({} locals)", self.what, self.locals))
            }
        }

        /// Variant tags, payload arity and record field indices agree with the declared types.
        fn shape(&self, e: &TExpr) -> R<()> {
            let bad = |m: String| Err(format!("{}: {m}", self.what));
            match &e.kind {
                TKind::Variant(tag, args) => {
                    let names = self.p.variant_names(&e.ty);
                    if *tag as usize >= names.len() {
                        return bad(format!("variant {tag} of `{}` out of range", e.ty));
                    }
                    if self.p.variant_fields(&e.ty, *tag).len() != args.len() {
                        return bad(format!("variant {tag} of `{}` with {} values", e.ty, args.len()));
                    }
                }
                TKind::Record(fs) => {
                    let n = self.p.record_fields(&e.ty).len();
                    if fs.len() != n || fs.iter().any(|(i, _)| *i as usize >= n) {
                        return bad(format!("record `{}` with fields out of range", e.ty));
                    }
                }
                TKind::Field(x, i) if *i as usize >= self.p.record_fields(&x.ty).len() => return bad(format!("field {i} of `{}` out of range", x.ty)),
                TKind::Builtin(b, args, _) if args.len() != b.arity() => return bad(format!("`{}` with {} arguments", b.name(), args.len())),
                _ => {}
            }
            Ok(())
        }

        /// A pattern against the type of the value it matches.
        fn pat_shape(&self, p: &TPat, ty: &Type) -> R<()> {
            match p {
                TPat::Variant(tag, ps) => {
                    let fields = self.p.variant_fields(ty, *tag);
                    if *tag as usize >= self.p.variant_names(ty).len() || fields.len() != ps.len() {
                        return Err(format!("{}: pattern variant {tag} does not fit `{ty}`", self.what));
                    }
                    ps.iter().zip(&fields).try_for_each(|(p, t)| self.pat_shape(p, t))
                }
                TPat::Record(fs) => {
                    let fields = self.p.record_fields(ty);
                    fs.iter().try_for_each(|(i, p)| match fields.get(*i as usize) {
                        Some((_, t)) => self.pat_shape(p, t),
                        None => Err(format!("{}: pattern field {i} does not fit `{ty}`", self.what)),
                    })
                }
                _ => Ok(()),
            }
        }

        fn e(&self, e: &TExpr) -> R<()> {
            self.shape(e)?;
            if let TKind::Match(s, arms) = &e.kind {
                arms.iter().try_for_each(|a| self.pat_shape(&a.pat, &s.ty))?;
            }
            let fns = self.p.fns.len();
            let generics = self.p.generics.len();
            match &e.kind {
                TKind::Int(_) | TKind::Float(_) | TKind::Bool(_) | TKind::Str(_) | TKind::Unit => Ok(()),
                TKind::Local(id) => self.local(*id),
                TKind::FnRef(i) if *i < fns => Ok(()),
                TKind::FnRefG(i, _) if *i < generics => Ok(()),
                TKind::FnRef(i) | TKind::FnRefG(i, _) => Err(format!("{}: function {i} out of range", self.what)),
                TKind::Bin(_, l, r, _) => {
                    self.e(l)?;
                    self.e(r)
                }
                TKind::Un(_, x, _) | TKind::Field(x, _) => self.e(x),
                TKind::If(c, t, f) => {
                    self.e(c)?;
                    self.e(t)?;
                    self.e(f)
                }
                TKind::Let(id, v, b) => {
                    self.local(*id)?;
                    self.e(v)?;
                    self.e(b)
                }
                TKind::Call(i, args) if *i < fns => args.iter().try_for_each(|a| self.e(a)),
                TKind::CallG(i, _, args) if *i < generics => args.iter().try_for_each(|a| self.e(a)),
                TKind::Call(i, _) | TKind::CallG(i, _, _) => Err(format!("{}: function {i} out of range", self.what)),
                TKind::Lambda(l) => {
                    l.params.iter().chain(&l.captures).try_for_each(|id| self.local(*id))?;
                    self.e(&l.body)
                }
                TKind::CallIndirect(f, args) => {
                    self.e(f)?;
                    args.iter().try_for_each(|a| self.e(a))
                }
                TKind::Builtin(_, xs, _) | TKind::List(xs) | TKind::Variant(_, xs) | TKind::Seq(xs) => xs.iter().try_for_each(|a| self.e(a)),
                TKind::Record(fs) => fs.iter().try_for_each(|(_, x)| self.e(x)),
                TKind::Match(s, arms) => {
                    self.e(s)?;
                    for a in arms {
                        self.pat(&a.pat)?;
                        if let Some(g) = &a.guard {
                            self.e(g)?;
                        }
                        self.e(&a.body)?;
                    }
                    Ok(())
                }
            }
        }

        fn pat(&self, p: &TPat) -> R<()> {
            match p {
                TPat::Bind(id) => self.local(*id),
                TPat::Variant(_, ps) => ps.iter().try_for_each(|p| self.pat(p)),
                TPat::Record(fs) => fs.iter().try_for_each(|(_, p)| self.pat(p)),
                _ => Ok(()),
            }
        }
    }
    let check_fn = |f: &TFn, kind: &str| -> R<()> {
        let v = V { p, locals: f.locals.len(), what: format!("{kind} `{}`", f.def.name) };
        if let Some(r) = f.result_local {
            v.local(r)?;
        }
        if let Some((g, _)) = &f.instance {
            if *g >= p.generics.len() {
                return Err(format!("{}: instance of generic {g} out of range", v.what));
            }
        }
        f.req.iter().chain(&f.ens).chain([&f.body]).try_for_each(|e| v.e(e))
    };
    for f in &p.fns {
        check_fn(f, "fn")?;
    }
    for f in &p.generics {
        check_fn(f, "generic")?;
    }
    for t in &p.tests {
        V { p, locals: t.locals.len(), what: format!("test `{}`", t.name) }.e(&t.body)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(n: i64) -> TExpr {
        TExpr { ty: Type::Int, kind: TKind::Int(n) }
    }

    fn sample() -> TProgram {
        let pair = TypeDef { name: "Pair".into(), params: vec!["A".into()], body: TypeBody::Record(vec![("a".into(), Type::Param("A".into())), ("n".into(), Type::Int)]) };
        let e = TypeDef { name: "E".into(), params: vec![], body: TypeBody::Enum(vec![Variant { name: "X".into(), fields: vec![] }, Variant { name: "Y".into(), fields: vec![Type::Str, Type::Bytes] }]) };
        let def = FnDef {
            name: "f".into(),
            tparams: vec![],
            eparams: vec![],
            params: vec![Param { name: "x".into(), ty: Type::Int }],
            ret: Type::Int,
            effects: vec!["io".into()],
            req: vec![Expr::Bin(BinOp::Ge, Box::new(Expr::Var("x".into())), Box::new(Expr::Int(-1)))],
            ens: vec![],
            body: Expr::Match(
                Box::new(Expr::Var("x".into())),
                vec![
                    Arm { pat: Pat::Int(i64::MIN), guard: Some(Expr::Bool(true)), body: Expr::Float(-0.0) },
                    Arm { pat: Pat::Record("R".into(), vec![("k".into(), Pat::Bind("y".into()))]), guard: None, body: Expr::Lambda(vec![("z".into(), Some(Type::Var(3)))], Box::new(Expr::Str("é".into()))) },
                ],
            ),
        };
        let body = TExpr {
            ty: Type::Int,
            kind: TKind::Let(
                1,
                Box::new(TExpr {
                    ty: Type::Fn(vec![Type::Int], Box::new(Type::Int), vec![]),
                    kind: TKind::Lambda(Box::new(TLambda { params: vec![2], captures: vec![0], body: TExpr { ty: Type::Int, kind: TKind::Bin(BinOp::Add, Box::new(TExpr { ty: Type::Int, kind: TKind::Local(2) }), Box::new(int(i64::MAX)), "body.value.body".into()) } })),
                }),
                Box::new(TExpr {
                    ty: Type::Int,
                    kind: TKind::Match(
                        Box::new(TExpr { ty: Type::option(Type::Int), kind: TKind::Variant(1, vec![int(5)]) }),
                        vec![
                            TArm { pat: TPat::Variant(1, vec![TPat::Bind(3)]), guard: Some(TExpr { ty: Type::Bool, kind: TKind::Bool(false) }), body: TExpr { ty: Type::Int, kind: TKind::CallIndirect(Box::new(TExpr { ty: Type::Unit, kind: TKind::Local(1) }), vec![int(1)]) } },
                            TArm { pat: TPat::Wild, guard: None, body: TExpr { ty: Type::Int, kind: TKind::Builtin(Builtin::BitXor, vec![int(-1), TExpr { ty: Type::Int, kind: TKind::Call(0, vec![int(0)]) }], "body.arms[1]".into()) } },
                        ],
                    ),
                }),
            ),
        };
        let f = TFn { def: def.clone(), locals: vec![Type::Int, Type::Unit, Type::Int, Type::Int, Type::Int], req: vec![int(1)], ens: vec![], result_local: Some(4), body, instance: Some((0, vec![Type::App("Pair".into(), vec![Type::Str])])) };
        let g = TFn { def, locals: vec![Type::Param("A".into())], req: vec![], ens: vec![], result_local: None, body: TExpr { ty: Type::Unit, kind: TKind::Seq(vec![TExpr { ty: Type::Unit, kind: TKind::FnRefG(0, vec![Type::Int]) }, TExpr { ty: Type::Unit, kind: TKind::CallG(0, vec![], vec![]) }]) }, instance: None };
        let t = TTest {
            name: "t".into(),
            locals: vec![Type::Named("Q".into())],
            body: TExpr {
                ty: Type::Bool,
                kind: TKind::Seq(vec![
                    TExpr { ty: Type::Str, kind: TKind::Str("".into()) },
                    TExpr { ty: Type::Float, kind: TKind::Float(f64::from_bits(0x7ff8_0000_0000_0001)) },
                    TExpr { ty: Type::Unit, kind: TKind::Unit },
                    TExpr { ty: Type::Int, kind: TKind::Field(Box::new(TExpr { ty: Type::Named("R".into()), kind: TKind::Record(vec![(1, int(2)), (0, int(3))]) }), 1) },
                    TExpr { ty: Type::Int, kind: TKind::If(Box::new(TExpr { ty: Type::Bool, kind: TKind::Un(UnOp::Not, Box::new(TExpr { ty: Type::Bool, kind: TKind::Bool(true) }), "p".into()) }), Box::new(int(0)), Box::new(TExpr { ty: Type::Unit, kind: TKind::FnRef(0) })) },
                    TExpr { ty: Type::list(Type::Int), kind: TKind::List(vec![int(0), int(-64), int(63), int(64)]) },
                    TExpr { ty: Type::Bool, kind: TKind::Match(Box::new(TExpr { ty: Type::Named("Q".into()), kind: TKind::Local(0) }), vec![TArm { pat: TPat::Record(vec![(0, TPat::Str("s".into())), (1, TPat::Bool(true)), (2, TPat::Unit), (3, TPat::Int(-7))]), guard: None, body: TExpr { ty: Type::Bool, kind: TKind::Bool(true) } }]) },
                ]),
            },
        };
        let r = TypeDef { name: "R".into(), params: vec![], body: TypeBody::Record(vec![("a".into(), Type::Int), ("b".into(), Type::Int)]) };
        let q = TypeDef { name: "Q".into(), params: vec![], body: TypeBody::Record(vec![("s".into(), Type::Str), ("b".into(), Type::Bool), ("u".into(), Type::Unit), ("n".into(), Type::Int)]) };
        TProgram { types: vec![pair, e, r, q], fns: vec![f], tests: vec![t], generics: vec![g] }
    }

    fn same_bits(a: &TProgram, b: &TProgram) -> bool {
        // Float NaN payloads compare unequal with `==`: compare the encodings too.
        encode(a) == encode(b)
    }

    #[test]
    fn typed_programs_round_trip() {
        let p = sample();
        let bytes = encode(&p);
        let back = decode(&bytes).unwrap();
        assert!(same_bits(&p, &back));
        assert_eq!(format!("{p:?}"), format!("{back:?}"));
        assert_eq!(decode(&encode(&TProgram::default())).unwrap(), TProgram::default());
        assert_eq!(&bytes[..5], b"QALT\x01");
    }

    #[test]
    fn leb128_is_minimal_both_ways() {
        for n in [0i64, 1, -1, 63, 64, -64, -65, 127, 128, i64::MAX, i64::MIN, 1 << 40, -(1 << 40)] {
            let mut b = Vec::new();
            put_int(&mut b, n);
            let mut r = Reader::new(&b);
            assert_eq!(r.int(), Ok(n), "{n}");
            assert!(r.at_end());
        }
        for n in [0u64, 1, 127, 128, 300, u32::MAX as u64, u64::MAX] {
            let mut b = Vec::new();
            put_uint(&mut b, n);
            let mut r = Reader::new(&b);
            assert_eq!(r.uint(), Ok(n), "{n}");
        }
        // Non-minimal, too long and out-of-range encodings are refused.
        for bad in [&[0x80, 0x00][..], &[0xff; 11][..], &[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f][..], &[0x80][..]] {
            assert!(Reader::new(bad).uint().is_err(), "{bad:?}");
        }
        for bad in [&[0x80, 0x00][..], &[0xff, 0x7f][..], &[0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7e][..]] {
            assert!(Reader::new(bad).int().is_err(), "{bad:?}");
        }
    }

    #[test]
    fn corrupt_forms_are_errors_not_panics() {
        let bytes = encode(&sample());
        assert!(decode(&bytes[..bytes.len() - 1]).is_err());
        assert!(decode(&[&bytes[..], &[0]].concat()).unwrap_err().contains("trailing"));
        assert!(decode(b"QALX\x01").is_err() && decode(b"QALT\x02\x00\x00\x00\x00").unwrap_err().contains("version"));
        // Every truncation and every single-byte change decodes to an error or a program, never a panic.
        for i in 0..bytes.len() {
            let _ = decode(&bytes[..i]);
            for v in [0u8, 1, 0x7f, 0x80, 0xff] {
                let mut b = bytes.clone();
                b[i] = v;
                let _ = decode(&b);
            }
        }
        // Out-of-range references are refused.
        let mut p = sample();
        p.generics.clear();
        assert!(decode(&encode(&p)).unwrap_err().contains("out of range"));
        let mut p = sample();
        p.fns[0].locals.truncate(2);
        assert!(decode(&encode(&p)).unwrap_err().contains("local"));
        // Shapes must fit the declared types (review M6).
        let mut p = sample();
        p.tests[0].body = TExpr { ty: Type::option(Type::Int), kind: TKind::Variant(2, vec![]) };
        assert!(decode(&encode(&p)).unwrap_err().contains("out of range"));
        let mut p = sample();
        p.tests[0].body = TExpr { ty: Type::option(Type::Int), kind: TKind::Variant(1, vec![]) };
        assert!(decode(&encode(&p)).unwrap_err().contains("with 0 values"));
        let mut p = sample();
        p.tests[0].body = TExpr { ty: Type::Named("R".into()), kind: TKind::Record(vec![(0, int(1))]) };
        assert!(decode(&encode(&p)).unwrap_err().contains("record"));
        // Deep nesting is bounded; `decode` brings its own stack, so even a small caller stack
        // (2 MiB) cannot overflow, whatever the input.
        let deep = |n: usize| {
            let mut e = int(0);
            for _ in 0..n {
                e = TExpr { ty: Type::list(Type::Int), kind: TKind::Seq(vec![e]) };
            }
            TProgram { tests: vec![TTest { name: "d".into(), locals: vec![], body: e }], ..TProgram::default() }
        };
        let big = std::thread::Builder::new().stack_size(256 << 20);
        let (deepest, too_deep) = big.spawn(move || (encode(&deep(MAX_DEPTH - 8)), encode(&deep(MAX_DEPTH)))).unwrap().join().unwrap();
        let small = std::thread::Builder::new().stack_size(2 << 20);
        let ok = small
            .spawn(move || {
                let malicious = [&b"QALT\x01\x00\x00\x01\x01d\x00"[..], &[T_LIST; 4 * MAX_DEPTH]].concat();
                let accepted = decode(&deepest).map(std::mem::forget).is_ok();
                (accepted, decode(&too_deep).unwrap_err(), decode(&malicious).unwrap_err())
            })
            .unwrap()
            .join()
            .unwrap();
        assert!(ok.0 && ok.1.contains("deeply") && ok.2.contains("deeply"), "{ok:?}");
        // Absurd counts fail before allocating.
        assert!(decode(b"QALT\x01\xff\xff\xff\xff\x0f").unwrap_err().contains("exceeds"));
    }

    #[test]
    fn builtin_indices_are_stable() {
        assert_eq!(builtin_index(Builtin::StrLen), 0);
        assert_eq!(builtin_index(Builtin::DbQuery), 56);
        assert_eq!(builtin_index(Builtin::Sha256), 70);
        for (i, b) in Builtin::ALL.iter().enumerate() {
            assert_eq!(builtin_index(*b), i);
        }
    }
}
