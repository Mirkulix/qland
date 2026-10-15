//! Canonical form and content hashes of definitions (spec §5).
//!
//! Every definition is encoded into a canonical binary form in which local variables are De Bruijn
//! indices (their names are kept separately as metadata), references to other definitions are
//! their hashes, and field and variant names stay (they are semantics). The hash of a definition is
//! the SHA-256 of that encoding, so renaming changes no hash and identical logic has one hash.
//!
//! A group of two or more mutually recursive definitions (a strongly connected component of the
//! reference graph) is encoded together: the group form holds all members, references inside the
//! group are member positions, and each member's canonical form is the group's size, the member's
//! position and the group form's hash. The positions come from a canonical labelling of the group
//! (refinement plus a search over all ways to break ties, keeping the smallest group form), so
//! neither the names nor the order of definition matter. A group with interchangeable members
//! (an automorphism: two members that are the same definition) is rejected, since its members
//! could not get name-independent hashes.
//!
//! ```text
//! canonical := "QAL\0" version:u8 1 0 member                      (a definition outside any group)
//!            | "QAL\0" version:u8 size:varint index:varint group-hash[32]   (member of a group, size >= 2)
//! group     := "QAL\0" version:u8 size:varint member{size}        (stored once per group)
//! member    := kind:u8 (0 type | 1 fn | 2 test) body
//!            | 3 tparams:n type-body | 4 tparams:n fn-body           (generic type / fn: n >= 1 type parameters,
//!              bound in order; a type parameter is written `T_PARAM index`)
//! type body := 0 n (field:str type){n}  |  1 n (variant:str k type{k}){n}
//! fn body   := n type{n} ret:type effects req:n expr{n} ens:n expr{n} body:expr   (params bound in order;
//!              `r` is bound for `ens` without being stored)
//! test body := expr
//! ref       := 0 hash[32] | 1 group-index:varint
//! effects   := n str{n}              (sorted, without duplicates)
//! str       := len:varint utf8-bytes; varints are minimal LEB128; ints are 8 bytes little endian;
//!              floats are their IEEE bit pattern; flags and Bools are 0 or 1
//! ```
//!
//! The expression, pattern and type tags are the constants in `tag` below.
//!
//! Versions: version 1 is the Phase 1 format. Version 2 adds generic members (kinds 3, 4), the types
//! `T_APP`, `T_PARAM`, `T_MAP`, `T_BYTES` and the expression `E_LAMBDA`. A form is written with the
//! lowest version that has every construct it uses (a group: the highest of its members), so a
//! definition without Phase 2 constructs keeps its Phase 1 bytes and hash. Decoding is strict both
//! ways: a version 1 form must not use version 2 constructs and a version 2 form must use one.

use crate::ast::*;
use crate::tast::Builtin;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;

pub const MAGIC: &[u8; 4] = b"QAL\0";
/// The highest canonical format version this code reads and writes (see the module documentation).
pub const VERSION: u8 = 2;

/// Nesting accepted while decoding: the parser's bounds (expression and pattern height 256, type
/// nesting 256), so nothing decodes that the text form could not express.
const MAX_DECODE_DEPTH: usize = 256;
const MAX_DECODE_TYPE_DEPTH: usize = 256;
/// Bounds on the canonical-labelling search of one group: search-tree nodes and member encodings.
const MAX_LABEL_NODES: usize = 1024;
const MAX_LABEL_WORK: usize = 2_000_000;

/// SHA-256 content hash of a definition's canonical form.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Hash(pub [u8; 32]);

impl Hash {
    pub fn of(bytes: &[u8]) -> Hash {
        let digest = Sha256::digest(bytes);
        let mut out = [0u8; 32];
        out.copy_from_slice(digest.as_slice());
        Hash(out)
    }

    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Parses exactly 64 hexadecimal digits.
    pub fn from_hex(s: &str) -> Option<Hash> {
        if s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        let mut out = [0u8; 32];
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
        }
        Some(Hash(out))
    }

    /// The first 12 hex digits, used as a placeholder name for definitions without a name.
    pub fn short(&self) -> String {
        self.to_hex()[..12].to_string()
    }
}

impl fmt::Display for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.short())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Type,
    Fn,
    Test,
}

impl Kind {
    pub fn of(d: &Def) -> Kind {
        match d {
            Def::Type(_) => Kind::Type,
            Def::Fn(_) => Kind::Fn,
            Def::Test(_) => Kind::Test,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Type => "type",
            Kind::Fn => "fn",
            Kind::Test => "test",
        }
    }

    pub fn parse(s: &str) -> Option<Kind> {
        match s {
            "type" => Some(Kind::Type),
            "fn" => Some(Kind::Fn),
            "test" => Some(Kind::Test),
            _ => None,
        }
    }

    fn tag(self) -> u8 {
        match self {
            Kind::Type => 0,
            Kind::Fn => 1,
            Kind::Test => 2,
        }
    }
}

/// The canonical form of one definition of a program, as computed by [`hash_program`].
#[derive(Debug, Clone, PartialEq)]
pub struct Canon {
    pub name: String,
    pub kind: Kind,
    pub hash: Hash,
    /// The bytes whose SHA-256 is `hash`.
    pub canonical: Vec<u8>,
    /// For a member of a group: the hash of the group form (in [`Hashed::groups`]).
    pub group: Option<Hash>,
    /// Names of the local variables in binding order (parameters first), for the text view.
    pub local_names: Vec<String>,
    /// Hashes of the other definitions this one refers to (sorted, without duplicates and itself).
    pub deps: Vec<Hash>,
    /// Names (in the program) of the definitions this one refers to, as written: sorted, without
    /// duplicates and itself. Several names can stand behind one dependency hash (identical
    /// definitions under two names); this says which ones the text used.
    pub refs: Vec<String>,
    /// The signature with type references written as `#<hash>`: `fn(Int, #…) -> Str !io` for
    /// functions, the body (`{ x: Int }`, `A | B(Int)`) for types, `Bool` for tests.
    pub signature: String,
    /// Effects of a function (sorted, without duplicates); empty for types and tests.
    pub effects: Vec<String>,
}

/// The result of [`hash_program`].
#[derive(Debug, Clone, PartialEq)]
pub struct Hashed {
    /// One entry per definition, in program order.
    pub defs: Vec<Canon>,
    /// Group forms by their hash (members refer to them through [`Canon::group`]).
    pub groups: std::collections::BTreeMap<Hash, Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HashError {
    /// The recursive group of these definitions has interchangeable members: some of them are the
    /// same definition under different names.
    Symmetric(Vec<String>),
    /// The canonical-labelling search of this recursive group exceeded its limit (very large groups
    /// whose members all look alike).
    TooLarge(Vec<String>),
    /// The program was not checked (an unresolved name) — never expected for checked programs.
    Internal(String),
}

impl fmt::Display for HashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HashError::Symmetric(names) => write!(f, "the recursive group {} has interchangeable members", names.join(", ")),
            HashError::TooLarge(names) => write!(f, "the recursive group of {} definitions is too large to label", names.len()),
            HashError::Internal(e) => f.write_str(e),
        }
    }
}

/// A stored canonical form: the member bytes and, for a member of a group, the group form.
#[derive(Debug, Clone, Copy)]
pub struct Form<'a> {
    pub canonical: &'a [u8],
    pub group: Option<&'a [u8]>,
}

impl<'a> Form<'a> {
    pub fn new(canonical: &'a [u8], group: Option<&'a [u8]>) -> Form<'a> {
        Form { canonical, group }
    }
}

/// What the header of a member's canonical form says.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Header {
    pub size: usize,
    pub index: usize,
    /// The group form's hash (groups of two or more).
    pub group: Option<Hash>,
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

    pub const E_INT: u8 = 0;
    pub const E_FLOAT: u8 = 1;
    pub const E_BOOL: u8 = 2;
    pub const E_STR: u8 = 3;
    pub const E_UNIT: u8 = 4;
    pub const E_LOCAL: u8 = 5;
    pub const E_FN_REF: u8 = 6;
    pub const E_VARIANT: u8 = 7;
    pub const E_CALL_LOCAL: u8 = 8;
    pub const E_CALL_FN: u8 = 9;
    pub const E_BUILTIN: u8 = 10;
    pub const E_BIN: u8 = 11;
    pub const E_UN: u8 = 12;
    pub const E_IF: u8 = 13;
    pub const E_LET: u8 = 14;
    pub const E_LIST: u8 = 15;
    pub const E_RECORD: u8 = 16;
    pub const E_FIELD: u8 = 17;
    pub const E_MATCH: u8 = 18;
    pub const E_SEQ: u8 = 19;
    pub const E_LAMBDA: u8 = 20;

    pub const P_WILD: u8 = 0;
    pub const P_BIND: u8 = 1;
    pub const P_INT: u8 = 2;
    pub const P_STR: u8 = 3;
    pub const P_BOOL: u8 = 4;
    pub const P_UNIT: u8 = 5;
    pub const P_VARIANT: u8 = 6;
    pub const P_RECORD: u8 = 7;

    /// Owner of a variant: the built-in Option, the built-in Result, or a declared enum (+ ref).
    pub const V_OPTION: u8 = 0;
    pub const V_RESULT: u8 = 1;
    pub const V_ENUM: u8 = 2;

    pub const R_HASH: u8 = 0;
    pub const R_GROUP: u8 = 1;
}
use tag::*;

const BINOPS: [BinOp; 14] = [
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

/// The operator's index in `BINOPS` (an exhaustive match, so a new operator cannot be forgotten).
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

/// A reference from one definition to another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ref {
    Hash(Hash),
    Group(u32),
}

/// What the names of a program refer to (variants and builtins are fixed by the program).
struct Env<'a> {
    fns: HashSet<&'a str>,
    types: HashSet<&'a str>,
    /// Variant name -> owning enum type name.
    variants: HashMap<&'a str, &'a str>,
}

impl<'a> Env<'a> {
    fn new(p: &'a Program) -> Env<'a> {
        let mut env = Env { fns: HashSet::new(), types: HashSet::new(), variants: HashMap::new() };
        for d in &p.defs {
            match d {
                Def::Fn(f) => {
                    env.fns.insert(&f.name);
                }
                Def::Type(t) => {
                    env.types.insert(&t.name);
                    if let TypeBody::Enum(vs) = &t.body {
                        for v in vs {
                            env.variants.insert(&v.name, &t.name);
                        }
                    }
                }
                Def::Test(_) => {}
            }
        }
        env
    }
}

/// Encodes one definition. Names of other definitions go through `refs`; the first unresolvable
/// name is kept in `err`.
struct Enc<'a, 'r> {
    env: &'a Env<'a>,
    refs: &'r mut dyn FnMut(&str) -> Option<Ref>,
    out: Vec<u8>,
    /// Visible local names, innermost last.
    scope: Vec<String>,
    /// Every binder's name in binding order (type parameters first).
    locals: Vec<String>,
    /// Type parameters of the encoded definition.
    tparams: Vec<String>,
    /// A version 2 construct was written.
    v2: bool,
    err: Option<String>,
}

impl Enc<'_, '_> {
    fn byte(&mut self, b: u8) {
        self.out.push(b);
    }

    fn varint(&mut self, n: u64) {
        put_varint(&mut self.out, n);
    }

    fn len(&mut self, n: usize) {
        self.varint(n as u64);
    }

    fn i64(&mut self, n: i64) {
        self.out.extend_from_slice(&n.to_le_bytes());
    }

    fn str(&mut self, s: &str) {
        self.len(s.len());
        self.out.extend_from_slice(s.as_bytes());
    }

    fn fail(&mut self, msg: String) {
        if self.err.is_none() {
            self.err = Some(msg);
        }
    }

    fn reference(&mut self, name: &str) {
        match (self.refs)(name) {
            Some(Ref::Hash(h)) => {
                self.byte(R_HASH);
                self.out.extend_from_slice(&h.0);
            }
            Some(Ref::Group(i)) => {
                self.byte(R_GROUP);
                self.varint(i as u64);
            }
            // Fail closed: never encode a placeholder reference.
            None => self.fail(format!("the reference to `{name}` cannot be resolved")),
        }
    }

    fn effects(&mut self, effs: &[String]) {
        let set: BTreeSet<&String> = effs.iter().collect();
        self.len(set.len());
        for e in set {
            self.str(e);
        }
    }

    fn ty(&mut self, t: &Type) {
        match t {
            Type::Int => self.byte(T_INT),
            Type::Float => self.byte(T_FLOAT),
            Type::Bool => self.byte(T_BOOL),
            Type::Unit => self.byte(T_UNIT),
            Type::Str => self.byte(T_STR),
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
            Type::Named(n) | Type::Param(n) if self.tparams.contains(n) => {
                self.v2 = true;
                self.byte(T_PARAM);
                let i = self.tparams.iter().position(|p| p == n).unwrap_or(0);
                self.len(i);
            }
            Type::App(n, args) => {
                self.v2 = true;
                if !self.env.types.contains(n.as_str()) {
                    self.fail(format!("unknown type `{n}`"));
                }
                self.byte(T_APP);
                self.reference(n);
                self.len(args.len());
                for a in args {
                    self.ty(a);
                }
            }
            Type::Map(k, v) => {
                self.v2 = true;
                self.byte(T_MAP);
                self.ty(k);
                self.ty(v);
            }
            Type::Bytes => {
                self.v2 = true;
                self.byte(T_BYTES);
            }
            Type::Param(n) => self.fail(format!("unknown type parameter `{n}`")),
            Type::Named(n) => {
                if !self.env.types.contains(n.as_str()) {
                    self.fail(format!("unknown type `{n}`"));
                }
                self.byte(T_NAMED);
                self.reference(n);
            }
            Type::Fn(ps, r, effs) => {
                self.byte(T_FN);
                self.len(ps.len());
                for p in ps {
                    self.ty(p);
                }
                self.ty(r);
                self.effects(effs);
            }
            Type::Var(_) => self.fail("inference variable in a written type".into()),
        }
    }

    fn bind(&mut self, name: &str) {
        self.scope.push(name.to_string());
        self.locals.push(name.to_string());
    }

    fn lookup(&self, name: &str) -> Option<usize> {
        self.scope.iter().rev().position(|n| n == name)
    }

    /// `None`/`Some`, `Ok`/`Err`, or a variant of a declared enum.
    fn variant_owner(&mut self, name: &str) -> bool {
        match name {
            "None" | "Some" => self.byte(V_OPTION),
            "Ok" | "Err" => self.byte(V_RESULT),
            _ => match self.env.variants.get(name).copied() {
                Some(owner) => {
                    self.byte(V_ENUM);
                    self.reference(owner);
                }
                None => return false,
            },
        }
        true
    }

    fn is_variant(&self, name: &str) -> bool {
        matches!(name, "None" | "Some" | "Ok" | "Err") || self.env.variants.contains_key(name)
    }

    fn exprs(&mut self, xs: &[Expr]) {
        self.len(xs.len());
        for x in xs {
            self.expr(x);
        }
    }

    fn variant(&mut self, name: &str, args: &[Expr]) {
        self.byte(E_VARIANT);
        self.variant_owner(name);
        self.str(name);
        self.exprs(args);
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Int(n) => {
                self.byte(E_INT);
                self.i64(*n);
            }
            Expr::Float(x) => {
                self.byte(E_FLOAT);
                self.out.extend_from_slice(&x.to_bits().to_le_bytes());
            }
            Expr::Bool(b) => {
                self.byte(E_BOOL);
                self.byte(*b as u8);
            }
            Expr::Str(s) => {
                self.byte(E_STR);
                self.str(s);
            }
            Expr::Unit => self.byte(E_UNIT),
            // Resolution order is the checker's: local, function, variant.
            Expr::Var(v) => {
                if let Some(i) = self.lookup(v) {
                    self.byte(E_LOCAL);
                    self.len(i);
                } else if self.env.fns.contains(v.as_str()) {
                    self.byte(E_FN_REF);
                    self.reference(v);
                } else if self.is_variant(v) {
                    self.variant(v, &[]);
                } else {
                    self.fail(format!("unknown name `{v}`"));
                }
            }
            // Resolution order is the checker's: local, function, variant, built-in.
            Expr::Call(name, args) => {
                if let Some(i) = self.lookup(name) {
                    self.byte(E_CALL_LOCAL);
                    self.len(i);
                    self.exprs(args);
                } else if self.env.fns.contains(name.as_str()) {
                    self.byte(E_CALL_FN);
                    self.reference(name);
                    self.exprs(args);
                } else if self.is_variant(name) {
                    self.variant(name, args);
                } else if Builtin::NAMES.contains(&name.as_str()) {
                    self.byte(E_BUILTIN);
                    self.str(name);
                    self.exprs(args);
                } else {
                    self.fail(format!("unknown name `{name}`"));
                }
            }
            Expr::Bin(op, l, r) => {
                self.byte(E_BIN);
                self.byte(binop_tag(*op));
                self.expr(l);
                self.expr(r);
            }
            Expr::Un(op, x) => {
                self.byte(E_UN);
                self.byte(match op {
                    UnOp::Neg => 0,
                    UnOp::Not => 1,
                });
                self.expr(x);
            }
            Expr::If(c, t, f) => {
                self.byte(E_IF);
                self.expr(c);
                self.expr(t);
                self.expr(f);
            }
            Expr::Let(name, ann, v, body) => {
                self.byte(E_LET);
                match ann {
                    Some(t) => {
                        self.byte(1);
                        self.ty(t);
                    }
                    None => self.byte(0),
                }
                self.expr(v);
                self.bind(name);
                self.expr(body);
                self.scope.pop();
            }
            Expr::List(items) => {
                self.byte(E_LIST);
                self.exprs(items);
            }
            Expr::Record(name, fields) => {
                self.byte(E_RECORD);
                if !self.env.types.contains(name.as_str()) {
                    self.fail(format!("unknown type `{name}`"));
                }
                self.reference(name);
                self.len(fields.len());
                for (f, x) in fields {
                    self.str(f);
                    self.expr(x);
                }
            }
            Expr::Field(target, field) => {
                self.byte(E_FIELD);
                self.expr(target);
                self.str(field);
            }
            Expr::Match(s, arms) => {
                self.byte(E_MATCH);
                self.expr(s);
                self.len(arms.len());
                for a in arms {
                    let mark = self.scope.len();
                    self.pat(&a.pat);
                    match &a.guard {
                        Some(g) => {
                            self.byte(1);
                            self.expr(g);
                        }
                        None => self.byte(0),
                    }
                    self.expr(&a.body);
                    self.scope.truncate(mark);
                }
            }
            Expr::Seq(items) => {
                self.byte(E_SEQ);
                self.exprs(items);
            }
            Expr::Lambda(params, body) => {
                self.v2 = true;
                self.byte(E_LAMBDA);
                self.len(params.len());
                for (_, ann) in params {
                    match ann {
                        Some(t) => {
                            self.byte(1);
                            self.ty(t);
                        }
                        None => self.byte(0),
                    }
                }
                for (n, _) in params {
                    self.bind(n);
                }
                self.expr(body);
                for _ in params {
                    self.scope.pop();
                }
            }
        }
    }

    fn pat(&mut self, p: &Pat) {
        match p {
            Pat::Wild => self.byte(P_WILD),
            Pat::Bind(n) => {
                self.byte(P_BIND);
                self.bind(n);
            }
            Pat::Int(n) => {
                self.byte(P_INT);
                self.i64(*n);
            }
            Pat::Str(s) => {
                self.byte(P_STR);
                self.str(s);
            }
            Pat::Bool(b) => {
                self.byte(P_BOOL);
                self.byte(*b as u8);
            }
            Pat::Unit => self.byte(P_UNIT),
            Pat::Variant(name, ps) => {
                self.byte(P_VARIANT);
                if !self.variant_owner(name) {
                    self.fail(format!("unknown variant `{name}`"));
                }
                self.str(name);
                self.len(ps.len());
                for p in ps {
                    self.pat(p);
                }
            }
            Pat::Record(name, fs) => {
                self.byte(P_RECORD);
                if !self.env.types.contains(name.as_str()) {
                    self.fail(format!("unknown type `{name}`"));
                }
                self.reference(name);
                self.len(fs.len());
                for (f, p) in fs {
                    self.str(f);
                    self.pat(p);
                }
            }
        }
    }

    /// Type parameters: the member kind says the definition is generic; their names are locals.
    fn tparams(&mut self, kind: u8, names: &[String]) {
        self.v2 = true;
        self.byte(kind);
        self.len(names.len());
        self.tparams = names.to_vec();
        self.locals.extend(names.iter().cloned());
    }

    fn member(&mut self, d: &Def) {
        match d {
            Def::Type(t) if !t.params.is_empty() => self.tparams(3, &t.params),
            Def::Fn(f) if !f.tparams.is_empty() => self.tparams(4, &fn_tparams(f)),
            _ => self.byte(Kind::of(d).tag()),
        }
        match d {
            Def::Type(t) => match &t.body {
                TypeBody::Record(fields) => {
                    self.byte(0);
                    self.len(fields.len());
                    for (f, ty) in fields {
                        self.str(f);
                        self.ty(ty);
                    }
                }
                TypeBody::Enum(vs) => {
                    self.byte(1);
                    self.len(vs.len());
                    for v in vs {
                        self.str(&v.name);
                        self.len(v.fields.len());
                        for ty in &v.fields {
                            self.ty(ty);
                        }
                    }
                }
            },
            Def::Fn(f) => {
                self.len(f.params.len());
                for p in &f.params {
                    self.ty(&p.ty);
                }
                self.ty(&f.ret);
                self.effects(&f.effects);
                for p in &f.params {
                    self.bind(&p.name);
                }
                self.exprs(&f.req);
                self.len(f.ens.len());
                if !f.ens.is_empty() {
                    // `r` is bound for the postconditions only and is not a stored local name.
                    self.scope.push(RESULT_VAR.to_string());
                    for c in &f.ens {
                        self.expr(c);
                    }
                    self.scope.pop();
                }
                self.expr(&f.body);
            }
            Def::Test(t) => self.expr(&t.body),
        }
    }
}

/// Encodes `d` (its member part only), returning (bytes, local names) or the first unresolved name.
fn encode_member(env: &Env, d: &Def, refs: &mut dyn FnMut(&str) -> Option<Ref>) -> Result<(Vec<u8>, Vec<String>), String> {
    encode_versioned(env, d, refs).map(|(b, l, _)| (b, l))
}

/// Like [`encode_member`], also returning the format version the member needs.
fn encode_versioned(env: &Env, d: &Def, refs: &mut dyn FnMut(&str) -> Option<Ref>) -> Result<(Vec<u8>, Vec<String>, u8), String> {
    let mut enc = Enc { env, refs, out: Vec::new(), scope: Vec::new(), locals: Vec::new(), tparams: Vec::new(), v2: false, err: None };
    enc.member(d);
    match enc.err {
        Some(e) => Err(format!("`{}`: {e}", d.name())),
        None => Ok((enc.out, enc.locals, if enc.v2 { 2 } else { 1 })),
    }
}

/// Names of the program's definitions that `d` refers to (as resolved by the encoder).
fn references(env: &Env, d: &Def) -> Result<BTreeSet<String>, String> {
    let mut names = BTreeSet::new();
    let mut record = |n: &str| {
        names.insert(n.to_string());
        Some(Ref::Group(0))
    };
    encode_member(env, d, &mut record)?;
    Ok(names)
}

fn put_varint(out: &mut Vec<u8>, mut n: u64) {
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

fn prefix(size: usize, version: u8) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    out.push(version);
    put_varint(&mut out, size as u64);
    out
}

/// Canonical form of a definition outside any group.
fn single_form(member: &[u8], version: u8) -> Vec<u8> {
    let mut out = prefix(1, version);
    out.push(0);
    out.extend_from_slice(member);
    out
}

fn group_form(members: &[Vec<u8>], version: u8) -> Vec<u8> {
    let mut out = prefix(members.len(), version);
    for m in members {
        out.extend_from_slice(m);
    }
    out
}

fn member_form(size: usize, index: usize, group: &Hash, version: u8) -> Vec<u8> {
    let mut out = prefix(size, version);
    put_varint(&mut out, index as u64);
    out.extend_from_slice(&group.0);
    out
}

/// Strongly connected components of `edges` (indices), each listed after every component it reaches.
fn sccs(n: usize, edges: &[Vec<usize>]) -> Vec<Vec<usize>> {
    // Iterative Tarjan: no recursion, so long dependency chains cannot overflow the stack.
    let mut index = vec![usize::MAX; n];
    let mut low = vec![0; n];
    let mut on_stack = vec![false; n];
    let mut stack = Vec::new();
    let mut out = Vec::new();
    let mut next = 0;
    for root in 0..n {
        if index[root] != usize::MAX {
            continue;
        }
        let mut work: Vec<(usize, usize)> = vec![(root, 0)];
        index[root] = next;
        low[root] = next;
        next += 1;
        stack.push(root);
        on_stack[root] = true;
        while let Some(&mut (v, ref mut i)) = work.last_mut() {
            if *i < edges[v].len() {
                let w = edges[v][*i];
                *i += 1;
                if index[w] == usize::MAX {
                    index[w] = next;
                    low[w] = next;
                    next += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    work.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(index[w]);
                }
            } else {
                work.pop();
                if let Some(&(parent, _)) = work.last() {
                    low[parent] = low[parent].min(low[v]);
                }
                if low[v] == index[v] {
                    let mut comp = Vec::new();
                    while let Some(w) = stack.pop() {
                        on_stack[w] = false;
                        comp.push(w);
                        if w == v {
                            break;
                        }
                    }
                    comp.sort_unstable();
                    out.push(comp);
                }
            }
        }
    }
    out
}

/// Computes the canonical form and hash of every definition of `p` (result in `p.defs` order).
/// `p` must have passed the checker; unresolved names are an error, never a panic.
pub fn hash_program(p: &Program) -> Result<Hashed, HashError> {
    let internal = HashError::Internal;
    let env = Env::new(p);
    let mut by_name: HashMap<&str, usize> = HashMap::new();
    for (i, d) in p.defs.iter().enumerate() {
        if by_name.insert(d.name(), i).is_some() {
            return Err(internal(format!("`{}` is defined more than once", d.name())));
        }
    }
    let mut edges = Vec::with_capacity(p.defs.len());
    let mut ref_names: Vec<Vec<String>> = Vec::with_capacity(p.defs.len());
    for d in &p.defs {
        let names = references(&env, d).map_err(internal)?;
        edges.push(names.iter().filter_map(|n| by_name.get(n.as_str()).copied()).collect::<Vec<_>>());
        ref_names.push(names.into_iter().filter(|n| n != d.name() && by_name.contains_key(n.as_str())).collect::<Vec<_>>());
    }
    let mut hashes: Vec<Option<Hash>> = vec![None; p.defs.len()];
    let mut out: Vec<Option<Canon>> = vec![None; p.defs.len()];
    let mut groups = std::collections::BTreeMap::new();
    for comp in sccs(p.defs.len(), &edges) {
        let order = order_group(&env, p, &comp, &by_name, &hashes)?;
        let pos: HashMap<usize, u32> = order.iter().enumerate().map(|(k, &d)| (d, k as u32)).collect();
        let mut members = Vec::new();
        let mut locals = Vec::new();
        let mut version = 1;
        for &d in &order {
            let mut refs = |n: &str| resolve(n, &by_name, &pos, &hashes);
            let (bytes, names, v) = encode_versioned(&env, &p.defs[d], &mut refs).map_err(internal)?;
            members.push(bytes);
            locals.push(names);
            version = version.max(v);
        }
        let (forms, group): (Vec<Vec<u8>>, Option<Hash>) = if order.len() == 1 {
            (vec![single_form(&members[0], version)], None)
        } else {
            let g = group_form(&members, version);
            let gh = Hash::of(&g);
            groups.insert(gh, g);
            ((0..order.len()).map(|k| member_form(order.len(), k, &gh, version)).collect(), Some(gh))
        };
        for (k, &d) in order.iter().enumerate() {
            hashes[d] = Some(Hash::of(&forms[k]));
        }
        for (k, &d) in order.iter().enumerate() {
            let def = &p.defs[d];
            let hash = hashes[d].ok_or_else(|| internal("unhashed member".into()))?;
            let mut deps: Vec<Hash> = edges[d].iter().filter_map(|&w| hashes[w]).filter(|h| *h != hash).collect();
            deps.sort();
            deps.dedup();
            let sig_name = |n: &str| by_name.get(n).and_then(|&i| hashes[i]).map(|h| format!("#{h}")).unwrap_or_else(|| n.to_string());
            out[d] = Some(Canon {
                name: def.name().to_string(),
                kind: Kind::of(def),
                hash,
                canonical: forms[k].clone(),
                group,
                local_names: std::mem::take(&mut locals[k]),
                deps,
                refs: std::mem::take(&mut ref_names[d]),
                signature: signature_text(def, &sig_name),
                effects: match def {
                    Def::Fn(f) => sorted_effects(&f.effects),
                    _ => Vec::new(),
                },
            });
        }
    }
    let defs = out.into_iter().collect::<Option<Vec<_>>>().ok_or_else(|| internal("a definition is in no component".into()))?;
    Ok(Hashed { defs, groups })
}

/// What `n` refers to: a member of the group being encoded, or an already hashed definition.
/// None (an internal error, never a placeholder hash) if neither holds.
fn resolve(n: &str, by_name: &HashMap<&str, usize>, pos: &HashMap<usize, u32>, hashes: &[Option<Hash>]) -> Option<Ref> {
    let i = *by_name.get(n)?;
    match pos.get(&i) {
        Some(k) => Some(Ref::Group(*k)),
        None => Some(Ref::Hash((*hashes.get(i)?)?)),
    }
}

/// Canonical labelling of a group: positions that depend only on the members' structure.
struct Labeller<'a> {
    env: &'a Env<'a>,
    p: &'a Program,
    comp: &'a [usize],
    by_name: &'a HashMap<&'a str, usize>,
    hashes: &'a [Option<Hash>],
    best: Option<(Vec<u8>, Vec<usize>)>,
    symmetric: bool,
    nodes: usize,
    work: std::cell::Cell<usize>,
}

impl Labeller<'_> {
    /// Definition index -> label, for `encode`.
    fn by_def(&self, labels: &[u32]) -> HashMap<usize, u32> {
        self.comp.iter().enumerate().map(|(j, &d)| (d, labels[j])).collect()
    }

    /// Member `i` encoded with in-group references written as the labels (`lab`) of their targets.
    fn encode(&self, i: usize, lab: &HashMap<usize, u32>) -> Result<Vec<u8>, HashError> {
        self.work.set(self.work.get() + 1);
        if self.work.get() > MAX_LABEL_WORK {
            return Err(HashError::TooLarge(self.names()));
        }
        let mut refs = |n: &str| resolve(n, self.by_name, lab, self.hashes);
        encode_member(self.env, &self.p.defs[self.comp[i]], &mut refs).map(|x| x.0).map_err(HashError::Internal)
    }

    /// Refines labels until stable: the new label ranks (old label, encoding with old labels).
    fn refine(&self, mut labels: Vec<u32>) -> Result<Vec<u32>, HashError> {
        loop {
            let before = labels.iter().collect::<BTreeSet<_>>().len();
            let lab = self.by_def(&labels);
            let mut keys = Vec::with_capacity(labels.len());
            for i in 0..labels.len() {
                keys.push((labels[i], self.encode(i, &lab)?));
            }
            let mut distinct: Vec<&(u32, Vec<u8>)> = keys.iter().collect();
            distinct.sort();
            distinct.dedup();
            labels = keys.iter().map(|k| distinct.binary_search(&k).unwrap_or(0) as u32).collect();
            if distinct.len() == before {
                return Ok(labels);
            }
        }
    }

    /// Individualization–refinement over every choice; keeps the smallest group form.
    fn search(&mut self, labels: Vec<u32>) -> Result<(), HashError> {
        self.nodes += 1;
        if self.nodes > MAX_LABEL_NODES {
            return Err(HashError::TooLarge(self.names()));
        }
        let labels = self.refine(labels)?;
        let k = labels.len();
        let mut count: HashMap<u32, usize> = HashMap::new();
        for l in &labels {
            *count.entry(*l).or_default() += 1;
        }
        let Some(target) = labels.iter().copied().filter(|l| count[l] > 1).min() else {
            // A leaf: labels are the positions 0..k.
            let mut order = vec![0; k];
            for (i, l) in labels.iter().enumerate() {
                order[*l as usize] = self.comp[i];
            }
            let lab = self.by_def(&labels);
            let mut at = vec![0; k];
            for (i, l) in labels.iter().enumerate() {
                at[*l as usize] = i;
            }
            let mut form = Vec::new();
            for i in at {
                form.extend(self.encode(i, &lab)?);
            }
            match &self.best {
                Some((b, o)) if form == *b => self.symmetric |= *o != order,
                Some((b, _)) if form > *b => {}
                _ => {
                    self.best = Some((form, order));
                    self.symmetric = false;
                }
            }
            return Ok(());
        };
        for i in 0..k {
            if labels[i] != target {
                continue;
            }
            let next: Vec<u32> = labels
                .iter()
                .enumerate()
                .map(|(j, &l)| match l.cmp(&target) {
                    std::cmp::Ordering::Less => 2 * l,
                    std::cmp::Ordering::Equal if j == i => 2 * l,
                    _ => 2 * l + 1,
                })
                .collect();
            self.search(next)?;
        }
        Ok(())
    }

    fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.comp.iter().map(|&d| self.p.defs[d].name().to_string()).collect();
        names.sort();
        names
    }
}

/// The members of a component in canonical order (see the module documentation).
fn order_group(env: &Env, p: &Program, comp: &[usize], by_name: &HashMap<&str, usize>, hashes: &[Option<Hash>]) -> Result<Vec<usize>, HashError> {
    if comp.len() == 1 {
        return Ok(comp.to_vec());
    }
    let mut l = Labeller { env, p, comp, by_name, hashes, best: None, symmetric: false, nodes: 0, work: std::cell::Cell::new(0) };
    l.search(vec![0; comp.len()])?;
    if l.symmetric {
        return Err(HashError::Symmetric(l.names()));
    }
    l.best.map(|(_, order)| order).ok_or_else(|| HashError::Internal("no labelling".into()))
}

/// The type parameters of a generic function in canonical order: by first occurrence in the
/// parameter types, then the result type (then any others, in written order). Type arguments of a
/// function are always inferred, never written, so their order carries no meaning.
pub fn fn_tparams(f: &FnDef) -> Vec<String> {
    fn walk(t: &Type, tps: &[String], out: &mut Vec<String>) {
        match t {
            Type::Named(n) | Type::Param(n) => {
                if tps.contains(n) && !out.contains(n) {
                    out.push(n.clone());
                }
            }
            Type::List(x) | Type::Option(x) => walk(x, tps, out),
            Type::Result(a, b) | Type::Map(a, b) => {
                walk(a, tps, out);
                walk(b, tps, out);
            }
            Type::App(_, args) => args.iter().for_each(|a| walk(a, tps, out)),
            Type::Fn(ps, r, _) => {
                ps.iter().for_each(|p| walk(p, tps, out));
                walk(r, tps, out);
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    for p in &f.params {
        walk(&p.ty, &f.tparams, &mut out);
    }
    walk(&f.ret, &f.tparams, &mut out);
    for p in &f.tparams {
        if !out.contains(p) {
            out.push(p.clone());
        }
    }
    out
}

/// Replaces every `Named` type name through `f`.
fn map_named(t: &Type, f: &dyn Fn(&str) -> String) -> Type {
    match t {
        Type::Named(n) => Type::Named(f(n)),
        Type::App(n, args) => Type::App(f(n), args.iter().map(|a| map_named(a, f)).collect()),
        Type::Map(k, v) => Type::map(map_named(k, f), map_named(v, f)),
        Type::List(x) => Type::list(map_named(x, f)),
        Type::Option(x) => Type::option(map_named(x, f)),
        Type::Result(a, b) => Type::result(map_named(a, f), map_named(b, f)),
        Type::Fn(ps, r, e) => Type::Fn(ps.iter().map(|p| map_named(p, f)).collect(), Box::new(map_named(r, f)), sorted_effects(e)),
        t => t.clone(),
    }
}

fn sorted_effects(e: &[String]) -> Vec<String> {
    e.iter().collect::<BTreeSet<_>>().into_iter().cloned().collect()
}

/// A definition's signature with type names mapped through `f` (see [`Canon::signature`]).
pub fn signature_text(d: &Def, f: &dyn Fn(&str) -> String) -> String {
    match d {
        Def::Fn(fd) => {
            let keep = |n: &str| if fd.tparams.iter().any(|p| p == n) { n.to_string() } else { f(n) };
            let t = Type::Fn(fd.params.iter().map(|p| p.ty.clone()).collect(), Box::new(fd.ret.clone()), fd.effects.clone());
            let s = map_named(&t, &keep).to_string();
            if fd.tparams.is_empty() {
                s
            } else {
                // `fn[T](T) -> T`
                format!("fn[{}]{}", fn_tparams(fd).join(", "), &s[2..])
            }
        }
        Def::Type(t) if !t.params.is_empty() => {
            let keep = |n: &str| if t.params.iter().any(|p| p == n) { n.to_string() } else { f(n) };
            let mono = Def::Type(TypeDef { name: t.name.clone(), params: vec![], body: t.body.clone() });
            format!("[{}] {}", t.params.join(", "), signature_text(&mono, &keep))
        }
        Def::Type(t) => match &t.body {
            TypeBody::Record(fields) => {
                let fs: Vec<String> = fields.iter().map(|(n, ty)| format!("{n}: {}", map_named(ty, f))).collect();
                format!("{{ {} }}", fs.join(", "))
            }
            TypeBody::Enum(vs) if vs.is_empty() => "|".to_string(),
            TypeBody::Enum(vs) => {
                let vs: Vec<String> = vs
                    .iter()
                    .map(|v| {
                        if v.fields.is_empty() {
                            v.name.clone()
                        } else {
                            let fs: Vec<String> = v.fields.iter().map(|t| map_named(t, f).to_string()).collect();
                            format!("{}({})", v.name, fs.join(", "))
                        }
                    })
                    .collect();
                vs.join(" | ")
            }
        },
        Def::Test(_) => "Bool".to_string(),
    }
}

// ---------------------------------------------------------------------------------------------
// Decoding

/// Parses the header of a member's canonical form.
pub fn header(canonical: &[u8]) -> Result<Header, String> {
    let mut d = Dec::new(canonical);
    d.header()
}

/// The hashes of the other definitions (including other members of its group) a canonical form
/// refers to, sorted and without duplicates.
pub fn references_of(form: Form) -> Result<Vec<Hash>, String> {
    let locals = vec![String::from("x"); count_binders(form)?];
    let mut seen = Vec::new();
    decode_raw(form, "self", &locals, &mut |h| {
        seen.push(*h);
        String::new()
    })?;
    seen.sort();
    seen.dedup();
    Ok(seen)
}

/// Number of local-variable binders of the member (the length its local names need). Also
/// validates the whole form, including the group form's hash.
pub fn count_binders(form: Form) -> Result<usize, String> {
    let mut d = Dec::new(form.canonical);
    d.open(form)?;
    let mut found = None;
    for k in 0..d.size {
        let before = d.binders;
        d.member_def("x", &mut |_| String::new())?;
        if k == d.index {
            found = Some(d.binders - before);
        }
    }
    d.finish()?;
    found.ok_or_else(|| "member index out of range".into())
}

/// Decodes the definition `form` encodes, naming it `name`, its locals from `local_names` (binding
/// order), and every referenced definition through `name_of`. Local names that would capture a
/// referenced definition's name are renamed (`x` -> `x_1`), so the result means what was stored.
pub fn decode(form: Form, name: &str, local_names: &[String], name_of: &mut dyn FnMut(&Hash) -> String) -> Result<Def, String> {
    decode_with(&|n, l, f| decode_raw(form, n, l, f), name, local_names, name_of)
}

type RawDecoder<'a> = dyn Fn(&str, &[String], &mut dyn FnMut(&Hash) -> String) -> Result<(Def, bool), String> + 'a;

fn decode_with(raw: &RawDecoder, name: &str, local_names: &[String], name_of: &mut dyn FnMut(&Hash) -> String) -> Result<Def, String> {
    // First pass: which definition names does the text refer to?
    let mut cache: HashMap<Hash, String> = HashMap::new();
    let mut globals: HashSet<String> = HashSet::new();
    let (_, self_ref) = raw(name, local_names, &mut |h| {
        let n = cache.entry(*h).or_insert_with(|| name_of(h)).clone();
        globals.insert(n.clone());
        n
    })?;
    if self_ref {
        globals.insert(name.to_string());
    }
    let locals = avoid_capture(local_names, &globals);
    let (def, _) = raw(name, &locals, &mut |h| cache.get(h).cloned().unwrap_or_default())?;
    Ok(def)
}

/// Renames locals whose name equals a referenced definition's name (consistently, so shadowing
/// between locals is kept).
fn avoid_capture(local_names: &[String], globals: &HashSet<String>) -> Vec<String> {
    let mut taken: HashSet<String> = local_names.iter().chain(globals.iter()).cloned().collect();
    let mut renamed: HashMap<String, String> = HashMap::new();
    local_names
        .iter()
        .map(|n| {
            if !globals.contains(n) {
                return n.clone();
            }
            renamed
                .entry(n.clone())
                .or_insert_with(|| {
                    let mut k = 1;
                    while taken.contains(&format!("{n}_{k}")) {
                        k += 1;
                    }
                    let c = format!("{n}_{k}");
                    taken.insert(c.clone());
                    c
                })
                .clone()
        })
        .collect()
}

/// Decodes the member; also returns whether it refers to itself.
fn decode_raw(form: Form, name: &str, local_names: &[String], name_of: &mut dyn FnMut(&Hash) -> String) -> Result<(Def, bool), String> {
    let mut d = Dec::new(form.canonical);
    d.open(form)?;
    let mut result = None;
    for k in 0..d.size {
        if k == d.index {
            d.counting = false;
            d.names = local_names.to_vec();
            d.self_refs = false;
            let def = d.member_def(name, name_of)?;
            if d.next_name != local_names.len() {
                return Err(format!("{} local names stored, {} binders", local_names.len(), d.next_name));
            }
            result = Some((def, d.self_refs));
            d.counting = true;
        } else {
            d.member_def(name, &mut |_| String::new())?;
        }
    }
    d.finish()?;
    result.ok_or_else(|| "member index out of range".into())
}

/// What a member of a group says about itself: its number of local binders and the hashes of the
/// definitions it refers to (sorted, without duplicates, without itself).
#[derive(Debug, Clone, PartialEq)]
pub struct MemberInfo {
    pub binders: usize,
    pub refs: Vec<Hash>,
}

/// A group form scanned once: per-member info and where each member starts, so that members can be
/// decoded one by one without re-reading the whole group each time.
#[derive(Debug, Clone)]
pub struct GroupIndex {
    pub members: Vec<MemberInfo>,
    offsets: Vec<usize>,
    hashes: Vec<Hash>,
}

impl GroupIndex {
    /// Scans (and validates) the group form `group`, which must hash to `gh`.
    pub fn new(group: &[u8], gh: &Hash) -> Result<GroupIndex, String> {
        if Hash::of(group) != *gh {
            return Err("the group form does not match its hash".into());
        }
        let mut d = Dec::new(group);
        let size = d.prefix()?;
        let hashes: Vec<Hash> = (0..size).map(|j| Hash::of(&member_form(size, j, gh, d.version))).collect();
        d.size = size;
        d.others = hashes.iter().map(|h| Some(*h)).collect();
        d.collect = true;
        let (mut members, mut offsets) = (Vec::new(), Vec::new());
        for k in 0..size {
            d.index = k;
            offsets.push(d.pos);
            let before = d.binders;
            let mut seen = Vec::new();
            d.member_def("x", &mut |h| {
                seen.push(*h);
                String::new()
            })?;
            seen.sort();
            seen.dedup();
            members.push(MemberInfo { binders: d.binders - before, refs: seen });
        }
        d.finish()?;
        Ok(GroupIndex { members, offsets, hashes })
    }

    /// Like [`decode`] for member `index` of the group form `group` (the one this was built from).
    pub fn decode(&self, group: &[u8], index: usize, name: &str, local_names: &[String], name_of: &mut dyn FnMut(&Hash) -> String) -> Result<Def, String> {
        let offset = *self.offsets.get(index).ok_or("member index out of range")?;
        let raw = |name: &str, locals: &[String], f: &mut dyn FnMut(&Hash) -> String| -> Result<(Def, bool), String> {
            let mut d = Dec::new(group);
            d.prefix()?;
            d.pos = offset;
            d.size = self.hashes.len();
            d.index = index;
            d.others = self.hashes.iter().enumerate().map(|(j, h)| if j == index { None } else { Some(*h) }).collect();
            d.counting = false;
            d.names = locals.to_vec();
            let def = d.member_def(name, f)?;
            if d.next_name != locals.len() {
                return Err(format!("{} local names stored, {} binders", locals.len(), d.next_name));
            }
            Ok((def, d.self_refs))
        };
        decode_with(&raw, name, local_names, name_of)
    }
}

struct Dec<'b> {
    b: &'b [u8],
    pos: usize,
    depth: usize,
    ty_depth: usize,
    size: usize,
    index: usize,
    /// Hashes of the other group members (None for the decoded member itself).
    others: Vec<Option<Hash>>,
    /// Skipping (not building) a member: names are not needed, binders are only counted.
    counting: bool,
    /// With `counting`: still report references (a scan of a whole group).
    collect: bool,
    binders: usize,
    self_refs: bool,
    names: Vec<String>,
    next_name: usize,
    scope: Vec<String>,
    /// Type parameters of the member being decoded.
    tparams: Vec<String>,
    /// Indices of the type parameters read so far, in order (checked for a generic function's
    /// parameter and result types).
    tparam_uses: Vec<usize>,
    /// Format version of the form (1 or 2), and whether a version 2 construct was read.
    version: u8,
    saw_v2: bool,
}

type R<T> = Result<T, String>;

#[cold]
fn bad_tag(what: &str, t: u8) -> String {
    format!("bad {what} tag {t}")
}

impl<'b> Dec<'b> {
    fn new(b: &'b [u8]) -> Dec<'b> {
        Dec {
            b,
            pos: 0,
            depth: 0,
            ty_depth: 0,
            size: 0,
            index: 0,
            others: Vec::new(),
            counting: true,
            collect: false,
            binders: 0,
            self_refs: false,
            names: Vec::new(),
            next_name: 0,
            scope: Vec::new(),
            tparams: Vec::new(),
            tparam_uses: Vec::new(),
            version: 1,
            saw_v2: false,
        }
    }

    /// A version 2 construct: not allowed in a version 1 form.
    fn v2(&mut self) -> R<()> {
        if self.version < 2 {
            return Err("version 2 construct in a version 1 canonical form".into());
        }
        self.saw_v2 = true;
        Ok(())
    }

    /// After the last member: no trailing bytes, and a version 2 form must need version 2.
    fn finish(&self) -> R<()> {
        if self.pos != self.b.len() {
            return Err("trailing bytes after the last group member".into());
        }
        if self.version == 2 && !self.saw_v2 {
            return Err("version 2 canonical form without version 2 constructs (not canonical)".into());
        }
        Ok(())
    }

    fn prefix(&mut self) -> R<usize> {
        if self.b.len() < 5 || &self.b[..4] != MAGIC {
            return Err("not a QAL canonical form".into());
        }
        if !(1..=VERSION).contains(&self.b[4]) {
            return Err(format!("unsupported canonical form version {}", self.b[4]));
        }
        self.version = self.b[4];
        self.pos = 5;
        let size = self.index()?;
        if size == 0 {
            return Err("empty group".into());
        }
        Ok(size)
    }

    fn header(&mut self) -> R<Header> {
        let size = self.prefix()?;
        let index = self.index()?;
        if index >= size {
            return Err(format!("bad header (size {size}, index {index})"));
        }
        let group = if size == 1 {
            None
        } else {
            let bytes = self.b.get(self.pos..self.pos + 32).ok_or("truncated canonical form")?;
            self.pos += 32;
            if self.pos != self.b.len() {
                return Err("trailing bytes after the group hash".into());
            }
            Some(Hash(bytes.try_into().map_err(|_| "bad group hash")?))
        };
        self.size = size;
        self.index = index;
        Ok(Header { size, index, group })
    }

    /// Reads the header and positions the decoder at the first member (of the group form, for a
    /// member of a group).
    fn open(&mut self, form: Form<'b>) -> R<()> {
        let h = self.header()?;
        let Some(gh) = h.group else { return Ok(()) };
        let group = form.group.ok_or("the group form is missing")?;
        if Hash::of(group) != gh {
            return Err("the group form does not match its hash".into());
        }
        self.b = group;
        let version = self.version;
        if self.prefix()? != h.size || self.version != version {
            return Err("the group form has another size or version".into());
        }
        self.others = (0..h.size).map(|j| if j == h.index { None } else { Some(Hash::of(&member_form(h.size, j, &gh, version))) }).collect();
        Ok(())
    }

    fn byte(&mut self) -> R<u8> {
        let b = *self.b.get(self.pos).ok_or("truncated canonical form")?;
        self.pos += 1;
        Ok(b)
    }

    fn varint(&mut self) -> R<u64> {
        let mut n: u64 = 0;
        for shift in (0..64).step_by(7) {
            let b = self.byte()?;
            n |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                if b == 0 && shift > 0 {
                    return Err("non-minimal varint".into());
                }
                return Ok(n);
            }
        }
        Err("varint too long".into())
    }

    /// A count of items (each at least one byte) or bytes: bounded by the remaining input, so
    /// corrupt input cannot request huge allocations.
    fn count(&mut self) -> R<usize> {
        let n = self.varint()?;
        if n > (self.b.len() - self.pos) as u64 {
            return Err("count exceeds the canonical form's size".into());
        }
        Ok(n as usize)
    }

    /// An index (De Bruijn or group); the caller checks its range.
    fn index(&mut self) -> R<usize> {
        usize::try_from(self.varint()?).map_err(|_| "index out of range".to_string())
    }

    fn i64(&mut self) -> R<i64> {
        let bytes: [u8; 8] = self.b.get(self.pos..self.pos + 8).and_then(|b| b.try_into().ok()).ok_or("truncated canonical form")?;
        self.pos += 8;
        Ok(i64::from_le_bytes(bytes))
    }

    fn str(&mut self) -> R<String> {
        let n = self.count()?;
        let bytes = self.b.get(self.pos..self.pos + n).ok_or("truncated canonical form")?;
        self.pos += n;
        String::from_utf8(bytes.to_vec()).map_err(|_| "invalid UTF-8 in canonical form".into())
    }

    fn enter(&mut self) -> R<()> {
        self.depth += 1;
        if self.depth > MAX_DECODE_DEPTH {
            return Err("canonical form nested too deeply".into());
        }
        Ok(())
    }

    fn flag(&mut self) -> R<bool> {
        match self.byte()? {
            0 => Ok(false),
            1 => Ok(true),
            t => Err(bad_tag("flag", t)),
        }
    }

    fn reference(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<String> {
        match self.byte()? {
            R_HASH => {
                let bytes: [u8; 32] = self.b.get(self.pos..self.pos + 32).and_then(|b| b.try_into().ok()).ok_or("truncated canonical form")?;
                self.pos += 32;
                let h = Hash(bytes);
                Ok(if self.counting && !self.collect { String::new() } else { name_of(&h) })
            }
            R_GROUP => {
                let j = self.index()?;
                if j >= self.size {
                    return Err(format!("group reference {j} out of range"));
                }
                if j == self.index {
                    self.self_refs = true;
                    return Ok(own.to_string());
                }
                if self.counting && !self.collect {
                    return Ok(String::new());
                }
                let h = self.others[j].ok_or("bad group reference")?;
                Ok(name_of(&h))
            }
            t => Err(bad_tag("reference", t)),
        }
    }

    /// Effects must be sorted and without duplicates (the canonical set).
    fn effects(&mut self) -> R<Vec<String>> {
        let n = self.count()?;
        let mut out: Vec<String> = Vec::new();
        for _ in 0..n {
            let e = self.str()?;
            if out.last().is_some_and(|prev| *prev >= e) {
                return Err("effects are not a sorted set".into());
            }
            out.push(e);
        }
        Ok(out)
    }

    fn ty(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<Type> {
        self.ty_depth += 1;
        if self.ty_depth > MAX_DECODE_TYPE_DEPTH {
            return Err("canonical form nested too deeply".into());
        }
        let t = self.ty_inner(own, name_of)?;
        self.ty_depth -= 1;
        Ok(t)
    }

    fn ty_inner(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<Type> {
        Ok(match self.byte()? {
            T_INT => Type::Int,
            T_FLOAT => Type::Float,
            T_BOOL => Type::Bool,
            T_UNIT => Type::Unit,
            T_STR => Type::Str,
            T_LIST => Type::List(Box::new(self.ty(own, name_of)?)),
            T_OPTION => Type::Option(Box::new(self.ty(own, name_of)?)),
            T_RESULT => {
                let a = self.ty(own, name_of)?;
                Type::Result(Box::new(a), Box::new(self.ty(own, name_of)?))
            }
            T_NAMED => Type::Named(self.reference(own, name_of)?),
            T_FN => return self.ty_fn(own, name_of),
            T_APP => {
                self.v2()?;
                let n = self.reference(own, name_of)?;
                let k = self.count()?;
                if k == 0 {
                    return Err("type application without arguments".into());
                }
                let args = (0..k).map(|_| self.ty(own, name_of)).collect::<R<Vec<_>>>()?;
                Type::App(n, args)
            }
            T_PARAM => {
                self.v2()?;
                let i = self.index()?;
                self.tparam_uses.push(i);
                Type::Named(self.tparams.get(i).cloned().ok_or_else(|| format!("type parameter {i} out of range"))?)
            }
            T_MAP => {
                self.v2()?;
                let k = self.ty(own, name_of)?;
                Type::map(k, self.ty(own, name_of)?)
            }
            T_BYTES => {
                self.v2()?;
                Type::Bytes
            }
            t => return Err(bad_tag("type", t)),
        })
    }

    fn ty_fn(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<Type> {
        let n = self.count()?;
        let mut ps = Vec::with_capacity(n.min(16));
        for _ in 0..n {
            ps.push(self.ty(own, name_of)?);
        }
        let r = self.ty(own, name_of)?;
        Ok(Type::Fn(ps, Box::new(r), self.effects()?))
    }

    fn bind(&mut self) -> R<String> {
        self.binders += 1;
        let n = if self.counting {
            String::new()
        } else {
            let n = self.names.get(self.next_name).cloned().ok_or("fewer local names stored than binders")?;
            self.next_name += 1;
            n
        };
        self.scope.push(n.clone());
        Ok(n)
    }

    fn local(&mut self) -> R<String> {
        let i = self.index()?;
        if i >= self.scope.len() {
            return Err(format!("local index {i} out of scope"));
        }
        Ok(self.scope[self.scope.len() - 1 - i].clone())
    }

    fn exprs(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<Vec<Expr>> {
        let n = self.count()?;
        (0..n).map(|_| self.expr(own, name_of)).collect()
    }

    /// Variant owner (+ reference) and name; the name must belong to a built-in owner, and a
    /// declared enum's variant cannot have a built-in variant's name.
    fn variant_name(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<String> {
        let owner = self.byte()?;
        match owner {
            V_OPTION | V_RESULT => {}
            V_ENUM => {
                self.reference(own, name_of)?;
            }
            t => return Err(bad_tag("variant owner", t)),
        }
        let name = self.str()?;
        let ok = match name.as_str() {
            "None" | "Some" => owner == V_OPTION,
            "Ok" | "Err" => owner == V_RESULT,
            _ => owner == V_ENUM,
        };
        if !ok {
            return Err(format!("variant `{name}` does not belong to its owner"));
        }
        Ok(name)
    }

    fn expr(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<Expr> {
        self.enter()?;
        let e = self.expr_inner(own, name_of)?;
        self.depth -= 1;
        Ok(e)
    }

    // Compound cases live in their own functions so the frames on the recursion path stay small.
    fn expr_inner(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<Expr> {
        Ok(match self.byte()? {
            E_INT => Expr::Int(self.i64()?),
            E_FLOAT => Expr::Float(f64::from_bits(self.i64()? as u64)),
            E_BOOL => Expr::Bool(self.flag()?),
            E_STR => Expr::Str(self.str()?),
            E_UNIT => Expr::Unit,
            E_LOCAL => Expr::Var(self.local()?),
            E_FN_REF => Expr::Var(self.reference(own, name_of)?),
            E_VARIANT => return self.e_variant(own, name_of),
            E_CALL_LOCAL | E_CALL_FN | E_BUILTIN => return self.e_call(own, name_of),
            E_BIN => return self.e_bin(own, name_of),
            E_UN => return self.e_un(own, name_of),
            E_IF => return self.e_if(own, name_of),
            E_LET => return self.e_let(own, name_of),
            E_LIST => Expr::List(self.exprs(own, name_of)?),
            E_RECORD => return self.e_record(own, name_of),
            E_FIELD => return self.e_field(own, name_of),
            E_MATCH => return self.e_match(own, name_of),
            E_SEQ => Expr::Seq(self.exprs(own, name_of)?),
            E_LAMBDA => return self.e_lambda(own, name_of),
            t => return Err(bad_tag("expression", t)),
        })
    }

    fn e_lambda(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<Expr> {
        self.v2()?;
        let n = self.count()?;
        let mut anns = Vec::with_capacity(n.min(16));
        for _ in 0..n {
            anns.push(if self.flag()? { Some(self.ty(own, name_of)?) } else { None });
        }
        let mut params = Vec::with_capacity(n.min(16));
        for ann in anns {
            params.push((self.bind()?, ann));
        }
        let body = self.expr(own, name_of)?;
        for _ in 0..n {
            self.scope.pop();
        }
        Ok(Expr::Lambda(params, Box::new(body)))
    }

    /// Type parameters of a generic member (kind 3 or 4): at least one; their names are locals.
    fn tparams(&mut self) -> R<Vec<String>> {
        self.v2()?;
        let n = self.count()?;
        if n == 0 {
            return Err("generic definition without type parameters".into());
        }
        for _ in 0..n {
            let name = self.bind()?;
            self.scope.pop();
            self.tparams.push(name);
        }
        Ok(self.tparams.clone())
    }

    fn e_variant(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<Expr> {
        let name = self.variant_name(own, name_of)?;
        let args = self.exprs(own, name_of)?;
        Ok(if args.is_empty() { Expr::Var(name) } else { Expr::Call(name, args) })
    }

    /// The tag byte was just read.
    fn e_call(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<Expr> {
        let f = match self.b[self.pos - 1] {
            E_CALL_LOCAL => self.local()?,
            E_CALL_FN => self.reference(own, name_of)?,
            _ => {
                let f = self.str()?;
                if !Builtin::NAMES.contains(&f.as_str()) {
                    return Err(format!("unknown built-in `{f}`"));
                }
                f
            }
        };
        Ok(Expr::Call(f, self.exprs(own, name_of)?))
    }

    fn e_bin(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<Expr> {
        let t = self.byte()?;
        let op = *BINOPS.get(t as usize).ok_or_else(|| bad_tag("operator", t))?;
        let l = self.expr(own, name_of)?;
        Ok(Expr::Bin(op, Box::new(l), Box::new(self.expr(own, name_of)?)))
    }

    fn e_un(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<Expr> {
        let op = match self.byte()? {
            0 => UnOp::Neg,
            1 => UnOp::Not,
            t => return Err(bad_tag("unary operator", t)),
        };
        Ok(Expr::Un(op, Box::new(self.expr(own, name_of)?)))
    }

    fn e_if(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<Expr> {
        let c = self.expr(own, name_of)?;
        let t = self.expr(own, name_of)?;
        Ok(Expr::If(Box::new(c), Box::new(t), Box::new(self.expr(own, name_of)?)))
    }

    fn e_let(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<Expr> {
        let ann = if self.flag()? { Some(self.ty(own, name_of)?) } else { None };
        let v = self.expr(own, name_of)?;
        let name = self.bind()?;
        let body = self.expr(own, name_of)?;
        self.scope.pop();
        Ok(Expr::Let(name, ann, Box::new(v), Box::new(body)))
    }

    fn e_record(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<Expr> {
        let name = self.reference(own, name_of)?;
        let n = self.count()?;
        let mut fields = Vec::with_capacity(n.min(16));
        for _ in 0..n {
            let f = self.str()?;
            fields.push((f, self.expr(own, name_of)?));
        }
        Ok(Expr::Record(name, fields))
    }

    fn e_field(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<Expr> {
        let t = self.expr(own, name_of)?;
        Ok(Expr::Field(Box::new(t), self.str()?))
    }

    fn e_match(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<Expr> {
        let s = self.expr(own, name_of)?;
        let n = self.count()?;
        let mut arms = Vec::with_capacity(n.min(16));
        for _ in 0..n {
            let mark = self.scope.len();
            let pat = self.pat(own, name_of)?;
            let guard = if self.flag()? { Some(self.expr(own, name_of)?) } else { None };
            let body = self.expr(own, name_of)?;
            self.scope.truncate(mark);
            arms.push(Arm { pat, guard, body });
        }
        Ok(Expr::Match(Box::new(s), arms))
    }

    fn pat(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<Pat> {
        self.enter()?;
        let p = self.pat_inner(own, name_of)?;
        self.depth -= 1;
        Ok(p)
    }

    fn pat_inner(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<Pat> {
        Ok(match self.byte()? {
            P_WILD => Pat::Wild,
            P_BIND => Pat::Bind(self.bind()?),
            P_INT => Pat::Int(self.i64()?),
            P_STR => Pat::Str(self.str()?),
            P_BOOL => Pat::Bool(self.flag()?),
            P_UNIT => Pat::Unit,
            P_VARIANT => return self.pat_variant(own, name_of),
            P_RECORD => return self.pat_record(own, name_of),
            t => return Err(bad_tag("pattern", t)),
        })
    }

    fn pat_variant(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<Pat> {
        let name = self.variant_name(own, name_of)?;
        let n = self.count()?;
        let mut ps = Vec::with_capacity(n.min(16));
        for _ in 0..n {
            ps.push(self.pat(own, name_of)?);
        }
        Ok(Pat::Variant(name, ps))
    }

    fn pat_record(&mut self, own: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<Pat> {
        let name = self.reference(own, name_of)?;
        let n = self.count()?;
        let mut fs = Vec::with_capacity(n.min(16));
        for _ in 0..n {
            let f = self.str()?;
            fs.push((f, self.pat(own, name_of)?));
        }
        Ok(Pat::Record(name, fs))
    }

    fn member_def(&mut self, name: &str, name_of: &mut dyn FnMut(&Hash) -> String) -> R<Def> {
        self.scope.clear();
        self.tparams.clear();
        let mut kind = self.byte()?;
        let tparams = match kind {
            3 | 4 => {
                kind -= 3;
                self.tparams()?
            }
            _ => Vec::new(),
        };
        Ok(match kind {
            0 => Def::Type(TypeDef {
                name: name.to_string(),
                params: tparams,
                body: match self.byte()? {
                    0 => {
                        let n = self.count()?;
                        let mut fields = Vec::with_capacity(n.min(16));
                        for _ in 0..n {
                            let f = self.str()?;
                            fields.push((f, self.ty(name, name_of)?));
                        }
                        TypeBody::Record(fields)
                    }
                    1 => {
                        let n = self.count()?;
                        let mut vs = Vec::with_capacity(n.min(16));
                        for _ in 0..n {
                            let vname = self.str()?;
                            let k = self.count()?;
                            let fields = (0..k).map(|_| self.ty(name, name_of)).collect::<R<Vec<_>>>()?;
                            vs.push(Variant { name: vname, fields });
                        }
                        TypeBody::Enum(vs)
                    }
                    t => return Err(bad_tag("type body", t)),
                },
            }),
            1 => {
                let n = self.count()?;
                self.tparam_uses.clear();
                let tys = (0..n).map(|_| self.ty(name, name_of)).collect::<R<Vec<_>>>()?;
                let ret = self.ty(name, name_of)?;
                // A generic function's type parameters are numbered by first occurrence in the
                // parameter and result types (`fn_tparams`), and all occur: otherwise reordering
                // them would give a second encoding of the same definition.
                if !tparams.is_empty() {
                    let mut first: Vec<usize> = Vec::new();
                    for &i in &self.tparam_uses {
                        if !first.contains(&i) {
                            first.push(i);
                        }
                    }
                    if !first.iter().copied().eq(0..tparams.len()) {
                        return Err("type parameters of a generic function out of first-occurrence order (not canonical)".into());
                    }
                }
                let effects = self.effects()?;
                let mut params = Vec::with_capacity(n.min(16));
                for ty in tys {
                    params.push(Param { name: self.bind()?, ty });
                }
                let req = self.exprs(name, name_of)?;
                let m = self.count()?;
                self.scope.push(RESULT_VAR.to_string());
                let ens = (0..m).map(|_| self.expr(name, name_of)).collect::<R<Vec<_>>>()?;
                self.scope.pop();
                let body = self.expr(name, name_of)?;
                Def::Fn(FnDef { name: name.to_string(), tparams, params, ret, effects, req, ens, body })
            }
            2 if tparams.is_empty() => Def::Test(TestDef { name: name.to_string(), body: self.expr(name, name_of)? }),
            t => return Err(bad_tag("definition kind", t)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(s: &str) -> Box<Expr> {
        Box::new(Expr::Var(s.into()))
    }

    fn func(name: &str, param: &str, body: Expr) -> Def {
        Def::Fn(FnDef { name: name.into(), tparams: vec![], params: vec![Param { name: param.into(), ty: Type::Int }], ret: Type::Int, effects: vec![], req: vec![], ens: vec![], body })
    }

    fn by_name(p: &Program) -> HashMap<String, Canon> {
        hash_program(p).unwrap().defs.into_iter().map(|c| (c.name.clone(), c)).collect()
    }

    fn single(c: &Canon) -> Form<'_> {
        Form::new(&c.canonical, None)
    }

    #[test]
    fn unresolvable_references_fail_closed() {
        let pos = HashMap::new();
        let by_name: HashMap<&str, usize> = [("f", 0)].into_iter().collect();
        assert_eq!(resolve("g", &by_name, &pos, &[None]), None, "unknown name");
        assert_eq!(resolve("f", &by_name, &pos, &[None]), None, "not hashed yet and not in the group");
        assert_eq!(resolve("f", &by_name, &pos, &[]), None, "out of range");
        let h = Hash::of(b"x");
        assert_eq!(resolve("f", &by_name, &pos, &[Some(h)]), Some(Ref::Hash(h)));
    }

    #[test]
    fn operator_tags_index_the_operator_table() {
        for (i, op) in BINOPS.iter().enumerate() {
            assert_eq!(binop_tag(*op) as usize, i, "{op:?}");
        }
    }

    #[test]
    fn hex_roundtrip_and_display() {
        let h = Hash::of(b"abc");
        assert_eq!(h.to_hex(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(Hash::from_hex(&h.to_hex()), Some(h));
        assert_eq!(Hash::from_hex("xyz"), None);
        assert_eq!(h.short(), "ba7816bf8f01");
    }

    #[test]
    fn local_names_do_not_change_the_hash() {
        let a = Program { defs: vec![func("f", "x", Expr::Bin(BinOp::Add, var("x"), Box::new(Expr::Int(1))))] };
        let b = Program { defs: vec![func("g", "y", Expr::Bin(BinOp::Add, var("y"), Box::new(Expr::Int(1))))] };
        let (ca, cb) = (&by_name(&a)["f"], &by_name(&b)["g"]);
        assert_eq!(ca.hash, cb.hash);
        assert_eq!(ca.canonical, cb.canonical);
        assert_eq!((ca.local_names.clone(), cb.local_names.clone()), (vec!["x".to_string()], vec!["y".to_string()]));
        assert_eq!(Hash::of(&ca.canonical), ca.hash);
    }

    #[test]
    fn effects_are_a_sorted_set() {
        let mk = |effs: &[&str]| {
            let mut f = func("f", "x", Expr::Unit);
            if let Def::Fn(fd) = &mut f {
                fd.effects = effs.iter().map(|s| s.to_string()).collect();
                fd.ret = Type::Unit;
            }
            by_name(&Program { defs: vec![f] })["f"].clone()
        };
        let (a, b) = (mk(&["io", "fs"]), mk(&["fs", "io", "fs"]));
        assert_eq!(a.hash, b.hash);
        assert_eq!(a.effects, vec!["fs".to_string(), "io".to_string()]);
        assert_eq!(a.signature, "fn(Int) -> Unit !fs !io");
    }

    #[test]
    fn decode_restores_the_definition() {
        let body = Expr::Let("y".into(), None, var("x"), Box::new(Expr::Call("g".into(), vec![Expr::Var("y".into())])));
        let p = Program { defs: vec![func("f", "x", body.clone()), func("g", "z", Expr::Var("z".into()))] };
        let c = by_name(&p);
        let g_hash = c["g"].hash;
        let d = decode(single(&c["f"]), "f", &c["f"].local_names, &mut |h| if *h == g_hash { "g".into() } else { "?".into() }).unwrap();
        assert_eq!(d, p.defs[0]);
        assert_eq!(references_of(single(&c["f"])).unwrap(), vec![g_hash]);
        assert_eq!(c["f"].deps, vec![g_hash]);
        assert_eq!(count_binders(single(&c["f"])).unwrap(), 2);
    }

    #[test]
    fn locals_that_would_capture_a_renamed_reference_are_renamed() {
        // f(x) = helper(x); after renaming `helper` to `x` the parameter must not capture it.
        let p = Program { defs: vec![func("f", "x", Expr::Call("helper".into(), vec![Expr::Var("x".into())])), func("helper", "n", Expr::Var("n".into()))] };
        let c = by_name(&p);
        let d = decode(single(&c["f"]), "f", &c["f"].local_names, &mut |_| "x".into()).unwrap();
        assert_eq!(d, func("f", "x_1", Expr::Call("x".into(), vec![Expr::Var("x_1".into())])));
    }

    #[test]
    fn corrupt_input_is_an_error_not_a_panic() {
        let p = Program { defs: vec![func("f", "x", Expr::Var("x".into()))] };
        let c = &by_name(&p)["f"];
        let dec = |bytes: &[u8], locals: &[String]| decode(Form::new(bytes, None), "f", locals, &mut |_| String::new());
        for cut in 0..c.canonical.len() {
            assert!(dec(&c.canonical[..cut], &c.local_names).is_err(), "cut at {cut}");
        }
        let mut bad = c.canonical.clone();
        *bad.last_mut().unwrap() = 9; // local index out of scope
        assert!(dec(&bad, &c.local_names).is_err());
        assert!(dec(&c.canonical, &[]).is_err());
        let test_form = |body: &[u8]| {
            let mut v = single_form(&[2], 1);
            v.extend_from_slice(body);
            v
        };
        let deep = test_form(&std::iter::repeat_n([E_UN, 0], 100_000).flatten().collect::<Vec<_>>());
        assert!(dec(&deep, &[]).unwrap_err().contains("nested too deeply"));
        // A deep type (let annotation) and a deep pattern.
        let mut body = vec![E_LET, 1];
        body.extend(std::iter::repeat_n(T_LIST, 100_000));
        assert!(dec(&test_form(&body), &[]).unwrap_err().contains("nested too deeply"));
        let mut body = vec![E_MATCH, E_UNIT, 1];
        body.extend(std::iter::repeat_n([P_VARIANT, V_OPTION, 4, b'S', b'o', b'm', b'e', 1], 100_000).flatten());
        assert!(dec(&test_form(&body), &[]).unwrap_err().contains("nested too deeply"));
        // Non-canonical bytes with a valid shape: non-minimal varint, Bool 2, unsorted effects,
        // a variant that does not belong to its owner.
        assert!(dec(&test_form(&[E_LIST, 0x80, 0x00]), &[]).unwrap_err().contains("non-minimal"));
        assert!(dec(&test_form(&[E_BOOL, 2]), &[]).is_err());
        let mut f = single_form(&[1, 0, T_UNIT, 2, 2, b'i', b'o', 2, b'f', b's', 0, 0, E_UNIT], 1);
        assert!(dec(&f, &[]).unwrap_err().contains("sorted set"));
        f = test_form(&[E_VARIANT, V_OPTION, 2, b'O', b'k', 0]);
        assert!(dec(&f, &[]).unwrap_err().contains("does not belong"));
    }

    fn generic_id(tp: &str, x: &str) -> Def {
        let t = Type::Named(tp.into());
        Def::Fn(FnDef { name: "id".into(), tparams: vec![tp.into()], params: vec![Param { name: x.into(), ty: t.clone() }], ret: t, effects: vec![], req: vec![], ens: vec![], body: Expr::Var(x.into()) })
    }

    #[test]
    fn phase1_definitions_keep_their_version_1_bytes() {
        let p = Program { defs: vec![func("f", "x", Expr::Var("x".into()))] };
        let c = &by_name(&p)["f"];
        // magic, version 1, size 1, index 0, kind fn, 1 param Int, ret Int, no effects, no req, no ens, local 0
        assert_eq!(c.canonical, [b'Q', b'A', b'L', 0, 1, 1, 0, 1, 1, T_INT, T_INT, 0, 0, 0, E_LOCAL, 0]);
    }

    #[test]
    fn generic_hashes_ignore_type_parameter_names_and_order() {
        let a = &by_name(&Program { defs: vec![generic_id("T", "x")] })["id"];
        let b = &by_name(&Program { defs: vec![generic_id("U", "y")] })["id"];
        assert_eq!(a.hash, b.hash);
        assert_eq!(a.canonical[4], 2, "generic definitions need version 2");
        assert_eq!(a.local_names, vec!["T".to_string(), "x".to_string()]);
        assert_eq!(a.signature, "fn[T](T) -> T");
        let back = decode(single(a), "id", &a.local_names, &mut |_| String::new()).unwrap();
        assert_eq!(back, generic_id("T", "x"));
        // Function type parameters are inferred, never written: their order does not matter.
        let k = |tps: [&str; 2]| {
            let (a, b) = (Type::Named("A".into()), Type::Named("B".into()));
            Def::Fn(FnDef { name: "k".into(), tparams: tps.iter().map(|s| s.to_string()).collect(), params: vec![Param { name: "a".into(), ty: a.clone() }, Param { name: "b".into(), ty: b }], ret: a, effects: vec![], req: vec![], ens: vec![], body: Expr::Var("a".into()) })
        };
        let (k1, k2) = (&by_name(&Program { defs: vec![k(["A", "B"])] })["k"], &by_name(&Program { defs: vec![k(["B", "A"])] })["k"]);
        assert_eq!(k1.hash, k2.hash);
        assert_eq!(k1.signature, k2.signature);
    }

    #[test]
    fn generic_types_lambdas_maps_and_bytes_round_trip() {
        let pair = Def::Type(TypeDef { name: "Pair".into(), params: vec!["A".into(), "B".into()], body: TypeBody::Record(vec![("fst".into(), Type::Named("A".into())), ("snd".into(), Type::map(Type::Named("B".into()), Type::Bytes))]) });
        let lam = Expr::Lambda(vec![("y".into(), Some(Type::Int)), ("z".into(), None)], Box::new(Expr::Bin(BinOp::Add, var("y"), var("n"))));
        let user = Def::Fn(FnDef {
            name: "mk".into(),
            tparams: vec![],
            params: vec![Param { name: "n".into(), ty: Type::Int }, Param { name: "p".into(), ty: Type::App("Pair".into(), vec![Type::Int, Type::Str]) }],
            ret: Type::Fn(vec![Type::Int, Type::Bool], Box::new(Type::Int), vec![]),
            effects: vec![],
            req: vec![],
            ens: vec![],
            body: lam,
        });
        let p = Program { defs: vec![pair.clone(), user.clone()] };
        let cs = by_name(&p);
        assert_eq!(cs["Pair"].signature, "[A, B] { fst: A, snd: Map[B, Bytes] }");
        assert_eq!(cs["mk"].local_names, vec!["n", "p", "y", "z"]);
        let names = |h: &Hash| if *h == cs["Pair"].hash { "Pair".to_string() } else { "?".into() };
        assert_eq!(decode(single(&cs["Pair"]), "Pair", &cs["Pair"].local_names, &mut |h| names(h)).unwrap(), pair);
        assert_eq!(decode(single(&cs["mk"]), "mk", &cs["mk"].local_names, &mut |h| names(h)).unwrap(), user);
        // Renaming the lambda's parameters changes no hash.
        let mut renamed = p.clone();
        if let Def::Fn(f) = &mut renamed.defs[1] {
            f.body = Expr::Lambda(vec![("a".into(), Some(Type::Int)), ("b".into(), None)], Box::new(Expr::Bin(BinOp::Add, var("a"), var("n"))));
        }
        assert_eq!(by_name(&renamed)["mk"].hash, cs["mk"].hash);
    }

    #[test]
    fn versions_are_strict_both_ways() {
        let p = Program { defs: vec![func("f", "x", Expr::Var("x".into()))] };
        let c = &by_name(&p)["f"];
        let dec = |bytes: &[u8], locals: &[String]| decode(Form::new(bytes, None), "f", locals, &mut |_| String::new());
        let mut as_v2 = c.canonical.clone();
        as_v2[4] = 2;
        assert!(dec(&as_v2, &c.local_names).unwrap_err().contains("not canonical"));
        // `test t = let b: Bytes = ... ` style: a version 2 type inside a version 1 form.
        let mut v1 = single_form(&[2, E_LET, 1, T_BYTES, E_UNIT, E_UNIT], 1);
        assert!(dec(&v1, &["b".into()]).unwrap_err().contains("version 2 construct"));
        v1[4] = 2;
        assert!(dec(&v1, &["b".into()]).is_ok());
        v1[4] = 3;
        assert!(dec(&v1, &["b".into()]).unwrap_err().contains("unsupported"));
    }

    #[test]
    fn generic_function_type_parameters_must_be_in_first_occurrence_order() {
        // fn k[A, B](a: A, b: B) -> A = a: A is parameter 0, B parameter 1 (review finding M1).
        let ty = |n: &str| Type::Named(n.into());
        let k = Def::Fn(FnDef {
            name: "k".into(),
            tparams: vec!["A".into(), "B".into()],
            params: vec![Param { name: "a".into(), ty: ty("A") }, Param { name: "b".into(), ty: ty("B") }],
            ret: ty("A"),
            effects: vec![],
            req: vec![],
            ens: vec![],
            body: Expr::Var("a".into()),
        });
        let c = &by_name(&Program { defs: vec![k.clone()] })["k"];
        let dec = |bytes: &[u8]| decode(Form::new(bytes, None), "k", &c.local_names, &mut |_| String::new());
        assert_eq!(dec(&c.canonical).unwrap(), k);
        // Swapping the indices gives `fn k[A, B](a: B, b: A) -> B`: the same definition, other bytes.
        let at: Vec<usize> = (0..c.canonical.len() - 1).filter(|&i| c.canonical[i] == T_PARAM && c.canonical[i + 1] <= 1).collect();
        assert_eq!(at.len(), 3, "{:?}", c.canonical);
        let mut swapped = c.canonical.clone();
        for i in at {
            swapped[i + 1] ^= 1;
        }
        assert!(dec(&swapped).unwrap_err().contains("not canonical"));
    }

    #[test]
    fn long_dependency_chains_do_not_overflow() {
        let mut defs = vec![func("f0", "x", Expr::Var("x".into()))];
        for i in 1..5000 {
            defs.push(func(&format!("f{i}"), "x", Expr::Call(format!("f{}", i - 1), vec![Expr::Var("x".into())])));
        }
        assert_eq!(hash_program(&Program { defs }).unwrap().defs.len(), 5000);
    }

    /// `name(n) = callee(n)`, or the base case `if n == 0 then 0 else callee(n - 1)`.
    fn call(name: &str, callee: &str, base: bool) -> Def {
        let arg = if base { Expr::Bin(BinOp::Sub, var("n"), Box::new(Expr::Int(1))) } else { Expr::Var("n".into()) };
        let c = Expr::Call(callee.into(), vec![arg]);
        let body = if base { Expr::If(Box::new(Expr::Bin(BinOp::Eq, var("n"), Box::new(Expr::Int(0)))), Box::new(Expr::Int(0)), Box::new(c)) } else { c };
        func(name, "n", body)
    }

    #[test]
    fn group_hashes_do_not_depend_on_names_or_order() {
        // A 4-cycle with one base case, written with two different namings and orders.
        let a = Program { defs: vec![call("a", "b", true), call("b", "c", false), call("c", "d", false), call("d", "a", false)] };
        let b = Program { defs: vec![call("w", "z", false), call("z", "y", false), call("y", "x", true), call("x", "w", false)] };
        let (ca, cb) = (by_name(&a), by_name(&b));
        // a->b->c->d->a corresponds to y->x->w->z->y.
        for (n, m) in [("a", "y"), ("b", "x"), ("c", "w"), ("d", "z")] {
            assert_eq!(ca[n].hash, cb[m].hash, "{n} vs {m}");
        }
        let ha = hash_program(&a).unwrap();
        assert_eq!(ha.groups.len(), 1);
        let (gh, g) = ha.groups.iter().next().unwrap();
        assert_eq!(ca["a"].group, Some(*gh));
        assert_eq!(header(&ca["a"].canonical).unwrap().size, 4);
        // Each member decodes from its small form plus the shared group form.
        let names: HashMap<Hash, String> = ca.values().map(|c| (c.hash, c.name.clone())).collect();
        for c in ca.values() {
            let d = decode(Form::new(&c.canonical, Some(g)), &c.name, &c.local_names, &mut |h| names[h].clone()).unwrap();
            assert_eq!(&d, a.defs.iter().find(|x| x.name() == c.name).unwrap());
        }
        assert!(decode(Form::new(&ca["a"].canonical, None), "a", &ca["a"].local_names, &mut |_| String::new()).unwrap_err().contains("missing"));
        assert!(decode(Form::new(&ca["a"].canonical, Some(&g[..g.len() - 1])), "a", &ca["a"].local_names, &mut |_| String::new()).is_err());
    }

    #[test]
    fn groups_with_interchangeable_members_are_rejected() {
        let two = Program { defs: vec![call("f", "g", false), call("g", "f", false)] };
        assert_eq!(hash_program(&two).unwrap_err(), HashError::Symmetric(vec!["f".into(), "g".into()]));
        let three = Program { defs: vec![call("a", "b", false), call("b", "c", false), call("c", "a", false)] };
        assert!(matches!(hash_program(&three), Err(HashError::Symmetric(_))));
        // A large group whose members all look alike exceeds the search limit.
        let n = 1100;
        let ring = Program { defs: (0..n).map(|i| call(&format!("f{i}"), &format!("f{}", (i + 1) % n), false)).collect() };
        assert!(matches!(hash_program(&ring), Err(HashError::TooLarge(names)) if names.len() == n));
        // Breaking the symmetry with one base case makes it hashable.
        let ok = Program { defs: vec![call("a", "b", true), call("b", "c", false), call("c", "a", false)] };
        assert!(hash_program(&ok).is_ok());
    }
}
