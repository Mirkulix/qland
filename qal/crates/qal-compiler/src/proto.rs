//! The compiler protocol `QALQ`/`QALR` (self-hosting spec §4.2-§4.5): requests to and responses
//! from a compiler that is a pure function `qc_main(req: Bytes) -> Bytes`.
//!
//! Integers are minimal LEB128 (`uint`), strings `uint utf8`, byte strings `uint bytes`, lists
//! `uint elem*`, flags and options one byte 0/1; types, expressions, definitions and typed programs
//! use the encodings of [`qal_core::tform`].
//!
//! ```text
//! req      := "QALQ" ver:u8(1) op:u8 payload
//!   1 PARSE       text:str                                   -> program
//!   2 DEFINE      state text:str options                     -> prepared
//!   3 PRINT       state names:n str                          -> texts
//!   4 TYPED       state roots:n str options                  -> typed form (QALT bytes)
//!   5 BUILD       state entry options                        -> module
//!   6 CHECK_FILE  text:str opt(entry) options                -> 0 QALT | 1 module
//!   7 JOIN        continuation:bytes results:n bytes         (a fork's continuation, §4.3)
//!   8 SIGNATURES  state names:n str                          -> texts
//! state    := n sentry{n} m group{m}                         (entries sorted by name, groups by hash, strictly)
//! sentry   := name:str hash[32] kind:u8(0 type | 1 fn | 2 test) canonical:bytes local_names:n str refs:n str
//! group    := hash[32] form:bytes
//! entry    := 0 name:str | 1 names:n str | 2 (tests) | 3 (all)
//! options  := memory_max_pages:uint                         (fixed field list of version 1; the
//!             checker's limits, e.g. MAX_INSTANCES, are constants of the language, not options)
//!
//! res      := "QALR" ver:u8(1) status:u8 payload
//!   0 DONE        payload:bytes                              (op-specific, below)
//!   1 DIAGNOSTICS n diag
//!   2 FORK        continuation:bytes items:n bytes           (each item is a request for a fresh instance)
//!   3 INTERNAL    message:str                                (a broken invariant: E0951)
//! diag     := code:str kind:str def:str path:str message:str opt(expected:str) opt(found:str)
//!             hints:n str dependents:n str opt(counterexample:str JSON)
//!
//! program  := n def                                          (tform definitions)
//! texts    := n (name:str text:str)
//! module   := wasm:bytes imports:n str table_len:uint memory_max_pages:uint types:n typedef funcs:n func
//! func     := name:str export:str is_test:bool params:n (str type) ret:type effects:n str
//!             req:n expr ens:n expr sites:n (path:str label:str opt(contract:uint) values:n (str type))
//! prepared := typed:bytes n canon m group k sig_old           (spec §4.5, what qal-store reads)
//! canon    := name:str kind:u8 hash[32] canonical:bytes signature:str effects:n str local_names:n str
//!             refs:n str deps:n hash[32] opt(group:hash[32])
//! sig_old  := name:str kind:u8 signature:str
//! ```
//!
//! Decoding is strict (exact lengths, known tags and codes, no trailing bytes): a response that
//! breaks it is `bad_compiler_response` (E0952). [`serve`] is the reference implementation of the
//! compiler side on the Rust pipeline (5a: every operation but `DEFINE` and `JOIN`, which arrive
//! with the QAL compiler in 5b/5d); the self-hosted compiler must answer byte for byte the same.

#[cfg(feature = "reference")]
use crate::CompileOptions;
use crate::{CompiledModule, Entry, FuncMeta, Site};
use qal_core::ast::{Def, Program};
use qal_core::canon::{self, Hash, Kind};
use qal_core::diag::Diagnostic;
use qal_core::tform::{self, Reader, Writer};
use std::collections::HashMap;

pub const REQ_MAGIC: &[u8; 4] = b"QALQ";
pub const RES_MAGIC: &[u8; 4] = b"QALR";
pub const VERSION: u8 = 1;

/// One definition of the store state, as the store holds it.
#[derive(Debug, Clone, PartialEq)]
pub struct StateEntry {
    pub name: String,
    pub hash: Hash,
    pub kind: Kind,
    pub canonical: Vec<u8>,
    pub local_names: Vec<String>,
    /// The names its references were written against (see `qal_core::canon::Canon::refs`).
    pub refs: Vec<String>,
}

/// The store state a request is about: definitions by name, group forms by hash.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StoreState {
    pub entries: Vec<StateEntry>,
    pub groups: Vec<(Hash, Vec<u8>)>,
}

/// The fixed option fields of version 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    pub memory_max_pages: u64,
}

impl Default for Options {
    fn default() -> Options {
        Options { memory_max_pages: crate::MAX_PAGES }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Request {
    Parse { text: String },
    Define { state: StoreState, text: String, options: Options, new_names: Option<Vec<String>> },
    Print { state: StoreState, names: Vec<String> },
    Typed { state: StoreState, roots: Vec<String>, options: Options },
    Build { state: StoreState, entry: Entry, options: Options },
    CheckFile { text: String, entry: Option<Entry>, options: Options, new_names: Option<Vec<String>> },
    Join { continuation: Vec<u8>, results: Vec<Vec<u8>> },
    Signatures { state: StoreState, names: Vec<String> },
    /// `CHECK_DELTA` (incremental define): checks `text` with the functions in `declared` reduced
    /// to their signatures (`FnDef::declared`); the result is the typed program (`QALT`).
    CheckDelta { text: String, new_names: Vec<String>, declared: Vec<String> },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Response {
    Done(Vec<u8>),
    Diagnostics(Vec<Diagnostic>),
    Fork { continuation: Vec<u8>, items: Vec<Vec<u8>> },
    Internal(String),
}

/// What `CHECK_FILE` returns when the file checks.
#[derive(Debug, Clone)]
pub enum Checked {
    Typed(Vec<u8>),
    Module(CompiledModule),
}

/// The result of `DEFINE` for one definition (spec §4.5).
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedCanon {
    pub name: String,
    pub kind: Kind,
    pub hash: Hash,
    pub canonical: Vec<u8>,
    pub signature: String,
    pub effects: Vec<String>,
    pub local_names: Vec<String>,
    pub refs: Vec<String>,
    pub deps: Vec<Hash>,
    pub group: Option<Hash>,
}

/// The result of `DEFINE` (spec §4.5).
#[derive(Debug, Clone, PartialEq)]
pub struct Prepared {
    pub typed: Vec<u8>,
    pub canons: Vec<PreparedCanon>,
    pub groups: Vec<(Hash, Vec<u8>)>,
    /// The signatures of the input names as stored now (for the replacement rule).
    pub sig_old: Vec<(String, Kind, String)>,
}

type R<T> = Result<T, String>;

fn kind_tag(k: Kind) -> u8 {
    match k {
        Kind::Type => 0,
        Kind::Fn => 1,
        Kind::Test => 2,
    }
}

fn kind_of(t: u8) -> R<Kind> {
    match t {
        0 => Ok(Kind::Type),
        1 => Ok(Kind::Fn),
        2 => Ok(Kind::Test),
        t => Err(format!("unknown definition kind {t}")),
    }
}

trait Put {
    fn hash(&mut self, h: &Hash);
    fn names(&mut self, xs: &[String]);
    fn opt_str(&mut self, s: &Option<String>);
    fn state(&mut self, s: &StoreState);
    fn entry(&mut self, e: &Entry);
    fn options(&mut self, o: &Options);
}

impl Put for Writer {
    fn hash(&mut self, h: &Hash) {
        self.out.extend_from_slice(&h.0);
    }

    fn names(&mut self, xs: &[String]) {
        self.strs(xs);
    }

    fn opt_str(&mut self, s: &Option<String>) {
        match s {
            Some(s) => {
                self.byte(1);
                self.str(s);
            }
            None => self.byte(0),
        }
    }

    fn state(&mut self, s: &StoreState) {
        self.len(s.entries.len());
        for e in &s.entries {
            self.str(&e.name);
            self.hash(&e.hash);
            self.byte(kind_tag(e.kind));
            self.bytes(&e.canonical);
            self.strs(&e.local_names);
            self.strs(&e.refs);
        }
        self.len(s.groups.len());
        for (h, form) in &s.groups {
            self.hash(h);
            self.bytes(form);
        }
    }

    fn entry(&mut self, e: &Entry) {
        match e {
            Entry::Fn(n) => {
                self.byte(0);
                self.str(n);
            }
            Entry::Fns(ns) => {
                self.byte(1);
                self.strs(ns);
            }
            Entry::Tests => self.byte(2),
            Entry::All => self.byte(3),
        }
    }

    fn options(&mut self, o: &Options) {
        self.uint(o.memory_max_pages);
    }
}

trait Get {
    fn hash(&mut self) -> R<Hash>;
    fn opt_str(&mut self) -> R<Option<String>>;
    fn blob(&mut self) -> R<Vec<u8>>;
    fn state(&mut self) -> R<StoreState>;
    fn entry(&mut self) -> R<Entry>;
    fn options(&mut self) -> R<Options>;
}

impl Get for Reader<'_> {
    fn hash(&mut self) -> R<Hash> {
        let b = self.bytes(32)?;
        let mut h = [0u8; 32];
        h.copy_from_slice(b);
        Ok(Hash(h))
    }

    fn opt_str(&mut self) -> R<Option<String>> {
        Ok(if self.bool()? { Some(self.str()?) } else { None })
    }

    fn blob(&mut self) -> R<Vec<u8>> {
        let n = self.count()?;
        Ok(self.bytes(n)?.to_vec())
    }

    fn state(&mut self) -> R<StoreState> {
        let n = self.count()?;
        let mut entries: Vec<StateEntry> = Vec::new();
        for _ in 0..n {
            let name = self.str()?;
            if entries.last().is_some_and(|p| p.name >= name) {
                return Err(format!("store state entries are not sorted by name at `{name}`"));
            }
            let hash = self.hash()?;
            let kind = kind_of(self.byte()?)?;
            let canonical = self.blob()?;
            let local_names = self.strs()?;
            let refs = self.strs()?;
            entries.push(StateEntry { name, hash, kind, canonical, local_names, refs });
        }
        let m = self.count()?;
        let mut groups: Vec<(Hash, Vec<u8>)> = Vec::new();
        for _ in 0..m {
            let h = self.hash()?;
            if groups.last().is_some_and(|(p, _)| *p >= h) {
                return Err("store state groups are not sorted by hash".into());
            }
            groups.push((h, self.blob()?));
        }
        Ok(StoreState { entries, groups })
    }

    fn entry(&mut self) -> R<Entry> {
        Ok(match self.byte()? {
            0 => Entry::Fn(self.str()?),
            1 => Entry::Fns(self.strs()?),
            2 => Entry::Tests,
            3 => Entry::All,
            t => return Err(format!("unknown entry tag {t}")),
        })
    }

    fn options(&mut self) -> R<Options> {
        Ok(Options { memory_max_pages: self.uint()? })
    }
}

fn magic(r: &mut Reader, m: &[u8; 4], what: &str) -> R<()> {
    if r.bytes(4)? != m {
        return Err(format!("not a {what}"));
    }
    match r.byte()? {
        VERSION => Ok(()),
        v => Err(format!("unsupported {what} version {v}")),
    }
}

/// Runs `f` on a thread with a stack large enough for the deepest accepted nesting.
fn big_stack<T: Send>(f: impl FnOnce() -> R<T> + Send) -> R<T> {
    std::thread::scope(|scope| match std::thread::Builder::new().name("qal-proto".into()).stack_size(tform::DECODE_STACK).spawn_scoped(scope, f) {
        Ok(h) => h.join().unwrap_or_else(|_| Err("internal error while decoding".into())),
        Err(e) => Err(format!("cannot start the decoding thread: {e}")),
    })
}

impl Request {
    pub fn op(&self) -> u8 {
        match self {
            Request::Parse { .. } => 1,
            Request::Define { .. } => 2,
            Request::Print { .. } => 3,
            Request::Typed { .. } => 4,
            Request::Build { .. } => 5,
            Request::CheckFile { .. } => 6,
            Request::Join { .. } => 7,
            Request::Signatures { .. } => 8,
            Request::CheckDelta { .. } => 11,
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::default();
        w.out.extend_from_slice(REQ_MAGIC);
        w.byte(VERSION);
        w.byte(self.op());
        match self {
            Request::Parse { text } => w.str(text),
            Request::Define { state, text, options, new_names } => {
                w.state(state);
                w.str(text);
                w.options(options);
                match new_names {
                    Some(names) => {
                        w.byte(1);
                        w.strs(names);
                    }
                    None => w.byte(0),
                }
            }
            Request::Print { state, names } | Request::Signatures { state, names } => {
                w.state(state);
                w.names(names);
            }
            Request::Typed { state, roots, options } => {
                w.state(state);
                w.names(roots);
                w.options(options);
            }
            Request::Build { state, entry, options } => {
                w.state(state);
                w.entry(entry);
                w.options(options);
            }
            Request::CheckFile { text, entry, options, new_names } => {
                w.str(text);
                match entry {
                    Some(e) => {
                        w.byte(1);
                        w.entry(e);
                    }
                    None => w.byte(0),
                }
                w.options(options);
                match new_names {
                    Some(names) => {
                        w.byte(1);
                        w.strs(names);
                    }
                    None => w.byte(0),
                }
            }
            Request::Join { continuation, results } => {
                w.bytes(continuation);
                w.len(results.len());
                for r in results {
                    w.bytes(r);
                }
            }
            Request::CheckDelta { text, new_names, declared } => {
                w.str(text);
                w.strs(new_names);
                w.strs(declared);
            }
        }
        w.out
    }

    pub fn decode(b: &[u8]) -> R<Request> {
        let mut r = Reader::new(b);
        magic(&mut r, REQ_MAGIC, "QALQ request")?;
        let req = match r.byte()? {
            1 => Request::Parse { text: r.str()? },
            2 => {
                let state = r.state()?;
                let text = r.str()?;
                let options = r.options()?;
                let new_names = if !r.at_end() && r.bool()? { Some(r.strs()?) } else { None };
                Request::Define { state, text, options, new_names }
            }
            3 => Request::Print { state: r.state()?, names: r.strs()? },
            4 => Request::Typed { state: r.state()?, roots: r.strs()?, options: r.options()? },
            5 => Request::Build { state: r.state()?, entry: r.entry()?, options: r.options()? },
            6 => {
                let text = r.str()?;
                let entry = if r.bool()? { Some(r.entry()?) } else { None };
                let options = r.options()?;
                let new_names = if r.bool()? { Some(r.strs()?) } else { None };
                Request::CheckFile { text, entry, options, new_names }
            }
            7 => {
                let continuation = r.blob()?;
                let n = r.count()?;
                let results = (0..n).map(|_| r.blob()).collect::<R<Vec<_>>>()?;
                Request::Join { continuation, results }
            }
            8 => Request::Signatures { state: r.state()?, names: r.strs()? },
            11 => Request::CheckDelta { text: r.str()?, new_names: r.strs()?, declared: r.strs()? },
            op => return Err(format!("unknown operation {op}")),
        };
        r.finish()?;
        Ok(req)
    }
}

/// The envelope of a `JOIN` request (what the host sends after a fork), without decoding the rest.
pub fn join(continuation: &[u8], results: &[Vec<u8>]) -> Vec<u8> {
    let mut w = Writer::default();
    w.out.extend_from_slice(REQ_MAGIC);
    w.byte(VERSION);
    w.byte(7);
    w.bytes(continuation);
    w.len(results.len());
    for r in results {
        w.bytes(r);
    }
    w.out
}

fn put_diag(w: &mut Writer, d: &Diagnostic) {
    w.str(d.code);
    w.str(d.kind);
    w.str(&d.def);
    w.str(&d.path);
    w.str(&d.message);
    w.opt_str(&d.expected);
    w.opt_str(&d.found);
    w.strs(&d.hints);
    w.strs(&d.dependents);
    w.opt_str(&d.counterexample.as_ref().map(|v| v.to_string()));
}

fn get_diag(r: &mut Reader) -> R<Diagnostic> {
    let (code, kind) = (r.str()?, r.str()?);
    let (code, kind) = qal_core::diag::known(&code, &kind).ok_or_else(|| format!("unknown diagnostic ({code}, {kind})"))?;
    let mut d = Diagnostic::new(code, kind, &r.str()?, &r.str()?, r.str()?);
    d.expected = r.opt_str()?;
    d.found = r.opt_str()?;
    d.hints = r.strs()?;
    d.dependents = r.strs()?;
    d.counterexample = match r.opt_str()? {
        Some(s) => Some(serde_json_value(&s)?),
        None => None,
    };
    Ok(d)
}

fn serde_json_value(s: &str) -> R<serde_json::Value> {
    serde_json::from_str(s).map_err(|_| "a counterexample is not JSON".to_string())
}

impl Response {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::default();
        w.out.extend_from_slice(RES_MAGIC);
        w.byte(VERSION);
        match self {
            Response::Done(payload) => {
                w.byte(0);
                w.bytes(payload);
            }
            Response::Diagnostics(ds) => {
                w.byte(1);
                w.len(ds.len());
                for d in ds {
                    put_diag(&mut w, d);
                }
            }
            Response::Fork { continuation, items } => {
                w.byte(2);
                w.bytes(continuation);
                w.len(items.len());
                for i in items {
                    w.bytes(i);
                }
            }
            Response::Internal(m) => {
                w.byte(3);
                w.str(m);
            }
        }
        w.out
    }

    /// Strict decoding of a compiler's response; an `Err` means E0952 `bad_compiler_response`.
    pub fn decode(b: &[u8]) -> R<Response> {
        let mut r = Reader::new(b);
        magic(&mut r, RES_MAGIC, "QALR response")?;
        let res = match r.byte()? {
            0 => Response::Done(r.blob()?),
            1 => {
                let n = r.count()?;
                Response::Diagnostics((0..n).map(|_| get_diag(&mut r)).collect::<R<Vec<_>>>()?)
            }
            2 => {
                let continuation = r.blob()?;
                let n = r.count()?;
                let items = (0..n).map(|_| r.blob()).collect::<R<Vec<_>>>()?;
                Response::Fork { continuation, items }
            }
            3 => Response::Internal(r.str()?),
            s => return Err(format!("unknown response status {s}")),
        };
        r.finish()?;
        Ok(res)
    }
}

// ---------------------------------------------------------------------------------------------
// DONE payloads

pub fn encode_program(p: &Program) -> Vec<u8> {
    let mut w = Writer::default();
    w.len(p.defs.len());
    for d in &p.defs {
        w.def(d);
    }
    w.out
}

pub fn decode_program(b: &[u8]) -> R<Program> {
    big_stack(|| {
        let mut r = Reader::new(b);
        let n = r.count()?;
        let defs = (0..n).map(|_| r.def()).collect::<R<Vec<Def>>>()?;
        r.finish()?;
        Ok(Program { defs })
    })
}

pub fn encode_texts(xs: &[(String, String)]) -> Vec<u8> {
    let mut w = Writer::default();
    w.len(xs.len());
    for (n, t) in xs {
        w.str(n);
        w.str(t);
    }
    w.out
}

pub fn decode_texts(b: &[u8]) -> R<Vec<(String, String)>> {
    let mut r = Reader::new(b);
    let n = r.count()?;
    let mut out = Vec::new();
    for _ in 0..n {
        let name = r.str()?;
        out.push((name, r.str()?));
    }
    r.finish()?;
    Ok(out)
}

/// The static name of a host function the module imports.
fn host_name(s: &str) -> Option<&'static str> {
    use crate::host;
    [host::FAIL, host::FLOAT_TO_STR, host::STR_TO_FLOAT, host::UTF8_VALID, host::SHA256].into_iter().chain(host::EFFECT_FNS).find(|n| *n == s)
}

fn put_module(w: &mut Writer, m: &CompiledModule) {
    w.bytes(&m.wasm);
    w.len(m.imports.len());
    for i in &m.imports {
        w.str(i);
    }
    w.len(m.table_len);
    w.uint(m.memory_max_pages);
    w.len(m.types.len());
    for t in &m.types {
        w.typedef(t);
    }
    w.len(m.funcs.len());
    for f in &m.funcs {
        w.str(&f.name);
        w.str(&f.export);
        w.byte(f.is_test as u8);
        w.len(f.params.len());
        for (n, t) in &f.params {
            w.str(n);
            w.ty(t);
        }
        w.ty(&f.ret);
        w.strs(&f.effects);
        w.exprs(&f.req);
        w.exprs(&f.ens);
        w.len(f.sites.len());
        for s in &f.sites {
            w.str(&s.path);
            w.str(&s.label);
            match s.contract {
                Some(c) => {
                    w.byte(1);
                    w.len(c);
                }
                None => w.byte(0),
            }
            w.len(s.values.len());
            for (n, t) in &s.values {
                w.str(n);
                w.ty(t);
            }
        }
    }
}

fn get_module(r: &mut Reader) -> R<CompiledModule> {
    let wasm = r.blob()?;
    let n = r.count()?;
    let mut imports = Vec::new();
    for _ in 0..n {
        let s = r.str()?;
        imports.push(host_name(&s).ok_or_else(|| format!("unknown host function `{s}`"))?);
    }
    let table_len = r.usize()?;
    let memory_max_pages = r.uint()?;
    let n = r.count()?;
    let types = (0..n).map(|_| r.typedef()).collect::<R<Vec<_>>>()?;
    let n = r.count()?;
    let mut funcs = Vec::new();
    for _ in 0..n {
        let name = r.str()?;
        let export = r.str()?;
        let is_test = r.bool()?;
        let k = r.count()?;
        let mut params = Vec::new();
        for _ in 0..k {
            let n = r.str()?;
            params.push((n, r.ty()?));
        }
        let ret = r.ty()?;
        let effects = r.strs()?;
        let req = r.exprs()?;
        let ens = r.exprs()?;
        let k = r.count()?;
        let mut sites = Vec::new();
        for _ in 0..k {
            let path = r.str()?;
            let label = r.str()?;
            let contract = if r.bool()? { Some(r.usize()?) } else { None };
            let j = r.count()?;
            let mut values = Vec::new();
            for _ in 0..j {
                let n = r.str()?;
                values.push((n, r.ty()?));
            }
            sites.push(Site { path, label, contract, values });
        }
        funcs.push(FuncMeta { name, export, is_test, params, ret, effects, req, ens, sites });
    }
    Ok(CompiledModule { wasm, funcs, types, imports, table_len, memory_max_pages })
}

pub fn encode_module(m: &CompiledModule) -> Vec<u8> {
    let mut w = Writer::default();
    put_module(&mut w, m);
    w.out
}

pub fn decode_module(b: &[u8]) -> R<CompiledModule> {
    big_stack(|| {
        let mut r = Reader::new(b);
        let m = get_module(&mut r)?;
        r.finish()?;
        Ok(m)
    })
}

pub fn encode_checked(c: &Checked) -> Vec<u8> {
    let mut w = Writer::default();
    match c {
        Checked::Typed(t) => {
            w.byte(0);
            w.bytes(t);
        }
        Checked::Module(m) => {
            w.byte(1);
            put_module(&mut w, m);
        }
    }
    w.out
}

pub fn decode_checked(b: &[u8]) -> R<Checked> {
    big_stack(|| {
        let mut r = Reader::new(b);
        let c = match r.byte()? {
            0 => Checked::Typed(r.blob()?),
            1 => Checked::Module(get_module(&mut r)?),
            t => return Err(format!("unknown CHECK_FILE result {t}")),
        };
        r.finish()?;
        Ok(c)
    })
}

pub fn encode_prepared(p: &Prepared) -> Vec<u8> {
    let mut w = Writer::default();
    w.bytes(&p.typed);
    w.len(p.canons.len());
    for c in &p.canons {
        w.str(&c.name);
        w.byte(kind_tag(c.kind));
        w.hash(&c.hash);
        w.bytes(&c.canonical);
        w.str(&c.signature);
        w.strs(&c.effects);
        w.strs(&c.local_names);
        w.strs(&c.refs);
        w.len(c.deps.len());
        for d in &c.deps {
            w.hash(d);
        }
        match &c.group {
            Some(g) => {
                w.byte(1);
                w.hash(g);
            }
            None => w.byte(0),
        }
    }
    w.len(p.groups.len());
    for (h, f) in &p.groups {
        w.hash(h);
        w.bytes(f);
    }
    w.len(p.sig_old.len());
    for (n, k, s) in &p.sig_old {
        w.str(n);
        w.byte(kind_tag(*k));
        w.str(s);
    }
    w.out
}

pub fn decode_prepared(b: &[u8]) -> R<Prepared> {
    let mut r = Reader::new(b);
    let typed = r.blob()?;
    let n = r.count()?;
    let mut canons = Vec::new();
    for _ in 0..n {
        let name = r.str()?;
        let kind = kind_of(r.byte()?)?;
        let hash = r.hash()?;
        let canonical = r.blob()?;
        let signature = r.str()?;
        let effects = r.strs()?;
        let local_names = r.strs()?;
        let refs = r.strs()?;
        let k = r.count()?;
        let deps = (0..k).map(|_| r.hash()).collect::<R<Vec<_>>>()?;
        let group = if r.bool()? { Some(r.hash()?) } else { None };
        canons.push(PreparedCanon { name, kind, hash, canonical, signature, effects, local_names, refs, deps, group });
    }
    let m = r.count()?;
    let mut groups = Vec::new();
    for _ in 0..m {
        let h = r.hash()?;
        groups.push((h, r.blob()?));
    }
    let k = r.count()?;
    let mut sig_old = Vec::new();
    for _ in 0..k {
        let n = r.str()?;
        let kind = kind_of(r.byte()?)?;
        sig_old.push((n, kind, r.str()?));
    }
    r.finish()?;
    Ok(Prepared { typed, canons, groups, sig_old })
}

// ---------------------------------------------------------------------------------------------
// The Rust reference server

/// A store state for a checked program (what a store holding exactly `p` would send).
#[cfg(feature = "reference")]
pub fn state_of(p: &Program) -> Result<StoreState, String> {
    let h = canon::hash_program(p).map_err(|e| e.to_string())?;
    let mut entries: Vec<StateEntry> = h
        .defs
        .iter()
        .map(|c| StateEntry { name: c.name.clone(), hash: c.hash, kind: c.kind, canonical: c.canonical.clone(), local_names: c.local_names.clone(), refs: c.refs.clone() })
        .collect();
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(StoreState { entries, groups: h.groups.into_iter().collect() })
}

/// The program a store state holds, decoded with its names (definitions in name order).
pub fn program_of(state: &StoreState) -> Result<Program, String> {
    let mut by_hash: HashMap<Hash, Vec<&str>> = HashMap::new();
    for e in &state.entries {
        by_hash.entry(e.hash).or_default().push(&e.name);
    }
    let hash_of: HashMap<&str, Hash> = state.entries.iter().map(|e| (e.name.as_str(), e.hash)).collect();
    let groups: HashMap<Hash, &[u8]> = state.groups.iter().map(|(h, f)| (*h, f.as_slice())).collect();
    let mut defs = Vec::with_capacity(state.entries.len());
    for e in &state.entries {
        if Hash::of(&e.canonical) != e.hash {
            return Err(format!("`{}`: the canonical form does not match its hash", e.name));
        }
        let header = canon::header(&e.canonical)?;
        let group = match header.group {
            Some(g) => Some(*groups.get(&g).ok_or_else(|| format!("`{}`: its group form is missing", e.name))?),
            None => None,
        };
        // A reference names the definition the text was written against (an alias may share the hash).
        let mut name_of = |h: &Hash| -> String {
            e.refs.iter().find(|r| hash_of.get(r.as_str()) == Some(h)).cloned().or_else(|| by_hash.get(h).and_then(|ns| ns.first()).map(|n| n.to_string())).unwrap_or_else(|| h.short())
        };
        defs.push(canon::decode(canon::Form::new(&e.canonical, group), &e.name, &e.local_names, &mut name_of).map_err(|m| format!("`{}`: {m}", e.name))?);
    }
    Ok(Program { defs })
}

#[cfg(feature = "reference")]
fn compile_options(o: &Options) -> CompileOptions {
    CompileOptions { memory_max_pages: o.memory_max_pages }
}

#[cfg(feature = "reference")]
fn diagnostics(ds: Vec<Diagnostic>) -> Response {
    Response::Diagnostics(ds)
}

/// The Rust pipeline answering a `QALQ` request (the reference for the self-hosted compiler).
/// Never panics: a malformed request is an `INTERNAL` response.
#[cfg(feature = "reference")]
pub fn serve(req: &[u8]) -> Vec<u8> {
    let res = match big_stack(|| Request::decode(req)) {
        Ok(r) => serve_request(r),
        Err(e) => Response::Internal(format!("malformed request: {e}")),
    };
    res.encode()
}

#[cfg(feature = "reference")]
fn serve_request(req: Request) -> Response {
    let backend = &crate::Rust;
    use crate::Backend;
    let named = |state: &StoreState, names: &[String], f: &dyn Fn(&Def) -> String| -> Response {
        let p = match program_of(state) {
            Ok(p) => p,
            Err(e) => return Response::Internal(e),
        };
        let mut out = Vec::new();
        for n in names {
            match p.defs.iter().find(|d| d.name() == n) {
                Some(d) => out.push((n.clone(), f(d))),
                None => return Response::Internal(format!("`{n}` is not in the store state")),
            }
        }
        Response::Done(encode_texts(&out))
    };
    match req {
        Request::Parse { text } => match backend.parse(&text) {
            Ok(p) => Response::Done(encode_program(&p)),
            Err(d) => diagnostics(vec![d]),
        },
        Request::Print { state, names } => {
            if state.entries.is_empty() && names.is_empty() && state.groups.len() == 1 {
                match decode_program(&state.groups[0].1) {
                    Ok(p) => {
                        let out: Vec<(String, String)> = p
                            .defs
                            .iter()
                            .map(|d| (d.name().to_string(), backend.print(&Program { defs: vec![d.clone()] }).trim_end().to_string()))
                            .collect();
                        Response::Done(encode_texts(&out))
                    }
                    Err(e) => Response::Internal(e),
                }
            } else {
                named(&state, &names, &|d| backend.print(&Program { defs: vec![d.clone()] }).trim_end().to_string())
            }
        }
        Request::Signatures { state, names } => named(&state, &names, &|d| canon::signature_text(d, &|n| n.to_string())),
        Request::Typed { state, roots, options: _ } => match program_of(&state) {
            Ok(p) => match roots.iter().find(|r| !p.defs.iter().any(|d| d.name() == r.as_str())) {
                Some(r) => Response::Internal(format!("`{r}` is not in the store state")),
                None => match backend.check(&p, &|_| false) {
                    Ok(tp) => Response::Done(tform::encode(&tp)),
                    Err(ds) => diagnostics(ds),
                },
            },
            Err(e) => Response::Internal(e),
        },
        Request::Build { state, entry, options } => {
            if state.entries.is_empty() && state.groups.len() == 1 && state.groups[0].1.starts_with(tform::MAGIC) {
                match tform::decode(&state.groups[0].1) {
                    Ok(tp) => match backend.compile(&tp, &entry, &compile_options(&options)) {
                        Ok(m) => Response::Done(encode_module(&m)),
                        Err(e) => Response::Internal(e),
                    },
                    Err(e) => Response::Internal(e),
                }
            } else {
                match program_of(&state) {
                    Ok(p) => match backend.check(&p, &|_| false) {
                        Ok(tp) => match backend.compile(&tp, &entry, &compile_options(&options)) {
                            Ok(m) => Response::Done(encode_module(&m)),
                            Err(e) => Response::Internal(e),
                        },
                        Err(ds) => diagnostics(ds),
                    },
                    Err(e) => Response::Internal(e),
                }
            }
        }
        Request::CheckDelta { text, new_names, declared } => match backend.parse(&text) {
            Err(d) => diagnostics(vec![d]),
            Ok(p) => {
                let new: std::collections::HashSet<String> = new_names.into_iter().collect();
                let declared: std::collections::HashSet<String> = declared.into_iter().collect();
                match backend.check(&p.declare(&|n| declared.contains(n)), &|n| new.contains(n)) {
                    Ok(tp) => Response::Done(tform::encode(&tp)),
                    Err(ds) => diagnostics(ds),
                }
            }
        },
        Request::CheckFile { text, entry, options, new_names } => match backend.parse(&text) {
            Err(d) => diagnostics(vec![d]),
            Ok(p) => {
                let is_new: Box<dyn Fn(&str) -> bool> = match new_names {
                    None => Box::new(|_| true),
                    Some(names) => {
                        let set: std::collections::HashSet<String> = names.into_iter().collect();
                        Box::new(move |n: &str| set.contains(n))
                    }
                };
                match backend.check(&p, &*is_new) {
                    Err(ds) => diagnostics(ds),
                    Ok(tp) => match entry {
                        None => Response::Done(encode_checked(&Checked::Typed(tform::encode(&tp)))),
                        Some(e) => match backend.compile(&tp, &e, &compile_options(&options)) {
                            Ok(m) => Response::Done(encode_checked(&Checked::Module(m))),
                            Err(e) => Response::Internal(e),
                        },
                    },
                }
            }
        },
        Request::Define { state, text, options: _, new_names } => {
            let input = match backend.parse(&text) {
                Ok(p) => p,
                Err(d) => return diagnostics(vec![d]),
            };
            let stored = match program_of(&state) {
                Ok(p) => p,
                Err(e) => return Response::Internal(e),
            };
            let input_names: std::collections::HashSet<&str> = match new_names {
                Some(ref names) => names.iter().map(|s| s.as_str()).collect(),
                None => input.defs.iter().map(|d| d.name()).collect(),
            };
            let mut combined_defs: Vec<Def> = stored.defs.iter().filter(|d| !input_names.contains(d.name())).cloned().collect();
            combined_defs.extend(input.defs.iter().cloned());
            let combined = Program { defs: combined_defs };
            let typed_bytes = match backend.check(&combined, &|n| input_names.contains(n)) {
                Ok(tp) => tform::encode(&tp),
                Err(ds) => {
                    let real: Vec<Diagnostic> = ds.into_iter().filter(|d| !(d.is_builtin_name() && !input_names.contains(d.def.as_str()))).collect();
                    if !real.is_empty() {
                        return diagnostics(real);
                    }
                    Vec::new()
                }
            };
            let hashed = match backend.hash(&combined) {
                Ok(h) => h,
                Err(e) => return Response::Internal(e.to_string()),
            };
            let mut canons = Vec::new();
            for d in &input.defs {
                let name = d.name();
                if let Some(c) = hashed.defs.iter().find(|c| c.name == name) {
                    canons.push(PreparedCanon {
                        name: c.name.clone(),
                        kind: c.kind,
                        hash: c.hash,
                        canonical: c.canonical.clone(),
                        signature: c.signature.clone(),
                        effects: c.effects.clone(),
                        local_names: c.local_names.clone(),
                        refs: c.refs.clone(),
                        deps: c.deps.clone(),
                        group: c.group,
                    });
                }
            }
            let mut sig_old = Vec::new();
            for e in &state.entries {
                if input_names.contains(e.name.as_str()) {
                    if let Some(d) = stored.defs.iter().find(|d| d.name() == e.name) {
                        let sig = canon::signature_text(d, &|n| n.to_string());
                        sig_old.push((e.name.clone(), e.kind, sig));
                    }
                }
            }
            let groups: Vec<(Hash, Vec<u8>)> = hashed.groups.into_iter().collect();
            let prepared = Prepared {
                typed: typed_bytes,
                canons,
                groups,
                sig_old,
            };
            Response::Done(encode_prepared(&prepared))
        }
        Request::Join { .. } => Response::Internal("the Rust pipeline never forks, so it receives no JOIN".into()),
    }
}

#[cfg(all(test, feature = "reference"))]
mod tests {
    use super::*;

    const SRC: &str = "type Shape = Dot | Circle(Int)\nfn area(s: Shape) -> Int req true = match s { Dot => 0, Circle(r) => 3 * r * r }\n\
                       fn even(n: Int) -> Bool = if n == 0 then true else odd(n - 1)\nfn odd(n: Int) -> Bool = if n == 0 then false else even(n - 1)\n\
                       fn twice(x: Int) -> Int = area(Circle(x)) + area(Circle(x))\ntest t = twice(1) == 6";

    fn state() -> StoreState {
        state_of(&qal_compiler_parse(SRC)).unwrap()
    }

    fn qal_compiler_parse(src: &str) -> Program {
        crate::parse_program(src).unwrap()
    }

    fn done(res: &[u8]) -> Vec<u8> {
        match Response::decode(res).unwrap() {
            Response::Done(p) => p,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn requests_and_responses_round_trip() {
        let st = state();
        assert_eq!(st.groups.len(), 1, "even/odd are a group");
        let opts = Options { memory_max_pages: 16384 };
        let reqs = [
            Request::Parse { text: SRC.into() },
            Request::Define { state: st.clone(), text: "fn x() -> Int = 1".into(), options: opts, new_names: None },
            Request::Print { state: st.clone(), names: vec!["area".into(), "odd".into()] },
            Request::Typed { state: st.clone(), roots: vec!["t".into()], options: Options::default() },
            Request::Build { state: st.clone(), entry: Entry::Fns(vec!["twice".into(), "even".into()]), options: opts },
            Request::Build { state: StoreState::default(), entry: Entry::Tests, options: opts },
            Request::CheckFile { text: SRC.into(), entry: Some(Entry::All), options: opts, new_names: None },
            Request::CheckFile { text: "".into(), entry: None, options: opts, new_names: None },
            Request::Join { continuation: vec![1, 2, 3], results: vec![vec![], vec![0xff; 300]] },
            Request::Signatures { state: st, names: vec!["Shape".into()] },
        ];
        for r in &reqs {
            let b = r.encode();
            assert_eq!(&b[..4], b"QALQ");
            assert_eq!(b[5], r.op());
            assert_eq!(&Request::decode(&b).unwrap(), r);
            assert!(Request::decode(&[&b[..], &[0]].concat()).unwrap_err().contains("trailing"));
            for i in 0..b.len().min(400) {
                let _ = Request::decode(&b[..i]);
            }
        }
        let mut d = Diagnostic::type_mismatch("f", "body", "Int", "Str").hint("h");
        d.dependents = vec!["g".into()];
        d.counterexample = Some(serde_json::json!({"x": [1, "é"]}));
        let ress = [
            Response::Done(vec![9; 40]),
            Response::Diagnostics(vec![d.clone(), Diagnostic::parse(1, 2, "m"), Diagnostic::protocol_limit("x")]),
            Response::Fork { continuation: b"k".to_vec(), items: vec![b"QALQ".to_vec(); 3] },
            Response::Internal("bug".into()),
        ];
        for r in &ress {
            let b = r.encode();
            assert_eq!(&Response::decode(&b).unwrap(), r);
        }
        // Codes the host does not know, unknown statuses and bad magic are malformed responses.
        let mut bad = d.clone();
        bad.code = "E9999";
        let b = Response::Diagnostics(vec![bad]).encode();
        assert!(Response::decode(&b).unwrap_err().contains("unknown diagnostic"));
        assert!(Response::decode(b"QALR\x01\x07").unwrap_err().contains("status"));
        assert!(Response::decode(b"QALQ\x01\x00\x00").is_err() && Response::decode(b"QALR\x02\x00\x00").unwrap_err().contains("version"));
    }

    #[test]
    fn store_states_are_strict() {
        let mut st = state();
        st.entries.swap(0, 1);
        let b = Request::Print { state: st, names: vec![] }.encode();
        assert!(Request::decode(&b).unwrap_err().contains("sorted"));
        let mut st = state();
        st.entries[0].canonical.push(0);
        assert!(program_of(&st).unwrap_err().contains("hash"));
        let mut st = state();
        st.groups.clear();
        assert!(program_of(&st).unwrap_err().contains("group"));
    }

    #[test]
    fn the_rust_server_answers_every_5a_operation() {
        let st = state();
        let p = program_of(&st).unwrap();
        assert_eq!(p.defs.len(), 6);
        // PARSE
        let prog = decode_program(&done(&serve(&Request::Parse { text: SRC.into() }.encode()))).unwrap();
        assert_eq!(prog, qal_compiler_parse(SRC));
        match Response::decode(&serve(&Request::Parse { text: "fn (".into() }.encode())).unwrap() {
            Response::Diagnostics(ds) => assert_eq!(ds[0].kind, "parse"),
            other => panic!("{other:?}"),
        }
        // PRINT and SIGNATURES
        let texts = decode_texts(&done(&serve(&Request::Print { state: st.clone(), names: vec!["odd".into(), "Shape".into()] }.encode()))).unwrap();
        assert_eq!(texts, vec![("odd".to_string(), "fn odd(n: Int) -> Bool\n= if n == 0 then false else even(n - 1)".to_string()), ("Shape".to_string(), "type Shape = Dot | Circle(Int)".to_string())]);
        let sigs = decode_texts(&done(&serve(&Request::Signatures { state: st.clone(), names: vec!["twice".into()] }.encode()))).unwrap();
        assert_eq!(sigs[0].1, "fn(Int) -> Int");
        // TYPED: the QALT of the checked program
        let typed = done(&serve(&Request::Typed { state: st.clone(), roots: vec!["t".into()], options: Options::default() }.encode()));
        assert_eq!(typed, tform::encode(&crate::check(&p).unwrap()));
        // BUILD: the module, metadata included
        let opts = Options { memory_max_pages: 16384 };
        let m = decode_module(&done(&serve(&Request::Build { state: st.clone(), entry: Entry::All, options: opts }.encode()))).unwrap();
        let reference = crate::compile_with(&crate::check(&p).unwrap(), &Entry::All, &CompileOptions::compiler()).unwrap();
        assert_eq!(m.wasm, reference.wasm);
        assert_eq!((m.funcs, m.types, m.imports, m.table_len, m.memory_max_pages), (reference.funcs, reference.types, reference.imports, reference.table_len, 16384));
        // CHECK_FILE: diagnostics, the typed form, or the module
        match Response::decode(&serve(&Request::CheckFile { text: "fn f() -> Int = \"x\"".into(), entry: None, options: opts, new_names: None }.encode())).unwrap() {
            Response::Diagnostics(ds) => assert_eq!((ds[0].code, ds[0].kind), ("E0203", "type_mismatch")),
            other => panic!("{other:?}"),
        }
        assert!(matches!(decode_checked(&done(&serve(&Request::CheckFile { text: SRC.into(), entry: None, options: opts, new_names: None }.encode()))).unwrap(), Checked::Typed(_)));
        assert!(matches!(decode_checked(&done(&serve(&Request::CheckFile { text: SRC.into(), entry: Some(Entry::Tests), options: opts, new_names: None }.encode()))).unwrap(), Checked::Module(_)));
        // DEFINE (Phase 5b): prepared definitions
        let def_res = serve(&Request::Define { state: st.clone(), text: "fn extra(x: Int) -> Int = x".into(), options: opts, new_names: None }.encode());
        let prep = decode_prepared(&done(&def_res)).unwrap();
        assert_eq!(prep.canons.len(), 1);
        assert_eq!(prep.canons[0].name, "extra");

        // Broken requests and unhandled operations (JOIN) are INTERNAL, never a panic.
        for req in [b"nope".to_vec(), join(b"", &[]), Request::Print { state: st.clone(), names: vec!["zz".into()] }.encode()] {
            assert!(matches!(Response::decode(&serve(&req)).unwrap(), Response::Internal(_)));
        }
        let bad = Options { memory_max_pages: 0 };
        assert!(matches!(Response::decode(&serve(&Request::Build { state: st, entry: Entry::All, options: bad }.encode())).unwrap(), Response::Internal(m) if m.contains("memory_max_pages")));
    }

    #[test]
    fn prepared_results_round_trip() {
        let h = canon::hash_program(&qal_compiler_parse(SRC)).unwrap();
        let canons = h.defs.iter().map(|c| PreparedCanon { name: c.name.clone(), kind: c.kind, hash: c.hash, canonical: c.canonical.clone(), signature: c.signature.clone(), effects: c.effects.clone(), local_names: c.local_names.clone(), refs: c.refs.clone(), deps: c.deps.clone(), group: c.group }).collect();
        let p = Prepared { typed: vec![1, 2], canons, groups: h.groups.into_iter().collect(), sig_old: vec![("area".into(), Kind::Fn, "fn(#abc) -> Int".into())] };
        assert_eq!(decode_prepared(&encode_prepared(&p)).unwrap(), p);
    }
}
