//! Self-hosting test support (spec §10.1): the QAL test corpus and a program generator. Test-only.
//!
//! - [`corpus`]: (1) every `.qal` file of `qal/examples/` and `qal/crates/*/tests/fixtures/`, the
//!   20 benchmark solutions, the packages `std` and `todo` (with `std`, which it imports), and (2)
//!   the programs extracted from the Rust tests into `qal/selfhost/corpus/` by
//!   `qal/selfhost/tools/extract_corpus.py`. (`shop.qalpkg` joins when it exists; `qalc.qalpkg` is left out, see `corpus()`.)
//! - [`well_typed`]: a random program built from a typed grammar; it always checks.
//! - [`mutate`]: a corpus program with a few token-level mutations (delete, swap, replace a name,
//!   exchange a type); usually ill-typed or unparsable, which is the point.
//!
//! Both are pure functions of a seed ([`Rng`]), so every failure reproduces from the seed in the log.

use std::path::{Path, PathBuf};

/// One program of the corpus.
#[derive(Debug, Clone)]
pub struct Source {
    /// Where it comes from, relative to `qal/` (e.g. `examples/age.qal`, `selfhost/corpus/…`).
    pub name: String,
    pub text: String,
}

/// The `qal/` directory of the repository.
pub fn qal_root() -> PathBuf {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    p.canonicalize().unwrap_or(p)
}

fn files(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir).map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == ext)).collect()).unwrap_or_default();
    out.sort();
    out
}

fn read(root: &Path, p: &Path) -> Source {
    let name = p.strip_prefix(root).unwrap_or(p).to_string_lossy().replace('\\', "/");
    Source { name, text: std::fs::read_to_string(p).unwrap_or_default() }
}

/// The corpus files of part (1) of spec §10.1, in a fixed order.
pub fn file_corpus() -> Vec<Source> {
    let root = qal_root();
    let mut out: Vec<Source> = files(&root.join("examples"), "qal").iter().map(|p| read(&root, p)).collect();
    let mut crates: Vec<PathBuf> = std::fs::read_dir(root.join("crates")).map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect()).unwrap_or_default();
    crates.sort();
    let std = read(&root, &root.join("std/std.qalpkg"));
    let todo = read(&root, &root.join("examples/todo/todo.qalpkg"));
    for c in crates {
        for p in files(&c.join("tests/fixtures"), "qal") {
            let mut s = read(&root, &p);
            // `todo_*.qal` fixtures extend the todo package (which imports std).
            if p.file_name().is_some_and(|f| f.to_string_lossy().starts_with("todo_")) {
                s.text = format!("{}\n{}\n{}", std.text, todo.text, s.text);
            }
            out.push(s);
        }
    }
    for p in files(&root.join("../bench/solutions/qal"), "qal") {
        let name = format!("bench/solutions/qal/{}", p.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default());
        out.push(Source { name, text: std::fs::read_to_string(&p).unwrap_or_default() });
    }
    // Packages; one that imports std is checked together with it.
    out.push(std.clone());
    // `selfhost/qalc.qalpkg` is not part of the corpus yet: the store operations TYPED, DEFINE and
    // HASH run as one execution and exceed 1 GiB on the compiler's own package (CHECK_FILE is
    // split via FORK/JOIN; these are not). The bootstrap fixpoint covers qalc (full_fixpoint.rs).
    for pkg in ["examples/todo/todo.qalpkg", "examples/shop/shop.qalpkg"] {
        let p = root.join(pkg);
        if p.exists() {
            let s = read(&root, &p);
            out.push(Source { name: s.name, text: format!("{}\n{}", std.text, s.text) });
        }
    }
    out
}

/// The programs of part (2): extracted from the Rust tests.
pub fn extracted_corpus() -> Vec<Source> {
    let root = qal_root();
    files(&root.join("selfhost/corpus"), "qal").iter().map(|p| read(&root, p)).collect()
}

/// The whole corpus: parts (1) and (2).
pub fn corpus() -> Vec<Source> {
    let mut c = file_corpus();
    c.extend(extracted_corpus());
    c
}

/// splitmix64: small, fast and the same everywhere.
#[derive(Debug, Clone)]
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (n > 0).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }

    pub fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }

    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
}

/// The types the generator writes.
#[derive(Debug, Clone, PartialEq)]
enum Ty {
    Int,
    Bool,
    Str,
    Float,
    Bytes,
    ListInt,
    ListStr,
    OptInt,
    Rec,
    Enum,
    BoxInt,
    AccInt,
}

const TYPES: [Ty; 12] = [Ty::Int, Ty::Bool, Ty::Str, Ty::Float, Ty::Bytes, Ty::ListInt, Ty::ListStr, Ty::OptInt, Ty::Rec, Ty::Enum, Ty::BoxInt, Ty::AccInt];

impl Ty {
    fn text(&self) -> &'static str {
        match self {
            Ty::Int => "Int",
            Ty::Bool => "Bool",
            Ty::Str => "Str",
            Ty::Float => "Float",
            Ty::Bytes => "Bytes",
            Ty::ListInt => "List[Int]",
            Ty::ListStr => "List[Str]",
            Ty::OptInt => "Option[Int]",
            Ty::Rec => "Rec",
            Ty::Enum => "Shape",
            Ty::BoxInt => "Cell[Int]",
            Ty::AccInt => "Accum[Int]",
        }
    }
}

/// A function the generated program has defined so far.
struct Sig {
    name: String,
    params: Vec<Ty>,
    ret: Ty,
}

struct Gen<'r> {
    rng: &'r mut Rng,
    fns: Vec<Sig>,
    /// Locals in scope: (name, type).
    env: Vec<(String, Ty)>,
    next_local: usize,
}

const PRELUDE: &str = "type Rec = { a: Int, b: Str, c: List[Int] }\n\
                       type Shape = Dot | Circle(Int) | Label(Str, Bool)\n\
                       type Cell[T] = { v: T, n: Int }\n";

impl Gen<'_> {
    fn fresh(&mut self) -> String {
        self.next_local += 1;
        format!("x{}", self.next_local)
    }

    fn local_of(&mut self, t: &Ty) -> Option<String> {
        let names: Vec<String> = self.env.iter().filter(|(_, u)| u == t).map(|(n, _)| n.clone()).collect();
        (!names.is_empty()).then(|| self.rng.pick(&names).clone())
    }

    fn int_lit(&mut self) -> String {
        let n = self.rng.below(2000) as i64 - 1000;
        if n < 0 {
            format!("(-{})", -n)
        } else {
            n.to_string()
        }
    }

    fn str_lit(&mut self) -> String {
        let parts = ["", "a", "bc", "x,y", " sp ", "\\n", "\\t", "\\\"q\\\"", "\\\\", "\\u{e9}", "日本", "12"];
        let k = 1 + self.rng.below(3);
        let s: String = (0..k).map(|_| *self.rng.pick(&parts)).collect();
        format!("\"{s}\"")
    }

    fn leaf(&mut self, t: &Ty) -> String {
        if self.rng.chance(50) {
            if let Some(n) = self.local_of(t) {
                return n;
            }
        }
        match t {
            Ty::Int => self.int_lit(),
            Ty::Bool => if self.rng.chance(50) { "true" } else { "false" }.to_string(),
            Ty::Str => self.str_lit(),
            Ty::Float => format!("{}.{}", self.rng.below(100), self.rng.below(100)),
            Ty::Bytes => format!("to_bytes({})", self.str_lit()),
            Ty::ListInt => {
                let k = self.rng.below(4);
                let xs: Vec<String> = (0..k).map(|_| self.int_lit()).collect();
                format!("[{}]", xs.join(", "))
            }
            Ty::ListStr => {
                let k = self.rng.below(4);
                let xs: Vec<String> = (0..k).map(|_| self.str_lit()).collect();
                if xs.is_empty() {
                    "split(\"\", \",\")".into()
                } else {
                    format!("[{}]", xs.join(", "))
                }
            }
            Ty::OptInt => if self.rng.chance(50) { "None".to_string() } else { format!("Some({})", self.int_lit()) },
            Ty::Rec => format!("Rec {{ a: {}, b: {}, c: [] }}", self.int_lit(), self.str_lit()),
            Ty::Enum => match self.rng.below(3) {
                0 => "Dot".into(),
                1 => format!("Circle({})", self.int_lit()),
                _ => format!("Label({}, true)", self.str_lit()),
            },
            Ty::BoxInt => format!("Cell {{ v: {}, n: 0 }}", self.int_lit()),
            Ty::AccInt => "AccEnd".into(),
        }
    }

    /// An expression of type `t`, at most `d` levels deep.
    fn expr(&mut self, t: &Ty, d: usize) -> String {
        if d == 0 || self.rng.chance(25) {
            return self.leaf(t);
        }
        let d = d - 1;
        // Shapes that work for every type.
        match self.rng.below(10) {
            0 => {
                let c = self.expr(&Ty::Bool, d);
                let a = self.expr(t, d);
                let b = self.expr(t, d);
                return format!("(if {c} then {a} else {b})");
            }
            1 => {
                let vt = self.rng.pick(&TYPES).clone();
                let v = self.expr(&vt, d);
                let x = self.fresh();
                self.env.push((x.clone(), vt.clone()));
                let body = self.expr(t, d);
                self.env.pop();
                return format!("(let {x}: {} = {v} in {body})", vt.text());
            }
            2 => {
                let s = self.expr(&Ty::OptInt, d);
                let x = self.fresh();
                self.env.push((x.clone(), Ty::Int));
                let some = self.expr(t, d);
                self.env.pop();
                let none = self.expr(t, d);
                return format!("(match ({s}) {{ Some({x}) => {some}, None => {none} }})");
            }
            3 => {
                let s = self.expr(&Ty::Enum, d);
                let (x, y) = (self.fresh(), self.fresh());
                let dot = self.expr(t, d);
                self.env.push((x.clone(), Ty::Int));
                let circle = self.expr(t, d);
                self.env.pop();
                self.env.push((y.clone(), Ty::Str));
                let label = self.expr(t, d);
                self.env.pop();
                return format!("(match ({s}) {{ Dot => {dot}, Circle({x}) if {x} > 0 => {circle}, Circle(_) => {dot}, Label({y}, _) => {label} }})");
            }
            4 => {
                let cands: Vec<usize> = (0..self.fns.len()).filter(|&i| self.fns[i].ret == *t).collect();
                if !cands.is_empty() {
                    let i = *self.rng.pick(&cands);
                    let params = self.fns[i].params.clone();
                    let args: Vec<String> = params.iter().map(|p| self.expr(p, d)).collect();
                    return format!("{}({})", self.fns[i].name, args.join(", "));
                }
            }
            _ => {}
        }
        match t {
            Ty::Int => match self.rng.below(14) {
                0 | 1 => {
                    let op = *self.rng.pick(&["+", "-", "*", "/", "%"]);
                    let (a, b) = (self.expr(t, d), self.expr(t, d));
                    format!("({a} {op} {b})")
                }
                2 => format!("len({})", self.expr(&Ty::Str, d)),
                3 => format!("len({})", self.expr(&Ty::ListInt, d)),
                4 => format!("bytes_len({})", self.expr(&Ty::Bytes, d)),
                5 => {
                    let f = *self.rng.pick(&["bit_and", "bit_or", "bit_xor"]);
                    let (a, b) = (self.expr(t, d), self.expr(t, d));
                    format!("{f}({a}, {b})")
                }
                6 => {
                    let f = *self.rng.pick(&["shl", "shr", "ushr"]);
                    format!("{f}({}, {})", self.expr(t, d), self.rng.below(64))
                }
                7 => format!("list_at({}, {})", self.expr(&Ty::ListInt, d), self.expr(t, d)),
                8 => {
                    let ct = self.rng.pick(&[Ty::Int, Ty::Str, Ty::Rec, Ty::ListStr, Ty::Enum, Ty::OptInt, Ty::AccInt]).clone();
                    let x = self.fresh();
                    format!("(let {x}: {} = {} in compare({x}, {}))", ct.text(), self.expr(&ct, d), self.expr(&ct, d))
                }
                9 => {
                    let (acc, x) = (self.fresh(), self.fresh());
                    format!("fold({}, {}, fn({acc}: Int, {x}: Int) => {acc} + {x})", self.expr(&Ty::ListInt, d), self.expr(t, d))
                }
                10 => format!("({}).a", self.expr(&Ty::Rec, d)),
                11 => format!("({}).v", self.expr(&Ty::BoxInt, d)),
                12 => format!("bytes_at({}, {})", self.expr(&Ty::Bytes, d), self.expr(t, d)),
                _ => format!("float_bits({})", self.expr(&Ty::Float, d)),
            },
            Ty::Bool => match self.rng.below(7) {
                0 => {
                    let op = *self.rng.pick(&["<", "<=", ">", ">=", "==", "!="]);
                    format!("({} {op} {})", self.expr(&Ty::Int, d), self.expr(&Ty::Int, d))
                }
                1 => {
                    let ct = self.rng.pick(&TYPES).clone();
                    let x = self.fresh();
                    format!("(let {x}: {} = {} in ({x} == {}))", ct.text(), self.expr(&ct, d), self.expr(&ct, d))
                }
                2 => format!("!({})", self.expr(t, d)),
                3 => {
                    let op = *self.rng.pick(&["&&", "||"]);
                    format!("({} {op} {})", self.expr(t, d), self.expr(t, d))
                }
                4 => format!("contains({}, {})", self.expr(&Ty::Str, d), self.expr(&Ty::Str, d)),
                5 => format!("starts_with({}, {})", self.expr(&Ty::Str, d), self.expr(&Ty::Str, d)),
                _ => format!("({} < {})", self.expr(&Ty::Str, d), self.expr(&Ty::Str, d)),
            },
            Ty::Str => match self.rng.below(8) {
                0 | 1 => format!("({} ++ {})", self.expr(t, d), self.expr(t, d)),
                2 => format!("from_int({})", self.expr(&Ty::Int, d)),
                3 => format!("join({}, {})", self.expr(&Ty::ListStr, d), self.expr(t, d)),
                4 => format!("trim({})", self.expr(t, d)),
                5 => format!("({}).b", self.expr(&Ty::Rec, d)),
                6 => format!("from_float({})", self.expr(&Ty::Float, d)),
                _ => format!("to_upper({})", self.expr(t, d)),
            },
            Ty::Float => match self.rng.below(3) {
                0 => format!("int_to_float({})", self.expr(&Ty::Int, d)),
                1 => format!("float_of_bits({})", self.expr(&Ty::Int, d)),
                _ => format!("({} * {})", self.expr(t, d), self.expr(t, d)),
            },
            Ty::Bytes => match self.rng.below(4) {
                0 => format!("bytes_join([{}, {}])", self.expr(t, d), self.expr(t, d)),
                1 => format!("sha256({})", self.expr(t, d)),
                2 => format!("({} ++ {})", self.expr(t, d), self.expr(t, d)),
                _ => format!("to_bytes({})", self.expr(&Ty::Str, d)),
            },
            Ty::ListInt => match self.rng.below(7) {
                0 => format!("range({}, {})", self.expr(&Ty::Int, d), self.expr(&Ty::Int, d)),
                1 => {
                    let x = self.fresh();
                    self.env.push((x.clone(), Ty::Int));
                    let body = self.expr(&Ty::Int, d);
                    self.env.pop();
                    format!("map({}, fn({x}: Int) => {body})", self.expr(t, d))
                }
                2 => {
                    let x = self.fresh();
                    self.env.push((x.clone(), Ty::Int));
                    let body = self.expr(&Ty::Bool, d);
                    self.env.pop();
                    format!("filter({}, fn({x}: Int) => {body})", self.expr(t, d))
                }
                3 => format!("push({}, {})", self.expr(t, d), self.expr(&Ty::Int, d)),
                4 => format!("acc_list({})", self.expr(&Ty::AccInt, d)),
                5 => format!("bytes_to_list({})", self.expr(&Ty::Bytes, d)),
                _ => format!("({}).c", self.expr(&Ty::Rec, d)),
            },
            Ty::ListStr => match self.rng.below(3) {
                0 => format!("split({}, \",\")", self.expr(&Ty::Str, d)),
                1 => format!("chars({})", self.expr(&Ty::Str, d)),
                _ => {
                    let x = self.fresh();
                    format!("map({}, fn({x}: Int) => from_int({x}))", self.expr(&Ty::ListInt, d))
                }
            },
            Ty::OptInt => match self.rng.below(3) {
                0 => format!("to_int({})", self.expr(&Ty::Str, d)),
                1 => format!("get({}, {})", self.expr(&Ty::ListInt, d), self.expr(&Ty::Int, d)),
                _ => format!("Some({})", self.expr(&Ty::Int, d)),
            },
            Ty::Rec => format!("Rec {{ c: {}, a: {}, b: {} }}", self.expr(&Ty::ListInt, d), self.expr(&Ty::Int, d), self.expr(&Ty::Str, d)),
            Ty::Enum => match self.rng.below(2) {
                0 => format!("Circle({})", self.expr(&Ty::Int, d)),
                _ => format!("Label({}, {})", self.expr(&Ty::Str, d), self.expr(&Ty::Bool, d)),
            },
            Ty::BoxInt => format!("Cell {{ v: {}, n: {} }}", self.expr(&Ty::Int, d), self.expr(&Ty::Int, d)),
            Ty::AccInt => format!("AccPush({}, {})", self.expr(t, d), self.expr(&Ty::Int, d)),
        }
    }
}

/// A random well-typed program: three declared types, 1–8 functions (each calls only earlier
/// ones), and 0–3 tests.
pub fn well_typed(rng: &mut Rng) -> String {
    let mut g = Gen { rng, fns: Vec::new(), env: Vec::new(), next_local: 0 };
    let mut out = String::from(PRELUDE);
    let n = 1 + g.rng.below(8);
    for i in 0..n {
        let k = g.rng.below(4);
        let params: Vec<Ty> = (0..k).map(|_| g.rng.pick(&TYPES).clone()).collect();
        let ret = g.rng.pick(&TYPES).clone();
        g.env = params.iter().enumerate().map(|(j, t)| (format!("p{j}"), t.clone())).collect();
        let depth = 1 + g.rng.below(5);
        let body = g.expr(&ret, depth);
        let ps: Vec<String> = params.iter().enumerate().map(|(j, t)| format!("p{j}: {}", t.text())).collect();
        out.push_str(&format!("fn f{i}({}) -> {} = {body}\n", ps.join(", "), ret.text()));
        g.fns.push(Sig { name: format!("f{i}"), params, ret });
    }
    g.env.clear();
    for i in 0..g.rng.below(4) {
        let body = g.expr(&Ty::Bool, 3);
        out.push_str(&format!("test t{i} = {body}\n"));
    }
    out
}

/// Tokens of QAL text, roughly: names, numbers, strings, and single punctuation characters.
fn tokens(src: &str) -> Vec<String> {
    let cs: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        let start = i;
        if c.is_whitespace() {
            while i < cs.len() && cs[i].is_whitespace() {
                i += 1;
            }
        } else if c == '"' {
            i += 1;
            while i < cs.len() && cs[i] != '"' {
                i += if cs[i] == '\\' { 2 } else { 1 };
            }
            i = (i + 1).min(cs.len());
        } else if c.is_alphanumeric() || c == '_' || c == '.' {
            while i < cs.len() && (cs[i].is_alphanumeric() || cs[i] == '_' || cs[i] == '.') {
                i += 1;
            }
        } else {
            i += 1;
        }
        out.push(cs[start..i.min(cs.len())].iter().collect());
    }
    out
}

/// `src` with one to three random mutations: delete a token, swap two neighbouring tokens,
/// replace a name by another name of the program, or exchange a type name.
pub fn mutate(src: &str, rng: &mut Rng) -> String {
    let mut ts = tokens(src);
    if ts.is_empty() {
        return src.to_string();
    }
    let names: Vec<String> = ts.iter().filter(|t| t.chars().next().is_some_and(|c| c.is_alphabetic())).cloned().collect();
    const TYPE_NAMES: [&str; 8] = ["Int", "Str", "Bool", "Float", "Bytes", "List", "Option", "Unit"];
    for _ in 0..1 + rng.below(3) {
        let i = rng.below(ts.len());
        match rng.below(4) {
            0 => {
                ts.remove(i);
                if ts.is_empty() {
                    break;
                }
            }
            1 if i + 1 < ts.len() => ts.swap(i, i + 1),
            2 if !names.is_empty() && ts[i].chars().next().is_some_and(|c| c.is_alphabetic()) => ts[i] = rng.pick(&names).clone(),
            _ => {
                if let Some(j) = (0..ts.len()).map(|k| (i + k) % ts.len()).find(|&k| TYPE_NAMES.contains(&ts[k].as_str())) {
                    ts[j] = rng.pick(&TYPE_NAMES).to_string();
                }
            }
        }
    }
    ts.concat()
}

/// The seed of a generator run: `QAL_GEN_SEED` if set, else the clock (printed by the caller).
pub fn seed() -> u64 {
    std::env::var("QAL_GEN_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1))
}

/// Number of generated programs per run: `QAL_GEN_COUNT`, else 10 000 (spec §10.1).
pub fn count() -> usize {
    std::env::var("QAL_GEN_COUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(10_000)
}
