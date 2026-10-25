//! A consistent, verified view of the store's contents (read inside one transaction), and decoding
//! of stored definitions back into source ASTs with the current names.

use crate::space::Space;
use crate::sql;
use qal_core::ast::{Def, Program};
use qal_core::canon::{self, Hash, Kind};
use qal_core::diag::Diagnostic;
use rusqlite::{params, Connection, OptionalExtension};
use std::cell::RefCell;
use std::rc::Rc;
use std::collections::{BTreeMap, HashMap};

/// A group form, scanned once; shared by all of its members.
#[derive(Debug)]
pub struct Group {
    form: Vec<u8>,
    index: canon::GroupIndex,
}

/// One row of `defs`, verified: `sha256(canonical) == hash`, and the canonical form decodes.
#[derive(Debug, Clone)]
pub struct Row {
    pub kind: Kind,
    pub canonical: Vec<u8>,
    /// For a member of a group: the group and the member's index in it.
    group: Option<(Rc<Group>, usize)>,
    pub local_names: Vec<String>,
    /// The hashes of the definitions it refers to (sorted, without duplicates).
    pub refs: Vec<Hash>,
}

impl Row {
    pub(crate) fn decode(&self, name: &str, locals: &[String], name_of: &mut dyn FnMut(&Hash) -> String) -> Result<Def, String> {
        match &self.group {
            None => canon::decode(canon::Form::new(&self.canonical, None), name, locals, name_of),
            Some((g, i)) => g.index.decode(&g.form, *i, name, locals, name_of),
        }
    }
}

/// What the store knows about one name.
#[derive(Debug, Clone, PartialEq)]
pub struct NameEntry {
    pub hash: Hash,
    /// Local-variable names of this name's latest text.
    pub locals: Vec<String>,
    /// The names its references were written against, one per referenced hash (sorted).
    pub refs: Vec<String>,
}

/// [`Snapshot::fingerprint`] of the names, hashes and local names `entries` (in any order).
pub fn fingerprint<'a>(entries: impl Iterator<Item = (&'a str, &'a Hash, &'a [String])>) -> Hash {
    let mut entries: Vec<_> = entries.collect();
    entries.sort_by_key(|e| e.0);
    let mut bytes = Vec::new();
    for (name, hash, locals) in entries {
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&hash.0);
        for l in locals {
            bytes.extend_from_slice(l.as_bytes());
            bytes.push(1);
        }
        bytes.push(0);
    }
    Hash::of(&bytes)
}

pub struct Snapshot {
    pub names: BTreeMap<String, NameEntry>,
    /// hash -> its names, sorted (the first one is the primary name used for hashes shown without
    /// a reference)
    pub by_hash: HashMap<Hash, Vec<String>>,
    /// name -> the names whose references were written against it, sorted
    pub users: HashMap<String, Vec<String>>,
    rows: RefCell<HashMap<Hash, Row>>,
    groups: RefCell<HashMap<Hash, Rc<Group>>>,
}

fn corrupt(what: impl std::fmt::Display) -> Vec<Diagnostic> {
    vec![Diagnostic::store_error(format!("the store is corrupt: {what}"))]
}

fn json_list(s: &str, what: &str) -> Result<Vec<String>, Vec<Diagnostic>> {
    serde_json::from_str(s).map_err(|e| corrupt(format!("{what} is not a JSON list of strings ({e})")))
}

fn parse_hash(s: &str) -> Result<Hash, Vec<Diagnostic>> {
    Hash::from_hex(s).ok_or_else(|| corrupt(format!("`{s}` is not a hash")))
}

fn index(names: &BTreeMap<String, NameEntry>) -> (HashMap<Hash, Vec<String>>, HashMap<String, Vec<String>>) {
    let mut by_hash: HashMap<Hash, Vec<String>> = HashMap::new();
    let mut users: HashMap<String, Vec<String>> = HashMap::new();
    for (n, e) in names {
        by_hash.entry(e.hash).or_default().push(n.clone());
        for r in &e.refs {
            users.entry(r.clone()).or_default().push(n.clone());
        }
    }
    (by_hash, users)
}

impl Snapshot {
    /// Reads all names of `space` and the definitions they point to.
    /// A hash of the codebase: every name with its definition's hash and local names (the key of
    /// the checked program a define keeps, see `incremental`).
    pub fn fingerprint(&self) -> Hash {
        fingerprint(self.names.iter().map(|(n, e)| (n.as_str(), &e.hash, e.locals.as_slice())))
    }

    /// The names that refer to one of `names`, directly or through others, and `names` themselves.
    pub fn with_dependents(&self, names: &std::collections::HashSet<&str>) -> std::collections::HashSet<String> {
        let mut out: std::collections::HashSet<String> = names.iter().map(|n| n.to_string()).collect();
        let mut todo: Vec<String> = out.iter().cloned().collect();
        while let Some(n) = todo.pop() {
            for u in self.users.get(&n).into_iter().flatten() {
                if out.insert(u.clone()) {
                    todo.push(u.clone());
                }
            }
        }
        out
    }

    /// True if `other` holds the same names with the same definitions and references: the same
    /// codebase (rows are content-addressed, so equal hashes are equal definitions).
    pub fn same_codebase(&self, other: &Snapshot) -> bool {
        self.names == other.names
    }

    pub fn load(conn: &Connection, space: &Space) -> Result<Snapshot, Vec<Diagnostic>> {
        crate::space::require(conn, space)?;
        let mut names = BTreeMap::new();
        for (name, hash, locals, refs) in crate::space::rows(conn, space)? {
            let entry = NameEntry { hash: parse_hash(&hash)?, locals: json_list(&locals, "names.local_names")?, refs: json_list(&refs, "names.refs")? };
            names.insert(name, entry);
        }
        let (by_hash, users) = index(&names);
        let snap = Snapshot { names, by_hash, users, rows: RefCell::new(HashMap::new()), groups: RefCell::new(HashMap::new()) };
        let hashes: Vec<Hash> = snap.by_hash.keys().copied().collect();
        for h in hashes {
            snap.row(conn, &h)?.ok_or_else(|| corrupt(format!("name(s) {} point to the missing definition #{h}", snap.by_hash[&h].join(", "))))?;
        }
        // Every name's references are names of the store, and stand for exactly the hashes its
        // definition refers to.
        for (name, e) in &snap.names {
            let row = snap.row(conn, &e.hash)?.ok_or_else(|| corrupt(format!("`{name}` points to the missing definition #{}", e.hash)))?;
            let mut hashes = Vec::new();
            for r in &e.refs {
                let target = snap.names.get(r).ok_or_else(|| corrupt(format!("`{name}` refers to the unknown name `{r}`")))?;
                hashes.push(target.hash);
            }
            hashes.sort();
            if hashes != row.refs {
                return Err(corrupt(format!("the names `{name}` refers to do not match its definition #{}", e.hash)));
            }
        }
        Ok(snap)
    }

    /// The same contents with `old` named `new` (for checking a rename before writing it).
    pub fn renamed(&self, old: &str, new: &str) -> Snapshot {
        let mut names = self.names.clone();
        if let Some(entry) = names.remove(old) {
            names.insert(new.to_string(), entry);
        }
        for e in names.values_mut() {
            if e.refs.iter().any(|r| r == old) {
                e.refs = e.refs.iter().map(|r| if r == old { new.to_string() } else { r.clone() }).collect();
                e.refs.sort();
            }
        }
        let (by_hash, users) = index(&names);
        Snapshot { names, by_hash, users, rows: RefCell::new(self.rows.borrow().clone()), groups: RefCell::new(self.groups.borrow().clone()) }
    }

    /// The verified definition with hash `h` (cached), or None if the store has none.
    pub fn row(&self, conn: &Connection, h: &Hash) -> Result<Option<Row>, Vec<Diagnostic>> {
        if let Some(r) = self.rows.borrow().get(h) {
            return Ok(Some(r.clone()));
        }
        let found = conn
            .query_row("SELECT kind, canonical, local_names FROM defs WHERE hash = ?1", params![h.to_hex()], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?, r.get::<_, String>(2)?))
            })
            .optional()
            .map_err(sql)?;
        let Some((kind, canonical, locals)) = found else { return Ok(None) };
        if Hash::of(&canonical) != *h {
            return Err(corrupt(format!("definition #{h} does not match its hash")));
        }
        let kind = Kind::parse(&kind).ok_or_else(|| corrupt(format!("definition #{h} has the unknown kind `{kind}`")))?;
        let local_names = json_list(&locals, "defs.local_names")?;
        let header = canon::header(&canonical).map_err(|e| corrupt(format!("definition #{h}: {e}")))?;
        let (group, binders, refs) = match header.group {
            None => {
                let form = canon::Form::new(&canonical, None);
                let binders = canon::count_binders(form).map_err(|e| corrupt(format!("definition #{h}: {e}")))?;
                (None, binders, canon::references_of(form).map_err(|e| corrupt(format!("definition #{h}: {e}")))?)
            }
            Some(g) => {
                let group = self.group(conn, h, &g)?;
                if group.index.members.len() != header.size {
                    return Err(corrupt(format!("definition #{h}: the group form has another size")));
                }
                let info = group.index.members.get(header.index).ok_or_else(|| corrupt(format!("definition #{h}: member index out of range")))?.clone();
                (Some((group, header.index)), info.binders, info.refs)
            }
        };
        if binders != local_names.len() {
            return Err(corrupt(format!("definition #{h} has {binders} locals but {} local names", local_names.len())));
        }
        let row = Row { kind, canonical, group, local_names, refs };
        self.rows.borrow_mut().insert(*h, row.clone());
        Ok(Some(row))
    }

    /// The name texts use for `h`: the first of its names, if it has any.
    pub fn primary(&self, h: &Hash) -> Option<&str> {
        self.by_hash.get(h).and_then(|ns| ns.first()).map(|s| s.as_str())
    }

    /// Decodes the definition stored under `name`. Each reference is named by the name this
    /// definition was written against; one without a name is a `missing_dep` diagnostic.
    pub fn decode_name(&self, conn: &Connection, name: &str) -> Result<Def, Vec<Diagnostic>> {
        let entry = self.names.get(name).ok_or_else(|| vec![Diagnostic::unknown_name(name, "", name)])?;
        let hash = &entry.hash;
        let row = self.row(conn, hash)?.ok_or_else(|| corrupt(format!("`{name}` points to the missing definition #{hash}")))?;
        let locals = if entry.locals.len() == row.local_names.len() { entry.locals.clone() } else { row.local_names.clone() };
        let by_target: HashMap<Hash, &str> = entry.refs.iter().rev().filter_map(|r| self.names.get(r).map(|t| (t.hash, r.as_str()))).collect();
        let mut missing = None;
        let def = row
            .decode(name, &locals, &mut |h| match by_target.get(h) {
                Some(n) => n.to_string(),
                None => {
                    missing.get_or_insert(*h);
                    format!("#{}", h.short())
                }
            })
            .map_err(|e| corrupt(format!("`{name}` (#{hash}): {e}")))?;
        if Kind::of(&def) != row.kind {
            return Err(corrupt(format!("definition #{hash} is stored as a {} but encodes a {}", row.kind.as_str(), Kind::of(&def).as_str())));
        }
        if let Some(h) = missing {
            return Err(vec![Diagnostic::missing_dep(name, &h.to_hex())]);
        }
        Ok(def)
    }

    /// The group form with hash `g` (cached; verified by the decoder against `g`).
    fn group(&self, conn: &Connection, member: &Hash, g: &Hash) -> Result<Rc<Group>, Vec<Diagnostic>> {
        if let Some(f) = self.groups.borrow().get(g) {
            return Ok(f.clone());
        }
        let form = conn
            .query_row("SELECT form FROM groups WHERE hash = ?1", params![g.to_hex()], |r| r.get::<_, Vec<u8>>(0))
            .optional()
            .map_err(sql)?
            .ok_or_else(|| corrupt(format!("definition #{member} belongs to the missing group #{g}")))?;
        let index = canon::GroupIndex::new(&form, g).map_err(|e| corrupt(format!("group #{g} (member #{member}): {e}")))?;
        let group = Rc::new(Group { form, index });
        self.groups.borrow_mut().insert(*g, group.clone());
        Ok(group)
    }

    /// The name a hash is shown with: its primary name, else the last name it had (history), else
    /// `#<hash prefix>`.
    fn display_name(&self, conn: &Connection, h: &Hash) -> Result<String, Vec<Diagnostic>> {
        if let Some(n) = self.primary(h) {
            return Ok(n.to_string());
        }
        let last: Option<String> = conn
            .query_row("SELECT name FROM history WHERE new_hash = ?1 ORDER BY id DESC LIMIT 1", params![h.to_hex()], |r| r.get(0))
            .optional()
            .map_err(sql)?;
        Ok(last.unwrap_or_else(|| format!("#{}", h.short())))
    }

    /// Decodes any stored definition by hash for display (old versions included): definitions
    /// without a current name are shown with the name they last had.
    pub fn decode_hash(&self, conn: &Connection, h: &Hash) -> Result<Option<Def>, Vec<Diagnostic>> {
        // A current definition is shown as one of its names wrote it (its own references), so a
        // std definition never shows a project type of the same structure (review N2).
        if let Some(name) = self.primary(h) {
            return self.decode_name(conn, name).map(Some);
        }
        let Some(row) = self.row(conn, h)? else { return Ok(None) };
        let own = self.display_name(conn, h)?;
        let mut names: HashMap<Hash, String> = HashMap::new();
        for r in &row.refs {
            names.insert(*r, self.display_name(conn, r)?);
        }
        let def = row
            .decode(&own, &row.local_names, &mut |r| names.get(r).cloned().unwrap_or_else(|| format!("#{}", r.short())))
            .map_err(|e| corrupt(format!("#{h}: {e}")))?;
        Ok(Some(def))
    }

    /// The whole named codebase as a program, except the names in `skip`: types, then functions,
    /// then tests, each sorted by name.
    pub fn program(&self, conn: &Connection, skip: &dyn Fn(&str) -> bool) -> Result<Program, Vec<Diagnostic>> {
        let mut defs = Vec::new();
        let mut diags = Vec::new();
        for name in self.names.keys() {
            if skip(name) {
                continue;
            }
            match self.decode_name(conn, name) {
                Ok(d) => defs.push(d),
                Err(d) => diags.extend(d),
            }
        }
        if !diags.is_empty() {
            return Err(diags);
        }
        defs.sort_by_key(|d| match d {
            Def::Type(_) => 0,
            Def::Fn(_) => 1,
            Def::Test(_) => 2,
        });
        Ok(Program { defs })
    }

    /// Names of the stored definitions written against `name`, sorted.
    pub fn dependents(&self, name: &str) -> Vec<String> {
        self.users.get(name).cloned().unwrap_or_default()
    }
}
