//! A consistent, verified view of the store's contents (read inside one transaction), and decoding
//! of stored definitions back into source ASTs with the current names.

use crate::sql;
use qal_core::ast::{Def, Program};
use qal_core::canon::{self, Hash, Kind};
use qal_core::diag::Diagnostic;
use rusqlite::{params, Connection, OptionalExtension};
use std::cell::RefCell;
use std::rc::Rc;
use std::collections::{BTreeMap, HashMap};

/// One row of `defs`, verified: `sha256(canonical) == hash`, and the canonical form decodes.
#[derive(Debug, Clone)]
pub struct Row {
    pub kind: Kind,
    pub canonical: Vec<u8>,
    /// The group form, for a member of a group (shared by all members).
    pub group: Option<Rc<Vec<u8>>>,
    pub local_names: Vec<String>,
}

impl Row {
    pub fn form(&self) -> canon::Form<'_> {
        canon::Form::new(&self.canonical, self.group.as_ref().map(|g| g.as_slice()))
    }
}

pub struct Snapshot {
    /// name -> (hash, local names of that name's text)
    pub names: BTreeMap<String, (Hash, Vec<String>)>,
    /// hash -> its names, sorted (the first one is the primary name used in texts)
    pub by_hash: HashMap<Hash, Vec<String>>,
    rows: RefCell<HashMap<Hash, Row>>,
    groups: RefCell<HashMap<Hash, Rc<Vec<u8>>>>,
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

impl Snapshot {
    /// Reads all names and the definitions they point to.
    pub fn load(conn: &Connection) -> Result<Snapshot, Vec<Diagnostic>> {
        let mut names = BTreeMap::new();
        let mut by_hash: HashMap<Hash, Vec<String>> = HashMap::new();
        let mut stmt = conn.prepare("SELECT name, hash, local_names FROM names ORDER BY name").map_err(sql)?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))).map_err(sql)?;
        for row in rows {
            let (name, hash, locals) = row.map_err(sql)?;
            let hash = parse_hash(&hash)?;
            by_hash.entry(hash).or_default().push(name.clone());
            names.insert(name, (hash, json_list(&locals, "names.local_names")?));
        }
        let snap = Snapshot { names, by_hash, rows: RefCell::new(HashMap::new()), groups: RefCell::new(HashMap::new()) };
        let hashes: Vec<Hash> = snap.by_hash.keys().copied().collect();
        for h in hashes {
            snap.row(conn, &h)?.ok_or_else(|| corrupt(format!("name(s) {} point to the missing definition #{h}", snap.by_hash[&h].join(", "))))?;
        }
        Ok(snap)
    }

    /// The same contents with `old` named `new` (for checking a rename before writing it).
    pub fn renamed(&self, old: &str, new: &str) -> Snapshot {
        let mut names = self.names.clone();
        if let Some(entry) = names.remove(old) {
            names.insert(new.to_string(), entry);
        }
        let mut by_hash: HashMap<Hash, Vec<String>> = HashMap::new();
        for (n, (h, _)) in &names {
            by_hash.entry(*h).or_default().push(n.clone());
        }
        Snapshot { names, by_hash, rows: RefCell::new(self.rows.borrow().clone()), groups: RefCell::new(self.groups.borrow().clone()) }
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
        let group = match header.group {
            None => None,
            Some(g) => Some(self.group(conn, h, &g)?),
        };
        let binders = canon::count_binders(canon::Form::new(&canonical, group.as_ref().map(|g| g.as_slice()))).map_err(|e| corrupt(format!("definition #{h}: {e}")))?;
        if binders != local_names.len() {
            return Err(corrupt(format!("definition #{h} has {binders} locals but {} local names", local_names.len())));
        }
        let row = Row { kind, canonical, group, local_names };
        self.rows.borrow_mut().insert(*h, row.clone());
        Ok(Some(row))
    }

    /// The name texts use for `h`: the first of its names, if it has any.
    pub fn primary(&self, h: &Hash) -> Option<&str> {
        self.by_hash.get(h).and_then(|ns| ns.first()).map(|s| s.as_str())
    }

    /// Decodes the definition stored under `name`. References are named through `overrides` first,
    /// then by their primary name; a reference without a name is a `missing_dep` diagnostic.
    pub fn decode_name(&self, conn: &Connection, name: &str, overrides: &HashMap<Hash, String>) -> Result<Def, Vec<Diagnostic>> {
        let (hash, locals) = self.names.get(name).ok_or_else(|| vec![Diagnostic::unknown_name(name, "", name)])?;
        let row = self.row(conn, hash)?.ok_or_else(|| corrupt(format!("`{name}` points to the missing definition #{hash}")))?;
        let locals = if locals.len() == row.local_names.len() { locals.clone() } else { row.local_names.clone() };
        let mut missing = None;
        let def = canon::decode(row.form(), name, &locals, &mut |h| match overrides.get(h).map(|s| s.as_str()).or_else(|| self.primary(h)) {
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
    fn group(&self, conn: &Connection, member: &Hash, g: &Hash) -> Result<Rc<Vec<u8>>, Vec<Diagnostic>> {
        if let Some(f) = self.groups.borrow().get(g) {
            return Ok(f.clone());
        }
        let form = conn
            .query_row("SELECT form FROM groups WHERE hash = ?1", params![g.to_hex()], |r| r.get::<_, Vec<u8>>(0))
            .optional()
            .map_err(sql)?
            .ok_or_else(|| corrupt(format!("definition #{member} belongs to the missing group #{g}")))?;
        let form = Rc::new(form);
        self.groups.borrow_mut().insert(*g, form.clone());
        Ok(form)
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
        let Some(row) = self.row(conn, h)? else { return Ok(None) };
        let own = self.display_name(conn, h)?;
        let mut names: HashMap<Hash, String> = HashMap::new();
        for r in canon::references_of(row.form()).map_err(|e| corrupt(format!("#{h}: {e}")))? {
            names.insert(r, self.display_name(conn, &r)?);
        }
        let def = canon::decode(row.form(), &own, &row.local_names, &mut |r| names.get(r).cloned().unwrap_or_else(|| format!("#{}", r.short())))
            .map_err(|e| corrupt(format!("#{h}: {e}")))?;
        Ok(Some(def))
    }

    /// The whole named codebase as a program, except the names in `skip`: types, then functions,
    /// then tests, each sorted by name.
    pub fn program(&self, conn: &Connection, overrides: &HashMap<Hash, String>, skip: &dyn Fn(&str) -> bool) -> Result<Program, Vec<Diagnostic>> {
        let mut defs = Vec::new();
        let mut diags = Vec::new();
        for name in self.names.keys() {
            if skip(name) {
                continue;
            }
            match self.decode_name(conn, name, overrides) {
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

    /// Names of the stored definitions that refer to `h` directly.
    pub fn dependents(&self, conn: &Connection, h: &Hash) -> Result<Vec<String>, Vec<Diagnostic>> {
        let mut stmt = conn
            .prepare("SELECT DISTINCT n.name FROM deps d JOIN names n ON n.hash = d.hash WHERE d.dep_hash = ?1 ORDER BY n.name")
            .map_err(sql)?;
        let rows = stmt.query_map(params![h.to_hex()], |r| r.get::<_, String>(0)).map_err(sql)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(sql)
    }
}
