//! Spaces (Phase 4 spec §3, §6.1): `main` (the table `names`) or a change set (its rows in
//! `change_names`). Every space has a complete name table; definitions, dependencies, groups and
//! proofs are shared by all spaces.

use crate::sql;
use qal_core::diag::Diagnostic;
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::BTreeMap;

/// The id of the main space.
pub const MAIN: &str = "main";

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum Space {
    #[default]
    Main,
    /// A change set by id (`c` + 12 hex digits).
    Change(String),
}

impl Space {
    /// `main` or a change set id.
    pub fn parse(id: &str) -> Space {
        if id == MAIN {
            Space::Main
        } else {
            Space::Change(id.to_string())
        }
    }

    pub fn id(&self) -> &str {
        match self {
            Space::Main => MAIN,
            Space::Change(c) => c,
        }
    }

    pub fn is_main(&self) -> bool {
        *self == Space::Main
    }
}

impl std::fmt::Display for Space {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.id())
    }
}

/// True if `id` looks like a change set id (`c` + 12 lowercase hex digits).
pub fn is_change_id(id: &str) -> bool {
    id.len() == 13 && id.starts_with('c') && id[1..].bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// E0201 for a change set that does not exist.
pub fn unknown_change(id: &str) -> Diagnostic {
    Diagnostic::new("E0201", "unknown_name", "", "change", format!("there is no change set `{id}`")).hint("qal_change {\"action\": \"list\"} lists the change sets")
}

/// Refuses a change space that does not exist.
pub fn require(conn: &Connection, space: &Space) -> Result<(), Vec<Diagnostic>> {
    if let Space::Change(c) = space {
        let found: Option<i64> = conn.query_row("SELECT 1 FROM changes WHERE id = ?1", params![c], |r| r.get(0)).optional().map_err(sql)?;
        if found.is_none() {
            return Err(vec![unknown_change(c)]);
        }
    }
    Ok(())
}

/// One name row of a space: hash, local names and refs (JSON lists as stored).
pub type NameRow = (String, String, String, String);

/// Every name of the space with its hash, local names and refs (JSON), by name.
pub fn rows(conn: &Connection, space: &Space) -> Result<Vec<NameRow>, Vec<Diagnostic>> {
    let (sql_text, id) = match space {
        Space::Main => ("SELECT name, hash, local_names, refs FROM names ORDER BY name", None),
        Space::Change(c) => ("SELECT name, hash, local_names, refs FROM change_names WHERE change = ?1 AND hash IS NOT NULL ORDER BY name", Some(c.as_str())),
    };
    let mut stmt = conn.prepare(sql_text).map_err(sql)?;
    let map = |r: &rusqlite::Row| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?, r.get::<_, String>(3)?));
    let it = match id {
        None => stmt.query_map([], map),
        Some(c) => stmt.query_map(params![c], map),
    }
    .map_err(sql)?;
    it.collect::<Result<Vec<_>, _>>().map_err(sql)
}

/// The current `name -> hash` of a space.
pub fn names(conn: &Connection, space: &Space) -> Result<BTreeMap<String, String>, Vec<Diagnostic>> {
    Ok(rows(conn, space)?.into_iter().map(|(n, h, _, _)| (n, h)).collect())
}

/// The hash of one name in a space.
pub fn hash_of(conn: &Connection, space: &Space, name: &str) -> Result<Option<String>, Vec<Diagnostic>> {
    match space {
        Space::Main => conn.query_row("SELECT hash FROM names WHERE name = ?1", params![name], |r| r.get(0)).optional().map_err(sql),
        Space::Change(c) => conn
            .query_row("SELECT hash FROM change_names WHERE change = ?1 AND name = ?2", params![c, name], |r| r.get::<_, Option<String>>(0))
            .optional()
            .map(Option::flatten)
            .map_err(sql),
    }
}

/// Writes the row of `name` (a new hash, or new local names / refs). In a change set the origin
/// follows §6.1: an input (`edited` if the name was in the base, else `added`; a conflict mark is
/// cleared) or a relinked dependent (`relinked` if it was unchanged from the base).
#[allow(clippy::too_many_arguments)]
pub fn put(conn: &Connection, space: &Space, name: &str, hash: &str, locals: &str, refs: &str, input: bool, now: i64) -> Result<(), Vec<Diagnostic>> {
    match space {
        Space::Main => conn.execute(
            "INSERT INTO names (name, hash, local_names, refs, updated_at) VALUES (?1, ?2, ?3, ?4, ?5) \
             ON CONFLICT(name) DO UPDATE SET hash = excluded.hash, local_names = excluded.local_names, refs = excluded.refs, updated_at = excluded.updated_at",
            params![name, hash, locals, refs, now],
        ),
        Space::Change(c) => conn.execute(
            "INSERT INTO change_names (change, name, hash, local_names, refs, origin, base_hash, conflict, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, 'added', NULL, NULL, ?6) \
             ON CONFLICT(change, name) DO UPDATE SET hash = excluded.hash, local_names = excluded.local_names, refs = excluded.refs, updated_at = excluded.updated_at, \
             origin = CASE WHEN ?7 THEN (CASE WHEN change_names.base_hash IS NOT NULL THEN 'edited' ELSE 'added' END) \
                           WHEN change_names.origin = 'base' THEN 'relinked' ELSE change_names.origin END, \
             conflict = CASE WHEN ?7 THEN NULL ELSE change_names.conflict END",
            params![c, name, hash, locals, refs, now, input],
        ),
    }
    .map(|_| ())
    .map_err(sql)
}

/// Removes `name` from a space (a delete, or the old name of a rename: `origin` `edited` or
/// `renamed`). In a change set a name of the base keeps its row with no hash; one added in the
/// change set goes.
pub fn remove(conn: &Connection, space: &Space, name: &str, origin: &str, now: i64) -> Result<(), Vec<Diagnostic>> {
    match space {
        Space::Main => conn.execute("DELETE FROM names WHERE name = ?1", params![name]).map(|_| ()),
        Space::Change(c) => {
            conn.execute("DELETE FROM change_names WHERE change = ?1 AND name = ?2 AND base_hash IS NULL", params![c, name])
                .and_then(|_| {
                    conn.execute(
                        "UPDATE change_names SET hash = NULL, local_names = '[]', refs = '[]', origin = ?3, conflict = NULL, updated_at = ?4 WHERE change = ?1 AND name = ?2",
                        params![c, name, origin, now],
                    )
                })
                .map(|_| ())
        }
    }
    .map_err(sql)
}

/// Sets the refs of `name` (a user of a renamed name): in a change set a base name becomes `relinked`.
pub fn set_refs(conn: &Connection, space: &Space, name: &str, refs: &str, now: i64) -> Result<(), Vec<Diagnostic>> {
    match space {
        Space::Main => conn.execute("UPDATE names SET refs = ?1 WHERE name = ?2", params![refs, name]),
        Space::Change(c) => conn.execute(
            "UPDATE change_names SET refs = ?1, updated_at = ?4, origin = CASE WHEN origin = 'base' THEN 'relinked' ELSE origin END WHERE change = ?2 AND name = ?3",
            params![refs, c, name, now],
        ),
    }
    .map(|_| ())
    .map_err(sql)
}

/// The SQL condition on `history` rows of a space (`?space` is bound by the caller as parameter
/// `n`): main's rows, or a change set's own rows plus main's rows up to its base.
pub fn history_filter(space: &Space, n: usize) -> String {
    match space {
        Space::Main => format!("space = ?{n}"),
        Space::Change(_) => format!(
            "(space = ?{n} OR (space = 'main' AND (record IS NULL OR record <= (SELECT base_seq FROM changes WHERE id = ?{n}))))"
        ),
    }
}
