//! Packages: named definitions exported as text with their hashes and origin, and imported with
//! the guarantee that the same text gives the same hashes (Phase 3 spec §4.3).
//!
//! ```text
//! # qal-pkg 1
//! # package <name> <sha256 of everything after this line>
//! # def <name> <hash> <agent>          (one line per name, sorted by name)
//!
//! <canonical text of all definitions, dependencies first, separated by blank lines>
//! ```
//!
//! `agent` is who last added or replaced the definition (the recorded origin). Importing defines
//! the text in one atomic `define` and keeps it only if every listed name is defined by the text
//! and gets exactly the recorded hash, and nothing else is defined (E0243). `std.` names: only the
//! package named `std` adds or replaces them (keeping their signatures, E0244); another package may
//! carry them only unchanged (E0240). Importing into the std project keeps each `# def` agent.

use crate::define::{self, Release, Rules, StdNames};
use qal_compiler::lexer::is_std_name;
use crate::snapshot::Snapshot;
use crate::{sql, Defined, Verify};
use qal_core::ast::Program;
use qal_core::canon::Hash;
use qal_core::diag::Diagnostic;
use rusqlite::{params, Connection, OptionalExtension};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

pub const FORMAT_LINE: &str = "# qal-pkg 1";

/// One `# def` line.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PackageDef {
    pub name: String,
    pub hash: String,
    pub agent: String,
}

/// A parsed, checksum-verified package.
#[derive(Debug, Clone, PartialEq)]
pub struct Package {
    pub name: String,
    pub hash: String,
    pub defs: Vec<PackageDef>,
    /// The std package the exporting project had (`# std <hash>`), if the package refers to
    /// `std.` names (which it never contains, except the std package itself).
    pub std: Option<String>,
    /// The definitions' text.
    pub text: String,
}

/// The result of an export.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Exported {
    pub package: String,
    pub hash: String,
    pub defs: usize,
    #[serde(skip)]
    pub file: String,
}

/// The result of an import.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Imported {
    pub package: String,
    pub hash: String,
    pub defined: Vec<Defined>,
}

fn mismatch(message: impl Into<String>) -> Vec<Diagnostic> {
    vec![Diagnostic::new("E0243", "package_mismatch", "", "", message).hint("the package file was changed or damaged; export it again from its source project")]
}

/// A package name: lowercase letters, digits, `_` and `-`, starting with a letter.
pub fn valid_package_name(n: &str) -> bool {
    (1..=64).contains(&n.len()) && n.starts_with(|c: char| c.is_ascii_lowercase()) && n.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
}

/// Parses and verifies a package file (format, checksum, `# def` lines).
pub fn parse(file: &str) -> Result<Package, Vec<Diagnostic>> {
    let (first, rest) = file.split_once('\n').ok_or_else(|| mismatch("not a QAL package: the header is missing"))?;
    if first != FORMAT_LINE {
        return Err(mismatch(format!("not a QAL package of format 1: the first line is `{}`", first.chars().take(40).collect::<String>())));
    }
    let (pkg_line, body) = rest.split_once('\n').unwrap_or((rest, ""));
    let mut parts = pkg_line.split(' ');
    let (Some("#"), Some("package"), Some(name), Some(hash), None) = (parts.next(), parts.next(), parts.next(), parts.next(), parts.next()) else {
        return Err(mismatch("the second line must be `# package <name> <sha256>`"));
    };
    if Hash::of(body.as_bytes()).to_hex() != hash {
        return Err(mismatch(format!("the checksum of package `{name}` does not match its content")));
    }
    let mut defs = Vec::new();
    let mut offset = 0;
    let mut std = None;
    if let Some(line) = body.split_inclusive('\n').next().filter(|l| l.starts_with("# std ")) {
        let h = line["# std ".len()..].trim_end_matches('\n');
        if !(h.len() == 64 && h.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))) {
            return Err(mismatch("malformed `# std` line: expected `# std <64 hex digits>`"));
        }
        std = Some(h.to_string());
        offset = line.len();
    }
    let mut lines = body[offset..].split_inclusive('\n');
    for line in lines.by_ref() {
        let Some(def) = line.strip_prefix("# def ") else { break };
        offset += line.len();
        let def = def.trim_end_matches('\n');
        let mut p = def.splitn(3, ' ');
        let shown = || def.chars().take(80).collect::<String>();
        let (Some(n), Some(h), Some(agent)) = (p.next(), p.next(), p.next()) else { return Err(mismatch(format!("malformed `# def` line `{}`", shown()))) };
        // The checksum covers these lines but anyone can recompute it: everything is untrusted.
        let is_name = n.len() <= 256 && (is_std_name(n) || (n.starts_with(|c: char| c.is_ascii_alphabetic()) && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')));
        let is_hash = h.len() == 64 && h.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        if !is_name || !is_hash || agent.is_empty() || agent.chars().any(char::is_control) {
            return Err(mismatch(format!("malformed `# def` line `{}`: expected `# def <name> <64 hex digits> <agent>`", shown())));
        }
        // Sorted by name without duplicates, as `export` writes them.
        if let Some(last) = defs.last().filter(|d: &&PackageDef| d.name.as_str() >= n) {
            return Err(mismatch(if last.name == n { format!("`{n}` is listed twice") } else { format!("the `# def` lines are not sorted by name (`{n}` after `{}`)", last.name) }));
        }
        defs.push(PackageDef { name: n.into(), hash: h.into(), agent: agent.into() });
    }
    Ok(Package { name: name.into(), hash: hash.into(), defs, std, text: body[offset..].to_string() })
}

/// The first 12 characters of a hash (never splits a character).
fn short(h: &str) -> &str {
    h.char_indices().nth(12).map_or(h, |(i, _)| &h[..i])
}

/// Builds the package file of `name` from the definitions in `text` and their `# def` lines.
fn assemble(name: &str, std: Option<&str>, defs: &[PackageDef], text: &str) -> (String, String) {
    let mut body = std.map(|h| format!("# std {h}\n")).unwrap_or_default();
    for d in defs {
        body.push_str(&format!("# def {} {} {}\n", d.name, d.hash, d.agent));
    }
    if !text.is_empty() {
        body.push('\n');
        body.push_str(text);
    }
    let hash = Hash::of(body.as_bytes()).to_hex();
    (format!("{FORMAT_LINE}\n# package {name} {hash}\n{body}"), hash)
}

/// The names a package of `prefix` holds: the matching names and everything they depend on;
/// `std.` names only if `with_std` (the std project), since every project has its own std.
fn closure(snap: &Snapshot, prefix: &str, with_std: bool) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut todo: Vec<String> = snap.names.keys().filter(|n| n.starts_with(prefix)).cloned().collect();
    while let Some(n) = todo.pop() {
        if !with_std && is_std_name(&n) {
            continue;
        }
        if out.insert(n.clone()) {
            if let Some(e) = snap.names.get(&n) {
                todo.extend(e.refs.iter().cloned());
            }
        }
    }
    out
}

/// `names` with dependencies first; ties (and mutually recursive groups) by name.
fn dependency_order(snap: &Snapshot, names: &BTreeSet<String>) -> Vec<String> {
    let mut deps: BTreeMap<&str, BTreeSet<&str>> = names.iter().map(|n| (n.as_str(), snap.names.get(n).map(|e| e.refs.iter().map(String::as_str).filter(|r| names.contains(*r) && *r != n).collect()).unwrap_or_default())).collect();
    let mut out = Vec::new();
    while !deps.is_empty() {
        let ready: Vec<&str> = deps.iter().filter(|(_, ds)| ds.is_empty()).map(|(n, _)| *n).collect();
        // A cycle (mutual recursion): take its smallest name; define handles the group at once.
        let take = if ready.is_empty() { deps.keys().take(1).copied().collect() } else { ready };
        for n in take {
            deps.remove(n);
            for ds in deps.values_mut() {
                ds.remove(n);
            }
            out.push(n.to_string());
        }
    }
    out
}

/// Exports every name starting with `prefix` (and its dependencies) as package `package`.
pub fn export(conn: &mut Connection, prefix: &str, package: &str) -> Result<Exported, Vec<Diagnostic>> {
    if !valid_package_name(package) {
        return Err(vec![Diagnostic::new("E0701", "bad_arguments", "", "", format!("`{package}` is not a package name")).hint("package names are 1 to 64 characters `a-z`, `0-9`, `_`, `-`, starting with a letter")]);
    }
    let tx = conn.transaction().map_err(sql)?;
    let snap = Snapshot::load(&tx)?;
    let std_store = define::is_std_store(&tx)?;
    let names = closure(&snap, prefix, std_store);
    // A project's package refers to its std by name: record which std that was.
    let uses_std = !std_store && names.iter().any(|n| snap.names[n].refs.iter().any(|r| is_std_name(r)));
    let std = if uses_std { meta(&tx, META_STD_PACKAGE)? } else { None };
    let mut defs = Vec::new();
    for n in &names {
        let agent: Option<String> = tx
            .query_row("SELECT agent FROM history WHERE name = ?1 AND action IN ('added', 'replaced') ORDER BY id DESC LIMIT 1", params![n], |r| r.get(0))
            .optional()
            .map_err(sql)?;
        defs.push(PackageDef { name: n.clone(), hash: snap.names[n].hash.to_hex(), agent: agent.unwrap_or_else(|| "unknown".into()) });
    }
    let mut texts = Vec::new();
    for n in dependency_order(&snap, &names) {
        let d = snap.decode_name(&tx, &n)?;
        texts.push(qal_compiler::print_program(&Program { defs: vec![d] }).trim_end().to_string());
    }
    let text = if texts.is_empty() { String::new() } else { texts.join("\n\n") + "\n" };
    let (file, hash) = assemble(package, std.as_deref(), &defs, &text);
    Ok(Exported { package: package.into(), hash, defs: defs.len(), file })
}

/// Imports a package: one atomic `define` of its text (agent `qal-<package>@<hash prefix>`), kept
/// only if every name gets the recorded hash (`package_mismatch`, E0243, otherwise; nothing is
/// written). Records `package:<name>` in `meta`. With `verify`, the definitions are verified (the
/// contracts proven) exactly as by `define`. `post` runs last inside the write transaction (the
/// signed record of a team store).
pub fn import(conn: &mut Connection, file: &str, verify: Option<&mut Verify>, release: Option<&Release>, post: &define::Check) -> Result<Imported, Vec<Diagnostic>> {
    let pkg = parse(file)?;
    if !valid_package_name(&pkg.name) {
        return Err(mismatch(format!("`{}` is not a package name", pkg.name)));
    }
    let is_std = pkg.name == STD_PACKAGE_NAME;
    let is_release = release.is_some_and(|r| r.hash == pkg.hash);
    // Only the released std (embedded in the binary) adds or replaces `std.` names in a project:
    // anyone can name a package `std` and recompute its checksum (review finding I1).
    if is_std && !is_release && !define::is_std_store(conn)? {
        let ours = release.map_or("none".to_string(), |r| short(&r.hash).to_string());
        return Err(vec![Diagnostic::new("E0243", "package_mismatch", "", "", format!("package `std` {} is not the standard library of this qal (std {ours})", short(&pkg.hash)))
            .hint("a project gets its standard library only from its qal (it is imported automatically); to change std, change the std project (qal/std) and rebuild qal")]);
    }
    let agent = format!("qal-{}@{}", pkg.name, &pkg.hash[..12]);
    let expected: BTreeMap<&str, &str> = pkg.defs.iter().map(|d| (d.name.as_str(), d.hash.as_str())).collect();
    let check = |defined: &[Defined], tx: &rusqlite::Transaction, before: &crate::team::Rows| -> Result<(), Vec<Diagnostic>> {
        // Every listed name is defined by the text (not only already stored).
        let given: BTreeSet<&str> = defined.iter().filter(|d| d.status != "relinked").map(|d| d.name.as_str()).collect();
        if let Some(missing) = expected.keys().find(|n| !given.contains(*n)) {
            return Err(mismatch(format!("`{missing}` is listed in the package but its text does not define it")));
        }
        let std_store = define::is_std_store(tx)?;
        for (name, hash) in &expected {
            let stored: Option<String> = tx.query_row("SELECT hash FROM names WHERE name = ?1", params![name], |r| r.get(0)).optional().map_err(sql)?;
            if stored.as_deref() != Some(*hash) {
                return Err(mismatch(format!("`{name}` was recorded with hash {} but defines as {}", short(hash), stored.as_deref().map_or("nothing", short))));
            }
        }
        // A package may replace only what an earlier import of the same package defined, never
        // the project's own definitions.
        // In the std project the std package replaces what the std agents wrote (a rebuild).
        let ours = format!("qal-{}@", pkg.name);
        for d in defined.iter().filter(|d| d.status == "replaced" && !(std_store && is_std)) {
            let last: Option<String> = tx
                .query_row("SELECT agent FROM history WHERE name = ?1 AND action IN ('added', 'replaced') AND new_hash != ?2 ORDER BY id DESC LIMIT 1", params![d.name, d.hash], |r| r.get(0))
                .optional()
                .map_err(sql)?;
            if !last.as_deref().is_some_and(|a| a.starts_with(&ours)) {
                return Err(vec![Diagnostic::new("E0243", "package_mismatch", &d.name, "", format!("importing package `{}` would replace the project's own `{}`", pkg.name, d.name))
                    .hint(format!("rename the project's `{}` first, or leave the package out", d.name))]);
            }
        }
        if let Some(extra) = defined.iter().find(|d| !expected.contains_key(d.name.as_str()) && d.status != "relinked") {
            return Err(mismatch(format!("`{}` is defined by the package text but not listed in it", extra.name)));
        }
        set_meta(tx, &format!("package:{}", pkg.name), &pkg.hash)?;
        if is_std {
            set_meta(tx, META_STD_PACKAGE, &pkg.hash)?;
        }
        // The std project keeps the recorded origin of each definition (`# def` agent) when it is
        // rebuilt from the released package, so its next export names the same agents. Only the
        // release: another file's agents are not trusted (review finding M3).
        if std_store && is_release {
            for d in &pkg.defs {
                tx.execute(
                    "UPDATE history SET agent = ?1 WHERE id = (SELECT max(id) FROM history WHERE name = ?2 AND agent = ?3 AND action IN ('added', 'replaced'))",
                    params![d.agent, d.name, agent],
                )
                .map_err(sql)?;
            }
        }
        post(defined, tx, before)
    };
    let rules = Rules { std: if is_std { StdNames::Package } else { StdNames::Unchanged }, release, breaking: false };
    let ours = meta(conn, META_STD_PACKAGE)?;
    let defined = define::define_checked(conn, &pkg.text, &agent, &rules, verify, &check).map_err(|mut d| {
        // Built against another std: say so first (the hashes or names differ because of it).
        if let Some(theirs) = pkg.std.as_deref().filter(|h| ours.as_deref() != Some(*h)) {
            let mine = ours.as_deref().map_or("none".to_string(), |h| short(h).to_string());
            d.insert(
                0,
                Diagnostic::new("E0243", "package_mismatch", "", "", format!("package `{}` was built against std {}, this project has std {mine}", pkg.name, short(theirs)))
                    .hint("the package refers to standard library definitions of another version: import it into a project with the same qal (std) version, or export it again from a project with this one"),
            );
        }
        d
    })?;
    Ok(Imported { package: pkg.name, hash: pkg.hash, defined })
}

/// The name of the standard library package.
pub const STD_PACKAGE_NAME: &str = "std";
/// `meta` key: hash of the std package last imported into the store.
pub const META_STD_PACKAGE: &str = "std_package";
/// `meta` key: `std` for the std project's store (absent for every other project).
pub const META_ROLE: &str = "role";

pub(crate) fn set_meta(tx: &rusqlite::Transaction, key: &str, value: &str) -> Result<(), Vec<Diagnostic>> {
    tx.execute("INSERT INTO meta (key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value", params![key, value]).map(|_| ()).map_err(sql)
}

/// A value of the `meta` table.
pub fn meta(conn: &Connection, key: &str) -> Result<Option<String>, Vec<Diagnostic>> {
    conn.query_row("SELECT value FROM meta WHERE key = ?1", params![key], |r| r.get(0)).optional().map_err(sql)
}
