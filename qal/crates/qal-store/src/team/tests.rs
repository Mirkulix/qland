use super::canonical::{self, root};
use super::keys::{KeyDir, SECRET_PREFIX};
use super::policy::{self, Role, Space};
use super::verify::{Outcome, Trust};
use super::*;
use crate::Store;
use rusqlite::Connection;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn temp(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qal-team-test-{tag}-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::SeqCst)));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A project and a key directory outside it (not created yet: `generate` creates it with 0700).
fn setup() -> (Store, KeyDir) {
    let project = temp("p");
    let keys = KeyDir::at(temp("k").join("keys"));
    (Store::init(&project).unwrap(), keys)
}

fn raw(s: &Store) -> Connection {
    Connection::open(s.path()).unwrap()
}

/// A team store with an architect `arch-1` and an implementer `impl-1`; returns their signers.
/// The architect has opened an own change set (record 4), and the store works in it.
fn team() -> (Store, KeyDir, Signer, Signer, Signer) {
    let (mut s, keys, owner, arch, imp) = bare_team();
    let c = s.change_open(&arch, &change::OpenArgs { title: "work".into(), ..Default::default() }).unwrap();
    s.set_space(crate::Space::Change(c["change"].as_str().unwrap().to_string()));
    (s, keys, owner, arch, imp)
}

/// [`team`] without the change set.
fn bare_team() -> (Store, KeyDir, Signer, Signer, Signer) {
    let (mut s, keys) = setup();
    s.define("fn base() -> Int = 1", "pre-team").unwrap();
    s.team_init("boss", &keys).unwrap();
    let owner = s.owner_signer(&keys).unwrap();
    let a = s.team_add(&owner, "arch-1", Role::Architect, &keys).unwrap();
    let i = s.team_add(&owner, "impl-1", Role::Implementer, &keys).unwrap();
    let project = s.project_dir().unwrap().to_path_buf();
    let arch = Signer { member: "arch-1".into(), session: "s000000000001".into(), key: keys.load(&project, &a.key_id).unwrap() };
    let imp = Signer { member: "impl-1".into(), session: "s000000000002".into(), key: keys.load(&project, &i.key_id).unwrap() };
    (s, keys, owner, arch, imp)
}

fn as_member(s: &mut Store, signer: Signer) {
    s.set_signer(Some(Arc::new(signer)));
}

fn verified(s: &mut Store, trust: Option<&Trust>) -> Outcome {
    s.verify(trust, 1, None).unwrap()
}

fn code(o: &Outcome) -> &'static str {
    o.error.as_ref().map_or("ok", |e| e.code)
}

#[test]
fn canonical_json_sorts_keys_and_has_no_whitespace() {
    // The golden bytes of a record body (spec §9.1): pinned so the signed format cannot drift.
    let body = json!({"v": 1, "seq": 2, "prev": ZERO_HASH, "at": 1_792_000_000_000_i64, "member": "impl-1", "key": "0123456789abcdef", "session": "s3f0000000000",
                      "op": "define", "space": "main", "args": {"entries": [{"name": "f", "old": null, "new": "ab", "action": "added", "refs": ["g"], "locals": ["x"]}]}, "root_before": "r0", "root_after": "r1", "names_after": "d1"});
    let bytes = record_bytes(&body).unwrap();
    let golden = concat!(
        "qal-record-v1\n",
        r#"{"args":{"entries":[{"action":"added","locals":["x"],"name":"f","new":"ab","old":null,"refs":["g"]}]},"at":1792000000000,"key":"0123456789abcdef","member":"impl-1","names_after":"d1","op":"define","prev":"0000000000000000000000000000000000000000000000000000000000000000","root_after":"r1","root_before":"r0","seq":2,"session":"s3f0000000000","space":"main","v":1}"#
    );
    assert_eq!(String::from_utf8(bytes).unwrap(), golden);
    // `preserve_order` is not active: insertion order does not matter.
    let mut m = serde_json::Map::new();
    m.insert("b".into(), json!(1));
    m.insert("a".into(), json!("é\n\"x\""));
    assert_eq!(canonical::to_string(&Value::Object(m)).unwrap(), r#"{"a":"é\n\"x\"","b":1}"#);
    // Only integers below 2^53.
    assert!(canonical::to_string(&json!({"x": 1.5})).is_err());
    assert!(canonical::to_string(&json!([9_007_199_254_740_992_u64])).is_err());
    assert!(canonical::to_string(&json!([9_007_199_254_740_991_u64, -9_007_199_254_740_991_i64])).is_ok());
}

#[test]
fn the_root_hashes_sorted_name_lines() {
    assert_eq!(root(&BTreeMap::new()), Hash::of(b"").to_hex());
    let m: BTreeMap<String, String> = [("b".to_string(), "h2".to_string()), ("a".to_string(), "h1".to_string())].into();
    assert_eq!(root(&m), Hash::of(b"a\0h1\nb\0h2\n").to_hex());
}

#[test]
fn key_files_are_private_and_never_shown() {
    let project = temp("p");
    let keys = KeyDir::at(temp("k").join("keys"));
    let k = keys.generate(&project).unwrap();
    let dir_mode = std::fs::metadata(keys.path()).unwrap().permissions().mode() & 0o777;
    assert_eq!(dir_mode, 0o700);
    let file = keys.path().join(format!("{}.ed25519", k.key_id()));
    assert_eq!(std::fs::metadata(&file).unwrap().permissions().mode() & 0o777, 0o600);
    let content = std::fs::read_to_string(&file).unwrap();
    let seed = content.trim_end().split(' ').nth(1).unwrap().to_string();
    assert!(content.starts_with(&format!("{SECRET_PREFIX} ")) && content.ends_with('\n') && canonical::is_hex64(&seed));
    assert_eq!(k.key_id().len(), 16);
    assert_eq!(k.key_id(), &Hash::of(k.public().as_bytes()).to_hex()[..16]);
    // Debug output shows only the key id.
    let shown = format!("{k:?}");
    assert!(shown.contains(k.key_id()) && !shown.contains(&seed), "{shown}");
    let loaded = keys.load(&project, k.key_id()).unwrap();
    assert_eq!(loaded.public(), k.public());
    assert_eq!(loaded.sign_hex(b"x"), k.sign_hex(b"x"));

    // A key file with other permissions, a symlink, a wrong content: E0920, naming only path and id.
    std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o644)).unwrap();
    let e = keys.load(&project, k.key_id()).unwrap_err();
    assert_eq!((e.code, e.kind), ("E0920", "key_unusable"));
    assert!(e.message.contains("644") && !e.message.contains(&seed));
    std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o600)).unwrap();
    let other = keys.generate(&project).unwrap();
    let link = keys.path().join("0000000000000000.ed25519");
    std::os::unix::fs::symlink(keys.path().join(format!("{}.ed25519", other.key_id())), &link).unwrap();
    assert!(keys.load(&project, "0000000000000000").unwrap_err().message.contains("symbolic link"));
    let renamed = keys.path().join("1111111111111111.ed25519");
    std::fs::copy(&file, &renamed).unwrap();
    std::fs::set_permissions(&renamed, std::fs::Permissions::from_mode(0o600)).unwrap();
    assert!(keys.load(&project, "1111111111111111").unwrap_err().message.contains("another key"));
    let garbage = keys.path().join("2222222222222222.ed25519");
    std::fs::write(&garbage, "hello world\n").unwrap();
    std::fs::set_permissions(&garbage, std::fs::Permissions::from_mode(0o600)).unwrap();
    assert!(keys.load(&project, "2222222222222222").unwrap_err().message.contains("not a QAL key file"));
    assert!(keys.load(&project, "3333333333333333").unwrap_err().message.contains("missing"));
    assert_eq!(keys.load(&project, "../../etc/passwd").unwrap_err().code, "E0920");

    // A key directory with group or other permissions.
    std::fs::set_permissions(keys.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
    let e = keys.load(&project, k.key_id()).unwrap_err();
    assert!(e.code == "E0920" && e.message.contains("755"), "{e:?}");
    std::fs::set_permissions(keys.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    // A key directory that is a symbolic link.
    let linked = KeyDir::at(temp("k").join("link"));
    std::os::unix::fs::symlink(keys.path(), linked.path()).unwrap();
    assert!(linked.load(&project, k.key_id()).unwrap_err().message.contains("symbolic link"));
    // A key directory inside the project.
    let inside = KeyDir::at(project.join("secrets"));
    assert_eq!(inside.generate(&project).unwrap_err().code, "E0926");
    assert_eq!(KeyDir::at(project.clone()).generate(&project).unwrap_err().code, "E0926");
}

#[test]
fn the_policy_matrix() {
    use Role::*;
    // Reads for everyone, anywhere.
    for tool in ["qal_view", "qal_query", "qal_history", "qal_whoami", "qal_verify", "qal_test", "qal_prove", "qal_run", "qal_call", "qal_request", "qal_build"] {
        for r in policy::ROLES {
            for sp in policy::SPACES {
                assert!(policy::allows(tool, None, r, sp), "{tool} {r:?} {sp:?}");
            }
        }
    }
    // Writes on main: nobody (main changes only by merge).
    for tool in ["qal_define", "qal_rename", "qal_delete"] {
        assert!(policy::allowed_roles(tool, None, Space::Main).is_empty(), "{tool}");
    }
    // In change sets: the matrix cells.
    assert_eq!(policy::allowed_roles("qal_define", None, Space::Own), [Architect, Implementer, Verifier]);
    assert_eq!(policy::allowed_roles("qal_define", None, Space::ForeignSubmitted), [Verifier]);
    assert!(policy::allowed_roles("qal_define", None, Space::ForeignOther).is_empty());
    assert_eq!(policy::allowed_roles("qal_rename", None, Space::Own), [Architect]);
    assert_eq!(policy::allowed_roles("qal_review", Some("approve"), Space::ForeignSubmitted), [Reviewer]);
    assert_eq!(policy::allowed_roles("qal_review", Some("verified"), Space::ForeignSubmitted), [Verifier]);
    assert_eq!(policy::allowed_roles("qal_merge", None, Space::ForeignSubmitted), [Architect, Reviewer]);
    assert_eq!(policy::allowed_roles("qal_change", Some("abandon"), Space::ForeignOther), [Architect]);
    assert_eq!(policy::allowed_roles("qal_lease", Some("break"), Space::Main), [Architect]);
    assert!(policy::rule("qal_lease", None, Architect).is_none(), "lease needs an action");
    // tools/list: only available tools; reviewers do not see writes.
    assert_eq!(policy::tools_for(Reviewer), ["qal_view", "qal_query", "qal_history", "qal_whoami", "qal_verify", "qal_test", "qal_prove", "qal_run", "qal_call", "qal_request", "qal_build", "qal_change", "qal_review", "qal_merge"]);
    assert!(policy::tools_for(Implementer).contains(&"qal_define") && !policy::tools_for(Implementer).contains(&"qal_rename"));
    assert_eq!(policy::tools_for(Architect).len(), 19);
    assert!(!policy::tools_for(Implementer).contains(&"qal_review") && policy::tools_for(Implementer).contains(&"qal_rebase"));
    // E0903 names the roles that may.
    let d = policy::forbidden("qal_define", None, Implementer, Space::Main);
    assert_eq!((d.code, d.kind, d.expected.as_deref(), d.found.as_deref()), ("E0903", "role_forbidden", Some("nobody"), Some("implementer")));
    assert!(d.hints[0].contains("change set"));
    let d = policy::forbidden("qal_rename", None, Reviewer, Space::Main);
    assert!(d.message.contains("reviewer") && d.hints[0].contains("architect"), "{d:?}");
    // Records.
    let ok = |op: &str, action: Option<&str>, role: &str, sp: Space| policy::record_allowed(op, action, role, sp);
    assert!(ok("define", None, "architect", Space::Own) && ok("define", None, "implementer", Space::Own) && !ok("define", None, "architect", Space::Main) && !ok("define", None, OWNER, Space::Own));
    assert!(ok("define", None, "verifier", Space::ForeignSubmitted) && !ok("define", None, "implementer", Space::ForeignSubmitted) && !ok("define", None, "reviewer", Space::Own));
    assert!(ok("change", Some("open"), "implementer", Space::Own) && !ok("change", Some("open"), "reviewer", Space::Own) && !ok("change", Some("list"), "architect", Space::Own));
    assert!(ok("change", Some("abandon"), "architect", Space::ForeignOther) && !ok("change", Some("abandon"), "implementer", Space::ForeignOther));
    assert!(ok("lease_break", None, "architect", Space::ForeignOther) && !ok("lease_break", None, "implementer", Space::ForeignOther));
    assert!(ok("member_add", None, OWNER, Space::Main) && !ok("member_add", None, "architect", Space::Main));
    assert!(!ok("frobnicate", None, OWNER, Space::Main));
    // Phase 4c: reviews by verdict, merges on main only, rebases by the owner, the merge's own records.
    assert!(ok("review", Some("approve"), "reviewer", Space::ForeignSubmitted) && !ok("review", Some("approve"), "verifier", Space::ForeignSubmitted) && !ok("review", None, "reviewer", Space::ForeignSubmitted));
    assert!(ok("review", Some("verified"), "verifier", Space::ForeignSubmitted) && ok("review", Some("comment"), "architect", Space::ForeignOther));
    assert!(ok("merge", None, "reviewer", Space::Main) && ok("merge", None, "architect", Space::Main) && !ok("merge", None, "reviewer", Space::Own) && !ok("merge", None, "implementer", Space::Main));
    assert!(ok("rebase", None, "implementer", Space::Own) && !ok("rebase", None, "implementer", Space::ForeignSubmitted) && !ok("rebase", None, "reviewer", Space::Own));
    assert!(ok("change", Some("merged"), "reviewer", Space::ForeignSubmitted) && ok("change", Some("needs_rebase"), "architect", Space::ForeignOther) && !ok("change", Some("merged"), "implementer", Space::Own));
}

#[test]
fn a_team_store_signs_every_write_and_verifies() {
    let (mut s, keys, owner, arch, imp) = team();
    let trust = Trust::parse(&s.team_trust().unwrap().to_string()).unwrap();
    assert_eq!(trust.owner, "boss");
    // genesis + 2 members + the architect's change set.
    let o = verified(&mut s, Some(&trust));
    assert!(o.ok, "{o:?}");
    assert_eq!((o.records, o.members, o.trust), (Some(4), Some(3), Some("file")));
    assert!(!o.warnings.iter().any(|w| w.contains("--trust")), "{o:?}");
    assert!(o.warnings.iter().any(|w| w.contains("--head")));
    let o = verified(&mut s, None);
    assert_eq!(o.trust, Some("store"));
    assert!(o.warnings[0].contains("--trust"));

    // Unsigned writes are refused in a team store.
    for e in [s.define("fn f() -> Int = 1", "cli").unwrap_err(), s.rename("base", "b2", "cli").unwrap_err(), s.delete(&["base".into()], "cli").unwrap_err()] {
        assert_eq!((e[0].code, e[0].expected.as_deref()), ("E0903", Some("architect")));
    }
    // An implementer cannot write in the architect's change set, nobody on main (records refuse it too).
    as_member(&mut s, imp);
    assert_eq!(s.define("fn f() -> Int = 1", "x").unwrap_err()[0].code, "E0903");
    assert_eq!(count(&s, "records"), 4);
    // The architect writes in its own change set; the history names the member, not the given agent.
    as_member(&mut s, arch);
    let d = s.define("fn f() -> Int = base() + 1\ntest f_works = f() == 2", "ignored").unwrap();
    assert_eq!(d.len(), 2);
    assert_eq!(s.history("f").unwrap()[0].agent, "arch-1");
    s.define("fn base() -> Int = 2", "x").unwrap(); // f relinked
    s.rename("f", "g", "x").unwrap();
    s.define("fn tmp() -> Int = 0", "x").unwrap();
    s.delete(&["tmp".into()], "x").unwrap();
    assert_eq!(count(&s, "records"), 9);
    // Every history row of the team era points to its record.
    let unlinked: i64 = raw(&s).query_row("SELECT count(*) FROM history WHERE record IS NULL AND agent != 'pre-team' AND agent NOT LIKE 'qal-std@%'", [], |r| r.get(0)).unwrap();
    assert_eq!(unlinked, 0);
    let pre: Option<i64> = raw(&s).query_row("SELECT record FROM history WHERE agent = 'pre-team'", [], |r| r.get(0)).unwrap();
    assert_eq!(pre, None);
    let o = verified(&mut s, Some(&trust));
    assert!(o.ok, "{o:?}");
    assert_eq!(o.records, Some(9));
    // The record's body: op, member, entries.
    let body: Vec<u8> = raw(&s).query_row("SELECT body FROM records WHERE seq = 7", [], |r| r.get(0)).unwrap();
    let v: Value = serde_json::from_slice(&body[RECORD_PREFIX.len()..]).unwrap();
    assert_eq!((v["op"].as_str(), v["member"].as_str(), v["space"].as_str()), (Some("rename"), Some("arch-1"), Some(s.space().id())));
    assert_eq!(v["args"]["entries"][0]["action"], "renamed");
    assert_eq!(v["args"]["entries"][0]["new"], Value::Null);
    // The users of the renamed name now refer to the new name: an `updated` entry with its refs.
    assert_eq!((v["args"]["entries"][1]["name"].as_str(), v["args"]["entries"][1]["action"].as_str()), (Some("f_works"), Some("updated")));
    assert_eq!(v["args"]["entries"][1]["refs"], json!(["g"]));
    assert_eq!((v["args"]["entries"][2]["name"].as_str(), v["args"]["entries"][2]["action"].as_str()), (Some("g"), Some("added")));

    // A foreign owner key in the trust file.
    let other = Trust { owner: "boss".into(), owner_key: keys::public_hex(&KeyDir::at(temp("k2").join("k")).generate(s.project_dir().unwrap()).unwrap().public()) };
    assert_eq!(code(&verified(&mut s, Some(&other))), "E0925");
    let wrong_owner = Trust { owner: "eve".into(), owner_key: trust.owner_key.clone() };
    assert_eq!(code(&verified(&mut s, Some(&wrong_owner))), "E0925");

    // Rekey: later records need the new key; revoke: the member's sessions end.
    let rekeyed = s.team_rekey(&owner, "arch-1", &keys).unwrap();
    assert_ne!(rekeyed.key_id, "");
    assert_eq!(s.define("fn h() -> Int = 3", "x").unwrap_err()[0].code, "E0920");
    let project = s.project_dir().unwrap().to_path_buf();
    as_member(&mut s, Signer { member: "arch-1".into(), session: "s1".into(), key: keys.load(&project, &rekeyed.key_id).unwrap() });
    s.define("fn h() -> Int = 3", "x").unwrap();
    let sess = s.team_session("arch-1", DEFAULT_SESSION_TTL_MS).unwrap();
    assert!(s.resolve_session(&token_hash(&sess.token)).is_ok());
    s.team_revoke(&owner, "arch-1").unwrap();
    assert_eq!(s.resolve_session(&token_hash(&sess.token)).unwrap_err().code, "E0902");
    assert_eq!(s.define("fn i() -> Int = 3", "x").unwrap_err()[0].code, "E0902");
    let o = verified(&mut s, Some(&trust));
    assert!(o.ok, "{o:?}");
    assert_eq!(o.records, Some(12));
    // Owner changes are refused; double init too.
    assert_eq!(s.team_rekey(&owner, "boss", &keys).unwrap_err()[0].code, "E0908");
    assert_eq!(s.team_init("boss", &keys).unwrap_err()[0].code, "E0908");
    assert_eq!(s.team_add(&owner, "impl-1", Role::Verifier, &keys).unwrap_err()[0].code, "E0908");
    assert_eq!(s.team_add(&owner, "Bad_Id", Role::Verifier, &keys).unwrap_err()[0].code, "E0701");
}

fn count(s: &Store, table: &str) -> i64 {
    raw(s).query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0)).unwrap()
}

/// A team store with a few signed writes, for tampering; also the architect's and owner's signers.
fn signed_store() -> (Store, Trust, Signer, Signer) {
    let (mut s, keys, owner, arch, _imp) = team();
    let project = s.project_dir().unwrap().to_path_buf();
    let again = Signer { member: arch.member.clone(), session: arch.session.clone(), key: keys.load(&project, arch.key.key_id()).unwrap() };
    as_member(&mut s, arch);
    s.define("fn f() -> Int = base() + 1", "x").unwrap();
    s.define("fn g() -> Int = f() * 2", "x").unwrap();
    let trust = Trust::parse(&s.team_trust().unwrap().to_string()).unwrap();
    assert!(verified(&mut s, Some(&trust)).ok);
    (s, trust, again, owner)
}

/// Flips one byte (`at`) of a BLOB or TEXT column of a row.
fn flip(s: &Store, table: &str, column: &str, key: &str, at: usize) {
    let c = raw(s);
    c.execute_batch("PRAGMA foreign_keys = OFF;").unwrap();
    let mut bytes: Vec<u8> = c.query_row(&format!("SELECT CAST({column} AS BLOB) FROM {table} WHERE {key}"), [], |r| r.get(0)).unwrap();
    let i = at.min(bytes.len() - 1);
    // Keep text columns valid UTF-8 hex: swap one hex digit for another.
    bytes[i] = match bytes[i] {
        b'0' => b'1',
        b'a' => b'b',
        // `f` ^ 1 would be `g`, which is not a hex digit (the test then hit another check).
        b'f' => b'e',
        b => b ^ 0x01,
    };
    let is_blob: bool = c.query_row(&format!("SELECT typeof({column}) = 'blob' FROM {table} WHERE {key}"), [], |r| r.get(0)).unwrap();
    if is_blob {
        c.execute(&format!("UPDATE {table} SET {column} = ?1 WHERE {key}"), [bytes]).unwrap();
    } else {
        c.execute(&format!("UPDATE {table} SET {column} = ?1 WHERE {key}"), [String::from_utf8(bytes).unwrap()]).unwrap();
    }
}

#[test]
fn tampering_with_records_is_detected() {
    // One byte in a record body (anywhere): the hash no longer matches.
    for at in [0, 20, 200, 10_000] {
        let (mut s, trust, _, _) = signed_store();
        flip(&s, "records", "body", "seq = 4", at);
        let o = verified(&mut s, Some(&trust));
        assert_eq!((code(&o), o.error.as_ref().unwrap().seq, o.checked), ("E0922", 4, Some(3)), "{o:?}");
    }
    // A body and hash rewritten consistently (but not re-signed): the signature fails.
    let (mut s, trust, _, _) = signed_store();
    let c = raw(&s);
    let body: Vec<u8> = c.query_row("SELECT body FROM records WHERE seq = 5", [], |r| r.get(0)).unwrap();
    let text = String::from_utf8(body).unwrap().replace("\"session\":\"s000000000001\"", "\"session\":\"s000000000009\"");
    c.execute("UPDATE records SET body = ?1, hash = ?2 WHERE seq = 5", rusqlite::params![text.as_bytes(), Hash::of(text.as_bytes()).to_hex()]).unwrap();
    drop(c);
    assert_eq!(code(&verified(&mut s, Some(&trust))), "E0921", "consistent hash, but not signed");
    // One byte in a signature.
    for seq in [1, 3, 5] {
        let (mut s, trust, _, _) = signed_store();
        flip(&s, "records", "sig", &format!("seq = {seq}"), 7);
        let o = verified(&mut s, Some(&trust));
        assert_eq!((code(&o), o.error.as_ref().unwrap().seq), ("E0921", seq), "{o:?}");
    }
    // A removed record.
    let (mut s, trust, _, _) = signed_store();
    raw(&s).execute("DELETE FROM records WHERE seq = 3", []).unwrap();
    assert_eq!(code(&verified(&mut s, Some(&trust))), "E0922");
    // The last record removed: the chain is fine, the names are not.
    let (mut s, trust, _, _) = signed_store();
    raw(&s).execute("DELETE FROM records WHERE seq = 6", []).unwrap();
    assert_eq!(code(&verified(&mut s, Some(&trust))), "E0923");
    // Two records swapped.
    let (mut s, trust, _, _) = signed_store();
    raw(&s).execute_batch("UPDATE records SET seq = -1 WHERE seq = 4; UPDATE records SET seq = 4 WHERE seq = 5; UPDATE records SET seq = 5 WHERE seq = -1;").unwrap();
    let o = verified(&mut s, Some(&trust));
    assert_eq!((code(&o), o.error.as_ref().unwrap().seq), ("E0922", 4), "{o:?}");
}

#[test]
fn tampering_with_names_and_definitions_is_detected() {
    // A name pointing to another hash: on main and in the change set.
    let (mut s, trust, _, _) = signed_store();
    flip(&s, "names", "hash", "name = 'base'", 5);
    assert_eq!(code(&verified(&mut s, Some(&trust))), "E0923");
    let (mut s, trust, _, _) = signed_store();
    flip(&s, "change_names", "hash", "name = 'f'", 5);
    assert_eq!(code(&verified(&mut s, Some(&trust))), "E0923");
    // A name moved to another stored definition (as a valid hash).
    let (mut s, trust, _, _) = signed_store();
    raw(&s).execute("UPDATE change_names SET hash = (SELECT hash FROM names WHERE name = 'base') WHERE name = 'g'", []).unwrap();
    assert_eq!(code(&verified(&mut s, Some(&trust))), "E0923");
    // A name added or removed behind the records' back.
    let (mut s, trust, _, _) = signed_store();
    raw(&s).execute("DELETE FROM change_names WHERE name = 'g'", []).unwrap();
    assert_eq!(code(&verified(&mut s, Some(&trust))), "E0923");
    let (mut s, trust, _, _) = signed_store();
    raw(&s).execute("DELETE FROM names WHERE name = 'base'", []).unwrap();
    assert_eq!(code(&verified(&mut s, Some(&trust))), "E0923");
    // A change set's status or scope changed behind the records' back.
    let (mut s, trust, _, _) = signed_store();
    raw(&s).execute("UPDATE changes SET status = 'submitted'", []).unwrap();
    assert_eq!(code(&verified(&mut s, Some(&trust))), "E0923");
    let (mut s, trust, _, _) = signed_store();
    raw(&s).execute("UPDATE changes SET scope = '[\"f\"]'", []).unwrap();
    assert_eq!(code(&verified(&mut s, Some(&trust))), "E0923");
    // A row in `change_names` no record made.
    let (mut s, trust, _, _) = signed_store();
    raw(&s).execute("INSERT INTO change_names (change, name, hash, local_names, refs, origin, updated_at) SELECT 'c000000000001', name, hash, local_names, refs, 'edited', 0 FROM names WHERE name = 'base'", []).unwrap();
    assert_eq!(code(&verified(&mut s, Some(&trust))), "E0923");
    // A change_names row changed by one byte (same detection).
    // One byte in a definition's canonical form.
    let (mut s, trust, _, _) = signed_store();
    let h: String = raw(&s).query_row("SELECT hash FROM change_names WHERE name = 'g'", [], |r| r.get(0)).unwrap();
    flip(&s, "defs", "canonical", &format!("hash = '{h}'"), 8);
    let o = verified(&mut s, Some(&trust));
    assert!(code(&o) == "E0923" && o.error.as_ref().unwrap().message.contains("does not match its hash"), "{o:?}");
    // A member's role raised in the table.
    let (mut s, trust, _, _) = signed_store();
    raw(&s).execute("UPDATE members SET role = 'architect' WHERE id = 'impl-1'", []).unwrap();
    assert_eq!(code(&verified(&mut s, Some(&trust))), "E0923");
}

#[test]
fn records_of_revoked_members_are_unauthorized() {
    let (mut s, trust, arch, owner) = signed_store();
    s.team_revoke(&owner, "arch-1").unwrap();
    assert!(verified(&mut s, Some(&trust)).ok);
    // A well-formed, correctly chained record signed with the revoked member's key.
    let c = raw(&s);
    let (seq, prev): (i64, String) = c.query_row("SELECT seq, hash FROM records ORDER BY seq DESC LIMIT 1", [], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
    let names = main_names(&c).unwrap();
    let body = json!({"v": 1, "seq": seq + 1, "prev": prev, "at": 1, "member": "arch-1", "key": arch.key.key_id(), "session": "s1", "op": "define", "space": "main",
                      "args": {"entries": []}, "root_before": root(&names), "root_after": root(&names), "names_after": names_digest(&c, &main_rows(&c).unwrap()).unwrap()});
    let bytes = record_bytes(&body).unwrap();
    c.execute("INSERT INTO records (seq, hash, body, sig) VALUES (?1, ?2, ?3, ?4)", rusqlite::params![seq + 1, Hash::of(&bytes).to_hex(), bytes, arch.key.sign_hex(&bytes)]).unwrap();
    drop(c);
    let o = verified(&mut s, Some(&trust));
    assert_eq!((code(&o), o.error.as_ref().unwrap().seq), ("E0924", seq + 1), "{o:?}");
    // Signed by the implementer for an op it may not make: also E0924; with a wrong key: E0921.
}

#[test]
fn sessions_store_only_the_token_hash() {
    let (mut s, keys, owner, _arch, _imp) = team();
    let sess = s.team_session("impl-1", 3_600_000).unwrap();
    assert!(sess.token.starts_with("qs_") && sess.token.len() == 67 && sess.id.starts_with('s') && sess.id.len() == 13);
    assert!(!format!("{sess:?}").contains(sess.token.as_str()));
    let who = s.resolve_session(&token_hash(&sess.token)).unwrap();
    assert_eq!((who.member.as_str(), who.role.as_str(), who.session.as_str()), ("impl-1", "implementer", sess.id.as_str()));
    // The token itself is nowhere in the database.
    let db = std::fs::read(s.path()).unwrap();
    let wal = std::fs::read(s.path().with_extension("db-wal")).unwrap_or_default();
    for bytes in [&db, &wal] {
        assert!(!contains(bytes, sess.token.as_bytes()));
    }
    // Unknown, expired, revoked.
    assert_eq!(s.resolve_session(&token_hash("qs_nope")).unwrap_err().code, "E0902");
    raw(&s).execute("UPDATE sessions SET expires_at = 1 WHERE id = ?1", [&sess.id]).unwrap();
    assert!(s.resolve_session(&token_hash(&sess.token)).unwrap_err().message.contains("expired"));
    let s2 = s.team_session("impl-1", 3_600_000).unwrap();
    s.team_revoke(&owner, "impl-1").unwrap();
    assert!(s.resolve_session(&token_hash(&s2.token)).unwrap_err().message.contains("revoked"));
    // Owners have no sessions; TTL limits.
    assert_eq!(s.team_session("boss", 3_600_000).unwrap_err()[0].code, "E0908");
    assert_eq!(s.team_session("arch-1", MAX_SESSION_TTL_MS + 1).unwrap_err()[0].code, "E0914");
    assert_eq!((parse_ttl("24h"), parse_ttl("30m"), parse_ttl("7d"), parse_ttl("90"), parse_ttl("x"), parse_ttl("0h")), (Some(86_400_000), Some(1_800_000), Some(604_800_000), Some(90_000), None, None));
    // Secrets: no seed of any key file and no key-file line in the store.
    let db = [std::fs::read(s.path()).unwrap(), std::fs::read(s.path().with_extension("db-wal")).unwrap_or_default()].concat();
    let mut seeds = 0;
    for e in std::fs::read_dir(keys.path()).unwrap().flatten() {
        let text = std::fs::read_to_string(e.path()).unwrap();
        let seed = text.trim_end().split(' ').nth(1).unwrap().to_string();
        assert!(!contains(&db, seed.as_bytes()), "a seed is in the store");
        seeds += 1;
    }
    assert_eq!(seeds, 3);
    assert!(!contains(&db, SECRET_PREFIX.as_bytes()));
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

#[test]
fn notices_are_capped_and_taken_once() {
    let (s, _keys, _owner, _arch, _imp) = team();
    let log = s.team_log().unwrap();
    assert!(!s.has_notices(&log, "impl-1").unwrap());
    for i in 0..1005 {
        s.notify("impl-1", "merged", &json!({"n": i})).unwrap();
    }
    assert_eq!(count(&s, "notices"), 1000);
    let (first, more) = s.take_notices(&log, "impl-1", NOTICES_PER_REPLY).unwrap();
    assert_eq!((first.len(), more), (10, 990));
    assert_eq!(first[0].body["n"], 5, "the oldest went");
    assert_eq!(serde_json::to_value(&first[0]).unwrap()["kind"], "merged");
    let (next, _) = s.take_notices(&log, "impl-1", 1).unwrap();
    assert_eq!(next[0].body["n"], 15);
    s.notify_once("arch-1", "std_outdated", &json!({"embedded": "x"})).unwrap();
    s.notify_once("arch-1", "std_outdated", &json!({"embedded": "x"})).unwrap();
    assert_eq!(s.take_notices(&log, "arch-1", 10).unwrap().0.len(), 1);
    assert!(!s.has_notices(&log, "arch-1").unwrap());
}

#[test]
fn a_team_store_does_not_import_std_automatically() {
    let (mut s, _keys, _o, _a, _i) = team();
    let pkg = "# qal-pkg 1\n# package std ".to_string();
    // Any std package: ensure_std leaves a team store alone.
    let body = "";
    let file = format!("{pkg}{}\n{body}", Hash::of(body.as_bytes()).to_hex());
    assert!(s.ensure_std(&file, None).unwrap().is_none());
    assert_eq!(s.meta(crate::package::META_STD_PACKAGE).unwrap(), None);
}

#[test]
fn stores_without_team_mode_write_no_records() {
    let (mut s, _keys) = setup();
    s.define("fn f() -> Int = 1", "a").unwrap();
    s.rename("f", "g", "a").unwrap();
    s.delete(&["g".into()], "a").unwrap();
    assert_eq!(count(&s, "records"), 0);
    assert!(!s.is_team().unwrap());
    assert_eq!(s.verify(None, 1, None).unwrap_err()[0].code, "E0900");
    let r: Option<i64> = raw(&s).query_row("SELECT max(record) FROM history", [], |r| r.get(0)).unwrap();
    assert_eq!(r, None);
}

#[test]
fn version_5_stores_are_upgraded_to_6() {
    let (mut s, _keys) = setup();
    s.define("fn f() -> Int = 1", "a").unwrap();
    let before = s.query(&crate::Query::default()).unwrap();
    let c = raw(&s);
    // The version-5 layout: no team tables, history without `space` and `record`.
    let mut drops = String::new();
    for t in ["members", "sessions", "changes", "change_names", "reviews", "leases", "seen", "notices", "records"] {
        drops.push_str(&format!("DROP TABLE {t}; "));
    }
    c.execute_batch(&drops).unwrap();
    c.execute_batch(
        "CREATE TABLE h5 (id INTEGER PRIMARY KEY, name TEXT NOT NULL, action TEXT NOT NULL CHECK (action IN ('added', 'replaced', 'relinked', 'renamed', 'deleted')), \
         old_hash TEXT, new_hash TEXT NOT NULL, from_name TEXT, to_name TEXT, agent TEXT NOT NULL, at INTEGER NOT NULL); \
         INSERT INTO h5 SELECT id, name, action, old_hash, new_hash, from_name, to_name, agent, at FROM history; DROP TABLE history; ALTER TABLE h5 RENAME TO history; \
         CREATE INDEX history_by_name ON history(name); PRAGMA user_version = 5;",
    )
    .unwrap();
    drop(c);
    let path = s.path().to_path_buf();
    drop(s);
    let mut s = Store::open(&path).unwrap();
    assert_eq!(raw(&s).query_row("PRAGMA user_version", [], |r| r.get::<_, i64>(0)).unwrap(), 6);
    assert_eq!(s.query(&crate::Query::default()).unwrap(), before, "same names and hashes");
    assert_eq!(count(&s, "records"), 0);
    let space: String = raw(&s).query_row("SELECT space FROM history WHERE name = 'f'", [], |r| r.get(0)).unwrap();
    assert_eq!(space, "main");
    assert!(!s.is_team().unwrap());
    s.define("fn g() -> Int = f()", "a").unwrap();
    assert_eq!(s.history("g").unwrap()[0].agent, "a");
}

#[test]
fn a_team_store_imports_only_std_signed_by_the_owner() {
    // Another project's package: refused in a team store, even for the owner.
    let (mut lib, _) = setup();
    lib.define("fn lib_one() -> Int = 1", "lib").unwrap();
    let pkg = lib.export("lib_", "lib").unwrap().file;
    // A standard library package (from a std store).
    let std_dir = temp("std");
    let mut stdlib = Store::init_std(&std_dir).unwrap();
    stdlib.define("fn std.a.one() -> Int = 1", "std-a").unwrap();
    let std_pkg = stdlib.export("std.", "std").unwrap().file;
    let (mut s, _keys, owner, arch, _imp) = team();
    s.set_release(&std_pkg).unwrap();
    as_member(&mut s, arch);
    assert_eq!(s.import(&std_pkg).unwrap_err()[0].code, "E0903", "architects make no import records");
    as_member(&mut s, owner);
    assert!(s.import(&pkg).unwrap_err()[0].message.contains("only the standard library"));
    let i = s.import(&std_pkg).unwrap();
    assert_eq!(i.defined.len(), 1);
    let o = verified(&mut s, None);
    assert!(o.ok && o.records == Some(5), "{o:?}");
    let body: Vec<u8> = raw(&s).query_row("SELECT body FROM records WHERE seq = 5", [], |r| r.get(0)).unwrap();
    let v: Value = serde_json::from_slice(&body[RECORD_PREFIX.len()..]).unwrap();
    assert_eq!((v["op"].as_str(), v["member"].as_str(), v["session"].as_str()), (Some("import"), Some("boss"), Some(CLI_SESSION)));
    assert_eq!((v["args"]["package"].as_str(), v["args"]["hash"].as_str()), (Some("std"), Some(i.hash.as_str())));
    assert_eq!(v["args"]["entries"][0]["name"], "std.a.one");
    // The history keeps the package's agent (later imports of the same package may replace it).
    s.set_space(crate::Space::Main);
    assert!(s.history("std.a.one").unwrap()[0].agent.starts_with("qal-std@"));
}

/// A signed store with a plain function, a user of it and a mutually recursive group.
fn group_store() -> (Store, Trust, Signer, Signer) {
    let (mut s, trust, arch, owner) = signed_store();
    s.define("fn is_even(n: Int) -> Bool = if n == 0 then true else is_odd(n - 1)\nfn is_odd(n: Int) -> Bool = if n == 0 then false else is_even(n - 1)", "x").unwrap();
    s.define("fn twice(x: Int) -> Int = f() + x + x", "x").unwrap();
    assert!(verified(&mut s, Some(&trust)).ok);
    (s, trust, arch, owner)
}

/// Review I1: every column that carries code or how it reads is bound by `qal verify`.
#[test]
fn tampering_with_any_code_bearing_column_is_detected() {
    let probes: &[(&str, &str)] = &[
        ("names.refs", "UPDATE change_names SET refs = '[]' WHERE name = 'twice'"),
        ("names.refs of a group member", "UPDATE change_names SET refs = '[\"is_even\",\"is_odd\"]' WHERE name = 'is_odd'"),
        ("names.refs re-spelled", "UPDATE change_names SET refs = '[ \"f\" ]' WHERE name = 'twice'"),
        ("names.local_names", "UPDATE change_names SET local_names = '[\"y\"]' WHERE name = 'twice'"),
        ("defs.signature", "UPDATE defs SET signature = 'fn(Int) -> Str' WHERE hash = (SELECT hash FROM change_names WHERE name = 'twice')"),
        ("defs.effects", "UPDATE defs SET effects = '[\"io\"]' WHERE hash = (SELECT hash FROM change_names WHERE name = 'twice')"),
        ("defs.kind", "UPDATE defs SET kind = 'test' WHERE hash = (SELECT hash FROM change_names WHERE name = 'twice')"),
        ("defs.local_names", "UPDATE defs SET local_names = '[\"z\"]' WHERE hash = (SELECT hash FROM change_names WHERE name = 'twice')"),
        ("groups.form", "UPDATE groups SET form = CAST(form AS BLOB) || x'00'"),
        ("groups deleted", "DELETE FROM groups"),
        ("deps deleted", "DELETE FROM deps"),
        ("one deps row deleted", "DELETE FROM deps WHERE hash = (SELECT hash FROM change_names WHERE name = 'twice')"),
        ("a deps row added", "INSERT INTO deps (hash, dep_hash) SELECT n.hash, b.hash FROM change_names n, change_names b WHERE n.name = 'f' AND b.name = 'twice'"),
        ("names.refs on main", "UPDATE names SET refs = '[\"x\"]' WHERE name = 'base'"),
        ("meta.team_allow", "UPDATE meta SET value = 'io,time,rand,db:mem,fs:.' WHERE key = 'team_allow'"),
        ("meta.owner", "UPDATE meta SET value = 'arch-1' WHERE key = 'owner'"),
    ];
    for (what, sql) in probes {
        let (mut s, trust, _, _) = group_store();
        let c = raw(&s);
        c.execute_batch(&format!("PRAGMA foreign_keys = OFF; {sql};")).unwrap();
        drop(c);
        let o = verified(&mut s, Some(&trust));
        assert!(!o.ok, "{what}: not detected");
        assert_eq!(code(&o), "E0923", "{what}: {o:?}");
    }
    // `meta.team` removed: still reported as tampering, not as "not a team store".
    let (mut s, trust, _, _) = group_store();
    raw(&s).execute("DELETE FROM meta WHERE key = 'team'", []).unwrap();
    let o = verified(&mut s, Some(&trust));
    assert_eq!(code(&o), "E0923", "{o:?}");
}

#[test]
fn signatures_have_one_spelling_and_from_says_what_it_skipped() {
    let (mut s, trust, _, _) = signed_store();
    raw(&s).execute("UPDATE records SET sig = upper(sig) WHERE seq = 2", []).unwrap();
    let o = verified(&mut s, Some(&trust));
    assert_eq!((code(&o), o.error.as_ref().unwrap().seq), ("E0921", 2), "{o:?}");
    let (mut s, trust, _, _) = signed_store();
    let o = s.verify(Some(&trust), 4, None).unwrap();
    assert!(o.ok && o.signatures_from == Some(4), "{o:?}");
    assert!(o.warnings.iter().any(|w| w.contains("only from record 4")));
    assert_eq!(verified(&mut s, Some(&trust)).signatures_from, None);
    // A bad signature before `from` is not checked (and the result says so).
    flip(&s, "records", "sig", "seq = 2", 9);
    let o3 = s.verify(Some(&trust), 3, None).unwrap();
    assert!(o3.ok, "{o3:?}");
    assert_eq!(code(&verified(&mut s, Some(&trust))), "E0921");
}

#[test]
fn a_pinned_head_finds_a_cut_off_tail() {
    let (mut s, trust, _, _) = signed_store();
    let head = verified(&mut s, Some(&trust)).head.unwrap();
    assert!(s.verify(Some(&trust), 1, Some(&head)).unwrap().ok);
    // The last record and its change removed together: consistent without the pin.
    let c = raw(&s);
    c.execute_batch("PRAGMA foreign_keys = OFF; DELETE FROM records WHERE seq = (SELECT max(seq) FROM records); DELETE FROM change_names WHERE name = 'g';").unwrap();
    drop(c);
    let o = verified(&mut s, Some(&trust));
    assert!(o.ok && o.warnings.iter().any(|w| w.contains("--head")), "{o:?}");
    let o = s.verify(Some(&trust), 1, Some(&head)).unwrap();
    assert_eq!(code(&o), "E0922", "{o:?}");
}

#[test]
fn a_record_over_one_mib_is_refused_and_nothing_is_written() {
    let (mut s, _keys, _owner, arch, _imp) = team();
    as_member(&mut s, arch);
    let src: String = (0..8000).map(|i| format!("fn generated_function_number_{i}() -> Int = {i}\n")).collect();
    let before = count(&s, "names");
    let e = s.define(&src, "x").unwrap_err();
    assert_eq!((e[0].code, e[0].kind), ("E0914", "team_limit"), "{e:?}");
    assert_eq!((count(&s, "names"), count(&s, "records")), (before, 4));
    assert!(verified(&mut s, None).ok);
}

/// Review I2: `qal verify` over 100 000 records (spec §14: ≤ 10 s in release). Run with
/// `cargo test --release -p qal-store verify_100k -- --ignored --nocapture`.
#[test]
#[ignore]
fn verify_100k_records() {
    let (mut s, _keys, _owner, arch, _imp) = team();
    let src: String = (0..300).map(|i| format!("fn n{i}() -> Int = {i}\n")).collect();
    as_member(&mut s, arch);
    s.define(&src, "x").unwrap();
    s.define("fn x() -> Int = 2", "x").unwrap();
    let h2 = main_names(&raw(&s)).unwrap()["x"].clone();
    s.define("fn x() -> Int = 1", "x").unwrap();
    let c = raw(&s);
    let rows1 = main_rows(&c).unwrap();
    let h1 = rows1["x"].hash.clone();
    let mut rows2 = rows1.clone();
    rows2.get_mut("x").unwrap().hash = h2.clone();
    let (r1, r2) = (root_of(&rows1), root_of(&rows2));
    let (d1, d2) = (names_digest(&c, &rows1).unwrap(), names_digest(&c, &rows2).unwrap());
    let signer = match &s.signer {
        Some(sg) => sg.clone(),
        None => panic!("no signer"),
    };
    let (mut seq, mut prev): (i64, String) = c.query_row("SELECT seq, hash FROM records ORDER BY seq DESC LIMIT 1", [], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
    let start_seq = seq;
    let total = 100_000 - start_seq;
    // Pairs of records flipping `x` to h2 and back: the store ends as it is now.
    c.execute_batch("BEGIN").unwrap();
    for i in 0..total {
        seq += 1;
        let (old, new, rb, ra, d) = if i % 2 == 0 { (&h1, &h2, &r1, &r2, &d2) } else { (&h2, &h1, &r2, &r1, &d1) };
        let entry = json!({"name": "x", "old": old, "new": new, "action": "replaced", "refs": [], "locals": []});
        let body = json!({"v": 1, "seq": seq, "prev": prev, "at": 1, "member": "arch-1", "key": signer.key.key_id(), "session": "s1", "op": "define", "space": "main",
                          "args": {"entries": [entry]}, "root_before": rb, "root_after": ra, "names_after": d});
        let bytes = record_bytes(&body).unwrap();
        let hash = Hash::of(&bytes).to_hex();
        c.execute("INSERT INTO records (seq, hash, body, sig) VALUES (?1, ?2, ?3, ?4)", rusqlite::params![seq, hash, bytes, signer.key.sign_hex(&bytes)]).unwrap();
        prev = hash;
    }
    c.execute_batch("COMMIT").unwrap();
    assert_eq!(total % 2, 0, "an odd number of flips would end on h2");
    drop(c);
    let t = std::time::Instant::now();
    let o = verified(&mut s, None);
    let full = t.elapsed();
    let t = std::time::Instant::now();
    let skipped = s.verify(None, i64::MAX, None).unwrap();
    let without = t.elapsed();
    eprintln!("verify of {:?} records: {full:?} (signatures skipped: {without:?})", o.records);
    assert!(o.ok && skipped.ok, "{o:?}");
    assert_eq!(o.records, Some(100_000));
    if !cfg!(debug_assertions) {
        assert!(full < std::time::Duration::from_secs(10), "{full:?}");
    }
}
/// [`bare_team`] plus a verifier `ver-1` and a second implementer `impl-2`, with signers.
fn full_team() -> (Store, KeyDir, Signer, Signer, Signer, Signer, Signer) {
    let (mut s, keys, owner, arch, imp) = bare_team();
    let v = s.team_add(&owner, "ver-1", Role::Verifier, &keys).unwrap();
    let i2 = s.team_add(&owner, "impl-2", Role::Implementer, &keys).unwrap();
    let project = s.project_dir().unwrap().to_path_buf();
    let ver = Signer { member: "ver-1".into(), session: "s000000000003".into(), key: keys.load(&project, &v.key_id).unwrap() };
    let imp2 = Signer { member: "impl-2".into(), session: "s000000000004".into(), key: keys.load(&project, &i2.key_id).unwrap() };
    (s, keys, owner, arch, imp, ver, imp2)
}

fn again(keys: &KeyDir, s: &Store, x: &Signer) -> Signer {
    Signer { member: x.member.clone(), session: x.session.clone(), key: keys.load(s.project_dir().unwrap(), x.key.key_id()).unwrap() }
}

fn open(s: &mut Store, who: &Signer, title: &str, scope: &[&str], assignee: Option<&str>) -> String {
    let a = change::OpenArgs { title: title.into(), scope: scope.iter().map(|x| x.to_string()).collect(), assignee: assignee.map(str::to_string), ..Default::default() };
    s.change_open(who, &a).unwrap()["change"].as_str().unwrap().to_string()
}

fn write_in(s: &mut Store, keys: &KeyDir, who: &Signer, change: &str, src: &str) -> Result<Vec<crate::Defined>, Vec<qal_core::diag::Diagnostic>> {
    let signer = again(keys, s, who);
    s.set_signer(Some(Arc::new(signer)));
    s.set_space(crate::Space::Change(change.to_string()));
    s.define(src, "x")
}

#[test]
fn change_sets_have_their_own_checked_space() {
    let (mut s, keys, _owner, arch, imp, ver, _imp2) = full_team();
    let trust = Trust::parse(&s.team_trust().unwrap().to_string()).unwrap();
    let c = open(&mut s, &arch, "skeleton", &[], None);
    // A copy of main: `base` is there, origin `base`.
    let d = write_in(&mut s, &keys, &arch, &c, "type Order = { qty: Int }\nfn make(q: Int) -> Order req q > 0 ens r.qty == q = todo()\nfn total(q: Int) -> Int req q > 0 = make(q).qty + base()").unwrap();
    assert_eq!(d.len(), 3);
    assert_eq!(s.last_write().stubs, vec!["make".to_string()]);
    assert_eq!(s.last_write().leases.iter().map(|l| l.name.as_str()).collect::<Vec<_>>(), ["Order", "make", "total"]);
    // Main is untouched; the change set sees base and its own names.
    s.set_space(crate::Space::Main);
    assert!(s.view(&["make".into()]).is_err());
    assert_eq!(s.query(&crate::Query::default()).unwrap().len(), 1);
    s.set_space(crate::Space::Change(c.clone()));
    assert_eq!(s.query(&crate::Query::default()).unwrap().len(), 4);
    assert_eq!(s.history("make").unwrap()[0].agent, "arch-1");
    let diff = s.change_diff(&c).unwrap();
    assert_eq!(diff["diff"].as_array().unwrap().len(), 3);
    assert_eq!(diff["diff"][0]["origin"], "added");
    // Replacing `base` relinks `total` in the change set only.
    write_in(&mut s, &keys, &arch, &c, "fn base() -> Int = 5").unwrap();
    let origin = |s: &Store, n: &str| -> String { raw(s).query_row("SELECT origin FROM change_names WHERE change = ?1 AND name = ?2", [&c, n], |r| r.get(0)).unwrap() };
    assert_eq!((origin(&s, "base").as_str(), origin(&s, "total").as_str()), ("edited", "added"));
    // Rename and delete in the change set.
    write_in(&mut s, &keys, &arch, &c, "fn tmp() -> Int = 1").unwrap();
    s.rename("tmp", "tmp2", "x").unwrap();
    s.delete(&["tmp2".into()], "x").unwrap();
    s.rename("base", "base0", "x").unwrap();
    let h: Option<String> = raw(&s).query_row("SELECT hash FROM change_names WHERE change = ?1 AND name = 'base'", [&c], |r| r.get(0)).unwrap();
    assert_eq!((h, origin(&s, "base").as_str(), origin(&s, "base0").as_str()), (None, "renamed", "added"));
    s.set_space(crate::Space::Main);
    assert_eq!(s.query(&crate::Query::default()).unwrap()[0].name, "base", "main keeps its names");
    // Submit; a write of the owner reopens it.
    s.change_submit(&arch, &c).unwrap();
    assert_eq!(s.change(&c).unwrap().unwrap().status, "submitted");
    assert!(s.has_notices(&s.team_log().unwrap(), "ver-1").unwrap());
    write_in(&mut s, &keys, &arch, &c, "fn total(q: Int) -> Int req q > 0 = make(q).qty + base0() + 1").unwrap();
    assert_eq!(s.change(&c).unwrap().unwrap().status, "open");
    // Others cannot write it; nobody writes main.
    assert_eq!(write_in(&mut s, &keys, &imp, &c, "fn x() -> Int = 1").unwrap_err()[0].code, "E0903");
    s.set_space(crate::Space::Main);
    assert_eq!(s.define("fn x() -> Int = 1", "x").unwrap_err()[0].code, "E0903");
    // A verifier contributes tests to a submitted change set, nothing else.
    s.change_submit(&arch, &c).unwrap();
    write_in(&mut s, &keys, &ver, &c, "test total_one = total(1) == 7").unwrap();
    assert_eq!(s.change(&c).unwrap().unwrap().status, "submitted", "a contribution keeps the status");
    let e = write_in(&mut s, &keys, &ver, &c, "fn y() -> Int = 1").unwrap_err();
    assert!(e[0].code == "E0903" && e[0].message.contains("only `test`"), "{e:?}");
    // Abandon: leases free, writes refused (E0908).
    s.change_abandon(&arch, &c).unwrap();
    assert!(s.leases(&c).unwrap().is_empty());
    assert_eq!(write_in(&mut s, &keys, &arch, &c, "fn z() -> Int = 1").unwrap_err()[0].code, "E0908");
    let o = verified(&mut s, Some(&trust));
    assert!(o.ok, "{o:?}");
}

#[test]
fn leases_are_all_or_nothing_and_never_wait() {
    let (mut s, keys, _owner, arch, imp, _ver, imp2) = full_team();
    let trust = Trust::parse(&s.team_trust().unwrap().to_string()).unwrap();
    let a = open(&mut s, &imp, "a", &[], None);
    let b = open(&mut s, &imp2, "b", &[], None);
    write_in(&mut s, &keys, &imp, &a, "fn shared() -> Int = 1").unwrap();
    // The same name in another change set: the whole define is refused, with the holder.
    let e = write_in(&mut s, &keys, &imp2, &b, "fn mine() -> Int = 2\nfn shared() -> Int = 3").unwrap_err();
    assert_eq!(e.len(), 1);
    assert_eq!((e[0].code, e[0].kind), ("E0901", "lease_held"));
    let detail = e[0].detail.clone().unwrap();
    assert_eq!((detail["name"].as_str(), detail["change"].as_str(), detail["member"].as_str()), (Some("shared"), Some(a.as_str()), Some("impl-1")));
    assert!(s.leases(&b).unwrap().is_empty(), "nothing was taken");
    // Explicit acquire: all or nothing.
    let names = |xs: &[&str]| xs.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let e = s.lease("impl-2", "acquire", &b, &names(&["free1", "shared"]), None).unwrap_err();
    assert_eq!(e[0].code, "E0901");
    assert!(s.leases(&b).unwrap().is_empty());
    let got = s.lease("impl-2", "acquire", &b, &names(&["free1", "free2"]), Some(120_000)).unwrap();
    assert_eq!(got["leases"].as_array().unwrap().len(), 2);
    assert_eq!(s.lease("impl-1", "acquire", &b, &names(&["x"]), None).unwrap_err()[0].code, "E0909", "only in an own change set");
    assert_eq!(s.lease("impl-2", "acquire", &b, &names(&["x"]), Some(1000)).unwrap_err()[0].code, "E0914");
    assert_eq!(s.lease("impl-2", "renew", &b, &names(&["shared"]), None).unwrap_err()[0].code, "E0901");
    s.lease("impl-2", "renew", &b, &names(&["free1"]), Some(3_600_000)).unwrap();
    let r = s.lease("impl-2", "release", &b, &names(&["free1", "shared"]), None).unwrap();
    assert_eq!(r["released"], json!(["free1"]));
    // An expired lease counts as free.
    raw(&s).execute("UPDATE leases SET expires_at = 1 WHERE name = 'shared'", []).unwrap();
    write_in(&mut s, &keys, &imp2, &b, "fn shared() -> Int = 3").unwrap();
    assert_eq!(s.leases(&b).unwrap().iter().filter(|l| l.name == "shared").count(), 1);
    // The architect breaks it: signed, the holder hears of it, its change set needs a rebase.
    assert_eq!(s.lease_break(&imp, &names(&["shared"])).unwrap_err()[0].code, "E0903");
    let broken = s.lease_break(&arch, &names(&["shared"])).unwrap();
    assert_eq!(broken["broken"][0]["holder"], "impl-2");
    assert!(s.change(&b).unwrap().unwrap().needs_rebase);
    let (n, _) = s.take_notices(&s.team_log().unwrap(), "impl-2", 10).unwrap();
    assert!(n.iter().any(|n| n.kind == "lease_broken"), "{n:?}");
    let o = verified(&mut s, Some(&trust));
    assert!(o.ok, "{o:?}");
}

#[test]
fn implementers_keep_scope_and_interfaces() {
    let (mut s, keys, _owner, arch, imp, _ver, imp2) = full_team();
    let trust = Trust::parse(&s.team_trust().unwrap().to_string()).unwrap();
    // The skeleton lands in main only by merge (4c); here a work package built on stubs the architect writes into it.
    let p = open(&mut s, &arch, "orders", &["make"], Some("impl-1"));
    write_in(&mut s, &keys, &arch, &p, "fn make(q: Int) -> Int req q > 0 ens r == q = todo()\nfn other() -> Int = 1").unwrap();
    assert!(s.change(&p).unwrap().unwrap().is_package());
    // Only the assignee claims it; it takes the scope's leases.
    assert_eq!(s.change_claim(&imp2, &p).unwrap_err()[0].code, "E0909");
    assert_eq!(s.change_claim(&arch, &p).unwrap_err()[0].code, "E0903");
    // The architect's stubs hold leases for the package itself: the claim may take them.
    let c = s.change_claim(&imp, &p).unwrap();
    assert_eq!(c["owner"], "impl-1");
    assert_eq!(s.change_claim(&imp, &p).unwrap_err()[0].code, "E0908");
    // The body may change; the signature, contracts and type not; outside the scope nothing existing.
    write_in(&mut s, &keys, &imp, &p, "fn make(q: Int) -> Int req q > 0 ens r == q = q").unwrap();
    for (src, code) in [
        ("fn make(q: Int) -> Str req q > 0 = \"x\"", "E0906"),
        ("fn make(q: Int) -> Int req q >= 0 ens r == q = q", "E0906"),
        ("fn other() -> Int = 2", "E0905"),
    ] {
        let e = write_in(&mut s, &keys, &imp, &p, src).unwrap_err();
        assert_eq!(e[0].code, code, "{src}: {e:?}");
    }
    // New names are free, and its own new names it may reshape.
    write_in(&mut s, &keys, &imp, &p, "fn helper(x: Int) -> Int = x").unwrap();
    write_in(&mut s, &keys, &imp, &p, "fn helper(x: Int, y: Int) -> Int = x + y").unwrap();
    s.change_scope(&arch, &p, &["other".into()], &[]).unwrap();
    write_in(&mut s, &keys, &imp, &p, "fn other() -> Int = 2").unwrap();
    assert_eq!(s.change_scope(&imp, &p, &["x".into()], &[]).unwrap_err()[0].code, "E0903");
    let o = verified(&mut s, Some(&trust));
    assert!(o.ok, "{o:?}");
}

#[test]
fn a_stale_base_is_refused() {
    let (mut s, keys, _owner, _arch, imp, ver, _imp2) = full_team();
    let c = open(&mut s, &imp, "work", &[], None);
    write_in(&mut s, &keys, &imp, &c, "fn f() -> Int = 1").unwrap();
    s.change_submit(&imp, &c).unwrap();
    write_in(&mut s, &keys, &ver, &c, "test f_one = f() == 1").unwrap();
    // The owner did not see the verifier's test: E0904; after viewing it, the write goes through.
    let e = write_in(&mut s, &keys, &imp, &c, "test f_one = f() >= 1").unwrap_err();
    assert_eq!((e[0].code, e[0].kind), ("E0904", "stale_base"));
    let h = s.view(&["f_one".into()]).unwrap()[0].hash.clone();
    s.mark_seen(&imp.session, &crate::Space::Change(c.clone()), &[("f_one".into(), h)]).unwrap();
    write_in(&mut s, &keys, &imp, &c, "test f_one = f() >= 1").unwrap();
    // Its own names are never stale for the writer; the verifier now has not seen the owner's version.
    write_in(&mut s, &keys, &imp, &c, "fn f() -> Int = 2").unwrap();
    s.change_submit(&imp, &c).unwrap();
    assert_eq!(write_in(&mut s, &keys, &ver, &c, "test f_one = f() == 2").unwrap_err()[0].code, "E0904");
}

/// Review I1: a relink is not a view. The owner replaces `f` (same signature), which relinks the
/// verifier's test; the owner has still not seen that test and may not overwrite it.
#[test]
fn a_relink_does_not_count_as_seen() {
    let (mut s, keys, _owner, _arch, imp, ver, _imp2) = full_team();
    let c = open(&mut s, &imp, "work", &[], None);
    write_in(&mut s, &keys, &imp, &c, "fn f() -> Int = 1").unwrap();
    s.change_submit(&imp, &c).unwrap();
    write_in(&mut s, &keys, &ver, &c, "test f_one = f() == 1").unwrap();
    let d = write_in(&mut s, &keys, &imp, &c, "fn f() -> Int = 2").unwrap();
    assert!(d.iter().any(|x| x.name == "f_one" && x.status == "relinked"), "{d:?}");
    let e = write_in(&mut s, &keys, &imp, &c, "test f_one = f() == 2").unwrap_err();
    assert_eq!(e[0].code, "E0904", "a relink is not a view: {e:?}");
    // Seen before the relink: the mark moves along with it.
    let h = s.view(&["f_one".into()]).unwrap()[0].hash.clone();
    s.mark_seen(&imp.session, &crate::Space::Change(c.clone()), &[("f_one".into(), h)]).unwrap();
    write_in(&mut s, &keys, &imp, &c, "fn f() -> Int = 3").unwrap();
    write_in(&mut s, &keys, &imp, &c, "test f_one = f() == 3").unwrap();
}

/// Review I2: the leases of a define count against the delta (2 000), not the 256 reservations,
/// and the delta is checked before anything is proven.
#[test]
fn large_defines_are_bounded_by_the_delta_not_by_leases() {
    let (mut s, keys, _owner, arch, _imp, _ver, _imp2) = full_team();
    let c = open(&mut s, &arch, "big", &[], None);
    let src = |from: usize, n: usize| -> String { (from..from + n).map(|i| format!("fn g{i}() -> Int = {i}\n")).collect() };
    write_in(&mut s, &keys, &arch, &c, &src(0, 300)).unwrap();
    assert_eq!(s.last_write().leases.len(), 300);
    write_in(&mut s, &keys, &arch, &c, &src(300, 1700)).unwrap();
    assert_eq!(s.leases(&c).unwrap().len(), 2000);
    // One more name is over the delta: refused before the prover (fast), nothing written.
    let start = std::time::Instant::now();
    let e = write_in(&mut s, &keys, &arch, &c, &src(2000, 1)).unwrap_err();
    assert_eq!((e[0].code, e[0].kind), ("E0914", "team_limit"), "{e:?}");
    assert!(e[0].message.contains("2000"), "{e:?}");
    eprintln!("refused in {:?}", start.elapsed());
    // Explicit reservations of names it has not changed: 256 on top.
    let names = |a: usize, n: usize| (a..a + n).map(|i| format!("r{i}")).collect::<Vec<_>>();
    for k in 0..4 {
        s.lease("arch-1", "acquire", &c, &names(k * 64, 64), None).unwrap();
    }
    assert_eq!(s.lease("arch-1", "acquire", &c, &names(256, 1), None).unwrap_err()[0].code, "E0914");
    let trust = Trust::parse(&s.team_trust().unwrap().to_string()).unwrap();
    assert!(verified(&mut s, Some(&trust)).ok);
}

/// Review I4: every column a change set's decisions rest on is bound by its records.
#[test]
fn tampering_with_change_set_metadata_is_detected() {
    let store = || {
        let (mut s, keys, _owner, arch, imp, _ver, _imp2) = full_team();
        // A deleted base name keeps a row without a hash (an architect's own change set).
        let own = open(&mut s, &arch, "own", &[], None);
        write_in(&mut s, &keys, &arch, &own, "fn spare() -> Int = 3").unwrap();
        s.delete(&["base".into()], "x").unwrap();
        s.lease("arch-1", "release", &own, &["base".into()], None).unwrap();
        let p = open(&mut s, &arch, "pkg", &["base"], Some("impl-1"));
        s.change_claim(&imp, &p).unwrap();
        write_in(&mut s, &keys, &imp, &p, "fn base() -> Int = 7\nfn added() -> Int = 1").unwrap();
        let trust = Trust::parse(&s.team_trust().unwrap().to_string()).unwrap();
        assert!(verified(&mut s, Some(&trust)).ok);
        (s, trust)
    };
    let probes: &[(&str, &str)] = &[
        ("origin of an added name", "UPDATE change_names SET origin = 'base' WHERE name = 'added'"),
        ("origin of an edited name", "UPDATE change_names SET origin = 'relinked' WHERE name = 'base'"),
        ("base_hash", "UPDATE change_names SET base_hash = NULL WHERE name = 'base'"),
        ("base_hash of an added name", "UPDATE change_names SET base_hash = hash WHERE name = 'added'"),
        ("conflict mark", "UPDATE change_names SET conflict = '{}' WHERE name = 'base'"),
        ("the row of a deleted base name", "UPDATE change_names SET origin = 'renamed' WHERE hash IS NULL"),
        ("ghost row without a hash", "INSERT INTO change_names (change, name, hash, local_names, refs, origin, base_hash, conflict, updated_at) SELECT change, 'ghost', NULL, '[]', '[]', 'renamed', NULL, NULL, 0 FROM change_names LIMIT 1"),
        ("refs re-spelled", "UPDATE change_names SET refs = '[ ]' WHERE name = 'added'"),
        ("changes.base_root", "UPDATE changes SET base_root = 'aa'"),
        ("changes.base_seq", "UPDATE changes SET base_seq = 0"),
        ("changes.claimed", "UPDATE changes SET claimed = 0"),
        ("changes.assignee", "UPDATE changes SET assignee = 'impl-2'"),
        ("changes.title", "UPDATE changes SET title = 'other'"),
        // Review N2: absent is not empty.
        ("base_hash = ''", "UPDATE change_names SET base_hash = '' WHERE name = 'added'"),
        ("conflict = ''", "UPDATE change_names SET conflict = '' WHERE name = 'added'"),
        ("assignee = ''", "UPDATE changes SET assignee = '' WHERE assignee IS NULL"),
    ];
    for (what, sql) in probes {
        let (mut s, trust) = store();
        raw(&s).execute_batch(&format!("PRAGMA foreign_keys = OFF; {sql};")).unwrap();
        let o = verified(&mut s, Some(&trust));
        assert_eq!(code(&o), "E0923", "{what}: {o:?}");
    }
}

/// Review M3, M4: claiming respects the per-member limit; an architect does not break its own leases.
#[test]
fn claims_count_toward_the_limit_and_own_leases_are_not_broken() {
    let (mut s, _keys, _owner, arch, imp, _ver, _imp2) = full_team();
    for k in 0..change::MAX_OPEN_PER_MEMBER {
        open(&mut s, &imp, &format!("own {k}"), &[], None);
    }
    let p = open(&mut s, &arch, "pkg", &[], Some("impl-1"));
    assert_eq!(s.change_claim(&imp, &p).unwrap_err()[0].code, "E0914");
    s.lease("arch-1", "acquire", &p, &["mine".into()], None).unwrap();
    let r = s.lease_break(&arch, &["mine".into()]).unwrap();
    assert_eq!((r["broken"].as_array().unwrap().len(), r["skipped"].as_array().unwrap().len()), (0, 1));
    assert!(!s.change(&p).unwrap().unwrap().needs_rebase);
}

/// Review N1, N4: renaming (or deleting) another session's unseen version and then writing the
/// name is E0904, like overwriting it directly.
#[test]
fn rename_and_delete_do_not_launder_an_unseen_version() {
    let (mut s, keys, _owner, arch, _imp, ver, _imp2) = full_team();
    let c = open(&mut s, &arch, "work", &[], None);
    write_in(&mut s, &keys, &arch, &c, "fn f() -> Int = 1").unwrap();
    s.change_submit(&arch, &c).unwrap();
    write_in(&mut s, &keys, &ver, &c, "test f_one = f() == 1\ntest f_two = f() >= 1").unwrap();
    assert_eq!(write_in(&mut s, &keys, &arch, &c, "test f_one = f() == 2").unwrap_err()[0].code, "E0904");
    // Rename (and rename back): refused before it can give the architect a mark.
    let e = s.rename("f_one", "f_three", "x").unwrap_err();
    assert_eq!(e[0].code, "E0904", "{e:?}");
    // Delete, then define afresh: refused too.
    let e = s.delete(&["f_two".into()], "x").unwrap_err();
    assert_eq!(e[0].code, "E0904", "{e:?}");
    // After a view, rename works; the mark moves with it, and the overwrite of the new name goes through.
    let h = s.view(&["f_one".into()]).unwrap()[0].hash.clone();
    s.mark_seen(&arch.session, &crate::Space::Change(c.clone()), &[("f_one".into(), h)]).unwrap();
    s.rename("f_one", "f_three", "x").unwrap();
    write_in(&mut s, &keys, &arch, &c, "test f_three = f() == 1").unwrap();
    // The architect's own names it renames freely, and a rename never creates a mark it had not.
    write_in(&mut s, &keys, &arch, &c, "fn g() -> Int = 2").unwrap();
    s.rename("g", "g2", "x").unwrap();
    s.rename("g2", "g", "x").unwrap();
    let trust = Trust::parse(&s.team_trust().unwrap().to_string()).unwrap();
    assert!(verified(&mut s, Some(&trust)).ok);
}

/// Review N3: change-set metadata is replayed from the records, so a tamper is found even after
/// later honest writes signed the tampered state.
#[test]
fn tampered_metadata_is_found_after_later_writes() {
    let (mut s, keys, _owner, _arch, imp, _ver, _imp2) = full_team();
    let trust = Trust::parse(&s.team_trust().unwrap().to_string()).unwrap();
    let c = open(&mut s, &imp, "work", &[], None);
    write_in(&mut s, &keys, &imp, &c, "fn added() -> Int = 1").unwrap();
    assert!(verified(&mut s, Some(&trust)).ok);
    // An added name made to look like one of the base (so it would be interface for implementers).
    raw(&s).execute("UPDATE change_names SET origin = 'edited', base_hash = hash WHERE name = 'added'", []).unwrap();
    assert_eq!(code(&verified(&mut s, Some(&trust))), "E0923");
    write_in(&mut s, &keys, &imp, &c, "fn more() -> Int = 2").unwrap();
    let o = verified(&mut s, Some(&trust));
    assert_eq!(code(&o), "E0923", "{o:?}");
    assert!(o.error.unwrap().message.contains("replayed"));
    // A changes column, then a write.
    let (mut s, keys, _owner, _arch, imp, _ver, _imp2) = full_team();
    let c = open(&mut s, &imp, "work", &[], None);
    raw(&s).execute("UPDATE changes SET base_root = 'aa'", []).unwrap();
    write_in(&mut s, &keys, &imp, &c, "fn x() -> Int = 1").unwrap();
    assert_eq!(code(&verified(&mut s, None)), "E0923");
}

// --- Phase 4c: review, merge, rebase ------------------------------------------------------------

/// [`full_team`] plus a reviewer `rev-1`: (store, keys, owner, arch, imp, ver, rev).
fn merge_team() -> (Store, KeyDir, Signer, Signer, Signer, Signer, Signer) {
    let (mut s, keys, owner, arch, imp, ver, _imp2) = full_team();
    let r = s.team_add(&owner, "rev-1", Role::Reviewer, &keys).unwrap();
    let rev = Signer { member: "rev-1".into(), session: "s000000000005".into(), key: keys.load(s.project_dir().unwrap(), &r.key_id).unwrap() };
    (s, keys, owner, arch, imp, ver, rev)
}

/// A merge whose check accepts (the store has no prover or runtime: the command line's check, with
/// fresh proofs and the tests, is tested over MCP).
fn merge_as(s: &mut Store, keys: &KeyDir, who: &Signer, c: &str) -> Result<Value, Vec<qal_core::diag::Diagnostic>> {
    let signer = again(keys, s, who);
    let mut run = |_: &qal_core::ast::Program, _: &qal_core::tast::TProgram, _: &[(String, String)]| Ok(merge::MergeChecked::default());
    let mut check = merge::MergeCheck { key: "test".into(), run: &mut run };
    s.merge(&signer, c, &mut check)
}

/// Submits `c` (by its owner `who`), then a verifier's `verified` and a reviewer's `approve`.
fn pass_gate(s: &mut Store, who: &Signer, ver: &Signer, rev: &Signer, c: &str) {
    if s.change(c).unwrap().unwrap().status == "open" {
        s.change_submit(who, c).unwrap();
    }
    s.review(ver, c, "verified", &[]).unwrap();
    s.review(rev, c, "approve", &[merge::Comment { name: None, text: "fine".into() }]).unwrap();
}

fn main_hash(s: &Store, n: &str) -> Option<String> {
    raw(s).query_row("SELECT hash FROM names WHERE name = ?1", [n], |r| r.get(0)).ok()
}

fn change_hash(s: &Store, c: &str, n: &str) -> Option<String> {
    raw(s).query_row("SELECT hash FROM change_names WHERE change = ?1 AND name = ?2", [c, n], |r| r.get::<_, Option<String>>(0)).ok().flatten()
}

fn ok_verify(s: &mut Store) {
    let trust = Trust::parse(&s.team_trust().unwrap().to_string()).unwrap();
    let o = verified(s, Some(&trust));
    assert!(o.ok, "{o:?}");
}

/// §7.2 row 1, the gate (§7.4) and the end of a merged change set.
#[test]
fn a_fast_forward_takes_the_change_set_as_it_is() {
    let (mut s, keys, _owner, arch, _imp, ver, rev) = merge_team();
    let c = open(&mut s, &arch, "skeleton", &[], None);
    write_in(&mut s, &keys, &arch, &c, "fn f() -> Int = base() + 1\ntest f_two = f() == 2").unwrap();
    // The gate: submitted, then verified and approved by non-owners for the current root.
    assert_eq!(merge_as(&mut s, &keys, &rev, &c).unwrap_err()[0].code, "E0913", "open");
    s.change_submit(&arch, &c).unwrap();
    assert_eq!(s.review(&rev, &c, "verified", &[]).unwrap_err()[0].code, "E0903", "a reviewer does not verify");
    assert_eq!(s.review(&ver, &c, "approve", &[]).unwrap_err()[0].code, "E0903", "a verifier does not approve");
    s.review(&ver, &c, "verified", &[]).unwrap();
    let e = merge_as(&mut s, &keys, &rev, &c).unwrap_err();
    assert!(e[0].code == "E0913" && e[0].message.contains("approve"), "{e:?}");
    assert_eq!(merge_as(&mut s, &keys, &ver, &c).unwrap_err()[0].code, "E0903", "a verifier does not merge");
    let r = s.review(&rev, &c, "approve", &[]).unwrap();
    assert_eq!(r["verdict"], "approve");
    let root = r["root"].as_str().unwrap().to_string();
    let f = change_hash(&s, &c, "f");
    let m = merge_as(&mut s, &keys, &rev, &c).unwrap();
    assert_eq!((m["kind"].as_str(), m["rehashed"].as_array().map(Vec::len)), (Some("fast_forward"), Some(0)), "{m}");
    // Main is the change set's view, hash for hash; the change set ends, its leases are free.
    assert_eq!(main_hash(&s, "f"), f);
    assert!(main_hash(&s, "f_two").is_some());
    let row = s.change(&c).unwrap().unwrap();
    assert_eq!(row.status, "merged");
    assert!(s.leases(&c).unwrap().is_empty());
    assert_eq!(s.root_of_space(&crate::Space::Main).unwrap(), root);
    // A merged change set takes nothing more.
    assert_eq!(merge_as(&mut s, &keys, &rev, &c).unwrap_err()[0].code, "E0908");
    assert_eq!(s.review(&rev, &c, "approve", &[]).unwrap_err()[0].code, "E0908");
    // Everyone else hears of it.
    let (n, _) = s.take_notices(&s.team_log().unwrap(), "impl-1", 50).unwrap();
    assert!(n.iter().any(|n| n.kind == "merged"), "{n:?}");
    let (n, _) = s.take_notices(&s.team_log().unwrap(), "arch-1", 50).unwrap();
    assert!(n.iter().filter(|n| n.kind == "review").count() >= 2, "the owner hears of reviews: {n:?}");
    ok_verify(&mut s);
}

/// §7.2 rows 2, 3 and 4: main moved since the base; clean names are replayed (and rehashed), a
/// definition both reached is converged, a name main only relinked takes the change set's text.
#[test]
fn a_merge_replays_onto_a_moved_main() {
    let (mut s, keys, _owner, arch, _imp, ver, rev) = seeded();
    // Two change sets from the same base (leases: b writes the shared names after a's merge).
    let a = open(&mut s, &arch, "a", &[], None);
    let b = open(&mut s, &arch, "b", &[], None);
    write_in(&mut s, &keys, &arch, &a, "fn base() -> Int = 2\nfn same() -> Int = 7").unwrap();
    write_in(&mut s, &keys, &arch, &b, "fn g() -> Int = base() + uses()\nfn uses() -> Int = base() * 100").unwrap();
    pass_gate(&mut s, &arch, &ver, &rev, &a);
    assert_eq!(merge_as(&mut s, &keys, &rev, &a).unwrap()["kind"], "fast_forward");
    write_in(&mut s, &keys, &arch, &b, "fn same() -> Int = 7").unwrap();
    // `uses` was only relinked in main (row 4), `g` is new (row 2), `same` converged (row 3).
    pass_gate(&mut s, &arch, &ver, &rev, &b);
    let ours_g = change_hash(&s, &b, "g");
    let m = merge_as(&mut s, &keys, &rev, &b).unwrap();
    assert_eq!(m["kind"], "replayed", "{m}");
    let rehashed: Vec<&str> = m["rehashed"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
    assert!(rehashed.contains(&"g") && rehashed.contains(&"uses"), "{m}");
    assert_ne!(main_hash(&s, "g"), ours_g, "g refers to the new base");
    // The merged main has both changes: base from a, uses from b (against a's base).
    s.set_space(crate::Space::Main);
    let text = s.view(&["uses".into(), "base".into()]).unwrap();
    assert!(text[0].text.contains("100") && text[1].text.contains("= 2"), "{text:?}");
    let entries = m["entries"].as_array().unwrap();
    assert!(!entries.iter().any(|e| e["name"] == "same"), "converged: nothing to do");
    ok_verify(&mut s);
}

/// [`merge_team`] whose main also has `spare` and `uses` (merged by a first change set).
fn seeded() -> (Store, KeyDir, Signer, Signer, Signer, Signer, Signer) {
    let (mut s, keys, owner, arch, imp, ver, rev) = merge_team();
    let c0 = open(&mut s, &arch, "seed", &[], None);
    write_in(&mut s, &keys, &arch, &c0, "fn spare() -> Int = 2\nfn uses() -> Int = base() * 10").unwrap();
    pass_gate(&mut s, &arch, &ver, &rev, &c0);
    merge_as(&mut s, &keys, &rev, &c0).unwrap();
    (s, keys, owner, arch, imp, ver, rev)
}

/// The open change sets `a` and `b` (both from the current main) for the conflict tests.
fn two_from_one_base(s: &mut Store, arch: &Signer) -> (String, String) {
    (open(s, arch, "a", &[], None), open(s, arch, "b", &[], None))
}

/// §7.2 rows 5, 6 and 7: edit/edit, add/add and rename conflicts are E0910 with their detail.
#[test]
fn real_conflicts_are_refused_with_what_main_did() {
    let (mut s, keys, _owner, arch, _imp, ver, rev) = seeded();
    let (a, b) = two_from_one_base(&mut s, &arch);
    write_in(&mut s, &keys, &arch, &a, "fn spare() -> Int = 5\nfn twin() -> Int = 1").unwrap();
    s.set_space(crate::Space::Change(a.clone()));
    s.rename("base", "base1", "x").unwrap();
    pass_gate(&mut s, &arch, &ver, &rev, &a);
    merge_as(&mut s, &keys, &rev, &a).unwrap();
    assert!(main_hash(&s, "base").is_none() && main_hash(&s, "base1").is_some(), "the rename reached main");
    // b (from the old base) edits `spare`, adds another `twin` and edits `base`.
    write_in(&mut s, &keys, &arch, &b, "fn spare() -> Int = 6\nfn twin() -> Int = 2\nfn base() -> Int = 3").unwrap();
    pass_gate(&mut s, &arch, &ver, &rev, &b);
    let e = merge_as(&mut s, &keys, &rev, &b).unwrap_err();
    let kinds: BTreeMap<String, String> = e.iter().map(|d| (d.def.clone(), d.detail.as_ref().unwrap()["conflict"].as_str().unwrap().to_string())).collect();
    assert_eq!(kinds, [("base".to_string(), "rename".to_string()), ("spare".into(), "edit_edit".into()), ("twin".into(), "add_add".into())].into(), "{e:?}");
    assert!(e.iter().all(|d| d.code == "E0910" && d.kind == "merge_conflict"));
    let spare = e.iter().find(|d| d.def == "spare").unwrap().detail.clone().unwrap();
    assert_eq!(spare["theirs_by"], "rev-1", "the merging member wrote main");
    assert!(spare["record"].as_i64().is_some() && spare["base"].is_string() && spare["ours"].is_string() && spare["theirs"].is_string(), "{spare}");
    assert_eq!(s.change(&b).unwrap().unwrap().status, "submitted", "nothing changed");
    ok_verify(&mut s);
}

/// §7.2 semantic conflicts: a signature b uses changed in main; a signature b changes got new
/// dependents in main. Both E0911 with the inner diagnostics; nothing is written.
#[test]
fn semantic_conflicts_fail_the_check() {
    let (mut s, keys, _owner, arch, _imp, ver, rev) = seeded();
    let (a, b) = two_from_one_base(&mut s, &arch);
    write_in(&mut s, &keys, &arch, &a, "fn spare(x: Int) -> Int = x").unwrap();
    pass_gate(&mut s, &arch, &ver, &rev, &a);
    merge_as(&mut s, &keys, &rev, &a).unwrap();
    write_in(&mut s, &keys, &arch, &b, "fn user() -> Int = spare() + 1").unwrap();
    pass_gate(&mut s, &arch, &ver, &rev, &b);
    let records = count(&s, "records");
    let e = merge_as(&mut s, &keys, &rev, &b).unwrap_err();
    assert_eq!((e[0].code, e[0].kind), ("E0911", "merge_check_failed"), "{e:?}");
    assert!(e.len() > 1 && e[1..].iter().any(|d| d.def == "user"), "the inner diagnostics follow: {e:?}");
    assert_eq!(count(&s, "records"), records, "nothing written");
    // A signature the change set changes got a new dependent in main.
    let (c, d) = two_from_one_base(&mut s, &arch);
    write_in(&mut s, &keys, &arch, &c, "fn newuser() -> Int = base() + 1").unwrap();
    pass_gate(&mut s, &arch, &ver, &rev, &c);
    merge_as(&mut s, &keys, &rev, &c).unwrap();
    write_in(&mut s, &keys, &arch, &d, "fn base(x: Int) -> Int = x\nfn uses() -> Int = base(1) * 10").unwrap();
    pass_gate(&mut s, &arch, &ver, &rev, &d);
    let e = merge_as(&mut s, &keys, &rev, &d).unwrap_err();
    assert_eq!(e[0].code, "E0911", "{e:?}");
    assert!(e.iter().any(|x| x.kind == "signature_change" && x.dependents.contains(&"newuser".to_string())), "{e:?}");
    ok_verify(&mut s);
}

/// §7.3: a rebase marks conflicts and keeps the change set's version; a merge refuses marks
/// (E0908); a define of the name resolves it; then the merge goes through. needs_rebase is set by
/// the merge that caused it (with a record in the flagged change set's space, review N6) and
/// cleared by the rebase.
#[test]
fn a_rebase_marks_conflicts_and_their_resolution_merges() {
    let (mut s, keys, _owner, arch, _imp, ver, rev) = seeded();
    let (a, b) = two_from_one_base(&mut s, &arch);
    write_in(&mut s, &keys, &arch, &b, "fn mine() -> Int = spare()").unwrap();
    write_in(&mut s, &keys, &arch, &a, "fn spare(x: Int) -> Int = x + 5").unwrap();
    pass_gate(&mut s, &arch, &ver, &rev, &a);
    let m = merge_as(&mut s, &keys, &rev, &a).unwrap();
    assert_eq!(m["needs_rebase"][0]["change"], b.as_str(), "{m}");
    assert!(s.change(&b).unwrap().unwrap().needs_rebase);
    let (n, _) = s.take_notices(&s.team_log().unwrap(), "arch-1", 50).unwrap();
    assert!(n.iter().any(|n| n.kind == "needs_rebase"), "{n:?}");
    ok_verify(&mut s);
    // b changes `spare` too (a's lease is free since its merge): a conflict to mark.
    write_in(&mut s, &keys, &arch, &b, "fn spare() -> Int = 6").unwrap();
    // Rebase: `spare` keeps b's version with a mark; `mine` is replayed.
    let ours = change_hash(&s, &b, "spare");
    let r = s.rebase(&again(&keys, &s, &arch), &b, None).unwrap();
    assert_eq!(r["conflicts"][0]["name"], "spare", "{r}");
    assert_eq!(r["conflicts"][0]["theirs"].as_str(), main_hash(&s, "spare").as_deref());
    assert_eq!(change_hash(&s, &b, "spare"), ours);
    let row = s.change(&b).unwrap().unwrap();
    assert!(!row.needs_rebase && row.base_root == s.root_of_space(&crate::Space::Main).unwrap());
    ok_verify(&mut s);
    pass_gate(&mut s, &arch, &ver, &rev, &b);
    assert_eq!(merge_as(&mut s, &keys, &rev, &b).unwrap_err()[0].code, "E0908", "conflict marks");
    // Resolve: define it again (the mark goes), submit, reviews again, merge.
    write_in(&mut s, &keys, &arch, &b, "fn spare() -> Int = 7").unwrap();
    let marks: i64 = raw(&s).query_row("SELECT count(*) FROM change_names WHERE change = ?1 AND conflict IS NOT NULL", [&b], |r| r.get(0)).unwrap();
    assert_eq!(marks, 0);
    pass_gate(&mut s, &arch, &ver, &rev, &b);
    let m = merge_as(&mut s, &keys, &rev, &b).unwrap();
    assert_eq!(m["kind"], "fast_forward", "rebased onto main: {m}");
    s.set_space(crate::Space::Main);
    assert!(s.view(&["spare".into()]).unwrap()[0].text.contains("7"));
    // A rebase with nothing to do changes nothing.
    let c = open(&mut s, &arch, "c", &[], None);
    let records = count(&s, "records");
    assert_eq!(s.rebase(&again(&keys, &s, &arch), &c, None).unwrap()["unchanged"], true);
    assert_eq!(count(&s, "records"), records);
    ok_verify(&mut s);
}

/// §7.3: a rebase whose replay does not check changes nothing and names what to adapt.
#[test]
fn a_rebase_that_does_not_check_changes_nothing() {
    let (mut s, keys, _owner, arch, _imp, ver, rev) = seeded();
    let (a, b) = two_from_one_base(&mut s, &arch);
    write_in(&mut s, &keys, &arch, &a, "fn spare(x: Int) -> Int = x").unwrap();
    write_in(&mut s, &keys, &arch, &b, "fn user() -> Int = spare() + 1").unwrap();
    pass_gate(&mut s, &arch, &ver, &rev, &a);
    merge_as(&mut s, &keys, &rev, &a).unwrap();
    let records = count(&s, "records");
    let before = s.change(&b).unwrap().unwrap();
    let e = s.rebase(&again(&keys, &s, &arch), &b, None).unwrap_err();
    assert_eq!(e[0].code, "E0911", "{e:?}");
    assert!(e[0].detail.as_ref().unwrap()["names"].as_array().unwrap().iter().any(|n| n == "user"), "{e:?}");
    assert_eq!((count(&s, "records"), s.change(&b).unwrap().unwrap().base_root), (records, before.base_root));
    // Others may not rebase it.
    assert_eq!(s.rebase(&again(&keys, &s, &rev), &b, None).unwrap_err()[0].code, "E0903");
    ok_verify(&mut s);
}

/// §7.4: the gate is not undermined by self-approval, stale reviews, `changes` or `failed`.
#[test]
fn the_merge_gate_needs_current_reviews_by_others() {
    let (mut s, keys, _owner, _arch, _imp, ver, rev) = merge_team();
    // A verifier's own change set: its own `verified` does not count.
    let c = open(&mut s, &ver, "tests", &[], None);
    write_in(&mut s, &keys, &ver, &c, "test base_one = base() == 1").unwrap();
    s.change_submit(&ver, &c).unwrap();
    s.review(&ver, &c, "verified", &[]).unwrap();
    s.review(&rev, &c, "approve", &[]).unwrap();
    let e = merge_as(&mut s, &keys, &rev, &c).unwrap_err();
    assert!(e[0].code == "E0913" && e[0].message.contains("verified"), "self-approval: {e:?}");
    // Another verifier... there is none: the owner adds one.
    let (mut s, keys, owner, arch, _imp, ver, rev) = merge_team();
    let c = open(&mut s, &arch, "work", &[], None);
    write_in(&mut s, &keys, &arch, &c, "fn f() -> Int = 1").unwrap();
    pass_gate(&mut s, &arch, &ver, &rev, &c);
    // A later write makes the reviews stale (and reopens the change set).
    write_in(&mut s, &keys, &arch, &c, "fn f() -> Int = 2").unwrap();
    s.change_submit(&arch, &c).unwrap();
    let e = merge_as(&mut s, &keys, &rev, &c).unwrap_err();
    assert!(e[0].code == "E0913", "stale reviews: {e:?}");
    // A verifier's contribution changes the root too: verified first, then approve.
    s.review(&ver, &c, "verified", &[]).unwrap();
    write_in(&mut s, &keys, &ver, &c, "test f_two = f() == 2").unwrap();
    s.review(&rev, &c, "approve", &[]).unwrap();
    assert_eq!(merge_as(&mut s, &keys, &rev, &c).unwrap_err()[0].code, "E0913", "the verified is for the old root");
    // `failed` and `changes` block until answered again.
    s.review(&ver, &c, "failed", &[merge::Comment { name: Some("f".into()), text: "wrong".into() }]).unwrap();
    let e = merge_as(&mut s, &keys, &rev, &c).unwrap_err();
    assert!(e[0].code == "E0913" && e[0].message.contains("failed"), "{e:?}");
    s.review(&ver, &c, "verified", &[]).unwrap();
    s.review(&rev, &c, "changes", &[]).unwrap();
    assert_eq!(s.change(&c).unwrap().unwrap().status, "open", "changes sends it back");
    assert_eq!(s.review(&rev, &c, "approve", &[]).unwrap_err()[0].code, "E0908", "only submitted");
    s.change_submit(&arch, &c).unwrap();
    s.review(&rev, &c, "approve", &[]).unwrap();
    // A revoked reviewer's approval no longer counts.
    s.team_revoke(&owner, "rev-1").unwrap();
    assert_eq!(merge_as(&mut s, &keys, &arch, &c).unwrap_err()[0].code, "E0913");
    // Comments are bounded; the architect only comments.
    let many: Vec<merge::Comment> = (0..33).map(|i| merge::Comment { name: None, text: format!("{i}") }).collect();
    assert_eq!(s.review(&arch, &c, "comment", &many).unwrap_err()[0].code, "E0914");
    assert_eq!(s.review(&arch, &c, "approve", &[]).unwrap_err()[0].code, "E0903");
    s.review(&arch, &c, "comment", &[merge::Comment { name: None, text: "ok".into() }]).unwrap();
    assert_eq!(s.reviews(&c).unwrap().len(), 9);
    ok_verify(&mut s);
}

/// Deletes and renames of a change set reach main by a merge, as a fast-forward and as a replay;
/// main's users of a renamed name follow it.
#[test]
fn deletes_and_renames_are_merged() {
    let (mut s, keys, _owner, arch, _imp, ver, rev) = merge_team();
    let c0 = open(&mut s, &arch, "users", &[], None);
    write_in(&mut s, &keys, &arch, &c0, "fn twice() -> Int = base() * 2\nfn tmp() -> Int = 0").unwrap();
    pass_gate(&mut s, &arch, &ver, &rev, &c0);
    merge_as(&mut s, &keys, &rev, &c0).unwrap();
    let (a, b) = two_from_one_base(&mut s, &arch);
    s.set_signer(Some(Arc::new(again(&keys, &s, &arch))));
    s.set_space(crate::Space::Change(a.clone()));
    s.rename("base", "one", "x").unwrap();
    s.delete(&["tmp".into()], "x").unwrap();
    write_in(&mut s, &keys, &arch, &b, "fn other() -> Int = 3").unwrap();
    pass_gate(&mut s, &arch, &ver, &rev, &b);
    merge_as(&mut s, &keys, &rev, &b).unwrap();
    // a is replayed onto main (which moved): `twice` follows the rename.
    pass_gate(&mut s, &arch, &ver, &rev, &a);
    let m = merge_as(&mut s, &keys, &rev, &a).unwrap();
    assert_eq!(m["kind"], "replayed", "{m}");
    assert!(main_hash(&s, "base").is_none() && main_hash(&s, "tmp").is_none() && main_hash(&s, "one").is_some() && main_hash(&s, "other").is_some());
    s.set_space(crate::Space::Main);
    assert!(s.view(&["twice".into()]).unwrap()[0].text.contains("one()"));
    assert_eq!(s.renamed_to("base").unwrap().as_deref(), Some("one"));
    ok_verify(&mut s);
}

/// Review N6 and criterion 5: `needs_rebase` and the reviews are bound; a merge record without its
/// gate, and an implementer's write outside its scope, are refused by verify even when signed.
#[test]
fn tampering_with_reviews_and_merge_gates_is_detected() {
    let store = || {
        let (mut s, keys, owner, arch, imp, ver, rev) = seeded();
        let (a, b) = two_from_one_base(&mut s, &arch);
        write_in(&mut s, &keys, &arch, &b, "fn mine() -> Int = spare()").unwrap();
        write_in(&mut s, &keys, &arch, &a, "fn spare(x: Int) -> Int = x").unwrap();
        pass_gate(&mut s, &arch, &ver, &rev, &a);
        merge_as(&mut s, &keys, &rev, &a).unwrap();
        ok_verify(&mut s);
        (s, keys, owner, arch, imp, ver, rev, b)
    };
    for (what, sql) in [
        ("needs_rebase", "UPDATE changes SET needs_rebase = 0 WHERE needs_rebase = 1"),
        ("a review's verdict", "UPDATE reviews SET verdict = 'changes' WHERE verdict = 'approve'"),
        ("a review's root", "UPDATE reviews SET root = 'aa' WHERE verdict = 'approve'"),
        ("a review's comments", "UPDATE reviews SET comments = '[]' WHERE verdict = 'approve'"),
        ("a forged review row", "INSERT INTO reviews (change, member, verdict, root, comments, at, record) SELECT change, 'rev-1', 'approve', root, '[]', 0, 2 FROM reviews LIMIT 1"),
        ("a status set to merged", "UPDATE changes SET status = 'merged' WHERE needs_rebase = 1"),
    ] {
        let (mut s, ..) = store();
        raw(&s).execute_batch(&format!("PRAGMA foreign_keys = OFF; {sql};")).unwrap();
        let trust = Trust::parse(&s.team_trust().unwrap().to_string()).unwrap();
        let o = verified(&mut s, Some(&trust));
        assert_eq!(code(&o), "E0923", "{what}: {o:?}");
    }
    // Hand-signed records that the store would have refused: E0924.
    let forge = |s: &Store, who: &Signer, op: &str, space: &str, args: Value| {
        let c = raw(s);
        let (seq, prev): (i64, String) = c.query_row("SELECT seq, hash FROM records ORDER BY seq DESC LIMIT 1", [], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
        let sp = crate::Space::parse(space);
        let rows = space_rows(&c, &sp).unwrap();
        let mut args = args;
        if !sp.is_main() {
            args["meta_after"] = json!(change::meta_digest(&c, space).unwrap());
        }
        let body = json!({"v": 1, "seq": seq + 1, "prev": prev, "at": 1, "member": who.member, "key": who.key.key_id(), "session": who.session, "op": op, "space": space,
                          "args": args, "root_before": root_of(&rows), "root_after": root_of(&rows), "names_after": names_digest(&c, &rows).unwrap()});
        let bytes = record_bytes(&body).unwrap();
        c.execute("INSERT INTO records (seq, hash, body, sig) VALUES (?1, ?2, ?3, ?4)", rusqlite::params![seq + 1, Hash::of(&bytes).to_hex(), bytes, who.key.sign_hex(&bytes)]).unwrap();
        seq + 1
    };
    // A merge of b (not submitted, no reviews) by the reviewer.
    let (mut s, _keys, _owner, _arch, _imp, _ver, rev, b) = store();
    let root = root_of(&space_rows(&raw(&s), &crate::Space::Change(b.clone())).unwrap());
    let seq = forge(&s, &rev, "merge", "main", json!({"change": b, "change_root": root, "kind": "replayed", "reviews": [], "entries": []}));
    let o = verified(&mut s, None);
    assert_eq!((code(&o), o.error.as_ref().map(|e| e.seq)), ("E0924", Some(seq)), "{o:?}");
    // An implementer's define outside its scope (base name `spare`), signed with its key.
    let (mut s, keys, _owner, _arch, imp, _ver, _rev, _b) = store();
    let c = open(&mut s, &imp, "own", &[], None);
    let h = main_hash(&s, "base").unwrap();
    let spare = main_hash(&s, "spare").unwrap();
    let _ = keys;
    // `spare` is replaced by `base`'s definition (a valid stored hash), outside the empty scope.
    let seq = forge(&s, &imp, "define", &c, json!({"entries": [{"name": "spare", "old": spare, "new": h, "action": "replaced", "refs": [], "locals": []}]}));
    let o = verified(&mut s, None);
    assert_eq!((code(&o), o.error.as_ref().map(|e| e.seq)), ("E0924", Some(seq)), "{o:?}");
}
