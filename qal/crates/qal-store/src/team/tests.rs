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
fn team() -> (Store, KeyDir, Signer, Signer, Signer) {
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
    // Writes on main: architects only, until change sets (4b).
    for tool in ["qal_define", "qal_rename", "qal_delete"] {
        assert_eq!(policy::allowed_roles(tool, None, Space::Main), [Architect]);
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
    assert_eq!(policy::tools_for(Reviewer), ["qal_view", "qal_query", "qal_history", "qal_whoami", "qal_verify", "qal_test", "qal_prove", "qal_run", "qal_call", "qal_request", "qal_build"]);
    assert!(policy::tools_for(Implementer).contains(&"qal_define") && !policy::tools_for(Implementer).contains(&"qal_rename"));
    assert_eq!(policy::tools_for(Architect).len(), 14);
    // E0903 names the roles that may.
    let d = policy::forbidden("qal_define", None, Implementer, Space::Main);
    assert_eq!((d.code, d.kind, d.expected.as_deref(), d.found.as_deref()), ("E0903", "role_forbidden", Some("architect"), Some("implementer")));
    assert!(d.hints[0].contains("change set"));
    let d = policy::forbidden("qal_rename", None, Reviewer, Space::Main);
    assert!(d.message.contains("reviewer") && d.hints[0].contains("architect"), "{d:?}");
    // Records.
    assert!(policy::record_allowed("define", "architect") && !policy::record_allowed("define", "implementer") && !policy::record_allowed("define", OWNER));
    assert!(policy::record_allowed("member_add", OWNER) && !policy::record_allowed("member_add", "architect"));
    assert!(!policy::record_allowed("frobnicate", OWNER));
}

#[test]
fn a_team_store_signs_every_write_and_verifies() {
    let (mut s, keys, owner, arch, imp) = team();
    let trust = Trust::parse(&s.team_trust().unwrap().to_string()).unwrap();
    assert_eq!(trust.owner, "boss");
    // genesis + 2 members.
    let o = verified(&mut s, Some(&trust));
    assert!(o.ok, "{o:?}");
    assert_eq!((o.records, o.members, o.trust), (Some(3), Some(3), Some("file")));
    assert!(!o.warnings.iter().any(|w| w.contains("--trust")), "{o:?}");
    assert!(o.warnings.iter().any(|w| w.contains("--head")));
    let o = verified(&mut s, None);
    assert_eq!(o.trust, Some("store"));
    assert!(o.warnings[0].contains("--trust"));

    // Unsigned writes are refused in a team store.
    for e in [s.define("fn f() -> Int = 1", "cli").unwrap_err(), s.rename("base", "b2", "cli").unwrap_err(), s.delete(&["base".into()], "cli").unwrap_err()] {
        assert_eq!((e[0].code, e[0].expected.as_deref()), ("E0903", Some("architect")));
    }
    // An implementer cannot write on main (records refuse it too).
    as_member(&mut s, imp);
    assert_eq!(s.define("fn f() -> Int = 1", "x").unwrap_err()[0].code, "E0903");
    assert_eq!(count(&s, "records"), 3);
    // An architect can; the history names the member, not the given agent.
    as_member(&mut s, arch);
    let d = s.define("fn f() -> Int = base() + 1\ntest f_works = f() == 2", "ignored").unwrap();
    assert_eq!(d.len(), 2);
    assert_eq!(s.history("f").unwrap()[0].agent, "arch-1");
    s.define("fn base() -> Int = 2", "x").unwrap(); // f relinked
    s.rename("f", "g", "x").unwrap();
    s.define("fn tmp() -> Int = 0", "x").unwrap();
    s.delete(&["tmp".into()], "x").unwrap();
    assert_eq!(count(&s, "records"), 8);
    // Every history row of the team era points to its record.
    let unlinked: i64 = raw(&s).query_row("SELECT count(*) FROM history WHERE record IS NULL AND agent != 'pre-team' AND agent NOT LIKE 'qal-std@%'", [], |r| r.get(0)).unwrap();
    assert_eq!(unlinked, 0);
    let pre: Option<i64> = raw(&s).query_row("SELECT record FROM history WHERE agent = 'pre-team'", [], |r| r.get(0)).unwrap();
    assert_eq!(pre, None);
    let o = verified(&mut s, Some(&trust));
    assert!(o.ok, "{o:?}");
    assert_eq!(o.records, Some(8));
    // The record's body: op, member, entries.
    let body: Vec<u8> = raw(&s).query_row("SELECT body FROM records WHERE seq = 6", [], |r| r.get(0)).unwrap();
    let v: Value = serde_json::from_slice(&body[RECORD_PREFIX.len()..]).unwrap();
    assert_eq!((v["op"].as_str(), v["member"].as_str(), v["space"].as_str()), (Some("rename"), Some("arch-1"), Some("main")));
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
    assert_eq!(o.records, Some(11));
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
    raw(&s).execute("DELETE FROM records WHERE seq = 5", []).unwrap();
    assert_eq!(code(&verified(&mut s, Some(&trust))), "E0923");
    // Two records swapped.
    let (mut s, trust, _, _) = signed_store();
    raw(&s).execute_batch("UPDATE records SET seq = -1 WHERE seq = 4; UPDATE records SET seq = 4 WHERE seq = 5; UPDATE records SET seq = 5 WHERE seq = -1;").unwrap();
    let o = verified(&mut s, Some(&trust));
    assert_eq!((code(&o), o.error.as_ref().unwrap().seq), ("E0922", 4), "{o:?}");
}

#[test]
fn tampering_with_names_and_definitions_is_detected() {
    // A name pointing to another hash.
    let (mut s, trust, _, _) = signed_store();
    flip(&s, "names", "hash", "name = 'f'", 5);
    assert_eq!(code(&verified(&mut s, Some(&trust))), "E0923");
    // A name moved to another stored definition (as a valid hash).
    let (mut s, trust, _, _) = signed_store();
    raw(&s).execute("UPDATE names SET hash = (SELECT hash FROM names WHERE name = 'base') WHERE name = 'g'", []).unwrap();
    assert_eq!(code(&verified(&mut s, Some(&trust))), "E0923");
    // A name added or removed behind the records' back.
    let (mut s, trust, _, _) = signed_store();
    raw(&s).execute("DELETE FROM names WHERE name = 'g'", []).unwrap();
    assert_eq!(code(&verified(&mut s, Some(&trust))), "E0923");
    // A row in `change_names` no record made.
    let (mut s, trust, _, _) = signed_store();
    raw(&s).execute("INSERT INTO change_names (change, name, hash, local_names, refs, origin, updated_at) SELECT 'c000000000001', name, hash, local_names, refs, 'edited', 0 FROM names WHERE name = 'f'", []).unwrap();
    assert_eq!(code(&verified(&mut s, Some(&trust))), "E0923");
    // A change_names row changed by one byte (same detection).
    // One byte in a definition's canonical form.
    let (mut s, trust, _, _) = signed_store();
    let h: String = raw(&s).query_row("SELECT hash FROM names WHERE name = 'g'", [], |r| r.get(0)).unwrap();
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
    assert!(o.ok && o.records == Some(4), "{o:?}");
    let body: Vec<u8> = raw(&s).query_row("SELECT body FROM records WHERE seq = 4", [], |r| r.get(0)).unwrap();
    let v: Value = serde_json::from_slice(&body[RECORD_PREFIX.len()..]).unwrap();
    assert_eq!((v["op"].as_str(), v["member"].as_str(), v["session"].as_str()), (Some("import"), Some("boss"), Some(CLI_SESSION)));
    assert_eq!((v["args"]["package"].as_str(), v["args"]["hash"].as_str()), (Some("std"), Some(i.hash.as_str())));
    assert_eq!(v["args"]["entries"][0]["name"], "std.a.one");
    // The history keeps the package's agent (later imports of the same package may replace it).
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
        ("names.refs", "UPDATE names SET refs = '[]' WHERE name = 'twice'"),
        ("names.refs of a group member", "UPDATE names SET refs = '[\"is_even\",\"is_odd\"]' WHERE name = 'is_odd'"),
        ("names.refs re-spelled", "UPDATE names SET refs = '[ \"f\" ]' WHERE name = 'twice'"),
        ("names.local_names", "UPDATE names SET local_names = '[\"y\"]' WHERE name = 'twice'"),
        ("defs.signature", "UPDATE defs SET signature = 'fn(Int) -> Str' WHERE hash = (SELECT hash FROM names WHERE name = 'twice')"),
        ("defs.effects", "UPDATE defs SET effects = '[\"io\"]' WHERE hash = (SELECT hash FROM names WHERE name = 'twice')"),
        ("defs.kind", "UPDATE defs SET kind = 'test' WHERE hash = (SELECT hash FROM names WHERE name = 'twice')"),
        ("defs.local_names", "UPDATE defs SET local_names = '[\"z\"]' WHERE hash = (SELECT hash FROM names WHERE name = 'twice')"),
        ("groups.form", "UPDATE groups SET form = CAST(form AS BLOB) || x'00'"),
        ("groups deleted", "DELETE FROM groups"),
        ("deps deleted", "DELETE FROM deps"),
        ("one deps row deleted", "DELETE FROM deps WHERE hash = (SELECT hash FROM names WHERE name = 'twice')"),
        ("a deps row added", "INSERT INTO deps (hash, dep_hash) SELECT n.hash, b.hash FROM names n, names b WHERE n.name = 'f' AND b.name = 'twice'"),
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
    assert!(s.verify(Some(&trust), 3, None).unwrap().ok);
    assert_eq!(code(&verified(&mut s, Some(&trust))), "E0921");
}

#[test]
fn a_pinned_head_finds_a_cut_off_tail() {
    let (mut s, trust, _, _) = signed_store();
    let head = verified(&mut s, Some(&trust)).head.unwrap();
    assert!(s.verify(Some(&trust), 1, Some(&head)).unwrap().ok);
    // The last record and its change removed together: consistent without the pin.
    let c = raw(&s);
    c.execute_batch("PRAGMA foreign_keys = OFF; DELETE FROM records WHERE seq = (SELECT max(seq) FROM records); DELETE FROM names WHERE name = 'g';").unwrap();
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
    assert_eq!((count(&s, "names"), count(&s, "records")), (before, 3));
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
