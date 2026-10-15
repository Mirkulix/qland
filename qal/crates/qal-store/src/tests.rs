use super::*;
use qal_core::canon::{hash_program, Hash};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

/// A fresh, empty project directory.
fn project() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qal-store-test-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::SeqCst)));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn store() -> Store {
    Store::init(&project()).unwrap()
}

fn ok(s: &mut Store, src: &str) -> Vec<Defined> {
    s.define(src, "test").unwrap_or_else(|d| panic!("unexpected diagnostics: {d:#?}"))
}

fn err(s: &mut Store, src: &str) -> Vec<Diagnostic> {
    match s.define(src, "test") {
        Ok(d) => panic!("expected diagnostics, defined {d:#?}"),
        Err(d) => d,
    }
}

fn statuses(d: &[Defined]) -> Vec<(&str, &str)> {
    d.iter().map(|d| (d.name.as_str(), d.status)).collect()
}

fn hashes(s: &mut Store) -> Vec<(String, String)> {
    s.query(&Query::default()).unwrap().into_iter().map(|e| (e.name, e.hash)).collect()
}

fn hash(s: &mut Store, name: &str) -> String {
    hashes(s).into_iter().find(|(n, _)| n == name).unwrap_or_else(|| panic!("no `{name}`")).1
}

fn text(s: &mut Store, name: &str) -> String {
    s.view(&[name.to_string()]).unwrap().remove(0).text
}

fn raw(s: &Store) -> Connection {
    Connection::open(s.path()).unwrap()
}

fn count(s: &Store, table: &str) -> i64 {
    raw(s).query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0)).unwrap()
}

const MATH: &str = include_str!("../../../examples/math.qal");

#[test]
fn define_stores_checked_definitions() {
    let mut s = store();
    let d = ok(&mut s, MATH);
    assert_eq!(d.len(), 8);
    assert!(d.iter().all(|d| d.status == "added" && d.old_hash.is_none()), "{d:#?}");
    assert_eq!((d[0].name.as_str(), d[0].kind), ("fact", "fn"));
    assert_eq!(text(&mut s, "fact"), "fn fact(n: Int) -> Int\n  req n >= 0\n  ens r >= 1\n= if n <= 1 then 1 else n * fact(n - 1)");
    // The stored canonical form is what the hash is computed from.
    let (canonical, h): (Vec<u8>, String) =
        raw(&s).query_row("SELECT d.canonical, d.hash FROM defs d JOIN names n ON n.hash = d.hash WHERE n.name = 'gcd'", [], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
    assert_eq!(Hash::of(&canonical).to_hex(), h);
    assert_eq!(count(&s, "history"), 8);
}

#[test]
fn nothing_is_stored_when_any_definition_fails() {
    let mut s = store();
    let d = err(&mut s, "fn good() -> Int = 1\nfn bad() -> Int = true");
    assert_eq!((d[0].kind, d[0].def.as_str()), ("type_mismatch", "bad"));
    assert_eq!(err(&mut s, "fn good( -> Int = 1")[0].kind, "parse");
    assert_eq!(err(&mut s, "fn f() -> Int = missing()")[0].kind, "unknown_name");
    for table in ["defs", "deps", "names", "history"] {
        assert_eq!(count(&s, table), 0, "{table}");
    }
}

#[test]
fn renaming_reformatting_and_local_names_do_not_change_the_hash() {
    let mut a = store();
    ok(&mut a, "fn inc(x: Int) -> Int = x + 1");
    let mut b = store();
    ok(&mut b, "# another agent's style\nfn   plus_one( n:Int )->Int=(n)+1");
    assert_eq!(hash(&mut a, "inc"), hash(&mut b, "plus_one"));
    // The same logic under a second name in one store: same hash, two names, own local names.
    let d = ok(&mut a, "fn plus_one(n: Int) -> Int = n + 1");
    assert_eq!(statuses(&d), vec![("plus_one", "added")]);
    assert_eq!(hash(&mut a, "inc"), hash(&mut a, "plus_one"));
    assert_eq!(count(&a, "defs"), 1);
    assert_eq!(text(&mut a, "plus_one"), "fn plus_one(n: Int) -> Int\n= n + 1");
    assert_eq!(text(&mut a, "inc"), "fn inc(x: Int) -> Int\n= x + 1");
}

#[test]
fn rename_changes_no_hash_and_moves_the_history() {
    let mut s = store();
    ok(&mut s, "fn helper(n: Int) -> Int = n * 2\nfn user(x: Int) -> Int = helper(x) + 1\ntest t = user(1) == 3");
    let before: Vec<String> = hashes(&mut s).into_iter().map(|(_, h)| h).collect();
    let r = s.rename("helper", "double", "agent-b").unwrap();
    assert_eq!((r.name.as_str(), r.kind), ("double", "fn"));
    let after = hashes(&mut s);
    assert_eq!(after.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(), vec!["double", "t", "user"]);
    let mut after_hashes: Vec<String> = after.into_iter().map(|(_, h)| h).collect();
    let mut before_sorted = before.clone();
    after_hashes.sort();
    before_sorted.sort();
    assert_eq!(after_hashes, before_sorted);
    assert_eq!(text(&mut s, "user"), "fn user(x: Int) -> Int\n= double(x) + 1");
    let h = s.history("double").unwrap();
    assert_eq!(h.iter().map(|e| e.action.as_str()).collect::<Vec<_>>(), vec!["added", "renamed"]);
    assert_eq!((h[1].from_name.as_deref(), h[1].agent.as_str()), (Some("helper"), "agent-b"));
    assert_eq!(h[1].old_hash.as_deref(), Some(h[1].new_hash.as_str()));
    assert_eq!(s.history("helper").unwrap_err()[0].kind, "unknown_name");
}

#[test]
fn renamed_references_never_capture_locals() {
    let mut s = store();
    ok(&mut s, "fn helper(n: Int) -> Int = n * 2\nfn f(x: Int) -> Int = let y = helper(x) in y + x");
    let h = hash(&mut s, "f");
    s.rename("helper", "x", "test").unwrap();
    let t = text(&mut s, "f");
    assert_eq!(t, "fn f(x_1: Int) -> Int\n= let y = x(x_1) in y + x_1");
    // The view is valid source for the same definition.
    let d = ok(&mut s, &t);
    assert_eq!((d[0].status, d[0].hash.as_str()), ("unchanged", h.as_str()));
}

#[test]
fn rename_rejects_taken_invalid_and_reserved_names() {
    let mut s = store();
    ok(&mut s, "type P = { x: Int }\nfn f(p: P) -> Int = p.x\nfn g() -> Int = 1\ntest t = g() == 1");
    let kind = |s: &mut Store, old: &str, new: &str| s.rename(old, new, "test").unwrap_err()[0].kind;
    assert_eq!(kind(&mut s, "f", "g"), "duplicate_name");
    assert_eq!(kind(&mut s, "nope", "x"), "unknown_name");
    assert_eq!(kind(&mut s, "f", "F"), "parse");
    assert_eq!(kind(&mut s, "P", "p"), "parse");
    assert_eq!(kind(&mut s, "f", "let"), "parse");
    assert_eq!(kind(&mut s, "f", "x'); DROP TABLE names; --"), "parse");
    assert_eq!(kind(&mut s, "g", "print"), "duplicate_name");
    assert_eq!(kind(&mut s, "g", "r"), "duplicate_name");
    assert_eq!(kind(&mut s, "P", "Int"), "duplicate_name");
    assert_eq!(kind(&mut s, "g", "main"), "type_mismatch");
    assert_eq!(hashes(&mut s).iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(), vec!["P", "f", "g", "t"]);
    s.rename("P", "Point", "test").unwrap();
    assert_eq!(text(&mut s, "f"), "fn f(p: Point) -> Int\n= p.x");
    s.rename("t", "g_is_one", "test").unwrap();
}

const EVEN_ODD: &str = "fn is_even(n: Int) -> Bool = if n == 0 then true else is_odd(n - 1)\n\
                        fn is_odd(n: Int) -> Bool = if n == 0 then false else is_even(n - 1)";

#[test]
fn mutual_recursion_is_hashed_as_a_group() {
    let mut a = store();
    ok(&mut a, EVEN_ODD);
    // Other names, other order of definition: the same two hashes.
    let mut b = store();
    ok(&mut b, "fn pong(k: Int) -> Bool = if k == 0 then false else ping(k - 1)\nfn ping(k: Int) -> Bool = if k == 0 then true else pong(k - 1)");
    assert_eq!(hash(&mut a, "is_even"), hash(&mut b, "ping"));
    assert_eq!(hash(&mut a, "is_odd"), hash(&mut b, "pong"));
    assert_ne!(hash(&mut a, "is_even"), hash(&mut a, "is_odd"));
    let v = a.view(&["is_even".to_string()]).unwrap().remove(0);
    assert_eq!(v.deps.iter().map(|d| (d.name.as_str(), d.signature.as_str())).collect::<Vec<_>>(), vec![("is_odd", "fn(Int) -> Bool")]);
    // Renaming one member changes neither hash; the other member's text follows.
    let before = hashes(&mut a);
    a.rename("is_odd", "odd", "test").unwrap();
    assert_eq!(text(&mut a, "is_even"), "fn is_even(n: Int) -> Bool\n= if n == 0 then true else odd(n - 1)");
    let after = hashes(&mut a);
    assert_eq!(before.iter().map(|(_, h)| h).collect::<Vec<_>>(), after.iter().map(|(_, h)| h).collect::<Vec<_>>());
    // Recursive and mutually recursive types.
    let mut c = store();
    ok(&mut c, "type Tree = Leaf | Node(Tree, Int, Tree)\ntype A = HasB(B) | NoB\ntype B = HasA(A)\nfn size(t: Tree) -> Int = match t { Leaf => 0, Node(l, _, r) => size(l) + 1 + size(r) }");
    let mut d = store();
    ok(&mut d, "type B = HasA(A)\ntype A = NoB | HasB(B)");
    assert_ne!(hash(&mut c, "A"), hash(&mut d, "A"), "variant order is semantics");
    assert_eq!(text(&mut c, "Tree"), "type Tree = Leaf | Node(Tree, Int, Tree)");
}

#[test]
fn same_signature_replacement_relinks_dependents() {
    let mut s = store();
    ok(&mut s, "fn base() -> Int = 1\nfn user() -> Int = base() + 1\ntest t = user() == 2\nfn other() -> Int = 7");
    let (old_base, old_user, old_other) = (hash(&mut s, "base"), hash(&mut s, "user"), hash(&mut s, "other"));
    let d = s.define("fn base() -> Int = 41", "agent-a").unwrap();
    assert_eq!(statuses(&d), vec![("base", "replaced"), ("t", "relinked"), ("user", "relinked")]);
    assert_eq!(d[0].old_hash.as_deref(), Some(old_base.as_str()));
    assert_ne!(hash(&mut s, "user"), old_user);
    assert_eq!(hash(&mut s, "other"), old_other);
    assert_eq!(text(&mut s, "user"), "fn user() -> Int\n= base() + 1");
    let h = s.history("user").unwrap();
    assert_eq!((h[1].action.as_str(), h[1].agent.as_str(), h[1].old_hash.as_deref()), ("relinked", "agent-a", Some(old_user.as_str())));
    // The old version is still stored and viewable by hash.
    let old = s.view(&[format!("#{old_base}")]).unwrap().remove(0);
    assert_eq!(old.text, "fn base() -> Int\n= 1");
    // Defining the same text again changes nothing.
    let d = ok(&mut s, "fn base() -> Int = 41");
    assert_eq!(statuses(&d), vec![("base", "unchanged")]);
    assert_eq!(s.history("base").unwrap().len(), 2);
}

#[test]
fn signature_change_needs_the_dependents_in_the_same_call() {
    let mut s = store();
    ok(&mut s, "fn base() -> Int = 1\nfn user() -> Int = base() + 1\nfn user2() -> Int = base()\ntest t = user() == 2");
    let before = hashes(&mut s);
    let d = err(&mut s, "fn base(x: Int) -> Int = x");
    assert_eq!(d.len(), 1);
    assert_eq!((d[0].kind, d[0].code, d[0].def.as_str()), ("signature_change", "E0401", "base"));
    assert_eq!(d[0].dependents, vec!["user".to_string(), "user2".to_string()]);
    assert_eq!((d[0].expected.as_deref(), d[0].found.as_deref()), (Some("fn() -> Int"), Some("fn(Int) -> Int")));
    // Effects are part of the signature.
    assert_eq!(err(&mut s, "fn base() -> Int !io = { print(\"x\"); 1 }")[0].kind, "signature_change");
    assert_eq!(hashes(&mut s), before);
    // With updated dependents it goes through; their own dependents are relinked.
    let d = ok(&mut s, "fn base(x: Int) -> Int = x\nfn user() -> Int = base(1) + 1\nfn user2() -> Int = base(2)");
    assert_eq!(statuses(&d), vec![("base", "replaced"), ("user", "replaced"), ("user2", "replaced"), ("t", "relinked")]);
    // An included dependent with a changed signature needs its own dependents too.
    let d = err(&mut s, "fn base() -> Int = 1\nfn user(y: Int) -> Int = base() + y\nfn user2() -> Int = base()");
    assert_eq!((d[0].def.as_str(), d[0].dependents.clone()), ("user", vec!["t".to_string()]));
}

#[test]
fn type_changes_follow_the_same_rule() {
    let mut s = store();
    ok(&mut s, "type P = { x: Int }\nfn mk() -> P = P { x: 1 }\nfn get_x(p: P) -> Int = p.x\ntest t = get_x(mk()) == 1");
    let d = err(&mut s, "type P = { x: Int, y: Int }");
    assert_eq!((d[0].kind, d[0].dependents.clone()), ("signature_change", vec!["get_x".to_string(), "mk".to_string()]));
    assert_eq!((d[0].expected.as_deref(), d[0].found.as_deref()), (Some("type = { x: Int }"), Some("type = { x: Int, y: Int }")));
    let d = ok(&mut s, "type P = { x: Int, y: Int }\nfn mk() -> P = P { x: 1, y: 2 }\nfn get_x(p: P) -> Int = p.x");
    assert_eq!(statuses(&d), vec![("P", "replaced"), ("mk", "replaced"), ("get_x", "replaced"), ("t", "relinked")]);
    // A replacement whose dependents do not check is rejected as a whole.
    let before = hashes(&mut s);
    let d = err(&mut s, "type P = { x: Int, z: Int }\nfn mk() -> P = P { x: 1, y: 2 }\nfn get_x(p: P) -> Int = p.x");
    assert_eq!((d[0].kind, d[0].def.as_str()), ("unknown_name", "mk"));
    assert_eq!(hashes(&mut s), before);
}

#[test]
fn identical_types_under_two_names_are_rejected() {
    let mut s = store();
    ok(&mut s, "type A = { x: Int }");
    let d = err(&mut s, "type B = { x: Int }");
    assert_eq!((d[0].kind, d[0].def.as_str(), d[0].message.as_str()), ("duplicate_name", "B", "type `B` is identical to type `A`"));
    let d = err(&mut s, "type C = Red | Green\ntype D = Red | Green");
    assert_eq!(d[0].kind, "duplicate_name");
    assert_eq!(count(&s, "names"), 1);
}

#[test]
fn view_shows_text_signature_effects_and_dependencies() {
    let mut s = store();
    ok(&mut s, "type P = { x: Int }\nfn show(p: P) -> Str = from_int(p.x)\nfn say(p: P) -> Unit !io = print(show(p))");
    let v = s.view(&["say".to_string(), "P".to_string()]).unwrap();
    assert_eq!((v[0].kind, v[0].signature.as_str(), v[0].effects.clone()), ("fn", "fn(P) -> Unit !io", vec!["io".to_string()]));
    assert_eq!(v[0].text, "fn say(p: P) -> Unit !io\n= print(show(p))");
    let deps: Vec<(&str, &str, &str)> = v[0].deps.iter().map(|d| (d.name.as_str(), d.kind, d.signature.as_str())).collect();
    assert_eq!(deps, vec![("P", "type", "{ x: Int }"), ("show", "fn", "fn(P) -> Str")]);
    assert_eq!((v[1].signature.as_str(), v[1].deps.len()), ("{ x: Int }", 0));
    let d = s.view(&["nope".to_string(), "#abc".to_string()]).unwrap_err();
    assert_eq!(d.iter().map(|d| d.kind).collect::<Vec<_>>(), vec!["unknown_name", "unknown_name"]);
}

#[test]
fn query_by_name_prefix_signature_and_uses() {
    let mut s = store();
    ok(&mut s, MATH);
    let names = |s: &mut Store, q: Query| s.query(&q).unwrap().into_iter().map(|e| e.name).collect::<Vec<_>>();
    assert_eq!(names(&mut s, Query { name: Some("fact".into()), ..Query::default() }), vec!["fact", "fact_5"]);
    assert_eq!(names(&mut s, Query { type_pattern: Some("fn(Int) -> Bool".into()), ..Query::default() }), vec!["is_prime"]);
    assert_eq!(names(&mut s, Query { type_pattern: Some("fn(_, _) -> _".into()), ..Query::default() }), vec!["gcd", "no_divisor"]);
    assert_eq!(names(&mut s, Query { uses: Some("is_prime".into()), ..Query::default() }), vec!["not_prime_91", "prime_97"]);
    let h = hash(&mut s, "no_divisor");
    // A recursive definition does not count as using itself.
    assert_eq!(names(&mut s, Query { uses: Some(format!("#{h}")), ..Query::default() }), vec!["is_prime"]);
    assert_eq!(names(&mut s, Query { name: Some("f".into()), type_pattern: Some("Bool".into()), ..Query::default() }), vec!["fact_5"]);
    // Prefixes are plain text: SQL wildcards and quotes match nothing.
    for p in ["%", "_", "fa%", "'", "\" OR 1=1 --"] {
        assert!(names(&mut s, Query { name: Some(p.into()), ..Query::default() }).is_empty(), "{p}");
    }
    assert_eq!(s.query(&Query { uses: Some("nope".into()), ..Query::default() }).unwrap_err()[0].kind, "unknown_name");
}

#[test]
fn stored_codebase_rehashes_to_the_stored_hashes() {
    for src in [MATH, include_str!("../../../examples/age.qal"), include_str!("../../../examples/wordcount.qal")] {
        let mut s = store();
        ok(&mut s, src);
        let program = s.program().unwrap();
        let mut rehashed: Vec<(String, String)> = hash_program(&program).unwrap().defs.into_iter().map(|c| (c.name, c.hash.to_hex())).collect();
        rehashed.sort();
        assert_eq!(rehashed, hashes(&mut s));
        // Every view is the canonical text and defines back to the same hash.
        for (name, h) in hashes(&mut s) {
            let t = text(&mut s, &name);
            assert_eq!(qal_text::print_program(&qal_text::parse_program(&t).unwrap()).trim_end(), t);
            assert_eq!(ok(&mut s, &t)[0].hash, h);
        }
    }
}

#[test]
fn concurrent_defines_serialize() {
    let dir = project();
    Store::init(&dir).unwrap();
    let path = dir.join(STORE_DIR).join(STORE_FILE);
    let workers: Vec<_> = (0..2)
        .map(|w| {
            let path = path.clone();
            std::thread::spawn(move || {
                let mut s = Store::open(&path).unwrap();
                for i in 0..10 {
                    let src = format!("fn w{w}_{i}() -> Int = {}\nfn shared() -> Int = {}", w * 100 + i, w * 100 + i);
                    s.define(&src, &format!("worker-{w}")).unwrap_or_else(|d| panic!("{d:#?}"));
                }
            })
        })
        .collect();
    for w in workers {
        w.join().unwrap();
    }
    let mut s = Store::open(&path).unwrap();
    assert_eq!(hashes(&mut s).len(), 21);
    assert_eq!(s.history("shared").unwrap().len(), 20);
}

#[test]
fn corrupt_stores_are_errors_not_panics() {
    let dir = project();
    std::fs::create_dir_all(dir.join(STORE_DIR)).unwrap();
    let path = dir.join(STORE_DIR).join(STORE_FILE);
    std::fs::write(&path, b"this is not a database, just text that is long enough to look like one.....").unwrap();
    assert_eq!(Store::open(&path).err().unwrap()[0].kind, "store_error");
    assert_eq!(Store::discover(&dir.join("sub")).err().unwrap()[0].kind, "store_error");
    assert!(Store::open(&dir.join("missing.db")).err().unwrap()[0].message.contains("no QAL store"));
    // A definition whose bytes no longer match its hash.
    let mut s = store();
    ok(&mut s, "fn f() -> Int = 1\nfn g() -> Int = f()");
    raw(&s).execute("UPDATE defs SET canonical = x'51414c0001000100' WHERE hash = (SELECT hash FROM names WHERE name = 'f')", []).unwrap();
    for d in [s.view(&["g".to_string()]).unwrap_err(), err(&mut s, "fn h() -> Int = 2"), s.program().unwrap_err()] {
        assert_eq!(d[0].kind, "store_error");
        assert!(d[0].message.contains("does not match its hash"), "{}", d[0].message);
    }
    // A name pointing nowhere, and a store from another version.
    let mut s = store();
    ok(&mut s, "fn f() -> Int = 1");
    let c = raw(&s);
    c.pragma_update(None, "foreign_keys", false).unwrap();
    c.execute("UPDATE names SET hash = 'ab' WHERE name = 'f'", []).unwrap();
    assert_eq!(s.program().unwrap_err()[0].kind, "store_error");
    c.pragma_update(None, "user_version", 99).unwrap();
    assert!(Store::open(s.path()).err().unwrap()[0].message.contains("found version 99"));
}

#[test]
fn a_failure_while_writing_rolls_everything_back() {
    let mut s = store();
    ok(&mut s, "fn base() -> Int = 1\nfn user() -> Int = base()");
    raw(&s).execute_batch("CREATE TRIGGER boom BEFORE INSERT ON names WHEN NEW.name = 'boom' BEGIN SELECT RAISE(ABORT, 'boom'); END;").unwrap();
    let before = (hashes(&mut s), count(&s, "defs"), count(&s, "deps"), count(&s, "history"));
    let d = err(&mut s, "fn base() -> Int = 2\nfn fresh() -> Int = 3\nfn boom() -> Int = 4");
    assert_eq!(d[0].kind, "store_error");
    assert!(d[0].message.contains("boom"), "{}", d[0].message);
    assert_eq!((hashes(&mut s), count(&s, "defs"), count(&s, "deps"), count(&s, "history")), before);
}

#[test]
fn init_is_idempotent_and_discover_searches_parents() {
    let dir = project();
    let mut s = Store::init(&dir).unwrap();
    ok(&mut s, "fn f() -> Int = 1");
    drop(s);
    let mut again = Store::init(&dir).unwrap();
    assert_eq!(hashes(&mut again).len(), 1);
    let sub = dir.join("a").join("b");
    std::fs::create_dir_all(&sub).unwrap();
    let mut found = Store::discover(&sub).unwrap();
    assert_eq!(hashes(&mut found).len(), 1);
}

#[test]
fn replacing_one_name_of_a_shared_hash_keeps_the_other_users() {
    let mut s = store();
    ok(&mut s, "fn one() -> Int = 1\nfn uno() -> Int = 1\nfn user() -> Int = one() + 1\nfn user2() -> Int = uno() * 2");
    assert_eq!(hash(&mut s, "one"), hash(&mut s, "uno"));
    let before_user = hash(&mut s, "user");
    // `one` keeps the old hash, so nothing that uses that hash changes (whichever name it was written with).
    let d = ok(&mut s, "fn uno() -> Int = 10");
    assert_eq!(statuses(&d), vec![("uno", "replaced"), ("user2", "relinked")]);
    assert_eq!(hash(&mut s, "user"), before_user);
    assert_eq!(text(&mut s, "user2"), "fn user2() -> Int\n= uno() * 2");
    // Users follow the name they were written against: only `user` is left on `one`.
    let d = err(&mut s, "fn one(s: Str) -> Int = len(s)");
    assert_eq!((d[0].kind, d[0].dependents.clone()), ("signature_change", vec!["user".to_string()]));
    // A different signature for one name of a shared hash is not a change for the users of the hash.
    let mut s = store();
    ok(&mut s, "fn one() -> Int = 1\nfn uno() -> Int = 1\nfn user() -> Int = one() + 1");
    assert_eq!(statuses(&ok(&mut s, "fn uno(s: Str) -> Int = len(s)")), vec![("uno", "replaced")]);
    // Once every name of a hash is redefined, its users follow (and need a compatible signature).
    let mut s = store();
    ok(&mut s, "fn one() -> Int = 1\nfn uno() -> Int = 1\nfn user() -> Int = one() + 1");
    let d = ok(&mut s, "fn one() -> Int = 5\nfn uno() -> Int = 5");
    assert_eq!(statuses(&d), vec![("one", "replaced"), ("uno", "replaced"), ("user", "relinked")]);
    assert_eq!(text(&mut s, "user"), "fn user() -> Int\n= one() + 1");
}

#[test]
fn interchangeable_recursive_definitions_are_rejected() {
    let mut s = store();
    let d = err(&mut s, "fn f(n: Int) -> Int = g(n)\nfn g(n: Int) -> Int = f(n)");
    assert_eq!((d[0].code, d[0].kind), ("E0209", "duplicate_name"));
    assert_eq!(count(&s, "names"), 0);
    // An asymmetric cycle is fine; every member can be renamed without changing any hash.
    ok(&mut s, "fn a(n: Int) -> Int = if n == 0 then 0 else b(n - 1)\nfn b(n: Int) -> Int = c(n)\nfn c(n: Int) -> Int = a(n)");
    let before: Vec<String> = hashes(&mut s).into_iter().map(|(_, h)| h).collect();
    for (old, new) in [("a", "z"), ("b", "y"), ("c", "x")] {
        s.rename(old, new, "test").unwrap();
    }
    let after: Vec<String> = hashes(&mut s).into_iter().map(|(_, h)| h).rev().collect();
    assert_eq!(after, before);
    assert_eq!(count(&s, "groups"), 1);
    // The same cycle written in another order and with other names has the same hashes.
    let mut t = store();
    ok(&mut t, "fn q(k: Int) -> Int = p(k)\nfn p(k: Int) -> Int = if k == 0 then 0 else r2(k - 1)\nfn r2(k: Int) -> Int = q(k)");
    assert_eq!(hash(&mut t, "p"), hash(&mut s, "z"));
    assert_eq!(hash(&mut t, "r2"), hash(&mut s, "y"));
}

#[test]
fn large_recursive_groups_are_stored_once_and_stay_fast() {
    let mut src = String::from("fn f0(n: Int) -> Int = if n == 0 then 0 else f1(n - 1)\n");
    for i in 1..200 {
        src += &format!("fn f{i}(n: Int) -> Int = f{}(n)\n", (i + 1) % 200);
    }
    let mut s = store();
    ok(&mut s, &src);
    assert_eq!(count(&s, "groups"), 1);
    let biggest: i64 = raw(&s).query_row("SELECT max(length(canonical)) FROM defs", [], |r| r.get(0)).unwrap();
    assert!(biggest < 64, "member forms name their group instead of containing it ({biggest} bytes)");
    let start = std::time::Instant::now();
    ok(&mut s, "fn unrelated() -> Int = 1");
    assert_eq!(hashes(&mut s).len(), 201);
    assert!(start.elapsed() < std::time::Duration::from_secs(10), "took {:?}", start.elapsed());
    assert_eq!(s.view(&["f7".to_string()]).unwrap()[0].text, "fn f7(n: Int) -> Int\n= f8(n)");
}

#[test]
fn definitions_at_the_parser_nesting_limit_round_trip() {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            let mut s = store();
            let src = format!("test deep = {}1 == 1", "-".repeat(250));
            let h = ok(&mut s, &src)[0].hash.clone();
            let t = text(&mut s, "deep");
            assert_eq!(ok(&mut s, &t)[0].hash, h);
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn tampered_effect_sets_are_store_errors() {
    let mut s = store();
    ok(&mut s, "fn f() -> Unit !io !fs = ()");
    let c = raw(&s);
    let blob: Vec<u8> = c.query_row("SELECT canonical FROM defs", [], |r| r.get(0)).unwrap();
    // Swap the two effects (`fs`, `io` -> `io`, `fs`) and store the result under its own hash.
    let i = blob.windows(2).position(|w| w == b"fs").unwrap();
    let mut bad = blob.clone();
    bad[i..i + 2].copy_from_slice(b"io");
    bad[i + 3..i + 5].copy_from_slice(b"fs");
    let h = Hash::of(&bad).to_hex();
    c.execute("INSERT INTO defs SELECT ?1, kind, ?2, signature, effects, local_names, created_at FROM defs", rusqlite::params![h, bad]).unwrap();
    c.execute("UPDATE names SET hash = ?1", rusqlite::params![h]).unwrap();
    let d = s.query(&Query::default()).unwrap_err();
    assert!(d[0].kind == "store_error" && d[0].message.contains("sorted set"), "{d:#?}");
}

#[test]
fn a_recursive_group_member_cannot_get_a_second_name() {
    let mut s = store();
    ok(&mut s, "fn a(n: Int) -> Int = if n <= 0 then 1 else b(n - 1)\nfn b(n: Int) -> Int = if n <= 1 then 0 else a(n - 2)");
    let d = err(&mut s, "fn c(n: Int) -> Int = if n <= 0 then 1 else d(n - 1)\nfn d(n: Int) -> Int = if n <= 1 then 0 else c(n - 2)");
    assert_eq!((d[0].kind, d[0].message.as_str()), ("duplicate_name", "`c` is identical to the recursive definition `a`"));
    // The store keeps working.
    ok(&mut s, "fn zz() -> Int = 1");
    s.rename("a", "x", "test").unwrap();
}

#[test]
fn restating_an_unchanged_alias_keeps_its_users() {
    let mut s = store();
    ok(&mut s, "fn f(n: Int) -> Int = n + 1\nfn g(n: Int) -> Int = n + 1\nfn user(x: Int) -> Int = g(x) * 2\ntest t = user(1) == 4");
    let before = hash(&mut s, "user");
    let d = ok(&mut s, "fn g(n: Int) -> Int = n + 1\nfn f(n: Int) -> Int = n + 3");
    assert_eq!(statuses(&d), vec![("g", "unchanged"), ("f", "replaced")]);
    assert_eq!(hash(&mut s, "user"), before);
    assert_eq!(text(&mut s, "user"), "fn user(x: Int) -> Int\n= g(x) * 2");
    // Both names changed, differently: the user follows the name it was written against.
    let mut s = store();
    ok(&mut s, "fn f(n: Int) -> Int = n + 1\nfn g(n: Int) -> Int = n + 1\nfn user(x: Int) -> Int = g(x) * 2");
    let d = ok(&mut s, "fn f(n: Int) -> Int = n + 3\nfn g(n: Int) -> Int = n + 4");
    assert_eq!(statuses(&d), vec![("f", "replaced"), ("g", "replaced"), ("user", "relinked")]);
    assert_eq!(text(&mut s, "user"), "fn user(x: Int) -> Int\n= g(x) * 2");
}

#[test]
fn an_accidental_alias_does_not_retarget_users() {
    let mut s = store();
    ok(&mut s, "fn a() -> Int = 1\nfn b() -> Int = 2\nfn ua() -> Int = a() + 1\nfn ub() -> Int = b() + 1\nfn top() -> Int = ub()");
    let top = hash(&mut s, "top");
    let d = ok(&mut s, "fn a() -> Int = 2");
    assert_eq!(statuses(&d), vec![("a", "replaced"), ("ua", "relinked")]);
    // `a` now has the hash of `b`, and `ua` that of `ub`; every text still reads as written.
    assert_eq!(hash(&mut s, "a"), hash(&mut s, "b"));
    assert_eq!(hash(&mut s, "ua"), hash(&mut s, "ub"));
    assert_eq!(hash(&mut s, "top"), top);
    assert_eq!(text(&mut s, "top"), "fn top() -> Int\n= ub()");
    assert_eq!(text(&mut s, "ub"), "fn ub() -> Int\n= b() + 1");
    assert_eq!(text(&mut s, "ua"), "fn ua() -> Int\n= a() + 1");
    let (ua, ub, a) = (hash(&mut s, "ua"), hash(&mut s, "ub"), hash(&mut s, "a"));
    // Replacing `b` moves the users of `b` (and theirs), not the users of `a`.
    let d = ok(&mut s, "fn b() -> Int = 3");
    assert_eq!(statuses(&d), vec![("b", "replaced"), ("top", "relinked"), ("ub", "relinked")]);
    assert_eq!((hash(&mut s, "ua"), hash(&mut s, "a")), (ua.clone(), a));
    assert_ne!(hash(&mut s, "ub"), ub);
    assert_ne!(hash(&mut s, "top"), top);
    assert_eq!(text(&mut s, "ub"), "fn ub() -> Int\n= b() + 1");
    assert_eq!(text(&mut s, "ua"), "fn ua() -> Int\n= a() + 1");
    // `ub` now has the shape of `b = 3`: viewing it shows `b` as its dependency.
    let v = s.view(&["ub".to_string()]).unwrap().remove(0);
    assert_eq!(v.deps.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(), vec!["b"]);
    let v = s.view(&["ua".to_string()]).unwrap().remove(0);
    assert_eq!(v.deps.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(), vec!["a"]);
    assert_eq!(s.query(&Query { uses: Some("a".into()), ..Query::default() }).unwrap().into_iter().map(|e| e.name).collect::<Vec<_>>(), vec!["ua"]);
    assert_eq!(s.program().unwrap().defs.len(), 5);
}

#[test]
fn a_deliberate_alias_relinks_only_its_own_users() {
    let mut s = store();
    ok(&mut s, "fn a() -> Int = 1\nfn b() -> Int = 1\nfn ua() -> Int = a() + 1\nfn ub() -> Int = b() + 1");
    assert_eq!(hash(&mut s, "a"), hash(&mut s, "b"));
    assert_eq!(hash(&mut s, "ua"), hash(&mut s, "ub"));
    let before = hash(&mut s, "ub");
    let d = ok(&mut s, "fn a() -> Int = 7");
    assert_eq!(statuses(&d), vec![("a", "replaced"), ("ua", "relinked")]);
    assert_eq!(hash(&mut s, "ub"), before);
    assert_eq!(text(&mut s, "ub"), "fn ub() -> Int\n= b() + 1");
    assert_eq!(text(&mut s, "ua"), "fn ua() -> Int\n= a() + 1");
}

#[test]
fn renamed_names_keep_their_users_attached() {
    let mut s = store();
    ok(&mut s, "fn a() -> Int = 1\nfn user() -> Int = a() + 1");
    s.rename("a", "alpha", "test").unwrap();
    assert_eq!(text(&mut s, "user"), "fn user() -> Int\n= alpha() + 1");
    let d = ok(&mut s, "fn alpha() -> Int = 5");
    assert_eq!(statuses(&d), vec![("alpha", "replaced"), ("user", "relinked")]);
    assert_eq!(text(&mut s, "user"), "fn user() -> Int\n= alpha() + 1");
}

#[test]
fn large_groups_load_in_linear_time() {
    let n = 1000;
    let mut src = String::from("fn f0(n: Int) -> Int = if n == 0 then 0 else f1(n - 1)\n");
    for i in 1..n {
        src += &format!("fn f{i}(n: Int) -> Int = f{}(n)\n", (i + 1) % n);
    }
    let mut s = store();
    ok(&mut s, &src);
    // Loading the group is linear: reading the whole codebase and viewing members is fast. (Hashing
    // the group again on `define` is `hash_program`'s cost and is not part of this bound.)
    let start = std::time::Instant::now();
    let all = s.query(&Query::default()).unwrap();
    assert_eq!(all.len(), n);
    assert_eq!(s.program().unwrap().defs.len(), n);
    for name in ["f0", "f500", "f999"] {
        assert!(s.view(&[name.to_string()]).unwrap()[0].text.starts_with(&format!("fn {name}(n: Int) -> Int")));
    }
    assert!(start.elapsed() < std::time::Duration::from_secs(2), "took {:?}", start.elapsed());
    let start = std::time::Instant::now();
    ok(&mut s, "fn unrelated() -> Int = 1");
    assert!(start.elapsed() < std::time::Duration::from_secs(20), "took {:?}", start.elapsed());
}

#[test]
fn rename_history_records_both_names() {
    let mut s = store();
    ok(&mut s, "fn g() -> Int = 1");
    s.rename("g", "h", "agent-1").unwrap();
    s.rename("h", "g", "agent-2").unwrap();
    let h = s.history("g").unwrap();
    assert_eq!(h.iter().map(|e| e.action.as_str()).collect::<Vec<_>>(), vec!["added", "renamed", "renamed"]);
    let renames: Vec<(Option<&str>, Option<&str>)> = h[1..].iter().map(|e| (e.from_name.as_deref(), e.to_name.as_deref())).collect();
    assert_eq!(renames, vec![(Some("g"), Some("h")), (Some("h"), Some("g"))]);
    assert_eq!((h[0].from_name.clone(), h[0].to_name.clone()), (None, None));
}

#[test]
fn init_validates_before_changing_anything() {
    let mode = |path: &std::path::Path| -> String { Connection::open(path).unwrap().query_row("PRAGMA journal_mode", [], |r| r.get(0)).unwrap() };
    // A store of another version is left alone.
    let dir = project();
    std::fs::create_dir_all(dir.join(STORE_DIR)).unwrap();
    let path = dir.join(STORE_DIR).join(STORE_FILE);
    let c = Connection::open(&path).unwrap();
    c.execute_batch("CREATE TABLE names (x); PRAGMA user_version = 99;").unwrap();
    drop(c);
    let d = Store::init(&dir).err().unwrap();
    assert!(d[0].kind == "store_error" && d[0].message.contains("found version 99"), "{d:#?}");
    assert_eq!(mode(&path), "delete");
    // A foreign database is refused and gets no QAL tables.
    let dir = project();
    std::fs::create_dir_all(dir.join(STORE_DIR)).unwrap();
    let path = dir.join(STORE_DIR).join(STORE_FILE);
    let c = Connection::open(&path).unwrap();
    c.execute_batch("CREATE TABLE notes (x);").unwrap();
    drop(c);
    let d = Store::init(&dir).err().unwrap();
    assert!(d[0].kind == "store_error" && d[0].message.contains("not a QAL store"), "{d:#?}");
    let c = Connection::open(&path).unwrap();
    let tables: i64 = c.query_row("SELECT count(*) FROM sqlite_master WHERE name IN ('names', 'defs', 'history')", [], |r| r.get(0)).unwrap();
    assert_eq!(tables, 0);
    assert_eq!(mode(&path), "delete");
    // An empty file is fine.
    let dir = project();
    std::fs::create_dir_all(dir.join(STORE_DIR)).unwrap();
    std::fs::write(dir.join(STORE_DIR).join(STORE_FILE), b"").unwrap();
    let mut s = Store::init(&dir).unwrap();
    assert_eq!(hashes(&mut s).len(), 0);
    assert_eq!(mode(s.path()), "wal");
}

#[test]
fn a_definition_cannot_name_two_names_of_one_definition() {
    // The defining call itself: `a` and `b` are the same definition.
    let mut s = store();
    let d = err(&mut s, "fn a() -> Int = 1\nfn b() -> Int = 1\nfn u() -> Int = a() * 10 + b()");
    assert_eq!((d[0].code, d[0].kind, d[0].def.as_str()), ("E0404", "alias_reference", "u"));
    assert!(d[0].message.starts_with("`u` refers to `a` and `b`, which are the same definition (#"), "{}", d[0].message);
    assert!(d[0].message.ends_with("a stored definition can name each referenced definition only once."), "{}", d[0].message);
    assert_eq!(d[0].hints, ["refer to only one of `a`, `b` (they are identical)"]);
    assert_eq!(count(&s, "names"), 0);
    // A later change that makes two referenced names identical is rejected, and nothing changes.
    ok(&mut s, "fn a() -> Int = 1\nfn b() -> Int = 2\nfn u() -> Int = a() * 10 + b()");
    let before = hashes(&mut s);
    let d = err(&mut s, "fn a() -> Int = 2");
    assert_eq!((d[0].code, d[0].kind, d[0].def.as_str()), ("E0404", "alias_reference", "u"));
    assert_eq!(d[0].hints, ["refer to one of them, or make them differ (redefine `u` in the same call)"]);
    assert_eq!(hashes(&mut s), before);
    assert_eq!(text(&mut s, "u"), "fn u() -> Int\n= a() * 10 + b()");
    // Redefining `u` in the same call is the way out.
    ok(&mut s, "fn a() -> Int = 2\nfn u() -> Int = a() * 10 + a()");
}

#[test]
fn alias_reference_hint_does_not_ask_to_redefine_a_definition_already_in_the_call() {
    let mut s = store();
    let d = err(&mut s, "fn mk() -> Int = 1\nfn mk2() -> Int = 1\nfn uses_mk() -> Int = mk() + mk2()");
    assert_eq!((d[0].code, d[0].def.as_str()), ("E0404", "uses_mk"));
    assert_eq!(d[0].hints, ["refer to only one of `mk`, `mk2` (they are identical)"]);
}

#[test]
fn a_name_cannot_change_its_kind() {
    let mut s = store();
    ok(&mut s, "fn check() -> Bool = true\ntest t = check()");
    let before = hashes(&mut s);
    let d = err(&mut s, "test check = true");
    assert_eq!((d[0].code, d[0].kind, d[0].def.as_str()), ("E0405", "kind_change", "check"));
    assert_eq!((d[0].expected.as_deref(), d[0].found.as_deref()), (Some("fn"), Some("test")));
    let d = err(&mut s, "fn t() -> Bool = false");
    assert_eq!((d[0].kind, d[0].message.as_str()), ("kind_change", "`t` is a test in the store and cannot be redefined as a fn"));
    assert_eq!(hashes(&mut s), before, "nothing changed");
}

#[test]
fn old_names_point_to_the_new_name() {
    let mut s = store();
    ok(&mut s, "fn one() -> Int = 1");
    s.rename("one", "uno", "a").unwrap();
    s.rename("uno", "eins", "a").unwrap();
    assert_eq!(s.renamed_to("one").unwrap(), Some("eins".to_string()));
    assert_eq!(s.renamed_to("uno").unwrap(), Some("eins".to_string()));
    assert_eq!(s.renamed_to("eins").unwrap(), None);
    assert_eq!(s.renamed_to("never").unwrap(), None);
    let hint = vec!["`one` was renamed to `eins`".to_string()];
    let d = s.view(&["one".to_string()]).unwrap_err();
    assert_eq!((d[0].kind, &d[0].hints), ("unknown_name", &hint));
    assert_eq!(s.history("one").unwrap_err()[0].hints, hint);
    assert_eq!(s.rename("one", "x", "a").unwrap_err()[0].hints, hint);
    assert_eq!(s.query(&Query { uses: Some("one".into()), ..Query::default() }).unwrap_err()[0].hints, hint);
    // Renames back and forth end at the name that exists now.
    ok(&mut s, "fn two() -> Int = 2");
    s.rename("two", "b", "a").unwrap();
    s.rename("b", "c", "a").unwrap();
    s.rename("c", "b", "a").unwrap();
    assert_eq!(s.renamed_to("two").unwrap(), Some("b".to_string()));
    assert_eq!(s.renamed_to("c").unwrap(), Some("b".to_string()));
    let long = "x".repeat(129);
    assert_eq!(s.rename("b", &long, "a").unwrap_err()[0].kind, "parse");
    // A name that was renamed away and then reused still reports where the renamed definition went
    // (`renamed_to` is history-based); `view` of the name shows the new definition, not the hint.
    ok(&mut s, "fn one() -> Int = 11");
    assert_eq!(s.renamed_to("one").unwrap(), Some("eins".to_string()));
    assert_eq!(text(&mut s, "one"), "fn one() -> Int\n= 11");
}

#[test]
fn invalid_signature_patterns_are_diagnostics() {
    let mut s = store();
    ok(&mut s, "fn f(x: Int) -> Int = x");
    let d = s.query(&Query { type_pattern: Some("fn(".into()), ..Query::default() }).unwrap_err();
    assert_eq!((d[0].code, d[0].kind, d[0].message.as_str()), ("E0102", "bad_pattern", "invalid signature pattern `fn(`: unclosed `(`"));
    assert_eq!(s.query(&Query { type_pattern: Some("fn(_) -> _".into()), ..Query::default() }).unwrap().len(), 1);
}

#[test]
fn definitions_named_like_new_builtins_can_be_renamed_one_by_one() {
    // A store from before `replace` and `to_upper` were built-ins: simulated by renaming in SQL.
    let dir = project();
    let mut s = Store::init(&dir).unwrap();
    ok(&mut s, "fn rep(s: Str) -> Str = s ++ \"!\"\nfn up(s: Str) -> Str = s\nfn shout(s: Str) -> Str = up(rep(s))\ntest t = shout(\"a\") == \"a!\"");
    let before = hash(&mut s, "shout");
    drop(s);
    let db = dir.join(".qal").join("store.db");
    let conn = rusqlite::Connection::open(&db).unwrap();
    conn.execute_batch(
        "UPDATE names SET name = 'replace' WHERE name = 'rep'; UPDATE names SET name = 'to_upper' WHERE name = 'up';\
         UPDATE names SET refs = replace(replace(refs, '\"rep\"', '\"replace\"'), '\"up\"', '\"to_upper\"');",
    )
    .unwrap();
    drop(conn);
    let mut s = Store::open(&db).unwrap();
    let d = err(&mut s, "fn helper() -> Int = 1");
    assert_eq!((d[0].kind, d[0].message.as_str()), ("duplicate_name", "`replace` is the name of a built-in function"));
    assert!(d[0].hints[0].contains("qal_rename"), "{:?}", d[0].hints);
    s.rename("replace", "bang", "test").unwrap();
    s.rename("to_upper", "same", "test").unwrap();
    ok(&mut s, "fn helper() -> Int = 1");
    assert_eq!(hash(&mut s, "shout"), before);
    assert_eq!(text(&mut s, "shout"), "fn shout(s: Str) -> Str\n= same(bang(s))");
    // Renaming onto a built-in name is still refused.
    assert_eq!(s.rename("bang", "index_of", "test").unwrap_err()[0].message, "`index_of` is the name of a built-in function");
}

// ---------------------------------------------------------------------------------------------
// Phase 2a: generic definitions, closures, maps and bytes in the store

const PHASE2A: &str = "type Pair[A, B] = { fst: A, snd: B }\n\
    type Tree[T] = Leaf | Node(Tree[T], T, Tree[T])\n\
    fn swap[A, B](p: Pair[A, B]) -> Pair[B, A] = Pair { fst: p.snd, snd: p.fst }\n\
    fn size[T](t: Tree[T]) -> Int = match t { Leaf => 0, Node(l, _, r) => size(l) + 1 + size(r) }\n\
    fn adder(n: Int) -> fn(Int) -> Int = fn(x) => x + n\n\
    fn tally(ws: List[Str]) -> Map[Str, Int] = fold(ws, map_new(), fn(m, w) => map_put(m, w, 1))\n\
    fn raw(s: Str) -> Bytes = to_bytes(s)\n\
    test t = size(Node(Leaf, swap(Pair { fst: 1, snd: true }), Leaf)) == 1";

#[test]
fn generic_and_closure_definitions_round_trip_through_the_store() {
    let mut s = store();
    ok(&mut s, PHASE2A);
    assert_eq!(text(&mut s, "swap"), "fn swap[A, B](p: Pair[A, B]) -> Pair[B, A]\n= Pair { fst: p.snd, snd: p.fst }");
    assert_eq!(text(&mut s, "adder"), "fn adder(n: Int) -> (fn(Int) -> Int)\n= fn(x) => x + n");
    assert_eq!(text(&mut s, "Tree"), "type Tree[T] = Leaf | Node(Tree[T], T, Tree[T])");
    let entries = s.query(&Query::default()).unwrap();
    let sig = |n: &str| entries.iter().find(|e| e.name == n).unwrap().signature.clone();
    assert_eq!(sig("swap"), "fn[A, B](Pair[A, B]) -> Pair[B, A]");
    // The whole stored codebase rehashes to its stored hashes and checks again.
    let p = s.program().unwrap();
    qal_check::check(&p).unwrap();
    let rehashed = hash_program(&p).unwrap();
    for c in rehashed.defs {
        assert_eq!(c.hash.to_hex(), hash(&mut s, &c.name), "{}", c.name);
    }
}

#[test]
fn renaming_type_parameters_and_closure_parameters_changes_no_hash() {
    let mut s = store();
    ok(&mut s, PHASE2A);
    let before = hashes(&mut s);
    let renamed = PHASE2A
        .replace("fn swap[A, B](p: Pair[A, B]) -> Pair[B, A]", "fn swap[X, Y](q: Pair[X, Y]) -> Pair[Y, X]")
        .replace("Pair { fst: p.snd, snd: p.fst }", "Pair { fst: q.snd, snd: q.fst }")
        .replace("fn(x) => x + n", "fn(y) => y + n")
        .replace("type Tree[T] = Leaf | Node(Tree[T], T, Tree[T])", "type Tree[E] = Leaf | Node(Tree[E], E, Tree[E])");
    let d = ok(&mut s, &renamed);
    assert!(d.iter().all(|d| d.status == "unchanged"), "{d:#?}");
    assert_eq!(hashes(&mut s), before);
    // The text shows the latest local names.
    assert!(text(&mut s, "swap").starts_with("fn swap[X, Y](q: Pair[X, Y])"));
}

#[test]
fn phase1_definitions_keep_their_hashes() {
    // Hash of `fn f(x: Int) -> Int = x` under the Phase 1 canonical format (version 1).
    let mut s = store();
    ok(&mut s, "fn f(x: Int) -> Int = x");
    let p = qal_text::parse_program("fn f(x: Int) -> Int = x").unwrap();
    let c = &hash_program(&p).unwrap().defs[0];
    assert_eq!(c.canonical[4], 1);
    assert_eq!(hash(&mut s, "f"), c.hash.to_hex());
}

#[test]
fn version_2_stores_are_upgraded_and_keep_proof_results() {
    let mut s = store();
    ok(&mut s, "fn f(x: Int) -> Int = x");
    // Downgrade to the Phase 1 layout, then open again.
    let c = raw(&s);
    c.execute_batch("DROP TABLE proofs; DROP TABLE meta; PRAGMA user_version = 2;").unwrap();
    drop(c);
    let path = s.path().to_path_buf();
    drop(s);
    // 2 -> 3 (proofs) -> 4 (meta) in one migration.
    let mut s = Store::open(&path).unwrap();
    assert_eq!(raw(&s).query_row("PRAGMA user_version", [], |r| r.get::<_, i64>(0)).unwrap(), 4);
    assert_eq!(s.meta("x").unwrap(), None);
    let h = hash(&mut s, "f");
    assert_eq!(s.proofs(&h, "p1").unwrap(), None);
    s.save_proofs(&h, "p1", "[1]").unwrap();
    s.save_proofs(&h, "p1", "[2]").unwrap();
    assert_eq!(s.proofs(&h, "p1").unwrap().as_deref(), Some("[2]"));
    assert_eq!(s.proofs(&h, "p2").unwrap(), None);
    // Results only exist for stored definitions.
    assert!(s.save_proofs(&"0".repeat(64), "p1", "[]").is_err());
}

#[test]
fn verification_runs_inside_define_and_can_reject_it() {
    let mut s = store();
    let mut seen: Vec<Vec<String>> = Vec::new();
    let verify_with = |s: &mut Store, src: &str, reject: Option<&str>, seen: &mut Vec<Vec<String>>| {
        let mut run = |tp: &qal_core::tast::TProgram, todo: &[(String, String)]| {
            assert!(todo.iter().all(|(n, _)| tp.fn_index(n).is_some() || tp.tests.iter().any(|t| &t.name == n)));
            seen.push(todo.iter().map(|(n, _)| n.clone()).collect());
            match reject {
                Some(n) if todo.iter().any(|(m, _)| m == n) => Err(vec![Diagnostic::new("E0801", "contract_refuted", n, "ens[0]", "refuted")]),
                _ => Ok(todo.iter().map(|(n, h)| (h.clone(), format!("[\"{n}\"]"))).collect()),
            }
        };
        let mut v = Verify { key: "k1".into(), run: &mut run };
        s.define_verified(src, "test", &mut v)
    };
    verify_with(&mut s, "type P = { x: Int }\nfn f(x: Int) -> Int = x\nfn g(x: Int) -> Int = f(x)\ntest t = g(1) == 1", None, &mut seen).unwrap();
    assert_eq!(seen.pop().unwrap(), ["f", "g", "t"], "types are not verified");
    let (f, g) = (hash(&mut s, "f"), hash(&mut s, "g"));
    assert_eq!(s.proofs(&f, "k1").unwrap().as_deref(), Some("[\"f\"]"));
    // Unchanged definitions with results are not verified again; relinked dependents are.
    verify_with(&mut s, "fn f(x: Int) -> Int = x + 0", None, &mut seen).unwrap();
    assert_eq!(seen.pop().unwrap(), ["f", "g", "t"]);
    assert_ne!(hash(&mut s, "g"), g);
    verify_with(&mut s, "fn f(x: Int) -> Int = x + 0", None, &mut seen).unwrap();
    assert!(seen.is_empty(), "nothing changed: the verifier is not called");
    // A rejection stores nothing: no definition, no name, no result.
    let before = hashes(&mut s);
    let d = verify_with(&mut s, "fn f(x: Int) -> Int = x + 1\nfn h(x: Int) -> Int = x", Some("g"), &mut seen).unwrap_err();
    assert_eq!(d[0].kind, "contract_refuted");
    assert_eq!(hashes(&mut s), before);
    assert_eq!(count(&s, "proofs"), 6);
    // Unchanged definitions without a result for this key are verified.
    let mut run = |_: &qal_core::tast::TProgram, todo: &[(String, String)]| Ok(todo.iter().map(|(_, h)| (h.clone(), "[]".to_string())).collect());
    let mut v = Verify { key: "k2".into(), run: &mut run };
    let out = s.define_verified("fn g(x: Int) -> Int = f(x)", "test", &mut v).unwrap();
    assert_eq!(out[0].status, "unchanged");
    let g = hash(&mut s, "g");
    assert_eq!(s.proofs(&g, "k2").unwrap().as_deref(), Some("[]"));
    assert_eq!(s.proofs(&f, "k2").unwrap(), None);
}

#[test]
fn verification_runs_without_the_write_lock() {
    // Review finding I4: the prover ran inside the IMMEDIATE transaction for up to its whole budget,
    // longer than other writers wait (10 s), so concurrent `define`s failed with "database is locked".
    let mut s = store();
    let path = s.path().to_path_buf();
    let mut calls = 0;
    let mut run = |_: &qal_core::tast::TProgram, todo: &[(String, String)]| {
        calls += 1;
        let other = rusqlite::Connection::open(&path).unwrap();
        other.busy_timeout(std::time::Duration::from_millis(100)).unwrap();
        other.execute_batch("BEGIN IMMEDIATE; ROLLBACK;").map_err(|e| vec![Diagnostic::store_error(format!("{e}"))])?;
        Ok(todo.iter().map(|(_, h)| (h.clone(), "[]".to_string())).collect())
    };
    let mut v = Verify { key: "k".into(), run: &mut run };
    let out = s.define_verified("fn f(x: Int) -> Int = x", "test", &mut v).unwrap_or_else(|d| panic!("{d:#?}"));
    assert_eq!((calls, out[0].status), (1, "added"));
    let f = hash(&mut s, "f");
    assert_eq!(s.proofs(&f, "k").unwrap().as_deref(), Some("[]"));
}

#[test]
fn verification_starts_over_when_another_writer_changes_its_input() {
    // While `g` is verified on a snapshot, another agent replaces `f`, which `g` uses: `g` gets a
    // new hash, so the results for the old one are not stored and `g` is verified again.
    let mut s = store();
    ok(&mut s, "fn f(x: Int) -> Int = x");
    let path = s.path().to_path_buf();
    let mut seen: Vec<Vec<String>> = Vec::new();
    let mut run = |_: &qal_core::tast::TProgram, todo: &[(String, String)]| {
        if seen.is_empty() {
            Store::open(&path).unwrap().define("fn f(x: Int) -> Int = x + 1", "other").unwrap();
        }
        seen.push(todo.iter().map(|(n, h)| format!("{n}:{h}")).collect());
        Ok(todo.iter().map(|(n, h)| (h.clone(), format!("[\"{n}\"]"))).collect())
    };
    let mut v = Verify { key: "k".into(), run: &mut run };
    let out = s.define_verified("fn g(x: Int) -> Int = f(x)", "test", &mut v).unwrap_or_else(|d| panic!("{d:#?}"));
    assert_eq!(seen.len(), 2, "{seen:?}");
    assert_ne!(seen[0], seen[1]);
    let g = hash(&mut s, "g");
    assert_eq!(seen[1], [format!("g:{g}")]);
    assert_eq!((out[0].hash.as_str(), s.proofs(&g, "k").unwrap().as_deref()), (g.as_str(), Some("[\"g\"]")));
}

#[test]
fn the_program_and_its_hashes_come_from_one_read() {
    // `qal prove` stores results under these hashes (re-review: two separate reads let a
    // concurrent `define` pair the old definition's results with the new hash).
    let mut s = store();
    ok(&mut s, "fn f(x: Int) -> Int = x\nfn g(x: Int) -> Int = f(x)");
    let (p, hashes) = s.program_with_hashes().unwrap();
    let hashed = hash_program(&p).unwrap();
    for c in &hashed.defs {
        assert_eq!(hashes.get(&c.name), Some(&c.hash.to_hex()), "{}", c.name);
    }
    assert_eq!(hashes.len(), 2);
}

#[test]
fn definitions_using_prelude_types_store_and_view() {
    let mut s = store();
    let src = "fn handle(rq: HttpRequest) -> HttpResponse = HttpResponse { status: 200, headers: [HttpHeader { name: \"x\", value: rq.path }], body: \"\" }\n\
               fn cell(v: SqlValue) -> Int = match v { SqlInt(n) => n, _ => 0 }";
    let d = ok(&mut s, src);
    assert_eq!(statuses(&d), vec![("handle", "added"), ("cell", "added")]);
    assert_eq!(count(&s, "deps"), 0, "prelude types are not stored dependencies");
    assert_eq!(text(&mut s, "handle"), "fn handle(rq: HttpRequest) -> HttpResponse\n= HttpResponse { status: 200, headers: [HttpHeader { name: \"x\", value: rq.path }], body: \"\" }");
    assert_eq!(statuses(&ok(&mut s, src)), vec![("handle", "unchanged"), ("cell", "unchanged")]);
    let e = err(&mut s, "type DbRows = { n: Int }");
    assert_eq!(e[0].code, "E0240");
    // A stored type cannot be renamed to a prelude name either.
    ok(&mut s, "type Row = { n: Int }");
    assert_eq!(s.rename("Row", "DbRows", "test").unwrap_err()[0].code, "E0240");
}

#[test]
fn packages_export_and_import_with_identical_hashes() {
    let mut s = store();
    ok(&mut s, "type P = { x: Int }\nfn lib_x(p: P) -> Int = p.x\nfn lib_two() -> Int = lib_x(P { x: 2 }) + helper()\nfn helper() -> Int = 0\nfn other() -> Int = 9\ntest lib_t = lib_two() == 2");
    let e = s.export("lib_", "lib").unwrap();
    assert_eq!((e.package.as_str(), e.defs), ("lib", 5), "the prefix plus dependencies P and helper");
    let lines: Vec<&str> = e.file.lines().collect();
    assert_eq!(lines[0], "# qal-pkg 1");
    assert_eq!(lines[1], format!("# package lib {}", e.hash));
    assert!(lines[2].starts_with("# def P ") && lines[2].ends_with(" test"), "{}", lines[2]);
    assert!(!e.file.contains("fn other"));
    let text = e.file.split("\n\n").skip(1).collect::<Vec<_>>().join("\n\n");
    assert!(text.find("type P").unwrap() < text.find("fn lib_x").unwrap() && text.find("fn helper").unwrap() < text.find("fn lib_two").unwrap(), "dependencies first");
    assert_eq!(s.export("lib_", "lib").unwrap().file, e.file, "deterministic");
    // Into a fresh store: same hashes, recorded origin, package hash in meta.
    let mut t = store();
    let i = t.import(&e.file).unwrap();
    assert_eq!((i.package.as_str(), i.hash.as_str(), i.defined.len()), ("lib", e.hash.as_str(), 5));
    for n in ["P", "lib_x", "lib_two", "helper", "lib_t"] {
        assert_eq!(hash(&mut t, n), hash(&mut s, n), "{n}");
    }
    assert_eq!(t.history("lib_x").unwrap()[0].agent, format!("qal-lib@{}", &e.hash[..12]));
    assert_eq!(t.meta("package:lib").unwrap(), Some(e.hash.clone()));
    assert_eq!(statuses(&t.import(&e.file).unwrap().defined)[0].1, "unchanged", "importing again changes nothing");
    // A changed file: wrong checksum, or right checksum but other hashes -> nothing is written.
    let tampered = e.file.replace("p.x", "p.x + 1");
    let d = t.import(&tampered).unwrap_err();
    assert_eq!((d[0].code, d[0].kind), ("E0243", "package_mismatch"));
    let body = tampered.split_once('\n').unwrap().1.split_once('\n').unwrap().1;
    let resealed = format!("# qal-pkg 1\n# package lib {}\n{body}", Hash::of(body.as_bytes()).to_hex());
    let mut u = store();
    let d = u.import(&resealed).unwrap_err();
    assert!(d[0].code == "E0243" && d[0].message.contains("was recorded with hash"), "{d:?}");
    assert_eq!(count(&u, "names"), 0, "rolled back");
    assert_eq!(u.import("not a package").unwrap_err()[0].code, "E0243");
    assert!(s.export("x", "Bad Name").is_err());
}

#[test]
fn an_empty_package_is_valid_and_old_stores_are_migrated() {
    let mut s = store();
    let e = s.export("std.", "std").unwrap();
    assert_eq!(e.file, format!("# qal-pkg 1\n# package std {}\n", Hash::of(b"").to_hex()));
    assert_eq!(s.import(&e.file).unwrap().defined.len(), 0);
    assert_eq!(s.meta("package:std").unwrap(), Some(Hash::of(b"").to_hex()));
    // A version-3 store (Phase 2b: `proofs`, no `meta`) is upgraded when opened.
    let mut old = store();
    ok(&mut old, "fn f() -> Int = 1");
    let h = hash(&mut old, "f");
    old.save_proofs(&h, "p", "[1]").unwrap();
    let c = raw(&old);
    c.execute_batch("DROP TABLE meta; PRAGMA user_version = 3;").unwrap();
    drop(c);
    let path = old.path().to_path_buf();
    drop(old);
    let mut again = Store::open(&path).unwrap();
    assert_eq!(again.meta("x").unwrap(), None);
    assert_eq!(hashes(&mut again).len(), 1);
    assert_eq!(again.proofs(&h, "p").unwrap().as_deref(), Some("[1]"));
    let v: i64 = raw(&again).query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
    assert_eq!(v, 4);
}

#[test]
fn imports_are_verified_like_define() {
    let mut src = store();
    ok(&mut src, "fn lib_one() -> Int = 1\nfn lib_two(x: Int) -> Int = x + lib_one()");
    let pkg = src.export("lib_", "lib").unwrap();
    // A rejecting verifier rejects the whole import: nothing is written, `meta` stays empty.
    let mut s = store();
    let mut reject = |_: &qal_core::tast::TProgram, _: &[(String, String)]| Err(vec![Diagnostic::store_error("refuted")]);
    let mut v = Verify { key: "k".into(), run: &mut reject };
    assert_eq!(s.import_verified(&pkg.file, &mut v).unwrap_err()[0].message, "refuted");
    assert!(hashes(&mut s).is_empty());
    assert_eq!(s.meta("package:lib").unwrap(), None);
    // An accepting one stores its results with the import.
    let mut seen = Vec::new();
    let mut accept = |_: &qal_core::tast::TProgram, todo: &[(String, String)]| {
        seen.extend(todo.iter().map(|(n, _)| n.clone()));
        Ok(todo.iter().map(|(_, h)| (h.clone(), "[]".to_string())).collect())
    };
    let mut v = Verify { key: "k".into(), run: &mut accept };
    assert_eq!(s.import_verified(&pkg.file, &mut v).unwrap().defined.len(), 2);
    seen.sort();
    assert_eq!(seen, ["lib_one", "lib_two"]);
    let h = hash(&mut s, "lib_two");
    assert_eq!(s.proofs(&h, "k").unwrap().as_deref(), Some("[]"));
    assert_eq!(s.meta("package:lib").unwrap(), Some(pkg.hash));
}

#[test]
fn packages_never_replace_the_projects_own_definitions() {
    let mut evil = store();
    ok(&mut evil, "fn lib_hello() -> Int = check_password(\"x\")\nfn check_password(p: Str) -> Int = 1");
    let pkg = evil.export("lib_", "lib").unwrap();
    let mut victim = store();
    ok(&mut victim, "fn check_password(p: Str) -> Int = if p == \"s3cret\" then 1 else 0");
    let before = hash(&mut victim, "check_password");
    let d = victim.import(&pkg.file).unwrap_err();
    assert!(d[0].code == "E0243" && d[0].message.contains("would replace the project's own `check_password`"), "{d:?}");
    assert_eq!(hash(&mut victim, "check_password"), before);
    // A newer version of the same package may replace what the older one defined.
    let mut lib = store();
    ok(&mut lib, "fn lib_one() -> Int = 1");
    let v1 = lib.export("lib_", "lib").unwrap();
    ok(&mut lib, "fn lib_one() -> Int = 2");
    let v2 = lib.export("lib_", "lib").unwrap();
    let mut user = store();
    user.import(&v1.file).unwrap();
    assert_eq!(statuses(&user.import(&v2.file).unwrap().defined), vec![("lib_one", "replaced")]);
}
