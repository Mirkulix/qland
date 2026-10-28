//! Phase 4 spec §20, risk 3: how long a `define` holds SQLite's write lock in a project of about
//! 150 definitions (plus the standard library), in a team store, in a change set. Target: median
//! at most 300 ms. Run with
//! `cargo test --release -p qal-cli --test lock_time -- --ignored --nocapture`.

use qal_cli::ops::{self, Ctx, StoreAt};
use qal_store::team::keys::KeyDir;
use qal_store::team::policy::Role;
use qal_store::team::Signer;
use std::sync::Arc;
use std::time::Duration;

/// About 150 definitions shaped like a service: records, enums, functions with contracts that
/// call each other, and tests.
fn project_source() -> String {
    let mut src = String::new();
    for i in 0..25 {
        src.push_str(&format!("type Item{i} = {{ id: Int, name: Str, qty: Int, price_cents: Int, kind{i}: Int }}\n"));
        src.push_str(&format!("fn total{i}(x: Item{i}) -> Int req x.qty >= 0 req x.price_cents >= 0 req x.qty <= 1000000 req x.price_cents <= 1000000 ens r >= 0 = x.qty * x.price_cents\n"));
        src.push_str(&format!("fn valid{i}(x: Item{i}) -> Result[Item{i}, Str] = if x.qty < 0 then Err(\"qty\") else if len(x.name) == 0 then Err(\"name\") else Ok(x)\n"));
        src.push_str(&format!("fn json{i}(x: Item{i}) -> Str = \"{{\\\"id\\\":\" ++ from_int(x.id) ++ \",\\\"name\\\":\\\"\" ++ x.name ++ \"\\\"}}\"\n"));
        src.push_str(&format!("fn bump{i}(n: Int) -> Int req n >= 0 req n < 1000 ens r > n = n + 1\n"));
        src.push_str(&format!("test t{i} = total{i}(Item{i} {{ id: 1, name: \"a\", qty: 2, price_cents: 3, kind{i}: 0 }}) == 6\n"));
    }
    src
}

#[test]
#[ignore]
fn define_holds_the_write_lock_briefly() {
    if !qal_compiler::has_qal_driver() {
        let _ = qal_run::selfhost::init_seed_backend(4);
    }
    let home = std::env::temp_dir().join(format!("qal-lock-time-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&home);
    let project = home.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let keys = KeyDir::at(home.join("keys"));
    let r = ops::init(&project, false);
    assert_eq!(r.code, 0, "{:?}", r.out);
    let mut store = qal_store::Store::init(&project).unwrap();
    store.team_init("boss", &keys).unwrap();
    let owner = store.owner_signer(&keys).unwrap();
    let a = store.team_add(&owner, "arch-1", Role::Architect, &keys).unwrap();
    let arch = Signer { member: "arch-1".into(), session: "s000000000001".into(), key: keys.load(&project, &a.key_id).unwrap() };
    let c = store.change_open(&arch, &qal_store::team::change::OpenArgs { title: "lock".into(), ..Default::default() }).unwrap();
    let change = c["change"].as_str().unwrap().to_string();
    let signer = Arc::new(arch);
    let ctx = Ctx { store: StoreAt::Root(project.clone()), stdin: None, timeout: None, test_budget: None, max_stdout: None, signer: Some(signer), space: Some(change), session: Some("s000000000001".into()), replay: false };
    let r = ops::define(&ctx, &project_source(), "x", false);
    assert_eq!(r.code, 0, "{:?}", r.out.diagnostics);
    let names = store.query(&qal_store::Query::default()).unwrap().len();
    let mut times: Vec<Duration> = Vec::new();
    for k in 0..15 {
        // A body change of one function: its users relink (same signature).
        let r = ops::define(&ctx, &format!("fn bump3(n: Int) -> Int req n >= 0 req n < 1000 ens r > n = n + 1 + {}", k % 2), "x", false);
        assert_eq!(r.code, 0, "{:?}", r.out.diagnostics);
        times.push(qal_store::last_lock_held());
    }
    times.sort();
    let median = times[times.len() / 2];
    eprintln!("lock held by define in a team change set ({names} names on main incl. std, +150 in the change set): median {median:?}, min {:?}, max {:?}", times[0], times[times.len() - 1]);
    let _ = std::fs::remove_dir_all(&home);
}
