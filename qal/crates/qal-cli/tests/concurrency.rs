//! Phase 4 spec §17 (Nebenläufigkeit), done criterion 3: several real `qal mcp` processes on one
//! team store. (a) 8 processes x 200 random operations on 20 names end without a hang, every
//! confirmed change is in the store and in the records, and `qal verify` is ok; (b) two processes
//! acquire overlapping name sets 1 000 times: never both; (c) a process killed with SIGKILL in the
//! middle of a `define` leaves a consistent store, and its leases expire.

use qal_store::team::keys::KeyDir;
use qal_store::team::policy::Role;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

struct Team {
    project: PathBuf,
    home: PathBuf,
    keys: PathBuf,
}

impl Team {
    /// A team store without the standard library (smaller, so defines are quick) and `members`.
    fn new(tag: &str, members: &[(&str, Role)]) -> Team {
        let home = std::env::temp_dir().join(format!("qal-conc-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        let project = home.join("project");
        std::fs::create_dir_all(&project).unwrap();
        let keys = home.join("keys");
        let mut s = qal_store::Store::init(&project).unwrap();
        s.define("fn base() -> Int = 1", "pre").unwrap();
        let kd = KeyDir::at(&keys);
        s.team_init("boss", &kd).unwrap();
        let owner = s.owner_signer(&kd).unwrap();
        for (id, role) in members {
            s.team_add(&owner, id, *role, &kd).unwrap();
        }
        Team { project, home, keys }
    }

    fn qal(&self, args: &[&str]) -> Value {
        let out = Command::new(env!("CARGO_BIN_EXE_qal")).current_dir(&self.project).env("QAL_KEY_DIR", &self.keys).args(args).stdin(Stdio::null()).output().unwrap();
        serde_json::from_slice(&out.stdout).unwrap_or_else(|e| panic!("not JSON ({e}): {}", String::from_utf8_lossy(&out.stdout)))
    }

    fn agent(&self, member: &str) -> Client {
        let file = self.home.join(format!("{member}.session"));
        let v = self.qal(&["team", "session", member, "--out", file.to_str().unwrap()]);
        assert!(v["session"]["id"].is_string(), "{v}");
        let child = Command::new(env!("CARGO_BIN_EXE_qal"))
            .args(["mcp", "--root"])
            .arg(&self.project)
            .arg("--session-file")
            .arg(&file)
            .env("QAL_KEY_DIR", &self.keys)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut c = Client::new(child);
        c.request("initialize", json!({"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "conc", "version": "1"}}));
        c
    }

    fn store_path(&self) -> PathBuf {
        self.project.join(".qal/store.db")
    }
}

impl Drop for Team {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.home);
    }
}

struct Client {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    next: i64,
}

impl Client {
    fn new(mut child: Child) -> Client {
        let stdin = child.stdin.take();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Client { child, stdin, stdout, next: 0 }
    }

    fn send(&mut self, method: &str, params: Value) {
        self.next += 1;
        let line = json!({"jsonrpc": "2.0", "id": self.next, "method": method, "params": params}).to_string();
        let stdin = self.stdin.as_mut().unwrap();
        stdin.write_all(line.as_bytes()).unwrap();
        stdin.write_all(b"\n").unwrap();
        stdin.flush().unwrap();
    }

    fn receive(&mut self) -> Value {
        let mut out = String::new();
        self.stdout.read_line(&mut out).unwrap();
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("not JSON ({e}): {out}"))
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        self.send(method, params);
        self.receive()
    }

    /// Sends a tool call without waiting for the answer.
    fn send_call(&mut self, tool: &str, args: Value) {
        self.send("tools/call", json!({"name": tool, "arguments": args}));
    }

    fn receive_call(&mut self) -> (bool, Value) {
        let r = self.receive();
        let result = &r["result"];
        assert!(result.is_object(), "{r}");
        (result["isError"].as_bool().unwrap(), result["structuredContent"].clone())
    }

    fn call(&mut self, tool: &str, args: Value) -> (bool, Value) {
        self.send_call(tool, args);
        self.receive_call()
    }

    fn open(&mut self, title: &str) -> String {
        let (err, v) = self.call("qal_change", json!({"action": "open", "title": title}));
        assert!(!err, "{v}");
        v["change"].as_str().unwrap().to_string()
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        drop(self.stdin.take());
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn code(v: &Value) -> &str {
    v["diagnostics"][0]["code"].as_str().unwrap_or("ok")
}

/// A small deterministic generator (the seed is printed with the test).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Every `define` entry in the records: (space, name, new hash).
fn recorded_entries(store: &Path) -> Vec<(String, String, Option<String>)> {
    let c = rusqlite::Connection::open(store).unwrap();
    let mut stmt = c.prepare("SELECT body FROM records ORDER BY seq").unwrap();
    let bodies: Vec<Vec<u8>> = stmt.query_map([], |r| r.get(0)).unwrap().map(Result::unwrap).collect();
    let mut out = Vec::new();
    for b in bodies {
        let v: Value = serde_json::from_slice(&b["qal-record-v1\n".len()..]).unwrap();
        for e in v["args"]["entries"].as_array().into_iter().flatten() {
            out.push((v["space"].as_str().unwrap().to_string(), e["name"].as_str().unwrap().to_string(), e["new"].as_str().map(str::to_string)));
        }
    }
    out
}

#[test]
fn eight_processes_with_random_operations_lose_nothing() {
    let seed: u64 = 0x5eed_4b;
    eprintln!("seed {seed:#x}");
    let members: Vec<(String, Role)> = (0..8).map(|i| (format!("agent-{i}"), if i < 2 { Role::Architect } else { Role::Implementer })).collect();
    let refs: Vec<(&str, Role)> = members.iter().map(|(m, r)| (m.as_str(), *r)).collect();
    let t = Team::new("a", &refs);
    let started = Instant::now();
    let results: Vec<(String, BTreeMap<String, String>, Vec<(String, String)>, BTreeMap<&'static str, usize>)> = std::thread::scope(|s| {
        let handles: Vec<_> = members
            .iter()
            .enumerate()
            .map(|(i, (member, role))| {
                let t = &t;
                let architect = *role == Role::Architect;
                s.spawn(move || {
                    let mut rng = Rng(seed ^ (i as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
                    let mut c = t.agent(member);
                    let change = c.open(&format!("work of {member}"));
                    // The last confirmed hash per name in this change set; every confirmed define.
                    let mut last: BTreeMap<String, String> = BTreeMap::new();
                    let mut confirmed: Vec<(String, String)> = Vec::new();
                    let mut codes: BTreeMap<&'static str, usize> = BTreeMap::new();
                    for _ in 0..200 {
                        let name = format!("n{}", rng.below(20));
                        let op = rng.below(100);
                        let (err, v) = if op < 50 {
                            let r = c.call("qal_define", json!({"source": format!("fn {name}() -> Int = {}", rng.below(1000))}));
                            if !r.0 {
                                let d = &r.1["defined"][0];
                                last.insert(name.clone(), d["hash"].as_str().unwrap().to_string());
                                confirmed.push((name.clone(), d["hash"].as_str().unwrap().to_string()));
                            }
                            r
                        } else if op < 65 {
                            c.call("qal_lease", json!({"action": "acquire", "names": [name, format!("n{}", rng.below(20))], "ttl_s": 120}))
                        } else if op < 80 {
                            c.call("qal_lease", json!({"action": "release", "names": [name]}))
                        } else if op < 90 {
                            c.call("qal_view", json!({"names": [name]}))
                        } else if op < 95 || !architect {
                            c.call("qal_change", json!({"action": if rng.below(2) == 0 { "submit" } else { "show" }, "change": change}))
                        } else {
                            c.call("qal_lease", json!({"action": "break", "names": [name]}))
                        };
                        let k = match code(&v) {
                            "ok" if !err => "ok",
                            "E0901" => "E0901",
                            "E0201" => "E0201",
                            "E0908" => "E0908",
                            other => panic!("{member}: unexpected {other}: {v}"),
                        };
                        *codes.entry(k).or_default() += 1;
                    }
                    (change, last, confirmed, codes)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let elapsed = started.elapsed();
    eprintln!("8 x 200 operations in {elapsed:?}");
    assert!(elapsed < Duration::from_secs(120), "took {elapsed:?}");
    let mut total: BTreeMap<&str, usize> = BTreeMap::new();
    for (_, _, _, codes) in &results {
        for (k, n) in codes {
            *total.entry(k).or_default() += n;
        }
    }
    eprintln!("results by code: {total:?}");
    // Names stay leased after a define (15 min), so most later writes of them by others conflict.
    assert!(total.get("ok").copied().unwrap_or(0) > 400, "{total:?}");
    let defines: usize = results.iter().map(|r| r.2.len()).sum();
    eprintln!("{defines} confirmed defines");
    assert!(defines > 50, "{defines}");
    assert!(total.get("E0901").copied().unwrap_or(0) > 0, "leases conflicted at least once: {total:?}");
    // Every confirmed define is in the records; each change set ends with its last confirmed version.
    let entries = recorded_entries(&t.store_path());
    let c = rusqlite::Connection::open(t.store_path()).unwrap();
    for (change, last, confirmed, _) in &results {
        for (name, hash) in confirmed {
            assert!(entries.iter().any(|(s, n, h)| s == change && n == name && h.as_deref() == Some(hash.as_str())), "{change} {name} {hash} has no record");
        }
        for (name, hash) in last {
            let stored: String = c.query_row("SELECT hash FROM change_names WHERE change = ?1 AND name = ?2", [change, name], |r| r.get(0)).unwrap();
            assert_eq!(&stored, hash, "{change} {name}: a confirmed change was lost");
        }
    }
    let v = t.qal(&["verify"]);
    assert_eq!(v["ok"], true, "{v}");
}

#[test]
fn overlapping_acquires_never_both_succeed() {
    let t = Team::new("b", &[("left", Role::Implementer), ("right", Role::Implementer)]);
    let mut a = t.agent("left");
    let mut b = t.agent("right");
    a.open("left");
    b.open("right");
    let (left, right) = (json!(["x0", "x1", "x2", "x3", "x4"]), json!(["x3", "x4", "x5", "x6", "x7"]));
    let mut wins = [0usize; 3];
    for i in 0..1000 {
        // Both requests are on their way before either answer is read: the processes race.
        a.send_call("qal_lease", json!({"action": "acquire", "names": left}));
        b.send_call("qal_lease", json!({"action": "acquire", "names": right}));
        let (ea, va) = a.receive_call();
        let (eb, vb) = b.receive_call();
        assert!(ea || eb, "iteration {i}: both acquired overlapping names: {va} {vb}");
        for (err, v) in [(ea, &va), (eb, &vb)] {
            assert!(!err || code(v) == "E0901", "iteration {i}: {v}");
        }
        wins[usize::from(!ea) + 2 * usize::from(!eb)] += 1;
        // Nothing was held by the loser (all or nothing); the winner releases.
        if !ea {
            assert_eq!(a.call("qal_lease", json!({"action": "release", "names": left})).1["released"].as_array().unwrap().len(), 5);
        }
        if !eb {
            assert_eq!(b.call("qal_lease", json!({"action": "release", "names": right})).1["released"].as_array().unwrap().len(), 5);
        }
    }
    eprintln!("left won {}, right won {}, neither {}", wins[1], wins[2], wins[0]);
    assert_eq!(wins[1] + wins[2] + wins[0], 1000);
    assert!(wins[1] > 0 && wins[2] > 0, "both sides won sometimes: {wins:?}");
}

#[test]
fn a_process_killed_in_a_define_leaves_a_consistent_store() {
    let t = Team::new("c", &[("victim", Role::Architect), ("other", Role::Implementer)]);
    let big: String = (0..1500).map(|i| format!("fn big{i}(x: Int) -> Int req x >= 0 req x < 1000 ens r > x = x + 1\n")).collect();
    let mut killed_in_flight = 0;
    for delay_ms in [30u64, 80, 150, 300, 600, 1200] {
        let mut v = t.agent("victim");
        let change = v.open("doomed");
        let (err, l) = v.call("qal_lease", json!({"action": "acquire", "names": ["held"], "ttl_s": 60}));
        assert!(!err, "{l}");
        let records_before: i64 = rusqlite::Connection::open(t.store_path()).unwrap().query_row("SELECT count(*) FROM records", [], |r| r.get(0)).unwrap();
        v.send_call("qal_define", json!({"source": big}));
        std::thread::sleep(Duration::from_millis(delay_ms));
        let _ = v.child.kill();
        let _ = v.child.wait();
        let c = rusqlite::Connection::open(t.store_path()).unwrap();
        let records_after: i64 = c.query_row("SELECT count(*) FROM records", [], |r| r.get(0)).unwrap();
        let names: i64 = c.query_row("SELECT count(*) FROM change_names WHERE change = ?1 AND name LIKE 'big%'", [&change], |r| r.get(0)).unwrap();
        // All or nothing: either the define committed with its record, or neither exists.
        assert!((records_after == records_before && names == 0) || (records_after == records_before + 1 && names == 1500), "{delay_ms} ms: {records_before} -> {records_after} records, {names} names");
        if records_after == records_before {
            killed_in_flight += 1;
        }
        let verify = t.qal(&["verify"]);
        assert_eq!(verify["ok"], true, "{delay_ms} ms: {verify}");
        // The killed agent's lease stays until it expires; then it counts as free.
        let mut o = t.agent("other");
        o.open(&format!("after {delay_ms}"));
        let (err, w) = o.call("qal_lease", json!({"action": "acquire", "names": ["held"]}));
        assert!(err && code(&w) == "E0901", "{w}");
        c.execute("UPDATE leases SET expires_at = 1 WHERE name = 'held'", []).unwrap();
        let (err, w) = o.call("qal_lease", json!({"action": "acquire", "names": ["held"]}));
        assert!(!err, "an expired lease is free: {w}");
        assert_eq!(o.call("qal_lease", json!({"action": "release", "names": ["held"]})).1["released"], json!(["held"]));
    }
    eprintln!("killed in flight {killed_in_flight} of 6 times");
    assert!(killed_in_flight > 0, "no kill happened during the define");
}

/// §5.1: a write lock held longer than the busy timeout (10 s) is E0907 `store_busy` (retry),
/// and the retry goes through.
#[test]
fn a_write_lock_held_too_long_is_store_busy() {
    let t = Team::new("busy", &[("arch-1", Role::Architect)]);
    let mut a = t.agent("arch-1");
    a.open("work");
    let (err, w) = a.call("qal_whoami", json!({}));
    assert!(!err && w["leases"] == json!([]), "{w}");
    let hold = rusqlite::Connection::open(t.store_path()).unwrap();
    hold.execute_batch("BEGIN IMMEDIATE").unwrap();
    let (err, v) = a.call("qal_define", json!({"source": "fn f() -> Int = 1"}));
    assert!(err && code(&v) == "E0907" && v["diagnostics"][0]["kind"] == "store_busy", "{v}");
    assert!(v["diagnostics"][0]["hints"][0].as_str().unwrap().contains("retry"));
    hold.execute_batch("ROLLBACK").unwrap();
    let (err, v) = a.call("qal_define", json!({"source": "fn f() -> Int = 1"}));
    assert!(!err, "{v}");
    let (_, w) = a.call("qal_whoami", json!({}));
    assert_eq!(w["leases"][0]["name"], "f", "{w}");
}
