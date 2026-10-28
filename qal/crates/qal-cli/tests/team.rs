//! Phase 4a/4b end to end: `qal team`, `qal verify` and `qal mcp --session-file` as real processes.
//! Every output of these runs (stdout, stderr, MCP lines) is kept and searched for secrets.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn temp(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qal-team-cli-{tag}-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::SeqCst)));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A project with a team: `qal init`, some code, `qal team init`, the four agent members.
struct Team {
    project: PathBuf,
    /// Where key files live (outside the project) and session files are written.
    home: PathBuf,
    keys: PathBuf,
    /// Everything the processes printed.
    transcript: Arc<Mutex<Vec<String>>>,
}

const BASE: &str = "fn base() -> Int = 1\nfn spare() -> Int = 2\nfn main(args: List[Str]) -> Int = 0\ntest base_works = base() == 1";

impl Team {
    fn bare() -> Team {
        if !qal_compiler::has_qal_driver() {
            let _ = qal_run::selfhost::init_seed_backend(4);
        }
        let home = temp("home");
        let keys = home.join("keys");
        Team { project: temp("project"), home, keys, transcript: Arc::new(Mutex::new(Vec::new())) }
    }

    fn new() -> Team {
        let t = Team::bare();
        assert_eq!(t.qal(&["init"]).0, 0);
        let f = t.write("base.qal", BASE);
        let (code, v) = t.qal(&["define", &f]);
        assert_eq!(code, 0, "{v}");
        let (code, v) = t.qal(&["team", "init", "--owner", "boss"]);
        assert_eq!(code, 0, "{v}");
        assert_eq!((v["team"]["owner"].as_str(), v["team"]["record"].as_i64()), (Some("boss"), Some(1)));
        assert!(v["note"].as_str().unwrap().contains("not a security boundary"));
        for (id, role) in [("arch-1", "architect"), ("impl-1", "implementer"), ("ver-1", "verifier"), ("rev-1", "reviewer")] {
            let (code, v) = t.qal(&["team", "add", id, "--role", role]);
            assert_eq!(code, 0, "{v}");
            assert_eq!(v["member"]["role"], role);
        }
        t
    }

    fn write(&self, name: &str, content: &str) -> String {
        let path = self.project.join(name);
        std::fs::write(&path, content).unwrap();
        path.to_string_lossy().into_owned()
    }

    fn command(&self) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_qal"));
        c.current_dir(&self.project).env("QAL_KEY_DIR", &self.keys).env_remove("QAL_SESSION_FILE");
        c
    }

    fn qal(&self, args: &[&str]) -> (i32, Value) {
        let out = self.command().args(args).stdin(Stdio::null()).output().unwrap();
        let stdout = String::from_utf8(out.stdout).unwrap();
        let stderr = String::from_utf8(out.stderr).unwrap();
        self.transcript.lock().unwrap().extend([stdout.clone(), stderr]);
        (out.status.code().unwrap(), serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("not JSON ({e}): {stdout}")))
    }

    /// A session file for `member` (written by `qal team session`).
    fn session(&self, member: &str) -> PathBuf {
        let file = self.home.join(format!("{member}.session"));
        let (code, v) = self.qal(&["team", "session", member, "--out", file.to_str().unwrap()]);
        assert_eq!(code, 0, "{v}");
        assert!(v["session"]["id"].as_str().unwrap().starts_with('s'));
        file
    }

    fn mcp(&self, session: Option<&Path>) -> Client {
        let mut c = self.command();
        c.args(["mcp", "--root"]).arg(&self.project);
        if let Some(s) = session {
            c.arg("--session-file").arg(s);
        }
        Client::spawn(c, self.transcript.clone())
    }

    fn agent(&self, member: &str) -> Client {
        let s = self.session(member);
        let mut c = self.mcp(Some(&s));
        c.initialize();
        c
    }
}

struct Client {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    next_id: i64,
    transcript: Arc<Mutex<Vec<String>>>,
}

impl Client {
    fn spawn(mut c: Command, transcript: Arc<Mutex<Vec<String>>>) -> Client {
        let mut child = c.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
        let stdin = child.stdin.take();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Client { child, stdin, stdout, next_id: 0, transcript }
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let line = json!({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params}).to_string();
        let stdin = self.stdin.as_mut().unwrap();
        stdin.write_all(line.as_bytes()).unwrap();
        stdin.write_all(b"\n").unwrap();
        stdin.flush().unwrap();
        let mut out = String::new();
        self.stdout.read_line(&mut out).unwrap();
        self.transcript.lock().unwrap().push(out.clone());
        let v: Value = serde_json::from_str(&out).unwrap_or_else(|e| panic!("not JSON ({e}): {out}"));
        assert_eq!(v["id"], self.next_id);
        v
    }

    fn initialize(&mut self) {
        self.request("initialize", json!({"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "team-test", "version": "1"}}));
    }

    fn tools(&mut self) -> Vec<String> {
        self.request("tools/list", json!({}))["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap().to_string()).collect()
    }

    fn call(&mut self, tool: &str, args: Value) -> (bool, Value) {
        let r = self.request("tools/call", json!({"name": tool, "arguments": args}));
        let result = &r["result"];
        assert!(result.is_object(), "{r}");
        let text: Value = serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap();
        (result["isError"].as_bool().unwrap(), text)
    }

    fn code(&mut self, tool: &str, args: Value) -> String {
        let (_, v) = self.call(tool, args);
        v["diagnostics"][0]["code"].as_str().unwrap_or("ok").to_string()
    }

    /// Opens an own change set (it becomes the session's current one); returns its id.
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
        let mut err = String::new();
        if let Some(mut e) = self.child.stderr.take() {
            let _ = e.read_to_string(&mut err);
        }
        if let Ok(mut t) = self.transcript.lock() {
            t.push(err);
        }
    }
}

fn diag(v: &Value) -> (&str, &str) {
    (v["diagnostics"][0]["code"].as_str().unwrap_or(""), v["diagnostics"][0]["expected"].as_str().unwrap_or(""))
}

#[test]
fn the_owner_sets_up_a_team_on_the_command_line() {
    let t = Team::new();
    let file = t.session("arch-1");
    // The token is only in the file (0600), never in the output.
    assert_eq!(std::fs::metadata(&file).unwrap().permissions().mode() & 0o777, 0o600);
    let token = std::fs::read_to_string(&file).unwrap();
    assert!(token.starts_with("qs_") && token.trim().len() == 67);
    let (code, status) = t.qal(&["team", "status"]);
    assert_eq!(code, 0);
    let members: Vec<&str> = status["status"]["members"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert_eq!(members, ["arch-1", "boss", "impl-1", "rev-1", "ver-1"]);
    assert_eq!(status["status"]["sessions"].as_array().unwrap().len(), 1);
    assert_eq!(status["status"]["records"], 5);
    assert_eq!(status["status"]["allow"], json!(["io", "time", "rand", "db:mem"]));
    // The key files: one per member, outside the project, 0600 in a 0700 directory.
    assert_eq!(std::fs::metadata(&t.keys).unwrap().permissions().mode() & 0o777, 0o700);
    let key_files: Vec<_> = std::fs::read_dir(&t.keys).unwrap().flatten().collect();
    assert_eq!(key_files.len(), 5);
    for f in &key_files {
        assert_eq!(f.metadata().unwrap().permissions().mode() & 0o777, 0o600);
    }

    // The command line does not write code in a team store.
    let f = t.write("more.qal", "fn more() -> Int = 3");
    for args in [vec!["define", f.as_str()], vec!["rename", "spare", "spare2"], vec!["delete", "spare"]] {
        let (code, v) = t.qal(&args);
        assert_eq!((code, diag(&v)), (1, ("E0903", "architect")), "{args:?}: {v}");
    }
    let (code, v) = t.qal(&["export", "base", "--out", t.home.join("p.qalpkg").to_str().unwrap()]);
    assert_eq!(code, 0, "{v}");
    let (code, v) = t.qal(&["import", t.home.join("p.qalpkg").to_str().unwrap()]);
    assert_eq!((code, diag(&v).0), (1, "E0903"), "{v}");
    // Reads still work; `init` does not import anything into a team store.
    assert_eq!(t.qal(&["view", "base"]).0, 0);
    assert_eq!(t.qal(&["init"]).0, 0);
    let (code, v) = t.qal(&["team", "sync-std"]);
    assert_eq!((code, v["std"]["status"].as_str()), (0, Some("up_to_date")), "{v}");

    // Trust file and verify.
    let trust = t.home.join("team.json");
    let (code, v) = t.qal(&["team", "export-trust", "--out", trust.to_str().unwrap()]);
    assert_eq!((code, v["trust"]["members"].as_i64()), (0, Some(5)), "{v}");
    let text = std::fs::read_to_string(&trust).unwrap();
    assert!(!text.contains("qal-ed25519-secret") && text.contains("owner_key"));
    let (code, v) = t.qal(&["verify", "--trust", trust.to_str().unwrap()]);
    assert_eq!((code, v["ok"].as_bool(), v["records"].as_i64(), v["trust"].as_str()), (0, Some(true), Some(5), Some("file")), "{v}");
    let (code, v) = t.qal(&["verify"]);
    assert_eq!((code, v["trust"].as_str()), (0, Some("store")));
    assert!(v["warnings"][0].as_str().unwrap().contains("--trust"));
    // Another team's trust file: E0925.
    let other = Team::new();
    let other_trust = other.home.join("team.json");
    other.qal(&["team", "export-trust", "--out", other_trust.to_str().unwrap()]);
    let (code, v) = t.qal(&["verify", "--trust", other_trust.to_str().unwrap()]);
    assert_eq!((code, v["ok"].as_bool(), v["error"]["code"].as_str(), v["error"]["seq"].as_i64(), v["checked"].as_i64()), (1, Some(false), Some("E0925"), Some(1), Some(0)), "{v}");

    // Revoke and rekey.
    let (code, v) = t.qal(&["team", "rekey", "impl-1"]);
    assert_eq!(code, 0, "{v}");
    let (code, v) = t.qal(&["team", "revoke", "rev-1"]);
    assert_eq!(code, 0, "{v}");
    let (code, v) = t.qal(&["team", "session", "rev-1", "--out", t.home.join("x.session").to_str().unwrap()]);
    assert_eq!((code, diag(&v).0), (1, "E0908"));
    assert!(!t.home.join("x.session").exists());
    assert_eq!(t.qal(&["verify", "--trust", trust.to_str().unwrap()]).1["records"], 7);
    // Usage errors.
    assert_eq!(t.qal_status(&["team"]), 2);
    assert_eq!(t.qal_status(&["team", "session", "arch-1"]), 2);
    assert_eq!(t.qal_status(&["team", "add", "x"]), 2);
    let (code, v) = t.qal(&["team", "add", "x", "--role", "boss"]);
    assert_eq!((code, diag(&v).0), (1, "E0701"));
    let (code, v) = t.qal(&["team", "session", "arch-1", "--ttl", "30d", "--out", t.home.join("y.session").to_str().unwrap()]);
    assert_eq!((code, diag(&v).0), (1, "E0914"));
    let (code, stats) = t.qal(&["team", "stats", "--json"]);
    assert_eq!((code, stats["stats"]["total"]["calls"].as_i64()), (0, Some(0)));
}

impl Team {
    fn qal_status(&self, args: &[&str]) -> i32 {
        self.command().args(args).stdin(Stdio::null()).output().unwrap().status.code().unwrap()
    }
}

#[test]
fn team_commands_need_a_team_store_and_keys_outside_the_project() {
    let t = Team::bare();
    t.qal(&["init"]);
    let (code, v) = t.qal(&["team", "status"]);
    assert_eq!((code, diag(&v).0), (1, "E0900"));
    let (code, v) = t.qal(&["verify"]);
    assert_eq!((code, diag(&v).0), (1, "E0900"));
    // A key directory in the project.
    let out = t.command().args(["team", "init"]).env("QAL_KEY_DIR", t.project.join("keys")).output().unwrap();
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(diag(&v).0, "E0926", "{v}");
    // A key directory others can read.
    std::fs::create_dir_all(&t.keys).unwrap();
    std::fs::set_permissions(&t.keys, std::fs::Permissions::from_mode(0o755)).unwrap();
    let (code, v) = t.qal(&["team", "init"]);
    assert_eq!((code, diag(&v).0), (1, "E0920"), "{v}");
    std::fs::set_permissions(&t.keys, std::fs::Permissions::from_mode(0o700)).unwrap();
    let (code, v) = t.qal(&["team", "init"]);
    assert_eq!((code, v["team"]["owner"].as_str()), (0, Some("owner")), "{v}");
    let (code, v) = t.qal(&["team", "init"]);
    assert_eq!((code, diag(&v).0), (1, "E0908"));
}

#[test]
fn sessions_decide_identity_and_roles_decide_tools() {
    let t = Team::new();
    let all = ["qal_define", "qal_define_ast", "qal_test", "qal_prove", "qal_run", "qal_call", "qal_request", "qal_view", "qal_query", "qal_rename", "qal_delete", "qal_history", "qal_build", "qal_recall", "qal_change", "qal_lease", "qal_review", "qal_merge", "qal_rebase", "qal_whoami", "qal_verify"];
    let reads = ["qal_test", "qal_prove", "qal_run", "qal_call", "qal_request", "qal_view", "qal_query", "qal_history", "qal_build", "qal_recall"];
    let mut arch = t.agent("arch-1");
    assert_eq!(arch.tools(), all);
    let mut imp = t.agent("impl-1");
    let with_define: Vec<&str> = std::iter::once("qal_define").chain(std::iter::once("qal_define_ast")).chain(reads).chain(["qal_change", "qal_lease", "qal_rebase", "qal_whoami", "qal_verify"]).collect();
    assert_eq!(imp.tools(), with_define);
    let mut rev = t.agent("rev-1");
    let rev_tools: Vec<&str> = reads.into_iter().chain(["qal_change", "qal_review", "qal_merge", "qal_whoami", "qal_verify"]).collect();
    assert_eq!(rev.tools(), rev_tools);
    // Every space tool takes `change` in a team store.
    let listed = arch.request("tools/list", json!({}));
    let view = listed["result"]["tools"].as_array().unwrap().iter().find(|t| t["name"] == "qal_view").unwrap().clone();
    assert!(view["inputSchema"]["properties"]["change"].is_object(), "{view}");
    // The role's rules are in qal_define's reference.
    let r = imp.request("tools/list", json!({}));
    let reference = r["result"]["tools"][0]["description"].as_str().unwrap();
    assert!(reference.contains("## Team") && reference.contains("You are an implementer"));

    let (err, who) = arch.call("qal_whoami", json!({}));
    assert!(!err, "{who}");
    assert_eq!((who["member"].as_str(), who["role"].as_str(), who["space"].as_str()), (Some("arch-1"), Some("architect"), Some("main")));
    assert!(who["session"].as_str().unwrap().starts_with('s') && who["leases"] == json!([]));

    // The architect writes in its own change set: signed records, history names the member.
    let c = arch.open("work");
    let (err, v) = arch.call("qal_define", json!({"source": "fn f() -> Int = base() + 1", "agent": "someone-else"}));
    assert!(!err, "{v}");
    assert_eq!((v["space"].as_str(), v["leases"][0]["name"].as_str()), (Some(c.as_str()), Some("f")));
    let (_, h) = arch.call("qal_history", json!({"name": "f"}));
    assert_eq!(h["history"][0]["agent"], "arch-1");
    assert_eq!(arch.call("qal_whoami", json!({})).1["change"], c.as_str());
    // Nobody writes on main (main changes only by merge); the refusal says where to write.
    let (err, v) = imp.call("qal_define", json!({"source": "fn g() -> Int = 1"}));
    assert!(err);
    assert_eq!(diag(&v), ("E0903", "nobody"));
    assert!(v["diagnostics"][0]["hints"][0].as_str().unwrap().contains("change set"));
    assert_eq!(rev.code("qal_define", json!({"source": "fn g() -> Int = 1"})), "E0903");
    assert_eq!(rev.code("qal_rename", json!({"old": "f", "new": "g"})), "E0903");
    assert_eq!(imp.code("qal_delete", json!({"name": "f"})), "E0903");
    // Nor in another member's change set.
    assert_eq!(imp.code("qal_define", json!({"source": "fn g() -> Int = 1", "change": c})), "E0903");
    // Reads for everyone, in any space; `name@space` in views.
    assert_eq!(rev.code("qal_view", json!({"names": ["f"]})), "E0201", "f is not on main");
    let (err, v) = rev.call("qal_view", json!({"names": [format!("f@{c}"), "base@main"]}));
    assert!(!err && v["views"][0]["space"] == c.as_str() && v["views"][1]["space"] == "main", "{v}");
    assert_eq!(rev.code("qal_view", json!({"names": ["f"], "change": c})), "ok");
    assert_eq!(rev.code("qal_test", json!({})), "ok");
    assert_eq!(rev.code("qal_view", json!({"names": ["f"], "change": "c000000000000"})), "E0201");

    // Verify over MCP.
    let (err, v) = rev.call("qal_verify", json!({}));
    assert!(!err && v["ok"] == true && v["records"] == 7 && v["trust"] == "store", "{v}");
    let (err, v) = rev.call("qal_verify", json!({"from": 0}));
    assert!(err && v["diagnostics"][0]["code"] == "E0701", "{v}");

    // No session, an unusable session file, a revoked member: E0902 for every tool.
    let mut none = t.mcp(None);
    none.initialize();
    assert_eq!(none.tools(), ["qal_whoami"]);
    for tool in ["qal_whoami", "qal_view", "qal_define"] {
        let (err, v) = none.call(tool, json!({"names": ["f"]}));
        assert!(err && v["diagnostics"][0]["code"] == "E0902", "{v}");
    }
    let loose = t.session("ver-1");
    std::fs::set_permissions(&loose, std::fs::Permissions::from_mode(0o644)).unwrap();
    let mut c = t.mcp(Some(&loose));
    c.initialize();
    let (_, v) = c.call("qal_whoami", json!({}));
    assert!(v["diagnostics"][0]["code"] == "E0902" && v["diagnostics"][0]["message"].as_str().unwrap().contains("644"), "{v}");
    let garbage = t.home.join("garbage.session");
    std::fs::write(&garbage, "qs_nothex").unwrap();
    std::fs::set_permissions(&garbage, std::fs::Permissions::from_mode(0o600)).unwrap();
    assert_eq!(t.mcp(Some(&garbage)).code("qal_whoami", json!({})), "E0902");
    t.qal(&["team", "revoke", "rev-1"]);
    assert_eq!(rev.code("qal_view", json!({"names": ["f"]})), "E0902");
    // The env variable works like the flag.
    let s = t.session("ver-1");
    let mut cmd = t.command();
    cmd.args(["mcp", "--root"]).arg(&t.project).env("QAL_SESSION_FILE", &s);
    let mut ver = Client::spawn(cmd, t.transcript.clone());
    ver.initialize();
    assert_eq!(ver.call("qal_whoami", json!({})).1["member"], "ver-1");

    // The call log.
    let (_, stats) = t.qal(&["team", "stats"]);
    let total = &stats["stats"]["total"];
    assert!(total["calls"].as_i64().unwrap() >= 15, "{stats}");
    assert!(total["failed_by_code"]["E0903"].as_i64().unwrap() >= 5 && total["failed_by_code"]["E0902"].as_i64().unwrap() >= 5, "{stats}");
    assert_eq!(stats["stats"]["members"]["arch-1"]["tools"]["qal_define"], 1);
}

/// Criterion 2: every tool (with each of its actions) with every agent role, compared with the
/// matrix. Each role works in an own change set (the reviewer, who opens none, on main); work
/// packages exist for claims, and a submitted change set of the architect for reviews and merges.
#[test]
fn every_tool_with_every_role_follows_the_matrix() {
    use qal_store::team::policy::{self, Role, Space};
    let t = Team::new();
    let mut arch0 = t.agent("arch-1");
    let mut packages = std::collections::BTreeMap::new();
    for who in ["impl-1", "ver-1", "arch-1", "rev-1"] {
        let (err, v) = arch0.call("qal_change", json!({"action": "open", "title": format!("for {who}"), "package": true}));
        assert!(!err, "{v}");
        packages.insert(who, v["change"].as_str().unwrap().to_string());
    }
    let submitted = arch0.open("to review");
    assert_eq!(arch0.code("qal_define", json!({"source": "fn reviewed() -> Int = 1"})), "ok");
    assert_eq!(arch0.code("qal_change", json!({"action": "submit"})), "ok");
    let args = |tool: &str, action: Option<&str>, role: &str, package: &str| -> Value {
        match (tool, action) {
            ("qal_review", Some(v)) => json!({"change": submitted, "verdict": v}),
            ("qal_merge", _) => json!({"change": submitted}),
            ("qal_rebase", _) => json!({}),
            // A verifier writes only tests.
            ("qal_define", _) if role == "verifier" => json!({"source": "test by_verifier = base() == 1"}),
            ("qal_define", _) => json!({"source": format!("fn by_{}() -> Int = 1", role)}),
            ("qal_define_ast", _) => json!({"ast": format!("{{\"tests\":[{{\"name\":\"ast_test_{role}\",\"body\":{{\"op\":\"==\",\"args\":[{{\"int\":1}},{{\"int\":1}}]}}}}]}}" )}),
            ("qal_view", _) => json!({"names": ["base"]}),
            ("qal_query", _) => json!({"name": "ba"}),
            ("qal_recall", _) => json!({"name": "ba"}),
            ("qal_history", _) => json!({"name": "base"}),
            ("qal_test", _) => json!({"names": ["base_works"]}),
            ("qal_prove", _) => json!({"names": ["base"]}),
            ("qal_run", _) => json!({"entry": "main"}),
            ("qal_call", _) => json!({"fn": "base"}),
            ("qal_request", _) => json!({"handler": "base", "requests": [{"path": "/"}]}),
            ("qal_build", _) => json!({"entry": "base"}),
            ("qal_rename", _) => json!({"old": "spare", "new": "spare2"}),
            ("qal_delete", _) => json!({"name": "spare2"}),
            ("qal_change", Some("open")) => json!({"action": "open", "title": "another"}),
            ("qal_change", Some("claim")) => json!({"action": "claim", "change": package}),
            ("qal_change", Some(a @ ("show" | "diff"))) => json!({"action": a, "change": package}),
            ("qal_change", Some("scope")) => json!({"action": "scope", "add": ["base"]}),
            ("qal_change", Some(a)) => json!({"action": a}),
            ("qal_lease", Some(a)) => json!({"action": a, "names": [format!("lease_{role}")]}),
            _ => json!({}),
        }
    };
    let rows: Vec<&policy::Row> = policy::POLICY.iter().filter(|r| r.available).collect();
    let mut checked = 0;
    for (member, role) in [("arch-1", "architect"), ("impl-1", "implementer"), ("ver-1", "verifier"), ("rev-1", "reviewer")] {
        let r = Role::parse(role).unwrap();
        let mut c = t.agent(member);
        let listed = c.tools();
        let space = if r == Role::Reviewer { Space::Main } else { c.open(&format!("own of {member}")); Space::Own };
        // Claim and abandon change the session's change set: they come last.
        let order = |row: &&policy::Row| match row.action {
            Some("claim") => 2,
            Some("abandon") => 3,
            _ => 1,
        };
        let mut rows = rows.clone();
        rows.sort_by_key(order);
        for row in rows {
            let allowed = policy::allows(row.tool, row.action, r, space);
            let (_, v) = c.call(row.tool, args(row.tool, row.action, role, &packages[member]));
            let code = v["diagnostics"][0]["code"].as_str().unwrap_or("ok");
            assert_eq!(code == "E0903", !allowed, "{role} {} {:?}: {v}", row.tool, row.action);
            if allowed {
                // A merge is allowed but its gate is not met (no reviews of that root yet): E0913.
                let expected = if row.tool == "qal_merge" { "E0913" } else { "ok" };
                assert_eq!(code, expected, "{role} {} {:?}: {v}", row.tool, row.action);
                assert!(listed.contains(&row.tool.to_string()), "{role} {} is allowed but not listed", row.tool);
            }
            if policy::tools_for(r).contains(&row.tool) != listed.contains(&row.tool.to_string()) {
                panic!("{role} {}: tools/list differs from the matrix", row.tool);
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 4 * rows.len());
    // Every write happened in change sets and is signed.
    let (code, v) = t.qal(&["verify"]);
    assert_eq!((code, v["ok"].as_bool()), (0, Some(true)), "{v}");
    assert!(t.project.join(".qal/build/main/base.wasm").is_file(), "the reviewer builds main");
    let spaces: Vec<_> = std::fs::read_dir(t.project.join(".qal/build")).unwrap().flatten().map(|e| e.file_name().into_string().unwrap()).collect();
    assert!(spaces.iter().filter(|s| s.starts_with('c')).count() >= 3, "team builds go to .qal/build/<space>/: {spaces:?}");
}

#[test]
fn agents_grant_only_the_team_effects() {
    let t = Team::new();
    let mut arch = t.agent("arch-1");
    arch.open("hello");
    assert_eq!(arch.code("qal_define", json!({"source": "fn hello() -> Int !io = { print(\"hi\"); 0 }"})), "ok");
    let (err, v) = arch.call("qal_call", json!({"fn": "hello", "allow": ["io"]}));
    assert!(!err, "{v}");
    for allow in [json!(["fs:data"]), json!(["io,net:example.com"]), json!(["db"])] {
        let (err, v) = arch.call("qal_call", json!({"fn": "hello", "allow": allow}));
        assert!(err && v["diagnostics"][0]["code"] == "E0915" && v["diagnostics"][0]["kind"] == "grant_forbidden", "{v}");
    }
    assert_eq!(arch.code("qal_run", json!({"entry": "main", "allow": ["fs:."]})), "E0915");
    assert_eq!(arch.code("qal_request", json!({"handler": "base", "allow": ["db:x"], "requests": [{"path": "/"}]})), "E0915");
}

#[test]
fn without_its_key_a_member_reads_but_cannot_write() {
    let t = Team::new();
    let (_, status) = t.qal(&["team", "status"]);
    let key_id = status["status"]["members"].as_array().unwrap().iter().find(|m| m["id"] == "arch-1").unwrap()["key_id"].as_str().unwrap().to_string();
    let mut arch = t.agent("arch-1");
    let c = arch.open("work");
    std::fs::remove_file(t.keys.join(format!("{key_id}.ed25519"))).unwrap();
    assert_eq!(arch.code("qal_view", json!({"names": ["base"]})), "ok");
    assert_eq!(arch.code("qal_change", json!({"action": "submit"})), "E0920");
    let (err, v) = arch.call("qal_define", json!({"source": "fn x() -> Int = 1"}));
    assert!(err && v["diagnostics"][0]["code"] == "E0920", "{v}");
    let message = v["diagnostics"][0]["message"].as_str().unwrap();
    assert!(message.contains(&key_id) && message.contains("missing"), "{message}");
    // The owner rekeys; a new server loads the new key.
    assert_eq!(t.qal(&["team", "rekey", "arch-1"]).0, 0);
    let mut again = t.agent("arch-1");
    assert_eq!(again.code("qal_define", json!({"source": "fn x() -> Int = 1", "change": c})), "ok");
    assert_eq!(t.qal(&["verify"]).1["ok"], true);
}

#[test]
fn whoami_waits_for_notices_and_results_carry_them() {
    let t = Team::new();
    let mut imp = t.agent("impl-1");
    // Notices of this std version have been produced or not; take whatever is there first.
    imp.call("qal_whoami", json!({}));
    let store_path = t.project.join(".qal/store.db");
    let started = Instant::now();
    let waiter = std::thread::spawn(move || {
        let r = imp.call("qal_whoami", json!({"wait_s": 20}));
        (r, imp)
    });
    std::thread::sleep(Duration::from_millis(600));
    let s = qal_store::Store::open(&store_path).unwrap();
    s.notify("impl-1", "change_opened", &json!({"change": "c0123456789ab", "title": "orders: create"})).unwrap();
    let ((err, v), mut imp) = waiter.join().unwrap();
    let waited = started.elapsed();
    assert!(!err, "{v}");
    assert!(waited < Duration::from_secs(10), "returned after the notice, not the timeout ({waited:?})");
    assert_eq!(v["notices"][0]["kind"], "change_opened");
    assert_eq!(v["notices"][0]["title"], "orders: create");
    // Seen once returned.
    let (_, v) = imp.call("qal_whoami", json!({}));
    assert_eq!(v["notices"], json!([]));
    // Without notices, wait_s is the timeout.
    let started = Instant::now();
    imp.call("qal_whoami", json!({"wait_s": 1}));
    assert!(started.elapsed() >= Duration::from_millis(900));
    assert_eq!(imp.code("qal_whoami", json!({"wait_s": 46})), "E0701");
    // Other results carry unseen notices (at most 10, `notices_more` counts the rest).
    for i in 0..12 {
        s.notify("impl-1", "merged", &json!({"n": i})).unwrap();
    }
    let (_, v) = imp.call("qal_query", json!({"name": "base"}));
    assert_eq!((v["notices"].as_array().unwrap().len(), v["notices_more"].as_i64()), (10, Some(2)), "{v}");
    let (_, v) = imp.call("qal_query", json!({"name": "base"}));
    assert_eq!(v["notices"].as_array().unwrap().len(), 2);
    let (_, v) = imp.call("qal_query", json!({"name": "base"}));
    assert!(v.get("notices").is_none() && v["space"] == "main");
}

#[test]
fn stores_without_team_mode_are_unchanged() {
    let t = Team::bare();
    t.qal(&["init"]);
    let f = t.write("base.qal", BASE);
    t.qal(&["define", &f]);
    // A session file is ignored by a store without team mode.
    let file = t.home.join("x.session");
    std::fs::write(&file, format!("qs_{}\n", "ab".repeat(32))).unwrap();
    std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o600)).unwrap();
    let mut c = t.mcp(Some(&file));
    c.initialize();
    assert_eq!(c.tools().len(), 14);
    let (err, v) = c.call("qal_define", json!({"source": "fn f() -> Int = 1", "agent": "me"}));
    assert!(!err && v.get("space").is_none() && v.get("notices").is_none(), "{v}");
    assert_eq!(c.call("qal_history", json!({"name": "f"})).1["history"][0]["agent"], "me");
    for tool in ["qal_whoami", "qal_verify"] {
        let (err, v) = c.call(tool, json!({}));
        assert!(err && v["diagnostics"][0]["code"] == "E0900", "{v}");
    }
    assert_eq!(c.request("tools/call", json!({"name": "qal_nope", "arguments": {}}))["error"]["code"], -32602);
}

fn is_hex(b: &[u8]) -> bool {
    b.iter().all(|c| c.is_ascii_hexdigit())
}

/// Every place a secret could leak to: the transcript of all processes, the store's bytes, and
/// every versioned file of the repository.
#[test]
fn secrets_never_leave_their_files() {
    let t = Team::new();
    let mut arch = t.agent("arch-1");
    arch.open("secrets");
    arch.call("qal_define", json!({"source": "fn f() -> Int = 1"}));
    arch.call("qal_whoami", json!({}));
    arch.call("qal_verify", json!({}));
    let mut imp = t.agent("impl-1");
    imp.call("qal_define", json!({"source": "fn g() -> Int = 1"}));
    t.qal(&["team", "status"]);
    t.qal(&["team", "stats"]);
    t.qal(&["team", "rekey", "ver-1"]);
    t.qal(&["verify"]);
    drop(arch);
    drop(imp);
    let mut secrets: Vec<String> = Vec::new();
    for e in std::fs::read_dir(&t.keys).unwrap().flatten() {
        let text = std::fs::read_to_string(e.path()).unwrap();
        secrets.push(text.trim_end().split(' ').nth(1).unwrap().to_string());
    }
    for e in std::fs::read_dir(&t.home).unwrap().flatten().filter(|e| e.path().extension().is_some_and(|x| x == "session")) {
        secrets.push(std::fs::read_to_string(e.path()).unwrap().trim().to_string());
    }
    assert_eq!(secrets.len(), 6 + 2, "6 keys (one rekeyed) and 2 sessions");
    let transcript = t.transcript.lock().unwrap().join("\n");
    assert!(transcript.len() > 1000);
    let store = [".qal/store.db", ".qal/store.db-wal", ".qal/store.db-shm"].iter().map(|f| std::fs::read(t.project.join(f)).unwrap_or_default()).collect::<Vec<_>>().concat();
    let marker = ["qal-ed25519-secret", "-v1"].concat();
    for s in &secrets {
        assert!(!transcript.contains(s.as_str()), "a secret is in the output");
        assert!(!store.windows(s.len()).any(|w| w == s.as_bytes()), "a secret is in the store");
    }
    assert!(!transcript.contains(&marker) && !store.windows(marker.len()).any(|w| w == marker.as_bytes()));
    // Versioned files: no key line (the marker followed by 64 hex digits), no test secret.
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let files = Command::new("git").arg("-C").arg(&repo).args(["ls-files", "-z"]).output().unwrap();
    assert!(files.status.success());
    let mut scanned = 0;
    for name in files.stdout.split(|b| *b == 0).filter(|n| !n.is_empty()) {
        let Ok(bytes) = std::fs::read(repo.join(String::from_utf8_lossy(name).as_ref())) else { continue };
        scanned += 1;
        let m = marker.as_bytes();
        for (i, w) in bytes.windows(m.len()).enumerate() {
            if w == m {
                let rest = &bytes[i + m.len()..];
                assert!(!(rest.len() > 64 && rest[0] == b' ' && is_hex(&rest[1..65])), "a key file line in {}", String::from_utf8_lossy(name));
            }
        }
        for s in &secrets {
            assert!(!bytes.windows(s.len()).any(|w| w == s.as_bytes()), "a test secret in {}", String::from_utf8_lossy(name));
        }
    }
    assert!(scanned > 100, "{scanned} files");
    let gitignore = std::fs::read_to_string(repo.join(".gitignore")).unwrap();
    assert!(gitignore.lines().any(|l| l.trim() == "*.ed25519"));
}

#[test]
fn the_standard_library_comes_only_by_a_signed_sync() {
    // A team store whose std is older than this qal's (here: none at all).
    let t = Team::bare();
    let mut s = qal_store::Store::init(&t.project).unwrap();
    s.define(BASE, "pre").unwrap();
    s.team_init("boss", &qal_store::team::keys::KeyDir::at(&t.keys)).unwrap();
    drop(s);
    assert_eq!(t.qal(&["team", "add", "arch-1", "--role", "architect"]).0, 0);
    let mut arch = t.agent("arch-1");
    // No automatic import on a write; agents are told instead (once).
    let (err, v) = arch.call("qal_change", json!({"action": "open", "title": "work"}));
    assert!(!err && v["notices"][0]["kind"] == "std_outdated", "{v}");
    let (err, v) = arch.call("qal_define", json!({"source": "fn f() -> Int = 1"}));
    assert!(!err && v.get("std_imported").is_none(), "{v}");
    let (_, v) = arch.call("qal_query", json!({"name": "std.", "change": "main"}));
    assert_eq!(v["results"], json!([]));
    let (_, status) = t.qal(&["team", "status"]);
    assert_eq!(status["status"]["records"], 4);
    let (code, v) = t.qal(&["team", "sync-std"]);
    assert_eq!((code, v["std"]["status"].as_str()), (0, Some("imported")), "{v}");
    let (_, v) = arch.call("qal_query", json!({"name": "std.", "change": "main"}));
    assert!(!v["results"].as_array().unwrap().is_empty());
    let (code, v) = t.qal(&["verify"]);
    assert_eq!((code, v["records"].as_i64()), (0, Some(5)), "{v}");
    assert_eq!(t.qal(&["team", "sync-std"]).1["std"]["status"], "up_to_date");
}

#[test]
fn the_first_result_after_a_newer_qal_says_std_is_outdated() {
    let t = Team::bare();
    let mut s = qal_store::Store::init(&t.project).unwrap();
    s.team_init("boss", &qal_store::team::keys::KeyDir::at(&t.keys)).unwrap();
    drop(s);
    t.qal(&["team", "add", "rev-1", "--role", "reviewer"]);
    let mut rev = t.agent("rev-1");
    let (_, v) = rev.call("qal_query", json!({}));
    assert_eq!(v["notices"][0]["kind"], "std_outdated", "{v}");
    assert!(v["notices"][0]["hint"].as_str().unwrap().contains("sync-std"));
    let (_, v) = rev.call("qal_query", json!({}));
    assert!(v.get("notices").is_none(), "once per version: {v}");
}

/// Review I3: in team mode a read never writes to `store.db`, so it never waits for a writer's
/// lock (spec §5.1); its bookkeeping (call log, seen notices) goes to `.qal/team-log.db`.
#[test]
fn team_reads_do_not_wait_for_the_write_lock() {
    let t = Team::new();
    let mut rev = t.agent("rev-1");
    rev.call("qal_whoami", json!({}));
    rev.call("qal_prove", json!({"names": ["base"]}));
    let s = qal_store::Store::open(&t.project.join(".qal/store.db")).unwrap();
    s.notify("rev-1", "merged", &json!({"n": 1})).unwrap();
    drop(s);
    // No cached proofs: `qal_prove` has to prove, and may not wait to cache the result (N1).
    rusqlite::Connection::open(t.project.join(".qal/store.db")).unwrap().execute("DELETE FROM proofs", []).unwrap();
    // Another process holds the write lock of the store for 4 s.
    let db = t.project.join(".qal/store.db");
    let (locked_tx, locked_rx) = std::sync::mpsc::channel();
    let holder = std::thread::spawn(move || {
        let c = rusqlite::Connection::open(&db).unwrap();
        c.execute_batch("BEGIN IMMEDIATE; INSERT INTO meta (key, value) VALUES ('held', '1');").unwrap();
        locked_tx.send(()).unwrap();
        std::thread::sleep(Duration::from_secs(4));
        c.execute_batch("ROLLBACK").unwrap();
    });
    locked_rx.recv().unwrap();
    let started = Instant::now();
    let (err, v) = rev.call("qal_view", json!({"names": ["base"]}));
    assert!(!err, "{v}");
    assert_eq!(v["notices"][0]["kind"], "merged", "notices still arrive: {v}");
    let (_, v) = rev.call("qal_whoami", json!({}));
    assert_eq!(v["notices"], json!([]), "seen once returned");
    assert_eq!(rev.code("qal_query", json!({"name": "ba"})), "ok");
    assert_eq!(rev.code("qal_verify", json!({})), "ok");
    let (err, v) = rev.call("qal_prove", json!({"names": ["base"]}));
    assert!(!err, "{v}");
    let took = started.elapsed();
    assert!(took < Duration::from_millis(2500), "reads waited for the lock: {took:?}");
    holder.join().unwrap();
    // The calls were logged (in the team log, not in the store).
    let (_, stats) = t.qal(&["team", "stats"]);
    assert!(stats["stats"]["members"]["rev-1"]["calls"].as_i64().unwrap() >= 5, "{stats}");
    assert!(t.project.join(".qal/team-log.db").is_file());
}

/// Review M4: a store that exists but cannot be read is an error, never "no team mode".
#[test]
fn the_team_check_fails_closed() {
    let t = Team::new();
    let mut imp = t.agent("impl-1");
    std::fs::write(t.project.join(".qal/store.db"), b"this is not a database at all, padded to look like a file of some size").unwrap();
    let _ = std::fs::remove_file(t.project.join(".qal/store.db-wal"));
    let _ = std::fs::remove_file(t.project.join(".qal/store.db-shm"));
    let (err, v) = imp.call("qal_run", json!({"entry": "main", "allow": ["fs:.", "net:example.com"]}));
    assert!(err && v["diagnostics"][0]["kind"] == "store_error" && v.get("exit_code").is_none(), "{v}");
    assert_eq!(imp.tools(), ["qal_whoami"]);
}

#[test]
fn session_files_stay_out_of_the_project_and_heads_can_be_pinned() {
    let t = Team::new();
    let inside = t.project.join("arch.session");
    let (code, v) = t.qal(&["team", "session", "arch-1", "--out", inside.to_str().unwrap()]);
    assert_eq!((code, diag(&v).0), (1, "E0926"), "{v}");
    assert!(!inside.exists());
    // A link at the target is replaced, not written through.
    let victim = t.home.join("victim.txt");
    std::fs::write(&victim, "keep").unwrap();
    let link = t.home.join("link.session");
    std::os::unix::fs::symlink(&victim, &link).unwrap();
    let (code, _) = t.qal(&["team", "session", "arch-1", "--out", link.to_str().unwrap()]);
    assert_eq!(code, 2, "a link is refused");
    assert_eq!(std::fs::read_to_string(&victim).unwrap(), "keep");
    // The head printed by verify (and status) can be pinned.
    let (_, v) = t.qal(&["verify"]);
    let head = v["head"].as_str().unwrap().to_string();
    assert!(v["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("--head")));
    let (code, v) = t.qal(&["verify", "--head", &head]);
    assert_eq!((code, v["ok"].as_bool()), (0, Some(true)), "{v}");
    let (code, v) = t.qal(&["verify", "--head", &"0".repeat(64)]);
    assert_eq!((code, v["error"]["code"].as_str()), (1, Some("E0922")), "{v}");
    let (code, v) = t.qal(&["verify", "--from", "3"]);
    assert_eq!((code, v["signatures_from"].as_i64()), (0, Some(3)), "{v}");
}

/// Phase 4c over MCP: review, merge (fast-forward and replay), the merge's own check with fresh
/// proofs and the tests (E0912 for a contract a dependency's new `req` refutes, and for a test that
/// fails only on the merged main), rebase, and `qal verify --trust` over all of it.
#[test]
fn a_team_reviews_and_merges_over_mcp() {
    let t = Team::new();
    let mut arch = t.agent("arch-1");
    let mut imp = t.agent("impl-1");
    let mut ver = t.agent("ver-1");
    let mut rev = t.agent("rev-1");
    let gate = |ver: &mut Client, rev: &mut Client, c: &str| {
        let (err, v) = ver.call("qal_test", json!({"change": c}));
        assert!(!err, "{v}");
        let (err, v) = ver.call("qal_review", json!({"change": c, "verdict": "verified"}));
        assert!(!err, "{v}");
        let (err, v) = rev.call("qal_review", json!({"change": c, "verdict": "approve", "comments": [{"name": "price", "text": "ok"}]}));
        assert!(!err, "{v}");
    };
    // A skeleton, reviewed and merged as a fast-forward.
    let c1 = arch.open("price");
    assert_eq!(arch.code("qal_define", json!({"source": "fn price(q: Int) -> Int req q > 0 = q * 2\ntest price_two = price(1) == 2"})), "ok");
    assert_eq!(rev.code("qal_merge", json!({"change": c1})), "E0913", "not submitted");
    assert_eq!(arch.code("qal_change", json!({"action": "submit"})), "ok");
    gate(&mut ver, &mut rev, &c1);
    assert_eq!(imp.code("qal_merge", json!({"change": c1})), "E0903");
    let (err, m) = rev.call("qal_merge", json!({"change": c1}));
    assert!(!err, "{m}");
    assert_eq!((m["merged"].as_bool(), m["kind"].as_str()), (Some(true), Some("fast_forward")), "{m}");
    assert!(m["tests"]["passed"].as_u64().unwrap() >= 2 && m["proofs"]["fresh"] == true, "{m}");
    let (_, v) = imp.call("qal_view", json!({"names": ["price@main"]}));
    assert!(v["views"][0]["text"].as_str().unwrap().contains("q * 2"), "{v}");
    // The implementer starts from that main; meanwhile the architect sharpens price's contract.
    let c2 = imp.open("buy");
    assert_eq!(imp.code("qal_define", json!({"source": "fn buy() -> Int = price(1)\ntest spare_two = spare() == 2"})), "ok");
    let c3 = arch.open("contract");
    assert_eq!(arch.code("qal_define", json!({"source": "fn price(q: Int) -> Int req q > 5 = q * 2\ntest price_two = price(6) == 12\nfn spare() -> Int = 3"})), "ok");
    assert_eq!(arch.code("qal_change", json!({"action": "submit"})), "ok");
    gate(&mut ver, &mut rev, &c3);
    let (err, m) = arch.call("qal_merge", json!({"change": c3}));
    assert!(!err && m["kind"] == "fast_forward", "{m}");
    // c2 is replayed onto the new main: `buy` now violates price's `req` (refuted, proven afresh).
    assert_eq!(imp.code("qal_change", json!({"action": "submit"})), "ok");
    gate(&mut ver, &mut rev, &c2);
    let (err, m) = rev.call("qal_merge", json!({"change": c2}));
    assert!(err, "{m}");
    assert_eq!((m["diagnostics"][0]["code"].as_str(), m["diagnostics"][0]["kind"].as_str()), (Some("E0912"), Some("merge_tests_failed")), "{m}");
    assert!(m["diagnostics"][0]["detail"]["refuted"].as_array().unwrap().iter().any(|n| n == "buy"), "{m}");
    assert!(m["diagnostics"].as_array().unwrap().iter().any(|d| d["code"] == "E0801"), "the refutation follows: {m}");
    // Fixed in the change set: now only the test fails on main as it would be (spare is 3 there).
    assert_eq!(imp.code("qal_define", json!({"source": "fn buy() -> Int = price(6)"})), "ok");
    assert_eq!(imp.code("qal_change", json!({"action": "submit"})), "ok");
    gate(&mut ver, &mut rev, &c2);
    let (err, m) = rev.call("qal_merge", json!({"change": c2}));
    assert!(err, "{m}");
    assert_eq!(m["diagnostics"][0]["code"], "E0912", "{m}");
    assert_eq!(m["diagnostics"][0]["detail"]["failed"][0]["name"], "spare_two", "{m}");
    // Rebase onto main, fix the test there, and merge as a fast-forward.
    let (err, r) = imp.call("qal_rebase", json!({}));
    assert!(!err && r["conflicts"].as_array().unwrap().is_empty(), "{r}");
    assert_eq!(imp.code("qal_define", json!({"source": "test spare_two = spare() == 3"})), "ok");
    assert_eq!(imp.code("qal_change", json!({"action": "submit"})), "ok");
    gate(&mut ver, &mut rev, &c2);
    let (err, m) = rev.call("qal_merge", json!({"change": c2}));
    assert!(!err && m["kind"] == "fast_forward", "{m}");
    // The board shows it merged; every record verifies against the trust file.
    let (_, v) = imp.call("qal_change", json!({"action": "list", "status": ["merged"]}));
    assert_eq!(v["changes"].as_array().unwrap().len(), 3, "{v}");
    let trust = t.home.join("team.json");
    assert_eq!(t.qal(&["team", "export-trust", "--out", trust.to_str().unwrap()]).0, 0);
    let (code, v) = t.qal(&["verify", "--trust", trust.to_str().unwrap()]);
    assert_eq!((code, v["ok"].as_bool()), (0, Some(true)), "{v}");
    // Everyone was told of the merges (notices ride along with every result). The notice JSON is
    // inside the text content, so parse it instead of matching the escaped string.
    let told = t.transcript.lock().unwrap().iter().filter(|line| {
        let Ok(envelope) = serde_json::from_str::<Value>(line) else { return false };
        let Some(text) = envelope["result"]["content"][0]["text"].as_str() else { return false };
        let Ok(body) = serde_json::from_str::<Value>(text) else { return false };
        body["notices"].as_array().is_some_and(|a| a.iter().any(|n| n["kind"] == "merged"))
    }).count();
    assert!(told >= 3, "{told}");
}

/// A reviewer's `changes` comments are readable by the change set's author through `qal_change show`.
#[test]
fn the_author_reads_review_comments_with_show() {
    let t = Team::new();
    let mut imp = t.agent("impl-1");
    let mut rev = t.agent("rev-1");
    let c = imp.open("buy");
    assert_eq!(imp.code("qal_define", json!({"source": "fn buy() -> Int = spare()"})), "ok");
    assert_eq!(imp.code("qal_change", json!({"action": "submit"})), "ok");
    let (err, v) = rev.call("qal_review", json!({"change": c, "verdict": "changes", "comments": [{"name": "buy", "text": "return price, not spare"}, {"text": "add a test"}]}));
    assert!(!err, "{v}");
    let (err, v) = imp.call("qal_change", json!({"action": "show", "change": c}));
    assert!(!err, "{v}");
    let r = &v["reviews"][0];
    assert_eq!((r["member"].as_str(), r["verdict"].as_str(), r["current"].as_bool()), (Some("rev-1"), Some("changes"), Some(true)), "{v}");
    assert!(r["at"].as_i64().is_some(), "{v}");
    assert_eq!((r["comments"][0]["name"].as_str(), r["comments"][0]["text"].as_str()), (Some("buy"), Some("return price, not spare")), "{v}");
    assert_eq!(r["comments"][1]["text"], "add a test", "{v}");
}

/// Spec §8: the section „Team“ of `qal_define`'s reference (in reference.md) has at most 12 lines
/// per role, with exactly that role's rules; the language reference itself is unchanged by it.
#[test]
fn the_team_reference_has_each_roles_rules_in_twelve_lines() {
    use qal_store::team::policy::{Role, ROLES};
    assert!(!qal_cli::mcp::REFERENCE.contains("## Team") && qal_cli::mcp::REFERENCE.ends_with("lists earlier corrections.\n"));
    assert!(qal_cli::mcp::TEAM_REFERENCE.starts_with("## Team\n"));
    for r in ROLES {
        let t = qal_cli::mcp::team_reference(r);
        assert!(t.lines().count() <= 12 && t.len() <= 3000, "{r:?}: {} lines, {} bytes", t.lines().count(), t.len());
        assert!(t.starts_with("## Team\n") && t.contains(&format!("You are a{} {}", if r == Role::Architect || r == Role::Implementer { "n" } else { "" }, r.as_str())), "{t}");
        assert!(!t.contains('@') || t.contains("name@main"), "no other role's line: {t}");
        for other in ROLES.into_iter().filter(|o| *o != r) {
            assert!(!t.contains(&format!("You are a {}", other.as_str())) && !t.contains(&format!("You are an {}", other.as_str())), "{r:?} sees {other:?}");
        }
        for code in ["E0901", "E0904", "E0908", "E0910", "E0911", "E0912", "E0913", "qal_rebase"] {
            assert!(t.contains(code), "{r:?}: {code}");
        }
    }
}
