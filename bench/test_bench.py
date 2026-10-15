#!/usr/bin/env python3
"""Tests of the benchmark harness itself: transcript audit, checker strictness, report metrics, and
the parse/print round trip of the QAL reference solutions (spec §12.3).

  python3 bench/test_bench.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH)
import check  # noqa: E402
import qalc  # noqa: E402
import report  # noqa: E402
import run  # noqa: E402

def transcript(calls):
    """A minimal subagent transcript: one assistant message per (tool, input, result text, is_error)."""
    lines = []
    for k, (tool, inp, text, err) in enumerate(calls):
        lines.append({"type": "assistant", "message": {"id": f"m{k}", "usage": {"input_tokens": 1, "cache_creation_input_tokens": 10, "cache_read_input_tokens": 100, "output_tokens": 2},
                                                       "content": [{"type": "tool_use", "id": f"t{k}", "name": tool, "input": inp}]}})
        lines.append({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": f"t{k}", "content": text, "is_error": err}]}})
    return "\n".join(json.dumps(l) for l in lines) + "\n"

class Audit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.wd = os.path.realpath(os.path.join(self.tmp, "w"))
        os.makedirs(self.wd)
        self.path = os.path.join(self.tmp, "agent-x.jsonl")
        run.transcript_path = lambda agent_id: self.path

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def audit(self, arm, calls):
        with open(self.path, "w") as f:
            f.write(transcript(calls + [("SubagentHandback", {"message": "DONE"}, "ok", False)]))
        return run.audit("x", self.wd, arm)

    def test_payloads_in_heredocs_and_quotes_are_not_violations(self):
        wd = self.wd
        a = self.audit("qal", [
            ("Bash", {"command": f"cd {wd} && python3 qalc.py define <<'EOF'\n# count all solutions in the bench, see /etc/x and a/b\nfn main(args: List[Str]) -> Int !io = 0\nEOF"}, "isError: false\n{}", False),
            ("Bash", {"command": f"cd {wd} && python3 qalc.py call qal_run '{{\"stdin\": \"the bench is long /tmp\\n\"}}'"}, "isError: true\n{}", False),
            ("Bash", {"command": f"cd {wd} && mkdir -p data && printf 'a b\\n' > data/x.txt"}, "", False),
        ])
        self.assertEqual(a["violations"], [])
        self.assertEqual((a["tool_calls"], a["failed_tool_calls"], a["api_calls"]), (3, 1, 4))
        self.assertEqual(a["total_tokens"], 4 * 113)

    def test_paths_outside_the_work_dir_are_violations(self):
        wd = self.wd
        a = self.audit("rust", [
            ("Bash", {"command": f"cd {wd} && cat {wd}/../other/src/main.rs"}, "", False),
            ("Bash", {"command": f"cd {wd} && ls /tmp"}, "", False),
            ("Bash", {"command": f"cd {wd} && cat ../x.prompt.md"}, "", False),
            ("Bash", {"command": f"cd {wd} && cat ~/notes"}, "", False),
            ("Bash", {"command": f"cd {wd} && cat \"/etc/passwd\""}, "", False),
            ("Read", {"file_path": "/etc/hosts"}, "", False),
            ("Bash", {"command": f"cd {wd} && cat {run.REPO}/README"}, "", False),
            ("WebFetch", {"url": "https://example.org"}, "", False),
            ("Bash", {"command": f"cd {wd} && cargo build && /usr/bin/env ls"}, "", False),
        ])
        self.assertGreaterEqual(len(a["violations"]), 8, a["violations"])
        for i in range(8):
            self.assertTrue(any(v.startswith(f"call {i + 1}:") for v in a["violations"]), (i, a["violations"]))
        self.assertFalse(any(v.startswith("call 9:") for v in a["violations"]), a["violations"])
        self.assertTrue(any("WebFetch" in v for v in a["violations"]))

    def test_rust_failures_are_compile_errors_panics_and_failed_tests_only(self):
        wd = self.wd
        a = self.audit("rust", [
            ("Bash", {"command": f"cd {wd} && cargo run -q < in.txt"}, "error: cannot read nope\n", False),
            ("Bash", {"command": f"cd {wd} && cargo build"}, "error[E0308]: mismatched types\n", False),
            ("Bash", {"command": f"cd {wd} && cargo run"}, "thread 'main' panicked at src/main.rs:3:5\n", False),
            ("Bash", {"command": f"cd {wd} && cargo test"}, "test result: FAILED. 1 passed; 1 failed\n", False),
        ])
        self.assertEqual((a["violations"], a["failed_tool_calls"]), ([], 3))

    def flagged(self, calls, arm="rust"):
        return self.audit(arm, calls)["violations"]

    def test_leakage_probes_are_flagged(self):
        wd = self.wd
        T = "bench/tasks/01-word-count/tests.json"
        probes = [
            ("Bash", {"command": f"cd {wd} && cat {T}"}),
            ("Bash", {"command": f"cd {wd} && grep -r tests.json ."}),
            ("Bash", {"command": f"cd {wd} && ls solutions/rust"}),
            ("Bash", {"command": f"cd {wd} && bash -c \"cat ../x\""}),
            ("Bash", {"command": f"cd {wd} && python3 - <<'EOF'\nprint(open('{T}').read())\nEOF"}),
            ("Bash", {"command": f"cd {wd} && cat \"$HOME/x\""}),
            ("Bash", {"command": f"cd {wd} && ls /tmp/qal-bench"}),
            ("Bash", {"command": f"cat {T}"}),
            ("Bash", {"command": "find . -name '*.json'"}),
            ("Bash", {"command": f"cd /tmp && ls"}),
            ("Glob", {"pattern": "/Users/x/**/tests.json", "path": wd}),
            ("Glob", {"pattern": "**/tests.json"}),
            ("Glob", {"pattern": "../**", "path": wd}),
            ("Grep", {"pattern": "expected"}),
            ("Grep", {"pattern": "x", "path": "/Users/x"}),
            ("Read", {"file_path": "src/main.rs"}),
            ("Write", {"file_path": "/tmp/x.rs", "content": ""}),
            ("Edit", {"file_path": f"{wd}/../x", "old_string": "a", "new_string": "b"}),
        ]
        for tool, inp in probes:
            self.assertGreaterEqual(len(self.flagged([(tool, inp, "", False)])), 1, (tool, inp))

    def test_normal_run_is_clean(self):
        wd = self.wd
        a = self.audit("rust", [
            ("Write", {"file_path": f"{wd}/src/main.rs", "content": "fn main() { let _ = 1 / 2; println!(\"a/b\"); }"}, "", False),
            ("Bash", {"command": f"cd {wd} && cargo build 2>&1 | tail -5"}, "", False),
            ("Bash", {"command": f"cd {wd} && printf 'x\\n' > in.txt && cargo run -q < in.txt"}, "", False),
            ("Bash", {"command": f"cd {wd} && cat > src/main.rs <<'EOF'\nfn main() {{ for i in 0..3 {{ println!(\"{{}}\", i / 2); }} }}\nEOF"}, "", False),
            ("Read", {"file_path": f"{wd}/src/main.rs"}, "", False),
            ("Edit", {"file_path": f"{wd}/src/main.rs", "old_string": "a", "new_string": "b"}, "", False),
            ("Glob", {"pattern": "**/*.rs", "path": wd}, "", False),
            ("Glob", {"pattern": f"{wd}/src/*.rs", "path": wd}, "", False),
            ("Grep", {"pattern": "fn main", "path": wd}, "", False),
            ("Bash", {"command": f"cd {wd} && /usr/bin/env ls"}, "", False),
            ("ToolSearch", {"query": "select:Bash"}, "", False),
        ])
        self.assertEqual(a["violations"], [])

    def test_source_text_is_symmetric_and_not_over_flagged(self):
        wd = self.wd
        qal = "# see tasks/ and ~\nfn f(x: T) -> Int = match x { A(..) => 1, _ => 0 }\nfn g() -> Str = \"..\"\nfn h() -> Str = '~'\n"
        clean = [
            ("Bash", {"command": f"cd {wd} && python3 qalc.py define <<'EOF'\n{qal}EOF"}),
            ("Bash", {"command": f"cd {wd} && python3 qalc.py call qal_run '{{\"stdin\": \"..\"}}'"}),
            ("Bash", {"command": f"cd {wd} && echo \"a .. ~ tasks/\" > data/x"}),
            ("Write", {"file_path": f"{wd}/src/main.rs", "content": "fn main() { let v = [1, 2, 3]; let _ = &v[..2]; for i in 0..3 { let _ = !i; } // ~ tasks/\n}"}),
            ("Edit", {"file_path": f"{wd}/src/main.rs", "old_string": "a", "new_string": "&v[1..]"}),
        ]
        for tool, inp in clean:
            self.assertEqual(self.flagged([(tool, inp, "", False)]), [], (tool, inp))
        flagged = [
            ("Write", {"file_path": f"{wd}/src/main.rs", "content": "const P: &str = \"x/tests.json\";"}),
            ("Edit", {"file_path": f"{wd}/src/main.rs", "old_string": "a", "new_string": "bench/tasks"}),
            ("Bash", {"command": f"cd {wd} && python3 qalc.py define <<'EOF'\n# tests.json\nEOF"}),
            ("Bash", {"command": f"cd {wd} && python3 qalc.py define <<'EOF'\n# solutions/x\nEOF"}),
            ("Bash", {"command": f"cd {wd} && ls tasks/"}),
            ("Bash", {"command": f"cd {wd} && ls .."}),
            ("Bash", {"command": f"cd {wd} && ls ~"}),
            ("Bash", {"command": f"cd {wd} && cd - && ls"}),
            ("Bash", {"command": f"cd {wd} && cd"}),
            ("Bash", {"command": f"cd {wd} && cd; ls"}),
            ("Bash", {"command": f"cd {wd} && pushd x"}),
            ("Bash", {"command": f"cd {wd} && popd"}),
            ("Bash", {"command": f"cd {wd} && ls $OLDPWD"}),
            ("Bash", {"command": f"cd {wd} && ls $(dirname {wd})"}),
            ("Bash", {"command": f"cd {wd} && ls `dirname {wd}`"}),
            ("Bash", {"command": f"cd {wd} && ls .*"}),
            ("Bash", {"command": f"cd {wd} && ls .??"}),
            ("Bash", {"command": f"cd {wd} && ls .{{a,b}}"}),
            ("Bash", {"command": f"cd {wd} && python3 -c 'import os; print(os.listdir(os.path.dirname(os.getcwd())))'"}),
            ("Bash", {"command": f"cd {wd} && python3 -c \"import pathlib; print(list(pathlib.Path('.').resolve().parent.iterdir()))\""}),
            ("Bash", {"command": f"cd {wd} && python3 - <<'EOF'\nimport os\nprint(os.getcwd())\nEOF"}),
            ("Bash", {"command": f"cd {wd} && bash -c \"ls ..\""}),
        ]
        for tool, inp in flagged:
            self.assertGreaterEqual(len(self.flagged([(tool, inp, "", False)])), 1, (tool, inp))

    def test_missing_cd_is_a_violation(self):
        v = self.flagged([("Bash", {"command": "cargo build"}, "", False)])
        self.assertEqual(len(v), 1)
        self.assertIn("cd", v[0])

class Archive(unittest.TestCase):
    def test_final_results_move_the_work_dir_to_the_archive(self):
        tmp = tempfile.mkdtemp()
        try:
            wd = os.path.join(tmp, "work", "abc")
            os.makedirs(wd)
            open(os.path.join(wd, "f"), "w").close()
            dest = run.archive_workdir(wd, "t-qal-run1", archive_root=os.path.join(tmp, "arch"))
            self.assertFalse(os.path.exists(wd))
            self.assertTrue(os.path.exists(os.path.join(dest, "f")))
            self.assertTrue(dest.startswith(os.path.join(tmp, "arch")))
        finally:
            shutil.rmtree(tmp)

class Robustness(unittest.TestCase):
    def test_client_request_times_out(self):
        tmp = tempfile.mkdtemp()
        try:
            fake = os.path.join(tmp, "fake")
            with open(fake, "w") as f:
                f.write("#!/bin/sh\nsleep 30\n")
            os.chmod(fake, 0o755)
            old = qalc.REQUEST_TIMEOUT
            qalc.REQUEST_TIMEOUT = 1
            try:
                with self.assertRaises(RuntimeError):
                    qalc.Client(root=tmp, binary=fake)
            finally:
                qalc.REQUEST_TIMEOUT = old
        finally:
            shutil.rmtree(tmp)

    def test_run_limited_caps_output_and_kills_the_group(self):
        import time
        t = time.time()
        rc, out, timed_out = check.run_limited(["sh", "-c", "yes"], cwd="/", stdin=b"", timeout=5, cap=1000)
        self.assertTrue(len(out) > 1000 and len(out) < 1 << 20 and not timed_out)
        rc, out, timed_out = check.run_limited(["sh", "-c", "sleep 30 & sleep 30"], cwd="/", stdin=b"", timeout=1, cap=1000)
        self.assertTrue(timed_out)
        self.assertLess(time.time() - t, 15)
        rc, out, timed_out = check.run_limited(["sh", "-c", "cat; exit 3"], cwd="/", stdin=b"hi\n", timeout=5, cap=1000)
        self.assertEqual((rc, out, timed_out), (3, b"hi\n", False))

class Checker(unittest.TestCase):
    def test_output_must_match_exactly(self):
        tmp = tempfile.mkdtemp()
        try:
            proj = os.path.join(tmp, "p")
            shutil.copytree(os.path.join(BENCH, "solutions", "rust", "01-word-count"), proj)
            src = os.path.join(proj, "src", "main.rs")
            with open(src) as f:
                text = f.read().replace('println!("{} {} {}"', 'print!("{} {} {}"')
            with open(src, "w") as f:
                f.write(text)
            res = check.check_rust("01-word-count", proj)
            self.assertEqual((res["error"], res["passed"]), (None, 0), "a missing final newline fails")
        finally:
            shutil.rmtree(tmp)

class ReferenceRoundTrip(unittest.TestCase):
    """Spec §12.3: parse(print(g)) == g for every QAL reference solution — the canonical print of a
    solution reprints identically and defines definitions with the same hashes."""

    def hashes(self, source):
        with tempfile.TemporaryDirectory() as root:
            c = qalc.Client(root=root)
            try:
                is_error, text, _ = c.call("qal_define", {"source": source})
                self.assertFalse(is_error, text)
                _, _, out = c.call("qal_query", {})
                return sorted((d["name"], d["hash"]) for d in out["results"])
            finally:
                c.close()

    def test_reference_solutions_round_trip(self):
        check.ensure_qal_binary()
        for task in check.task_ids():
            path = os.path.join(BENCH, "solutions", "qal", f"{task}.qal")
            printed = subprocess.run([qalc.BIN, "fmt", path], capture_output=True, text=True, check=True).stdout
            with tempfile.NamedTemporaryFile("w", suffix=".qal", delete=False) as f:
                f.write(printed)
            try:
                again = subprocess.run([qalc.BIN, "fmt", f.name], capture_output=True, text=True, check=True).stdout
            finally:
                os.unlink(f.name)
            self.assertEqual(again, printed, task)
            with open(path, encoding="utf-8") as f:
                self.assertEqual(self.hashes(printed), self.hashes(f.read()), task)

class Report(unittest.TestCase):
    def rec(self, task, arm, attempt, passed, tokens, valid=True):
        return {"task": task, "arm": arm, "run": 1, "attempt": attempt, "passed": passed, "valid": valid,
                "dispatch": {"total_tokens": tokens, "tool_uses": 3, "duration_ms": 1000},
                "transcript": {"failed_tool_calls": 0, "total_tokens": tokens * 2, "violations": [] if valid else ["x"]}}

    def test_last_attempt_counts_and_a_redo_starts_fresh(self):
        runs = report.runs_of([self.rec("t", "qal", 1, True, 10, valid=False), self.rec("t", "qal", 1, False, 20), self.rec("t", "qal", 2, True, 30)])
        r = runs[("t", "qal", 1)]
        self.assertEqual((r["attempts"], r["tokens"], r["passed"], r["first_pass"], r["valid"]), (2, 30, True, False, True))

    def test_over_budget_runs_are_listed(self):
        rec = self.rec("t", "qal", 1, True, 10)
        rec["transcript"]["over_budget"] = True
        lines = report.over_budget_lines(report.runs_of([rec, self.rec("u", "rust", 1, True, 10)]))
        self.assertEqual(len(lines), 1)
        self.assertIn("t qal", lines[0])

    def test_winner_ranks_by_solve_rate_then_tokens(self):
        a = lambda passed, tokens: {"passed": passed, "tokens": tokens}
        self.assertEqual(report.winner([a(True, 100), a(False, 50)], [a(False, 10), a(False, 10)]), "QAL")
        self.assertEqual(report.winner([a(True, 100)], [a(True, 95)]), "tie")
        self.assertEqual(report.winner([a(True, 100)], [a(True, 50)]), "Rust")
        self.assertEqual(report.winner([a(False, 100)], [a(False, 50)]), "none")

if __name__ == "__main__":
    unittest.main()
