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
            ("Bash", {"command": f"python3 {wd}/qalc.py define <<'EOF'\n# count all solutions in the bench, see /etc/x and ../y\nfn main(args: List[Str]) -> Int !io = 0\nEOF"}, "isError: false\n{}", False),
            ("Bash", {"command": f"python3 {wd}/qalc.py call qal_run '{{\"stdin\": \"the bench is long ../x /tmp\\n\"}}'"}, "isError: true\n{}", False),
            ("Bash", {"command": f"cd {wd} && mkdir -p data && printf 'a b\\n' > data/x.txt"}, "", False),
        ])
        self.assertEqual(a["violations"], [])
        self.assertEqual((a["tool_calls"], a["failed_tool_calls"], a["api_calls"]), (3, 1, 4))
        self.assertEqual(a["total_tokens"], 4 * 113)

    def test_paths_outside_the_work_dir_are_violations(self):
        wd = self.wd
        a = self.audit("rust", [
            ("Bash", {"command": f"cat {wd}/../other/src/main.rs"}, "", False),
            ("Bash", {"command": "ls /tmp"}, "", False),
            ("Bash", {"command": f"cd {wd} && cat ../x.prompt.md"}, "", False),
            ("Bash", {"command": "cat ~/notes"}, "", False),
            ("Bash", {"command": "cat \"/etc/passwd\""}, "", False),
            ("Read", {"file_path": "/etc/hosts"}, "", False),
            ("Bash", {"command": f"cat {run.REPO}/README"}, "", False),
            ("WebFetch", {"url": "https://example.org"}, "", False),
            ("Bash", {"command": f"cd {wd} && cargo build && /usr/bin/env ls"}, "", False),
        ])
        self.assertEqual(len(a["violations"]), 9, a["violations"])
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

    def test_winner_ranks_by_solve_rate_then_tokens(self):
        a = lambda passed, tokens: {"passed": passed, "tokens": tokens}
        self.assertEqual(report.winner([a(True, 100), a(False, 50)], [a(False, 10), a(False, 10)]), "QAL")
        self.assertEqual(report.winner([a(True, 100)], [a(True, 95)]), "tie")
        self.assertEqual(report.winner([a(True, 100)], [a(True, 50)]), "Rust")
        self.assertEqual(report.winner([a(False, 100)], [a(False, 50)]), "none")

if __name__ == "__main__":
    unittest.main()
