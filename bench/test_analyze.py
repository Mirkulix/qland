#!/usr/bin/env python3
"""Tests of the token attribution (Phase 6, Task 1):

  cd bench && python3 -m unittest test_analyze -v
"""

import json
import os
import sys
import tempfile
import unittest

BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH)
import analyze  # noqa: E402
from test_bench import transcript  # noqa: E402

USAGE = {"input_tokens": 1, "cache_creation_input_tokens": 10, "cache_read_input_tokens": 100, "output_tokens": 2}
PROCESSED, NEW = 113, 11


def qal_define(err_text=None, body="0"):
    """One qalc.py define call; `err_text` is the diagnostic JSON of a failed call."""
    cmd = f"cd /w && python3 qalc.py define <<'EOF'\nfn main(args: List[Str]) -> Int = {body}\nEOF"
    if err_text is None:
        return ("Bash", {"command": cmd}, 'isError: false\n{"defined": [{"name": "main"}]}', False)
    return ("Bash", {"command": cmd}, f"isError: true\n{err_text}", False)


HANDBACK = ("SubagentHandback", {"message": "DONE"}, "ok", False)


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


class Classify(unittest.TestCase):
    def test_classify_rules(self):
        c = analyze.classify_result
        self.assertEqual(c("qal", "Bash", {"command": "cd w && python3 qalc.py define <<'EOF'"}, 'isError: true\n{"diagnostics":[{"code":"E0203"}]}', False), "lang.type")
        self.assertEqual(c("qal", "Bash", {"command": "cd w && python3 qalc.py define <<'EOF'"}, 'isError: true\n{"diagnostics":[{"code":"E0100"}]}', False), "lang.parse")
        self.assertEqual(c("qal", "Bash", {"command": "cd w && python3 qalc.py define <<'EOF'"}, 'isError: true\n{"diagnostics":[{"code":"E0301"}]}', False), "lang.effect")
        self.assertEqual(c("qal", "Bash", {"command": "cd w && python3 qalc.py call qal_test '{}'"}, 'isError: false\n{"failed": ["t1"]}', False), "lang.test_failed")
        self.assertEqual(c("qal", "Bash", {"command": "cd w && python3 qalc.py call qal_test '{}'"}, 'isError: false\n{"failed": []}', False), "ok.test")
        self.assertEqual(c("qal", "Bash", {"command": "cd w && python3 qalc.py call qal_run '{}'"}, 'isError: false\n{"error": {"kind": "trap"}}', False), "lang.run_failed")
        self.assertEqual(c("qal", "Bash", {"command": "cd w && python3 qalc.py define <<'EOF'"}, 'isError: false\n{}', False), "ok.define")
        self.assertEqual(c("rust", "Bash", {"command": "cd w && cargo test"}, "error[E0308]: mismatched types", False), "rust.compile")
        self.assertEqual(c("rust", "Bash", {"command": "cd w && cargo test"}, "thread 'main' panicked at src/main.rs", False), "rust.panic")
        self.assertEqual(c("rust", "Bash", {"command": "cd w && cargo test"}, "test result: FAILED. 1 passed; 1 failed", False), "rust.test_failed")
        self.assertEqual(c("rust", "Bash", {"command": "cd w && cargo test"}, "test result: ok. 3 passed", False), "ok.other")
        self.assertEqual(c("qal", "Read", {"file_path": "/w/TASK.md"}, "You are taking part…", False), "env.read_task")
        self.assertEqual(c("qal", "Bash", {"command": "cd w && mkdir -p data"}, "", False), "ok.other")
        self.assertEqual(c("qal", "Bash", {"command": "cd w && cat nothing"}, "cat: nothing: No such file", True), "env.tool_error")

    def test_guard_refusal_beats_cargo_error(self):
        text = "This session is isolated in the worktree /x, but this command ... error[E0001]"
        self.assertEqual(analyze.classify_result("rust", "Bash", {"command": "cd w && cargo build"}, text, True), "env.guard")


class Attribute(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "agent-x.jsonl")

    def attribute(self, calls, arm="qal", reference_chars=0, prompt_chars=1):
        write(self.path, transcript(calls))
        return analyze.attribute(self.path, arm, reference_chars, prompt_chars)

    def test_next_call_is_charged_to_the_previous_result(self):
        a = self.attribute([qal_define('{"diagnostics":[{"code":"E0203"}]}'), qal_define(), HANDBACK])
        by = a["by_category"]
        self.assertEqual(a["api_calls"], 3)
        self.assertEqual(by["lang.type"]["processed"], PROCESSED)
        self.assertEqual(by["lang.type"]["calls"], 1)
        # the last round (the DONE handback) is charged to the result before it, like every round
        self.assertEqual(by["ok.define"]["processed"], PROCESSED)
        self.assertNotIn("final", by)
        self.assertEqual(by["prompt.task"]["processed"], PROCESSED)
        self.assertEqual(sum(c["processed"] for c in by.values()), 3 * PROCESSED)
        self.assertEqual(a["processed_per_round"], [PROCESSED, PROCESSED, PROCESSED])

    def test_streamed_chunks_count_once(self):
        lines = [
            {"type": "assistant", "message": {"id": "m0", "usage": USAGE, "content": [{"type": "text", "text": "thinking"}]}},
            {"type": "assistant", "message": {"id": "m0", "usage": USAGE, "content": [{"type": "tool_use", "id": "t0", "name": "Bash", "input": {"command": "cd w && python3 qalc.py define <<'EOF'"}}]}},
            {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t0", "content": "isError: false\n{}", "is_error": False}]}},
            {"type": "assistant", "message": {"id": "m1", "usage": USAGE, "content": [{"type": "text", "text": "DONE"}]}},
        ]
        write(self.path, "\n".join(json.dumps(l) for l in lines) + "\n")
        a = analyze.attribute(self.path, "qal", 0, 1)
        self.assertEqual(a["api_calls"], 2)
        self.assertEqual(sum(c["processed"] for c in a["by_category"].values()), 2 * PROCESSED)
        self.assertEqual(a["by_category"]["ok.define"]["processed"], PROCESSED)

    def test_text_only_continuation_is_final(self):
        lines = [
            {"type": "assistant", "message": {"id": "m0", "usage": USAGE, "content": [{"type": "tool_use", "id": "t0", "name": "Bash", "input": {"command": "cd w && ls"}}]}},
            {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t0", "content": "", "is_error": False}]}},
            {"type": "assistant", "message": {"id": "m1", "usage": USAGE, "content": [{"type": "text", "text": "Now I will..."}]}},
            {"type": "assistant", "message": {"id": "m2", "usage": USAGE, "content": [{"type": "text", "text": "DONE"}]}},
        ]
        write(self.path, "\n".join(json.dumps(l) for l in lines) + "\n")
        a = analyze.attribute(self.path, "qal", 0, 1)
        self.assertEqual(a["by_category"]["ok.other"]["processed"], PROCESSED)
        self.assertEqual(a["by_category"]["final"]["processed"], PROCESSED)

    def test_orphan_results_are_unclassified(self):
        lines = [
            {"type": "assistant", "message": {"id": "m0", "usage": USAGE, "content": [{"type": "tool_use", "id": "t0", "name": "Bash", "input": {"command": "cd w && ls"}}]}},
            {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "zzz", "content": "orphan", "is_error": False}]}},
            {"type": "assistant", "message": {"id": "m1", "usage": USAGE, "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "cd w && ls"}}]}},
            {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "", "is_error": False}]}},
            {"type": "assistant", "message": {"id": "m2", "usage": USAGE, "content": [{"type": "text", "text": "DONE"}]}},
        ]
        write(self.path, "\n".join(json.dumps(l) for l in lines) + "\n")
        a = analyze.attribute(self.path, "qal", 0, 1)
        self.assertEqual([u["text"] for u in a["unclassified"]], ["orphan"])
        self.assertEqual(a["by_category"]["unclassified"]["processed"], PROCESSED)
        self.assertEqual(a["api_calls"], 3)

    def test_prompt_split_estimates_reference_tokens_from_its_chars(self):
        # reference_chars / 4 tokens, never more than the first call's new tokens
        a = self.attribute([qal_define(), HANDBACK], reference_chars=16, prompt_chars=400)
        self.assertAlmostEqual(a["by_category"]["prompt.reference"]["new"], 4, delta=0.01)
        self.assertAlmostEqual(a["by_category"]["prompt.task"]["new"], NEW - 4, delta=0.01)
        self.assertAlmostEqual(a["by_category"]["prompt.reference"]["processed"] + a["by_category"]["prompt.task"]["processed"], PROCESSED, delta=0.01)
        a = self.attribute([qal_define(), HANDBACK], reference_chars=4000, prompt_chars=4100)
        self.assertAlmostEqual(a["by_category"]["prompt.reference"]["new"], NEW, delta=0.01)
        self.assertAlmostEqual(a["by_category"]["prompt.task"]["new"], 0, delta=0.01)

    def test_reclassify_lists_tool_errors_and_other_diags(self):
        a = self.attribute([
            ("Bash", {"command": "cd w && cat nothing"}, "cat: nothing: No such file", True),
            qal_define('{"diagnostics":[{"code":"E0601"}]}'),
            qal_define(body="1"), HANDBACK])
        self.assertEqual([(r["category"], r["text"][:12]) for r in a["reclassify"]], [("env.tool_error", "cat: nothing"), ("lang.other_diag", "isError: tru")])
        self.assertEqual(a["unclassified"], [])

    def test_diag_codes_are_counted(self):
        a = self.attribute([qal_define('{"diagnostics":[{"code":"E0203"},{"code":"E0201"}]}'), qal_define('{"diagnostics":[{"code":"E0203"}]}'), qal_define(body="1"), HANDBACK])
        self.assertEqual(a["diag_codes"], {"E0203": 2, "E0201": 1})

    def test_retries_are_identical_inputs_after_a_failure(self):
        bad = qal_define('{"diagnostics":[{"code":"E0100"}]}')
        # the same input again after a failure is a retry; a changed input is a fix
        a = self.attribute([bad, bad, qal_define(body="1"), HANDBACK])
        self.assertEqual(a["retries"], 1)

    def test_guard_refusals_are_counted(self):
        refused = ("Bash", {"command": "cd /w && cargo test"}, "This session is isolated in the worktree /x, but ...", True)
        a = self.attribute([refused, ("Bash", {"command": "cd /w && cargo test"}, "test result: ok. 1 passed", False), HANDBACK], arm="rust")
        self.assertEqual(a["by_category"]["env.guard"]["calls"], 1)
        self.assertEqual(a["retries"], 1)


def row(arm, processed_by_cat, rounds):
    by = {cat: {"processed": p, "new": p // 10, "output": 1, "calls": 1} for cat, p in processed_by_cat.items()}
    return {"arm": arm, "by_category": by, "api_calls": rounds, "processed_per_round": [100 * (k + 1) for k in range(rounds)], "retries": 0, "dispatch_tokens": 1000, "diag_codes": {"E0203": 1}}


class Summarize(unittest.TestCase):
    def rows(self):
        return [
            row("qal", {"prompt.reference": 300, "prompt.task": 100, "lang.type": 200, "ok.define": 400}, 2),
            row("qal", {"prompt.reference": 300, "prompt.task": 100, "env.guard": 500, "ok.define": 100}, 4),
            row("qal", {"prompt.reference": 300, "prompt.task": 100, "final": 50, "ok.run": 150}, 6),
            row("rust", {"prompt.task": 100, "rust.compile": 100, "ok.other": 300}, 3),
        ]

    def test_summary_top3_and_shares(self):
        s = analyze.summarize(self.rows())
        self.assertEqual(len(s["qal"]["top3"]), 3)
        self.assertEqual(s["qal"]["top3"][0][0], "prompt.reference")
        self.assertAlmostEqual(sum(s["qal"]["shares"].values()), 1.0, places=6)
        self.assertAlmostEqual(s["qal"]["shares"]["env"], 500 / 2600, places=6)
        self.assertEqual(s["qal"]["totals"]["ok.define"]["processed"], 500)
        self.assertAlmostEqual(s["qal"]["means"]["ok.define"]["processed"], 500 / 3, places=6)
        self.assertEqual(s["qal"]["runs"], 3)
        self.assertEqual(s["rust"]["runs"], 1)
        self.assertEqual(s["qal"]["diag_codes"], {"E0203": 3})

    def test_rounds_and_round_cost(self):
        s = analyze.summarize(self.rows())
        self.assertEqual(s["qal"]["rounds"]["mean"], 4.0)
        self.assertEqual(s["qal"]["rounds"]["median"], 4)
        self.assertEqual(s["qal"]["rounds"]["max"], 6)
        self.assertEqual(set(s["qal"]["round_cost"]), {"1", "5", "10"})
        self.assertEqual(s["qal"]["round_cost"]["1"], 100.0)
        self.assertEqual(s["qal"]["round_cost"]["5"], 500.0)
        self.assertIsNone(s["qal"]["round_cost"]["10"])


if __name__ == "__main__":
    unittest.main()
