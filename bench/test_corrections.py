#!/usr/bin/env python3
"""Tests of the correction extractor (Phase 6b, Task 3):

  python3 bench/test_corrections.py
"""

import json
import os
import sys
import tempfile
import unittest

BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH)
import corrections  # noqa: E402
from test_bench import transcript  # noqa: E402


def define(source):
    return {"command": f"cd /w && python3 qalc.py define <<'EOF'\n{source}\nEOF"}


def fail(code, df, path="body", message="unknown name `x`"):
    return "isError: true\n" + json.dumps({"diagnostics": [{"code": code, "kind": "k", "def": df, "path": path, "message": message}]})


def done(*names):
    return "isError: false\n" + json.dumps({"defined": [{"name": n, "status": "added"} for n in names]})


BAD = "fn helper() -> Int = 1\nfn f(n: Int) -> Int = { let x = n + 1; x }\nfn tail() -> Int = 2"
GOOD = "fn helper() -> Int = 1\nfn f(n: Int) -> Int = let x = n + 1 in x\nfn tail() -> Int = 2"


class Extract(unittest.TestCase):
    def extract(self, calls, name="agent-a1.jsonl"):
        d = tempfile.mkdtemp()
        p = os.path.join(d, name)
        with open(p, "w") as f:
            f.write(transcript(calls))
        return corrections.extract(p)

    def test_failed_then_fixed_define_is_one_record_cut_to_the_definition(self):
        r = self.extract([
            ("Bash", define(BAD), fail("E0201", "f", "body.items[1]", "unknown name `x`"), False),
            ("Bash", define(GOOD), done("helper", "f", "tail"), False),
        ])
        self.assertEqual(r, [{
            "code": "E0201", "path": "body.items[1]", "message": "unknown name `x`",
            "before": "fn f(n: Int) -> Int = { let x = n + 1; x }",
            "after": "fn f(n: Int) -> Int = let x = n + 1 in x",
            "agent": "a1",
        }])

    def test_a_failed_define_never_fixed_yields_none(self):
        self.assertEqual(self.extract([("Bash", define(BAD), fail("E0201", "f"), False)]), [])

    def test_json_calls_and_the_newest_failure_count(self):
        call = lambda src: {"command": "python3 qalc.py call qal_define '" + json.dumps({"source": src}) + "'"}
        r = self.extract([
            ("Bash", call("fn f() -> Int = y"), fail("E0201", "f", "a", "m1"), False),
            ("Bash", call("fn f() -> Int = z"), fail("E0203", "f", "b", "m2"), False),
            ("Bash", call("fn f() -> Int = 1"), done("f"), False),
        ])
        self.assertEqual([(x["code"], x["before"], x["after"]) for x in r], [("E0203", "fn f() -> Int = z", "fn f() -> Int = 1")])

    def test_other_names_empty_defs_and_unchanged_text_record_nothing(self):
        r = self.extract([
            ("Bash", define("fn f() -> Int = y"), fail("E0201", "f"), False),
            ("Bash", define("fn g() -> Int = 1"), done("g"), False),       # another name
            ("Bash", define("fn h() -> Int = 1 +"), fail("E0101", ""), False),  # parse error: no def
            ("Bash", define("fn h() -> Int = 1"), done("h"), False),
            ("Bash", define("fn f() -> Int = y"), done("f"), False),       # same text: no change
        ])
        self.assertEqual(r, [])

    def test_type_and_test_definitions_are_cut_too(self):
        r = self.extract([
            ("Bash", define("type T = A | B(String)\nfn k() -> Int = 1"), fail("E0204", "T", "t", "unknown type `String`"), False),
            ("Bash", define("type T = A | B(Str)\nfn k() -> Int = 1"), done("T", "k"), False),
        ])
        self.assertEqual((r[0]["before"], r[0]["after"]), ("type T = A | B(String)", "type T = A | B(Str)"))

    def test_a_failure_whose_definition_is_not_in_the_source_drops_the_older_pending_one(self):
        r = self.extract([
            ("Bash", define("fn f() -> Int = y"), fail("E0201", "f"), False),
            ("Bash", define("fn g() -> Int = 1"), fail("E0201", "f"), False),  # f is not in this source
            ("Bash", define("fn f() -> Int = 1"), done("f"), False),
        ])
        self.assertEqual(r, [])

    def test_placeholders_and_rewrites_are_skipped(self):
        long_bad = "fn r(n: Int) -> Int = {\n  let a = n in\n  let b = a in\n  print(a);\n  b\n}"
        rewrite = "fn r(n: Int) -> Int =\n  if n > 0\n  then 1\n  else 0"
        reindented = "fn r(n: Int) -> Int = {\n    let a = n in\n    let b = a in\n    print(a);\n    b + 0\n}"
        r = self.extract([
            # A placeholder the agent invented (`fn_*`), named by the diagnostic or as the definition.
            ("Bash", define("fn f() -> Int = fn_dummy(1)"), fail("E0201", "f", message="unknown name `fn_dummy`"), False),
            ("Bash", define("fn f() -> Int = 1"), done("f"), False),
            ("Bash", define("fn fn_x() -> Int = y"), fail("E0201", "fn_x"), False),
            ("Bash", define("fn fn_x() -> Int = 1"), done("fn_x"), False),
            # A rewrite: before and after share fewer than half of their lines.
            ("Bash", define(long_bad), fail("E0201", "r", message="unknown name `a`"), False),
            ("Bash", define(rewrite), done("r"), False),
            # Indentation does not count, and one-line texts are compared by their words.
            ("Bash", define(long_bad), fail("E0201", "r", message="unknown name `a`"), False),
            ("Bash", define(reindented), done("r"), False),
            ("Bash", define("fn g(xs: List[Str]) -> List[Int] = map(xs, len)"), fail("E0201", "g", message="built-in `len` is not a value"), False),
            ("Bash", define("fn g(xs: List[Str]) -> List[Int] = map(xs, slen)"), done("g"), False),
        ])
        self.assertEqual([(x["before"].split("\n")[0], x["after"].split("\n")[0]) for x in r], [
            ("fn r(n: Int) -> Int = {", "fn r(n: Int) -> Int = {"),
            ("fn g(xs: List[Str]) -> List[Int] = map(xs, len)", "fn g(xs: List[Str]) -> List[Int] = map(xs, slen)"),
        ])

    def test_shared_part(self):
        self.assertEqual(corrections.shared("a\n b\nc", "a\nb\nd"), 2 / 3)
        self.assertEqual(corrections.shared("f(x, len)", "f(x, slen)"), 1 / 2)


class Main(unittest.TestCase):
    def test_results_file_with_blank_and_non_json_lines_is_read(self):
        d = tempfile.mkdtemp()
        res = os.path.join(d, "results.jsonl")
        out = os.path.join(d, "out.json")
        with open(res, "w") as f:
            f.write('\n   \nnot json\n{"agent_id": null}\n[1, 2]\n')
        corrections.main(["--results", res, "--out", out])
        with open(out) as f:
            self.assertEqual(json.load(f), [])


if __name__ == "__main__":
    unittest.main()
