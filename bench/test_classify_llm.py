#!/usr/bin/env python3
"""Tests of the Haiku pass over results no rule classifies (Phase 6, Task 2):

  cd bench && python3 -m unittest test_classify_llm -v
"""

import json
import os
import sys
import tempfile
import unittest

BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH)
import classify_llm  # noqa: E402


class FakeMessages:
    def __init__(self, answer, usage):
        self.answer, self.usage, self.calls = answer, usage, 0

    def create(self, **kw):
        self.calls += 1
        self.kw = kw
        return type("R", (), {"content": [type("B", (), {"type": "text", "text": self.answer})()], "usage": type("U", (), self.usage)()})()


class FakeClient:
    def __init__(self, answer, usage=None):
        self.messages = FakeMessages(answer, usage or {"input_tokens": 500, "output_tokens": 20})

    @property
    def calls(self):
        return self.messages.calls


class Classify(unittest.TestCase):
    def test_answer_is_parsed(self):
        cat, reason, usage = classify_llm.classify(FakeClient("lang.type | missing annotation"), "isError: true …", "Bash", "qal")
        self.assertEqual((cat, reason), ("lang.type", "missing annotation"))
        self.assertEqual(usage, {"input_tokens": 500, "output_tokens": 20})

    def test_bad_answer_stays_unclassified(self):
        cat, _, usage = classify_llm.classify(FakeClient("banana"), "…", "Bash", "qal")
        self.assertEqual(cat, "unclassified")
        self.assertGreater(usage["input_tokens"], 0)

    def test_request_uses_haiku_low_effort(self):
        c = FakeClient("ok.other | fine")
        classify_llm.classify(c, "text", "Bash", "qal")
        kw = c.messages.kw
        self.assertEqual(kw["model"], "claude-haiku-5-5")
        self.assertEqual(kw["extra_body"], {"output_config": {"effort": "low"}, "thinking": {"type": "disabled"}})
        self.assertLessEqual(kw["max_tokens"], 150)
        self.assertIn("lang.type", kw["system"])

    def test_angle_brackets_and_backticks_around_the_category_are_tolerated(self):
        cat, reason, _ = classify_llm.classify(FakeClient("<lang.run_failed> | trap"), "…", "Bash", "qal")
        self.assertEqual((cat, reason), ("lang.run_failed", "trap"))
        cat, _, _ = classify_llm.classify(FakeClient("`env.tool_error` | missing file"), "…", "Bash", "qal")
        self.assertEqual(cat, "env.tool_error")


class BudgetAndRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cache = os.path.join(self.tmp, "cache.jsonl")
        self.rows = [
            {"key": "a", "arm": "qal", "unclassified": [{"call": 2, "tool": "Bash", "text": "one"}, {"call": 3, "tool": "Bash", "text": "two"}]},
            {"key": "b", "arm": "rust", "unclassified": [{"call": 2, "tool": "Bash", "text": "one"}]},
        ]

    def test_budget_charges_haiku_prices(self):
        b = classify_llm.Budget(limit_eur=20.0)
        cost = b.charge({"input_tokens": 1_000_000, "output_tokens": 1_000_000})
        self.assertAlmostEqual(cost, 0.60, places=6)
        self.assertAlmostEqual(b.spent_eur, 0.60, places=6)
        self.assertFalse(b.exhausted())

    def test_budget_stops_calls(self):
        b = classify_llm.Budget(limit_eur=0.000001)
        b.charge({"input_tokens": 10_000, "output_tokens": 100})
        self.assertTrue(b.exhausted())
        c = FakeClient("ok.other | x")
        n_open = classify_llm.run(self.rows, c, b, self.cache)
        self.assertEqual(c.calls, 0)
        self.assertEqual(n_open, 2)

    def test_cache_prevents_second_call(self):
        c = FakeClient("ok.other | x")
        n_open = classify_llm.run(self.rows, c, classify_llm.Budget(limit_eur=20.0), self.cache)
        self.assertEqual(n_open, 0)
        self.assertEqual(c.calls, 2)  # "one" twice, "two" once -> two distinct texts
        lines = [json.loads(l) for l in open(self.cache)]
        self.assertEqual(len(lines), 2)
        self.assertEqual(set(lines[0]), {"sha256", "category", "reason", "usage"})
        c2 = FakeClient("ok.other | x")
        classify_llm.run(self.rows, c2, classify_llm.Budget(limit_eur=20.0), self.cache)
        self.assertEqual(c2.calls, 0)

    def test_run_also_sends_the_reclassify_texts(self):
        rows = [{"key": "a", "arm": "qal", "unclassified": [], "reclassify": [{"call": 2, "tool": "Bash", "category": "lang.other_diag", "text": "three"}]}]
        c = FakeClient("lang.run_failed | trap")
        self.assertEqual(classify_llm.run(rows, c, classify_llm.Budget(limit_eur=20.0), self.cache), 0)
        self.assertEqual(c.calls, 1)
        self.assertEqual(json.loads(open(self.cache).readline())["category"], "lang.run_failed")

    def test_cached_unclassified_verdicts_are_asked_again(self):
        c = FakeClient("banana")
        classify_llm.run(self.rows, c, classify_llm.Budget(limit_eur=20.0), self.cache)
        self.assertEqual(c.calls, 2)
        c2 = FakeClient("ok.other | fine")
        classify_llm.run(self.rows, c2, classify_llm.Budget(limit_eur=20.0), self.cache)
        self.assertEqual(c2.calls, 2)
        self.assertEqual(sum(1 for l in open(self.cache) if json.loads(l)["category"] == "ok.other"), 2)

    def test_run_reports_open_rest_when_budget_ends_midway(self):
        b = classify_llm.Budget(limit_eur=0.0004)  # one call of 500 in + 20 out costs 0.00006 -> budget ends after the first
        c = FakeClient("ok.other | x", usage={"input_tokens": 4_000_000, "output_tokens": 1})
        n_open = classify_llm.run(self.rows, c, b, self.cache)
        self.assertEqual(c.calls, 1)
        self.assertEqual(n_open, 1)
        self.assertEqual(len(open(self.cache).read().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
