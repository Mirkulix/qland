#!/usr/bin/env python3
"""Builds the task set of the living program: bench/living/tasks/<NN-stage>/{task.md, tests.json}
for stages 0..12, from `spec.py` (text) and `oracle.py` (expected outputs). Each stage's hidden
tests are hand-written scripts for its rules plus regression scripts of the earlier stages
(re-evaluated under the new rules) plus seeded random command streams.

  python3 bench/living/build.py            # writes the task set, prints a summary
  python3 bench/living/build.py --check    # only checks the public examples against the oracle
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oracle  # noqa: E402
import spec  # noqa: E402

PRESSURE = os.environ.get("LIVING_MODE") == "pressure"
TASKS = os.path.join(HERE, "tasks-pressure" if PRESSURE else "tasks")
STAGES = 13

BASE = "product p1 widget 1000 5\nproduct p2 gadget 250 20\ncustomer c1 ann\ncustomer c2 bob\n"

# Hand-written scripts per stage: (name, stdin). Two public ones per stage (`example`, `second`).
SCRIPTS = {
    0: [
        ("example", BASE + "order o1 c1 p1 2\norder o2 c2 p2 25\norder o3 c1 p2 20\nreport\n"),
        ("second", "product p1 a 10 1\ncustomer c1 x\norder o1 c1 p1 1\norder o1 c1 p1 1\norder o2 c9 p1 1\norder o3 c1 p9 1\nreport\n"),
        ("empty", ""),
        ("invalid_lines", "product p1 a 0 1\nproduct p2 b 10 -1\nproduct p3 c 10\ncustomer\norder o1 c1 p1 0\nfoo\nproduct p-1 a 1 1\n\nreport extra\nreport\n"),
        ("duplicates", BASE + "product p1 again 1 1\ncustomer c1 again\norder o1 c1 p1 1\norder o1 c2 p2 1\nreport\n"),
        ("error_order", BASE + "order o1 c9 p9 1\norder o2 c1 p9 1\norder o1 c9 p9 1\nreport\n"),
        ("stock_edge", BASE + "order o1 c1 p1 5\norder o2 c1 p1 1\norder o3 c2 p1 0\norder o4 c2 p2 20\norder o5 c2 p2 1\nreport\n"),
        ("big_numbers", "product p1 a 999999999 999999999\ncustomer c1 x\norder o1 c1 p1 999999999\nreport\n"),
        ("no_report", BASE + "order o1 c1 p1 1\n"),
        ("order_of_ids", "product B b 1 9\nproduct a a 1 9\nproduct _ u 1 9\ncustomer Z z\ncustomer b b\norder o1 b B 1\norder o2 Z a 2\nreport\n"),
    ],
    1: [
        ("example", BASE + "order o1 c1 p2 9\norder o2 c1 p2 10\norder o3 c2 p1 3\nreport\n"),
        ("second", "product p1 a 7 100\ncustomer c1 x\norder o1 c1 p1 10\norder o2 c1 p1 11\norder o3 c1 p1 1\nreport\n"),
        ("rounding", "product p1 a 1 100\nproduct p2 b 3 100\ncustomer c1 x\norder o1 c1 p1 10\norder o2 c1 p2 13\norder o3 c1 p1 19\nreport\n"),
        ("discount_not_on_reject", BASE + "order o1 c1 p1 10\norder o2 c1 p2 20\nreport\n"),
    ],
    2: [
        ("example", "product p1 a 1000 100\ncustomer c1 ann gold\ncustomer c2 bob silver\ncustomer c3 cy\norder o1 c1 p1 1\norder o2 c2 p1 1\norder o3 c3 p1 1\norder o4 c1 p1 10\nreport\n"),
        ("second", "product p1 a 999 100\ncustomer c1 ann basic\ncustomer c2 bob platinum\ncustomer c3 cy gold extra\norder o1 c1 p1 10\norder o2 c2 p1 1\nreport\n"),
        ("tier_rounding", "product p1 a 101 100\ncustomer g g gold\ncustomer s s silver\norder o1 g p1 3\norder o2 s p1 3\norder o3 g p1 10\norder o4 s p1 10\nreport\n"),
    ],
    3: [
        ("example", BASE + "order o1 c1 p1 2\ncancel o1\ncancel o1\ncancel o9\norder o2 c1 p1 5\norder o3 c2 p1 1\ncancel o3\norder o4 c2 p1 1\nreport\n"),
        ("second", "product p1 a 10 1\ncustomer c1 x gold\norder o1 c1 p1 1\ncancel o1\norder o2 c1 p1 1\ncancel o2\ncancel\ncancel o2 now\nreport\n"),
        ("cancel_restores_discounted", "product p1 a 100 20\ncustomer c1 x silver\norder o1 c1 p1 10\ncancel o1\norder o2 c1 p1 10\nreport\n"),
    ],
    4: [
        ("example", BASE + "tax 19\norder o1 c1 p1 1\norder o2 c2 p2 3\ntax 0\norder o3 c1 p2 1\ncancel o1\nreport\n"),
        ("second", "product p1 a 999 100\ncustomer c1 x gold\ntax 7\norder o1 c1 p1 10\ntax 101\ntax -1\ntax 100\norder o2 c1 p1 1\nreport\n"),
        ("tax_at_order_time", "product p1 a 100 10\ncustomer c1 x\ntax 10\norder o1 c1 p1 1\ntax 20\ncancel o1\norder o2 c1 p1 1\nreport\n"),
    ],
    5: [
        ("example", BASE + "limit c1 1500\ntax 10\norder o1 c1 p1 1\norder o2 c1 p1 1\npay c1 600\norder o3 c1 p1 1\npay c1 2000\nlimit c9 1\nreport\n"),
        ("second", "product p1 a 100 10\ncustomer c1 x\nlimit c1 0\norder o1 c1 p1 1\npay c1 0\npay c9 1\nlimit c1 100\norder o2 c1 p1 1\norder o3 c1 p1 1\ncancel o2\norder o4 c1 p1 1\nreport\n"),
        ("credit_after_stock", "product p1 a 100 1\ncustomer c1 x\nlimit c1 50\norder o1 c1 p1 2\norder o2 c1 p1 1\nreport\n"),
        ("cancel_not_below_zero", "product p1 a 100 10\ncustomer c1 x\norder o1 c1 p1 1\npay c1 100\ncancel o1\norder o2 c1 p1 2\npay c1 50\nreport\n"),
    ],
    6: [
        ("example", BASE + "order o1 c1 p1 5\norder o2 c1 p1 1\nrestock p1 3\norder o3 c1 p1 3\nrestock p9 1\nrestock p1 0\nreport\n"),
        ("second", "product p1 a 1 0\ncustomer c1 x\nrestock p1 999999999\norder o1 c1 p1 999999999\nreport\n"),
    ],
    7: [
        ("example", BASE + "order o1 c1 p1 6\norder o2 c2 p1 2\norder o3 c1 p1 1\nrestock p1 2\nrestock p1 10\ncancel o1\nreport\n"),
        ("second", "product p1 a 100 0\ncustomer c1 x\nlimit c1 150\norder o1 c1 p1 1\norder o2 c1 p1 1\nrestock p1 5\ncancel o2\ncancel o1\nreport\n"),
        ("fifo_skip", "product p1 a 10 0\ncustomer c1 x\norder o1 c1 p1 5\norder o2 c1 p1 1\nrestock p1 1\nrestock p1 4\nreport\n"),
        ("backorder_exit_code", "product p1 a 10 0\ncustomer c1 x\norder o1 c1 p1 1\nreport\n"),
    ],
    8: [
        ("example", BASE + "order o1 c1 p1 6\nprice p1 2000\ntax 10\nrestock p1 1\nprice p9 1\nprice p1 0\norder o2 c2 p1 0\nreport\n"),
        ("second", "product p1 a 100 0\ncustomer c1 x silver\norder o1 c1 p1 10\nprice p1 50\nrestock p1 10\nreport\n"),
    ],
    9: [
        ("example", BASE + "product p3 c 1 50\norder o1 c1 p1 3\norder o2 c2 p2 3\norder o3 c1 p3 4\ntop\ncancel o3\ntop\norder o4 c2 p1 1\ntop\nreport\n"),
        ("second", "product p1 a 1 9\ntop\ntop now\norder o1 c9 p1 1\nreport\n"),
    ],
    10: [
        ("example", BASE + "tax 10\norder o1 c1 p2 10\nreturn o1 3\nreturn o1 7\nreturn o1 1\ncancel o1\nreturn o9 1\nreport\n"),
        ("second", "product p1 a 999 10\ncustomer c1 x gold\nlimit c1 100000\norder o1 c1 p1 3\npay c1 2000\nreturn o1 2\nreturn o1 0\nreturn o1\norder o2 c1 p1 10\nreturn o2 1\nreport\n"),
        ("return_pending", "product p1 a 10 0\ncustomer c1 x\norder o1 c1 p1 1\nreturn o1 1\ncancel o1\nreturn o1 1\nreport\n"),
    ],
    11: [
        ("example", BASE + "promo p1 20\norder o1 c1 p1 1\npromo p1 0\norder o2 c1 p1 1\npromo p2 50\ncustomer c3 cy gold\norder o3 c3 p2 10\npromo p2 51\npromo p9 1\nreport\n"),
        ("second", "product p1 a 1000 100\ncustomer c1 x\npromo p1 50\norder o1 c1 p1 10\npromo p1 45\norder o2 c1 p1 10\nreport\n"),
    ],
    12: [
        ("example", BASE + "order o1 c1 p1:2,p2:8\norder o2 c2 p1:3,p9:1\norder o3 c2 p1:4,p2:1\nrestock p1 10\nreturn o1 3\nreport\ntop\n"),
        ("second", "product p1 a 100 10\nproduct p2 b 10 10\ncustomer c1 x silver\npromo p2 40\ntax 10\norder o1 c1 p1:5,p2:5\norder o2 c1 p1 5\norder o3 c1 p1:0\norder o4 c1 p1:1,\norder o5 c1 p1:1,p1:1\nreport\n"),
    ],
}

COMMANDS = ["product", "customer", "order", "report", "cancel", "tax", "limit", "pay", "restock", "price", "top", "return", "promo"]


def random_script(rng, stage, n_lines):
    pids = [f"p{i}" for i in range(1, 6)]
    cids = [f"c{i}" for i in range(1, 4)]
    lines = []
    oids = []
    for pid in pids[:3]:
        lines.append(f"product {pid} n{pid} {rng.choice([1, 7, 99, 250, 1000, 12345])} {rng.choice([0, 1, 3, 10, 50])}")
    for cid in cids[:2]:
        tier = f" {rng.choice(['basic', 'silver', 'gold'])}" if stage >= 2 and rng.random() < 0.6 else ""
        lines.append(f"customer {cid} n{cid}{tier}")
    k = 0
    while len(lines) < n_lines:
        cmd = rng.choice(COMMANDS)
        if cmd == "product":
            lines.append(f"product {rng.choice(pids)} x {rng.choice([1, 50, 1000])} {rng.choice([0, 2, 20])}")
        elif cmd == "customer":
            lines.append(f"customer {rng.choice(cids)} y")
        elif cmd == "order":
            k += 1
            oid = f"o{k}" if rng.random() < 0.9 else rng.choice(oids or ["o1"])
            oids.append(oid)
            if stage >= 12 and rng.random() < 0.4:
                parts = ",".join(f"{rng.choice(pids)}:{rng.choice([1, 2, 5, 10])}" for _ in range(rng.choice([1, 2, 3])))
                lines.append(f"order {oid} {rng.choice(cids)} {parts}")
            else:
                lines.append(f"order {oid} {rng.choice(cids)} {rng.choice(pids)} {rng.choice([1, 2, 5, 9, 10, 11, 25])}")
        elif cmd == "report":
            lines.append("report")
        elif cmd == "cancel":
            lines.append(f"cancel {rng.choice(oids or ['o1'])}")
        elif cmd == "tax":
            lines.append(f"tax {rng.choice([0, 7, 19, 100])}")
        elif cmd == "limit":
            lines.append(f"limit {rng.choice(cids)} {rng.choice([0, 500, 5000, 100000])}")
        elif cmd == "pay":
            lines.append(f"pay {rng.choice(cids)} {rng.choice([1, 100, 1000])}")
        elif cmd == "restock":
            lines.append(f"restock {rng.choice(pids)} {rng.choice([1, 5, 30])}")
        elif cmd == "price":
            lines.append(f"price {rng.choice(pids)} {rng.choice([1, 60, 2000])}")
        elif cmd == "top":
            lines.append("top")
        elif cmd == "return":
            lines.append(f"return {rng.choice(oids or ['o1'])} {rng.choice([1, 2, 10])}")
        elif cmd == "promo":
            lines.append(f"promo {rng.choice(pids)} {rng.choice([0, 10, 50])}")
    lines.append("report")
    return "\n".join(lines) + "\n"


def cases_of(stage):
    cases = []
    seen = set()

    def add(name, stdin, public):
        if name in seen:
            name = f"{name}_{len(seen)}"
        seen.add(name)
        stdout, code = oracle.run(stdin, stage)
        cases.append({"name": name, "args": [], "stdin": stdin, "files": {}, "public": public, "stdout": stdout, "exit_code": code})

    for name, stdin in SCRIPTS[stage]:
        add(name, stdin, name in ("example", "second"))
    # Regression: the scripts of earlier stages, re-evaluated under this stage's rules.
    for s in range(stage):
        for name, stdin in SCRIPTS[s]:
            add(f"s{s}_{name}", stdin, False)
    rng = random.Random(f"living-{stage}")
    for i in range(6):
        add(f"random_{i}", random_script(rng, stage, 25 + 10 * i), False)
    return cases


def main(argv):
    check_only = "--check" in argv
    for stage in range(STAGES):
        cases = cases_of(stage)
        public = [(c["name"], c["stdin"], c["stdout"], c["exit_code"]) for c in cases if c["public"]]
        text = spec.statement(stage, public, PRESSURE)
        # The statement's examples must be what the oracle computes.
        for name, stdin, stdout, code in public:
            block = text.split(f"### Example `{name}`")[1]
            got = block.split("Standard output (exit code")[1].split("```")[1]
            assert got == "\n" + stdout or (stdout == "" and got == "\n"), (stage, name, got, stdout)
        tid = f"{stage + 30:02d}-living-{'abcdefghijklm'[stage]}"
        if not check_only:
            d = os.path.join(TASKS, tid)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "task.md"), "w", encoding="utf-8") as f:
                f.write(text)
            with open(os.path.join(d, "tests.json"), "w", encoding="utf-8") as f:
                json.dump({"id": tid, "title": f"{spec.TITLE} stage {stage}", "effects": ["io"], "cases": cases}, f, indent=1, ensure_ascii=False)
                f.write("\n")
        fails = sum(c["exit_code"] for c in cases)
        print(f"{tid}: {len(cases)} cases ({fails} with exit 1), statement {len(text)} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
