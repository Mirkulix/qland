#!/usr/bin/env python3
"""Verify that the seed corpus still matches `qal check`.

The builder stores each program's status and diagnostics. This checker re-runs a
sample (or all programs) from `programs.jsonl`, writing the source to a temporary
file, and compares only the status: `ok` must be ok, `error` must produce at
least one diagnostic. Diagnostics text is not frozen, so improvements to the
checker do not fail the corpus.

  python3 bench/corpus/verify.py             # 50 evenly spaced programs
  python3 bench/corpus/verify.py --all       # all programs
  python3 bench/corpus/verify.py --limit 10  # a smaller sample
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_QAL = os.path.join(REPO, "qal", "target", "release", "qal")
CORPUS = os.path.join(REPO, "bench", "corpus", "programs.jsonl")


def load(limit, all_programs):
    with open(CORPUS, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if all_programs or limit <= 0 or limit >= len(records):
        return records
    step = max(1, len(records) // limit)
    return records[::step][:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qal", default=os.environ.get("QAL_BIN", DEFAULT_QAL))
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(args.qal):
        sys.exit(f"verify: {args.qal} is missing; run ./install.sh or cargo build --release")
    if not os.path.exists(CORPUS):
        sys.exit(f"verify: {CORPUS} is missing; run python3 bench/corpus/build.py")

    checked = failed = 0
    for record in load(args.limit, args.all):
        checked += 1
        with tempfile.NamedTemporaryFile("w", suffix=".qal", delete=False, encoding="utf-8") as f:
            f.write(record["source"])
            path = f.name
        try:
            res = subprocess.run([args.qal, "check", path], capture_output=True, text=True, timeout=60)
            data = json.loads(res.stdout)
            ok = "ok" in data
            diagnostics = [] if ok else data.get("diagnostics", [])
            good = (record["status"] == "ok" and ok) or (record["status"] == "error" and not ok and diagnostics)
            if not good:
                failed += 1
                print(f"MISMATCH {record['id']} ({record['origin']}): stored {record['status']}, "
                      f"now {'ok' if ok else 'error'}", file=sys.stderr)
        finally:
            os.unlink(path)

    print(json.dumps({"checked": checked, "failed": failed}, indent=2))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
