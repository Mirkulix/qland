#!/usr/bin/env python3
"""Build the verified QAL seed corpus for Phase 8.

The corpus is a JSONL index of QAL programs with their origin and the result of
`qal check`: `ok` or the exact diagnostics. It intentionally includes the
invalid programs of the extracted corpus, because those are the error traces a
model has to learn from.

This is the repository-side seed, not the 5,000-20,000-program corpus of the
Phase 8 specification. Model-written solutions and LoRA traces are added to the
same format later; see README.md.

  python3 bench/corpus/build.py [--qal PATH] [--out DIR] [--limit N]
"""

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_QAL = os.path.join(REPO, "qal", "target", "release", "qal")


def sources():
    patterns = [
        "qal/selfhost/corpus/*.qal",
        "bench/solutions/qal/*.qal",
        "qal/examples/*.qal",
    ]
    out = []
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(REPO, pattern))):
            out.append((os.path.relpath(path, REPO), path))
    return out


def check(qal, path):
    res = subprocess.run([qal, "check", path], capture_output=True, text=True, timeout=60)
    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError:
        return "error", [{"code": "E0000", "kind": "check_failed",
                          "message": (res.stdout or res.stderr).strip()[:500]}]
    if "ok" in data:
        return "ok", []
    return "error", data.get("diagnostics", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qal", default=os.environ.get("QAL_BIN", DEFAULT_QAL))
    ap.add_argument("--out", default=os.path.join(REPO, "bench", "corpus"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.qal):
        sys.exit(f"build: {args.qal} is missing; run ./install.sh or cargo build --release")

    os.makedirs(args.out, exist_ok=True)
    records, seen = [], set()
    for origin, path in sources():
        if args.limit and len(records) >= args.limit:
            break
        with open(path, encoding="utf-8") as f:
            source = f.read()
        digest = hashlib.sha256(source.encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        status, diagnostics = check(args.qal, path)
        records.append({
            "id": digest[:16],
            "origin": origin,
            "status": status,
            "diagnostics": diagnostics,
            "source": source,
        })

    out = os.path.join(args.out, "programs.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    corrections_path = os.path.join(args.out, "corrections.jsonl")
    corrections = 0
    if os.path.exists(corrections_path):
        with open(corrections_path, encoding="utf-8") as f:
            corrections = sum(1 for line in f if line.strip())
    summary = {
        "programs": len(records),
        "ok": sum(1 for r in records if r["status"] == "ok"),
        "error": sum(1 for r in records if r["status"] == "error"),
        "corrections": corrections,
        "sources": sorted({r["origin"].split("/")[0] for r in records}),
        "phase8_target": [5000, 20000],
        "model_written_solutions": 0,
        "note": "Repo-side seed; model-written solutions and LoRA traces are not part of this file yet. corrections.jsonl holds the real fixes from the benchmark transcripts.",
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
