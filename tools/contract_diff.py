#!/usr/bin/env python3
"""Refuse silent deletion or weakening of QAL req/ens contracts.

  python3 tools/contract_diff.py --before old.qal --after new.qal
  python3 tools/contract_diff.py --base HEAD~1        # all changed *.qal in git

Exit 1 if a contract line disappears, or a comparison is weakened
(>= -> >, <= -> <, == -> !=), unless the change is explicitly acknowledged
with --allow-contract-change (still reported, exit 0).
"""
import argparse, json, re, subprocess, sys
from pathlib import Path

CONTRACT = re.compile(r"^\s*(req|ens)\b")


def contract_lines(path):
    out = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if CONTRACT.match(line):
            out.append((n, line.strip()))
    return out


def weakened(before, after):
    pairs = {">=": ">", "<=": "<", "==": "!="}
    for op, weak in pairs.items():
        if op in before and weak in after and op not in after:
            return True
    return False


def compare(before_path, after_path, label):
    before = contract_lines(before_path)
    after = contract_lines(after_path)
    before_set = {t for _, t in before}
    after_set = {t for _, t in after}
    findings = []
    for line_no, text in before:
        if text not in after_set:
            findings.append({"kind": "removed", "file": label, "line": line_no, "contract": text})
    for i, (line_no, text) in enumerate(after):
        if i < len(before) and text != before[i][1] and weakened(before[i][1], text):
            findings.append({"kind": "weakened", "file": label, "line": line_no,
                             "before": before[i][1], "after": text})
    return findings


def git_changed(base):
    r = subprocess.run(["git", "diff", "--name-only", base, "--", "*.qal"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(r.stderr or "git diff failed")
    return [line for line in r.stdout.splitlines() if line.endswith(".qal")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--before")
    ap.add_argument("--after")
    ap.add_argument("--allow-contract-change", action="store_true")
    args = ap.parse_args()
    findings = []
    if not (args.base or (args.before and args.after)):
        sys.exit("use --base, or --before and --after")
    if args.before or args.after:
        if not (args.before and args.after):
            sys.exit("--before and --after are required together")
        findings += compare(args.before, args.after, args.after)
    else:
        for path in git_changed(args.base):
            r = subprocess.run(["git", "show", f"{args.base}:{path}"], capture_output=True, text=True)
            if r.returncode != 0:
                continue
            import tempfile
            with tempfile.NamedTemporaryFile("w", suffix=".qal", delete=False, encoding="utf-8") as f:
                f.write(r.stdout)
                old = f.name
            try:
                findings += compare(old, path, path)
            finally:
                Path(old).unlink()
    print(json.dumps({"findings": findings, "allowed": args.allow_contract_change}, indent=2))
    if findings and not args.allow_contract_change:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
