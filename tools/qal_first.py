#!/usr/bin/env python3
"""Inventory of the QAL-first surface: QAL vs host code, packages, tests."""
import argparse, json, os, re
from pathlib import Path

SKIP = {".git", "target", "node_modules", ".qal", "__pycache__", ".venv", "dist"}
EXTS = {".qal": "qal", ".rs": "rust", ".py": "python", ".ts": "typescript", ".tsx": "typescript", ".js": "javascript", ".md": "docs"}


def scan(root):
    counts = {}
    lines = {}
    files = []
    packages = []
    tests = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP]
        for name in filenames:
            p = Path(dirpath) / name
            ext = p.suffix
            if ext == ".qalpkg":
                packages.append(p.relative_to(root).as_posix())
                continue
            kind = EXTS.get(ext)
            if not kind:
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            counts[kind] = counts.get(kind, 0) + 1
            lines[kind] = lines.get(kind, 0) + text.count("\n") + 1
            if ext == ".qal":
                tests += len(re.findall(r"(?m)^\s*test\s+", text))
    return {"files": counts, "lines": lines, "qal_tests": tests, "packages": sorted(packages)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    data = scan(Path(a.root).resolve())
    if a.json:
        print(json.dumps(data, indent=2))
    else:
        print(f"qal: {data['files'].get('qal',0)} files, {data['lines'].get('qal',0)} lines, {data['qal_tests']} tests")
        print(f"rust: {data['files'].get('rust',0)} files, {data['lines'].get('rust',0)} lines")
        print(f"python: {data['files'].get('python',0)} files, {data['lines'].get('python',0)} lines")
        print(f"typescript: {data['files'].get('typescript',0)} files")
        print(f"packages: {len(data['packages'])}")


if __name__ == "__main__":
    main()
