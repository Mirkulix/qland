#!/usr/bin/env python3
"""Extracts the QAL programs embedded in the Rust tests into qal/selfhost/corpus/ (spec §10.1 (2)).

Every Rust string literal in qal/crates/**/*.rs that holds QAL source (a line starting with
`fn`, `type` or `test` followed by a name, or a `#` comment followed by one) becomes one file
`corpus/<first 16 hex digits of its SHA-256>.qal`, deduplicated by content. Literals that are
format strings (`format!(`, `println!(`, `write!(`, assertion messages) are skipped: their
`{}` holes are not QAL. Programs that do not parse or check are kept on purpose: they cover the
diagnostic paths.

Usage: python3 qal/selfhost/tools/extract_corpus.py [--check]
  --check  only verify that corpus/ matches the sources (exit 1 if not); writes nothing.
"""
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
QAL = os.path.normpath(os.path.join(HERE, "..", ".."))
CRATES = os.path.join(QAL, "crates")
OUT = os.path.join(QAL, "selfhost", "corpus")

DEF = re.compile(r"(?m)^[ \t]*(fn|type|test)[ \t]+[A-Za-z_]")
FORMAT_CALLS = ("format!(", "println!(", "eprintln!(", "print!(", "write!(", "writeln!(", "panic!(", "format_args!(")


def unescape(body):
    out, i = [], 0
    while i < len(body):
        c = body[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        n = body[i + 1]
        if n == "\n":  # line continuation: skip the newline and leading whitespace
            i += 2
            while i < len(body) and body[i] in " \t\n\r":
                i += 1
            continue
        simple = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\", '"': '"', "'": "'"}
        if n in simple:
            out.append(simple[n])
            i += 2
        elif n == "x":
            out.append(chr(int(body[i + 2 : i + 4], 16)))
            i += 4
        elif n == "u":
            end = body.index("}", i)
            out.append(chr(int(body[i + 3 : end], 16)))
            i = end + 1
        else:
            raise ValueError("unknown escape \\" + n)
    return "".join(out)


def literals(src):
    """(start offset, text) of every string literal, skipping comments and char literals."""
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            i = src.find("\n", i)
            i = n if i < 0 else i
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        m = re.match(r'b?r(#*)"', src[i:])
        if m and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            hashes = m.group(1)
            start = i + m.end()
            end = src.find('"' + hashes, start)
            yield i, src[start:end], src[i] == "b"
            i = end + 1 + len(hashes)
            continue
        if c == "'":
            # char literal ('a', '\n', '\u{..}') or a lifetime ('a)
            m = re.match(r"'(\\u\{[0-9a-fA-F]+\}|\\.|[^\\'])'", src[i:])
            i += m.end() if m else 1
            continue
        if c == '"' or (c == "b" and src.startswith('b"', i) and not (src[i - 1].isalnum() or src[i - 1] == "_")):
            is_bytes = c == "b"
            start = i + (2 if is_bytes else 1)
            j = start
            while src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            yield i, unescape(src[start:j]), is_bytes
            i = j + 1
            continue
        i += 1


def is_format_string(src, at):
    before = src[max(0, at - 40) : at].rstrip()
    return any(before.endswith(f.rstrip("(")) or before.endswith(f) for f in FORMAT_CALLS)


def corpus():
    found = {}
    for root, _, files in os.walk(CRATES):
        for f in sorted(files):
            if not f.endswith(".rs"):
                continue
            path = os.path.join(root, f)
            src = open(path, encoding="utf-8").read()
            for at, text, is_bytes in literals(src):
                if is_bytes or not DEF.search(text) or is_format_string(src, at):
                    continue
                text = text if text.endswith("\n") else text + "\n"
                h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                found.setdefault(h, text)
    return found


def main():
    found = corpus()
    existing = {f[:-4] for f in os.listdir(OUT) if f.endswith(".qal")} if os.path.isdir(OUT) else set()
    if "--check" in sys.argv:
        ok = existing == set(found) and all(open(os.path.join(OUT, h + ".qal"), encoding="utf-8").read() == t for h, t in found.items())
        print(f"{len(found)} programs extracted, corpus {'up to date' if ok else 'OUT OF DATE'}")
        sys.exit(0 if ok else 1)
    os.makedirs(OUT, exist_ok=True)
    for h in existing - set(found):
        os.remove(os.path.join(OUT, h + ".qal"))
    for h, text in found.items():
        with open(os.path.join(OUT, h + ".qal"), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    print(f"{len(found)} programs in {os.path.relpath(OUT, QAL)}")


if __name__ == "__main__":
    main()
