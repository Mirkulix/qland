#!/usr/bin/env python3
"""A tiny content-addressed QAL package registry.

Packages are the compiler's own `.qalpkg` files. The registry stores them by
SHA-256 and keeps an index of package name -> hash. It does not replace
`qal import`; it verifies and serves the bytes that `qal import` consumes.

  python3 qal-registry/registry.py publish math.qalpkg
  python3 qal-registry/registry.py index
  python3 qal-registry/registry.py fetch <sha256> --out math.qalpkg
  python3 qal-registry/registry.py verify <sha256>
"""
import argparse, hashlib, json, os, shutil, sys, tempfile
from pathlib import Path

ROOT = Path(os.environ.get("QAL_REGISTRY", Path(__file__).resolve().parent / "store"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def header(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()[:3]
    if len(lines) < 2 or not lines[0].startswith("# qal-pkg"):
        sys.exit(f"{path}: not a qal-pkg file")
    parts = lines[1].split()
    if len(parts) < 2 or parts[0] != "#":
        sys.exit(f"{path}: malformed package header")
    name = parts[2] if len(parts) > 2 else ""
    defs = sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines()
               if line.startswith("# def "))
    return name, defs


def load_index():
    p = ROOT / "index.json"
    if not p.exists():
        return {"packages": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def save_index(idx):
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "index.json").write_text(json.dumps(idx, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cmd_publish(args):
    src = Path(args.package)
    name, defs = header(src)
    digest = sha256(src)
    dest = ROOT / "packages" / f"{digest}.qalpkg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(src, dest)
    idx = load_index()
    idx["packages"][digest] = {"name": name, "defs": defs, "file": f"packages/{digest}.qalpkg"}
    save_index(idx)
    print(json.dumps({"published": name, "sha256": digest, "defs": defs, "path": str(dest)}, indent=2))


def cmd_fetch(args):
    idx = load_index()
    ent = idx["packages"].get(args.sha256)
    if not ent:
        sys.exit("unknown sha256")
    src = ROOT / ent["file"]
    if sha256(src) != args.sha256:
        sys.exit("registry corruption: hash mismatch")
    out = Path(args.out)
    shutil.copy2(src, out)
    print(json.dumps({"fetched": ent["name"], "sha256": args.sha256, "out": str(out)}, indent=2))


def cmd_verify(args):
    idx = load_index()
    ent = idx["packages"].get(args.sha256)
    if not ent:
        sys.exit("unknown sha256")
    got = sha256(ROOT / ent["file"])
    print(json.dumps({"sha256": args.sha256, "ok": got == args.sha256, "name": ent["name"]}, indent=2))
    sys.exit(0 if got == args.sha256 else 1)


def cmd_index(args):
    print(json.dumps(load_index(), indent=2, sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish"); p.add_argument("package"); p.set_defaults(fn=cmd_publish)
    f = sub.add_parser("fetch"); f.add_argument("sha256"); f.add_argument("--out", required=True); f.set_defaults(fn=cmd_fetch)
    v = sub.add_parser("verify"); v.add_argument("sha256"); v.set_defaults(fn=cmd_verify)
    i = sub.add_parser("index"); i.set_defaults(fn=cmd_index)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
