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
import argparse, base64, datetime, hashlib, json, os, shutil, sys, tempfile
from pathlib import Path

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    from cryptography.hazmat.primitives import serialization
except Exception:
    Ed25519PrivateKey = Ed25519PublicKey = serialization = None

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


def _need_crypto():
    if Ed25519PrivateKey is None:
        sys.exit("python cryptography is required for signing")


def public_b64(key):
    raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


def sign_digest(digest, key_path):
    _need_crypto()
    key = serialization.load_pem_private_key(Path(key_path).read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        sys.exit("key is not Ed25519")
    return base64.b64encode(key.sign(digest.encode())).decode(), public_b64(key)


def verify_digest(digest, signature, public_key_b64):
    _need_crypto()
    try:
        pub = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64))
        pub.verify(base64.b64decode(signature), digest.encode())
        return True
    except Exception:
        return False


def trusted_keys():
    p = ROOT / "trusted.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("keys", [])


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
    entry = {"name": name, "defs": defs, "file": f"packages/{digest}.qalpkg",
             "version": args.version, "published_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if args.key:
        sig, pub = sign_digest(digest, args.key)
        entry["signature"] = sig
        entry["public_key"] = pub
    idx = load_index()
    idx["packages"][digest] = entry
    save_index(idx)
    print(json.dumps({"published": name, "sha256": digest, "defs": defs, "path": str(dest),
                      "signed": bool(args.key)}, indent=2))


def cmd_fetch(args):
    idx = load_index()
    ent = idx["packages"].get(args.sha256)
    if not ent:
        sys.exit("unknown sha256")
    src = ROOT / ent["file"]
    if sha256(src) != args.sha256:
        sys.exit("registry corruption: hash mismatch")
    if args.trust:
        if not ent.get("signature") or ent.get("public_key") not in trusted_keys():
            sys.exit("package is not signed by a trusted key")
        if not verify_digest(args.sha256, ent["signature"], ent["public_key"]):
            sys.exit("package signature is invalid")
    out = Path(args.out)
    shutil.copy2(src, out)
    print(json.dumps({"fetched": ent["name"], "sha256": args.sha256, "out": str(out)}, indent=2))


def cmd_verify(args):
    idx = load_index()
    ent = idx["packages"].get(args.sha256)
    if not ent:
        sys.exit("unknown sha256")
    got = sha256(ROOT / ent["file"])
    ok = got == args.sha256
    signed = False
    if args.trust and ok and ent.get("signature"):
        ok = ent.get("public_key") in trusted_keys() and verify_digest(args.sha256, ent["signature"], ent["public_key"])
        signed = ok
    print(json.dumps({"sha256": args.sha256, "ok": ok, "name": ent["name"],
                      "signed": signed, "trust_required": args.trust}, indent=2))
    sys.exit(0 if ok else 1)


def cmd_trust(args):
    _need_crypto()
    pub = serialization.load_pem_public_key(Path(args.public).read_bytes())
    raw = base64.b64encode(pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
    keys = trusted_keys()
    if raw not in keys:
        keys.append(raw)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "trusted.json").write_text(json.dumps({"keys": keys}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"trusted": raw}, indent=2))


def cmd_resolve(args):
    idx = load_index()
    hits = [(h, e) for h, e in idx["packages"].items() if e.get("name") == args.name]
    if not hits:
        sys.exit("unknown package name")
    hits.sort(key=lambda he: (he[1].get("version", ""), he[1].get("published_at", "")))
    h, e = hits[-1]
    print(json.dumps({"name": e["name"], "version": e.get("version"), "sha256": h,
                      "signed": bool(e.get("signature")), "defs": e.get("defs")}, indent=2))


def cmd_index(args):
    print(json.dumps(load_index(), indent=2, sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish"); p.add_argument("package"); p.add_argument("--key"); p.add_argument("--version", default="0.1.0"); p.set_defaults(fn=cmd_publish)
    f = sub.add_parser("fetch"); f.add_argument("sha256"); f.add_argument("--out", required=True); f.add_argument("--trust", action="store_true"); f.set_defaults(fn=cmd_fetch)
    v = sub.add_parser("verify"); v.add_argument("sha256"); v.add_argument("--trust", action="store_true"); v.set_defaults(fn=cmd_verify)
    t = sub.add_parser("trust"); t.add_argument("--public", required=True); t.set_defaults(fn=cmd_trust)
    r = sub.add_parser("resolve"); r.add_argument("--name", required=True); r.set_defaults(fn=cmd_resolve)
    i = sub.add_parser("index"); i.set_defaults(fn=cmd_index)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
