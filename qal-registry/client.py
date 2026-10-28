#!/usr/bin/env python3
"""HTTP client: resolve, verify and install a signed QAL package graph."""
import argparse, base64, hashlib, json, os, subprocess, sys, tempfile, urllib.request
from pathlib import Path

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
except Exception:
    Ed25519PublicKey = None


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def trusted(path):
    if not path:
        return set()
    d = json.loads(Path(path).read_text())
    return set(d.get("keys", []))


def check(entry, blob, digest, trust):
    if sha(blob) != digest:
        raise SystemExit(f"hash mismatch for {entry.get('name')}")
    if not trust:
        return
    if Ed25519PublicKey is None:
        raise SystemExit("cryptography is required for --trusted")
    if not entry.get("signature") or entry.get("public_key") not in trust:
        raise SystemExit(f"untrusted package {entry.get('name')}")
    try:
        Ed25519PublicKey.from_public_bytes(base64.b64decode(entry["public_key"])).verify(
            base64.b64decode(entry["signature"]), digest.encode())
    except Exception:
        raise SystemExit(f"bad signature for {entry.get('name')}")


def order(sha_key, index, trust, seen=None, out=None):
    if seen is None:
        seen = set()
    if out is None:
        out = []
    if sha_key in seen:
        raise SystemExit("dependency cycle")
    seen.add(sha_key)
    entry = index["packages"].get(sha_key)
    if not entry:
        raise SystemExit(f"missing dependency {sha_key}")
    for dep in entry.get("dependencies", []):
        order(dep, index, trust, seen, out)
    out.append(sha_key)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--trusted", help="JSON file with a keys array")
    ap.add_argument("--insecure", action="store_true", help="skip signature verification")
    ap.add_argument("--qal", default=os.environ.get("QAL_BIN", "qal"))
    args = ap.parse_args()
    base = args.url.rstrip("/")
    index = json.loads(get(base + "/index.json"))
    hits = [(h, e) for h, e in index["packages"].items() if e.get("name") == args.name]
    if not hits:
        raise SystemExit(f"unknown package {args.name}")
    hits.sort(key=lambda he: (he[1].get("version", ""), he[1].get("published_at", "")))
    root_sha = hits[-1][0]
    keys = trusted(args.trusted)
    if not keys and not args.insecure:
        raise SystemExit("no trusted keys: pass --trusted keys.json or --insecure")
    graph = order(root_sha, index, keys)
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"root does not exist: {root}")
    installed = []
    with tempfile.TemporaryDirectory(prefix="qal-http-install-") as tmp:
        for digest in graph:
            entry = index["packages"][digest]
            blob = get(f"{base}/packages/{digest}.qalpkg")
            check(entry, blob, digest, keys)
            path = Path(tmp) / f"{digest}.qalpkg"
            path.write_bytes(blob)
            r = subprocess.run([args.qal, "import", str(path)], cwd=root, capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit(f"qal import failed for {entry['name']}: {(r.stdout or r.stderr)[-400:]}")
            installed.append({"name": entry["name"], "sha256": digest})
    print(json.dumps({"installed": installed, "root": str(root)}, indent=2))


if __name__ == "__main__":
    main()
