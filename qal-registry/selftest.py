#!/usr/bin/env python3
"""End-to-end self-test for the signed, dependency-aware QAL registry."""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QAL = Path(os.environ.get("QAL_BIN", REPO / "qal" / "target" / "release" / "qal"))
REG = REPO / "qal-registry" / "registry.py"


def run(cmd, cwd, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    r = subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True, timeout=120)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def check(name, ok, detail=""):
    return {"case": name, "ok": bool(ok), "detail": detail[-300:]}


def main():
    if not QAL.exists():
        sys.exit(f"qal binary missing: {QAL}")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization

    base = Path(tempfile.mkdtemp(prefix="qal-reg-selftest-"))
    store = base / "store"
    pkgs = base / "pkgs"
    pkgs.mkdir()
    key = base / "key.pem"
    pub = base / "pub.pem"
    k = Ed25519PrivateKey.generate()
    key.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    pub.write_bytes(k.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    env = {"QAL_REGISTRY": str(store)}
    results = []

    code, out = run([str(QAL), "init"], pkgs)
    results.append(check("qal init", code == 0, out))
    for name, example in (("math", "math.qal"), ("age", "age.qal")):
        code, out = run([str(QAL), "define", str(REPO / "qal" / "examples" / example)], pkgs)
        results.append(check(f"define {name}", code == 0, out))
        code, out = run([str(QAL), "export", "", "--out", f"{name}.qalpkg", "--name", name], pkgs)
        results.append(check(f"export {name}", code == 0, out))

    code, out = run([sys.executable, str(REG), "publish", str(pkgs / "math.qalpkg"),
                     "--key", str(key), "--version", "1.0.0"], REPO, env)
    results.append(check("publish math signed", code == 0, out))
    math_sha = ""
    if code == 0:
        try:
            math_sha = json.loads(out[out.index("{"):])["sha256"]
        except Exception:
            pass
    code, out = run([sys.executable, str(REG), "publish", str(pkgs / "age.qalpkg"),
                     "--key", str(key), "--version", "1.0.0", "--depends", math_sha], REPO, env)
    results.append(check("publish age depends math", code == 0, out))
    code, out = run([sys.executable, str(REG), "trust", "--public", str(pub)], REPO, env)
    results.append(check("trust key", code == 0, out))
    code, out = run([sys.executable, str(REG), "resolve", "--name", "age", "--trust"], REPO, env)
    results.append(check("resolve age graph ok", code == 0 and '"ok": true' in out, out))

    # install the whole graph into a fresh project
    proj = base / "proj"
    proj.mkdir()
    code, out = run([str(QAL), "init"], proj)
    code, out = run([sys.executable, str(REG), "install", "--name", "age", "--root", str(proj), "--trust"], REPO, env)
    results.append(check("install age graph", code == 0 and '"name": "math"' in out and '"name": "age"' in out, out))
    code, out = run([str(QAL), "query", "--name", "fact"], proj)
    results.append(check("installed fact queryable", code == 0 and '"fact"' in out, out))

    # tamper with the dependency file
    shutil.rmtree(base / "store-broken", ignore_errors=True)
    shutil.copytree(store, base / "store-broken")
    (base / "store-broken" / "packages" / f"{math_sha}.qalpkg").write_bytes(b"tampered")
    code, out = run([sys.executable, str(REG), "resolve", "--name", "age", "--trust"], REPO,
                    {"QAL_REGISTRY": str(base / "store-broken")})
    results.append(check("tampered dependency rejected", code != 0 and '"check": "hash"' in out, out))

    # remove the dependency from the index
    shutil.rmtree(base / "store-missing", ignore_errors=True)
    shutil.copytree(store, base / "store-missing")
    idx = base / "store-missing" / "index.json"
    d = json.loads(idx.read_text())
    d["packages"] = {h: e for h, e in d["packages"].items() if e["name"] != "math"}
    idx.write_text(json.dumps(d))
    code, out = run([sys.executable, str(REG), "resolve", "--name", "age", "--trust"], REPO,
                    {"QAL_REGISTRY": str(base / "store-missing")})
    results.append(check("missing dependency rejected", code != 0 and '"check": "missing"' in out, out))

    summary = {"ok": all(r["ok"] for r in results), "results": results}
    print(json.dumps(summary, indent=2))
    shutil.rmtree(base, ignore_errors=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
