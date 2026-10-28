#!/usr/bin/env python3
"""Signed registry install with a dependency, end to end, over MCP."""
import base64, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

REPO = Path(__file__).resolve().parents[2]
QAL = Path(os.environ.get("QAL_BIN", REPO / "qal" / "target" / "release" / "qal"))
REG = REPO / "qal-registry" / "registry.py"


def run(cmd, cwd=None, env=None):
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{cmd}: {(r.stdout or r.stderr)[-800:]}")
    return r.stdout


def rpc(proc, obj):
    proc.stdin.write(json.dumps(obj) + "\n"); proc.stdin.flush()
    return json.loads(proc.stdout.readline())


def main():
    with tempfile.TemporaryDirectory(prefix="qal-install-verify-") as tmp:
        tmp = Path(tmp)
        key = Ed25519PrivateKey.generate()
        (tmp / "key.pem").write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        pub = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        (tmp / "pub.pem").write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
        (tmp / "trusted.json").write_text(json.dumps({"keys": [base64.b64encode(pub).decode()]}))
        reg = tmp / "registry"; reg.mkdir()
        env = dict(os.environ, QAL_REGISTRY=str(reg))

        a = tmp / "a"; a.mkdir(); run([str(QAL), "init"], cwd=a)
        (a / "a.qal").write_text("fn base() -> Int = 7\n")
        run([str(QAL), "define", "a.qal"], cwd=a)
        run([str(QAL), "export", "", "--out", str(tmp / "a.qalpkg"), "--name", "qal_base"], cwd=a)
        out_a = json.loads(run(["python3", str(REG), "publish", str(tmp / "a.qalpkg"), "--key", str(tmp / "key.pem"), "--version", "1.0.0"], env=env))
        b = tmp / "b"; b.mkdir(); run([str(QAL), "init"], cwd=b)
        run([str(QAL), "import", str(tmp / "a.qalpkg")], cwd=b)
        (b / "b.qal").write_text("fn use_base() -> Int = base()\n")
        run([str(QAL), "define", "b.qal"], cwd=b)
        run([str(QAL), "export", "", "--out", str(tmp / "b.qalpkg"), "--name", "qal_use"], cwd=b)
        run(["python3", str(REG), "publish", str(tmp / "b.qalpkg"), "--key", str(tmp / "key.pem"),
             "--version", "1.0.0", "--depends", out_a["sha256"]], env=env)
        run(["python3", str(REG), "trust", "--public", str(tmp / "pub.pem")], env=env)

        srv = subprocess.Popen(["python3", str(REPO / "qal-registry" / "server.py"), "--registry", str(reg), "--port", "8898"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proj = tmp / "proj"; proj.mkdir(); run([str(QAL), "init"], cwd=proj)
        shutil.copy2(tmp / "trusted.json", proj / "trusted.json")
        time.sleep(0.8)
        proc = subprocess.Popen([str(QAL), "mcp", "--root", str(proj)], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        try:
            rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "v", "version": "1"}}})
            proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"); proc.stdin.flush()
            ins = rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "qal_install", "arguments": {"url": "http://127.0.0.1:8898", "name": "qal_use", "trusted": "trusted.json"}}})
            assert not ins["result"]["isError"], ins
            body = json.loads(ins["result"]["content"][0]["text"])
            assert [x["name"] for x in body["installed"]] == ["qal_base", "qal_use"], body
            for name in ("base", "use_base"):
                q = rpc(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "qal_query", "arguments": {"name": name}}})
                assert not q["result"]["isError"], q
        finally:
            proc.stdin.close(); proc.wait(timeout=10)
            srv.terminate(); srv.wait(timeout=10)
    print("mcp: signed dependency install qal_base -> qal_use verified")


if __name__ == "__main__":
    main()
