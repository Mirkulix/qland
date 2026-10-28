#!/usr/bin/env python3
"""Read-only HTTP front end for a QAL registry store.

  python3 qal-registry/server.py --registry /path/to/store --port 8899

Endpoints:
  GET /health
  GET /index.json
  GET /packages/<sha256>
The package bytes are hashed before serving; a mismatch is a 500.
"""
import argparse, hashlib, json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(".")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self.send(200, b'{"ok":true}\n')
        if self.path == "/index.json":
            p = ROOT / "index.json"
            if not p.exists():
                return self.send(404, b'{"error":"no index"}\n')
            return self.send(200, p.read_bytes())
        if self.path.startswith("/packages/"):
            name = self.path.rsplit("/", 1)[-1]
            if not name.endswith(".qalpkg") or "/" in name:
                return self.send(400, b'{"error":"bad package name"}\n')
            sha = name[:-len(".qalpkg")]
            p = ROOT / "packages" / name
            if not p.exists():
                return self.send(404, b'{"error":"not found"}\n')
            if sha256(p) != sha:
                return self.send(500, b'{"error":"hash mismatch"}\n')
            return self.send(200, p.read_bytes(), "application/octet-stream")
        return self.send(404, b'{"error":"not found"}\n')

    def log_message(self, fmt, *args):
        pass


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(Path(__file__).resolve().parent / "store"))
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    args = ap.parse_args()
    ROOT = Path(args.registry)
    if not (ROOT / "index.json").exists():
        raise SystemExit(f"no index.json in {ROOT}")
    srv = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(json.dumps({"listening": f"http://{args.bind}:{srv.server_port}", "registry": str(ROOT)}), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
