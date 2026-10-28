#!/usr/bin/env python3
"""Apple GPU tensor sidecar: same protocol as server.py, MLX/Metal matmul."""
import argparse, json, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mlx.core as mx
import numpy as np


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/matmul":
            self.send_response(404); self.end_headers(); return
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n).decode()
        try:
            size, a, b = body.split(";")
            k = int(size)
            av = mx.array(np.array([float(x) for x in a.split(",")], dtype=np.float32).reshape(k, k))
            bv = mx.array(np.array([float(x) for x in b.split(",")], dtype=np.float32).reshape(k, k))
            out = mx.matmul(av, bv)
            mx.eval(out)
            text = ",".join(repr(float(x)) for x in np.array(out).reshape(-1))
            self.send_response(200)
        except Exception as e:
            text = json.dumps({"error": str(e)}); self.send_response(400)
        self.send_header("content-type", "text/plain")
        self.send_header("content-length", str(len(text)))
        self.end_headers()
        self.wfile.write(text.encode())

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8901)
    a = ap.parse_args()
    s = ThreadingHTTPServer((a.bind, a.port), Handler)
    print(json.dumps({"listening": f"http://{a.bind}:{s.server_port}", "backend": "mlx-metal"}), flush=True)
    s.serve_forever()


if __name__ == "__main__":
    main()
