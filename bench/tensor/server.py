#!/usr/bin/env python3
"""NumPy tensor sidecar: POST /matmul with body `n;a_csv;b_csv`."""
import argparse, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
            av = np.array([float(x) for x in a.split(",")], dtype=float)
            bv = np.array([float(x) for x in b.split(",")], dtype=float)
            out = np.matmul(av.reshape(k, k), bv.reshape(k, k)).reshape(-1)
            text = ",".join(repr(float(x)) for x in out)
            self.send_response(200)
        except Exception as e:
            text = json.dumps({"error": str(e)})
            self.send_response(400)
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
    print(json.dumps({"listening": f"http://{a.bind}:{s.server_port}"}), flush=True)
    s.serve_forever()


if __name__ == "__main__":
    main()
