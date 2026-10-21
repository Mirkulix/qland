#!/usr/bin/env python3
"""
Production API Gateway Sidecar for QAL Escrow Engine.
Provides:
  - Authentication: API-Key verification (X-API-Key or Bearer token)
  - Rate Limiting: Token-bucket per IP
  - Observability: /healthz health check & /metrics Prometheus metrics
  - Reverse Proxy: Forwards valid requests to upstream `qal serve`
"""

import http.server
import json
import os
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, Tuple

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8290")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8080"))
GATEWAY_HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
API_KEY = os.getenv("ESCROW_API_KEY", "escrow-prod-secret-key-2026")
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "true").lower() in ("true", "1", "yes")

OPS_DIR = os.path.dirname(os.path.abspath(__file__))
PILOT_DIR = os.path.abspath(os.path.join(OPS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(PILOT_DIR, "..", ".."))

# Rate limiter settings: 50 requests/sec with burst of 100
RATE_LIMIT_RPS = float(os.getenv("RATE_LIMIT_RPS", "50.0"))
RATE_LIMIT_BURST = float(os.getenv("RATE_LIMIT_BURST", "100.0"))

START_TIME = time.time()


class TokenBucket:
    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = threading.Lock()

    def consume(self, count: float = 1.0) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            if self.tokens >= count:
                self.tokens -= count
                return True
            return False


class RateLimiter:
    def __init__(self, rps: float, burst: float):
        self.rps = rps
        self.burst = burst
        self.buckets: Dict[str, TokenBucket] = {}
        self.lock = threading.Lock()

    def check(self, client_ip: str) -> bool:
        with self.lock:
            if client_ip not in self.buckets:
                self.buckets[client_ip] = TokenBucket(self.burst, self.rps)
            bucket = self.buckets[client_ip]
        return bucket.consume()


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests_total = 0
        self.status_2xx = 0
        self.status_4xx = 0
        self.status_5xx = 0
        self.auth_failures = 0
        self.rate_limited = 0
        self.latency_sum = 0.0

    def record(self, status: int, latency: float):
        with self.lock:
            self.requests_total += 1
            self.latency_sum += latency
            if 200 <= status < 300:
                self.status_2xx += 1
            elif 400 <= status < 500:
                self.status_4xx += 1
            elif status >= 500:
                self.status_5xx += 1

    def record_auth_failure(self):
        with self.lock:
            self.auth_failures += 1
            self.status_4xx += 1
            self.requests_total += 1

    def record_rate_limit(self):
        with self.lock:
            self.rate_limited += 1
            self.status_4xx += 1
            self.requests_total += 1

    def export_prometheus(self, backend_balanced: int) -> str:
        with self.lock:
            avg_lat = (self.latency_sum / self.requests_total) if self.requests_total > 0 else 0.0
            return (
                "# HELP escrow_requests_total Total HTTP requests handled by gateway\n"
                "# TYPE escrow_requests_total counter\n"
                f"escrow_requests_total {self.requests_total}\n"
                "# HELP escrow_status_2xx_total Total successful 2xx responses\n"
                "# TYPE escrow_status_2xx_total counter\n"
                f"escrow_status_2xx_total {self.status_2xx}\n"
                "# HELP escrow_status_4xx_total Total client error 4xx responses\n"
                "# TYPE escrow_status_4xx_total counter\n"
                f"escrow_status_4xx_total {self.status_4xx}\n"
                "# HELP escrow_status_5xx_total Total server error 5xx responses\n"
                "# TYPE escrow_status_5xx_total counter\n"
                f"escrow_status_5xx_total {self.status_5xx}\n"
                "# HELP escrow_auth_failures_total Total authentication rejections\n"
                "# TYPE escrow_auth_failures_total counter\n"
                f"escrow_auth_failures_total {self.auth_failures}\n"
                "# HELP escrow_rate_limited_total Total requests rejected by rate limiter\n"
                "# TYPE escrow_rate_limited_total counter\n"
                f"escrow_rate_limited_total {self.rate_limited}\n"
                "# HELP escrow_latency_seconds_sum Total request latency sum in seconds\n"
                "# TYPE escrow_latency_seconds_sum counter\n"
                f"escrow_latency_seconds_sum {self.latency_sum:.6f}\n"
                "# HELP escrow_zero_sum_balanced Gauge indicating if treasury is zero-sum balanced (1=yes, 0=no)\n"
                "# TYPE escrow_zero_sum_balanced gauge\n"
                f"escrow_zero_sum_balanced {backend_balanced}\n"
            )


rate_limiter = RateLimiter(RATE_LIMIT_RPS, RATE_LIMIT_BURST)
metrics = Metrics()


class EscrowGatewayHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        # Silent standard logging to keep output clean, errors logged to stderr
        pass

    def check_auth(self) -> bool:
        if not REQUIRE_AUTH:
            return True
        # Check X-API-Key header
        key = self.headers.get("X-API-Key")
        if not key:
            auth_header = self.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                key = auth_header[7:].strip()
        return key == API_KEY

    def query_backend_health(self) -> Tuple[bool, dict]:
        try:
            url = f"{BACKEND_URL.rstrip('/')}/reconciliation"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                balanced = bool(data.get("balanced", False))
                return True, data
        except Exception as e:
            return False, {"error": str(e)}

    def send_json(self, status: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")
        self.end_headers()

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = self.path.split("?")[0]

        # 0. Static Web Cockpit & Interactive Showcases (Public)
        static_files = {
            "/": "dashboard.html",
            "/dashboard": "dashboard.html",
            "/dashboard.html": "dashboard.html",
            "/preview": "preview.html",
            "/preview.html": "preview.html",
            "/proof": "proof.html",
            "/proof.html": "proof.html",
        }
        if path in static_files:
            file_name = static_files[path]
            file_path = os.path.join(REPO_ROOT, file_name)
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(content)
                return

        # 1. Health endpoint
        if path == "/healthz":
            alive, details = self.query_backend_health()
            uptime = round(time.time() - START_TIME, 2)
            if alive:
                self.send_json(200, {
                    "status": "healthy",
                    "service": "qal-escrow-gateway",
                    "backend": BACKEND_URL,
                    "uptime_seconds": uptime,
                    "reconciliation": details
                })
            else:
                self.send_json(503, {
                    "status": "unhealthy",
                    "backend": BACKEND_URL,
                    "uptime_seconds": uptime,
                    "error": details.get("error", "backend unreachable")
                })
            return

        # 2. Metrics endpoint
        if path == "/metrics":
            alive, details = self.query_backend_health()
            balanced = 1 if (alive and details.get("balanced")) else 0
            body = metrics.export_prometheus(balanced).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # 3. Proxied endpoints
        self.proxy_request("GET")

    def do_POST(self):
        self.proxy_request("POST")

    def do_PUT(self):
        self.proxy_request("PUT")

    def do_DELETE(self):
        self.proxy_request("DELETE")

    def proxy_request(self, method: str):
        client_ip = self.client_address[0] if self.client_address else "127.0.0.1"

        # Rate Limiting
        if not rate_limiter.check(client_ip):
            metrics.record_rate_limit()
            self.send_json(429, {
                "error": "rate_limited",
                "message": "Too many requests. Limit exceeded (Token Bucket)."
            })
            return

        # Authentication
        if not self.check_auth():
            metrics.record_auth_failure()
            self.send_json(401, {
                "error": "unauthorized",
                "message": "Authentication required. Provide valid API key in 'X-API-Key' or 'Authorization: Bearer <token>' header."
            })
            return

        # Read Request Body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # Build Backend Request
        target_url = f"{BACKEND_URL.rstrip('/')}{self.path}"
        headers = {}
        for h, v in self.headers.items():
            if h.lower() not in ("host", "authorization", "x-api-key"):
                headers[h] = v
        headers["X-Forwarded-For"] = client_ip

        t_start = time.time()
        try:
            req = urllib.request.Request(target_url, data=body, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                resp_body = resp.read()
                resp_status = resp.status
                resp_headers = resp.headers
        except urllib.error.HTTPError as e:
            resp_body = e.read()
            resp_status = e.code
            resp_headers = e.headers
        except Exception as e:
            metrics.record(502, time.time() - t_start)
            self.send_json(502, {
                "error": "bad_gateway",
                "message": f"Failed to reach QAL backend at {BACKEND_URL}: {str(e)}"
            })
            return

        latency = time.time() - t_start
        metrics.record(resp_status, latency)

        self.send_response(resp_status)
        for h, v in resp_headers.items():
            if h.lower() not in ("content-length", "transfer-encoding", "connection"):
                self.send_header(h, v)
        self.send_header("Content-Length", str(len(resp_body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(resp_body)


def run_gateway():
    server = http.server.ThreadingHTTPServer((GATEWAY_HOST, GATEWAY_PORT), EscrowGatewayHandler)
    print(f"[Gateway] QAL Escrow API Gateway listening on {GATEWAY_HOST}:{GATEWAY_PORT} (upstream: {BACKEND_URL})")
    print(f"[Gateway] Auth required: {REQUIRE_AUTH} (API Key: {API_KEY[:4]}***{API_KEY[-4:] if len(API_KEY) > 8 else ''})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Gateway] Shutting down gateway server.")
        server.server_close()


if __name__ == "__main__":
    run_gateway()
