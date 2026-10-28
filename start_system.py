#!/usr/bin/env python3
"""
Unified Live System Launcher for QLand & QAL.
Boots:
  1. QAL Escrow Verified Backend (`qal serve` on port 8090)
  2. Production API Gateway & Live Agent Cockpit (Port 8080)
  3. Verifies Ollama connection on http://localhost:11434
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
QAL_BIN = os.path.join(REPO_ROOT, "bin", "qal")
RUNTIME_DIR = os.path.join(REPO_ROOT, "pilot", "escrow", "ops", "runtime")
GATEWAY_SCRIPT = os.path.join(REPO_ROOT, "pilot", "escrow", "ops", "gateway.py")

BACKEND_PORT = 8290
GATEWAY_PORT = 8080
# A fresh key per start unless the operator provides one; it is printed once below.
API_KEY = os.environ.get("ESCROW_API_KEY") or __import__("secrets").token_urlsafe(32)

processes = []


def signal_handler(sig, frame):
    print("\n[System] Shutting down QLand & QAL services...")
    for p in processes:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                p.kill()
    print("[System] All services stopped. Goodbye!")
    sys.exit(0)


def check_ollama():
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name") for m in data.get("models", [])]
            return True, models
    except Exception:
        return False, []


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("\n" + "="*75)
    print("🚀 STARTING QLAND & QAL LIVE SYSTEM")
    print("="*75)

    # 1. Check Ollama
    ollama_ok, models = check_ollama()
    if ollama_ok:
        print(f"[✓] Ollama Engine online on http://localhost:11434")
        print(f"    Available models: {', '.join(models)}")
    else:
        print("[!] Ollama not detected on http://localhost:11434 (Fallback mode active)")

    # 2. Ensure Runtime Store
    if not os.path.exists(os.path.join(RUNTIME_DIR, ".qal", "store.db")):
        os.makedirs(RUNTIME_DIR, exist_ok=True)
        src_qal = os.path.join(REPO_ROOT, "pilot", "escrow", "qal", ".qal")
        dst_qal = os.path.join(RUNTIME_DIR, ".qal")
        if os.path.exists(dst_qal):
            import shutil
            shutil.rmtree(dst_qal)
        import shutil
        shutil.copytree(src_qal, dst_qal)

    # 3. Start Backend: qal serve
    serve_cmd = [
        QAL_BIN, "serve", "handle",
        "--listen", f"127.0.0.1:{BACKEND_PORT}",
        "--allow", "db:escrow,time",
        "--init", "setup"
    ]
    print(f"[*] Starting QAL Backend on 127.0.0.1:{BACKEND_PORT}...")
    p_backend = subprocess.Popen(serve_cmd, cwd=RUNTIME_DIR)
    processes.append(p_backend)
    time.sleep(0.8)

    # 4. Start Gateway
    env = os.environ.copy()
    env["BACKEND_URL"] = f"http://127.0.0.1:{BACKEND_PORT}"
    env["GATEWAY_PORT"] = str(GATEWAY_PORT)
    env["GATEWAY_HOST"] = "0.0.0.0"
    env["ESCROW_API_KEY"] = API_KEY
    env["REQUIRE_AUTH"] = "true"

    print(f"[*] Starting API Gateway & Visual Cockpit on http://0.0.0.0:{GATEWAY_PORT}...")
    p_gateway = subprocess.Popen([sys.executable, GATEWAY_SCRIPT], env=env)
    processes.append(p_gateway)
    time.sleep(0.8)

    # 5. Success Banner
    print("\n" + "="*75)
    print("🎉 QLAND & QAL SYSTEM IST JETZT BEREIT UND EINSATZFÄHIG!")
    print("="*75)
    print(f"\n  👉 Visual Cockpit (Dashboard):   http://localhost:{GATEWAY_PORT}/")
    print(f"  👉 Interaktive Leistungsschau:   http://localhost:{GATEWAY_PORT}/preview")
    print(f"  👉 Health Check:                 http://localhost:{GATEWAY_PORT}/healthz")
    print(f"  👉 Prometheus Metriken:          http://localhost:{GATEWAY_PORT}/metrics")
    print(f"  👉 Escrow Engine API:            http://localhost:{GATEWAY_PORT}/accounts")
    print(f"  🔑 API-Key für Requests:         {API_KEY}")
    print("\nDrücke Strg+C zum Beenden aller Dienste.\n")

    # Keep running
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
