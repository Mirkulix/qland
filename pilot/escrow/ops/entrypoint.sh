#!/bin/bash
set -e

DATA_DIR=${DATA_DIR:-/data}
mkdir -p "$DATA_DIR/.qal/db" "$DATA_DIR/backups"

# If project store is not in volume, seed it from /app/project
if [ ! -f "$DATA_DIR/.qal/store.db" ]; then
    echo "[Entrypoint] Seeding persistent store from verified image template..."
    cp -r /app/project/.qal "$DATA_DIR/"
fi

# Run schema migration check
python3 /app/ops/backup_restore.py migrate --db "$DATA_DIR/.qal/db/escrow.sqlite"

# Start QAL serve in background
echo "[Entrypoint] Starting QAL serve (upstream 127.0.0.1:8090)..."
cd "$DATA_DIR"
qal serve handle --listen 127.0.0.1:8090 --allow db:escrow,time --init setup &
QAL_PID=$!

# Wait for backend to be ready
sleep 1

# Start API Gateway in foreground
echo "[Entrypoint] Starting Escrow API Gateway on 0.0.0.0:${GATEWAY_PORT:-8080}..."
export BACKEND_URL="http://127.0.0.1:8090"
export GATEWAY_HOST="0.0.0.0"

# Trap SIGTERM to shut down QAL cleanly
trap 'kill -TERM $QAL_PID; exit 0' SIGTERM SIGINT

python3 /app/ops/gateway.py &
GATEWAY_PID=$!

wait $GATEWAY_PID
