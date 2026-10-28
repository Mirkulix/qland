# ==============================================================================
# Production Dockerfile for Operational QAL Escrow Engine
# Bundles:
#   - QAL Engine with Z3 Theorem Prover
#   - Verified Escrow Project Store (22 Proven Contracts)
#   - Hardened Python API Gateway Sidecar (Auth, Rate-Limiting, Observability)
#   - Atomic Backup & Restore Tools
# ==============================================================================

# Stage 1: Build QAL compiler & runtime for Linux
FROM rust:1.80-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    clang \
    libclang-dev \
    libsqlite3-dev \
    pkg-config \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY qal /workspace/qal

# Build optimized release binary
RUN cd /workspace/qal && cargo build --release -p qal-cli

# Stage 2: Minimal hardened production image
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libsqlite3-0 \
    curl \
    python3 \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /workspace/qal/target/release/qal /usr/local/bin/qal

# Copy template verified escrow project
COPY pilot/escrow/qal /app/project

# Copy operational scripts
COPY pilot/escrow/ops /app/ops
RUN chmod +x /app/ops/*.sh /app/ops/*.py

# Set up runtime volume mount point
VOLUME ["/data"]

EXPOSE 8080

ENV DATA_DIR="/data"
ENV GATEWAY_PORT="8080"
ENV ESCROW_API_KEY="escrow-prod-secret-key-2026"
ENV REQUIRE_AUTH="true"

ENTRYPOINT ["/app/ops/entrypoint.sh"]
