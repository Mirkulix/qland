# ==============================================================================
# Production Multi-Stage Dockerfile for QAL & Generation-3 TGS Ecosystem
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder
# ------------------------------------------------------------------------------
FROM rust:1.80-slim-bookworm AS builder

# Install build dependencies: C/C++ compiler, cmake, clang, sqlite, python
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

# Copy QAL Rust source workspace
COPY qal /workspace/qal

# Build QAL release binary (with statically linked Z3 theorem prover)
RUN cd /workspace/qal && cargo build --release -p qal-cli

# Copy Gen-3 TGS Substrate Engine
COPY gen3 /workspace/gen3

# Build TGS release binary
RUN cd /workspace/gen3 && cargo build --release

# ------------------------------------------------------------------------------
# Stage 2: Minimal Production Runtime
# ------------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libsqlite3-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged application user
RUN groupadd -g 10001 qal && \
    useradd -u 10001 -g qal -s /bin/bash -m qaluser

WORKDIR /home/qaluser/app

# Copy compiled binaries from builder
COPY --from=builder /workspace/qal/target/release/qal /usr/local/bin/qal
COPY --from=builder /workspace/gen3/target/release/tgs-engine /usr/local/bin/tgs

# Prepare persistent data directory
RUN mkdir -p /home/qaluser/app/.qal && \
    chown -R qaluser:qal /home/qaluser/app

USER qaluser

# Expose HTTP service port
EXPOSE 8080

# Environment defaults
ENV QAL_ROOT="/home/qaluser/app"
ENV RUST_BACKTRACE="1"

# Default container entrypoint
ENTRYPOINT ["/usr/local/bin/qal"]
CMD ["serve", "--listen", "0.0.0.0:8080"]
