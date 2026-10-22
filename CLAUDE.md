# CLAUDE.md — Agent Instructions for QAL & qland

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview: `qland`

**QAL (Query & Agent Language)** — A deterministic, formally verifiable programming language and self-hosted runtime designed for autonomous AI agents.

* **Core Stack:** ~46,000 lines of Rust in `qal/crates/`.
* **Self-Hosting Compiler:** Written in QAL (`qal/selfhost/qalc.qal`, 17 modules).
* **Bootstrap Fixpoint:** Stage 0 (Rust) == Stage 1 (QAL WASM) == Stage 2 (QAL WASM) byte-identical bootstrap fixpoint verified.
* **Formal Verification:** Contracts (`req` preconditions, `ens` postconditions) proven via embedded **Z3 Theorem Prover** (`qal-prove`).
* **Sandbox:** Hermetic WebAssembly runtime via **Wasmtime** with strict fuel and memory limits.
* **Canonical Storage:** Content-addressed Merkle store (`.qal/store.db`) identified by SHA-256 hashes.
* **Sibling Workspace:** Production binaries, packaging, and the **Generation-3 AI-Native Substrate (ARM64 JIT, 2.35ns)** live in `../QAL/`.

---

## Workspace Layout

```
/Users/aleksandarbarisic/Workspace/qland/
├── qal/                        # Rust compiler source & self-hosting compiler
│   ├── Cargo.toml              # Workspace manifest
│   ├── crates/                 # 10 core crates (core, text, check, wasm, run, prove, store, serve, compiler, cli)
│   ├── selfhost/               # 17 self-hosted compiler modules (qalc.qal)
│   └── tests/                  # Parity & differential test suites
├── docs/                       # Architectural specifications & Phase 1-5 documentation
├── bench/                      # Benchmarks and evaluation tasks
└── IGQK_Complete_Package/      # Theoretical quantum compression research (mathematical foundations)
```

---

## Common Commands & Workflows

### 1. Build and Check
```bash
# Check compiler crates
cargo check --manifest-path qal/Cargo.toml

# Run compiler tests
cargo test --manifest-path qal/Cargo.toml
```

### 2. Verify Self-Hosting Compiler
```bash
# Verify the 17-module self-hosting compiler bundle
cargo run --manifest-path qal/Cargo.toml -p qal-cli -- check qal/selfhost/qalc.qal
```

### 3. Run Examples with Verification
```bash
# Test, prove contracts with Z3, and execute sandboxed
cargo run --manifest-path qal/Cargo.toml -p qal-cli -- test qal/examples/math.qal
cargo run --manifest-path qal/Cargo.toml -p qal-cli -- prove qal/examples/math.qal
cargo run --manifest-path qal/Cargo.toml -p qal-cli -- call qal/examples/math.qal fact 6
```

### 4. Generation-3 AI-Native Substrate (in sibling directory)
```bash
# Hardware benchmark (Apple Silicon ARM64 JIT: 2.35 ns)
/Users/aleksandarbarisic/Workspace/QAL/bin/tgs bench

# Autonomous AI synthesis loop
python3 /Users/aleksandarbarisic/Workspace/QAL/gen3/synth.py

# Interactive 13-tool launcher
python3 /Users/aleksandarbarisic/Workspace/QAL/start.py
```

---

## Critical Rules & Guidelines

1. **Never touch `qlang/web/index.html`:** This file is protected and must remain untouched.
2. **Avoid `git add -A`:** Always stage specific, intentional files.
3. **Preserve Documentation Integrity:** Keep all existing comments and mathematical proofs.
4. **Zero Compiler Warnings:** All Rust crates must compile cleanly with zero warnings and zero panics.
5. **Disk Cleanliness:** Run `cargo clean` inside build directories if temporary target files consume significant disk space.
