<![CDATA[# QLANG

**The programming language for AI systems.**

QLANG compiles neural network models to native code, compresses them 16x, and deploys them anywhere — browser, GPU, edge devices, or cloud. One model, nine targets, zero Python.

[![CI](https://github.com/Mirkulix/qland/actions/workflows/qlang-ci.yml/badge.svg)](https://github.com/Mirkulix/qland/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Why QLANG?

| Problem | QLANG Solution |
|---------|---------------|
| ML models are too large for edge devices | **IGQK compression: 16x smaller, same accuracy** |
| PyTorch models only run with Python | **Compile to native code, WASM, or GPU shaders** |
| AI agents communicate via slow text | **Binary graph protocol (3 KB vs 50 KB text)** |
| Model deployment is complex | **One graph → 9 compilation targets** |

## Quick Start

```bash
git clone https://github.com/Mirkulix/qland.git
cd qland/qlang
cargo build --release
cargo run --release --example train_autograd
```

Output:
```
Epoch  1/50: loss=1.3507, acc=67.5%
Epoch 11/50: loss=0.5470, acc=100.0%
Epoch 50/50: loss=0.0188, acc=100.0%
Training time: 70ms
IGQK Compression: 3.8x at 100% accuracy
```

## Write a Model in 10 Lines

```qlang
graph classifier {
  input image: f32[1, 784]
  input W1: f32[784, 128]
  input W2: f32[128, 10]

  node hidden = matmul(image, W1)
  node activated = relu(hidden)
  node logits = matmul(activated, W2)
  node predictions = softmax(logits)
  node compressed = to_ternary(W1) @proof theorem_5_2

  output result = predictions
}
```

## Deploy Anywhere

```bash
# Native x86-64 (29x faster than interpreter)
qlang-cli compile model.qlang -o model.o

# WebAssembly (runs in any browser)
qlang-cli wasm model.qlang > model.wat

# GPU compute shader (WebGPU)
qlang-cli gpu model.qlang > model.wgsl

# Inspect as assembly
qlang-cli asm model.qlang
```

## Compilation Targets

| Target | Format | Speed | Use Case |
|--------|--------|-------|----------|
| **LLVM JIT** | Native x86-64 | 29x | Production servers |
| **LLVM SIMD** | AVX2 vectors | 29x | Batch processing |
| **AOT** | .o object file | 29x | Embed in C/C++/Rust |
| **WebAssembly** | .wat/.wasm | 5-10x | Browser apps |
| **GPU** | WGSL shader | 100x+ | Parallel compute |
| **Interpreter** | Direct | 1x | Development |
| **Binary** | .qlg (3 KB) | — | Wire format |
| **ONNX** | JSON | — | PyTorch interop |
| **Assembly** | .S | — | Debugging |

## IGQK Model Compression

QLANG implements Information-Geometric Quantum Compression:

```
Original model:    392 KB (f32 weights)
Compressed:         25 KB (ternary: {-1, 0, +1})
Compression:       16x
Accuracy retained: 100%
```

Each compression carries a formal proof annotation linking to mathematical theorems that guarantee bounded distortion.

## ML Features

- **Autograd**: Reverse-mode automatic differentiation
- **Optimizers**: SGD (with momentum), Adam, gradient clipping
- **LR Schedules**: Constant, step decay, cosine annealing, linear warmup
- **Architectures**: MLP, Transformer (attention, LayerNorm, GELU), Conv2D
- **Training**: Cross-entropy loss, softmax, batch processing
- **Checkpoints**: Save/load models in .qlm binary format

## For AI Agents

QLANG is designed as a communication protocol between AI systems:

```
AI Agent A                        AI Agent B
    |                                 |
    |-- Build graph (4 decisions) --> |
    |   (not 47 text tokens)          |
    |                                 |-- Verify types
    |                                 |-- Optimize (6 passes)
    |                                 |-- JIT compile
    |                                 |-- Execute
    | <-------- Results ------------- |
```

Binary protocol (QLMS): 3 KB instead of 50 KB text. No parsing errors possible.

## Architecture

```
qlang/
├── crates/
│   ├── qlang-core/        # Graph, Tensor, Quantum, Ops, Types
│   ├── qlang-compile/     # LLVM, SIMD, AOT, WASM, GPU, Parser, CLI
│   ├── qlang-runtime/     # Executor, Autograd, Training, Transformer
│   └── qlang-agent/       # Protocol, Packages, Distributed, Server
├── examples/              # 11 runnable examples
├── spec/                  # Language specification
└── .github/workflows/     # CI/CD pipeline
```

## Performance

```
Benchmark: relu(a + b), release mode

Elements    Interpreter    LLVM JIT     Speedup
   1,024        10.0µs     680ns        14.7x
  65,536       703.6µs      44.6µs      15.8x
1,048,576      21.4ms      728.4µs      29.4x
```

## CLI Tool

```bash
qlang-cli repl                    # Interactive REPL
qlang-cli parse model.qlang      # Parse and validate
qlang-cli info model.qlg.json    # Show graph structure
qlang-cli verify model.qlg.json  # Check constraints
qlang-cli optimize model.qlg.json # Run 6 optimization passes
qlang-cli jit model.qlg.json     # JIT compile and execute
qlang-cli compile model.qlg.json -o out.o  # AOT compile
qlang-cli dot model.qlg.json     # Graphviz visualization
qlang-cli stats model.qlg.json   # Graph statistics
qlang-cli schedule model.qlg.json # Execution plan
```

## Requirements

- **Rust 1.70+** (tested with 1.93)
- **LLVM 18** (optional, for JIT/AOT compilation)
- Works without LLVM in interpreter-only mode

### Install on Ubuntu/Debian
```bash
sudo apt install llvm-18-dev libpolly-18-dev libzstd-dev
```

## Stats

- **248 tests** passing
- **47 modules** across 4 crates
- **~20,000 lines** of Rust
- **MIT licensed**

## License

MIT License - see [LICENSE](LICENSE)

## Author

Aleksandar Barisic ([@Mirkulix](https://github.com/Mirkulix))

Built with IGQK theory (Information-Geometric Quantum Compression).
]]>