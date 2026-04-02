# QLANG Quick Start Guide

## Installation

### Linux (Ubuntu/Debian)
```bash
git clone https://github.com/Mirkulix/qland.git
cd qland/qlang
./setup.sh
```

### macOS
```bash
brew install llvm@18 rust
git clone https://github.com/Mirkulix/qland.git
cd qland/qlang
cargo build --release
```

### Windows
1. Install Rust: https://rustup.rs
2. Install Visual Studio C++ Build Tools
3. Install LLVM 18: https://github.com/llvm/llvm-project/releases
4. ```powershell
   git clone https://github.com/Mirkulix/qland.git
   cd qland\qlang
   cargo build --release
   ```

## Your First Model (2 minutes)

Create `hello.qlang`:
```qlang
graph add_relu {
  input x: f32[4]
  input y: f32[4]

  node sum = add(x, y)
  node result = relu(sum)

  output out = result
}
```

Parse and run:
```bash
cargo run --release --bin qlang-cli -p qlang-compile -- parse hello.qlang
```

## Train a Neural Network (5 minutes)

```bash
cargo run --release --example train_autograd
```

This trains a 64->32->4 MLP to 100% accuracy in 70ms.

## Interactive REPL

```bash
cargo run --release --bin qlang-cli -p qlang-compile -- repl
```

Type commands interactively:
```
qlang> input x: f32[4]
qlang> input y: f32[4]
qlang> node sum = add(x, y)
qlang> output result = sum
qlang> run
```

## Compress a Model with IGQK

```qlang
graph compress {
  input weights: f32[768, 768]
  node compressed = to_ternary(weights) @proof theorem_5_2
  output small = compressed
}
```

Result: 768x768 x 4 bytes = 2.4 MB -> 150 KB (16x compression).

## Compile to Native Code

```bash
# Generate object file
cargo run --release --bin qlang-cli -p qlang-compile -- compile model.qlg.json -o model.o

# Link with your C program
cc -o myapp main.c model.o -lm

# In your C code:
# extern void qlang_graph(float* a, float* b, float* out, uint64_t n);
```

## All Examples

```bash
cargo run --release --example hello_qlang        # Simple graph
cargo run --release --example neural_network     # MLP + compression
cargo run --release --example train_autograd     # Backpropagation
cargo run --release --example train_mnist        # MNIST (784->128->10)
cargo run --release --example transformer        # Transformer encoder
cargo run --release --example jit_compile        # LLVM JIT demo
cargo run --release --example benchmark          # Performance test
cargo run --release --example full_pipeline      # Everything
```

## Next Steps

- Read the [Language Specification](../spec/QLANG_SPEC.md)
- See the [Pitch Deck](PITCH.md) for business context
- Browse the [Examples](../examples/)
- Join the discussion on GitHub Issues
