# Changelog

All notable changes to QLANG will be documented in this file.

## [0.1.0] - 2026-04-02

### Added
- **Core Language**: Graph-based program representation (DAG), 36+ operations
- **Type System**: Tensor types (f32, f64, i32, ternary, etc.) with shape inference
- **Parser**: .qlang text syntax with proof annotations (@proof theorem_5_2)
- **Compiler**: LLVM JIT compilation (29x faster than interpreter)
- **SIMD**: AVX2 vectorization with aligned memory allocator
- **AOT**: Ahead-of-time compilation to native object files (.o)
- **WebAssembly**: WAT code generation for browser deployment
- **GPU**: WGSL compute shader generation (WebGPU)
- **Runtime**: Graph executor with 20+ tensor operations
- **Autograd**: Reverse-mode automatic differentiation (backpropagation)
- **Training**: MLP training with SGD, Adam optimizer, gradient clipping
- **Transformer**: Multi-head attention, LayerNorm, GELU, positional encoding
- **Conv2D**: 2D convolution, max pooling, causal attention masking
- **IGQK**: Ternary compression (4-16x), quantum state operations
- **ONNX**: Import/export for PyTorch/TensorFlow interoperability
- **Agent Protocol**: QLMS binary protocol for AI-to-AI communication
- **Network Server**: TCP-based graph exchange and remote execution
- **Package System**: Registry with standard library
- **Graph Diff**: Version control for computation graphs
- **Distributed Training**: Data-parallel training with gradient aggregation
- **CLI**: Interactive REPL, parser, optimizer, visualizer, profiler
- **LSP**: Language server foundation for IDE integration
- **CI/CD**: GitHub Actions pipeline
- **Optimizer**: 6 passes (DCE, constant folding, CSE, op fusion, identity elimination)
- **Diagnostics**: Deep graph validation with error recovery
- **Scheduler**: Wavefront parallelism detection and memory planning
- **Benchmarks**: Comprehensive performance measurement suite
- **Checkpoints**: Model save/load in .qlm binary format
- **Visualization**: Graphviz DOT and ASCII terminal output
- **Self-hosting**: Compiler expressed as QLANG graph (foundation)

### Performance
- JIT compilation: 29x faster than interpreter (1M elements)
- Training: 100% accuracy on toy dataset in 70ms
- MNIST: 784->128->10 MLP converges in 15.9s
- IGQK compression: 4x-16x with accuracy retention
- Binary graph format: 3.2 KB for complete MNIST model
