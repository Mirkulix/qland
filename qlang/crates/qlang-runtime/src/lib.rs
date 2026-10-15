//! QLANG Runtime — Graph executor
//!
//! Executes QLANG graphs by:
//! 1. Topologically sorting nodes
//! 2. Executing each node in order
//! 3. Flowing tensor data along edges
//!
//! This is the interpreter backend (Phase 1).
//! Phase 2 will add LLVM JIT compilation.

// Numeric ML code with parallel index arrays; `clamp` panics when min > max, so rewriting could change behavior.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_clamp
)]

pub mod accel;
pub mod autograd;
pub mod bench;
pub mod bytecode;
pub mod checkpoint;
pub mod concurrency;
pub mod config;
pub mod conv;
pub mod debugger;
pub mod diagnostics;
pub mod diffusion;
pub mod distributed_train;
pub mod executor;
pub mod fisher;
pub mod gpu_compute;
pub mod gpu_runtime;
pub mod graph_ops;
pub mod graph_train;
pub mod hebbian;
pub mod hub;
pub mod igqk;
pub mod linalg;
pub mod mnist;
pub mod ollama;
pub mod optimizers;
pub mod orchestrator;
pub mod parallel;
pub mod profiler;
pub mod providers;
pub mod quantum_flow;
pub mod registry;
pub mod scheduler;
pub mod stdlib;
pub mod theorems;
pub mod tokenizer;
pub mod training;
pub mod transformer;
pub mod transformer_train;
pub mod types;
pub mod unified;
pub mod vm;
pub mod web_server;
