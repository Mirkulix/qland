//! QLANG Runtime — Graph executor
//!
//! Executes QLANG graphs by:
//! 1. Topologically sorting nodes
//! 2. Executing each node in order
//! 3. Flowing tensor data along edges
//!
//! This is the interpreter backend (Phase 1).
//! Phase 2 will add LLVM JIT compilation.

pub mod autograd;
pub mod checkpoint;
pub mod optimizers;
pub mod diagnostics;
pub mod executor;
pub mod graph_train;
pub mod mnist;
pub mod profiler;
pub mod scheduler;
pub mod training;
pub mod transformer;
pub mod conv;
pub mod bench;
pub mod stdlib;
pub mod vm;
pub mod debugger;
pub mod linalg;
pub mod fisher;
pub mod quantum_flow;
pub mod theorems;
pub mod igqk;
pub mod config;
pub mod types;
pub mod unified;
pub mod concurrency;
pub mod graph_ops;
pub mod gpu_runtime;
pub mod registry;
pub mod parallel;
pub mod hub;
