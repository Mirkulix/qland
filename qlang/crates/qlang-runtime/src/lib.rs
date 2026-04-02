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
pub mod executor;
pub mod training;
