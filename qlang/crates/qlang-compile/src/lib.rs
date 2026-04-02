//! QLANG Compiler — Graph to machine code
//!
//! Phase 1: Scaffold and IR representation
//! Phase 2: LLVM IR code generation
//! Phase 3: Direct machine code emission
//!
//! The compilation pipeline:
//!   Graph → Optimize → Schedule → LLVM IR → Machine Code
//!
//! For now, this crate provides graph optimization passes.
//! LLVM code generation will be added in Phase 2 when we integrate
//! the `inkwell` crate (Rust LLVM bindings).

pub mod optimize;
