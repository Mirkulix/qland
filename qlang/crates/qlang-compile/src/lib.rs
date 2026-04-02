//! QLANG Compiler — Graph to machine code via LLVM
//!
//! Compilation pipeline:
//!   QLANG Graph → Optimize → Schedule → LLVM IR → Machine Code
//!
//! Uses inkwell (Rust LLVM bindings) for native code generation.
//! The compiled code runs at the same speed as hand-written C/Rust —
//! because it IS the same LLVM backend that C and Rust use.

pub mod codegen;
pub mod optimize;
pub mod visualize;
