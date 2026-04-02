//! QLANG Core — Graph-based AI-to-AI programming language
//!
//! This crate defines the fundamental data structures:
//! - Graph, Node, Edge (the program representation)
//! - TensorType, Dtype, Shape (the type system)
//! - QuantumState / DensityMatrix (probabilistic values)
//! - Constraint, Proof (verification primitives)

pub mod graph;
pub mod tensor;
pub mod quantum;
pub mod ops;
pub mod verify;
pub mod serial;
