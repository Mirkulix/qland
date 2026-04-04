//! QLANG Core — Graph-based AI-to-AI programming language
//!
//! This crate defines the fundamental data structures:
//! - Graph, Node, Edge (the program representation)
//! - TensorType, Dtype, Shape (the type system)
//! - QuantumState / DensityMatrix (probabilistic values)
//! - Constraint, Proof (verification primitives)
//! - Crypto: SHA-256 hashing, HMAC-SHA256 signatures (protocol security)

pub mod crypto;
pub mod errors;
pub mod graph;
pub mod ops;
pub mod quantum;
pub mod serial;
pub mod stats;
pub mod tensor;
pub mod shape_inference;
pub mod type_check;
pub mod verify;
pub mod ffi;
