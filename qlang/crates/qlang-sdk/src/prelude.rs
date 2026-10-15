//! Convenience re-exports for common QLANG SDK usage.
//!
//! ```rust
//! use qlang_sdk::prelude::*;
//! ```

// SDK top-level functions
pub use crate::{compress_ternary, sign, verify};

// Core types
pub use crate::{Dim, Dtype, Graph, Op, Shape, TensorData, TensorType};

// Crypto
pub use crate::{hash_graph, sha256, Keypair, SignedGraph};

// Agent types
pub use crate::GraphEmitter;
pub use crate::{AgentId, Capability, GraphMessage, MessageIntent};
