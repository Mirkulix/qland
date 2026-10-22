//! Dense Binary DAG (Directed Acyclic Graph) Representation.
//!
//! Unlike human-centric languages (Python/Rust/C) which parse linear ASCII text into an AST,
//! the Gen-3 Substrate receives code as a flat vector of topological nodes directly from AI latent space.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    I64,
    F64,
    Bool,
    Tensor1D,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OpCode {
    ConstI64(i64),
    ConstF64(f64),
    ConstBool(bool),
    Param(u32), // Input argument slot
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Neg,
    CmpEq,
    CmpLt,
    CmpLe,
    CmpGt,
    CmpGe,
    Select,     // (cond, if_true, if_false) - Branch-free dataflow selection
    Clamp,      // (val, low, high)
    Abs,
    Min,
    Max,
    TensorMapAdd(f64),
    TensorSum,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Constraint {
    NonNegative,            // r >= 0
    StrictlyPositive,       // r > 0
    BoundedRange(i64, i64), // low <= r <= high
    NonZero,                // r != 0
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: u32,
    pub op: OpCode,
    pub inputs: Vec<u32>,
    pub ty: DataType,
    pub constraint: Option<Constraint>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    pub name: String,
    pub input_types: Vec<DataType>,
    pub return_type: DataType,
    pub nodes: Vec<Node>,
    pub return_node: u32,
}

impl Graph {
    pub fn new(name: impl Into<String>, input_types: Vec<DataType>, return_type: DataType) -> Self {
        Self {
            name: name.into(),
            input_types,
            return_type,
            nodes: Vec::new(),
            return_node: 0,
        }
    }

    /// Appends a node into the graph in topological order.
    pub fn add_node(&mut self, op: OpCode, inputs: Vec<u32>, ty: DataType, constraint: Option<Constraint>) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(Node {
            id,
            op,
            inputs,
            ty,
            constraint,
        });
        self.return_node = id;
        id
    }

    /// Expands a bounded loop of N static iterations into the DAG.
    /// Eliminates branching and guarantees termination while maintaining full interval verification.
    pub fn unroll_loop<F>(&mut self, initial_node: u32, iterations: u32, mut step_fn: F) -> u32
    where
        F: FnMut(&mut Graph, u32, u32) -> u32,
    {
        let mut current = initial_node;
        for step in 0..iterations {
            current = step_fn(self, current, step);
        }
        self.return_node = current;
        current
    }

    /// Serializes the graph to raw dense binary (no JSON, no text, minimal byte overhead).
    pub fn to_dense_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // Magic header 'TGS1'
        buf.extend_from_slice(b"TGS1");
        buf.extend_from_slice(&(self.nodes.len() as u32).to_le_bytes());
        for node in &self.nodes {
            buf.extend_from_slice(&node.id.to_le_bytes());
            buf.push(match node.ty {
                DataType::I64 => 1,
                DataType::F64 => 2,
                DataType::Bool => 3,
                DataType::Tensor1D => 4,
            });
            buf.push(node.inputs.len() as u8);
            for &inp in &node.inputs {
                buf.extend_from_slice(&inp.to_le_bytes());
            }
            // Opcode encoding
            match &node.op {
                OpCode::ConstI64(v) => { buf.push(1); buf.extend_from_slice(&v.to_le_bytes()); }
                OpCode::ConstF64(v) => { buf.push(2); buf.extend_from_slice(&v.to_le_bytes()); }
                OpCode::ConstBool(v) => { buf.push(3); buf.push(if *v { 1 } else { 0 }); }
                OpCode::Param(idx) => { buf.push(4); buf.extend_from_slice(&idx.to_le_bytes()); }
                OpCode::Add => buf.push(10),
                OpCode::Sub => buf.push(11),
                OpCode::Mul => buf.push(12),
                OpCode::Div => buf.push(13),
                OpCode::Mod => buf.push(14),
                OpCode::Neg => buf.push(15),
                OpCode::CmpEq => buf.push(20),
                OpCode::CmpLt => buf.push(21),
                OpCode::CmpLe => buf.push(22),
                OpCode::CmpGt => buf.push(23),
                OpCode::CmpGe => buf.push(24),
                OpCode::Select => buf.push(30),
                OpCode::Clamp => buf.push(31),
                OpCode::Abs => buf.push(32),
                OpCode::Min => buf.push(33),
                OpCode::Max => buf.push(34),
                OpCode::TensorMapAdd(val) => { buf.push(40); buf.extend_from_slice(&val.to_le_bytes()); }
                OpCode::TensorSum => buf.push(41),
            }
        }
        buf
    }

    /// Estimated AI Token count: In dense vector representation,
    /// each node is a single embedding/vector token rather than 10-25 text tokens.
    pub fn dense_token_count(&self) -> usize {
        self.nodes.len()
    }
}
