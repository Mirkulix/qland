//! Dataflow Graph Virtual Machine (DVM).
//!
//! Executes the DAG topologically over a flat pre-allocated value buffer.
//! Zero heap allocations during execution, zero instruction-pointer branching jumps.

use crate::dag::{Graph, OpCode};

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    I64(i64),
    F64(f64),
    Bool(bool),
    Tensor(Vec<f64>),
    Unit,
}

impl Value {
    pub fn as_i64(&self) -> i64 {
        match self {
            Value::I64(v) => *v,
            _ => panic!("Expected I64"),
        }
    }

    pub fn as_f64(&self) -> f64 {
        match self {
            Value::F64(v) => *v,
            _ => panic!("Expected F64"),
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            Value::Bool(v) => *v,
            _ => panic!("Expected Bool"),
        }
    }
}

pub struct Engine;

impl Engine {
    /// Executes the graph given the input values.
    /// Returns the result of the designated return node.
    pub fn execute(graph: &Graph, args: &[Value]) -> Result<Value, String> {
        let mut buffer: Vec<Value> = Vec::with_capacity(graph.nodes.len());

        for node in &graph.nodes {
            let val = match &node.op {
                OpCode::ConstI64(v) => Value::I64(*v),
                OpCode::ConstF64(v) => Value::F64(*v),
                OpCode::ConstBool(v) => Value::Bool(*v),
                OpCode::Param(slot) => {
                    args.get(*slot as usize).cloned().ok_or_else(|| {
                        format!("Missing argument at slot {slot}")
                    })?
                }
                OpCode::Add => {
                    let a = buffer[node.inputs[0] as usize].as_i64();
                    let b = buffer[node.inputs[1] as usize].as_i64();
                    Value::I64(a.wrapping_add(b))
                }
                OpCode::Sub => {
                    let a = buffer[node.inputs[0] as usize].as_i64();
                    let b = buffer[node.inputs[1] as usize].as_i64();
                    Value::I64(a.wrapping_sub(b))
                }
                OpCode::Mul => {
                    let a = buffer[node.inputs[0] as usize].as_i64();
                    let b = buffer[node.inputs[1] as usize].as_i64();
                    Value::I64(a.wrapping_mul(b))
                }
                OpCode::Div => {
                    let a = buffer[node.inputs[0] as usize].as_i64();
                    let b = buffer[node.inputs[1] as usize].as_i64();
                    if b == 0 {
                        return Err(format!("Division by zero at node {}", node.id));
                    }
                    Value::I64(a / b)
                }
                OpCode::Mod => {
                    let a = buffer[node.inputs[0] as usize].as_i64();
                    let b = buffer[node.inputs[1] as usize].as_i64();
                    if b == 0 {
                        return Err(format!("Modulo by zero at node {}", node.id));
                    }
                    Value::I64(a % b)
                }
                OpCode::Neg => {
                    let a = buffer[node.inputs[0] as usize].as_i64();
                    Value::I64(-a)
                }
                OpCode::CmpEq => {
                    let a = &buffer[node.inputs[0] as usize];
                    let b = &buffer[node.inputs[1] as usize];
                    Value::Bool(a == b)
                }
                OpCode::CmpLt => {
                    let a = buffer[node.inputs[0] as usize].as_i64();
                    let b = buffer[node.inputs[1] as usize].as_i64();
                    Value::Bool(a < b)
                }
                OpCode::CmpLe => {
                    let a = buffer[node.inputs[0] as usize].as_i64();
                    let b = buffer[node.inputs[1] as usize].as_i64();
                    Value::Bool(a <= b)
                }
                OpCode::CmpGt => {
                    let a = buffer[node.inputs[0] as usize].as_i64();
                    let b = buffer[node.inputs[1] as usize].as_i64();
                    Value::Bool(a > b)
                }
                OpCode::CmpGe => {
                    let a = buffer[node.inputs[0] as usize].as_i64();
                    let b = buffer[node.inputs[1] as usize].as_i64();
                    Value::Bool(a >= b)
                }
                OpCode::Select => {
                    // Branch-free dataflow ternary multiplexer
                    let cond = buffer[node.inputs[0] as usize].as_bool();
                    let on_true = &buffer[node.inputs[1] as usize];
                    let on_false = &buffer[node.inputs[2] as usize];
                    if cond { on_true.clone() } else { on_false.clone() }
                }
                OpCode::Clamp => {
                    let val = buffer[node.inputs[0] as usize].as_i64();
                    let low = buffer[node.inputs[1] as usize].as_i64();
                    let high = buffer[node.inputs[2] as usize].as_i64();
                    Value::I64(val.max(low).min(high))
                }
                OpCode::Abs => {
                    let val = buffer[node.inputs[0] as usize].as_i64();
                    Value::I64(val.abs())
                }
                OpCode::Min => {
                    let a = buffer[node.inputs[0] as usize].as_i64();
                    let b = buffer[node.inputs[1] as usize].as_i64();
                    Value::I64(a.min(b))
                }
                OpCode::Max => {
                    let a = buffer[node.inputs[0] as usize].as_i64();
                    let b = buffer[node.inputs[1] as usize].as_i64();
                    Value::I64(a.max(b))
                }
                OpCode::TensorMapAdd(bias) => {
                    let tensor = match &buffer[node.inputs[0] as usize] {
                        Value::Tensor(vec) => vec,
                        _ => return Err("Expected Tensor value".into()),
                    };
                    let mapped: Vec<f64> = tensor.iter().map(|&x| x + bias).collect();
                    Value::Tensor(mapped)
                }
                OpCode::TensorSum => {
                    let tensor = match &buffer[node.inputs[0] as usize] {
                        Value::Tensor(vec) => vec,
                        _ => return Err("Expected Tensor value".into()),
                    };
                    let sum: f64 = tensor.iter().sum();
                    Value::F64(sum)
                }
            };
            buffer.push(val);
        }

        let ret = buffer
            .get(graph.return_node as usize)
            .cloned()
            .ok_or_else(|| "Invalid return node index".to_string())?;

        Ok(ret)
    }
}
