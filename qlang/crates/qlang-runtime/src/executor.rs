use std::collections::HashMap;

use qlang_core::graph::{Graph, NodeId};
use qlang_core::ops::Op;
use qlang_core::tensor::TensorData;
use qlang_core::verify;

/// Result of executing a QLANG graph.
#[derive(Debug)]
pub struct ExecutionResult {
    /// Output tensors, keyed by output node name.
    pub outputs: HashMap<String, TensorData>,
    /// Execution statistics.
    pub stats: ExecutionStats,
}

#[derive(Debug, Default)]
pub struct ExecutionStats {
    pub nodes_executed: usize,
    pub quantum_ops: usize,
    pub total_flops: u64,
}

/// Errors during graph execution.
#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error("graph verification failed")]
    VerificationFailed(String),

    #[error("no input provided for node {0}: {1}")]
    MissingInput(NodeId, String),

    #[error("unsupported operation: {0}")]
    UnsupportedOp(String),

    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("runtime error: {0}")]
    RuntimeError(String),
}

/// Execute a QLANG graph with the given inputs.
///
/// This is the Phase 1 interpreter. It executes nodes one-by-one
/// in topological order, passing tensors along edges.
///
/// Phase 2 will JIT-compile the graph to native code via LLVM.
pub fn execute(
    graph: &Graph,
    inputs: HashMap<String, TensorData>,
) -> Result<ExecutionResult, ExecutionError> {
    // 1. Verify the graph
    let verification = verify::verify_graph(graph);
    if !verification.is_ok() {
        return Err(ExecutionError::VerificationFailed(format!(
            "{}",
            verification
        )));
    }

    // 2. Topological sort
    let order = graph
        .topological_sort()
        .map_err(|e| ExecutionError::RuntimeError(e.to_string()))?;

    // 3. Execute nodes in order
    let mut node_outputs: HashMap<(NodeId, u8), TensorData> = HashMap::new();
    let mut stats = ExecutionStats::default();

    for &node_id in &order {
        let node = graph
            .node(node_id)
            .ok_or(ExecutionError::RuntimeError(format!(
                "Node {node_id} not found"
            )))?;

        match &node.op {
            Op::Input { name } => {
                let data = inputs
                    .get(name)
                    .ok_or_else(|| {
                        ExecutionError::MissingInput(node_id, name.clone())
                    })?
                    .clone();
                node_outputs.insert((node_id, 0), data);
            }

            Op::Output { name: _ } => {
                // Output nodes just pass through their input
                let incoming = graph.incoming_edges(node_id);
                if let Some(edge) = incoming.first() {
                    if let Some(data) = node_outputs.get(&(edge.from_node, edge.from_port)) {
                        node_outputs.insert((node_id, 0), data.clone());
                    }
                }
            }

            Op::Constant => {
                // Constants have their data embedded (via metadata or separate storage)
                // For Phase 1, we skip — constants should be provided as inputs
            }

            Op::Add => {
                let (a, b) = get_two_inputs(graph, node_id, &node_outputs)?;
                let result = tensor_add(&a, &b)?;
                node_outputs.insert((node_id, 0), result);
                stats.total_flops += a.shape.numel().unwrap_or(0) as u64;
            }

            Op::Mul => {
                let (a, b) = get_two_inputs(graph, node_id, &node_outputs)?;
                let result = tensor_mul(&a, &b)?;
                node_outputs.insert((node_id, 0), result);
                stats.total_flops += a.shape.numel().unwrap_or(0) as u64;
            }

            Op::MatMul => {
                let (a, b) = get_two_inputs(graph, node_id, &node_outputs)?;
                let result = tensor_matmul(&a, &b)?;
                let flops = match (a.shape.numel(), b.shape.0.last()) {
                    (Some(m), Some(qlang_core::tensor::Dim::Fixed(n))) => (m as u64) * (*n as u64) * 2,
                    _ => 0,
                };
                node_outputs.insert((node_id, 0), result);
                stats.total_flops += flops;
            }

            Op::Relu => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_relu(&input)?;
                node_outputs.insert((node_id, 0), result);
            }

            Op::Neg => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_neg(&input)?;
                node_outputs.insert((node_id, 0), result);
            }

            Op::ToTernary => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_to_ternary(&input)?;
                node_outputs.insert((node_id, 0), result);
                stats.quantum_ops += 1;
            }

            other => {
                return Err(ExecutionError::UnsupportedOp(format!("{other}")));
            }
        }

        stats.nodes_executed += 1;
    }

    // 4. Collect outputs
    let mut outputs = HashMap::new();
    for node in graph.output_nodes() {
        if let Op::Output { name } = &node.op {
            if let Some(data) = node_outputs.get(&(node.id, 0)) {
                outputs.insert(name.clone(), data.clone());
            }
        }
    }

    Ok(ExecutionResult { outputs, stats })
}

// === Helper functions to get inputs from edges ===

fn get_one_input(
    graph: &Graph,
    node_id: NodeId,
    outputs: &HashMap<(NodeId, u8), TensorData>,
) -> Result<TensorData, ExecutionError> {
    let incoming = graph.incoming_edges(node_id);
    let edge = incoming
        .first()
        .ok_or(ExecutionError::MissingInput(node_id, "input 0".into()))?;
    outputs
        .get(&(edge.from_node, edge.from_port))
        .cloned()
        .ok_or(ExecutionError::MissingInput(
            node_id,
            format!("from node {}", edge.from_node),
        ))
}

fn get_two_inputs(
    graph: &Graph,
    node_id: NodeId,
    outputs: &HashMap<(NodeId, u8), TensorData>,
) -> Result<(TensorData, TensorData), ExecutionError> {
    let incoming = graph.incoming_edges(node_id);
    if incoming.len() < 2 {
        return Err(ExecutionError::MissingInput(
            node_id,
            "need 2 inputs".into(),
        ));
    }

    let a = outputs
        .get(&(incoming[0].from_node, incoming[0].from_port))
        .cloned()
        .ok_or(ExecutionError::MissingInput(node_id, "input a".into()))?;
    let b = outputs
        .get(&(incoming[1].from_node, incoming[1].from_port))
        .cloned()
        .ok_or(ExecutionError::MissingInput(node_id, "input b".into()))?;

    Ok((a, b))
}

// === Tensor operations (Phase 1: pure Rust, no SIMD yet) ===

fn tensor_add(a: &TensorData, b: &TensorData) -> Result<TensorData, ExecutionError> {
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "add: input a not f32".into(),
    ))?;
    let vb = b.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "add: input b not f32".into(),
    ))?;

    if va.len() != vb.len() {
        return Err(ExecutionError::ShapeMismatch {
            expected: format!("{}", a.shape),
            got: format!("{}", b.shape),
        });
    }

    let result: Vec<f32> = va.iter().zip(vb.iter()).map(|(x, y)| x + y).collect();
    Ok(TensorData::from_f32(a.shape.clone(), &result))
}

fn tensor_mul(a: &TensorData, b: &TensorData) -> Result<TensorData, ExecutionError> {
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "mul: input a not f32".into(),
    ))?;
    let vb = b.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "mul: input b not f32".into(),
    ))?;

    if va.len() != vb.len() {
        return Err(ExecutionError::ShapeMismatch {
            expected: format!("{}", a.shape),
            got: format!("{}", b.shape),
        });
    }

    let result: Vec<f32> = va.iter().zip(vb.iter()).map(|(x, y)| x * y).collect();
    Ok(TensorData::from_f32(a.shape.clone(), &result))
}

fn tensor_matmul(a: &TensorData, b: &TensorData) -> Result<TensorData, ExecutionError> {
    use qlang_core::tensor::{Dim, Shape};

    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "matmul: input a not f32".into(),
    ))?;
    let vb = b.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "matmul: input b not f32".into(),
    ))?;

    // Get dimensions: a is [m, k], b is [k, n]
    let (m, k) = match a.shape.0.as_slice() {
        [Dim::Fixed(m), Dim::Fixed(k)] => (*m, *k),
        _ => {
            return Err(ExecutionError::RuntimeError(
                "matmul: input a must be 2D".into(),
            ))
        }
    };
    let (k2, n) = match b.shape.0.as_slice() {
        [Dim::Fixed(k2), Dim::Fixed(n)] => (*k2, *n),
        _ => {
            return Err(ExecutionError::RuntimeError(
                "matmul: input b must be 2D".into(),
            ))
        }
    };

    if k != k2 {
        return Err(ExecutionError::ShapeMismatch {
            expected: format!("[{m}, {k}] × [{k}, ?]"),
            got: format!("[{m}, {k}] × [{k2}, {n}]"),
        });
    }

    // Naive matmul: O(m*n*k) — Phase 2 will use BLAS/LLVM vectorization
    let mut result = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += va[i * k + p] * vb[p * n + j];
            }
            result[i * n + j] = sum;
        }
    }

    Ok(TensorData::from_f32(Shape::matrix(m, n), &result))
}

fn tensor_relu(a: &TensorData) -> Result<TensorData, ExecutionError> {
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "relu: input not f32".into(),
    ))?;
    let result: Vec<f32> = va.iter().map(|&x| x.max(0.0)).collect();
    Ok(TensorData::from_f32(a.shape.clone(), &result))
}

fn tensor_neg(a: &TensorData) -> Result<TensorData, ExecutionError> {
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "neg: input not f32".into(),
    ))?;
    let result: Vec<f32> = va.iter().map(|&x| -x).collect();
    Ok(TensorData::from_f32(a.shape.clone(), &result))
}

/// IGQK ternary compression: project f32 weights to {-1, 0, +1}.
///
/// Uses threshold-based projection:
///   w > threshold  →  +1
///   w < -threshold →  -1
///   otherwise      →   0
///
/// The threshold is computed as mean(|w|) * 0.7 (a common heuristic).
/// Theorem 5.2 bounds the distortion of this projection.
fn tensor_to_ternary(a: &TensorData) -> Result<TensorData, ExecutionError> {
    use qlang_core::tensor::{Dtype, Shape};

    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "to_ternary: input not f32".into(),
    ))?;

    // Compute threshold
    let mean_abs: f32 = va.iter().map(|x| x.abs()).sum::<f32>() / va.len() as f32;
    let threshold = mean_abs * 0.7;

    // Project to ternary
    let result: Vec<u8> = va
        .iter()
        .map(|&x| {
            if x > threshold {
                1u8 // represents +1 as i8
            } else if x < -threshold {
                255u8 // represents -1 as i8 (two's complement)
            } else {
                0u8 // represents 0
            }
        })
        .collect();

    Ok(TensorData {
        dtype: Dtype::Ternary,
        shape: a.shape.clone(),
        data: result,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::graph::Graph;
    use qlang_core::ops::Op;
    use qlang_core::tensor::{Dtype, Shape, TensorData, TensorType};

    #[test]
    fn execute_add_graph() {
        // Build graph: a + b = y
        let mut g = Graph::new("test_add");

        let a = g.add_node(
            Op::Input { name: "a".into() },
            vec![],
            vec![TensorType::f32_vector(4)],
        );
        let b = g.add_node(
            Op::Input { name: "b".into() },
            vec![],
            vec![TensorType::f32_vector(4)],
        );
        let add = g.add_node(
            Op::Add,
            vec![TensorType::f32_vector(4), TensorType::f32_vector(4)],
            vec![TensorType::f32_vector(4)],
        );
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![TensorType::f32_vector(4)],
            vec![],
        );

        g.add_edge(a, 0, add, 0, TensorType::f32_vector(4));
        g.add_edge(b, 0, add, 1, TensorType::f32_vector(4));
        g.add_edge(add, 0, out, 0, TensorType::f32_vector(4));

        // Execute
        let mut inputs = HashMap::new();
        inputs.insert(
            "a".to_string(),
            TensorData::from_f32(Shape::vector(4), &[1.0, 2.0, 3.0, 4.0]),
        );
        inputs.insert(
            "b".to_string(),
            TensorData::from_f32(Shape::vector(4), &[10.0, 20.0, 30.0, 40.0]),
        );

        let result = execute(&g, inputs).unwrap();
        let y = result.outputs.get("y").unwrap();
        let values = y.as_f32_slice().unwrap();

        assert_eq!(values, vec![11.0, 22.0, 33.0, 44.0]);
        assert_eq!(result.stats.nodes_executed, 4);
    }

    #[test]
    fn execute_matmul_graph() {
        let mut g = Graph::new("test_matmul");

        let a = g.add_node(
            Op::Input { name: "a".into() },
            vec![],
            vec![TensorType::f32_matrix(2, 3)],
        );
        let b = g.add_node(
            Op::Input { name: "b".into() },
            vec![],
            vec![TensorType::f32_matrix(3, 2)],
        );
        let mm = g.add_node(
            Op::MatMul,
            vec![TensorType::f32_matrix(2, 3), TensorType::f32_matrix(3, 2)],
            vec![TensorType::f32_matrix(2, 2)],
        );
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![TensorType::f32_matrix(2, 2)],
            vec![],
        );

        g.add_edge(a, 0, mm, 0, TensorType::f32_matrix(2, 3));
        g.add_edge(b, 0, mm, 1, TensorType::f32_matrix(3, 2));
        g.add_edge(mm, 0, out, 0, TensorType::f32_matrix(2, 2));

        // A = [[1,2,3],[4,5,6]], B = [[1,0],[0,1],[1,1]]
        // A*B = [[4,5],[10,11]]
        let mut inputs = HashMap::new();
        inputs.insert(
            "a".to_string(),
            TensorData::from_f32(Shape::matrix(2, 3), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );
        inputs.insert(
            "b".to_string(),
            TensorData::from_f32(Shape::matrix(3, 2), &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        );

        let result = execute(&g, inputs).unwrap();
        let y = result.outputs.get("y").unwrap();
        let values = y.as_f32_slice().unwrap();

        assert_eq!(values, vec![4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn execute_ternary_compression() {
        let mut g = Graph::new("test_ternary");

        let input = g.add_node(
            Op::Input { name: "weights".into() },
            vec![],
            vec![TensorType::f32_vector(8)],
        );
        let ternary = g.add_node(
            Op::ToTernary,
            vec![TensorType::f32_vector(8)],
            vec![TensorType::new(Dtype::Ternary, Shape::vector(8))],
        );
        let out = g.add_node(
            Op::Output { name: "compressed".into() },
            vec![TensorType::new(Dtype::Ternary, Shape::vector(8))],
            vec![],
        );

        g.add_edge(input, 0, ternary, 0, TensorType::f32_vector(8));
        g.add_edge(ternary, 0, out, 0, TensorType::new(Dtype::Ternary, Shape::vector(8)));

        let mut inputs = HashMap::new();
        // Large positive, small, large negative values
        inputs.insert(
            "weights".to_string(),
            TensorData::from_f32(
                Shape::vector(8),
                &[0.9, -0.8, 0.01, -0.02, 0.7, -0.6, 0.0, 0.5],
            ),
        );

        let result = execute(&g, inputs).unwrap();
        let compressed = result.outputs.get("compressed").unwrap();

        assert_eq!(compressed.dtype, Dtype::Ternary);
        assert_eq!(result.stats.quantum_ops, 1);
    }
}
