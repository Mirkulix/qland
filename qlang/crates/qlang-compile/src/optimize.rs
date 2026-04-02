use qlang_core::graph::Graph;
use qlang_core::ops::Op;

/// Optimization passes that transform the graph before code generation.
/// These operate on the graph structure — no tensor data needed.

/// Run all optimization passes on a graph.
pub fn optimize(graph: &mut Graph) {
    eliminate_dead_nodes(graph);
    fuse_consecutive_ops(graph);
}

/// Remove nodes that have no outgoing edges and are not output nodes.
fn eliminate_dead_nodes(graph: &mut Graph) {
    loop {
        let dead: Vec<u32> = graph
            .nodes
            .iter()
            .filter(|n| {
                !matches!(n.op, Op::Output { .. })
                    && graph.outgoing_edges(n.id).is_empty()
            })
            .map(|n| n.id)
            .collect();

        if dead.is_empty() {
            break;
        }

        // Remove dead nodes and their incoming edges
        for &id in &dead {
            graph.edges.retain(|e| e.to_node != id && e.from_node != id);
        }
        graph.nodes.retain(|n| !dead.contains(&n.id));
    }
}

/// Fuse consecutive operations where possible.
/// Example: MatMul followed by Add can become FusedMatMulAdd (for LLVM FMA).
///
/// Phase 1: Just identifies fusable pairs and logs them.
/// Phase 2: Actually transforms the graph.
fn fuse_consecutive_ops(graph: &Graph) {
    for node in &graph.nodes {
        if matches!(node.op, Op::MatMul) {
            let outgoing = graph.outgoing_edges(node.id);
            for edge in outgoing {
                if let Some(target) = graph.node(edge.to_node) {
                    if matches!(target.op, Op::Add) {
                        // MatMul → Add is fusable to FMA
                        // Phase 2: transform graph here
                        let _ = (node.id, target.id); // placeholder
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::graph::Graph;
    use qlang_core::ops::Op;
    use qlang_core::tensor::TensorType;

    #[test]
    fn dead_node_elimination() {
        let mut g = Graph::new("dce_test");

        let input = g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![TensorType::f32_vector(4)],
        );
        let relu = g.add_node(
            Op::Relu,
            vec![TensorType::f32_vector(4)],
            vec![TensorType::f32_vector(4)],
        );
        // Dead node: not connected to anything
        let _dead = g.add_node(
            Op::Neg,
            vec![TensorType::f32_vector(4)],
            vec![TensorType::f32_vector(4)],
        );
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![TensorType::f32_vector(4)],
            vec![],
        );

        g.add_edge(input, 0, relu, 0, TensorType::f32_vector(4));
        g.add_edge(relu, 0, out, 0, TensorType::f32_vector(4));

        assert_eq!(g.nodes.len(), 4); // including dead node
        optimize(&mut g);
        assert_eq!(g.nodes.len(), 3); // dead node removed
    }
}
