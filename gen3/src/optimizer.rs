//! Graph Optimizer & Rewrite Engine.
//!
//! Performs algebraic simplifications, constant folding, common subexpression
//! elimination (CSE), and dead code pruning on the DAG while preserving all invariants.

use std::collections::{HashMap, HashSet};
use crate::dag::{Graph, Node, OpCode};

pub struct Optimizer;

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationStats {
    pub initial_nodes: usize,
    pub final_nodes: usize,
    pub constants_folded: usize,
    pub algebraic_reductions: usize,
    pub cse_eliminations: usize,
    pub dead_nodes_removed: usize,
}

impl Optimizer {
    pub fn optimize(graph: &Graph) -> (Graph, OptimizationStats) {
        let initial_nodes = graph.nodes.len();
        let mut stats = OptimizationStats {
            initial_nodes,
            final_nodes: 0,
            constants_folded: 0,
            algebraic_reductions: 0,
            cse_eliminations: 0,
            dead_nodes_removed: 0,
        };

        let mut current = graph.clone();

        // 1. Iterative simplification (Constant folding + Algebraic Identities + CSE)
        loop {
            let (next, changed) = Self::pass_simplify(&current, &mut stats);
            current = next;
            if !changed {
                break;
            }
        }

        // 2. Dead code elimination (remove unreachable nodes from return_node)
        current = Self::pass_dce(&current, &mut stats);

        stats.final_nodes = current.nodes.len();
        (current, stats)
    }

    fn pass_simplify(graph: &Graph, stats: &mut OptimizationStats) -> (Graph, bool) {
        let mut changed = false;
        let mut new_nodes = Vec::with_capacity(graph.nodes.len());
        // Map from old node ID to replacement node ID (or folded constant)
        let mut alias_map: HashMap<u32, u32> = HashMap::new();

        // Store signatures for CSE: (OpCode, Vec<InputId>) -> NodeId
        let mut cse_table: HashMap<(String, Vec<u32>), u32> = HashMap::new();

        for node in &graph.nodes {
            let remapped_inputs: Vec<u32> = node
                .inputs
                .iter()
                .map(|inp| *alias_map.get(inp).unwrap_or(inp))
                .collect();

            // Try Constant Folding
            if let Some(folded) = Self::try_fold_constant(&node.op, &remapped_inputs, &new_nodes) {
                let folded_id = new_nodes.len() as u32;
                new_nodes.push(Node {
                    id: folded_id,
                    op: folded,
                    inputs: vec![],
                    ty: node.ty,
                    constraint: node.constraint.clone(),
                });
                alias_map.insert(node.id, folded_id);
                stats.constants_folded += 1;
                changed = true;
                continue;
            }

            // Try Algebraic Simplification
            if let Some(simplified_target) = Self::try_algebraic_identity(&node.op, &remapped_inputs, &new_nodes) {
                alias_map.insert(node.id, simplified_target);
                stats.algebraic_reductions += 1;
                changed = true;
                continue;
            }

            // Try Common Subexpression Elimination (CSE)
            let key = (format!("{:?}", node.op), remapped_inputs.clone());
            if let Some(&existing_id) = cse_table.get(&key) {
                alias_map.insert(node.id, existing_id);
                stats.cse_eliminations += 1;
                changed = true;
                continue;
            }

            let new_id = new_nodes.len() as u32;
            cse_table.insert(key, new_id);
            alias_map.insert(node.id, new_id);
            new_nodes.push(Node {
                id: new_id,
                op: node.op.clone(),
                inputs: remapped_inputs,
                ty: node.ty,
                constraint: node.constraint.clone(),
            });
        }

        let new_return = *alias_map.get(&graph.return_node).unwrap_or(&graph.return_node);

        let new_graph = Graph {
            name: graph.name.clone(),
            input_types: graph.input_types.clone(),
            return_type: graph.return_type,
            nodes: new_nodes,
            return_node: new_return,
        };

        (new_graph, changed)
    }

    fn try_fold_constant(op: &OpCode, inputs: &[u32], nodes: &[Node]) -> Option<OpCode> {
        match op {
            OpCode::Add if inputs.len() == 2 => {
                let a = &nodes[inputs[0] as usize].op;
                let b = &nodes[inputs[1] as usize].op;
                if let (OpCode::ConstI64(v1), OpCode::ConstI64(v2)) = (a, b) {
                    return Some(OpCode::ConstI64(v1.saturating_add(*v2)));
                }
            }
            OpCode::Sub if inputs.len() == 2 => {
                let a = &nodes[inputs[0] as usize].op;
                let b = &nodes[inputs[1] as usize].op;
                if let (OpCode::ConstI64(v1), OpCode::ConstI64(v2)) = (a, b) {
                    return Some(OpCode::ConstI64(v1.saturating_sub(*v2)));
                }
            }
            OpCode::Mul if inputs.len() == 2 => {
                let a = &nodes[inputs[0] as usize].op;
                let b = &nodes[inputs[1] as usize].op;
                if let (OpCode::ConstI64(v1), OpCode::ConstI64(v2)) = (a, b) {
                    return Some(OpCode::ConstI64(v1.saturating_mul(*v2)));
                }
            }
            OpCode::Div if inputs.len() == 2 => {
                let a = &nodes[inputs[0] as usize].op;
                let b = &nodes[inputs[1] as usize].op;
                if let (OpCode::ConstI64(v1), OpCode::ConstI64(v2)) = (a, b) {
                    if *v2 != 0 {
                        return Some(OpCode::ConstI64(v1 / v2));
                    }
                }
            }
            OpCode::Clamp if inputs.len() == 3 => {
                let v = &nodes[inputs[0] as usize].op;
                let low = &nodes[inputs[1] as usize].op;
                let high = &nodes[inputs[2] as usize].op;
                if let (OpCode::ConstI64(val), OpCode::ConstI64(l), OpCode::ConstI64(h)) = (v, low, high) {
                    let clamped = (*val).max(*l).min(*h);
                    return Some(OpCode::ConstI64(clamped));
                }
            }
            OpCode::Abs if inputs.len() == 1 => {
                if let OpCode::ConstI64(v) = &nodes[inputs[0] as usize].op {
                    return Some(OpCode::ConstI64(v.abs()));
                }
            }
            _ => {}
        }
        None
    }

    fn try_algebraic_identity(op: &OpCode, inputs: &[u32], nodes: &[Node]) -> Option<u32> {
        match op {
            // x + 0 = x
            OpCode::Add if inputs.len() == 2 => {
                let a = &nodes[inputs[0] as usize].op;
                let b = &nodes[inputs[1] as usize].op;
                if matches!(b, OpCode::ConstI64(0)) {
                    return Some(inputs[0]);
                }
                if matches!(a, OpCode::ConstI64(0)) {
                    return Some(inputs[1]);
                }
            }
            // x - 0 = x
            OpCode::Sub if inputs.len() == 2 => {
                let b = &nodes[inputs[1] as usize].op;
                if matches!(b, OpCode::ConstI64(0)) {
                    return Some(inputs[0]);
                }
            }
            // x * 1 = x, 1 * x = x
            OpCode::Mul if inputs.len() == 2 => {
                let a = &nodes[inputs[0] as usize].op;
                let b = &nodes[inputs[1] as usize].op;
                if matches!(b, OpCode::ConstI64(1)) {
                    return Some(inputs[0]);
                }
                if matches!(a, OpCode::ConstI64(1)) {
                    return Some(inputs[1]);
                }
            }
            // x / 1 = x
            OpCode::Div if inputs.len() == 2 => {
                let b = &nodes[inputs[1] as usize].op;
                if matches!(b, OpCode::ConstI64(1)) {
                    return Some(inputs[0]);
                }
            }
            _ => {}
        }
        None
    }

    /// Dead Code Elimination: Keeps only nodes strictly reachable from return_node.
    fn pass_dce(graph: &Graph, stats: &mut OptimizationStats) -> Graph {
        let mut reachable: HashSet<u32> = HashSet::new();
        let mut stack = vec![graph.return_node];

        while let Some(curr) = stack.pop() {
            if reachable.insert(curr) {
                if let Some(node) = graph.nodes.get(curr as usize) {
                    for &inp in &node.inputs {
                        stack.push(inp);
                    }
                }
            }
        }

        // Remap IDs
        let mut new_nodes = Vec::new();
        let mut remap: HashMap<u32, u32> = HashMap::new();

        for node in &graph.nodes {
            if reachable.contains(&node.id) {
                let new_id = new_nodes.len() as u32;
                remap.insert(node.id, new_id);
                new_nodes.push(node.clone());
            } else {
                stats.dead_nodes_removed += 1;
            }
        }

        // Fix node IDs and inputs
        for (idx, node) in new_nodes.iter_mut().enumerate() {
            node.id = idx as u32;
            node.inputs = node
                .inputs
                .iter()
                .map(|inp| *remap.get(inp).expect("Broken graph reference during DCE"))
                .collect();
        }

        let new_return = *remap.get(&graph.return_node).unwrap_or(&0);

        Graph {
            name: graph.name.clone(),
            input_types: graph.input_types.clone(),
            return_type: graph.return_type,
            nodes: new_nodes,
            return_node: new_return,
        }
    }
}
