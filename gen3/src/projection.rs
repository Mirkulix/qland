//! Human View Projection Engine.
//!
//! Decompiles the machine-native dense DAG into an intuitive, human-readable
//! mathematical representation and ASCII dataflow graph.

use crate::dag::{Constraint, Graph, Node, OpCode};

pub struct Projection;

impl Projection {
    /// Decompiles the DAG into human-readable SSA (Static Single Assignment) mathematical pseudo-code.
    pub fn decompile(graph: &Graph) -> String {
        let mut out = String::new();
        out.push_str(&format!("fn {}(", graph.name));
        for (i, ty) in graph.input_types.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!("arg{i}: {ty:?}"));
        }
        out.push_str(&format!(") -> {:?} {{\n", graph.return_type));

        for node in &graph.nodes {
            let expr = Self::format_node_expr(node);
            let constraint_str = match &node.constraint {
                Some(Constraint::NonNegative) => "  # invariant: >= 0",
                Some(Constraint::StrictlyPositive) => "  # invariant: > 0",
                Some(Constraint::NonZero) => "  # invariant: != 0",
                Some(Constraint::BoundedRange(l, h)) => &format!("  # invariant: [{l}, {h}]"),
                None => "",
            };
            out.push_str(&format!("  v{:02} = {:<28}{}\n", node.id, expr, constraint_str));
        }

        out.push_str(&format!("  return v{:02}\n}}\n", graph.return_node));
        out
    }

    /// Generates a visual ASCII Dataflow Graph for terminal inspection.
    pub fn visual_graph(graph: &Graph) -> String {
        let mut out = String::new();
        out.push_str(&format!("┌─ [GEN-3 DATAFLOW DAG: {}] ({} Nodes, Return: v{:02})\n", graph.name, graph.nodes.len(), graph.return_node));
        for node in &graph.nodes {
            let inputs_str = if node.inputs.is_empty() {
                "".to_string()
            } else {
                let list: Vec<String> = node.inputs.iter().map(|id| format!("v{id:02}")).collect();
                format!(" ◄─ [{}]", list.join(", "))
            };
            let constr = match &node.constraint {
                Some(c) => format!(" 🛡️ {c:?}"),
                None => "".to_string(),
            };
            out.push_str(&format!("│  [v{:02}] {:<16} ({:?}){}{}\n", node.id, format!("{:?}", node.op), node.ty, inputs_str, constr));
        }
        out.push_str("└───────────────────────────────────────────────────────────\n");
        out
    }

    fn format_node_expr(node: &Node) -> String {
        match &node.op {
            OpCode::ConstI64(v) => format!("{v}"),
            OpCode::ConstF64(v) => format!("{v}"),
            OpCode::ConstBool(v) => format!("{v}"),
            OpCode::Param(slot) => format!("arg{slot}"),
            OpCode::Add => format!("v{:02} + v{:02}", node.inputs[0], node.inputs[1]),
            OpCode::Sub => format!("v{:02} - v{:02}", node.inputs[0], node.inputs[1]),
            OpCode::Mul => format!("v{:02} * v{:02}", node.inputs[0], node.inputs[1]),
            OpCode::Div => format!("v{:02} / v{:02}", node.inputs[0], node.inputs[1]),
            OpCode::Mod => format!("v{:02} % v{:02}", node.inputs[0], node.inputs[1]),
            OpCode::Neg => format!("-v{:02}", node.inputs[0]),
            OpCode::CmpEq => format!("v{:02} == v{:02}", node.inputs[0], node.inputs[1]),
            OpCode::CmpLt => format!("v{:02} < v{:02}", node.inputs[0], node.inputs[1]),
            OpCode::CmpLe => format!("v{:02} <= v{:02}", node.inputs[0], node.inputs[1]),
            OpCode::CmpGt => format!("v{:02} > v{:02}", node.inputs[0], node.inputs[1]),
            OpCode::CmpGe => format!("v{:02} >= v{:02}", node.inputs[0], node.inputs[1]),
            OpCode::Select => format!("v{:02} ? v{:02} : v{:02}", node.inputs[0], node.inputs[1], node.inputs[2]),
            OpCode::Clamp => format!("clamp(v{:02}, v{:02}, v{:02})", node.inputs[0], node.inputs[1], node.inputs[2]),
            OpCode::Abs => format!("abs(v{:02})", node.inputs[0]),
            OpCode::Min => format!("min(v{:02}, v{:02})", node.inputs[0], node.inputs[1]),
            OpCode::Max => format!("max(v{:02}, v{:02})", node.inputs[0], node.inputs[1]),
            OpCode::TensorMapAdd(bias) => format!("tensor_map(v{:02}, +{bias})", node.inputs[0]),
            OpCode::TensorSum => format!("tensor_sum(v{:02})", node.inputs[0]),
        }
    }
}
