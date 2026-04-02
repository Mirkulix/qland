//! ONNX-compatible graph import/export for QLANG.
//!
//! Provides conversion between QLANG's internal graph representation and an
//! ONNX-like intermediate format, enabling interoperability with PyTorch and
//! TensorFlow models.
//!
//! This module does NOT depend on ONNX protobuf libraries. Instead it defines
//! lightweight Rust structs that mirror the ONNX spec and offers a JSON
//! serialization path for easy exchange with Python tooling.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use qlang_core::graph::{Graph, NodeId};
use qlang_core::ops::Op;
use qlang_core::tensor::{Dim, Dtype, Shape, TensorType};

// ---------------------------------------------------------------------------
// ONNX-like data structures
// ---------------------------------------------------------------------------

/// A single attribute value on an ONNX node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum OnnxAttribute {
    Int(i64),
    Float(f64),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
}

/// A node in an ONNX graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnnxNode {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, OnnxAttribute>,
}

/// A named tensor value-info (type descriptor) used for graph inputs/outputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnnxValueInfo {
    pub name: String,
    pub elem_type: String,
    pub shape: Vec<OnnxDim>,
}

/// A dimension in an ONNX tensor shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OnnxDim {
    Fixed(usize),
    Dynamic(String),
}

/// A named initializer (constant tensor) in an ONNX graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnnxInitializer {
    pub name: String,
    pub elem_type: String,
    pub shape: Vec<usize>,
    /// Raw data encoded as base64 in JSON; left empty for structural conversion.
    pub data_base64: Option<String>,
}

/// Top-level ONNX-like graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnnxGraph {
    pub name: String,
    pub nodes: Vec<OnnxNode>,
    pub inputs: Vec<OnnxValueInfo>,
    pub outputs: Vec<OnnxValueInfo>,
    pub initializers: Vec<OnnxInitializer>,
}

// ---------------------------------------------------------------------------
// Conversion errors
// ---------------------------------------------------------------------------

/// Errors produced when converting between ONNX and QLANG representations.
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    #[error("unsupported ONNX op type: {0}")]
    UnsupportedOp(String),

    #[error("missing required attribute '{attr}' on node '{node}'")]
    MissingAttribute { node: String, attr: String },

    #[error("attribute type mismatch for '{attr}' on node '{node}'")]
    AttributeTypeMismatch { node: String, attr: String },

    #[error("unknown element type: {0}")]
    UnknownElemType(String),

    #[error("edge wiring error: {0}")]
    WiringError(String),
}

// ---------------------------------------------------------------------------
// Op mapping: QLANG Op <-> ONNX op_type string
// ---------------------------------------------------------------------------

/// Map a QLANG Op to an ONNX op_type string and optional attributes.
fn op_to_onnx(op: &Op) -> Option<(String, HashMap<String, OnnxAttribute>)> {
    let mut attrs = HashMap::new();
    let op_type = match op {
        // Tensor ops
        Op::Add => "Add",
        Op::Sub => "Sub",
        Op::Mul => "Mul",
        Op::Div => "Div",
        Op::Neg => "Neg",
        Op::MatMul => "MatMul",
        Op::Transpose => "Transpose",
        Op::Reshape { target_shape } => {
            attrs.insert(
                "shape".to_string(),
                OnnxAttribute::Ints(target_shape.iter().map(|&d| d as i64).collect()),
            );
            "Reshape"
        }
        Op::Slice { start, end } => {
            attrs.insert(
                "starts".to_string(),
                OnnxAttribute::Ints(start.iter().map(|&s| s as i64).collect()),
            );
            attrs.insert(
                "ends".to_string(),
                OnnxAttribute::Ints(end.iter().map(|&e| e as i64).collect()),
            );
            "Slice"
        }
        Op::Concat { axis } => {
            attrs.insert("axis".to_string(), OnnxAttribute::Int(*axis as i64));
            "Concat"
        }
        Op::ReduceSum { axis } => {
            if let Some(a) = axis {
                attrs.insert("axes".to_string(), OnnxAttribute::Ints(vec![*a as i64]));
            }
            "ReduceSum"
        }
        Op::ReduceMean { axis } => {
            if let Some(a) = axis {
                attrs.insert("axes".to_string(), OnnxAttribute::Ints(vec![*a as i64]));
            }
            "ReduceMean"
        }
        Op::ReduceMax { axis } => {
            if let Some(a) = axis {
                attrs.insert("axes".to_string(), OnnxAttribute::Ints(vec![*a as i64]));
            }
            "ReduceMax"
        }
        // Activations
        Op::Relu => "Relu",
        Op::Sigmoid => "Sigmoid",
        Op::Tanh => "Tanh",
        Op::Softmax { axis } => {
            attrs.insert("axis".to_string(), OnnxAttribute::Int(*axis as i64));
            "Softmax"
        }
        // Transformer ops
        Op::LayerNorm { eps } => {
            attrs.insert("epsilon".to_string(), OnnxAttribute::Float(*eps));
            "LayerNormalization"
        }
        Op::Gelu => "Gelu",
        Op::Dropout { rate } => {
            attrs.insert("ratio".to_string(), OnnxAttribute::Float(*rate));
            "Dropout"
        }
        // Constant
        Op::Constant => "Constant",
        // Residual mapped as Add (it's x + f(x))
        Op::Residual => "Add",
        // Ops that have no standard ONNX equivalent get a prefixed domain.
        Op::Superpose
        | Op::Evolve { .. }
        | Op::Measure
        | Op::Entangle
        | Op::Collapse
        | Op::Entropy
        | Op::ToTernary
        | Op::ToLowRank { .. }
        | Op::ToSparse { .. }
        | Op::FisherMetric
        | Op::Project { .. }
        | Op::Attention { .. }
        | Op::Embedding { .. }
        | Op::Cond
        | Op::Scan { .. }
        | Op::SubGraph { .. } => {
            return Some((format!("com.qlang.{op}"), op_to_qlang_attrs(op)));
        }
        // Input/Output are not emitted as ONNX nodes.
        Op::Input { .. } | Op::Output { .. } => return None,
    };
    Some((op_type.to_string(), attrs))
}

/// Produce QLANG-specific attributes for quantum/custom ops.
fn op_to_qlang_attrs(op: &Op) -> HashMap<String, OnnxAttribute> {
    let mut attrs = HashMap::new();
    match op {
        Op::Evolve { gamma, dt } => {
            attrs.insert("gamma".to_string(), OnnxAttribute::Float(*gamma));
            attrs.insert("dt".to_string(), OnnxAttribute::Float(*dt));
        }
        Op::ToLowRank { rank } => {
            attrs.insert("rank".to_string(), OnnxAttribute::Int(*rank as i64));
        }
        Op::ToSparse { sparsity } => {
            attrs.insert("sparsity".to_string(), OnnxAttribute::Float(*sparsity));
        }
        Op::Softmax { axis } => {
            attrs.insert("axis".to_string(), OnnxAttribute::Int(*axis as i64));
        }
        Op::Attention { n_heads, d_model } => {
            attrs.insert("n_heads".to_string(), OnnxAttribute::Int(*n_heads as i64));
            attrs.insert("d_model".to_string(), OnnxAttribute::Int(*d_model as i64));
        }
        Op::Embedding { vocab_size, d_model } => {
            attrs.insert(
                "vocab_size".to_string(),
                OnnxAttribute::Int(*vocab_size as i64),
            );
            attrs.insert("d_model".to_string(), OnnxAttribute::Int(*d_model as i64));
        }
        Op::Scan { n_iterations } => {
            attrs.insert(
                "n_iterations".to_string(),
                OnnxAttribute::Int(*n_iterations as i64),
            );
        }
        Op::SubGraph { graph_id } => {
            attrs.insert(
                "graph_id".to_string(),
                OnnxAttribute::String(graph_id.clone()),
            );
        }
        _ => {}
    }
    attrs
}

/// Map an ONNX op_type back to a QLANG Op, using provided attributes.
fn onnx_op_to_qlang(
    op_type: &str,
    attrs: &HashMap<String, OnnxAttribute>,
    node_name: &str,
) -> Result<Op, ConversionError> {
    match op_type {
        "Add" => Ok(Op::Add),
        "Sub" => Ok(Op::Sub),
        "Mul" => Ok(Op::Mul),
        "Div" => Ok(Op::Div),
        "Neg" => Ok(Op::Neg),
        "MatMul" => Ok(Op::MatMul),
        "Transpose" => Ok(Op::Transpose),
        "Reshape" => {
            let shape = get_ints_attr(attrs, "shape", node_name)?;
            Ok(Op::Reshape {
                target_shape: shape.iter().map(|&v| v as usize).collect(),
            })
        }
        "Slice" => {
            let starts = get_ints_attr(attrs, "starts", node_name)?;
            let ends = get_ints_attr(attrs, "ends", node_name)?;
            Ok(Op::Slice {
                start: starts.iter().map(|&v| v as usize).collect(),
                end: ends.iter().map(|&v| v as usize).collect(),
            })
        }
        "Concat" => {
            let axis = get_int_attr(attrs, "axis", node_name)?;
            Ok(Op::Concat {
                axis: axis as usize,
            })
        }
        "ReduceSum" => {
            let axis = optional_axes_attr(attrs);
            Ok(Op::ReduceSum { axis })
        }
        "ReduceMean" => {
            let axis = optional_axes_attr(attrs);
            Ok(Op::ReduceMean { axis })
        }
        "ReduceMax" => {
            let axis = optional_axes_attr(attrs);
            Ok(Op::ReduceMax { axis })
        }
        "Relu" => Ok(Op::Relu),
        "Sigmoid" => Ok(Op::Sigmoid),
        "Tanh" => Ok(Op::Tanh),
        "Softmax" => {
            let axis = get_int_attr(attrs, "axis", node_name).unwrap_or(1);
            Ok(Op::Softmax {
                axis: axis as usize,
            })
        }
        "LayerNormalization" => {
            let eps = get_float_attr(attrs, "epsilon", node_name).unwrap_or(1e-5);
            Ok(Op::LayerNorm { eps })
        }
        "Gelu" => Ok(Op::Gelu),
        "Dropout" => {
            let rate = get_float_attr(attrs, "ratio", node_name).unwrap_or(0.5);
            Ok(Op::Dropout { rate })
        }
        "Constant" => Ok(Op::Constant),
        other if other.starts_with("com.qlang.") => {
            qlang_custom_op(other.strip_prefix("com.qlang.").unwrap(), attrs, node_name)
        }
        _ => Err(ConversionError::UnsupportedOp(op_type.to_string())),
    }
}

/// Convert QLANG custom-domain ops back to Op.
fn qlang_custom_op(
    tag: &str,
    attrs: &HashMap<String, OnnxAttribute>,
    node_name: &str,
) -> Result<Op, ConversionError> {
    // The tag is the Display representation of the Op.
    // We match on known prefixes.
    if tag.starts_with("superpose") {
        return Ok(Op::Superpose);
    }
    if tag.starts_with("evolve") {
        let gamma = get_float_attr(attrs, "gamma", node_name)?;
        let dt = get_float_attr(attrs, "dt", node_name)?;
        return Ok(Op::Evolve { gamma, dt });
    }
    if tag.starts_with("measure") {
        return Ok(Op::Measure);
    }
    if tag.starts_with("entangle") {
        return Ok(Op::Entangle);
    }
    if tag.starts_with("collapse") {
        return Ok(Op::Collapse);
    }
    if tag.starts_with("entropy") {
        return Ok(Op::Entropy);
    }
    if tag.starts_with("to_ternary") {
        return Ok(Op::ToTernary);
    }
    if tag.starts_with("to_lowrank") {
        let rank = get_int_attr(attrs, "rank", node_name)? as usize;
        return Ok(Op::ToLowRank { rank });
    }
    if tag.starts_with("to_sparse") {
        let sparsity = get_float_attr(attrs, "sparsity", node_name)?;
        return Ok(Op::ToSparse { sparsity });
    }
    if tag.starts_with("fisher_metric") {
        return Ok(Op::FisherMetric);
    }
    if tag.starts_with("attention") {
        let n_heads = get_int_attr(attrs, "n_heads", node_name)? as usize;
        let d_model = get_int_attr(attrs, "d_model", node_name)? as usize;
        return Ok(Op::Attention { n_heads, d_model });
    }
    if tag.starts_with("embedding") {
        let vocab_size = get_int_attr(attrs, "vocab_size", node_name)? as usize;
        let d_model = get_int_attr(attrs, "d_model", node_name)? as usize;
        return Ok(Op::Embedding {
            vocab_size,
            d_model,
        });
    }
    if tag.starts_with("scan") {
        let n = get_int_attr(attrs, "n_iterations", node_name)? as usize;
        return Ok(Op::Scan { n_iterations: n });
    }
    if tag.starts_with("subgraph") {
        let gid = get_string_attr(attrs, "graph_id", node_name)?;
        return Ok(Op::SubGraph { graph_id: gid });
    }
    Err(ConversionError::UnsupportedOp(format!(
        "com.qlang.{tag}"
    )))
}

// ---------------------------------------------------------------------------
// Attribute helpers
// ---------------------------------------------------------------------------

fn get_int_attr(
    attrs: &HashMap<String, OnnxAttribute>,
    key: &str,
    node: &str,
) -> Result<i64, ConversionError> {
    match attrs.get(key) {
        Some(OnnxAttribute::Int(v)) => Ok(*v),
        Some(_) => Err(ConversionError::AttributeTypeMismatch {
            node: node.to_string(),
            attr: key.to_string(),
        }),
        None => Err(ConversionError::MissingAttribute {
            node: node.to_string(),
            attr: key.to_string(),
        }),
    }
}

fn get_float_attr(
    attrs: &HashMap<String, OnnxAttribute>,
    key: &str,
    node: &str,
) -> Result<f64, ConversionError> {
    match attrs.get(key) {
        Some(OnnxAttribute::Float(v)) => Ok(*v),
        Some(_) => Err(ConversionError::AttributeTypeMismatch {
            node: node.to_string(),
            attr: key.to_string(),
        }),
        None => Err(ConversionError::MissingAttribute {
            node: node.to_string(),
            attr: key.to_string(),
        }),
    }
}

fn get_ints_attr(
    attrs: &HashMap<String, OnnxAttribute>,
    key: &str,
    node: &str,
) -> Result<Vec<i64>, ConversionError> {
    match attrs.get(key) {
        Some(OnnxAttribute::Ints(v)) => Ok(v.clone()),
        Some(_) => Err(ConversionError::AttributeTypeMismatch {
            node: node.to_string(),
            attr: key.to_string(),
        }),
        None => Err(ConversionError::MissingAttribute {
            node: node.to_string(),
            attr: key.to_string(),
        }),
    }
}

fn get_string_attr(
    attrs: &HashMap<String, OnnxAttribute>,
    key: &str,
    node: &str,
) -> Result<String, ConversionError> {
    match attrs.get(key) {
        Some(OnnxAttribute::String(v)) => Ok(v.clone()),
        Some(_) => Err(ConversionError::AttributeTypeMismatch {
            node: node.to_string(),
            attr: key.to_string(),
        }),
        None => Err(ConversionError::MissingAttribute {
            node: node.to_string(),
            attr: key.to_string(),
        }),
    }
}

fn optional_axes_attr(attrs: &HashMap<String, OnnxAttribute>) -> Option<usize> {
    if let Some(OnnxAttribute::Ints(axes)) = attrs.get("axes") {
        axes.first().map(|&a| a as usize)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Dtype <-> ONNX elem_type mapping
// ---------------------------------------------------------------------------

fn dtype_to_onnx(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F16 => "FLOAT16",
        Dtype::F32 => "FLOAT",
        Dtype::F64 => "DOUBLE",
        Dtype::I8 => "INT8",
        Dtype::I16 => "INT16",
        Dtype::I32 => "INT32",
        Dtype::I64 => "INT64",
        Dtype::Bool => "BOOL",
        Dtype::Ternary => "INT8", // closest standard type
    }
}

fn onnx_to_dtype(elem_type: &str) -> Result<Dtype, ConversionError> {
    match elem_type {
        "FLOAT16" => Ok(Dtype::F16),
        "FLOAT" => Ok(Dtype::F32),
        "DOUBLE" => Ok(Dtype::F64),
        "INT8" => Ok(Dtype::I8),
        "INT16" => Ok(Dtype::I16),
        "INT32" => Ok(Dtype::I32),
        "INT64" => Ok(Dtype::I64),
        "BOOL" => Ok(Dtype::Bool),
        _ => Err(ConversionError::UnknownElemType(elem_type.to_string())),
    }
}

fn shape_to_onnx(shape: &Shape) -> Vec<OnnxDim> {
    shape
        .0
        .iter()
        .map(|d| match d {
            Dim::Fixed(n) => OnnxDim::Fixed(*n),
            Dim::Dynamic => OnnxDim::Dynamic("?".to_string()),
        })
        .collect()
}

fn onnx_to_shape(dims: &[OnnxDim]) -> Shape {
    Shape(
        dims.iter()
            .map(|d| match d {
                OnnxDim::Fixed(n) => Dim::Fixed(*n),
                OnnxDim::Dynamic(_) => Dim::Dynamic,
            })
            .collect(),
    )
}

fn tensor_type_to_value_info(name: &str, tt: &TensorType) -> OnnxValueInfo {
    OnnxValueInfo {
        name: name.to_string(),
        elem_type: dtype_to_onnx(tt.dtype).to_string(),
        shape: shape_to_onnx(&tt.shape),
    }
}

// ---------------------------------------------------------------------------
// Wire-name helpers
// ---------------------------------------------------------------------------

/// Produce a deterministic edge wire name: `node{id}_out{port}`.
fn wire_name(node_id: NodeId, port: u8) -> String {
    format!("node{node_id}_out{port}")
}

// ---------------------------------------------------------------------------
// QLANG Graph -> OnnxGraph
// ---------------------------------------------------------------------------

/// Convert a QLANG [`Graph`] into an [`OnnxGraph`] for export.
///
/// Input and Output nodes are mapped to graph-level inputs/outputs rather than
/// ONNX operator nodes.
pub fn to_onnx(graph: &Graph) -> OnnxGraph {
    let mut onnx_nodes = Vec::new();
    let mut onnx_inputs = Vec::new();
    let mut onnx_outputs = Vec::new();

    // Pre-build a map: for each (to_node, to_port) find the producing wire.
    let mut input_wire: HashMap<(NodeId, u8), String> = HashMap::new();
    for edge in &graph.edges {
        input_wire.insert(
            (edge.to_node, edge.to_port),
            wire_name(edge.from_node, edge.from_port),
        );
    }

    for node in &graph.nodes {
        match &node.op {
            Op::Input { name } => {
                // Register as graph input.
                let out_name = wire_name(node.id, 0);
                if let Some(tt) = node.output_types.first() {
                    onnx_inputs.push(tensor_type_to_value_info(&out_name, tt));
                } else {
                    onnx_inputs.push(OnnxValueInfo {
                        name: out_name.clone(),
                        elem_type: "FLOAT".to_string(),
                        shape: vec![],
                    });
                }
                // Store the friendly name in metadata so roundtrip preserves it.
                // We add a minimal "Identity" passthrough only if we need to
                // keep the name mapping; skip for cleanliness.
                let _ = name; // used implicitly via node metadata
            }
            Op::Output { name } => {
                // Register as graph output.
                let in_wire = input_wire
                    .get(&(node.id, 0))
                    .cloned()
                    .unwrap_or_else(|| format!("output_{name}"));
                if let Some(tt) = node.input_types.first() {
                    onnx_outputs.push(tensor_type_to_value_info(&in_wire, tt));
                } else {
                    onnx_outputs.push(OnnxValueInfo {
                        name: in_wire,
                        elem_type: "FLOAT".to_string(),
                        shape: vec![],
                    });
                }
            }
            op => {
                if let Some((op_type, mut attributes)) = op_to_onnx(op) {
                    // Build input wire list (in port order).
                    let n_in = op.n_inputs();
                    let inputs: Vec<String> = (0..n_in as u8)
                        .map(|port| {
                            input_wire
                                .get(&(node.id, port))
                                .cloned()
                                .unwrap_or_default()
                        })
                        .collect();

                    // Build output wire list.
                    let n_out = op.n_outputs();
                    let outputs: Vec<String> =
                        (0..n_out as u8).map(|port| wire_name(node.id, port)).collect();

                    // Carry over QLANG metadata as string attributes.
                    for (k, v) in &node.metadata {
                        attributes
                            .entry(format!("qlang.{k}"))
                            .or_insert_with(|| OnnxAttribute::String(v.clone()));
                    }

                    onnx_nodes.push(OnnxNode {
                        name: format!("node_{}", node.id),
                        op_type,
                        inputs,
                        outputs,
                        attributes,
                    });
                }
            }
        }
    }

    OnnxGraph {
        name: graph.id.clone(),
        nodes: onnx_nodes,
        inputs: onnx_inputs,
        outputs: onnx_outputs,
        initializers: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// OnnxGraph -> QLANG Graph
// ---------------------------------------------------------------------------

/// Convert an [`OnnxGraph`] into a QLANG [`Graph`].
pub fn from_onnx(onnx: &OnnxGraph) -> Result<Graph, ConversionError> {
    let mut graph = Graph::new(&onnx.name);

    // Maps wire-name -> (producing NodeId, port).
    let mut wire_producers: HashMap<String, (NodeId, u8)> = HashMap::new();

    // 1. Create Input nodes for each graph input.
    for vi in &onnx.inputs {
        let dtype = onnx_to_dtype(&vi.elem_type)?;
        let shape = onnx_to_shape(&vi.shape);
        let tt = TensorType::new(dtype, shape);
        let nid = graph.add_node(
            Op::Input {
                name: vi.name.clone(),
            },
            vec![],
            vec![tt],
        );
        wire_producers.insert(vi.name.clone(), (nid, 0));
    }

    // 2. Process ONNX nodes in order (assumed topological).
    for onode in &onnx.nodes {
        let op = onnx_op_to_qlang(&onode.op_type, &onode.attributes, &onode.name)?;

        // Build input types from producers where possible; default to f32 scalar.
        let input_types: Vec<TensorType> = onode
            .inputs
            .iter()
            .map(|wire| {
                if let Some(&(pid, _port)) = wire_producers.get(wire) {
                    if let Some(pnode) = graph.node(pid) {
                        pnode.output_types.first().cloned().unwrap_or_else(TensorType::f32_scalar)
                    } else {
                        TensorType::f32_scalar()
                    }
                } else {
                    TensorType::f32_scalar()
                }
            })
            .collect();

        // Output types: for simplicity, propagate the first input type (correct
        // for element-wise ops; callers can refine via type inference).
        let output_types: Vec<TensorType> = onode
            .outputs
            .iter()
            .map(|_| input_types.first().cloned().unwrap_or_else(TensorType::f32_scalar))
            .collect();

        let nid = graph.add_node(op, input_types, output_types);

        // Register output wires.
        for (port, wire) in onode.outputs.iter().enumerate() {
            wire_producers.insert(wire.clone(), (nid, port as u8));
        }

        // Create edges for inputs.
        for (to_port, wire) in onode.inputs.iter().enumerate() {
            if let Some(&(from_node, from_port)) = wire_producers.get(wire) {
                let tt = if let Some(pnode) = graph.node(from_node) {
                    pnode
                        .output_types
                        .get(from_port as usize)
                        .cloned()
                        .unwrap_or_else(TensorType::f32_scalar)
                } else {
                    TensorType::f32_scalar()
                };
                graph.add_edge(from_node, from_port, nid, to_port as u8, tt);
            }
        }
    }

    // 3. Create Output nodes for each graph output.
    for vi in &onnx.outputs {
        let dtype = onnx_to_dtype(&vi.elem_type)?;
        let shape = onnx_to_shape(&vi.shape);
        let tt = TensorType::new(dtype, shape);
        let nid = graph.add_node(
            Op::Output {
                name: vi.name.clone(),
            },
            vec![tt.clone()],
            vec![],
        );
        if let Some(&(from_node, from_port)) = wire_producers.get(&vi.name) {
            graph.add_edge(from_node, from_port, nid, 0, tt);
        }
    }

    Ok(graph)
}

// ---------------------------------------------------------------------------
// JSON export
// ---------------------------------------------------------------------------

/// Serialize a QLANG [`Graph`] as ONNX-like JSON.
///
/// This produces a JSON string representing the [`OnnxGraph`] that can be
/// consumed by Python tooling (e.g., `json.loads(...)` and then fed into
/// `onnx.helper.make_graph`).
pub fn to_onnx_json(graph: &Graph) -> String {
    let onnx = to_onnx(graph);
    serde_json::to_string_pretty(&onnx).expect("OnnxGraph serialization cannot fail")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::graph::Graph;
    use qlang_core::ops::Op;
    use qlang_core::tensor::{Dtype, Shape, TensorType};

    fn f32_vec(n: usize) -> TensorType {
        TensorType::new(Dtype::F32, Shape::vector(n))
    }

    fn f32_mat(m: usize, n: usize) -> TensorType {
        TensorType::new(Dtype::F32, Shape::matrix(m, n))
    }

    /// Build a simple graph: input -> relu -> output
    fn simple_relu_graph() -> Graph {
        let mut g = Graph::new("relu_test");
        let inp = g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32_vec(4)]);
        let relu = g.add_node(Op::Relu, vec![f32_vec(4)], vec![f32_vec(4)]);
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![f32_vec(4)],
            vec![],
        );
        g.add_edge(inp, 0, relu, 0, f32_vec(4));
        g.add_edge(relu, 0, out, 0, f32_vec(4));
        g
    }

    #[test]
    fn test_to_onnx_simple() {
        let g = simple_relu_graph();
        let onnx = to_onnx(&g);

        assert_eq!(onnx.name, "relu_test");
        assert_eq!(onnx.inputs.len(), 1);
        assert_eq!(onnx.outputs.len(), 1);
        assert_eq!(onnx.nodes.len(), 1);
        assert_eq!(onnx.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_to_onnx_json_parses() {
        let g = simple_relu_graph();
        let json = to_onnx_json(&g);
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed["name"], "relu_test");
        assert!(parsed["nodes"].as_array().unwrap().len() == 1);
    }

    #[test]
    fn test_roundtrip_relu() {
        let g = simple_relu_graph();
        let onnx = to_onnx(&g);
        let g2 = from_onnx(&onnx).expect("conversion succeeds");

        // Should have input + relu + output = 3 nodes.
        assert_eq!(g2.nodes.len(), 3);
        // Should have 2 edges.
        assert_eq!(g2.edges.len(), 2);
        // The relu node should exist.
        assert!(g2.nodes.iter().any(|n| n.op == Op::Relu));
    }

    #[test]
    fn test_roundtrip_matmul_add() {
        let mut g = Graph::new("linear");
        let x = g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32_mat(1, 4)]);
        let w = g.add_node(Op::Input { name: "w".into() }, vec![], vec![f32_mat(4, 2)]);
        let b = g.add_node(
            Op::Input { name: "b".into() },
            vec![],
            vec![f32_vec(2)],
        );
        let mm = g.add_node(
            Op::MatMul,
            vec![f32_mat(1, 4), f32_mat(4, 2)],
            vec![f32_mat(1, 2)],
        );
        let add = g.add_node(
            Op::Add,
            vec![f32_mat(1, 2), f32_vec(2)],
            vec![f32_mat(1, 2)],
        );
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![f32_mat(1, 2)],
            vec![],
        );
        g.add_edge(x, 0, mm, 0, f32_mat(1, 4));
        g.add_edge(w, 0, mm, 1, f32_mat(4, 2));
        g.add_edge(mm, 0, add, 0, f32_mat(1, 2));
        g.add_edge(b, 0, add, 1, f32_vec(2));
        g.add_edge(add, 0, out, 0, f32_mat(1, 2));

        let onnx = to_onnx(&g);
        assert_eq!(onnx.nodes.len(), 2); // MatMul + Add
        assert_eq!(onnx.inputs.len(), 3); // x, w, b

        let g2 = from_onnx(&onnx).expect("roundtrip succeeds");
        assert_eq!(g2.nodes.len(), 6); // 3 inputs + 2 ops + 1 output
        assert!(g2.nodes.iter().any(|n| n.op == Op::MatMul));
        assert!(g2.nodes.iter().any(|n| n.op == Op::Add));
    }

    #[test]
    fn test_op_mapping_coverage() {
        // Verify that standard ops map to known ONNX names.
        let cases: Vec<(Op, &str)> = vec![
            (Op::Add, "Add"),
            (Op::Sub, "Sub"),
            (Op::Mul, "Mul"),
            (Op::Div, "Div"),
            (Op::Neg, "Neg"),
            (Op::MatMul, "MatMul"),
            (Op::Relu, "Relu"),
            (Op::Sigmoid, "Sigmoid"),
            (Op::Tanh, "Tanh"),
            (Op::Transpose, "Transpose"),
            (Op::Constant, "Constant"),
            (Op::Gelu, "Gelu"),
        ];
        for (op, expected) in cases {
            let (onnx_type, _) = op_to_onnx(&op).expect("op should map");
            assert_eq!(onnx_type, expected, "mismatch for {op:?}");
        }
    }

    #[test]
    fn test_quantum_op_roundtrip() {
        let mut g = Graph::new("quantum");
        let inp = g.add_node(Op::Input { name: "rho".into() }, vec![], vec![f32_mat(4, 4)]);
        let ent = g.add_node(
            Op::Entropy,
            vec![f32_mat(4, 4)],
            vec![TensorType::f32_scalar()],
        );
        let out = g.add_node(
            Op::Output { name: "s".into() },
            vec![TensorType::f32_scalar()],
            vec![],
        );
        g.add_edge(inp, 0, ent, 0, f32_mat(4, 4));
        g.add_edge(ent, 0, out, 0, TensorType::f32_scalar());

        let onnx = to_onnx(&g);
        assert_eq!(onnx.nodes.len(), 1);
        assert!(onnx.nodes[0].op_type.starts_with("com.qlang."));

        let g2 = from_onnx(&onnx).expect("roundtrip");
        assert!(g2.nodes.iter().any(|n| n.op == Op::Entropy));
    }

    #[test]
    fn test_unsupported_op_error() {
        let onnx = OnnxGraph {
            name: "bad".into(),
            nodes: vec![OnnxNode {
                name: "n0".into(),
                op_type: "FooBarBaz".into(),
                inputs: vec![],
                outputs: vec!["out".into()],
                attributes: HashMap::new(),
            }],
            inputs: vec![],
            outputs: vec![],
            initializers: vec![],
        };
        let result = from_onnx(&onnx);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ConversionError::UnsupportedOp(ref s) if s == "FooBarBaz"),
            "expected UnsupportedOp, got {err:?}"
        );
    }

    #[test]
    fn test_json_roundtrip() {
        let g = simple_relu_graph();
        let json = to_onnx_json(&g);
        let onnx: OnnxGraph = serde_json::from_str(&json).expect("deserialize");
        let g2 = from_onnx(&onnx).expect("from_onnx");
        assert_eq!(g2.nodes.len(), 3);
    }

    #[test]
    fn test_softmax_with_axis() {
        let mut g = Graph::new("sm");
        let inp = g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32_mat(2, 3)]);
        let sm = g.add_node(
            Op::Softmax { axis: 1 },
            vec![f32_mat(2, 3)],
            vec![f32_mat(2, 3)],
        );
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![f32_mat(2, 3)],
            vec![],
        );
        g.add_edge(inp, 0, sm, 0, f32_mat(2, 3));
        g.add_edge(sm, 0, out, 0, f32_mat(2, 3));

        let onnx = to_onnx(&g);
        assert_eq!(onnx.nodes[0].op_type, "Softmax");
        assert_eq!(
            onnx.nodes[0].attributes.get("axis"),
            Some(&OnnxAttribute::Int(1))
        );

        let g2 = from_onnx(&onnx).expect("roundtrip");
        let sm_node = g2.nodes.iter().find(|n| matches!(n.op, Op::Softmax { .. }));
        assert!(sm_node.is_some());
        assert_eq!(sm_node.unwrap().op, Op::Softmax { axis: 1 });
    }
}
