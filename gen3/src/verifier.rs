//! Topological Invariant Verifier & Interval Bound Propagation Engine.
//!
//! Evaluates formal mathematical constraints directly on the Graph representation
//! before code can ever reach hardware execution.

use crate::dag::{Constraint, DataType, Graph, Node, OpCode};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub min: i64,
    pub max: i64,
}

impl Interval {
    pub fn unbounded() -> Self {
        Self { min: i64::MIN, max: i64::MAX }
    }

    pub fn point(v: i64) -> Self {
        Self { min: v, max: v }
    }

    pub fn satisfies(&self, constraint: &Constraint) -> bool {
        match constraint {
            Constraint::NonNegative => self.min >= 0,
            Constraint::StrictlyPositive => self.min > 0,
            Constraint::NonZero => self.min > 0 || self.max < 0,
            Constraint::BoundedRange(low, high) => self.min >= *low && self.max <= *high,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum VerifyError {
    CyclicOrFutureReference { node: u32, target: u32 },
    TypeMismatch { node: u32, expected: DataType, found: DataType },
    InvalidArity { node: u32, expected: usize, found: usize },
    InvariantRefuted { node: u32, constraint: Constraint, range: Interval },
    PotentialZeroDivision { node: u32, divisor_range: Interval },
}

pub struct Verifier;

impl Verifier {
    pub fn verify(graph: &Graph, input_bounds: &[Interval]) -> Result<Vec<Interval>, VerifyError> {
        let mut bounds: Vec<Interval> = Vec::with_capacity(graph.nodes.len());

        for (idx, node) in graph.nodes.iter().enumerate() {
            let nid = node.id;
            if nid as usize != idx {
                return Err(VerifyError::CyclicOrFutureReference { node: nid, target: idx as u32 });
            }

            // 1. Structural topological check: all inputs must refer to preceding nodes
            for &inp in &node.inputs {
                if inp >= nid {
                    return Err(VerifyError::CyclicOrFutureReference { node: nid, target: inp });
                }
            }

            // 2. Compute interval bounds for node
            let node_bound = match &node.op {
                OpCode::ConstI64(v) => Interval::point(*v),
                OpCode::ConstF64(_) | OpCode::ConstBool(_) => Interval::unbounded(),
                OpCode::Param(slot) => {
                    input_bounds.get(*slot as usize).copied().unwrap_or(Interval::unbounded())
                }
                OpCode::Add => {
                    Self::check_arity(node, 2)?;
                    let a = bounds[node.inputs[0] as usize];
                    let b = bounds[node.inputs[1] as usize];
                    Interval {
                        min: a.min.saturating_add(b.min),
                        max: a.max.saturating_add(b.max),
                    }
                }
                OpCode::Sub => {
                    Self::check_arity(node, 2)?;
                    let a = bounds[node.inputs[0] as usize];
                    let b = bounds[node.inputs[1] as usize];
                    Interval {
                        min: a.min.saturating_sub(b.max),
                        max: a.max.saturating_sub(b.min),
                    }
                }
                OpCode::Clamp => {
                    Self::check_arity(node, 3)?;
                    let low = bounds[node.inputs[1] as usize].min;
                    let high = bounds[node.inputs[2] as usize].max;
                    Interval { min: low, max: high }
                }
                OpCode::Abs => {
                    Self::check_arity(node, 1)?;
                    let a = bounds[node.inputs[0] as usize];
                    let max_abs = a.min.abs().max(a.max.abs());
                    let min_abs = if a.min <= 0 && a.max >= 0 { 0 } else { a.min.abs().min(a.max.abs()) };
                    Interval { min: min_abs, max: max_abs }
                }
                OpCode::Mul => {
                    Self::check_arity(node, 2)?;
                    let a = bounds[node.inputs[0] as usize];
                    let b = bounds[node.inputs[1] as usize];
                    let p1 = a.min.saturating_mul(b.min);
                    let p2 = a.min.saturating_mul(b.max);
                    let p3 = a.max.saturating_mul(b.min);
                    let p4 = a.max.saturating_mul(b.max);
                    let min = p1.min(p2).min(p3).min(p4);
                    let max = p1.max(p2).max(p3).max(p4);
                    Interval { min, max }
                }
                OpCode::Div => {
                    Self::check_arity(node, 2)?;
                    let a = bounds[node.inputs[0] as usize];
                    let den = bounds[node.inputs[1] as usize];
                    if den.min <= 0 && den.max >= 0 {
                        return Err(VerifyError::PotentialZeroDivision {
                            node: nid,
                            divisor_range: den,
                        });
                    }
                    let candidates = [
                        a.min / den.min,
                        a.min / den.max,
                        a.max / den.min,
                        a.max / den.max,
                    ];
                    let min = candidates.iter().copied().min().unwrap();
                    let max = candidates.iter().copied().max().unwrap();
                    Interval { min, max }
                }
                OpCode::Min => {
                    Self::check_arity(node, 2)?;
                    let a = bounds[node.inputs[0] as usize];
                    let b = bounds[node.inputs[1] as usize];
                    Interval {
                        min: a.min.min(b.min),
                        max: a.max.min(b.max),
                    }
                }
                OpCode::Max => {
                    Self::check_arity(node, 2)?;
                    let a = bounds[node.inputs[0] as usize];
                    let b = bounds[node.inputs[1] as usize];
                    Interval {
                        min: a.min.max(b.min),
                        max: a.max.max(b.max),
                    }
                }
                OpCode::Neg => {
                    Self::check_arity(node, 1)?;
                    let a = bounds[node.inputs[0] as usize];
                    Interval {
                        min: a.max.saturating_neg(),
                        max: a.min.saturating_neg(),
                    }
                }
                _ => Interval::unbounded(),
            };

            // 3. Verify embedded formal constraint
            if let Some(constraint) = &node.constraint {
                if !node_bound.satisfies(constraint) {
                    return Err(VerifyError::InvariantRefuted {
                        node: nid,
                        constraint: constraint.clone(),
                        range: node_bound,
                    });
                }
            }

            bounds.push(node_bound);
        }

        Ok(bounds)
    }

    fn check_arity(node: &Node, expected: usize) -> Result<(), VerifyError> {
        if node.inputs.len() != expected {
            Err(VerifyError::InvalidArity {
                node: node.id,
                expected,
                found: node.inputs.len(),
            })
        } else {
            Ok(())
        }
    }
}
