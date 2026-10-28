//! Bidirectional Human SSA Parser.
//!
//! Parses decompiled human-readable SSA pseudo-code back into a machine-native
//! dense binary DAG (`Graph`), completing the round-trip isomorphism.

use crate::dag::{Constraint, DataType, Graph, OpCode};

#[derive(Debug, PartialEq, Eq)]
pub enum ParseError {
    InvalidHeader(String),
    InvalidNode(String),
    MissingReturn,
    UnknownOp(String),
}

pub struct Parser;

impl Parser {
    pub fn parse(source: &str) -> Result<Graph, ParseError> {
        let lines: Vec<&str> = source.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
        if lines.is_empty() {
            return Err(ParseError::InvalidHeader("Empty source".to_string()));
        }

        // 1. Parse signature: fn name(arg0: I64, ...) -> RetType {
        let header = lines[0];
        let (name, input_types, return_type) = Self::parse_header(header)?;

        let mut graph = Graph::new(&name, input_types, return_type);
        let mut return_node: Option<u32> = None;

        // 2. Parse body lines
        for line in &lines[1..] {
            if *line == "}" {
                continue;
            }

            if line.starts_with("return ") {
                let ret_part = line.trim_start_matches("return ").trim();
                let nid = Self::parse_vid(ret_part)?;
                return_node = Some(nid);
                continue;
            }

            // Line format: v04 = expr   # invariant: ...
            Self::parse_node_line(line, &mut graph)?;
        }

        if let Some(rn) = return_node {
            graph.return_node = rn;
            Ok(graph)
        } else {
            Err(ParseError::MissingReturn)
        }
    }

    fn parse_header(line: &str) -> Result<(String, Vec<DataType>, DataType), ParseError> {
        // e.g.: fn calculate_discount(arg0: I64, arg1: I64) -> I64 {
        if !line.starts_with("fn ") {
            return Err(ParseError::InvalidHeader("Expected 'fn '".to_string()));
        }
        let open_paren = line.find('(').ok_or_else(|| ParseError::InvalidHeader("Missing '('".to_string()))?;
        let close_paren = line.find(')').ok_or_else(|| ParseError::InvalidHeader("Missing ')'".to_string()))?;

        let name = line[3..open_paren].trim().to_string();
        let args_str = &line[open_paren + 1..close_paren];

        let mut input_types = Vec::new();
        if !args_str.trim().is_empty() {
            for arg in args_str.split(',') {
                let parts: Vec<&str> = arg.split(':').collect();
                if parts.len() == 2 {
                    input_types.push(Self::parse_data_type(parts[1].trim())?);
                }
            }
        }

        let ret_str = if let Some(arrow) = line.find("->") {
            let after_arrow = &line[arrow + 2..];
            let brace = after_arrow.find('{').unwrap_or(after_arrow.len());
            after_arrow[..brace].trim()
        } else {
            "I64"
        };
        let return_type = Self::parse_data_type(ret_str)?;

        Ok((name, input_types, return_type))
    }

    fn parse_data_type(s: &str) -> Result<DataType, ParseError> {
        match s {
            "I64" => Ok(DataType::I64),
            "F64" => Ok(DataType::F64),
            "Bool" => Ok(DataType::Bool),
            "Tensor1D" => Ok(DataType::Tensor1D),
            _ => Err(ParseError::InvalidHeader(format!("Unknown DataType: {s}"))),
        }
    }

    fn parse_node_line(line: &str, graph: &mut Graph) -> Result<(), ParseError> {
        // Example: v04 = clamp(v01, v02, v03)          # invariant: [0, 100]
        let (code_part, comment_part) = if let Some(pos) = line.find('#') {
            (&line[..pos], Some(&line[pos..]))
        } else {
            (line, None)
        };

        let equals = code_part.find('=').ok_or_else(|| ParseError::InvalidNode(line.to_string()))?;
        let vid_str = code_part[..equals].trim();
        let expected_id = Self::parse_vid(vid_str)?;
        let expr = code_part[equals + 1..].trim();

        // Invariant constraint parsing
        let constraint = if let Some(cmt) = comment_part {
            Self::parse_constraint(cmt)
        } else {
            None
        };

        // Parse expression
        let (op, inputs, ty) = Self::parse_expr(expr)?;
        let actual_id = graph.add_node(op, inputs, ty, constraint);
        if actual_id != expected_id {
            return Err(ParseError::InvalidNode(format!("Node id mismatch: expected v{expected_id:02}, got v{actual_id:02}")));
        }

        Ok(())
    }

    fn parse_constraint(comment: &str) -> Option<Constraint> {
        let cmt = comment.trim().trim_start_matches('#').trim();
        if !cmt.starts_with("invariant:") {
            return None;
        }
        let inv = cmt.trim_start_matches("invariant:").trim();
        if inv == ">= 0" {
            Some(Constraint::NonNegative)
        } else if inv == "> 0" {
            Some(Constraint::StrictlyPositive)
        } else if inv == "!= 0" {
            Some(Constraint::NonZero)
        } else if inv.starts_with('[') && inv.ends_with(']') {
            let inner = &inv[1..inv.len() - 1];
            let parts: Vec<&str> = inner.split(',').collect();
            if parts.len() == 2 {
                if let (Ok(l), Ok(h)) = (parts[0].trim().parse::<i64>(), parts[1].trim().parse::<i64>()) {
                    return Some(Constraint::BoundedRange(l, h));
                }
            }
            None
        } else {
            None
        }
    }

    fn parse_expr(expr: &str) -> Result<(OpCode, Vec<u32>, DataType), ParseError> {
        // argN
        if expr.starts_with("arg") {
            if let Ok(slot) = expr[3..].parse::<u32>() {
                return Ok((OpCode::Param(slot), vec![], DataType::I64));
            }
        }

        // Integer literal
        if let Ok(v) = expr.parse::<i64>() {
            return Ok((OpCode::ConstI64(v), vec![], DataType::I64));
        }

        // Float literal
        if let Ok(v) = expr.parse::<f64>() {
            return Ok((OpCode::ConstF64(v), vec![], DataType::F64));
        }

        // Bool literal
        if expr == "true" {
            return Ok((OpCode::ConstBool(true), vec![], DataType::Bool));
        }
        if expr == "false" {
            return Ok((OpCode::ConstBool(false), vec![], DataType::Bool));
        }

        // clamp(v01, v02, v03)
        if expr.starts_with("clamp(") && expr.ends_with(')') {
            let inner = &expr[6..expr.len() - 1];
            let args = Self::parse_vids(inner)?;
            if args.len() == 3 {
                return Ok((OpCode::Clamp, args, DataType::I64));
            }
        }

        // abs(v01)
        if expr.starts_with("abs(") && expr.ends_with(')') {
            let inner = &expr[4..expr.len() - 1];
            let args = Self::parse_vids(inner)?;
            if args.len() == 1 {
                return Ok((OpCode::Abs, args, DataType::I64));
            }
        }

        // min(v01, v02)
        if expr.starts_with("min(") && expr.ends_with(')') {
            let inner = &expr[4..expr.len() - 1];
            let args = Self::parse_vids(inner)?;
            if args.len() == 2 {
                return Ok((OpCode::Min, args, DataType::I64));
            }
        }

        // max(v01, v02)
        if expr.starts_with("max(") && expr.ends_with(')') {
            let inner = &expr[4..expr.len() - 1];
            let args = Self::parse_vids(inner)?;
            if args.len() == 2 {
                return Ok((OpCode::Max, args, DataType::I64));
            }
        }

        // tensor_map(v00, +0.5)
        if expr.starts_with("tensor_map(") && expr.ends_with(')') {
            let inner = &expr[11..expr.len() - 1];
            let parts: Vec<&str> = inner.split(',').collect();
            if parts.len() == 2 {
                let vid = Self::parse_vid(parts[0].trim())?;
                let bias_str = parts[1].trim().trim_start_matches('+');
                let bias: f64 = bias_str.parse().map_err(|_| ParseError::UnknownOp(expr.to_string()))?;
                return Ok((OpCode::TensorMapAdd(bias), vec![vid], DataType::Tensor1D));
            }
        }

        // tensor_sum(v01)
        if expr.starts_with("tensor_sum(") && expr.ends_with(')') {
            let inner = &expr[11..expr.len() - 1];
            let args = Self::parse_vids(inner)?;
            if args.len() == 1 {
                return Ok((OpCode::TensorSum, args, DataType::F64));
            }
        }

        // Binary operations: vXX OP vYY
        let tokens: Vec<&str> = expr.split_whitespace().collect();
        if tokens.len() == 3 {
            let left = Self::parse_vid(tokens[0])?;
            let op_str = tokens[1];
            let right = Self::parse_vid(tokens[2])?;
            let inputs = vec![left, right];

            let (opcode, ty) = match op_str {
                "+" => (OpCode::Add, DataType::I64),
                "-" => (OpCode::Sub, DataType::I64),
                "*" => (OpCode::Mul, DataType::I64),
                "/" => (OpCode::Div, DataType::I64),
                "%" => (OpCode::Mod, DataType::I64),
                "==" => (OpCode::CmpEq, DataType::Bool),
                "<" => (OpCode::CmpLt, DataType::Bool),
                "<=" => (OpCode::CmpLe, DataType::Bool),
                ">" => (OpCode::CmpGt, DataType::Bool),
                ">=" => (OpCode::CmpGe, DataType::Bool),
                _ => return Err(ParseError::UnknownOp(expr.to_string())),
            };
            return Ok((opcode, inputs, ty));
        }

        // Negation: -vXX
        if expr.starts_with("-v") {
            let vid = Self::parse_vid(&expr[1..])?;
            return Ok((OpCode::Neg, vec![vid], DataType::I64));
        }

        Err(ParseError::UnknownOp(expr.to_string()))
    }

    fn parse_vid(s: &str) -> Result<u32, ParseError> {
        let trimmed = s.trim();
        if trimmed.starts_with('v') {
            trimmed[1..]
                .parse::<u32>()
                .map_err(|_| ParseError::InvalidNode(format!("Invalid vid: {s}")))
        } else {
            Err(ParseError::InvalidNode(format!("Expected 'v' prefix: {s}")))
        }
    }

    fn parse_vids(s: &str) -> Result<Vec<u32>, ParseError> {
        let mut list = Vec::new();
        for item in s.split(',') {
            list.push(Self::parse_vid(item.trim())?);
        }
        Ok(list)
    }
}
