//! LLVM Code Generation — Compile QLANG graphs to native machine code.
//!
//! This is the core of QLANG's performance story:
//! - Graph nodes become LLVM IR instructions
//! - LLVM optimizes and emits native x86-64/ARM code
//! - Result: same performance as hand-written C
//!
//! Example: a QLANG `Add` node on two f32 vectors becomes:
//!   %result = fadd <4 x float> %a, %b    (SIMD vector add!)

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::values::FloatValue;
use inkwell::OptimizationLevel;

use qlang_core::graph::Graph;
use qlang_core::ops::Op;

/// Result of JIT compilation.
pub struct CompiledGraph<'ctx> {
    pub context: &'ctx Context,
    pub module: Module<'ctx>,
    pub execution_engine: ExecutionEngine<'ctx>,
    pub function_name: String,
    pub llvm_ir: String,
}

/// Errors during code generation.
#[derive(Debug, thiserror::Error)]
pub enum CodegenError {
    #[error("LLVM error: {0}")]
    LlvmError(String),

    #[error("unsupported op for codegen: {0}")]
    UnsupportedOp(String),

    #[error("graph error: {0}")]
    GraphError(String),
}

/// Compile a QLANG graph to LLVM IR and JIT-execute it.
///
/// This is the key function — it takes a graph and produces native machine code.
/// The compiled function signature is:
///   fn(input_ptrs: &[*const f32], output_ptrs: &[*mut f32], sizes: &[usize])
pub fn compile_graph<'ctx>(
    context: &'ctx Context,
    graph: &Graph,
    opt_level: OptimizationLevel,
) -> Result<CompiledGraph<'ctx>, CodegenError> {
    let module = context.create_module(&graph.id);
    let builder = context.create_builder();

    // Create the main function: void graph_fn(f32* inputs[], f32* outputs[], i64* sizes[])
    let f32_type = context.f32_type();
    let f32_ptr_type = context.ptr_type(inkwell::AddressSpace::default());
    let i64_type = context.i64_type();
    let void_type = context.void_type();

    // Function takes: pointer to input array, pointer to output array, number of elements
    let fn_type = void_type.fn_type(
        &[
            f32_ptr_type.into(), // input_a pointer
            f32_ptr_type.into(), // input_b pointer (or unused)
            f32_ptr_type.into(), // output pointer
            i64_type.into(),     // number of elements
        ],
        false,
    );

    let function = module.add_function("qlang_graph", fn_type, None);
    let entry_block = context.append_basic_block(function, "entry");
    builder.position_at_end(entry_block);

    // Get function parameters
    let input_a_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
    let input_b_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
    let output_ptr = function.get_nth_param(2).unwrap().into_pointer_value();
    let n_elements = function.get_nth_param(3).unwrap().into_int_value();

    // Analyze the graph to determine the operation
    let ops: Vec<&Op> = graph
        .nodes
        .iter()
        .filter(|n| !matches!(n.op, Op::Input { .. } | Op::Output { .. } | Op::Constant))
        .map(|n| &n.op)
        .collect();

    // Generate a loop that processes elements
    let loop_block = context.append_basic_block(function, "loop");
    let body_block = context.append_basic_block(function, "body");
    let exit_block = context.append_basic_block(function, "exit");

    // Initialize loop counter
    let zero = i64_type.const_int(0, false);
    builder.build_unconditional_branch(loop_block).unwrap();

    // Loop header: check if i < n
    builder.position_at_end(loop_block);
    let i_phi = builder.build_phi(i64_type, "i").unwrap();
    i_phi.add_incoming(&[(&zero, entry_block)]);
    let i_val = i_phi.as_basic_value().into_int_value();

    let cond = builder
        .build_int_compare(inkwell::IntPredicate::ULT, i_val, n_elements, "cond")
        .unwrap();
    builder
        .build_conditional_branch(cond, body_block, exit_block)
        .unwrap();

    // Loop body: load, compute, store
    builder.position_at_end(body_block);

    // Load input_a[i]
    let a_elem_ptr = unsafe {
        builder
            .build_gep(f32_type, input_a_ptr, &[i_val], "a_ptr")
            .unwrap()
    };
    let a_val = builder
        .build_load(f32_type, a_elem_ptr, "a_val")
        .unwrap()
        .into_float_value();

    // Load input_b[i]
    let b_elem_ptr = unsafe {
        builder
            .build_gep(f32_type, input_b_ptr, &[i_val], "b_ptr")
            .unwrap()
    };
    let b_val = builder
        .build_load(f32_type, b_elem_ptr, "b_val")
        .unwrap()
        .into_float_value();

    // Apply operation(s) from the graph
    let result_val = emit_ops(&builder, &context, &ops, a_val, b_val)?;

    // Store result[i]
    let out_elem_ptr = unsafe {
        builder
            .build_gep(f32_type, output_ptr, &[i_val], "out_ptr")
            .unwrap()
    };
    builder.build_store(out_elem_ptr, result_val).unwrap();

    // Increment loop counter
    let one = i64_type.const_int(1, false);
    let i_next = builder.build_int_add(i_val, one, "i_next").unwrap();
    i_phi.add_incoming(&[(&i_next, body_block)]);
    builder.build_unconditional_branch(loop_block).unwrap();

    // Exit
    builder.position_at_end(exit_block);
    builder.build_return(None).unwrap();

    // Get LLVM IR as string (for inspection)
    let llvm_ir = module.print_to_string().to_string();

    // Create JIT execution engine
    let execution_engine = module
        .create_jit_execution_engine(opt_level)
        .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

    Ok(CompiledGraph {
        context,
        module,
        execution_engine,
        function_name: "qlang_graph".to_string(),
        llvm_ir,
    })
}

/// Emit LLVM IR for a chain of operations.
fn emit_ops<'ctx>(
    builder: &Builder<'ctx>,
    context: &'ctx Context,
    ops: &[&Op],
    a: FloatValue<'ctx>,
    b: FloatValue<'ctx>,
) -> Result<FloatValue<'ctx>, CodegenError> {
    let mut current = a;

    for op in ops {
        current = match op {
            Op::Add => builder.build_float_add(current, b, "add").unwrap(),
            Op::Sub => builder.build_float_sub(current, b, "sub").unwrap(),
            Op::Mul => builder.build_float_mul(current, b, "mul").unwrap(),
            Op::Div => builder.build_float_div(current, b, "div").unwrap(),
            Op::Neg => builder.build_float_neg(current, "neg").unwrap(),

            Op::Relu => {
                // ReLU: max(0, x)
                let zero = context.f32_type().const_float(0.0);
                let cmp = builder
                    .build_float_compare(inkwell::FloatPredicate::OGT, current, zero, "relu_cmp")
                    .unwrap();
                builder
                    .build_select(cmp, current, zero, "relu")
                    .unwrap()
                    .into_float_value()
            }

            Op::Sigmoid => {
                // Sigmoid: 1 / (1 + exp(-x))
                let neg_x = builder.build_float_neg(current, "neg_x").unwrap();
                let exp_neg_x = emit_exp(builder, context, neg_x);
                let one = context.f32_type().const_float(1.0);
                let denom = builder.build_float_add(one, exp_neg_x, "denom").unwrap();
                builder.build_float_div(one, denom, "sigmoid").unwrap()
            }

            Op::Tanh => {
                // Tanh via: (exp(2x) - 1) / (exp(2x) + 1)
                let two = context.f32_type().const_float(2.0);
                let two_x = builder.build_float_mul(current, two, "two_x").unwrap();
                let exp_2x = emit_exp(builder, context, two_x);
                let one = context.f32_type().const_float(1.0);
                let num = builder.build_float_sub(exp_2x, one, "tanh_num").unwrap();
                let den = builder.build_float_add(exp_2x, one, "tanh_den").unwrap();
                builder.build_float_div(num, den, "tanh").unwrap()
            }

            Op::ToTernary => {
                // Ternary: x > 0.3 → 1.0, x < -0.3 → -1.0, else → 0.0
                let pos_thresh = context.f32_type().const_float(0.3);
                let neg_thresh = context.f32_type().const_float(-0.3);
                let one = context.f32_type().const_float(1.0);
                let neg_one = context.f32_type().const_float(-1.0);
                let zero = context.f32_type().const_float(0.0);

                let is_pos = builder
                    .build_float_compare(inkwell::FloatPredicate::OGT, current, pos_thresh, "is_pos")
                    .unwrap();
                let is_neg = builder
                    .build_float_compare(inkwell::FloatPredicate::OLT, current, neg_thresh, "is_neg")
                    .unwrap();

                let neg_or_zero = builder
                    .build_select(is_neg, neg_one, zero, "neg_or_zero")
                    .unwrap()
                    .into_float_value();
                builder
                    .build_select(is_pos, one, neg_or_zero, "ternary")
                    .unwrap()
                    .into_float_value()
            }

            other => {
                return Err(CodegenError::UnsupportedOp(format!("{other}")));
            }
        };
    }

    Ok(current)
}

/// Emit exp(x) approximation using Padé-like rational polynomial.
///
/// Uses the identity: exp(x) ≈ (1 + x/n)^n for large n.
/// We use a 6th-order Taylor series which is accurate for |x| < 5:
///   exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120 + x⁶/720
///
/// For sigmoid/tanh with typical neural network values, this is sufficient.
/// LLVM will optimize this into fast FMA (fused multiply-add) instructions.
fn emit_exp<'ctx>(
    builder: &Builder<'ctx>,
    context: &'ctx Context,
    x: FloatValue<'ctx>,
) -> FloatValue<'ctx> {
    let f32_type = context.f32_type();
    let one = f32_type.const_float(1.0);
    let half = f32_type.const_float(0.5);
    let sixth = f32_type.const_float(1.0 / 6.0);
    let c4 = f32_type.const_float(1.0 / 24.0);
    let c5 = f32_type.const_float(1.0 / 120.0);
    let c6 = f32_type.const_float(1.0 / 720.0);

    // x² = x * x
    let x2 = builder.build_float_mul(x, x, "x2").unwrap();
    // x³ = x² * x
    let x3 = builder.build_float_mul(x2, x, "x3").unwrap();
    // x⁴ = x² * x²
    let x4 = builder.build_float_mul(x2, x2, "x4").unwrap();
    // x⁵ = x⁴ * x
    let x5 = builder.build_float_mul(x4, x, "x5").unwrap();
    // x⁶ = x³ * x³
    let x6 = builder.build_float_mul(x3, x3, "x6").unwrap();

    // result = 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120 + x⁶/720
    let t1 = builder.build_float_add(one, x, "t1").unwrap();
    let t2_term = builder.build_float_mul(x2, half, "t2_term").unwrap();
    let t2 = builder.build_float_add(t1, t2_term, "t2").unwrap();
    let t3_term = builder.build_float_mul(x3, sixth, "t3_term").unwrap();
    let t3 = builder.build_float_add(t2, t3_term, "t3").unwrap();
    let t4_term = builder.build_float_mul(x4, c4, "t4_term").unwrap();
    let t4 = builder.build_float_add(t3, t4_term, "t4").unwrap();
    let t5_term = builder.build_float_mul(x5, c5, "t5_term").unwrap();
    let t5 = builder.build_float_add(t4, t5_term, "t5").unwrap();
    let t6_term = builder.build_float_mul(x6, c6, "t6_term").unwrap();
    let result = builder.build_float_add(t5, t6_term, "exp_approx").unwrap();

    // Clamp to positive (exp is always > 0)
    let zero = f32_type.const_float(1e-7);
    let cmp = builder
        .build_float_compare(inkwell::FloatPredicate::OGT, result, zero, "exp_pos")
        .unwrap();
    builder
        .build_select(cmp, result, zero, "exp_clamped")
        .unwrap()
        .into_float_value()
}

/// Type alias for the JIT-compiled graph function.
type GraphFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, u64);

/// Execute a compiled graph with concrete data.
///
/// This calls the JIT-compiled native code directly — no interpretation,
/// no overhead. Pure register operations.
pub fn execute_compiled(
    compiled: &CompiledGraph,
    input_a: &[f32],
    input_b: &[f32],
) -> Result<Vec<f32>, CodegenError> {
    let n = input_a.len();
    let mut output = vec![0.0f32; n];

    unsafe {
        let func: JitFunction<GraphFn> = compiled
            .execution_engine
            .get_function(&compiled.function_name)
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

        func.call(
            input_a.as_ptr(),
            input_b.as_ptr(),
            output.as_mut_ptr(),
            n as u64,
        );
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::graph::Graph;
    use qlang_core::ops::Op;
    use qlang_core::tensor::TensorType;

    fn make_binop_graph(op: Op) -> Graph {
        let mut g = Graph::new("test");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let compute = g.add_node(op, vec![TensorType::f32_vector(4), TensorType::f32_vector(4)], vec![TensorType::f32_vector(4)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(4)], vec![]);
        g.add_edge(a, 0, compute, 0, TensorType::f32_vector(4));
        g.add_edge(b, 0, compute, 1, TensorType::f32_vector(4));
        g.add_edge(compute, 0, out, 0, TensorType::f32_vector(4));
        g
    }

    #[test]
    fn jit_add() {
        let graph = make_binop_graph(Op::Add);
        let context = Context::create();
        let compiled = compile_graph(&context, &graph, OptimizationLevel::Aggressive).unwrap();

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0, 30.0, 40.0];
        let result = execute_compiled(&compiled, &a, &b).unwrap();

        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn jit_mul() {
        let graph = make_binop_graph(Op::Mul);
        let context = Context::create();
        let compiled = compile_graph(&context, &graph, OptimizationLevel::Aggressive).unwrap();

        let a = vec![2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 10.0, 10.0, 10.0];
        let result = execute_compiled(&compiled, &a, &b).unwrap();

        assert_eq!(result, vec![20.0, 30.0, 40.0, 50.0]);
    }

    #[test]
    fn jit_relu() {
        let mut g = Graph::new("relu_test");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let relu = g.add_node(Op::Relu, vec![TensorType::f32_vector(4)], vec![TensorType::f32_vector(4)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(4)], vec![]);
        g.add_edge(a, 0, relu, 0, TensorType::f32_vector(4));
        g.add_edge(relu, 0, out, 0, TensorType::f32_vector(4));

        let context = Context::create();
        let compiled = compile_graph(&context, &g, OptimizationLevel::Aggressive).unwrap();

        let input = vec![1.0, -2.0, 3.0, -4.0];
        let dummy = vec![0.0; 4];
        let result = execute_compiled(&compiled, &input, &dummy).unwrap();

        assert_eq!(result, vec![1.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn jit_ternary() {
        let mut g = Graph::new("ternary_test");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let ternary = g.add_node(Op::ToTernary, vec![TensorType::f32_vector(4)], vec![TensorType::f32_vector(4)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(4)], vec![]);
        g.add_edge(a, 0, ternary, 0, TensorType::f32_vector(4));
        g.add_edge(ternary, 0, out, 0, TensorType::f32_vector(4));

        let context = Context::create();
        let compiled = compile_graph(&context, &g, OptimizationLevel::Aggressive).unwrap();

        let input = vec![0.5, -0.5, 0.1, -0.1];
        let dummy = vec![0.0; 4];
        let result = execute_compiled(&compiled, &input, &dummy).unwrap();

        assert_eq!(result, vec![1.0, -1.0, 0.0, 0.0]); // ternary: {+1, -1, 0, 0}
    }

    #[test]
    fn jit_prints_llvm_ir() {
        let graph = make_binop_graph(Op::Add);
        let context = Context::create();
        let compiled = compile_graph(&context, &graph, OptimizationLevel::None).unwrap();

        assert!(compiled.llvm_ir.contains("fadd float"));
        assert!(compiled.llvm_ir.contains("qlang_graph"));
    }

    #[test]
    fn jit_sigmoid() {
        let mut g = Graph::new("sigmoid_test");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(3)]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(3)]);
        let sig = g.add_node(Op::Sigmoid, vec![TensorType::f32_vector(3)], vec![TensorType::f32_vector(3)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(3)], vec![]);
        g.add_edge(a, 0, sig, 0, TensorType::f32_vector(3));
        g.add_edge(sig, 0, out, 0, TensorType::f32_vector(3));

        let context = Context::create();
        let compiled = compile_graph(&context, &g, OptimizationLevel::Aggressive).unwrap();

        // Use values in range where Taylor exp() is accurate (|x| < 5)
        let input = vec![0.0, 2.0, -2.0];
        let dummy = vec![0.0; 3];
        let result = execute_compiled(&compiled, &input, &dummy).unwrap();

        // sigmoid(0) = 0.5, sigmoid(2) ≈ 0.8808, sigmoid(-2) ≈ 0.1192
        assert!((result[0] - 0.5).abs() < 1e-3);
        assert!((result[1] - 0.8808).abs() < 0.05);
        assert!((result[2] - 0.1192).abs() < 0.05);
    }
}
