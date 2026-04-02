//! LLVM JIT code generation for matrix multiplication.
//!
//! Generates a triple-loop matmul:
//!   for i in 0..M:
//!     for j in 0..N:
//!       sum = 0
//!       for k in 0..K:
//!         sum += A[i*K+k] * B[k*N+j]
//!       C[i*N+j] = sum

use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
use inkwell::OptimizationLevel;

use crate::codegen::CodegenError;

/// JIT-compiled matrix multiplication.
pub struct CompiledMatMul<'ctx> {
    _context: &'ctx Context,
    _module: inkwell::module::Module<'ctx>,
    execution_engine: inkwell::execution_engine::ExecutionEngine<'ctx>,
    pub llvm_ir: String,
}

type MatMulFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, u64, u64, u64);

/// Compile a matrix multiplication kernel via LLVM.
pub fn compile_matmul(context: &Context) -> Result<CompiledMatMul, CodegenError> {
    let module = context.create_module("qlang_matmul");
    let builder = context.create_builder();

    let f32_type = context.f32_type();
    let f32_ptr = context.ptr_type(inkwell::AddressSpace::default());
    let i64_type = context.i64_type();
    let void_type = context.void_type();

    let fn_type = void_type.fn_type(
        &[f32_ptr.into(), f32_ptr.into(), f32_ptr.into(),
          i64_type.into(), i64_type.into(), i64_type.into()],
        false,
    );

    let function = module.add_function("qlang_matmul", fn_type, None);

    // Blocks
    let entry = context.append_basic_block(function, "entry");
    let loop_i_header = context.append_basic_block(function, "loop_i");
    let loop_j_header = context.append_basic_block(function, "loop_j");
    let loop_k_header = context.append_basic_block(function, "loop_k");
    let k_body = context.append_basic_block(function, "k_body");
    let k_done = context.append_basic_block(function, "k_done");
    let j_inc = context.append_basic_block(function, "j_inc");
    let i_inc = context.append_basic_block(function, "i_inc");
    let exit = context.append_basic_block(function, "exit");

    let a_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
    let b_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
    let c_ptr = function.get_nth_param(2).unwrap().into_pointer_value();
    let m_param = function.get_nth_param(3).unwrap().into_int_value();
    let k_param = function.get_nth_param(4).unwrap().into_int_value();
    let n_param = function.get_nth_param(5).unwrap().into_int_value();

    let zero_i64 = i64_type.const_int(0, false);
    let one_i64 = i64_type.const_int(1, false);
    let zero_f32 = f32_type.const_float(0.0);

    // entry → loop_i
    builder.position_at_end(entry);
    builder.build_unconditional_branch(loop_i_header).unwrap();

    // loop_i: i = phi(0, i+1); if i < M goto loop_j else exit
    builder.position_at_end(loop_i_header);
    let i_phi = builder.build_phi(i64_type, "i").unwrap();
    i_phi.add_incoming(&[(&zero_i64, entry)]);
    let i_val = i_phi.as_basic_value().into_int_value();
    let cond_i = builder.build_int_compare(inkwell::IntPredicate::ULT, i_val, m_param, "ci").unwrap();
    builder.build_conditional_branch(cond_i, loop_j_header, exit).unwrap();

    // loop_j: j = phi(0, j+1); if j < N goto loop_k else i_inc
    builder.position_at_end(loop_j_header);
    let j_phi = builder.build_phi(i64_type, "j").unwrap();
    j_phi.add_incoming(&[(&zero_i64, loop_i_header)]);
    let j_val = j_phi.as_basic_value().into_int_value();
    let cond_j = builder.build_int_compare(inkwell::IntPredicate::ULT, j_val, n_param, "cj").unwrap();
    builder.build_conditional_branch(cond_j, loop_k_header, i_inc).unwrap();

    // loop_k: k = phi(0, k+1), sum = phi(0.0, new_sum); if k < K goto k_body else k_done
    builder.position_at_end(loop_k_header);
    let k_phi = builder.build_phi(i64_type, "k").unwrap();
    k_phi.add_incoming(&[(&zero_i64, loop_j_header)]);
    let sum_phi = builder.build_phi(f32_type, "sum").unwrap();
    sum_phi.add_incoming(&[(&zero_f32, loop_j_header)]);
    let k_val = k_phi.as_basic_value().into_int_value();
    let sum_val = sum_phi.as_basic_value().into_float_value();
    let cond_k = builder.build_int_compare(inkwell::IntPredicate::ULT, k_val, k_param, "ck").unwrap();
    builder.build_conditional_branch(cond_k, k_body, k_done).unwrap();

    // k_body: sum += A[i*K+k] * B[k*N+j]; k++; goto loop_k
    builder.position_at_end(k_body);
    let ik = builder.build_int_mul(i_val, k_param, "ik").unwrap();
    let a_idx = builder.build_int_add(ik, k_val, "aidx").unwrap();
    let a_gep = unsafe { builder.build_gep(f32_type, a_ptr, &[a_idx], "agep").unwrap() };
    let a_val = builder.build_load(f32_type, a_gep, "av").unwrap().into_float_value();

    let kn = builder.build_int_mul(k_val, n_param, "kn").unwrap();
    let b_idx = builder.build_int_add(kn, j_val, "bidx").unwrap();
    let b_gep = unsafe { builder.build_gep(f32_type, b_ptr, &[b_idx], "bgep").unwrap() };
    let b_val = builder.build_load(f32_type, b_gep, "bv").unwrap().into_float_value();

    let prod = builder.build_float_mul(a_val, b_val, "prod").unwrap();
    let new_sum = builder.build_float_add(sum_val, prod, "nsum").unwrap();
    let k_next = builder.build_int_add(k_val, one_i64, "knxt").unwrap();

    k_phi.add_incoming(&[(&k_next, k_body)]);
    sum_phi.add_incoming(&[(&new_sum, k_body)]);
    builder.build_unconditional_branch(loop_k_header).unwrap();

    // k_done: store C[i*N+j] = sum; goto j_inc
    builder.position_at_end(k_done);
    let in_val = builder.build_int_mul(i_val, n_param, "in").unwrap();
    let c_idx = builder.build_int_add(in_val, j_val, "cidx").unwrap();
    let c_gep = unsafe { builder.build_gep(f32_type, c_ptr, &[c_idx], "cgep").unwrap() };
    builder.build_store(c_gep, sum_val).unwrap();
    builder.build_unconditional_branch(j_inc).unwrap();

    // j_inc: j++; goto loop_j
    builder.position_at_end(j_inc);
    let j_next = builder.build_int_add(j_val, one_i64, "jnxt").unwrap();
    j_phi.add_incoming(&[(&j_next, j_inc)]);
    builder.build_unconditional_branch(loop_j_header).unwrap();

    // i_inc: i++; goto loop_i
    builder.position_at_end(i_inc);
    let i_next = builder.build_int_add(i_val, one_i64, "inxt").unwrap();
    i_phi.add_incoming(&[(&i_next, i_inc)]);
    builder.build_unconditional_branch(loop_i_header).unwrap();

    // exit
    builder.position_at_end(exit);
    builder.build_return(None).unwrap();

    let llvm_ir = module.print_to_string().to_string();
    let execution_engine = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

    Ok(CompiledMatMul {
        _context: context,
        _module: module,
        execution_engine,
        llvm_ir,
    })
}

/// Execute JIT-compiled matmul: C[m×n] = A[m×k] × B[k×n]
pub fn execute_matmul(
    compiled: &CompiledMatMul,
    a: &[f32], b: &[f32],
    m: usize, k: usize, n: usize,
) -> Result<Vec<f32>, CodegenError> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut c = vec![0.0f32; m * n];

    unsafe {
        let func: JitFunction<MatMulFn> = compiled.execution_engine
            .get_function("qlang_matmul")
            .map_err(|e| CodegenError::LlvmError(e.to_string()))?;
        func.call(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m as u64, k as u64, n as u64);
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jit_matmul_2x3_times_3x2() {
        let context = Context::create();
        let compiled = compile_matmul(&context).unwrap();

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let c = execute_matmul(&compiled, &a, &b, 2, 3, 2).unwrap();
        assert_eq!(c, vec![4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn jit_matmul_1x1() {
        let context = Context::create();
        let compiled = compile_matmul(&context).unwrap();
        let c = execute_matmul(&compiled, &[3.0], &[5.0], 1, 1, 1).unwrap();
        assert_eq!(c, vec![15.0]);
    }

    #[test]
    fn jit_matmul_identity() {
        let context = Context::create();
        let compiled = compile_matmul(&context).unwrap();

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let id = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let c = execute_matmul(&compiled, &a, &id, 2, 3, 3).unwrap();
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn jit_matmul_large_correct() {
        let context = Context::create();
        let compiled = compile_matmul(&context).unwrap();

        let m = 32;
        let k = 64;
        let n = 16;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.007).cos()).collect();

        let c = execute_matmul(&compiled, &a, &b, m, k, n).unwrap();

        // Verify against naive
        for i in 0..m {
            for j in 0..n {
                let mut expected = 0.0f32;
                for p in 0..k {
                    expected += a[i * k + p] * b[p * n + j];
                }
                assert!((c[i * n + j] - expected).abs() < 1e-3,
                    "Mismatch at [{i},{j}]");
            }
        }
    }

    #[test]
    fn jit_matmul_ir_contains_fmul() {
        let context = Context::create();
        let compiled = compile_matmul(&context).unwrap();
        assert!(compiled.llvm_ir.contains("qlang_matmul"));
        assert!(compiled.llvm_ir.contains("fmul"));
    }
}
