use tgs_engine::dag::{Constraint, DataType, Graph, OpCode};
use tgs_engine::engine::{Engine, Value};
use tgs_engine::projection::Projection;
use tgs_engine::verifier::{Interval, Verifier, VerifyError};

#[test]
fn test_dense_dag_serialization() {
    let mut g = Graph::new("simple_add", vec![DataType::I64, DataType::I64], DataType::I64);
    let p0 = g.add_node(OpCode::Param(0), vec![], DataType::I64, None);
    let p1 = g.add_node(OpCode::Param(1), vec![], DataType::I64, None);
    let res = g.add_node(OpCode::Add, vec![p0, p1], DataType::I64, Some(Constraint::NonNegative));
    g.return_node = res;

    let dense = g.to_dense_bytes();
    assert!(!dense.is_empty());
    assert_eq!(g.dense_token_count(), 3);
}

#[test]
fn test_cycle_detection() {
    let mut g = Graph::new("invalid_cycle", vec![DataType::I64], DataType::I64);
    // Node referencing future node 2
    g.add_node(OpCode::Add, vec![2, 0], DataType::I64, None);
    let err = Verifier::verify(&g, &[Interval::unbounded()]);
    assert!(matches!(err, Err(VerifyError::CyclicOrFutureReference { .. })));
}

#[test]
fn test_zero_division_detection() {
    let mut g = Graph::new("div_test", vec![DataType::I64, DataType::I64], DataType::I64);
    let p0 = g.add_node(OpCode::Param(0), vec![], DataType::I64, None);
    let p1 = g.add_node(OpCode::Param(1), vec![], DataType::I64, None);
    g.add_node(OpCode::Div, vec![p0, p1], DataType::I64, None);

    // Divisor interval [-5, 5] includes 0
    let bounds = vec![
        Interval { min: 10, max: 20 },
        Interval { min: -5, max: 5 },
    ];
    let err = Verifier::verify(&g, &bounds);
    assert!(matches!(err, Err(VerifyError::PotentialZeroDivision { .. })));
}

#[test]
fn test_invariant_refutation() {
    let mut g = Graph::new("constraint_fail", vec![DataType::I64], DataType::I64);
    let p0 = g.add_node(OpCode::Param(0), vec![], DataType::I64, None);
    let c_minus5 = g.add_node(OpCode::ConstI64(-5), vec![], DataType::I64, None);
    // Add x + (-5) with constraint NonNegative
    g.add_node(OpCode::Add, vec![p0, c_minus5], DataType::I64, Some(Constraint::NonNegative));

    // Input x in [0, 2] -> result in [-5, -3] which violates NonNegative!
    let bounds = vec![Interval { min: 0, max: 2 }];
    let err = Verifier::verify(&g, &bounds);
    assert!(matches!(err, Err(VerifyError::InvariantRefuted { .. })));
}

#[test]
fn test_dvm_execution() {
    let mut g = Graph::new("calc", vec![DataType::I64, DataType::I64], DataType::I64);
    let p0 = g.add_node(OpCode::Param(0), vec![], DataType::I64, None);
    let p1 = g.add_node(OpCode::Param(1), vec![], DataType::I64, None);
    let mul = g.add_node(OpCode::Mul, vec![p0, p1], DataType::I64, None);
    let c10 = g.add_node(OpCode::ConstI64(10), vec![], DataType::I64, None);
    let add = g.add_node(OpCode::Add, vec![mul, c10], DataType::I64, None);
    g.return_node = add;

    let res = Engine::execute(&g, &[Value::I64(7), Value::I64(6)]).unwrap();
    assert_eq!(res, Value::I64(52)); // 7 * 6 + 10 = 52
}

#[test]
fn test_tensor_simd_execution() {
    let mut g = Graph::new("tensor_op", vec![DataType::Tensor1D], DataType::F64);
    let p0 = g.add_node(OpCode::Param(0), vec![], DataType::Tensor1D, None);
    let add_const = g.add_node(OpCode::TensorMapAdd(2.0), vec![p0], DataType::Tensor1D, None);
    let sum = g.add_node(OpCode::TensorSum, vec![add_const], DataType::F64, None);
    g.return_node = sum;

    // [10.0, 20.0, 30.0] -> [12.0, 22.0, 32.0] -> sum = 66.0
    let res = Engine::execute(&g, &[Value::Tensor(vec![10.0, 20.0, 30.0])]).unwrap();
    assert_eq!(res, Value::F64(66.0));
}

#[test]
fn test_projection_decompile() {
    let mut g = Graph::new("demo", vec![DataType::I64], DataType::I64);
    let p0 = g.add_node(OpCode::Param(0), vec![], DataType::I64, None);
    let c2 = g.add_node(OpCode::ConstI64(2), vec![], DataType::I64, None);
    let mul = g.add_node(OpCode::Mul, vec![p0, c2], DataType::I64, Some(Constraint::NonNegative));
    g.return_node = mul;

    let code = Projection::decompile(&g);
    assert!(code.contains("fn demo(arg0: I64) -> I64"));
    assert!(code.contains("v02 = v00 * v01"));
    assert!(code.contains("# invariant: >= 0"));

    let visual = Projection::visual_graph(&g);
    assert!(visual.contains("[GEN-3 DATAFLOW DAG: demo]"));
}

#[test]
fn test_optimizer_reductions() {
    use tgs_engine::optimizer::Optimizer;

    let mut g = Graph::new("redundant", vec![DataType::I64], DataType::I64);
    let p0 = g.add_node(OpCode::Param(0), vec![], DataType::I64, None);
    let c10 = g.add_node(OpCode::ConstI64(10), vec![], DataType::I64, None);
    let c20 = g.add_node(OpCode::ConstI64(20), vec![], DataType::I64, None);
    // Constant fold: 10 + 20 -> 30
    let c_add = g.add_node(OpCode::Add, vec![c10, c20], DataType::I64, None);

    // Algebraic identity: p0 * 1 -> p0
    let c1 = g.add_node(OpCode::ConstI64(1), vec![], DataType::I64, None);
    let mul1 = g.add_node(OpCode::Mul, vec![p0, c1], DataType::I64, None);

    // Dead node
    let dead_c = g.add_node(OpCode::ConstI64(888), vec![], DataType::I64, None);
    let _dead = g.add_node(OpCode::Add, vec![dead_c, dead_c], DataType::I64, None);

    // Final result: (p0 * 1) + (10 + 20)
    let ret = g.add_node(OpCode::Add, vec![mul1, c_add], DataType::I64, None);
    g.return_node = ret;

    let (opt_g, stats) = Optimizer::optimize(&g);
    assert!(stats.constants_folded >= 1);
    assert!(stats.algebraic_reductions >= 1);
    assert!(stats.dead_nodes_removed >= 1);
    assert!(opt_g.nodes.len() < g.nodes.len());

    // Check numerical equivalence
    let r1 = Engine::execute(&g, &[Value::I64(5)]).unwrap();
    let r2 = Engine::execute(&opt_g, &[Value::I64(5)]).unwrap();
    assert_eq!(r1, r2);
}

#[test]
fn test_parser_and_roundtrip_isomorphism() {
    use tgs_engine::parser::Parser;

    let mut g = Graph::new("calc_price", vec![DataType::I64, DataType::I64], DataType::I64);
    let p0 = g.add_node(OpCode::Param(0), vec![], DataType::I64, None);
    let p1 = g.add_node(OpCode::Param(1), vec![], DataType::I64, None);
    let c0 = g.add_node(OpCode::ConstI64(0), vec![], DataType::I64, None);
    let c100 = g.add_node(OpCode::ConstI64(100), vec![], DataType::I64, None);
    let clamped = g.add_node(OpCode::Clamp, vec![p1, c0, c100], DataType::I64, Some(Constraint::BoundedRange(0, 100)));
    let rem = g.add_node(OpCode::Sub, vec![c100, clamped], DataType::I64, Some(Constraint::BoundedRange(0, 100)));
    let mul = g.add_node(OpCode::Mul, vec![p0, rem], DataType::I64, Some(Constraint::NonNegative));
    let final_price = g.add_node(OpCode::Div, vec![mul, c100], DataType::I64, Some(Constraint::NonNegative));
    g.return_node = final_price;

    let ssa = Projection::decompile(&g);
    let parsed = Parser::parse(&ssa).expect("Parsing SSA should succeed");

    assert_eq!(parsed.nodes.len(), g.nodes.len());
    assert_eq!(parsed.name, g.name);

    let args = [Value::I64(300), Value::I64(10)];
    let res_orig = Engine::execute(&g, &args).unwrap();
    let res_parsed = Engine::execute(&parsed, &args).unwrap();
    assert_eq!(res_orig, res_parsed);
    assert_eq!(res_orig, Value::I64(270));
}

#[test]
fn test_arm64_jit_execution() {
    use tgs_engine::jit::JitEngine;

    if !JitEngine::is_supported() {
        eprintln!("Skipping JIT test: not running on macOS ARM64");
        return;
    }

    let mut g = Graph::new("calculate_discount", vec![DataType::I64, DataType::I64], DataType::I64);
    let p_price = g.add_node(OpCode::Param(0), vec![], DataType::I64, None);
    let p_pct = g.add_node(OpCode::Param(1), vec![], DataType::I64, None);
    let c_0 = g.add_node(OpCode::ConstI64(0), vec![], DataType::I64, None);
    let c_100 = g.add_node(OpCode::ConstI64(100), vec![], DataType::I64, None);
    let clamped_pct = g.add_node(OpCode::Clamp, vec![p_pct, c_0, c_100], DataType::I64, None);
    let rem_pct = g.add_node(OpCode::Sub, vec![c_100, clamped_pct], DataType::I64, None);
    let mul = g.add_node(OpCode::Mul, vec![p_price, rem_pct], DataType::I64, None);
    let final_price = g.add_node(OpCode::Div, vec![mul, c_100], DataType::I64, None);
    g.return_node = final_price;

    let compiled = JitEngine::compile(&g).expect("ARM64 JIT compilation should succeed");
    unsafe {
        let res = compiled.call_i64_2(250, 20);
        assert_eq!(res, 200);

        let res2 = compiled.call_i64_2(1000, 50);
        assert_eq!(res2, 500);

        // Clamp test: pct > 100 should be clamped to 100
        let res3 = compiled.call_i64_2(1000, 150);
        assert_eq!(res3, 0);

        // Clamp test: pct < 0 should be clamped to 0
        let res4 = compiled.call_i64_2(1000, -25);
        assert_eq!(res4, 1000);
    }
}

#[test]
fn test_arm64_jit_tensor_reduction() {
    use tgs_engine::jit::JitEngine;

    if !JitEngine::is_supported() {
        return;
    }

    let mut g = Graph::new("tensor_activation", vec![DataType::Tensor1D], DataType::F64);
    let p_tensor = g.add_node(OpCode::Param(0), vec![], DataType::Tensor1D, None);
    let biased = g.add_node(OpCode::TensorMapAdd(0.5), vec![p_tensor], DataType::Tensor1D, None);
    let sum = g.add_node(OpCode::TensorSum, vec![biased], DataType::F64, None);
    g.return_node = sum;

    let compiled = JitEngine::compile(&g).expect("Tensor JIT compilation should succeed");
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    unsafe {
        let res = compiled.call_tensor_reduce(input.as_ptr(), input.len());
        eprintln!("Tensor JIT result: got {}, expected 17.5", res);
        assert!((res - 17.5).abs() < 1e-6);
    }
}

#[test]
fn test_jit_content_addressed_cache() {
    use tgs_engine::cache::JitCache;

    let mut cache = JitCache::new();

    let mut g = Graph::new("simple", vec![DataType::I64, DataType::I64], DataType::I64);
    let p0 = g.add_node(OpCode::Param(0), vec![], DataType::I64, None);
    let p1 = g.add_node(OpCode::Param(1), vec![], DataType::I64, None);
    let add = g.add_node(OpCode::Add, vec![p0, p1], DataType::I64, None);
    g.return_node = add;

    // First lookup: compilation miss
    let fn1 = cache.get_or_compile(&g).expect("First compile should succeed");
    assert_eq!(cache.stats.cache_misses, 1);
    assert_eq!(cache.stats.cache_hits, 0);

    // Second lookup with identical graph: instant cache hit!
    let fn2 = cache.get_or_compile(&g).expect("Second lookup should hit cache");
    assert_eq!(cache.stats.cache_hits, 1);
    assert_eq!(cache.stats.total_lookups, 2);

    unsafe {
        assert_eq!(fn1.call_i64_2(15, 27), 42);
        assert_eq!(fn2.call_i64_2(15, 27), 42);
    }
}

#[test]
fn test_bounded_loop_unrolling() {
    let mut g = Graph::new("compound_5_years", vec![DataType::I64], DataType::I64);
    let p0 = g.add_node(OpCode::Param(0), vec![], DataType::I64, Some(Constraint::NonNegative));

    // Unroll 3 iterations of 10% compound interest
    g.unroll_loop(p0, 3, |graph, curr_state, _step| {
        let c_110 = graph.add_node(OpCode::ConstI64(110), vec![], DataType::I64, None);
        let c_100 = graph.add_node(OpCode::ConstI64(100), vec![], DataType::I64, None);
        let mul = graph.add_node(OpCode::Mul, vec![curr_state, c_110], DataType::I64, Some(Constraint::NonNegative));
        graph.add_node(OpCode::Div, vec![mul, c_100], DataType::I64, Some(Constraint::NonNegative))
    });

    // Verification check
    let bounds = vec![Interval { min: 1000, max: 10000 }];
    let verified = Verifier::verify(&g, &bounds).expect("Bounded loop must verify");
    assert!(!verified.is_empty());

    // Execute via DVM: 1000 * 1.1 * 1.1 * 1.1 = 1331
    let res = Engine::execute(&g, &[Value::I64(1000)]).unwrap();
    assert_eq!(res, Value::I64(1331));
}




