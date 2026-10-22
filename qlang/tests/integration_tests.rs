use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Shape, TensorData, TensorType};
use std::collections::HashMap;

fn f32v(n: usize) -> TensorType {
    TensorType::f32_vector(n)
}

fn f32m(m: usize, n: usize) -> TensorType {
    TensorType::f32_matrix(m, n)
}

// ---------------------------------------------------------------------------
// 1. Build a graph with add_node/add_edge, execute, verify output
// ---------------------------------------------------------------------------

#[test]
fn test_graph_build_and_execute() {
    let mut g = Graph::new("integ_add");
    let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![f32v(4)]);
    let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![f32v(4)]);
    let add = g.add_node(Op::Add, vec![f32v(4), f32v(4)], vec![f32v(4)]);
    let out = g.add_node(Op::Output { name: "y".into() }, vec![f32v(4)], vec![]);
    g.add_edge(a, 0, add, 0, f32v(4));
    g.add_edge(b, 0, add, 1, f32v(4));
    g.add_edge(add, 0, out, 0, f32v(4));

    let mut inputs = HashMap::new();
    inputs.insert(
        "a".into(),
        TensorData::from_f32(Shape::vector(4), &[1.0, 2.0, 3.0, 4.0]),
    );
    inputs.insert(
        "b".into(),
        TensorData::from_f32(Shape::vector(4), &[10.0, 20.0, 30.0, 40.0]),
    );

    let result = qlang_runtime::executor::execute(&g, inputs).unwrap();
    let y = result.outputs.get("y").unwrap().as_f32_slice().unwrap();
    assert_eq!(y, vec![11.0, 22.0, 33.0, 44.0]);
}

// ---------------------------------------------------------------------------
// 2. Graph → JSON → Graph round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_serialize_json_roundtrip() {
    let mut g = Graph::new("json_rt");
    g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32v(8)]);
    g.add_node(Op::Relu, vec![f32v(8)], vec![f32v(8)]);
    g.add_edge(0, 0, 1, 0, f32v(8));

    let json = qlang_core::serial::to_json(&g).unwrap();
    let g2 = qlang_core::serial::from_json(&json).unwrap();

    assert_eq!(g.id, g2.id);
    assert_eq!(g.nodes.len(), g2.nodes.len());
    assert_eq!(g.edges.len(), g2.edges.len());
}

// ---------------------------------------------------------------------------
// 3. Graph → binary → Graph round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_serialize_binary_roundtrip() {
    let mut g = Graph::new("bin_rt");
    g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32m(4, 4)]);
    g.add_node(
        Op::ToTernary,
        vec![f32m(4, 4)],
        vec![TensorType::ternary_matrix(4, 4)],
    );
    g.add_edge(0, 0, 1, 0, f32m(4, 4));

    let bin = qlang_core::serial::to_binary(&g).unwrap();
    let g2 = qlang_core::serial::from_binary(&bin).unwrap();

    assert_eq!(g.id, g2.id);
    assert_eq!(g.nodes.len(), g2.nodes.len());
    assert_eq!(g.edges.len(), g2.edges.len());
}

// ---------------------------------------------------------------------------
// 4. Train MLP, verify loss decreases
// ---------------------------------------------------------------------------

#[test]
fn test_train_mlp_loss_decreases() {
    use qlang_runtime::training::{generate_toy_dataset, MlpWeights};

    let dim = 16;
    let mut mlp = MlpWeights::new(dim, 8, 4);
    let (images, labels) = generate_toy_dataset(8, dim);

    let probs_before = mlp.forward(&images);
    let loss_before = mlp.loss(&probs_before, &labels);

    for _ in 0..3 {
        mlp.train_step(&images, &labels, 0.001);
    }

    let probs_after = mlp.forward(&images);
    let loss_after = mlp.loss(&probs_after, &labels);
    assert!(
        loss_after < loss_before,
        "Loss did not decrease: {loss_before} -> {loss_after}"
    );
}

// ---------------------------------------------------------------------------
// 5. Ternary compression: all values in {-1, 0, +1}
// ---------------------------------------------------------------------------

#[test]
fn test_ternary_compression() {
    use qlang_runtime::training::MlpWeights;

    let mlp = MlpWeights::new(16, 8, 4);
    let compressed = mlp.compress_ternary();

    for &w in &compressed.w1 {
        assert!(
            w == -1.0 || w == 0.0 || w == 1.0,
            "Non-ternary w1 value: {w}"
        );
    }
    for &w in &compressed.w2 {
        assert!(
            w == -1.0 || w == 0.0 || w == 1.0,
            "Non-ternary w2 value: {w}"
        );
    }
}

// ---------------------------------------------------------------------------
// 6. VM: fibonacci(10) == 55
// ---------------------------------------------------------------------------

#[test]
fn test_vm_fibonacci() {
    let src = r#"
fn fib(n) {
    if n <= 1.0 {
        return n
    }
    return fib(n - 1.0) + fib(n - 2.0)
}
return fib(10.0)
"#;
    let (val, _output) = qlang_runtime::vm::run_qlang_script(src).unwrap();
    assert_eq!(val.as_number().unwrap(), 55.0);
}

// ---------------------------------------------------------------------------
// 7. VM: sum an array
// ---------------------------------------------------------------------------

#[test]
fn test_vm_array_sum() {
    let src = r#"
let arr = [10.0, 20.0, 30.0, 40.0]
let total = 0.0
let i = 0.0
while i < len(arr) {
    total = total + arr[i]
    i = i + 1.0
}
return total
"#;
    let (val, _output) = qlang_runtime::vm::run_qlang_script(src).unwrap();
    assert_eq!(val.as_number().unwrap(), 100.0);
}

// ---------------------------------------------------------------------------
// 8. Type checker catches MatMul dimension mismatch
// ---------------------------------------------------------------------------

#[test]
fn test_type_check_catches_mismatch() {
    let mut g = Graph::new("tc_bad");
    let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![f32m(2, 3)]);
    let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![f32m(5, 4)]);
    let mm = g.add_node(Op::MatMul, vec![], vec![]);
    g.add_edge(a, 0, mm, 0, f32m(2, 3));
    g.add_edge(b, 0, mm, 1, f32m(5, 4));

    let errors = qlang_core::type_check::type_check(&g);
    assert!(!errors.is_empty());
    assert!(errors[0].message.contains("dimension mismatch"));
}

// ---------------------------------------------------------------------------
// 9. Shape inference through a MatMul chain
// ---------------------------------------------------------------------------

#[test]
fn test_shape_inference() {
    let mut g = Graph::new("shape_chain");
    let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![f32m(4, 8)]);
    let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![f32m(8, 3)]);
    let mm = g.add_node(Op::MatMul, vec![f32m(4, 8), f32m(8, 3)], vec![f32m(4, 3)]);
    let out = g.add_node(Op::Output { name: "y".into() }, vec![f32m(4, 3)], vec![]);
    g.add_edge(a, 0, mm, 0, f32m(4, 8));
    g.add_edge(b, 0, mm, 1, f32m(8, 3));
    g.add_edge(mm, 0, out, 0, f32m(4, 3));

    let shapes = qlang_core::shape_inference::infer_shapes(&g).unwrap();
    let mm_out = &shapes[&mm];
    assert_eq!(mm_out.len(), 1);
    assert_eq!(mm_out[0].shape, Shape::matrix(4, 3));
}

// ---------------------------------------------------------------------------
// 10. Optimizer removes dead nodes
// ---------------------------------------------------------------------------

#[test]
fn test_optimizer_removes_dead_nodes() {
    let mut g = Graph::new("dce_integ");
    let inp = g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32v(4)]);
    let relu = g.add_node(Op::Relu, vec![f32v(4)], vec![f32v(4)]);
    let _dead = g.add_node(Op::Neg, vec![f32v(4)], vec![f32v(4)]);
    let out = g.add_node(Op::Output { name: "y".into() }, vec![f32v(4)], vec![]);
    g.add_edge(inp, 0, relu, 0, f32v(4));
    g.add_edge(relu, 0, out, 0, f32v(4));

    let before = g.nodes.len();
    let report = qlang_compile::optimize::optimize(&mut g);
    assert!(g.nodes.len() < before);
    assert!(report.dead_nodes_removed >= 1);
}

// ---------------------------------------------------------------------------
// 11. WASM output contains "(module"
// ---------------------------------------------------------------------------

#[test]
fn test_wasm_output_valid() {
    let mut g = Graph::new("wasm_integ");
    let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![f32v(4)]);
    let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![f32v(4)]);
    let add = g.add_node(Op::Add, vec![f32v(4); 2], vec![f32v(4)]);
    let out = g.add_node(Op::Output { name: "y".into() }, vec![f32v(4)], vec![]);
    g.add_edge(a, 0, add, 0, f32v(4));
    g.add_edge(b, 0, add, 1, f32v(4));
    g.add_edge(add, 0, out, 0, f32v(4));

    let wat = qlang_compile::wasm::to_wat(&g);
    assert!(wat.contains("(module"), "WAT missing module header");
}

// ---------------------------------------------------------------------------
// 12. GPU shader output contains "@compute"
// ---------------------------------------------------------------------------

#[test]
fn test_gpu_shader_valid() {
    let mut g = Graph::new("gpu_integ");
    let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![f32v(256)]);
    let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![f32v(256)]);
    let add = g.add_node(Op::Add, vec![f32v(256); 2], vec![f32v(256)]);
    let out = g.add_node(Op::Output { name: "y".into() }, vec![f32v(256)], vec![]);
    g.add_edge(a, 0, add, 0, f32v(256));
    g.add_edge(b, 0, add, 1, f32v(256));
    g.add_edge(add, 0, out, 0, f32v(256));

    let wgsl = qlang_compile::gpu::to_wgsl(&g);
    assert!(wgsl.contains("@compute"), "WGSL missing @compute");
}

// ---------------------------------------------------------------------------
// 13. Autograd gradients are all finite
// ---------------------------------------------------------------------------

#[test]
fn test_autograd_gradients_finite() {
    use qlang_runtime::autograd::Tape;

    let mut tape = Tape::new();
    let a = tape.variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = tape.variable(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]);
    let c = tape.matmul(a, b);
    let d = tape.relu(c);

    tape.backward(d);

    for id in [a, b] {
        let grad = tape.grad(id).expect("gradient should exist");
        assert!(
            grad.iter().all(|g| g.is_finite()),
            "non-finite gradient found"
        );
    }
}

// ---------------------------------------------------------------------------
// 14. Checkpoint save/load round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_checkpoint_save_load() {
    use qlang_runtime::checkpoint::{Checkpoint, WeightTensor};

    let g = Graph::new("ckpt_test");
    let mut ckpt = Checkpoint::new(g);
    ckpt.metadata.final_loss = 0.5; // avoid NaN which doesn't round-trip in JSON
    let weights = vec![0.1f32, -0.5, 0.9, 0.0];
    ckpt.add_weight(WeightTensor::from_f32("w1", vec![2, 2], &weights));

    let path = "/tmp/qlang_integ_test_ckpt.json";
    ckpt.save(path).unwrap();
    let loaded = Checkpoint::load(path).unwrap();

    let w = loaded.weights.get("w1").unwrap().as_f32().unwrap();
    assert_eq!(w, weights);
    std::fs::remove_file(path).ok();
}

// ---------------------------------------------------------------------------
// 15. Graph Display contains node names
// ---------------------------------------------------------------------------

#[test]
fn test_graph_display() {
    let mut g = Graph::new("display_integ");
    g.add_node(
        Op::Input {
            name: "features".into(),
        },
        vec![],
        vec![f32v(10)],
    );
    g.add_node(Op::Relu, vec![f32v(10)], vec![f32v(10)]);
    g.add_node(
        Op::Output {
            name: "result".into(),
        },
        vec![f32v(10)],
        vec![],
    );
    g.add_edge(0, 0, 1, 0, f32v(10));
    g.add_edge(1, 0, 2, 0, f32v(10));

    let display = format!("{g}");
    assert!(
        display.contains("input(features)"),
        "missing input node name"
    );
    assert!(
        display.contains("output(result)"),
        "missing output node name"
    );
    assert!(display.contains("relu"), "missing relu node");
}
