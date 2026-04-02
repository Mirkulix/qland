//! QLANG CLI — Compile, visualize and execute QLANG graph files.
//!
//! Usage:
//!   qlang-cli info     <file.qlg.json>                    Show graph info
//!   qlang-cli verify   <file.qlg.json>                    Verify constraints
//!   qlang-cli optimize <file.qlg.json> -o <output.json>   Optimize graph
//!   qlang-cli run      <file.qlg.json>                    Execute (interpreter)
//!   qlang-cli jit      <file.qlg.json>                    Execute (JIT/native)
//!   qlang-cli dot      <file.qlg.json>                    Output Graphviz DOT
//!   qlang-cli ascii    <file.qlg.json>                    ASCII visualization
//!   qlang-cli llvm-ir  <file.qlg.json>                    Show LLVM IR output

use std::collections::HashMap;
use std::env;
use std::fs;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        print_usage();
        process::exit(1);
    }

    let command = &args[1];
    let file_path = &args[2];

    // Read the graph file
    let content = match fs::read_to_string(file_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading {file_path}: {e}");
            process::exit(1);
        }
    };

    let graph = match qlang_core::serial::from_json(&content) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error parsing graph: {e}");
            process::exit(1);
        }
    };

    match command.as_str() {
        "info" => cmd_info(&graph),
        "verify" => cmd_verify(&graph),
        "optimize" => {
            let output = args.get(4).map(|s| s.as_str());
            cmd_optimize(graph, output);
        }
        "run" => cmd_run(&graph),
        "jit" => cmd_jit(&graph),
        "dot" => cmd_dot(&graph),
        "ascii" => cmd_ascii(&graph),
        "llvm-ir" => cmd_llvm_ir(&graph),
        _ => {
            eprintln!("Unknown command: {command}");
            print_usage();
            process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("QLANG CLI v0.2 — Graph-based AI-to-AI programming language\n");
    eprintln!("Usage:");
    eprintln!("  qlang-cli info     <file.qlg.json>                    Show graph info");
    eprintln!("  qlang-cli verify   <file.qlg.json>                    Verify constraints");
    eprintln!("  qlang-cli optimize <file.qlg.json> -o <output.json>   Optimize graph");
    eprintln!("  qlang-cli run      <file.qlg.json>                    Execute (interpreter)");
    eprintln!("  qlang-cli jit      <file.qlg.json>                    Execute (JIT/native)");
    eprintln!("  qlang-cli dot      <file.qlg.json>                    Output Graphviz DOT");
    eprintln!("  qlang-cli ascii    <file.qlg.json>                    ASCII visualization");
    eprintln!("  qlang-cli llvm-ir  <file.qlg.json>                    Show LLVM IR output");
}

fn cmd_info(graph: &qlang_core::graph::Graph) {
    println!("{graph}");

    let inputs = graph.input_nodes();
    let outputs = graph.output_nodes();
    let quantum_ops: Vec<_> = graph.nodes.iter().filter(|n| n.op.is_quantum()).collect();

    println!("Summary:");
    println!("  Inputs:      {}", inputs.len());
    println!("  Outputs:     {}", outputs.len());
    println!("  Total nodes: {}", graph.nodes.len());
    println!("  Total edges: {}", graph.edges.len());
    println!("  Quantum ops: {}", quantum_ops.len());

    if let Ok(binary) = qlang_core::serial::to_binary(graph) {
        println!("  Binary size: {} bytes", binary.len());
    }
}

fn cmd_verify(graph: &qlang_core::graph::Graph) {
    let result = qlang_core::verify::verify_graph(graph);
    println!("{result}");

    if result.is_ok() {
        println!("Graph verification PASSED.");
    } else {
        println!("Graph verification FAILED.");
        process::exit(1);
    }
}

fn cmd_optimize(mut graph: qlang_core::graph::Graph, output: Option<&str>) {
    let before = graph.nodes.len();
    qlang_compile::optimize::optimize(&mut graph);
    let after = graph.nodes.len();

    println!("Optimization complete:");
    println!("  Nodes before: {before}");
    println!("  Nodes after:  {after}");
    println!("  Removed:      {}", before - after);

    if let Some(path) = output {
        let json = qlang_core::serial::to_json(&graph).unwrap();
        fs::write(path, json).unwrap();
        println!("  Saved to: {path}");
    }
}

fn cmd_run(graph: &qlang_core::graph::Graph) {
    let mut inputs = HashMap::new();
    for node in graph.input_nodes() {
        if let qlang_core::ops::Op::Input { name } = &node.op {
            if let Some(tt) = node.output_types.first() {
                if let Some(data) = qlang_core::tensor::TensorData::zeros(tt) {
                    println!("  Input '{name}': {} (zeros)", tt);
                    inputs.insert(name.clone(), data);
                }
            }
        }
    }

    match qlang_runtime::executor::execute(graph, inputs) {
        Ok(result) => {
            println!("\nExecution complete (interpreter):");
            println!("  Nodes executed: {}", result.stats.nodes_executed);
            println!("  Quantum ops:    {}", result.stats.quantum_ops);
            println!("  Total FLOPs:    {}", result.stats.total_flops);

            for (name, tensor) in &result.outputs {
                println!("\n  Output '{name}':");
                println!("    dtype: {}", tensor.dtype);
                println!("    shape: {}", tensor.shape);
                if let Some(vals) = tensor.as_f32_slice() {
                    if vals.len() <= 20 {
                        println!("    values: {:?}", vals);
                    } else {
                        println!("    values: [{}, {}, ... {} total]", vals[0], vals[1], vals.len());
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Execution failed: {e}");
            process::exit(1);
        }
    }
}

fn cmd_jit(graph: &qlang_core::graph::Graph) {
    use inkwell::context::Context;
    use inkwell::OptimizationLevel;

    println!("JIT compiling graph '{}'...", graph.id);

    let context = Context::create();
    match qlang_compile::codegen::compile_graph(&context, graph, OptimizationLevel::Aggressive) {
        Ok(compiled) => {
            println!("  Compilation successful!");
            println!("  LLVM IR size: {} bytes", compiled.llvm_ir.len());

            // Determine input sizes from graph
            let input_nodes = graph.input_nodes();
            let n = input_nodes.first()
                .and_then(|n| n.output_types.first())
                .and_then(|t| t.shape.numel())
                .unwrap_or(4);

            let input_a = vec![0.0f32; n];
            let input_b = vec![0.0f32; n];

            println!("  Executing with {} zero-filled elements...", n);

            match qlang_compile::codegen::execute_compiled(&compiled, &input_a, &input_b) {
                Ok(result) => {
                    println!("\n  JIT execution complete (native code):");
                    if result.len() <= 20 {
                        println!("    output: {:?}", result);
                    } else {
                        println!("    output: [{}, {}, ... {} total]", result[0], result[1], result.len());
                    }
                }
                Err(e) => eprintln!("  JIT execution failed: {e}"),
            }
        }
        Err(e) => {
            eprintln!("  JIT compilation failed: {e}");
            process::exit(1);
        }
    }
}

fn cmd_dot(graph: &qlang_core::graph::Graph) {
    print!("{}", qlang_compile::visualize::to_dot(graph));
}

fn cmd_ascii(graph: &qlang_core::graph::Graph) {
    print!("{}", qlang_compile::visualize::to_ascii(graph));
}

fn cmd_llvm_ir(graph: &qlang_core::graph::Graph) {
    use inkwell::context::Context;
    use inkwell::OptimizationLevel;

    let context = Context::create();
    match qlang_compile::codegen::compile_graph(&context, graph, OptimizationLevel::None) {
        Ok(compiled) => {
            println!("{}", compiled.llvm_ir);
        }
        Err(e) => {
            eprintln!("Codegen failed: {e}");
            process::exit(1);
        }
    }
}
