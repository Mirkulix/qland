# QLANG API Reference

## Language Syntax

### Graph Definition
```qlang
graph <name> {
  input <name>: <type>
  node <name> = <op>(<args>) [@proof <theorem>]
  output <name> = <node>
}
```

### Script (VM)
```qlang
let x = 5.0                    // variable
x = x + 1.0                   // assignment
if x > 3.0 { ... } else { ... } // conditional
for i in 0..10 { ... }        // for loop
while x > 0.0 { ... }         // while loop
fn foo(a, b) { return a + b } // function
print(x)                      // output
let arr = [1.0, 2.0, 3.0]    // array
arr[0]                         // indexing
```

## Types

| Type | Description | Size |
|------|-------------|------|
| `f16` | 16-bit float | 2 bytes |
| `f32` | 32-bit float | 4 bytes |
| `f64` | 64-bit float | 8 bytes |
| `i8` | 8-bit integer | 1 byte |
| `i16` | 16-bit integer | 2 bytes |
| `i32` | 32-bit integer | 4 bytes |
| `i64` | 64-bit integer | 8 bytes |
| `bool` | Boolean | 1 byte |
| `ternary` | {-1, 0, +1} | 1 byte |

Shapes: `f32[784]` (vector), `f32[28, 28]` (matrix), `f32[?]` (dynamic)

## Operations (36+)

### Tensor Ops
| Op | Inputs | Output | Description |
|----|--------|--------|-------------|
| `add` | a, b | a+b | Element-wise add |
| `sub` | a, b | a-b | Element-wise subtract |
| `mul` | a, b | a*b | Element-wise multiply |
| `div` | a, b | a/b | Element-wise divide |
| `neg` | a | -a | Negate |
| `matmul` | a, b | a@b | Matrix multiplication |
| `transpose` | a | a^T | Transpose |
| `reshape` | a | reshaped | Reshape tensor |
| `concat` | a, b | [a,b] | Concatenate |
| `reduce_sum` | a | scalar | Sum all elements |
| `reduce_mean` | a | scalar | Mean of elements |
| `reduce_max` | a | scalar | Max element |

### Activations
| Op | Formula | Description |
|----|---------|-------------|
| `relu` | max(0, x) | Rectified linear unit |
| `sigmoid` | 1/(1+e^-x) | Sigmoid |
| `tanh` | tanh(x) | Hyperbolic tangent |
| `gelu` | x*Phi(x) | Gaussian error linear unit |
| `softmax` | e^x/sum(e^x) | Softmax probabilities |

### IGQK / Quantum
| Op | Description |
|----|-------------|
| `to_ternary` | Compress to {-1, 0, +1} |
| `to_lowrank` | Low-rank approximation |
| `to_sparse` | Sparsification |
| `superpose` | Quantum superposition |
| `evolve` | Quantum gradient flow |
| `measure` | Born rule measurement |
| `entangle` | Quantum entanglement |
| `entropy` | Von Neumann entropy |
| `fisher_metric` | Fisher information |
| `project` | Project onto manifold |

### Transformer
| Op | Description |
|----|-------------|
| `layer_norm` | Layer normalization |
| `attention` | Multi-head attention |
| `embedding` | Token embedding |
| `residual` | Residual connection |
| `dropout` | Dropout (training) |

## Standard Library (44 functions)

### Math
`abs`, `sqrt`, `pow`, `min`, `max`, `floor`, `ceil`, `round`, `sin`, `cos`, `log`, `exp`

### Arrays
`len`, `sum`, `mean`, `max_val`, `min_val`, `sort`, `reverse`, `range`, `zeros`, `ones`, `linspace`

### Strings
`str`, `concat`, `split`, `trim`, `contains`, `replace`, `to_upper`, `to_lower`, `starts_with`, `ends_with`

### I/O
`print`, `println`, `read_file`, `write_file`

### Types
`type_of`, `is_number`, `is_array`, `is_string`, `to_number`, `to_string`

### Tensors
`shape`, `reshape`, `transpose`, `dot`, `matmul`

### Random
`random`, `random_range`, `random_array`

### Time
`clock`

## CLI Commands

```
qlang-cli repl                     Interactive REPL
qlang-cli parse <file.qlang>       Parse and validate
qlang-cli info <file.qlg.json>     Graph information
qlang-cli verify <file.qlg.json>   Check constraints
qlang-cli optimize <file>          Run 6 optimization passes
qlang-cli run <file.qlg.json>      Execute (interpreter)
qlang-cli jit <file.qlg.json>      Execute (LLVM JIT)
qlang-cli compile <file> -o out.o  Compile to object file
qlang-cli asm <file.qlg.json>      Show assembly
qlang-cli wasm <file.qlg.json>     WebAssembly output
qlang-cli gpu <file.qlg.json>      GPU WGSL shader
qlang-cli dot <file.qlg.json>      Graphviz visualization
qlang-cli ascii <file.qlg.json>    ASCII visualization
qlang-cli stats <file.qlg.json>    Graph statistics
qlang-cli schedule <file>          Execution plan
```

## Rust API

### qlang-core
```rust
use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Dtype, Shape, TensorType, TensorData};
use qlang_core::verify::verify_graph;
use qlang_core::serial::{to_json, from_json, to_binary, from_binary};
use qlang_core::type_check::type_check;
use qlang_core::shape_inference::infer_shapes;
use qlang_core::stats::compute_stats;
```

### qlang-runtime
```rust
use qlang_runtime::executor::execute;
use qlang_runtime::autograd::{Tape, train_mlp_autograd};
use qlang_runtime::vm::run_qlang_script;
use qlang_runtime::training::{MlpWeights, generate_toy_dataset};
use qlang_runtime::transformer::*;
use qlang_runtime::checkpoint::Checkpoint;
use qlang_runtime::unified::execute_unified;
```

### qlang-agent
```rust
use qlang_agent::emitter::GraphEmitter;
use qlang_agent::protocol::{AgentConversation, MessageIntent};
use qlang_agent::server::{Server, Client};
use qlang_agent::packages::{Package, Registry};
```

## Python API

```python
import qlang

# Graph construction
g = qlang.Graph("name")
node_id = g.add_input("x", "f32", [784])
node_id = g.add_matmul(a_id, b_id)
node_id = g.add_relu(input_id)
node_id = g.add_add(a_id, b_id)
node_id = g.add_softmax(input_id)
node_id = g.add_to_ternary(input_id)
g.add_output("y", source_id)

# Execution
result = g.execute({"x": [1.0, 2.0], "y": [3.0, 4.0]})

# Serialization
json_str = g.to_json()
text = g.to_qlang_text()
n = g.num_nodes()
ok = g.verify()

# Standalone functions
compressed = qlang.compress_ternary([0.5, -0.3, 0.8])
result = qlang.train_mlp(x_data, y_data, targets, epochs=50, lr=0.01)
```

## C FFI

```c
#include "qlang.h"

QlangGraph* qlang_graph_new(const char* name);
void qlang_graph_free(QlangGraph* graph);
uint32_t qlang_graph_add_input(QlangGraph*, const char*, uint32_t dtype, const uint32_t* shape, uint32_t shape_len);
uint32_t qlang_graph_add_op(QlangGraph*, const char* op, uint32_t input_a, uint32_t input_b);
uint32_t qlang_graph_add_output(QlangGraph*, const char* name, uint32_t source);
char* qlang_graph_to_json(QlangGraph*);
void qlang_free_string(char*);
uint32_t qlang_graph_num_nodes(QlangGraph*);
int32_t qlang_graph_verify(QlangGraph*);
const char* qlang_version(void);
```

## Configuration

### Environment Variables
```
QLANG_LOG_LEVEL=info        # error, warn, info, debug, trace
QLANG_MAX_MEMORY_MB=1024    # memory limit
QLANG_MAX_TIME_MS=30000     # execution timeout
QLANG_ENABLE_JIT=true       # JIT compilation
QLANG_THREADS=4             # thread count
```

### JSON Config File
```json
{
  "log_level": "info",
  "max_memory_mb": 2048,
  "max_execution_time_ms": 60000,
  "enable_jit": true,
  "num_threads": 4,
  "model_cache_dir": ".qlang/cache"
}
```

## Wire Formats

### Binary .qlg
```
[0x51, 0x4C, 0x41, 0x4E]  "QLAN" magic
[u16 version]              Wire version
[u16 flags]                Reserved
[JSON payload]             Graph data
```

### Checkpoint .qlm
```
[0x51, 0x4C, 0x4D, 0x44]  "QLMD" magic
[u32 version]              Format version
[u32 payload_size]         Size in bytes
[JSON payload]             Graph + weights + metadata
```

### Agent Protocol QLMS
```
[0x51, 0x4C, 0x4D, 0x53]  "QLMS" magic
[u32 n_messages]           Message count
[JSON payload]             Message array
```
