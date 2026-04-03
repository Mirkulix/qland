# QLANG Tutorial

## 1. Installation

```bash
git clone https://github.com/Mirkulix/qland.git
cd qland/qlang

# Linux/Mac (with LLVM for JIT):
./setup.sh

# Windows (without LLVM):
cargo build --release --no-default-features
```

## 2. Hello World

Create `hello.qlang`:
```qlang
graph hello {
  input x: f32[4]
  input y: f32[4]
  node sum = add(x, y)
  node result = relu(sum)
  output out = result
}
```

```bash
cargo run --release --bin qlang-cli -p qlang-compile -- parse hello.qlang
```

## 3. Interactive REPL

```bash
cargo run --release --bin qlang-cli -p qlang-compile -- repl
```

```
qlang> input x: f32[4]
qlang> input y: f32[4]
qlang> node sum = add(x, y)
qlang> output result = sum
qlang> set x = [1, -2, 3, -4]
qlang> set y = [10, 20, 30, 40]
qlang> run
  result = [11.0, 18.0, 33.0, 36.0]
```

## 4. Write a Script

QLANG is a full programming language:
```qlang
fn fibonacci(n) {
  if n <= 1.0 { return n }
  return fibonacci(n - 1.0) + fibonacci(n - 2.0)
}

let result = fibonacci(10.0)
print(result)

let data = [1.0, 2.0, 3.0, 4.0, 5.0]
let sum = 0.0
for i in 0..len(data) {
  sum = sum + data[i]
}
print(sum)
```

## 5. Train a Neural Network

```bash
cargo run --release --example train_autograd
```

Output:
```
Epoch  1/50: loss=1.3507, acc=67.5%
Epoch 50/50: loss=0.0188, acc=100.0%
Training time: 70ms
```

## 6. Compress with IGQK

```qlang
graph compress {
  input weights: f32[768, 768]
  node compressed = to_ternary(weights) @proof theorem_5_2
  output small = compressed
}
```

Result: 2.4 MB -> 150 KB (16x compression) with mathematical proof.

## 7. Deploy

```bash
# Native object file (link with C/C++)
qlang-cli compile model.qlg.json -o model.o

# WebAssembly (browser)
qlang-cli wasm model.qlg.json > model.wat

# GPU shader
qlang-cli gpu model.qlg.json > model.wgsl
```

## 8. Use from Python

```python
import qlang

g = qlang.Graph("demo")
x = g.add_input("x", "f32", [4])
y = g.add_input("y", "f32", [4])
s = g.add_add(x, y)
r = g.add_relu(s)
g.add_output("result", r)

result = g.execute({"x": [1,-2,3,-4], "y": [10,20,30,40]})
print(result)  # {"result": [11.0, 18.0, 33.0, 36.0]}
```

## 9. Use from C

```c
#include "qlang.h"

int main() {
    QlangGraph* g = qlang_graph_new("demo");
    uint32_t shape[] = {4};
    uint32_t x = qlang_graph_add_input(g, "x", 2, shape, 1);  // f32
    // ... add ops ...
    qlang_graph_free(g);
}
```

## 10. All Examples

```bash
cargo run --release --example hello_qlang
cargo run --release --example train_autograd
cargo run --release --example train_mnist
cargo run --release --example transformer
cargo run --release --example benchmark
cargo run --release --example full_pipeline
```
