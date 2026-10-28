#!/usr/bin/env python3
"""Compare NumPy CPU and MLX/Metal matmul on this Mac."""
import argparse, json, time
import numpy as np
import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--iters", type=int, default=20)
a = ap.parse_args()
n, iters = a.n, a.iters
rng = np.random.default_rng(0)
x = rng.standard_normal((n, n), dtype=np.float32)
y = rng.standard_normal((n, n), dtype=np.float32)
# warmup
np.matmul(x, y)
mxa, mya = mx.array(x), mx.array(y)
mx.eval(mx.matmul(mxa, mya))
t0 = time.perf_counter()
for _ in range(iters):
    np.matmul(x, y)
cpu = (time.perf_counter() - t0) / iters
t0 = time.perf_counter()
for _ in range(iters):
    mx.eval(mx.matmul(mxa, mya))
mlx_t = (time.perf_counter() - t0) / iters
flops = 2 * n ** 3
print(json.dumps({
    "n": n, "iters": iters, "device": "Apple M1 / Metal",
    "numpy_ms": round(cpu * 1000, 3), "mlx_ms": round(mlx_t * 1000, 3),
    "numpy_gflops": round(flops / cpu / 1e9, 1), "mlx_gflops": round(flops / mlx_t / 1e9, 1),
    "speedup": round(cpu / mlx_t, 2),
}, indent=2))
