# NumPy tensor sidecar

QAL keeps the shapes and contracts; NumPy does the matmul.

- `server.py`: `POST /matmul` with `n;a_csv;b_csv`, returns flat CSV.
- `host.qal`: `matmul_remote(a, b, n) !net` with
  `req n > 0 && len(a) == n*n && len(b) == n*n` and `ens len(r) == n*n`.
- `main.qal`: 2x2 demo.

```sh
./bench/tensor/verify.sh
# tensor: NumPy sidecar 2x2 -> 19.0,22.0,43.0,50.0
```

This uses the existing `!net` capability and a host process, not a new
`!tensor` effect: QAL owns the shape contract, the sidecar owns the numeric
kernel. A native `!tensor` capability can replace `!net` later without changing
the contract.

## Apple Silicon / MLX

`server_mlx.py` speaks the same protocol and runs the matmul on the Apple GPU via
MLX/Metal; `bench_mlx.py` compares it with NumPy on the M1:

```json
{"n":512,"numpy_ms":3.046,"mlx_ms":0.884,
 "numpy_gflops":88.1,"mlx_gflops":303.7,"speedup":3.45}
```

`verify_mlx.sh` runs the QAL 2x2 demo through MLX:

```text
tensor-mlx: 2x2 via MLX/Metal, 466.1 GFLOPS vs numpy 135.3 (3.45x)
```

The QAL contract (`req len == n*n`, `ens len(r) == n*n`) is unchanged; only the
host backend changes.

### Scaling on the M1 (8-core GPU)

| n | NumPy ms | MLX ms | NumPy GFLOPS | MLX GFLOPS | speedup |
|---:|---:|---:|---:|---:|---:|
| 256 | 13.84 | 4.24 | 2.4 | 7.9 | 3.27x |
| 512 | 10.80 | 0.85 | 24.9 | 315.7 | 12.7x |
| 1024 | 13.13 | 3.16 | 163.5 | 680.8 | 4.16x |
| 2048 | 69.87 | 11.96 | 245.9 | **1436.8** | 5.84x |

At 2048×2048 MLX reaches **1.44 TFLOPS** on the M1; the small-size numbers are
overhead-dominated, the large ones show the GPU.

The helper is also a package: `qal/packages/tensor_remote.qalpkg`
(`qal_tensor_remote`), 3 definitions, signed install verified. `matmul_remote`
reports effect `!net`; the caller must grant `net:127.0.0.1:8901`.

## Native `!tensor`

`native.qal` calls the real builtin `tensor_matmul(a_csv, b_csv, n) -> Str`:

```qal
fn matmul(a, b, n) -> List[Float] !tensor
```

Verified:

```text
qal run native.qal --allow io,tensor -> 19.0,22.0,43.0,50.0
qal run native.qal --allow io        -> E0508 permission_denied
```

On macOS the native kernel calls **Apple Accelerate** `cblas_sgemm`; elsewhere it
is a naive Rust loop. `native_bench.qal` measures the whole QAL path
(build + CSV encode/decode + matmul): 128→22 ms, 256→80 ms, 512→316 ms,
1024→1300 ms. The CSV encode/decode dominates for large matrices; the MLX
sidecar is the fast GPU path.
