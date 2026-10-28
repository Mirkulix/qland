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
