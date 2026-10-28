# QAL packages

## `prelude.qalpkg` (`qal_prelude`)

The helper library used by the 100-task AST pilot:

- 22 definitions: `parse_ints`, `sum_all`, `max_all`, `min_all`, `product_all`,
  `fact_first`, `count_words`, `reverse_line`, `abs_diff`, `count_chars`,
  `first_int`, and their helpers.
- Exported from `bench/compare/prelude.qal` with `qal export "" --name qal_prelude`.
- Header hash `cb68f252…`; registry SHA-256 `2211e122…` when signed.

Verified flow:

```sh
python3 qal-registry/registry.py publish qal/packages/prelude.qalpkg \
  --key key.pem --version 1.0.0
python3 qal-registry/registry.py trust --public pub.pem
python3 qal-registry/registry.py install --name qal_prelude --root <project> --trust
qal query --name sum_all          # fn(Str) -> Int
qal define main.json --ast        # uses sum_all
echo '2 3' | qal run main --allow io   # 5
```

The package is signed and resolved through the same registry trust path; the AST
definition only composes verified definitions.

## `event_queue.qalpkg` (`qal_event_queue`)

Deterministic event queue: `Event`, `Queue`, `Popped`, `enqueue`, `pop`, `size`,
with three tests. Exported from `bench/sim/queue.qal`, 10 definitions, header
hash `beadff7c…`, signed registry SHA-256 `d8e8b761…`.

## `tensor.qalpkg` (`qal_tensor`)

Row-major tensor with `at`, `dot`, `matmul`, `identity`, shape postcondition and
three tests. 11 definitions, signed registry SHA-256 `09cc3dcf…`.

## `numerics.qalpkg` (`qal_numerics`)

`integrate(f, a, b, n)` with range contracts, plus `square`, `linear`, `absf`;
two integral tests. 6 definitions, signed registry SHA-256 `550726e1…`.

## `tensor_remote.qalpkg` (`qal_tensor_remote`)

`matmul_remote(a, b, n) !net` with shape `req`/`ens`, backed by the NumPy
sidecar in `bench/tensor/server.py`. 3 definitions, signed registry SHA-256
`51c31794…`; `qal query matmul_remote` shows `!net`, `local_shape` passes.
