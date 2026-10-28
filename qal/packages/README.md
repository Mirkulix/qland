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

## `sim_core.qalpkg` (`qal_sim`)

Reusable simulation core: `Agent`, `Sim`, `init`, `total`, `transfer`, `step`,
two tests (`transfer_conserves`, `self_transfer_is_noop`). 10 definitions,
signed registry SHA-256 `9d905d7e…`; `transfer` keeps `ens total`.

`qal_import` over MCP installs a local `.qalpkg` from the project (relative
path, no signature check; use the registry client for signed HTTP installs).
Verified with `event_queue.qalpkg`: import succeeds and its three tests pass.

MCP package roundtrip: `qal_import` a local `.qalpkg`, `qal_export` a prefix
and dependencies to another `.qalpkg`, then import that elsewhere. Verified with
`enqueue` (3 defs) between two projects.

`qal_install` over MCP installs from a signed HTTP registry: SHA-256 + Ed25519
against a trusted-keys JSON, dependency order, then `ops::import`. Verified with
the local `qal_event_queue` store (3 tests pass); no trust file is refused.

## `sim_node.qalpkg` (`qal_sim_node`)

File-backed Raft follower endpoint: `POST /heartbeat` (wire `hb:...`) persists
`node_state.txt`, `GET /state` returns JSON, `setup` initializes. 7 definitions,
signed registry SHA-256 `ce039a02…`; verified with `qal request handle --init
setup --allow fs:.`.
