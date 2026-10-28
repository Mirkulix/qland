# QAL 100-agent cash simulation

`sim.qal` is a discrete, auditable simulation:

- 100 agents, each starting with 1000 cash, in a `Map[Int, Agent]`;
- 1000 randomized transfer steps (`rand_int`), effect `!rand`;
- pure `transfer` with `ens total(r) == total(sim)`;
- `main` prints the total and final tick.

## Measured

```text
qal check sim.qal  -> {"ok":true}
qal run sim.qal --allow io,rand
  -> agents=100 total=100000 tick=989
qal test sim.qal
  -> transfer_conserves passed
  -> self_transfer_is_noop passed
```

## The contract caught a real bug

The first run failed with:

```json
{"kind":"contract_violation","def":"transfer","path":"ens[0]",
 "message":"`ens total(r) == total(sim)` of `transfer` is false",
 "values":{"from":91,"to":91,"amount":50,"tick":19}}
```

A self-transfer (`from == to`) subtracted from the source and added the old
destination balance, creating 50 cash. Python tests would only catch this if
they happened to test `from == to`; the QAL postcondition caught it immediately.
After the fix:

```qal
= if from == to then sim else match map_get(sim.agents, from) { ... }
```

the run conserves 100,000 exactly.

## What this demonstrates

- QAL can express a discrete event/agent simulation;
- the invariant is checked at runtime, not just in tests;
- deterministic replay is possible with a fixed `rand` seed;
- the same pattern scales to scheduler/event-queue libraries and host numerics.

It is **not** a high-performance simulator: the interpreter, immutable maps and
recursion are fine for 100 agents/1000 steps, not millions.

## Event queue package

`queue.qal` defines a deterministic event queue over `Map[Int, Event]`:

- `enqueue(q, at, kind, payload)` keys by `at * 1_000_000 + seq`,
- `pop(q)` sorts keys, removes the first event and returns `Option[Popped]`,
- tests: `push_size`, `pop_order`, `pop_empty` all pass.

It is exported as `qal/packages/event_queue.qalpkg` (`qal_event_queue`, 10
definitions) and installs through the signed registry:

```text
installed qal_event_queue d8e8b761…
qal query --name enqueue -> fn(Queue, Int, Str, Str) -> Queue
qal test push_size pop_order pop_empty -> 3 passed
```

The queue is QAL-owned logic; time and worker execution stay in the host.

## Tensor package

`tensor.qal` is a row-major `Tensor { rows, cols, data }` package:

- `at`, `dot`, `matmul`, `identity`;
- `matmul` has `ens r.rows == a.rows && r.cols == b.cols`;
- tests: identity shape, identity multiplication and the 2x2 product (43).

Exported as `qal/packages/tensor.qalpkg` (`qal_tensor`, 11 definitions), signed
and installed through the registry:

```text
qal query --name matmul -> fn(Tensor, Tensor) -> Tensor
qal test identity_shape identity_matmul matmul_2x2 -> 3 passed
```

This is the QAL-owned shape/contract layer; a GPU/NumPy host would implement the
inner multiplication later.

## Numerics package

`numerics.qal` adds `integrate(f, a, b, n)` (left Riemann sum) with
`req n > 0 && n <= 1_000_000 && a <= b`, plus `square`, `linear` and `absf`.
Tests: integral of x^2 on [0,1] and x on [0,2], both within 0.01.

Exported as `qal/packages/numerics.qalpkg` (`qal_numerics`, 6 definitions),
signed and installable:

```text
qal query --name integrate
  -> fn(fn(Float) -> Float, Float, Float, Int) -> Float
qal test square_integral linear_integral -> 2 passed
```

QAL owns the contract and the loop shape; a host GPU/NumPy implementation can
replace the inner numerical work later.

## Queue-driven simulation

`sim_queue.qal` imports `qal_event_queue`, schedules 1000 tick events and processes
them in order; each event triggers a randomized transfer.

```sh
qal init
qal import qal/packages/event_queue.qalpkg
qal define bench/sim/sim_queue.qal
qal run main --allow io,rand
```

Verified:

```text
agents=100 total=100000 events=1000 tick=1000
```

The queue is the scheduler; `transfer` keeps the conservation contract.
`add_events` carries a bounded-range precondition; the prover reports `transfer`'s
map-based conservation as `unknown` (checked at runtime), which is honest.

## Dashboard endpoint

`dashboard.qal` serves both the dashboard shell and its metrics:

- `GET /` → HTML page that fetches `/metrics` and renders the JSON;
- `GET /metrics` → `{"agents":100,"total":100000,"tick":1000,
  "invariant":"total == 100000","source":"qal"}`.

```sh
qal define bench/sim/dashboard.qal
qal request handle reqs.json --allow io
```

Verified:

```text
GET /        -> 200 text/html
GET /metrics -> 200 application/json, total=100000
qal test total_100 -> passed
```

The HTML/JS is a shell; QAL owns the metric values, the invariant and the JSON
contract. Charts/updates remain a browser concern.

## Scaling limit (measured)

Same `sim.qal`, varying `init(agents)` and `run(..., steps)` on the release
binary, 256 MiB Wasm memory limit:

| agents | steps | result | wall | total |
|---:|---:|---|---:|---:|
| 100 | 1,000 | ok | 0.24 s | 100,000 |
| 500 | 5,000 | ok | 0.50 s | 500,000 |
| 1,000 | 5,000 | ok | 0.76 s | 1,000,000 |
| 2,000 | 1,000 | ok | 0.43 s | 2,000,000 |
| 5,000 | 1,000 | ok | 0.75 s | 5,000,000 |
| 2,000 | 5,000 | **OOM** (256 MiB) | 0.80 s | – |
| 1,000 | 10,000 | **OOM** (256 MiB) | 0.89 s | – |

The wall-clock time is not the problem; the **persistent immutable map** grows
until the execution's 256 MiB limit. So QAL's simulation package is good for
thousands of agents/steps, not millions. A million-agent runtime needs native
mutable state, sharding and a different memory model; QAL should define the
rules/contracts, not interpret every agent.

The core (without `main`) is also `qal/packages/sim_core.qalpkg` (`qal_sim`),
10 definitions; signed install verified, `transfer_conserves` and
`self_transfer_is_noop` pass after import.

## Million-agent sharding

`shard.qal` is one independent shard; `shard.py` runs many processes so each has
its own 256 MiB budget.

```sh
./bench/sim/shard.py --shards 100 --agents 10000 --steps 100
# {"shards":100,"agents":10000,"steps":100,"total_agents":1000000,
#  "total":1000000000,"expected":1000000000,"seconds":34.81}
```

- **1,000,000 agents**, 100 steps each, 34.81 s,
- total conserved: **1,000,000,000**,
- each shard is a fresh QAL execution, so the persistent-map OOM is avoided.

Caveat: shards are independent; transfers do not cross shard boundaries. A
production sharded runtime must handle cross-shard messages explicitly.
