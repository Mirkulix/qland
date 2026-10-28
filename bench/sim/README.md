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
