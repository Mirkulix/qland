# Live 6-agent swarm, 2026-10-27

Six DeepSeek agents worked on one QAL team store with real role sessions. Unlike the
scripted `bench/team/run.py`, every action was decided and executed by an LLM agent
through the team MCP tools.

## Roles

| Member | Role | Session |
|---|---|---|
| `arch` | architect | `/tmp/qal-swarm6-keys/arch.session` |
| `impl1` | implementer | `/tmp/qal-swarm6-keys/impl1.session` |
| `impl2` | implementer | `/tmp/qal-swarm6-keys/impl2.session` |
| `ver` | verifier | `/tmp/qal-swarm6-keys/ver.session` |
| `rev` | reviewer | `/tmp/qal-swarm6-keys/rev.session` |
| `obs` | verifier (observer) | `/tmp/qal-swarm6-keys/obs.session` |

Keys and sessions live only in `/tmp`, never in the repository.

## Sequence

1. **arch** opened `base`, defined `fn base() -> Int = 10` + `base_ok`, tested and submitted
   (`c0defc9d4fcbd`).
2. **ver** verified it; **rev** approved and merged (**fast_forward**).
3. **impl1** and **impl2** ran **in parallel** from the same main:
   - `impl1`: `fn double() -> Int = base() * 2` + `double_ok` (`c0e5312393356`),
   - `impl2`: `fn triple() -> Int = base() * 3` + `triple_ok` (`c31901683232e`).
4. **ver** verified both submitted changes.
5. **rev** merged `impl1` (**fast_forward**) and then `impl2` (**replayed**, because main
   had advanced; no conflict, tests re-ran).
6. Independent check:

```json
qal_test main ["base_ok","double_ok","triple_ok"] -> passed 3, failed 0
qal_verify --trust team.json --head b63f7bbb536077dd70d29f7f187843591bf26f7a0755f60f76879ff1b2086b3b
  -> ok=true, records=28, members=7
```

## What this proves

- **Live role protocol:** separate LLM agents, separate signed sessions, verifier gate,
  reviewer merge.
- **Real concurrency:** two implementers worked from the same main state; the second merge
  was **replayed** onto the new main and still passed.
- **Audit:** 28 signed records, 7 members, pinned-head verification green.

## What it does not prove

- No emergence, scheduler, reputation or selection; the agents were coordinated by prompts.
- The store and keys are temporary (`/tmp/qal-swarm6*`); only this report is committed.
- The observer role was configured but did not add a distinct audit step in this run.
