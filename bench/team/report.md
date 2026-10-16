# QAL Phase 4 Benchmark Report: Team vs. Solo

**Date:** 2026-10-16  
**Service:** `shop` (REST API with SQLite persistence, inventory reservation, and rollback)  
**Binary:** `qal`  
**Evaluation Scope:** 2 Team runs (6 agents: `arch-1`, `impl-1..3`, `ver-1`, `rev-1`) vs. 2 Solo runs (1 agent)  

## Summary of Results

| Run | Arm | Agents | Wall Time (s) | Tool Calls | Failed Calls | Tokens (est.) | Active Writers (5 min) | Replayed Merges | Gate Approvals | Scenarios (20) | Verify Crypto |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Team | 6 | 86.08s | 31 | 0 | 12,531 | 3 | 2 | 4/4 | 20/20 | OK |
| 2 | Team | 6 | 90.44s | 31 | 0 | 12,531 | 3 | 2 | 4/4 | 20/20 | OK |
| 1 | Solo | 1 | 79.05s | 2 | 1 | 7,952 | 1 | 0 | N/A | 20/20 | OK |
| 2 | Solo | 1 | 71.36s | 2 | 1 | 7,952 | 1 | 0 | N/A | 20/20 | OK |

### Averages

| Arm | Avg Wall Time (s) | Avg Tool Calls | Avg Tokens (est.) | Scenarios Passed |
|---|---|---|---|---|
| **Team (6 agents)** | **88.26s** | **31.0** | **12,531** | **100% (20/20)** |
| **Solo (1 agent)** | **75.20s** | **2.0** | **7,952** | **100% (20/20)** |

---

## Criterion 6 Fulfillment Evidence

1. **Scenarios & HTTP Harness:** All 20 scenarios (5 public `p01..p05`, 15 hidden `h01..h15`) pass 100% in all runs.
2. **Concurrency / Parallel Writers:** At least 3 members (`impl-1`, `impl-2`, `impl-3`) authored write records concurrently within a 5-minute window.
3. **Automatic Replayed Merge:** Non-fast-forward merges occurred on branches W2 and W3 after W1 advanced `main`. Both were replayed and succeeded automatically without conflict (`kind: "replayed"`).
4. **Gate Compliance:** Every merged change set has verified review (`verified` by `ver-1`) and approval (`approve` by `rev-1`), satisfying the gate policy (§7.4).
5. **Cryptographic Integrity:** `qal verify --trust team.json` successfully verified the entire Ed25519 hash chain of signed records.
6. **Provenance Integrity:** 0 QAL definitions written by controller (`cli`/`owner`). Every definition in `shop.qalpkg` is attributed to a team member identity (`arch-1`, `impl-1`, `impl-2`, `impl-3`, `rev-1`).
