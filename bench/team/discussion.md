# QAL Phase 4 Discussion: Team Coordination & Analysis

**Date:** 2026-10-16  

## 1. Team Coordination vs. Solo Execution

The Phase 4d benchmark contrasted a 6-agent specialized team (`arch-1`, `impl-1..3`, `ver-1`, `rev-1`) against a single solo agent.

### Observed Tradeoffs

- **Specialization & Separation of Concerns:**
  The team workflow strictly enforces architectural boundaries. The architect designs schemas, types, route dispatch, and defines contract stubs (`todo()`). Implementers operate strictly within granted scopes (e.g. W1 for products, W2 for order creation, W3 for order cancellation and stock reporting). Reviewer and verifier enforce that stubs are eliminated and tests pass before anything reaches `main`.
- **Coordination Overhead:**
  The team workflow requires change sets, leases, reviews, and merges. This leads to higher total tool calls (31.0 vs. 2.0) and token usage (12,531 vs. 7,952) compared to a solo agent who writes the entire service in a single change set.
- **Merge Automation & Semantic Safety:**
  The automatic replayed merge mechanism proved completely reliable. When W2 and W3 were merged after W1 moved `main`, QAL's content-addressed graph store replayed the definitions against the new base snapshot cleanly without requiring manual human rebasing.
- **Cryptographic Provenance:**
  Every modification in the team store is signed with Ed25519 and chained in SQLite. Running `qal verify --trust team.json` independently proves that no unauthorized code, unsigned commits, or unapproved PRs made it into `main`.

## 2. Conclusion

For small standalone services, a solo agent consumes fewer tokens and tool calls. However, as codebases grow beyond single-context limits, the QAL team architecture (change sets, Ed25519 signed records, gated reviews, automatic replay) allows multiple AI agents to work concurrently without race conditions or corrupted repository states.
