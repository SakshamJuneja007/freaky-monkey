# P2.5 Dependency-Aware Workflow Execution

P2.5 keeps one DEIMOS orchestrator and adds dependency-aware scheduling above the existing policy, execution, BrowserSkill, verification, recovery, and LangGraph layers.

## Scheduling rules

- `WorkflowStep` is the backward-compatible TaskNode representation. It exposes stable `node_id`, intent/capability/arguments, dependencies, dependency reasons/types, policy state, result, verification, recovery, and lifecycle state.
- Explicit `then`/`after that` ordering is preserved. Linguistic adjacency alone does not create a dependency.
- State dependencies (for example launch-app -> type), explicit data dependencies, and authentication/session dependencies are represented separately.
- Exclusive resources are protected by `ResourceLockManager`; the scheduler never creates a second browser mutation path.
- Independent resource-safe branches can execute concurrently. Same-resource branches serialize.
- Maximum workflow concurrency is bounded by `DEIMOS_WORKFLOW_MAX_CONCURRENCY`.

The default is **2 concurrent branches**. Values are clamped to the safe range 1..32.

## Safety invariants

Approval is not execution success; execution success is not verification success. Only a verified `PASS` becomes `COMPLETED`. `FAILED`/`UNKNOWN` branches never become runnable again without an explicit recovery transition. Completed side effects are never replayed by the scheduler.

The LangGraph graph retains the durable workflow thread/checkpoint identity. P2.5 adds a batch execution node for policy-allowed ready branches; approval-required branches remain on the existing approval interrupt path.
