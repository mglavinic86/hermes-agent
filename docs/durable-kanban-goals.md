# Durable Kanban Goals

Durable Kanban goals are an explicit gateway-only opt-in for trusted Workflow
V2 contracts:

```text
/goal durable contract <opaque-ref>
```

Legacy `/goal <text>` behavior is unchanged. `/goal durable complete <goal-id>
<candidate-sha>` is also preserved and only records owner completion; it does
not merge, push, deploy, or run an operation executor.

Free-form `/goal durable <objective>` is not a trusted V2 entrypoint. In
trusted-contract mode it fails closed before creating any goal, task, or outbox
row.

## Configuration

```yaml
kanban:
  durable_goals:
    workflow_version: 2
    start_mode: trusted_contract
    board: default
    task_contract_resolver: local-contract-registry
    verify_promote_adapter: local-verify-promote
    orchestrator_profile: durable_orchestrator
    builder_profile: durable_builder
    reviewer_profile: durable_reviewer
    reviewer_skill_digest: 1e74b219dbf5377886fde11fcc873c673d3c01178a007ed9667a0942f8f10ec1
    repair_budget: 1
    review_retry_budget: 1
```

`task_contract_resolver` and `verify_promote_adapter` are static registry ids
supplied by the gateway runtime. They are not module paths, file paths, shell
commands, URLs, or dynamic imports. Workflow V2 rejects `verifier_profile`: the
only LLM authorities are `orchestrator`, `builder`, and `reviewer`.

The reviewer digest pins the reviewer profile's isolated
`immutable-change-reviews` skill snapshot. Reviewer skills must be profile-local
snapshots, not symlinks or live inheritance from another profile.

## Workflow V2

Workflow V2 starts with a PLAN task assigned to the orchestrator. A successful
PLAN creates `BUILD_CANDIDATE` for the builder. Builder completion enters the
taskless `VERIFY_PROMOTE` boundary; the gateway supervisor does not dispatch a
verifier task. A safe injected/fake boundary applies deterministic
verify/promote evidence with one of these protocol classifications:

- `PASS` creates `REVIEW` for the reviewer.
- `RETRYABLE` leaves the same taskless `VERIFY_PROMOTE` state unchanged.
- `REPAIRABLE_FAILURE` creates `ADJUDICATE` for the orchestrator; only `REPAIR`
  or `HUMAN_GATE` can advance from that evidence.
- `HARD_BLOCK` atomically terminalizes to a human gate without creating a
  builder or reviewer task.

Reviewer verdicts are `APPROVE`, `CHANGES_REQUIRED`, and `BLOCKED`. Finding
severities are `BLOCKER`, `MAJOR`, `MINOR`, and `NIT`. Review completion always
creates `ADJUDICATE`; it never makes a goal ready directly.

Adjudication decisions are `READY_FOR_OWNER`, `REPAIR`, and `HUMAN_GATE`.
`READY_FOR_OWNER` is accepted only for the current goal state when the current
candidate, contract, run, PASS promotion evidence, and APPROVE review all match,
there are no `BLOCKER` or `MAJOR` findings, and no human gate is requested.
`REPAIR` is bounded by `repair_budget` and creates `REPAIR_BUILD_n` for the
builder. `HUMAN_GATE` terminalizes safely and enqueues one logical owner
notification.

## Rollout And Migration

Every gateway instance that can acquire the machine-wide Kanban dispatcher lock
must be upgraded before trusted V2 durable goals are enabled. Creation checks a
fresh `kanban_goal_runtime` lease with the exact schema and protocol version
before inserting anything.

The migration is additive. V1 terminal history remains readable. If an active
V1 durable goal is observed by a V2 supervisor, it fails closed once to a human
gate, enqueues one logical durable owner notification, and never spawns a V2
verifier. Ordinary Kanban tasks are unaffected.
