# Durable Kanban Goals

Durable Kanban goals are an explicit gateway-only opt-in:

```text
/goal durable <objective>
```

Legacy `/goal <text>` behavior is unchanged. The durable form records the
gateway origin, creates one Kanban BUILD task, and relies on the gateway Kanban
watcher to advance a deterministic supervisor state machine. It does not enqueue
a synthetic chat turn.

Required configuration:

```yaml
kanban:
  durable_goals:
    board: turpi-v2
    builder_profile: turpi_builder
    verifier_profile: turpi_verify
    reviewer_profile: turpi_review
    reviewer_skill_digest: 1e74b219dbf5377886fde11fcc873c673d3c01178a007ed9667a0942f8f10ec1
    repair_budget: 1
    review_retry_budget: 1
```

The selected board must have an existing absolute Git checkout configured as
its `default_workdir`. Each stage inherits the same resolved worktree/branch
lineage. A route with no known gateway destination, no board checkout, invalid
budget, or an incompatible dispatcher lease fails without creating a goal or
worker task.

## State and authority

The deterministic stages are `BUILD -> REPAIR_BUILD_n -> VERIFY -> REVIEW ->
READY_FOR_OWNER`; repair is omitted when unnecessary and malformed reviews may
use bounded `REVIEW_RETRY_n` stages. Attempt reservation and successor creation
commit in the same CAS transaction, so a crash cannot refund an attempt or mint
a duplicate successor. `BLOCKED_CAPABILITY`, `BLOCKED`, and `READY_FOR_OWNER`
are durable database states.

Workers must complete with `metadata.durable_goal` structured payloads. The
supervisor rejects prose verdicts, stale run ids, non-positive run ids, malformed
payloads, candidate SHA mismatches, foreign boards, and protocol version skew.
Review tasks additionally require reviewer dispatch-role provenance and the
reviewer's isolated `immutable-change-reviews` skill snapshot matching the
pinned `reviewer_skill_digest`. Capability preflight runs before claim/spawn;
failure creates no run, preserves the prerequisite as a durable block, and
enqueues one owner notification.

`READY_FOR_OWNER` grants no Git or deployment authority. After separately
performing or observing the owner action, the exact originating chat may record
final completion for the exact candidate:

```text
/goal durable complete <goal-id> <candidate-sha>
```

This command only records `COMPLETED`; it does not merge, push, or deploy.

Terminal owner states are delivered through the durable goal notification
outbox. Delivery is intentionally separate from state transition so restart or
replay at successor creation and pre-delivery boundaries cannot create duplicate
successors or outbox rows. Notifications are limited to blocking/major findings,
unrecoverable or capability failures, `READY_FOR_OWNER`, and final completion.
Outbox claims have a bounded lease; if a gateway crashes after reserving a row
but before calling `adapter.send`, a restarted watcher can reclaim the expired
row and send it once. If the adapter send succeeds and the process crashes
before the durable ack, delivery is necessarily at-least-once unless that
adapter provides its own idempotent send key.

## Mixed-version rollout gate

Every gateway instance that can acquire the machine-wide Kanban dispatcher lock
must be upgraded before `kanban.durable_goals` is enabled. Mixed lock-eligible
gateway versions are intentionally fail-closed: private task states protect
already-created supervised tasks from older dispatchers, while the runtime
lease/config gate prevents new durable goal creation under a stale owner.

1. Deploy/start the upgraded singleton gateway dispatcher first.
2. Wait for it to acquire the existing machine-wide dispatcher lock and write a
   fresh `kanban_goal_runtime` lease with the exact schema and protocol version.
3. Only then enable or use `/goal durable`.

The public route checks the lease before inserting anything. Supervised tasks
use private `durable_ready`/`durable_review` queue states, so an older dispatcher
cannot claim them. The upgraded dispatcher independently checks board,
protocol, goal state, current run, role, and required skill before every spawn;
it does not infer compatibility from profile identity. Ordinary `/goal` and
ordinary Kanban task states retain their existing behavior.
