"""Durable VERIFY_PROMOTE operation executor/replay contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_goal_supervisor as supervisor
from hermes_cli.kanban_goal_operations import (
    OperationState,
    claim_due_operation,
    execute_due_operation_once,
    get_operation,
)
from hermes_cli.kanban_goal_supervisor import (
    DURABLE_GOAL_PROTOCOL_VERSION,
    GoalOrigin,
    create_trusted_durable_goal,
    get_durable_goal,
    list_durable_goal_tasks,
    list_goal_notifications,
    supervise_goal_once,
)
from hermes_cli.kanban_trusted_stages import (
    PromotionEvidence,
    PromotionRequest,
    ResultClassification,
    TaskContract,
    TrustedStageRegistry,
)


def _contract() -> TaskContract:
    return TaskContract.create(
        reference="issue:223b",
        objective="Durable trusted operation",
        base_revision="1" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    )


def _create_goal(conn, contract: TaskContract):
    return create_trusted_durable_goal(
        conn,
        contract=contract,
        origin=GoalOrigin(platform="telegram", chat_id="owner"),
        board="default",
        resolver_id="fixture-resolver",
        verify_promote_adapter_id="fixture-adapter",
        orchestrator_profile="orchestrator",
        builder_profile="builder",
        reviewer_profile="reviewer",
        reviewer_skill_digest="f" * 64,
        repair_budget=1,
        review_retry_budget=1,
    )


def _complete_bound_task(
    conn,
    *,
    task_id: str,
    profile: str,
    contract: TaskContract,
    stage: str,
    fields: dict[str, object],
) -> None:
    task = (
        kb.claim_review_task(conn, task_id, claimer=profile, allow_durable=True)
        if stage == "REVIEW"
        else kb.claim_task(conn, task_id, claimer=profile)
    )
    assert task is not None and task.current_run_id is not None
    payload = {
        "workflow_version": 2,
        "protocol_version": DURABLE_GOAL_PROTOCOL_VERSION,
        "stage": stage,
        "run_id": task.current_run_id,
        "contract_hash": contract.contract_hash,
        "base_revision": contract.base_revision,
        "scope": list(contract.scope),
        "gates": list(contract.gates),
        "authority": profile,
        **fields,
    }
    assert kb.complete_task(
        conn,
        task.id,
        summary=f"completed {stage}",
        metadata={"durable_goal": payload},
        expected_run_id=task.current_run_id,
    )


def _build_fields(
    *,
    candidate_sha: str = "2" * 40,
    candidate_tree: str = "3" * 40,
    branch_identity: str = "turpi/223b",
    pr_identity: str = "pr-223b",
    remote_base: str = "1" * 40,
    remote_head: str | None = None,
    remote_tree: str = "3" * 40,
    deterministic_gate_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "candidate_sha": candidate_sha,
        "candidate_tree": candidate_tree,
        "branch_identity": branch_identity,
        "pr_identity": pr_identity,
        "remote_base": remote_base,
        "remote_head": remote_head or candidate_sha,
        "remote_tree": remote_tree,
        "deterministic_gate_evidence": deterministic_gate_evidence
        or {"focused-tests": "passed"},
    }


def _create_waiting_operation(
    conn,
    *,
    candidate_sha: str = "2" * 40,
    build_fields: dict[str, object] | None = None,
    expect_waiting: bool = True,
):
    contract = _contract()
    created = _create_goal(conn, contract)
    _complete_bound_task(
        conn,
        task_id=created.task_id,
        profile="orchestrator",
        contract=contract,
        stage="PLAN",
        fields={"decision": "PLAN_ACCEPTED"},
    )
    build_id = supervise_goal_once(
        conn,
        created.goal_id,
        board="default",
        runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
    ).task_id
    assert build_id is not None
    _complete_bound_task(
        conn,
        task_id=build_id,
        profile="builder",
        contract=contract,
        stage="BUILD_CANDIDATE",
        fields=build_fields or _build_fields(candidate_sha=candidate_sha),
    )
    result = supervise_goal_once(
        conn,
        created.goal_id,
        board="default",
        runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
    )
    if expect_waiting:
        assert result.action == "AWAITING_TRUSTED_RESULT"
    return created, contract


def _setup_home(tmp_path, monkeypatch) -> None:
    home = tmp_path / ".hermes"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()


def test_schema_and_handoff_create_stable_idempotent_operation(tmp_path, monkeypatch):
    _setup_home(tmp_path, monkeypatch)

    with kb.connect() as conn:
        created, contract = _create_waiting_operation(conn)
        row = conn.execute(
            "SELECT * FROM kanban_goal_operations WHERE goal_id = ?",
            (created.goal_id,),
        ).fetchone()
        duplicate = kb.create_or_reuse_goal_operation(
            conn,
            goal_id=created.goal_id,
            kind="VERIFY_PROMOTE",
            stage_attempt=0,
            request_payload=json.loads(row["request_payload"]),
        )
        count = conn.execute(
            "SELECT COUNT(*) FROM kanban_goal_operations WHERE goal_id = ?",
            (created.goal_id,),
        ).fetchone()[0]

    assert row is not None
    assert row["operation_id"] == duplicate["operation_id"]
    assert row["kind"] == "VERIFY_PROMOTE"
    assert row["state"] == OperationState.PENDING.value
    assert row["request_hash"] == duplicate["request_hash"]
    assert count == 1
    assert contract.contract_hash in row["request_payload"]


def test_operation_schema_version_blocks_active_pr4_goal_once(tmp_path, monkeypatch):
    _setup_home(tmp_path, monkeypatch)
    assert supervisor.DURABLE_GOAL_SCHEMA_VERSION == 3
    assert "apply_trusted_stage_result" not in supervisor.__all__

    with kb.connect() as conn:
        created = _create_goal(conn, _contract())
        conn.execute(
            "UPDATE kanban_goals SET schema_version = 2 WHERE id = ?",
            (created.goal_id,),
        )
        conn.commit()

        first = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        replay = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)

    assert first.action == "HUMAN_GATE"
    assert first.reason == "durable goal schema mismatch: goal=2, runtime=3"
    assert replay.action == "NOOP" and replay.reason == "terminal_goal"
    assert goal is not None and goal.status == "BLOCKED"
    assert goal.current_stage == "HUMAN_GATE"
    assert [item.kind for item in notifications] == ["HUMAN_GATE"]


def test_build_handoff_requires_exact_canonical_operation_fields(tmp_path, monkeypatch):
    _setup_home(tmp_path, monkeypatch)
    required = {
        "candidate_sha",
        "candidate_tree",
        "branch_identity",
        "pr_identity",
        "remote_base",
        "remote_head",
        "remote_tree",
    }
    for missing in sorted(required):
        with kb.connect() as conn:
            fields = _build_fields()
            fields.pop(missing)
            created, _contract = _create_waiting_operation(
                conn,
                build_fields=fields,
                expect_waiting=False,
            )
            goal = get_durable_goal(conn, created.goal_id)
            operations = conn.execute(
                "SELECT COUNT(*) FROM kanban_goal_operations WHERE goal_id = ?",
                (created.goal_id,),
            ).fetchone()[0]
            outbox = conn.execute(
                "SELECT COUNT(*) FROM kanban_goal_notification_outbox WHERE goal_id = ?",
                (created.goal_id,),
            ).fetchone()[0]
        assert goal is not None
        assert goal.status == "BLOCKED", missing
        assert operations == 0, missing
        assert outbox == 1, missing


def test_build_handoff_rejects_foreign_remote_readback(tmp_path, monkeypatch):
    cases = {
        "remote_base": "9" * 40,
        "remote_head": "8" * 40,
        "remote_tree": "7" * 40,
    }
    for field, foreign_value in cases.items():
        _setup_home(tmp_path / field, monkeypatch)
        with kb.connect() as conn:
            fields = _build_fields()
            fields[field] = foreign_value
            created, _contract = _create_waiting_operation(
                conn,
                build_fields=fields,
                expect_waiting=False,
            )
            goal = get_durable_goal(conn, created.goal_id)
            operations = conn.execute(
                "SELECT COUNT(*) FROM kanban_goal_operations WHERE goal_id = ?",
                (created.goal_id,),
            ).fetchone()[0]
            outbox = conn.execute(
                "SELECT COUNT(*) FROM kanban_goal_notification_outbox WHERE goal_id = ?",
                (created.goal_id,),
            ).fetchone()[0]
        assert goal is not None and goal.status == "BLOCKED", field
        assert operations == 0, field
        assert outbox == 1, field


def test_operation_request_uses_task_contract_base_revision_only(tmp_path, monkeypatch):
    _setup_home(tmp_path, monkeypatch)
    with kb.connect() as conn:
        _created, contract = _create_waiting_operation(conn)
        row = conn.execute("SELECT request_payload FROM kanban_goal_operations").fetchone()
        payload = json.loads(row["request_payload"])
    assert payload["base_revision"] == contract.base_revision
    assert payload["expected_base_sha"] == contract.base_revision
    assert payload["adapter_id"] == "fixture-adapter"
    assert payload["remote_base"] == contract.base_revision
    assert payload["remote_head"] == payload["candidate_sha"]
    assert payload["remote_tree"] == payload["candidate_tree"]
    assert isinstance(payload["task_contract"], dict)
    assert payload["task_contract"]["base_revision"] == contract.base_revision


def test_claim_is_atomic_and_stale_claim_cannot_ack(tmp_path, monkeypatch):
    _setup_home(tmp_path, monkeypatch)

    with kb.connect() as conn:
        _create_waiting_operation(conn)
        first = claim_due_operation(conn, now=100, lease_seconds=30, claim_token="first")
        second = claim_due_operation(conn, now=101, lease_seconds=30, claim_token="second")
        takeover = claim_due_operation(conn, now=131, lease_seconds=30, claim_token="second")
        assert first is not None
        assert second is None
        assert takeover is not None
        stale = kb.ack_goal_operation_result(
            conn,
            operation_id=first.operation_id,
            claim_token="first",
            request_hash=first.request_hash,
            response_payload={"classification": "RETRYABLE", "summary": "stale"},
            now=132,
        )
        row = get_operation(conn, first.operation_id)

    assert stale is None
    assert row is not None
    assert row.state == OperationState.CLAIMED
    assert row.claim_token == "second"


def test_expired_claim_cannot_ack_before_takeover(tmp_path, monkeypatch):
    _setup_home(tmp_path, monkeypatch)
    with kb.connect() as conn:
        _create_waiting_operation(conn)
        claimed = claim_due_operation(
            conn,
            now=100,
            lease_seconds=30,
            claim_token="expired",
        )
        assert claimed is not None
        request = _request_from_claimed(claimed, conn)
        evidence = PromotionEvidence.create(
            adapter_id="fixture-adapter",
            request=request,
            classification=ResultClassification.PASS,
            summary="late ack must not win",
        )
        acked = kb.ack_goal_operation_result(
            conn,
            operation_id=claimed.operation_id,
            claim_token="expired",
            request_hash=claimed.request_hash,
            response_payload=evidence.as_payload(),
            now=131,
        )
        row = get_operation(conn, claimed.operation_id)

    assert acked is None
    assert row is not None
    assert row.state == OperationState.CLAIMED
    assert row.claim_token == "expired"


def test_two_connection_claim_takeover_and_stale_ack_are_race_safe(
    tmp_path, monkeypatch
):
    _setup_home(tmp_path, monkeypatch)
    with kb.connect() as setup:
        _create_waiting_operation(setup)
    with kb.connect() as conn_a, kb.connect() as conn_b:
        first = claim_due_operation(conn_a, now=100, lease_seconds=30, claim_token="a")
        concurrent = claim_due_operation(
            conn_b, now=100, lease_seconds=30, claim_token="b"
        )
        takeover = claim_due_operation(
            conn_b, now=131, lease_seconds=30, claim_token="b"
        )
        assert first is not None
        assert concurrent is None
        assert takeover is not None
        request = _request_from_claimed(takeover, conn_b)
        evidence = PromotionEvidence.create(
            adapter_id="fixture-adapter",
            request=request,
            classification=ResultClassification.PASS,
            summary="winner",
        )
        stale = kb.ack_goal_operation_result(
            conn_a,
            operation_id=first.operation_id,
            claim_token="a",
            request_hash=first.request_hash,
            response_payload=evidence.as_payload(),
            now=132,
        )
        winner = kb.ack_goal_operation_result(
            conn_b,
            operation_id=takeover.operation_id,
            claim_token="b",
            request_hash=takeover.request_hash,
            response_payload=evidence.as_payload(),
            now=133,
        )
    assert stale is None
    assert winner is not None
    assert winner["state"] == OperationState.SUCCEEDED.value


def test_retryable_backoff_is_durable_bounded_exponential(tmp_path, monkeypatch):
    _setup_home(tmp_path, monkeypatch)
    with kb.connect() as conn:
        _create_waiting_operation(conn)
        for attempt, expected_delay in ((1, 60), (2, 120), (3, 240), (8, 3600)):
            claimed = claim_due_operation(
                conn,
                now=1000,
                lease_seconds=30,
                claim_token=f"claim-{attempt}",
            )
            assert claimed is not None
            conn.execute(
                "UPDATE kanban_goal_operations SET attempt_count = ? WHERE operation_id = ?",
                (attempt, claimed.operation_id),
            )
            request = _request_from_claimed(claimed, conn)
            evidence = PromotionEvidence.create(
                adapter_id="fixture-adapter",
                request=request,
                classification=ResultClassification.RETRYABLE,
                summary=f"transient {attempt}",
            )
            row = kb.ack_goal_operation_result(
                conn,
                operation_id=claimed.operation_id,
                claim_token=f"claim-{attempt}",
                request_hash=claimed.request_hash,
                response_payload=evidence.as_payload(),
                now=1000,
            )
            assert row is not None
            assert row["state"] == "RETRYABLE"
            assert row["next_attempt_at"] == 1000 + expected_delay
            conn.execute(
                "UPDATE kanban_goal_operations SET state = 'PENDING', next_attempt_at = 1000 "
                "WHERE operation_id = ?",
                (claimed.operation_id,),
            )


def _request_from_claimed(claimed, conn) -> PromotionRequest:
    row = conn.execute(
        "SELECT request_payload FROM kanban_goal_operations WHERE operation_id = ?",
        (claimed.operation_id,),
    ).fetchone()
    payload = json.loads(row["request_payload"])
    return PromotionRequest(
        contract_hash=payload["contract_hash"],
        base_revision=payload["base_revision"],
        scope=tuple(payload["scope"]),
        gates=tuple(payload["gates"]),
        candidate_sha=payload["candidate_sha"],
        attempt=payload["stage_attempt"],
        operation_id=claimed.operation_id,
        request_hash=claimed.request_hash,
        contract_version=payload["contract_version"],
        candidate_tree=payload["candidate_tree"],
        branch_identity=payload["branch_identity"],
        pr_identity=payload["pr_identity"],
        remote_base=payload["remote_base"],
        remote_head=payload["remote_head"],
        remote_tree=payload["remote_tree"],
        gate_evidence=payload["deterministic_gate_evidence"],
    )


class _PassAdapter:
    adapter_id = "fixture-adapter"

    def __init__(self) -> None:
        self.calls = 0

    def classify(self, request: PromotionRequest) -> PromotionEvidence:
        self.calls += 1
        return PromotionEvidence.create(
            adapter_id=self.adapter_id,
            request=request,
            classification=ResultClassification.PASS,
            summary="all deterministic gates passed",
        )


class _LeaseExpiringAdapter:
    adapter_id = "fixture-adapter"

    def classify(self, request: PromotionRequest) -> PromotionEvidence:
        return PromotionEvidence.create(
            adapter_id=self.adapter_id,
            request=request,
            classification=ResultClassification.PASS,
            summary="completed after lease expiry",
        )


def test_executor_rechecks_clock_before_ack_and_rejects_expired_claim(
    tmp_path, monkeypatch
):
    _setup_home(tmp_path, monkeypatch)
    ticks = iter((100, 102))
    registry = TrustedStageRegistry(
        resolvers={}, adapters={"fixture-adapter": _LeaseExpiringAdapter()}
    )
    with kb.connect() as conn:
        created, _contract = _create_waiting_operation(conn)
        result = execute_due_operation_once(
            conn,
            registry=registry,
            clock=lambda: next(ticks),
            lease_seconds=1,
            token_factory=lambda: "expired-claim",
        )
        row = conn.execute(
            "SELECT * FROM kanban_goal_operations WHERE goal_id = ?",
            (created.goal_id,),
        ).fetchone()

    assert result is None
    assert row is not None
    assert row["state"] == OperationState.CLAIMED.value
    assert row["response_payload"] is None


class _SequenceAdapter:
    adapter_id = "fixture-adapter"

    def __init__(self, classifications: list[ResultClassification]) -> None:
        self.classifications = list(classifications)
        self.requests: list[PromotionRequest] = []

    def classify(self, request: PromotionRequest) -> PromotionEvidence:
        self.requests.append(request)
        classification = self.classifications.pop(0)
        summary = {
            ResultClassification.PASS: "pass",
            ResultClassification.REPAIRABLE_FAILURE: "repairable",
            ResultClassification.HARD_BLOCK: "hard block",
            ResultClassification.RETRYABLE: "retryable",
        }[classification]
        return PromotionEvidence.create(
            adapter_id=self.adapter_id,
            request=request,
            classification=classification,
            summary=summary,
        )


def _review_and_ready(conn, *, created, contract: TaskContract, candidate_sha: str):
    review_id = supervise_goal_once(
        conn,
        created.goal_id,
        board="default",
        runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
    ).task_id
    assert review_id is not None
    _complete_bound_task(
        conn,
        task_id=review_id,
        profile="reviewer",
        contract=contract,
        stage="REVIEW",
        fields={
            "candidate_sha": candidate_sha,
            "verdict": "APPROVE",
            "findings": [],
            "reviewer_skill_digest": "f" * 64,
        },
    )
    adjudicate_id = supervise_goal_once(
        conn,
        created.goal_id,
        board="default",
        runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
    ).task_id
    assert adjudicate_id is not None
    _complete_bound_task(
        conn,
        task_id=adjudicate_id,
        profile="orchestrator",
        contract=contract,
        stage="ADJUDICATE",
        fields={"candidate_sha": candidate_sha, "decision": "READY_FOR_OWNER"},
    )
    ready = supervise_goal_once(
        conn,
        created.goal_id,
        board="default",
        runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
    )
    replay = supervise_goal_once(
        conn,
        created.goal_id,
        board="default",
        runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
    )
    return ready, replay


def test_synthetic_e2e_happy_operation_to_ready_once(tmp_path, monkeypatch):
    _setup_home(tmp_path, monkeypatch)
    adapter = _SequenceAdapter([ResultClassification.PASS])
    registry = TrustedStageRegistry(resolvers={}, adapters={"fixture-adapter": adapter})
    with kb.connect() as conn:
        created, contract = _create_waiting_operation(conn, candidate_sha="4" * 40)
        executed = execute_due_operation_once(
            conn, registry=registry, now=200, token_factory=lambda: "claim"
        )
        review = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        assert executed is not None and executed.state == OperationState.SUCCEEDED
        assert review.action == "CREATED_REVIEW"
        ready, replay = _review_and_ready(
            conn, created=created, contract=contract, candidate_sha="4" * 40
        )
        goal = get_durable_goal(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)
        bindings = list_durable_goal_tasks(conn, created.goal_id)
    assert ready.action == "READY_FOR_OWNER"
    assert replay.action == "NOOP"
    assert goal is not None and goal.status == "READY_FOR_OWNER"
    assert [item.kind for item in notifications] == ["READY_FOR_OWNER"]
    assert [binding.stage for binding in bindings] == [
        "PLAN",
        "BUILD_CANDIDATE",
        "REVIEW",
        "ADJUDICATE",
    ]
    assert len(adapter.requests) == 1


def test_synthetic_e2e_repair_uses_distinct_stable_operation_attempt(
    tmp_path, monkeypatch
):
    _setup_home(tmp_path, monkeypatch)
    adapter = _SequenceAdapter(
        [ResultClassification.REPAIRABLE_FAILURE, ResultClassification.PASS]
    )
    registry = TrustedStageRegistry(resolvers={}, adapters={"fixture-adapter": adapter})
    candidate_a = "4" * 40
    candidate_b = "5" * 40
    with kb.connect() as conn:
        created, contract = _create_waiting_operation(conn, candidate_sha=candidate_a)
        first = execute_due_operation_once(
            conn, registry=registry, now=200, token_factory=lambda: "claim-a"
        )
        adjudicate = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        assert first is not None and first.state == OperationState.SUCCEEDED
        assert adjudicate.action == "CREATED_ADJUDICATE"
        _complete_bound_task(
            conn,
            task_id=adjudicate.task_id,
            profile="orchestrator",
            contract=contract,
            stage="ADJUDICATE",
            fields={"candidate_sha": candidate_a, "decision": "REPAIR"},
        )
        repair = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        assert repair.action == "CREATED_REPAIR"
        _complete_bound_task(
            conn,
            task_id=repair.task_id,
            profile="builder",
            contract=contract,
            stage="REPAIR_BUILD_1",
            fields=_build_fields(
                candidate_sha=candidate_b,
                candidate_tree="6" * 40,
                branch_identity="turpi/repair",
                pr_identity="pr-repair",
                remote_head=candidate_b,
                remote_tree="6" * 40,
            )
            | {"input_candidate_sha": candidate_a},
        )
        waiting_b = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        second = execute_due_operation_once(
            conn, registry=registry, now=300, token_factory=lambda: "claim-b"
        )
        review_b = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        ready, replay = _review_and_ready(
            conn, created=created, contract=contract, candidate_sha=candidate_b
        )
        operations = conn.execute(
            "SELECT operation_id, stage_attempt FROM kanban_goal_operations "
            "WHERE goal_id = ? ORDER BY stage_attempt",
            (created.goal_id,),
        ).fetchall()
        notifications = list_goal_notifications(conn, created.goal_id)
    assert waiting_b.action == "AWAITING_TRUSTED_RESULT"
    assert second is not None and second.state == OperationState.SUCCEEDED
    assert review_b.action == "CREATED_REVIEW"
    assert ready.action == "READY_FOR_OWNER"
    assert replay.action == "NOOP"
    assert [(row["stage_attempt"]) for row in operations] == [0, 1]
    assert operations[0]["operation_id"] != operations[1]["operation_id"]
    assert [request.candidate_sha for request in adapter.requests] == [
        candidate_a,
        candidate_b,
    ]
    assert [item.kind for item in notifications] == ["READY_FOR_OWNER"]


def test_restart_after_operation_succeeded_before_review_dedupes_review_and_outbox(
    tmp_path, monkeypatch
):
    _setup_home(tmp_path, monkeypatch)
    adapter = _SequenceAdapter([ResultClassification.PASS])
    registry = TrustedStageRegistry(resolvers={}, adapters={"fixture-adapter": adapter})
    with kb.connect() as conn:
        created, contract = _create_waiting_operation(conn, candidate_sha="7" * 40)
        executed = execute_due_operation_once(
            conn, registry=registry, now=200, token_factory=lambda: "claim"
        )
        assert executed is not None and executed.state == OperationState.SUCCEEDED
    with kb.connect() as restarted:
        first = supervise_goal_once(
            restarted,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        second = supervise_goal_once(
            restarted,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        ready, replay = _review_and_ready(
            restarted, created=created, contract=contract, candidate_sha="7" * 40
        )
        bindings = list_durable_goal_tasks(restarted, created.goal_id)
        notifications = list_goal_notifications(restarted, created.goal_id)
    assert first.action == "CREATED_REVIEW"
    assert second.action == "NOOP"
    assert ready.action == "READY_FOR_OWNER"
    assert replay.action == "NOOP"
    assert [binding.stage for binding in bindings].count("REVIEW") == 1
    assert [item.kind for item in notifications] == ["READY_FOR_OWNER"]


class _WrongCandidateAdapter:
    adapter_id = "fixture-adapter"

    def classify(self, request: PromotionRequest) -> PromotionEvidence:
        wrong = PromotionRequest(
            contract_hash=request.contract_hash,
            base_revision=request.base_revision,
            scope=request.scope,
            gates=request.gates,
            candidate_sha="9" * 40,
            attempt=request.attempt,
            operation_id=request.operation_id,
            request_hash=request.request_hash,
            contract_version=request.contract_version,
            candidate_tree=request.candidate_tree,
            branch_identity=request.branch_identity,
            pr_identity=request.pr_identity,
            remote_base=request.remote_base,
            remote_head=request.remote_head,
            remote_tree=request.remote_tree,
            gate_evidence=request.gate_evidence,
        )
        return PromotionEvidence.create(
            adapter_id=self.adapter_id,
            request=wrong,
            classification=ResultClassification.PASS,
            summary="wrong candidate",
        )


def test_execute_due_operation_pass_acks_and_review_is_replay_safe(tmp_path, monkeypatch):
    _setup_home(tmp_path, monkeypatch)
    adapter = _PassAdapter()
    registry = TrustedStageRegistry(resolvers={}, adapters={"fixture-adapter": adapter})

    with kb.connect() as conn:
        created, _contract = _create_waiting_operation(conn)
        first = execute_due_operation_once(
            conn,
            registry=registry,
            now=200,
            token_factory=lambda: "claim-pass",
        )
        second = execute_due_operation_once(
            conn,
            registry=registry,
            now=201,
            token_factory=lambda: "claim-dup",
        )
        supervised = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        again = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(conn, created.goal_id)
        bindings = list_durable_goal_tasks(conn, created.goal_id)

    assert first is not None and first.state == OperationState.SUCCEEDED
    assert second is None
    assert supervised.action == "CREATED_REVIEW"
    assert again.action == "NOOP"
    assert goal is not None and goal.current_stage == "REVIEW"
    assert [binding.stage for binding in bindings].count("REVIEW") == 1
    assert adapter.calls == 1


def test_execute_due_operation_wrong_readback_fails_closed_without_review(
    tmp_path, monkeypatch
):
    _setup_home(tmp_path, monkeypatch)
    registry = TrustedStageRegistry(
        resolvers={}, adapters={"fixture-adapter": _WrongCandidateAdapter()}
    )

    with kb.connect() as conn:
        created, _contract = _create_waiting_operation(conn)
        result = execute_due_operation_once(
            conn,
            registry=registry,
            now=300,
            token_factory=lambda: "claim-wrong",
        )
        supervised = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(conn, created.goal_id)
        bindings = list_durable_goal_tasks(conn, created.goal_id)
        outbox_count = conn.execute(
            "SELECT COUNT(*) FROM kanban_goal_notification_outbox WHERE goal_id = ?",
            (created.goal_id,),
        ).fetchone()[0]

    assert result is not None and result.state == OperationState.BLOCKED
    assert supervised.action == "HUMAN_GATE"
    assert goal is not None and goal.status == "BLOCKED"
    assert [binding.stage for binding in bindings] == ["PLAN", "BUILD_CANDIDATE"]
    assert outbox_count == 1


def test_ack_revalidates_every_authoritative_response_field(tmp_path, monkeypatch):
    _setup_home(tmp_path, monkeypatch)
    fields = (
        ("adapter_id", "other-adapter"),
        ("contract_hash", "9" * 64),
        ("contract_version", 99),
        ("base_revision", "8" * 40),
        ("scope", ["other/"]),
        ("gates", ["other-gate"]),
        ("candidate_sha", "7" * 40),
        ("candidate_tree", "6" * 40),
        ("branch_identity", "other-branch"),
        ("pr_identity", "other-pr"),
        ("remote_base", "origin/other-base"),
        ("remote_head", "origin/other-head"),
        ("remote_tree", "5" * 40),
        ("gate_evidence", {"focused-tests": "failed"}),
    )
    for field, bad_value in fields:
        with kb.connect() as conn:
            _create_waiting_operation(conn)
            claimed = claim_due_operation(conn, now=10, claim_token=f"claim-{field}")
            assert claimed is not None
            request = _request_from_claimed(claimed, conn)
            payload = PromotionEvidence.create(
                adapter_id="fixture-adapter",
                request=request,
                classification=ResultClassification.PASS,
                summary="pass",
            ).as_payload()
            payload[field] = bad_value
            payload.pop("evidence_hash", None)
            payload["evidence_hash"] = "0" * 64
            row = kb.ack_goal_operation_result(
                conn,
                operation_id=claimed.operation_id,
                claim_token=f"claim-{field}",
                request_hash=claimed.request_hash,
                response_payload=payload,
                now=11,
            )
            current = get_operation(conn, claimed.operation_id)
        assert row is None, field
        assert current is not None and current.state == OperationState.CLAIMED, field


class _TransactionAssertingAdapter:
    adapter_id = "fixture-adapter"

    def __init__(self, conn) -> None:
        self.conn = conn

    def classify(self, request: PromotionRequest) -> PromotionEvidence:
        assert self.conn.in_transaction is False
        return PromotionEvidence.create(
            adapter_id=self.adapter_id,
            request=request,
            classification=ResultClassification.PASS,
            summary="no write transaction at adapter boundary",
        )


def test_adapter_invocation_is_outside_sqlite_transaction(tmp_path, monkeypatch):
    _setup_home(tmp_path, monkeypatch)
    with kb.connect() as conn:
        _create_waiting_operation(conn)
        registry = TrustedStageRegistry(
            resolvers={},
            adapters={"fixture-adapter": _TransactionAssertingAdapter(conn)},
        )
        result = execute_due_operation_once(
            conn, registry=registry, now=20, token_factory=lambda: "claim"
        )
    assert result is not None
    assert result.state == OperationState.SUCCEEDED


class _Crash(BaseException):
    pass


class _StatefulFakeRemoteAdapter:
    adapter_id = "fixture-adapter"

    def __init__(self, *, crash_at: str | None = None, collision: bool = False) -> None:
        self.branches: dict[str, dict[str, str]] = {}
        self.prs: dict[str, dict[str, str]] = {}
        self.calls = 0
        self.branch_creates = 0
        self.pr_creates = 0
        self.crash_at = crash_at
        self.collision = collision

    def classify(self, request: PromotionRequest) -> PromotionEvidence:
        self.calls += 1
        if self.crash_at == "before_remote" and self.calls == 1:
            raise _Crash("before remote change")
        expected_branch = str(request.branch_identity)
        expected_pr = str(request.pr_identity)
        candidate = str(request.candidate_sha)
        tree = str(request.candidate_tree)
        if self.collision and self.calls == 1:
            self.branches[expected_branch] = {
                "candidate_sha": "0" * 40,
                "candidate_tree": "0" * 40,
                "operation_id": "other-operation",
            }
        branch = self.branches.get(expected_branch)
        if branch is None:
            self.branches[expected_branch] = {
                "candidate_sha": candidate,
                "candidate_tree": tree,
                "operation_id": str(request.operation_id),
            }
            self.branch_creates += 1
            if self.crash_at == "after_branch" and self.calls == 1:
                raise _Crash("after branch")
        elif branch != {
            "candidate_sha": candidate,
            "candidate_tree": tree,
            "operation_id": str(request.operation_id),
        }:
            return PromotionEvidence.create(
                adapter_id=self.adapter_id,
                request=request,
                classification=ResultClassification.HARD_BLOCK,
                summary="expected branch collision",
            )
        pr = self.prs.get(expected_pr)
        if pr is None:
            self.prs[expected_pr] = {
                "branch_identity": expected_branch,
                "candidate_sha": candidate,
                "candidate_tree": tree,
                "operation_id": str(request.operation_id),
            }
            self.pr_creates += 1
            if self.crash_at == "after_pr" and self.calls == 1:
                raise _Crash("after pr")
        elif pr != {
            "branch_identity": expected_branch,
            "candidate_sha": candidate,
            "candidate_tree": tree,
            "operation_id": str(request.operation_id),
        }:
            return PromotionEvidence.create(
                adapter_id=self.adapter_id,
                request=request,
                classification=ResultClassification.HARD_BLOCK,
                summary="expected pr collision",
            )
        return PromotionEvidence.create(
            adapter_id=self.adapter_id,
            request=request,
            classification=ResultClassification.PASS,
            summary="stable branch and PR readback",
        )


def test_stateful_fake_remote_replays_crash_checkpoints_a_to_d(tmp_path, monkeypatch):
    for crash_at in ("before_remote", "after_branch", "after_pr", None):
        _setup_home(tmp_path / str(crash_at or "done"), monkeypatch)
        adapter = _StatefulFakeRemoteAdapter(crash_at=crash_at)
        registry = TrustedStageRegistry(resolvers={}, adapters={"fixture-adapter": adapter})
        with kb.connect() as conn:
            _create_waiting_operation(conn)
            if crash_at is None:
                first = execute_due_operation_once(
                    conn, registry=registry, now=100, token_factory=lambda: "first"
                )
                assert first is not None and first.state == OperationState.SUCCEEDED
            else:
                with pytest.raises(_Crash):
                    execute_due_operation_once(
                        conn, registry=registry, now=100, token_factory=lambda: "first"
                    )
                claimed = conn.execute(
                    "SELECT * FROM kanban_goal_operations"
                ).fetchone()
                assert claimed["state"] == OperationState.CLAIMED.value
                assert claimed["claim_token"] == "first"
                assert claimed["lease_expires_at"] == 400
                assert execute_due_operation_once(
                    conn, registry=registry, now=399, token_factory=lambda: "too-early"
                ) is None
                second = execute_due_operation_once(
                    conn, registry=registry, now=400, token_factory=lambda: "second"
                )
                assert second is not None and second.state == OperationState.SUCCEEDED
            assert adapter.branch_creates == 1
            assert adapter.pr_creates == 1


def test_stateful_fake_remote_result_done_crash_before_ack_replays_without_duplicates(
    tmp_path, monkeypatch
):
    _setup_home(tmp_path, monkeypatch)
    adapter = _StatefulFakeRemoteAdapter()
    registry = TrustedStageRegistry(resolvers={}, adapters={"fixture-adapter": adapter})
    original_ack = kb.ack_goal_operation_result
    crashed = False

    def crash_before_ack(*args, **kwargs):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise _Crash("after adapter result before ack")
        return original_ack(*args, **kwargs)

    monkeypatch.setattr(kb, "ack_goal_operation_result", crash_before_ack)
    with kb.connect() as conn:
        _create_waiting_operation(conn)
        try:
            execute_due_operation_once(
                conn, registry=registry, now=100, token_factory=lambda: "first"
            )
        except _Crash:
            pass
        else:
            raise AssertionError("expected crash before ack")
        replay = execute_due_operation_once(
            conn, registry=registry, now=500, token_factory=lambda: "second"
        )
    assert replay is not None and replay.state == OperationState.SUCCEEDED
    assert adapter.branch_creates == 1
    assert adapter.pr_creates == 1


def test_stateful_fake_remote_collision_hard_blocks_without_alternate_identity(
    tmp_path, monkeypatch
):
    _setup_home(tmp_path, monkeypatch)
    adapter = _StatefulFakeRemoteAdapter(collision=True)
    registry = TrustedStageRegistry(resolvers={}, adapters={"fixture-adapter": adapter})
    with kb.connect() as conn:
        _create_waiting_operation(conn)
        result = execute_due_operation_once(
            conn, registry=registry, now=100, token_factory=lambda: "claim"
        )
        supervised = supervise_goal_once(
            conn,
            conn.execute("SELECT id FROM kanban_goals").fetchone()["id"],
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
    assert result is not None and result.state == OperationState.BLOCKED
    assert supervised.action == "HUMAN_GATE"
    assert adapter.branch_creates == 0
    assert adapter.pr_creates == 0
