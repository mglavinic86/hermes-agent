"""Public contracts for the generic durable-goal Workflow V2 boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import sqlite3
from typing import cast

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.config import parse_durable_goals_v2_config
from hermes_cli.kanban_goal_supervisor import (
    DURABLE_GOAL_PROTOCOL_VERSION,
    DURABLE_GOAL_SCHEMA_VERSION,
    GoalOrigin,
    apply_trusted_stage_result,
    create_durable_goal,
    create_trusted_durable_goal,
    get_durable_goal,
    list_durable_goal_tasks,
    list_goal_notifications,
    preflight_durable_dispatch,
    supervise_goal_once,
)
from hermes_cli.kanban_trusted_stages import (
    AdjudicationDecision,
    Authority,
    FindingSeverity,
    PlanDecision,
    PromotionEvidence,
    PromotionRequest,
    ReviewVerdict,
    ResultClassification,
    StaticTaskContractResolver,
    TaskContract,
    TrustedStageRegistry,
)


def _create_v2_goal(conn, contract: TaskContract, *, repair_budget: int = 1):
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
        repair_budget=repair_budget,
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
):
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
    return task


def _create_goal_waiting_for_evidence(
    conn,
    *,
    contract: TaskContract,
    candidate_sha: str,
    repair_budget: int = 1,
):
    created = _create_v2_goal(conn, contract, repair_budget=repair_budget)
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
        fields={"candidate_sha": candidate_sha},
    )
    waiting = supervise_goal_once(
        conn,
        created.goal_id,
        board="default",
        runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
    )
    assert waiting.action == "AWAITING_TRUSTED_RESULT"
    return created


def _create_goal_at_review(
    conn,
    *,
    contract: TaskContract,
    candidate_sha: str,
    repair_budget: int = 1,
):
    created = _create_goal_waiting_for_evidence(
        conn,
        contract=contract,
        candidate_sha=candidate_sha,
        repair_budget=repair_budget,
    )
    evidence = PromotionEvidence.create(
        adapter_id="fixture-adapter",
        request=PromotionRequest(
            contract_hash=contract.contract_hash,
            base_revision=contract.base_revision,
            scope=contract.scope,
            gates=contract.gates,
            candidate_sha=candidate_sha,
            attempt=0,
        ),
        classification=ResultClassification.PASS,
        summary="deterministic gates passed",
    )
    review_id = apply_trusted_stage_result(conn, created.goal_id, evidence).task_id
    assert review_id is not None
    return created, review_id


def _create_goal_at_repair(
    conn,
    *,
    contract: TaskContract,
    candidate_sha: str,
):
    created, review_id = _create_goal_at_review(
        conn,
        contract=contract,
        candidate_sha=candidate_sha,
        repair_budget=1,
    )
    _complete_bound_task(
        conn,
        task_id=review_id,
        profile="reviewer",
        contract=contract,
        stage="REVIEW",
        fields={
            "candidate_sha": candidate_sha,
            "verdict": "CHANGES_REQUIRED",
            "findings": [{"severity": "MAJOR", "summary": "repair required"}],
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
        fields={"candidate_sha": candidate_sha, "decision": "REPAIR"},
    )
    repair_id = supervise_goal_once(
        conn,
        created.goal_id,
        board="default",
        runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
    ).task_id
    assert repair_id is not None
    return created, repair_id


def test_static_resolver_returns_an_immutable_hash_bound_task_contract():
    contract = TaskContract.create(
        reference="issue:synthetic-223-a",
        objective="Implement the accepted synthetic workflow slice",
        base_revision="a" * 40,
        scope=("hermes_cli/", "tests/hermes_cli/"),
        gates=("focused-tests", "no-remote-writes"),
        authority=Authority.ORCHESTRATOR,
    )
    resolver = StaticTaskContractResolver(
        resolver_id="synthetic-static",
        contracts={contract.reference: contract},
    )

    resolved = resolver.resolve(contract.reference)

    assert resolved is contract
    assert resolved.verify_hash()
    assert len(resolved.contract_hash) == 64
    assert tuple(classification.value for classification in ResultClassification) == (
        "PASS",
        "RETRYABLE",
        "REPAIRABLE_FAILURE",
        "HARD_BLOCK",
    )
    with pytest.raises(FrozenInstanceError):
        resolved.objective = "mutated"  # type: ignore[misc]


def test_static_registry_selects_protocol_implementations_by_id_without_executing_them():
    contract = TaskContract.create(
        reference="issue:registry-fixture",
        objective="Prove registry selection is static",
        base_revision="b" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    )
    resolver = StaticTaskContractResolver(
        resolver_id="fixture-resolver",
        contracts={contract.reference: contract},
    )

    class FakeVerifyPromoteAdapter:
        adapter_id = "fixture-adapter"

        def __init__(self) -> None:
            self.calls = 0

        def classify(self, request: PromotionRequest) -> PromotionEvidence:
            self.calls += 1
            return PromotionEvidence.create(
                adapter_id=self.adapter_id,
                request=request,
                classification=ResultClassification.PASS,
                summary="synthetic deterministic pass",
            )

    adapter = FakeVerifyPromoteAdapter()
    registry = TrustedStageRegistry(
        resolvers={resolver.resolver_id: resolver},
        adapters={adapter.adapter_id: adapter},
    )

    assert registry.resolver("fixture-resolver") is resolver
    assert registry.adapter("fixture-adapter") is adapter
    assert adapter.calls == 0
    assert tuple(decision.value for decision in PlanDecision) == (
        "PLAN_ACCEPTED",
        "HUMAN_GATE",
    )
    assert tuple(verdict.value for verdict in ReviewVerdict) == (
        "APPROVE",
        "CHANGES_REQUIRED",
        "BLOCKED",
    )
    assert tuple(severity.value for severity in FindingSeverity) == (
        "BLOCKER",
        "MAJOR",
        "MINOR",
        "NIT",
    )
    assert tuple(decision.value for decision in AdjudicationDecision) == (
        "READY_FOR_OWNER",
        "REPAIR",
        "HUMAN_GATE",
    )


def test_db_boundary_rejects_a_tampered_contract_without_creating_rows(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    contract = TaskContract.create(
        reference="issue:tampered",
        objective="Reject a modified contract",
        base_revision="a" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    ).as_payload()
    contract["objective"] = "Mutated after hashing"

    with kb.connect() as conn:
        with pytest.raises(ValueError, match="contract hash"):
            kb.create_trusted_durable_goal_record(
                conn,
                task_contract=contract,
                board_slug="default",
                origin={"platform": "telegram", "chat_id": "owner"},
                resolver_id="fixture-resolver",
                verify_promote_adapter_id="fixture-adapter",
                orchestrator_profile="orchestrator",
                builder_profile="builder",
                reviewer_profile="reviewer",
                reviewer_skill_digest="f" * 64,
                repair_budget=1,
                review_retry_budget=1,
                schema_version=2,
                protocol_version=2,
                workflow_version=2,
            )
        counts = tuple(
            conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("kanban_goals", "kanban_goal_tasks", "tasks")
        )

    assert counts == (0, 0, 0)


def test_creation_canonicalizes_valid_contract_payload_across_module_identity(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    contract = TaskContract.create(
        reference="issue:reloaded-contract-class",
        objective="Accept only the canonical immutable payload",
        base_revision="1" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    )

    class ContractFromReloadedModule:
        def as_payload(self):
            return contract.as_payload()

    with kb.connect() as conn:
        created = create_trusted_durable_goal(
            conn,
            contract=cast(TaskContract, ContractFromReloadedModule()),
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
        goal = get_durable_goal(conn, created.goal_id)

    assert goal is not None
    assert goal.current_stage == "PLAN"
    assert goal.task_contract == contract


def test_persisted_v2_contract_binding_cannot_be_mutated(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    contract = TaskContract.create(
        reference="issue:immutable",
        objective="Keep the persisted contract immutable",
        base_revision="b" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    )

    with kb.connect() as conn:
        created = _create_v2_goal(conn, contract)
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE kanban_goals SET task_contract_hash = ? WHERE id = ?",
                ("0" * 64, created.goal_id),
            )
        conn.rollback()
        goal = get_durable_goal(conn, created.goal_id)

    assert goal is not None
    assert goal.task_contract is not None
    assert goal.task_contract.contract_hash == contract.contract_hash


def test_origin_message_replay_cannot_rebind_to_a_different_contract(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    first_contract = TaskContract.create(
        reference="issue:first",
        objective="Keep the first binding",
        base_revision="c" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    )
    second_contract = TaskContract.create(
        reference="issue:second",
        objective="Attempt a conflicting replay",
        base_revision="c" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    )
    origin = GoalOrigin(
        platform="telegram",
        chat_id="owner",
        message_id="same-message",
    )

    with kb.connect() as conn:
        created = create_trusted_durable_goal(
            conn,
            contract=first_contract,
            origin=origin,
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
        with pytest.raises(ValueError, match="different Workflow V2 contract"):
            create_trusted_durable_goal(
                conn,
                contract=second_contract,
                origin=origin,
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
        goal = get_durable_goal(conn, created.goal_id)
        goal_count = conn.execute("SELECT COUNT(*) FROM kanban_goals").fetchone()[0]

    assert goal_count == 1
    assert goal is not None and goal.task_contract == first_contract


def test_v2_config_is_generic_and_rejects_legacy_verifier_authority():
    parsed = parse_durable_goals_v2_config({
        "kanban": {
            "durable_goals": {
                "workflow_version": 2,
                "start_mode": "trusted_contract",
                "board": "default",
                "orchestrator_profile": "orchestrator",
                "builder_profile": "builder",
                "reviewer_profile": "reviewer",
                "task_contract_resolver": "fixture-resolver",
                "verify_promote_adapter": "fixture-adapter",
                "reviewer_skill_digest": "a" * 64,
                "repair_budget": 2,
                "review_retry_budget": 1,
            }
        }
    })

    assert parsed["workflow_version"] == 2
    assert parsed["orchestrator_profile"] == "orchestrator"
    assert parsed["verify_promote_adapter"] == "fixture-adapter"
    assert tuple(authority.value for authority in Authority) == (
        "orchestrator",
        "builder",
        "reviewer",
    )
    assert {key for key in parsed if key.endswith("_profile")} == {
        "orchestrator_profile",
        "builder_profile",
        "reviewer_profile",
    }
    assert "verifier_profile" not in parsed

    with pytest.raises(ValueError, match="verifier_profile"):
        parse_durable_goals_v2_config({
            "kanban": {
                "durable_goals": {
                    **parsed,
                    "verifier_profile": "legacy-verifier",
                }
            }
        })


def test_v2_creation_binds_contract_and_starts_one_orchestrator_plan_task(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    contract = TaskContract.create(
        reference="issue:trusted-create",
        objective="Create the bound workflow",
        base_revision="c" * 40,
        scope=("hermes_cli/kanban_goal_supervisor.py",),
        gates=("focused-tests", "owner-completion"),
    )

    with kb.connect() as conn:
        created = create_trusted_durable_goal(
            conn,
            contract=contract,
            origin=GoalOrigin(platform="telegram", chat_id="owner"),
            board="default",
            resolver_id="fixture-resolver",
            verify_promote_adapter_id="fixture-adapter",
            orchestrator_profile="orchestrator",
            builder_profile="builder",
            reviewer_profile="reviewer",
            reviewer_skill_digest="d" * 64,
            repair_budget=2,
            review_retry_budget=1,
        )
        goal = get_durable_goal(conn, created.goal_id)
        task = kb.get_task(conn, created.task_id)
        bindings = list_durable_goal_tasks(conn, created.goal_id)

    assert DURABLE_GOAL_SCHEMA_VERSION == 2
    assert DURABLE_GOAL_PROTOCOL_VERSION == 2
    assert goal is not None
    assert goal.workflow_version == 2
    assert goal.current_stage == "PLAN"
    assert goal.orchestrator_profile == "orchestrator"
    assert goal.verifier_profile == ""
    assert goal.resolver_id == "fixture-resolver"
    assert goal.verify_promote_adapter_id == "fixture-adapter"
    assert goal.task_contract == contract
    assert task is not None and task.assignee == "orchestrator"
    assert task.current_step_key == "PLAN"
    assert [(binding.stage, binding.attempt) for binding in bindings] == [("PLAN", 0)]


def test_plan_accepted_is_contract_bound_and_creates_builder_candidate_task(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    contract = TaskContract.create(
        reference="issue:plan-accepted",
        objective="Plan then build the candidate",
        base_revision="e" * 40,
        scope=("hermes_cli/", "tests/hermes_cli/"),
        gates=("focused-tests", "scope-check"),
    )

    with kb.connect() as conn:
        created = create_trusted_durable_goal(
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
        plan = kb.claim_task(conn, created.task_id, claimer="orchestrator")
        assert plan is not None and plan.current_run_id is not None
        assert kb.complete_task(
            conn,
            plan.id,
            summary="accepted the immutable plan",
            metadata={
                "durable_goal": {
                    "workflow_version": 2,
                    "protocol_version": DURABLE_GOAL_PROTOCOL_VERSION,
                    "stage": "PLAN",
                    "run_id": plan.current_run_id,
                    "decision": "PLAN_ACCEPTED",
                    "contract_hash": contract.contract_hash,
                    "base_revision": contract.base_revision,
                    "scope": list(contract.scope),
                    "gates": list(contract.gates),
                    "authority": "orchestrator",
                }
            },
            expected_run_id=plan.current_run_id,
        )

        result = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(conn, created.goal_id)
        tasks = list_durable_goal_tasks(conn, created.goal_id)
        build = kb.get_task(conn, result.task_id) if result.task_id else None
        assignees = [kb.get_task(conn, binding.task_id).assignee for binding in tasks]

    assert result.action == "CREATED_SUCCESSOR"
    assert goal is not None and goal.current_stage == "BUILD_CANDIDATE"
    assert build is not None and build.assignee == "builder"
    assert build.current_step_key == "BUILD_CANDIDATE"
    assert [binding.stage for binding in tasks] == ["PLAN", "BUILD_CANDIDATE"]
    assert assignees == ["orchestrator", "builder"]


def test_plan_human_gate_blocks_once_without_builder_task(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    contract = TaskContract.create(
        reference="issue:plan-human-gate",
        objective="Ask the owner when planning cannot proceed",
        base_revision="f" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    )

    with kb.connect() as conn:
        created = _create_v2_goal(conn, contract)
        _complete_bound_task(
            conn,
            task_id=created.task_id,
            profile="orchestrator",
            contract=contract,
            stage="PLAN",
            fields={
                "decision": "HUMAN_GATE",
                "reason": "owner must choose the contract-preserving option",
            },
        )

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
        bindings = list_durable_goal_tasks(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)

    assert first.action == "HUMAN_GATE"
    assert replay.action == "NOOP"
    assert goal is not None and goal.status == "BLOCKED"
    assert [binding.stage for binding in bindings] == ["PLAN"]
    assert [notification.kind for notification in notifications] == ["HUMAN_GATE"]


def test_build_candidate_completion_waits_for_deterministic_evidence_without_task(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    contract = TaskContract.create(
        reference="issue:build-boundary",
        objective="Build before deterministic classification",
        base_revision="1" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    )
    candidate_sha = "2" * 40

    with kb.connect() as conn:
        created = _create_v2_goal(conn, contract)
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
            fields={"candidate_sha": candidate_sha},
        )

        result = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(conn, created.goal_id)
        bindings = list_durable_goal_tasks(conn, created.goal_id)

    assert result.action == "AWAITING_TRUSTED_RESULT"
    assert result.task_id is None
    assert goal is not None and goal.current_stage == "VERIFY_PROMOTE"
    assert goal.candidate_sha == candidate_sha
    assert [binding.stage for binding in bindings] == ["PLAN", "BUILD_CANDIDATE"]


def test_injected_pass_evidence_creates_review_for_reviewer(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    contract = TaskContract.create(
        reference="issue:pass-to-review",
        objective="Classify a candidate then review it",
        base_revision="3" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    )
    candidate_sha = "4" * 40

    with kb.connect() as conn:
        created = _create_goal_waiting_for_evidence(
            conn, contract=contract, candidate_sha=candidate_sha
        )
        evidence = PromotionEvidence.create(
            adapter_id="fixture-adapter",
            request=PromotionRequest(
                contract_hash=contract.contract_hash,
                base_revision=contract.base_revision,
                scope=contract.scope,
                gates=contract.gates,
                candidate_sha=candidate_sha,
                attempt=0,
            ),
            classification=ResultClassification.PASS,
            summary="all deterministic gates passed",
        )

        result = apply_trusted_stage_result(conn, created.goal_id, evidence)
        goal = get_durable_goal(conn, created.goal_id)
        review = kb.get_task(conn, result.task_id) if result.task_id else None
        bindings = list_durable_goal_tasks(conn, created.goal_id)

    assert result.action == "CREATED_REVIEW"
    assert goal is not None and goal.current_stage == "REVIEW"
    assert goal.promotion_evidence == evidence.as_payload()
    assert review is not None and review.assignee == "reviewer"
    assert review.status == "durable_review"
    assert [binding.stage for binding in bindings] == [
        "PLAN",
        "BUILD_CANDIDATE",
        "REVIEW",
    ]


def test_repairable_failure_creates_adjudicate_for_orchestrator(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    contract = TaskContract.create(
        reference="issue:repairable-adjudication",
        objective="Let the orchestrator adjudicate a repairable result",
        base_revision="5" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    )
    candidate_sha = "6" * 40

    with kb.connect() as conn:
        created = _create_goal_waiting_for_evidence(
            conn, contract=contract, candidate_sha=candidate_sha
        )
        evidence = PromotionEvidence.create(
            adapter_id="fixture-adapter",
            request=PromotionRequest(
                contract_hash=contract.contract_hash,
                base_revision=contract.base_revision,
                scope=contract.scope,
                gates=contract.gates,
                candidate_sha=candidate_sha,
                attempt=0,
            ),
            classification=ResultClassification.REPAIRABLE_FAILURE,
            summary="a bounded repair may satisfy the contract",
        )

        result = apply_trusted_stage_result(conn, created.goal_id, evidence)
        goal = get_durable_goal(conn, created.goal_id)
        adjudicate = kb.get_task(conn, result.task_id) if result.task_id else None
        bindings = list_durable_goal_tasks(conn, created.goal_id)

    assert result.action == "CREATED_ADJUDICATE"
    assert goal is not None and goal.current_stage == "ADJUDICATE"
    assert adjudicate is not None and adjudicate.assignee == "orchestrator"
    assert [binding.stage for binding in bindings] == [
        "PLAN",
        "BUILD_CANDIDATE",
        "ADJUDICATE",
    ]


def test_hard_block_evidence_routes_once_to_human_gate_without_llm_task(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    contract = TaskContract.create(
        reference="issue:hard-block",
        objective="Stop on deterministic hard block",
        base_revision="7" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    )
    candidate_sha = "8" * 40

    with kb.connect() as conn:
        created = _create_goal_waiting_for_evidence(
            conn, contract=contract, candidate_sha=candidate_sha
        )
        evidence = PromotionEvidence.create(
            adapter_id="fixture-adapter",
            request=PromotionRequest(
                contract_hash=contract.contract_hash,
                base_revision=contract.base_revision,
                scope=contract.scope,
                gates=contract.gates,
                candidate_sha=candidate_sha,
                attempt=0,
            ),
            classification=ResultClassification.HARD_BLOCK,
            summary="deterministic policy gate cannot be repaired",
        )

        first = apply_trusted_stage_result(conn, created.goal_id, evidence)
        replay = apply_trusted_stage_result(conn, created.goal_id, evidence)
        goal = get_durable_goal(conn, created.goal_id)
        bindings = list_durable_goal_tasks(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)

    assert first.action == "HUMAN_GATE"
    assert first.task_id is None
    assert replay.action == "NOOP"
    assert goal is not None and goal.status == "BLOCKED"
    assert [binding.stage for binding in bindings] == ["PLAN", "BUILD_CANDIDATE"]
    assert [notification.kind for notification in notifications] == ["HUMAN_GATE"]


def test_retryable_evidence_stays_non_llm_and_creates_no_task(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    contract = TaskContract.create(
        reference="issue:retryable",
        objective="Retry deterministic classification without an LLM",
        base_revision="0" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    )
    candidate_sha = "1" * 40

    with kb.connect() as conn:
        created = _create_goal_waiting_for_evidence(
            conn, contract=contract, candidate_sha=candidate_sha
        )
        before = list_durable_goal_tasks(conn, created.goal_id)
        evidence = PromotionEvidence.create(
            adapter_id="fixture-adapter",
            request=PromotionRequest(
                contract_hash=contract.contract_hash,
                base_revision=contract.base_revision,
                scope=contract.scope,
                gates=contract.gates,
                candidate_sha=candidate_sha,
                attempt=0,
            ),
            classification=ResultClassification.RETRYABLE,
            summary="trusted service is temporarily unavailable",
        )

        result = apply_trusted_stage_result(conn, created.goal_id, evidence)
        goal = get_durable_goal(conn, created.goal_id)
        after = list_durable_goal_tasks(conn, created.goal_id)

    assert result.action == "RETRYABLE"
    assert result.task_id is None
    assert goal is not None and goal.current_stage == "VERIFY_PROMOTE"
    assert goal.promotion_evidence is None
    assert [binding.task_id for binding in after] == [
        binding.task_id for binding in before
    ]


def test_review_completion_always_creates_adjudicate_and_never_ready(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    contract = TaskContract.create(
        reference="issue:review-adjudicate",
        objective="Adjudicate every review",
        base_revision="9" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    )
    candidate_sha = "a" * 40

    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn, contract=contract, candidate_sha=candidate_sha
        )
        _complete_bound_task(
            conn,
            task_id=review_id,
            profile="reviewer",
            contract=contract,
            stage="REVIEW",
            fields={
                "candidate_sha": candidate_sha,
                "verdict": "APPROVE",
                "findings": [
                    {
                        "severity": "MINOR",
                        "summary": "small follow-up is non-blocking",
                        "resolved": False,
                    }
                ],
                "reviewer_skill_digest": "f" * 64,
            },
        )

        result = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(conn, created.goal_id)
        adjudicate = kb.get_task(conn, result.task_id) if result.task_id else None
        notifications = list_goal_notifications(conn, created.goal_id)

    assert result.action == "CREATED_ADJUDICATE"
    assert goal is not None and goal.status == "ACTIVE"
    assert goal.current_stage == "ADJUDICATE"
    assert adjudicate is not None and adjudicate.assignee == "orchestrator"
    assert notifications == []


def test_adjudicate_ready_requires_pass_approved_current_candidate_without_major_findings(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    contract = TaskContract.create(
        reference="issue:ready-adjudication",
        objective="Ready only after current pass and approval",
        base_revision="b" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    )
    candidate_sha = "c" * 40

    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn, contract=contract, candidate_sha=candidate_sha
        )
        _complete_bound_task(
            conn,
            task_id=review_id,
            profile="reviewer",
            contract=contract,
            stage="REVIEW",
            fields={
                "candidate_sha": candidate_sha,
                "verdict": "APPROVE",
                "findings": [{"severity": "MINOR", "summary": "ok"}],
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
            fields={
                "candidate_sha": candidate_sha,
                "decision": "READY_FOR_OWNER",
            },
        )

        result = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)

    assert result.action == "READY_FOR_OWNER"
    assert goal is not None and goal.status == "READY_FOR_OWNER"
    assert [notification.kind for notification in notifications] == ["READY_FOR_OWNER"]


def test_adjudicate_ready_rejects_major_finding_without_terminalizing(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    contract = TaskContract.create(
        reference="issue:major-finding",
        objective="Major findings block readiness",
        base_revision="d" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    )
    candidate_sha = "e" * 40

    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn, contract=contract, candidate_sha=candidate_sha
        )
        _complete_bound_task(
            conn,
            task_id=review_id,
            profile="reviewer",
            contract=contract,
            stage="REVIEW",
            fields={
                "candidate_sha": candidate_sha,
                "verdict": "APPROVE",
                "findings": [{"severity": "MAJOR", "summary": "must fix"}],
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
            fields={
                "candidate_sha": candidate_sha,
                "decision": "READY_FOR_OWNER",
            },
        )

        result = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)

    assert result.action == "NOOP"
    assert result.reason == "ready_validation_failed"
    assert goal is not None and goal.status == "ACTIVE"
    assert goal.current_stage == "ADJUDICATE"
    assert notifications == []


def test_adjudicate_repair_is_bounded_and_creates_builder_repair_task(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    contract = TaskContract.create(
        reference="issue:bounded-repair",
        objective="Repair only within budget",
        base_revision="f" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    )
    candidate_sha = "1" * 40

    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn, contract=contract, candidate_sha=candidate_sha, repair_budget=1
        )
        _complete_bound_task(
            conn,
            task_id=review_id,
            profile="reviewer",
            contract=contract,
            stage="REVIEW",
            fields={
                "candidate_sha": candidate_sha,
                "verdict": "CHANGES_REQUIRED",
                "findings": [{"severity": "MAJOR", "summary": "repair needed"}],
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
            fields={"candidate_sha": candidate_sha, "decision": "REPAIR"},
        )

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
        repair = kb.get_task(conn, first.task_id) if first.task_id else None

    assert first.action == "CREATED_REPAIR"
    assert replay.action == "NOOP"
    assert goal is not None and goal.current_stage == "REPAIR_BUILD_1"
    assert goal.repair_attempts_reserved == 1
    assert repair is not None and repair.assignee == "builder"


def test_retryable_evidence_keeps_same_taskless_verify_promote_state(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    contract = TaskContract.create(
        reference="issue:retryable",
        objective="Retry deterministic boundary without LLM mutation",
        base_revision="2" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    )
    candidate_sha = "3" * 40

    with kb.connect() as conn:
        created = _create_goal_waiting_for_evidence(
            conn, contract=contract, candidate_sha=candidate_sha
        )
        before = list_durable_goal_tasks(conn, created.goal_id)
        evidence = PromotionEvidence.create(
            adapter_id="fixture-adapter",
            request=PromotionRequest(
                contract_hash=contract.contract_hash,
                base_revision=contract.base_revision,
                scope=contract.scope,
                gates=contract.gates,
                candidate_sha=candidate_sha,
                attempt=0,
            ),
            classification=ResultClassification.RETRYABLE,
            summary="temporary deterministic service outage",
        )

        result = apply_trusted_stage_result(conn, created.goal_id, evidence)
        goal = get_durable_goal(conn, created.goal_id)
        after = list_durable_goal_tasks(conn, created.goal_id)

    assert result.action == "RETRYABLE"
    assert goal is not None and goal.current_stage == "VERIFY_PROMOTE"
    assert before == after


def test_active_v1_goal_fails_closed_once_under_v2_supervisor(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()

    with kb.connect() as conn:
        created = kb.create_durable_goal_record(
            conn,
            objective="Legacy V1 history",
            board_slug="default",
            origin={"platform": "telegram", "chat_id": "owner"},
            builder_profile="builder",
            verifier_profile="verifier",
            reviewer_profile="reviewer",
            repair_budget=1,
            review_retry_budget=1,
            schema_version=1,
            protocol_version=1,
        )
        goal_id, task_id = created

        first = supervise_goal_once(
            conn,
            goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        second = supervise_goal_once(
            conn,
            goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(conn, goal_id)
        notifications = list_goal_notifications(conn, goal_id)
        tasks = list_durable_goal_tasks(conn, goal_id)

    assert first.action == "HUMAN_GATE"
    assert second.action == "NOOP"
    assert goal is not None and goal.status == "BLOCKED"
    assert goal.workflow_version == 1
    assert [notification.kind for notification in notifications] == ["HUMAN_GATE"]
    assert [(binding.stage, binding.task_id) for binding in tasks] == [
        ("BUILD", task_id)
    ]


def test_adjudicate_ready_requires_approved_review_without_unresolved_major_findings(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    contract = TaskContract.create(
        reference="issue:ready-hard-checks",
        objective="Become ready only after acceptable review",
        base_revision="b" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    )
    candidate_sha = "c" * 40

    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn, contract=contract, candidate_sha=candidate_sha
        )
        _complete_bound_task(
            conn,
            task_id=review_id,
            profile="reviewer",
            contract=contract,
            stage="REVIEW",
            fields={
                "candidate_sha": candidate_sha,
                "verdict": "APPROVE",
                "findings": [
                    {
                        "severity": "NIT",
                        "summary": "non-blocking wording",
                        "resolved": False,
                    }
                ],
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
            fields={
                "candidate_sha": candidate_sha,
                "decision": "READY_FOR_OWNER",
            },
        )

        result = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(conn, created.goal_id)
        bindings = list_durable_goal_tasks(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)

    assert result.action == "READY_FOR_OWNER"
    assert result.task_id == adjudicate_id
    assert goal is not None and goal.status == "READY_FOR_OWNER"
    assert goal.candidate_sha == candidate_sha
    assert [binding.stage for binding in bindings] == [
        "PLAN",
        "BUILD_CANDIDATE",
        "REVIEW",
        "ADJUDICATE",
    ]
    assert [notification.kind for notification in notifications] == ["READY_FOR_OWNER"]


def test_adjudicate_repair_is_current_contract_bound_and_reserves_budget(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    contract = TaskContract.create(
        reference="issue:bounded-repair",
        objective="Repair only the current reviewed candidate",
        base_revision="d" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    )
    candidate_sha = "e" * 40

    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn,
            contract=contract,
            candidate_sha=candidate_sha,
            repair_budget=1,
        )
        _complete_bound_task(
            conn,
            task_id=review_id,
            profile="reviewer",
            contract=contract,
            stage="REVIEW",
            fields={
                "candidate_sha": candidate_sha,
                "verdict": "CHANGES_REQUIRED",
                "findings": [
                    {
                        "severity": "MAJOR",
                        "summary": "contract gate is not yet satisfied",
                        "resolved": False,
                    }
                ],
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
            fields={"candidate_sha": candidate_sha, "decision": "REPAIR"},
        )

        result = supervise_goal_once(
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
        repair = kb.get_task(conn, result.task_id) if result.task_id else None
        repair_binding = list_durable_goal_tasks(conn, created.goal_id)[-1]

    assert result.action == "CREATED_REPAIR"
    assert replay.action == "NOOP"
    assert goal is not None and goal.current_stage == "REPAIR_BUILD_1"
    assert goal.repair_attempts_reserved == 1
    assert repair is not None and repair.assignee == "builder"
    assert repair_binding.stage == "REPAIR_BUILD_1"
    assert repair_binding.expected_candidate_sha == candidate_sha


def test_repair_build_rejects_a_noncurrent_input_candidate(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    contract = TaskContract.create(
        reference="issue:repair-current-candidate",
        objective="Repair only the candidate adjudicated for repair",
        base_revision="2" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    )
    candidate_sha = "3" * 40

    with kb.connect() as conn:
        created, repair_id = _create_goal_at_repair(
            conn, contract=contract, candidate_sha=candidate_sha
        )
        _complete_bound_task(
            conn,
            task_id=repair_id,
            profile="builder",
            contract=contract,
            stage="REPAIR_BUILD_1",
            fields={
                "input_candidate_sha": "4" * 40,
                "candidate_sha": "5" * 40,
            },
        )

        result = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(conn, created.goal_id)
        bindings = list_durable_goal_tasks(conn, created.goal_id)

    assert result.action == "BLOCKED"
    assert result.reason == "repair_input_candidate_mismatch"
    assert goal is not None and goal.current_stage == "BLOCKED"
    assert goal.status == "BLOCKED"
    assert goal.candidate_sha == candidate_sha
    assert [binding.stage for binding in bindings][-1] == "REPAIR_BUILD_1"


def test_active_v1_history_fails_closed_once_and_never_spawns_verifier(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    spawned: list[str] = []
    with kb.connect() as conn:
        created = create_durable_goal(
            conn,
            objective="Readable legacy verifier history",
            origin=GoalOrigin(platform="telegram", chat_id="legacy-owner"),
            board="default",
            builder_profile="legacy-builder",
            verifier_profile="legacy-verifier",
            reviewer_profile="legacy-reviewer",
            repair_budget=1,
            review_retry_budget=1,
        )
        conn.execute(
            "UPDATE kanban_goals SET workflow_version = 1, schema_version = 1, "
            "protocol_version = 1, current_stage = 'VERIFY' WHERE id = ?",
            (created.goal_id,),
        )
        conn.execute(
            "UPDATE kanban_goal_tasks SET stage = 'VERIFY' WHERE task_id = ?",
            (created.task_id,),
        )
        conn.execute(
            "UPDATE tasks SET assignee = 'legacy-verifier', "
            "current_step_key = 'VERIFY' WHERE id = ?",
            (created.task_id,),
        )
        conn.commit()

        first_preflight = preflight_durable_dispatch(
            conn,
            created.task_id,
            board="default",
            runtime_protocol_version=1,
        )
        replay_preflight = preflight_durable_dispatch(
            conn,
            created.task_id,
            board="default",
            runtime_protocol_version=1,
        )
        dispatch = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=lambda task, *_args, **_kwargs: spawned.append(task.id),
            durable_goal_protocol_version=1,
        )
        goal = get_durable_goal(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)
        runs = kb.list_runs(conn, created.task_id)

    assert spawned == []
    assert first_preflight.allowed is False and first_preflight.blocked_now is True
    assert replay_preflight.allowed is False and replay_preflight.blocked_now is False
    assert dispatch.spawned == [] and runs == []
    assert goal is not None and goal.workflow_version == 1
    assert goal.verifier_profile == "legacy-verifier"
    assert goal.status == "BLOCKED_CAPABILITY"
    assert len(notifications) == 1
