"""Behavior tests for the deterministic durable Kanban goal supervisor."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_goal_supervisor import (
    DURABLE_GOAL_PROTOCOL_VERSION,
    GoalOrigin,
    apply_trusted_stage_result,
    create_durable_goal,
    create_trusted_durable_goal,
    get_durable_goal,
    _profile_skill_digest,
    list_durable_goal_tasks,
    list_goal_notifications,
    mark_durable_goal_completed_by_owner,
    supervise_board_once,
    supervise_goal_once,
)
from hermes_cli.kanban_trusted_stages import (
    Authority,
    PromotionEvidence,
    PromotionRequest,
    ResultClassification,
    TaskContract,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "--allow-empty", "-m", "base"],
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
        },
        capture_output=True,
    )
    kb.write_board_metadata("default", default_workdir=str(repo))
    return home


def _v2_contract(candidate_sha: str) -> TaskContract:
    return TaskContract.create(
        reference=f"test-contract:{candidate_sha}",
        objective="Exercise the Workflow V2 supervisor",
        base_revision="a" * 40,
        scope=("hermes_cli/", "tests/hermes_cli/"),
        gates=("focused-tests",),
        authority=Authority.ORCHESTRATOR,
    )


def _complete_v2_task(
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
    assert kb.complete_task(
        conn,
        task.id,
        summary=f"completed {stage}",
        metadata={
            "durable_goal": {
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
        },
        expected_run_id=task.current_run_id,
    )
    return task


def _create_v2_goal_at_build(
    conn,
    *,
    candidate_key: str,
    origin: GoalOrigin | None = None,
    reviewer_skill_digest: str | None = None,
    repair_budget: int = 1,
    review_retry_budget: int = 1,
):
    contract = _v2_contract(candidate_key)
    created = create_trusted_durable_goal(
        conn,
        contract=contract,
        origin=origin or GoalOrigin(platform="telegram", chat_id="owner-chat"),
        board="default",
        resolver_id="fixture-resolver",
        verify_promote_adapter_id="fixture-adapter",
        orchestrator_profile="orchestrator",
        builder_profile="builder",
        reviewer_profile="reviewer",
        reviewer_skill_digest=reviewer_skill_digest,
        repair_budget=repair_budget,
        review_retry_budget=review_retry_budget,
    )
    _complete_v2_task(
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
    return created, build_id, contract


def _complete_v2_review(
    conn,
    *,
    created,
    review_id: str,
    candidate_sha: str,
    verdict: str,
    findings: list[dict[str, object]],
):
    goal = get_durable_goal(conn, created.goal_id)
    assert goal is not None and goal.task_contract is not None
    return _complete_v2_task(
        conn,
        task_id=review_id,
        profile="reviewer",
        contract=goal.task_contract,
        stage="REVIEW",
        fields={
            "candidate_sha": candidate_sha,
            "verdict": verdict,
            "findings": findings,
            **(
                {"reviewer_skill_digest": goal.reviewer_skill_digest}
                if goal.reviewer_skill_digest
                else {}
            ),
        },
    )


def _adjudicate_v2(
    conn,
    *,
    created,
    candidate_sha: str,
    decision: str,
):
    adjudicate_id = supervise_goal_once(
        conn,
        created.goal_id,
        board="default",
        runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
    ).task_id
    assert adjudicate_id is not None
    goal = get_durable_goal(conn, created.goal_id)
    assert goal is not None and goal.task_contract is not None
    _complete_v2_task(
        conn,
        task_id=adjudicate_id,
        profile="orchestrator",
        contract=goal.task_contract,
        stage="ADJUDICATE",
        fields={"candidate_sha": candidate_sha, "decision": decision},
    )
    return adjudicate_id


def _create_goal_at_review(
    conn,
    *,
    candidate_sha: str,
    review_retry_budget: int = 1,
    reviewer_skill_digest: str | None = None,
    origin: GoalOrigin | None = None,
):
    contract = _v2_contract(candidate_sha)
    created = create_trusted_durable_goal(
        conn,
        contract=contract,
        origin=origin or GoalOrigin(platform="telegram", chat_id="owner-chat"),
        board="default",
        resolver_id="fixture-resolver",
        verify_promote_adapter_id="fixture-adapter",
        orchestrator_profile="orchestrator",
        builder_profile="builder",
        reviewer_profile="reviewer",
        reviewer_skill_digest=reviewer_skill_digest,
        repair_budget=1,
        review_retry_budget=review_retry_budget,
    )
    _complete_v2_task(
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
    _complete_v2_task(
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


def _create_goal_at_repair(conn, *, candidate_sha: str):
    created, review_id = _create_goal_at_review(
        conn,
        candidate_sha=candidate_sha,
    )
    _complete_v2_review(
        conn,
        created=created,
        review_id=review_id,
        candidate_sha=candidate_sha,
        verdict="CHANGES_REQUIRED",
        findings=[{"severity": "MAJOR", "summary": "repair required"}],
    )
    _adjudicate_v2(
        conn,
        created=created,
        candidate_sha=candidate_sha,
        decision="REPAIR",
    )
    repair_id = supervise_goal_once(
        conn,
        created.goal_id,
        board="default",
        runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
    ).task_id
    assert repair_id is not None
    return created, repair_id


def _create_profile_skill(home: Path, profile: str, skill_name: str) -> Path:
    skill_dir = home / "profiles" / profile / "skills" / skill_name
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        f"---\nname: {skill_name}\n---\n# {skill_name}\n",
        encoding="utf-8",
    )
    return skill_file


def _create_reviewer_snapshot(home: Path) -> Path:
    skill_dir = home / "profiles" / "reviewer" / "skills" / "immutable-change-reviews"
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: immutable-change-reviews\n---\n# Immutable Review\n",
        encoding="utf-8",
    )
    (skill_dir / "references" / "rules.md").write_text(
        "review exact immutable changes\n",
        encoding="utf-8",
    )
    return skill_dir


def _dispatch_pinned_reviewer_and_capture_child_env(
    kanban_home,
    monkeypatch,
    *,
    skill_text: str | None = None,
):
    """Run the real durable dispatcher/default-spawn path up to child exec."""
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    skill_dir = _create_reviewer_snapshot(kanban_home)
    if skill_text is not None:
        (skill_dir / "SKILL.md").write_text(skill_text, encoding="utf-8")
    expected_digest = _profile_skill_digest("reviewer", "immutable-change-reviews")
    assert expected_digest is not None
    captured: dict = {}

    class _FakeProc:
        pid = 24680

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return _FakeProc()

    def _spawn(task, workspace, **kwargs):
        # Let dispatcher worktree setup use the real subprocess machinery;
        # intercept only the final child exec inside the production spawner.
        with patch("subprocess.Popen", _fake_popen):
            return kb._default_spawn(task, workspace, board=kwargs.get("board"))

    with kb.connect() as conn:
        _created, review_id = _create_goal_at_review(
            conn,
            candidate_sha="e" * 40,
            reviewer_skill_digest=expected_digest,
        )
        result = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )

    assert result.spawned and result.spawned[0][0] == review_id
    assert captured["env"]["HERMES_HOME"].endswith("/profiles/reviewer")
    assert "--skills" in captured["cmd"]
    return skill_dir, expected_digest, captured["env"]


def _create_legacy_ready_task(conn, *, title: str, assignee: str) -> str:
    task_id = kb.create_task(
        conn,
        title=title,
        assignee=assignee,
        workspace_kind="scratch",
    )
    conn.execute(
        "UPDATE tasks SET status = 'ready', claim_lock = NULL WHERE id = ?",
        (task_id,),
    )
    conn.commit()
    return task_id


def test_gateway_origin_creates_one_durable_build_task(kanban_home):
    origin = GoalOrigin(
        platform="telegram",
        chat_id="owner-chat",
        chat_type="dm",
        thread_id="topic-42",
        user_id="owner-7",
        message_id="origin-message-42",
        notifier_profile="gateway-owner",
        delivery_metadata={"thread_id": "topic-42"},
    )

    with kb.connect() as conn:
        created = create_durable_goal(
            conn,
            objective="Ship the candidate without merging or deploying",
            origin=origin,
            board="default",
            builder_profile="turpi_builder",
            verifier_profile="turpi_verify",
            reviewer_profile="turpi_review",
            repair_budget=1,
            review_retry_budget=1,
        )
        replayed = create_durable_goal(
            conn,
            objective="Ship the candidate without merging or deploying",
            origin=origin,
            board="default",
            builder_profile="turpi_builder",
            verifier_profile="turpi_verify",
            reviewer_profile="turpi_review",
            repair_budget=1,
            review_retry_budget=1,
        )
        goal = get_durable_goal(conn, created.goal_id)
        tasks = list_durable_goal_tasks(conn, created.goal_id)

    assert goal is not None
    assert goal.status == "ACTIVE"
    assert goal.current_stage == "BUILD"
    assert goal.origin == origin
    assert replayed == created
    assert created.task_id == tasks[0].task_id
    assert [(task.stage, task.attempt) for task in tasks] == [("BUILD", 0)]

    with kb.connect() as conn:
        build = kb.get_task(conn, created.task_id)
        assert build is not None
        assert build.current_step_key == "BUILD"
        assert build.assignee == "turpi_builder"
        assert build.status == "durable_ready"
        assert build.workspace_kind == "worktree"
        assert kb.list_notify_subs(conn, created.task_id) == []


def test_build_gave_up_reserves_one_repair_and_replay_creates_no_duplicate(
    kanban_home, monkeypatch
):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    origin = GoalOrigin(platform="telegram", chat_id="owner-chat")
    with kb.connect() as conn:
        created, build_id, _contract = _create_v2_goal_at_build(
            conn,
            candidate_key="builder-gave-up",
            origin=origin,
            repair_budget=1,
        )

        def _spawn_failure(*_args, **_kwargs):
            raise RuntimeError("worker could not start")

        result = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn_failure,
            failure_limit=1,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        assert result.auto_blocked == [build_id]
        build_run = kb.latest_run(conn, build_id)
        assert build_run is not None and build_run.ended_at is not None
        gave_up_event = next(
            event
            for event in reversed(kb.list_events(conn, build_id))
            if event.kind == "gave_up" and event.run_id == build_run.id
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

    assert first.action == "BLOCKED"
    assert replay.action == "NOOP"
    assert goal is not None and goal.status == "BLOCKED"
    assert goal.current_stage == "BLOCKED"
    assert goal.repair_attempts_reserved == 0
    assert "gave up" in (goal.blocked_reason or "")
    assert [(item.stage, item.attempt) for item in bindings] == [
        ("PLAN", 0),
        ("BUILD_CANDIDATE", 0),
    ]
    assert bindings[1].expected_run_id == build_run.id
    assert bindings[1].completion_event_id == gave_up_event.id
    assert all(not item.stage.startswith("REPAIR_BUILD_") for item in bindings)
    assert [item.kind for item in notifications] == ["HUMAN_GATE"]
    assert notifications[0].origin == origin


@pytest.mark.parametrize(
    ("terminal_event", "block_kind"),
    [
        ("blocked", None),
        ("dependency_wait", "dependency"),
        ("block_loop_detected", "needs_input"),
        ("scheduled", None),
        ("status:triage", None),
    ],
)
def test_current_run_park_or_block_terminates_goal_and_notifies_once(
    kanban_home, monkeypatch, terminal_event, block_kind
):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    with kb.connect() as conn:
        created, build_id, _contract = _create_v2_goal_at_build(
            conn,
            candidate_key=f"park-{terminal_event}",
            origin=GoalOrigin(platform="telegram", chat_id="owner-chat"),
        )
        task = kb.claim_task(conn, build_id, claimer="builder")
        assert task is not None and task.current_run_id is not None
        run_id = task.current_run_id

        if terminal_event == "scheduled":
            assert kb.schedule_task(
                conn,
                task.id,
                reason="wait for owner window",
                expected_run_id=task.current_run_id,
            )
        elif terminal_event == "status:triage":
            from plugins.kanban.dashboard.plugin_api import _set_status_direct

            assert _set_status_direct(conn, task.id, "triage")
        else:
            if terminal_event == "block_loop_detected":
                conn.execute(
                    "UPDATE tasks SET block_kind = ?, block_recurrences = ? "
                    "WHERE id = ?",
                    (block_kind, kb.BLOCK_RECURRENCE_LIMIT - 1, task.id),
                )
                conn.commit()
            assert kb.block_task(
                conn,
                task.id,
                reason="owner input required",
                kind=block_kind,
                expected_run_id=task.current_run_id,
            )

        expected_event_kind = (
            "status" if terminal_event == "status:triage" else terminal_event
        )
        closing_event = next(
            event
            for event in reversed(kb.list_events(conn, task.id))
            if event.kind == expected_event_kind and event.run_id == run_id
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

    assert first.action == "BLOCKED"
    assert replay.action == "NOOP"
    assert goal is not None and goal.status == "BLOCKED"
    assert expected_event_kind in (goal.blocked_reason or "")
    assert [binding.stage for binding in bindings] == ["PLAN", "BUILD_CANDIDATE"]
    assert bindings[1].expected_run_id == run_id
    assert bindings[1].completion_event_id == closing_event.id
    assert bindings[1].completion_payload["event_kind"] == expected_event_kind
    assert all(not binding.stage.startswith("REPAIR_BUILD_") for binding in bindings)
    assert [item.kind for item in notifications] == ["HUMAN_GATE"]
    assert notifications[0].payload["reason"] == goal.blocked_reason


def test_newer_runless_status_event_does_not_mask_run_closing_event(
    kanban_home, monkeypatch
):
    from hermes_cli import profiles
    from plugins.kanban.dashboard.plugin_api import _set_status_direct

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    with kb.connect() as conn:
        created, build_id, _contract = _create_v2_goal_at_build(
            conn,
            candidate_key="run-closing-terminal-proof",
            origin=GoalOrigin(platform="telegram", chat_id="owner-chat"),
        )
        task = kb.claim_task(conn, build_id, claimer="builder")
        assert task is not None and task.current_run_id is not None
        run_id = task.current_run_id
        assert _set_status_direct(conn, task.id, "todo")
        run_closing_event = next(
            event
            for event in reversed(kb.list_events(conn, task.id))
            if event.kind == "status" and event.run_id == run_id
        )
        assert _set_status_direct(conn, task.id, "triage")
        assert not kb.archive_task(conn, task.id)
        newest_status_event = next(
            event
            for event in reversed(kb.list_events(conn, task.id))
            if event.kind == "status"
        )
        assert newest_status_event.id > run_closing_event.id
        assert newest_status_event.run_id is None

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

    assert first.action == "BLOCKED"
    assert replay.action == "NOOP"
    assert goal is not None and goal.status == "BLOCKED"
    assert "status" in (goal.blocked_reason or "")
    assert [binding.stage for binding in bindings] == ["PLAN", "BUILD_CANDIDATE"]
    assert bindings[1].expected_run_id == run_id
    assert bindings[1].completion_event_id == run_closing_event.id
    assert bindings[1].completion_event_id != newest_status_event.id
    assert all(not binding.stage.startswith("REPAIR_BUILD_") for binding in bindings)
    assert [item.kind for item in notifications] == ["HUMAN_GATE"]


def test_db_transition_cas_rejects_completion_after_newer_run_claimed(
    kanban_home, monkeypatch
):
    from hermes_cli import profiles
    from plugins.kanban.dashboard.plugin_api import _set_status_direct

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    with kb.connect() as conn:
        created = create_durable_goal(
            conn,
            objective="Reject stale completion inside the write transaction",
            origin=GoalOrigin(platform="telegram", chat_id="owner-chat"),
            board="default",
            builder_profile="builder",
            verifier_profile="verifier",
            reviewer_profile="reviewer",
            repair_budget=1,
            review_retry_budget=1,
        )
        run_one_task = kb.claim_task(conn, created.task_id, claimer="builder")
        assert run_one_task is not None and run_one_task.current_run_id is not None
        run_one = run_one_task.current_run_id
        assert _set_status_direct(conn, created.task_id, "todo")
        run_one_event = next(
            event
            for event in reversed(kb.list_events(conn, created.task_id))
            if event.kind == "status" and event.run_id == run_one
        )
        assert _set_status_direct(conn, created.task_id, "ready")
        run_two_task = kb.claim_task(conn, created.task_id, claimer="builder")
        assert run_two_task is not None and run_two_task.current_run_id != run_one

        assert (
            kb.transition_durable_goal_to_successor(
                conn,
                goal_id=created.goal_id,
                expected_state_version=1,
                predecessor_task_id=created.task_id,
                completion_event_id=run_one_event.id,
                expected_run_id=run_one,
                next_stage="VERIFY",
                next_attempt=0,
                assignee="verifier",
                completion_payload={"status": "PASS"},
            )
            is None
        )
        assert not kb.transition_durable_goal_to_terminal(
            conn,
            goal_id=created.goal_id,
            expected_state_version=1,
            predecessor_task_id=created.task_id,
            completion_event_id=run_one_event.id,
            expected_run_id=run_one,
            completion_payload={"event": "status:todo"},
            terminal_status="BLOCKED",
            notification_kind="BLOCKED",
            notification_payload={"status": "BLOCKED"},
            blocked_reason="stale run must not terminate",
        )
        goal = get_durable_goal(conn, created.goal_id)
        bindings = list_durable_goal_tasks(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)

    assert goal is not None and goal.status == "ACTIVE"
    assert goal.current_stage == "BUILD" and goal.state_version == 1
    assert len(bindings) == 1 and bindings[0].completion_event_id is None
    assert notifications == []


def test_direct_status_helper_rejects_structured_terminal_statuses(
    kanban_home, monkeypatch
):
    from hermes_cli import profiles
    from plugins.kanban.dashboard.plugin_api import _set_status_direct

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    with kb.connect() as conn:
        created = create_durable_goal(
            conn,
            objective="Structured statuses use guarded verbs",
            origin=GoalOrigin(platform="telegram", chat_id="owner-chat"),
            board="default",
            builder_profile="builder",
            verifier_profile="verifier",
            reviewer_profile="reviewer",
            repair_budget=1,
            review_retry_budget=1,
        )
        task = kb.claim_task(conn, created.task_id, claimer="builder")
        assert task is not None
        for status in ("archived", "done", "blocked", "scheduled", "running"):
            assert not _set_status_direct(conn, task.id, status)
        current = kb.get_task(conn, task.id)

    assert current is not None and current.status == "running"


def test_dispatcher_scoped_worker_cannot_archive_own_or_foreign_task(
    kanban_home, monkeypatch
):
    from argparse import Namespace
    from hermes_cli import profiles
    from hermes_cli.kanban import _cmd_archive

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    with kb.connect() as conn:
        own = create_durable_goal(
            conn,
            objective="own durable task",
            origin=GoalOrigin(platform="telegram", chat_id="owner-chat"),
            board="default",
            builder_profile="builder",
            verifier_profile="verifier",
            reviewer_profile="reviewer",
            repair_budget=1,
            review_retry_budget=1,
        )
        foreign = create_durable_goal(
            conn,
            objective="foreign durable task",
            origin=GoalOrigin(platform="telegram", chat_id="owner-chat-2"),
            board="default",
            builder_profile="builder",
            verifier_profile="verifier",
            reviewer_profile="reviewer",
            repair_budget=1,
            review_retry_budget=1,
        )
        own_task = kb.claim_task(conn, own.task_id, claimer="builder")
        foreign_task = kb.claim_task(conn, foreign.task_id, claimer="builder")
        assert own_task is not None and own_task.current_run_id is not None
        assert foreign_task is not None
        ordinary_foreign_id = _create_legacy_ready_task(
            conn,
            title="ordinary foreign task",
            assignee="builder",
        )
        conn.execute(
            "UPDATE tasks SET worker_pid = ? WHERE id = ?",
            (os.getpid(), own.task_id),
        )
        conn.commit()

    monkeypatch.setenv("HERMES_KANBAN_TASK", own.task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(own_task.current_run_id))
    assert (
        _cmd_archive(Namespace(task_ids=[own.task_id, foreign.task_id], purge_ids=[]))
        == 1
    )
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID")
    assert _cmd_archive(Namespace(task_ids=[own.task_id], purge_ids=[])) == 1
    assert _cmd_archive(Namespace(task_ids=[ordinary_foreign_id], purge_ids=[])) == 1

    with kb.connect() as conn:
        own_after = kb.get_task(conn, own.task_id)
        foreign_after = kb.get_task(conn, foreign.task_id)
        assert own_after is not None and own_after.status == "running"
        assert foreign_after is not None and foreign_after.status == "running"
        ordinary_after = kb.get_task(conn, ordinary_foreign_id)
        assert ordinary_after is not None and ordinary_after.status == "ready"


def test_durable_task_hard_delete_is_controlled_and_non_mutating(
    kanban_home, monkeypatch
):
    from fastapi import HTTPException
    from hermes_cli import profiles
    from plugins.kanban.dashboard.plugin_api import delete_task as dashboard_delete_task

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    with kb.connect() as conn:
        created = create_durable_goal(
            conn,
            objective="Preserve durable audit history",
            origin=GoalOrigin(platform="telegram", chat_id="owner-chat"),
            board="default",
            builder_profile="builder",
            verifier_profile="verifier",
            reviewer_profile="reviewer",
            repair_budget=1,
            review_retry_budget=1,
        )
        assert not kb.delete_task(conn, created.task_id)
        assert kb.get_task(conn, created.task_id) is not None

    with pytest.raises(HTTPException) as exc_info:
        dashboard_delete_task(created.task_id, board="default")
    assert exc_info.value.status_code == 409

    with kb.connect() as conn:
        assert kb.get_task(conn, created.task_id) is not None


def test_structured_build_completion_binds_current_run_and_creates_verify(
    kanban_home,
):
    candidate_sha = "a" * 40
    with kb.connect() as conn:
        created, build_id, contract = _create_v2_goal_at_build(
            conn,
            candidate_key=candidate_sha,
            origin=GoalOrigin(platform="discord", chat_id="owner-channel"),
        )
        build = _complete_v2_task(
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
    assert goal is not None
    assert goal.current_stage == "VERIFY_PROMOTE"
    assert goal.candidate_sha == candidate_sha
    assert [(item.stage, item.attempt) for item in bindings] == [
        ("PLAN", 0),
        ("BUILD_CANDIDATE", 0),
    ]
    assert sum(item.stage == "VERIFY" for item in bindings) == 0
    assert bindings[1].expected_run_id == build.current_run_id
    assert bindings[1].completion_payload == {
        "workflow_version": 2,
        "protocol_version": DURABLE_GOAL_PROTOCOL_VERSION,
        "stage": "BUILD_CANDIDATE",
        "run_id": build.current_run_id,
        "contract_hash": contract.contract_hash,
        "base_revision": contract.base_revision,
        "scope": list(contract.scope),
        "gates": list(contract.gates),
        "authority": "builder",
        "candidate_sha": candidate_sha,
    }


def test_board_tick_restart_after_terminal_build_creates_one_verify(
    kanban_home,
):
    candidate_sha = "1" * 40
    with kb.connect() as conn:
        created, build_id, contract = _create_v2_goal_at_build(
            conn,
            candidate_key=candidate_sha,
            origin=GoalOrigin(platform="telegram", chat_id="owner"),
        )
        _complete_v2_task(
            conn,
            task_id=build_id,
            profile="builder",
            contract=contract,
            stage="BUILD_CANDIDATE",
            fields={"candidate_sha": candidate_sha},
        )

    with kb.connect() as restarted:
        first = supervise_board_once(
            restarted,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
    with kb.connect() as replayed:
        replay = supervise_board_once(
            replayed,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        bindings = list_durable_goal_tasks(replayed, created.goal_id)
        goal = get_durable_goal(replayed, created.goal_id)

    assert [result.action for result in first] == ["AWAITING_TRUSTED_RESULT"]
    assert first[0].task_id is None
    assert [result.action for result in replay] == ["NOOP"]
    assert goal is not None and goal.current_stage == "VERIFY_PROMOTE"
    assert [binding.stage for binding in bindings] == ["PLAN", "BUILD_CANDIDATE"]
    assert sum(binding.stage == "VERIFY" for binding in bindings) == 0


def test_restart_after_terminal_build_gave_up_creates_one_repair(
    kanban_home, monkeypatch
):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    with kb.connect() as conn:
        created, build_id, _contract = _create_v2_goal_at_build(
            conn,
            candidate_key="pre-restart-builder-gave-up",
            origin=GoalOrigin(platform="telegram", chat_id="owner"),
            repair_budget=1,
        )

        def _spawn_failure(*_args, **_kwargs):
            raise RuntimeError("build worker failed before restart")

        dispatch = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn_failure,
            failure_limit=1,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        assert dispatch.auto_blocked == [build_id]

    with kb.connect() as restarted:
        first = supervise_board_once(
            restarted,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(restarted, created.goal_id)
        notifications = list_goal_notifications(restarted, created.goal_id)
    with kb.connect() as replayed:
        replay = supervise_board_once(
            replayed,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        bindings = list_durable_goal_tasks(replayed, created.goal_id)

    assert [result.action for result in first] == ["BLOCKED"]
    assert replay == []
    assert goal is not None and goal.status == "BLOCKED"
    assert goal.repair_attempts_reserved == 0
    assert [binding.stage for binding in bindings] == ["PLAN", "BUILD_CANDIDATE"]
    assert all(not binding.stage.startswith("REPAIR_BUILD_") for binding in bindings)
    assert [item.kind for item in notifications] == ["HUMAN_GATE"]


def test_restart_after_successor_creation_before_next_stage_creates_no_duplicate(
    kanban_home,
):
    candidate_sha = "7" * 40
    with kb.connect() as conn:
        created, build_id, _contract = _create_v2_goal_at_build(
            conn,
            candidate_key=candidate_sha,
            origin=GoalOrigin(platform="telegram", chat_id="owner"),
        )
        bindings_before_restart = list_durable_goal_tasks(conn, created.goal_id)

    assert build_id != created.task_id
    assert [binding.stage for binding in bindings_before_restart] == [
        "PLAN",
        "BUILD_CANDIDATE",
    ]
    assert (
        sum(binding.stage == "BUILD_CANDIDATE" for binding in bindings_before_restart)
        == 1
    )

    with kb.connect() as restarted:
        replay = supervise_board_once(
            restarted,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        bindings = list_durable_goal_tasks(restarted, created.goal_id)

    assert [result.action for result in replay] == ["NOOP"]
    assert [binding.stage for binding in bindings] == ["PLAN", "BUILD_CANDIDATE"]
    assert sum(binding.stage == "BUILD_CANDIDATE" for binding in bindings) == 1


def test_verify_pass_for_exact_candidate_creates_capability_pinned_review(
    kanban_home,
):
    candidate_sha = "b" * 40
    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn,
            candidate_sha=candidate_sha,
            origin=GoalOrigin(platform="slack", chat_id="owner-channel"),
        )
        goal = get_durable_goal(conn, created.goal_id)
        bindings = list_durable_goal_tasks(conn, created.goal_id)
        review = kb.get_task(conn, review_id)

    assert goal is not None and goal.current_stage == "REVIEW"
    assert [binding.stage for binding in bindings] == [
        "PLAN",
        "BUILD_CANDIDATE",
        "REVIEW",
    ]
    assert all(binding.stage != "VERIFY" for binding in bindings)
    assert all(
        (kb_task := review if binding.task_id == review_id else None) is None
        or kb_task.assignee != "verifier"
        for binding in bindings
    )
    assert bindings[2].expected_candidate_sha == candidate_sha
    assert review is not None
    assert review.status == "durable_review"
    assert review.assignee == "reviewer"
    assert review.skills == ["immutable-change-reviews"]
    assert "Do not merge, push, or deploy" in (review.body or "")


def test_verify_fail_reserves_one_build_repair_and_replay_is_idempotent(
    kanban_home,
):
    candidate_sha = "2" * 40
    with kb.connect() as conn:
        created, build_id, contract = _create_v2_goal_at_build(
            conn,
            candidate_key=candidate_sha,
            origin=GoalOrigin(platform="slack", chat_id="owner-channel"),
            repair_budget=1,
        )
        _complete_v2_task(
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
        waiting_goal = get_durable_goal(conn, created.goal_id)
        waiting_bindings = list_durable_goal_tasks(conn, created.goal_id)

        assert waiting.action == "AWAITING_TRUSTED_RESULT"
        assert waiting.task_id is None
        assert waiting_goal is not None
        assert waiting_goal.current_stage == "VERIFY_PROMOTE"
        assert [binding.stage for binding in waiting_bindings] == [
            "PLAN",
            "BUILD_CANDIDATE",
        ]

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
            summary="acceptance check failed but a bounded repair is possible",
        )
        adjudicate_result = apply_trusted_stage_result(conn, created.goal_id, evidence)
        adjudicate_goal = get_durable_goal(conn, created.goal_id)
        adjudicate_bindings = list_durable_goal_tasks(conn, created.goal_id)

        assert adjudicate_result.action == "CREATED_ADJUDICATE"
        assert adjudicate_result.task_id is not None
        assert adjudicate_goal is not None
        assert adjudicate_goal.current_stage == "ADJUDICATE"
        assert [binding.stage for binding in adjudicate_bindings] == [
            "PLAN",
            "BUILD_CANDIDATE",
            "ADJUDICATE",
        ]

        _complete_v2_task(
            conn,
            task_id=adjudicate_result.task_id,
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
        bindings = list_durable_goal_tasks(conn, created.goal_id)
        repair = kb.get_task(conn, result.task_id) if result.task_id else None

    assert result.action == "CREATED_REPAIR"
    assert replay.action == "NOOP"
    assert goal is not None
    assert goal.current_stage == "REPAIR_BUILD_1"
    assert goal.repair_attempts_reserved == goal.repair_budget == 1
    assert [binding.stage for binding in bindings] == [
        "PLAN",
        "BUILD_CANDIDATE",
        "ADJUDICATE",
        "REPAIR_BUILD_1",
    ]
    assert sum(binding.stage == "REPAIR_BUILD_1" for binding in bindings) == 1
    assert all(binding.stage != "VERIFY" for binding in bindings)
    assert repair is not None and repair.assignee == "builder"


def test_repair_binds_failed_input_sha_and_advances_new_candidate_to_verify(
    kanban_home,
):
    old_sha = "c" * 40
    new_sha = "d" * 40
    with kb.connect() as conn:
        created, repair_id = _create_goal_at_repair(
            conn,
            candidate_sha=old_sha,
        )
        goal_before_repair = get_durable_goal(conn, created.goal_id)
        assert goal_before_repair is not None
        assert goal_before_repair.task_contract is not None
        _complete_v2_task(
            conn,
            task_id=repair_id,
            profile="builder",
            contract=goal_before_repair.task_contract,
            stage="REPAIR_BUILD_1",
            fields={
                "input_candidate_sha": old_sha,
                "candidate_sha": new_sha,
            },
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
        bindings = list_durable_goal_tasks(conn, created.goal_id)

    assert result.action == "AWAITING_TRUSTED_RESULT"
    assert result.task_id is None
    assert replay.action == "NOOP"
    assert replay.task_id is None
    assert goal is not None
    assert goal.current_stage == "VERIFY_PROMOTE"
    assert goal.candidate_sha == new_sha
    assert [binding.stage for binding in bindings] == [
        "PLAN",
        "BUILD_CANDIDATE",
        "REVIEW",
        "ADJUDICATE",
        "REPAIR_BUILD_1",
    ]
    assert all(binding.stage != "VERIFY" for binding in bindings)
    assert bindings[-1].expected_candidate_sha == old_sha


def test_repair_for_different_input_candidate_sha_fails_closed(kanban_home):
    old_sha = "c" * 40
    with kb.connect() as conn:
        created, repair_id = _create_goal_at_repair(
            conn,
            candidate_sha=old_sha,
        )
        goal_before_repair = get_durable_goal(conn, created.goal_id)
        assert goal_before_repair is not None
        assert goal_before_repair.task_contract is not None
        _complete_v2_task(
            conn,
            task_id=repair_id,
            profile="builder",
            contract=goal_before_repair.task_contract,
            stage="REPAIR_BUILD_1",
            fields={
                "input_candidate_sha": "e" * 40,
                "candidate_sha": "d" * 40,
            },
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
        bindings = list_durable_goal_tasks(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)

    assert result.action == "BLOCKED"
    assert replay.action == "NOOP"
    assert goal is not None and goal.status == "BLOCKED"
    assert goal.blocked_reason == "repair_input_candidate_mismatch"
    assert [binding.stage for binding in bindings] == [
        "PLAN",
        "BUILD_CANDIDATE",
        "REVIEW",
        "ADJUDICATE",
        "REPAIR_BUILD_1",
    ]
    assert all(binding.stage != "VERIFY" for binding in bindings)
    assert [notification.kind for notification in notifications] == [
        "UNRECOVERABLE_FAILURE"
    ]
    assert notifications[0].payload["candidate_sha"] == old_sha
    assert notifications[0].payload["reason"] == "repair_input_candidate_mismatch"


def test_malformed_verify_payload_blocks_without_successor_and_notifies_once(
    kanban_home,
):
    candidate_sha = "3" * 40
    with kb.connect() as conn:
        created, build_id, contract = _create_v2_goal_at_build(
            conn,
            candidate_key=candidate_sha,
            origin=GoalOrigin(platform="slack", chat_id="owner-channel"),
        )
        _complete_v2_task(
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
        assert waiting.task_id is None
        mismatched_evidence = PromotionEvidence.create(
            adapter_id="fixture-adapter",
            request=PromotionRequest(
                contract_hash=contract.contract_hash,
                base_revision=contract.base_revision,
                scope=contract.scope,
                gates=contract.gates,
                candidate_sha="4" * 40,
                attempt=0,
            ),
            classification=ResultClassification.PASS,
            summary="candidate identity does not match the built candidate",
        )

        result = apply_trusted_stage_result(conn, created.goal_id, mismatched_evidence)
        replay = apply_trusted_stage_result(conn, created.goal_id, mismatched_evidence)
        goal = get_durable_goal(conn, created.goal_id)
        bindings = list_durable_goal_tasks(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)

    assert result.action == "NOOP"
    assert result.task_id is None
    assert result.reason == "trusted_evidence_mismatch"
    assert replay.action == "NOOP"
    assert replay.task_id is None
    assert replay.reason == "trusted_evidence_mismatch"
    assert goal is not None and goal.status == "ACTIVE"
    assert goal.current_stage == "VERIFY_PROMOTE"
    assert goal.candidate_sha == candidate_sha
    assert [binding.stage for binding in bindings] == ["PLAN", "BUILD_CANDIDATE"]
    assert all(
        binding.stage not in {"VERIFY", "REVIEW", "ADJUDICATE", "READY_FOR_OWNER"}
        for binding in bindings
    )
    assert notifications == []


def test_review_approve_reaches_ready_for_owner_with_one_replay_safe_outbox(
    kanban_home,
):
    candidate_sha = "c" * 40
    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn,
            candidate_sha=candidate_sha,
            origin=GoalOrigin(
                platform="telegram",
                chat_id="owner-chat",
                thread_id="topic-9",
                notifier_profile="gateway-owner",
            ),
        )
        _complete_v2_review(
            conn,
            created=created,
            review_id=review_id,
            candidate_sha=candidate_sha,
            verdict="APPROVE",
            findings=[],
        )
        review_result = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal_at_adjudicate = get_durable_goal(conn, created.goal_id)
        bindings_at_adjudicate = list_durable_goal_tasks(conn, created.goal_id)
        assert review_result.action == "CREATED_ADJUDICATE"
        assert review_result.task_id is not None
        assert goal_at_adjudicate is not None
        assert goal_at_adjudicate.current_stage == "ADJUDICATE"
        assert goal_at_adjudicate.task_contract is not None
        assert [binding.stage for binding in bindings_at_adjudicate] == [
            "PLAN",
            "BUILD_CANDIDATE",
            "REVIEW",
            "ADJUDICATE",
        ]
        _complete_v2_task(
            conn,
            task_id=review_result.task_id,
            profile="orchestrator",
            contract=goal_at_adjudicate.task_contract,
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
        replay = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)
        bindings = list_durable_goal_tasks(conn, created.goal_id)

    assert result.action == "READY_FOR_OWNER"
    assert replay.action == "NOOP"
    assert goal is not None
    assert goal.status == "READY_FOR_OWNER"
    assert goal.current_stage == "READY_FOR_OWNER"
    assert [binding.stage for binding in bindings] == [
        "PLAN",
        "BUILD_CANDIDATE",
        "REVIEW",
        "ADJUDICATE",
    ]
    assert all(binding.stage != "VERIFY" for binding in bindings)
    assert len(notifications) == 1
    notification = notifications[0]
    assert notification.kind == "READY_FOR_OWNER"
    assert notification.origin.platform == "telegram"
    assert notification.origin.chat_id == "owner-chat"
    assert notification.origin.thread_id == "topic-9"
    assert notification.payload["candidate_sha"] == candidate_sha
    assert notification.delivered_at is None


def test_restart_after_review_ready_outbox_before_delivery_creates_no_duplicate(
    kanban_home,
):
    candidate_sha = "c" * 40
    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn,
            candidate_sha=candidate_sha,
        )
        _complete_v2_review(
            conn,
            created=created,
            review_id=review_id,
            candidate_sha=candidate_sha,
            verdict="APPROVE",
            findings=[],
        )
        review_result = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal_at_adjudicate = get_durable_goal(conn, created.goal_id)
        assert review_result.action == "CREATED_ADJUDICATE"
        assert review_result.task_id is not None
        assert goal_at_adjudicate is not None
        assert goal_at_adjudicate.current_stage == "ADJUDICATE"
        assert goal_at_adjudicate.task_contract is not None
        _complete_v2_task(
            conn,
            task_id=review_result.task_id,
            profile="orchestrator",
            contract=goal_at_adjudicate.task_contract,
            stage="ADJUDICATE",
            fields={
                "candidate_sha": candidate_sha,
                "decision": "READY_FOR_OWNER",
            },
        )
        first = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        assert first.action == "READY_FOR_OWNER"

    with kb.connect() as restarted:
        replay = supervise_board_once(
            restarted,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(restarted, created.goal_id)
        notifications = list_goal_notifications(restarted, created.goal_id)
        bindings = list_durable_goal_tasks(restarted, created.goal_id)

    assert replay == []
    assert goal is not None and goal.status == "READY_FOR_OWNER"
    assert [binding.stage for binding in bindings] == [
        "PLAN",
        "BUILD_CANDIDATE",
        "REVIEW",
        "ADJUDICATE",
    ]
    assert all(binding.stage != "VERIFY" for binding in bindings)
    assert [notification.kind for notification in notifications] == ["READY_FOR_OWNER"]
    assert notifications[0].delivered_at is None


def test_only_exact_origin_and_candidate_can_mark_ready_goal_completed(
    kanban_home,
):
    candidate_sha = "5" * 40
    owner = GoalOrigin(
        platform="telegram",
        chat_id="owner-chat",
        user_id="owner-7",
    )
    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn,
            candidate_sha=candidate_sha,
            origin=owner,
        )
        _complete_v2_review(
            conn,
            created=created,
            review_id=review_id,
            candidate_sha=candidate_sha,
            verdict="APPROVE",
            findings=[],
        )
        review_result = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal_at_adjudicate = get_durable_goal(conn, created.goal_id)
        assert review_result.action == "CREATED_ADJUDICATE"
        assert review_result.task_id is not None
        assert goal_at_adjudicate is not None
        assert goal_at_adjudicate.current_stage == "ADJUDICATE"
        assert goal_at_adjudicate.task_contract is not None
        _complete_v2_task(
            conn,
            task_id=review_result.task_id,
            profile="orchestrator",
            contract=goal_at_adjudicate.task_contract,
            stage="ADJUDICATE",
            fields={
                "candidate_sha": candidate_sha,
                "decision": "READY_FOR_OWNER",
            },
        )
        assert (
            supervise_goal_once(
                conn,
                created.goal_id,
                board="default",
                runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
            ).action
            == "READY_FOR_OWNER"
        )

        wrong_origin = mark_durable_goal_completed_by_owner(
            conn,
            goal_id=created.goal_id,
            origin=GoalOrigin(platform="telegram", chat_id="different-chat"),
            candidate_sha=candidate_sha,
        )
        wrong_sha = mark_durable_goal_completed_by_owner(
            conn,
            goal_id=created.goal_id,
            origin=owner,
            candidate_sha="6" * 40,
        )
        wrong_user = mark_durable_goal_completed_by_owner(
            conn,
            goal_id=created.goal_id,
            origin=GoalOrigin(
                platform="telegram",
                chat_id="owner-chat",
                user_id="different-user",
            ),
            candidate_sha=candidate_sha,
        )
        completed = mark_durable_goal_completed_by_owner(
            conn,
            goal_id=created.goal_id,
            origin=owner,
            candidate_sha=candidate_sha,
        )
        replay = mark_durable_goal_completed_by_owner(
            conn,
            goal_id=created.goal_id,
            origin=owner,
            candidate_sha=candidate_sha,
        )
        goal = get_durable_goal(conn, created.goal_id)
        bindings = list_durable_goal_tasks(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)

    assert wrong_origin is False
    assert wrong_sha is False
    assert wrong_user is False
    assert completed is True
    assert replay is False
    assert goal is not None and goal.status == "COMPLETED"
    assert [binding.stage for binding in bindings] == [
        "PLAN",
        "BUILD_CANDIDATE",
        "REVIEW",
        "ADJUDICATE",
    ]
    assert all(binding.stage != "VERIFY" for binding in bindings)
    assert [notification.kind for notification in notifications] == [
        "READY_FOR_OWNER",
        "COMPLETED",
    ]
    assert all(
        notification.payload["candidate_sha"] == candidate_sha
        for notification in notifications
    )


def test_missing_reviewer_skill_blocks_before_claim_or_spawn_and_notifies_once(
    kanban_home, monkeypatch
):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    candidate_sha = "d" * 40
    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn,
            candidate_sha=candidate_sha,
            reviewer_skill_digest=None,
            origin=GoalOrigin(platform="discord", chat_id="owner-channel"),
        )
        spawned: list[str] = []

        def _spawn(task, _workspace, **_kwargs):
            spawned.append(task.id)
            return 123

        first = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
            durable_goal_skill_available=lambda _profile, _skill: False,
        )
        replay = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
            durable_goal_skill_available=lambda _profile, _skill: False,
        )
        goal = get_durable_goal(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)
        runs = kb.list_runs(conn, review_id)

    assert first.auto_blocked == [review_id]
    assert replay.auto_blocked == []
    assert spawned == []
    assert runs == []
    assert goal is not None
    assert goal.status == "BLOCKED_CAPABILITY"
    assert goal.blocked_reason == "missing required skill: immutable-change-reviews"
    assert [item.kind for item in notifications] == ["BLOCKED_CAPABILITY"]


def test_missing_reviewer_skill_uses_target_profile_filesystem_catalog(
    kanban_home,
):
    reviewer_profile = kanban_home / "profiles" / "reviewer"
    (reviewer_profile / "skills").mkdir(parents=True)
    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn,
            candidate_sha="e" * 40,
        )
        spawned: list[str] = []

        def _spawn(task, _workspace, **_kwargs):
            spawned.append(task.id)
            return 123

        first = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        replay = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)
        review_task = kb.get_task(conn, review_id)
        runs = kb.list_runs(conn, review_id)

    assert first.auto_blocked == [review_id]
    assert replay.auto_blocked == []
    assert spawned == []
    assert runs == []
    assert review_task is not None
    assert review_task.status == "blocked"
    assert goal is not None
    assert goal.status == "BLOCKED_CAPABILITY"
    assert goal.blocked_reason == "missing required skill: immutable-change-reviews"
    assert [item.kind for item in notifications] == ["BLOCKED_CAPABILITY"]


def test_pinned_reviewer_skill_digest_exact_tree_allows_dispatch(
    kanban_home, monkeypatch
):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    _create_reviewer_snapshot(kanban_home)
    expected_digest = _profile_skill_digest("reviewer", "immutable-change-reviews")
    assert expected_digest is not None
    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn,
            candidate_sha="e" * 40,
            reviewer_skill_digest=expected_digest,
        )
        spawned: list[str] = []

        def _spawn(task, _workspace, **_kwargs):
            spawned.append(task.id)
            return 123

        result = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(conn, created.goal_id)
        runs = kb.list_runs(conn, review_id)

    assert result.spawned and result.spawned[0][0] == review_id
    assert spawned == [review_id]
    assert len(runs) == 1
    assert goal is not None and goal.status == "ACTIVE"
    assert goal.reviewer_skill_digest == expected_digest


@pytest.mark.parametrize("mutation", ["changed_byte", "added_file", "symlink"])
def test_pinned_reviewer_skill_digest_mutation_fails_closed(
    kanban_home, monkeypatch, mutation
):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    skill_dir = _create_reviewer_snapshot(kanban_home)
    expected_digest = _profile_skill_digest("reviewer", "immutable-change-reviews")
    assert expected_digest is not None
    if mutation == "changed_byte":
        (skill_dir / "references" / "rules.md").write_text(
            "review changed bytes\n",
            encoding="utf-8",
        )
    elif mutation == "added_file":
        (skill_dir / "references" / "extra.md").write_text(
            "extra file changes digest\n",
            encoding="utf-8",
        )
    else:
        (skill_dir / "references" / "linked.md").symlink_to("rules.md")

    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn,
            candidate_sha="e" * 40,
            reviewer_skill_digest=expected_digest,
        )
        spawned: list[str] = []

        def _spawn(task, _workspace, **_kwargs):
            spawned.append(task.id)
            return 123

        first = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        replay = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(conn, created.goal_id)
        runs = kb.list_runs(conn, review_id)
        notifications = list_goal_notifications(conn, created.goal_id)

    assert first.auto_blocked == [review_id]
    assert replay.auto_blocked == []
    assert spawned == []
    assert runs == []
    assert goal is not None and goal.status == "BLOCKED_CAPABILITY"
    assert goal.blocked_reason == "missing required skill: immutable-change-reviews"
    assert [item.kind for item in notifications] == ["BLOCKED_CAPABILITY"]


def test_pinned_reviewer_skill_digest_hardlink_fails_closed(kanban_home, monkeypatch):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    skill_dir = _create_reviewer_snapshot(kanban_home)
    expected_digest = _profile_skill_digest("reviewer", "immutable-change-reviews")
    assert expected_digest is not None
    try:
        os.link(
            skill_dir / "references" / "rules.md",
            skill_dir / "references" / "rules-hardlink.md",
        )
    except OSError:
        pytest.skip("hardlinks are not supported on this filesystem")

    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn,
            candidate_sha="e" * 40,
            reviewer_skill_digest=expected_digest,
        )
        spawned: list[str] = []

        def _spawn(task, _workspace, **_kwargs):
            spawned.append(task.id)
            return 123

        result = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(conn, created.goal_id)
        runs = kb.list_runs(conn, review_id)
        notifications = list_goal_notifications(conn, created.goal_id)

    assert result.auto_blocked == [review_id]
    assert spawned == []
    assert runs == []
    assert goal is not None and goal.status == "BLOCKED_CAPABILITY"
    assert [item.kind for item in notifications] == ["BLOCKED_CAPABILITY"]


def test_pinned_reviewer_skill_mutation_after_preflight_fails_child_preload(
    kanban_home, monkeypatch
):
    """The spawned reviewer revalidates the parent-pinned tree, not just its name."""
    from agent.skill_commands import build_preloaded_skills_prompt
    from agent.skill_integrity import PINNED_SKILL_DIGESTS_ENV

    skill_dir, expected_digest, child_env = (
        _dispatch_pinned_reviewer_and_capture_child_env(kanban_home, monkeypatch)
    )
    (skill_dir / "references" / "rules.md").write_text(
        "mutated after dispatcher preflight\n",
        encoding="utf-8",
    )
    for key, value in child_env.items():
        monkeypatch.setenv(key, value)

    prompt, loaded, missing = build_preloaded_skills_prompt([
        "immutable-change-reviews"
    ])

    assert json.loads(child_env[PINNED_SKILL_DIGESTS_ENV]) == {
        "immutable-change-reviews": expected_digest,
    }
    assert prompt == ""
    assert loaded == []
    assert missing == ["immutable-change-reviews"]


@pytest.mark.parametrize("digest_bridge", [None, "{}"], ids=["absent", "empty"])
def test_scoped_reviewer_preload_requires_nonempty_dispatcher_digest_bridge(
    kanban_home, monkeypatch, digest_bridge
):
    """A durable reviewer child must never downgrade to an unpinned preload."""
    from agent.skill_commands import build_preloaded_skills_prompt
    from agent.skill_integrity import PINNED_SKILL_DIGESTS_ENV

    _skill_dir, _expected_digest, child_env = (
        _dispatch_pinned_reviewer_and_capture_child_env(kanban_home, monkeypatch)
    )
    for key, value in child_env.items():
        monkeypatch.setenv(key, value)
    assert child_env["HERMES_KANBAN_TASK"]
    assert child_env["HERMES_KANBAN_ROLE"] == "reviewer"
    if digest_bridge is None:
        monkeypatch.delenv(PINNED_SKILL_DIGESTS_ENV, raising=False)
    else:
        monkeypatch.setenv(PINNED_SKILL_DIGESTS_ENV, digest_bridge)

    prompt, loaded, missing = build_preloaded_skills_prompt([
        "immutable-change-reviews"
    ])

    assert prompt == ""
    assert loaded == []
    assert missing == ["immutable-change-reviews"]


def test_valid_reviewer_digest_bridge_does_not_pin_ordinary_sibling_skills(
    kanban_home, monkeypatch
):
    from agent.skill_commands import build_preloaded_skills_prompt

    _skill_dir, _expected_digest, child_env = (
        _dispatch_pinned_reviewer_and_capture_child_env(kanban_home, monkeypatch)
    )
    _create_profile_skill(kanban_home, "reviewer", "review-notes")
    for key, value in child_env.items():
        monkeypatch.setenv(key, value)

    prompt, loaded, missing = build_preloaded_skills_prompt([
        "immutable-change-reviews",
        "review-notes",
    ])

    assert prompt
    assert loaded == ["immutable-change-reviews", "review-notes"]
    assert missing == []


def test_unscoped_manual_reviewer_skill_preload_remains_unpinned(
    kanban_home, monkeypatch
):
    from agent.skill_commands import build_preloaded_skills_prompt
    from agent.skill_integrity import PINNED_SKILL_DIGESTS_ENV

    _create_reviewer_snapshot(kanban_home)
    monkeypatch.setenv("HERMES_HOME", str(kanban_home / "profiles" / "reviewer"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_ROLE", raising=False)
    monkeypatch.delenv(PINNED_SKILL_DIGESTS_ENV, raising=False)

    prompt, loaded, missing = build_preloaded_skills_prompt([
        "immutable-change-reviews"
    ])

    assert prompt
    assert loaded == ["immutable-change-reviews"]
    assert missing == []


def test_pinned_reviewer_skill_mutation_after_preload_fails_linked_read(
    kanban_home, monkeypatch
):
    """Every later linked-file read revalidates the full pinned skill tree."""
    from agent.skill_commands import build_preloaded_skills_prompt
    from tools.skills_tool import skill_view

    skill_dir, _expected_digest, child_env = (
        _dispatch_pinned_reviewer_and_capture_child_env(kanban_home, monkeypatch)
    )
    for key, value in child_env.items():
        monkeypatch.setenv(key, value)

    prompt, loaded, missing = build_preloaded_skills_prompt([
        "immutable-change-reviews"
    ])
    assert prompt, json.loads(skill_view("immutable-change-reviews"))
    assert loaded == ["immutable-change-reviews"]
    assert missing == []

    (skill_dir / "references" / "rules.md").write_text(
        "mutated after child preload\n",
        encoding="utf-8",
    )
    linked = json.loads(
        skill_view(
            "immutable-change-reviews",
            file_path="references/rules.md",
        )
    )

    assert linked["success"] is False
    assert "digest" in linked["error"].lower()


def test_pinned_reviewer_consumes_snapshot_not_reversible_read_text_bytes(
    kanban_home, monkeypatch
):
    """Consumer-only read_text substitution cannot bypass pinned-byte verification."""
    from agent.skill_commands import build_preloaded_skills_prompt
    from tools.skills_tool import skill_view

    skill_dir, _expected_digest, child_env = (
        _dispatch_pinned_reviewer_and_capture_child_env(kanban_home, monkeypatch)
    )
    reference = skill_dir / "references" / "rules.md"
    for key, value in child_env.items():
        monkeypatch.setenv(key, value)

    original_read_text = Path.read_text

    def reversible_consumer_bytes(path, *args, **kwargs):
        if path == skill_dir / "SKILL.md":
            return "---\nname: immutable-change-reviews\n---\nMUTATED_UNCHECKED_BODY\n"
        if path == reference:
            return "MUTATED_UNCHECKED_REFERENCE"
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reversible_consumer_bytes)

    prompt, loaded, missing = build_preloaded_skills_prompt([
        "immutable-change-reviews"
    ])
    assert loaded == ["immutable-change-reviews"]
    assert missing == []
    assert "MUTATED_UNCHECKED_BODY" not in prompt

    response = json.loads(
        skill_view("immutable-change-reviews", file_path="references/rules.md")
    )
    assert response["success"] is True
    assert response["content"] == "review exact immutable changes\n"
    assert "MUTATED_UNCHECKED_REFERENCE" not in response["content"]


def test_pinned_reviewer_rejects_mutated_verified_snapshot_bytes(
    kanban_home, monkeypatch
):
    """The exact byte map handed to consumers must itself match the pinned digest."""
    from agent import skill_integrity
    from agent.skill_commands import build_preloaded_skills_prompt

    skill_dir, _expected_digest, child_env = (
        _dispatch_pinned_reviewer_and_capture_child_env(kanban_home, monkeypatch)
    )
    for key, value in child_env.items():
        monkeypatch.setenv(key, value)

    original_snapshot = skill_integrity._read_skill_tree_snapshot

    def poisoned_snapshot(path):
        snapshot = original_snapshot(path)
        if snapshot is not None and path.resolve() == skill_dir.resolve():
            snapshot = dict(snapshot)
            snapshot["SKILL.md"] += b"\nMUTATED_UNCHECKED_BODY\n"
        return snapshot

    monkeypatch.setattr(skill_integrity, "_read_skill_tree_snapshot", poisoned_snapshot)

    prompt, loaded, missing = build_preloaded_skills_prompt([
        "immutable-change-reviews"
    ])
    assert prompt == ""
    assert loaded == []
    assert missing == ["immutable-change-reviews"]


def test_pinned_reviewer_disables_inline_shell_after_snapshot_verification(
    kanban_home, monkeypatch
):
    """Pinned bytes must not trigger post-verification reads from the live tree."""
    from agent import skill_commands, skill_preprocessing
    from tools.skills_tool import skill_view

    inline_skill = (
        "---\nname: immutable-change-reviews\n---\n"
        "# Immutable Review\n"
        "Verified literal: !`cat SKILL.md`\n"
        "Verified directory token: ${HERMES_SKILL_DIR}\n"
    )
    _skill_dir, _expected_digest, child_env = (
        _dispatch_pinned_reviewer_and_capture_child_env(
            kanban_home,
            monkeypatch,
            skill_text=inline_skill,
        )
    )
    for key, value in child_env.items():
        monkeypatch.setenv(key, value)

    shell_calls: list[str] = []
    discovery_calls: list[str] = []
    original_glob = Path.glob

    def live_glob(path, pattern):
        if path == _skill_dir / "references":
            discovery_calls.append(pattern)
            return iter([_skill_dir / "references" / "MUTATED_UNCHECKED_FILENAME.md"])
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", live_glob)
    monkeypatch.setattr(
        skill_commands,
        "_load_skills_config",
        lambda: {"template_vars": True, "inline_shell": True},
    )
    monkeypatch.setattr(
        skill_commands,
        "_expand_inline_shell",
        lambda content, *_args: (
            shell_calls.append("preload") or "MUTATED_UNCHECKED_BODY"
        ),
    )
    monkeypatch.setattr(
        skill_preprocessing,
        "load_skills_config",
        lambda: {"template_vars": True, "inline_shell": True},
    )
    monkeypatch.setattr(
        skill_preprocessing,
        "run_inline_shell",
        lambda *_args, **_kwargs: (
            shell_calls.append("skill_view") or "MUTATED_UNCHECKED_BODY"
        ),
    )

    prompt, loaded, missing = skill_commands.build_preloaded_skills_prompt([
        "immutable-change-reviews"
    ])
    assert loaded == ["immutable-change-reviews"]
    assert missing == []
    assert "Verified literal: !`cat SKILL.md`" in prompt
    assert "Verified directory token: ${HERMES_SKILL_DIR}" in prompt
    assert "MUTATED_UNCHECKED_BODY" not in prompt
    assert "MUTATED_UNCHECKED_FILENAME" not in prompt
    assert "references/rules.md" in prompt
    assert "dispatcher-verified snapshot" in prompt
    assert str(_skill_dir) not in prompt
    assert "run scripts directly" not in prompt

    viewed = json.loads(skill_view("immutable-change-reviews", preprocess=True))
    assert viewed["success"] is True
    assert "Verified literal: !`cat SKILL.md`" in viewed["content"]
    assert "Verified directory token: ${HERMES_SKILL_DIR}" in viewed["content"]
    assert "MUTATED_UNCHECKED_BODY" not in viewed["content"]
    assert "path" not in viewed
    assert "skill_dir" not in viewed
    assert shell_calls == []
    assert discovery_calls == []


def test_pinned_reviewer_missing_file_inventory_uses_verified_snapshot(
    kanban_home, monkeypatch
):
    """Missing-file hints must not reopen the mutable live skill tree."""
    from agent import skill_integrity
    from tools.skills_tool import skill_view

    skill_dir, _expected_digest, child_env = (
        _dispatch_pinned_reviewer_and_capture_child_env(kanban_home, monkeypatch)
    )
    for key, value in child_env.items():
        monkeypatch.setenv(key, value)

    original_snapshot = skill_integrity._read_skill_tree_snapshot
    injected = False

    def snapshot_then_live_filename(path):
        nonlocal injected
        snapshot = original_snapshot(path)
        if (
            snapshot is not None
            and not injected
            and path.resolve() == skill_dir.resolve()
        ):
            injected = True
            (skill_dir / "references" / "LIVE_UNCHECKED_FILENAME.md").write_text(
                "unchecked",
                encoding="utf-8",
            )
        return snapshot

    monkeypatch.setattr(
        skill_integrity,
        "_read_skill_tree_snapshot",
        snapshot_then_live_filename,
    )

    response = json.loads(
        skill_view(
            "immutable-change-reviews",
            file_path="references/does-not-exist.md",
        )
    )
    assert response["success"] is False
    assert response["available_files"] == {"references": ["references/rules.md"]}
    assert "LIVE_UNCHECKED_FILENAME" not in json.dumps(response)


def test_pinned_reviewer_resolution_ignores_live_name_collisions(
    kanban_home, monkeypatch
):
    """Pinned lookup must select only the exact profile-local pinned directory."""
    from agent import skill_utils
    from tools.skills_tool import skill_view

    skill_dir, _expected_digest, child_env = (
        _dispatch_pinned_reviewer_and_capture_child_env(kanban_home, monkeypatch)
    )
    for key, value in child_env.items():
        monkeypatch.setenv(key, value)

    initial = json.loads(skill_view("immutable-change-reviews"))
    assert initial["success"] is True
    assert initial["_pinned_snapshot_verified"] is True
    expected_rules = (skill_dir / "references" / "rules.md").read_text(encoding="utf-8")

    collision_dir = (
        kanban_home
        / "profiles"
        / "reviewer"
        / "skills"
        / "LIVE_UNCHECKED_PROMPT_MARKER"
    )
    collision_dir.mkdir(parents=True)
    (collision_dir / "SKILL.md").write_text(
        "---\nname: immutable-change-reviews\n---\nLIVE_UNCHECKED_COLLISION\n",
        encoding="utf-8",
    )

    def forbidden_discovery(*_args, **_kwargs):
        raise AssertionError("pinned skill lookup must not scan the mutable skill tree")

    monkeypatch.setattr(skill_utils, "iter_skill_index_files", forbidden_discovery)

    main = json.loads(skill_view("immutable-change-reviews"))
    linked = json.loads(
        skill_view("immutable-change-reviews", file_path="references/rules.md")
    )
    assert main["success"] is True
    assert linked["success"] is True
    assert linked["content"] == expected_rules
    combined = json.dumps([main, linked])
    assert "LIVE_UNCHECKED" not in combined
    assert "Ambiguous skill name" not in combined
    assert str(collision_dir) not in combined


def test_durable_review_without_pinned_digest_never_spawns(kanban_home, monkeypatch):
    """A durable REVIEW binding cannot silently degrade to an unpinned child."""
    with kb.connect() as conn:
        _created, review_id = _create_goal_at_review(
            conn,
            candidate_sha="e" * 40,
            reviewer_skill_digest=None,
        )
        review = kb.get_task(conn, review_id)
    assert review is not None

    monkeypatch.setattr(
        "subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("unpinned reviewer spawned"),
    )
    with pytest.raises(RuntimeError, match="no valid pinned skill digest"):
        kb._default_spawn(review, str(kanban_home.parent / "repo"), board="default")


def test_durable_dispatch_foreign_board_fails_closed_but_legacy_task_spawns(
    kanban_home, monkeypatch
):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    kb.write_board_metadata("foreign", default_workdir=str(kanban_home.parent / "repo"))
    with kb.connect() as conn:
        # V1_MIGRATION: V1 retirement is mandatory before any board-specific check.
        v1_created = create_durable_goal(
            conn,
            objective="Do not dispatch from the wrong board",
            origin=GoalOrigin(platform="telegram", chat_id="owner"),
            board="default",
            builder_profile="builder",
            verifier_profile="verifier",
            reviewer_profile="reviewer",
            repair_budget=1,
            review_retry_budget=1,
        )
        legacy_id = _create_legacy_ready_task(
            title="ordinary legacy dispatch still works",
            conn=conn,
            assignee="builder",
        )
        spawned: list[str] = []

        def _spawn(task, _workspace, **_kwargs):
            spawned.append(task.id)
            return 123

        result = kb.dispatch_once(
            conn,
            board="foreign",
            spawn_fn=_spawn,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(conn, v1_created.goal_id)
        bindings = list_durable_goal_tasks(conn, v1_created.goal_id)
        notifications = list_goal_notifications(conn, v1_created.goal_id)
        runs = kb.list_runs(conn, v1_created.task_id)
        legacy = kb.get_task(conn, legacy_id)

    assert result.auto_blocked == [v1_created.task_id]
    assert spawned == [legacy_id]
    assert runs == []
    assert legacy is not None and legacy.status == "running"
    assert goal is not None
    assert goal.status == "BLOCKED_CAPABILITY"
    assert (
        goal.blocked_reason
        == "durable workflow V1 is retired; owner intervention is required"
    )
    assert [binding.stage for binding in bindings] == ["BUILD"]
    assert all(binding.stage != "VERIFY" for binding in bindings)
    assert [item.kind for item in notifications] == ["BLOCKED_CAPABILITY"]
    assert notifications[0].payload["reason"] == goal.blocked_reason


def test_durable_dispatch_protocol_skew_fails_closed_but_legacy_task_spawns(
    kanban_home, monkeypatch
):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    with kb.connect() as conn:
        created = create_durable_goal(
            conn,
            objective="Do not dispatch from a skewed supervisor runtime",
            origin=GoalOrigin(platform="telegram", chat_id="owner"),
            board="default",
            builder_profile="builder",
            verifier_profile="verifier",
            reviewer_profile="reviewer",
            repair_budget=1,
            review_retry_budget=1,
        )
        legacy_id = _create_legacy_ready_task(
            title="ordinary task ignores durable supervisor skew",
            conn=conn,
            assignee="builder",
        )
        spawned: list[str] = []

        def _spawn(task, _workspace, **_kwargs):
            spawned.append(task.id)
            return 123

        result = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION + 1,
        )
        goal = get_durable_goal(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)
        runs = kb.list_runs(conn, created.task_id)
        legacy = kb.get_task(conn, legacy_id)

    assert result.auto_blocked == [created.task_id]
    assert spawned == [legacy_id]
    assert runs == []
    assert legacy is not None and legacy.status == "running"
    assert goal is not None
    assert goal.status == "BLOCKED_CAPABILITY"
    assert "workflow V1 is retired" in (goal.blocked_reason or "")
    assert [item.kind for item in notifications] == ["BLOCKED_CAPABILITY"]


def test_durable_dispatch_preflight_checks_review_skill_once(kanban_home, monkeypatch):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(conn, candidate_sha="4" * 40)
        calls: list[tuple[str, str]] = []
        spawned: list[str] = []

        def _skill_available(profile, skill):
            calls.append((profile, skill))
            return True

        def _spawn(task, _workspace, **_kwargs):
            spawned.append(task.id)
            return 123

        result = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
            durable_goal_skill_available=_skill_available,
        )

    assert result.spawned and result.spawned[0][0] == review_id
    assert spawned == [review_id]
    assert calls == [("reviewer", "immutable-change-reviews")]
    with kb.connect() as conn:
        goal = get_durable_goal(conn, created.goal_id)
        assert goal is not None and goal.status == "ACTIVE"


def test_old_dispatcher_without_protocol_cannot_see_durable_review(
    kanban_home, monkeypatch
):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(conn, candidate_sha="4" * 40)
        spawned: list[str] = []

        def _spawn(task, _workspace, **_kwargs):
            spawned.append(task.id)
            return 123

        result = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn,
        )
        goal = get_durable_goal(conn, created.goal_id)
        review = kb.get_task(conn, review_id)
        runs = kb.list_runs(conn, review_id)
        notifications = list_goal_notifications(conn, created.goal_id)

    assert result.spawned == []
    assert result.auto_blocked == []
    assert spawned == []
    assert runs == []
    assert goal is not None and goal.status == "ACTIVE"
    assert review is not None
    assert review.status == "durable_review"
    assert review.skills == ["immutable-change-reviews"]
    assert notifications == []


def test_new_dispatcher_claims_durable_review_after_single_preflight(
    kanban_home, monkeypatch
):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(conn, candidate_sha="4" * 40)
        calls: list[tuple[str, str]] = []
        spawned: list[tuple[str, list[str], str | None]] = []

        def _skill_available(profile, skill):
            calls.append((profile, skill))
            return True

        def _spawn(task, _workspace, **_kwargs):
            spawned.append((task.id, task.skills, task.dispatch_role))
            return 123

        result = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
            durable_goal_skill_available=_skill_available,
        )
        goal = get_durable_goal(conn, created.goal_id)
        review = kb.get_task(conn, review_id)
        runs = kb.list_runs(conn, review_id)

    assert result.spawned and result.spawned[0][0] == review_id
    assert spawned == [(review_id, ["immutable-change-reviews"], "reviewer")]
    assert calls == [("reviewer", "immutable-change-reviews")]
    assert len(runs) == 1
    assert review is not None and review.status == "running"
    assert goal is not None and goal.status == "ACTIVE"


def test_durable_review_protocol_skew_blocks_once_before_run(kanban_home, monkeypatch):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(conn, candidate_sha="4" * 40)
        spawned: list[str] = []

        def _spawn(task, _workspace, **_kwargs):
            spawned.append(task.id)
            return 123

        first = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION + 1,
        )
        replay = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION + 1,
        )
        goal = get_durable_goal(conn, created.goal_id)
        review = kb.get_task(conn, review_id)
        runs = kb.list_runs(conn, review_id)
        notifications = list_goal_notifications(conn, created.goal_id)

    assert first.auto_blocked == [review_id]
    assert replay.auto_blocked == []
    assert spawned == []
    assert runs == []
    assert review is not None and review.status == "blocked"
    assert goal is not None and goal.status == "BLOCKED_CAPABILITY"
    assert "protocol mismatch" in (goal.blocked_reason or "")
    assert [item.kind for item in notifications] == ["BLOCKED_CAPABILITY"]


def test_ordinary_review_dispatch_keeps_legacy_sdlc_review_path(
    kanban_home, monkeypatch
):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    with kb.connect() as conn:
        review_id = kb.create_task(
            conn,
            title="ordinary review",
            assignee="reviewer",
        )
        conn.execute(
            "UPDATE tasks SET status = 'review', claim_lock = NULL WHERE id = ?",
            (review_id,),
        )
        conn.commit()
        spawned: list[tuple[str, list[str], str | None]] = []

        def _spawn(task, _workspace, **_kwargs):
            spawned.append((task.id, task.skills, task.dispatch_role))
            return 123

        result = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn,
        )
        review = kb.get_task(conn, review_id)
        runs = kb.list_runs(conn, review_id)

    assert result.spawned and result.spawned[0][0] == review_id
    assert spawned == [(review_id, ["sdlc-review"], "reviewer")]
    assert len(runs) == 1
    assert review is not None and review.status == "running"


def test_exhausted_repair_budget_blocks_unrecoverable_without_successor(
    kanban_home, monkeypatch
):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    candidate_sha = "d" * 40
    with kb.connect() as conn:
        created, build_id, contract = _create_v2_goal_at_build(
            conn,
            candidate_key=candidate_sha,
            origin=GoalOrigin(platform="telegram", chat_id="owner-chat"),
            repair_budget=0,
        )
        _complete_v2_task(
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
        assert waiting.task_id is None

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
            summary="deterministic evidence allows a bounded repair",
        )
        adjudicate = apply_trusted_stage_result(conn, created.goal_id, evidence)
        assert adjudicate.action == "CREATED_ADJUDICATE"
        assert adjudicate.task_id is not None
        _complete_v2_task(
            conn,
            task_id=adjudicate.task_id,
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
        bindings = list_durable_goal_tasks(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)

    assert result.action == "BLOCKED"
    assert replay.action == "NOOP"
    assert goal is not None and goal.status == "BLOCKED"
    assert goal.repair_attempts_reserved == goal.repair_budget == 0
    assert "repair" in (goal.blocked_reason or "").lower()
    assert "budget" in (goal.blocked_reason or "").lower()
    assert [binding.stage for binding in bindings] == [
        "PLAN",
        "BUILD_CANDIDATE",
        "ADJUDICATE",
    ]
    assert all(binding.stage != "VERIFY" for binding in bindings)
    assert all(not binding.stage.startswith("REPAIR_BUILD_") for binding in bindings)
    assert [item.kind for item in notifications] == ["HUMAN_GATE"]


@pytest.mark.parametrize("payload_run_id", [0, -1, "malformed", 999999])
def test_build_payload_with_noncurrent_run_id_fails_closed(kanban_home, payload_run_id):
    with kb.connect() as conn:
        created, build_id, contract = _create_v2_goal_at_build(
            conn,
            candidate_key=f"stale-run-{payload_run_id}",
            origin=GoalOrigin(platform="telegram", chat_id="owner-chat"),
        )
        build = _complete_v2_task(
            conn,
            task_id=build_id,
            profile="builder",
            contract=contract,
            stage="BUILD_CANDIDATE",
            fields={"run_id": payload_run_id, "candidate_sha": "e" * 40},
        )
        assert build.current_run_id is not None
        assert payload_run_id != build.current_run_id
        completed_run = kb.get_run(conn, build.current_run_id)
        assert completed_run is not None
        assert completed_run.metadata["durable_goal"]["run_id"] == payload_run_id
        closing_event = next(
            event
            for event in reversed(kb.list_events(conn, build.id))
            if event.kind == "completed" and event.run_id == build.current_run_id
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

    assert result.action == "BLOCKED"
    assert goal is not None and goal.status == "BLOCKED"
    assert goal.blocked_reason == "invalid_structured_build"
    assert [binding.stage for binding in bindings] == ["PLAN", "BUILD_CANDIDATE"]
    assert bindings[1].expected_run_id == build.current_run_id
    assert bindings[1].completion_event_id == closing_event.id
    assert bindings[1].completion_payload["run_id"] == build.current_run_id
    assert all(binding.stage != "VERIFY" for binding in bindings)
    assert all(not binding.stage.startswith("REPAIR_BUILD_") for binding in bindings)
    assert [item.kind for item in notifications] == ["UNRECOVERABLE_FAILURE"]


def test_unstructured_review_reserves_one_retry_before_successor_and_replays(
    kanban_home,
):
    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn, candidate_sha="f" * 40, review_retry_budget=1
        )
        review = kb.claim_review_task(
            conn, review_id, claimer="reviewer", allow_durable=True
        )
        assert review is not None and review.current_run_id is not None
        assert kb.complete_task(
            conn,
            review.id,
            summary="APPROVE — prose is not structured authority",
            metadata={"_kanban_dispatch_role": "reviewer"},
            expected_run_id=review.current_run_id,
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
        bindings = list_durable_goal_tasks(conn, created.goal_id)
        retry = kb.get_task(conn, result.task_id) if result.task_id else None

    assert result.action == "CREATED_SUCCESSOR"
    assert replay.action == "NOOP"
    assert goal is not None
    assert goal.current_stage == "REVIEW_RETRY_1"
    assert goal.review_attempts_reserved == 1
    assert [binding.stage for binding in bindings] == [
        "PLAN",
        "BUILD_CANDIDATE",
        "REVIEW",
        "REVIEW_RETRY_1",
    ]
    assert all(binding.stage != "VERIFY" for binding in bindings)
    assert retry is not None and retry.status == "durable_review"
    assert retry.skills == ["immutable-change-reviews"]


def test_review_gave_up_reserves_one_retry_and_replay_creates_no_duplicate(
    kanban_home, monkeypatch
):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    candidate_sha = "6" * 40
    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn, candidate_sha=candidate_sha, review_retry_budget=1
        )

        def _spawn_failure(*_args, **_kwargs):
            raise RuntimeError("review worker crashed until the breaker gave up")

        dispatched = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn_failure,
            failure_limit=1,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
            durable_goal_skill_available=lambda _profile, _skill: True,
        )
        assert dispatched.auto_blocked == [review_id]

    with kb.connect() as restarted:
        result = supervise_goal_once(
            restarted,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
    with kb.connect() as replayed:
        replay = supervise_goal_once(
            replayed,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(replayed, created.goal_id)
        bindings = list_durable_goal_tasks(replayed, created.goal_id)
        retry = kb.get_task(replayed, result.task_id) if result.task_id else None

    assert result.action == "CREATED_SUCCESSOR"
    assert replay.action == "NOOP"
    assert goal is not None
    assert goal.candidate_sha == candidate_sha
    assert goal.current_stage == "REVIEW_RETRY_1"
    assert goal.review_attempts_reserved == 1
    assert [binding.stage for binding in bindings] == [
        "PLAN",
        "BUILD_CANDIDATE",
        "REVIEW",
        "REVIEW_RETRY_1",
    ]
    assert all(binding.stage != "VERIFY" for binding in bindings)
    assert retry is not None
    assert retry.status == "durable_review"
    assert retry.skills == ["immutable-change-reviews"]


def test_review_gave_up_blocks_once_when_retry_budget_is_exhausted(
    kanban_home, monkeypatch
):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    candidate_sha = "6" * 40
    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn,
            candidate_sha=candidate_sha,
            review_retry_budget=0,
        )

        def _spawn_failure(*_args, **_kwargs):
            raise RuntimeError("review worker gave up with no retry budget")

        dispatched = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn_failure,
            failure_limit=1,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
            durable_goal_skill_available=lambda _profile, _skill: True,
        )
        assert dispatched.auto_blocked == [review_id]

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
        bindings = list_durable_goal_tasks(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)

    assert result.action == "BLOCKED"
    assert replay.action == "NOOP"
    assert goal is not None and goal.status == "BLOCKED"
    assert goal.review_attempts_reserved == goal.review_retry_budget == 0
    assert goal.blocked_reason == "durable stage REVIEW gave up"
    assert [binding.stage for binding in bindings] == [
        "PLAN",
        "BUILD_CANDIDATE",
        "REVIEW",
    ]
    assert all(binding.stage != "VERIFY" for binding in bindings)
    assert all(binding.stage != "ADJUDICATE" for binding in bindings)
    assert all(binding.stage != "READY_FOR_OWNER" for binding in bindings)
    assert [notification.kind for notification in notifications] == ["HUMAN_GATE"]
    assert notifications[0].payload["reason"] == goal.blocked_reason


@pytest.mark.parametrize("event_run_id", [0, -1, 999999])
def test_review_gave_up_with_noncurrent_run_id_does_not_consume_retry_budget(
    kanban_home,
    event_run_id,
):
    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn, candidate_sha="6" * 40, review_retry_budget=1
        )
        # Circuit-breaker review failures are surfaced to the supervisor as a
        # terminal gave_up event. Stale/non-positive/noncurrent event run IDs
        # must be ignored by the same exact-current-run validation used for
        # structured completions.
        kb._append_event(
            conn,
            review_id,
            "gave_up",
            {"error": "stale review failure"},
            run_id=event_run_id,
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

    assert result.action == "NOOP"
    assert goal is not None
    assert goal.status == "ACTIVE"
    assert goal.current_stage == "REVIEW"
    assert goal.review_attempts_reserved == 0
    assert [binding.stage for binding in bindings] == [
        "PLAN",
        "BUILD_CANDIDATE",
        "REVIEW",
    ]
    assert all(binding.stage != "VERIFY" for binding in bindings)
    assert notifications == []


def test_unstructured_review_blocks_once_when_retry_budget_is_exhausted(
    kanban_home,
):
    candidate_sha = "8" * 40
    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn,
            candidate_sha=candidate_sha,
            review_retry_budget=0,
        )
        review = kb.claim_review_task(
            conn, review_id, claimer="reviewer", allow_durable=True
        )
        assert review is not None and review.current_run_id is not None
        assert kb.complete_task(
            conn,
            review.id,
            summary="APPROVE in prose is not a structured verdict",
            metadata={"_kanban_dispatch_role": "reviewer"},
            expected_run_id=review.current_run_id,
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
        bindings = list_durable_goal_tasks(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)

    assert result.action == "BLOCKED"
    assert replay.action == "NOOP"
    assert goal is not None and goal.status == "BLOCKED"
    assert goal.review_attempts_reserved == goal.review_retry_budget == 0
    assert goal.blocked_reason == "invalid_structured_review"
    assert [binding.stage for binding in bindings] == [
        "PLAN",
        "BUILD_CANDIDATE",
        "REVIEW",
    ]
    assert all(binding.stage != "VERIFY" for binding in bindings)
    assert all(binding.stage != "ADJUDICATE" for binding in bindings)
    assert all(binding.stage != "READY_FOR_OWNER" for binding in bindings)
    assert [notification.kind for notification in notifications] == ["HUMAN_GATE"]
    assert notifications[0].payload["reason"] == goal.blocked_reason


@pytest.mark.parametrize("severity", ["BLOCKER", "MAJOR"])
def test_structured_negative_review_verdict_blocks_and_notifies_once(
    kanban_home,
    severity,
):
    candidate_sha = "9" * 40
    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn,
            candidate_sha=candidate_sha,
        )
        _complete_v2_review(
            conn,
            created=created,
            review_id=review_id,
            candidate_sha=candidate_sha,
            verdict="CHANGES_REQUIRED",
            findings=[{"severity": severity, "summary": "must fix"}],
        )

        review_result = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        review_replay = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal_at_adjudicate = get_durable_goal(conn, created.goal_id)
        bindings_at_adjudicate = list_durable_goal_tasks(conn, created.goal_id)
        notifications_before_adjudication = list_goal_notifications(
            conn, created.goal_id
        )

        assert review_result.action == "CREATED_ADJUDICATE"
        assert review_result.task_id is not None
        assert review_replay.action == "NOOP"
        assert goal_at_adjudicate is not None
        assert goal_at_adjudicate.status == "ACTIVE"
        assert goal_at_adjudicate.current_stage == "ADJUDICATE"
        assert notifications_before_adjudication == []
        assert all(
            binding.stage != "READY_FOR_OWNER" for binding in bindings_at_adjudicate
        )
        assert goal_at_adjudicate.task_contract is not None

        human_gate_reason = f"{severity} review finding requires owner adjudication"
        _complete_v2_task(
            conn,
            task_id=review_result.task_id,
            profile="orchestrator",
            contract=goal_at_adjudicate.task_contract,
            stage="ADJUDICATE",
            fields={
                "candidate_sha": candidate_sha,
                "decision": "HUMAN_GATE",
                "reason": human_gate_reason,
            },
        )
        adjudication_result = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        adjudication_replay = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(conn, created.goal_id)
        bindings = list_durable_goal_tasks(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)

    assert adjudication_result.action == "HUMAN_GATE"
    assert adjudication_replay.action == "NOOP"
    assert goal is not None and goal.status == "BLOCKED"
    assert goal.blocked_reason == human_gate_reason
    assert [binding.stage for binding in bindings] == [
        "PLAN",
        "BUILD_CANDIDATE",
        "REVIEW",
        "ADJUDICATE",
    ]
    assert all(binding.stage != "VERIFY" for binding in bindings)
    assert all(binding.stage != "READY_FOR_OWNER" for binding in bindings)
    review_binding = next(binding for binding in bindings if binding.stage == "REVIEW")
    assert review_binding.completion_payload["verdict"] == "CHANGES_REQUIRED"
    assert review_binding.completion_payload["findings"] == [
        {"severity": severity, "summary": "must fix", "resolved": False}
    ]
    assert [notification.kind for notification in notifications] == ["HUMAN_GATE"]
    assert notifications[0].payload["candidate_sha"] == candidate_sha
    assert notifications[0].payload["reason"] == human_gate_reason


def test_review_verdict_for_different_candidate_sha_fails_closed(
    kanban_home,
):
    candidate_sha = "a" * 40
    forged_candidate_sha = "b" * 40
    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn,
            candidate_sha=candidate_sha,
            review_retry_budget=0,
        )
        review = _complete_v2_review(
            conn,
            created=created,
            review_id=review_id,
            candidate_sha=forged_candidate_sha,
            verdict="APPROVE",
            findings=[],
        )
        assert review.current_run_id is not None
        completed_run = kb.get_run(conn, review.current_run_id)
        assert completed_run is not None
        assert (
            completed_run.metadata["durable_goal"]["candidate_sha"]
            == forged_candidate_sha
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
        bindings = list_durable_goal_tasks(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)

    assert result.action == "BLOCKED"
    assert replay.action == "NOOP"
    assert goal is not None and goal.status == "BLOCKED"
    assert goal.review_attempts_reserved == goal.review_retry_budget == 0
    assert goal.blocked_reason == "invalid_structured_review"
    assert [binding.stage for binding in bindings] == [
        "PLAN",
        "BUILD_CANDIDATE",
        "REVIEW",
    ]
    assert bindings[-1].expected_run_id == review.current_run_id
    assert all(binding.stage != "VERIFY" for binding in bindings)
    assert all(binding.stage != "ADJUDICATE" for binding in bindings)
    assert all(binding.stage != "READY_FOR_OWNER" for binding in bindings)
    assert [notification.kind for notification in notifications] == ["HUMAN_GATE"]
    assert notifications[0].payload["candidate_sha"] == candidate_sha
    assert notifications[0].payload["reason"] == goal.blocked_reason


def test_review_completion_without_reviewer_role_fails_closed(kanban_home):
    candidate_sha = "7" * 40
    with kb.connect() as conn:
        created, review_id = _create_goal_at_review(
            conn,
            candidate_sha=candidate_sha,
            review_retry_budget=0,
        )
        goal_before = get_durable_goal(conn, created.goal_id)
        assert goal_before is not None and goal_before.task_contract is not None
        contract = goal_before.task_contract
        review = kb.claim_review_task(
            conn, review_id, claimer="reviewer", allow_durable=True
        )
        assert review is not None and review.current_run_id is not None
        assert kb.complete_task(
            conn,
            review.id,
            summary="valid verdict from a forged run profile",
            metadata={
                "durable_goal": {
                    "workflow_version": 2,
                    "protocol_version": DURABLE_GOAL_PROTOCOL_VERSION,
                    "stage": "REVIEW",
                    "run_id": review.current_run_id,
                    "contract_hash": contract.contract_hash,
                    "base_revision": contract.base_revision,
                    "scope": list(contract.scope),
                    "gates": list(contract.gates),
                    "authority": "reviewer",
                    "candidate_sha": candidate_sha,
                    "verdict": "APPROVE",
                    "findings": [],
                },
            },
            expected_run_id=review.current_run_id,
        )
        conn.execute(
            "UPDATE task_runs SET profile = 'builder' WHERE id = ?",
            (review.current_run_id,),
        )
        conn.commit()

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
        bindings = list_durable_goal_tasks(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)

    assert result.action == "BLOCKED"
    assert replay.action == "NOOP"
    assert goal is not None and goal.status == "BLOCKED"
    assert goal.current_stage == "BLOCKED"
    assert all(binding.stage != "ADJUDICATE" for binding in bindings)
    assert all(binding.stage != "READY_FOR_OWNER" for binding in bindings)
    assert len(notifications) == 1
    assert notifications[0].kind == "HUMAN_GATE"
    assert notifications[0].payload["reason"] == "invalid_structured_review"
