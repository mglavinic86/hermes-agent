"""Hermetic restart/replay acceptance for the opt-in durable goal route."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_goal_supervisor import (
    DURABLE_GOAL_PROTOCOL_VERSION,
    DURABLE_GOAL_SCHEMA_VERSION,
    get_durable_goal,
    list_durable_goal_tasks,
    list_goal_notifications,
    supervise_goal_once,
)
from tests.gateway.test_kanban_notifier import (
    RecordingAdapter,
    _make_runner,
    _run_one_notifier_tick,
)


class _SessionEntry:
    session_id = "durable-acceptance-origin"


class _SessionStore:
    def get_or_create_session(self, _source):
        return _SessionEntry()

    def _generate_session_key(self, _source):
        return "agent:main:telegram:dm:owner-chat"


def _init_checkout(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "--allow-empty", "-m", "base"],
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


def _run_real_gateway_tick(home: Path) -> dict:
    helper = Path(__file__).with_name("durable_goal_gateway_process.py")
    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    env.pop("HERMES_KANBAN_BOARD", None)
    env.pop("HERMES_KANBAN_DISPATCH_IN_GATEWAY", None)
    repo_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (repo_root, env.get("PYTHONPATH", "")) if part
    )
    completed = subprocess.run(
        [sys.executable, str(helper), "watch-once"],
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, completed.stderr
    return json.loads(lines[-1])


@pytest.mark.asyncio
async def test_one_route_survives_stage_restarts_and_notifies_owner_once(
    tmp_path,
    monkeypatch,
):
    from hermes_cli import profiles

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
kanban:
  dispatch_in_gateway: true
  dispatch_interval_seconds: 1
  auto_decompose: false
  durable_goals:
    board: default
    builder_profile: builder
    verifier_profile: verifier
    reviewer_profile: reviewer
    reviewer_skill_digest: 1e74b219dbf5377886fde11fcc873c673d3c01178a007ed9667a0942f8f10ec1
    repair_budget: 1
    review_retry_budget: 1
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "durable-acceptance.db"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    kb.init_db()
    checkout = tmp_path / "candidate-repo"
    _init_checkout(checkout)
    kb.write_board_metadata("default", default_workdir=str(checkout))
    with kb.connect() as conn:
        kb.upsert_durable_goal_runtime(
            conn,
            runtime_id="singleton-runtime-not-profile-identity",
            schema_version=DURABLE_GOAL_SCHEMA_VERSION,
            protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
            lease_seconds=300,
        )

    route_runner = object.__new__(GatewayRunner)
    route_runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="token")}
    )
    route_runner.session_store = _SessionStore()
    route_runner.adapters = {}
    route_runner._queued_events = {}
    route_runner._active_profile_name = lambda: "gateway-owner"
    route_runner._enqueue_fifo = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("durable goals must not enqueue a synthetic user turn")
    )
    initial_messages = [
        MessageEvent(
            text="/goal durable Ship the candidate without merge or deploy",
            message_type=MessageType.COMMAND,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="owner-chat",
                chat_type="dm",
                user_id="owner-7",
                thread_id="topic-42",
                profile="gateway-owner",
            ),
            message_id="only-inbound-message",
        )
    ]
    response = await GatewayRunner._handle_goal_command(
        route_runner,
        initial_messages.pop(),
    )
    goal_id = response.split()[3]
    assert route_runner._queued_events == {}
    assert initial_messages == []

    # BUILD terminal event, then a process/connection restart before supervision.
    with kb.connect() as conn:
        [build_binding] = list_durable_goal_tasks(conn, goal_id)

        def _build_gave_up(*_args, **_kwargs):
            raise RuntimeError("build worker gave up")

        dispatched = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_build_gave_up,
            failure_limit=1,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        assert dispatched.auto_blocked == [build_binding.task_id]

    with kb.connect() as conn:
        repair_result = supervise_goal_once(
            conn,
            goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        assert repair_result.action == "CREATED_SUCCESSOR"

    # Restart after successor creation; replay cannot mint another repair.
    with kb.connect() as conn:
        assert supervise_goal_once(
            conn,
            goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        ).action == "NOOP"
        repair = kb.claim_task(conn, repair_result.task_id, claimer="builder")
        assert repair is not None and repair.current_run_id is not None
        candidate_sha = "a" * 40
        assert kb.complete_task(
            conn,
            repair.id,
            summary="repair built candidate",
            metadata={
                "durable_goal": {
                    "protocol_version": DURABLE_GOAL_PROTOCOL_VERSION,
                    "stage": "BUILD",
                    "run_id": repair.current_run_id,
                    "candidate_sha": candidate_sha,
                    "outcome": "BUILT",
                }
            },
            expected_run_id=repair.current_run_id,
        )
        verify_result = supervise_goal_once(
            conn,
            goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        verify = kb.claim_task(conn, verify_result.task_id, claimer="verifier")
        assert verify is not None and verify.current_run_id is not None
        assert kb.complete_task(
            conn,
            verify.id,
            summary="verified",
            metadata={
                "durable_goal": {
                    "protocol_version": DURABLE_GOAL_PROTOCOL_VERSION,
                    "stage": "VERIFY",
                    "run_id": verify.current_run_id,
                    "candidate_sha": candidate_sha,
                    "verdict": "PASS",
                }
            },
            expected_run_id=verify.current_run_id,
        )
        review_result = supervise_goal_once(
            conn,
            goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        review = kb.claim_review_task(
            conn,
            review_result.task_id,
            claimer="reviewer",
            allow_durable=True,
        )
        assert review is not None and review.current_run_id is not None
        assert kb.complete_task(
            conn,
            review.id,
            summary="approved",
            metadata={
                "_kanban_dispatch_role": "reviewer",
                "durable_goal": {
                    "protocol_version": DURABLE_GOAL_PROTOCOL_VERSION,
                    "stage": "REVIEW",
                    "run_id": review.current_run_id,
                    "candidate_sha": candidate_sha,
                    "verdict": "APPROVE",
                },
            },
            expected_run_id=review.current_run_id,
        )

    # Crash/restart after REVIEW completion but before READY transition/delivery.
    first_gateway = _run_real_gateway_tick(home)
    second_gateway = _run_real_gateway_tick(home)
    assert first_gateway["runtime_id"].startswith(f"{first_gateway['pid']}:")
    assert second_gateway["runtime_id"].startswith(f"{second_gateway['pid']}:")
    assert second_gateway["runtime_id"] != first_gateway["runtime_id"]
    with kb.connect() as conn:
        goal = get_durable_goal(conn, goal_id)
        bindings = list_durable_goal_tasks(conn, goal_id)
        notifications = list_goal_notifications(conn, goal_id)

    assert goal is not None and goal.status == "READY_FOR_OWNER"
    assert [binding.stage for binding in bindings] == [
        "BUILD",
        "REPAIR_BUILD_1",
        "VERIFY",
        "REVIEW",
    ]
    assert len({binding.task_id for binding in bindings}) == 4
    assert [notification.kind for notification in notifications] == [
        "READY_FOR_OWNER"
    ]

    adapter = RecordingAdapter()
    first_notifier = _make_runner(adapter)
    first_notifier._active_profile_name = lambda: "gateway-owner"
    await _run_one_notifier_tick(monkeypatch, first_notifier)
    replayed_notifier = _make_runner(adapter)
    replayed_notifier._active_profile_name = lambda: "gateway-owner"
    await _run_one_notifier_tick(monkeypatch, replayed_notifier)
    assert len(adapter.sent) == 1
    assert goal_id in adapter.sent[0]["text"]
    with kb.connect() as conn:
        [notification] = kb.list_durable_goal_notification_rows(conn, goal_id)
        assert notification["delivered_at"] is not None
        assert notification["delivery_attempts"] == 1
        assert len(list_durable_goal_tasks(conn, goal_id)) == 4
