"""Durable singleton-runtime compatibility contract for supervised goals."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_goal_supervisor import (
    DURABLE_GOAL_PROTOCOL_VERSION,
    DURABLE_GOAL_SCHEMA_VERSION,
    GoalOrigin,
    check_durable_goal_runtime,
    create_durable_goal,
    get_durable_goal,
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


def test_runtime_lease_requires_exact_fresh_schema_and_protocol(kanban_home):
    with kb.connect() as conn:
        kb.upsert_durable_goal_runtime(
            conn,
            runtime_id="singleton-runtime-1",
            schema_version=DURABLE_GOAL_SCHEMA_VERSION,
            protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
            lease_seconds=30,
            now=100,
        )

        compatible = check_durable_goal_runtime(conn, now=110)
        expired = check_durable_goal_runtime(conn, now=131)
        conn.execute(
            "UPDATE kanban_goal_runtime SET schema_version = schema_version + 1"
        )
        wrong_schema = check_durable_goal_runtime(conn, now=110)
        conn.execute(
            "UPDATE kanban_goal_runtime SET schema_version = ?, protocol_version = ?",
            (DURABLE_GOAL_SCHEMA_VERSION, DURABLE_GOAL_PROTOCOL_VERSION + 1),
        )
        wrong_protocol = check_durable_goal_runtime(conn, now=110)

    assert compatible.compatible is True
    assert compatible.runtime_id == "singleton-runtime-1"
    assert expired.compatible is False and expired.reason == "runtime_lease_expired"
    assert wrong_schema.compatible is False and wrong_schema.reason == "schema_mismatch"
    assert wrong_protocol.compatible is False
    assert wrong_protocol.reason == "protocol_mismatch"


def test_supervised_task_is_invisible_without_protocol_and_claimable_with_exact_runtime(
    kanban_home,
    monkeypatch,
):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    spawned: list[str] = []

    def _spawn(task, _workspace, **_kwargs):
        spawned.append(task.id)
        return 123

    with kb.connect() as conn:
        incompatible = create_durable_goal(
            conn,
            objective="Do not expose this to an unstamped dispatcher",
            origin=GoalOrigin(platform="telegram", chat_id="owner"),
            board="default",
            builder_profile="builder",
            verifier_profile="verifier",
            reviewer_profile="reviewer",
            repair_budget=1,
            review_retry_budget=1,
        )
        initial = kb.get_task(conn, incompatible.task_id)
        unstamped = kb.dispatch_once(conn, board="default", spawn_fn=_spawn)
        blocked_goal = get_durable_goal(conn, incompatible.goal_id)

        compatible = create_durable_goal(
            conn,
            objective="Dispatch only with the exact runtime protocol",
            origin=GoalOrigin(platform="telegram", chat_id="owner-2"),
            board="default",
            builder_profile="builder",
            verifier_profile="verifier",
            reviewer_profile="reviewer",
            repair_budget=1,
            review_retry_budget=1,
        )
        exact = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
            durable_goal_skill_available=lambda _profile, _skill: True,
        )

    assert initial is not None and initial.status == "durable_ready"
    assert unstamped.spawned == []
    assert unstamped.auto_blocked == [incompatible.task_id]
    assert blocked_goal is not None and blocked_goal.status == "BLOCKED_CAPABILITY"
    assert exact.spawned[0][0] == compatible.task_id
    assert spawned == [compatible.task_id]


def test_supervised_spawn_failure_requeues_to_private_status(
    kanban_home,
    monkeypatch,
):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    with kb.connect() as conn:
        created = create_durable_goal(
            conn,
            objective="Remain hidden after a transient spawn failure",
            origin=GoalOrigin(platform="telegram", chat_id="owner"),
            board="default",
            builder_profile="builder",
            verifier_profile="verifier",
            reviewer_profile="reviewer",
            repair_budget=1,
            review_retry_budget=1,
        )

        result = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("transient spawn failure")
            ),
            failure_limit=2,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
            durable_goal_skill_available=lambda _profile, _skill: True,
        )
        task = kb.get_task(conn, created.task_id)

    assert result.spawned == []
    assert result.auto_blocked == []
    assert task is not None and task.status == "durable_ready"


def test_foreign_board_preflight_blocks_before_worker_run(
    kanban_home,
    monkeypatch,
):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    spawned: list[str] = []
    with kb.connect() as conn:
        created = create_durable_goal(
            conn,
            objective="Never cross the board authority boundary",
            origin=GoalOrigin(platform="telegram", chat_id="owner"),
            board="expected-board",
            builder_profile="builder",
            verifier_profile="verifier",
            reviewer_profile="reviewer",
            repair_budget=1,
            review_retry_budget=1,
        )
        result = kb.dispatch_once(
            conn,
            board="foreign-board",
            spawn_fn=lambda task, *_args, **_kwargs: spawned.append(task.id),
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(conn, created.goal_id)
        runs = kb.list_runs(conn, created.task_id)

    assert result.spawned == []
    assert result.auto_blocked == [created.task_id]
    assert spawned == []
    assert runs == []
    assert goal is not None and goal.status == "BLOCKED_CAPABILITY"
    assert goal.blocked_reason == (
        "foreign board: expected expected-board, got foreign-board"
    )
