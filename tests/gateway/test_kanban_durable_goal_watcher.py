"""Gateway singleton preparation for durable goal boards."""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.kanban_watchers import _prepare_durable_goal_board_tick
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_goal_supervisor import (
    DURABLE_GOAL_PROTOCOL_VERSION,
    GoalOrigin,
    create_durable_goal,
    list_durable_goal_tasks,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_singleton_board_tick_stamps_runtime_and_supervises_before_dispatch(
    kanban_home,
):
    candidate_sha = "4" * 40
    with kb.connect() as conn:
        created = create_durable_goal(
            conn,
            objective="Advance before the dispatcher scans successors",
            origin=GoalOrigin(platform="telegram", chat_id="owner"),
            board="default",
            builder_profile="builder",
            verifier_profile="verifier",
            reviewer_profile="reviewer",
            repair_budget=1,
            review_retry_budget=1,
        )
        build = kb.claim_task(conn, created.task_id, claimer="builder")
        assert build is not None and build.current_run_id is not None
        assert kb.complete_task(
            conn,
            build.id,
            summary="built",
            metadata={
                "durable_goal": {
                    "protocol_version": DURABLE_GOAL_PROTOCOL_VERSION,
                    "stage": "BUILD",
                    "run_id": build.current_run_id,
                    "candidate_sha": candidate_sha,
                    "outcome": "BUILT",
                }
            },
            expected_run_id=build.current_run_id,
        )

        results = _prepare_durable_goal_board_tick(
            conn,
            board="default",
            runtime_id="singleton-without-profile-identity",
            lease_seconds=180,
        )
        runtime = kb.get_durable_goal_runtime_row(conn)
        bindings = list_durable_goal_tasks(conn, created.goal_id)

    assert [result.action for result in results] == ["CREATED_SUCCESSOR"]
    assert runtime is not None
    assert runtime["runtime_id"] == "singleton-without-profile-identity"
    assert runtime["protocol_version"] == DURABLE_GOAL_PROTOCOL_VERSION
    assert [binding.stage for binding in bindings] == ["BUILD", "VERIFY"]
