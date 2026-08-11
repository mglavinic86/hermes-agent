"""Gateway singleton preparation for durable goal boards."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gateway.kanban_watchers import _prepare_durable_goal_board_tick
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_goal_supervisor import (
    DURABLE_GOAL_PROTOCOL_VERSION,
    GoalOrigin,
    create_trusted_durable_goal,
    list_durable_goal_tasks,
)
from hermes_cli.kanban_trusted_stages import TaskContract


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
    contract = TaskContract.create(
        reference="watcher-contract",
        objective="Advance before the dispatcher scans successors",
        base_revision="4" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
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
            reviewer_skill_digest=None,
            repair_budget=1,
            review_retry_budget=1,
        )
        plan = kb.claim_task(conn, created.task_id, claimer="orchestrator")
        assert plan is not None and plan.current_run_id is not None
        assert kb.complete_task(
            conn,
            plan.id,
            summary="planned",
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
    assert [binding.stage for binding in bindings] == ["PLAN", "BUILD_CANDIDATE"]


def _gateway_process_env(kanban_home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["HERMES_HOME"] = str(kanban_home)
    env.pop("HERMES_KANBAN_DB", None)
    env.pop("HERMES_KANBAN_BOARD", None)
    env.pop("HERMES_KANBAN_DISPATCH_IN_GATEWAY", None)
    repo_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (repo_root, env.get("PYTHONPATH", "")) if part
    )
    return env


def _run_gateway_harness(
    kanban_home: Path, mode: str, *, timeout: float = 15
) -> tuple[subprocess.CompletedProcess[str], dict]:
    helper = Path(__file__).with_name("durable_goal_gateway_process.py")
    completed = subprocess.run(
        [sys.executable, str(helper), mode],
        env=_gateway_process_env(kanban_home),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1]) if lines else {}
    return completed, payload


@pytest.mark.live_system_guard_bypass
def test_real_gateway_process_restart_reacquires_lock_lease_and_dedupes_successor(
    kanban_home,
):
    (kanban_home / "config.yaml").write_text(
        "kanban:\n"
        "  dispatch_in_gateway: true\n"
        "  dispatch_interval_seconds: 1\n"
        "  auto_decompose: false\n"
        "  max_in_progress: 1\n"
    )
    contract = TaskContract.create(
        reference="watcher-restart-contract",
        objective="Resume a terminal plan after a real gateway crash",
        base_revision="7" * 40,
        scope=("hermes_cli/",),
        gates=("focused-tests",),
    )
    with kb.connect() as conn:
        # Keep the actual dispatcher from spawning a profile worker; the test
        # exercises lock/lease/supervisor restart semantics, not child exec.
        guard_id = kb.create_task(conn, title="running guard", assignee="guard")
        assert kb.claim_task(conn, guard_id, claimer="guard") is not None
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
            reviewer_skill_digest=None,
            repair_budget=1,
            review_retry_budget=1,
        )
        plan = kb.claim_task(conn, created.task_id, claimer="orchestrator")
        assert plan is not None and plan.current_run_id is not None
        assert kb.complete_task(
            conn,
            plan.id,
            summary="planned before gateway crash",
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

    helper = Path(__file__).with_name("durable_goal_gateway_process.py")
    old = subprocess.Popen(
        [sys.executable, str(helper), "old-hold"],
        env=_gateway_process_env(kanban_home),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert old.stdout is not None
        held = json.loads(old.stdout.readline())
        assert held["lock_state"] == "held"

        contended, contended_payload = _run_gateway_harness(
            kanban_home, "watch-once", timeout=5
        )
        assert contended.returncode == 0, contended.stderr
        assert contended_payload["runtime_id"] is None
        with kb.connect() as conn:
            assert kb.get_durable_goal_runtime_row(conn) is None
            assert [
                binding.stage
                for binding in list_durable_goal_tasks(conn, created.goal_id)
            ] == ["PLAN"]
    finally:
        if old.poll() is None:
            old.kill()
            old.wait(timeout=5)

    restarted, restarted_payload = _run_gateway_harness(kanban_home, "watch-once")
    assert restarted.returncode == 0, restarted.stderr
    assert restarted_payload["runtime_id"].startswith(f"{restarted_payload['pid']}:")
    with kb.connect() as conn:
        stages = [
            binding.stage for binding in list_durable_goal_tasks(conn, created.goal_id)
        ]
        runtime = kb.get_durable_goal_runtime_row(conn)
        assert runtime is not None
        first_runtime_id = runtime["runtime_id"]
    assert stages == ["PLAN", "BUILD_CANDIDATE"]

    replayed, replayed_payload = _run_gateway_harness(kanban_home, "watch-once")
    assert replayed.returncode == 0, replayed.stderr
    assert replayed_payload["runtime_id"].startswith(f"{replayed_payload['pid']}:")
    assert replayed_payload["runtime_id"] != first_runtime_id
    with kb.connect() as conn:
        stages = [
            binding.stage for binding in list_durable_goal_tasks(conn, created.goal_id)
        ]
    assert stages == ["PLAN", "BUILD_CANDIDATE"]
