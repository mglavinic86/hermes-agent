"""Gateway-only opt-in surface for deterministic durable Kanban goals."""

from __future__ import annotations

import os
import subprocess
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
    GoalOrigin,
    create_durable_goal,
    get_durable_goal,
)

PINNED_REVIEWER_DIGEST = (
    "1e74b219dbf5377886fde11fcc873c673d3c01178a007ed9667a0942f8f10ec1"
)


class _SessionEntry:
    session_id = "durable-goal-origin-session"


class _SessionStore:
    def get_or_create_session(self, source):
        return _SessionEntry()

    def _generate_session_key(self, source):
        return "agent:main:telegram:dm:owner-chat"


@pytest.mark.asyncio
async def test_goal_durable_creates_kanban_goal_without_agent_kickoff(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
kanban:
  durable_goals:
    builder_profile: turpi_builder
    verifier_profile: turpi_verify
    reviewer_profile: turpi_review
    reviewer_skill_digest: 1e74b219dbf5377886fde11fcc873c673d3c01178a007ed9667a0942f8f10ec1
    repair_budget: 1
    review_retry_budget: 1
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kb.connect() as conn:
        kb.upsert_durable_goal_runtime(
            conn,
            runtime_id="compatible-singleton",
            schema_version=DURABLE_GOAL_SCHEMA_VERSION,
            protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
            lease_seconds=300,
        )
    repo = tmp_path / "candidate-repo"
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

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _SessionStore()
    runner.adapters = {}
    runner._queued_events = {}
    runner._active_profile_name = lambda: "gateway-owner"
    runner._enqueue_fifo = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("durable goals must not enqueue a synthetic user turn")
    )

    event = MessageEvent(
        text="/goal durable Ship the candidate without merging",
        message_type=MessageType.COMMAND,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="owner-chat",
            chat_type="dm",
            user_id="owner-7",
            thread_id="topic-42",
            profile="gateway-owner",
        ),
        message_id="origin-message-1",
    )

    response = await GatewayRunner._handle_goal_command(runner, event)

    assert "Durable goal created" in response
    goal_id = response.split()[3]
    with kb.connect() as conn:
        goal = get_durable_goal(conn, goal_id)
    assert goal is not None
    assert goal.objective == "Ship the candidate without merging"
    assert goal.origin.message_id == "origin-message-1"
    assert goal.reviewer_skill_digest == PINNED_REVIEWER_DIGEST
    assert runner._queued_events == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "digest_line",
    ["", "    reviewer_skill_digest: ABCD\n", "    reviewer_skill_digest: 1234\n"],
)
async def test_goal_durable_requires_pinned_reviewer_skill_digest(
    tmp_path, monkeypatch, digest_line
):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        f"""
kanban:
  durable_goals:
    builder_profile: turpi_builder
    verifier_profile: turpi_verify
    reviewer_profile: turpi_review
{digest_line}""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kb.connect() as conn:
        kb.upsert_durable_goal_runtime(
            conn,
            runtime_id="compatible-singleton",
            schema_version=DURABLE_GOAL_SCHEMA_VERSION,
            protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
            lease_seconds=300,
        )
    repo = tmp_path / "candidate-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    kb.write_board_metadata("default", default_workdir=str(repo))

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _SessionStore()
    runner.adapters = {}
    runner._queued_events = {}
    runner._active_profile_name = lambda: "gateway-owner"

    response = await GatewayRunner._handle_goal_command(
        runner,
        MessageEvent(
            text="/goal durable Missing pinned reviewer digest",
            message_type=MessageType.COMMAND,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="owner-chat",
                chat_type="dm",
                profile="gateway-owner",
            ),
            message_id="origin-missing-digest",
        ),
    )

    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM kanban_goals").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    assert "reviewer_skill_digest" in response


@pytest.mark.asyncio
async def test_goal_durable_requires_explicit_board_workdir(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
kanban:
  durable_goals:
    builder_profile: turpi_builder
    verifier_profile: turpi_verify
    reviewer_profile: turpi_review
    reviewer_skill_digest: 1e74b219dbf5377886fde11fcc873c673d3c01178a007ed9667a0942f8f10ec1
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kb.connect() as conn:
        kb.upsert_durable_goal_runtime(
            conn,
            runtime_id="compatible-singleton",
            schema_version=DURABLE_GOAL_SCHEMA_VERSION,
            protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
            lease_seconds=300,
        )
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _SessionStore()
    runner.adapters = {}
    runner._queued_events = {}
    runner._active_profile_name = lambda: "gateway-owner"

    response = await GatewayRunner._handle_goal_command(
        runner,
        MessageEvent(
            text="/goal durable Must have a known checkout",
            message_type=MessageType.COMMAND,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="owner-chat",
                chat_type="dm",
                profile="gateway-owner",
            ),
            message_id="origin-no-workdir",
        ),
    )

    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM kanban_goals").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    assert "board default_workdir" in response


@pytest.mark.asyncio
async def test_goal_durable_complete_is_origin_and_candidate_bound(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
kanban:
  durable_goals:
    board: default
    builder_profile: builder
    verifier_profile: verifier
    reviewer_profile: reviewer
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    candidate_sha = "7" * 40
    with kb.connect() as conn:
        created = create_durable_goal(
            conn,
            objective="Owner completes without merge authority",
            origin=GoalOrigin(platform="telegram", chat_id="owner-chat"),
            board="default",
            builder_profile="builder",
            verifier_profile="verifier",
            reviewer_profile="reviewer",
            repair_budget=1,
            review_retry_budget=1,
        )
        conn.execute(
            "UPDATE kanban_goals SET status = 'READY_FOR_OWNER', "
            "current_stage = 'READY_FOR_OWNER', candidate_sha = ? WHERE id = ?",
            (candidate_sha, created.goal_id),
        )
        conn.commit()

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _SessionStore()
    runner.adapters = {}
    runner._queued_events = {}
    runner._active_profile_name = lambda: "gateway-owner"
    response = await GatewayRunner._handle_goal_command(
        runner,
        MessageEvent(
            text=f"/goal durable complete {created.goal_id} {candidate_sha}",
            message_type=MessageType.COMMAND,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="owner-chat",
                chat_type="dm",
                profile="gateway-owner",
            ),
            message_id="owner-complete-1",
        ),
    )

    with kb.connect() as conn:
        goal = get_durable_goal(conn, created.goal_id)
    assert "marked COMPLETED" in response
    assert goal is not None and goal.status == "COMPLETED"
    assert runner._queued_events == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_state", ["missing", "protocol_skew"])
async def test_goal_durable_refuses_incompatible_singleton_without_creating_task(
    tmp_path,
    monkeypatch,
    runtime_state,
):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
kanban:
  durable_goals:
    builder_profile: turpi_builder
    verifier_profile: turpi_verify
    reviewer_profile: turpi_review
    reviewer_skill_digest: 1e74b219dbf5377886fde11fcc873c673d3c01178a007ed9667a0942f8f10ec1
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    if runtime_state == "protocol_skew":
        with kb.connect() as conn:
            kb.upsert_durable_goal_runtime(
                conn,
                runtime_id="older-singleton",
                schema_version=DURABLE_GOAL_SCHEMA_VERSION,
                protocol_version=DURABLE_GOAL_PROTOCOL_VERSION + 1,
                lease_seconds=300,
            )

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _SessionStore()
    runner.adapters = {}
    runner._queued_events = {}
    runner._active_profile_name = lambda: "gateway-owner"

    response = await GatewayRunner._handle_goal_command(
        runner,
        MessageEvent(
            text="/goal durable Must not dispatch under skew",
            message_type=MessageType.COMMAND,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="owner-chat",
                chat_type="dm",
                user_id="owner-7",
                profile="gateway-owner",
            ),
            message_id=f"origin-{runtime_state}",
        ),
    )

    with kb.connect() as conn:
        goal_count = conn.execute("SELECT COUNT(*) FROM kanban_goals").fetchone()[0]
        task_count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    assert "incompatible singleton dispatcher" in response
    assert goal_count == 0
    assert task_count == 0
    assert runner._queued_events == {}
