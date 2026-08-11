"""Gateway-only opt-in surface for deterministic durable Kanban goals."""

from __future__ import annotations

import logging
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
from hermes_cli.kanban_trusted_stages import (
    PromotionEvidence,
    PromotionRequest,
    ResultClassification,
    StaticTaskContractResolver,
    TaskContract,
    TrustedStageRegistry,
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
async def test_goal_durable_contract_resolves_static_v2_contract_without_adapter_execution(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
kanban:
  durable_goals:
    workflow_version: 2
    start_mode: trusted_contract
    orchestrator_profile: orchestrator
    builder_profile: builder
    reviewer_profile: reviewer
    task_contract_resolver: fixture-resolver
    verify_promote_adapter: fixture-adapter
    reviewer_skill_digest: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
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
            runtime_id="compatible-v2-singleton",
            schema_version=DURABLE_GOAL_SCHEMA_VERSION,
            protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
            lease_seconds=300,
        )
    contract = TaskContract.create(
        reference="issue:gateway-fixture",
        objective="Create a trusted PLAN task",
        base_revision="b" * 40,
        scope=("hermes_cli/", "gateway/"),
        gates=("focused-tests",),
    )
    resolver = StaticTaskContractResolver(
        resolver_id="fixture-resolver",
        contracts={contract.reference: contract},
    )

    class FakeAdapter:
        adapter_id = "fixture-adapter"

        def __init__(self):
            self.calls = 0

        def classify(self, _request):
            self.calls += 1
            raise AssertionError("PR 223-A must not execute adapters")

    adapter = FakeAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _SessionStore()
    runner.adapters = {}
    runner._queued_events = {}
    runner._active_profile_name = lambda: "gateway-owner"
    runner._trusted_stage_registry = TrustedStageRegistry(
        resolvers={resolver.resolver_id: resolver},
        adapters={adapter.adapter_id: adapter},
    )

    response = await GatewayRunner._handle_goal_command(
        runner,
        MessageEvent(
            text="/goal durable contract issue:gateway-fixture",
            message_type=MessageType.COMMAND,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="owner-chat",
                chat_type="dm",
                user_id="owner-7",
                profile="gateway-owner",
            ),
            message_id="trusted-contract-origin",
        ),
    )

    assert "Durable goal created" in response
    goal_id = response.split()[3]
    with kb.connect() as conn:
        goal = get_durable_goal(conn, goal_id)
        rows = conn.execute(
            "SELECT stage FROM kanban_goal_tasks WHERE goal_id = ?", (goal_id,)
        ).fetchall()
    assert goal is not None and goal.task_contract == contract
    assert goal.current_stage == "PLAN"
    assert [row["stage"] for row in rows] == ["PLAN"]
    assert adapter.calls == 0


@pytest.mark.asyncio
async def test_goal_durable_v2_free_form_fails_closed_with_zero_rows(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
kanban:
  durable_goals:
    workflow_version: 2
    start_mode: trusted_contract
    orchestrator_profile: orchestrator
    builder_profile: builder
    reviewer_profile: reviewer
    task_contract_resolver: fixture-resolver
    verify_promote_adapter: fixture-adapter
    reviewer_skill_digest: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
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
            text="/goal durable this free-form objective is not a contract",
            message_type=MessageType.COMMAND,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="owner-chat",
                chat_type="dm",
                profile="gateway-owner",
            ),
            message_id="free-form-must-not-persist",
        ),
    )

    with kb.connect() as conn:
        table_names = {
            str(row["name"])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "kanban_goals",
                "tasks",
                "kanban_goal_tasks",
                "kanban_goal_notification_outbox",
            )
        }
        counts["kanban_goal_operations"] = (
            conn.execute("SELECT COUNT(*) FROM kanban_goal_operations").fetchone()[0]
            if "kanban_goal_operations" in table_names
            else 0
        )
    assert response == "Usage: /goal durable contract <opaque-ref>"
    assert counts == {
        "kanban_goals": 0,
        "tasks": 0,
        "kanban_goal_tasks": 0,
        "kanban_goal_notification_outbox": 0,
        "kanban_goal_operations": 0,
    }
    assert runner._queued_events == {}


@pytest.mark.asyncio
async def test_goal_durable_v1_start_is_retired_without_creating_rows(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
kanban:
  durable_goals:
    workflow_version: 1
    builder_profile: legacy-builder
    verifier_profile: legacy-verifier
    reviewer_profile: legacy-reviewer
    reviewer_skill_digest: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
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
            text="/goal durable legacy active start",
            message_type=MessageType.COMMAND,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="owner-chat",
                chat_type="dm",
                profile="gateway-owner",
            ),
            message_id="retired-v1-start",
        ),
    )

    with kb.connect() as conn:
        counts = tuple(
            conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("kanban_goals", "kanban_goal_tasks", "tasks")
        )
    assert "Workflow V1 is retired" in response
    assert counts == (0, 0, 0)


class _FakeVerifyPromoteAdapter:
    adapter_id = "fixture-adapter"

    def classify(self, request: PromotionRequest) -> PromotionEvidence:
        return PromotionEvidence.create(
            adapter_id=self.adapter_id,
            request=request,
            classification=ResultClassification.PASS,
            summary="fixture pass",
        )


def _trusted_registry(reference: str = "fixture-contract") -> TrustedStageRegistry:
    contract = TaskContract.create(
        reference=reference,
        objective="Ship the candidate without merging",
        base_revision="a" * 40,
        scope=("hermes_cli/", "tests/"),
        gates=("focused-tests",),
    )
    resolver = StaticTaskContractResolver(
        resolver_id="fixture-resolver",
        contracts={contract.reference: contract},
    )
    adapter = _FakeVerifyPromoteAdapter()
    return TrustedStageRegistry(
        resolvers={resolver.resolver_id: resolver},
        adapters={adapter.adapter_id: adapter},
    )


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
    workflow_version: 2
    start_mode: trusted_contract
    task_contract_resolver: fixture-resolver
    verify_promote_adapter: fixture-adapter
    orchestrator_profile: orchestrator
    builder_profile: builder
    reviewer_profile: reviewer
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
    runner._trusted_stage_registry = _trusted_registry()
    runner._enqueue_fifo = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("durable goals must not enqueue a synthetic user turn")
    )

    event = MessageEvent(
        text="/goal durable contract fixture-contract",
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
    assert goal.workflow_version == 2
    assert goal.current_stage == "PLAN"
    assert goal.origin.message_id == "origin-message-1"
    assert goal.reviewer_skill_digest == PINNED_REVIEWER_DIGEST
    assert runner._queued_events == {}


@pytest.mark.asyncio
async def test_goal_durable_free_form_fails_closed_in_trusted_v2_mode(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
kanban:
  durable_goals:
    workflow_version: 2
    start_mode: trusted_contract
    task_contract_resolver: fixture-resolver
    verify_promote_adapter: fixture-adapter
    orchestrator_profile: orchestrator
    builder_profile: builder
    reviewer_profile: reviewer
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
    runner._trusted_stage_registry = _trusted_registry()

    response = await GatewayRunner._handle_goal_command(
        runner,
        MessageEvent(
            text="/goal durable Ship free form",
            message_type=MessageType.COMMAND,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="owner-chat",
                chat_type="dm",
                profile="gateway-owner",
            ),
            message_id="origin-free-form",
        ),
    )

    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM kanban_goals").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM kanban_goal_notification_outbox"
            ).fetchone()[0]
            == 0
        )
    assert response == "Usage: /goal durable contract <opaque-ref>"


@pytest.mark.asyncio
async def test_goal_durable_create_internal_error_is_logged_without_chat_leak(
    tmp_path, monkeypatch, caplog
):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
kanban:
  durable_goals:
    workflow_version: 2
    start_mode: trusted_contract
    task_contract_resolver: fixture-resolver
    verify_promote_adapter: fixture-adapter
    orchestrator_profile: orchestrator
    builder_profile: builder
    reviewer_profile: reviewer
    reviewer_skill_digest: 1e74b219dbf5377886fde11fcc873c673d3c01178a007ed9667a0942f8f10ec1
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kb.connect_closing() as conn:
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

    sensitive_error = (
        "permission denied reading /private/secret/.hermes/config.yaml "
        "with token=fake-sensitive-token"
    )
    from hermes_cli import kanban_goal_supervisor

    def _raise_sensitive_os_error(*_args, **_kwargs):
        raise OSError(sensitive_error)

    monkeypatch.setattr(
        kanban_goal_supervisor,
        "create_trusted_durable_goal",
        _raise_sensitive_os_error,
    )

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _SessionStore()
    runner.adapters = {}
    runner._queued_events = {}
    runner._active_profile_name = lambda: "gateway-owner"
    runner._trusted_stage_registry = _trusted_registry()
    event = MessageEvent(
        text="/goal durable contract fixture-contract",
        message_type=MessageType.COMMAND,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="owner-chat",
            chat_type="dm",
            user_id="owner-7",
            profile="gateway-owner",
        ),
        message_id="origin-internal-error",
    )

    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        response = await GatewayRunner._handle_goal_command(runner, event)

    assert response == (
        "Durable goal unavailable due to an internal error. No goal was created."
    )
    assert sensitive_error in caplog.text
    assert "/private/secret" not in response
    assert "fake-sensitive-token" not in response
    with kb.connect_closing() as conn:
        assert conn.execute("SELECT COUNT(*) FROM kanban_goals").fetchone()[0] == 0


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
    workflow_version: 2
    start_mode: trusted_contract
    task_contract_resolver: fixture-resolver
    verify_promote_adapter: fixture-adapter
    orchestrator_profile: orchestrator
    builder_profile: builder
    reviewer_profile: reviewer
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
    runner._trusted_stage_registry = _trusted_registry()

    response = await GatewayRunner._handle_goal_command(
        runner,
        MessageEvent(
            text="/goal durable contract fixture-contract",
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
async def test_goal_durable_v2_rejects_legacy_verifier_profile(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
kanban:
  durable_goals:
    workflow_version: 2
    start_mode: trusted_contract
    task_contract_resolver: fixture-resolver
    verify_promote_adapter: fixture-adapter
    orchestrator_profile: orchestrator
    builder_profile: builder
    verifier_profile: verifier
    reviewer_profile: reviewer
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
    runner._trusted_stage_registry = _trusted_registry()

    response = await GatewayRunner._handle_goal_command(
        runner,
        MessageEvent(
            text="/goal durable contract fixture-contract",
            message_type=MessageType.COMMAND,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="owner-chat",
                chat_type="dm",
                profile="gateway-owner",
            ),
            message_id="origin-legacy-verifier",
        ),
    )

    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM kanban_goals").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    assert "verifier_profile is not valid for Workflow V2" in response


@pytest.mark.asyncio
async def test_goal_durable_complete_is_origin_and_candidate_bound(
    tmp_path, monkeypatch
):
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
    workflow_version: 2
    start_mode: trusted_contract
    task_contract_resolver: fixture-resolver
    verify_promote_adapter: fixture-adapter
    orchestrator_profile: orchestrator
    builder_profile: builder
    reviewer_profile: reviewer
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
    runner._trusted_stage_registry = _trusted_registry()

    response = await GatewayRunner._handle_goal_command(
        runner,
        MessageEvent(
            text="/goal durable contract fixture-contract",
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
