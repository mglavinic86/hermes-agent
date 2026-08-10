"""Security contracts for dispatcher-assigned Kanban roles.

Explicit roles are trusted dispatcher context, not model-provided labels. Legacy
workers without HERMES_KANBAN_ROLE retain the pre-hardening tool surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from tools import kanban_tools as kt


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_role_resolution_defaults_to_worker_and_honors_profile_mapping():
    cfg = {
        "kanban": {
            "orchestrator_profile": "legacy-lead",
            "role_profiles": {
                "orchestrator": ["planner"],
                "reviewer": ["auditor"],
            },
        }
    }

    assert kb.resolve_dispatched_role("builder", config=cfg) == "worker"
    assert kb.resolve_dispatched_role("planner", config=cfg) == "orchestrator"
    assert kb.resolve_dispatched_role("legacy-lead", config=cfg) == "orchestrator"
    assert kb.resolve_dispatched_role("auditor", config=cfg) == "reviewer"
    assert kb.resolve_dispatched_role("builder", config=cfg, forced_role="reviewer") == "reviewer"
    assert kb.resolve_dispatched_role("builder", config={"kanban": {}}) is None


def test_overlapping_role_mapping_fails_closed_to_worker():
    cfg = {
        "kanban": {
            "role_profiles": {
                "orchestrator": ["ambiguous"],
                "reviewer": ["ambiguous"],
            }
        }
    }

    assert kb.resolve_dispatched_role("ambiguous", config=cfg) == "worker"


def test_worker_cannot_route_board_but_keeps_own_lifecycle(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    monkeypatch.setenv("HERMES_KANBAN_ROLE", "worker")

    assert kt._check_kanban_mode() is True
    assert kt._check_kanban_routing_mode() is False
    assert kt._require_kanban_capability("kanban_complete") is None
    denied = kt._require_kanban_capability("kanban_create")
    assert denied is not None and "orchestrator" in denied.lower()


def test_reviewer_is_read_only_except_verdict_lifecycle(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_review")
    monkeypatch.setenv("HERMES_KANBAN_ROLE", "reviewer")

    for allowed in (
        "kanban_show",
        "kanban_complete",
        "kanban_block",
        "kanban_heartbeat",
        "kanban_comment",
        "kanban_attachments",
    ):
        assert kt._require_kanban_capability(allowed) is None

    for denied_tool in (
        "kanban_list",
        "kanban_create",
        "kanban_link",
        "kanban_unblock",
        "kanban_attach",
        "kanban_attach_url",
    ):
        denied = kt._require_kanban_capability(denied_tool)
        assert denied is not None, denied_tool


def test_orchestrator_has_routing_capabilities(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_plan")
    monkeypatch.setenv("HERMES_KANBAN_ROLE", "orchestrator")

    assert kt._check_kanban_routing_mode() is True
    for tool_name in kt.KANBAN_ROLE_CAPABILITIES["orchestrator"]:
        assert kt._require_kanban_capability(tool_name) is None


def test_invalid_explicit_role_denies_every_kanban_tool(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_invalid")
    monkeypatch.setenv("HERMES_KANBAN_ROLE", "admin")

    assert kt._check_kanban_mode() is False
    assert kt._require_kanban_capability("kanban_show") is not None


def test_legacy_worker_without_role_keeps_old_routing_surface(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_legacy")
    monkeypatch.delenv("HERMES_KANBAN_ROLE", raising=False)

    assert kt._check_kanban_mode() is True
    assert kt._check_kanban_routing_mode() is True
    assert kt._require_kanban_capability("kanban_create") is None


def test_default_spawn_stamps_configured_role_into_trusted_env(
    kanban_home, monkeypatch
):
    captured = {}

    class _FakeProc:
        pid = 7331

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"role_profiles": {"reviewer": ["auditor"]}}},
    )
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda _cmd, **kwargs: captured.setdefault("env", kwargs["env"])
        and _FakeProc(),
    )

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="review", assignee="auditor")
        task = kb.get_task(conn, tid)
        assert task is not None

    kb._default_spawn(task, str(kanban_home))
    assert captured["env"]["HERMES_KANBAN_ROLE"] == "reviewer"


def test_default_spawn_preserves_legacy_surface_without_role_mapping(
    kanban_home, monkeypatch
):
    captured = {}

    class _FakeProc:
        pid = 7332

    monkeypatch.setenv("HERMES_KANBAN_ROLE", "orchestrator")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"kanban": {}})

    def _fake_popen(_cmd, **kwargs):
        captured["env"] = kwargs["env"]
        return _FakeProc()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="legacy", assignee="builder")
        task = kb.get_task(conn, tid)
        assert task is not None

    kb._default_spawn(task, str(kanban_home))
    assert "HERMES_KANBAN_ROLE" not in captured["env"]
