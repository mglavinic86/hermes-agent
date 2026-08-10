"""Tests for kanban lifecycle plugin hooks.

Verifies that claim/complete/block transitions fire the
kanban_task_claimed / kanban_task_completed / kanban_task_blocked plugin
hooks AFTER the board DB change is committed, with the documented kwargs,
and that a misbehaving hook callback never breaks the transition.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def captured_hooks(monkeypatch):
    """Register capturing callbacks for the three kanban lifecycle hooks.

    Patches the plugin manager's _hooks dict directly (the same registry
    invoke_hook reads) and restores it afterward.
    """
    mgr = get_plugin_manager()
    events: list[tuple[str, dict]] = []
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    for hook in ("kanban_task_claimed", "kanban_task_completed", "kanban_task_blocked"):
        mgr._hooks.setdefault(hook, []).append(
            lambda _h=hook, **kw: events.append((_h, kw))
        )
    try:
        yield events
    finally:
        mgr._hooks = saved


def test_hooks_are_registered_as_valid():
    """The three lifecycle hook names are part of VALID_HOOKS."""
    assert "kanban_task_claimed" in VALID_HOOKS
    assert "kanban_task_completed" in VALID_HOOKS
    assert "kanban_task_blocked" in VALID_HOOKS


def test_claim_fires_hook(kanban_home, captured_hooks):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
    finally:
        conn.close()
    fired = [e for e in captured_hooks if e[0] == "kanban_task_claimed"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["assignee"] == "worker"
    assert "profile_name" in kw
    assert kw["run_id"] is not None


def test_complete_fires_hook_with_summary(kanban_home, captured_hooks):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker")
        kb.claim_task(conn, tid)
        assert kb.complete_task(conn, tid, summary="all done")
    finally:
        conn.close()
    fired = [e for e in captured_hooks if e[0] == "kanban_task_completed"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["summary"] == "all done"
    assert kw["assignee"] == "worker"


def test_block_fires_hook_with_reason(kanban_home, captured_hooks):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker")
        kb.claim_task(conn, tid)
        assert kb.block_task(conn, tid, reason="needs human")
    finally:
        conn.close()
    fired = [e for e in captured_hooks if e[0] == "kanban_task_blocked"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["reason"] == "needs human"


def test_dependency_block_commit_failure_fires_no_hook(
    kanban_home, captured_hooks, monkeypatch
):
    """A dependency hook must never announce a transaction that rolled back."""
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="dependency wait", assignee="worker")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        before_events = [
            (event.kind, event.payload) for event in kb.list_events(conn, task_id)
        ]
        real_boundary = kb._execute_boundary_with_retry

        def fail_commit(connection, sql):
            if sql.strip().upper() == "COMMIT":
                raise sqlite3.OperationalError("injected dependency commit failure")
            return real_boundary(connection, sql)

        monkeypatch.setattr(kb, "_execute_boundary_with_retry", fail_commit)
        with pytest.raises(sqlite3.OperationalError, match="dependency commit failure"):
            kb.block_task(
                conn,
                task_id,
                reason="waiting on parent",
                kind="dependency",
                comment_author="operator",
                comment_body="BLOCKED: waiting on parent",
            )

        assert kb.get_task(conn, task_id).status == "running"
        assert kb.list_comments(conn, task_id) == []
        assert [
            (event.kind, event.payload) for event in kb.list_events(conn, task_id)
        ] == before_events

    assert [e for e in captured_hooks if e[0] == "kanban_task_blocked"] == []


def test_dependency_block_fires_one_hook_after_atomic_reason_comment(
    kanban_home, captured_hooks
):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="dependency wait", assignee="worker")
        assert kb.claim_task(conn, task_id) is not None
        assert kb.block_task(
            conn,
            task_id,
            reason="waiting on parent",
            kind="dependency",
            comment_author="operator",
            comment_body="BLOCKED: waiting on parent",
        )
        assert kb.get_task(conn, task_id).status == "todo"
        comments = kb.list_comments(conn, task_id)

    assert [(comment.author, comment.body) for comment in comments] == [
        ("operator", "BLOCKED: waiting on parent")
    ]
    fired = [e for e in captured_hooks if e[0] == "kanban_task_blocked"]
    assert len(fired) == 1
    assert fired[0][1]["task_id"] == task_id
    assert fired[0][1]["reason"] == "waiting on parent"


def test_no_hook_on_failed_transition(kanban_home, captured_hooks):
    """complete_task on an unclaimed/nonexistent task fires no hook."""
    conn = kb.connect()
    try:
        # Completing a task that doesn't exist returns False without firing.
        assert kb.complete_task(conn, "t_doesnotexist", summary="x") is False
    finally:
        conn.close()
    assert [e for e in captured_hooks if e[0] == "kanban_task_completed"] == []


def test_misbehaving_hook_does_not_break_transition(kanban_home, monkeypatch):
    """A hook callback that raises must not break the board transition."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    def _boom(**kw):
        raise RuntimeError("plugin exploded")

    mgr._hooks.setdefault("kanban_task_completed", []).append(_boom)
    try:
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="worker")
            kb.claim_task(conn, tid)
            # Despite the raising hook, completion succeeds and persists.
            assert kb.complete_task(conn, tid, summary="ok") is True
            assert kb.get_task(conn, tid).status == "done"
        finally:
            conn.close()
    finally:
        mgr._hooks = saved
