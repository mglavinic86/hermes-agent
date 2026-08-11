import asyncio
import os
import sqlite3
import subprocess
from pathlib import Path


from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_goal_supervisor import (
    DURABLE_GOAL_PROTOCOL_VERSION,
    GoalOrigin,
    create_durable_goal,
    get_durable_goal,
    list_durable_goal_tasks,
    list_goal_notifications,
    supervise_goal_once,
)


class RecordingAdapter:
    def __init__(self):
        self.sent = []
        self.handled = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})

    async def handle_message(self, event):
        self.handled.append(event)


class DisconnectedAdapters(dict):
    """Expose a platform during collection, then simulate disconnect on get()."""

    def get(self, key, default=None):
        return None


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _make_runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    return runner


def _create_completed_subscription(summary="done once"):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="notify once", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(conn, tid, summary=summary)
        return tid
    finally:
        conn.close()


def _insert_goal_outbox(conn, *, goal_id="g_notify_once", kind="READY_FOR_OWNER"):
    now = 1_700_000_000
    conn.execute(
        """
        INSERT INTO kanban_goals (
            id, objective, status, current_stage, schema_version,
            protocol_version, board_slug, origin_platform, origin_chat_id,
            origin_thread_id, builder_profile, verifier_profile,
            reviewer_profile, repair_budget, review_retry_budget,
            created_at, updated_at
        ) VALUES (?, 'notify owner', 'READY_FOR_OWNER', 'READY_FOR_OWNER',
                  1, 1, 'default', 'telegram', 'owner-chat', '',
                  'builder', 'verifier', 'reviewer', 1, 1, ?, ?)
        """,
        (goal_id, now, now),
    )
    conn.execute(
        """
        INSERT INTO kanban_goal_notification_outbox (
            goal_id, kind, dedupe_key, payload, platform, chat_id,
            thread_id, created_at
        ) VALUES (?, ?, ?, ?, 'telegram', 'owner-chat', '', ?)
        """,
        (
            goal_id,
            kind,
            f"durable-goal:{goal_id}:{kind}",
            '{"candidate_sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}',
            now,
        ),
    )
    conn.commit()
    return goal_id


def _init_durable_goal_board(tmp_path, monkeypatch, db_name="durable-goal.db"):
    db_path = tmp_path / db_name
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
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
    return db_path


def test_goal_outbox_delivers_once_and_acks_across_notifier_replay(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "goal-outbox-once.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        goal_id = _insert_goal_outbox(conn)
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    assert goal_id in adapter.sent[0]["text"]
    assert "READY_FOR_OWNER" in adapter.sent[0]["text"]
    conn = kb.connect()
    try:
        [row] = kb.list_durable_goal_notification_rows(conn, goal_id)
        assert row["delivered_at"] is not None
        assert row["delivery_attempts"] == 1
    finally:
        conn.close()


def test_goal_outbox_expired_claim_reclaims_delivers_and_acks_once(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "goal-outbox-expired-claim.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        goal_id = _insert_goal_outbox(conn, goal_id="g_expired_claim")
        [claimed] = kb.claim_pending_durable_goal_notifications(
            conn,
            now=100,
            claim_timeout_seconds=120,
        )
        assert claimed["claim_token"]
        assert kb.count_pending_durable_goal_notifications(
            db_path=db_path,
            now=119,
            claim_timeout_seconds=120,
        ) == 1
        assert kb.count_pending_durable_goal_notifications(
            db_path=db_path,
            now=221,
            claim_timeout_seconds=120,
        ) == 1
    finally:
        conn.close()

    monkeypatch.setattr(kb.time, "time", lambda: 221)
    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    assert goal_id in adapter.sent[0]["text"]
    conn = kb.connect()
    try:
        [row] = kb.list_durable_goal_notification_rows(conn, goal_id)
        goal = kb.get_durable_goal_row(conn, goal_id)
        assert row["delivered_at"] is not None
        assert row["claim_token"] is None
        assert row["delivery_attempts"] == 2
        assert goal["status"] == "READY_FOR_OWNER"
    finally:
        conn.close()


def test_goal_outbox_send_failure_retries_without_reverting_goal(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "goal-outbox-retry.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        goal_id = _insert_goal_outbox(conn, goal_id="g_retry_owner")
    finally:
        conn.close()

    clock = {"now": 10_000}
    monkeypatch.setattr(kb.time, "time", lambda: clock["now"])
    adapter = FailingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert adapter.attempts == 1
    conn = kb.connect()
    try:
        [row] = kb.list_durable_goal_notification_rows(conn, goal_id)
        goal = kb.get_durable_goal_row(conn, goal_id)
        assert row["delivered_at"] is None
        assert row["claim_token"] is None
        assert row["delivery_attempts"] == 1
        assert row["next_attempt_at"] == (
            clock["now"] + kb.DURABLE_GOAL_NOTIFICATION_RETRY_BASE_SECONDS
        )
        assert "simulated send failure" in row["last_error"]
        assert goal["status"] == "READY_FOR_OWNER"
    finally:
        conn.close()

    clock["now"] = int(row["next_attempt_at"])
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert adapter.attempts == 2


def test_goal_outbox_retries_beyond_thirteen_then_recovers_without_duplicate(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "goal-outbox-indefinite-retry.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    with kb.connect_closing() as conn:
        goal_id = _insert_goal_outbox(conn, goal_id="g_indefinite_retry")

    clock = {"now": 20_000}
    monkeypatch.setattr(kb.time, "time", lambda: clock["now"])
    unavailable_adapter = FailingAdapter()

    for expected_attempts in range(1, 14):
        asyncio.run(
            _run_one_notifier_tick(
                monkeypatch,
                _make_runner(unavailable_adapter),
            )
        )
        with kb.connect_closing() as conn:
            [row] = kb.list_durable_goal_notification_rows(conn, goal_id)
        assert row["delivered_at"] is None
        assert row["delivery_attempts"] == expected_attempts
        assert row["next_attempt_at"] > clock["now"]
        if expected_attempts < 13:
            clock["now"] = int(row["next_attempt_at"])

    assert unavailable_adapter.attempts == 13
    assert kb.count_pending_durable_goal_notifications(
        db_path=db_path,
        max_attempts=1,
        now=clock["now"],
    ) == 1
    with kb.connect_closing() as conn:
        assert kb.claim_pending_durable_goal_notifications(
            conn,
            max_attempts=1,
            now=int(row["next_attempt_at"]) - 1,
        ) == []

    clock["now"] = int(row["next_attempt_at"])
    recovered_adapter = RecordingAdapter()
    asyncio.run(
        _run_one_notifier_tick(monkeypatch, _make_runner(recovered_adapter))
    )
    asyncio.run(
        _run_one_notifier_tick(monkeypatch, _make_runner(recovered_adapter))
    )

    assert len(recovered_adapter.sent) == 1
    assert goal_id in recovered_adapter.sent[0]["text"]
    with kb.connect_closing() as conn:
        [row] = kb.list_durable_goal_notification_rows(conn, goal_id)
    assert row["delivered_at"] is not None
    assert row["delivery_attempts"] == 14
    assert row["next_attempt_at"] is None
    assert kb.count_pending_durable_goal_notifications(db_path=db_path) == 0


def test_goal_outbox_failure_backoff_delays_increases_and_caps(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "goal-outbox-backoff.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    with kb.connect_closing() as conn:
        goal_id = _insert_goal_outbox(conn, goal_id="g_backoff_owner")
        now = 10_000

        [claimed] = kb.claim_pending_durable_goal_notifications(conn, now=now)
        assert kb.release_durable_goal_notification_claim(
            conn,
            notification_id=int(claimed["id"]),
            claim_token=str(claimed["claim_token"]),
            error="first failure",
            now=now,
        )
        [row] = kb.list_durable_goal_notification_rows(conn, goal_id)
        assert row["next_attempt_at"] == (
            now + kb.DURABLE_GOAL_NOTIFICATION_RETRY_BASE_SECONDS
        )
        assert kb.count_pending_durable_goal_notifications(
            db_path=db_path,
            now=now,
        ) == 1
        assert kb.claim_pending_durable_goal_notifications(
            conn,
            now=int(row["next_attempt_at"]) - 1,
        ) == []

        now = int(row["next_attempt_at"])
        [claimed] = kb.claim_pending_durable_goal_notifications(conn, now=now)
        assert kb.release_durable_goal_notification_claim(
            conn,
            notification_id=int(claimed["id"]),
            claim_token=str(claimed["claim_token"]),
            error="second failure",
            now=now,
        )
        [row] = kb.list_durable_goal_notification_rows(conn, goal_id)
        assert row["next_attempt_at"] - now == (
            kb.DURABLE_GOAL_NOTIFICATION_RETRY_BASE_SECONDS * 2
        )

        while int(row["delivery_attempts"]) < 13:
            now = int(row["next_attempt_at"])
            [claimed] = kb.claim_pending_durable_goal_notifications(conn, now=now)
            assert kb.release_durable_goal_notification_claim(
                conn,
                notification_id=int(claimed["id"]),
                claim_token=str(claimed["claim_token"]),
                error="repeated failure",
                now=now,
            )
            [row] = kb.list_durable_goal_notification_rows(conn, goal_id)

        assert row["delivery_attempts"] == 13
        assert (
            row["next_attempt_at"] - now
            == kb.DURABLE_GOAL_NOTIFICATION_RETRY_CAP_SECONDS
        )
        assert kb.claim_pending_durable_goal_notifications(
            conn,
            now=int(row["next_attempt_at"]) - 1,
        ) == []


def test_goal_outbox_legacy_row_migrates_due_and_immediately_claimable(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "goal-outbox-legacy.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    with kb.connect_closing() as conn:
        goal_id = _insert_goal_outbox(conn, goal_id="g_legacy_outbox")
        conn.execute(
            "UPDATE kanban_goal_notification_outbox "
            "SET delivery_attempts = 12 WHERE goal_id = ?",
            (goal_id,),
        )
        conn.execute("DROP INDEX IF EXISTS idx_goal_outbox_due")
        conn.execute("DROP INDEX IF EXISTS idx_goal_outbox_pending")
        conn.execute(
            "ALTER TABLE kanban_goal_notification_outbox "
            "RENAME TO kanban_goal_notification_outbox_newer"
        )
        conn.execute(
            """
            CREATE TABLE kanban_goal_notification_outbox (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id           TEXT NOT NULL,
                kind              TEXT NOT NULL,
                dedupe_key        TEXT NOT NULL UNIQUE,
                payload           TEXT NOT NULL,
                platform          TEXT NOT NULL,
                chat_id           TEXT NOT NULL,
                chat_type         TEXT,
                thread_id         TEXT NOT NULL DEFAULT '',
                user_id           TEXT,
                notifier_profile  TEXT,
                delivery_metadata TEXT,
                claimed_at        INTEGER,
                claim_token       TEXT,
                delivered_at      INTEGER,
                delivery_attempts INTEGER NOT NULL DEFAULT 0,
                last_error        TEXT,
                created_at        INTEGER NOT NULL,
                FOREIGN KEY (goal_id) REFERENCES kanban_goals(id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO kanban_goal_notification_outbox (
                id, goal_id, kind, dedupe_key, payload, platform, chat_id,
                chat_type, thread_id, user_id, notifier_profile,
                delivery_metadata, claimed_at, claim_token, delivered_at,
                delivery_attempts, last_error, created_at
            )
            SELECT id, goal_id, kind, dedupe_key, payload, platform, chat_id,
                   chat_type, thread_id, user_id, notifier_profile,
                   delivery_metadata, claimed_at, claim_token, delivered_at,
                   delivery_attempts, last_error, created_at
              FROM kanban_goal_notification_outbox_newer
            """
        )
        conn.execute("DROP TABLE kanban_goal_notification_outbox_newer")
        conn.commit()

    kb.init_db()

    with kb.connect_closing() as conn:
        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(kanban_goal_notification_outbox)"
            )
        }
        [row] = kb.list_durable_goal_notification_rows(conn, goal_id)
        assert "next_attempt_at" in columns
        assert row["next_attempt_at"] is None
        [claimed] = kb.claim_pending_durable_goal_notifications(conn, now=30_000)
        assert claimed["goal_id"] == goal_id
        assert claimed["delivery_attempts"] == 13


def test_durable_goal_acceptance_chain_delivers_once_without_inbound_replay(
    tmp_path, monkeypatch
):
    from hermes_cli import profiles

    _init_durable_goal_board(tmp_path, monkeypatch, "durable-acceptance.db")
    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    candidate_sha = "a" * 40
    inbound_user_messages = ["synthetic /goal durable ship deterministic candidate"]
    with kb.connect() as conn:
        assert len(inbound_user_messages) == 1
        created = create_durable_goal(
            conn,
            objective="Ship a deterministic candidate without merge or deploy",
            origin=GoalOrigin(platform="telegram", chat_id="owner-chat"),
            board="default",
            builder_profile="builder",
            verifier_profile="verifier",
            reviewer_profile="reviewer",
            repair_budget=1,
            review_retry_budget=1,
        )
        inbound_user_messages.clear()

        def _spawn_failure(*_args, **_kwargs):
            raise RuntimeError("build worker gave up before producing a candidate")

        gave_up = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=_spawn_failure,
            failure_limit=1,
            durable_goal_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        assert gave_up.auto_blocked == [created.task_id]
        assert inbound_user_messages == []

        repair_result = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        assert repair_result.action == "CREATED_SUCCESSOR"
        repair_replay = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        assert repair_replay.action == "NOOP"
        assert [
            task.stage for task in list_durable_goal_tasks(conn, created.goal_id)
        ] == ["BUILD", "REPAIR_BUILD_1"]

        repair = kb.claim_task(conn, repair_result.task_id, claimer="builder")
        assert repair is not None and repair.current_run_id is not None
        assert kb.complete_task(
            conn,
            repair.id,
            summary="repair produced candidate",
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
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        assert verify_result.action == "CREATED_SUCCESSOR"
        assert [
            task.stage for task in list_durable_goal_tasks(conn, created.goal_id)
        ] == ["BUILD", "REPAIR_BUILD_1", "VERIFY"]

        verify = kb.claim_task(conn, verify_result.task_id, claimer="verifier")
        assert verify is not None and verify.current_run_id is not None
        assert kb.complete_task(
            conn,
            verify.id,
            summary="candidate passed verification",
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
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        assert review_result.action == "CREATED_SUCCESSOR"
        review_task = kb.get_task(conn, review_result.task_id)
        assert review_task is not None
        assert review_task.skills == ["immutable-change-reviews"]

        review = kb.claim_review_task(
            conn, review_result.task_id, claimer="reviewer", allow_durable=True
        )
        assert review is not None and review.current_run_id is not None
        assert kb.complete_task(
            conn,
            review.id,
            summary="review approved candidate",
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
        ready = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        ready_replay = supervise_goal_once(
            conn,
            created.goal_id,
            board="default",
            runtime_protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        )
        goal = get_durable_goal(conn, created.goal_id)
        notifications = list_goal_notifications(conn, created.goal_id)
        tasks = list_durable_goal_tasks(conn, created.goal_id)

    assert ready.action == "READY_FOR_OWNER"
    assert ready_replay.action == "NOOP"
    assert inbound_user_messages == []
    assert goal is not None and goal.status == "READY_FOR_OWNER"
    assert [task.stage for task in tasks] == [
        "BUILD",
        "REPAIR_BUILD_1",
        "VERIFY",
        "REVIEW",
    ]
    assert [notification.kind for notification in notifications] == [
        "READY_FOR_OWNER"
    ]

    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    assert created.goal_id in adapter.sent[0]["text"]
    with kb.connect() as conn:
        [row] = kb.list_durable_goal_notification_rows(conn, created.goal_id)
        assert row["delivered_at"] is not None
        assert row["delivery_attempts"] == 1
        assert [
            task.stage for task in list_durable_goal_tasks(conn, created.goal_id)
        ] == ["BUILD", "REPAIR_BUILD_1", "VERIFY", "REVIEW"]


def _unseen_terminal_events(tid):
    conn = kb.connect()
    try:
        _, events = kb.unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


def test_kanban_notifier_dedupes_board_slugs_pointing_to_same_db(tmp_path, monkeypatch):
    db_path = tmp_path / "shared-kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    kb.write_board_metadata("alias-a", name="Alias A")
    kb.write_board_metadata("alias-b", name="Alias B")

    tid = _create_completed_subscription()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert "Kanban" in adapter.sent[0]["text"]
    assert tid in adapter.sent[0]["text"]


def test_kanban_notifier_claim_prevents_second_watcher_send(tmp_path, monkeypatch):
    db_path = tmp_path / "single-owner.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    tid = _create_completed_subscription()

    adapter1 = RecordingAdapter()
    adapter2 = RecordingAdapter()

    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter1)))
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter2)))

    assert len(adapter1.sent) == 1
    assert adapter2.sent == []


def test_kanban_notifier_replays_telegram_dm_topic_delivery_metadata(tmp_path, monkeypatch):
    db_path = tmp_path / "dm-topic-metadata.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="dm topic task",
            assignee="worker",
            session_id="agent:main:telegram:dm:chat-1",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            thread_id="20197",
            delivery_metadata={
                "chat_type": "dm",
                "direct_messages_topic_id": "20197",
                "telegram_dm_topic_reply_fallback": True,
                "telegram_reply_to_message_id": "462",
                "thread_id": "20197",
            },
        )
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert adapter.sent[0]["metadata"] == {
        "chat_type": "dm",
        "direct_messages_topic_id": "20197",
        "telegram_dm_topic_reply_fallback": True,
        "telegram_reply_to_message_id": "462",
        "thread_id": "20197",
    }
    assert len(adapter.handled) == 1
    assert adapter.handled[0].source.chat_type == "dm"
    assert adapter.handled[0].source.thread_id == "20197"


def test_kanban_notifier_rewinds_claim_if_adapter_disconnects(tmp_path, monkeypatch):
    db_path = tmp_path / "adapter-disconnect.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = DisconnectedAdapters({Platform.TELEGRAM: RecordingAdapter()})
    runner._kanban_sub_fail_counts = {}

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert [ev.kind for ev in _unseen_terminal_events(tid)] == ["completed"]


def test_active_named_profile_subscription_is_delivered(tmp_path, monkeypatch):
    """A sub stamped with the gateway's own named profile uses self.adapters.

    Regression for #71340: on a standalone (non-multiplex) gateway running a
    named profile, _authorization_adapter() used to treat the active name as a
    multiplex secondary, find no _profile_adapters entry, fail closed, and
    rewind the claim forever — silent zero-delivery.
    """
    db_path = tmp_path / "actionable-block.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    reason = "AGE-39 — https://linear.example/AGE-39 — publishing verified."
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="approval", assignee="publisher")
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            notifier_profile="main",
        )
        kb.block_task(conn, tid, reason=reason, kind="needs_input")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "main"

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    message = adapter.sent[0]["text"]
    assert tid in message
    assert "blocked" in message


def test_kanban_db_path_is_test_isolated_from_real_home():
    hermes_home = Path(kb.kanban_home())
    production_db = Path.home() / ".hermes" / "kanban.db"
    assert kb.kanban_db_path().resolve() != production_db.resolve()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
    finally:
        conn.close()

    assert kb.kanban_db_path().resolve().is_relative_to(hermes_home.resolve())
    assert kb.kanban_db_path().resolve() != production_db.resolve()


class FailingAdapter:
    """Adapter whose send() always raises, simulating a transient send error."""

    def __init__(self):
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        self.attempts += 1
        raise RuntimeError("simulated send failure")


def test_kanban_notifier_rewinds_claim_on_send_exception(tmp_path, monkeypatch):
    """A raising adapter rewinds the claim so the next tick can retry.

    This is the second rewind path (distinct from the adapter-disconnect path
    in test_kanban_notifier_rewinds_claim_if_adapter_disconnects). Here the
    adapter is connected and the send call actually fires; the claim must
    still rewind so the event isn't lost when send() raises mid-tick.
    """
    db_path = tmp_path / "send-failure.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    adapter = FailingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # Send was attempted (so we exercised the failure path, not just the
    # disconnect path) and the claim was rewound — the unseen-events query
    # still returns the event for retry on the next tick.
    assert adapter.attempts >= 1, "send should have been attempted at least once"
    assert [ev.kind for ev in _unseen_terminal_events(tid)] == ["completed"]


class ReportedFailureAdapter:
    """Adapter that REPORTS failure via SendResult(success=False) instead of
    raising — the exact contract the Telegram adapter uses for 'Not connected'
    and degraded-send paths."""

    def __init__(self):
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        self.attempts += 1
        from gateway.platforms.base import SendResult
        return SendResult(success=False, error="Not connected")


def test_kanban_notifier_rewinds_claim_on_reported_send_failure(tmp_path, monkeypatch):
    """A non-raising SendResult(success=False) must NOT advance the cursor.

    Regression for the silent-drop bug: the notifier used to discard send()'s
    return value, so a reported (not raised) failure — e.g. Telegram mid-
    reconnect after a gateway restart — fell through to the success branch,
    marked the event seen, and lost the notification forever. The event must
    remain unseen for retry, exactly like the raised-exception path.
    """
    db_path = tmp_path / "reported-failure.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    adapter = ReportedFailureAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.attempts >= 1, "send should have been attempted"
    assert [ev.kind for ev in _unseen_terminal_events(tid)] == ["completed"], (
        "a reported send failure must rewind the claim, not silently drop the event"
    )


def test_notifier_redelivers_same_kind_on_dispatch_cycle(tmp_path, monkeypatch):
    """A retry cycle (crashed → reclaimed → crashed) notifies the user twice.

    Before #21398 the notifier auto-unsubscribed on any terminal event kind
    (gave_up / crashed / timed_out), so the second crash in a respawn cycle
    silently dropped — the subscription was already gone. This test pins the
    new contract: subscription survives non-final terminal events; the
    cursor handles dedup.

    Two crashes ten seconds apart on the same task — both should land on
    the adapter.
    """
    db_path = tmp_path / "redeliver-cycle.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="cycle test", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        # First crash — fired by the dispatcher when the worker PID dies.
        kb._append_event(conn, tid, kind="crashed")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # First crash delivered.
    assert len(adapter.sent) == 1
    assert "crashed" in adapter.sent[0]["text"].lower()

    # Subscription survives — the cursor advanced past event #1, but the
    # row is still there.
    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1, (
            "Subscription must survive a crashed event so a respawn-cycle "
            "second crash also notifies the user (issue #21398)."
        )

        # Second crash — same task, same dispatcher (or a respawn). Append
        # another event to simulate the dispatcher firing crashed a second
        # time during retry.
        kb._append_event(conn, tid, kind="crashed")
    finally:
        conn.close()

    # New tick: the second event has a fresh id past the cursor advance,
    # so it gets claimed and delivered.
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 2, (
        f"Second crashed event should also notify; got {len(adapter.sent)} "
        f"deliveries (texts: {[d['text'] for d in adapter.sent]})"
    )
    assert "crashed" in adapter.sent[1]["text"].lower()


def test_notifier_delivers_subscription_owned_by_active_profile(tmp_path, monkeypatch):
    """A single-profile gateway stamps active profile but keeps adapters primary."""
    db_path = tmp_path / "active-profile-owner.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="owned by active profile", assignee="worker")
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            notifier_profile="dev",
        )
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "dev"

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert tid in adapter.sent[0]["text"]


def test_notifier_owning_profile_adapter_no_default_fallback(tmp_path, monkeypatch):
    """A subscription owned by a secondary profile whose profile-adapter
    registry entry EXISTS but lacks this platform must NOT fall back to the
    default profile's same-platform adapter — the notifier must route through
    the shared ``_authorization_adapter`` chokepoint, which forbids that
    fallback (gateway/authz_mixin.py). Delivering via the default profile's bot
    is the exact cross-profile mis-delivery this whole change exists to fix
    (`[230002] Bot can NOT be out of the chat`).

    Mutation check: reverting kanban_watchers.py's adapter selection to the old
    inline ``if adapter is None: adapter = self.adapters.get(plat)`` fallback
    makes this test FAIL (the default adapter receives the delivery).
    """
    db_path = tmp_path / "profile-no-fallback.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="owned by beta", assignee="worker")
        # Subscription is owned by profile "beta".
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-beta",
            notifier_profile="beta",
        )
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    default_adapter = RecordingAdapter()
    other_adapter = RecordingAdapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    # Default profile has a telegram adapter …
    runner.adapters = {Platform.TELEGRAM: default_adapter}
    # … and profile "beta" HAS a non-empty registry entry (so it passes the
    # notifier's upstream skip-filter, which only skips owning profiles with NO
    # adapter at all), but that entry does NOT contain a telegram adapter — beta
    # connected a different platform (discord). The telegram sub owned by beta
    # must therefore resolve to NO adapter, not silently borrow the default
    # profile's telegram bot.
    runner._profile_adapters = {"beta": {Platform.DISCORD: other_adapter}}
    runner._kanban_sub_fail_counts = {}

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # The default profile's adapter must never receive beta's notification.
    assert default_adapter.sent == [], (
        "Owning-profile subscription must not fall back to the default "
        f"profile's adapter; got {default_adapter.sent!r}"
    )
    assert other_adapter.sent == [], (
        f"beta's discord adapter must not receive a telegram sub; got {other_adapter.sent!r}"
    )
    # The claim is rewound (adapter resolved to None → treated as disconnected),
    # so the event is still unseen and will deliver once beta's adapter connects.
    assert [ev.kind for ev in _unseen_terminal_events_for(tid, "chat-beta")] == ["completed"]


def test_notifier_claims_platform_only_a_secondary_profile_owns(tmp_path, monkeypatch):
    """A subscription owned by a secondary profile on a platform the DEFAULT
    profile never connected must still be claimed and delivered.

    Regression: the ``_collect()`` pre-filter built ``active_platforms``
    solely from ``self.adapters`` (the default profile). A sub owned by
    profile "beta" on "discord", where beta genuinely has a live discord
    adapter but the default profile has no discord adapter at all, was
    dropped by that pre-filter (``platform not in active_platforms``)
    before ``claim_unseen_events_for_sub`` ever ran — unlike the
    disconnected-adapter path, an unclaimed event is never rewound, so this
    was a permanent, silent notification loss, not a retryable one. This
    directly contradicts the feature's own purpose (routing notifications
    via the owning profile), and is the same cross-profile-adapter-lookup
    class the delivery-side chokepoint in
    ``test_notifier_owning_profile_adapter_no_default_fallback`` already
    guards — just one gate earlier.
    """
    db_path = tmp_path / "secondary-only-platform.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="owned by beta on discord", assignee="worker")
        kb.add_notify_sub(
            conn, task_id=tid, platform="discord", chat_id="chat-beta",
            notifier_profile="beta",
        )
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    beta_adapter = RecordingAdapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    # Default profile has NO discord adapter at all.
    runner.adapters = {Platform.TELEGRAM: RecordingAdapter()}
    # Secondary profile "beta" has a live discord adapter.
    runner._profile_adapters = {"beta": {Platform.DISCORD: beta_adapter}}
    runner._kanban_sub_fail_counts = {}

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(beta_adapter.sent) == 1, (
        f"beta's discord adapter should have received the notification; got {beta_adapter.sent!r}"
    )


def test_notifier_wakeup_uses_subscription_chat_type(tmp_path, monkeypatch):
    db_path = tmp_path / "chat-type-wakeup.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="dm requester",
            assignee="worker",
            session_id="origin-session",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-dm",
            chat_type="dm",
        )
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1
    assert adapter.handled[0].source.chat_type == "dm"

    # The wake must resume the creator's real DM session key — the whole bug
    # was that a hardcoded chat_type="group" made build_session_key() produce
    # a group-scoped key (a NEW session) instead of the ":dm:<chat_id>" shape
    # the original conversation runs under (#56580 / #68874).
    from gateway.session import build_session_key

    wake_key = build_session_key(adapter.handled[0].source)
    assert wake_key == "agent:main:telegram:dm:chat-dm"
    assert ":group:" not in wake_key


def test_auto_subscribe_persists_session_chat_type(tmp_path, monkeypatch):
    db_path = tmp_path / "auto-sub-chat-type.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import kanban_tools

    monkeypatch.setattr(
        kanban_tools,
        "load_config",
        lambda: {"kanban": {"auto_subscribe_on_create": True}},
    )

    tokens = set_session_vars(
        platform="telegram",
        chat_id="chat-dm",
        chat_type="dm",
    )
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="auto sub", assignee="worker")

        assert kanban_tools._maybe_auto_subscribe(conn, tid) is True
        [sub] = kb.list_notify_subs(conn, task_id=tid)
        assert sub["chat_type"] == "dm"
    finally:
        conn.close()
        clear_session_vars(tokens)


def test_notify_sub_migration_adds_chat_type_to_legacy_table(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy-notify-sub.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))

    legacy = sqlite3.connect(db_path)
    try:
        legacy.execute(
            """
            CREATE TABLE kanban_notify_subs (
                task_id       TEXT NOT NULL,
                platform      TEXT NOT NULL,
                chat_id       TEXT NOT NULL,
                thread_id     TEXT NOT NULL DEFAULT '',
                user_id       TEXT,
                notifier_profile TEXT,
                created_at    INTEGER NOT NULL,
                last_event_id INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (task_id, platform, chat_id, thread_id)
            )
            """
        )
        legacy.commit()
    finally:
        legacy.close()

    kb.init_db()
    conn = kb.connect()
    try:
        cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(kanban_notify_subs)")
        }
        assert "chat_type" in cols

        tid = kb.create_task(conn, title="legacy sub", assignee="worker")
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-dm",
            chat_type="dm",
        )
        [sub] = kb.list_notify_subs(conn, task_id=tid)
        assert sub["chat_type"] == "dm"
    finally:
        conn.close()


def _unseen_terminal_events_for(tid, chat_id):
    conn = kb.connect()
    try:
        _, events = kb.unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id=chat_id,
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


def test_kanban_notifier_isolates_per_subscription_failure(tmp_path, monkeypatch):
    """One bad subscription must not block delivery for all others.

    Regression for #59269: when claim_unseen_events_for_sub raises for one
    subscription, the entire notifier tick used to abort — silently blocking
    delivery for every other subscription.
    """
    db_path = tmp_path / "isolation.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    # Create two tasks with subscriptions and complete both. The BAD task is
    # created first: list_notify_subs() has no ORDER BY, so SQLite's natural
    # scan returns insertion order — the failing subscription must be
    # processed BEFORE the good one or this test passes even without the
    # per-subscription isolation (the good delivery happens before the tick
    # aborts). A deterministic-order shim below removes the reliance on the
    # scan order entirely.
    conn = kb.connect()
    try:
        tid_bad = kb.create_task(conn, title="bad task", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid_bad, platform="telegram", chat_id="chat-bad")
        kb.complete_task(conn, tid_bad, summary="done")

        tid_good = kb.create_task(conn, title="good task", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid_good, platform="telegram", chat_id="chat-good")
        kb.complete_task(conn, tid_good, summary="done")
    finally:
        conn.close()

    original_claim = kb.claim_unseen_events_for_sub

    def selective_claim(conn, task_id, **kwargs):
        if task_id == tid_bad:
            raise RuntimeError("simulated DB corruption for bad task")
        return original_claim(conn, task_id=task_id, **kwargs)

    monkeypatch.setattr(kb, "claim_unseen_events_for_sub", selective_claim)

    # Force the failing subscription to be iterated FIRST regardless of the
    # unordered SELECT's scan order.
    original_list = kb.list_notify_subs

    def bad_first(conn, task_id=None):
        subs = original_list(conn, task_id)
        return sorted(subs, key=lambda s: 0 if s["task_id"] == tid_bad else 1)

    monkeypatch.setattr(kb, "list_notify_subs", bad_first)

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # The good task must still be delivered despite the bad task failing.
    assert len(adapter.sent) == 1
    assert tid_good in adapter.sent[0]["text"]


def test_notifier_delivers_block_loop_detected_triage_ping(tmp_path, monkeypatch):
    """A `block_loop_detected` event must reach the subscriber as a triage ping.

    Regression for the silent-triage gap (PR #62712): kanban_db routes a task
    to `triage` after BLOCK_RECURRENCE_LIMIT re-blocks for the same cause and
    emits ONLY a `block_loop_detected` event — no `blocked`/`status` event.
    Before `block_loop_detected` joined TERMINAL_KINDS with its own message
    branch, that one transition (the whole point of which is to force human
    attention) produced zero notification and the task stalled in triage
    silently.
    """
    db_path = tmp_path / "block-loop.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="loops forever", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb._append_event(
            conn, tid, "block_loop_detected",
            {"reason": "needs credentials", "kind": "needs_input",
             "recurrences": 2, "limit": kb.BLOCK_RECURRENCE_LIMIT},
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1, "block_loop_detected must produce a notification"
    text = adapter.sent[0]["text"]
    assert "TRIAGE" in text
    assert tid in text
    assert "needs credentials" in text
    # Cursor advanced: the event is claimed and not re-delivered.
    conn = kb.connect()
    try:
        _, remaining = kb.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-1",
            kinds=["block_loop_detected"],
        )
    finally:
        conn.close()
    assert remaining == []
