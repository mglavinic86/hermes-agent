"""Deterministic supervision for explicitly opted-in durable Kanban goals.

The supervisor is a state machine over existing Kanban tasks and run records.
It never calls an LLM, merges code, pushes, or deploys. Ordinary tasks are not
present in ``kanban_goal_tasks`` and therefore never enter this module's flow.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from hermes_cli import kanban_db as kb
from agent.skill_integrity import compute_skill_tree_digest


DURABLE_GOAL_SCHEMA_VERSION = 1
DURABLE_GOAL_PROTOCOL_VERSION = 1
_CANDIDATE_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SKILL_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GoalOrigin:
    platform: str
    chat_id: str
    chat_type: Optional[str] = None
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    message_id: Optional[str] = None
    notifier_profile: Optional[str] = None
    delivery_metadata: Optional[dict[str, Any]] = None

    def as_record(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "chat_id": self.chat_id,
            "chat_type": self.chat_type,
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "message_id": self.message_id,
            "notifier_profile": self.notifier_profile,
            "delivery_metadata": self.delivery_metadata,
        }


@dataclass(frozen=True)
class DurableGoal:
    id: str
    objective: str
    status: str
    current_stage: str
    board: str
    origin: GoalOrigin
    builder_profile: str
    verifier_profile: str
    reviewer_profile: str
    reviewer_skill_digest: Optional[str]
    repair_budget: int
    repair_attempts_reserved: int
    review_retry_budget: int
    review_attempts_reserved: int
    candidate_sha: Optional[str]
    blocked_reason: Optional[str]
    schema_version: int
    protocol_version: int
    state_version: int


@dataclass(frozen=True)
class DurableGoalTask:
    goal_id: str
    task_id: str
    stage: str
    attempt: int
    expected_run_id: Optional[int]
    expected_candidate_sha: Optional[str]
    completion_event_id: Optional[int]
    completion_payload: Optional[dict[str, Any]]


@dataclass(frozen=True)
class DurableGoalCreated:
    goal_id: str
    task_id: str


@dataclass(frozen=True)
class SupervisionResult:
    action: str
    goal_id: str
    task_id: Optional[str] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class GoalNotification:
    id: int
    goal_id: str
    kind: str
    payload: dict[str, Any]
    origin: GoalOrigin
    delivered_at: Optional[int]


@dataclass(frozen=True)
class DispatchPreflight:
    allowed: bool
    supervised: bool
    reason: Optional[str] = None
    blocked_now: bool = False


@dataclass(frozen=True)
class RuntimeCompatibility:
    compatible: bool
    reason: Optional[str] = None
    runtime_id: Optional[str] = None


def _decode_object(raw: Any) -> Optional[dict[str, Any]]:
    if not raw:
        return None
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        value = json.loads(str(raw))
    except (TypeError, ValueError):
        return None
    return dict(value) if isinstance(value, dict) else None


def _goal_from_row(row) -> DurableGoal:
    return DurableGoal(
        id=str(row["id"]),
        objective=str(row["objective"]),
        status=str(row["status"]),
        current_stage=str(row["current_stage"]),
        board=str(row["board_slug"]),
        origin=GoalOrigin(
            platform=str(row["origin_platform"]),
            chat_id=str(row["origin_chat_id"]),
            chat_type=row["origin_chat_type"],
            thread_id=row["origin_thread_id"] or None,
            user_id=row["origin_user_id"],
            message_id=row["origin_message_id"],
            notifier_profile=row["notifier_profile"],
            delivery_metadata=_decode_object(row["delivery_metadata"]),
        ),
        builder_profile=str(row["builder_profile"]),
        verifier_profile=str(row["verifier_profile"]),
        reviewer_profile=str(row["reviewer_profile"]),
        reviewer_skill_digest=(
            str(row["reviewer_skill_digest"])
            if "reviewer_skill_digest" in row.keys()
            and row["reviewer_skill_digest"] is not None
            else None
        ),
        repair_budget=int(row["repair_budget"]),
        repair_attempts_reserved=int(row["repair_attempts_reserved"]),
        review_retry_budget=int(row["review_retry_budget"]),
        review_attempts_reserved=int(row["review_attempts_reserved"]),
        candidate_sha=row["candidate_sha"],
        blocked_reason=row["blocked_reason"],
        schema_version=int(row["schema_version"]),
        protocol_version=int(row["protocol_version"]),
        state_version=int(row["state_version"]),
    )


def _task_from_row(row) -> DurableGoalTask:
    return DurableGoalTask(
        goal_id=str(row["goal_id"]),
        task_id=str(row["task_id"]),
        stage=str(row["stage"]),
        attempt=int(row["attempt"]),
        expected_run_id=(
            int(row["expected_run_id"])
            if row["expected_run_id"] is not None
            else None
        ),
        expected_candidate_sha=row["expected_candidate_sha"],
        completion_event_id=(
            int(row["completion_event_id"])
            if row["completion_event_id"] is not None
            else None
        ),
        completion_payload=_decode_object(row["completion_payload"]),
    )


def create_durable_goal(
    conn,
    *,
    objective: str,
    origin: GoalOrigin,
    board: str,
    builder_profile: str,
    verifier_profile: str,
    reviewer_profile: str,
    reviewer_skill_digest: Optional[str] = None,
    repair_budget: int,
    review_retry_budget: int,
) -> DurableGoalCreated:
    """Create one gateway-routed durable goal and its initial BUILD task."""
    goal_id, task_id = kb.create_durable_goal_record(
        conn,
        objective=objective,
        board_slug=board,
        origin=origin.as_record(),
        builder_profile=builder_profile,
        verifier_profile=verifier_profile,
        reviewer_profile=reviewer_profile,
        reviewer_skill_digest=reviewer_skill_digest,
        repair_budget=repair_budget,
        review_retry_budget=review_retry_budget,
        schema_version=DURABLE_GOAL_SCHEMA_VERSION,
        protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
    )
    return DurableGoalCreated(goal_id=goal_id, task_id=task_id)


def get_durable_goal(conn, goal_id: str) -> Optional[DurableGoal]:
    row = kb.get_durable_goal_row(conn, goal_id)
    return _goal_from_row(row) if row is not None else None


def mark_durable_goal_completed_by_owner(
    conn,
    *,
    goal_id: str,
    origin: GoalOrigin,
    candidate_sha: str,
) -> bool:
    """Record owner completion only for the exact READY_FOR_OWNER candidate."""
    goal = get_durable_goal(conn, goal_id)
    normalized_sha = str(candidate_sha or "").strip().lower()
    if (
        goal is None
        or goal.status != "READY_FOR_OWNER"
        or not _CANDIDATE_SHA_RE.fullmatch(normalized_sha)
    ):
        return False
    return kb.complete_durable_goal_by_owner_record(
        conn,
        goal_id=goal.id,
        expected_state_version=goal.state_version,
        platform=str(origin.platform or "").strip().lower(),
        chat_id=str(origin.chat_id or "").strip(),
        thread_id=str(origin.thread_id or "").strip(),
        user_id=str(origin.user_id or "").strip() or None,
        candidate_sha=normalized_sha,
    )


def list_durable_goal_tasks(conn, goal_id: str) -> list[DurableGoalTask]:
    return [
        _task_from_row(row)
        for row in kb.list_durable_goal_task_rows(conn, goal_id)
    ]


def list_goal_notifications(conn, goal_id: str) -> list[GoalNotification]:
    notifications: list[GoalNotification] = []
    for row in kb.list_durable_goal_notification_rows(conn, goal_id):
        notifications.append(
            GoalNotification(
                id=int(row["id"]),
                goal_id=str(row["goal_id"]),
                kind=str(row["kind"]),
                payload=_decode_object(row["payload"]) or {},
                origin=GoalOrigin(
                    platform=str(row["platform"]),
                    chat_id=str(row["chat_id"]),
                    chat_type=row["chat_type"],
                    thread_id=row["thread_id"] or None,
                    user_id=row["user_id"],
                    notifier_profile=row["notifier_profile"],
                    delivery_metadata=_decode_object(row["delivery_metadata"]),
                ),
                delivered_at=(
                    int(row["delivered_at"])
                    if row["delivered_at"] is not None
                    else None
                ),
            )
        )
    return notifications


def check_durable_goal_runtime(
    conn,
    *,
    now: Optional[int] = None,
) -> RuntimeCompatibility:
    """Require the singleton dispatcher's exact, fresh durable protocol lease."""
    row = kb.get_durable_goal_runtime_row(conn)
    if row is None:
        return RuntimeCompatibility(False, "runtime_lease_missing")
    try:
        schema_version = int(row["schema_version"])
        protocol_version = int(row["protocol_version"])
        lease_expires_at = int(row["lease_expires_at"])
        current_time = int(time.time()) if now is None else int(now)
    except (TypeError, ValueError):
        return RuntimeCompatibility(False, "runtime_lease_malformed")
    runtime_id = str(row["runtime_id"] or "").strip() or None
    if schema_version != DURABLE_GOAL_SCHEMA_VERSION:
        return RuntimeCompatibility(False, "schema_mismatch", runtime_id)
    if protocol_version != DURABLE_GOAL_PROTOCOL_VERSION:
        return RuntimeCompatibility(False, "protocol_mismatch", runtime_id)
    if lease_expires_at < current_time:
        return RuntimeCompatibility(False, "runtime_lease_expired", runtime_id)
    if runtime_id is None:
        return RuntimeCompatibility(False, "runtime_lease_malformed")
    return RuntimeCompatibility(True, runtime_id=runtime_id)


def _profile_skill_available(profile: str, skill_name: str) -> bool:
    """Check the selected profile's own non-symlink skill snapshot."""
    return _profile_skill_digest(profile, skill_name) is not None


def _profile_skill_digest(profile: str, skill_name: str) -> Optional[str]:
    """Return the deterministic snapshot digest for a profile skill tree."""
    try:
        from agent.skill_utils import parse_frontmatter
        from hermes_cli.profiles import resolve_profile_env

        profile_home = Path(resolve_profile_env(profile)).resolve()
    except Exception:
        return False
    skills_root = profile_home / "skills"
    if not skills_root.is_dir() or skills_root.is_symlink():
        return None
    skill_root = skills_root / skill_name
    try:
        skill_root_resolved = skill_root.resolve(strict=True)
        skills_root_resolved = skills_root.resolve(strict=True)
    except OSError:
        return None
    try:
        skill_root_resolved.relative_to(skills_root_resolved)
    except ValueError:
        return None
    if (
        skill_root.is_symlink()
        or not skill_root_resolved.is_dir()
        or not skill_root_resolved.samefile(skill_root)
    ):
        return None
    root_skill = skill_root_resolved / "SKILL.md"
    if not root_skill.is_file() or root_skill.is_symlink():
        return None
    try:
        content = root_skill.read_text(encoding="utf-8")
        frontmatter, _body = parse_frontmatter(content)
    except (OSError, UnicodeError):
        return None
    declared = str(frontmatter.get("name") or "").strip()
    if declared != skill_name:
        return None

    return compute_skill_tree_digest(skill_root_resolved)


def preflight_durable_dispatch(
    conn,
    task_id: str,
    *,
    board: str,
    runtime_protocol_version: Optional[int],
    skill_available=None,
) -> DispatchPreflight:
    """Fail closed for a supervised task before a run can be claimed."""
    row = conn.execute(
        "SELECT g.* FROM kanban_goal_tasks gt "
        "JOIN kanban_goals g ON g.id = gt.goal_id "
        "WHERE gt.task_id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return DispatchPreflight(True, False)
    goal = _goal_from_row(row)
    if goal.status != "ACTIVE":
        return DispatchPreflight(False, True, "durable goal is not active")

    reason = None
    if str(board) != goal.board:
        reason = f"foreign board: expected {goal.board}, got {board}"
    elif runtime_protocol_version is None:
        reason = "durable goal protocol unavailable in dispatcher runtime"
    else:
        try:
            runtime_version = int(runtime_protocol_version)
        except (TypeError, ValueError):
            runtime_version = 0
        if runtime_version < 1 or runtime_version != goal.protocol_version:
            reason = (
                "durable goal protocol mismatch: "
                f"goal={goal.protocol_version}, runtime={runtime_version}"
            )

    binding_row = conn.execute(
        "SELECT stage FROM kanban_goal_tasks WHERE task_id = ?", (task_id,)
    ).fetchone()
    stage = str(binding_row["stage"]) if binding_row else ""
    if reason is None and stage.startswith("REVIEW"):
        required_skill = "immutable-change-reviews"
        if skill_available is not None:
            available = bool(skill_available(goal.reviewer_profile, required_skill))
        else:
            actual_digest = _profile_skill_digest(goal.reviewer_profile, required_skill)
            available = actual_digest is not None
            if available and goal.reviewer_skill_digest:
                available = actual_digest == goal.reviewer_skill_digest
        if not available:
            reason = f"missing required skill: {required_skill}"

    if reason is None:
        return DispatchPreflight(True, True)
    blocked = kb.block_durable_goal_capability(
        conn,
        goal_id=goal.id,
        task_id=task_id,
        expected_state_version=goal.state_version,
        reason=reason,
    )
    return DispatchPreflight(
        False,
        True,
        reason if blocked else "transition_lost",
        blocked_now=blocked,
    )


def _current_binding(conn, goal: DurableGoal) -> Optional[DurableGoalTask]:
    row = conn.execute(
        "SELECT * FROM kanban_goal_tasks "
        "WHERE goal_id = ? AND stage = ? "
        "ORDER BY attempt DESC LIMIT 1",
        (goal.id, goal.current_stage),
    ).fetchone()
    return _task_from_row(row) if row is not None else None


def _validated_terminal_event(conn, binding: DurableGoalTask):
    """Return the newest terminal event only when its run is exact/current."""
    rows = conn.execute(
        "SELECT * FROM task_events WHERE task_id = ? "
        "AND kind IN ('completed', 'gave_up', 'blocked', 'dependency_wait', "
        "'block_loop_detected', 'scheduled', 'archived', 'status') "
        "ORDER BY id DESC",
        (binding.task_id,),
    ).fetchall()
    latest = kb.latest_run(conn, binding.task_id)
    if latest is None or latest.ended_at is None:
        return None
    for row in rows:
        if row["run_id"] is None:
            continue
        try:
            run_id = int(row["run_id"])
        except (TypeError, ValueError):
            continue
        if run_id < 1 or latest.id != run_id:
            continue
        run = kb.get_run(conn, run_id)
        if (
            run is not None
            and run.task_id == binding.task_id
            and run.ended_at is not None
        ):
            return row
    return None


def _structured_completion_payload(
    conn,
    binding: DurableGoalTask,
    event,
    *,
    expected_stage: str,
    expected_outcome: str,
) -> Optional[dict[str, Any]]:
    run = kb.get_run(conn, int(event["run_id"]))
    payload = (
        run.metadata.get("durable_goal")
        if run is not None and isinstance(run.metadata, dict)
        else None
    )
    if not isinstance(payload, dict):
        return None
    try:
        protocol_version = int(payload.get("protocol_version"))
        payload_run_id = int(payload.get("run_id"))
    except (TypeError, ValueError):
        return None
    candidate_sha = str(payload.get("candidate_sha") or "").strip().lower()
    if (
        protocol_version != DURABLE_GOAL_PROTOCOL_VERSION
        or payload_run_id < 1
        or payload_run_id != int(event["run_id"])
        or str(payload.get("stage") or "").strip().upper() != expected_stage
        or str(payload.get("outcome") or "").strip().upper() != expected_outcome
        or not _CANDIDATE_SHA_RE.fullmatch(candidate_sha)
    ):
        return None
    normalized = dict(payload)
    normalized["candidate_sha"] = candidate_sha
    normalized["protocol_version"] = protocol_version
    normalized["run_id"] = payload_run_id
    normalized["stage"] = expected_stage
    normalized["outcome"] = expected_outcome
    return normalized


def _structured_verdict_payload(
    conn,
    event,
    *,
    expected_stage: str,
    allowed_verdicts: set[str],
) -> Optional[dict[str, Any]]:
    run = kb.get_run(conn, int(event["run_id"]))
    payload = (
        run.metadata.get("durable_goal")
        if run is not None and isinstance(run.metadata, dict)
        else None
    )
    if not isinstance(payload, dict):
        return None
    try:
        protocol_version = int(payload.get("protocol_version"))
        payload_run_id = int(payload.get("run_id"))
    except (TypeError, ValueError):
        return None
    candidate_sha = str(payload.get("candidate_sha") or "").strip().lower()
    verdict = str(payload.get("verdict") or "").strip().upper()
    if (
        protocol_version != DURABLE_GOAL_PROTOCOL_VERSION
        or payload_run_id < 1
        or payload_run_id != int(event["run_id"])
        or str(payload.get("stage") or "").strip().upper() != expected_stage
        or verdict not in allowed_verdicts
        or not _CANDIDATE_SHA_RE.fullmatch(candidate_sha)
    ):
        return None
    normalized = dict(payload)
    normalized.update(
        {
            "candidate_sha": candidate_sha,
            "protocol_version": protocol_version,
            "run_id": payload_run_id,
            "stage": expected_stage,
            "verdict": verdict,
        }
    )
    return normalized


def supervise_goal_once(
    conn,
    goal_id: str,
    *,
    board: str,
    runtime_protocol_version: int,
) -> SupervisionResult:
    """Advance at most one deterministic stage for one durable goal."""
    goal = get_durable_goal(conn, goal_id)
    if goal is None:
        return SupervisionResult("NOOP", goal_id, reason="unknown_goal")
    if goal.status != "ACTIVE":
        return SupervisionResult("NOOP", goal_id, reason="terminal_goal")
    if str(board) != goal.board:
        return SupervisionResult("NOOP", goal_id, reason="foreign_board")
    if int(runtime_protocol_version) != goal.protocol_version:
        return SupervisionResult("NOOP", goal_id, reason="protocol_mismatch")
    binding = _current_binding(conn, goal)
    if binding is None or binding.completion_event_id is not None:
        return SupervisionResult("NOOP", goal_id, reason="no_current_binding")
    event = _validated_terminal_event(conn, binding)
    if event is None:
        return SupervisionResult("NOOP", goal_id, task_id=binding.task_id)

    if event["kind"] in {
        "blocked",
        "dependency_wait",
        "block_loop_detected",
        "scheduled",
        "archived",
        "status",
    }:
        event_payload = _decode_object(event["payload"]) or {}
        event_reason = str(event_payload.get("reason") or "").strip()
        event_label = str(event["kind"])
        if event_label == "status":
            current_task = kb.get_task(conn, binding.task_id)
            direct_status = str(
                (current_task.status if current_task is not None else None)
                or event_payload.get("status")
                or "unknown"
            ).strip()
            event_label = f"status:{direct_status}"
        reason = f"durable task {event_label} during {binding.stage}"
        if event_reason:
            reason = f"{reason}: {event_reason}"
        completion_payload = dict(event_payload)
        completion_payload["event"] = event_label
        blocked = kb.transition_durable_goal_to_terminal(
            conn,
            goal_id=goal.id,
            expected_state_version=goal.state_version,
            predecessor_task_id=binding.task_id,
            completion_event_id=int(event["id"]),
            expected_run_id=int(event["run_id"]),
            completion_payload=completion_payload,
            terminal_status="BLOCKED",
            notification_kind="BLOCKED",
            notification_payload={
                "goal_id": goal.id,
                "objective": goal.objective,
                "candidate_sha": goal.candidate_sha,
                "status": "BLOCKED",
                "event": event_label,
                "reason": reason,
            },
            blocked_reason=reason,
        )
        if not blocked:
            return SupervisionResult("NOOP", goal_id, reason="transition_lost")
        return SupervisionResult("BLOCKED", goal_id, binding.task_id, reason)

    if event["kind"] == "gave_up" and (
        binding.stage == "BUILD" or binding.stage.startswith("REPAIR_BUILD_")
    ):
        next_attempt = goal.repair_attempts_reserved + 1
        if next_attempt > goal.repair_budget:
            reason = f"repair budget exhausted after {binding.stage}"
            payload = _decode_object(event["payload"]) or {
                "event": "gave_up"
            }
            blocked = kb.transition_durable_goal_to_terminal(
                conn,
                goal_id=goal.id,
                expected_state_version=goal.state_version,
                predecessor_task_id=binding.task_id,
                completion_event_id=int(event["id"]),
                expected_run_id=int(event["run_id"]),
                completion_payload=payload,
                terminal_status="BLOCKED",
                notification_kind="UNRECOVERABLE_FAILURE",
                notification_payload={
                    "goal_id": goal.id,
                    "objective": goal.objective,
                    "candidate_sha": goal.candidate_sha,
                    "status": "BLOCKED",
                    "reason": reason,
                },
                blocked_reason=reason,
            )
            if not blocked:
                return SupervisionResult("NOOP", goal_id, reason="transition_lost")
            return SupervisionResult("BLOCKED", goal_id, binding.task_id, reason)
        next_stage = f"REPAIR_BUILD_{next_attempt}"
        successor = kb.transition_durable_goal_to_successor(
            conn,
            goal_id=goal.id,
            expected_state_version=goal.state_version,
            predecessor_task_id=binding.task_id,
            completion_event_id=int(event["id"]),
            expected_run_id=int(event["run_id"]),
            next_stage=next_stage,
            next_attempt=next_attempt,
            assignee=goal.builder_profile,
            reserve_budget="repair",
        )
        if successor is None:
            return SupervisionResult("NOOP", goal_id, reason="transition_lost")
        return SupervisionResult("CREATED_SUCCESSOR", goal_id, successor)

    if event["kind"] == "completed" and (
        binding.stage == "BUILD" or binding.stage.startswith("REPAIR_BUILD_")
    ):
        payload = _structured_completion_payload(
            conn,
            binding,
            event,
            expected_stage="BUILD",
            expected_outcome="BUILT",
        )
        if payload is None:
            reason = "invalid structured BUILD payload for current run"
            blocked = kb.transition_durable_goal_to_terminal(
                conn,
                goal_id=goal.id,
                expected_state_version=goal.state_version,
                predecessor_task_id=binding.task_id,
                completion_event_id=int(event["id"]),
                expected_run_id=int(event["run_id"]),
                completion_payload={"validation_error": reason},
                terminal_status="BLOCKED",
                notification_kind="UNRECOVERABLE_FAILURE",
                notification_payload={
                    "goal_id": goal.id,
                    "objective": goal.objective,
                    "status": "BLOCKED",
                    "reason": reason,
                },
                blocked_reason=reason,
            )
            if not blocked:
                return SupervisionResult("NOOP", goal_id, reason="transition_lost")
            return SupervisionResult("BLOCKED", goal_id, binding.task_id, reason)
        candidate_sha = payload["candidate_sha"]
        is_repair = binding.stage.startswith("REPAIR_BUILD_")
        if is_repair:
            input_candidate_sha = str(
                payload.get("input_candidate_sha") or ""
            ).strip().lower()
            if goal.candidate_sha and input_candidate_sha != goal.candidate_sha:
                reason = (
                    "structured REPAIR input candidate SHA does not match "
                    "the failed candidate"
                )
                blocked = kb.transition_durable_goal_to_terminal(
                    conn,
                    goal_id=goal.id,
                    expected_state_version=goal.state_version,
                    predecessor_task_id=binding.task_id,
                    completion_event_id=int(event["id"]),
                    expected_run_id=int(event["run_id"]),
                    completion_payload=payload,
                    terminal_status="BLOCKED",
                    notification_kind="UNRECOVERABLE_FAILURE",
                    notification_payload={
                        "goal_id": goal.id,
                        "objective": goal.objective,
                        "status": "BLOCKED",
                        "reason": reason,
                        "expected_candidate_sha": goal.candidate_sha,
                        "reported_input_candidate_sha": input_candidate_sha,
                    },
                    blocked_reason=reason,
                )
                if not blocked:
                    return SupervisionResult(
                        "NOOP", goal_id, reason="transition_lost"
                    )
                return SupervisionResult(
                    "BLOCKED", goal_id, binding.task_id, reason
                )
        elif goal.candidate_sha and goal.candidate_sha != candidate_sha:
            return SupervisionResult(
                "NOOP", goal_id, binding.task_id, "candidate_sha_mismatch"
            )
        successor = kb.transition_durable_goal_to_successor(
            conn,
            goal_id=goal.id,
            expected_state_version=goal.state_version,
            predecessor_task_id=binding.task_id,
            completion_event_id=int(event["id"]),
            expected_run_id=int(event["run_id"]),
            next_stage="VERIFY",
            next_attempt=binding.attempt if is_repair else 0,
            assignee=goal.verifier_profile,
            completion_payload=payload,
            candidate_sha=candidate_sha,
        )
        if successor is None:
            return SupervisionResult("NOOP", goal_id, reason="transition_lost")
        return SupervisionResult("CREATED_SUCCESSOR", goal_id, successor)

    if event["kind"] == "completed" and binding.stage == "VERIFY":
        payload = _structured_verdict_payload(
            conn,
            event,
            expected_stage="VERIFY",
            allowed_verdicts={"PASS", "FAIL"},
        )
        if payload is None:
            reason = "invalid structured VERIFY payload for current run"
            blocked = kb.transition_durable_goal_to_terminal(
                conn,
                goal_id=goal.id,
                expected_state_version=goal.state_version,
                predecessor_task_id=binding.task_id,
                completion_event_id=int(event["id"]),
                expected_run_id=int(event["run_id"]),
                completion_payload={"validation_error": reason},
                terminal_status="BLOCKED",
                notification_kind="UNRECOVERABLE_FAILURE",
                notification_payload={
                    "goal_id": goal.id,
                    "objective": goal.objective,
                    "candidate_sha": goal.candidate_sha,
                    "status": "BLOCKED",
                    "reason": reason,
                },
                blocked_reason=reason,
            )
            if not blocked:
                return SupervisionResult("NOOP", goal_id, reason="transition_lost")
            return SupervisionResult("BLOCKED", goal_id, binding.task_id, reason)
        if payload["candidate_sha"] != goal.candidate_sha:
            reason = "structured VERIFY candidate SHA does not match built candidate"
            blocked = kb.transition_durable_goal_to_terminal(
                conn,
                goal_id=goal.id,
                expected_state_version=goal.state_version,
                predecessor_task_id=binding.task_id,
                completion_event_id=int(event["id"]),
                expected_run_id=int(event["run_id"]),
                completion_payload=payload,
                terminal_status="BLOCKED",
                notification_kind="UNRECOVERABLE_FAILURE",
                notification_payload={
                    "goal_id": goal.id,
                    "objective": goal.objective,
                    "status": "BLOCKED",
                    "reason": reason,
                    "expected_candidate_sha": goal.candidate_sha,
                    "reported_candidate_sha": payload["candidate_sha"],
                },
                blocked_reason=reason,
            )
            if not blocked:
                return SupervisionResult("NOOP", goal_id, reason="transition_lost")
            return SupervisionResult("BLOCKED", goal_id, binding.task_id, reason)
        if payload["verdict"] != "PASS":
            next_attempt = goal.repair_attempts_reserved + 1
            if next_attempt <= goal.repair_budget:
                successor = kb.transition_durable_goal_to_successor(
                    conn,
                    goal_id=goal.id,
                    expected_state_version=goal.state_version,
                    predecessor_task_id=binding.task_id,
                    completion_event_id=int(event["id"]),
                    expected_run_id=int(event["run_id"]),
                    next_stage=f"REPAIR_BUILD_{next_attempt}",
                    next_attempt=next_attempt,
                    assignee=goal.builder_profile,
                    reserve_budget="repair",
                    completion_payload=payload,
                    candidate_sha=goal.candidate_sha,
                )
                if successor is None:
                    return SupervisionResult(
                        "NOOP", goal_id, reason="transition_lost"
                    )
                return SupervisionResult(
                    "CREATED_SUCCESSOR", goal_id, successor
                )
            reason = "repair budget exhausted after structured VERIFY failure"
            blocked = kb.transition_durable_goal_to_terminal(
                conn,
                goal_id=goal.id,
                expected_state_version=goal.state_version,
                predecessor_task_id=binding.task_id,
                completion_event_id=int(event["id"]),
                expected_run_id=int(event["run_id"]),
                completion_payload=payload,
                terminal_status="BLOCKED",
                notification_kind="UNRECOVERABLE_FAILURE",
                notification_payload={
                    "goal_id": goal.id,
                    "objective": goal.objective,
                    "candidate_sha": goal.candidate_sha,
                    "status": "BLOCKED",
                    "reason": reason,
                    "failures": payload.get("failures", []),
                },
                blocked_reason=reason,
            )
            if not blocked:
                return SupervisionResult("NOOP", goal_id, reason="transition_lost")
            return SupervisionResult("BLOCKED", goal_id, binding.task_id, reason)
        successor = kb.transition_durable_goal_to_successor(
            conn,
            goal_id=goal.id,
            expected_state_version=goal.state_version,
            predecessor_task_id=binding.task_id,
            completion_event_id=int(event["id"]),
            expected_run_id=int(event["run_id"]),
            next_stage="REVIEW",
            next_attempt=0,
            assignee=goal.reviewer_profile,
            completion_payload=payload,
            candidate_sha=goal.candidate_sha,
            task_status="review",
            skills=["immutable-change-reviews"],
        )
        if successor is None:
            return SupervisionResult("NOOP", goal_id, reason="transition_lost")
        return SupervisionResult("CREATED_SUCCESSOR", goal_id, successor)

    if event["kind"] == "gave_up" and binding.stage.startswith("REVIEW"):
        # The dispatcher may emit transient crashed/timed_out events while a
        # worker is being retried. The supervisor only acts on the circuit
        # breaker's terminal gave_up event, which is already exact-current-run
        # validated by _validated_terminal_event().
        next_attempt = goal.review_attempts_reserved + 1
        event_payload = _decode_object(event["payload"]) or {"event": "gave_up"}
        if next_attempt <= goal.review_retry_budget:
            next_stage = f"REVIEW_RETRY_{next_attempt}"
            successor = kb.transition_durable_goal_to_successor(
                conn,
                goal_id=goal.id,
                expected_state_version=goal.state_version,
                predecessor_task_id=binding.task_id,
                completion_event_id=int(event["id"]),
                expected_run_id=int(event["run_id"]),
                next_stage=next_stage,
                next_attempt=next_attempt,
                assignee=goal.reviewer_profile,
                reserve_budget="review",
                completion_payload=event_payload,
                candidate_sha=goal.candidate_sha,
                task_status="review",
                skills=["immutable-change-reviews"],
            )
            if successor is None:
                return SupervisionResult("NOOP", goal_id, reason="transition_lost")
            return SupervisionResult("CREATED_SUCCESSOR", goal_id, successor)
        reason = f"review retry budget exhausted after {binding.stage} gave_up"
        blocked = kb.transition_durable_goal_to_terminal(
            conn,
            goal_id=goal.id,
            expected_state_version=goal.state_version,
            predecessor_task_id=binding.task_id,
            completion_event_id=int(event["id"]),
            expected_run_id=int(event["run_id"]),
            completion_payload=event_payload,
            terminal_status="BLOCKED",
            notification_kind="UNRECOVERABLE_FAILURE",
            notification_payload={
                "goal_id": goal.id,
                "objective": goal.objective,
                "candidate_sha": goal.candidate_sha,
                "status": "BLOCKED",
                "reason": reason,
            },
            blocked_reason=reason,
        )
        if not blocked:
            return SupervisionResult("NOOP", goal_id, reason="transition_lost")
        return SupervisionResult("BLOCKED", goal_id, binding.task_id, reason)

    if event["kind"] == "completed" and binding.stage.startswith("REVIEW"):
        event_payload = _decode_object(event["payload"])
        if not event_payload or event_payload.get("dispatch_role") != "reviewer":
            reason = "structured REVIEW completion is missing reviewer role authority"
            blocked = kb.transition_durable_goal_to_terminal(
                conn,
                goal_id=goal.id,
                expected_state_version=goal.state_version,
                predecessor_task_id=binding.task_id,
                completion_event_id=int(event["id"]),
                expected_run_id=int(event["run_id"]),
                completion_payload={"validation_error": reason},
                terminal_status="BLOCKED",
                notification_kind="UNRECOVERABLE_FAILURE",
                notification_payload={
                    "goal_id": goal.id,
                    "objective": goal.objective,
                    "candidate_sha": goal.candidate_sha,
                    "status": "BLOCKED",
                    "reason": reason,
                },
                blocked_reason=reason,
            )
            if not blocked:
                return SupervisionResult("NOOP", goal_id, reason="transition_lost")
            return SupervisionResult("BLOCKED", goal_id, binding.task_id, reason)
        payload = _structured_verdict_payload(
            conn,
            event,
            expected_stage="REVIEW",
            allowed_verdicts={"APPROVE", "BLOCKER", "MAJOR"},
        )
        if payload is None:
            next_attempt = goal.review_attempts_reserved + 1
            if next_attempt <= goal.review_retry_budget:
                next_stage = f"REVIEW_RETRY_{next_attempt}"
                successor = kb.transition_durable_goal_to_successor(
                    conn,
                    goal_id=goal.id,
                    expected_state_version=goal.state_version,
                    predecessor_task_id=binding.task_id,
                    completion_event_id=int(event["id"]),
                    expected_run_id=int(event["run_id"]),
                    next_stage=next_stage,
                    next_attempt=next_attempt,
                    assignee=goal.reviewer_profile,
                    reserve_budget="review",
                    completion_payload={
                        "validation_error": "malformed structured REVIEW payload"
                    },
                    candidate_sha=goal.candidate_sha,
                    task_status="review",
                    skills=["immutable-change-reviews"],
                )
                if successor is None:
                    return SupervisionResult(
                        "NOOP", goal_id, reason="transition_lost"
                    )
                return SupervisionResult(
                    "CREATED_SUCCESSOR", goal_id, successor
                )
            reason = "review retry budget exhausted after malformed structured REVIEW payload"
            blocked = kb.transition_durable_goal_to_terminal(
                conn,
                goal_id=goal.id,
                expected_state_version=goal.state_version,
                predecessor_task_id=binding.task_id,
                completion_event_id=int(event["id"]),
                expected_run_id=int(event["run_id"]),
                completion_payload={"validation_error": reason},
                terminal_status="BLOCKED",
                notification_kind="UNRECOVERABLE_FAILURE",
                notification_payload={
                    "goal_id": goal.id,
                    "objective": goal.objective,
                    "candidate_sha": goal.candidate_sha,
                    "status": "BLOCKED",
                    "reason": reason,
                },
                blocked_reason=reason,
            )
            if not blocked:
                return SupervisionResult("NOOP", goal_id, reason="transition_lost")
            return SupervisionResult("BLOCKED", goal_id, binding.task_id, reason)
        if payload["candidate_sha"] != goal.candidate_sha:
            reason = "structured REVIEW candidate SHA does not match verified candidate"
            blocked = kb.transition_durable_goal_to_terminal(
                conn,
                goal_id=goal.id,
                expected_state_version=goal.state_version,
                predecessor_task_id=binding.task_id,
                completion_event_id=int(event["id"]),
                expected_run_id=int(event["run_id"]),
                completion_payload=payload,
                terminal_status="BLOCKED",
                notification_kind="UNRECOVERABLE_FAILURE",
                notification_payload={
                    "goal_id": goal.id,
                    "objective": goal.objective,
                    "status": "BLOCKED",
                    "reason": reason,
                    "expected_candidate_sha": goal.candidate_sha,
                    "reported_candidate_sha": payload["candidate_sha"],
                },
                blocked_reason=reason,
            )
            if not blocked:
                return SupervisionResult("NOOP", goal_id, reason="transition_lost")
            return SupervisionResult("BLOCKED", goal_id, binding.task_id, reason)
        if payload["verdict"] != "APPROVE":
            verdict = payload["verdict"]
            reason = f"structured REVIEW verdict {verdict}"
            blocked = kb.transition_durable_goal_to_terminal(
                conn,
                goal_id=goal.id,
                expected_state_version=goal.state_version,
                predecessor_task_id=binding.task_id,
                completion_event_id=int(event["id"]),
                expected_run_id=int(event["run_id"]),
                completion_payload=payload,
                terminal_status="BLOCKED",
                notification_kind="BLOCKED" if verdict == "BLOCKER" else "MAJOR",
                notification_payload={
                    "goal_id": goal.id,
                    "objective": goal.objective,
                    "candidate_sha": goal.candidate_sha,
                    "status": "BLOCKED",
                    "verdict": verdict,
                    "findings": payload.get("findings", []),
                    "reason": reason,
                },
                blocked_reason=reason,
            )
            if not blocked:
                return SupervisionResult("NOOP", goal_id, reason="transition_lost")
            return SupervisionResult("BLOCKED", goal_id, binding.task_id, reason)
        ready = kb.transition_durable_goal_to_terminal(
            conn,
            goal_id=goal.id,
            expected_state_version=goal.state_version,
            predecessor_task_id=binding.task_id,
            completion_event_id=int(event["id"]),
            expected_run_id=int(event["run_id"]),
            completion_payload=payload,
            terminal_status="READY_FOR_OWNER",
            notification_kind="READY_FOR_OWNER",
            notification_payload={
                "goal_id": goal.id,
                "objective": goal.objective,
                "candidate_sha": goal.candidate_sha,
                "status": "READY_FOR_OWNER",
            },
        )
        if not ready:
            return SupervisionResult("NOOP", goal_id, reason="transition_lost")
        return SupervisionResult("READY_FOR_OWNER", goal_id, binding.task_id)

    return SupervisionResult(
        "NOOP", goal_id, binding.task_id, f"unhandled_{event['kind']}"
    )


def supervise_board_once(
    conn,
    *,
    board: str,
    runtime_protocol_version: int,
) -> list[SupervisionResult]:
    """Advance each active goal on one board by at most one CAS transition."""
    goal_rows = conn.execute(
        "SELECT id FROM kanban_goals "
        "WHERE status = 'ACTIVE' AND board_slug = ? ORDER BY created_at, id",
        (str(board),),
    ).fetchall()
    return [
        supervise_goal_once(
            conn,
            str(row["id"]),
            board=board,
            runtime_protocol_version=runtime_protocol_version,
        )
        for row in goal_rows
    ]


__all__ = [
    "DURABLE_GOAL_PROTOCOL_VERSION",
    "DURABLE_GOAL_SCHEMA_VERSION",
    "DurableGoal",
    "DurableGoalCreated",
    "DurableGoalTask",
    "GoalOrigin",
    "GoalNotification",
    "RuntimeCompatibility",
    "SupervisionResult",
    "create_durable_goal",
    "check_durable_goal_runtime",
    "get_durable_goal",
    "list_durable_goal_tasks",
    "list_goal_notifications",
    "mark_durable_goal_completed_by_owner",
    "supervise_board_once",
    "supervise_goal_once",
]
