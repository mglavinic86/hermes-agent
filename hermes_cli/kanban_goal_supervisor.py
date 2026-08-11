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
from hermes_cli.kanban_trusted_stages import (
    AdjudicationDecision,
    Authority,
    FindingSeverity,
    PlanDecision,
    PromotionEvidence,
    ReviewVerdict,
    ResultClassification,
    TaskContract,
)


DURABLE_GOAL_SCHEMA_VERSION = 2
DURABLE_GOAL_PROTOCOL_VERSION = 2
DURABLE_GOAL_WORKFLOW_VERSION = 2
_CANDIDATE_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


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
    workflow_version: int
    board: str
    origin: GoalOrigin
    task_contract: Optional[TaskContract]
    resolver_id: Optional[str]
    orchestrator_profile: Optional[str]
    verify_promote_adapter_id: Optional[str]
    promotion_evidence: Optional[dict[str, Any]]
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
class _ValidatedReviewEvidence:
    payload: dict[str, Any]
    snapshot: dict[str, Any]


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
    contract = None
    if "task_contract_json" in row.keys() and row["task_contract_json"]:
        try:
            raw_contract = json.loads(str(row["task_contract_json"]))
            if isinstance(raw_contract, dict):
                contract = TaskContract.from_payload(raw_contract)
        except (TypeError, ValueError, json.JSONDecodeError):
            contract = None
    return DurableGoal(
        id=str(row["id"]),
        objective=str(row["objective"]),
        status=str(row["status"]),
        current_stage=str(row["current_stage"]),
        workflow_version=(
            int(row["workflow_version"]) if "workflow_version" in row.keys() else 1
        ),
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
        task_contract=contract,
        resolver_id=(
            str(row["resolver_id"])
            if "resolver_id" in row.keys() and row["resolver_id"] is not None
            else None
        ),
        orchestrator_profile=(
            str(row["orchestrator_profile"])
            if "orchestrator_profile" in row.keys()
            and row["orchestrator_profile"] is not None
            else None
        ),
        verify_promote_adapter_id=(
            str(row["verify_promote_adapter_id"])
            if "verify_promote_adapter_id" in row.keys()
            and row["verify_promote_adapter_id"] is not None
            else None
        ),
        promotion_evidence=(
            _decode_object(row["promotion_evidence"])
            if "promotion_evidence" in row.keys()
            else None
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
            int(row["expected_run_id"]) if row["expected_run_id"] is not None else None
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
    """Create a history-compatible V1 record that V2 will never execute."""
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
        schema_version=1,
        protocol_version=1,
    )
    return DurableGoalCreated(goal_id=goal_id, task_id=task_id)


def create_trusted_durable_goal(
    conn,
    *,
    contract: TaskContract,
    origin: GoalOrigin,
    board: str,
    resolver_id: str,
    verify_promote_adapter_id: str,
    orchestrator_profile: str,
    builder_profile: str,
    reviewer_profile: str,
    reviewer_skill_digest: Optional[str],
    repair_budget: int,
    review_retry_budget: int,
) -> DurableGoalCreated:
    """Create one immutable-contract Workflow V2 goal and initial PLAN task."""
    try:
        contract_payload = contract.as_payload()
        canonical_contract = TaskContract.from_payload(contract_payload)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("trusted task contract hash is invalid") from None
    contract = canonical_contract
    goal_id, task_id = kb.create_trusted_durable_goal_record(
        conn,
        task_contract=contract.as_payload(),
        board_slug=board,
        origin=origin.as_record(),
        resolver_id=resolver_id,
        verify_promote_adapter_id=verify_promote_adapter_id,
        orchestrator_profile=orchestrator_profile,
        builder_profile=builder_profile,
        reviewer_profile=reviewer_profile,
        reviewer_skill_digest=reviewer_skill_digest,
        repair_budget=repair_budget,
        review_retry_budget=review_retry_budget,
        schema_version=DURABLE_GOAL_SCHEMA_VERSION,
        protocol_version=DURABLE_GOAL_PROTOCOL_VERSION,
        workflow_version=DURABLE_GOAL_WORKFLOW_VERSION,
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
        _task_from_row(row) for row in kb.list_durable_goal_task_rows(conn, goal_id)
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


def _latest_completed_build_binding(conn, goal_id: str) -> Optional[DurableGoalTask]:
    source_rows = [
        binding
        for binding in list_durable_goal_tasks(conn, goal_id)
        if binding.completion_event_id is not None
        and (
            binding.stage == "BUILD_CANDIDATE"
            or binding.stage.startswith("REPAIR_BUILD_")
        )
    ]
    return (
        max(source_rows, key=lambda binding: binding.attempt) if source_rows else None
    )


def apply_trusted_stage_result(
    conn,
    goal_id: str,
    evidence: PromotionEvidence,
) -> SupervisionResult:
    """Apply precomputed deterministic evidence; never execute an adapter."""
    goal = get_durable_goal(conn, goal_id)
    if goal is None:
        return SupervisionResult("NOOP", goal_id, reason="unknown_goal")
    if (
        goal.status != "ACTIVE"
        or goal.workflow_version != DURABLE_GOAL_WORKFLOW_VERSION
        or goal.current_stage != "VERIFY_PROMOTE"
        or goal.task_contract is None
        or not isinstance(evidence, PromotionEvidence)
        or not evidence.verify_hash()
    ):
        return SupervisionResult("NOOP", goal_id, reason="invalid_trusted_stage_state")
    contract = goal.task_contract
    if (
        evidence.adapter_id != goal.verify_promote_adapter_id
        or evidence.contract_hash != contract.contract_hash
        or evidence.base_revision != contract.base_revision
        or evidence.scope != contract.scope
        or evidence.gates != contract.gates
        or evidence.candidate_sha != goal.candidate_sha
    ):
        return SupervisionResult("NOOP", goal_id, reason="trusted_evidence_mismatch")
    source = _latest_completed_build_binding(conn, goal.id)
    if source is None:
        return SupervisionResult("NOOP", goal_id, reason="missing_build_source")
    if evidence.attempt != source.attempt:
        return SupervisionResult("NOOP", goal_id, reason="stale_trusted_evidence")
    if evidence.classification == ResultClassification.PASS:
        successor = kb.transition_durable_goal_from_waiting_to_successor(
            conn,
            goal_id=goal.id,
            expected_state_version=goal.state_version,
            expected_waiting_stage="VERIFY_PROMOTE",
            source_task_id=source.task_id,
            next_stage="REVIEW",
            next_attempt=source.attempt,
            assignee=goal.reviewer_profile,
            evidence=evidence.as_payload(),
            candidate_sha=evidence.candidate_sha,
            task_status="review",
            skills=("immutable-change-reviews",),
        )
        if successor is None:
            return SupervisionResult("NOOP", goal.id, reason="transition_lost")
        return SupervisionResult("CREATED_REVIEW", goal.id, successor)
    if evidence.classification == ResultClassification.REPAIRABLE_FAILURE:
        successor = kb.transition_durable_goal_from_waiting_to_successor(
            conn,
            goal_id=goal.id,
            expected_state_version=goal.state_version,
            expected_waiting_stage="VERIFY_PROMOTE",
            source_task_id=source.task_id,
            next_stage="ADJUDICATE",
            next_attempt=source.attempt,
            assignee=str(goal.orchestrator_profile or ""),
            evidence=evidence.as_payload(),
            candidate_sha=evidence.candidate_sha,
        )
        if successor is None:
            return SupervisionResult("NOOP", goal.id, reason="transition_lost")
        return SupervisionResult("CREATED_ADJUDICATE", goal.id, successor)
    if evidence.classification == ResultClassification.HARD_BLOCK:
        transitioned = kb.transition_durable_goal_waiting_to_terminal(
            conn,
            goal_id=goal.id,
            expected_state_version=goal.state_version,
            expected_waiting_stage="VERIFY_PROMOTE",
            evidence=evidence.as_payload(),
            terminal_status="BLOCKED",
            notification_kind="HUMAN_GATE",
            notification_payload={
                "goal_id": goal.id,
                "objective": goal.objective,
                "candidate_sha": goal.candidate_sha,
                "status": "BLOCKED",
                "classification": ResultClassification.HARD_BLOCK.value,
                "reason": evidence.summary,
            },
            blocked_reason=evidence.summary,
        )
        if not transitioned:
            return SupervisionResult("NOOP", goal.id, reason="transition_lost")
        return SupervisionResult("HUMAN_GATE", goal.id, reason=evidence.summary)
    return SupervisionResult(
        evidence.classification.value, goal.id, reason=evidence.summary
    )


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
    if goal.workflow_version != DURABLE_GOAL_WORKFLOW_VERSION:
        reason = (
            f"durable workflow V{goal.workflow_version} is retired; "
            "owner intervention is required"
        )
    elif str(board) != goal.board:
        reason = f"foreign board: expected {goal.board}, got {board}"
    elif goal.schema_version != DURABLE_GOAL_SCHEMA_VERSION:
        reason = (
            "durable goal schema mismatch: "
            f"goal={goal.schema_version}, runtime={DURABLE_GOAL_SCHEMA_VERSION}"
        )
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


def _structured_v2_authority_payload(
    conn,
    goal: DurableGoal,
    binding: DurableGoalTask,
    event,
    *,
    expected_stage: str,
    expected_authority: str,
    expected_profile: str,
) -> Optional[dict[str, Any]]:
    """Validate a V2 LLM result against its immutable/current authority."""
    contract = goal.task_contract
    if contract is None:
        return None
    run = kb.get_run(conn, int(event["run_id"]))
    payload = (
        run.metadata.get("durable_goal")
        if run is not None and isinstance(run.metadata, dict)
        else None
    )
    task = kb.get_task(conn, binding.task_id)
    if not isinstance(payload, dict) or run is None or task is None:
        return None
    try:
        workflow_version = int(payload.get("workflow_version"))
        protocol_version = int(payload.get("protocol_version"))
        payload_run_id = int(payload.get("run_id"))
    except (TypeError, ValueError):
        return None
    if (
        workflow_version != DURABLE_GOAL_WORKFLOW_VERSION
        or protocol_version != DURABLE_GOAL_PROTOCOL_VERSION
        or payload_run_id != int(event["run_id"])
        or str(payload.get("stage") or "").strip().upper() != expected_stage
        or str(payload.get("authority") or "").strip().lower() != expected_authority
        or str(run.profile or "").strip().lower() != expected_profile.lower()
        or str(task.assignee or "").strip().lower() != expected_profile.lower()
        or str(payload.get("contract_hash") or "").strip() != contract.contract_hash
        or str(payload.get("base_revision") or "").strip() != contract.base_revision
        or tuple(str(item) for item in (payload.get("scope") or ())) != contract.scope
        or tuple(str(item) for item in (payload.get("gates") or ())) != contract.gates
    ):
        return None
    normalized = dict(payload)
    normalized.update({
        "authority": expected_authority,
        "base_revision": contract.base_revision,
        "contract_hash": contract.contract_hash,
        "gates": list(contract.gates),
        "protocol_version": protocol_version,
        "run_id": payload_run_id,
        "scope": list(contract.scope),
        "stage": expected_stage,
        "workflow_version": workflow_version,
    })
    return normalized


def _validate_review_payload(
    payload: Optional[dict[str, Any]],
    goal: DurableGoal,
) -> Optional[dict[str, Any]]:
    if payload is None:
        return None
    candidate_sha = str(payload.get("candidate_sha") or "").strip().lower()
    verdict_raw = str(payload.get("verdict") or "").strip().upper()
    if candidate_sha != str(goal.candidate_sha or ""):
        return None
    try:
        verdict = ReviewVerdict(verdict_raw)
    except ValueError:
        return None
    if (
        goal.reviewer_skill_digest
        and str(payload.get("reviewer_skill_digest") or "").strip()
        != goal.reviewer_skill_digest
    ):
        return None
    raw_findings = payload.get("findings") or []
    if not isinstance(raw_findings, list):
        return None
    findings: list[dict[str, Any]] = []
    for finding in raw_findings:
        if not isinstance(finding, Mapping):
            return None
        severity_raw = str(finding.get("severity") or "").strip().upper()
        summary = str(finding.get("summary") or "").strip()
        resolved = finding.get("resolved", False)
        try:
            severity = FindingSeverity(severity_raw)
        except ValueError:
            return None
        if not summary or not isinstance(resolved, bool):
            return None
        normalized_finding = dict(finding)
        normalized_finding["severity"] = severity.value
        normalized_finding["summary"] = summary
        normalized_finding["resolved"] = resolved
        findings.append(normalized_finding)
    normalized = dict(payload)
    normalized["candidate_sha"] = candidate_sha
    normalized["verdict"] = verdict.value
    normalized["findings"] = findings
    return normalized


def _validated_current_review_payload(
    conn,
    *,
    goal: DurableGoal,
    adjudicate_binding: DurableGoalTask,
) -> Optional[_ValidatedReviewEvidence]:
    rows = conn.execute(
        "SELECT * FROM kanban_goal_tasks "
        "WHERE goal_id = ? AND (stage = 'REVIEW' OR stage LIKE 'REVIEW_RETRY_%') "
        "AND attempt = ? "
        "AND completion_event_id IS NOT NULL ORDER BY created_at DESC",
        (goal.id, adjudicate_binding.attempt),
    ).fetchall()
    if len(rows) != 1:
        return None
    review_binding = _task_from_row(rows[0])
    if (
        review_binding.expected_run_id is None
        or review_binding.completion_event_id is None
        or review_binding.expected_candidate_sha != str(goal.candidate_sha or "")
        or adjudicate_binding.expected_candidate_sha != str(goal.candidate_sha or "")
    ):
        return None
    event = _validated_terminal_event(conn, review_binding)
    if (
        event is None
        or int(event["id"]) != review_binding.completion_event_id
        or str(event["kind"] or "") != "completed"
        or str(event["task_id"] or "") != review_binding.task_id
        or int(event["run_id"]) != review_binding.expected_run_id
    ):
        return None
    task = kb.get_task(conn, review_binding.task_id)
    run = kb.get_run(conn, int(event["run_id"]))
    if task is None or run is None:
        return None
    authoritative = _structured_v2_authority_payload(
        conn,
        goal,
        review_binding,
        event,
        expected_stage=review_binding.stage,
        expected_authority=Authority.REVIEWER.value,
        expected_profile=goal.reviewer_profile,
    )
    authoritative = _validate_review_payload(authoritative, goal)
    if authoritative is None or review_binding.completion_payload != authoritative:
        return None
    return _ValidatedReviewEvidence(
        payload=authoritative,
        snapshot={
            "binding": {
                "goal_id": review_binding.goal_id,
                "task_id": review_binding.task_id,
                "stage": review_binding.stage,
                "attempt": review_binding.attempt,
                "expected_run_id": review_binding.expected_run_id,
                "expected_candidate_sha": review_binding.expected_candidate_sha,
                "completion_event_id": review_binding.completion_event_id,
                "completion_payload": review_binding.completion_payload,
            },
            "task": {
                "id": task.id,
                "assignee": task.assignee,
                "status": task.status,
                "current_run_id": task.current_run_id,
            },
            "event": {
                "id": int(event["id"]),
                "task_id": str(event["task_id"]),
                "run_id": int(event["run_id"]),
                "kind": str(event["kind"]),
                "payload": event["payload"],
            },
            "run": {
                "id": run.id,
                "task_id": run.task_id,
                "profile": run.profile,
                "status": run.status,
                "ended_at": run.ended_at,
                "outcome": run.outcome,
                "metadata": run.metadata,
            },
        },
    )


def _ready_authority_snapshot(
    goal: DurableGoal,
    adjudicate_binding: DurableGoalTask,
    review: _ValidatedReviewEvidence,
) -> dict[str, Any]:
    contract = goal.task_contract
    assert contract is not None
    assert isinstance(goal.promotion_evidence, Mapping)
    return {
        "goal": {
            "current_stage": goal.current_stage,
            "candidate_sha": goal.candidate_sha,
            "task_contract": contract.as_payload(),
            "verify_promote_adapter_id": goal.verify_promote_adapter_id,
            "promotion_evidence": dict(goal.promotion_evidence),
            "reviewer_profile": goal.reviewer_profile,
            "reviewer_skill_digest": goal.reviewer_skill_digest,
        },
        "adjudicate_binding": {
            "goal_id": adjudicate_binding.goal_id,
            "task_id": adjudicate_binding.task_id,
            "stage": adjudicate_binding.stage,
            "attempt": adjudicate_binding.attempt,
            "expected_run_id": adjudicate_binding.expected_run_id,
            "expected_candidate_sha": adjudicate_binding.expected_candidate_sha,
            "completion_event_id": adjudicate_binding.completion_event_id,
        },
        "review": review.snapshot,
    }


def _validated_current_promotion_evidence(
    conn,
    goal: DurableGoal,
    binding: DurableGoalTask,
) -> Optional[PromotionEvidence]:
    contract = goal.task_contract
    if contract is None or not isinstance(goal.promotion_evidence, Mapping):
        return None
    try:
        evidence = PromotionEvidence.from_payload(goal.promotion_evidence)
    except (TypeError, ValueError):
        return None
    source = _latest_completed_build_binding(conn, goal.id)
    if (
        source is None
        or evidence.classification != ResultClassification.PASS
        or evidence.adapter_id != str(goal.verify_promote_adapter_id or "")
        or evidence.contract_hash != contract.contract_hash
        or evidence.base_revision != contract.base_revision
        or evidence.scope != contract.scope
        or evidence.gates != contract.gates
        or evidence.candidate_sha != str(goal.candidate_sha or "")
        or binding.expected_candidate_sha != str(goal.candidate_sha or "")
        or evidence.attempt != source.attempt
    ):
        return None
    return evidence


def _terminal_event_payload(
    goal: DurableGoal, binding: DurableGoalTask, event
) -> dict[str, Any]:
    return {
        "workflow_version": DURABLE_GOAL_WORKFLOW_VERSION,
        "protocol_version": DURABLE_GOAL_PROTOCOL_VERSION,
        "stage": binding.stage,
        "run_id": int(event["run_id"]),
        "event_kind": str(event["kind"]),
        "candidate_sha": goal.candidate_sha,
    }


def _block_v2_goal_from_current_event(
    conn,
    goal: DurableGoal,
    binding: DurableGoalTask,
    event,
    *,
    reason: str,
    notification_kind: str = "HUMAN_GATE",
) -> SupervisionResult:
    payload = _terminal_event_payload(goal, binding, event)
    payload["reason"] = reason
    transitioned = kb.transition_durable_goal_to_terminal(
        conn,
        goal_id=goal.id,
        expected_state_version=goal.state_version,
        predecessor_task_id=binding.task_id,
        completion_event_id=int(event["id"]),
        expected_run_id=int(event["run_id"]),
        completion_payload=payload,
        terminal_status="BLOCKED",
        notification_kind=notification_kind,
        notification_payload={
            "goal_id": goal.id,
            "objective": goal.objective,
            "candidate_sha": goal.candidate_sha,
            "status": "BLOCKED",
            "reason": reason,
        },
        blocked_reason=reason,
    )
    if not transitioned:
        return SupervisionResult("NOOP", goal.id, reason="transition_lost")
    return SupervisionResult("BLOCKED", goal.id, binding.task_id, reason)


def _retry_v2_review_from_current_event(
    conn,
    goal: DurableGoal,
    binding: DurableGoalTask,
    event,
    *,
    reason: str,
) -> SupervisionResult:
    next_attempt = goal.review_attempts_reserved + 1
    successor = kb.transition_durable_goal_to_successor(
        conn,
        goal_id=goal.id,
        expected_state_version=goal.state_version,
        predecessor_task_id=binding.task_id,
        completion_event_id=int(event["id"]),
        expected_run_id=int(event["run_id"]),
        next_stage=f"REVIEW_RETRY_{next_attempt}",
        next_attempt=next_attempt,
        assignee=goal.reviewer_profile,
        reserve_budget="review",
        completion_payload={
            **_terminal_event_payload(goal, binding, event),
            "reason": reason,
        },
        candidate_sha=goal.candidate_sha,
        task_status="review",
        skills=("immutable-change-reviews",),
    )
    if successor is None:
        return _block_v2_goal_from_current_event(
            conn,
            goal,
            binding,
            event,
            reason=reason,
            notification_kind="HUMAN_GATE",
        )
    return SupervisionResult("CREATED_SUCCESSOR", goal.id, successor)


def _payload_has_blocking_findings(payload: Mapping[str, Any]) -> bool:
    for finding in payload.get("findings") or []:
        if not isinstance(finding, Mapping):
            return True
        if (
            str(finding.get("severity") or "").strip().upper() in {"BLOCKER", "MAJOR"}
            and finding.get("resolved") is not True
        ):
            return True
    return False


def _supervise_v2_goal_once(
    conn,
    goal: DurableGoal,
    binding: DurableGoalTask,
    event,
) -> SupervisionResult:
    if event["kind"] in {
        "blocked",
        "dependency_wait",
        "block_loop_detected",
        "scheduled",
        "archived",
        "status",
    }:
        return _block_v2_goal_from_current_event(
            conn,
            goal,
            binding,
            event,
            reason=f"durable stage {binding.stage} ended with {event['kind']}",
            notification_kind="HUMAN_GATE",
        )
    if event["kind"] == "gave_up":
        if binding.stage == "BUILD_CANDIDATE" or binding.stage.startswith(
            "REPAIR_BUILD_"
        ):
            return _block_v2_goal_from_current_event(
                conn,
                goal,
                binding,
                event,
                reason=f"durable stage {binding.stage} gave up; owner adjudication required",
                notification_kind="HUMAN_GATE",
            )
        if binding.stage.startswith("REVIEW"):
            return _retry_v2_review_from_current_event(
                conn,
                goal,
                binding,
                event,
                reason=f"durable stage {binding.stage} gave up",
            )
        return _block_v2_goal_from_current_event(
            conn,
            goal,
            binding,
            event,
            reason=f"durable stage {binding.stage} gave up",
            notification_kind="HUMAN_GATE",
        )
    if event["kind"] == "completed" and binding.stage == "PLAN":
        payload = _structured_v2_authority_payload(
            conn,
            goal,
            binding,
            event,
            expected_stage="PLAN",
            expected_authority=Authority.ORCHESTRATOR.value,
            expected_profile=str(goal.orchestrator_profile or ""),
        )
        decision = str((payload or {}).get("decision") or "").strip().upper()
        if payload is None:
            return _block_v2_goal_from_current_event(
                conn,
                goal,
                binding,
                event,
                reason="invalid_structured_plan",
                notification_kind="UNRECOVERABLE_FAILURE",
            )
        if decision == PlanDecision.HUMAN_GATE.value:
            reason = str(payload.get("reason") or "human gate requested").strip()
            transitioned = kb.transition_durable_goal_to_terminal(
                conn,
                goal_id=goal.id,
                expected_state_version=goal.state_version,
                predecessor_task_id=binding.task_id,
                completion_event_id=int(event["id"]),
                expected_run_id=int(event["run_id"]),
                completion_payload=payload,
                terminal_status="BLOCKED",
                notification_kind="HUMAN_GATE",
                notification_payload={
                    "goal_id": goal.id,
                    "objective": goal.objective,
                    "status": "BLOCKED",
                    "reason": reason,
                },
                blocked_reason=reason,
            )
            if not transitioned:
                return SupervisionResult("NOOP", goal.id, reason="transition_lost")
            return SupervisionResult("HUMAN_GATE", goal.id, binding.task_id, reason)
        if decision != PlanDecision.PLAN_ACCEPTED.value:
            return SupervisionResult(
                "NOOP", goal.id, binding.task_id, "invalid_structured_plan"
            )
        successor = kb.transition_durable_goal_to_successor(
            conn,
            goal_id=goal.id,
            expected_state_version=goal.state_version,
            predecessor_task_id=binding.task_id,
            completion_event_id=int(event["id"]),
            expected_run_id=int(event["run_id"]),
            next_stage="BUILD_CANDIDATE",
            next_attempt=0,
            assignee=goal.builder_profile,
            completion_payload=payload,
        )
        if successor is None:
            return SupervisionResult("NOOP", goal.id, reason="transition_lost")
        return SupervisionResult("CREATED_SUCCESSOR", goal.id, successor)
    if event["kind"] == "completed" and binding.stage.startswith("REVIEW"):
        payload = _structured_v2_authority_payload(
            conn,
            goal,
            binding,
            event,
            expected_stage=binding.stage,
            expected_authority=Authority.REVIEWER.value,
            expected_profile=goal.reviewer_profile,
        )
        payload = _validate_review_payload(payload, goal)
        if payload is None:
            return _retry_v2_review_from_current_event(
                conn,
                goal,
                binding,
                event,
                reason="invalid_structured_review",
            )
        successor = kb.transition_durable_goal_to_successor(
            conn,
            goal_id=goal.id,
            expected_state_version=goal.state_version,
            predecessor_task_id=binding.task_id,
            completion_event_id=int(event["id"]),
            expected_run_id=int(event["run_id"]),
            next_stage="ADJUDICATE",
            next_attempt=binding.attempt,
            assignee=str(goal.orchestrator_profile or ""),
            completion_payload=payload,
            candidate_sha=payload["candidate_sha"],
        )
        if successor is None:
            return SupervisionResult("NOOP", goal.id, reason="transition_lost")
        return SupervisionResult("CREATED_ADJUDICATE", goal.id, successor)
    if event["kind"] == "completed" and binding.stage == "ADJUDICATE":
        payload = _structured_v2_authority_payload(
            conn,
            goal,
            binding,
            event,
            expected_stage="ADJUDICATE",
            expected_authority=Authority.ORCHESTRATOR.value,
            expected_profile=str(goal.orchestrator_profile or ""),
        )
        candidate_sha = str((payload or {}).get("candidate_sha") or "").strip().lower()
        decision_raw = str((payload or {}).get("decision") or "").strip().upper()
        try:
            decision = AdjudicationDecision(decision_raw)
        except ValueError:
            return _block_v2_goal_from_current_event(
                conn,
                goal,
                binding,
                event,
                reason="invalid_structured_adjudication",
                notification_kind="UNRECOVERABLE_FAILURE",
            )
        if payload is None or candidate_sha != str(goal.candidate_sha or ""):
            return _block_v2_goal_from_current_event(
                conn,
                goal,
                binding,
                event,
                reason="invalid_structured_adjudication",
                notification_kind="UNRECOVERABLE_FAILURE",
            )
        if decision == AdjudicationDecision.READY_FOR_OWNER:
            evidence = _validated_current_promotion_evidence(conn, goal, binding)
            if evidence is None:
                return _block_v2_goal_from_current_event(
                    conn,
                    goal,
                    binding,
                    event,
                    reason="invalid_promotion_evidence",
                    notification_kind="HUMAN_GATE",
                )
            review = _validated_current_review_payload(
                conn,
                goal=goal,
                adjudicate_binding=binding,
            )
            if review is None:
                return _block_v2_goal_from_current_event(
                    conn,
                    goal,
                    binding,
                    event,
                    reason="invalid_review_evidence",
                    notification_kind="HUMAN_GATE",
                )
            review_payload = review.payload
            if (
                str(review_payload.get("verdict") or "") != ReviewVerdict.APPROVE.value
                or str(review_payload.get("candidate_sha") or "") != candidate_sha
                or _payload_has_blocking_findings(review_payload)
                or bool(payload.get("human_gate"))
            ):
                return SupervisionResult(
                    "NOOP", goal.id, binding.task_id, "ready_validation_failed"
                )
            transitioned = kb.transition_durable_goal_to_terminal(
                conn,
                goal_id=goal.id,
                expected_state_version=goal.state_version,
                predecessor_task_id=binding.task_id,
                completion_event_id=int(event["id"]),
                expected_run_id=int(event["run_id"]),
                completion_payload=payload,
                terminal_status="READY_FOR_OWNER",
                ready_authority_snapshot=_ready_authority_snapshot(
                    goal, binding, review
                ),
                notification_kind="READY_FOR_OWNER",
                notification_payload={
                    "goal_id": goal.id,
                    "objective": goal.objective,
                    "candidate_sha": candidate_sha,
                    "status": "READY_FOR_OWNER",
                },
            )
            if not transitioned:
                return _block_v2_goal_from_current_event(
                    conn,
                    goal,
                    binding,
                    event,
                    reason="ready_authority_changed",
                    notification_kind="HUMAN_GATE",
                )
            return SupervisionResult("READY_FOR_OWNER", goal.id, binding.task_id)
        if decision == AdjudicationDecision.REPAIR:
            next_attempt = goal.repair_attempts_reserved + 1
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
                candidate_sha=candidate_sha,
            )
            if successor is None:
                return _block_v2_goal_from_current_event(
                    conn,
                    goal,
                    binding,
                    event,
                    reason="repair budget exhausted; owner adjudication required",
                    notification_kind="HUMAN_GATE",
                )
            return SupervisionResult("CREATED_REPAIR", goal.id, successor)
        reason = str(payload.get("reason") or "human gate requested").strip()
        transitioned = kb.transition_durable_goal_to_terminal(
            conn,
            goal_id=goal.id,
            expected_state_version=goal.state_version,
            predecessor_task_id=binding.task_id,
            completion_event_id=int(event["id"]),
            expected_run_id=int(event["run_id"]),
            completion_payload=payload,
            terminal_status="BLOCKED",
            notification_kind="HUMAN_GATE",
            notification_payload={
                "goal_id": goal.id,
                "objective": goal.objective,
                "candidate_sha": candidate_sha,
                "status": "BLOCKED",
                "reason": reason,
            },
            blocked_reason=reason,
        )
        if not transitioned:
            return SupervisionResult("NOOP", goal.id, reason="transition_lost")
        return SupervisionResult("HUMAN_GATE", goal.id, binding.task_id, reason)
    if event["kind"] == "completed" and (
        binding.stage == "BUILD_CANDIDATE" or binding.stage.startswith("REPAIR_BUILD_")
    ):
        payload = _structured_v2_authority_payload(
            conn,
            goal,
            binding,
            event,
            expected_stage=binding.stage,
            expected_authority=Authority.BUILDER.value,
            expected_profile=goal.builder_profile,
        )
        candidate_sha = str((payload or {}).get("candidate_sha") or "").strip().lower()
        if payload is None or not _CANDIDATE_SHA_RE.fullmatch(candidate_sha):
            return _block_v2_goal_from_current_event(
                conn,
                goal,
                binding,
                event,
                reason="invalid_structured_build",
                notification_kind="UNRECOVERABLE_FAILURE",
            )
        if binding.stage.startswith("REPAIR_BUILD_"):
            input_candidate_sha = (
                str(payload.get("input_candidate_sha") or "").strip().lower()
            )
            expected_input = (
                str(binding.expected_candidate_sha or goal.candidate_sha or "")
                .strip()
                .lower()
            )
            if (
                not _CANDIDATE_SHA_RE.fullmatch(input_candidate_sha)
                or input_candidate_sha != expected_input
            ):
                return _block_v2_goal_from_current_event(
                    conn,
                    goal,
                    binding,
                    event,
                    reason="repair_input_candidate_mismatch",
                    notification_kind="UNRECOVERABLE_FAILURE",
                )
        transitioned = kb.transition_durable_goal_to_waiting_stage(
            conn,
            goal_id=goal.id,
            expected_state_version=goal.state_version,
            predecessor_task_id=binding.task_id,
            completion_event_id=int(event["id"]),
            expected_run_id=int(event["run_id"]),
            completion_payload=payload,
            waiting_stage="VERIFY_PROMOTE",
            candidate_sha=candidate_sha,
        )
        if not transitioned:
            return SupervisionResult("NOOP", goal.id, reason="transition_lost")
        return SupervisionResult("AWAITING_TRUSTED_RESULT", goal.id)
    return SupervisionResult("NOOP", goal.id, binding.task_id)


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
    if goal.workflow_version != DURABLE_GOAL_WORKFLOW_VERSION:
        blocked = kb.block_legacy_durable_goal_under_v2(
            conn,
            goal_id=goal.id,
            expected_state_version=goal.state_version,
            reason="legacy durable goal cannot run under Workflow V2 supervisor",
        )
        return SupervisionResult(
            "HUMAN_GATE" if blocked else "NOOP",
            goal.id,
            reason=(
                "legacy durable goal cannot run under Workflow V2 supervisor"
                if blocked
                else "transition_lost"
            ),
        )
    if goal.schema_version != DURABLE_GOAL_SCHEMA_VERSION:
        return SupervisionResult("NOOP", goal_id, reason="schema_mismatch")
    if int(runtime_protocol_version) != goal.protocol_version:
        return SupervisionResult("NOOP", goal_id, reason="protocol_mismatch")
    binding = _current_binding(conn, goal)
    if binding is None or binding.completion_event_id is not None:
        return SupervisionResult("NOOP", goal_id, reason="no_current_binding")
    event = _validated_terminal_event(conn, binding)
    if event is None:
        return SupervisionResult("NOOP", goal_id, task_id=binding.task_id)

    return _supervise_v2_goal_once(conn, goal, binding, event)


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
    "DURABLE_GOAL_WORKFLOW_VERSION",
    "DurableGoal",
    "DurableGoalCreated",
    "DurableGoalTask",
    "GoalOrigin",
    "GoalNotification",
    "RuntimeCompatibility",
    "SupervisionResult",
    "create_durable_goal",
    "create_trusted_durable_goal",
    "apply_trusted_stage_result",
    "check_durable_goal_runtime",
    "get_durable_goal",
    "list_durable_goal_tasks",
    "list_goal_notifications",
    "mark_durable_goal_completed_by_owner",
    "supervise_board_once",
    "supervise_goal_once",
]
