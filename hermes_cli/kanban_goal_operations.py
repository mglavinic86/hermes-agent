"""Durable taskless operation replay for Workflow V2 trusted stages."""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Optional

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_trusted_stages import (
    PromotionEvidence,
    PromotionRequest,
    ResultClassification,
    TrustedStageRegistry,
    get_trusted_stage_registry,
)


class OperationState(str, Enum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    RETRYABLE = "RETRYABLE"
    SUCCEEDED = "SUCCEEDED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class GoalOperation:
    operation_id: str
    goal_id: str
    kind: str
    stage_attempt: int
    request_hash: str
    request_payload: dict[str, object]
    state: OperationState
    claim_token: Optional[str]
    claimed_at: Optional[int]
    lease_expires_at: Optional[int]
    attempt_count: int
    next_attempt_at: Optional[int]
    response_hash: Optional[str]
    response_payload: Optional[dict[str, object]]
    last_error: Optional[str]
    created_at: int
    updated_at: int
    completed_at: Optional[int]


def _decode_object(raw: object) -> dict[str, object]:
    if isinstance(raw, Mapping):
        return dict(raw)
    value = json.loads(str(raw))
    if not isinstance(value, dict):
        raise ValueError("operation JSON payload is not an object")
    return value


def _operation_from_row(row) -> GoalOperation:
    return GoalOperation(
        operation_id=str(row["operation_id"]),
        goal_id=str(row["goal_id"]),
        kind=str(row["kind"]),
        stage_attempt=int(row["stage_attempt"]),
        request_hash=str(row["request_hash"]),
        request_payload=_decode_object(row["request_payload"]),
        state=OperationState(str(row["state"])),
        claim_token=row["claim_token"],
        claimed_at=int(row["claimed_at"]) if row["claimed_at"] is not None else None,
        lease_expires_at=(
            int(row["lease_expires_at"]) if row["lease_expires_at"] is not None else None
        ),
        attempt_count=int(row["attempt_count"]),
        next_attempt_at=(
            int(row["next_attempt_at"]) if row["next_attempt_at"] is not None else None
        ),
        response_hash=row["response_hash"],
        response_payload=(
            _decode_object(row["response_payload"])
            if row["response_payload"] is not None
            else None
        ),
        last_error=row["last_error"],
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
        completed_at=(
            int(row["completed_at"]) if row["completed_at"] is not None else None
        ),
    )


def get_operation(conn, operation_id: str) -> Optional[GoalOperation]:
    row = conn.execute(
        "SELECT * FROM kanban_goal_operations WHERE operation_id = ?",
        (operation_id,),
    ).fetchone()
    return _operation_from_row(row) if row is not None else None


def claim_due_operation(
    conn,
    *,
    now: Optional[int] = None,
    lease_seconds: int = 300,
    claim_token: Optional[str] = None,
) -> Optional[GoalOperation]:
    current_time = int(time.time()) if now is None else int(now)
    token = claim_token or secrets.token_urlsafe(24)
    with kb.write_txn(conn):
        row = conn.execute(
            """
            SELECT * FROM kanban_goal_operations
             WHERE (
                    state = 'PENDING'
                    OR (state = 'RETRYABLE' AND COALESCE(next_attempt_at, 0) <= ?)
                    OR (state = 'CLAIMED' AND lease_expires_at IS NOT NULL
                        AND lease_expires_at <= ?)
                   )
             ORDER BY created_at, operation_id
             LIMIT 1
            """,
            (current_time, current_time),
        ).fetchone()
        if row is None:
            return None
        updated = conn.execute(
            """
            UPDATE kanban_goal_operations
               SET state = 'CLAIMED', claim_token = ?, claimed_at = ?,
                   lease_expires_at = ?, attempt_count = attempt_count + 1,
                   updated_at = ?
             WHERE operation_id = ? AND request_hash = ?
               AND (
                    state = 'PENDING'
                    OR (state = 'RETRYABLE' AND COALESCE(next_attempt_at, 0) <= ?)
                    OR (state = 'CLAIMED' AND lease_expires_at IS NOT NULL
                        AND lease_expires_at <= ?)
                   )
            """,
            (
                token,
                current_time,
                current_time + int(lease_seconds),
                current_time,
                row["operation_id"],
                row["request_hash"],
                current_time,
                current_time,
            ),
        )
        if updated.rowcount != 1:
            return None
        claimed = conn.execute(
            "SELECT * FROM kanban_goal_operations WHERE operation_id = ?",
            (row["operation_id"],),
        ).fetchone()
    return _operation_from_row(claimed) if claimed is not None else None


def _request_from_operation(operation: GoalOperation) -> PromotionRequest:
    payload = operation.request_payload
    return PromotionRequest(
        contract_hash=str(payload["contract_hash"]),
        base_revision=str(payload["base_revision"]),
        scope=tuple(str(item) for item in payload.get("scope") or ()),
        gates=tuple(str(item) for item in payload.get("gates") or ()),
        candidate_sha=str(payload["candidate_sha"]),
        attempt=int(payload["stage_attempt"]),
        operation_id=operation.operation_id,
        request_hash=operation.request_hash,
        contract_version=int(payload.get("contract_version") or 1),
        candidate_tree=str(payload.get("candidate_tree") or ""),
        branch_identity=str(payload.get("branch_identity") or ""),
        pr_identity=str(payload.get("pr_identity") or ""),
        remote_base=str(payload.get("remote_base") or ""),
        remote_head=str(payload.get("remote_head") or ""),
        remote_tree=str(payload.get("remote_tree") or ""),
        gate_evidence=dict(payload.get("deterministic_gate_evidence") or {}),
    )


def _hard_block_payload(operation: GoalOperation, summary: str) -> dict[str, object]:
    request = _request_from_operation(operation)
    evidence = PromotionEvidence.create(
        adapter_id=str(operation.request_payload.get("adapter_id") or "unknown-adapter"),
        request=request,
        classification=ResultClassification.HARD_BLOCK,
        summary=summary,
    )
    return evidence.as_payload()


def _validate_response_payload(
    operation: GoalOperation,
    response_payload: Mapping[str, object],
) -> dict[str, object]:
    evidence = PromotionEvidence.from_payload(response_payload)
    request = _request_from_operation(operation)
    if (
        evidence.operation_id != operation.operation_id
        or evidence.request_hash != operation.request_hash
        or evidence.adapter_id != str(operation.request_payload.get("adapter_id") or "")
        or evidence.contract_hash != request.contract_hash
        or evidence.contract_version != request.contract_version
        or evidence.base_revision != request.base_revision
        or evidence.scope != request.scope
        or evidence.gates != request.gates
        or evidence.candidate_sha != request.candidate_sha
        or evidence.attempt != request.attempt
        or evidence.candidate_tree != request.candidate_tree
        or evidence.branch_identity != request.branch_identity
        or evidence.pr_identity != request.pr_identity
        or evidence.remote_base != request.remote_base
        or evidence.remote_head != request.remote_head
        or evidence.remote_tree != request.remote_tree
        or dict(evidence.gate_evidence or {}) != dict(request.gate_evidence or {})
    ):
        raise ValueError("trusted operation response does not match request")
    return evidence.as_payload()


def execute_due_operation_once(
    conn,
    *,
    registry: Optional[TrustedStageRegistry] = None,
    now: Optional[int] = None,
    token_factory: Callable[[], str] | None = None,
    lease_seconds: int = 300,
) -> Optional[GoalOperation]:
    current_time = int(time.time()) if now is None else int(now)
    token = token_factory() if token_factory is not None else secrets.token_urlsafe(24)
    operation = claim_due_operation(
        conn, now=current_time, lease_seconds=lease_seconds, claim_token=token
    )
    if operation is None:
        return None

    selected_registry = registry or get_trusted_stage_registry()
    adapter_id = str(operation.request_payload.get("adapter_id") or "")
    try:
        adapter = selected_registry.adapter(adapter_id)
        evidence = adapter.classify(_request_from_operation(operation))
        response_payload = _validate_response_payload(operation, evidence.as_payload())
    except Exception as exc:
        response_payload = _hard_block_payload(operation, f"trusted adapter failed closed: {exc}")

    row = kb.ack_goal_operation_result(
        conn,
        operation_id=operation.operation_id,
        claim_token=operation.claim_token or token,
        request_hash=operation.request_hash,
        response_payload=response_payload,
        now=current_time,
    )
    return _operation_from_row(row) if row is not None else None


__all__ = [
    "GoalOperation",
    "OperationState",
    "claim_due_operation",
    "execute_due_operation_once",
    "get_operation",
]
