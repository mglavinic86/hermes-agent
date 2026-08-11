"""Trusted, generic protocol types for durable Kanban Workflow V2.

This module defines data contracts and static registries only.  It does not
execute adapters, import implementations dynamically, run shell commands, or
perform remote operations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Protocol


class Authority(str, Enum):
    ORCHESTRATOR = "orchestrator"
    BUILDER = "builder"
    REVIEWER = "reviewer"


class ResultClassification(str, Enum):
    PASS = "PASS"
    RETRYABLE = "RETRYABLE"
    REPAIRABLE_FAILURE = "REPAIRABLE_FAILURE"
    HARD_BLOCK = "HARD_BLOCK"


class PlanDecision(str, Enum):
    PLAN_ACCEPTED = "PLAN_ACCEPTED"
    HUMAN_GATE = "HUMAN_GATE"


class ReviewVerdict(str, Enum):
    APPROVE = "APPROVE"
    CHANGES_REQUIRED = "CHANGES_REQUIRED"
    BLOCKED = "BLOCKED"


class FindingSeverity(str, Enum):
    BLOCKER = "BLOCKER"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    NIT = "NIT"


class AdjudicationDecision(str, Enum):
    READY_FOR_OWNER = "READY_FOR_OWNER"
    REPAIR = "REPAIR"
    HUMAN_GATE = "HUMAN_GATE"


def _canonical_contract_payload(
    *,
    reference: str,
    objective: str,
    base_revision: str,
    scope: tuple[str, ...],
    gates: tuple[str, ...],
    authority: Authority,
) -> dict[str, object]:
    return {
        "authority": authority.value,
        "base_revision": base_revision,
        "gates": list(gates),
        "objective": objective,
        "reference": reference,
        "scope": list(scope),
    }


@dataclass(frozen=True)
class TaskContract:
    reference: str
    objective: str
    base_revision: str
    scope: tuple[str, ...]
    gates: tuple[str, ...]
    authority: Authority
    contract_hash: str

    @classmethod
    def create(
        cls,
        *,
        reference: str,
        objective: str,
        base_revision: str,
        scope: tuple[str, ...],
        gates: tuple[str, ...],
        authority: Authority = Authority.ORCHESTRATOR,
    ) -> "TaskContract":
        normalized_reference = str(reference or "").strip()
        normalized_objective = str(objective or "").strip()
        normalized_base = str(base_revision or "").strip()
        normalized_scope = tuple(
            str(item).strip() for item in scope if str(item).strip()
        )
        normalized_gates = tuple(
            str(item).strip() for item in gates if str(item).strip()
        )
        normalized_authority = Authority(authority)
        if not normalized_reference:
            raise ValueError("task contract reference is required")
        if not normalized_objective:
            raise ValueError("task contract objective is required")
        if not normalized_base:
            raise ValueError("task contract base_revision is required")
        if not normalized_scope:
            raise ValueError("task contract scope is required")
        if not normalized_gates:
            raise ValueError("task contract gates are required")
        payload = _canonical_contract_payload(
            reference=normalized_reference,
            objective=normalized_objective,
            base_revision=normalized_base,
            scope=normalized_scope,
            gates=normalized_gates,
            authority=normalized_authority,
        )
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(
            reference=normalized_reference,
            objective=normalized_objective,
            base_revision=normalized_base,
            scope=normalized_scope,
            gates=normalized_gates,
            authority=normalized_authority,
            contract_hash=digest,
        )

    def as_payload(self) -> dict[str, object]:
        payload = _canonical_contract_payload(
            reference=self.reference,
            objective=self.objective,
            base_revision=self.base_revision,
            scope=self.scope,
            gates=self.gates,
            authority=self.authority,
        )
        payload["contract_hash"] = self.contract_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TaskContract":
        rebuilt = cls.create(
            reference=str(payload.get("reference") or ""),
            objective=str(payload.get("objective") or ""),
            base_revision=str(payload.get("base_revision") or ""),
            scope=tuple(str(item) for item in (payload.get("scope") or ())),
            gates=tuple(str(item) for item in (payload.get("gates") or ())),
            authority=Authority(str(payload.get("authority") or "")),
        )
        stored_hash = str(payload.get("contract_hash") or "").strip()
        if stored_hash != rebuilt.contract_hash:
            raise ValueError("task contract hash does not match its payload")
        return rebuilt

    def verify_hash(self) -> bool:
        rebuilt = type(self).create(
            reference=self.reference,
            objective=self.objective,
            base_revision=self.base_revision,
            scope=self.scope,
            gates=self.gates,
            authority=self.authority,
        )
        return self.contract_hash == rebuilt.contract_hash


class StaticTaskContractResolver:
    """Resolve opaque references only from an explicitly supplied mapping."""

    def __init__(
        self,
        *,
        resolver_id: str,
        contracts: Mapping[str, TaskContract],
    ) -> None:
        normalized_id = str(resolver_id or "").strip()
        if not normalized_id:
            raise ValueError("resolver_id is required")
        copied = dict(contracts)
        if any(
            reference != contract.reference for reference, contract in copied.items()
        ):
            raise ValueError(
                "task contract registry keys must match contract references"
            )
        if any(not contract.verify_hash() for contract in copied.values()):
            raise ValueError("task contract registry contains an invalid contract hash")
        self.resolver_id = normalized_id
        self._contracts = MappingProxyType(copied)

    def resolve(self, opaque_reference: str) -> TaskContract:
        reference = str(opaque_reference or "").strip()
        if not reference:
            raise ValueError("opaque task contract reference is required")
        try:
            return self._contracts[reference]
        except KeyError as exc:
            raise KeyError(
                f"unknown trusted task contract reference: {reference}"
            ) from exc


class TaskContractResolver(Protocol):
    resolver_id: str

    def resolve(self, opaque_reference: str) -> TaskContract: ...


@dataclass(frozen=True)
class PromotionRequest:
    contract_hash: str
    base_revision: str
    scope: tuple[str, ...]
    gates: tuple[str, ...]
    candidate_sha: str
    attempt: int


@dataclass(frozen=True)
class PromotionEvidence:
    adapter_id: str
    contract_hash: str
    base_revision: str
    scope: tuple[str, ...]
    gates: tuple[str, ...]
    candidate_sha: str
    attempt: int
    classification: ResultClassification
    summary: str
    evidence_hash: str

    @classmethod
    def create(
        cls,
        *,
        adapter_id: str,
        request: PromotionRequest,
        classification: ResultClassification,
        summary: str,
    ) -> "PromotionEvidence":
        normalized_adapter = str(adapter_id or "").strip()
        normalized_summary = str(summary or "").strip()
        normalized_classification = ResultClassification(classification)
        if not normalized_adapter:
            raise ValueError("adapter_id is required")
        if not normalized_summary:
            raise ValueError("promotion evidence summary is required")
        payload = {
            "adapter_id": normalized_adapter,
            "attempt": int(request.attempt),
            "base_revision": request.base_revision,
            "candidate_sha": request.candidate_sha,
            "classification": normalized_classification.value,
            "contract_hash": request.contract_hash,
            "gates": list(request.gates),
            "scope": list(request.scope),
            "summary": normalized_summary,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(
            adapter_id=normalized_adapter,
            contract_hash=request.contract_hash,
            base_revision=request.base_revision,
            scope=tuple(request.scope),
            gates=tuple(request.gates),
            candidate_sha=request.candidate_sha,
            attempt=int(request.attempt),
            classification=normalized_classification,
            summary=normalized_summary,
            evidence_hash=digest,
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "adapter_id": self.adapter_id,
            "attempt": self.attempt,
            "base_revision": self.base_revision,
            "candidate_sha": self.candidate_sha,
            "classification": self.classification.value,
            "contract_hash": self.contract_hash,
            "evidence_hash": self.evidence_hash,
            "gates": list(self.gates),
            "scope": list(self.scope),
            "summary": self.summary,
        }

    def verify_hash(self) -> bool:
        rebuilt = type(self).create(
            adapter_id=self.adapter_id,
            request=PromotionRequest(
                contract_hash=self.contract_hash,
                base_revision=self.base_revision,
                scope=self.scope,
                gates=self.gates,
                candidate_sha=self.candidate_sha,
                attempt=self.attempt,
            ),
            classification=self.classification,
            summary=self.summary,
        )
        return self.evidence_hash == rebuilt.evidence_hash


class VerifyPromoteAdapter(Protocol):
    adapter_id: str

    def classify(self, request: PromotionRequest) -> PromotionEvidence: ...


class TrustedStageRegistry:
    """Immutable ID lookup for explicitly provided trusted implementations."""

    def __init__(
        self,
        *,
        resolvers: Mapping[str, TaskContractResolver],
        adapters: Mapping[str, VerifyPromoteAdapter],
    ) -> None:
        resolver_copy = dict(resolvers)
        adapter_copy = dict(adapters)
        if any(key != value.resolver_id for key, value in resolver_copy.items()):
            raise ValueError("resolver registry keys must match resolver_id")
        if any(key != value.adapter_id for key, value in adapter_copy.items()):
            raise ValueError("adapter registry keys must match adapter_id")
        self._resolvers = MappingProxyType(resolver_copy)
        self._adapters = MappingProxyType(adapter_copy)

    def resolver(self, resolver_id: str) -> TaskContractResolver:
        normalized = str(resolver_id or "").strip()
        try:
            return self._resolvers[normalized]
        except KeyError as exc:
            raise KeyError(
                f"unknown trusted task contract resolver: {normalized}"
            ) from exc

    def adapter(self, adapter_id: str) -> VerifyPromoteAdapter:
        normalized = str(adapter_id or "").strip()
        try:
            return self._adapters[normalized]
        except KeyError as exc:
            raise KeyError(
                f"unknown trusted verify/promote adapter: {normalized}"
            ) from exc


# PR 223-A intentionally ships no production resolver or adapter
# implementation. Deployments may replace this process-local static registry
# during trusted bootstrap; config can only select IDs already present here.
TRUSTED_STAGE_REGISTRY = TrustedStageRegistry(resolvers={}, adapters={})


def get_trusted_stage_registry() -> TrustedStageRegistry:
    return TRUSTED_STAGE_REGISTRY


__all__ = [
    "AdjudicationDecision",
    "Authority",
    "FindingSeverity",
    "PlanDecision",
    "PromotionEvidence",
    "PromotionRequest",
    "ReviewVerdict",
    "ResultClassification",
    "StaticTaskContractResolver",
    "TaskContract",
    "TaskContractResolver",
    "TRUSTED_STAGE_REGISTRY",
    "TrustedStageRegistry",
    "VerifyPromoteAdapter",
    "get_trusted_stage_registry",
]
