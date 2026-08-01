"""Typed schemas for the policy engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PolicyDecision:
    """A deterministic decision produced by the policy engine.

    When a policy rule fires, the LLM is skipped and this decision is
    converted directly into a RoutingDecision.
    """

    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: list[str]
    rule_name: str
    priority: int = 0  # higher priority rules run first

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "action": self.action,
            "message_type": self.message_type,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "evidence_message_ids": self.evidence_message_ids,
            "rule_name": self.rule_name,
            "priority": self.priority,
        }