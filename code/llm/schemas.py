"""Typed schemas for the LLM routing layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Allowed values for routing decisions.
VALID_ACTIONS = {"notify", "digest", "mute"}
VALID_MESSAGE_TYPES = {
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
}


@dataclass
class LLMResponse:
    """Raw output from the LLM provider."""

    action: str = ""
    message_type: str = ""
    reason: str = ""
    confidence: float = 0.0
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "message_type": self.message_type,
            "reason": self.reason,
            "confidence": self.confidence,
            "raw": self.raw,
        }


@dataclass
class RoutingDecision:
    """Final validated routing decision for one message."""

    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "action": self.action,
            "message_type": self.message_type,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "evidence_message_ids": self.evidence_message_ids,
        }

    def to_output_row(self) -> dict[str, Any]:
        """Return a dict matching the output.csv contract."""
        return {
            "message_id": self.message_id,
            "action": self.action,
            "message_type": self.message_type,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "evidence_message_ids": (
                ";".join(self.evidence_message_ids)
                if self.evidence_message_ids
                else "none"
            ),
        }


@dataclass
class RouterTrace:
    """Trace of the routing pipeline for debugging/inspection."""

    message_id: str
    prompt_summary: str
    llm_response: LLMResponse
    validated: LLMResponse
    final_decision: RoutingDecision
    calibration_notes: list[str] = field(default_factory=list)
    policy_rule: str | None = None
    llm_skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "prompt_summary": self.prompt_summary,
            "llm_response": self.llm_response.to_dict(),
            "validated": self.validated.to_dict(),
            "final_decision": self.final_decision.to_dict(),
            "calibration_notes": self.calibration_notes,
            "policy_rule": self.policy_rule,
            "llm_skipped": self.llm_skipped,
        }
