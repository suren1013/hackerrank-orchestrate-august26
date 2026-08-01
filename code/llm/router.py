"""Router stub.

Later phases will implement the LLM-based routing decision that maps a
MessageContext to an action, message_type, reason, confidence, and evidence.
This stub keeps the module importable and defines the interface.
"""

from __future__ import annotations

from typing import Any

from models.schemas import MessageContext
from utils.logger import get_logger

logger = get_logger(__name__)


class Router:
    """Placeholder for the routing engine."""

    def __init__(self) -> None:
        logger.info("Router initialized (stub).")

    def route(self, context: MessageContext) -> dict[str, Any]:
        """Return a routing decision for a message context.

        Stub implementation returns a placeholder decision.
        """
        return {
            "message_id": context.message_id,
            "action": "digest",
            "message_type": "unknown",
            "reason": "Router not yet implemented.",
            "confidence": 0.5,
            "evidence_message_ids": "none",
        }