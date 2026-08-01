"""Retriever stub.

Later phases will implement retrieval of relevant historical messages
(e.g. by sender, conversation, text similarity, or media similarity) to
support evidence selection for the router.

This stub exists so the module structure is in place and importable.
"""

from __future__ import annotations

from typing import Any

from models.schemas import MessageContext
from utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    """Placeholder for the retrieval engine."""

    def __init__(self) -> None:
        logger.info("Retriever initialized (stub).")

    def retrieve_evidence(
        self, context: MessageContext, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Return candidate evidence messages for a given context.

        Stub implementation: returns the historical messages already attached
        to the context, truncated to ``top_k``.
        """
        return context.historical_messages[:top_k]