"""Evidence ranking: select the most useful historical messages as evidence.

Ranks similar messages by a usefulness score that combines:
- similarity score (from the retrieval step)
- user engagement signals (opened, replied, dismissed, muted, reported)
- recency (more recent messages are more useful)
"""

from __future__ import annotations

from typing import Any

from retrieval.types import SimilarMessage
from utils.helpers import is_empty, safe_str


def _engagement_penalty(event: dict[str, Any] | None) -> float:
    """Return a penalty (0..1) based on negative engagement signals."""
    if not event:
        return 0.0
    penalty = 0.0
    if event.get("notification_dismissed") == 1:
        penalty += 0.3
    if event.get("muted_after_message") == 1:
        penalty += 0.3
    if event.get("message_reported") == 1:
        penalty += 0.4
    return min(1.0, penalty)


def _recency_bonus(created_at: Any, reference_ts: Any) -> float:
    """Return a 0..1 recency bonus based on days between reference and message."""
    try:
        import pandas as pd

        ref = pd.Timestamp(reference_ts)
        msg_ts = pd.Timestamp(created_at)
        days = (ref - msg_ts).days
        if days < 0:
            return 0.0
        # 1.0 for same day, decaying to ~0.1 after 30 days.
        return max(0.0, 1.0 - days / 30.0)
    except Exception:
        return 0.0


def rank_evidence(
    similar_messages: list[SimilarMessage],
    events_by_message: dict[str, dict[str, Any]],
    reference_ts: Any,
    top_k: int = 5,
) -> list[SimilarMessage]:
    """Rank similar messages by usefulness and return the top-K.

    Parameters
    ----------
    similar_messages : list[SimilarMessage]
        Candidate messages from the similarity step.
    events_by_message : dict[str, dict[str, Any]]
        Map of message_id -> event dict for engagement signals.
    reference_ts : Any
        Timestamp of the incoming message (for recency).
    top_k : int
        Maximum number of evidence messages to return.

    Returns
    -------
    list[SimilarMessage]
        Ranked evidence messages (highest usefulness first).
    """
    ranked: list[SimilarMessage] = []

    for sm in similar_messages:
        event = events_by_message.get(sm.message_id, {})
        penalty = _engagement_penalty(event)
        recency = _recency_bonus(
            sm.message.get("created_at"), reference_ts
        )

        # Usefulness = similarity - engagement penalty + recency bonus.
        usefulness = sm.score - penalty + 0.2 * recency
        ranked.append(
            SimilarMessage(
                message_id=sm.message_id,
                score=usefulness,
                reason=sm.reason,
                message=sm.message,
            )
        )

    # Sort by usefulness descending, then message_id for determinism.
    ranked.sort(key=lambda m: (-m.score, m.message_id))
    return ranked[:top_k]