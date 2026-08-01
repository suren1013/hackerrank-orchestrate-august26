"""Similar message retrieval using RapidFuzz.

Finds historical messages that are similar to the incoming message by:
- same sender
- same business
- same group
- similar text (fuzzy string matching)
"""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz

from models.schemas import MessageContext
from retrieval.types import SimilarMessage
from utils.helpers import is_empty, normalize_text, safe_str


def _text_similarity(a: str, b: str) -> float:
    """Return a 0..1 similarity score between two text strings."""
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na or not nb:
        return 0.0
    return fuzz.ratio(na, nb) / 100.0


def find_similar_messages(
    context: MessageContext, top_k: int = 5
) -> list[SimilarMessage]:
    """Find the top-K most relevant historical messages.

    Scoring strategy (deterministic, weighted):
    - +0.5 if same sender_user_id
    - +0.5 if same business_id
    - +0.4 if same group_id
    - +0.3 if same conversation_type
    - text similarity (0..1) weighted by 0.6
    - +0.1 if same media_type
    """
    incoming_text = safe_str(context.message_text)
    incoming_media = safe_str(context.media_type)

    scored: list[SimilarMessage] = []

    for hist in context.historical_messages:
        score = 0.0
        reasons: list[str] = []

        # Same sender.
        if (
            context.sender_user_id
            and safe_str(hist.get("sender_user_id")) == context.sender_user_id
        ):
            score += 0.5
            reasons.append("same_sender")

        # Same business.
        if (
            context.business_id
            and safe_str(hist.get("business_id")) == context.business_id
        ):
            score += 0.5
            reasons.append("same_business")

        # Same group.
        if (
            context.group_id
            and safe_str(hist.get("group_id")) == context.group_id
        ):
            score += 0.4
            reasons.append("same_group")

        # Same conversation type.
        if safe_str(hist.get("conversation_type")) == context.conversation_type:
            score += 0.3
            reasons.append("same_conversation_type")

        # Text similarity.
        hist_text = safe_str(hist.get("message_text"))
        if incoming_text and hist_text:
            sim = _text_similarity(incoming_text, hist_text)
            score += 0.6 * sim
            if sim >= 0.5:
                reasons.append("similar_text")

        # Same media type.
        if incoming_media and safe_str(hist.get("media_type")) == incoming_media:
            score += 0.1
            reasons.append("same_media_type")

        if score > 0:
            scored.append(
                SimilarMessage(
                    message_id=safe_str(hist.get("message_id")),
                    score=score,
                    reason=";".join(reasons) if reasons else "weak_match",
                    message=hist,
                )
            )

    # Sort by score descending, then by message_id for determinism.
    scored.sort(key=lambda m: (-m.score, m.message_id))
    return scored[:top_k]