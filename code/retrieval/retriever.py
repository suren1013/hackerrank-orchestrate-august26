"""RetrievalEngine: converts a MessageContext into a compact evidence package.

The engine orchestrates the analysis modules:
- sender analysis
- business analysis
- group analysis
- engagement summary
- similar message retrieval (RapidFuzz)
- evidence ranking
- retrieval summary generation

It is deterministic, uses no LLM/OCR/ASR, and is efficient for hundreds of
historical messages.
"""

from __future__ import annotations

from typing import Any

from models.schemas import MessageContext
from retrieval.analyzers import (
    analyze_business,
    analyze_engagement,
    analyze_group,
    analyze_sender,
)
from retrieval.evidence import rank_evidence
from retrieval.similarity import find_similar_messages
from retrieval.summarizer import build_retrieval_summary
from retrieval.types import RetrievalResult, SimilarMessage
from utils.helpers import safe_str
from utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalEngine:
    """Builds a compact evidence package for a single message context."""

    def __init__(
        self,
        top_similar: int = 5,
        top_evidence: int = 5,
    ) -> None:
        self.top_similar = top_similar
        self.top_evidence = top_evidence
        logger.info(
            "RetrievalEngine initialized (top_similar=%d, top_evidence=%d).",
            top_similar,
            top_evidence,
        )

    def retrieve(self, context: MessageContext) -> RetrievalResult:
        """Produce a RetrievalResult for a message context."""
        # 1. Relationship analyses.
        sender_rel = analyze_sender(context)
        business_rel = analyze_business(context)
        group_rel = analyze_group(context)
        engagement = analyze_engagement(context)

        # 2. Similar message retrieval.
        similar = find_similar_messages(context, top_k=self.top_similar)

        # 3. Evidence ranking (re-rank by usefulness).
        events_by_message: dict[str, dict[str, Any]] = {}
        for ev in context.historical_events:
            mid = safe_str(ev.get("message_id"))
            if mid:
                events_by_message[mid] = ev

        evidence = rank_evidence(
            similar_messages=similar,
            events_by_message=events_by_message,
            reference_ts=context.created_at,
            top_k=self.top_evidence,
        )

        # 4. Evidence message IDs.
        evidence_ids = [m.message_id for m in evidence]

        # 5. Trust and interest scores.
        trust_score = self._compute_trust_score(
            sender_rel=sender_rel,
            business_rel=business_rel,
            group_rel=group_rel,
            engagement=engagement,
        )
        interest_score = self._compute_interest_score(
            sender_rel=sender_rel,
            business_rel=business_rel,
            group_rel=group_rel,
            engagement=engagement,
        )

        # 6. Retrieval summary.
        summary = build_retrieval_summary(
            sender=sender_rel,
            business=business_rel,
            group=group_rel,
            engagement=engagement,
        )

        return RetrievalResult(
            message_id=context.message_id,
            evidence_message_ids=evidence_ids,
            top_similar_messages=evidence,
            sender_relationship=sender_rel,
            business_relationship=business_rel,
            group_relationship=group_rel,
            engagement_summary=engagement,
            trust_score=trust_score,
            interest_score=interest_score,
            retrieval_summary=summary,
        )

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def _compute_trust_score(
        self,
        sender_rel,
        business_rel,
        group_rel,
        engagement,
    ) -> float:
        """Compute a 0..1 trust score for the sender/business/group."""
        score = 0.0

        # Sender trust.
        if sender_rel is not None:
            if sender_rel.total_messages == 0:
                score += 0.1  # unknown sender, low trust
            else:
                # Higher reply/read rates -> more trust.
                score += 0.3 * sender_rel.reply_frequency
                score += 0.2 * sender_rel.read_frequency
                # High ignore/archive rates -> less trust.
                score -= 0.2 * sender_rel.ignore_frequency
                score -= 0.2 * sender_rel.archive_frequency

        # Business trust.
        if business_rel is not None:
            score += 0.5 * business_rel.trust_score

        # Group trust.
        if group_rel is not None:
            if group_rel.is_admin:
                score += 0.3
            elif group_rel.is_member:
                score += 0.2
            if group_rel.is_muted:
                score -= 0.1

        # Engagement trust: high report/mute rates reduce trust.
        if engagement.total_historical > 0:
            score -= 0.3 * engagement.report_rate
            score -= 0.2 * engagement.mute_rate

        return max(0.0, min(1.0, score))

    def _compute_interest_score(
        self,
        sender_rel,
        business_rel,
        group_rel,
        engagement,
    ) -> float:
        """Compute a 0..1 interest score for the message topic/sender."""
        score = 0.0

        # Sender interest.
        if sender_rel is not None:
            if sender_rel.total_messages > 0:
                score += 0.3 * sender_rel.reply_frequency
                score += 0.2 * sender_rel.read_frequency

        # Business interest.
        if business_rel is not None:
            if business_rel.is_active_customer:
                score += 0.3
            if business_rel.has_frequent_interactions:
                score += 0.2
            if business_rel.allows_promotions:
                score += 0.1
            if business_rel.opted_out:
                score -= 0.2

        # Group interest.
        if group_rel is not None:
            if group_rel.is_active:
                score += 0.2
            if group_rel.is_admin:
                score += 0.1
            if group_rel.is_muted:
                score -= 0.2

        # Engagement interest: high open/reply rates -> more interest.
        if engagement.total_historical > 0:
            score += 0.2 * engagement.open_rate
            score += 0.2 * engagement.reply_rate

        return max(0.0, min(1.0, score))