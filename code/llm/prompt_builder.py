"""Builds structured prompts for the routing LLM.

Organizes the incoming message, conversation type, retrieval summary,
media summary, relationship summaries, engagement data, urgency, and
safety signals into readable sections — not raw JSON dumps.
"""

from __future__ import annotations

from typing import Any

from media.types import MediaResult
from models.schemas import MessageContext
from retrieval.types import RetrievalResult
from utils.helpers import safe_str

SYSTEM_PROMPT = """You are a message notification router for WhatsApp.

For each incoming message, decide how it should be handled for the receiving user:

- notify: important enough to interrupt the user now
- digest: useful but can be shown later
- mute: low-value, repetitive, unwanted, suspicious, or unsafe

Return a JSON object with exactly these fields:
{
  "action": "notify" | "digest" | "mute",
  "message_type": "personal" | "urgent" | "event" | "payment" | "business_update" | "promotion" | "greeting" | "forward" | "spam" | "scam" | "unknown",
  "reason": "short human-readable explanation (1-2 sentences)",
  "confidence": 0.0 to 1.0
}

Rules:
- Risky messages (OTP requests, payment pressure, phishing links, QR payment prompts) must be muted with message_type scam or spam.
- Personalized decisions: use the user's relationship with the sender/business/group.
- A trusted sender with an urgent request should be notified.
- Repeated forwards/greetings from a sender the user ignores should be muted.
- A verified business the user actively uses should be notified for transactional updates.
- Promotions from businesses the user opted out of should be muted.
"""


def build_user_prompt(
    context: MessageContext,
    retrieval: RetrievalResult,
    media: MediaResult | None = None,
) -> str:
    """Build a structured user prompt for the routing LLM."""
    sections: list[str] = []

    # 1. Incoming message.
    sections.append("## INCOMING MESSAGE")
    sections.append(f"Message ID: {context.message_id}")
    sections.append(f"Conversation type: {context.conversation_type}")
    sections.append(f"Created at: {context.created_at}")
    sections.append(f"Forwarded count: {context.forwarded_count}")
    if context.message_text:
        sections.append(f"Text: {context.message_text}")
    else:
        sections.append("Text: (none)")

    # 2. Media summary.
    if media is not None and media.has_media:
        sections.append("## MEDIA CONTENT")
        sections.append(f"Media type: {media.media_type}")
        if media.summary:
            sections.append(f"Summary: {media.summary}")
        if media.entities:
            for key, values in media.entities.items():
                if values:
                    sections.append(f"{key}: {', '.join(values[:5])}")
        if media.urgency_indicators:
            sections.append(f"Urgency indicators: {', '.join(media.urgency_indicators)}")
        if media.safety_indicators:
            sections.append(f"Safety indicators: {', '.join(media.safety_indicators)}")

    # 3. Retrieval summary.
    sections.append("## RETRIEVAL SUMMARY")
    sections.append(retrieval.retrieval_summary)

    # 4. Sender relationship.
    if retrieval.sender_relationship is not None:
        sr = retrieval.sender_relationship
        sections.append("## SENDER RELATIONSHIP")
        sections.append(
            f"Sender: {sr.sender_id} | Label: {sr.relationship_label} | "
            f"Messages: {sr.total_messages} | Reply rate: {sr.reply_frequency:.0%} | "
            f"Read rate: {sr.read_frequency:.0%} | Ignore rate: {sr.ignore_frequency:.0%}"
        )

    # 5. Business relationship.
    if retrieval.business_relationship is not None:
        br = retrieval.business_relationship
        sections.append("## BUSINESS RELATIONSHIP")
        sections.append(
            f"Business: {br.display_name or br.business_id} | "
            f"Category: {br.category} | Verified: {br.verified} | "
            f"Label: {br.relationship_label} | Trust: {br.trust_score:.2f}"
        )

    # 6. Group relationship.
    if retrieval.group_relationship is not None:
        gr = retrieval.group_relationship
        sections.append("## GROUP RELATIONSHIP")
        sections.append(
            f"Group: {gr.group_name or gr.group_id} | Type: {gr.group_type} | "
            f"Importance: {gr.importance_label} | Muted: {gr.is_muted}"
        )

    # 7. Engagement summary.
    es = retrieval.engagement_summary
    sections.append("## ENGAGEMENT SUMMARY")
    sections.append(
        f"Historical messages: {es.total_historical} | "
        f"Open rate: {es.open_rate:.0%} | Reply rate: {es.reply_rate:.0%} | "
        f"Dismiss rate: {es.dismiss_rate:.0%} | Mute rate: {es.mute_rate:.0%} | "
        f"Report rate: {es.report_rate:.0%}"
    )

    # 8. Trust / interest scores.
    sections.append("## SCORES")
    sections.append(f"Trust score: {retrieval.trust_score:.2f}")
    sections.append(f"Interest score: {retrieval.interest_score:.2f}")

    # 9. Evidence.
    if retrieval.evidence_message_ids:
        sections.append("## EVIDENCE MESSAGE IDS")
        sections.append(";".join(retrieval.evidence_message_ids))

    sections.append("")
    sections.append("Return the routing decision as JSON.")

    return "\n\n".join(sections)


def build_prompt_summary(
    context: MessageContext,
    retrieval: RetrievalResult,
    media: MediaResult | None = None,
) -> str:
    """Build a short human-readable summary of the prompt for inspection."""
    parts = [
        f"Message {context.message_id} ({context.conversation_type})",
    ]
    if context.message_text:
        parts.append(f"text={context.message_text[:80]!r}")
    if media is not None and media.has_media:
        parts.append(f"media={media.media_type}")
    if retrieval.sender_relationship:
        parts.append(f"sender={retrieval.sender_relationship.relationship_label}")
    if retrieval.business_relationship:
        parts.append(f"business={retrieval.business_relationship.relationship_label}")
    if retrieval.group_relationship:
        parts.append(f"group={retrieval.group_relationship.importance_label}")
    parts.append(f"trust={retrieval.trust_score:.2f}")
    parts.append(f"interest={retrieval.interest_score:.2f}")
    return " | ".join(parts)