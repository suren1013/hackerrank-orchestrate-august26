"""Builds structured prompts for the routing LLM.

Organizes the incoming message, conversation type, retrieval summary,
media summary, relationship summaries, engagement data, urgency, and
safety signals into readable sections—not raw JSON dumps.
"""

from __future__ import annotations

from typing import Any

from media.types import MediaResult
from models.schemas import MessageContext
from retrieval.types import RetrievalResult
from utils.helpers import safe_str

SYSTEM_PROMPT = """You are the AI Notification Routing Engine.

Your responsibility is to decide whether a WhatsApp message should:
- notify
- digest
- mute

Your primary objective is to minimize unnecessary interruptions while never missing genuinely important messages.

Use retrieval history, engagement history, media analysis, and policy signals together.

DECISION GUIDE

NOTIFY
Use notify only if:
- immediate user attention is valuable
- message is urgent
- trusted sender with important information
- family emergency
- security alert
- payment due today
- travel changes
- OTP explicitly requested by the user

DIGEST
Use digest when:
- useful
- informative
- reminder
- newsletters
- promotions from known businesses
- order updates
- routine banking
- event reminders
- messages that can wait

MUTE
Use mute when:
- scams
- phishing
- fake OTP
- spam
- malicious links
- repeated forwards
- low-value mass broadcasts
- suspicious payment requests

EXAMPLE DECISION MATRIX

Trusted + Urgent
→ notify

Trusted + Non-Urgent
→ digest

Unknown + Promotional
→ digest

Unknown + Scam Indicators
→ mute

Known Spam
→ mute

Routine Reminder
→ digest

Emergency from Family
→ notify

SCORE EXPLANATIONS

Business Verification Score
Measures sender authenticity.
It DOES NOT mean the notification deserves interruption.

Overall Notification Priority Score
Measures whether interrupting the user is worthwhile.
A verified sender may still deserve digest.

User Interest Score
Measures historical engagement.
High interest increases notify likelihood.

DO NOT

Do NOT assume every verified business deserves notify.
Do NOT confuse authenticity with urgency.
Do NOT classify advertisements as urgent.
Do NOT overreact to capital letters.
Do NOT infer emergencies that are not stated.
Do NOT invent facts.
Do NOT invent evidence.
Do NOT reference information outside the prompt.

CONFIDENCE GUIDE

0.95-1.00
Obvious decision
Policy-like certainty

0.80-0.94
Strong evidence
Little ambiguity

0.60-0.79
Moderate confidence
Mixed evidence

0.40-0.59
Weak evidence
Many conflicting signals

Below 0.40
Very uncertain
Use only when evidence is poor.

OUTPUT RULES

Return ONLY valid JSON.
Never output Markdown.
Never explain reasoning.
Never output code fences.
Never output comments.

Return exactly:
{
  "action":"",
  "message_type":"",
  "reason":"",
  "confidence":0.0
}

Allowed values:
- action: notify | digest | mute
- message_type: personal | urgent | event | payment | business_update | promotion | greeting | forward | spam | scam | unknown

FEW-SHOT EXAMPLES

Example 1
Verified bank payment reminder due tomorrow
→ digest

Example 2
Mother asking to call immediately
→ notify

Example 3
OTP + PIN + payment link
→ mute

Example 4
Movie promotion
→ digest

Example 5
Group announcement about office fire drill today
→ notify

Example 6
Repeated forwarded greeting
→ mute
"""


def build_user_prompt(
    context: MessageContext,
    retrieval: RetrievalResult,
    media: MediaResult | None = None,
) -> str:
    """Build a structured user prompt for the routing LLM."""
    sections: list[str] = []

    # 1. USER PROFILE (from users.csv).
    user = context.user or {}
    sections.append("USER PROFILE")
    sections.append("-" * 12)
    dnd = safe_str(user.get("do_not_disturb_window")) or "Unknown"
    sections.append(f"Do Not Disturb: {dnd}")
    sections.append(
        f"Messages opened: {user.get('messages_opened_30d', '?')}"
    )
    sections.append(f"Replies: {user.get('messages_replied_30d', '?')}")
    sections.append(
        f"Dismissed notifications: {user.get('notifications_dismissed_30d', '?')}"
    )
    sections.append(f"Reported messages: {user.get('messages_reported_30d', '?')}")

    # 2. SENDER/BUSINESS/GROUP RELATIONSHIP context.
    if retrieval.business_relationship is not None:
        br = retrieval.business_relationship
        sections.append("BUSINESS CONTEXT")
        sections.append("-" * 16)
        sections.append(f"Business: {br.display_name or br.business_id}")
        sections.append(f"Category: {br.category}")
        sections.append(f"Verified Business: {'Yes' if br.verified else 'No'}")
        sections.append(f"Business Verification Score: {br.trust_score:.2f}")
        sections.append(f"Relationship: {br.relationship_label}")
        sections.append(f"Promotions Allowed: {'Yes' if br.allows_promotions else 'No'}")

    if retrieval.sender_relationship is not None:
        sr = retrieval.sender_relationship
        sections.append("SENDER CONTEXT")
        sections.append("-" * 14)
        sections.append(f"Sender: {sr.sender_id}")
        sections.append(f"Relationship: {sr.relationship_label}")
        sections.append(f"Historical Messages: {sr.total_messages}")
        sections.append(f"Reply Rate: {sr.reply_frequency:.0%}")
        sections.append(f"Read Rate: {sr.read_frequency:.0%}")
        sections.append(f"Ignore Rate: {sr.ignore_frequency:.0%}")

    if retrieval.group_relationship is not None:
        gr = retrieval.group_relationship
        sections.append("GROUP CONTEXT")
        sections.append("-" * 13)
        sections.append(f"Group: {gr.group_name or gr.group_id}")
        sections.append(f"Type: {gr.group_type}")
        sections.append(f"Importance: {gr.importance_label}")
        sections.append(f"Muted: {'Yes' if gr.is_muted else 'No'}")

    # 3. USER PRIORITY SIGNALS.
    sections.append("USER PRIORITY SIGNALS")
    sections.append("-" * 21)
    sections.append(f"Overall Notification Priority Score: {retrieval.trust_score:.2f}")
    sections.append(f"User Interest Score: {retrieval.interest_score:.2f}")

    # 4. CURRENT MESSAGE.
    sections.append("CURRENT MESSAGE")
    sections.append("-" * 15)
    sections.append(f"Message ID: {context.message_id}")
    sections.append(f"Conversation type: {context.conversation_type}")
    sections.append(f"Created at: {context.created_at}")
    sections.append(
        f"Media: {safe_str(context.media_type) or 'none'}"
    )
    sections.append(
        f"Forwarded: {'Yes' if context.forwarded_count > 0 else 'No'} "
        f"({context.forwarded_count} times)"
    )
    if context.message_text:
        sections.append(f"Text: {context.message_text}")

    # 5. MEDIA CONTENT.
    if media is not None and media.has_media:
        sections.append("MEDIA CONTENT")
        sections.append("-" * 13)
        sections.append(f"Media type: {media.media_type}")
        if media.summary:
            sections.append(f"Summary: {media.summary}")
        if media.entities:
            for key, values in media.entities.items():
                if values:
                    sections.append(f"{key}: {', '.join(values[:5])}")
        if media.urgency_indicators:
            sections.append(
                f"Urgency indicators: {', '.join(media.urgency_indicators)}"
            )
        if media.safety_indicators:
            sections.append(
                f"Safety indicators: {', '.join(media.safety_indicators)}"
            )

    # 6. RETRIEVAL SUMMARY (natural language).
    sections.append("RETRIEVAL SUMMARY")
    sections.append("-" * 17)
    sections.append(retrieval.retrieval_summary)

    # 7. SIMILAR HISTORY (evidence).
    if retrieval.top_similar_messages:
        sections.append("SIMILAR HISTORY")
        sections.append("-" * 15)
        for sm in retrieval.top_similar_messages:
            hist = sm.message or {}
            hist_date = ""
            if hist.get("created_at"):
                hist_date = str(hist.get("created_at"))[:10]
            sections.append(
                f"- {sm.message_id} ({hist_date}): "
                f"{safe_str(hist.get('message_text'))[:120]}"
            )

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
    parts.append(
        f"priority={retrieval.trust_score:.2f}"
    )
    parts.append(f"interest={retrieval.interest_score:.2f}")
    return " | ".join(parts)