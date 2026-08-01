"""Natural-language retrieval summary generator.

Produces a concise 4-6 sentence summary of the user's historical relationship
with the sender/business/group using natural language, not dense percentages.

This summary is intended for the LLM prompt in Phase 4.
"""

from __future__ import annotations

from typing import Any

from models.schemas import MessageContext
from retrieval.types import (
    BusinessRelationship,
    EngagementSummary,
    GroupRelationship,
    SenderRelationship,
)
from utils.helpers import safe_str


def build_retrieval_summary(
    sender: SenderRelationship | None,
    business: BusinessRelationship | None,
    group: GroupRelationship | None,
    engagement: EngagementSummary,
    context: MessageContext | None = None,
) -> str:
    """Build a natural-language summary of the user's historical relationship.

    Parameters
    ----------
    sender : SenderRelationship | None
        Sender relationship summary.
    business : BusinessRelationship | None
        Business relationship summary.
    group : GroupRelationship | None
        Group relationship summary.
    engagement : EngagementSummary
        Aggregate engagement metrics.
    context : MessageContext | None
        Full message context (used to count similar historical messages and
        find the most recent matching message).
    """
    parts: list[str] = []

    # ------------------------------------------------------------------
    # Business relationship (natural language).
    # ------------------------------------------------------------------
    if business is not None:
        biz_msgs = _matching_history(
            context,
            business_id=business.business_id,
        )
        parts.append(
            _business_narrative(business, biz_msgs, context)
        )

    # ------------------------------------------------------------------
    # Sender relationship (natural language).
    # ------------------------------------------------------------------
    if sender is not None:
        sender_msgs = _matching_history(
            context,
            sender_id=sender.sender_id,
        )
        parts.append(
            _sender_narrative(sender, sender_msgs, context)
        )

    # ------------------------------------------------------------------
    # Group relationship.
    # ------------------------------------------------------------------
    if group is not None:
        group_msgs = _matching_history(
            context,
            group_id=group.group_id,
        )
        parts.append(
            _group_narrative(group, group_msgs)
        )

    # ------------------------------------------------------------------
    # Engagement summary (natural language).
    # ------------------------------------------------------------------
    parts.append(_engagement_narrative(engagement, context))

    # Filter empty sentences and return.
    sentences = [p for p in parts if p.strip()]
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Narrative builders
# ---------------------------------------------------------------------------


def _business_narrative(
    business: BusinessRelationship,
    biz_msgs: list[dict[str, Any]],
    context: MessageContext | None,
) -> str:
    """Build a natural-language sentence for the business relationship."""
    name = business.display_name or business.business_id

    if not business.is_known_business:
        return f"{name} is an unfamiliar business with no prior user history."

    # Relationship label → natural phrasing.
    if business.relationship_label == "trusted_active_customer":
        rel_phrase = "a trusted active customer"
    elif business.relationship_label == "active_customer":
        rel_phrase = "an active customer"
    elif business.relationship_label == "promotional_subscriber":
        rel_phrase = "a promotional subscriber"
    elif business.relationship_label == "opted_out":
        rel_phrase = "opted out of"
    elif business.relationship_label == "known_business":
        rel_phrase = "a known customer"
    else:
        rel_phrase = business.relationship_label.replace("_", " ")

    parts = [f"The user is {rel_phrase} of {name}."]

    if biz_msgs:
        parts.append(f"Previously received {len(biz_msgs)} messages from {name}.")

        # Count opened/replied among these messages.
        opened = 0
        replied = 0
        events = _events_by_message(context)
        for m in biz_msgs:
            ev = events.get(safe_str(m.get("message_id")), {})
            if ev.get("message_opened") == 1:
                opened += 1
            if ev.get("message_replied") == 1:
                replied += 1
        if opened:
            parts.append(f"Opened {opened} of them.")
        if replied:
            parts.append(f"Replied to {replied}.")

        # Most recent matching message date.
        latest = _latest_date(biz_msgs)
        if latest:
            parts.append(f"The most recent {name} message was on {latest}.")

    return " ".join(parts)


def _sender_narrative(
    sender: SenderRelationship,
    sender_msgs: list[dict[str, Any]],
    context: MessageContext | None,
) -> str:
    """Build a natural-language sentence for the sender relationship."""
    if sender.total_messages == 0:
        return f"The sender {sender.sender_id} has no prior message history with this user."

    parts = [
        f"The sender {sender.sender_id} has sent {sender.total_messages} historical messages."
    ]
    if sender.reply_count:
        parts.append(f"The user replied to {sender.reply_count} of them.")
    if sender.read_count:
        parts.append(f"Read {sender.read_count}.")

    # Count distinct messages with any negative engagement (dismissed/muted).
    # A single message can have both dismissed and muted events, so use
    # a cap to avoid implying more ignored messages than were received.
    if sender.total_messages > 0:
        neg_events = sender.ignore_count + sender.archive_count
        ignored = min(neg_events, sender.total_messages)
        if ignored > 0:
            parts.append(
                f"About {ignored} of them were ignored, dismissed, or muted by the user."
            )

    return " ".join(parts)


def _group_narrative(
    group: GroupRelationship,
    group_msgs: list[dict[str, Any]],
) -> str:
    """Build a natural-language sentence for the group relationship."""
    name = group.group_name or group.group_id

    if not group.is_member:
        return f"The user is not a member of {name}."

    parts = [f"The user is a member of {name} ({group.group_type})."]
    parts.append(f"The group is {group.importance_label}.")

    if group.is_muted:
        parts.append("The user has muted this group.")

    if group_msgs:
        parts.append(f"Previously received {len(group_msgs)} messages from this group.")

    return " ".join(parts)


def _engagement_narrative(
    engagement: EngagementSummary,
    context: MessageContext | None,
) -> str:
    """Build a natural-language sentence for aggregate engagement."""
    if engagement.total_historical == 0:
        return "No historical engagement data is available for this user."

    parts = [
        f"Across {engagement.total_historical} historical messages, "
        f"the user opened {engagement.opened}."
    ]
    if engagement.replied:
        parts.append(f"Replied to {engagement.replied}.")
    if engagement.dismissed:
        parts.append(f"Dismissed {engagement.dismissed} notifications.")
    if engagement.muted:
        parts.append(f"Muted {engagement.muted}.")
    if engagement.reported:
        parts.append(
            f"Reported {engagement.reported} suspicious messages."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matching_history(
    context: MessageContext | None,
    business_id: str | None = None,
    sender_id: str | None = None,
    group_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return historical messages matching the given identifiers."""
    if context is None:
        return []
    matches = []
    for m in context.historical_messages:
        if business_id and safe_str(m.get("business_id")) == business_id:
            matches.append(m)
        elif sender_id and safe_str(m.get("sender_user_id")) == sender_id:
            matches.append(m)
        elif group_id and safe_str(m.get("group_id")) == group_id:
            matches.append(m)
    return matches


def _events_by_message(
    context: MessageContext | None,
) -> dict[str, dict[str, Any]]:
    """Build a map of message_id -> event dict from the context."""
    if context is None:
        return {}
    events: dict[str, dict[str, Any]] = {}
    for ev in context.historical_events:
        mid = safe_str(ev.get("message_id"))
        if mid:
            events[mid] = ev
    return events


def _latest_date(messages: list[dict[str, Any]]) -> str:
    """Return the most recent created_at date (YYYY-MM-DD) among messages."""
    dates = []
    for m in messages:
        ts = m.get("created_at")
        if ts is not None:
            dates.append(str(ts)[:10])
    return max(dates) if dates else ""