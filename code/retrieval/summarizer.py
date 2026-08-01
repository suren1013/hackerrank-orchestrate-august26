"""Natural-language retrieval summary generator.

Produces a concise 2-4 sentence summary of the user's historical relationship
with the sender/business/group. This summary is intended for the LLM prompt
in Phase 4.
"""

from __future__ import annotations

from retrieval.types import (
    BusinessRelationship,
    EngagementSummary,
    GroupRelationship,
    SenderRelationship,
)


def build_retrieval_summary(
    sender: SenderRelationship | None,
    business: BusinessRelationship | None,
    group: GroupRelationship | None,
    engagement: EngagementSummary,
) -> str:
    """Build a 2-4 sentence natural-language summary."""
    parts: list[str] = []

    # Sender relationship.
    if sender is not None:
        if sender.total_messages == 0:
            parts.append(
                f"The sender {sender.sender_id} has no prior message history with this user."
            )
        else:
            parts.append(
                f"The sender {sender.sender_id} has sent {sender.total_messages} "
                f"historical messages; the user replied to {sender.reply_count} "
                f"({sender.reply_frequency:.0%}) and read {sender.read_count} "
                f"({sender.read_frequency:.0%})."
            )

    # Business relationship.
    if business is not None:
        if business.is_known_business:
            parts.append(
                f"{business.display_name or business.business_id} is a "
                f"{'verified' if business.verified else 'unverified'} "
                f"{business.category} business the user has interacted with "
                f"({business.relationship_label})."
            )
        else:
            parts.append(
                f"{business.display_name or business.business_id} is an "
                f"unfamiliar business with no prior user history."
            )

    # Group relationship.
    if group is not None:
        if group.is_member:
            parts.append(
                f"The user is a member of {group.group_name or group.group_id} "
                f"({group.group_type}); the group is {group.importance_label}."
            )
        else:
            parts.append(
                f"The user is not a member of {group.group_name or group.group_id}."
            )

    # Engagement summary.
    if engagement.total_historical > 0:
        parts.append(
            f"Across {engagement.total_historical} historical messages, the user "
            f"opened {engagement.open_rate:.0%}, replied to {engagement.reply_rate:.0%}, "
            f"dismissed {engagement.dismiss_rate:.0%}, muted {engagement.mute_rate:.0%}, "
            f"and reported {engagement.report_rate:.0%}."
        )
    else:
        parts.append("No historical engagement data is available for this user.")

    return " ".join(parts)