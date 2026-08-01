"""Typed schemas for the retrieval layer.

These dataclasses describe the output of the RetrievalEngine: a compact
evidence package that the router (Phase 4) can consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SenderRelationship:
    """Summary of the receiver's historical interaction with a sender."""

    sender_id: str
    total_messages: int = 0
    reply_count: int = 0
    read_count: int = 0
    ignore_count: int = 0
    archive_count: int = 0
    reply_frequency: float = 0.0
    read_frequency: float = 0.0
    ignore_frequency: float = 0.0
    archive_frequency: float = 0.0
    relationship_label: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender_id": self.sender_id,
            "total_messages": self.total_messages,
            "reply_count": self.reply_count,
            "read_count": self.read_count,
            "ignore_count": self.ignore_count,
            "archive_count": self.archive_count,
            "reply_frequency": round(self.reply_frequency, 4),
            "read_frequency": round(self.read_frequency, 4),
            "ignore_frequency": round(self.ignore_frequency, 4),
            "archive_frequency": round(self.archive_frequency, 4),
            "relationship_label": self.relationship_label,
        }


@dataclass
class BusinessRelationship:
    """Summary of the receiver's relationship with a business sender."""

    business_id: str
    display_name: str = ""
    verified: bool = False
    category: str = ""
    is_known_business: bool = False
    is_active_customer: bool = False
    has_frequent_interactions: bool = False
    allows_promotions: bool = False
    opted_out: bool = False
    trust_score: float = 0.0
    relationship_label: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "business_id": self.business_id,
            "display_name": self.display_name,
            "verified": self.verified,
            "category": self.category,
            "is_known_business": self.is_known_business,
            "is_active_customer": self.is_active_customer,
            "has_frequent_interactions": self.has_frequent_interactions,
            "allows_promotions": self.allows_promotions,
            "opted_out": self.opted_out,
            "trust_score": round(self.trust_score, 4),
            "relationship_label": self.relationship_label,
        }


@dataclass
class GroupRelationship:
    """Summary of the receiver's relationship with a group."""

    group_id: str
    group_name: str = ""
    group_type: str = ""
    is_member: bool = False
    is_admin: bool = False
    is_muted: bool = False
    is_active: bool = False
    is_announcement_group: bool = False
    importance_label: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "group_name": self.group_name,
            "group_type": self.group_type,
            "is_member": self.is_member,
            "is_admin": self.is_admin,
            "is_muted": self.is_muted,
            "is_active": self.is_active,
            "is_announcement_group": self.is_announcement_group,
            "importance_label": self.importance_label,
        }


@dataclass
class EngagementSummary:
    """Aggregate engagement metrics for the user's historical messages."""

    total_historical: int = 0
    opened: int = 0
    replied: int = 0
    dismissed: int = 0
    muted: int = 0
    reported: int = 0
    open_rate: float = 0.0
    reply_rate: float = 0.0
    dismiss_rate: float = 0.0
    mute_rate: float = 0.0
    report_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_historical": self.total_historical,
            "opened": self.opened,
            "replied": self.replied,
            "dismissed": self.dismissed,
            "muted": self.muted,
            "reported": self.reported,
            "open_rate": round(self.open_rate, 4),
            "reply_rate": round(self.reply_rate, 4),
            "dismiss_rate": round(self.dismiss_rate, 4),
            "mute_rate": round(self.mute_rate, 4),
            "report_rate": round(self.report_rate, 4),
        }


@dataclass
class SimilarMessage:
    """A historical message ranked by relevance to the incoming message."""

    message_id: str
    score: float
    reason: str
    message: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "score": round(self.score, 4),
            "reason": self.reason,
            "message": self.message,
        }


@dataclass
class RetrievalResult:
    """Compact evidence package produced by the RetrievalEngine."""

    message_id: str
    evidence_message_ids: list[str]
    top_similar_messages: list[SimilarMessage]
    sender_relationship: SenderRelationship | None
    business_relationship: BusinessRelationship | None
    group_relationship: GroupRelationship | None
    engagement_summary: EngagementSummary
    trust_score: float
    interest_score: float
    retrieval_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "evidence_message_ids": self.evidence_message_ids,
            "top_similar_messages": [m.to_dict() for m in self.top_similar_messages],
            "sender_relationship": (
                self.sender_relationship.to_dict()
                if self.sender_relationship
                else None
            ),
            "business_relationship": (
                self.business_relationship.to_dict()
                if self.business_relationship
                else None
            ),
            "group_relationship": (
                self.group_relationship.to_dict()
                if self.group_relationship
                else None
            ),
            "engagement_summary": self.engagement_summary.to_dict(),
            "trust_score": round(self.trust_score, 4),
            "interest_score": round(self.interest_score, 4),
            "retrieval_summary": self.retrieval_summary,
        }