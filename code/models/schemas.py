"""Typed schemas for the unified message context.

These dataclasses describe the shape of the fully-assembled context that the
DataLoader produces for each incoming message. They are intentionally
lightweight — no routing logic lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MessageContext:
    """A single incoming message joined with all available context.

    Attributes
    ----------
    message_id : str
        Unique incoming message ID.
    user_id : str
        User receiving the message.
    conversation_type : str
        One of ``personal``, ``group``, or ``business``.
    created_at : str
        ISO timestamp of the message.
    message_text : str
        Text content (empty for voice-note messages).
    media_type : str
        Empty, ``image``, or ``voice``.
    media_id : str
        Linked image or voice-note ID, if present.
    forwarded_count : int
        Forwarding signal.
    user : dict[str, Any] | None
        Row from users.csv for the receiving user.
    group : dict[str, Any] | None
        Row from groups.csv if the message is from a group.
    group_member : dict[str, Any] | None
        Row from group_members.csv for (user, group) if present.
    business : dict[str, Any] | None
        Row from business_accounts.csv if the message is from a business.
    user_business : dict[str, Any] | None
        Row from user_business_history.csv for (user, business) if present.
    image : dict[str, Any] | None
        Row from images.csv if the message has an image.
    voice_note : dict[str, Any] | None
        Row from voice_notes.csv if the message has a voice note.
    daily_notification : dict[str, Any] | None
        Row from daily_notification_summary.csv for the user on the message date.
    historical_messages : list[dict[str, Any]]
        Past messages for the same user/conversation/sender (from message_history.csv).
    historical_events : list[dict[str, Any]]
        User reactions to historical messages (from message_events.csv).
    """

    message_id: str
    user_id: str
    conversation_type: str
    created_at: str
    message_text: str
    media_type: str
    media_id: str
    forwarded_count: int
    group_id: str = ""
    business_id: str = ""
    sender_user_id: str = ""

    user: dict[str, Any] | None = None
    group: dict[str, Any] | None = None
    group_member: dict[str, Any] | None = None
    business: dict[str, Any] | None = None
    user_business: dict[str, Any] | None = None
    image: dict[str, Any] | None = None
    voice_note: dict[str, Any] | None = None
    daily_notification: dict[str, Any] | None = None

    historical_messages: list[dict[str, Any]] = field(default_factory=list)
    historical_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation (useful for inspection/logging)."""
        return {
            "message_id": self.message_id,
            "user_id": self.user_id,
            "conversation_type": self.conversation_type,
            "group_id": self.group_id,
            "business_id": self.business_id,
            "sender_user_id": self.sender_user_id,
            "created_at": self.created_at,
            "message_text": self.message_text,
            "media_type": self.media_type,
            "media_id": self.media_id,
            "forwarded_count": self.forwarded_count,
            "user": self.user,
            "group": self.group,
            "group_member": self.group_member,
            "business": self.business,
            "user_business": self.user_business,
            "image": self.image,
            "voice_note": self.voice_note,
            "daily_notification": self.daily_notification,
            "historical_messages": self.historical_messages,
            "historical_events": self.historical_events,
        }