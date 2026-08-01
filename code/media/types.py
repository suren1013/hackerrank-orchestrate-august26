"""Typed schemas for the multimodal media processing layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MediaResult:
    """Output of processing an image or voice message.

    Attributes
    ----------
    message_id : str
        Incoming message ID this media belongs to ('' if unknown).
    media_type : str
        One of ``image``, ``voice``, or ``none``.
    media_id : str
        Linked image or voice-note ID, if present.
    extracted_text : str
        OCR / ASR output text.
    summary : str
        Concise summary (LLM summary if available, else OCR/ASR output).
    entities : dict[str, list[str]]
        Detected dates, times, money, links, phone numbers.
    urgency_indicators : list[str]
        Urgency signals (e.g. deadline words).
    safety_indicators : list[str]
        Safety signals (e.g. OTP request, phishing links).
    confidence : float
        Confidence 0..1 in the extraction (empty media = 0.0).
    error : str | None
        Error message if processing failed.
    """

    message_id: str = ""
    media_type: str = "none"
    media_id: str = ""
    extracted_text: str = ""
    summary: str = ""
    entities: dict[str, list[str]] = field(default_factory=dict)
    urgency_indicators: list[str] = field(default_factory=list)
    safety_indicators: list[str] = field(default_factory=list)
    confidence: float = 0.0
    error: str | None = None

    @property
    def has_media(self) -> bool:
        return self.media_type in {"image", "voice"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "media_type": self.media_type,
            "media_id": self.media_id,
            "extracted_text": self.extracted_text,
            "summary": self.summary,
            "entities": self.entities,
            "urgency_indicators": self.urgency_indicators,
            "safety_indicators": self.safety_indicators,
            "confidence": round(self.confidence, 4),
            "error": self.error,
        }


def empty_media_result(message_id: str = "") -> MediaResult:
    """Return an empty MediaResult for messages without media."""
    return MediaResult(message_id=message_id, media_type="none", media_id="")


def media_result_from_error(
    message_id: str, media_type: str, media_id: str, error: str
) -> MediaResult:
    """Return a MediaResult populated with an error message."""
    return MediaResult(
        message_id=message_id,
        media_type=media_type,
        media_id=media_id,
        error=error,
    )