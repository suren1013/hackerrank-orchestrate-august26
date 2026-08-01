"""Validates LLM routing output.

Checks that the LLM response conforms to the expected schema:
- action ∈ {notify, digest, mute}
- message_type ∈ allowed set
- confidence ∈ [0, 1]
- reason is non-empty and within a reasonable length
- required fields present

Malformed outputs are rejected and a fallback decision is returned.
"""

from __future__ import annotations

from llm.schemas import LLMResponse, VALID_ACTIONS, VALID_MESSAGE_TYPES
from utils.logger import get_logger

logger = get_logger(__name__)

MAX_REASON_LENGTH = 500
MIN_REASON_LENGTH = 5


class ValidationError(Exception):
    """Raised when an LLM response fails validation."""


def validate_response(response: LLMResponse) -> LLMResponse:
    """Validate an LLMResponse and return a cleaned version.

    Raises
    ------
    ValidationError
        If the response is malformed or missing required fields.
    """
    errors: list[str] = []

    # Required fields.
    if not response.action:
        errors.append("missing action")
    if not response.message_type:
        errors.append("missing message_type")
    if not response.reason:
        errors.append("missing reason")

    # Action must be valid.
    if response.action and response.action not in VALID_ACTIONS:
        errors.append(f"invalid action '{response.action}'")

    # Message type must be valid.
    if response.message_type and response.message_type not in VALID_MESSAGE_TYPES:
        errors.append(f"invalid message_type '{response.message_type}'")

    # Confidence must be a number in [0, 1].
    try:
        conf = float(response.confidence)
        if not (0.0 <= conf <= 1.0):
            errors.append(f"confidence {conf} out of range [0,1]")
    except (TypeError, ValueError):
        errors.append("confidence is not a number")

    # Reason length.
    reason_len = len(response.reason.strip())
    if reason_len < MIN_REASON_LENGTH:
        errors.append("reason too short")
    if reason_len > MAX_REASON_LENGTH:
        errors.append("reason too long")

    if errors:
        raise ValidationError("; ".join(errors))

    # Clean the response.
    return LLMResponse(
        action=response.action,
        message_type=response.message_type,
        reason=response.reason.strip(),
        confidence=float(response.confidence),
        raw=response.raw,
    )


def fallback_response(message_id: str, error: str) -> LLMResponse:
    """Return a safe fallback decision when validation fails."""
    logger.warning("Validation failed for %s: %s", message_id, error)
    return LLMResponse(
        action="digest",
        message_type="unknown",
        reason="Routing decision could not be validated; defaulting to digest.",
        confidence=0.5,
        raw="",
    )