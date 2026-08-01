"""Deterministic policy rules.

Each rule is a pure function that evaluates a MessageContext,
RetrievalResult, and MediaResult and returns a PolicyDecision or None.

Rules are ordered by priority (highest first) in the engine. Rules here
follow a consistent signature so they are easy to extend.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from media.types import MediaResult
from models.schemas import MessageContext
from policy.types import PolicyDecision
from retrieval.types import RetrievalResult
from utils.helpers import safe_str

# Rule signature: (context, retrieval, media) -> PolicyDecision | None
RuleFn = Callable[
    [MessageContext, RetrievalResult, MediaResult], PolicyDecision | None
]

# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _get_evidence(retrieval: RetrievalResult) -> list[str]:
    return list(retrieval.evidence_message_ids)


def _has_safety(safety: list[str], *labels: str) -> bool:
    return any(s in labels for s in safety)


# ---------------------------------------------------------------------------
# Rule 1: Obvious scam detection
# ---------------------------------------------------------------------------

SCAM_TRIGGERS = {
    "otp_request",
    "pin_request",
    "account_block",
    "phishing_links",
    "qr_payment_prompt",
    "pay_now",
}

HIGH_RISK_SENDER_LABELS = {"new_sender", "ignored", "casual"}


def rule_scam_detection(
    context: MessageContext,
    retrieval: RetrievalResult,
    media: MediaResult,
) -> PolicyDecision | None:
    """Mute obvious scams based on safety indicators + low sender trust.

    Fires when:
    - Media/text contains high-risk safety indicators (OTP, PIN, phishing,
      QR payment, pay now, account block), AND
    - The sender/business trust is low (trust_score < 0.4), OR the sender
      is new/unknown.
    """
    safety = media.safety_indicators if media else []

    if not _has_safety(safety, *SCAM_TRIGGERS):
        return None

    # Determine sender risk.
    sender_label = (
        retrieval.sender_relationship.relationship_label
        if retrieval.sender_relationship
        else None
    )
    sender_risky = (
        sender_label in HIGH_RISK_SENDER_LABELS
        and sender_label is not None
    )

    business_risky = False
    if retrieval.business_relationship is not None:
        br = retrieval.business_relationship
        business_risky = (
            not br.verified
            and br.trust_score < 0.5
        ) or br.relationship_label == "unfamiliar_business"

    trust_low = retrieval.trust_score < 0.4

    if not (sender_risky or business_risky or trust_low):
        return None

    message_type = "scam" if _has_safety(
        safety, "otp_request", "pin_request", "phishing_links", "qr_payment_prompt"
    ) else "spam"

    return PolicyDecision(
        message_id=context.message_id,
        action="mute",
        message_type=message_type,
        reason=(
            f"Policy detected high-risk safety signals ({', '.join(sorted(safety))}) "
            f"with low sender trust ({retrieval.trust_score:.2f})."
        ),
        confidence=0.92,
        evidence_message_ids=_get_evidence(retrieval),
        rule_name="scam_detection",
        priority=100,
    )


# ---------------------------------------------------------------------------
# Rule 2: Trusted urgent messages
# ---------------------------------------------------------------------------

TRUSTED_FAMILY_TYPES = {"family", "extended_family"}
TRUSTED_ADMIN_GROUPS = {"society", "school_group", "college_faculty", "safety"}
TRUSTED_BANK_CATEGORIES = {"bank", "credit_card", "payments", "fintech"}
URGENCY_KEYWORDS = re.compile(
    r"\b(?:urgent|immediately|asap|emergency|deadline|"
    r"closes? (?:today|tonight|now)|before \d{1,2}(?::\d{2})?(?: ?(?:am|pm))?|"
    r"last (?:minute|day|chance)|right now|call me now)\b",
    re.IGNORECASE,
)


def rule_trusted_urgent(
    context: MessageContext,
    retrieval: RetrievalResult,
    media: MediaResult,
) -> PolicyDecision | None:
    """Notify for trusted urgent messages.

    Fires when:
    - The message contains urgency keywords, AND
    - The sender/group/business is trusted:
      * Group is family/extended_family with trusted admin, OR
      * Group is a society/school/safety announcement group with admin sender, OR
      * Business is a verified bank/fintech the user actively uses
    """
    text = safe_str(context.message_text)
    if not URGENCY_KEYWORDS.search(text):
        return None

    trusted = False
    reason_ctx = ""

    # Group trust.
    gr = retrieval.group_relationship
    if gr is not None:
        if gr.group_type in TRUSTED_FAMILY_TYPES and gr.is_active:
            trusted = True
            reason_ctx = f"trusted family group {gr.group_name or gr.group_id}"
        elif (
            gr.group_type in TRUSTED_ADMIN_GROUPS
            and gr.is_admin
        ):
            trusted = True
            reason_ctx = f"trusted admin of {gr.group_name or gr.group_id}"

    # Business trust (verified bank/fintech the user actively uses).
    br = retrieval.business_relationship
    if br is not None:
        if (
            br.verified
            and br.category in TRUSTED_BANK_CATEGORIES
            and br.is_active_customer
        ):
            trusted = True
            reason_ctx = f"verified {br.category} {br.display_name or br.business_id}"

    # Sender trust (close contact with high reply/read rates).
    sr = retrieval.sender_relationship
    if sr is not None:
        if sr.relationship_label == "highly_engaged" and sr.reply_frequency >= 0.5:
            trusted = True
            reason_ctx = f"highly-engaged sender {sr.sender_id}"

    if not trusted:
        return None

    return PolicyDecision(
        message_id=context.message_id,
        action="notify",
        message_type="urgent",
        reason=f"Policy classified as trusted urgent message from {reason_ctx}.",
        confidence=0.88,
        evidence_message_ids=_get_evidence(retrieval),
        rule_name="trusted_urgent",
        priority=90,
    )


# ---------------------------------------------------------------------------
# Rule 3: Malformed media / extraction failure
# ---------------------------------------------------------------------------

def rule_media_failure(
    context: MessageContext,
    retrieval: RetrievalResult,
    media: MediaResult,
) -> PolicyDecision | None:
    """Handle malformed media or extraction failures.

    Fires when:
    - The message has media (image/voice) but the MediaResult has an error
      or empty extraction with zero confidence.

    Decision: digest by default (do not interrupt, do not mute aggressively).
    """
    if not media.has_media:
        return None

    extraction_failed = media.error is not None or (
        not media.extracted_text.strip() and media.confidence == 0.0
    )

    if not extraction_failed:
        return None

    return PolicyDecision(
        message_id=context.message_id,
        action="digest",
        message_type="unknown",
        reason=(
            f"Media extraction failed ({media.error or 'no text detected'}); "
            "routing via policy digest."
        ),
        confidence=0.55,
        evidence_message_ids=_get_evidence(retrieval),
        rule_name="media_failure",
        priority=80,
    )


# ---------------------------------------------------------------------------
# Rule 4: Highly repetitive spam
# ---------------------------------------------------------------------------

def rule_repetitive_spam(
    context: MessageContext,
    retrieval: RetrievalResult,
    media: MediaResult,
) -> PolicyDecision | None:
    """Mute highly repetitive spam.

    Fires when:
    - The incoming message is heavily forwarded (forwarded_count >= threshold),
      AND the sender's historical messages were mostly ignored/dismissed/muted.
    """
    forwarded_count = context.forwarded_count
    if forwarded_count < 5:
        return None

    sr = retrieval.sender_relationship
    if sr is None:
        return None

    if sr.total_messages == 0:
        return None

    # The sender's messages were largely ignored / muted / archived.
    ignored_ratio = (
        sr.ignore_frequency + sr.archive_frequency
    ) / 2.0
    if ignored_ratio < 0.5:
        return None

    return PolicyDecision(
        message_id=context.message_id,
        action="mute",
        message_type="forward",
        reason=(
            f"Highly forwarded message ({forwarded_count} forwards) from a sender "
            f"the user mostly ignores (ignore+archive rate {ignored_ratio:.0%})."
        ),
        confidence=0.82,
        evidence_message_ids=_get_evidence(retrieval),
        rule_name="repetitive_spam",
        priority=70,
    )