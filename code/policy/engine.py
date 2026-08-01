"""PolicyEngine: runs deterministic rules before the LLM.

Evaluates MessageContext, RetrievalResult, and MediaResult. If a rule
fires, returns a PolicyDecision and the LLM is skipped.

The engine also analyzes the incoming message text for safety indicators
when no media is present, so scam detection works on plain-text messages.
"""

from __future__ import annotations

from media.types import MediaResult, empty_media_result
from media.text_analysis import analyze_text
from models.schemas import MessageContext
from policy.rules import RuleFn, rule_media_failure, rule_repetitive_spam, rule_scam_detection, rule_trusted_urgent
from policy.types import PolicyDecision
from retrieval.types import RetrievalResult
from utils.helpers import safe_str
from utils.logger import get_logger

logger = get_logger(__name__)

# Default rule set, ordered by priority (highest first).
DEFAULT_RULES: list[RuleFn] = [
    rule_scam_detection,
    rule_trusted_urgent,
    rule_media_failure,
    rule_repetitive_spam,
]


class PolicyEngine:
    """Deterministic policy engine that runs before the LLM."""

    def __init__(self, rules: list[RuleFn] | None = None) -> None:
        self.rules = rules if rules is not None else list(DEFAULT_RULES)
        logger.info("PolicyEngine initialized with %d rules.", len(self.rules))

    def evaluate(
        self,
        context: MessageContext,
        retrieval: RetrievalResult,
        media: MediaResult | None = None,
    ) -> PolicyDecision | None:
        """Evaluate all rules in priority order.

        Parameters
        ----------
        context : MessageContext
            Fully assembled message context.
        retrieval : RetrievalResult
            Retrieval result from the RetrievalEngine.
        media : MediaResult | None
            Media result from the MediaProcessor. If None, the message text
            is analyzed for safety indicators so scam rules still work on
            plain-text messages.

        Returns
        -------
        PolicyDecision | None
            First firing decision, or None if no rule applies.
        """
        # If media is None OR the message has no real media, build a result
        # enriched with text analysis of the incoming message so safety
        # indicators from plain text still feed the rules.
        if media is None or not media.has_media:
            media = self._build_text_media(context)

        for rule in self.rules:
            decision = rule(context, retrieval, media)
            if decision is not None:
                logger.info(
                    "Policy rule '%s' fired for %s -> %s/%s",
                    decision.rule_name,
                    context.message_id,
                    decision.action,
                    decision.message_type,
                )
                return decision

        logger.info("No policy rule fired for %s.", context.message_id)
        return None

    def _build_text_media(self, context: MessageContext) -> MediaResult:
        """Build a MediaResult with text-based safety/urgency analysis."""
        message_text = safe_str(context.message_text)
        if not message_text:
            return empty_media_result(context.message_id)

        analysis = analyze_text(message_text)
        return MediaResult(
            message_id=context.message_id,
            media_type="none",
            extracted_text=message_text,
            summary="",
            entities=analysis["entities"],
            urgency_indicators=analysis["urgency_indicators"],
            safety_indicators=analysis["safety_indicators"],
            confidence=0.0,
        )