"""AI Notification Router: produces the final routing decision.

Pipeline:
1. PolicyEngine evaluates deterministic rules (scam detection, trusted
   urgent, media failures, repetitive spam). If a rule fires, the LLM is
   skipped entirely and the policy decision becomes the final decision.
2. Only ambiguous messages (no policy rule fired) go to the LLM:
   - prompt building (structured, readable sections)
   - LLM call via a provider abstraction (OpenAI / Gemini / Ollama / Mock)
   - validation of the LLM output
   - confidence calibration (retrieval + safety + media signals)
   - postprocessing (attach evidence IDs from retrieval only)

The router is provider-independent: switch models by changing the
LLM_PROVIDER environment variable.
"""

from __future__ import annotations

from typing import Any

from llm.confidence import calibrate_confidence
from llm.postprocessor import build_final_decision
from llm.prompt_builder import SYSTEM_PROMPT, build_prompt_summary, build_user_prompt
from llm.providers import LLMProvider, create_provider
from llm.schemas import LLMResponse, RouterTrace, RoutingDecision
from llm.validator import ValidationError, fallback_response, validate_response
from media.processor import MediaProcessor
from media.types import MediaResult
from models.schemas import MessageContext
from policy.engine import PolicyEngine
from policy.types import PolicyDecision
from retrieval.retriever import RetrievalEngine
from retrieval.types import RetrievalResult
from utils.logger import get_logger

logger = get_logger(__name__)


class Router:
    """Routes a message context to a final notification decision."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        retrieval_engine: RetrievalEngine | None = None,
        media_processor: MediaProcessor | None = None,
        policy_engine: PolicyEngine | None = None,
    ) -> None:
        self.provider = provider or create_provider()
        self.retrieval_engine = retrieval_engine or RetrievalEngine()
        self.media_processor = media_processor or MediaProcessor()
        self.policy_engine = policy_engine or PolicyEngine()
        logger.info("Router initialized with provider=%s.", type(self.provider).__name__)

    def route(
        self,
        context: MessageContext,
        retrieval: RetrievalResult | None = None,
        media: MediaResult | None = None,
    ) -> RouterTrace:
        """Run the full routing pipeline for one message context.

        Parameters
        ----------
        context : MessageContext
            Fully assembled message context.
        retrieval : RetrievalResult | None
            Pre-computed retrieval result (computed if not provided).
        media : MediaResult | None
            Pre-computed media result (computed if not provided).

        Returns
        -------
        RouterTrace
            Full trace including prompt summary, LLM output, validated
            output, final decision, and calibration notes.
        """
        # 1. Compute retrieval and media if not provided.
        if retrieval is None:
            retrieval = self.retrieval_engine.retrieve(context)
        if media is None:
            media = self.media_processor.process(context)

        # 2. Run the deterministic policy engine.
        policy_decision = self.policy_engine.evaluate(context, retrieval, media)

        if policy_decision is not None:
            # Skip the LLM entirely — produce the final decision directly.
            return self._from_policy(context.message_id, retrieval, policy_decision)

        # 3. Build the prompt (only for ambiguous messages).
        user_prompt = build_user_prompt(context, retrieval, media)
        prompt_summary = build_prompt_summary(context, retrieval, media)

        # 4. Call the LLM provider.
        try:
            llm_response = self.provider.complete(SYSTEM_PROMPT, user_prompt)
        except Exception as exc:
            logger.error("LLM call failed for %s: %s", context.message_id, exc)
            llm_response = fallback_response(context.message_id, str(exc))

        # 5. Validate the LLM output.
        try:
            validated = validate_response(llm_response)
        except ValidationError as exc:
            validated = fallback_response(context.message_id, str(exc))

        # 6. Calibrate confidence.
        calibrated, notes = calibrate_confidence(
            validated.confidence, retrieval, media
        )

        # 7. Build the final decision (evidence from retrieval only).
        decision = build_final_decision(
            message_id=context.message_id,
            validated=validated,
            retrieval=retrieval,
            calibrated_confidence=calibrated,
        )

        return RouterTrace(
            message_id=context.message_id,
            prompt_summary=prompt_summary,
            llm_response=llm_response,
            validated=validated,
            final_decision=decision,
            calibration_notes=notes,
            policy_rule=None,
            llm_skipped=False,
        )

    # ------------------------------------------------------------------
    # Policy path
    # ------------------------------------------------------------------

    def _from_policy(
        self,
        message_id: str,
        retrieval: RetrievalResult,
        policy: PolicyDecision,
    ) -> RouterTrace:
        """Build a RouterTrace from a policy decision (LLM skipped)."""
        decision = RoutingDecision(
            message_id=message_id,
            action=policy.action,
            message_type=policy.message_type,
            reason=policy.reason,
            confidence=policy.confidence,
            evidence_message_ids=policy.evidence_message_ids,
        )

        # The policy decision acts as both the LLM "response" and validated output.
        policy_response = LLMResponse(
            action=policy.action,
            message_type=policy.message_type,
            reason=policy.reason,
            confidence=policy.confidence,
            raw="policy",
        )

        logger.info(
            "Policy decision for %s: rule=%s action=%s type=%s confidence=%.2f "
            "(LLM skipped)",
            message_id,
            policy.rule_name,
            policy.action,
            policy.message_type,
            policy.confidence,
        )

        return RouterTrace(
            message_id=message_id,
            prompt_summary=f"Policy rule '{policy.rule_name}' fired; LLM skipped.",
            llm_response=policy_response,
            validated=policy_response,
            final_decision=decision,
            calibration_notes=["policy_override"],
            policy_rule=policy.rule_name,
            llm_skipped=True,
        )