"""AI Notification Router: produces the final routing decision.

Orchestrates:
1. Prompt building (structured, readable sections)
2. LLM call via a provider abstraction (OpenAI / Gemini / Ollama / Mock)
3. Validation of the LLM output
4. Confidence calibration (retrieval + safety + media signals)
5. Postprocessing (attach evidence IDs from retrieval only)

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
    ) -> None:
        self.provider = provider or create_provider()
        self.retrieval_engine = retrieval_engine or RetrievalEngine()
        self.media_processor = media_processor or MediaProcessor()
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

        # 2. Build the prompt.
        user_prompt = build_user_prompt(context, retrieval, media)
        prompt_summary = build_prompt_summary(context, retrieval, media)

        # 3. Call the LLM provider.
        try:
            llm_response = self.provider.complete(SYSTEM_PROMPT, user_prompt)
        except Exception as exc:
            logger.error("LLM call failed for %s: %s", context.message_id, exc)
            llm_response = fallback_response(context.message_id, str(exc))

        # 4. Validate the LLM output.
        try:
            validated = validate_response(llm_response)
        except ValidationError as exc:
            validated = fallback_response(context.message_id, str(exc))

        # 5. Calibrate confidence.
        calibrated, notes = calibrate_confidence(
            validated.confidence, retrieval, media
        )

        # 6. Build the final decision (evidence from retrieval only).
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
        )