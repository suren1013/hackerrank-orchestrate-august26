"""Postprocessor: attaches evidence IDs and builds the final decision.

Evidence IDs come exclusively from the RetrievalResult — never from the LLM.
This prevents hallucinated evidence IDs.
"""

from __future__ import annotations

from llm.schemas import LLMResponse, RoutingDecision
from retrieval.types import RetrievalResult
from utils.logger import get_logger

logger = get_logger(__name__)


def build_final_decision(
    message_id: str,
    validated: LLMResponse,
    retrieval: RetrievalResult,
    calibrated_confidence: float,
) -> RoutingDecision:
    """Build the final RoutingDecision from validated LLM output.

    Parameters
    ----------
    message_id : str
        Incoming message ID.
    validated : LLMResponse
        Validated LLM response.
    retrieval : RetrievalResult
        Retrieval result (source of evidence IDs).
    calibrated_confidence : float
        Calibrated confidence score.

    Returns
    -------
    RoutingDecision
        Final decision with evidence IDs from retrieval only.
    """
    # Evidence IDs come ONLY from the retrieval engine.
    evidence_ids = list(retrieval.evidence_message_ids)

    logger.info(
        "Final decision for %s: action=%s, type=%s, confidence=%.2f, evidence=%d",
        message_id,
        validated.action,
        validated.message_type,
        calibrated_confidence,
        len(evidence_ids),
    )

    return RoutingDecision(
        message_id=message_id,
        action=validated.action,
        message_type=validated.message_type,
        reason=validated.reason,
        confidence=calibrated_confidence,
        evidence_message_ids=evidence_ids,
    )