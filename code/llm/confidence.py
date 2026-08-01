"""Confidence calibration for routing decisions.

Combines the LLM's raw confidence with deterministic signals:
- retrieval quality (evidence similarity scores)
- safety signal strength
- media extraction confidence
- trust/interest scores

Produces a calibrated confidence in [0, 1].
"""

from __future__ import annotations

from media.types import MediaResult
from retrieval.types import RetrievalResult
from utils.logger import get_logger

logger = get_logger(__name__)


def calibrate_confidence(
    raw_confidence: float,
    retrieval: RetrievalResult,
    media: MediaResult | None = None,
) -> tuple[float, list[str]]:
    """Calibrate the LLM confidence using retrieval and media signals.

    Returns
    -------
    tuple[float, list[str]]
        Calibrated confidence (0..1) and a list of calibration notes.
    """
    notes: list[str] = []
    score = float(raw_confidence)

    # 1. Retrieval quality: average similarity of top evidence.
    if retrieval.top_similar_messages:
        avg_sim = sum(m.score for m in retrieval.top_similar_messages) / len(
            retrieval.top_similar_messages
        )
        # Normalize similarity (scores can exceed 1.0 due to weighted sums).
        norm_sim = min(1.0, avg_sim / 2.0)
        # Blend: 20% weight on retrieval quality.
        score = 0.8 * score + 0.2 * norm_sim
        notes.append(f"retrieval_similarity={norm_sim:.2f}")

    # 2. Safety signal strength: strong safety signals should boost
    #    confidence in mute decisions and reduce confidence otherwise.
    safety = media.safety_indicators if media else []
    if safety:
        safety_strength = min(1.0, len(safety) * 0.15)
        if retrieval.trust_score < 0.5:
            # Low trust + safety signals -> high confidence in mute.
            score = min(1.0, score + safety_strength)
            notes.append(f"safety_boost={safety_strength:.2f}")
        else:
            # High trust but safety signals -> slightly reduce confidence.
            score = max(0.0, score - safety_strength * 0.5)
            notes.append(f"safety_penalty={safety_strength * 0.5:.2f}")

    # 3. Media extraction confidence.
    if media is not None and media.has_media:
        if media.confidence > 0:
            # Blend 10% of media confidence.
            score = 0.9 * score + 0.1 * media.confidence
            notes.append(f"media_confidence={media.confidence:.2f}")
        else:
            # Failed/empty media extraction reduces confidence.
            score = max(0.0, score - 0.1)
            notes.append("media_extraction_low")

    # 4. Trust/interest scores: anchor confidence.
    #    High trust + high interest -> more confident in notify/digest.
    #    Low trust -> more confident in mute.
    trust = retrieval.trust_score
    interest = retrieval.interest_score
    if trust < 0.3:
        score = min(1.0, score + 0.05)
        notes.append("low_trust_anchor")
    elif trust > 0.7:
        score = max(0.0, score - 0.05)
        notes.append("high_trust_anchor")

    # Clamp to [0, 1].
    calibrated = max(0.0, min(1.0, score))
    notes.append(f"final={calibrated:.2f}")
    return calibrated, notes