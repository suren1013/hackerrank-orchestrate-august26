"""Image processor: runs OCR on image messages and extracts insights.

Uses Tesseract OCR (via pytesseract). If a vision-capable LLM is available
later, a summary can be generated from the OCR output; for now the OCR text
is used as the summary.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pytesseract
from PIL import Image

from config import TESSERACT_CMD
from media.text_analysis import analyze_text
from media.types import MediaResult, media_result_from_error
from utils.logger import get_logger

logger = get_logger(__name__)

# Configure the Tesseract binary path.
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


class ImageProcessor:
    """Runs OCR on image files and produces a typed MediaResult."""

    def __init__(self) -> None:
        logger.info("ImageProcessor initialized (Tesseract OCR).")

    def process(
        self,
        image_path: str | Path,
        message_id: str = "",
        media_id: str = "",
    ) -> MediaResult:
        """Process an image file and return a MediaResult.

        Parameters
        ----------
        image_path : str | Path
            Path to the image file.
        message_id : str
            Incoming message ID (for traceability).
        media_id : str
            Image ID from images.csv.

        Returns
        -------
        MediaResult
            Typed result with OCR text, summary, entities, and signals.
        """
        path = Path(image_path)
        if not path.exists():
            logger.warning("Image not found: %s", path)
            return media_result_from_error(
                message_id, "image", media_id, f"image not found: {path}"
            )

        try:
            ocr_text = self._run_ocr(path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("OCR failed for %s: %s", path, exc)
            return media_result_from_error(
                message_id, "image", media_id, f"OCR failed: {exc}"
            )

        analysis = analyze_text(ocr_text)

        # Confidence: Tesseract provides per-word confidences; we estimate
        # a simple heuristic based on how much text was extracted.
        confidence = _estimate_ocr_confidence(ocr_text)

        summary = _build_summary(ocr_text, analysis)

        return MediaResult(
            message_id=message_id,
            media_type="image",
            media_id=media_id,
            extracted_text=ocr_text,
            summary=summary,
            entities=analysis["entities"],
            urgency_indicators=analysis["urgency_indicators"],
            safety_indicators=analysis["safety_indicators"],
            confidence=confidence,
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def _run_ocr(image_path: Path) -> str:
        """Run Tesseract OCR on an image path (cached for repeated work)."""
        with Image.open(image_path) as img:
            return pytesseract.image_to_string(img)


def _estimate_ocr_confidence(text: str) -> float:
    """Estimate extraction confidence from the OCR text.

    Heuristic: 0.0 for empty text, escalating with more extracted content.
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return 0.0
    if len(cleaned) < 20:
        return 0.4
    if len(cleaned) < 80:
        return 0.6
    return 0.8


def _build_summary(ocr_text: str, analysis: dict[str, Any]) -> str:
    """Build a concise summary from OCR output and detected signals.

    Currently uses the OCR text (truncated) because no vision LLM is
    configured. Later phases can swap in an LLM summary.
    """
    cleaned = " ".join(ocr_text.split())
    if not cleaned:
        return "No text detected in image."

    parts = ["OCR extracted text from the image."]
    if analysis["entities"]["dates"]:
        parts.append(f"Contains dates: {', '.join(analysis['entities']['dates'][:3])}.")
    if analysis["entities"]["money"]:
        parts.append(f"Contains money references: {', '.join(analysis['entities']['money'][:3])}.")
    if analysis["entities"]["links"]:
        parts.append(f"Contains links: {', '.join(analysis['entities']['links'][:2])}.")
    if analysis["urgency_indicators"]:
        parts.append(f"Urgency signals: {', '.join(analysis['urgency_indicators'])}.")
    if analysis["safety_indicators"]:
        parts.append(f"Safety signals: {', '.join(analysis['safety_indicators'])}.")

    # Include a truncated version of the raw OCR text.
    snippet = cleaned[:300]
    parts.append(f"Text: {snippet}{'...' if len(cleaned) > 300 else ''}")

    return " ".join(parts)