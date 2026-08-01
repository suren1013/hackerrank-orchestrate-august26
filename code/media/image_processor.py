"""Image processor stub.

Later phases will add OCR / image understanding (e.g. reading posters,
screenshots, and safety-relevant content). This stub keeps the module
importable and defines the interface.
"""

from __future__ import annotations

from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


class ImageProcessor:
    """Placeholder for image understanding."""

    def __init__(self) -> None:
        logger.info("ImageProcessor initialized (stub).")

    def process(self, image_path: str) -> dict[str, Any]:
        """Return extracted information from an image.

        Stub implementation returns an empty result.
        """
        return {"image_path": image_path, "ocr_text": "", "description": ""}