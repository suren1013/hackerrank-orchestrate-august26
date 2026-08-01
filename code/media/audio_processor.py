"""Audio processor stub.

Later phases will add speech-to-text (Whisper) and audio understanding for
voice notes. This stub keeps the module importable and defines the interface.
"""

from __future__ import annotations

from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


class AudioProcessor:
    """Placeholder for voice-note transcription."""

    def __init__(self) -> None:
        logger.info("AudioProcessor initialized (stub).")

    def transcribe(self, audio_path: str) -> dict[str, Any]:
        """Return transcription and metadata for an audio file.

        Stub implementation returns an empty result.
        """
        return {"audio_path": audio_path, "transcript": "", "language": ""}