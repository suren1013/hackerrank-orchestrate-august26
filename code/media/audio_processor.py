"""Audio processor: transcribes voice notes using Whisper.

Uses faster-whisper (CTranslate2-backed, lighter than openai-whisper).
If the model cannot be loaded (e.g. offline), a graceful fallback returns
a MediaResult with an error and empty text.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from config import WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_MODEL_SIZE
from media.text_analysis import analyze_text
from media.types import MediaResult, media_result_from_error
from utils.logger import get_logger

logger = get_logger(__name__)


class AudioProcessor:
    """Transcribes voice notes and produces a typed MediaResult."""

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

        # Attempt to load the Whisper model lazily on first use.
        try:
            from faster_whisper import WhisperModel

            self._whisper_cls = WhisperModel
            logger.info(
                "AudioProcessor initialized "
                "(model=%s, device=%s, compute_type=%s).",
                model_size,
                device,
                compute_type,
            )
        except ImportError:
            self._whisper_cls = None
            logger.warning(
                "faster-whisper not installed; voice transcription disabled."
            )

    def transcribe(
        self,
        audio_path: str | Path,
        message_id: str = "",
        media_id: str = "",
    ) -> MediaResult:
        """Transcribe an audio file and return a MediaResult.

        Parameters
        ----------
        audio_path : str | Path
            Path to the audio file.
        message_id : str
            Incoming message ID (for traceability).
        media_id : str
            Voice note ID from voice_notes.csv.

        Returns
        -------
        MediaResult
            Typed result with transcript, summary, entities, and signals.
        """
        path = Path(audio_path)
        if not path.exists():
            logger.warning("Audio not found: %s", path)
            return media_result_from_error(
                message_id, "voice", media_id, f"audio not found: {path}"
            )

        if self._whisper_cls is None:
            return media_result_from_error(
                message_id,
                "voice",
                media_id,
                "faster-whisper not available; transcription disabled.",
            )

        try:
            transcript = self._transcribe(path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Transcription failed for %s: %s", path, exc)
            return media_result_from_error(
                message_id, "voice", media_id, f"transcription failed: {exc}"
            )

        analysis = analyze_text(transcript)

        confidence = _estimate_transcript_confidence(transcript)
        summary = _build_summary(transcript, analysis)

        return MediaResult(
            message_id=message_id,
            media_type="voice",
            media_id=media_id,
            extracted_text=transcript,
            summary=summary,
            entities=analysis["entities"],
            urgency_indicators=analysis["urgency_indicators"],
            safety_indicators=analysis["safety_indicators"],
            confidence=confidence,
        )

    def _get_model(self):
        """Lazily load and cache the Whisper model."""
        if self._model is None:
            self._model = self._whisper_cls(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("Whisper model '%s' loaded.", self.model_size)
        return self._model

    @lru_cache(maxsize=64)
    def _transcribe(self, audio_path: Path) -> str:
        """Transcribe an audio file path (cached for repeated work)."""
        model = self._get_model()
        segments, _info = model.transcribe(str(audio_path))
        return " ".join(segment.text.strip() for segment in segments).strip()


def _estimate_transcript_confidence(transcript: str) -> float:
    """Estimate confidence for a transcript.

    Heuristic: 0.0 for empty transcript, escalating with content length.
    """
    cleaned = " ".join(transcript.split())
    if not cleaned:
        return 0.0
    if len(cleaned.split()) < 5:
        return 0.4
    if len(cleaned.split()) < 20:
        return 0.6
    return 0.8


def _build_summary(transcript: str, analysis: dict[str, Any]) -> str:
    """Build a concise summary from the transcript and detected signals."""
    cleaned = " ".join(transcript.split())
    if not cleaned:
        return "No speech detected in voice note."

    parts = ["Transcribed voice note."]
    if analysis["entities"]["times"]:
        parts.append(f"Contains times: {', '.join(analysis['entities']['times'][:3])}.")
    if analysis["entities"]["links"]:
        parts.append(f"Contains links: {', '.join(analysis['entities']['links'][:2])}.")
    if analysis["entities"]["phones"]:
        parts.append(f"Contains phone numbers: {', '.join(analysis['entities']['phones'][:2])}.")
    if analysis["urgency_indicators"]:
        parts.append(f"Urgency signals: {', '.join(analysis['urgency_indicators'])}.")
    if analysis["safety_indicators"]:
        parts.append(f"Safety signals: {', '.join(analysis['safety_indicators'])}.")

    snippet = cleaned[:300]
    parts.append(f"Transcript: {snippet}{'...' if len(cleaned) > 300 else ''}")

    return " ".join(parts)