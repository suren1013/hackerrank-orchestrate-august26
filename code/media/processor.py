"""MediaProcessor: routes messages to the correct media handler.

Routes based on ``media_type`` from the MessageContext:
- ``image`` -> ImageProcessor (OCR)
- ``voice`` -> AudioProcessor (Whisper transcription)
- otherwise -> empty MediaResult

Results are cached on disk keyed by (media_id, file mtime) to avoid
re-running OCR/ASR on repeated messages.
"""

from __future__ import annotations

import json
from pathlib import Path

from config import DATASET_DIR, MEDIA_CACHE_DIR
from media.audio_processor import AudioProcessor
from media.image_processor import ImageProcessor
from media.types import MediaResult, empty_media_result
from models.schemas import MessageContext
from utils.helpers import safe_str
from utils.logger import get_logger

logger = get_logger(__name__)


class MediaProcessor:
    """Routes message contexts to the appropriate media processing engine."""

    def __init__(
        self,
        use_cache: bool = True,
        enable_voice: bool = True,
    ) -> None:
        self.use_cache = use_cache
        self.enable_voice = enable_voice
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor() if enable_voice else None

        if self.use_cache:
            MEDIA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(
            "MediaProcessor initialized (cache=%s, voice=%s).",
            use_cache,
            enable_voice,
        )

    def process(self, context: MessageContext) -> MediaResult:
        """Process media for a message context based on its media_type."""
        media_type = safe_str(context.media_type)
        media_id = safe_str(context.media_id)
        message_id = context.message_id

        # No media -> empty result.
        if not media_type or media_type == "none":
            return empty_media_result(message_id)

        if media_type == "image":
            return self._process_image(context, media_id, message_id)

        if media_type == "voice":
            return self._process_voice(context, media_id, message_id)

        logger.warning("Unknown media_type '%s' for %s.", media_type, message_id)
        return empty_media_result(message_id)

    # ------------------------------------------------------------------
    # Image handling
    # ------------------------------------------------------------------

    def _process_image(
        self, context: MessageContext, media_id: str, message_id: str
    ) -> MediaResult:
        image_meta = context.image or {}
        file_path = safe_str(image_meta.get("file_path"))
        if not file_path:
            logger.warning(
                "No file_path for image media_id=%s (msg=%s).", media_id, message_id
            )
            return empty_media_result(message_id)

        full_path = self._resolve_media_path(file_path)

        cache_key = self._cache_key(media_id, full_path)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        result = self.image_processor.process(
            full_path, message_id=message_id, media_id=media_id
        )
        self._save_cache(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Voice handling
    # ------------------------------------------------------------------

    def _process_voice(
        self, context: MessageContext, media_id: str, message_id: str
    ) -> MediaResult:
        if self.audio_processor is None:
            logger.warning("Voice processing disabled.")
            return empty_media_result(message_id)

        voice_meta = context.voice_note or {}
        file_path = safe_str(voice_meta.get("file_path"))
        if not file_path:
            logger.warning(
                "No file_path for voice media_id=%s (msg=%s).", media_id, message_id
            )
            return empty_media_result(message_id)

        full_path = self._resolve_media_path(file_path)

        cache_key = self._cache_key(media_id, full_path)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        result = self.audio_processor.transcribe(
            full_path, message_id=message_id, media_id=media_id
        )
        self._save_cache(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Path + cache helpers
    # ------------------------------------------------------------------

    def _resolve_media_path(self, file_path: str) -> Path:
        """Resolve a relative media path against the dataset media root."""
        p = Path(file_path)
        if p.is_absolute():
            return p
        # file_paths look like media/images/img_001.jpg relative to dataset/.
        return (DATASET_DIR / p).resolve()

    def _cache_key(self, media_id: str, path: Path) -> str:
        """Build a cache key based on media_id and file modification time."""
        mtime = ""
        try:
            mtime = str(path.stat().st_mtime_ns)
        except OSError:
            pass
        return f"{media_id}__{mtime}"

    def _load_cache(self, key: str) -> MediaResult | None:
        """Load a cached MediaResult from disk if present."""
        if not self.use_cache:
            return None
        cache_file = MEDIA_CACHE_DIR / f"{key}.json"
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            result = MediaResult(**data)
            # Never serve stale error results from cache.
            if result.error:
                return None
            return result
        except Exception as exc:
            logger.warning("Failed to load media cache %s: %s", cache_file, exc)
            return None

    def _save_cache(self, key: str, result: MediaResult) -> None:
        """Persist a MediaResult to disk (only successful results)."""
        if not self.use_cache:
            return
        if result.error:
            # Don't cache failures; they may be transient (e.g. missing lib).
            return
        cache_file = MEDIA_CACHE_DIR / f"{key}.json"
        try:
            cache_file.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Failed to write media cache %s: %s", cache_file, exc)
