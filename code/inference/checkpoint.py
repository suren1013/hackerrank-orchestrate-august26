"""Checkpointing for the inference pipeline.

Saves partial output every N messages and supports resuming by skipping
already-processed message IDs.
"""

from __future__ import annotations

import csv
from pathlib import Path

from config import OUTPUT_COLUMNS, OUTPUT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

PARTIAL_FILENAME = "partial_output.csv"
CHECKPOINT_INTERVAL = 10


class CheckpointManager:
    """Manages partial output checkpoints for resume support."""

    def __init__(
        self,
        output_dir: Path | str | None = None,
        interval: int = CHECKPOINT_INTERVAL,
    ) -> None:
        self.output_dir = Path(output_dir or OUTPUT_DIR).resolve()
        self.interval = interval
        self.partial_path = self.output_dir / PARTIAL_FILENAME
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def has_checkpoint(self) -> bool:
        """Return True if a partial output checkpoint exists."""
        return self.partial_path.exists()

    def load_processed_ids(self) -> set[str]:
        """Load the set of already-processed message IDs from the checkpoint."""
        if not self.has_checkpoint():
            return set()

        processed: set[str] = set()
        try:
            with open(self.partial_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    mid = row.get("message_id", "").strip()
                    if mid:
                        processed.add(mid)
            logger.info("Loaded %d processed message IDs from checkpoint.", len(processed))
        except Exception as exc:
            logger.warning("Failed to load checkpoint: %s", exc)
        return processed

    def save_checkpoint(self, rows: list[dict]) -> None:
        """Write the current rows to the partial output file."""
        try:
            with open(self.partial_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            logger.info("Checkpoint saved: %d rows -> %s", len(rows), self.partial_path)
        except Exception as exc:
            logger.error("Failed to save checkpoint: %s", exc)

    def clear_checkpoint(self) -> None:
        """Remove the partial output file (after successful completion)."""
        if self.partial_path.exists():
            self.partial_path.unlink()
            logger.info("Checkpoint cleared: %s", self.partial_path)