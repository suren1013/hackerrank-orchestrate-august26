"""Trace logger for the inference pipeline.

Writes a JSON trace file per message to output/traces/msg_xxx.json.
Never stores API keys or secrets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import OUTPUT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

TRACES_DIRNAME = "traces"


class TraceLogger:
    """Writes per-message trace files for debugging and inspection."""

    def __init__(self, output_dir: Path | str | None = None) -> None:
        self.output_dir = Path(output_dir or OUTPUT_DIR).resolve()
        self.traces_dir = self.output_dir / TRACES_DIRNAME
        self.traces_dir.mkdir(parents=True, exist_ok=True)

    def save_trace(self, message_id: str, trace: dict[str, Any]) -> Path:
        """Save a trace dict to output/traces/{message_id}.json.

        Parameters
        ----------
        message_id : str
            Message ID (used as the filename).
        trace : dict[str, Any]
            Trace data to persist.

        Returns
        -------
        Path
            Path to the written trace file.
        """
        # Sanitize the message_id for use as a filename.
        safe_id = message_id.replace("/", "_").replace("\\", "_")
        trace_path = self.traces_dir / f"{safe_id}.json"

        try:
            trace_path.write_text(
                json.dumps(trace, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Failed to write trace for %s: %s", message_id, exc)
        return trace_path

    def load_trace(self, message_id: str) -> dict[str, Any] | None:
        """Load a saved trace for a message ID.

        Returns
        -------
        dict[str, Any] | None
            The trace dict, or None if not found.
        """
        safe_id = message_id.replace("/", "_").replace("\\", "_")
        trace_path = self.traces_dir / f"{safe_id}.json"
        if not trace_path.exists():
            return None
        try:
            return json.loads(trace_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load trace for %s: %s", message_id, exc)
            return None

    def pretty_print_trace(self, message_id: str) -> None:
        """Pretty-print a saved trace for a message ID."""
        trace = self.load_trace(message_id)
        if trace is None:
            print(f"No trace found for {message_id}.")
            return
        print(json.dumps(trace, indent=2, default=str))