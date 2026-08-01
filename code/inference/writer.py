"""Output writer for the inference pipeline.

Writes the final output.csv with the exact HackerRank contract columns.
"""

from __future__ import annotations

import csv
from pathlib import Path

from config import OUTPUT_COLUMNS, OUTPUT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_FILENAME = "output.csv"


class OutputWriter:
    """Writes routing decisions to output.csv."""

    def __init__(self, output_dir: Path | str | None = None) -> None:
        self.output_dir = Path(output_dir or OUTPUT_DIR).resolve()
        self.output_path = self.output_dir / OUTPUT_FILENAME
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, rows: list[dict]) -> Path:
        """Write rows to output.csv and return the output path.

        Parameters
        ----------
        rows : list[dict]
            List of output rows with keys matching OUTPUT_COLUMNS.

        Returns
        -------
        Path
            Path to the written output.csv.
        """
        with open(self.output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Output written: %d rows -> %s", len(rows), self.output_path)
        return self.output_path